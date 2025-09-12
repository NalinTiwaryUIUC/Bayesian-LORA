#!/usr/bin/env python3
"""
Training script for MRPC RoBERTa LoRA SGLD experiment.
Implements both MAP training and SGLD sampling as described in the experiment outline.
"""

import argparse
import logging
import math
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)

from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.data.glue_datasets import MRPCDataset
from bayesian_lora.samplers.sgld import SGLDSampler, SAMSGLDRank1Sampler
from bayesian_lora.utils.lora_params import LoRAParams

# ESS computation moved to evaluation phase

# Setup logging to both console and file
def setup_logging(experiment_name="mrpc_roberta_lora_samsgld_rank1"):
    """Setup logging to both console and file."""
    # Create logs directory with experiment-specific subfolder
    logs_dir = Path(f"logs/{experiment_name}")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(logs_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    def convert_scientific_notation(loader, node):
        """Convert scientific notation strings to floats."""
        value = loader.construct_scalar(node)
        try:
            return float(value)
        except ValueError:
            return value
    
    # Create custom loader that handles scientific notation
    class ScientificNotationLoader(yaml.SafeLoader):
        pass
    
    # Register the converter for scalar nodes
    ScientificNotationLoader.add_constructor('tag:yaml.org,2002:str', convert_scientific_notation)
    
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=ScientificNotationLoader)


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Initialize RoBERTa model and tokenizer with LoRA."""
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['backbone'])
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = RobertaForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2  # MRPC is binary classification
    )
    
    # Apply LoRA
    lora_config = config['model']['lora']
    lora_model = LoRAModel(
        model,
        r=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=lora_config['inject_into']
    )
    
    return lora_model, tokenizer


def train_map_lora(model: LoRAModel, train_dataloader: DataLoader, 
                   val_dataloader: DataLoader, config: Dict[str, Any], 
                   device: torch.device, logger):
    """Train LoRA using MAP estimation."""
    logger.info("Starting MAP LoRA training...")
    
    train_config = config['training']['map_lora']
    
    # Setup optimizer
    if train_config['optimizer'] == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            betas=(train_config['beta1'], train_config['beta2']),
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config.get('momentum', 0.9),
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
    
    # Setup scheduler
    if train_config.get('scheduler', 'linear') == 'none':
        scheduler = None
    else:
        total_steps = len(train_dataloader) * config['data']['max_epochs']
        warmup_steps = int(total_steps * train_config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    # Training loop with gradient accumulation
    model.train()
    gradient_accumulation_steps = config['data'].get('gradient_accumulation_steps', 1)
    
    for epoch in range(config['data']['max_epochs']):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Increased from 1.0 to allow larger updates
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs.logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        logger.info(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_dataloader):.4f}, "
                   f"Val Loss: {val_loss/len(val_dataloader):.4f}, "
                   f"Val Acc: {val_accuracy:.4f}")
        
        model.train()
    
    return model


def train_sgld_lora(model: LoRAModel, train_dataloader: DataLoader,
                     config: Dict[str, Any], device: torch.device, logger, output_dir: Path) -> List[Dict[str, Any]]:
    """Train LoRA using SGLD or SAM-SGLD sampling."""
    
    # Determine which sampler to use based on config
    if 'sgld_lora' in config['training']:
        logger.info("Starting SGLD LoRA training...")
        sgld_config = config['training']['sgld_lora']
        sampler_class = SGLDSampler
    elif 'samsgld_rank1_lora' in config['training']:
        logger.info("Starting SAM-SGLD Rank-1 LoRA training...")
        sgld_config = config['training']['samsgld_rank1_lora']
        sampler_class = SAMSGLDRank1Sampler
    else:
        raise ValueError("No SGLD or SAM-SGLD configuration found in training config")
    
    # Initialize sampler with proper configuration
    if sampler_class == SGLDSampler:
        sampler = SGLDSampler(
            model=model,
            temperature=sgld_config['temperature'],
            step_size=sgld_config['learning_rate'],
            noise_scale=sgld_config['noise_scale'],
            prior_std=sgld_config['prior_std'],
            gradient_clip_norm=sgld_config['gradient_clip_norm']
        )
    elif sampler_class == SAMSGLDRank1Sampler:
        sampler = SAMSGLDRank1Sampler(
            model=model,
            temperature=sgld_config['temperature'],
            step_size=sgld_config['learning_rate'],
            noise_scale=sgld_config['noise_scale'],
            rho=sgld_config['rho'],
            lambd=sgld_config['lambd'],
            sigma_dir=sgld_config['sigma_dir'],
            gradient_clip_norm=sgld_config['gradient_clip_norm'],
            prior_std=sgld_config['prior_std']
        )
    
    # Step size schedule parameters
    initial_step_size = sgld_config['step_size_schedule']['initial']
    decay_rate = sgld_config['step_size_schedule']['decay_rate']
    decay_steps = sgld_config['step_size_schedule']['decay_steps']
    
    # Run multiple chains
    all_samples = []
    for chain in range(sgld_config['chains']):
        logger.info(f"Running chain {chain + 1}/{sgld_config['chains']}")
        
        # No per-chain seeding (matching working version)

        # Reset model to MAP state for each chain
        # Load MAP model state for fresh start (including chain 0)
        map_state_path = output_dir / "map_model.pth"
        map_state = torch.load(map_state_path, map_location=device)
        model.load_state_dict(map_state)
        
        # Verify MAP model loading by checking a few parameter norms
        param_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                param_norms.append(param.norm().item())
                if len(param_norms) >= 3:  # Check first 3 LoRA parameters
                    break
        
        logger.info(f"Chain {chain + 1}: MAP model loaded - LoRA param norms: {param_norms[:3]}")
        
        # Verify MAP model performance by testing on a small batch
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(train_dataloader))
            test_input_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)
            test_labels = test_batch['labels'].to(device)
            
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask)
            test_loss = F.cross_entropy(test_outputs.logits, test_labels)
            test_preds = torch.argmax(test_outputs.logits, dim=1)
            test_acc = (test_preds == test_labels).float().mean()
            
            logger.info(f"Chain {chain + 1}: MAP model test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        # Log initial configuration for this chain
        logger.info(f"Chain {chain + 1}: Initial configuration:")
        logger.info(f"  Initial step size: {initial_step_size:.2e}")
        logger.info(f"  Temperature: {sampler.temperature:.1f}")
        logger.info(f"  Noise scale: {sampler.noise_scale:.2e}")
        logger.info(f"  Prior std: {sampler.prior_std:.3f}")
        logger.info(f"  SAM rho: {sampler.rho:.3f}")
        logger.info(f"  Decay rate: {decay_rate:.3f}")
        
        # Burn-in phase
        logger.info(f"Chain {chain + 1}: Burn-in phase ({sgld_config['burn_in_steps']} steps)")
        for step in range(sgld_config['burn_in_steps']):
            # Update step size according to schedule using correct power law decay
            current_step_size = initial_step_size * (1 + step / decay_steps) ** (-decay_rate)
            sampler.step_size = current_step_size
            
            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            sampler.step(input_ids, attention_mask, labels)
            
            # Clear GPU cache more frequently to prevent memory issues
            if step % 50 == 0:
                torch.cuda.empty_cache()
            
            if step % 500 == 0:
                # Get actual drift and noise from last step
                actual_drift_norm = sampler.last_drift_norm
                actual_noise_norm = sampler.last_noise_norm
                actual_drift_noise_ratio = actual_drift_norm / actual_noise_norm if actual_noise_norm > 0 else float('inf')
                
                # Calculate theoretical values for comparison
                theoretical_noise_std = math.sqrt(2 * current_step_size / sampler.temperature) * sampler.noise_scale
                theoretical_step_noise_ratio = current_step_size / theoretical_noise_std
                prior_likelihood_ratio = (sampler.prior_std ** 2) / (2 * sampler.temperature)
                
                logger.info(f"Burn-in step {step}/{sgld_config['burn_in_steps']}:")
                logger.info(f"  Actual drift norm: {actual_drift_norm:.2e}")
                logger.info(f"  Actual noise norm: {actual_noise_norm:.2e}")
                logger.info(f"  Actual drift/noise ratio: {actual_drift_noise_ratio:.3f}")
                logger.info(f"  Theoretical step/noise ratio: {theoretical_step_noise_ratio:.3f}")
                logger.info(f"  Prior/Likelihood ratio: {prior_likelihood_ratio:.3f}")
        
        # Sampling phase
        logger.info(f"Chain {chain + 1}: Sampling phase ({sgld_config['sampling_steps']} steps)")
        chain_samples = []
        sample_count = 0
        
        # Track diagnostic values during sampling (disabled to avoid interference)
        # log_posterior_values = []
        # l2_norm_values = []
        
        # No sample collection logic (matching working version)
        
        # Use fixed step size during sampling for better sample independence
        # Calculate the step size at the end of burn-in using correct power law decay
        burn_in_end_step_size = initial_step_size * (1 + sgld_config['burn_in_steps'] / decay_steps) ** (-decay_rate)
        sampler.step_size = burn_in_end_step_size
        
        # Log step size change
        step_size_reduction = (initial_step_size - burn_in_end_step_size) / initial_step_size * 100
        logger.info(f"Chain {chain + 1}: Step size after burn-in:")
        logger.info(f"  Initial: {initial_step_size:.2e}")
        logger.info(f"  Final: {burn_in_end_step_size:.2e}")
        logger.info(f"  Reduction: {step_size_reduction:.1f}%")
        
        for step in range(sgld_config['sampling_steps']):
            # Keep step size constant during sampling for better sample independence
            # (No step size update during sampling phase)
            
            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            sampler.step(input_ids, attention_mask, labels)
            
            # Log key diagnostic ratios every 100 steps
            if step % 100 == 0:
                # Get actual drift and noise from last step
                actual_drift_norm = sampler.last_drift_norm
                actual_noise_norm = sampler.last_noise_norm
                actual_step_size = sampler.last_step_size
                
                # Calculate actual ratios
                actual_drift_noise_ratio = actual_drift_norm / actual_noise_norm if actual_noise_norm > 0 else float('inf')
                
                # Calculate theoretical values for comparison
                theoretical_noise_std = math.sqrt(2 * actual_step_size / sampler.temperature) * sampler.noise_scale
                theoretical_step_noise_ratio = actual_step_size / theoretical_noise_std
                
                # Calculate prior/likelihood ratio (approximate)
                # Get current model parameters for prior calculation
                total_param_norm = 0
                total_params = 0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        total_param_norm += param.norm().item() ** 2
                        total_params += param.numel()
                
                avg_param_norm = math.sqrt(total_param_norm / total_params) if total_params > 0 else 0
                prior_likelihood_ratio = (sampler.prior_std ** 2) / (2 * sampler.temperature)
                
                # Log diagnostic information
                logger.info(f"Chain {chain + 1}, Step {step}:")
                logger.info(f"  Actual drift norm: {actual_drift_norm:.2e}")
                logger.info(f"  Actual noise norm: {actual_noise_norm:.2e}")
                logger.info(f"  Actual drift/noise ratio: {actual_drift_noise_ratio:.3f}")
                logger.info(f"  Theoretical step/noise ratio: {theoretical_step_noise_ratio:.3f}")
                logger.info(f"  Prior/Likelihood ratio: {prior_likelihood_ratio:.3f}")
                logger.info(f"  SAM rho: {sampler.rho:.3f}")
                logger.info(f"  Avg param norm: {avg_param_norm:.3f}")
                logger.info(f"  Temperature: {sampler.temperature:.1f}")
            
            # Clear GPU cache more frequently to prevent memory issues
            if step % 50 == 0:
                torch.cuda.empty_cache()
            
            # Keep samples based on thinning
            if step % sgld_config['thinning'] == 0:
                # Save current model state
                sample_state = sampler.get_current_state()
                chain_samples.append(sample_state)
                sample_count += 1
                
                if sample_count % 100 == 0:
                    logger.info(f"Chain {chain + 1}: Collected {sample_count} samples")
            
            if step % 1000 == 0:
                logger.info(f"Sampling step {step}/{sgld_config['sampling_steps']}, "
                           f"step_size: {sampler.step_size:.2e}")
        
        # Keep only the specified number of samples per chain
        samples_per_chain = sgld_config['samples_to_retain'] // sgld_config['chains']
        chain_samples = chain_samples[-samples_per_chain:]
        all_samples.extend(chain_samples)
        
        # Diagnostic values collection disabled to avoid interference with sampling
        # all_log_posterior_values.extend(log_posterior_values)
        # all_l2_norm_values.extend(l2_norm_values)
        
        # Log sample collection (ESS computation moved to evaluation phase)
        logger.info(f"Chain {chain + 1}: Collected {len(chain_samples)} samples")
        
        # Evaluate model metrics after each chain (disabled for now to avoid performance issues)
        # logger.info(f"Chain {chain + 1}: Evaluating model metrics...")
        # try:
        #     metrics = evaluate_model_metrics(model, train_dataloader, device, sgld_config['prior_std'], logger)
        #     logger.info(f"Chain {chain + 1}: Loss = {metrics['loss']:.4f}")
        #     logger.info(f"Chain {chain + 1}: Accuracy = {metrics['accuracy']:.4f}")
        #     logger.info(f"Chain {chain + 1}: NLL = {metrics['nll']:.4f}")
        # except Exception as e:
        #     logger.warning(f"Chain {chain + 1}: Failed to evaluate metrics: {e}")
        
        # Clear GPU cache after each chain
        torch.cuda.empty_cache()
    
    logger.info(f"Total samples collected: {len(all_samples)}")
    
    # Compute overall ESS across all chains
    if len(all_samples) > 0:
        logger.info("Computing overall MCMC diagnostics...")
        # Note: Full ESS computation will be done during evaluation
        # Here we just log that samples were collected successfully
        sampler_name = "SGLD" if sampler_class == SGLDSampler else "SAM-SGLD"
        logger.info(f"{sampler_name} sampling completed with {len(all_samples)} total samples")
        
        # Check final model performance after sampling
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(train_dataloader))
            test_input_ids = test_batch['input_ids'].to(device)
            test_attention_mask = test_batch['attention_mask'].to(device)
            test_labels = test_batch['labels'].to(device)
            
            test_outputs = model(test_input_ids, attention_mask=test_attention_mask)
            test_loss = F.cross_entropy(test_outputs.logits, test_labels)
            test_preds = torch.argmax(test_outputs.logits, dim=1)
            test_acc = (test_preds == test_labels).float().mean()
            
            logger.info(f"Final model after sampling - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        logger.info("ESS and R-hat will be computed during evaluation phase")
    
    # Return samples only (diagnostics disabled to avoid interference)
    return all_samples


def evaluate_model_metrics(model: LoRAModel, dataloader: DataLoader, device: torch.device, 
                          prior_std: float, logger: logging.Logger) -> Dict[str, float]:
    """Evaluate model and return loss, accuracy, and NLL metrics."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_nll = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            
            # Compute likelihood loss (NLL)
            likelihood_loss = F.cross_entropy(logits, labels, reduction='sum')
            
            # Compute prior loss
            prior_loss = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    prior_loss += torch.sum(param ** 2) / (2 * prior_std ** 2)
            
            # Total loss (negative log posterior)
            total_loss += likelihood_loss.item() + prior_loss.item()
            
            # NLL (negative log likelihood without prior)
            total_nll += likelihood_loss.item()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().sum().item()
            total_accuracy += accuracy
            
            total_samples += labels.size(0)
    
    # Average metrics
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_nll = total_nll / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'nll': avg_nll
    }


def main():
    parser = argparse.ArgumentParser(description="Train MRPC LoRA with SGLD or SAM-SGLD")
    parser.add_argument("--config", type=str, default="configs/mrpc_roberta_lora_samsgld_rank1.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="runs/mrpc_roberta_lora_samsgld_rank1",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging with experiment name from config
    experiment_name = config['experiment']['name']
    logger = setup_logging(experiment_name)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    model.to(device)
    
    # Setup datasets
    train_dataset = MRPCDataset(split="train", tokenizer=tokenizer, 
                               max_length=config['model']['max_sequence_length'])
    val_dataset = MRPCDataset(split="validation", tokenizer=tokenizer,
                              max_length=config['model']['max_sequence_length'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], 
                                 shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'])
    
    # Train MAP LoRA
    map_model = train_map_lora(model, train_dataloader, val_dataloader, config, device, logger)
    
    # Save MAP model
    map_save_path = output_dir / "map_model.pth"
    torch.save(map_model.state_dict(), map_save_path)
    logger.info(f"MAP model saved to {map_save_path}")
    
    # Clear GPU cache before sampling training
    torch.cuda.empty_cache()
    
    # Determine sampler type for logging
    if 'sgld_lora' in config['training']:
        sampler_name = "SGLD"
        samples_filename = "sgld_samples.pth"
    elif 'samsgld_rank1_lora' in config['training']:
        sampler_name = "SAM-SGLD"
        samples_filename = "samsgld_samples.pth"
    else:
        sampler_name = "Unknown"
        samples_filename = "samples.pth"
    
    # Train sampler LoRA
    samples = train_sgld_lora(model, train_dataloader, config, device, logger, output_dir)
    
    # Save samples and diagnostic values
    samples_save_path = output_dir / samples_filename
    torch.save(samples, samples_save_path)
    logger.info(f"{sampler_name} samples saved to {samples_save_path}")
    
    # Diagnostic values saving disabled to avoid interference with sampling
    # diagnostics_save_path = output_dir / "diagnostics.pth"
    # torch.save(diagnostics, diagnostics_save_path)
    # logger.info(f"Diagnostic values saved to {diagnostics_save_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
