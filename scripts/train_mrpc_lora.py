#!/usr/bin/env python3
"""
Training script for MRPC RoBERTa LoRA SGLD experiment.
Implements both MAP training and SGLD sampling as described in the experiment outline.
"""

import argparse
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)

from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.data.glue_datasets import MRPCDataset
from bayesian_lora.samplers.sgld import SGLDSampler
from bayesian_lora.utils.lora_params import LoRAParams

# Setup logging to both console and file
def setup_logging():
    """Setup logging to both console and file."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
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

logger = setup_logging()


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
                   device: torch.device):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
                     config: Dict[str, Any], device: torch.device):
    """Train LoRA using SGLD sampling."""
    logger.info("Starting SGLD LoRA training...")
    
    sgld_config = config['training']['sgld_lora']
    
    # Initialize SGLD sampler with proper configuration
    sampler = SGLDSampler(
            model=model,
            temperature=sgld_config['temperature'],
            step_size=sgld_config['learning_rate'],
            noise_scale=sgld_config['noise_scale'],  # Use configurable noise scale
            prior_std=sgld_config['prior_std'],
            gradient_clip_norm=sgld_config['gradient_clip_norm']
        )
    
    # Step size schedule parameters
    initial_step_size = sgld_config['step_size_schedule']['initial']
    decay_rate = sgld_config['step_size_schedule']['decay_rate']
    decay_steps = sgld_config['step_size_schedule']['decay_steps']
    
    # Run multiple chains
    all_samples = []
    for chain in range(sgld_config['chains']):
        logger.info(f"Running chain {chain + 1}/{sgld_config['chains']}")
        
        # Reset model to MAP state for each chain
        # Load MAP model state for fresh start (including chain 0)
        map_state = torch.load(f"runs/mrpc_roberta_lora_sgld/map_model.pth", map_location=device)
        model.load_state_dict(map_state)
        
        # Burn-in phase
        logger.info(f"Chain {chain + 1}: Burn-in phase ({sgld_config['burn_in_steps']} steps)")
        for step in range(sgld_config['burn_in_steps']):
            # Update step size according to schedule
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
                logger.info(f"Burn-in step {step}/{sgld_config['burn_in_steps']}, "
                           f"step_size: {current_step_size:.2e}")
        
        # Sampling phase
        logger.info(f"Chain {chain + 1}: Sampling phase ({sgld_config['sampling_steps']} steps)")
        chain_samples = []
        sample_count = 0
        
        # Track ESS during sampling
        log_posterior_values = []
        l2_norm_values = []
        
        for step in range(sgld_config['sampling_steps']):
            # Update step size according to schedule
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
            
            # Keep samples based on thinning
            if step % sgld_config['thinning'] == 0:
                # Save current model state
                sample_state = sampler.get_current_state()
                chain_samples.append(sample_state)
                sample_count += 1
                
                # Compute MCMC diagnostics for this sample
                model.eval()
                with torch.no_grad():
                    output = model(input_ids, attention_mask=attention_mask)
                    logits = output.logits
                    loss = F.cross_entropy(logits, labels)
                    
                    # Log posterior (negative loss + prior)
                    prior_loss = sum(torch.sum(param ** 2) / (2 * sgld_config['prior_std'] ** 2) 
                                   for param in model.parameters() if param.requires_grad)
                    log_posterior = -(loss + prior_loss).item()
                    log_posterior_values.append(log_posterior)
                    
                    # L2 norm of parameters
                    l2_norm = sum(torch.norm(param).item() for param in model.parameters() if param.requires_grad)
                    l2_norm_values.append(l2_norm)
                model.train()
                
                if sample_count % 100 == 0:
                    logger.info(f"Chain {chain + 1}: Collected {sample_count} samples")
            
            if step % 1000 == 0:
                logger.info(f"Sampling step {step}/{sgld_config['sampling_steps']}, "
                           f"step_size: {current_step_size:.2e}")
        
        # Keep only the specified number of samples per chain
        samples_per_chain = sgld_config['samples_to_retain'] // sgld_config['chains']
        chain_samples = chain_samples[-samples_per_chain:]
        all_samples.extend(chain_samples)
        
        # Compute ESS for this chain
        if len(log_posterior_values) > 0:
            from scripts.eval_mrpc_lora import compute_ess
            import numpy as np
            
            # Keep only the diagnostics for the retained samples
            log_posterior_chain = np.array(log_posterior_values[-samples_per_chain:])
            l2_norm_chain = np.array(l2_norm_values[-samples_per_chain:])
            
            ess_log_posterior = compute_ess(log_posterior_chain)
            ess_l2_norm = compute_ess(l2_norm_chain)
            min_ess = min(ess_log_posterior, ess_l2_norm)
            
            logger.info(f"Chain {chain + 1}: Collected {len(chain_samples)} samples")
            logger.info(f"Chain {chain + 1}: ESS (log_posterior) = {ess_log_posterior}")
            logger.info(f"Chain {chain + 1}: ESS (l2_norm) = {ess_l2_norm}")
            logger.info(f"Chain {chain + 1}: Min ESS = {min_ess}")
        else:
            logger.info(f"Chain {chain + 1}: Collected {len(chain_samples)} samples")
        
        # Clear GPU cache after each chain
        torch.cuda.empty_cache()
    
    logger.info(f"Total samples collected: {len(all_samples)}")
    
    # Compute overall ESS across all chains
    if len(all_samples) > 0:
        logger.info("Computing overall MCMC diagnostics...")
        # Note: Full ESS computation will be done during evaluation
        # Here we just log that samples were collected successfully
        logger.info(f"SGLD sampling completed with {len(all_samples)} total samples")
        logger.info("ESS and R-hat will be computed during evaluation phase")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Train MRPC LoRA with SGLD")
    parser.add_argument("--config", type=str, default="configs/mrpc_roberta_lora_sgld.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="runs/mrpc_roberta_lora_sgld",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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
    map_model = train_map_lora(model, train_dataloader, val_dataloader, config, device)
    
    # Save MAP model
    map_save_path = output_dir / "map_model.pth"
    torch.save(map_model.state_dict(), map_save_path)
    logger.info(f"MAP model saved to {map_save_path}")
    
    # Clear GPU cache before SGLD training
    torch.cuda.empty_cache()
    
    # Train SGLD LoRA
    sgld_samples = train_sgld_lora(model, train_dataloader, config, device)
    
    # Save SGLD samples
    samples_save_path = output_dir / "sgld_samples.pth"
    torch.save(sgld_samples, samples_save_path)
    logger.info(f"SGLD samples saved to {samples_save_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
