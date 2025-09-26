#!/usr/bin/env python3
"""
Training script for MRPC LoRA with online SAM-SGLD.
Based on sam_sgld_experiment_specifications.md
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_lora.data.glue_datasets import MRPCDataset
from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.samplers.sgld import SAMSGLDRank1Sampler
from bayesian_lora.utils.online_estimators import OnlineMetricsTracker, CumulativeEnsemble


def setup_model_and_tokenizer(config):
    """Setup model and tokenizer based on config."""
    # Create base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2
    )
    
    # Create LoRA model
    lora_config = config['model']['lora']
    model = LoRAModel(
        base_model,
        r=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=lora_config['inject_into']
    )
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
    
    return model, tokenizer


def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_per_sample_metrics(model, dataloader, device):
    """Compute per-sample accuracy, NLL, and ECE."""
    model.eval()
    all_logits = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    # Concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    # Compute metrics
    predictions = torch.argmax(all_probs, dim=-1)
    accuracy = (predictions == all_labels).float().mean().item()
    
    # NLL
    log_probs = torch.log_softmax(all_logits, dim=-1)
    nll = -log_probs[range(len(all_labels)), all_labels].mean().item()
    
    # ECE (simplified - bin-based calibration error)
    confidences = torch.max(all_probs, dim=-1)[0]
    ece = compute_ece(all_probs, all_labels, confidences)
    
    return {
        'accuracy': accuracy,
        'nll': nll,
        'ece': ece,
        'logits': all_logits,
        'labels': all_labels
    }


def compute_ece(probs, labels, confidences, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (labels[in_bin] == torch.argmax(probs[in_bin], dim=-1)).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def extract_diagnostic_scalars(model, sampler):
    """Extract diagnostic scalars for ESS estimation."""
    scalars = {}
    
    # Log posterior (simplified - just log prior for now)
    total_params = 0
    prior_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:
            prior_loss += (param ** 2).sum().item() / (2 * sampler.prior_std ** 2)
            total_params += param.numel()
    
    scalars['log_posterior'] = -prior_loss
    
    # Parameter L2 norm
    param_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm += (param ** 2).sum().item()
    scalars['param_l2_norm'] = math.sqrt(param_norm)
    
    # SAM-specific diagnostics (only if sampler has SAM attributes)
    if hasattr(sampler, 'last_sam_perturbation_norm'):
        scalars['sam_perturbation_norm'] = sampler.last_sam_perturbation_norm
    if hasattr(sampler, 'last_sam_gradient_norm'):
        scalars['sam_gradient_norm'] = sampler.last_sam_gradient_norm
    if hasattr(sampler, 'last_drift_norm'):
        scalars['drift_norm'] = sampler.last_drift_norm
    if hasattr(sampler, 'last_noise_norm'):
        scalars['noise_norm'] = sampler.last_noise_norm
    
    # LoRA block Frobenius norms (if available)
    if hasattr(model, 'lora_modules'):
        for i, module in enumerate(model.lora_modules):
            if hasattr(module, 'weight'):
                frobenius_norm = torch.norm(module.weight, p='fro').item()
                scalars[f'lora_block_{i}_frobenius'] = frobenius_norm
    
    # Random projections of LoRA parameters
    all_params = []
    for param in model.parameters():
        if param.requires_grad:
            all_params.append(param.flatten())
    
    if all_params:
        param_vector = torch.cat(all_params, dim=0)
        # Use fixed random projections
        torch.manual_seed(42)
        projection_0 = torch.randn_like(param_vector)
        projection_1 = torch.randn_like(param_vector)
        
        scalars['lora_projection_0'] = torch.dot(param_vector, projection_0).item()
        scalars['lora_projection_1'] = torch.dot(param_vector, projection_1).item()
    
    return scalars


def train_map_lora(model, train_dataloader, val_dataloader, config, device, logger):
    """Train MAP LoRA model."""
    logger.info("Starting MAP LoRA training...")
    
    # Setup optimizer
    map_config = config['training']['map_lora']
    if map_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(map_config['learning_rate']),
            weight_decay=float(map_config['weight_decay']),
            betas=(float(map_config['beta1']), float(map_config['beta2'])),
            eps=float(map_config['eps'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {map_config['optimizer']}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(int(config['data']['max_epochs'])):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_dataloader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_dataloader):.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    logger.info("MAP training completed!")
    return model


def run_samsgld_sampling(model, train_dataloader, val_dataloader, config, device, logger):
    """Run SAM-SGLD sampling with online ESS tracking."""
    logger.info("Starting SAM-SGLD sampling...")
    
    # Initialize sampler
    sampler = SAMSGLDRank1Sampler(
        model=model,
        temperature=float(config['temperature']),
        step_size=float(config['step_size_schedule']['initial']),
        noise_scale=float(config['noise_scale']),
        prior_std=float(config['prior_std']),
        gradient_clip_norm=float(config['gradient_clip_norm']),
        rho=float(config['rho']),
        lambd=float(config['lambd']),
        sigma_dir=float(config['sigma_dir'])
    )
    
    # Initialize online estimators
    diagnostic_scalars = config['online_ess']['scalars']
    # Include metrics in the tracked scalars
    all_scalars = diagnostic_scalars + ['accuracy', 'nll', 'ece']
    metrics_tracker = OnlineMetricsTracker(all_scalars, block_size_growth=float(config['online_ess']['block_size_growth']))
    ensemble = CumulativeEnsemble()
    
    # Burn-in phase
    logger.info(f"Burn-in phase ({config['burn_in_steps']} steps)")
    initial_step_size = float(config['step_size_schedule']['initial'])
    decay_rate = float(config['step_size_schedule']['decay_rate'])
    decay_steps = int(config['step_size_schedule']['decay_steps'])
    
    for step in tqdm(range(int(config['burn_in_steps'])), desc="Burn-in"):
        # Update step size according to schedule
        current_step_size = initial_step_size * (1 + step / decay_steps) ** (-decay_rate)
        sampler.step_size = current_step_size
        
        # Get batch and update
        batch = next(iter(train_dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # SAM-SGLD update
        loss = sampler.step(input_ids, attention_mask, labels)
        
        if step % 1000 == 0:
            logger.info(f"Burn-in step {step}: loss={loss:.4f}, step_size={current_step_size:.6f}")
    
    logger.info("Burn-in completed!")
    
    # Set constant step size and SAM radius for sampling
    sampler.step_size = float(config['sampling_step_size'])
    sampler.rho = float(config['rho'])  # Constant SAM radius
    
    logger.info(f"Sampling phase: {config['sampling_steps']} steps, "
               f"step_size={config['sampling_step_size']}, rho={config['rho']}")
    
    # Sampling phase
    kept_samples = 0
    milestone_idx = 0
    milestones = config['milestones']
    
    for step in tqdm(range(int(config['sampling_steps'])), desc="Sampling"):
        # Get batch and update
        batch = next(iter(train_dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # SAM-SGLD update
        loss = sampler.step(input_ids, attention_mask, labels)
        
        # Check if we should save a sample
        if step % int(config['thinning']) == 0:
            kept_samples += 1
            
            # Compute per-sample metrics
            metrics = compute_per_sample_metrics(model, val_dataloader, device)
            
            # Update online estimators
            metrics_tracker.update({
                'accuracy': metrics['accuracy'],
                'nll': metrics['nll'],
                'ece': metrics['ece']
            })
            
            # Update ensemble
            ensemble.update(metrics['logits'], metrics['labels'])
            
            # Extract diagnostic scalars
            scalars = extract_diagnostic_scalars(model, sampler)
            metrics_tracker.update(scalars)
            
            # Log milestone summaries
            if milestone_idx < len(milestones) and step >= milestones[milestone_idx]:
                logger.info(f"\n=== MILESTONE {milestone_idx + 1}: {step} sampling steps ===")
                logger.info(f"Kept samples: {kept_samples}")
                
                # Get summary statistics
                summary = metrics_tracker.get_summary(config['thinning'])
                
                logger.info("ESS Summary:")
                for scalar, stats in summary.items():
                    logger.info(f"  {scalar}: ESS={stats['ess']:.2f}, "
                              f"ESS/step={stats['ess_per_step']:.6f}, "
                              f"IACT={stats['iact']:.2f}")
                
                logger.info("Metric Summary:")
                for metric in ['accuracy', 'nll', 'ece']:
                    if metric in summary:
                        stats = summary[metric]
                        logger.info(f"  {metric}: mean={stats['mean']:.4f}±{stats['std']:.4f}, "
                                  f"MCSE={stats['mcse']:.4f}")
                
                # Ensemble metrics
                if ensemble.count > 0:
                    ensemble_probs = ensemble.get_ensemble_probs()
                    ensemble_preds = torch.argmax(ensemble_probs, dim=-1)
                    ensemble_acc = (ensemble_preds == ensemble.labels).float().mean().item()
                    logger.info(f"  Ensemble accuracy: {ensemble_acc:.4f}")
                
                milestone_idx += 1
    
    logger.info(f"Sampling completed! Total kept samples: {kept_samples}")
    
    # Final summary
    final_summary = metrics_tracker.get_summary(config['thinning'])
    logger.info("\n=== FINAL SUMMARY ===")
    for scalar, stats in final_summary.items():
        logger.info(f"{scalar}: ESS={stats['ess']:.2f}, ESS/step={stats['ess_per_step']:.6f}")
    
    return model, final_summary


def main():
    parser = argparse.ArgumentParser(description="Train MRPC LoRA with online SAM-SGLD")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = args.output_dir or config['output']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "training.log")
    logger = setup_logging(config['output']['logging']['level'], log_file)
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Output directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup model and data
    model, tokenizer = setup_model_and_tokenizer(config)
    model = model.to(device)  # Move model to device
    
    train_dataset = MRPCDataset(split='train', tokenizer=tokenizer, max_length=config['model']['max_sequence_length'])
    val_dataset = MRPCDataset(split='validation', tokenizer=tokenizer, max_length=config['model']['max_sequence_length'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    # Train MAP model (if max_epochs > 0)
    if config['data']['max_epochs'] > 0:
        logger.info("Starting MAP training...")
        map_model = train_map_lora(model, train_dataloader, val_dataloader, 
                                  config, device, logger)
        
        # Save MAP model
        map_path = os.path.join(output_dir, "map_model.pth")
        torch.save(map_model.state_dict(), map_path)
        logger.info(f"MAP model saved to {map_path}")
    else:
        logger.info("Skipping MAP training (max_epochs = 0)")
        map_model = model  # Use untrained model
    
    # Run SAM-SGLD sampling
    final_model, summary = run_samsgld_sampling(map_model, train_dataloader, val_dataloader,
                                               config['training']['samsgld_rank1_lora'], device, logger)
    
    # Save final model and summary
    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save(final_model.state_dict(), final_path)
    
    summary_path = os.path.join(output_dir, "online_summary.yaml")
    # Convert numpy scalars to Python types for YAML serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    summary_converted = convert_numpy_types(summary)
    with open(summary_path, 'w') as f:
        yaml.dump(summary_converted, f)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
