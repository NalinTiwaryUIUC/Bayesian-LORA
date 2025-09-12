#!/usr/bin/env python3
"""
Systematic SAM-SGLD Diagnostics
Tests key aspects of SAM-SGLD to identify why it's not exploring properly.
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import logging

from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.data.glue_datasets import MRPCDataset
from bayesian_lora.samplers.sgld import SAMSGLDRank1Sampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def setup_logging():
    """Setup logging for diagnostics."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_config():
    """Load the SAM-SGLD configuration."""
    config_path = 'configs/mrpc_roberta_lora_samsgld_rank1.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def create_model_and_data(config, device):
    """Create model and data for testing."""
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['backbone'])
    
    # Create base model
    base_model = RobertaForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2
    )
    
    # Apply LoRA
    lora_config = config['model']['lora']
    model = LoRAModel(
        base_model,
        r=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=lora_config['inject_into']
    )
    
    # Create dataset
    try:
        dataset = MRPCDataset("train", tokenizer, max_length=config['model']['max_sequence_length'])
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['data']['batch_size'], shuffle=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {e}")
    
    return model, train_dataloader, tokenizer

def train_map_model(model, dataloader, config, device, logger):
    """Train MAP model for diagnostics."""
    logger.info("=== Training MAP Model for Diagnostics ===")
    
    model.to(device)
    model.train()
    
    # Setup optimizer
    map_config = config['training']['map_lora']
    if map_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(map_config['learning_rate']),
            weight_decay=float(map_config['weight_decay']),
            momentum=float(map_config['momentum'])
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(map_config['learning_rate']),
            weight_decay=float(map_config['weight_decay'])
        )
    
    # Training loop
    num_epochs = min(3, config['data']['max_epochs'])  # Quick training for diagnostics
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Only train on a few batches for diagnostics
            if num_batches >= 10:
                break
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")
    
    logger.info("MAP model training completed")
    return model

def test_sam_perturbation(model, dataloader, config, device, logger):
    """Test if SAM perturbation is working correctly."""
    logger.info("=== Testing SAM Perturbation ===")
    
    model.to(device)
    model.train()
    
    # Get a batch
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Get initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            initial_params[name] = param.data.clone()
    
    # Test SAM perturbation
    rho = config['training']['samsgld_rank1_lora']['rho']
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, labels)
    
    # Compute gradients
    loss.backward()
    
    # Get gradient norm for LoRA parameters only
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower() and param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    logger.info(f"Initial loss: {loss.item():.4f}")
    logger.info(f"Gradient norm: {grad_norm:.4f}")
    logger.info(f"SAM rho: {rho:.3f}")
    
    # Compute SAM perturbation
    # Normalize gradients for LoRA parameters only
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower() and param.grad is not None:
            param.grad.data = param.grad.data / grad_norm
    
    # Apply perturbation
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower() and param.grad is not None:
            param.data = param.data + rho * param.grad.data
    
    # Check perturbation norm
    perturbation_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            perturbation_norm += (param.data - initial_params[name]).norm().item() ** 2
    perturbation_norm = perturbation_norm ** 0.5
    
    logger.info(f"Actual perturbation norm: {perturbation_norm:.4f}")
    logger.info(f"Expected perturbation norm: {rho:.3f}")
    logger.info(f"Perturbation ratio: {perturbation_norm / rho:.3f}")
    
    # Test perturbed loss
    model.zero_grad()
    outputs_pert = model(input_ids, attention_mask=attention_mask)
    loss_pert = F.cross_entropy(outputs_pert.logits, labels)
    
    logger.info(f"Perturbed loss: {loss_pert.item():.4f}")
    logger.info(f"Loss change: {loss_pert.item() - loss.item():.4f}")
    
    # Restore parameters
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            param.data = initial_params[name]
    
    return {
        'grad_norm': grad_norm,
        'perturbation_norm': perturbation_norm,
        'expected_perturbation': rho,
        'perturbation_ratio': perturbation_norm / rho,
        'loss_change': loss_pert.item() - loss.item()
    }

def test_sgld_step(model, dataloader, config, device, logger):
    """Test SGLD step to see actual movement."""
    logger.info("=== Testing SGLD Step ===")
    
    model.to(device)
    model.train()
    
    # Create sampler
    sgld_config = config['training']['samsgld_rank1_lora']
    sampler = SAMSGLDRank1Sampler(
        model=model,
        temperature=float(sgld_config['temperature']),
        step_size=float(sgld_config['learning_rate']),
        noise_scale=float(sgld_config['noise_scale']),
        rho=float(sgld_config['rho']),
        lambd=float(sgld_config['lambd']),
        sigma_dir=float(sgld_config['sigma_dir']),
        gradient_clip_norm=float(sgld_config['gradient_clip_norm']),
        prior_std=float(sgld_config['prior_std'])
    )
    
    # Get initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            initial_params[name] = param.data.clone()
    
    # Get a batch
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Calculate prior and likelihood terms before step
    with torch.no_grad():
        # Prior term
        prior_loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                prior_loss += (param ** 2).sum() / (2 * float(sgld_config['prior_std']) ** 2)
        
        # Likelihood term
        outputs = model(input_ids, attention_mask=attention_mask)
        likelihood_loss = F.cross_entropy(outputs.logits, labels)
        
        logger.info(f"Prior loss: {prior_loss.item():.6f}")
        logger.info(f"Likelihood loss: {likelihood_loss.item():.6f}")
        logger.info(f"Prior/Likelihood ratio: {prior_loss.item() / likelihood_loss.item():.3f}")
    
    # Take SGLD step
    sampler.step(input_ids, attention_mask, labels)
    
    # Check parameter movement
    total_movement = 0.0
    max_movement = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            movement = (param.data - initial_params[name]).norm().item()
            total_movement += movement ** 2
            max_movement = max(max_movement, movement)
            param_count += 1
    
    total_movement = total_movement ** 0.5
    avg_movement = total_movement / param_count if param_count > 0 else 0.0
    
    logger.info(f"Total parameter movement: {total_movement:.6f}")
    logger.info(f"Average parameter movement: {avg_movement:.6f}")
    logger.info(f"Max parameter movement: {max_movement:.6f}")
    logger.info(f"Parameters moved: {param_count}")
    
    # Check sampler diagnostics
    logger.info(f"Sampler drift norm: {sampler.last_drift_norm:.6f}")
    logger.info(f"Sampler noise norm: {sampler.last_noise_norm:.6f}")
    logger.info(f"Sampler step size: {sampler.last_step_size:.6f}")
    
    if sampler.last_noise_norm > 0:
        logger.info(f"Actual drift/noise ratio: {sampler.last_drift_norm / sampler.last_noise_norm:.3f}")
    
    return {
        'total_movement': total_movement,
        'avg_movement': avg_movement,
        'max_movement': max_movement,
        'drift_norm': sampler.last_drift_norm,
        'noise_norm': sampler.last_noise_norm,
        'step_size': sampler.last_step_size,
        'prior_loss': prior_loss.item(),
        'likelihood_loss': likelihood_loss.item(),
        'prior_likelihood_ratio': prior_loss.item() / likelihood_loss.item()
    }

def test_multiple_steps(model, dataloader, config, device, logger, num_steps=10):
    """Test multiple SGLD steps to see cumulative movement."""
    logger.info(f"=== Testing {num_steps} SGLD Steps ===")
    
    model.to(device)
    model.train()
    
    # Create sampler
    sgld_config = config['training']['samsgld_rank1_lora']
    sampler = SAMSGLDRank1Sampler(
        model=model,
        temperature=float(sgld_config['temperature']),
        step_size=float(sgld_config['learning_rate']),
        noise_scale=float(sgld_config['noise_scale']),
        rho=float(sgld_config['rho']),
        lambd=float(sgld_config['lambd']),
        sigma_dir=float(sgld_config['sigma_dir']),
        gradient_clip_norm=float(sgld_config['gradient_clip_norm']),
        prior_std=float(sgld_config['prior_std'])
    )
    
    # Get initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            initial_params[name] = param.data.clone()
    
    # Get a batch
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Track movement over steps
    movements = []
    drift_norms = []
    noise_norms = []
    prior_likelihood_ratios = []
    
    for step in range(num_steps):
        # Calculate prior/likelihood ratio before step
        with torch.no_grad():
            prior_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    prior_loss += (param ** 2).sum() / (2 * float(sgld_config['prior_std']) ** 2)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            likelihood_loss = F.cross_entropy(outputs.logits, labels)
            prior_likelihood_ratio = prior_loss.item() / likelihood_loss.item()
            prior_likelihood_ratios.append(prior_likelihood_ratio)
        
        # Take SGLD step
        sampler.step(input_ids, attention_mask, labels)
        
        # Calculate movement
        total_movement = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                movement = (param.data - initial_params[name]).norm().item()
                total_movement += movement ** 2
        total_movement = total_movement ** 0.5
        
        movements.append(total_movement)
        drift_norms.append(sampler.last_drift_norm)
        noise_norms.append(sampler.last_noise_norm)
        
        if step % 5 == 0:
            logger.info(f"Step {step}: Movement={total_movement:.6f}, Drift={sampler.last_drift_norm:.6f}, Noise={sampler.last_noise_norm:.6f}, Prior/Likelihood={prior_likelihood_ratio:.3f}")
    
    # Analyze movement pattern
    final_movement = movements[-1]
    movement_trend = np.polyfit(range(len(movements)), movements, 1)[0]
    
    logger.info(f"Final movement after {num_steps} steps: {final_movement:.6f}")
    logger.info(f"Movement trend (slope): {movement_trend:.6f}")
    logger.info(f"Average drift norm: {np.mean(drift_norms):.6f}")
    logger.info(f"Average noise norm: {np.mean(noise_norms):.6f}")
    logger.info(f"Average prior/likelihood ratio: {np.mean(prior_likelihood_ratios):.3f}")
    logger.info(f"Prior/likelihood ratio range: {np.min(prior_likelihood_ratios):.3f} - {np.max(prior_likelihood_ratios):.3f}")
    
    return {
        'final_movement': final_movement,
        'movement_trend': movement_trend,
        'avg_drift_norm': np.mean(drift_norms),
        'avg_noise_norm': np.mean(noise_norms),
        'avg_prior_likelihood_ratio': np.mean(prior_likelihood_ratios),
        'prior_likelihood_ratio_range': (np.min(prior_likelihood_ratios), np.max(prior_likelihood_ratios)),
        'movements': movements
    }

def main():
    """Run systematic SAM-SGLD diagnostics."""
    logger = setup_logging()
    logger.info("Starting systematic SAM-SGLD diagnostics...")
    
    try:
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = load_config()
        
        logger.info(f"Using device: {device}")
        logger.info(f"Configuration: {config['experiment']['name']}")
        
        # Create model and data
        model, dataloader, tokenizer = create_model_and_data(config, device)
        
        # Train MAP model first
        map_model = train_map_model(model, dataloader, config, device, logger)
        
        # Test 1: SAM perturbation
        sam_results = test_sam_perturbation(map_model, dataloader, config, device, logger)
        
        # Test 2: Single SGLD step
        sgld_results = test_sgld_step(map_model, dataloader, config, device, logger)
        
        # Test 3: Multiple SGLD steps
        multi_results = test_multiple_steps(map_model, dataloader, config, device, logger, num_steps=20)
        
        # Summary
        logger.info("=== DIAGNOSTIC SUMMARY ===")
        logger.info(f"SAM perturbation ratio: {sam_results['perturbation_ratio']:.3f} (should be ~1.0)")
        logger.info(f"SAM loss change: {sam_results['loss_change']:.4f} (should be non-zero)")
        logger.info(f"Single step movement: {sgld_results['total_movement']:.6f}")
        logger.info(f"20-step movement: {multi_results['final_movement']:.6f}")
        logger.info(f"Movement trend: {multi_results['movement_trend']:.6f} (should be positive)")
        logger.info(f"Avg drift/noise ratio: {multi_results['avg_drift_norm'] / multi_results['avg_noise_norm']:.3f}")
        logger.info(f"Prior/Likelihood ratio: {sgld_results['prior_likelihood_ratio']:.3f}")
        logger.info(f"Avg Prior/Likelihood ratio (20 steps): {multi_results['avg_prior_likelihood_ratio']:.3f}")
        logger.info(f"Prior/Likelihood ratio range: {multi_results['prior_likelihood_ratio_range'][0]:.3f} - {multi_results['prior_likelihood_ratio_range'][1]:.3f}")
        
        # Recommendations
        logger.info("=== RECOMMENDATIONS ===")
        if sam_results['perturbation_ratio'] < 0.5:
            logger.info("❌ SAM perturbation too small - increase rho")
        elif sam_results['perturbation_ratio'] > 2.0:
            logger.info("❌ SAM perturbation too large - decrease rho")
        else:
            logger.info("✅ SAM perturbation looks good")
        
        if multi_results['movement_trend'] < 1e-6:
            logger.info("❌ No cumulative movement - increase step size or noise scale")
        else:
            logger.info("✅ Cumulative movement detected")
        
        if multi_results['avg_drift_norm'] / multi_results['avg_noise_norm'] > 100:
            logger.info("❌ Drift dominates noise - increase noise scale")
        elif multi_results['avg_drift_norm'] / multi_results['avg_noise_norm'] < 0.1:
            logger.info("❌ Noise dominates drift - decrease noise scale")
        else:
            logger.info("✅ Drift/noise ratio looks balanced")
        
        # Prior/Likelihood recommendations
        if sgld_results['prior_likelihood_ratio'] > 1.0:
            logger.info("❌ Prior dominates likelihood - reduce prior_std")
        elif sgld_results['prior_likelihood_ratio'] < 0.1:
            logger.info("❌ Likelihood dominates prior - increase prior_std")
        else:
            logger.info("✅ Prior/Likelihood ratio looks balanced")
    
    except Exception as e:
        logger.error(f"Diagnostic failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
