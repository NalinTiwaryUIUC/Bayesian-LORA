#!/usr/bin/env python
"""
Training script for HuggingFace LoRA models with Bayesian sampling.
"""

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from bayesian_lora.models.hf_lora import build_huggingface_lora_model
from bayesian_lora.data.glue_datasets import create_dataloaders, get_dataset_metadata
from bayesian_lora.samplers.sgld import (
    sgld_step, asgld_step, sam_sgld_step, sam_sgld_rank_1_step
)
from bayesian_lora.utils.lora_params import get_lora_parameters, count_lora_parameters

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            if labels.dim() == 1:  # Classification
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
            else:  # Regression
                pred = logits.squeeze()
                correct += ((pred - labels).abs() < 0.5).sum().item()
            
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def train_epoch(model, train_loader, optimizer, device, sampler_name, sampler_params):
    """Train for one epoch using specified sampler."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Training ({sampler_name})"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply sampler-specific update
        lora_params = get_lora_parameters(model)
        
        if sampler_name == 'sgld':
            sgld_step(lora_params, sampler_params['step_size'])
        elif sampler_name == 'asgld':
            asgld_step(lora_params, sampler_params['step_size'], sampler_params.get('momentum', 0.9))
        elif sampler_name == 'sam_sgld':
            sam_sgld_step(lora_params, sampler_params['step_size'], sampler_params.get('rho', 0.1))
        elif sampler_name == 'sam_sgld_r1':
            sam_sgld_rank_1_step(lora_params, sampler_params['step_size'], sampler_params.get('rho', 0.1))
        else:
            # Standard SGD update
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['out']['dir'], exist_ok=True)
    
    # Get dataset metadata
    dataset_name = config['data']['name']
    dataset_metadata = get_dataset_metadata(dataset_name)
    num_labels = dataset_metadata['num_labels']
    
    print(f"Dataset: {dataset_metadata['name']}")
    print(f"Task: {dataset_metadata['task']}")
    print(f"Number of labels: {num_labels}")
    
    # Build model
    model_config = {
        'name': config['model']['name'],
        'num_labels': num_labels,
        'lora': config['model']['lora']
    }
    
    model = build_huggingface_lora_model(model_config)
    model = model.to(device)
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=config['data']['batch_size'],
        max_length=config['data'].get('max_length', None)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initial evaluation
    print("\nInitial evaluation:")
    initial_loss, initial_acc = evaluate_model(model, val_loader, device)
    print(f"Initial loss: {initial_loss:.4f}, accuracy: {initial_acc:.4f}")
    
    # Training loop
    sampler_config = config['sampler']
    sampler_name = sampler_config['name']
    num_epochs = sampler_config.get('num_epochs', 10)
    
    print(f"\nTraining with {sampler_name} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, None, device, sampler_name, sampler_config
        )
        
        # Evaluate
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
    
    # Sampling phase
    print(f"\nSampling with {sampler_name}...")
    
    burn_in = sampler_config.get('burn_in', 1000)
    thin = sampler_config.get('thin', 200)
    num_samples = sampler_config.get('num_samples', 20)
    
    # Burn-in phase
    print(f"Burn-in phase: {burn_in} steps")
    for step in tqdm(range(burn_in), desc="Burn-in"):
        # Sample one batch for burn-in
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        model.zero_grad()
        loss.backward()
        
        # Apply sampler step
        lora_params = get_lora_parameters(model)
        if sampler_name == 'sgld':
            sgld_step(lora_params, sampler_config['step_size'])
        elif sampler_name == 'asgld':
            asgld_step(lora_params, sampler_config['step_size'], sampler_config.get('momentum', 0.9))
        elif sampler_name == 'sam_sgld':
            sam_sgld_step(lora_params, sampler_config['step_size'], sampler_config.get('rho', 0.1))
        elif sampler_name == 'sam_sgld_r1':
            sam_sgld_rank_1_step(lora_params, sampler_config['step_size'], sampler_config.get('rho', 0.1))
    
    # Collect samples
    print(f"Collecting {num_samples} samples (thinning: {thin})")
    samples = []
    
    for sample_idx in range(num_samples):
        # Thinning phase
        for step in range(thin):
            batch = next(iter(train_loader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            model.zero_grad()
            loss.backward()
            
            # Apply sampler step
            lora_params = get_lora_parameters(model)
            if sampler_name == 'sgld':
                sgld_step(lora_params, sampler_config['step_size'])
            elif sampler_name == 'asgld':
                asgld_step(lora_params, sampler_config['step_size'], sampler_config.get('momentum', 0.9))
            elif sampler_name == 'sam_sgld':
                sam_sgld_step(lora_params, sampler_config['step_size'], sampler_config.get('rho', 0.1))
            elif sampler_name == 'sam_sgld_r1':
                sam_sgld_rank_1_step(lora_params, sampler_config['step_size'], sampler_config.get('rho', 0.1))
        
        # Save sample
        sample_path = os.path.join(config['out']['dir'], f'sample_{sample_idx+1:04d}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'sample_idx': sample_idx,
            'sampler': sampler_name,
            'config': config
        }, sample_path)
        
        samples.append(sample_path)
        print(f"Saved sample {sample_idx+1}/{num_samples}: {sample_path}")
    
    print(f"\nTraining and sampling completed!")
    print(f"Saved {len(samples)} samples to {config['out']['dir']}")
    print(f"LoRA parameters: {count_lora_parameters(model):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HuggingFace LoRA model with Bayesian sampling")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    
    args = parser.parse_args()
    main(args.config)