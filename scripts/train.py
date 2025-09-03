#!/usr/bin/env python3
"""
Training script for CIFAR experiments with SGLD variants.
Supports SGLD, ASGLD, SAM-SGLD, and SAM-SGLD Rank-1 samplers.
"""

import argparse
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from bayesian_lora.data.cifar import get_cifar_dataset
from bayesian_lora.models.resnet_cifar import ResNetCIFAR
from bayesian_lora.models.wide_resnet import WideResNetCIFAR
from bayesian_lora.samplers.sgld import SGLDSampler, ASGLDSampler, SAMSGLDSampler, SAMSGLDRank1Sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model(config: Dict[str, Any], device: torch.device):
    """Initialize model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] == 'resnet18_cifar':
        model = ResNetCIFAR(
            depth=18,
            num_classes=model_config['num_classes']
        )
    elif model_config['name'] == 'wrn_28_10_cifar':
        model = WideResNetCIFAR(
            depth=model_config['depth'],
            widen_factor=model_config['widen_factor'],
            num_classes=model_config['num_classes']
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    model.to(device)
    return model


def setup_data(config: Dict[str, Any]):
    """Setup data loaders."""
    data_config = config['data']
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Get datasets
    train_dataset, test_dataset = get_cifar_dataset(
        name=data_config['name'],
        root=data_config['root'],
        transform_train=transform_train,
        transform_test=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers']
    )
    
    return train_loader, test_loader


def pretrain_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                  config: Dict[str, Any], device: torch.device):
    """Pretrain model if specified in config."""
    if config['train']['epochs'] == 0:
        logger.info("No pretraining specified, skipping...")
        return model
    
    logger.info(f"Starting pretraining for {config['train']['epochs']} epochs...")
    
    # Setup optimizer
    if config['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['train']['optimizer']}")
    
    # Setup scheduler
    if config['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['train']['epochs']
        )
    else:
        scheduler = None
    
    # Training loop
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        if scheduler:
            scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, "
                   f"Test Accuracy: {accuracy:.2f}%")
        
        model.train()
    
    logger.info("Pretraining completed!")
    return model


def setup_sampler(model: nn.Module, config: Dict[str, Any], device: torch.device):
    """Setup SGLD sampler based on configuration."""
    sampler_config = config['sampler']
    sampler_type = sampler_config['type']
    
    if sampler_type == 'sgld':
        sampler = SGLDSampler(
            model=model,
            temperature=sampler_config['temperature'],
            step_size=sampler_config['step_size'],
            noise_scale=sampler_config['noise_scale']
        )
    elif sampler_type == 'asgld':
        sampler = ASGLDSampler(
            model=model,
            temperature=sampler_config['temperature'],
            step_size=sampler_config['step_size'],
            noise_scale=sampler_config['noise_scale']
        )
    elif sampler_type == 'sam-sgld':
        sampler = SAMSGLDSampler(
            model=model,
            temperature=sampler_config['temperature'],
            step_size=sampler_config['step_size'],
            noise_scale=sampler_config['noise_scale'],
            rho=sampler_config['rho'],
            beta1=sampler_config['beta1'],
            beta2=sampler_config['beta2']
        )
    elif sampler_type == 'sam-sgld-r1':
        sampler = SAMSGLDRank1Sampler(
            model=model,
            temperature=sampler_config['temperature'],
            step_size=sampler_config['step_size'],
            noise_scale=sampler_config['noise_scale'],
            rho=sampler_config['rho'],
            beta1=sampler_config['beta1'],
            beta2=sampler_config['beta2']
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    return sampler


def run_sampling(model: nn.Module, sampler, train_loader: DataLoader, config: Dict[str, Any],
                device: torch.device, output_dir: Path):
    """Run SGLD sampling."""
    sampler_config = config['sampler']
    
    logger.info(f"Starting {sampler_config['type'].upper()} sampling...")
    logger.info(f"Burn-in steps: {sampler_config['burn_in']}")
    logger.info(f"Sampling steps: {sampler_config['num_samples'] * sampler_config['thin']}")
    logger.info(f"Thinning: {sampler_config['thin']}")
    
    # Burn-in phase
    logger.info("Phase 1: Burn-in...")
    for step in range(sampler_config['burn_in']):
        batch = next(iter(train_loader))
        data, target = batch[0].to(device), batch[1].to(device)
        
        sampler.step(data, target)
        
        if step % 500 == 0:
            logger.info(f"Burn-in step {step}/{sampler_config['burn_in']}")
    
    # Sampling phase
    logger.info("Phase 2: Sampling...")
    samples = []
    for step in range(sampler_config['num_samples'] * sampler_config['thin']):
        batch = next(iter(train_loader))
        data, target = batch[0].to(device), batch[1].to(device)
        
        sampler.step(data, target)
        
        # Save sample based on thinning
        if step % sampler_config['thin'] == 0:
            sample_state = sampler.get_current_state()
            samples.append(sample_state)
            
            # Save individual sample
            sample_path = output_dir / f"sample_{len(samples):04d}.pth"
            torch.save(sample_state, sample_path)
            
            logger.info(f"Saved sample {len(samples)}/{sampler_config['num_samples']}")
    
    logger.info(f"Sampling completed! {len(samples)} samples saved to {output_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR models with SGLD variants")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['out']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Setup model
    model = setup_model(config, device)
    
    # Setup data
    train_loader, test_loader = setup_data(config)
    
    # Pretrain if specified
    model = pretrain_model(model, train_loader, test_loader, config, device)
    
    # Setup sampler
    sampler = setup_sampler(model, config, device)
    
    # Run sampling
    samples = run_sampling(model, sampler, train_loader, config, device, output_dir)

    # Save manifest
    manifest = {
        'config': config,
        'num_samples': len(samples),
        'device': str(device),
        'sampler_type': config['sampler']['type']
    }
    
    manifest_path = output_dir / "manifest.json"
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
