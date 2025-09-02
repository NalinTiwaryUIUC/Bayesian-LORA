#!/usr/bin/env python
"""
Evaluation script for HuggingFace LoRA models with Bayesian sampling.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import numpy as np

from bayesian_lora.data.glue_datasets import create_dataloaders, get_dataset_metadata, evaluate_predictions
from bayesian_lora.models.hf_lora import build_huggingface_lora_model
from bayesian_lora.utils.lora_params import get_lora_parameters, count_lora_parameters

def evaluate_lora_sample(model: nn.Module, sample: dict, val_loader, device: str, dataset_name: str) -> tuple:
    """
    Evaluate a single LoRA sample.
    
    Args:
        model: LoRA model
        sample: Sample dictionary containing model state dict
        val_loader: Validation data loader
        device: Device to run evaluation on
        dataset_name: Name of the dataset for proper evaluation
        
    Returns:
        (accuracy, nll, metrics) tuple
    """
    # Load model state from sample
    model.load_state_dict(sample["model_state_dict"])
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            
            # Get predictions
            if labels.dim() == 1:  # Classification
                _, predicted = logits.max(1)
            else:  # Regression
                predicted = logits.squeeze()
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Convert to numpy for evaluation
    all_predictions = all_predictions.numpy()
    all_labels = all_labels.numpy()
    
    # Compute accuracy
    if labels.dim() == 1:  # Classification
        accuracy = (all_predictions == all_labels).mean()
    else:  # Regression
        accuracy = ((all_predictions - all_labels).abs() < 0.5).mean()
    
    # Compute NLL
    nll = total_loss / len(all_labels)
    
    # Compute dataset-specific metrics
    metric_results = evaluate_predictions(dataset_name, all_predictions, all_labels)
    
    return accuracy, nll, metric_results

def evaluate_lora_ensemble(model: nn.Module, samples: list, val_loader, device: str, dataset_name: str) -> tuple:
    """
    Evaluate an ensemble of LoRA samples.
    
    Args:
        model: LoRA model
        samples: List of sample dictionaries
        val_loader: Validation data loader
        device: Device to run evaluation on
        dataset_name: Name of the dataset for proper evaluation
        
    Returns:
        (accuracy, nll, metric_results) tuple
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Ensemble prediction
            ensemble_logits = 0.0
            for sample in samples:
                model.load_state_dict(sample["model_state_dict"])
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ensemble_logits += outputs.logits
            
            ensemble_logits /= len(samples)
            
            # Compute metrics
            if labels.dim() == 1:  # Classification
                _, predicted = ensemble_logits.max(1)
            else:  # Regression
                predicted = ensemble_logits.squeeze()
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
            
            loss = criterion(ensemble_logits, labels)
            total_loss += loss.item()
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Convert to numpy for evaluation
    all_predictions = all_predictions.numpy()
    all_labels = all_labels.numpy()
    
    # Compute accuracy
    if labels.dim() == 1:  # Classification
        accuracy = (all_predictions == all_labels).mean()
    else:  # Regression
        accuracy = ((all_predictions - all_labels).abs() < 0.5).mean()
    
    # Compute NLL
    nll = total_loss / len(all_labels)
    
    # Compute dataset-specific metrics
    metric_results = evaluate_predictions(dataset_name, all_predictions, all_labels)
    
    return accuracy, nll, metric_results

def main(config_path: str, samples_dir: str = None, k: int = 20, single: bool = False):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get dataset metadata
    dataset_name = config["data"]["name"]
    dataset_metadata = get_dataset_metadata(dataset_name)
    
    print(f"Dataset: {dataset_metadata['name']}")
    print(f"Task: {dataset_metadata['task']}")
    print(f"Number of labels: {dataset_metadata['num_labels']}")
    print()

    # Build model
    model_config = {
        'name': config["model"]["name"],
        'num_labels': dataset_metadata['num_labels'],
        'lora': config["model"]["lora"]
    }
    
    print(f"Building {config['model']['name']} with LoRA...")
    model = build_huggingface_lora_model(model_config).to(device)
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Print LoRA information
    lora_param_count = count_lora_parameters(model)
    print(f"LoRA Model Info:")
    print(f"  - LoRA parameters: {lora_param_count:,}")
    print()

    # Create validation dataloader
    _, val_loader = create_dataloaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"].get("max_length", None)
    )

    # Determine sample paths
    if samples_dir is None:
        samples_dir = config["out"]["dir"]
    paths = sorted(Path(samples_dir).glob("sample_*.pth"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No samples found under {samples_dir}")

    print(f"Found {len(paths)} LoRA samples in {samples_dir}")
    print()

    if single:
        # Evaluate just the first sample
        print("Evaluating single LoRA sample...")
        sample_0 = torch.load(paths[0], map_location=device)
        acc, nll, metrics = evaluate_lora_sample(model, sample_0, val_loader, device, dataset_name)
        print(f"Single LoRA sample - Acc: {acc:.3%} | NLL: {nll:.4f}")
        
        # Print dataset-specific metrics
        print(f"Dataset metrics: {metrics}")
        return

    # Ensemble evaluation
    if k is not None and k > 0:
        paths = paths[:k]
    
    print(f"Evaluating ensemble of {len(paths)} LoRA samples...")
    samples = [torch.load(p, map_location=device) for p in paths]
    
    acc, nll, metrics = evaluate_lora_ensemble(model, samples, val_loader, device, dataset_name)
    print(f"LoRA Ensemble@{len(samples)} — Acc: {acc:.3%} | NLL: {nll:.4f}")
    print(f"Dataset metrics: {metrics}")
    
    # Additional LoRA analysis
    print(f"\nLoRA Analysis:")
    print(f"  - Number of samples: {len(samples)}")
    print(f"  - Ensemble accuracy: {acc:.3%}")
    
    # Check if samples have sample index info
    if "sample_idx" in samples[0]:
        sample_indices = [s["sample_idx"] for s in samples]
        print(f"  - Sample range: {min(sample_indices)} - {max(sample_indices)}")
    
    # Analyze LoRA parameter statistics across samples
    if len(samples) > 1:
        print(f"\nLoRA Parameter Statistics:")
        
        # Get LoRA parameters from first sample for analysis
        model.load_state_dict(samples[0]["model_state_dict"])
        lora_params = get_lora_parameters(model)
        
        # Compute parameter statistics across samples
        param_norms = []
        for sample in samples:
            model.load_state_dict(sample["model_state_dict"])
            lora_params = get_lora_parameters(model)
            param_norm = sum(p.norm().item() for p in lora_params)
            param_norms.append(param_norm)
        
        param_norms = np.array(param_norms)
        print(f"  - Parameter norm mean: {param_norms.mean():.4f}")
        print(f"  - Parameter norm std: {param_norms.std():.4f}")
        print(f"  - Parameter norm CV: {param_norms.std() / param_norms.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace LoRA models with Bayesian sampling")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--samples_dir", type=str, default=None, help="Directory containing LoRA samples")
    parser.add_argument("--k", type=int, default=20, help="Use first k samples for ensemble")
    parser.add_argument("--single", action="store_true", help="Evaluate only the first sample")
    
    args = parser.parse_args()
    main(args.config, args.samples_dir, args.k, args.single)
