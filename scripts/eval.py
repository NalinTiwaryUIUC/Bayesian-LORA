#!/usr/bin/env python3
"""
Evaluation script for CIFAR experiments with SGLD variants.
Supports single sample and ensemble evaluation.
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
from torchvision import transforms

from bayesian_lora.data.cifar import get_cifar_dataset
from bayesian_lora.models.resnet_cifar import ResNetCIFAR
from bayesian_lora.models.wide_resnet import WideResNetCIFAR

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
    
    # Data augmentation for test (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Get test dataset only
    _, test_dataset = get_cifar_dataset(
        name=data_config['name'],
        root=data_config['root'],
        transform_train=None,  # Not needed for evaluation
        transform_test=transform_test
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers']
    )
    
    return test_loader


def evaluate_single_sample(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    """Evaluate a single model sample."""
    model.eval()
    all_predictions = []
    all_confidences = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
            
            probs = F.softmax(output, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    nll = total_loss / len(test_loader)
    ece = compute_ece(all_confidences, all_predictions == all_labels)
    
    return {
        'accuracy': accuracy,
        'nll': nll,
        'ece': ece,
        'predictions': all_predictions,
        'confidences': all_confidences,
        'labels': all_labels
    }


def evaluate_ensemble(model: nn.Module, samples: List[Dict], test_loader: torch.utils.data.DataLoader,
                     device: torch.device, k: int = None):
    """Evaluate ensemble of model samples."""
    if k is not None:
        samples = samples[:k]
        logger.info(f"Using first {k} samples for ensemble")
    
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_logits = []
    
    # Collect all labels first
    for data, target in test_loader:
        all_labels.extend(target.cpu().numpy())
    
    all_labels = np.array(all_labels)
    
    # Evaluate each sample
    for i, sample_state in enumerate(samples):
        logger.info(f"Evaluating sample {i+1}/{len(samples)}")
        
        # Load sample state
        model.load_state_dict(sample_state)
        model.eval()
        
        sample_predictions = []
        sample_confidences = []
        sample_logits = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                
                output = model(data)
                probs = F.softmax(output, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                sample_predictions.extend(predictions.cpu().numpy())
                sample_confidences.extend(confidences.cpu().numpy())
                sample_logits.extend(output.cpu().numpy())
        
        all_predictions.append(sample_predictions)
        all_confidences.append(sample_confidences)
        all_logits.append(sample_logits)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)  # [n_samples, n_examples]
    all_confidences = np.array(all_confidences)  # [n_samples, n_examples]
    all_logits = np.array(all_logits)  # [n_samples, n_examples, n_classes]
    
    # Ensemble predictions
    ensemble_logits = np.mean(all_logits, axis=0)  # [n_examples, n_classes]
    ensemble_probs = F.softmax(torch.tensor(ensemble_logits), dim=1).numpy()
    ensemble_confidences = np.max(ensemble_probs, axis=1)
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)
    
    # Compute ensemble metrics
    accuracy = accuracy_score(all_labels, ensemble_predictions)
    
    # Compute ensemble NLL
    nll = log_loss(all_labels, ensemble_probs)
    
    # Compute ensemble ECE
    ece = compute_ece(ensemble_confidences, ensemble_predictions == all_labels)
    
    return {
        'accuracy': accuracy,
        'nll': nll,
        'ece': ece,
        'predictions': ensemble_predictions,
        'confidences': ensemble_confidences,
        'labels': all_labels,
        'num_samples': len(samples)
    }


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)
    
    return ece


def load_samples(output_dir: Path) -> List[Dict]:
    """Load all samples from output directory."""
    samples = []
    
    # Look for sample files
    sample_files = sorted(output_dir.glob("sample_*.pth"))
    
    for sample_file in sample_files:
        sample_state = torch.load(sample_file, map_location='cpu')
        samples.append(sample_state)
    
    logger.info(f"Loaded {len(samples)} samples from {output_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR experiments with SGLD variants")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory containing samples")
    parser.add_argument("--single", action="store_true",
                       help="Evaluate single sample (first sample)")
    parser.add_argument("--k", type=int, default=None,
                       help="Number of samples to use for ensemble (default: all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Setup model
    model = setup_model(config, device)
    
    # Setup data
    test_loader = setup_data(config)
    
    # Load samples
    output_dir = Path(args.output_dir)
    samples = load_samples(output_dir)
    
    if not samples:
        logger.error(f"No samples found in {output_dir}")
        return

    # Evaluate based on mode
    if args.single:
        logger.info("Evaluating single sample...")
        # Load first sample
        model.load_state_dict(samples[0])
        results = evaluate_single_sample(model, test_loader, device)
        
        logger.info("Single Sample Results:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"NLL: {results['nll']:.4f}")
        logger.info(f"ECE: {results['ece']:.4f}")
        
    else:
        logger.info("Evaluating ensemble...")
        results = evaluate_ensemble(model, samples, test_loader, device, args.k)
        
        logger.info("Ensemble Results:")
        logger.info(f"Number of samples: {results['num_samples']}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"NLL: {results['nll']:.4f}")
        logger.info(f"ECE: {results['ece']:.4f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
