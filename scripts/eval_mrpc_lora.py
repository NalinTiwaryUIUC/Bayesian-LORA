#!/usr/bin/env python3
"""
Evaluation script for MRPC RoBERTa LoRA SGLD experiment.
Computes accuracy, NLL, ECE, and MCMC diagnostics as described in the experiment outline.
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from scipy import stats
from transformers import RobertaTokenizer

from bayesian_lora.models.hf_lora import LoRAModel
from bayesian_lora.data.glue_datasets import MRPCDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    def convert_scientific_notation(loader, node):
        """Convert scientific notation strings to floats."""
        value = loader.construct_scalar(node)
        try:
            if any(char in value.lower() for char in '0123456789.e+-'):
                return float(value)
            else:
                return value
        except ValueError:
            return value
    
    # Create custom loader that handles scientific notation
    class ScientificNotationLoader(yaml.SafeLoader):
        pass
    
    # Register the converter for scalar nodes
    ScientificNotationLoader.add_constructor('tag:yaml.org,2002:str', convert_scientific_notation)
    
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=ScientificNotationLoader)


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


def compute_r_hat(chains: List[np.ndarray]) -> float:
    """Compute R-hat (Gelman-Rubin statistic) for MCMC diagnostics."""
    if len(chains) < 2:
        return 1.0
    
    # Convert to numpy arrays if needed
    chains = [np.array(chain) for chain in chains]
    
    # Compute within-chain variance
    chain_means = [np.mean(chain) for chain in chains]
    chain_vars = [np.var(chain, ddof=1) for chain in chains]
    
    n = len(chains[0])  # Assuming all chains have same length
    m = len(chains)
    
    # Within-chain variance
    W = np.mean(chain_vars)
    
    # Between-chain variance
    overall_mean = np.mean(chain_means)
    B = (n / (m - 1)) * np.sum((chain_means - overall_mean) ** 2)
    
    # Pooled variance
    V = ((n - 1) / n) * W + (1 / n) * B
    
    # R-hat
    r_hat = np.sqrt(V / W) if W > 0 else 1.0
    
    return r_hat


def compute_ess(chain: np.ndarray) -> int:
    """Compute Effective Sample Size using autocorrelation."""
    if len(chain) < 2:
        return len(chain)
    
    # Compute autocorrelation
    acf = np.correlate(chain - np.mean(chain), chain - np.mean(chain), mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]  # Normalize
    
    # Find first crossing of 0.05 threshold
    threshold = 0.05
    cutoff = np.where(acf < threshold)[0]
    if len(cutoff) > 0:
        lag = cutoff[0]
    else:
        lag = len(acf) - 1
    
    # ESS = N / (1 + 2 * sum of autocorrelations)
    ess = len(chain) / (1 + 2 * np.sum(acf[1:lag+1]))
    
    return int(max(1, ess))


def evaluate_map_model(model: LoRAModel, dataloader: torch.utils.data.DataLoader, 
                      device: torch.device) -> Tuple[float, float, float]:
    """Evaluate MAP model and compute metrics."""
    model.eval()
    all_predictions = []
    all_confidences = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(outputs.logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = np.mean(all_predictions == all_labels)
    nll = total_loss / len(dataloader)
    ece = compute_ece(all_confidences, all_predictions == all_labels)
    
    return accuracy, nll, ece


def evaluate_sgld_samples(model: LoRAModel, samples: List[Dict], dataloader: torch.utils.data.DataLoader,
                         device: torch.device, num_chains: int, samples_per_chain: int) -> Tuple[float, float, float, Dict]:
    """Evaluate SGLD samples and compute ensemble metrics."""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_logits = []
    
    # Collect all labels first
    for batch in dataloader:
        all_labels.extend(batch['labels'].cpu().numpy())
    
    all_labels = np.array(all_labels)
    
    # Evaluate each sample
    for i, sample_state in enumerate(samples):
        logger.info(f"Evaluating sample {i+1}/{len(samples)}")
        
        # Load sample state
        # Load LoRA-only weights; ignore missing base model keys
        model.load_state_dict(sample_state, strict=False)
        model.eval()
        
        sample_predictions = []
        sample_confidences = []
        sample_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                sample_predictions.extend(predictions.cpu().numpy())
                sample_confidences.extend(confidences.cpu().numpy())
                sample_logits.extend(outputs.logits.cpu().numpy())
        
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
    accuracy = np.mean(ensemble_predictions == all_labels)
    
    # Compute ensemble NLL
    nll = 0
    for i, label in enumerate(all_labels):
        nll -= np.log(ensemble_probs[i, label])
    nll /= len(all_labels)
    
    # Compute ensemble ECE
    ece = compute_ece(ensemble_confidences, ensemble_predictions == all_labels)
    
    # MCMC diagnostics
    mcmc_diagnostics = {}
    
    # Use probe set for diagnostics (first 512 examples)
    probe_size = min(512, len(all_labels))
    probe_logits = all_logits[:, :probe_size, :]
    
    # Summary 1: Log posterior (simplified as negative log-likelihood)
    # Collect per-sample scalar metric first, then partition by chains
    log_posterior_values = []
    for sample_logits in probe_logits:
        sample_probs = F.softmax(torch.tensor(sample_logits), dim=1).numpy()
        sample_nll = -np.mean([np.log(sample_probs[i, all_labels[i]]) 
                              for i in range(probe_size)])
        log_posterior_values.append(sample_nll)
    
    # Summary 2: L2 norm of LoRA parameters
    l2_norm_values = []
    for sample_state in samples:
        l2_norm = 0
        for param in sample_state.values():
            if isinstance(param, torch.Tensor):
                l2_norm += torch.norm(param).item() ** 2
        l2_norm_values.append(np.sqrt(l2_norm))

    # Partition metrics into chains
    def partition_into_chains(values: List[float], num_chains: int, samples_per_chain: int) -> List[np.ndarray]:
        chains = []
        for c in range(num_chains):
            start = c * samples_per_chain
            end = start + samples_per_chain
            chains.append(np.array(values[start:end]))
        return chains

    log_posterior_chains = partition_into_chains(log_posterior_values, num_chains, samples_per_chain)
    l2_norm_chains = partition_into_chains(l2_norm_values, num_chains, samples_per_chain)
    
    # Compute R-hat and ESS for each summary
    mcmc_diagnostics['r_hat_log_posterior'] = compute_r_hat(log_posterior_chains)
    mcmc_diagnostics['r_hat_l2_norm'] = compute_r_hat(l2_norm_chains)
    # Compute ESS per chain and report the minimum as conservative estimate
    mcmc_diagnostics['ess_log_posterior'] = min(compute_ess(chain) for chain in log_posterior_chains)
    mcmc_diagnostics['ess_l2_norm'] = min(compute_ess(chain) for chain in l2_norm_chains)
    
    return accuracy, nll, ece, mcmc_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MRPC LoRA with SGLD")
    parser.add_argument("--config", type=str, default="configs/mrpc_roberta_lora_sgld.yaml",
                       help="Path to configuration file")
    parser.add_argument("--map_model_path", type=str, required=True,
                       help="Path to MAP model checkpoint")
    parser.add_argument("--sgld_samples_path", type=str, required=True,
                       help="Path to SGLD samples checkpoint")
    parser.add_argument("--output_dir", type=str, default="runs/mrpc_roberta_lora_sgld",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MAP model
    logger.info("Loading MAP model...")
    
    # Create LoRA model with proper configuration
    from transformers import RobertaForSequenceClassification
    base_model = RobertaForSequenceClassification.from_pretrained(
        config['model']['backbone'],
        num_labels=2  # MRPC is binary classification
    )
    
    lora_config = config['model']['lora']
    map_model = LoRAModel(
        base_model,
        r=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        target_modules=lora_config['inject_into']
    )
    
    # Load the trained state
    map_model.load_state_dict(torch.load(args.map_model_path, map_location=device))
    map_model.to(device)
    
    # Load SGLD samples
    logger.info("Loading SGLD samples...")
    sgld_samples = torch.load(args.sgld_samples_path, map_location=device)
    
    # Setup datasets
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['backbone'])
    eval_dataset = MRPCDataset(split="validation", tokenizer=tokenizer,
                               max_length=config['model']['max_sequence_length'])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, 
                                                 batch_size=config['data']['batch_size'])
    
    # Evaluate MAP model
    logger.info("Evaluating MAP model...")
    map_accuracy, map_nll, map_ece = evaluate_map_model(map_model, eval_dataloader, device)
    
    # Determine chain partitioning from config
    num_chains = int(config['training']['sgld_lora']['chains'])
    samples_total = int(config['training']['sgld_lora']['samples_to_retain'])
    samples_per_chain = samples_total // num_chains
    sgld_accuracy, sgld_nll, sgld_ece, mcmc_diagnostics = evaluate_sgld_samples(
        map_model, sgld_samples, eval_dataloader, device, num_chains, samples_per_chain
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"{'Method':<15} {'Accuracy':<10} {'NLL':<10} {'ECE':<10}")
    logger.info("-" * 50)
    logger.info(f"{'MAP-LoRA':<15} {map_accuracy:<10.4f} {map_nll:<10.4f} {map_ece:<10.4f}")
    logger.info(f"{'SGLD-LoRA':<15} {sgld_accuracy:<10.4f} {sgld_nll:<10.4f} {sgld_ece:<10.4f}")
    logger.info("\nMCMC Diagnostics:")
    logger.info(f"R-hat (log posterior): {mcmc_diagnostics['r_hat_log_posterior']:.4f}")
    logger.info(f"R-hat (L2 norm): {mcmc_diagnostics['r_hat_l2_norm']:.4f}")
    logger.info(f"ESS (log posterior): {mcmc_diagnostics['ess_log_posterior']}")
    logger.info(f"ESS (L2 norm): {mcmc_diagnostics['ess_l2_norm']}")
    
    # Convert numpy values to Python native types for human-readable YAML
    def convert_numpy_to_python(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(item) for item in obj]
        else:
            return obj
    
    # Save results
    results = {
        'map_lora': {
            'accuracy': map_accuracy,
            'nll': map_nll,
            'ece': map_ece
        },
        'sgld_lora': {
            'accuracy': sgld_accuracy,
            'nll': sgld_nll,
            'ece': sgld_ece,
            'mcmc_diagnostics': mcmc_diagnostics
        }
    }
    
    # Convert all numpy values to Python native types
    results = convert_numpy_to_python(results)
    
    results_path = output_dir / "evaluation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
