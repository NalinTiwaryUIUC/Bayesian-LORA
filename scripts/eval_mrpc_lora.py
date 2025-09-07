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

# Setup logging to both console and file
def setup_logging(experiment_name="mrpc_roberta_lora_sgld"):
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
    file_handler = logging.FileHandler(logs_dir / "evaluation.log")
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
    """Compute rank-normalized, split-chain R-hat (Gelman-Rubin statistic) for MCMC diagnostics.
    
    This implements the rank-normalized, split-chain R-hat as specified in the convergence document.
    """
    if len(chains) < 2:
        return 1.0
    
    # Convert to numpy arrays if needed
    chains = [np.array(chain) for chain in chains]
    
    # Step 1: Rank-normalize all values
    # Pool all values and convert to ranks
    all_values = np.concatenate(chains)
    ranks = np.argsort(np.argsort(all_values)) + 1  # 1-based ranks
    n_total = len(all_values)
    
    # Map ranks to (0,1) via r/(n_total+1), then apply standard normal inverse CDF
    from scipy.stats import norm
    normalized_values = norm.ppf(ranks / (n_total + 1))
    
    # Split back into chains
    rank_normalized_chains = []
    start_idx = 0
    for chain in chains:
        end_idx = start_idx + len(chain)
        rank_normalized_chains.append(normalized_values[start_idx:end_idx])
        start_idx = end_idx
    
    # Step 2: Split each chain in half
    split_chains = []
    for chain in rank_normalized_chains:
        n = len(chain)
        if n < 2:
            continue
        # Split chain in half
        half_length = n // 2
        split_chains.append(chain[:half_length])
        split_chains.append(chain[half_length:])
    
    if len(split_chains) < 2:
        return 1.0
    
    # Step 3: Compute within/between variances for split chains
    n_half = len(split_chains[0])  # Length of each half-chain
    m_half = len(split_chains)     # Number of half-chains (2K)
    
    # Within-chain variance (W)
    chain_vars = [np.var(chain, ddof=1) for chain in split_chains]
    W = np.mean(chain_vars)
    
    # Between-chain variance (B)
    chain_means = [np.mean(chain) for chain in split_chains]
    overall_mean = np.mean(chain_means)
    B = (n_half / (m_half - 1)) * np.sum((chain_means - overall_mean) ** 2)
    
    # Step 4: Variance estimate and R-hat
    V = ((n_half - 1) / n_half) * W + (1 / n_half) * B
    r_hat = np.sqrt(V / W) if W > 0 else 1.0
    
    return r_hat


def compute_ess(chain: np.ndarray) -> int:
    """Compute rank-normalized, split ESS (bulk-ESS) with Geyer IPS truncation.
    
    This implements the rank-normalized, split ESS as specified in the convergence document.
    """
    if len(chain) < 2:
        return len(chain)
    
    # For single chain, use rank normalization
    from scipy.stats import norm
    
    # Rank-normalize the chain
    ranks = np.argsort(np.argsort(chain)) + 1  # 1-based ranks
    n = len(chain)
    normalized_chain = norm.ppf(ranks / (n + 1))
    
    # Split chain in half
    half_length = n // 2
    if half_length < 2:
        return len(chain)
    
    half_chain1 = normalized_chain[:half_length]
    half_chain2 = normalized_chain[half_length:]
    
    # Compute ESS for each half-chain and average
    ess1 = _compute_ess_single_chain(half_chain1)
    ess2 = _compute_ess_single_chain(half_chain2)
    
    # Average the ESS values (weighted by within-chain variance)
    var1 = np.var(half_chain1, ddof=1)
    var2 = np.var(half_chain2, ddof=1)
    total_var = var1 + var2
    
    if total_var > 0:
        weighted_ess = (ess1 * var1 + ess2 * var2) / total_var
    else:
        weighted_ess = (ess1 + ess2) / 2
    
    # Scale back to original sample size
    ess = weighted_ess * (n / half_length)
    
    return int(max(1, ess))


def _compute_ess_single_chain(chain: np.ndarray) -> float:
    """Compute ESS for a single chain using Geyer IPS truncation."""
    if len(chain) < 2:
        return len(chain)
    
    # Compute autocorrelations using FFT-based autocovariance
    n = len(chain)
    centered_chain = chain - np.mean(chain)
    
    # FFT-based autocovariance
    fft_chain = np.fft.fft(centered_chain, n=2*n)
    autocov = np.fft.ifft(fft_chain * np.conj(fft_chain))[:n]
    autocov = np.real(autocov)
    
    # Normalize to get autocorrelations
    autocorr = autocov / autocov[0]
    
    # Geyer IPS truncation: find first odd lag m where ρ_{m-1} + ρ_m < 0
    m = 1
    while m < len(autocorr) - 1:
        if autocorr[m-1] + autocorr[m] < 0:
            break
        m += 2  # Only check odd lags
    
    # Truncate at m-1
    truncation_point = max(1, m - 1)
    
    # Integrated autocorrelation time
    tau_int = 1 + 2 * np.sum(autocorr[1:truncation_point+1])
    
    # ESS
    ess = n / tau_int
    
    return ess


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
    
    # Ensemble predictions - average probabilities, not logits
    # Convert logits to probabilities for each sample
    all_probs = []
    for sample_logits in all_logits:
        sample_probs = F.softmax(torch.tensor(sample_logits), dim=1).numpy()
        all_probs.append(sample_probs)
    
    all_probs = np.array(all_probs)  # [n_samples, n_examples, n_classes]
    
    # Average probabilities across samples (proper ensemble)
    ensemble_probs = np.mean(all_probs, axis=0)  # [n_examples, n_classes]
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
    
    # Compute ESS per chain for multimodal assessment
    ess_log_posterior_per_chain = [compute_ess(chain) for chain in log_posterior_chains]
    ess_l2_norm_per_chain = [compute_ess(chain) for chain in l2_norm_chains]
    
    mcmc_diagnostics['ess_log_posterior'] = min(ess_log_posterior_per_chain)
    mcmc_diagnostics['ess_l2_norm'] = min(ess_l2_norm_per_chain)
    mcmc_diagnostics['ess_log_posterior_per_chain'] = ess_log_posterior_per_chain
    mcmc_diagnostics['ess_l2_norm_per_chain'] = ess_l2_norm_per_chain
    
    # Multimodal-specific diagnostics
    mcmc_diagnostics['avg_ess_log_posterior'] = np.mean(ess_log_posterior_per_chain)
    mcmc_diagnostics['avg_ess_l2_norm'] = np.mean(ess_l2_norm_per_chain)
    mcmc_diagnostics['ess_std_log_posterior'] = np.std(ess_log_posterior_per_chain)
    mcmc_diagnostics['ess_std_l2_norm'] = np.std(ess_l2_norm_per_chain)
    
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
    
    # Setup logging with experiment name from config
    experiment_name = config['experiment']['name']
    logger = setup_logging(experiment_name)
    
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
    
    # Compute and report min ESS
    min_ess = min(mcmc_diagnostics['ess_log_posterior'], mcmc_diagnostics['ess_l2_norm'])
    logger.info(f"Min ESS: {min_ess}")
    
    # Multimodal MCMC Quality Assessment
    r_hat_threshold = config['evaluation']['mcmc_diagnostics']['r_hat_threshold']
    ess_threshold = config['evaluation']['mcmc_diagnostics']['ess_threshold']
    
    # For multimodal distributions, we expect R-hat > 1.05 (chains in different modes)
    r_hat_log_posterior = mcmc_diagnostics['r_hat_log_posterior']
    r_hat_l2_norm = mcmc_diagnostics['r_hat_l2_norm']
    
    # Check if chains are exploring different modes (R-hat > 1.1 but not too extreme)
    multimodal_r_hat_good = (1.1 < r_hat_log_posterior < 3.0 and 1.1 < r_hat_l2_norm < 3.0)
    
    # Check within-mode mixing (ESS per chain)
    avg_ess_log_posterior = mcmc_diagnostics['avg_ess_log_posterior']
    avg_ess_l2_norm = mcmc_diagnostics['avg_ess_l2_norm']
    within_mode_mixing_good = (avg_ess_log_posterior >= ess_threshold/4 and avg_ess_l2_norm >= ess_threshold/4)
    
    logger.info(f"\nMultimodal MCMC Quality Assessment:")
    logger.info(f"R-hat (log posterior): {r_hat_log_posterior:.3f}")
    logger.info(f"R-hat (L2 norm): {r_hat_l2_norm:.3f}")
    logger.info(f"Multimodal R-hat (1.1 < R-hat < 3.0): {'✅ GOOD' if multimodal_r_hat_good else '⚠️  CHECK'}")
    
    logger.info(f"\nWithin-mode mixing:")
    logger.info(f"Avg ESS (log posterior): {avg_ess_log_posterior:.1f} ± {mcmc_diagnostics['ess_std_log_posterior']:.1f}")
    logger.info(f"Avg ESS (L2 norm): {avg_ess_l2_norm:.1f} ± {mcmc_diagnostics['ess_std_l2_norm']:.1f}")
    logger.info(f"Within-mode mixing (ESS ≥ {ess_threshold/4}): {'✅ GOOD' if within_mode_mixing_good else '⚠️  NEEDS IMPROVEMENT'}")
    
    logger.info(f"\nESS Statistics Summary:")
    logger.info(f"  Log Posterior ESS: {mcmc_diagnostics['ess_log_posterior']} (min), {avg_ess_log_posterior:.1f} ± {mcmc_diagnostics['ess_std_log_posterior']:.1f} (mean ± std)")
    logger.info(f"  L2 Norm ESS: {mcmc_diagnostics['ess_l2_norm']} (min), {avg_ess_l2_norm:.1f} ± {mcmc_diagnostics['ess_std_l2_norm']:.1f} (mean ± std)")
    
    # Interpret standard deviations
    std_lp = mcmc_diagnostics['ess_std_log_posterior']
    std_l2 = mcmc_diagnostics['ess_std_l2_norm']
    logger.info(f"\nESS Variability Interpretation:")
    if std_lp > avg_ess_log_posterior * 0.5:
        logger.info(f"  • Log posterior ESS varies significantly (std={std_lp:.1f}) - different modes have different mixing quality")
    else:
        logger.info(f"  • Log posterior ESS is consistent across modes (std={std_lp:.1f}) - uniform mixing quality")
    
    if std_l2 > avg_ess_l2_norm * 0.5:
        logger.info(f"  • L2 norm ESS varies significantly (std={std_l2:.1f}) - parameter space mixing differs across modes")
    else:
        logger.info(f"  • L2 norm ESS is consistent across modes (std={std_l2:.1f}) - uniform parameter space mixing")
    
    logger.info(f"\nPer-chain ESS details:")
    for i, (ess_lp, ess_l2) in enumerate(zip(mcmc_diagnostics['ess_log_posterior_per_chain'], 
                                           mcmc_diagnostics['ess_l2_norm_per_chain'])):
        logger.info(f"  Chain {i+1}: ESS(log_posterior)={ess_lp}, ESS(l2_norm)={ess_l2}")
    
    overall_multimodal_quality = multimodal_r_hat_good and within_mode_mixing_good
    logger.info(f"\nOverall Multimodal Quality: {'✅ GOOD' if overall_multimodal_quality else '⚠️  NEEDS IMPROVEMENT'}")
    
    if not multimodal_r_hat_good:
        if r_hat_log_posterior <= 1.1 or r_hat_l2_norm <= 1.1:
            logger.warning("R-hat values suggest chains may be in the same mode (not multimodal)")
        elif r_hat_log_posterior >= 3.0 or r_hat_l2_norm >= 3.0:
            logger.warning("R-hat values are very high - chains may not have converged to any mode")
    
    if not within_mode_mixing_good:
        logger.warning(f"Within-mode ESS is low - samples within each mode are too correlated")
    
    # Multimodal Pass/Fail criteria validation
    logger.info(f"\nMultimodal Pass/Fail Criteria Assessment:")
    
    # Criterion 1: Multimodal R-hat (1.1 < R-hat < 3.0) - chains in different modes
    r_hat_pass = bool(multimodal_r_hat_good)
    logger.info(f"  Multimodal R-hat (1.1 < R-hat < 3.0): {'✅ PASS' if r_hat_pass else '❌ FAIL'}")
    
    # Criterion 2: Within-mode mixing (ESS ≥ threshold/4 per chain)
    ess_pass = bool(within_mode_mixing_good)
    logger.info(f"  Within-mode ESS ≥ {ess_threshold/4}: {'✅ PASS' if ess_pass else '❌ FAIL'}")
    
    # Criterion 3: Accuracy within ±0.3% of MAP-LoRA
    accuracy_diff = abs(sgld_accuracy - map_accuracy)
    accuracy_pass = bool(accuracy_diff <= 0.003)  # 0.3%
    logger.info(f"  Accuracy within ±0.3% of MAP: {'✅ PASS' if accuracy_pass else '❌ FAIL'} (diff: {accuracy_diff:.4f})")
    
    # Criterion 4: NLL and ECE lower than MAP-LoRA
    nll_pass = bool(sgld_nll < map_nll)
    ece_pass = bool(sgld_ece < map_ece)
    logger.info(f"  NLL lower than MAP: {'✅ PASS' if nll_pass else '❌ FAIL'} (SGLD: {sgld_nll:.4f}, MAP: {map_nll:.4f})")
    logger.info(f"  ECE lower than MAP: {'✅ PASS' if ece_pass else '❌ FAIL'} (SGLD: {sgld_ece:.4f}, MAP: {map_ece:.4f})")
    
    # Overall multimodal pass/fail
    overall_pass = bool(r_hat_pass and ess_pass and accuracy_pass and nll_pass and ece_pass)
    logger.info(f"\nOverall Multimodal Convergence Assessment: {'🎉 PASS' if overall_pass else '⚠️  FAIL'}")
    
    if not overall_pass:
        logger.warning("Multimodal convergence criteria not met. Consider:")
        if not r_hat_pass:
            if r_hat_log_posterior <= 1.1 or r_hat_l2_norm <= 1.1:
                logger.warning("  - Chains are in same mode - increase step size or reduce prior strength")
            elif r_hat_log_posterior >= 3.0 or r_hat_l2_norm >= 3.0:
                logger.warning("  - Chains not converged to any mode - reduce step size or increase burn-in")
        if not ess_pass:
            logger.warning("  - Within-mode mixing is poor - reduce step size or increase sampling steps")
        if not accuracy_pass:
            logger.warning("  - Check if SGLD is learning properly")
        if not nll_pass or not ece_pass:
            logger.warning("  - SGLD should outperform MAP - check implementation")
    
    # Convert numpy values to Python native types for human-readable YAML
    def convert_numpy_to_python(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer, np.bool_)):
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
            'mcmc_diagnostics': mcmc_diagnostics,
            'min_ess': min_ess,
            'pass_fail_criteria': {
                'multimodal_r_hat_pass': r_hat_pass,
                'within_mode_ess_pass': ess_pass,
                'accuracy_pass': accuracy_pass,
                'nll_pass': nll_pass,
                'ece_pass': ece_pass,
                'overall_multimodal_pass': overall_pass
            }
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
