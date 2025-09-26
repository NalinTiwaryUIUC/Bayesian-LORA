"""
Online estimators for SAM-SGLD experiment.
Implements Welford's algorithm for running mean/variance and OBM for ESS estimation.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple


class WelfordEstimator:
    """
    Welford's online algorithm for computing running mean and variance.
    """
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
    
    def update(self, value: float) -> None:
        """Update the estimator with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    def get_mean(self) -> float:
        """Get the current mean."""
        return self.mean
    
    def get_variance(self) -> float:
        """Get the current variance (population variance)."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count
    
    def get_sample_variance(self) -> float:
        """Get the current sample variance (unbiased)."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
    def get_std(self) -> float:
        """Get the current standard deviation."""
        return math.sqrt(self.get_variance())


class OBMEstimator:
    """
    Overlapping Batch Means (OBM) estimator for ESS calculation.
    """
    
    def __init__(self, block_size_growth: float = 0.5):
        self.block_size_growth = block_size_growth
        self.values = []  # Store all values for OBM calculation
        self.marginal_var_estimator = WelfordEstimator()
        
    def update(self, value: float) -> None:
        """Update the estimator with a new value."""
        self.values.append(value)
        self.marginal_var_estimator.update(value)
    
    def get_block_size(self) -> int:
        """Calculate block size based on current sample count."""
        m = len(self.values)
        if m < 2:
            return 1
        # Use larger block sizes for better autocorrelation capture
        block_size = max(10, int(m ** self.block_size_growth))
        # Ensure we don't use too large blocks
        return min(block_size, m // 4)
    
    def get_iact(self) -> float:
        """
        Calculate Integrated Autocorrelation Time (IACT) using OBM.
        Returns tau_hat = m * Var_hat(mean) / s2
        """
        m = len(self.values)
        if m < 2:
            return 1.0
        
        b = self.get_block_size()
        s2 = self.marginal_var_estimator.get_sample_variance()
        
        if s2 <= 0:
            return 1.0
        
        # Calculate overlapping batch means
        num_batches = m - b + 1
        if num_batches < 2:
            return 1.0
            
        batch_means = []
        for i in range(num_batches):
            batch_mean = np.mean(self.values[i:i + b])
            batch_means.append(batch_mean)
        
        # Calculate variance of batch means using numpy for accuracy
        batch_var = np.var(batch_means, ddof=1) if len(batch_means) > 1 else 0.0
        
        # OBM estimate of variance of sample mean
        var_hat_mean = batch_var / num_batches
        
        # IACT estimate
        tau_hat = m * var_hat_mean / s2
        
        return max(1.0, tau_hat)
    
    def get_ess(self) -> float:
        """Calculate Effective Sample Size (ESS)."""
        m = len(self.values)
        if m < 2:
            return 1.0
        
        tau_hat = self.get_iact()
        return m / tau_hat
    
    def get_ess_per_step(self, thinning_interval: int) -> float:
        """Calculate ESS per SGD step."""
        ess = self.get_ess()
        m = len(self.values)
        if m == 0 or thinning_interval == 0:
            return 0.0
        return ess / (m * thinning_interval)


class OnlineMetricsTracker:
    """
    Tracks multiple metrics with online ESS estimation.
    """
    
    def __init__(self, metric_names: List[str], block_size_growth: float = 0.5):
        self.metric_names = metric_names
        self.metric_estimators = {
            name: WelfordEstimator() for name in metric_names
        }
        self.obm_estimators = {
            name: OBMEstimator(block_size_growth) for name in metric_names
        }
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update all metric estimators."""
        for name, value in metrics.items():
            if name in self.metric_estimators:
                self.metric_estimators[name].update(value)
                self.obm_estimators[name].update(value)
    
    def get_summary(self, thinning_interval: int) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for name in self.metric_names:
            welford = self.metric_estimators[name]
            obm = self.obm_estimators[name]
            
            summary[name] = {
                'mean': welford.get_mean(),
                'std': welford.get_std(),
                'count': welford.count,
                'ess': obm.get_ess(),
                'ess_per_step': obm.get_ess_per_step(thinning_interval),
                'iact': obm.get_iact(),
                'mcse': welford.get_std() / math.sqrt(obm.get_ess()) if obm.get_ess() > 0 else float('inf')
            }
        
        return summary


class CumulativeEnsemble:
    """
    Maintains cumulative ensemble predictions for accuracy/NLL/ECE calculation.
    """
    
    def __init__(self):
        self.logits_sum = None
        self.count = 0
        self.labels = None
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Add new sample to the cumulative ensemble."""
        if self.logits_sum is None:
            self.logits_sum = torch.zeros_like(logits)
            self.labels = labels.clone()
        
        self.logits_sum += logits
        self.count += 1
    
    def get_ensemble_logits(self) -> torch.Tensor:
        """Get ensemble average logits."""
        if self.count == 0:
            raise ValueError("No samples added to ensemble")
        return self.logits_sum / self.count
    
    def get_ensemble_probs(self) -> torch.Tensor:
        """Get ensemble average probabilities."""
        logits = self.get_ensemble_logits()
        return torch.softmax(logits, dim=-1)
    
    def reset(self) -> None:
        """Reset the ensemble."""
        self.logits_sum = None
        self.count = 0
        self.labels = None
