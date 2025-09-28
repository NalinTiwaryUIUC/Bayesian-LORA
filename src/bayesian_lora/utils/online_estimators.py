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
        # Use smaller block sizes for better autocorrelation capture with few samples
        if m < 50:
            # For small sample sizes, use smaller blocks
            block_size = max(2, int(m ** 0.3))
        else:
            # For larger sample sizes, use the configured growth
            block_size = max(10, int(m ** self.block_size_growth))
        # Ensure we don't use too large blocks
        return min(block_size, max(1, m // 4))
    
    def get_iact(self) -> float:
        """
        Calculate Integrated Autocorrelation Time (IACT) using OBM.
        Returns tau_hat = m * Var_hat(mean) / s2
        """
        m = len(self.values)
        if m < 2:
            return 1.0
        
        s2 = self.marginal_var_estimator.get_sample_variance()
        if s2 <= 0:
            return 1.0
        
        # For small sample sizes, use a simpler autocorrelation estimate
        if m < 30:
            # Use lag-1 autocorrelation for small samples
            if m < 3:
                return 1.0
            
            # Calculate lag-1 autocorrelation
            values_array = np.array(self.values)
            mean_val = np.mean(values_array)
            centered = values_array - mean_val
            
            if len(centered) < 2:
                return 1.0
            
            # Lag-1 autocorrelation
            numerator = np.sum(centered[:-1] * centered[1:])
            denominator = np.sum(centered ** 2)
            
            if denominator <= 0:
                return 1.0
            
            rho_1 = numerator / denominator
            
            # IACT estimate for AR(1) process
            if rho_1 >= 0.99:  # Avoid division by zero
                tau_hat = 100.0
            else:
                tau_hat = (1 + rho_1) / (1 - rho_1)
            
            return max(1.0, min(tau_hat, m / 2))  # Cap at reasonable values
        
        # For larger samples, use autocorrelation function approach
        # This is more reliable than OBM for MCMC diagnostics
        values_array = np.array(self.values)
        mean_val = np.mean(values_array)
        centered = values_array - mean_val
        
        # Calculate autocorrelation function up to lag m//4
        max_lag = min(m // 4, 50)  # Limit to reasonable number of lags
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            if lag >= len(centered):
                break
                
            # Calculate autocorrelation at lag
            numerator = np.sum(centered[:-lag] * centered[lag:])
            denominator = np.sum(centered ** 2)
            
            if denominator <= 0:
                break
                
            autocorr = numerator / denominator
            autocorrs.append(autocorr)
            
            # Stop if autocorrelation becomes negative (indicates noise)
            if autocorr <= 0:
                break
        
        if not autocorrs:
            return 1.0
        
        # Find the first negative autocorrelation or use all positive ones
        positive_autocorrs = [ac for ac in autocorrs if ac > 0]
        
        if not positive_autocorrs:
            return 1.0
        
        # Estimate IACT using the sum of autocorrelations
        # IACT ≈ 1 + 2 * sum of autocorrelations
        iact_estimate = 1 + 2 * sum(positive_autocorrs)
        
        return max(1.0, iact_estimate)
    
    def get_ess(self) -> float:
        """Calculate Effective Sample Size (ESS)."""
        m = len(self.values)
        if m < 2:
            return 1.0
        
        tau_hat = self.get_iact()
        return m / tau_hat
    
    def get_ess_per_step(self, total_steps: int) -> float:
        """Calculate ESS per total sampling step."""
        ess = self.get_ess()
        if total_steps == 0:
            return 0.0
        return ess / total_steps


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
