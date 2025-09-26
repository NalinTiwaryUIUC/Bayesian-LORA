#!/usr/bin/env python3
"""
Evaluation script for MRPC LoRA with online SAM-SGLD results.
Analyzes the online summary and generates reports.
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_online_summary(summary_path):
    """Load the online summary from YAML file."""
    with open(summary_path, 'r') as f:
        return yaml.safe_load(f)


def generate_ess_report(summary, output_dir, logger):
    """Generate ESS report and plots."""
    logger.info("Generating ESS report...")
    
    # Extract ESS data
    ess_data = {}
    for scalar, stats in summary.items():
        if 'ess' in stats:
            ess_data[scalar] = {
                'ess': stats['ess'],
                'ess_per_step': stats['ess_per_step'],
                'iact': stats['iact'],
                'count': stats['count']
            }
    
    # Create ESS report
    report_path = os.path.join(output_dir, "ess_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== EFFECTIVE SAMPLE SIZE (ESS) REPORT ===\n\n")
        
        f.write("Scalar Diagnostics:\n")
        f.write("-" * 60 + "\n")
        for scalar, data in ess_data.items():
            f.write(f"{scalar}:\n")
            f.write(f"  ESS (kept samples): {data['ess']:.2f}\n")
            f.write(f"  ESS per SGD step: {data['ess_per_step']:.6f}\n")
            f.write(f"  Integrated Autocorrelation Time: {data['iact']:.2f}\n")
            f.write(f"  Sample count: {data['count']}\n\n")
        
        # Summary statistics
        ess_values = [data['ess'] for data in ess_data.values()]
        ess_per_step_values = [data['ess_per_step'] for data in ess_data.values()]
        
        f.write("Summary Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Min ESS: {min(ess_values):.2f}\n")
        f.write(f"Max ESS: {max(ess_values):.2f}\n")
        f.write(f"Mean ESS: {np.mean(ess_values):.2f}\n")
        f.write(f"Std ESS: {np.std(ess_values):.2f}\n\n")
        
        f.write(f"Min ESS/step: {min(ess_per_step_values):.6f}\n")
        f.write(f"Max ESS/step: {max(ess_per_step_values):.6f}\n")
        f.write(f"Mean ESS/step: {np.mean(ess_per_step_values):.6f}\n")
        f.write(f"Std ESS/step: {np.std(ess_per_step_values):.6f}\n")
    
    logger.info(f"ESS report saved to {report_path}")
    
    # Create ESS plots
    plot_ess_metrics(ess_data, output_dir, logger)


def plot_ess_metrics(ess_data, output_dir, logger):
    """Create ESS visualization plots."""
    logger.info("Creating ESS plots...")
    
    # Extract data for plotting
    scalars = list(ess_data.keys())
    ess_values = [ess_data[s]['ess'] for s in scalars]
    ess_per_step_values = [ess_data[s]['ess_per_step'] for s in scalars]
    iact_values = [ess_data[s]['iact'] for s in scalars]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ESS Analysis', fontsize=16)
    
    # ESS values
    axes[0, 0].bar(range(len(scalars)), ess_values)
    axes[0, 0].set_title('ESS (kept samples)')
    axes[0, 0].set_xlabel('Scalar')
    axes[0, 0].set_ylabel('ESS')
    axes[0, 0].set_xticks(range(len(scalars)))
    axes[0, 0].set_xticklabels(scalars, rotation=45, ha='right')
    
    # ESS per step
    axes[0, 1].bar(range(len(scalars)), ess_per_step_values)
    axes[0, 1].set_title('ESS per SGD step')
    axes[0, 1].set_xlabel('Scalar')
    axes[0, 1].set_ylabel('ESS per step')
    axes[0, 1].set_xticks(range(len(scalars)))
    axes[0, 1].set_xticklabels(scalars, rotation=45, ha='right')
    
    # IACT values
    axes[1, 0].bar(range(len(scalars)), iact_values)
    axes[1, 0].set_title('Integrated Autocorrelation Time')
    axes[1, 0].set_xlabel('Scalar')
    axes[1, 0].set_ylabel('IACT')
    axes[1, 0].set_xticks(range(len(scalars)))
    axes[1, 0].set_xticklabels(scalars, rotation=45, ha='right')
    
    # ESS vs IACT scatter
    axes[1, 1].scatter(iact_values, ess_values, alpha=0.7)
    axes[1, 1].set_xlabel('IACT')
    axes[1, 1].set_ylabel('ESS')
    axes[1, 1].set_title('ESS vs IACT')
    
    # Add labels to points
    for i, scalar in enumerate(scalars):
        axes[1, 1].annotate(scalar, (iact_values[i], ess_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "ess_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ESS plots saved to {plot_path}")


def generate_metrics_report(summary, output_dir, logger):
    """Generate metrics report with MCSE analysis."""
    logger.info("Generating metrics report...")
    
    # Extract metrics data
    metrics_data = {}
    for metric in ['accuracy', 'nll', 'ece']:
        if metric in summary:
            metrics_data[metric] = summary[metric]
    
    # Create metrics report
    report_path = os.path.join(output_dir, "metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== METRICS REPORT WITH MCSE ===\n\n")
        
        f.write("Per-Sample Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, data in metrics_data.items():
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Mean: {data['mean']:.6f}\n")
            f.write(f"  Standard Deviation: {data['std']:.6f}\n")
            f.write(f"  MCSE (Monte Carlo Standard Error): {data['mcse']:.6f}\n")
            f.write(f"  Sample count: {data['count']}\n")
            
            # MCSE interpretation
            mcse_ratio = data['mcse'] / data['std'] if data['std'] > 0 else float('inf')
            f.write(f"  MCSE/Std ratio: {mcse_ratio:.4f}\n")
            if mcse_ratio < 0.1:
                f.write("  ✅ MCSE is small relative to standard deviation\n")
            elif mcse_ratio < 0.3:
                f.write("  ⚠️  MCSE is moderate relative to standard deviation\n")
            else:
                f.write("  ❌ MCSE is large relative to standard deviation\n")
            f.write("\n")
        
        # Summary statistics
        if metrics_data:
            mcse_values = [data['mcse'] for data in metrics_data.values()]
            f.write("MCSE Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Min MCSE: {min(mcse_values):.6f}\n")
            f.write(f"Max MCSE: {max(mcse_values):.6f}\n")
            f.write(f"Mean MCSE: {np.mean(mcse_values):.6f}\n")
    
    logger.info(f"Metrics report saved to {report_path}")


def generate_convergence_assessment(summary, output_dir, logger):
    """Generate convergence assessment report."""
    logger.info("Generating convergence assessment...")
    
    # Extract relevant data
    ess_values = []
    for scalar, stats in summary.items():
        if 'ess' in stats:
            ess_values.append(stats['ess'])
    
    # Create convergence report
    report_path = os.path.join(output_dir, "convergence_assessment.txt")
    with open(report_path, 'w') as f:
        f.write("=== CONVERGENCE ASSESSMENT ===\n\n")
        
        if ess_values:
            min_ess = min(ess_values)
            mean_ess = np.mean(ess_values)
            max_ess = max(ess_values)
            
            f.write("ESS Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Minimum ESS: {min_ess:.2f}\n")
            f.write(f"Mean ESS: {mean_ess:.2f}\n")
            f.write(f"Maximum ESS: {max_ess:.2f}\n\n")
            
            # Convergence criteria
            f.write("Convergence Criteria:\n")
            f.write("-" * 25 + "\n")
            
            if min_ess >= 100:
                f.write("✅ Minimum ESS ≥ 100: EXCELLENT convergence\n")
            elif min_ess >= 50:
                f.write("✅ Minimum ESS ≥ 50: GOOD convergence\n")
            elif min_ess >= 25:
                f.write("⚠️  Minimum ESS ≥ 25: MODERATE convergence\n")
            elif min_ess >= 10:
                f.write("⚠️  Minimum ESS ≥ 10: POOR convergence\n")
            else:
                f.write("❌ Minimum ESS < 10: VERY POOR convergence\n")
            
            # ESS variability
            ess_cv = np.std(ess_values) / np.mean(ess_values) if np.mean(ess_values) > 0 else float('inf')
            f.write(f"\nESS Coefficient of Variation: {ess_cv:.3f}\n")
            if ess_cv < 0.5:
                f.write("✅ Low ESS variability: Consistent mixing across diagnostics\n")
            elif ess_cv < 1.0:
                f.write("⚠️  Moderate ESS variability: Some diagnostics mix better than others\n")
            else:
                f.write("❌ High ESS variability: Inconsistent mixing across diagnostics\n")
        
        # Recommendations
        f.write("\nRecommendations:\n")
        f.write("-" * 15 + "\n")
        if ess_values and min(ess_values) < 50:
            f.write("- Consider increasing sampling steps or reducing thinning interval\n")
            f.write("- Check if step size is appropriate for the target distribution\n")
            f.write("- Verify that burn-in was sufficient\n")
        if ess_values and np.std(ess_values) / np.mean(ess_values) > 0.5:
            f.write("- High ESS variability suggests some diagnostics are more correlated than others\n")
            f.write("- Consider using different diagnostic scalars or increasing sample size\n")
    
    logger.info(f"Convergence assessment saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate online SAM-SGLD results")
    parser.add_argument("--summary", type=str, required=True, help="Path to online summary YAML file")
    parser.add_argument("--output_dir", type=str, help="Output directory for reports")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use same directory as summary file
        output_dir = os.path.dirname(args.summary)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging("INFO")
    
    logger.info(f"Evaluating online summary: {args.summary}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load summary
    try:
        summary = load_online_summary(args.summary)
        logger.info("Online summary loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load online summary: {e}")
        return 1
    
    # Generate reports
    try:
        generate_ess_report(summary, output_dir, logger)
        generate_metrics_report(summary, output_dir, logger)
        generate_convergence_assessment(summary, output_dir, logger)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
