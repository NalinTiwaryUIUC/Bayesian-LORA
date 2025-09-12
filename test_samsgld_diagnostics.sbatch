#!/bin/bash
#Batch Job Script for SAM-SGLD Diagnostics
#
#SBATCH --job-name=samsgld_diagnostics
#SBATCH --time=00:30:00                    # 30 minutes should be enough
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/samsgld_diagnostics_%j.log

# Create logs directory
mkdir -p logs

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Check GPU
echo "=== GPU Information ==="
nvidia-smi

# Activate virtual environment
echo "=== Setting up environment ==="
source .venv/bin/activate

# Install package
echo "=== Installing package ==="
pip3 install -e .

# Run diagnostics
echo "=== Running SAM-SGLD Diagnostics ==="
python3 test_samsgld_diagnostics.py

echo "=== Diagnostics completed ==="
echo "Job completed at: $(date)"
