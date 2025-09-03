#!/bin/bash
#Batch Job Script for Bayesian LoRA Experiments
#
#SBATCH --job-name=run_bayesian_lora

#SBATCH --time=10:00:00                    # Job run time (hh:mm:ss) - 10 hours
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu" # Email address to alert when job starts/finishes
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --ntasks-per-node=1                # Number of cores per node
#SBATCH --job-name=nalint2-lora-exp        # Name of job
#SBATCH --account=arindamb-cs-eng          # Account
#SBATCH --partition=eng-research-gpu       # Partition
#SBATCH --output=nalint2_lora_log_%j       # output file name with job ID
#

# Load any necessary modules (uncomment if needed)
# module load python/3.9
# module load cuda/11.8

# Set environment variables to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Check GPU and memory of node
echo "=== GPU Information ==="
nvidia-smi
echo "=== Memory Information ==="
free -h
echo "=== Current Directory ==="
pwd
echo "=== Python Version ==="
python3 --version

# Activate virtual environment and install dependencies
echo "=== Setting up environment ==="
source .venv/bin/activate

echo "=== Upgrading pip to fix version issues ==="
python3 -m pip install --upgrade pip

echo "=== Installing dependencies ==="
echo "Installing package in editable mode (proper way)..."
pip3 install -e .

# Check if the installation actually succeeded
echo "=== Verifying pip installation ==="
pip3 show bayesian-lora || {
    echo "❌ pip install -e . failed or didn't complete properly!"
    echo "=== Debugging pip install failure ==="
    echo "Python version: $(python3 --version)"
    echo "Pip version: $(pip3 --version)"
    echo "Current directory: $(pwd)"
    echo "Virtual environment: $VIRTUAL_ENV"
    
    echo "=== Running deep debug analysis ==="
    python3 debug/deep_debug.py
    
    echo "=== Trying alternative approaches ==="
    echo "Attempting pip install . (non-editable)..."
    pip3 install . || {
        echo "❌ pip install . also failed. Installing dependencies only..."
        pip3 install -r requirements_lora.txt
        
        echo "=== Setting PYTHONPATH as fallback ==="
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
        echo "PYTHONPATH set to: $PYTHONPATH"
    }
}

echo "=== Verifying installation ==="
echo "Testing package installation status..."

# Test the proper import (should work if pip install -e . succeeded)
python3 -c "
import bayesian_lora
print('✅ bayesian_lora module imported successfully!')
print(f'Version: {bayesian_lora.__version__}')
print(f'Author: {bayesian_lora.__author__}')
" && {
    echo "✅ Package properly installed! Testing submodules..."
    python3 debug/test_installation.py
} || {
    echo "❌ Direct import failed. Let's debug..."
    echo "Current PYTHONPATH: $PYTHONPATH"
    echo "Current directory: $(pwd)"
    
    echo "=== Running debug tests ==="
    echo "Running installation test..."
    python3 debug/test_installation.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Installation test failed! Exiting."
        exit 1
    fi
}

echo "=== Checking key dependencies ==="
python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python3 -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')"
python3 -c "import peft; print(f'✓ PEFT version: {peft.__version__}')"

echo "=== Running quick debug check ==="
python3 debug/cluster_troubleshooting.py

# Run your experiment using Make commands
# Example: Running SST-2 BERT SGLD experiment
echo "=== Starting SST-2 BERT SGLD Experiment ==="
make experiment-sst2-bert-sgld

# Alternative examples you can uncomment and use:
# echo "=== Starting SST-2 RoBERTa SAM-SGLD Experiment ==="
# make experiment-sst2-roberta-sam-sgld

# echo "=== Starting MRPC DistilBERT ASGLD Experiment ==="
# make experiment-mrpc-distilbert-asgld

# echo "=== Starting IMDB BERT SAM-SGLD-R1 Experiment ==="
# make experiment-imdb-bert-sam-sgld-r1

echo "=== Experiment Completed ==="
