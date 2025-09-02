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
echo "Installing dependencies directly (skipping problematic package install)..."
pip3 install -r requirements_lora.txt

echo "=== Setting up Python path for imports ==="
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "=== Verifying installation ==="
echo "Testing import with PYTHONPATH set..."

# First try the simple import test
echo "Running simple import test..."
python3 debug/simple_import_test.py

if [ $? -ne 0 ]; then
    echo "❌ Simple import test failed!"
    echo "Trying to debug the issue..."
    echo "Current PYTHONPATH: $PYTHONPATH"
    echo "Current directory: $(pwd)"
    echo "Listing src directory:"
    ls -la src/
    echo "Listing src/bayesian_lora directory:"
    ls -la src/bayesian_lora/
    
    echo "Trying alternative import test..."
    python3 debug/test_import.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Both import tests failed! Exiting."
        exit 1
    fi
else
    echo "✅ Simple import test successful!"
fi

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
