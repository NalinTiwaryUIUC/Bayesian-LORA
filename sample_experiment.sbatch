#!/bin/bash
#Sample Batch Job Script for Bayesian LoRA Experiments
#
#SBATCH --time=00:10:00                    # Job run time (hh:mm:ss) - 10 minutes
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

echo "=== Installing dependencies ==="
make install
make install-lora

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
