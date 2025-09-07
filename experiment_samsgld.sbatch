#!/bin/bash
#Batch Job Script for SAM-SGLD Rank-1 LoRA Experiment
#
#SBATCH --job-name=samsgld_lora_experiment
#SBATCH --time=15:00:00                    # Job run time (hh:mm:ss) - 15 hours for LoRA only
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu" # Email address to alert when job starts/finishes
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --ntasks-per-node=1                # Number of cores per node
#SBATCH --account=arindamb-cs-eng          # Account
#SBATCH --partition=eng-research-gpu       # Partition
#SBATCH --output=logs/mrpc_roberta_lora_samsgld_rank1/samsgld_lora_experiment_log_%j    # output file name with job ID

# Load any necessary modules (uncomment if needed)
# module load python/3.9
# module load cuda/11.8

# Create logs directory structure if it doesn't exist
mkdir -p logs/mrpc_roberta_lora_samsgld_rank1

# Set environment variables to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Check GPU and memory of node
echo "=== GPU Information ==="
nvidia-smi
echo "=== Memory Information ==="
free -h
echo "=== CPU Information ==="
lscpu | grep "Model name"
echo "=== Python Version ==="
python3 --version
echo "=== PyTorch Version ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo "=== Transformers Version ==="
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Change to the project directory
cd /home/nalint2/Bayesian-LORA

# Install the package in development mode
echo "=== Installing Bayesian LoRA Package ==="
pip3 install -e .

# Run the SAM-SGLD LoRA experiment
echo "=== Starting SAM-SGLD LoRA Experiment ==="
python3 scripts/train_mrpc_lora.py \
    --config configs/mrpc_roberta_lora_samsgld_rank1.yaml \
    --output_dir runs/mrpc_roberta_lora_samsgld_rank1 \
    --device cuda

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully ==="
    
    # Run evaluation
    echo "=== Starting Evaluation ==="
    python3 scripts/eval_mrpc_lora.py \
        --config configs/mrpc_roberta_lora_samsgld_rank1.yaml \
        --map_model_path runs/mrpc_roberta_lora_samsgld_rank1/map_model.pth \
        --sgld_samples_path runs/mrpc_roberta_lora_samsgld_rank1/sgld_samples.pth \
        --output_dir runs/mrpc_roberta_lora_samsgld_rank1 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "=== Evaluation completed successfully ==="
        echo "=== Experiment completed successfully ==="
    else
        echo "=== Evaluation failed ==="
        exit 1
    fi
else
    echo "=== Training failed ==="
    exit 1
fi

echo "=== Job completed ==="
