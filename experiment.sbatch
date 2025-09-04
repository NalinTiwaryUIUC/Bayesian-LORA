#!/bin/bash
#Batch Job Script for LoRA Experiment Only
#
#SBATCH --job-name=bayesian_lora_experiment
#SBATCH --time=10:00:00                    # Job run time (hh:mm:ss) - 10 hours for LoRA only
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu" # Email address to alert when job starts/finishes
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --ntasks-per-node=1                # Number of cores per node
#SBATCH --account=arindamb-cs-eng          # Account
#SBATCH --partition=eng-research-gpu       # Partition
#SBATCH --output=bayesian_lora_experiment_log_%j    # output file name with job ID

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
echo "Installing package in editable mode..."
pip3 install -e .

# Check if the installation succeeded
echo "=== Verifying pip installation ==="
pip3 show bayesian-lora || {
    echo "❌ pip install -e . failed!"
    echo "=== Debugging pip install failure ==="
    echo "Python version: $(python3 --version)"
    echo "Pip version: $(pip3 --version)"
    echo "Current directory: $(pwd)"
    echo "Virtual environment: $VIRTUAL_ENV"
    
    echo "=== File permissions check ==="
    echo "Checking src directory permissions..."
    ls -la src/ 2>/dev/null || echo "❌ Cannot access src directory"
    
    echo "=== Running emergency debug analysis ==="
    python3 debug/debug_suite.py --quick
    
    if [ $? -ne 0 ]; then
        echo "❌ Emergency debug failed! Trying alternative approaches..."
        echo "Attempting pip install . (non-editable)..."
        pip3 install . || {
            echo "❌ pip install . also failed. Installing dependencies only..."
            pip3 install -r requirements_lora.txt
            
            echo "=== Setting PYTHONPATH as fallback ==="
            export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
            echo "PYTHONPATH set to: $PYTHONPATH"
            
            echo "=== Testing fallback import ==="
            python3 -c "
import sys
sys.path.insert(0, '$(pwd)/src')
try:
    import bayesian_lora
    print('✅ Fallback import successful!')
except Exception as e:
    print(f'❌ Fallback import failed: {e}')
"
        }
    fi
}

echo "=== Verifying installation ==="
echo "Testing package installation status..."

# Test the proper import
python3 -c "
import bayesian_lora
print('✅ bayesian_lora module imported successfully!')
print(f'Version: {bayesian_lora.__version__}')
print(f'Author: {bayesian_lora.__author__}')
" && {
    echo "✅ Package installation verified successfully!"
} || {
    echo "❌ Package import failed even after installation attempts!"
    echo "=== Final debugging attempt ==="
    python3 debug/debug_suite.py
    exit 1
}

echo "=== Installing additional requirements ==="
pip3 install -r requirements_lora.txt

echo "=== Environment setup complete ==="
echo "Python executable: $(which python3)"
echo "Pip executable: $(which pip3)"
echo "Virtual environment: $VIRTUAL_ENV"

# Run the LoRA experiment only
echo "=== Starting LoRA Experiment ==="
echo "Configuration: configs/mrpc_roberta_lora_sgld.yaml"
echo "Output directory: runs/mrpc_roberta_lora_sgld"

# Phase 1: LoRA Training
echo "=== Phase 1: Training MAP LoRA and SGLD Sampling ==="
python3 scripts/train_mrpc_lora.py \
    --config configs/mrpc_roberta_lora_sgld.yaml \
    --output_dir runs/mrpc_roberta_lora_sgld

if [ $? -eq 0 ]; then
    echo "✅ LoRA training completed successfully!"
else
    echo "❌ LoRA training failed!"
    exit 1
fi

# Phase 2: LoRA Evaluation
echo "=== Phase 2: Evaluating LoRA Results ==="
python3 scripts/eval_mrpc_lora.py \
    --config configs/mrpc_roberta_lora_sgld.yaml \
    --map_model_path runs/mrpc_roberta_lora_sgld/map_model.pth \
    --sgld_samples_path runs/mrpc_roberta_lora_sgld/sgld_samples.pth \
    --output_dir runs/mrpc_roberta_lora_sgld

if [ $? -eq 0 ]; then
    echo "✅ LoRA evaluation completed successfully!"
else
    echo "❌ LoRA evaluation failed!"
    exit 1
fi

echo "=== LoRA experiment completed successfully! ==="
echo "Results saved to: runs/mrpc_roberta_lora_sgld/"
echo "Job completed at: $(date)"
