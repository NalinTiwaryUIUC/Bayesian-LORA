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

# Test the proper import with timeout and better error handling
echo "=== Testing bayesian_lora import ==="
python3 -c "
import sys
print(f'Python path: {sys.path[:3]}...')
try:
    import bayesian_lora
    print('✅ bayesian_lora module imported successfully!')
    print(f'Version: {getattr(bayesian_lora, \"__version__\", \"Unknown\")}')
    print(f'Author: {getattr(bayesian_lora, \"__author__\", \"Unknown\")}')
    print(f'Module location: {bayesian_lora.__file__}')
except ImportError as e:
    print(f'❌ ImportError: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)
" 

if [ $? -eq 0 ]; then
    echo "✅ Package installation verified successfully!"
else
    echo "❌ Package import failed!"
    echo "=== Debugging import failure ==="
    
    echo "=== Checking Python path ==="
    python3 -c "import sys; print('Python path:'); [print(f'  {p}') for p in sys.path[:10]]"
    
    echo "=== Checking if package files exist ==="
    ls -la src/bayesian_lora/ 2>/dev/null || echo "❌ src/bayesian_lora/ not found"
    
    echo "=== Running debug suite ==="
    python3 debug/debug_suite.py --quick || echo "❌ Debug suite failed"
    
    echo "=== Final attempt: direct import test ==="
    cd src && python3 -c "
try:
    import bayesian_lora
    print('✅ Direct import from src successful!')
except Exception as e:
    print(f'❌ Direct import failed: {e}')
" || echo "❌ Direct import test failed"
    
    cd ..
    exit 1
fi

echo "=== Installing additional requirements ==="
pip3 install -r requirements_lora.txt

echo "=== Environment setup complete ==="
echo "Python executable: $(which python3)"
echo "Pip executable: $(which pip3)"
echo "Virtual environment: $VIRTUAL_ENV"

# Run the SAM-SGLD LoRA experiment
echo "=== Starting SAM-SGLD LoRA Experiment ==="
echo "Configuration: configs/mrpc_roberta_lora_samsgld_rank1.yaml"
echo "Output directory: runs/mrpc_roberta_lora_samsgld_rank1"

# Phase 1: SAM-SGLD LoRA Training
echo "=== Phase 1: Training MAP LoRA and SAM-SGLD Sampling ==="
python3 scripts/train_mrpc_lora.py \
    --config configs/mrpc_roberta_lora_samsgld_rank1.yaml \
    --output_dir runs/mrpc_roberta_lora_samsgld_rank1

if [ $? -eq 0 ]; then
    echo "✅ SAM-SGLD LoRA training completed successfully!"
else
    echo "❌ SAM-SGLD LoRA training failed!"
    exit 1
fi

# Phase 2: SAM-SGLD LoRA Evaluation
echo "=== Phase 2: Evaluating SAM-SGLD LoRA Results ==="
python3 scripts/eval_mrpc_lora.py \
    --config configs/mrpc_roberta_lora_samsgld_rank1.yaml \
    --map_model_path runs/mrpc_roberta_lora_samsgld_rank1/map_model.pth \
    --sgld_samples_path runs/mrpc_roberta_lora_samsgld_rank1/samsgld_samples.pth \
    --output_dir runs/mrpc_roberta_lora_samsgld_rank1

if [ $? -eq 0 ]; then
    echo "✅ SAM-SGLD LoRA evaluation completed successfully!"
else
    echo "❌ SAM-SGLD LoRA evaluation failed!"
    exit 1
fi

echo "=== SAM-SGLD LoRA experiment completed successfully! ==="
echo "Results saved to: runs/mrpc_roberta_lora_samsgld_rank1/"
echo "Job completed at: $(date)"
