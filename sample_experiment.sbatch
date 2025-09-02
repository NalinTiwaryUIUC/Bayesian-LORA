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

# Activate virtual environment if you have one (uncomment if needed)
# source /path/to/your/venv/bin/activate

# Run your experiment
# Example: Running CIFAR-10 ResNet18 experiment with SGLD
echo "=== Starting CIFAR-10 ResNet18 SGLD Experiment ==="
python3 scripts/train.py --config configs/cifar10_resnet18_sgld.yaml

# Alternative examples you can uncomment and use:
# echo "=== Starting SST-2 BERT SGLD Experiment ==="
# python3 scripts/train.py --config configs/experiment_sst2_bert_sgld.yaml

# echo "=== Starting CIFAR-100 Wide ResNet SGLD Experiment ==="
# python3 scripts/train.py --config configs/cifar100_wrn2810_sgld.yaml

echo "=== Experiment Completed ==="
