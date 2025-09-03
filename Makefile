venv:
	python3 -m venv .venv

install:
	pip3 install -e .
	pip3 install -r requirements.txt

install-lora:
	pip3 install -r requirements_lora.txt

test:
	pytest -q

format:
	black . && ruff check --fix .

help:
	@echo "Bayesian LoRA - Available Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  venv          - Create virtual environment"
	@echo "  install       - Install package in editable mode"
	@echo "  install-lora  - Install LoRA-specific dependencies"
	@echo "  test          - Run tests"
	@echo "  format        - Format code with black and ruff"
	@echo ""
	@echo "CIFAR Experiments:"
	@echo "  experiment-cifar10-resnet18-sgld      - CIFAR-10 ResNet-18 SGLD"
	@echo "  experiment-cifar100-wrn2810-sgld      - CIFAR-100 WideResNet-28-10 SGLD"
	@echo "  experiment-cifar100-wrn2810-sam-sgld  - CIFAR-100 WideResNet-28-10 SAM-SGLD"
	@echo "  experiment-cifar100-wrn2810-sam-sgld-r1 - CIFAR-100 WideResNet-28-10 SAM-SGLD Rank-1"
	@echo "  experiments-cifar                      - Run all CIFAR experiments"
	@echo ""
	@echo "LoRA Experiments:"
	@echo "  train-mrpc-lora                        - Train MRPC LoRA with SGLD"
	@echo "  eval-mrpc-lora                         - Evaluate MRPC LoRA experiment"
	@echo "  experiment-mrpc-lora                   - Run complete MRPC LoRA experiment"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval-cifar10-resnet18-sgld            - Evaluate CIFAR-10 ResNet-18 SGLD"
	@echo "  eval-cifar100-wrn2810-sgld            - Evaluate CIFAR-100 WideResNet-28-10 SGLD"
	@echo "  eval-cifar100-wrn2810-sam-sgld        - Evaluate CIFAR-100 WideResNet-28-10 SAM-SGLD"
	@echo "  eval-cifar100-wrn2810-sam-sgld-r1     - Evaluate CIFAR-100 WideResNet-28-10 SAM-SGLD Rank-1"
	@echo "  eval-cifar                             - Evaluate all CIFAR experiments"
	@echo "  eval-all                               - Evaluate all experiments"
	@echo ""
	@echo "Aggregate Commands:"
	@echo "  experiments-all                         - Run all experiments (CIFAR + LoRA)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean                                   - Clean LoRA experiment runs"
	@echo "  clean-cifar                            - Clean CIFAR experiment runs"
	@echo "  clean-all                              - Clean all experiment runs"

# =============================================================================
# CIFAR Experiments with SGLD Variants
# =============================================================================

# CIFAR-10 ResNet-18 SGLD
experiment-cifar10-resnet18-sgld:
	python3 scripts/train.py --config configs/cifar10_resnet18_sgld.yaml

# CIFAR-100 WideResNet-28-10 SGLD
experiment-cifar100-wrn2810-sgld:
	python3 scripts/train.py --config configs/cifar100_wrn2810_sgld.yaml

# CIFAR-100 WideResNet-28-10 SAM-SGLD
experiment-cifar100-wrn2810-sam-sgld:
	python3 scripts/train.py --config configs/cifar100_wrn2810_sam_sgld.yaml

# CIFAR-100 WideResNet-28-10 SAM-SGLD Rank-1
experiment-cifar100-wrn2810-sam-sgld-r1:
	python3 scripts/train.py --config configs/cifar100_wrn2810_sam_sgld_r1.yaml

# =============================================================================
# MRPC LoRA SGLD Experiment
# =============================================================================

# Train MRPC LoRA with SGLD
train-mrpc-lora:
	python3 scripts/train_mrpc_lora.py --config configs/mrpc_roberta_lora_sgld.yaml

# Evaluate MRPC LoRA experiment
eval-mrpc-lora:
	python3 scripts/eval_mrpc_lora.py \
		--config configs/mrpc_roberta_lora_sgld.yaml \
		--map_model_path runs/mrpc_roberta_lora_sgld/map_model.pth \
		--sgld_samples_path runs/mrpc_roberta_lora_sgld/sgld_samples.pth

# Run complete MRPC LoRA experiment (train + evaluate)
experiment-mrpc-lora: train-mrpc-lora eval-mrpc-lora

# =============================================================================
# Evaluation Commands
# =============================================================================

# Evaluate CIFAR experiments
eval-cifar10-resnet18-sgld:
	python3 scripts/eval.py --config configs/cifar10_resnet18_sgld.yaml --output_dir runs/c10_r18_sgld

eval-cifar100-wrn2810-sgld:
	python3 scripts/eval.py --config configs/cifar100_wrn2810_sgld.yaml --output_dir runs/c100_wrn2810_sgld

eval-cifar100-wrn2810-sam-sgld:
	python3 scripts/eval.py --config configs/cifar100_wrn2810_sam_sgld.yaml --output_dir runs/c100_wrn2810_sam_sgld

eval-cifar100-wrn2810-sam-sgld-r1:
	python3 scripts/eval.py --config configs/cifar100_wrn2810_sam_sgld_r1.yaml --output_dir runs/c100_wrn2810_sam_sgld_r1

# =============================================================================
# Quick Commands
# =============================================================================

# Run all CIFAR experiments
experiments-cifar: experiment-cifar10-resnet18-sgld experiment-cifar100-wrn2810-sgld experiment-cifar100-wrn2810-sam-sgld experiment-cifar100-wrn2810-sam-sgld-r1

# Evaluate all CIFAR experiments
eval-cifar: eval-cifar10-resnet18-sgld eval-cifar100-wrn2810-sgld eval-cifar100-wrn2810-sam-sgld eval-cifar100-wrn2810-sam-sgld-r1

# Run all experiments (CIFAR + LoRA)
experiments-all: experiments-cifar experiment-mrpc-lora

# Evaluate all experiments
eval-all: eval-cifar eval-mrpc-lora

# -----------------------------------------------------------------------------
# Clean up
# -----------------------------------------------------------------------------

clean:
	rm -rf runs/mrpc_roberta_lora_sgld

clean-cifar:
	rm -rf runs/c10_r18_sgld runs/c100_wrn2810_sgld runs/c100_wrn2810_sam_sgld runs/c100_wrn2810_sam_sgld_r1

clean-all:
	rm -rf runs/*
