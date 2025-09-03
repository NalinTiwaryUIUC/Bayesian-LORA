# Bayesian LORA

Bayesian posterior sampling for deep neural networks using **SGLD (Stochastic Gradient Langevin Dynamics)** applied to both **ResNet/CIFAR classification** and **LoRA (Low-Rank Adaptation)** on transformer models.

This repository provides:
- A clean **experiment bench** for Bayesian posterior sampling in deep learning
- **CIFAR experiments** with ResNet and WideResNet using SGLD variants
- **LoRA experiments** with transformers using SGLD sampling
- Modularized code for **models, samplers, utilities, and evaluation**
- Scripts for **training/sampling** and **evaluation/ensembling**
- Config-driven experiments for reproducibility

---

## Experiments Available

### 1. CIFAR Classification with SGLD Variants
- **CIFAR-10 + ResNet-18 + SGLD**
- **CIFAR-100 + WideResNet-28-10 + SGLD**
- **CIFAR-100 + WideResNet-28-10 + SAM-SGLD**
- **CIFAR-100 + WideResNet-28-10 + SAM-SGLD Rank-1**

### 2. MRPC RoBERTa LoRA SGLD
- **Dataset**: GLUE MRPC (Microsoft Research Paraphrase Corpus)
- **Model**: RoBERTa-base with LoRA adaptation
- **Methods**: MAP-LoRA (baseline) vs. SGLD-LoRA (Bayesian)

---

## Getting Started

### 1. Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -e .
pip3 install -r requirements_lora.txt
```

### 2. Run CIFAR Experiments
```bash
# Individual experiments
make experiment-cifar10-resnet18-sgld
make experiment-cifar100-wrn2810-sgld
make experiment-cifar100-wrn2810-sam-sgld
make experiment-cifar100-wrn2810-sam-sgld-r1

# Or run all CIFAR experiments
make experiments-cifar
```

### 3. Run LoRA Experiment
```bash
# Complete MRPC LoRA experiment
make experiment-mrpc-lora

# Or run phases separately
make train-mrpc-lora
make eval-mrpc-lora
```

### 4. Run All Experiments
```bash
make experiments-all
```

## Configuration

### CIFAR Experiments
Each CIFAR experiment has its own YAML config file in `configs/`:
- `cifar10_resnet18_sgld.yaml`
- `cifar100_wrn2810_sgld.yaml`
- `cifar100_wrn2810_sam_sgld.yaml`
- `cifar100_wrn2810_sam_sgld_r1.yaml`

### LoRA Experiment
The MRPC LoRA experiment uses:
- `configs/mrpc_roberta_lora_sgld.yaml`

## Methods Implemented

### CIFAR Experiments
- **SGLD**: Stochastic Gradient Langevin Dynamics
- **ASGLD**: Adaptive SGLD with Adam-like moments
- **SAM-SGLD**: Sharpness-Aware SGLD with adversarial perturbations
- **SAM-SGLD Rank-1**: Directional low-rank noise variant

### LoRA Experiment
- **MAP-LoRA**: Standard LoRA training with AdamW optimizer
- **SGLD-LoRA**: Stochastic Gradient Langevin Dynamics in LoRA subspace

## Evaluation

### CIFAR Experiments
```bash
# Single sample evaluation
make eval-cifar10-resnet18-sgld
make eval-cifar100-wrn2810-sgld
make eval-cifar100-wrn2810-sam-sgld
make eval-cifar100-wrn2810-sam-sgld-r1

# Or evaluate all CIFAR experiments
make eval-cifar
```

### LoRA Experiment
```bash
make eval-mrpc-lora
```

### All Experiments
```bash
make eval-all
```

## Metrics & Diagnostics

### Predictive Performance
- **Accuracy**: Classification accuracy
- **NLL**: Negative Log-Likelihood
- **ECE**: Expected Calibration Error

### MCMC Diagnostics (SGLD only)
- **R-hat**: Gelman-Rubin statistic for convergence
- **ESS**: Effective Sample Size for mixing quality
- **Trace plots**: Parameter evolution across chains

## Example Workflow

### CIFAR Experiments
1. **Choose sampler** via config (`sampler.type`)
2. **Run sampling** with `scripts/train.py`
3. **Evaluate** with `scripts/eval.py` (single or ensemble)

### LoRA Experiment
1. **Train MAP LoRA** to establish baseline
2. **Sample with SGLD** to explore posterior in LoRA subspace
3. **Compare performance** and calibration
4. **Analyze MCMC diagnostics**

## Debugging

### **Quick Health Check:**
```bash
python3 debug/debug_suite.py --quick
```

### **Full System Validation:**
```bash
python3 debug/debug_suite.py
```

### **Comprehensive Guide:**
See `debug/DEBUG_GUIDE.md` for detailed debugging information, common issues, and solutions.

## Clean Up
```bash
make clean          # Remove LoRA experiment results
make clean-cifar    # Remove CIFAR experiment results
make clean-all      # Remove all runs
```