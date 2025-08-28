# Bayesian LORA

Bayesian posterior sampling for deep neural networks with **SGLD variants** (SGLD, ASGLD, SAM-SGLD, SAM-SGLD Rank-1), applied to CIFAR classification using ResNet and WideResNet backbones.

This repository provides:
- A clean **experiment bench** for posterior sampling in deep learning.
- Modularized code for **models, samplers, utilities, and evaluation**.
- Scripts for **training/sampling** and **evaluation/ensembling**.
- Config-driven experiments for reproducibility.

---

## Getting Started

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Run Training & Sampling
Example: ResNet-18 + CIFAR-10 + SGLD
```bash
python scripts/train.py --config configs/cifar10_resnet18_sgld.yaml
```

### 3. Evaluate
Single sample:
```bash
python scripts/eval.py --config configs/cifar10_resnet18_sgld.yaml --single
```

Ensemble (first K samples):
```bash
python scripts/eval.py --config configs/cifar10_resnet18_sgld.yaml --k 20
```

## Configuration
Experiments are fully driven by YAML config files in `configs/`.

Key sections:
- **data**: dataset name, root path, batch size, augmentation.
- **model**: backbone (`resnet18_cifar`, `resnet34_cifar`, `wrn_28_10_cifar`).
- **train** *(optional)*: supervised pretraining before sampling (epochs, lr, weight_decay, etc.).
- **sampler**: type & hyperparameters (e.g., `type: sgld|asgld|sam-sgld|sam-sgld-r1`, `step_size`, `burn_in`, `thin`, `rho`, `beta1`, `beta2`, noise settings).
- **out**: directory for saving samples and `manifest.json`.

Example (minimal):
```yaml
data:
  name: cifar10
  root: /path/to/data
  batch_size: 128
model:
  name: resnet18_cifar
train:
  epochs: 0
sampler:
  type: sgld
  step_size: 1e-4
  burn_in: 2000
  thin: 200
out:
  dir: runs/c10_r18_sgld
```

## Samplers Implemented
- **SGLD** — Stochastic Gradient Langevin Dynamics  
- **ASGLD** — Adaptive SGLD with Adam-like moments  
- **SAM-SGLD** — Sharpness-Aware SGLD with adversarial perturbations  
- **SAM-SGLD Rank-1** — Directional low-rank noise variant

## Example Workflow
1. **(Optional) Pretrain**  
   Set `train.epochs > 0` to get a good initialization near a high-probability region.

2. **Sample**  
   Choose a sampler via the config (`sampler.type`) and run `scripts/train.py` to generate `sample_XXXX.pth` files under `out.dir`.

3. **Evaluate**  
   Use `scripts/eval.py` for single-sample or K-sample ensembles (accuracy, calibration, etc.).

4. **Tune/Repeat**  
   Adjust `step_size`, `burn_in`, `thin`, and (for SAM variants) perturbation radius to trade off mixing vs. bias.


## TODO
- Implement LORA experiments with transformers