# Experiment Outline: SGLD for Bayesian LoRA on MRPC

---

## Objective
Evaluate whether **SGLD sampling in the LoRA subspace** produces a well-mixed Bayesian posterior that improves calibration (ECE, NLL) while maintaining accuracy, compared to MAP-trained LoRA.

---

## Model & Data
- **Backbone**: RoBERTa-base (HuggingFace).
- **Dataset**: GLUE MRPC (train/dev splits).
- **Max sequence length**: 256.
- **Tokenizer**: RoBERTa tokenizer.

### LoRA Configuration
- Inject into: q, k, v, o projections (all layers).
- Rank (**r**): 8.
- Scaling (**α**): 16.
- Dropout: 0.05.

---

## Methods
### 1. MAP-LoRA (Baseline)
- Optimizer: AdamW (β1=0.9, β2=0.999, weight decay=0.01).
- Learning rate: 5e-5.
- Batch size: 32.
- Epochs: 20.
- Scheduler: linear (10% warmup).
- Trainables: LoRA A/B only; base frozen.

### 2. SGLD-LoRA (Bayesian)
- Trainables: LoRA A/B only; base frozen.
- Loss: minibatch NLL + Gaussian prior
  - Prior: \( \tfrac{1}{2\sigma^2}\|\Delta W\|_2^2 \), with \(\sigma=0.1\).
- Update rule:
  \[ \theta \leftarrow \theta - \eta_t g + \sqrt{2\eta_t/\tau}\,\xi_t, \quad \xi_t \sim \mathcal{N}(0,I) \]
- Step size schedule: \(\eta_t = \eta_0 (1 + t/1000)^{-0.6}\), \(\eta_0 = 1e-4\).
- Temperature: τ = 1.0.
- Gradient clipping: global norm ≤ 1.0.
- Batch size: 32.
- Chains: 4 independent chains.
- Burn-in: 2,000 update steps (discard).
- Sampling: 10,000 update steps.
- Thinning: keep every 20th sample → 500 total; retain 200 for evaluation.
- Ensemble: average predictive probabilities across samples from all chains.

---

## Metrics
### Predictive Performance
- **Accuracy** (extrinsic).
- **Negative Log-Likelihood (NLL)** (intrinsic):
  \[ -\tfrac{1}{N}\sum_{i=1}^N \log p_{ens}(y_i \mid x_i) \]
- **Expected Calibration Error (ECE)**:
  - 15 equal-width bins.
  - \( \text{ECE} = \sum_m \tfrac{|B_m|}{N}\, |\text{acc}(B_m) - \text{conf}(B_m)| \).

### MCMC Diagnostics (SGLD only)
Probe set: fixed 512 examples from MRPC dev.

- **Summaries to track**:
  - S1: log posterior (−NLL_probe − log prior).
  - S2: \(\|\Delta W\|_2\).

- **R-hat (per summary)**:
  - Compute within- and between-chain variance.
  - \( \hat{R} = \sqrt{\hat{V}/W} \).
  - Target: ≤ 1.05.

- **Effective Sample Size (ESS, per summary)**:
  - Autocorrelation-based estimate.
  - Target: ≥ 200.

### Stability Check
- Repeat one chain with step size \(\eta_0/2\).
- Confirm NLL and ECE remain stable → guards against discretization bias.

---

## Deliverables
- **Metrics Table**:
  | Method       | Accuracy ↑ | NLL ↓ | ECE ↓ | R-hat (S1) ↓ | R-hat (S2) ↓ | ESS (S1) ↑ | ESS (S2) ↑ |
  |--------------|------------|-------|-------|--------------|--------------|------------|------------|
  | MAP-LoRA     |            |       |       |     –        |     –        |    –       |    –       |
  | SGLD-LoRA    |            |       |       |              |              |            |            |

- **Reliability Diagram**: MAP vs SGLD.
- **Trace Plots**: S1 and S2 across chains.
- **ACF Plots**: for S1, S2.

---

## Success Criteria
- Accuracy within ±0.3% of MAP-LoRA.
- NLL and ECE lower than MAP-LoRA.
- R-hat ≤ 1.05 and ESS ≥ 200 for S1 and S2.
- η/2 test shows stable predictive metrics.

