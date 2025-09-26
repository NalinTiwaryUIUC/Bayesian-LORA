# SAM-SGLD (Single-Chain, 1.01M Steps) — Experiment Spec

Single-chain Bayesian-LoRA using **SAM-SGLD** with **constant step size and constant SAM radius during sampling**, thinned saving, online **ESS (OBM)**, **per-sample metrics (accuracy/NLL/ECE)**, and **cumulative-ensemble** reporting at fixed sampling-step milestones.

---

## 0) Goals & Outputs

**Goals**
- Track **ESS/step** vs sampling steps at 1k, 10k, 100k, 1,000k (SGD steps, post-burn-in).
- Quantify dispersion of **per-sample accuracy/NLL/ECE** and report **MCSE** via ESS.
- Plot **cumulative-K ensemble** convergence for accuracy/NLL/ECE.

**Primary outputs**
- Milestone tables: **ESS (kept units), ESS/step, mean±SD & MCSE** for accuracy/NLL/ECE; cumulative-K curves.
- Lightweight parameter checkpoints (optional) and scalar logs sufficient to recompute all summaries.

---

## 1) Setup & Assumptions

- **Model**: Baseline network with **LoRA adapters** (no preconditioning).
- **Warm-start**: AdamW to a high-density region.
- **Burn-in**: Decaying step size; samples during burn-in are **not** used for ESS/metrics.
- **Sampling**: **Constant** SAM-SGLD step size and **constant** SAM radius.
- **Thinning**: Save one sample every fixed number of SGD steps (you choose).
- **Milestones**: Sampling steps at {1k, 10k, 100k, 1,000k}. Convert to kept-sample counts by dividing by the thinning interval.

---

## 2) Configuration (description only)

- **Data**: dataset, split, evaluation loader settings (you choose).
- **Optimization**:
  - Warm-start schedule (you choose).
  - Burn-in length and decay schedule (you choose).
  - Sampling phase: constant step size and constant SAM radius (you choose).
  - SAM inner step: norm type and number of ascent steps (you choose; typically 1, ℓ2).
- **Thinning**: interval in SGD steps (you choose).
- **Evaluation**:
  - Whether to evaluate per-sample metrics on the full validation set or a fixed, large subset.
  - Whether and how often to save parameter checkpoints (optional).
- **Diagnostics**:
  - Scalars to drive ESS: choose a small set (5–10) such as log posterior, parameter L2 norm, Frobenius norms of a few LoRA blocks, and a few fixed random 1-D projections of the LoRA parameter vector.
  - Online estimators for metrics (Welford) and ESS (OBM) enabled.

---

## 3) Run Pipeline

1. **Warm-start** with AdamW to your usual criterion.
2. **Burn-in** (decay step size; constant or separate SAM radius is fine). Do **not** include these samples in ESS/metrics.
3. **Freeze** to **constant step size and SAM radius** for the entire sampling phase.
4. **Sampling**:
   - Perform SAM-SGLD updates every SGD step.
   - **Thinning**: every *thinning interval* steps, emit a **kept sample**.
   - For each kept sample:
     - Compute **per-sample accuracy, NLL, ECE** (on full val set or fixed subset).
     - Update **online (Welford) mean/variance** for each metric.
     - Update **OBM** state for each chosen scalar (for ESS).
     - Update **cumulative-ensemble** metrics (add this sample to the ensemble).
     - Optionally save a sparse parameter checkpoint.
5. **Milestones** (at specified sampling steps):
   - Convert sampling steps → kept-sample counts by dividing by thinning interval.
   - Emit summaries: **ESS, ESS/step, τ̂**, **mean±SD & MCSE** for metrics, and cumulative-K curves up to that point.

---

## 4) Online ESS (OBM) & MCSE

**Scalars for ESS** (post burn-in, on kept samples):
- Log posterior (up to constant),
- Global parameter L2 norm,
- Frobenius norms for a few LoRA blocks,
- A few fixed random 1-D projections of the LoRA parameter vector.

**OBM (per scalar, online)**
- Maintain running **marginal variance** of the scalar (Welford).
- Maintain **overlapping block means** with block size `b` that grows slowly with kept-sample count `m` (e.g., `b = floor(m^0.5)`).
- Let `Var_hat(mean)` be the OBM estimate of the variance of the sample mean; let `s2` be the marginal variance.
- **IACT**: `tau_hat = m * Var_hat(mean) / s2`.
- **ESS (kept units)**: `ESS = m / tau_hat`.
- **ESS per SGD step**: `ESS_per_step = ESS / (m * thinning_interval)`.

**MCSE for metrics**
- For each per-sample metric sequence, maintain Welford mean/variance and OBM-based ESS.
- **MCSE of the mean** ≈ `sqrt( Var(y) / ESS_y )`.

---

## 5) Logging Schema (quantities only)

For **each kept sample** (post burn-in), log:
- **Iteration indices**: total SGD step, kept-sample index.
- **Per-sample evaluation metrics**: accuracy, NLL, ECE (on the chosen validation set).
- **Cumulative-ensemble metrics**: ensemble accuracy, NLL, ECE using all kept samples so far.
- **Tracked scalars for ESS**: values of selected diagnostics (e.g., log posterior, norms, projections).
- **Online estimator states (optional to log each time)**: current running means/variances for metrics; current block size and counts for OBM per scalar.
- **Sampler context** (for provenance): current constant step size, SAM radius, and optionally gradient/SAM perturbation norms.

At **milestones**, log summaries:
- **For each ESS scalar**: ESS (kept), ESS/step, and estimated IACT.
- **For each evaluation metric**: mean, standard deviation (across single samples), and **MCSE**.
- **Cumulative-K ensemble** snapshots or series up to that milestone.

---

## 6) Quick Functional Tests — *Only* for Rolling Metrics

> These tests are fast, minimal, and verify that **rolling per-sample metrics and ESS** work end-to-end. No testing of burn-in, milestones, or other trivial plumbing.

### (a) Welford (running mean/variance) sanity
- **What**: Stream a constant scalar sequence and a known-variance sequence through the metric accumulator.
- **Pass if**: Running mean matches the true mean; variance tends to 0 for the constant stream and to the known variance for the noisy stream (within numerical tolerance).

### (b) OBM-based ESS on a controlled dependent stream
- **What**: Feed an **AR(1)** synthetic scalar sequence with known autocorrelation into the OBM estimator.
- **Pass if**: Estimated IACT is close to `(1 + rho) / (1 - rho)` for sufficiently many samples; ESS ≈ `m / IACT` (within ~10–20% after a few thousand points).

### (c) Per-sample metric correctness on fixed predictions
- **What**: Provide fixed logits/probabilities and labels to the evaluation routines as if they were one kept sample (repeatable input).
- **Pass if**: Reported accuracy, NLL, and ECE equal the analytically computed ground truth for those logits.

### (d) Cumulative-ensemble update logic
- **What**: Use two synthetic “samples” with complementary predictions (e.g., one confident-correct, one near-uniform). Feed them in sequence.
- **Pass if**: The cumulative ensemble metric updates are consistent (e.g., ensemble NLL/ECE improves when adding the confident-correct sample to the uniform one), and the cumulative value after K samples equals the recomputed ensemble from the first K samples.

### (e) End-to-end rolling test on a tiny run
- **What**: Run a very short sampling phase with frequent kept samples on a small validation subset.
- **Pass if**: Per-sample metrics and OBM/ESS update at each kept sample; milestone summaries (when triggered) reflect exactly the rolling state (no recomputation from history required).