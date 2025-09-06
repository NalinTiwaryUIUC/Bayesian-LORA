# Convergence Metrics for SGLD–LoRA (MRPC / RoBERTa‑base)

This document specifies **exactly what convergence metrics to track** for SGLD sampling in the LoRA subspace, and **how to compute them**. It is designed to be copy‑pasted into your repo as a reference.

---

## Scope & Setup

* **Setting:** LoRA‑only finetuning on MRPC with RoBERTa‑base; base weights frozen.
* **Chains:** $K = 4$ independent SGLD chains (different seeds & data orders).
* **Kept draws:** After burn‑in & thinning, retain $T$ draws per chain (typical: $T = 200$); total retained $M = K\,T$.
* **Probe set:** Fixed subset of **512 MRPC dev** examples used to compute **predictive summaries** consistently over time (same subset for all steps & chains).

---

## Summaries to Monitor (per retained draw)

Define two core scalar summaries; add more later if desired.

1. **S1: Log posterior on probe**

   $$
   S1_t = -\,\mathrm{NLL}_{\text{probe}}(\theta_t)\; -\; \log \mathrm{Prior}(\theta_t),
   \quad \text{with}\quad \log \mathrm{Prior}(\theta) = -\tfrac{1}{2\sigma^2}\,\lVert \Delta W(\theta) \rVert_2^2 + C.
   $$

   Here $\Delta W$ denotes concatenated LoRA deltas across all targeted modules.

2. **S2: Parameter scale**

   $$
   S2_t = \lVert \Delta W(\theta_t) \rVert_2.\
   $$

> **Why these?** S1 is a **predictive** summary (more robust under multi‑modality), S2 is a **parameter‑space** summary (sensitive to stickiness). Together they capture both views.

Optional stress‑test: add **10 random projections** $S^{(j)}_t = u_j^\top \operatorname{vec}(\Delta W(\theta_t))$ with $u_j \sim \mathcal N(0, I)$. Report min/median/5th‑pct ESS across all summaries if you include these.

---

## Rank‑Normalized, Split‑Chain $\hat R$ (Gelman–Rubin)

**Goal:** Cross‑chain agreement. **Target:** $\hat R \le 1.05$.

### Procedure (per summary S ∈ {S1, S2})

1. **Collect series:** For each chain $k\in\{1,\dots,K\}$, obtain the length‑$T$ post‑burn‑in, thinned series $\{S_{k,t}\}_{t=1}^T$.
2. **Rank‑normalize:** Pool all $K\,T$ values, convert each to its rank $r\in\{1,\dots,KT\}$, map to $(0,1)$ via $r/(KT+1)$, then apply the standard normal inverse CDF $\Phi^{-1}(\cdot)$.
3. **Split chains:** Cut each chain’s series in half → $2K$ half‑chains of length $T/2$.
4. **Within/between variances:** For half‑chains indexed by $m=1,\dots,2K$ with means $\bar S_m$ and variances $s_m^2$:

   $$
   W = \frac{1}{2K}\sum_{m=1}^{2K} s_m^2,\quad
   B = \frac{T/2}{2K-1}\sum_{m=1}^{2K} (\bar S_m - \bar S_{\cdot})^2,\quad
   \bar S_{\cdot} = \frac{1}{2K}\sum_m \bar S_m.
   $$
5. **Variance estimate & R‑hat:**

   $$
   \hat V = \frac{T/2 - 1}{T/2} W + \frac{1}{T/2} B,\qquad
   \boxed{\hat R = \sqrt{\hat V / W}}.
   $$
6. **Windowed $\hat R$:** Recompute on prefixes (25/50/75/100% of draws) to verify a trend $\downarrow 1$.

**Interpretation & Actions**

* $\hat R \le 1.05$: good.
* $1.05{-}1.10$: extend sampling and/or reduce initial step size $\eta_0$ (×0.5–0.67); consider more burn‑in.
* $>1.10$: likely non‑mixing or persistent multi‑modality in predictions; reduce $\eta_0$, warm‑start closer, or try pSGLD (diag precond.).

---

## Rank‑Normalized, Split **ESS** (bulk‑ESS)

**Goal:** Effective number of independent draws. **Target:** **ESS ≥ 200** per summary.

### Procedure (per summary S ∈ {S1, S2})

1. Use the **same rank‑normalized, split** half‑chain series from $\hat R$ (length $T' = T/2$ per half‑chain).
2. **Autocorrelations:** For each half‑chain, compute lag‑$\ell$ autocorrelations $\hat\rho_\ell$ for $\ell = 1,2,\dots$ (FFT‑based autocovariance recommended; unbiased at lag 0).
3. **Geyer IPS truncation:** Find first odd lag $m$ where $\hat\rho_{m-1} + \hat\rho_m < 0$; truncate the sum at $m-1$.
4. **Integrated autocorrelation time:**

   $$
   \tau_{\text{int}} = 1 + 2 \sum_{\ell=1}^{m-1} \hat\rho_\ell.
   $$
5. **Combine half‑chains** (e.g., via averaging $\tau_{\text{int}}$ weighted by within‑chain variance), then compute

   $$
   \boxed{\mathrm{ESS} = \frac{M}{\tau_{\text{int}}}},\quad M = K\,T\; (\text{total retained draws before splitting}).
   $$
6. **Report:** ESS for **S1** and **S2**; optionally **ESS/sec** using wall‑clock for the sampling phase only.

**Interpretation & Actions**

* S1 typically achieves higher ESS than S2 (predictive vs parameter‑space).
* If ESS < 200: increase total updates; or reduce $\eta_0$ \~1.5×; or use pSGLD; ensure batch $\approx 32$ (very large batches hurt mixing).

**Optional aggregate reporting** (if tracking >2 summaries): report **min ESS**, **median ESS**, and **5th‑percentile ESS** across summaries; pair with $\hat R$ for context.

---

## Posterior Predictive Stability (Accuracy / NLL / ECE vs #samples)

**Goal:** Verify convergence of **predictive metrics** as more samples are aggregated.

### Procedure

For prefixes of retained samples (e.g., 10%, 25%, 50%, 100% of $M$):

1. Form the **ensemble predictive** by **averaging class probabilities** over samples (never average logits).
2. Compute on the full MRPC dev set:

   * **Accuracy** (% correct),
   * **NLL**: $-\tfrac{1}{N}\sum_i \log p_{\text{ens}}(y_i\mid x_i)$,
   * **ECE** with **15 equal‑width bins** over confidence $\hat p = \max_c p_{\text{ens}}(y=c\mid x)$:

     $$
     \mathrm{ECE} = \sum_{m=1}^{15} \frac{|B_m|}{N}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|.
     $$
3. Plot each metric against #samples. Curves should **stabilize** (flat after \~100–200 samples).

**Action:** If curves drift with more samples → add burn‑in, reduce $\eta_0$, or increase total sampling.

---

## Discretization‑Bias Check (SGLD Sanity)

**Goal:** Ensure conclusions aren’t artifacts of step size.

### Procedure

* Duplicate one configuration with **$\eta_0/2$**, keeping all else fixed (same #kept samples).
* Compare final **NLL**/**ECE** on dev; pass if changes are small (e.g., $|\Delta\mathrm{NLL}| < 0.01$, $|\Delta\mathrm{ECE}| < 0.005$).

---

## Optional: Predictive Overlap Across Chains

**Goal:** Detect multi‑modal **predictive** behavior (parameters can be multi‑modal without harm).

### Procedure

* For each chain’s own ensemble (averaging its retained samples), compute pairwise **KL divergence** or **Wasserstein‑1** between predictive distributions on the **probe set**; report mean and 95th percentile.
* Low distances ⇒ chains agree **predictively**, even if parameter values differ.

---

## What to Log per Kept Draw

* Chain ID, step index, current $\eta_t$.
* **S1**, **S2** (and any extra projections) on the **probe set**.
* Dev **NLL** on probe (cheap); optionally full dev every N draws.
* Wall‑clock timestamp (for **ESS/sec**).

---

## Pass / Fail Criteria

* **$\hat R \le 1.05$** for **S1** and **S2**.
* **ESS ≥ 200** for **S1** and **S2** (if reporting additional summaries, ensure **min ESS ≥ 200** or document exceptions with stability justification).
* **Posterior predictive**: Accuracy within **±0.3%** of MAP‑LoRA; **NLL** and **ECE** **lower** than MAP‑LoRA.
* **Discretization check** ($\eta_0/2$) shows negligible changes in NLL/ECE.

---

## Notes & Conventions

* Always **average probabilities** across samples when computing ensemble predictions.
* Use **rank‑normalized, split** definitions for both $\hat R$ and ESS (per Vehtari et al.).
* When adding extra summaries (random projections), report **min/median/5th‑pct ESS** alongside individual S1/S2 values.
