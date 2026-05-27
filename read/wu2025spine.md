# SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization

> **Added to survey on:** 2026-03-11

> arXiv 2511.17938, Nov 2025
> Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, Jianfei Cai
> Monash University, Imperial College London

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| SPINE | Policy Opt. | test-time | Token |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | benchmark cohort / rollout group |
| Persistence | full parameter accumulate across test-time episodes; no per-sample reset |
| Inference Coupling | adapt within the cohort, then infer/evaluate with the updated model |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | protocol-inferred |
| Timing Regime (auxiliary taxonomy) | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** updates fire iteratively per rollout group / mini-batch on the target benchmark or its unlabeled cohort — not sample-local adapt-and-reset.
- **Whose sample it serves:** updates produced by the current sample or mini-batch mainly serve subsequent rounds within the same cohort and the final evaluation, not the current sample alone.
- **Whether parameters/state accumulate:** parameters accumulate throughout the test-time RL run; typically re-initialized only when switching to a new benchmark, model run, or independent experiment.
- **Reset boundary:** so it is closer to a cohort-level cumulative TTA than an arrival-by-arrival streaming-reset protocol.

## 1. UPT Assignment Rationale

**Family II — Sample-Relation Supervision (population consensus)**

> **Reclassification note (2026-05-19):** previously classified under Family I (then Prediction-Statistic Optimization). After closer review of the reward construction, the dominant-artifact rule reassigns it to Family II: SPINE's reward proper is the self-consistency majority vote (`r=1[y=maj]`), identical to TTRL / RoiRL / ECHO; the entropy band serves only for token-level selection (deciding which "forking tokens" receive gradients) — a selection mask, not the reward itself. The dominant artifact is therefore the majority-vote statistic across rollouts.

SPINE differs from TTRL / RoiRL not in the reward signal itself but in **gradient routing**:

- **Reward source:** majority-voting pseudo-reward — the reward proper, identical to TTRL.
- **Innovation (secondary mechanism):** token-level selective updates — only high-entropy "forking tokens" (reasoning branch points) receive gradient updates; low-entropy "flowing tokens" stay unchanged. Entropy-band regularization further constrains forking-token entropy to a reasonable range.
- **Policy-optimization carrier:** GRPO-style policy-optimization objective.
- **No external labels:** relies on no ground-truth labels, external reward models, or human annotation.

---

## 2. Problem Addressed

Standard Test-Time Reinforcement Learning (TTRL) uses self-consistency majority voting as the pseudo-reward and updates the policy uniformly over all tokens. In practice, this causes **characteristic collapse**:

1. **Majority-vote-reward saturation:** the majority-vote reward grows rapidly to 1.0, yet Pass@1 accuracy drops.
2. **Response length shrinks sharply:** the model tends to produce short, consistent-but-wrong answers.
3. **Loss of reasoning diversity:** the policy over-optimizes pseudo-consensus rather than true correctness.

The authors attribute these issues to **the fundamental flaw of updating all tokens uniformly:** the vast majority of CoT tokens are low-entropy "flowing tokens" (~80%); only the ~20% high-entropy "forking tokens" actually decide reasoning branch directions. Gradient updates on low-entropy tokens are redundant and may be harmful.

---

## 3. Method

SPINE (**S**elective **P**olicy **I**mprovements at **N**odes of **E**ntropy) has two core modules:

### 3.1 Self-consistency reward and GRPO objective

- For each input x, sample N responses {y_i} and obtain the consensus answer y* via majority vote.
- Each response receives a rule-based reward r_i = r(y_i, y*) (exact match, code unit tests, etc.).
- Optionally use the leave-one-out variant y*_{-i} to mitigate self-inclusion bias.
- Use GRPO to compute the group-wise normalized advantage: A_hat_i = (r_i − mean) / (std + epsilon).
- Update the policy with the standard clipped PPO surrogate objective.

### 3.2 Forking-token selection

- Compute the predictive entropy at each token position t: H_t = −∑ pi(v|s_t) log pi(v|s_t).
- Select the top **20%** highest-entropy tokens as forking tokens, marked by a binary mask m_t.
- Gradients only update forking tokens; flowing tokens have their gradients stopped (stop-gradient).
- Simultaneously apply a **masked KL divergence** constraint on forking tokens (anchored at the pre-adaptation model) to prevent policy drift at high-uncertainty positions.

### 3.3 Entropy-band regularization (quantile band-pass regularization)

- For each sample i's forking-token set S_i, compute two quantile thresholds:
  - H_low = 10% quantile (lower bound).
  - H_high = 50% quantile (upper bound = median).
- Use hinge losses to constrain forking-token entropy:
  - l_high = max(0, H_t − H_high): suppress excessive entropy (reduce noisy supervision).
  - l_low = max(0, H_low − H_t): prevent entropy from collapsing (preserve exploration).
- Quantile thresholds are computed adaptively per sample, with no task-specific tuning.

### 3.4 Overall objective

L = L_core + R_band

where L_core = −E[m_t · l_PPO] + lambda_KL · l_KL^fork, and R_band is the entropy-band regularization term.

### Test-time update procedure

For each unlabeled input: (i) sample N=8 responses; (ii) form a pseudo-label by majority vote and compute the reward; (iii) compute the advantage; (iv) compute token entropies, select forking tokens (top 20%), and compute quantile thresholds; (v) minimize the total objective L to update parameters θ. Iterate a few rounds on the test split.

---

## 4. Datasets

The paper evaluates on **10 benchmarks** across 4 task families:

| Task type | Datasets |
|---------|--------|
| Multimodal VQA | MathVision, SLAKE, MedXpertQA-MM |
| Math reasoning | AIME 2025, AMC, MATH-500 |
| General / expert QA | GPQA, MMLU |
| Medical QA | MedQA (USMLE), PubMedQA |

**Base models:**
- Qwen2.5-VL-3B-Instruct (multimodal).
- Qwen3-1.7B (general text).
- Qwen2.5-Math-1.5B (math-specialized).

All experiments run on 4 × NVIDIA A100 80GB with the EasyR1 framework.

---

## 5. Evaluation metrics and main results

**Metric:** Pass@1 accuracy (greedy decoding, temperature=0); matched after standard normalization, case folding, LaTeX normalization, and SymPy algebraic-equivalence checks.

### Main results

**Multimodal VQA (Qwen2.5-VL-3B-Instruct):**

| Method | MathVision | SLAKE | MedXpertQA-MM | MedQA | PubMedQA | Avg |
|------|-----------|-------|--------------|-------|----------|-----|
| No adaptation | 19.65 | 26.17 | 17.17 | 30.40 | 68.00 | 32.28 |
| TTRL | 22.73 | 30.00 | 22.61 | 51.88 | 71.50 | 39.74 |
| **SPINE** | **27.26** | **38.66** | **23.92** | **55.40** | **76.20** | **44.29** |
| Delta vs TTRL | +4.5 | +8.7 | +1.3 | +3.5 | +4.7 | +4.6 |

**Math reasoning (Qwen2.5-Math-1.5B):**

| Method | AIME 2025 | AMC | MATH-500 | GPQA | Avg |
|------|----------|-----|----------|------|-----|
| No adaptation | 10.00 | 28.91 | 30.20 | 4.06 | 18.29 |
| TTRL | 16.67 | 49.88 | 66.42 | 25.38 | 39.59 |
| **SPINE** | **20.00** | **59.03** | **77.00** | **30.96** | **46.75** |
| Delta vs TTRL | +3.3 | +9.2 | +10.6 | +5.6 | +7.2 |

**Math reasoning (Qwen3-1.7B):**

| Method | AIME 2025 | AMC | MATH-500 | GPQA | MMLU | Avg |
|------|----------|-----|----------|------|------|-----|
| TTRL | 26.67 | 53.01 | 79.86 | 29.94 | 71.19 | 52.13 |
| **SPINE** | **36.67** | **61.46** | **81.40** | **36.04** | **72.66** | **57.65** |
| Delta | +10.0 | +8.5 | +1.5 | +6.1 | +1.5 | +5.5 |

### Key findings

1. **SPINE consistently beats TTRL on all 10 benchmarks**, with an average gain of 4.6–7.2 points.
2. **More stable training dynamics:** SPINE avoids TTRL's reward saturation, response shrinkage, and entropy explosion.
3. **Good cross-task generalization:** adapting on AIME 2025 yields positive transfer to AMC, MATH-500, and GPQA (Table 3, average rises from 32.31 to 46.96).
4. **Ablation shows two modules are complementary:** forking-token selection contributes +3.2 (from 27.56 to 30.76); the entropy band adds another +3.2 (to 33.99), totaling +15.7 over the base.
5. **Clear advantage over SFT-based methods (LMSI, SEALONG)**, which can even underperform the no-adaptation baseline on some benchmarks.
