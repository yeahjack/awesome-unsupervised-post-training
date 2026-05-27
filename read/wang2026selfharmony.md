# Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning

> **Added to survey on:** 2026-03-11

> **Method:** Self-Harmony | **Carrier:** Policy Opt. | **Regime:** Test-time | **Level:** Semantic
>
> Ru Wang, Wei Huang, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo (U of Tokyo, RIKEN, ISM, Google Research Australia)
>
> Published as a conference paper at ICLR 2026. arXiv: 2511.01191v2

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

Self-Harmony relies on no external ground-truth labels, no external reward model, and no human annotation. Its core supervision signal comes from the relation between the model's own answer-frequency distributions under two "views" — the original question and the model's self-generated paraphrase:

- **Paraphrase consistency:** the same model performs rollouts on the original question and on its self-generated paraphrase; correct answers should appear stably under both views, while wrong answers tend to be frequent under only one phrasing.
- **Self-supervision:** the same model plays both Solver and Reframer roles via prompt switching, with no external model.
- **Self-play:** Solver and Reframer cooperate in self-play — the Solver learns to align with the harmonic-mean pseudo-label, the Reframer learns to produce paraphrases that expose the Solver's bias.

The final pseudo-label is selected by the **harmonic mean score (HMS)** aggregating answer frequencies across the two views — a relational intrinsic reward: it rewards answers that appear consistently under both distributions and penalizes spurious answers popular under only one view. The whole process is population-level sample-relation internal supervision.

---

## 2. Problem Addressed

**The fragility of pseudo-label selection in test-time reinforcement learning (TTRL).**

- In TTRL settings, the model must self-adapt at inference using only unlabeled test data; pseudo-label quality directly determines training stability and final performance.
- Mainstream methods use **majority voting** as the pseudo-label selection strategy. However, when the model has a systematic reasoning bias (i.e. p(Correct | x) < p(Wrong | x)), majority voting amplifies the error — as the number of samples grows, the probability of recovering the correct answer approaches zero (the "majority-vote trap").
- Existing alternatives (using an external reward model or a strong LLM to verify) violate the test-time self-contained principle.
- Self-Harmony proposes a fully unsupervised solution: combining paraphrase consistency and harmonic-mean aggregation to build a pseudo-label more robust than majority voting.

---

## 3. Method

### 3.1 Core intuition

Correct answers should remain stable across semantically equivalent but differently phrased questions, while wrong answers tend to depend on a specific surface phrasing (style-sensitive). Therefore, pseudo-labels should not be based on popularity under a single view but on invariance across views.

### 3.2 Theoretical foundation: from majority voting to harmonic mean

- Propose the **View-Invariance Assumption:** for semantically equivalent queries x and x', the generation probability of the correct label C is approximately invariant (p(A=C|x) ≈ p(A=C|x')), while the probability of incorrect labels varies across views.
- Define the **View-Invariant Infomax objective:** J_λ(a) = I(Z_a; A) − λI(Z_a; X), maximizing answer information while penalizing view dependence.
- **Theorem 3.2** proves: at λ=2, the second-order approximate optimum of this objective is exactly the harmonic mean:

  y* = arg max_a  2p₀(a)p₁(a) / (p₀(a) + p₁(a))

  where p₀(a) and p₁(a) are the empirical frequencies of answer a under the original and paraphrased questions respectively.

### 3.3 Self-Harmony framework

**One model, two roles:**
- **Solver (π_θ):** generates N rollout answers {y_i} for the original question x.
- **Reframer (ρ_θ):** rewrites the original question into a semantically equivalent paraphrase x' (via prompt switching; shared parameters).
- The Solver then generates N rollout answers {y'_i} for the paraphrased question x'.

**Pseudo-label selection:** compute each candidate answer a's empirical frequencies p̂₀(a) and p̂₁(a) under the two views, and pick the pseudo-label by HMS: y* = arg max_a HMS(p̂₀(a), p̂₁(a)).

**Practical implementation optimization:** fuse "reframe → solve" into a single generation action (the model first rewrites and then solves in one generation), compressing the three-step pipeline into two model calls.

### 3.4 Reward design

- **Solver reward:** R_solve(y) = I[y = y*] (whether the answer matches the pseudo-label).
- **Fused reframe-and-solve reward:** R_fused(y') = (1 − w_f · R^penalty_format(y')) · (1 − w_d · R^penalty_div(y', y)) · I[y' = y*]
  - Format Penalty: penalize structurally invalid paraphrases.
  - Diversity Penalty: based on Jensen-Shannon divergence, penalize paraphrases too similar to the original.
  - Reward is granted only when the answer is correct (correctness serves as the success gate).

**Policy optimization:** use the GRPO objective; compute rewards separately for the original and paraphrased branches and optimize jointly.

---

## 4. Datasets

| Dataset | Type | Description |
|--------|------|------|
| **MATH500** | Math reasoning | Hendrycks et al., 2021; 500 math problems. |
| **GSM8K** | Math reasoning | Cobbe et al., 2021; grade-school math word problems. |
| **AIME 2024** | Math competition | High-difficulty math competition problems. |
| **AMC** | Math competition | American math competition problems. |
| **GPQA-Diamond** | Multi-discipline | Rein et al., 2023; graduate-level science Q&A. |
| **MMLU-Pro** | Multi-task | Wang et al., 2024; multi-domain multi-task reasoning. |

All experiments are in the label-free setting: each dataset trains a separate model, and ground-truth labels are never used to produce pseudo-labels or compute correctness signals during training.

---

## 5. Evaluation metrics and main results

### Metrics
- **pass@1 accuracy** (mean pass@1 over 16 rollouts).
- Training stability (zero training failure).
- Pseudo-label quality (accuracy, F1-score, Spearman correlation).

### Main results (Table 1)

**Five base models:** Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct.

**Baselines:** Intuitor, Rent, Majority-Voting (TTRL), Co-Reward.

**Key numbers:**

| Model + Dataset | Before RL | Self-Harmony | Gain |
|---|---|---|---|
| Qwen3-1.7B + MATH500 | 42.70 | **69.60** | +26.9 |
| Qwen3-4B + MATH500 | 60.20 | **78.50** | +18.3 |
| Qwen3-8B + MATH500 | 66.80 | **80.00** | +13.2 |
| Llama-3.1-8B + GSM8K | 95.60 | **91.59** | — |
| Llama-3.1-8B + MATH500 | 41.46 | **50.40** | +8.94 |

- Across 30 configurations (5 models × 6 benchmarks), Self-Harmony **ranks first 28 times and second 2 times**.
- All experiments show **zero training failure** — unprecedented robustness; some baselines degrade significantly after peak.
- Pseudo-label accuracy is consistently the highest (about 80–85% on MATH500 Level 3), substantially beating Co-Reward and TTRL.

### Ablation (Table 2, MATH500)

| Variant | Qwen3-4B | Qwen3-8B |
|---|---|---|
| Self-Harmony (Full) | 78.50 | 79.80 |
| w/o Format Reward | 78.40 | 77.46 |
| w/o Diversity Reward | 78.20 | 78.90 |
| Cross Selection instead of HMS | 76.50 | 78.40 |
| Majority Voting instead of HMS | 77.30 | 79.00 |

All components (HMS, Format Reward, Diversity Reward) contribute to the final performance, validating the necessity of the framework design.
