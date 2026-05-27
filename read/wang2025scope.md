# SCOPE: Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning

> **Added to survey on:** 2026-04-14

**Paper:** arXiv 2512.15146v2
**Authors:** Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang
**Date:** 2025-12-18

| Method | Carrier | Regime | Level |
|------|----------------|-------------|------------|
| SCOPE | Policy Opt. | test-time | Traj. |

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

**Family II (Sample-Relation Supervision, population consensus / subgroup consensus).**

SCOPE remains a strict UPT method for the following reasons:

- **A real update-bearing optimization exists:** the method updates policy parameters with GRPO at test time; it is not merely search or reranking.
- **The supervision signal comes from the model's own rollouts and token probabilities:** pseudo-labels come from multiple model-generated answers under the same query, plus step-wise confidence derived from the model's own decoding distribution.
- **No external ground truth / verifier / tool / environment reward:** no human labels, no external reward model, no code executor, no environment returns, no external teacher.
- **The core is not minimizing the uncertainty of a single sequence but reconstructing sample-relation supervision:** it rebuilds TTRL's single majority vote as confidence-weighted consensus + subgroup-local consensus, placing it closer to Family II's Sample-Relation Supervision than Family I's Prediction-Statistic Optimization.

---

## 2. Problem Addressed

SCOPE targets two issues with TTRL-style methods:

- **Confirmation bias:** majority voting treats all rollouts as equal weight; wrong but frequent answers get falsely reinforced.
- **Reward sparsity:** the whole rollout group corresponds to a single global consensus label, making supervision coarse and exploration limited.

The paper's core idea: on one hand, use step-wise confidence to distinguish "high-quality, high-confidence reasoning paths" from "answers that happen to be frequent"; on the other hand, split the entire rollout group into subgroups and produce local consensus for each subgroup, increasing reward density and exploration diversity.

---

## 3. Method

### 3.1 Step-wise confidence

- For each token, compute the average negative log of the top-k token probabilities to obtain a token confidence.
- Aggregate by reasoning step, then average over the full response to obtain the **average step confidence**.
- Replace naive vote counts with this confidence and perform **confidence-weighted voting** over candidate answers to produce the global pseudo-label.

### 3.2 Subgroup-specific reward

- Partition all generations under the same query into several subgroups.
- For each subgroup, instead of voting within the subgroup directly, perform bootstrap sampling from the global rollout pool and use confidence-weighted voting to obtain the subgroup's local consensus.
- Each rollout's reward is determined by whether it matches its subgroup's local consensus, providing finer-grained supervision than TTRL's single global label.

### 3.3 Automatic subgroup-size selection

- For candidate subgroup sizes, simultaneously measure:
  - **Quality rate:** the level of agreement between outputs and the subgroup consensus.
  - **Exploration rate:** the proportion of subgroups whose consensus differs.
- Use a Pareto front + trade-off distance to automatically pick the subgroup size, balancing correctness and diversity.

### 3.4 Optimization

- Use GRPO to update the policy at test time.
- It is therefore not a mere reward redesign but a complete test-time RL self-bootstrapping loop.

---

## 4. Datasets

- **AIME 2024**
- **AIME 2025**
- **AMC**
- **MATH-500**

Evaluated models:

- Qwen2.5-Math-1.5B.
- Qwen3-1.7B.
- Llama-3.1-8B-Instruct.
- Qwen3-8B.

---

## 5. Evaluation metrics and main results

The paper's main table shows SCOPE outperforms TTRL and several unsupervised RL baselines across all models.

Representative results:

- **Qwen2.5-Math-1.5B:** average rises from **36.95** to **41.36**; AIME 2024 from **16.48** to **22.50**.
- **Qwen3-1.7B:** average rises from **41.91** to **44.02**.
- **Qwen3-8B:** average rises from **58.21** to **62.20**; AIME 2024 from **47.13** to **52.70**; AIME 2025 from **27.40** to **31.00**.
- **Llama-3.1-8B-Instruct:** average rises from **26.38** to **28.18**; although it falls slightly on the easier MATH-500, gains on harder competition problems are more pronounced.

Ablations also support the two core components:

- Removing **confidence weighting** causes a clear regression, indicating naive majority voting is still the bottleneck.
- Removing **subgroup partitioning** also causes a clear regression, indicating finer-grained rewards are genuinely important for exploration.
