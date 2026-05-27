# DARE: Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning

> **Added to survey on:** 2026-03-11

**Paper:** Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning
**Authors:** Bodong Du, Xuanqi Huang, Xiaomeng Li (HKUST)
**ArXiv:** 2601.21804
**Date:** 2026-01-30

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| DARE | Policy Opt. | Test-time | Traj. |

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

DARE's core reward signal comes entirely from **empirical-distribution statistics** of the model's own rollouts, with no reliance on external ground truth, verifier, or human annotation. Specifically:

- Given a test query, the policy generates M rollouts; the **empirical frequency** of each candidate answer is recorded together with the **trace-level uncertainty** (mean token-level entropy).
- Rewards are computed from the **distributional relations** of answers within the rollout population: high-frequency, low-uncertainty answers receive high reward; low-frequency but low-uncertainty answers receive an extra exploration bonus.
- The entire signal construction is **population-level relational**: each rollout's reward depends on its distributional relation to all other rollouts of the same query, not on an independent external judgment.

DARE therefore belongs to Family II, building supervision signals from intra-population statistical relations (consensus / distributional structure).

---

## 2. Problem Addressed

Existing Test-Time Reinforcement Learning (TTRL) methods commonly use **Majority Voting (MV)** as the reward estimator: take the modal answer over multiple rollouts as the pseudo-label and assign a binary reward. The paper identifies two fundamental flaws of MV:

1. **Information collapse:** MV compresses the full rollout distribution into a single mode, discarding information from non-modal but correct rollouts. Theorem 2.1 shows that the mutual information carried by the MV reward is strictly smaller than that of the original reward.
2. **Latent-conditioned bias:** when rollouts are positively correlated (driven by a latent variable Z), MV estimates the latent-conditional mode rather than the marginal expected reward, systematically overestimating the correctness of frequent answers and triggering **confirmation collapse** — once a wrong answer is in the majority, it keeps reinforcing itself.

DARE aims to replace point-level consensus with **distribution-aware** reward estimation, providing a more informative and robust learning signal that improves both convergence stability and final performance of test-time adaptation.

---

## 3. Method

The DARE framework has five steps (corresponding to Figure 2 (a)–(e)):

### 3.1 Rollout sampling & uncertainty-aware distribution

For a test query q, the policy $\pi_\theta$ produces M rollouts $\{\tau_1, \dots, \tau_M\}$, each producing a final answer $\hat{y}_i$.

- **Empirical frequency:** $n(\hat{y}) = \sum_{k=1}^{M} \mathbf{1}[\hat{y}_k = \hat{y}]$.
- **Trace-level uncertainty:** average token-level entropy per rollout: $u(\hat{y}) = \frac{1}{n(\hat{y})} \sum_{k: \hat{y}_k = \hat{y}} \frac{1}{|\tau_k|} \sum_{i \in \tau_k} \sum_j -P_i(j) \log P_i(j)$.
- **Uncertainty-aware empirical distribution:** divide frequency by uncertainty and renormalize: $\hat{p}(\hat{y}) = \frac{n(\hat{y})/(u(\hat{y}) + \epsilon)}{\sum_{\hat{y}'} n(\hat{y}')/(u(\hat{y}') + \epsilon)}$.

In this way, high-frequency low-uncertainty answers receive higher probability, reducing preference for frequent but unreliable answers.

### 3.2 Distribution-based reward & exploration bonus

- **Base reward:** use the uncertainty-aware distribution probability directly as the reward: $r_{\text{dis}}(y_i) = \hat{p}(y_i)$.
- **Exploration bonus:** encourage the model to explore low-frequency, low-uncertainty rollouts: $b(y_i) = \left(1 - \frac{n(y_i)}{M}\right) \cdot (1 - u(y_i))$.

  This bonus is largest when the answer frequency is low and uncertainty is low, avoiding amplification of noisy rollouts.
- **Combined reward:** $r(y_i) = r_{\text{dis}}(y_i) + \alpha \, b(y_i)$, with $\alpha \in [0, 1]$ controlling exploration strength.

### 3.3 Distribution support pruning

Remove rollouts whose empirical probability is below a threshold $\tau$, and renormalize the remaining rollouts:

$$\tilde{p}(y_i) = \frac{\hat{p}(y_i) \, \mathbf{1}[\hat{p}(y_i) \geq \tau]}{\sum_{k=1}^{M} \hat{p}(y_k) \, \mathbf{1}[\hat{p}(y_k) \geq \tau]}$$

After pruning, recompute all distribution-dependent statistics ($\tilde{n}$, $\tilde{u}$, $\tilde{b}$); the final reward is $r(y_i) = \tilde{p}(y_i) + \alpha \, \tilde{b}(y_i)$.

This step removes degenerate low-quality rollouts, lowering reward variance and stabilizing optimization.

### 3.4 Test-time policy optimization

Use GRPO to update the policy at test time with the refined rollout-level rewards above. The whole process requires no external labels and depends only on the model's own generated distributional information.

---

## 4. Datasets

The paper evaluates on five benchmarks across three reasoning domains:

| Domain | Dataset | Description |
|------|--------|------|
| General Reasoning | **MMLU-Pro** | Multi-task language understanding (harder version). |
| Mathematical Reasoning | **MATH-500** | 500-problem math reasoning subset. |
| Mathematical Reasoning | **AIME 2024** | American Invitational Mathematics Examination (high difficulty). |
| Mathematical Reasoning | **AMC** | American Mathematics Competition. |
| Scientific Reasoning | **GPQA** | Graduate-level science Q&A. |

**Backbone models:** Qwen2.5-Math-1.5B and Qwen3-1.7B.

OOD generalization experiments: adapt on one benchmark and evaluate on the others.

---

## 5. Evaluation metrics and main results

### Metrics

- **Pass@1** (stochastic decoding, temperature=1.0, top-p=0.95).
- Maximum generation length 3,072 tokens.

### Main results (Table 1)

**Qwen2.5-Math-1.5B:**

| Method | MMLU-Pro | MATH-500 | AIME 2024 | AMC | GPQA | Avg |
|------|----------|----------|-----------|-----|------|-----|
| TTRL | 35.6 | 73.0 | 15.8 | 47.3 | 26.1 | 41.5 |
| **DARE** | **38.9** | **73.6** | **19.8** | **50.2** | **28.5** | **44.2** |

**Qwen3-1.7B:**

| Method | MMLU-Pro | MATH-500 | AIME 2024 | AMC | GPQA | Avg |
|------|----------|----------|-----------|-----|------|-----|
| TTRL | 46.9 | 78.2 | 24.0 | 52.9 | 31.2 | 48.6 |
| **DARE** | **48.8** | **79.6** | **26.3** | **55.7** | **32.7** | **50.6** |

### Key findings

1. **Beats all baselines:** DARE achieves the best average across two backbones and five benchmarks, gaining 2.0–2.7 points over TTRL on average; relative gain of 25.3% on AIME 2024 and 5.3% on AMC.
2. **Ablation** (Table 2): adding components incrementally, the distribution reward contributes the most (AIME 2024 rises from 7.7 to 16.6); exploration bonus and pruning provide complementary gains; the full DARE is best.
3. **OOD generalization:** after adapting on one benchmark, DARE consistently beats TTRL on unseen benchmarks, typically by 2–5 points, indicating distribution-aware-reward behavior is more transferable.
4. **Robustness to rollout correlation:** as rollout correlation grows, TTRL performance drops sharply while DARE degrades more slowly (Figure 5), validating its mitigation of confirmation collapse.
5. **Convergence efficiency:** DARE reaches the same accuracy threshold about 12–26 update steps fewer than TTRL (Figure 6), indicating distribution-aware reward provides a more efficient learning signal.
