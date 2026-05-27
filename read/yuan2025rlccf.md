# RLCCF — Reinforcement Learning from Coevolutionary Collective Feedback

> **Added to survey on:** 2026-03-11

**Paper:** Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback
**Authors:** Wenzhen Yuan, Shengji Tang, Weihao Lin, Jiacheng Ruan, Ganqu Cui, Bo Zhang, Tao Chen, Ting Liu, Yuzhuo Fu, Peng Ye, Lei Bai
**Affiliations:** Shanghai Jiao Tong University, Shanghai AI Lab, CUHK, Fudan University
**arXiv:** 2508.12338 (Aug 2025)

| Attribute | Value |
|---|---|
| Method | RLCCF |
| Carrier | Policy Opt. |
| Regime | training-time |
| Level | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / rollout group |
| Persistence | full parameter accumulate across RL steps |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates fire during an offline RL stage before deployment; the basic unit is a group of rollouts per prompt batch.
- **Whose sample it serves:** the current rollout group's update serves subsequent training steps and the final deployed model, not the immediate inference of any single sample.
- **Whether parameters/state accumulate:** parameters accumulate across RL training; no per-sample reset.
- **Reset boundary:** it is an offline RL-style UPT schedule, not test-time arrival-by-arrival adaptation.

## 1. UPT Assignment Rationale

RLCCF belongs to **Family II (Sample-Relation Supervision)**, sub-class population consensus / collective agreement.

Core reason: RLCCF relies on no external ground truth, no human annotation, and no external reward model. Its supervision signal comes entirely from the **Collective Consistency (CC)** within a group of models. Concretely, multiple heterogeneous LLMs each independently sample candidate answers, and a Self-Consistency (SC)-weighted majority vote produces a pseudo-label shared as the reward signal across all models. Each model's voting weight is determined by its own SC score (the frequency of the model's most common answer) — a textbook **inter-model sample-relation supervision**: the reward comes from group consensus, not from any external verifier. Each model then uses this pseudo-label to perform GRPO-based policy optimization, achieving coevolution.

---

## 2. Problem Addressed

- **Limitations of single-model self-feedback:** TTRL, EMPO, Intuitor, and others rely on a single model's own outputs to construct the reward, which easily leads to:
  - Overconfidence in incorrect outputs.
  - Reward hacking.
  - Training collapse (uncontrolled positive feedback loop).
- **Scalability bottleneck of external supervision:** RLHF needs expensive human annotation to train a reward model; RLVR needs a rule-based verifier or labeled data, limiting applicable domains.
- RLCCF proposes to leverage the **complementarity and diversity of a group of models** to produce more robust, more accurate pseudo-labels while avoiding external supervision.

---

## 3. Method

The RLCCF framework has two core stages:

### 3.1 Pseudo-label estimation (SC-weighted majority voting)

Given a model set M = {M_1, ..., M_N}, for each query q:

1. **Independent sampling:** each model M_n independently produces K candidate answers {o_{n,k}}.
2. **Compute SC scores:** for each model, compute its Self-Consistency score SC_n, defined as the frequency of its most common answer (a measure of the model's internal consistency / reliability).
3. **SC-weighted voting:** the pseudo-label is decided by SC-weighted voting across all models:

   â = argmax_a Σ_{n=1}^{N} Σ_{k=1}^{K} SC_n · I[a = o_{n,k}]

   Models with higher SC scores have larger voting weight, ensuring that more reliable models contribute more to the collective decision.

### 3.2 Policy optimization (based on GRPO)

Using the pseudo-label â as the shared supervision signal, each model is independently optimized:

- **Reward:** r_{n,k} = 1 if o_{n,k} = â, else 0 (binary reward — whether the answer matches the pseudo-label).
- **Objective:** under the GRPO framework, each model M_n maximizes

  J_n(q) = (1/K) Σ_{k=1}^{K} Â_{n,k} − β · KL(π_{θ_n}(·|q) || π_{ref_n}(·|q))

  where Â_{n,k} is the clipped advantage (PPO-style) and β controls KL regularization.

### 3.3 Theoretical analysis: why multi-model collaboration works

- **Bias Reduction:** modeling each model's output as X_{n,k} ~ N(GT + ε_n, σ_n²), as the number of models N grows, the mean of aggregated outputs approaches the ground truth (E[X] → GT as N → ∞), because biases of different models cancel out.
- **Model Complementarity:** different models have complementary strengths in different domains (e.g. math vs. code); collaborative training substantially lifts each model's weak areas.

---

## 4. Datasets

### Training data
- **MATH-700**: 100 problems each randomly sampled from 7 sub-categories of the MATH training set, totaling 700 problems (unlabeled, questions only).

### Evaluation benchmarks
| Dataset | Description | Difficulty |
|---|---|---|
| **AIME 2024** | American Invitational Mathematics Examination. | Highest |
| **OlympiadBench** | Olympiad-level multimodal scientific problems. | High |
| **AMC 2024** | American Mathematics Competition. | Medium-high |
| **MATH-500** | MATH test-set subset. | Medium |

---

## 5. Evaluation metrics and main results

### Metrics
- **Average Accuracy:** mean accuracy over 32 sampled answers per problem.
- **Group Majority Vote Accuracy:** aggregated vote over 32 sampled answers from each of the 4 models.
- **Label Accuracy:** accuracy of the pseudo-labels during training.
- **Reward Accuracy:** accuracy of the reward signal during training.

### Main results (Table 1)

| Model | Base AVG | RLCCF AVG | Relative gain |
|---|---|---|---|
| Qwen2.5-7B | 31.92 | 40.60 | +27.20% |
| GLM-4-9B | 32.99 | 38.48 | +16.63% |
| InternLM3-8B-Instruct | 34.51 | 39.80 | +15.31% |
| LLaMA-3.1-8B-Instruct | 23.68 | 25.51 | +7.74% |
| Group Majority Vote | 48.70 | **50.90** | +4.51% |

### Key findings

1. **Comprehensively beats self-feedback baselines:** RLCCF's average exceeds Intuitor (+38.99%), EMPO (+2.82%), and TTRL (+2.45%).
2. **Expands collective ability:** Group Majority Vote rises from 48.70% to 50.90%, while single-model self-feedback methods mostly fail to improve — and sometimes hurt — group-vote accuracy.
3. **Training stability:** unlike Intuitor's training-collapse issues, RLCCF provides stable and sustained gains.
4. **SC-weighted voting beats simple voting:** SC-weighted voting outperforms simple majority voting on every model (Table 2), with a +1.29% gain at the group level.
5. **Collective Consistency strengthened:** after training, the group's answer distributions converge to a single consistent correct peak, whereas TTRL-trained models, although more consistent, may peak at different (incorrect) answers.
