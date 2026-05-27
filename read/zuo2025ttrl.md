# TTRL: Test-Time Reinforcement Learning

> **Added to survey on:** 2026-03-11

> **Method:** TTRL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv 2504.16084 — Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, et al. (Tsinghua University, Shanghai AI Lab)

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

TTRL belongs to **Family II (Sample-Relation Supervision)**, sub-class population consensus / majority-vote.

Core mechanism: at test time, sample multiple completions for the same question and aggregate a consensus answer from the model's own outputs via majority voting; use this consensus as a pseudo-label to construct a rule-based reward (match=1, mismatch=0) that drives online RL training. The pipeline relies on no external ground-truth label, no human annotation, and no external verifier; the reward signal comes entirely from **sample-relation internal supervision** among the model's own outputs.

---

## 2. Problem Addressed

- Traditional RL for reasoning (DeepSeek-R1, GRPO, etc.) relies on labeled data to provide the reward signal — annotation is costly and not scalable.
- Test-Time Scaling (TTS) methods (majority voting, best-of-N, etc.) only use multi-sampling at inference to lift performance, never updating model parameters; ability is not durably improved.
- In reality, more and more hard problems lack labeled answers; models need to self-evolve on **unlabeled test data**.

TTRL combines TTS with Test-Time Training (TTT): use RL during inference to update model parameters directly, enabling the model to self-improve on unlabeled data.

---

## 3. Method

### 3.1 Overall framework

TTRL runs the following loop at test time:

1. **Label Estimation:** for a question $x$, sample $N$ candidate outputs $\{y_1, y_2, \dots, y_N\}$ from the current policy $\pi_\theta$; extract each answer and use majority voting to obtain the consensus answer $y^*$.
2. **Reward Calculation:** for each sampled output $\hat{y}_i$, use a rule-based reward:
   $$R(\hat{y}_i, y^*) = \begin{cases} 1, & \text{if } \hat{y}_i = y^* \\ 0, & \text{otherwise} \end{cases}$$
3. **Policy Optimization:** drive online RL with this reward (GRPO by default), maximizing expected reward via gradient ascent:
   $$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(y, y^*)]$$

### 3.2 Key design details

- **Vote-then-sample:** first sample 64 outputs for majority-vote label estimation, then sub-sample 32 for RL training to cut compute.
- **Online learning:** the model keeps updating throughout training, so the quality of majority-vote labels improves with the model's ability — a positive self-reinforcing loop.
- **"Lucky Hit" phenomenon:** even when majority-vote labels are wrong, wrong predictions usually scatter across different answers, so most rewards remain correct (in experiments, when label accuracy is just 37%, reward accuracy is still 92%).
- **Compatible with multiple RL algorithms:** beyond GRPO, also compatible with PPO and PRIME, with highly consistent training curves.

### 3.3 Hyperparameters

- Learning rate: cosine schedule, peak $5 \times 10^{-7}$; AdamW optimizer.
- Rollout temperature: 0.6 (1.0 for Qwen2.5-Math and LRMs).
- Maximum generation length: 32,768 tokens for LRMs; 3,072 for other models.
- Number of episodes: 10 for MATH-500, 30 for AMC, 80 for AIME 2024.
- Hardware: 8 × NVIDIA A100 80GB GPUs.

---

## 4. Datasets

| Dataset | Domain | Description |
|--------|------|------|
| **AIME 2024** | Math competition | American Invitational Mathematics Examination — high difficulty. |
| **AMC** | Math competition | American Mathematics Competition. |
| **MATH-500** | Math reasoning | 500-problem subset of MATH covering 5 difficulty levels. |
| **GPQA-Diamond** | Science Q&A | The high-quality subset of Graduate-Level Google-Proof Q&A. |

All datasets are used **without labels** — only the questions are provided; no ground-truth answers are used.

---

## 5. Evaluation metrics and main results

### Metrics

- **pass@1:** main experiments use non-zero-temperature sampling to generate 16 answers (4 in 32k context); average accuracy is reported.
- **maj@n:** majority-voting accuracy (n=16 or n=64).
- **avg@n:** mean accuracy over n samples.

### Main results

**Math base models (Qwen2.5-Math series):**

| Model | AIME 2024 | AMC | MATH-500 | GPQA | Average gain |
|------|-----------|-----|----------|------|---------|
| Qwen2.5-Math-1.5B | 7.7 → 15.8 | 28.6 → 48.9 | 32.7 → 73.0 | 24.9 → 26.1 | +74.4% |
| Qwen2.5-Math-7B | 12.9 → 40.2 | 35.6 → 68.1 | 46.7 → 83.4 | 29.1 → 27.7 | +76.5% |
| **Qwen2.5-Math-7B AIME** | **+211.6%** | | | | |

**Vanilla base / instruct models:**

| Model | AIME 2024 | AMC | MATH-500 | GPQA | Average gain |
|------|-----------|-----|----------|------|---------|
| Qwen2.5-7B | +194.9% | +62.6% | +33.1% | +5.7% | +43.7% |
| Qwen2.5-32B | +203.8% | +81.9% | +49.1% | +13.6% | +57.7% |
| LLaMA3.1-8B-Instruct | +117.4% | +38.6% | +31.1% | +10.7% | +30.6% |

**Cross-family generalization (6 additional models):** LLaMA, Mistral, and DeepSeek families all show consistent gains.

### Key findings

1. **Surpassing the maj@n upper bound:** after TTRL training, the model's avg@64 consistently exceeds the original model's maj@64 — the model "bootstraps" past the upper bound of its initial supervision signal via the self-reinforcing loop.
2. **Approaching the RL (leakage) upper bound:** TTRL's MATH-500 performance curve approaches the leakage upper bound of RL with ground-truth labels.
3. **Good OOD generalization:** after training on one benchmark, pass@1 improves on other benchmarks too — not overfitting.
4. **Scaling with model size:** monotonic improvement from 1.5B → 7B → 32B.
5. **Failure cases:** when the model's priors are insufficient (e.g. very hard AIME problems) or hyperparameters are inappropriate (too-high temperature, too-large batch size), TTRL can fail.
