# RLSC: Reinforcement Learning via Self-Confidence

> **Added to survey on:** 2026-03-11

**Paper:** Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models
**arXiv:** 2506.06395
**Method:** RLSC | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Token

| When to Adapt | Few-Sample Target Adaptation before held-out inference |
|---|---|
| Trigger Unit | small unlabeled task dataset / rollout batch |
| Persistence | full parameter accumulate across short RL runs |
| Inference Coupling | adapt first on the task cohort, then evaluate on downstream benchmarks |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Few-Sample Target Adaptation |
| Visibility Scope | Few-sample target subset |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Few-Sample Target Adaptation`; `Visibility Scope=Few-sample target subset`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** the update is triggered on a very small unlabeled task cohort, typically running only a small number of RL / optimization steps.
- **Whose sample it serves:** these updates mainly serve a subsequent evaluation on a larger benchmark set, not the immediate inference of any single prompt.
- **Whether parameters/state accumulate:** parameters accumulate continuously during the short training run and reset only at the end of one experiment.
- **Reset boundary:** it sits between offline adaptation and test-time micro-adaptation, but the primary coupling is still "adapt first, then evaluate as a whole".

## 1. UPT Assignment Rationale

RLSC belongs to **Family I (Prediction-Statistic Optimization)**, sub-class intrinsic predictive statistics via policy optimization.

Core reason: RLSC directly uses the model's own average log-probability / self-confidence as the reward, with no reliance on external labels, human feedback, external reward models, or verifiers. Its objective maximizes the model's confidence in its own sampled outputs (mode sharpening); in mathematical form, $F(p_\theta) = \mathbb{E}_{y \sim p_\theta(\cdot|x)}[p_\theta(y|x)]$, equivalent to maximizing the probability that two independent samples from the output distribution agree. This objective is driven entirely by the model's intrinsic predictive statistics and implemented via policy optimization — a textbook intrinsic-statistic signal.

---

## 2. Problem Addressed

Existing RL post-training methods have the following limitations:

- **RLHF** requires large amounts of human annotation and a preference model — costly.
- **RLVR (Reinforcement Learning with Verifiable Rewards)** still needs ground-truth labels to compute the reward.
- **TTRL (Test-Time Reinforcement Learning)** generates pseudo-labels via majority voting but needs 64 samples per question, which is computationally expensive and requires extra answer-extraction preprocessing.

RLSC aims to provide a **label-free, low-sample, low-compute** RL fine-tuning method that improves reasoning ability using only the model's own confidence signal.

---

## 3. Method

### 3.1 From majority voting to mode sharpening

RLSC's core insight: majority voting essentially selects the mode of the output distribution, implicitly concentrating probability mass on the most likely answer. RLSC formalizes this process as a differentiable self-supervised objective — **mode sharpening**.

Define the self-confidence objective:

$$F(p_\theta) = \mathbb{E}_{y \sim p_\theta(\cdot|x)}[p_\theta(y|x)] = \sum_y p_\theta(y|x)^2$$

The objective is maximized when the distribution degenerates into a delta function (the model is fully confident about one answer).

### 3.2 Self-confidence loss

Taking gradients via the log-trick yields the training loss:

$$\mathcal{L}_1 = -\sum_y p_{\text{old}}(y|x) \cdot \log p_\theta(y|x)$$

where $p_{\text{old}}$ is a frozen copy of the old model used for sampling and weighting (no gradient flows through it). The loss encourages the model to assign higher log-probability to answers that the old model already assigns high confidence to.

A smoothed variant adds a constant $\alpha > 0$:

$$\mathcal{L}_2 = -\sum_y (p_{\text{old}}(y|x) + \alpha) \cdot \log p_\theta(y|x)$$

Empirically, $\alpha = 0.1$ improves convergence and generalization.

### 3.3 Training pipeline

1. For each problem, generate 16 candidate responses from the base model at temperature 0.5.
2. For each (prompt + answer) pair, compute token-level log-probabilities.
3. Apply an assistant mask to keep only the answer-part tokens.
4. Sum masked log-probs to obtain the log-likelihood of the response.
5. Backpropagate the self-confidence loss to update parameters.

Training takes only **10 or 20 steps**, using 8 NVIDIA A100 GPUs (80GB), AdamW optimizer, learning rate $1 \times 10^{-5}$, generation length capped at 3072 tokens.

---

## 4. Datasets

**Training data:**
- **AIME2024** training set (competition math problems from NuminaMath) — questions only, no labels used.

**Evaluation benchmarks:**
- AIME2024
- MATH500
- AMC23
- GSM8K
- Minerva Math
- Olympiadbench
- MMLU Stem
- GPQADiamond

---

## 5. Evaluation metrics and main results

**Metrics:** Accuracy (correct / total), Pass@1.

### Main results (Qwen2.5-Math-7B)

| Benchmark | Baseline | RLSC | Gain |
|---|---|---|---|
| AIME2024 | 13.3 | 26.7 | **+13.4** |
| MATH500 | 51.4 | 72.6 | **+21.2** |
| AMC23 | 45.0 | 54.7 | **+9.7** |
| GSM8K | 84.3 | 86.3 | +2.0 |
| Olympiadbench | 15.1 | 35.9 | **+20.8** |
| Minerva Math | 10.7 | 32.4 | **+21.7** |
| MMLU Stem | 52.3 | 57.6 | +5.3 |

### Main results (Qwen2.5-Math-1.5B)

| Benchmark | Baseline | RLSC | Gain |
|---|---|---|---|
| AIME2024 | 3.3 | 6.7 | +3.4 |
| MATH500 | 35.6 | 62.4 | **+26.8** |
| AMC23 | 34.7 | 46.2 | **+11.5** |
| Minerva Math | 11.4 | 26.1 | **+14.7** |
| MMLU Stem | 34.1 | 48.6 | **+14.5** |

### Key findings

- At 7B scale, Minerva Math improves most (+21.7%); AIME2024 and Olympiadbench also gain substantially.
- At 1.5B scale, MATH500 improves the most (+26.8%).
- **Emergent behavior:** after RLSC fine-tuning, the model tends to produce shorter, more confident answers and can reason concisely without prompts like "Let's think step by step".
- Just 16 samples + 10–20 training steps deliver substantial gains — extremely low resource demand.
