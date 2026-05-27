# Model Whisper: Steering Vectors Unlock Large Language Models' Potential in Test-time

> **Added to survey on:** 2026-03-11

**Paper:** arXiv:2512.04748
**Authors:** Xinyue Kang, Diwei Shi, Li Chen (Tsinghua University)
**Method:** Model Whisper | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token

| When to Adapt | multi-protocol: Full-Cohort Transductive Adaptation + Test-Time Instance Adaptation |
|---|---|
| Trigger Unit | main: whole test set / test-distribution mini-batches; secondary: single instance |
| Persistence | main: optimized steering vectors persist across full test set; secondary: sample-specific vectors serve one instance |
| Inference Coupling | main: adapt first on target distribution, then infer on all samples; secondary: adapt for current instance |
| Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Multi-protocol: Cumulative + Non-Cumulative |
| Reset Boundary | Multi-protocol: Target-Distribution Boundary + Sample Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Multi-protocol: Full-Cohort Transductive Adaptation + Test-Time Instance Adaptation |
| Visibility Scope | Multi-protocol: Full target cohort + Current Instance Only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** this paper contains multiple protocol entries: `Timing Regime=Multi-protocol: Full-Cohort Transductive Adaptation + Test-Time Instance Adaptation`; `Visibility Scope=Multi-protocol: Full target cohort + Current Instance Only`.
- **Two-axis coding:** `Input Visibility=Multi-protocol: Offline + Online`; `Update Persistence=Multi-protocol: Cumulative + Non-Cumulative`; `Reset Boundary=Multi-protocol: Target-Distribution Boundary + Sample Boundary`.

| Protocol Entry | Timing Regime | Visibility Scope | Input Visibility | Update Persistence | Reset Boundary | Note |
|---|---|---|---|---|---|---|
| Model Whisper / distribution-level TTSV | Full-Cohort Transductive Adaptation | Full target cohort | Offline | Cumulative | Target-Distribution Boundary | Optimize steering vectors on the target distribution first, then apply once to the entire sample set. |
| Model Whisper / sample-specific TTSV | Test-Time Instance Adaptation | Current Instance Only | Online | Non-Cumulative | Sample Boundary | Optimize a steering vector for the current instance and discard afterwards. |

- **When the update is triggered:** the update concentrates into a single optimization pass over the target test distribution, rather than updating sample by sample.
- **Whose sample it serves:** the optimized steering vectors then serve the unified inference over the entire test set; the current mini-batch's update mainly serves all subsequent samples.
- **Whether parameters/state accumulate:** the base model's parameters stay frozen; what truly persists is the optimized steering vectors, which are reused across the test set.
- **Reset boundary:** the main work explicitly adopts optimize-once, apply-to-all as the default protocol, whose boundary is the target-distribution boundary; the sample-specific variant optimizes within a single sample and serves that instance, bounded at the sample boundary.

## 1. UPT Assignment Rationale

Model Whisper belongs to **Family I — Prediction-Statistic Optimization** (local-state / adapter shaping).

Core justification:

- **No external labels/verifiers:** the method runs on fully unlabeled test data with no reliance on ground truth, human feedback, or external AI annotations.
- **Driven by intrinsic signal:** the objective is the token-level conditional entropy of the model's own output — a pure intrinsic-statistic signal that guides the model toward a high-confidence state by minimizing output entropy.
- **Local-state shaping:** the optimization target is a set of learnable steering vectors (TTSV, Test-Time Steering Vectors) concatenated as a prefix in front of the input embedding. The model's parameters stay completely frozen; only the local state of the input space is shaped to activate the model's latent capabilities — a classic adapter / state-shaping paradigm.
- **Sample-specific adaptation:** TTSV is optimized for the specific test distribution, providing per-distribution or per-sample adaptation.

---

## 2. Problem Addressed

LLMs internalize substantial knowledge and reasoning patterns during pretraining, but these latent capabilities often fail to be automatically activated on specific tasks or new data distributions, leaving a significant gap between actual performance and latent capability.

Existing test-time adaptation (TTA) methods have the following limitations:

1. **Full-parameter fine-tuning methods** (e.g. Entropy Minimization, EM): high compute cost, gradient backpropagation through all parameters, and risk of catastrophic forgetting.
2. **Output-layer calibration methods** (e.g. SLOT): only add a parameter vector at the last hidden layer for next-token-prediction fine-tuning; their effect is local and cannot guide the multi-layer reasoning process upstream.
3. **Prompt engineering:** interacts with the model via discrete text tokens; expressiveness is limited to the discrete vocabulary and cannot convey continuous, dense guidance.

Model Whisper's core motivation: can we, **without modifying any model parameters**, guide the model's multi-layer reasoning from the computation origin (input layer) using signals in the continuous embedding space?

---

## 3. Method

### 3.1 TTSV definition

Introduce a set of learnable continuous vectors $V_{\text{steer}} \in \mathbb{R}^{L \times d}$, where $L$ is the steering-vector sequence length (default $L=20$) and $d$ is the model's embedding dimension. For any input $X$, prepend TTSV to its embedding sequence, forming the augmented input:

$$E' = [V_{\text{steer}}; E(X)]$$

This augmented sequence is fed into the frozen LLM for inference. TTSV participates in every layer's attention computation, with a bias-amplification effect that scales the guidance signal layer by layer.

### 3.2 Entropy minimization optimization

The optimization is based on the core assumption that the model is highly confident (low output entropy) on tasks it does well. So the objective is to minimize the average conditional entropy of output tokens:

$$\mathcal{L}(V_{\text{steer}}) = \frac{\sum_{i=1}^{B} \sum_{t \in \mathcal{I}_i} H_{i,t}}{\sum_{i=1}^{B} |\mathcal{I}_i|}$$

where $H_{i,t} = -\sum_{v \in V} p(v|y_{<t}, E') \log p(v|y_{<t}, E')$ is the conditional entropy at step $t$ of sample $i$. Gradients are computed and applied only to $V_{\text{steer}}$; the model parameters $\theta$ stay frozen.

### 3.3 Optimize-once, apply-to-all

- **Optimization phase:** iteratively optimize TTSV on the test dataset (20 epochs, AdamW, batch size 16) to obtain a fixed $V^*_{\text{steer}}$.
- **Inference phase:** prepend $V^*_{\text{steer}}$ as a fixed prefix to every test sample — plug-and-play, with no extra inference-time latency.

### 3.4 Theoretical analysis

In the first attention layer, TTSV introduces a linear bias:

$$t'_i = (1 - \alpha_i) t_i + \alpha_i b$$

where $b = W_V p$ is the bias direction (determined by the learnable vector) and $\alpha_i$ is the attention weight of position $i$ on TTSV. This bias signal propagates through the deep network and is amplified layer by layer (bias amplification), eventually altering the model's computation trajectory significantly.

### 3.5 Initialization strategy

- **Qwen series:** random initialization from a standard normal $\mathcal{N}(0,1)$.
- **LLaMA series:** data-driven initialization — compute the mean and variance of the test-data token embeddings, sample the initial TTSV from this distribution, and avoid sensitive models falling into bad local optima from random initialization.

---

## 4. Datasets

### Mathematical reasoning

| Dataset | Description |
|--------|------|
| **MATH500** | Classic math problem benchmark (Hendrycks et al., 2021). |
| **Minerva Math** | Quantitative reasoning problem set (Lewkowycz et al., 2022). |
| **Olympiad Bench** | Olympiad-level math and science (He et al., 2024). |
| **AMC23** | AMC competition math problems. |
| **AIME24** | American Invitational Mathematics Examination 2024 (high difficulty). |

### Cross-domain generalization

| Dataset | Description |
|--------|------|
| **GPQA Diamond** | Graduate-level Google-proof Q&A, covering physics and biology (Rein et al., 2024). |

### Base models

- Qwen2.5-Math (1.5B, 7B).
- LLaMA-3.1-8B-Instruct.
- Qwen3-4B (reasoning-enhanced, thinking mode).

---

## 5. Evaluation metrics and main results

### Metrics

- **Accuracy:** absolute correctness on each benchmark.
- **Relative Gain:** percentage improvement over the baseline.
- **Absolute Improvement:** accuracy-point difference.

### Main results

#### vs. full-parameter TTA (EM) — Qwen2.5-Math-7B

| Benchmark | Baseline | +TTSV | +EM | TTSV relative gain |
|-----------|----------|-------|-----|-------------|
| MATH500 | 51.00 | 74.40 | +15.00 | +45.88% |
| Minerva Math | 12.90 | 22.80 | +7.80 | +76.74% |
| Olympiad | 16.70 | 29.80 | +16.00 | +78.44% |
| AMC23 | 42.50 | 65.00 | +17.50 | +52.94% |
| **Avg.** | **30.78** | **48.00** | +14.08 | **+55.95%** |

On Qwen2.5-Math-7B, TTSV delivers an average absolute gain of +17.22% (relative +55.95%), surpassing full-parameter EM across the board.

#### vs. SLOT — Qwen2.5-Math-1.5B / 7B

| Model | SLOT Avg. gain | TTSV Avg. gain | TTSV relative gain |
|------|--------------|--------------|-------------|
| Qwen2.5-Math-1.5B | +0.94 | +10.51 | +42.99% |
| Qwen2.5-Math-7B | +4.98 | +12.14 | +39.12% |

TTSV substantially beats SLOT at both scales.

#### Reasoning model — Qwen3-4B (Thinking Mode)

| Benchmark | Baseline | +TTSV | Relative gain |
|-----------|----------|-------|--------|
| MATH500 | 51.80 | 60.20 | +16.22% |
| Olympiad | 18.10 | 30.70 | +69.61% |
| AIME24 | 56.67 | 60.00 | +5.88% |
| GPQA | 41.92 | 47.98 | +14.46% |
| **Avg.** | **38.43** | **44.05** | **+14.62%** |

Equally effective on reasoning-enhanced models, with a relative gain of up to 69.61% on Olympiad Bench.

### Key findings

1. **Strong cross-distribution generalization:** TTSV optimized on MATH500 lifts AMC23 accuracy from 42.5% to 62.5% — OOD performance can even exceed in-distribution optimization.
2. **TTSV-length sensitivity:** $L=20$ is optimal; even $L=1$ yields substantial gain (39.4% → 55.4%); but $L=40$ degrades performance — a trade-off between expressiveness and stability.
3. **Stable training:** entropy loss decreases steadily while accuracy keeps rising, unlike EM which can degrade late in training (overfitting to confident but incorrect predictions).
4. **t-SNE visualization** shows that TTSVs from different tasks steer model activations to a shared robust reasoning subspace, explaining the strong cross-distribution generalization.
