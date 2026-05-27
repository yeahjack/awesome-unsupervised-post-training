# One-shot Entropy Minimization

> **Added to survey on:** 2026-03-11

> **Method:** One-shot EM | **Title:** One-shot Entropy Minimization
> **Carrier:** Direct Opt. | **Regime:** test-time | **Level:** Token
> **arXiv:** 2505.20282 | **Authors:** Zitian Gao, Lynx Chen, Haoming Luo, Joey Zhou, Bryan Dai (Ubiquant)

| When to Adapt | Few-Sample Target Adaptation before held-out inference |
|---|---|
| Trigger Unit | one unlabeled prompt or tiny prompt pool |
| Persistence | full parameter update persists across subsequent evaluation until manual reset |
| Inference Coupling | adapt first on the selected prompt(s), then evaluate on downstream benchmarks |
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

- **When the update is triggered:** the update is triggered by an extremely small prompt pool; the core setting is to perform a short adaptation on one or very few unlabeled prompts first.
- **Whose sample it serves:** these updates then serve a broader benchmark evaluation, not just that one prompt.
- **Whether parameters/state accumulate:** parameters are retained after the short adaptation through the subsequent evaluation; the paper does not perform per-sample reset.
- **Reset boundary:** it is more of a micro-dataset adapt-then-evaluate than a standard streaming TTA.

## 1. UPT Assignment Rationale

This paper belongs to **Family I — Prediction-Statistic Optimization (entropy minimization)**.

Reasons:
- The training process is fully **unsupervised**, requiring no ground-truth labels, external verifier, or reward model.
- The objective is the **conditional entropy** of the model's own generated tokens — an intrinsic statistic.
- It uses only **one unlabeled prompt** and a very small number of gradient steps (about 10), directly minimizing token-level entropy over the generated sequence.
- The signal comes entirely from the model's own predictive distribution (intrinsic statistics) with no external feedback.

---

## 2. Problem Addressed

Traditional post-training methods (e.g. RL) require large amounts of high-quality annotated data and carefully designed rule-based rewards, with high preparation cost and risk of reward hacking. This paper studies an extreme question: **can we improve an LLM's reasoning ability with a single unlabeled example and very few training steps, without any external supervision?**

The authors observe that LLM generation is fundamentally stochastic, and the correct answer typically corresponds to lower token entropy. Based on this, directly minimizing entropy can let the model "lock onto" high-probability correct reasoning paths and unleash the latent ability of the pretrained model.

---

## 3. Method

### 3.1 Entropy Minimization algorithm

Given a pretrained model $p_\theta$ and an input prompt $x$, the model autoregressively generates a response $y = (y_1, \dots, y_T)$. For each generation position $t$, compute the conditional entropy:

$$H_t = -\sum_{v \in \mathcal{V}} p_\theta(v \mid y_{<t}, x) \log p_\theta(v \mid y_{<t}, x)$$

Entropy is computed only over **generated tokens after the prompt** (excluding the prompt and PAD tokens); the EM loss is:

$$\mathcal{L}_{\text{EM}}(x; \theta) = \frac{1}{|\mathcal{I}|} \sum_{t \in \mathcal{I}} H_t$$

This loss is fully differentiable; its gradient is similar to the score-function estimator in entropy-regularized RL but requires no external reward or value baseline.

### 3.2 Data-selection strategy

Use **variance-based data selection**: for each prompt in the candidate pool, sample $k$ times and compute the pass@k variance:

$$x^* = \arg\max_{x \in \mathcal{D}} \text{Var}_{\text{pass@k}}(x)$$

High variance means the model is most uncertain about this prompt (sometimes right, sometimes wrong); such "entropy-sensitive" samples provide the largest gradient signal.

### 3.3 Training setup

- Use only **1 unlabeled example** (selected from the NuminaMath dataset).
- Training converges in **10 steps** (beyond 10 steps, over-confidence sets in and performance drops).
- Learning rate: $2 \times 10^{-5}$; temperature: 0.5; batch size: 64.
- Implemented with the Accelerate framework.

### 3.4 Key findings

- **Logit shift:** EM training causes the logit distribution to **shift right** (skewness increases), concentrating probability on high-probability tokens — opposite to the leftward shift induced by RL. This makes greedy decoding particularly effective after EM.
- **Over-confidence effect:** EM loss keeps decreasing but performance starts to drop after ~10 steps, showing that EM is essentially a **distribution-shaping tool** rather than a traditional learning method.
- **Temperature trend reversal:** EM-trained models perform best at **low inference temperature** (opposite to RL models, which prefer high temperature), because EM has already concentrated probability on the correct token.
- **EM → RL beats RL → EM:** running EM first then RL works well (EM strengthens reasoning, RL refines further); the reverse — RL first then EM — destroys the distribution learned by RL.

---

## 4. Datasets

### Training data
- **NuminaMath:** 1 (1-shot) or a few (multi-shot) examples; only the prompt is used, not the label.

### Evaluation benchmarks
| Benchmark | Type |
|---|---|
| MATH500 | Mathematical reasoning |
| Minerva Math | Mathematical reasoning |
| OlympiadBench | Math competitions |
| AMC23 | Math competitions |
| KK | Logical reasoning |
| MBPP | Code generation |

---

## 5. Evaluation metrics and main results

### Metrics
- Per-benchmark accuracy, using **avg@8** (mean of 8 samples) to reduce stochasticity.
- All experiments are repeated over 16 random seeds for reliability.

### Main results (Qwen2.5-Math-7B as base)

| Model | MATH500 | Minerva Math | Olympiad Bench | AMC23 | KK | MBPP | Avg. |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B (base) | 53.0 | 11.0 | 17.2 | 44.1 | 1.0 | 48.9 | 29.2 |
| **+ EM 1-shot** | **78.8** | **35.3** | **39.7** | **70.3** | **17.4** | **65.1** | **51.1** |
| Gain | +25.8 | +24.3 | +22.5 | +26.2 | +16.4 | +16.2 | +21.9 |

- Using just 1 example and 10 training steps, EM delivers an average gain of **+21.9 points**, on several benchmarks matching or exceeding RL methods that require thousands of examples and hundreds of steps (e.g. RLVR, OpenReasoner-Zero, SimpleRL-Zoo).
- Effective across multiple base models: LLaMA-3.1-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, SimpleRL-Zoo, Qwen2.5-Math-7B.
- 1-shot EM usually beats multi-shot EM: single-sample training is more stable, with smaller prompt/output-length variation and smoother loss convergence.
- However, on models already heavily RL-trained (e.g. SimpleRL-Zoo), EM can lead to a slight drop (49.3 → 44.5), indicating that EM may negatively affect an already highly optimized distribution.
