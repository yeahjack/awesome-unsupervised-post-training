# E2-LLM: Efficient and Extreme-Length Extension of Large Language Models

> **Added to survey on:** 2026-03-11

**Method:** E2-LLM | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

> Findings of ACL 2024, pp. 4243–4253

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | unlabeled corpus / continual training batch |
| Persistence | full parameter accumulate across corpus stages; no sample-level reset |
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

- **When the update is triggered:** updates are triggered during the pre-deployment continued-pretraining / CPT stage, batch by batch over the corpus, not by any individual test sample.
- **Whose sample it serves:** updates from the current batch serve subsequent training batches and the final deployed model, not the immediate inference of the current sample.
- **Whether parameters/state accumulate:** parameters accumulate continuously across the corpus stream / stage sequence; any stage switch is only a training-stage transition, not a sample-level reset.
- **Reset boundary:** so it addresses the adaptation schedule *before* deployment, not online TTA *during* deployment.

## 1. UPT Assignment Rationale

E2-LLM belongs to **Family I (Prediction-Statistic Optimization, predictive likelihood minimization)**. Its core training objective is the standard next-token-prediction loss (language-modeling likelihood), with no reliance on external annotation, external verifier, or human/AI preference signal. Training uses only the model's own predictive likelihood as the optimization signal: it fine-tunes on short sequences to extend long-context ability. The dominant artifact is predictive likelihood, optimized directly at the token level.

---

## 2. Problem Addressed

Existing methods for extending LLM context length face two bottlenecks:

1. **Heavy data requirement:** training data matching the target context length (e.g. 32K/64K tokens) is hard to collect.
2. **High GPU memory cost:** training on long sequences requires memory that scales linearly or quadratically with length; e.g. extending to 64K can require 68.8 GB of memory or even OOM.
3. **Poor generalization:** Position Interpolation (PI) and similar methods must be trained separately for each target context length, and perform poorly on out-of-distribution (OOD) lengths.

E2-LLM aims to use **extremely short training sequences** (e.g. 4K tokens) and **a single training run** to extend the model to **arbitrary long context** (16K/32K/64K up to 120K).

---

## 3. Method

E2-LLM is based on RoPE (Rotary Position Embedding) and proposes a **dual augmentation strategy** that augments both scale and position offset during training, exposing the model to a diverse distribution of relative positions while training only on short sequences.

### 3.1 Core formula

The standard RoPE position embedding is $\mathbf{f}(\mathbf{x}, m)$. E2-LLM rewrites it as:

$$\mathbf{f}'(\mathbf{x}, m; t, g) = \mathbf{f}\left(\mathbf{x}, \frac{m+t}{g}\right)$$

where $g$ is the scale parameter and $t$ is the position offset.

### 3.2 Scale augmentation (over $g$)

- At each training iteration $i$, sample $g_i$ uniformly from $\mathcal{G} = \{1, 2, \ldots, G_{max}\}$.
- Different $g$ values correspond to different interpolated context-window sizes (e.g. $g=8$ for 32K, $g=16$ for 64K).
- This exposes the model to many position densities during training, avoiding overfitting to one specific interpolation ratio (analogous to the Runge phenomenon).

### 3.3 Position-offset augmentation (over $t$)

- Keep the position offsets of the first 4 tokens at 0 (attention-sink mechanism).
- For the remaining tokens, sample $t_i \in \{0, \ldots, T_{max}\}$ uniformly per iteration.
- $T_{max}$ is set to the difference between the current interpolated context window and the training window.
- This exposes the model to different absolute position values, helping it generalize to larger relative position gaps.

### 3.4 Training and inference

- **Training:** standard next-token-prediction fine-tuning on short sequences (4K/8K tokens); randomly sample $g_i$ and $t_i$ per iteration to modify RoPE.
- **Inference:** no additional parameters; simply set the scale parameter $g$ according to the target context length (e.g. $g=8$ for 32K, $g=16$ for 64K). The same model weights support multiple context windows.

---

## 4. Datasets

### Training data

| Dataset | Use |
|--------|------|
| **Pile** (Gao et al., 2020) | Pretraining data, general corpus |
| **ShareGPT** (Zheng et al., 2023) | Fine-tuning data, improves QA ability |
| **Long summarization datasets** (Cohan et al., 2018) | Fine-tuning data, improves long-text processing |

The maximum training sequence length $R$ is only **4K tokens** (some experiments use 8K).

### Evaluation data

| Dataset | Description |
|--------|------|
| **LongBench** (Bai et al., 2023) | Bilingual long-context understanding benchmark, covering single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, code tasks. |
| **Arxiv Proof-Pile** (Azerbayev et al., 2022) | ArXiv math papers, used to evaluate long-sequence perplexity (each paper ≥ 64K tokens). |
| **Needle In A Haystack** | Stress test for retrieval accuracy under varying context lengths and document depths. |

---

## 5. Evaluation metrics and main results

### Metrics

- **LongBench accuracy** (0–100, higher is better): covers six sub-tasks (single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic, code), split into English (EN), Chinese (ZH), and overall (All).
- **Perplexity (PPL)** (lower is better): on Arxiv Proof-Pile across context windows.
- **Needle In A Haystack accuracy:** retrieval accuracy at different document depths and context lengths.

### Main results

#### LongBench (Llama2-13B series)

| Model | EN | ZH | All |
|------|----|----|-----|
| PI-Llama2-13B-16K | 40.88 | 26.35 | 37.65 |
| **E2-LLM-Llama2-13B-16K** | **44.73** | **28.56** | **41.13** |
| **E2-LLM-Llama2-13B-32K** | **44.55** | **31.93** | **41.74** |
| GPT-3.5-Turbo-16K | 44.60 | 33.78 | 42.19 |

- E2-LLM-Llama2-13B-32K matches GPT-3.5-Turbo-16K overall (41.74 vs 42.19).
- E2-LLM-13B-16K outperforms PI-Llama2-13B-16K by ~**9%** on average under the same setting.

#### Arxiv Proof-Pile PPL (Llama2-13B, training window 4K)

| Method | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
|------|-------|-------|--------|--------|--------|
| E2-LLM-16K | 2.82 | 2.59 | 2.43 | - | - |
| E2-LLM-32K | 2.85 | 2.61 | 2.44 | 2.34 | - |
| E2-LLM-64K | 2.91 | 2.67 | 2.49 | 2.39 | 2.44 |

- As the evaluation context window grows, PPL continues to drop, showing the model effectively uses long-context information.

#### Generalization

- With $G_{max}=20$ (supporting up to an 80K interpolation window), inference-time scales of 30/40/50 correspond to 120K/160K/200K context.
- Within **120K tokens**, PPL remains at a reasonable level, demonstrating generalization to scales unseen during training.

#### Ablation

| Variant | EN | ZH | All |
|------|----|----|-----|
| **E2-LLM (full)** | **44.55** | **31.93** | **41.74** |
| E2-LLM (no offset) | 42.28 | 29.49 | 39.44 |
| E2-LLM (fixed scale) | 41.66 | 28.33 | 38.77 |

- Scale augmentation and position-offset augmentation each contribute independently; combining them yields the best result.
- Setting the number of initial fixed tokens to 4 is optimal.
