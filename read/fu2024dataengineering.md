# Data Engineering for Scaling Language Models to 128K Context

> **Added to survey on:** 2026-03-11

**Paper:** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng (2024)
**arXiv:** 2402.10171
**Method:** Data Eng 128K | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

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

This paper belongs to **Family I — Prediction-Statistic Optimization (predictive likelihood minimization)**.

- The core practice is **continual pretraining** on top of a pretrained LLaMA-2, with the standard **next-token prediction (language-modeling loss)** as the objective.
- All training data come from SlimPajama (an open reproduction of LLaMA's pretraining data); no external annotation, human preference, external verifier, or external AI label is introduced.
- The only innovation lies at the data-engineering level (length upsampling + domain balance); the training signal comes entirely from the intrinsic statistics of the text itself (token-level predictive likelihood).
- Therefore it fits the UPT definition: no external ground truth, purely the model's internal LM loss driving post-training optimization.

---

## 2. Problem Addressed

Existing open-source long-context models (LongChat, Together LLaMA-2, YaRN Mistral, LongLoRA), although claimed to support 32K–128K contexts, perform poorly on the **Needle-in-a-Haystack** precise retrieval test and cannot reliably retrieve information at arbitrary positions in long documents. The authors argue:

1. **Long-context ability does not require massive training:** the model already roughly acquires the ability to leverage information at arbitrary positions during 4K pretraining; a small amount of continual pretraining suffices to extend this ability to 128K.
2. **Existing work overlooks data-engineering details:** training lengths are too short (Together only 32K), domain mixes are imbalanced (YaRN uses only books), and no length upsampling is performed (LongLoRA).
3. A **low-cost, academic-budget-friendly** data recipe is needed to make open-source models approach GPT-4 Turbo 128K on Needle-in-a-Haystack.

---

## 3. Method

### 3.1 Data recipe: Per-source Length Upsampling

Starting from SlimPajama (82% web / 15% C4 / 4.5% code / 4.5% Wikipedia / 4.5% books / 2.5% Arxiv / 2.0% StackExchange), the authors compare several data-mixing strategies:

| Strategy | Description |
|---|---|
| **Cut at 4K** | Cut all documents into 4K chunks (standard practice; breaks long-range dependencies). |
| **Cut at 128K** | Keep natural long documents, leaving the domain distribution unchanged (but natural long documents are scarce). |
| **Global Upsampling** | Globally upsample long sequences; this shifts the domain distribution. |
| **Upsample Arxiv/Book/Github** | Upsample only specific domains, changing both domain and length distributions. |
| **Per-source Upsampling** (recommended) | Upsample long documents *within each source*, keeping domain proportions unchanged and only altering the length distribution. |

The key to **per-source upsampling**: within each data source, upsample documents longer than 4K from ~30% to ~70%, while keeping the original domain ratios (67% CC / 15% C4 / 4.5% Github / etc.). This increases long-sequence training data while avoiding the short-context degradation that domain shift would cause.

### 3.2 Training setup

- **Base models:** LLaMA-2 7B / 13B.
- **Training context length:** 80K for 7B and 64K for 13B (memory-limited).
- **RoPE base adjustment:** modify positional encoding following Xiong et al. (2023).
- **Data volume:** 1B–5B tokens (about 2000 optimization steps; batch size 4M tokens).
- **Hardware:** 8 × 80G A100 (about 5 days for 7B, 10 days for 13B); cost is only ~1% of Xiong et al. (400B tokens).
- **Framework:** Huggingface Transformers + DeepSpeed Zero 3 + FlashAttention 2 + Gradient Checkpointing + CPU Offloading.
- **Learning rate:** constant 2e-5.

### 3.3 Key findings

- **Data volume:** 500M tokens already unlock precise retrieval within 80K; 5B tokens generalize to unseen lengths in the 80K–128K range; 10B tokens cause overfitting and degrade generalization.
- **Domain balance:** upsampling a single domain (e.g. Books/Code) does not transfer to other domains and may even hurt them; per-source upsampling is the only strategy that improves across all domains and lengths.
- **Validation loss is insufficient:** two recipes can yield very similar validation losses but show drastically different Needle-in-a-Haystack performance (Figure 4).

---

## 4. Datasets

### Training data

- **SlimPajama** (Soboleva et al., 2023): an open reproduction of LLaMA's pretraining data, 627B tokens, comprising CommonCrawl (67%), C4 (15%), Wikipedia (4.5%), Github (4.5%), Books (4.5%), Arxiv (2.5%), StackExchange (2.0%).
- After per-source length upsampling, packed into 80K / 64K chunks; total training volume **5B tokens**.

### Evaluation data

| Benchmark | Description |
|---|---|
| **Needle-in-a-Haystack** (Kamradt, 2023) | Insert a sentence (needle) at arbitrary positions in 1K–128K documents and test whether the model can recite it precisely; the primary metric. |
| **InfiniBench BookQA** (Zhang et al., 2023) | Book-long question answering at 128K. |
| **MMLU** (Hendrycks et al., 2020) | Short-context general-ability benchmark, used to verify no degradation. |
| **Per-domain validation loss** | Perplexity per domain over 0–4K and 4K–128K. |

---

## 5. Evaluation metrics and main results

### Primary metrics

- **Needle-in-a-Haystack accuracy:** retrieval success rate over a grid of document lengths × needle positions.
- **BookQA accuracy:** QA accuracy on 128K books.
- **MMLU score:** short-context general ability.

### Core results

| Model | Context | Needle Acc | MMLU |
|---|---|---|---|
| GPT-4-Turbo | 128K | 87.1 | 86.4 |
| Together LLaMA-2 7B | 32K | 27.9 | 44.8 |
| LongChat v1.5 7B | 32K | 18.0 | 42.3 |
| LongLoRA 7B | 100K | 70.0 | 37.9 |
| YaRN Mistral 7B | 128K | 57.4 | 59.4 |
| **Ours LLaMA-2 7B** | **80K** | **88.0** | **43.3** |
| LongLoRA 13B | 64K | 54.1 | 50.1 |
| **Ours LLaMA-2 13B** | **64K** | **90.0** | **52.4** |

| Model | BookQA (128K) |
|---|---|
| GPT-4-Turbo 128K | 37.4 |
| LongLoRA 7B 100K | 24.3 |
| Ours LLaMA-2 7B 80K | 27.4 |
| YaRN Mistral 7B 128K | 26.3 |
| **Ours LLaMA-2 13B 64K** | **31.1** |

### Key conclusions

1. On Needle-in-a-Haystack, the 7B model reaches **88.0** accuracy (trained at 80K context), approaching GPT-4 Turbo's 87.1 and dramatically beating all open-source baselines.
2. **No short-context degradation:** MMLU scores match the base model (7B: 43.3 vs base ~45; 13B: 52.4 vs base ~55).
3. Per-source length upsampling is the only recipe that avoids significant degradation on any domain (0–4K and 4K–128K).
4. Only **5B tokens / 5 days / 8× A100** are needed, at about 1% of the cost of large-scale recipes, proving 128K long-context modeling is feasible on an academic budget.
