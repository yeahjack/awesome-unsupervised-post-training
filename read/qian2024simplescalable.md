# Simple and Scalable Strategies to Continually Pre-train Large Language Models

> **Added to survey on:** 2026-03-11

**Method name:** Simple CPT
**Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token
**Venue:** Transactions on Machine Learning Research (06/2024)
**Authors:** Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, Irina Rish
**arXiv:** 2403.08763

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

This paper belongs to **Family I: Prediction-Statistic Optimization (predictive likelihood minimization)** for the following reasons:

- The method's core objective is always the **causal language modeling (CLM) loss**, i.e. the next-token-prediction negative log-likelihood on text-only corpora. No external annotation, external verifier, human preference, or external AI label is introduced.
- The training signal comes entirely from the data's own **intrinsic statistics** — the conditional probability distribution of token sequences. This is the most primitive prediction-statistic form of UPT: simply continuing to minimize the LM loss on new data.
- The replay mechanism only replays raw tokens from the old corpus; it involves no model-generated content and no preference judgment, and remains a predictive likelihood minimization over raw observations.
- Learning-rate re-warming / re-decaying is a pure optimization-level schedule and does not alter the nature of the objective.

This paper is therefore the most typical Family I representative: **continual pre-training that keeps minimizing the LM loss on a new text-only corpus.**

---

## 2. Problem Addressed

Large language models (LLMs) are typically pre-trained from scratch on hundreds of billions of tokens. When new data becomes available, retraining from scratch is computationally prohibitive. **Continual pre-training (CPT)** is a more efficient alternative but faces two core challenges:

1. **Catastrophic forgetting:** continuing training on new data degrades performance on old data.
2. **Poor adaptation:** simply continuing training on new data (especially with a near-zero learning rate) leaves the model under-adapted to the new data.

These two problems are most acute at scale (200B+ tokens) and under distribution shifts of varying strength (weak: English→English; strong: English→German). The paper sets out to answer: **when simple and scalable continual-learning techniques are applied, how large is the performance gap between a CPT'd model and a model trained from scratch on all the data?**

---

## 3. Method

The paper proposes a combination of three simple and scalable strategies:

### 3.1 Learning Rate (LR) Re-warming

Most open-source LLMs use a cosine decay schedule that ends training at a tiny LR (eta_min). Continuing training at this low LR causes under-adaptation. The paper proposes **re-warming** the LR back to a high value (eta_max) when switching to the new dataset, typically restoring the original pre-training eta_max (e.g. 3e-4).

### 3.2 Learning Rate (LR) Re-decaying

After re-warming, cosine-anneal the LR from eta_max back down to eta_min = 0.1 * eta_max. The combination of re-warming and re-decaying is essential for maximizing adaptation to the new data. Experiments find that:
- A higher eta_max yields both stronger forgetting and stronger adaptation.
- A lower eta_max yields both weaker forgetting and weaker adaptation.
- Linear-warmup duration (0%–2%) has little impact on the final performance; the default is set to 1%.

### 3.3 Compute-equivalent Replay

While training on new data, mix in a small fraction of old data (replay). For a fair comparison, the paper adopts **compute-equivalent replay**: the token budget consumed by replay is deducted from the new-data token budget, keeping total compute constant. For example, 5% replay means each batch contains 5% samples from the old dataset D_0 and 95% from the new dataset D_1.

Key findings:
- Even a tiny amount of replay (1%) substantially mitigates forgetting.
- Under weak distribution shift, 5% replay suffices; under strong shift, 25% replay is needed.
- Replay has minimal impact on adaptation to the new data.

### 3.4 Infinite Learning Rate Schedules

As a complement, the paper also proposes **infinite LR schedules** as an alternative to cosine decay. These schedules transition to a constant high LR after the pre-training stage, avoiding the optimization difficulty caused by re-warming and not being tied to a fixed token budget. They are especially useful when many dataset switches (N > 2) are needed.

---

## 4. Datasets

The paper uses three large-scale pre-training datasets:

| Dataset | Scale | Use |
|--------|------|------|
| **Pile** | ~330B tokens (300B subset used) | D_0: initial pre-training data (English) |
| **SlimPajama** | 606B tokens (downsampled to ~299B) | D_1: weak-shift target (English→English) |
| **German Common Crawl** (Oscar) | ~195B tokens | D_1: strong-shift target (English→German) |

Domain composition of the SlimPajama 300B subset:

| Domain | Size | Sampling % |
|--------|------|-----------|
| Common Crawl | 155.89B | 52.09 |
| C4 | 79.87B | 26.69 |
| GitHub | 15.63B | 5.22 |
| Book | 12.58B | 4.20 |
| Wikipedia | 11.96B | 4.00 |
| Arxiv | 13.25B | 4.43 |
| Stack Exchange | 10.09B | 3.37 |

Experimental settings include:
- **Two datasets, weak shift:** Pile (300B) → SlimPajama (300B)
- **Two datasets, strong shift:** Pile (300B) → German (200B)
- **Three datasets, no shift:** three 100B splits of SlimPajama
- **Domain incremental:** SlimPajama trained domain by domain

---

## 5. Evaluation metrics and main results

### Metrics

- **Validation loss** on D_0 (Pile) and D_1 (SlimPajama / German): final validation loss (averaged over the last 100 iterations).
- **English LM evaluation benchmarks (zero-/few-shot accuracy):**
  - Commonsense Reasoning (0-shot): HellaSwag, Winogrande, PIQA, OpenBookQA, ARC-Easy, ARC-Challenge
  - World Knowledge (5-shot): NaturalQuestions, TriviaQA
  - Reading Comprehension (0-shot): BoolQ
  - Math: MathQA
  - Aggregated: MMLU (5-shot)
- **German benchmarks:** HellaSwag-DE, ARC-Challenge-DE, TriviaQA-DE, MMLU-DE

### Main results

**405M model, weak shift (Pile→SlimPajama):**
- CPT with LR re-warming + re-decaying + 5% replay reaches almost the same average validation loss (2.37 vs 2.35) and downstream evaluation accuracy as a baseline trained from scratch on Pile ∪ SP (600B).
- Uses only about half the compute.

**405M model, strong shift (Pile→German):**
- CPT with LR re-warming + re-decaying + 25% replay reaches the same average validation loss as the Pile ∪ German (500B) baseline (1.75 vs 1.75).
- English evaluation accuracy: 32.48 vs 32.43; German HellaSwag: 31.04 vs 30.45.

**10B model, weak shift (Pile→SlimPajama):**
- CPT + 5% replay again matches the from-scratch baseline, confirming the method scales to larger model sizes.

**Effect of replay (405M, Table 2):**
- Weak shift: no replay AVG loss = 2.47, 5% replay = 2.37, 50% replay = 2.35 (≈ union baseline 2.35).
- Strong shift: no replay AVG loss = 2.34, 25% replay = 1.75, 50% replay = 1.73 (≈ union baseline 1.75).
- Replay substantially reduces forgetting while barely affecting adaptation.

**Core conclusion:** the simple combination of LR re-warming, LR re-decaying, and a small fraction of replay is enough to make a continually pre-trained LLM match — on both validation loss and downstream evaluation — a model trained from scratch on all the data, while saving substantial compute.
