# ReplayAlign CPT — Revisiting Replay and Gradient Alignment for Continual Pretraining of Large Language Models

> **Added to survey on:** 2026-03-11

> **Paper:** Revisiting Replay and Gradient Alignment for Continual Pretraining of Large Language Models
> **Authors:** Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Thérien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish
> **Affiliations:** Université de Montréal, Mila, IBM Research, Fujitsu Research, Polytechnique Montréal
> **Venue:** 4th Conference on Lifelong Learning Agents (CoLLAs), 2025
> **arXiv:** 2508.01908
> **Method:** ReplayAlign CPT | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

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

This paper belongs to **Family I (Prediction-Statistic Optimization)**, specifically the **predictive likelihood minimization** sub-class.

- **Core training objective:** the standard language-modeling cross-entropy loss (i.e. next-token prediction), without reliance on any external annotation, human preference, or external verifier.
- **Replay mechanism:** an unbounded on-disk replay buffer is mixed with new data at proportion α per batch, optimizing the LM loss directly. This is a purely intrinsic-statistics optimization strategy (the model's own predictive likelihood).
- **Gradient Alignment (Reptile meta-update):** periodic parameter interpolation via Reptile first-order meta-learning (every k=500 steps), whose regularization objective is to maximize the dot product between gradients of old and new batches, promoting transfer and reducing interference. This regularization signal is derived entirely from the model's own gradient statistics, requiring no external supervision.
- **No external signal:** the method uses no ground-truth labels, no human feedback, no external AI labels, and no external verifier, fully matching the UPT definition.

---

## 2. Problem Addressed

**Catastrophic forgetting and the stability–plasticity dilemma in continual pre-training (CPT).**

- LLMs need to be continually updated on new data / new domains, yet retraining from scratch is prohibitively expensive. Continual pre-training is a more efficient alternative, but introducing new data tends to make the model forget old knowledge (catastrophic forgetting).
- In the existing continual-learning literature, experience replay and gradient-alignment methods work well at small model sizes, but **have not been systematically evaluated at modern LLM scale.**
- Key question: under a fixed compute budget, which strategy is most efficient — increasing the replay ratio, scaling the model up, or adding gradient alignment?

---

## 3. Method

### 3.1 Experience Replay

- Maintain a disk-based "unbounded" replay buffer that stores all training data seen so far.
- At each training step, a fraction α of the batch is sampled randomly from the replay buffer; the rest comes from the current new-data stream p(x|t).
- Experiments test α ∈ {0, 0.25, 0.5} (0%, 25%, 50% replay).
- The disk implementation uses asynchronous prefetching and caching, compatible with the Megatron/NeoX framework, avoiding GPU/RAM bottlenecks.

### 3.2 Meta-Experience Replay (MER)

- Based on Riemer et al. (2019a), the paper introduces **Reptile first-order meta-learning** for gradient alignment.
- Approximate Reptile objective:

  $$\arg\min_{\theta_t} \mathbb{E}_{B_1,...,B_k}\left[2\sum_{i=1}^{k}L(B_i) - \sum_{j=1}^{i-1}\beta\frac{\partial L(B_i)}{\partial \theta_t}\cdot\frac{\partial L(B_j)}{\partial \theta_t}\right]$$

  The second term promotes alignment between gradients of different batches (maximizing dot product), thereby encouraging positive transfer and reducing interference.

- **Implementation:** every k=500 steps, perform a parameter interpolation θ_t ← θ_{t-k} + ε(θ_t − θ_{t-k}), with ε=0.1.
- Computational overhead is tiny: every 500 batches adds only ~3× model-size FLOPs (negligible relative to gradient-update cost).

### 3.3 Full Method: Replay + Reptile (MER)

- Combines the replay buffer with the Reptile meta-update: replay places α% of old data in the batch sequence, and Reptile periodically aligns gradients on top of this.
- The two cooperate: replay provides a direct optimization signal on old data, while gradient alignment encourages parameters that generalize to both old and new data.

---

## 4. Datasets

### Training data (sequential CPT, 100B tokens per task)

| Task | Dataset | Language | Tokens |
|------|--------|------|--------|
| A | DCLM-Baseline (DataComp-LM, CommonCrawl subset) | English | 100B |
| B | OSCAR (CommonCrawl extraction) | French | 100B |
| C | OSCAR | German | 100B |
| D (extended) | Aloui et al. (2024) | Arabic | 100B |
| E (extended) | CommonCrawl-derived corpus (Hattori, 2024) | Japanese | 100B |

- 3-task setting: English → French → German.
- 5-task extended setting: English → French → German → Arabic → Japanese.

### Models

- **Spectra LLM suite** (Llama-architecture family): 99M, 560M, 1B, 6B.
- AdamW optimizer, cosine learning-rate schedule, 357-step warmup, batch size 4096.
- One epoch per stage; no gradient/optimizer reset between stages.

---

## 5. Evaluation metrics and main results

### Metrics

- **Forgetting Score:** the increment in current validation loss relative to the historical best validation loss (lower is better).
- **Retained Loss:** average validation cross-entropy loss across all tasks at the end of training (measures stability).
- **Learned Loss:** validation loss on each task immediately after training on that task (measures plasticity).
- **Downstream Performance:** zero-shot results on HellaSwag, PiQA, PubMedQA, etc.

### Main results

**Q1: Effect of Experience Replay**
- Replay significantly reduces forgetting: the 560M model's final validation loss is ≈3.3 with no replay vs. ≈3.0 with 50% replay.
- 560M + 50% replay approaches the performance of the 1B no-replay model, showing replay is more compute-efficient than simply scaling the model.

**Q2: Synergy between Gradient Alignment and Replay**
- 50% replay + Reptile consistently achieves the lowest average forgetting score across all model sizes.
- 560M model: 25% replay + Reptile reaches a downstream average of 67.5, beating the 560M joint-training baseline (67.33), pure replay (66.4), and pure Reptile (65.4).
- 6B model shows the largest effect: joint-training baseline average is 72.6; 25% replay + Reptile reaches 76.8; 50% replay + Reptile reaches 77.1, **outperforming joint training**.
- In the 5-task extended experiments, 50% replay + Reptile even achieves **negative forgetting** (backward transfer), showing the method scales to longer task sequences.

**Q3: Gradient Alignment promotes generalization**
- Adding Reptile brings the task-specific validation-loss curves closer to the joint-training baseline (especially on French/German), indicating that gradient alignment effectively encourages cross-task generalization.

**Q4: Computational scaling**
- Both stability and plasticity scale as an inverse power law of FLOPs/token.
- 25% replay adds only 1.33× FLOPs/token, more efficient than 50% replay (2× FLOPs/token).
- Reptile's gain is almost "free" — it brings consistent improvements on both stability and plasticity scaling curves without notable extra cost.
- The scaling trend of 25% replay + Reptile is even better than 25% replay without Reptile, with the gap widening as the model size grows.
