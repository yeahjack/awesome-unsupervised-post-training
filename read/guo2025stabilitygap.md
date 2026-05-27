# Stability-Gap CPT

> **Added to survey on:** 2026-03-11

**Paper:** Efficient Domain Continual Pretraining by Mitigating the Stability Gap
**Authors:** Yiduo Guo, Jie Fu, Huishuai Zhang, Dongyan Zhao (Peking University, HKUST)
**Venue:** ACL 2025 (Long Paper)
**Method:** Stability-Gap CPT | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

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
This paper belongs to **Family I (Prediction-Statistic Optimization)**: the core objective is always **token-level language-modeling loss (predictive likelihood minimization)**. Although the paper introduces three stabilization mechanisms (multi-epoch subset training, domain-PPL-based data selection, and pretraining-mixture-rate matching), all of them serve to minimize the LM loss on the domain corpus more efficiently, and none relies on external annotation, external verifier, or human preference signal. The KenLM perplexity used for data selection is itself an intrinsic statistic (the model's own n-gram probability), fully consistent with the UPT definition.

---

## 2. Problem Addressed
Continual pretraining (CPT) is a common way to adapt a general LLM to a specific domain (medical, legal, etc.). However, the authors observe a phenomenon that is widespread across model scales and domains: **early in CPT, target-domain task performance first drops, then gradually recovers and eventually surpasses the original model**, tracing a V-shaped curve. The authors draw an analogy with the **stability gap** seen in visual continual learning, attributing the root cause to **the plasticity gradient (learning new knowledge) overwhelming the stability gradient (retaining old knowledge) at the start of training**, which temporarily degrades the model's general abilities (instruction-following, commonsense).

This stability gap wastes compute budget — the model needs many tokens to recover from the initial drop. The paper aims to:
1. systematically identify and explain the stability gap in CPT;
2. propose efficient strategies under a **fixed compute budget** that mitigate the stability gap so the model reaches higher performance more quickly.

---

## 3. Method
The authors propose three complementary efficient CPT strategies:

### Strategy I: Multi-epoch Subset Training
Instead of training for one epoch on the entire large corpus, **train for multiple epochs on a smaller subset**. Key insight: a large corpus requires sustained high plasticity gradients for a long time, whereas multi-epoch training on a small subset can shorten the duration of the plasticity gradient after the first epoch, allowing the stability gradient to rise and performance to recover sooner. Experiments show peak performance around the 4th epoch.

### Strategy II: Domain PPL-based Data Selection
Use **KenLM** (an n-gram model trained on a medical Wikipedia corpus) to compute the **domain perplexity (PPL)** of each sample in RefinedWeb, and select the lowest-PPL subset as the domain corpus. A lower domain PPL implies a better match with the target-domain distribution, which speeds recovery and raises peak performance.

### Strategy III: Pretraining Mixture Rate Matching
Construct the CPT data mixture according to the original pretraining data's **mixture rate** (proportions of each data source). Concretely: first sample 5B tokens at the Llama mixture rate, then replace the CC and C4 portions (about 82%) with the KenLM-selected medical tokens from Strategy II. Re-sample and re-replace at every epoch to reduce distribution shift and stabilize instruction-following ability. Two variants are proposed:
- **Rate-Fixed-Data-Fixed:** sample once; all epochs use the same corpus.
- **Rate-Fixed-Data-Dynamic:** sample independently for each epoch, producing a dynamically changing training corpus.

The three strategies can be combined to jointly mitigate the stability gap.

---

## 4. Datasets
### Pretraining data
- **Base models:** OpenLLaMA-3B-v2 (trained on RefinedWeb), TinyLlama-1.1B, Llama-3-8B.
- **Source of the domain corpus:** RefinedWeb dataset, sorted by domain PPL via KenLM (trained on a medical Wikipedia corpus); the lowest-PPL subset is selected.
- **Compute budget:** the baseline uses 50B medical tokens (1 epoch); the proposed strategies use 5B tokens × 4 epochs = **20B tokens** (only 40% of the baseline).
- **Domains:** main experiments are on **medical**; supplementary experiments cover the **legal** domain and a **general** continual-pretraining setting.

### Evaluation data
- **MMLU-Medical:** medical genetics, anatomy, clinical knowledge, professional medicine, college medicine.
- **PubMedQA** (Jin et al., 2019).
- **MedMCQA** (Pal et al., 2022).
- **MedQA-4-Option** (Jin et al., 2021a).
- Evaluation framework: lm-evaluation-harness (Gao et al., 2023).

### Llama-3-Physician additional evaluation
- Classification (HOC), Relation Extraction (DDI-2013), BioNLI, Summarization (MIMIC-CXR).

---

## 5. Evaluation metrics and main results
### Metrics
- **Zero-shot accuracy** on the medical benchmarks.
- **Average medical performance:** mean accuracy across the four medical benchmarks above.
- **Commonsense task performance:** measures retention of general ability.
- **Medical perplexity (PPL):** perplexity on the medical Wikipedia corpus.

### Main results

**OpenLLaMA-3B experiments (Table 1):**

| Method | Tokens | MMLU-Med | PubMedQA | MedMCQA | MedQA-4-Opt | Avg |
|---|---|---|---|---|---|---|
| OpenLLaMA-3B (original) | - | 25.6 | 68.4 | 25.4 | 25.4 | 36.2 |
| Full token baseline | 50B | 26.1 | 70.4 | 26.1 | 27.1 | 37.4 |
| Replay 10B data | 50B | 29.3 | 71.0 | 30.4 | 27.6 | 39.5 |
| **Our strategies** | **20B** | **30.0** | **71.2** | **34.0** | **27.8** | **40.7** |

- Using only **40% of the compute budget (20B vs 50B tokens)**, the average medical performance rises from 36.2% to **40.7%** (+4.5%), beating all baselines.
- **No catastrophic forgetting of general ability:** commonsense performance even improves.

**Llama-3-Physician-8B (Table 2, task-specific fine-tuning):**

| Model | MMLU-Med | PubMedQA | MedMCQA | MedQA-4-Opt | Avg |
|---|---|---|---|---|---|
| Llama-3-8B base | 47.2 | 52.1 | 38.2 | 35.5 | 43.3 |
| Llama-3-Physician-8B (ours) | **85.0** | **79.1** | **81.4** | 61.5 | **76.7** |
| GPT-3.5-turbo-finetuned | 70.5 | 71.4 | 61.8 | 63.3 | 66.7 |
| Llama-2-70B | - | 78.0 | 62.7 | 61.3 | 67.2 |

- Llama-3-Physician-8B is the best among open-source models in the same scale class (7–8B).
- Average performance approaches GPT-4; on classification, relation extraction, NLI, and summarization tasks it substantially exceeds GPT-4.
- **The 7B model even beats several 70B models on average.**

### Key ablation findings
- **Multi-epoch training** accelerates recovery, with peak performance at epoch 4.
- The **KenLM-selected subset** recovers faster and peaks higher than a random subset.
- **Data-mixture-rate matching** preserves medical performance while improving commonsense performance.
- A learning rate that is too high causes a large drop in general ability; too low blocks new-knowledge acquisition; too-large subsets introduce a stability gap; too-small subsets lead to late-stage overfitting.
