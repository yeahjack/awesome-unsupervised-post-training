# CPT-LM: Continual Pre-Training of Language Models

> **Added to survey on:** 2026-03-11

- **Method name in taxonomy:** CPT-LM
- **Full title:** Continual Pre-Training of Language Models
- **Authors:** Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, Bing Liu
- **Venue:** ICLR 2023
- **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token
- **Dominant artifact:** directly minimizes the LM loss on unlabeled text; the training signal is token NLL / perplexity.

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

This paper belongs to **Unsupervised Post-Training (UPT)** and is placed in **Family I: Prediction-Statistic Optimization**, specifically the **predictive likelihood minimization** sub-class. Reasons:

- **No external labels:** the entire continual DAP-training stage uses only unlabeled domain corpora; it relies on no human labels, no external verifier, and no external AI labels.
- **Training signal is token-level NLL:** the core training objective is the Masked Language Model (MLM) loss, i.e. minimizing the token-level negative log-likelihood on unlabeled text. This is the most primitive intrinsic-statistic signal of a language model — the predictive likelihood drawn directly from the data itself.
- **Auxiliary signals are also unsupervised:** the additional contrastive loss compares the model's own representations under different dropout masks (full knowledge vs. previously learned knowledge), which is a self-comparison over model-generated content and introduces no external signal.
- **Importance computation uses an intrinsic proxy:** the proxy KL-divergence loss used to identify the importance of general knowledge also exploits only output differences of the model itself under different dropout masks (robustness); it requires no pre-training data and no external annotation.

In sum, every training signal comes from intrinsic statistics (token NLL/perplexity) and model-generated content (self-comparison of representations), fully consistent with the UPT definition and with Family I's predictive likelihood minimization paradigm.

---

## 2. Problem Addressed

The core problem is **continual domain-adaptive pre-training (continual DAP-training)**: how can a pretrained language model adapt to a sequence of unlabeled domain corpora while neither forgetting older domains nor losing the ability to transfer knowledge? Concretely, the method must simultaneously:

1. **Overcome catastrophic forgetting (CF):** when learning a new domain, do not forget (a) prior-domain knowledge or (b) the general knowledge from the original pretraining.
2. **Achieve knowledge transfer (KT):** including forward transfer (old-domain knowledge helps new-domain learning) and backward transfer (learning a new domain improves old-domain performance).
3. **Require no domain-ID:** at end-task fine-tuning time, the sample's domain need not be known.

Existing continual-learning methods (parameter-isolation, regularization, replay) are ill-suited here: parameter-isolation methods isolate sub-networks and prevent the end-task from leveraging all knowledge; regularization methods (e.g. EWC) trade learning capacity for forgetting prevention; replay methods require storing large amounts of old data.

---

## 3. Method

The proposed method is called **DAS (Continual DA-pre-training of LMs with Soft-masking)** and has two main stages:

### 3.1 Initialization: compute unit importance for general knowledge

- Before continual learning begins, compute the importance $I_l^{(0)}$ of each unit (attention head and neuron) in the Transformer with respect to general knowledge.
- Since the original pre-training data is not accessible, a **proxy KL-divergence loss** is proposed: feed a subset of the current-domain data through the LM twice with different dropout masks, and compute the KL divergence between the two outputs. Units with larger gradients are more important for the model's robustness, i.e. they carry more general knowledge.

### 3.2 Continual Learning: Soft-masking + Contrastive Loss

For each new domain $t$, perform two steps:

**(a) Domain Training:**

- **Soft-masking:** using the accumulated importance $I_l^{(\leq t-1)}$ (aggregated by element-wise max), soft-mask the gradients:
  $$\hat{\nabla}_l = (1 - I_l^{(\leq t-1)}) \otimes \nabla_l$$
  Gradients of important units are suppressed (but not completely blocked) to prevent forgetting; unimportant units update freely, promoting knowledge transfer. The soft-mask is applied only in the backward pass, leaving the forward pass intact so the end-task can still use the full network.

- **Contrastive loss:** contrasts the full-knowledge representation ($o^{\text{full}}$) with the previously-learned-knowledge representation ($o^{\text{prev}}$, obtained by importance-weighted masking), encouraging the new domain to learn a representation that complements existing knowledge:
  $$\mathcal{L}_{\text{DAP-train}} = \mathcal{L}_{\text{MLM}} + \lambda \mathcal{L}_{\text{contrast}}$$

**(b) Importance Computation:**

- After training on the current domain, compute the per-domain unit importance $I_l^{(t)}$ via gradient-based importance (Eq. 3) on the current-domain data, to be accumulated in the next round.

### Key characteristics

- The soft-mask values are continuous in $[0,1]$ (not binary), giving fine-grained control.
- Sub-networks are not isolated; all knowledge accumulates within the same full LM.
- No replay memory and no domain-ID are required.

---

## 4. Datasets

### Unlabeled Domain Corpora for DAP-training (6 domains)

| Source | Dataset | Size |
|--------|---------|------|
| Reviews | Yelp Restaurant | 758MB |
| Reviews | Amazon Phone | 724MB |
| Reviews | Amazon Camera | 319MB |
| Academic Papers | ACL Papers | 867MB |
| Academic Papers | AI Papers | 507MB |
| Academic Papers | PubMed Papers | 989MB |

### Supervised Classification Datasets for End-task Evaluation (6 tasks)

| Dataset | Task | #Training | #Testing | #Classes |
|---------|------|-----------|----------|----------|
| Restaurant | Aspect Sentiment Classification (ASC) | 3,452 | 1,120 | 3 |
| Phone | Aspect Sentiment Classification (ASC) | 239 | 553 | 2 |
| Camera | Aspect Sentiment Classification (ASC) | 230 | 626 | 2 |
| ACL | Citation Intent Classification | 1,520 | 421 | 6 |
| AI | Relation Classification | 2,260 | 2,388 | 7 |
| PubMed | Chemical-protein Interaction Prediction (CHEMPORT) | 2,667 | 7,398 | 13 |

The base model is **RoBERTa-base**. The proxy KL-divergence initialization stage additionally uses Wiki data as a stand-in for $D_0$.

---

## 5. Evaluation metrics and main results

### Metrics

- **Macro-F1 (MF1)** and **Accuracy (Acc)** on the 6 end-task classification datasets (after continual DAP-training is complete, fine-tune and test on each domain).
- **Forgetting Rate (Forget R.)** measures the degree of catastrophic forgetting: $\text{Forget R.} = \frac{1}{t-1}\sum_{k=1}^{t-1}(A_{k,k} - A_{t,k})$, where $A_{k,k}$ is the end-task accuracy on domain $k$ right after training on it, and $A_{t,k}$ is the accuracy on domain $k$ after all domains have been trained. Negative values indicate positive transfer.

### Main results (Table 2, averaged over 5 random seeds)

| Model | Avg MF1 | Avg Acc | Forget R. (MF1) | Forget R. (Acc) |
|-------|---------|---------|-----------------|-----------------|
| RoBERTa (no DAP) | 77.25 | — | — | — |
| Pool (all domains together) | 80.63 | 90.83 | — | — |
| NCL (naive continual) | 80.70 | 76.66 | 1.14 | 1.05 |
| DEMIX | 74.70 | — | 0.15 | — |
| EWC | 74.84 | — | 0.02 | -0.01 |
| DER++ | 75.51 | — | 2.36 | 1.53 |
| **DAS (proposed)** | **81.91** | **77.93** | **-1.09** | **-0.60** |

### Key findings

1. **DAS achieves the highest average performance among all baselines** (MF1 81.91) while also attaining the strongest knowledge transfer (negative forgetting rate = -1.09, i.e. positive transfer).
2. **DAS handles CF and KT simultaneously:** forgetting-focused methods (KD, EWC, DER++) sacrifice learning capacity; transfer-focused methods (BCL, CLASSIC, DEMIX) still forget. DAS achieves both.
3. **Soft-masking outperforms parameter-isolation:** HAT and other binary-mask / sub-network methods perform worse because the end-task cannot exploit the full network's knowledge.
4. **Proxy KL-divergence is effective:** compared with using Wiki data + MLM loss to estimate general-knowledge importance, the proxy KL-divergence is better — it measures robustness, which is domain-agnostic and therefore better reflects general knowledge.
5. **Ablation confirms every component contributes:** removing initialization, soft-masking, or contrastive learning all degrade performance.
