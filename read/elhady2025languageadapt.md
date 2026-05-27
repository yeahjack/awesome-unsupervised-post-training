# LangAdapt CPT — Emergent Abilities of Large Language Models under Continued Pre-Training for Language Adaptation

> **Added to survey on:** 2026-03-11

- **Method:** LangAdapt CPT
- **Carrier:** Direct Opt.
- **Regime:** Training-time
- **Level:** Token
- **Authors:** Ahmed Elhady, Eneko Agirre, Mikel Artetxe
- **Affiliations:** HiTZ Center, University of the Basque Country (UPV/EHU); Reka AI
- **Venue:** ACL 2025 (Long Papers)

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

Core reasons:
- The training signal comes entirely from the **language-modeling objective (LM loss / next-token prediction)**, i.e. predictive likelihood minimization; it relies on no external annotation, verifier, or human/AI preference label.
- The internal artifact is **predictive likelihood**: the model directly minimizes cross-entropy loss on a target-language corpus, and its perplexity (the exponentiated cross-entropy) is the intrinsic statistic.
- The entire continued-pretraining (CPT) process applies only standard LM-loss updates to the weights, i.e. direct optimization on intrinsic statistics.
- The two proposed alternatives (curriculum learning and EMA) only manipulate the training schedule and parameter averaging; they introduce no external supervision signal.

---

## 2. Problem Addressed

Existing LLMs (e.g. Llama 2) are English-centric, with substantially lower performance on low-resource languages. CPT is the dominant approach to adapting an LLM to a new language; in practice English data is often mixed into the target-language corpus, yet **the specific role of English data during CPT has not been systematically studied.**

The paper answers the following key questions:
1. **Is mixing in English data really necessary?** The authors find that the English-free CPT variant matches the English-mixed variant on target-language validation perplexity but underperforms substantially on downstream-task accuracy.
2. **What is the root cause of the gap?** English-free CPT suffers catastrophic forgetting in the very first training steps, manifested as a sharp loss of in-context-learning (ICL) ability and a drastic increase in parameter shift.
3. **Can other techniques substitute for English data?** The authors explore curriculum learning and EMA (exponential moving average) to mitigate parameter drift, thereby reducing or removing the dependence on English data.

---

## 3. Method

### 3.1 Basic CPT pipeline

Starting from an existing English LLM, perform full-parameter continued pretraining on a target-language corpus with the standard next-token-prediction loss as the objective.

Two common variants:
- **CPT (lang):** target-language data only.
- **CPT (lang+en):** target-language data + 20% English data mixed in.

### 3.2 Copain Benchmark

The authors propose **Copain (Contextual Pattern Inference)** — a language-agnostic ICL evaluation benchmark with 7 tasks (150 examples each, 1050 test items total) — to disentangle ICL ability from target-language knowledge:
- Identify min/max/median in an integer list.
- Identify numbers at odd/even positions.
- Identify alphabetical first/last characters.

Inputs are lists of numbers or characters, with no natural-language instruction; the model must infer the task rule from few-shot demonstrations.

### 3.3 Key findings

1. **Perplexity does not tell the whole story:** the two CPT variants have nearly identical validation perplexity but dramatically different downstream and Copain performance.
2. **Catastrophic forgetting of ICL:** in the English-free CPT, ICL ability collapses to near zero within the first few steps (Copain accuracy drops from ~44 to ~0) and recovers only partially and slowly thereafter.
3. **Parameter shift:** the L2 parameter distance of the English-free variant in the first 10 steps already exceeds the total change accumulated by the English-mixed variant over the entire training; at step 100 the L2 distance is 7× that of the English-mixed variant, and at step 1000 it is 15×.
4. **Critical period:** distribution-shift damage concentrates in the first 1k CPT steps.

### 3.4 Alternatives

#### Curriculum Learning
- Mix English data only during the first 10% of training steps (the first 1k steps); switch to pure target-language training afterward.
- Achieves results comparable to mixing English throughout, confirming that the critical period is at the start.

#### EMA of Model Parameters
- Every η steps, take a weighted average of the current and the η-steps-ago parameters: θ_t = α·θ_{t-η} + (1-α)·θ'_t, with α = 0.92.
- Requires no English data at all; mitigates catastrophic forgetting by constraining parameter shift.
- Basque and Indonesian use η=1; Arabic uses η=10.
- Matches the English-mixed CPT on validation perplexity and downstream accuracy, but Copain results fluctuate and are sensitive to hyperparameters.

---

## 4. Datasets

### Training data
| Language | Corpus | Tokens |
|------|------|-----------|
| Basque (eu) | Latxa corpus (Etxaniz et al., 2024) | 4.7B tokens |
| Arabic (ar) | CulturaX (Nguyen et al., 2023), random sample | ~4.5–4.7B tokens |
| Indonesian (id) | CulturaX (Nguyen et al., 2023), random sample | ~4.5–4.7B tokens |
| English (en) | The Pile (Gao et al., 2020), 500k document sample | 20% of the CPT total |

### Evaluation benchmarks
| Language | Benchmark | Type |
|------|-----------|------|
| Basque | EusTrivia, EusProficiency, EusExams, EusReading | Multiple-choice (native Basque) |
| Basque | MGSM-eu (Baucells et al., 2025) | Generative math (translated from English) |
| Arabic | ArabicMMLU (Koto et al., 2024) | Multiple-choice, 5 sub-tasks |
| Indonesian | IndoMMLU (Koto et al., 2023) | Multiple-choice, 5 sub-tasks |
| All languages | Copain (proposed) | Language-agnostic ICL, 7 tasks |

---

## 5. Evaluation metrics and main results

### Metrics
- **Validation Perplexity (PPL):** perplexity on the target-language validation set.
- **Downstream Accuracy (Dwn):** mean accuracy on target-language multiple-choice benchmarks (5-shot; EusReading 1-shot).
- **Copain Accuracy (Cop):** exact-match accuracy on the language-agnostic ICL benchmark (3-shot).

### Main results (Llama 2 7B as the base model)

| Configuration | PPL (eu) | Dwn (eu) | Cop (eu) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 23.64 | 27.43 | 44.67 |
| + CPT (eu+en) | **3.35** | **34.14** | 43.43 |
| + CPT (eu) | 3.58 | 28.89 | 20.12 |

| Configuration | PPL (ar) | Dwn (ar) | Cop (ar) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 4.36 | 32.45 | 44.67 |
| + CPT (ar+en) | 2.09 | **34.34** | 32.60 |
| + CPT (ar) | 2.12 | 32.67 | 23.80 |

| Configuration | PPL (id) | Dwn (id) | Cop (id) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 6.27 | 26.65 | 44.67 |
| + CPT (id+en) | 3.25 | **30.79** | 30.79 |
| + CPT (id) | **3.05** | 26.92 | 27.34 |

### Core conclusions

1. **English data does not affect perplexity but markedly improves downstream performance:** the two CPT variants reach comparable perplexity, yet the English-mixed version has higher downstream accuracy across all languages and models, with the largest gap on Basque (34.14 vs 28.89).
2. **The weaker the base model, the more critical English data is:** Llama 2 13B shows a 7-point downstream gap on Basque (42.52 vs 35.20), whereas Llama 3.1 8B and Gemma 2 9B have smaller gaps thanks to their stronger initial multilingual ability.
3. **Curriculum learning works:** mixing in English only during the first 10% of training steps reaches results comparable to mixing throughout (Basque: Dwn 35.12 vs 34.14).
4. **EMA can fully replace English data:** without any English, EMA performs best on perplexity and is comparable to the English-mixed version on downstream (Arabic: Dwn 33.36 vs 34.34), although Copain results are sensitive to the hyperparameter η.
5. **Emergent abilities surface abruptly:** the improvement in downstream accuracy shows abrupt emergence (the Basque English-mixed version gains 8 points between steps 2k and 4k), challenging the prior assumption that "similar perplexity implies similar downstream performance."
