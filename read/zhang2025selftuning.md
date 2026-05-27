# Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching

> **Added to survey on:** 2026-03-11

> **Method:** Self-Tuning | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | document corpus / self-teaching task batch |
| Persistence | full parameter accumulate across Stage 1–3 |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | note-explicit |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates are organized around the document corpus and self-teaching tasks in a staged manner, with the core being an offline Stage 1→Stage 2→Stage 3 schedule.
- **Serving the current sample or future ones:** updates in the current stage primarily serve subsequent stages and the final deployed model, not the immediate inference of a particular test sample.
- **Whether parameters/state accumulate:** parameters accumulate continuously across the three stages, with no sample-level reset in the paper.
- **Reset boundary:** consequently it is closer to staged document adaptation than to general online TTA.

## 1. UPT Assignment Rationale

Self-Tuning belongs to **Family III — Self-Generated Target Bootstrapping (knowledge/instruction self-curation)**.

Core rationale: the key component of Self-Tuning, the Self-Teaching strategy, generates knowledge-intensive tasks (memorization, comprehension, self-reflection subtasks) from raw documents in a fully self-supervised manner, with no external annotation, external verifier, or human labels. Specifically:

- **Memorization**: direct next-token prediction on raw documents — an intrinsic signal.
- **Comprehension**: automatically generates summarization (document title as ground truth), gist identification (Spacy-extracted entities as answers), and NLI (randomly sampled sentences and entity replacement for positive/negative examples) — all targets derived automatically from document content, with no dependence on external models or humans.
- **Self-Reflection**: automatically constructs closed-book generation tasks such as Teaching, Flashcards, Fill-in-the-Blank, Multi-choice QA, and Sentence Completion, with targets derived from document content.

The model creates knowledge/instruction targets for itself and feeds them back into self-teaching, fully matching the definition of self-generated target bootstrapping.

---

## 2. Problem Addressed

LLMs suffer from a knowledge cutoff due to one-shot pre-training and struggle to continually acquire new knowledge. Existing approaches (e.g., continued pre-training, standard instruction-tuning) can inject new documents into model parameters but perform poorly at the **extraction** stage — the model fails to effectively recall the injected knowledge at inference time. Self-Tuning aims to address:

1. **Insufficient knowledge absorption**: plain continued pre-training only achieves surface memorization, lacking deep understanding.
2. **Difficulty of knowledge extraction**: injected knowledge cannot be effectively retrieved and used in QA settings.
3. **Knowledge forgetting**: after learning new knowledge, old knowledge is prone to catastrophic forgetting.

---

## 3. Method

Inspired by the Feynman Technique, Self-Tuning consists of **three training stages** and a core **Self-Teaching strategy**.

### Three-stage training pipeline

**Stage 1: Learn How to Effectively Absorb Knowledge from Raw Documents**
- Jointly trains on training documents $D_{\text{train}}^{\text{Doc}}$, training QA data $D_{\text{train}}^{\text{QA}}$, and Self-Teaching-generated tasks $D_{\text{train}}^{\text{Self}}$.
- Goal: teach the model how to efficiently absorb knowledge from raw documents while retaining QA ability.
- Objective: $L_\theta^{\text{Stage1}} = L_\theta(D_{\text{train}}^{\text{Doc}}) + L_\theta(D_{\text{train}}^{\text{Self}}) + L_\theta(D_{\text{train}}^{\text{QA}})$

**Stage 2: Learn New Knowledge while Reviewing QA Skills**
- Continues pre-training on previously unseen test documents $D_{\text{test}}^{\text{Doc}}$ while mixing in QA data to preserve answering ability.
- Goal: transfer the absorption strategy learned in Stage 1 to new documents.
- Objective: $L_\theta^{\text{Stage2}} = L_\theta(D_{\text{test}}^{\text{Doc}}) + L_\theta(D_{\text{train}}^{\text{QA}})$

**Stage 3: Continually Learn**
- Trains only on test documents $D_{\text{test}}^{\text{Doc}}$ to ensure thorough absorption of new knowledge.
- Objective: $L_\theta^{\text{Stage3}} = L_\theta(D_{\text{test}}^{\text{Doc}})$

### Self-Teaching strategy (core)

Generates knowledge-intensive tasks for documents in a self-supervised way along three dimensions:

**Memorization**:
- Next-token prediction on raw documents.

**Comprehension**:
- *Summarization*: prompt `Write a title:`, with the document title as target.
- *Gist Identification*: prompt `Highlight the key information within the article:`, with Spacy-extracted entities as target.
- *NLI*: random document sentences as positive examples, entity replacement to construct negative examples, model judges Yes/No/Impossible.

**Self-Reflection**:
- *Teaching*: prompt `Tell me about {topic}`, with document content as target (closed-book generation).
- *Flashcards*: prompt `Generate a concrete description about {topic} based on the following keywords:`, document as target.
- *Fill-in-the-Blank*: random entity replacement with blanks, model fills in.
- *Multi-choice QA*: replace entities with `–`, provide four choices (correct entity + three distractors).
- *Sentence Completion*: truncate a document sentence, model completes the following phrase.

None of the tasks rely on external mining patterns; the strategy applies to arbitrary raw text.

---

## 4. Datasets

### Training/evaluation data: Wiki-Newpages-2023-QA

Collected from Wikipedia NewPages, covering articles newly released in September–October 2023 (4,257 in total), ensuring zero overlap with LLM pre-training data. Three sub-datasets are constructed:

| Dataset | Domain | # Train Docs | # Train QA | # Test QA |
|---|---|---|---|---|
| **Wiki-Bio** | Single-domain (biographies) | 1,263 | 6,136 QA + 1,136 Doc | 663 QA + 127 Doc |
| **Wiki-Multi** | Multi-domain (news, sports, etc.) | 2,104 | 10,004 QA + 1,823 Doc | 1,502 QA + 281 Doc |
| **Wiki-Film** | Single-domain (films), test only | — | — | 955 QA + 169 Doc |

- Wiki-Bio and Wiki-Multi are used for single-domain and multi-domain evaluation.
- Wiki-Film is used only as a test set, for cross-domain evaluation (trained on Wiki-Bio).
- QA pairs are generated by GPT-4 from handcrafted prompts, covering all factual information in the documents (open-ended generation + NLI).

### Knowledge retention evaluation data
- **Natural Questions (NQ)**: evaluates old knowledge extraction (EM, F1).
- **CommonsenseQA (CSQA)**: evaluates commonsense-reasoning retention (Accuracy).

---

## 5. Evaluation metrics and main results
### Metrics

| Task | Metric |
|---|---|
| **Memorization** | Perplexity (PPL, ↓) |
| **Extraction** (open-ended generation) | Accuracy, Exact Match (EM), F1, Recall, Rouge-L |
| **Reasoning** (NLI) | Accuracy |
| **Knowledge Retention (Extraction)** | EM, F1 (on NQ) |
| **Knowledge Retention (Reasoning)** | Accuracy (on CSQA) |

### Main results (LLaMA2-7B, 5-shot, Wiki-Bio single-domain)

| Method | PPL ↓ | Extraction EM | Extraction F1 | Reasoning Acc. |
|---|---|---|---|---|
| Cont. Pre-training | 7.28 | 3.62 | 15.96 | 53.40 |
| Standard Ins.-tuning | 6.83 | 5.13 | 19.15 | 51.84 |
| PIT | 2.08 | 11.61 | 27.15 | 57.58 |
| **Self-Tuning** | **1.11** | **31.52** | **50.83** | **66.01** |

### Key findings

1. **Comprehensive lead on knowledge acquisition**: Self-Tuning drives PPL close to 1, improves EM by about 20 points over PIT, and reaches 66.01% reasoning accuracy.
2. **Strong cross-domain generalization**: in the cross-domain setting (Wiki-Bio → Wiki-Film), Self-Tuning still achieves the best performance (EM 16.44, Reasoning 66.34).
3. **Excellent knowledge retention**: on NQ and CSQA, Self-Tuning does not damage old knowledge and even slightly improves it (NQ F1: 25.67, CSQA: 66.01), with no catastrophic forgetting.
4. **Multi-model generalization**: also achieves the best results on Qwen2-7B and Mistral-7B-v0.1, and is effective on the WebNews-2023 corpus.
5. **Not overfitting**: training-dynamics analysis shows Self-Tuning surpasses the open-book baseline within 5 epochs, peaks around 25 epochs, and retains old knowledge with only a 2–3% EM drop under long training.
