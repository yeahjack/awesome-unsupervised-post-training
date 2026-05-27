# TTT-NN: Test-Time Training on Nearest Neighbors for Large Language Models

> **Added to survey on:** 2026-03-11

**Method:** TTT-NN | **Carrier:** Direct Opt. | **Regime:** test-time | **Level:** Token

**Authors:** Moritz Hardt, Yu Sun
**Venue:** ICLR 2024
**arXiv:** 2305.18466

| When to Adapt | Test-Time Instance Adaptation with reset at instance boundary |
|---|---|
| Trigger Unit | arriving sample / prompt |
| Persistence | sample-local parameter, adapter, or state update; reset after inference |
| Inference Coupling | adapt on the current sample for the current sample |
| Input Visibility | Online |
| Update Persistence | Non-Cumulative |
| Reset Boundary | Sample Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Test-Time Instance Adaptation |
| Visibility Scope | Current Instance Only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Test-Time Instance Adaptation`; `Visibility Scope=Current Instance Only`.
- **Two-axis coding:** `Input Visibility=Online`; `Update Persistence=Non-Cumulative`; `Reset Boundary=Sample Boundary`.

- **When the update is triggered:** the update is triggered by a single arriving sample / prompt; a small amount of optimization is done directly around the current sample.
- **Whose sample it serves:** the update from the current sample mainly serves the current sample itself, not later samples.
- **Whether parameters/state accumulate:** parameters / adapters / local state exist only briefly within a sample and are reset / discarded after inference.
- **Reset boundary:** this method is the most typical test-time-instance adapt-and-reset protocol.

## 1. UPT Assignment Rationale

TTT-NN belongs to **Family I (Prediction-Statistic Optimization)**, specifically predictive likelihood minimization. Its core mechanism: at test time, for each test sample, retrieve nearest-neighbor sequences from a large training corpus, then run a few gradient steps of language-modeling loss (LM loss / NLL) on these neighbors. The process relies on no external labels, no human feedback, and no external verifier; the only training signal is the model's own predictive NLL on the retrieved neighbor text. After each test instance, model parameters are reset to the pretrained state — a purely intrinsic-statistics-based test-time adaptation.

---

## 2. Problem Addressed

Existing retrieval-augmented LM methods concatenate retrieved text into the input context, which grows the input length linearly and self-attention quadratically, and usually also requires retrieval at training time (train–test consistency). TTT-NN targets:

- **How can we improve LM performance with retrieved neighbors without modifying the architecture or extending the context length?**
- **How can a standard pretrained model (with no special retrieval-aware training) adapt to relevant data at test time?**
- **How can we process multiple neighbors at linear (not quadratic) cost?**

---

## 3. Method

### 3.1 Building the nearest-neighbor index

- Build a large neighbor index over the Pile training set (~210M sequences, 1.3 TB of text).
- Use RoBERTa-large (355M parameters, trained on Pile) to produce 1024-dim text embeddings.
- Store all embeddings in a FAISS Flat L2 index (~810 GB; 2.1 TB total with vectors).
- Deploy a distributed client–server setup on 180 servers; a single query takes ~1 s.

### 3.2 Test-Time Training pipeline

For each test sequence:

1. **Retrieve:** query the index with the test sequence's embedding to get k nearest neighbors (default k=50).
2. **Order and sequential training:** process neighbors in **increasing-distance** order (farthest to nearest), one gradient update each (standard LM loss). "Far-to-near" is better than "near-to-far": large steps on farther data first guide fine-tuning into a better region of the loss landscape.
3. **Long-sequence chunking:** if a neighbor sequence exceeds the model's maximum length, chunk it and apply one gradient update per chunk.
4. **Evaluate:** measure the perplexity of the test sequence under the fine-tuned model.
5. **Reset parameters:** after evaluation, reset the model to the original pretrained state and proceed to the next test instance.

### 3.3 Key design choices

- Use the model's default optimizer and hyperparameters — **no extra tuning needed**.
- Each neighbor goes through an independent forward + backward pass; compute scales **linearly** with the number of neighbors (vs. quadratic for in-context methods).
- 20 neighbors already capture most of the gain, with just 1 gradient iteration per neighbor.

---

## 4. Datasets

- **Training corpus / index source:** the Pile training set (~210M sequences, 1.3 TB), covering 22 sub-tasks (arxiv, github, books3, pile-cc, pubmed, etc.).
- **Evaluation set:** the Pile test set (214,584 sequences); a 20% subset (42,916 sequences) is used for actual evaluation.
- The index does **not** contain validation or test data.

---

## 5. Evaluation metrics and main results

### Metrics

- **Bits per byte (BPB):** NLL divided by a dataset-specific normalization constant — a standardized perplexity measure. Computed with Eleuther-AI's `lm-evaluation-harness`.

### Main results

**Model scale and overall effect:**

| Model | Params | BPB (pile_all) before TTT-NN | BPB (pile_all) after TTT-NN | Reduction |
|------|--------|--------------------------|--------------------------|---------|
| GPT-2 Small | 117M | — | — | 82% retained |
| GPT-2 Large | 774M | 1.07 | **0.85** | — |
| GPT-Neo | 1.3B | — | — | ~97% retained |

**GPT-2 Large vs. baselines (pile_all BPB):**

| Method | BPB |
|------|-----|
| Base Only | 1.07 |
| In-Context (neighbors concatenated) | 1.06 |
| Interpolate (KNN-LM variant) | 0.99 |
| Dynamic Evaluation | 1.03 |
| **TTT-NN (this paper)** | **0.85** |

**Key findings:**

- TTT-NN lowers BPB on all 22 Pile sub-tasks; overall reduction is about **20%**.
- **Code generation (pile_github) benefits the most:** GPT-2 Small's BPB drops by over 60% (from 1.95 to 0.51); GPT-2 Large drops to 26% of its original BPB.
- A small GPT-2 (117M) with TTT-NN substantially closes the gap to the 10× larger GPT-Neo (1.3B).
- 20 neighbors already capture most of the gain; adding more (up to 50) gives marginal improvements.
- For domains the model has not seen during training (e.g. github for GPT-2), TTT-NN's gain is especially pronounced; for seen domains (e.g. pile-cc) the gain is more modest but consistent.
- TTT-NN **outperforms all three baselines** (In-Context, Interpolate, Dynamic Evaluation) on the vast majority of tasks.
