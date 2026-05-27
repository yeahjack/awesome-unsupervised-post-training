# KBAlign: Efficient Self Adaptation on Specific Textual Knowledge Bases

> **Added to survey on:** 2026-03-11

> **Method:** KBAlign | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | self-generated data batch / iteration round |
| Persistence | full parameter accumulate across synthesis / refinement rounds |
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

- **When the update is triggered:** updates run in an offline pre-deployment bootstrapping loop, typically a round-based schedule of "generate data / score / filter / retrain".
- **Serving the current sample or future ones:** synthetic samples or pseudo-targets produced in the current round mainly serve the next training round and the final deployed model, not the immediate inference of a particular test sample.
- **Whether parameters/state accumulate:** parameters accumulate continuously across bootstrapping rounds, with no sample-level reset.
- **Reset boundary:** the `When to Adapt` of such methods is centered on offline iterative bootstrapping rather than online test-time adaptation.

## 1. UPT Assignment Rationale

**Family III — Self-Generated Target Bootstrapping (knowledge/instruction self-curation)**

KBAlign relies on no external supervisory signal (no human labels, no external AI annotation). Its core pipeline: given a domain-specific textual knowledge base (KB), the model **self-generates instruction–response training data around the KB text** (self-annotation), then fine-tunes itself on that data. It further self-verifies and corrects errors through self-verify tuning. All supervision comes from model-generated content, matching the definition of self-generated target bootstrapping.

---

## 2. Problem Addressed

On small-scale domain-specific text KBs (e.g., legal, biomedical documents), existing RAG systems face:

- **Vanilla unsupervised training** (plain language modeling) yields limited benefit and fails to fully align the model with domain knowledge.
- **Supervised fine-tuning** (e.g., GPT-4-annotated training data) is costly and infeasible under confidentiality or compute constraints.
- The model often fails at RAG-time when the retriever returns imprecise context.

KBAlign aims to adapt LLMs efficiently to domain text KBs **at very low cost and without external supervision**, improving knowledge-based QA accuracy.

---

## 3. Method

KBAlign comprises three core modules:

### 3.1 Self Annotation

Using KB raw text, the model automatically generates Q&A training data with a multi-granularity strategy:

- **Short-dependency Annotation**: chunk the KB into fixed-length pieces ≤ 1,024 words; each chunk serves as golden context $C_g$; the model generates a question $Q$ from it, the retriever fetches related context $C_R$, and the model annotates the answer $A$ from $C_g + C_R$. Suitable for precise local factual knowledge.
- **Long-dependency Annotation**: chunk into short < 256-word segments, concatenate paragraphs from the same hierarchical directory or with high embedding similarity as $C_g$, then require the model to generate multi-hop questions covering multiple segments together with long-form integrated answers. Suitable for tasks requiring global information aggregation.

### 3.2 Iterative Tuning

- **Initial Tuning**: fine-tune the model on self-annotated $\langle Q, A \rangle$ data to obtain $M_1$; during training, randomly concatenate golden or retrieved context to bridge the training/inference gap.
- **Self-Verify Tuning**: split the annotated data into $k$ parts; the model trained on the first part performs RAG inference on the second part to obtain predictions $P_2$; the model then compares $P_2$ with ground truth $A_2$ and generates a correction rationale $V_2$. In the next stage, $\langle Q_2, P_2 \rangle \to V_2$ is used as training data. Experiments use a 25% verify + 75% QA mixture, iterating 2–3 times.

### 3.3 Targeted Inference

- **Query Expansion (QE)**: the model first generates a preliminary prediction $P$ (using memorized domain knowledge); the expanded query $Q + P$ is sent to the retriever for more accurate retrieval.
- **Self Verification**: the model leverages the verification capability acquired during iterative tuning to judge confidence over multiple samples and self-select.

---

## 4. Datasets

| Dataset | Type | Description |
|--------|------|------|
| **LooGLE** | Long-context QA | Long-text material as KB; evaluates the model's memorization of specific knowledge |
| **ASQA** | Long-form QA | Evaluates knowledge recall and information organization (uses only the test set, not training data) |
| **JEC-QA** | Chinese legal multiple-choice | Evaluates domain-specialized learning and instruction following (single + multi-choice) |
| **BioASQ** | Biomedical QA | Evaluates biomedical knowledge retrieval and reasoning |
| **WebASQ** | General KBQA benchmark | Validates generalization of the method |

KB sizes range from 0.41M to 21M tokens.

---

## 5. Evaluation metrics and main results
### Metrics

- **Rule metrics**: F1 score, Match score (recall of key elements in long-form answers); JEC-QA uses choice accuracy.
- **Similarity metrics**: BERT score, BLEU, ROUGE, text2vec cosine similarity.
- **Intelligent metrics**: semantic judgment score by GPT-4o (LLM score).

### Main results (Table 1)

| Model / Method | LooGLE F1 | ASQA Match | JEC-QA Total | BioASQ F1 |
|---|---|---|---|---|
| Vanilla RAG (MiniCPM-2B) | 30.92 | 11.91 | 25.69 | 29.27 |
| **KBAlign (MiniCPM-2B)** | **54.09** | **15.68** | **28.91** | **61.38** |
| Δ | +23.17 | +3.77 | +3.22 | +32.11 |
| Vanilla RAG (LLaMA-3.1-8B) | 40.46 | 20.21 | 23.73 | 27.96 |
| **KBAlign (LLaMA-3.1-8B)** | **62.07** | **25.23** | **23.83** | **70.97** |
| Δ | +21.61 | +5.02 | +0.10 | +43.01 |

Key findings:

- KBAlign brings >20% F1 gain on LooGLE; the self-adapted 2B model even surpasses GPT-4o.
- Gains on BioASQ are especially large (MiniCPM-2B: +32.11 F1, LLaMA-3.1-8B: +43.01 F1).
- KBAlign retains about **90%** of the gain achievable by GPT-4-supervised adaptation while relying entirely on self-annotation, without any external large model.
- Plain language modeling (LM) gives limited gains, confirming the necessity of self-annotation for constructing instruction data.
- Self-verify iterative tuning accelerates convergence; at least 3 iterations are recommended.
- Time cost is much lower than direct LM training (entire KBAlign pipeline < 480 min vs. LM 480 min on A100).
