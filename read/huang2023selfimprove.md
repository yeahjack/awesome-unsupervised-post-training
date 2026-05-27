# Large Language Models Can Self-Improve (LMSI)

> **Added to survey on:** 2026-03-11

> Huang et al., EMNLP 2023

| Property | Value |
|---|---|
| Method | Self-Improve |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |

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

**Family III — Self-Generated Target Bootstrapping (rationale/critique/latent-thought self-training)**

LMSI is fully free of external ground-truth labels. Its core pipeline uses the LLM's own Chain-of-Thought (CoT) reasoning ability: it samples multiple reasoning paths via multiple-path decoding, then selects high-confidence answers and their reasoning paths via self-consistency (majority voting), and fine-tunes the model on these model-generated reasoning–answer pairs as new learning targets (synthetic targets). The whole process uses no external annotation, external verifier, or human feedback; signals come entirely from model-generated content and intrinsic statistics (the self-consistency voting confidence).

---

## 2. Problem Addressed

- LLM fine-tuning typically requires large amounts of high-quality annotated data and is severely limited in low-resource scenarios or domains with no labels.
- Existing methods (e.g., FLAN, InstructGPT, Minerva) all require large supervised datasets.
- This paper explores: can an LLM be self-trained using only an unlabeled question set, leveraging its own reasoning ability, and thereby improve reasoning performance without any ground-truth labels?

---

## 3. Method

LMSI includes the following key steps:

### 3.1 Generating and filtering multiple reasoning paths
- Given a question-only training set $D^{\text{train}} = \{x_i\}$, use few-shot CoT prompting to sample $m=32$ reasoning paths per question (sampling temperature $T=0.7$).
- For each question, use majority voting (self-consistency) to choose the most consistent answer $\tilde{y}_i$, and keep all reasoning paths that lead to that answer as self-training data.
- High-confidence answers are more likely correct; low-confidence wrong answers have few supporting paths and contribute limited training noise.

### 3.2 Mixed-format training data augmentation
To prevent overfitting to a single prompting format, each reasoning path is augmented into four formats:
1. **Format 1**: CoT prompting examples + Question → CoT reasoning path
2. **Format 2**: Standard prompting examples + Question → Direct answer
3. **Format 3**: Question + "Let's think step by step" → CoT reasoning (zero-shot)
4. **Format 4**: Question → Direct answer (zero-shot)

### 3.3 Self-generated questions and prompts (low-resource extension)
- **Question generation**: when training questions are limited, randomly concatenate existing questions as a prompt to let the LLM generate new ones, then filter high-confidence questions via self-consistency.
- **Prompt generation**: when no human CoT exemplars are available, use "Let's think step by step" (Kojima et al., 2022) to let the model self-generate CoT examples as few-shot exemplars.

### 3.4 Fine-tuning
- Fine-tune the pre-trained LLM on self-generated reasoning–answer pairs.
- Train 10k steps, learning rate $5\text{e-}5$, batch size 32.
- After fine-tuning, sample at inference with $T=1.2$ (higher than the training $T=0.7$), because the post–self-improvement output distribution has lower entropy.

---

## 4. Datasets

| Task type | Dataset | Description |
|---|---|---|
| Arithmetic reasoning | **GSM8K** | Grade-school math word problems |
| Arithmetic reasoning | **DROP** | Reading comprehension (numerical reasoning), split into football / non-football subsets |
| Commonsense reasoning | **OpenBookQA** | Multiple choice |
| Commonsense reasoning | **ARC-c** (AI2 Reasoning Challenge) | Multiple choice (Challenge subset) |
| NLI | **ANLI-A2, ANLI-A3** | Adversarial natural language inference |
| OOD evaluation | **AQUA, SVAMP, StrategyQA, ANLI-A1, RTE, MNLI-M/MM** | Generalization tests |

Note: training uses questions only (no ground-truth labels).

---

## 5. Evaluation metrics and main results
**Metrics**: Accuracy (all tasks report accuracy), evaluated with three inference modes: Standard Prompting, CoT-Prompting, Self-Consistency.

### In-domain results (Table 3, PaLM 540B)

| Dataset | w/o LMSI (Self-Consistency) | w/ LMSI (Self-Consistency) | Gain |
|---|---|---|---|
| GSM8K | 74.4% | **82.1%** | +7.7 |
| DROP | 78.2% | **83.0%** | +4.8 |
| ARC-c | 88.7% | **89.8%** | +1.1 |
| OpenBookQA | 90.0% | **94.4%** | +4.4 |
| ANLI-A2 | 64.5% | **66.5%** | +2.0 |
| ANLI-A3 | 63.4% | **67.9%** | +4.5 |

### Out-of-Domain generalization (Table 4, after multi-task joint training)
- All 6 OOD tasks (AQUA, SVAMP, StrategyQA, ANLI-A1, RTE, MNLI) improve, showing that LMSI strengthens general reasoning ability rather than merely fitting the training distribution.

### Key ablations
- **Importance of mixed format (Table 5)**: training jointly on the four formats (73.5% CoT on GSM8K) beats CoT-only (69.4%) or direct-answer-only (61.6%).
- **Self-generated questions (Table 6)**: using self-generated questions still improves performance (GSM8K CoT: 66.2%), though not as much as the real training set (73.5%).
- **Zero-shot self-generated prompts (Fig. 3)**: reaches 74.2% on GSM8K (zero-shot SOTA), with no human CoT exemplars.
- **Distillation to smaller models (Table 7)**: training data generated by LMSI 540B is used to fine-tune 8B and 62B models; the distilled 62B (57.4%) surpasses the original 540B pre-trained model (56.5%).
- **Number of sampled paths (Fig. 5)**: $m=15$ already gives good results; after LMSI, just 5 self-consistency paths outperform 32 paths without LMSI.
