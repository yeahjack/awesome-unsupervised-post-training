# Effective Long-Context Scaling of Foundation Models

> **Added to survey on:** 2026-03-11

**Paper:** Wenhan Xiong, Jingyu Liu, Igor Molybog et al. (GenAI, Meta), arXiv:2309.16039, 2023

| Attribute | Value |
|---|---|
| Method | LongContext Scaling |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Token |
| Family | I — Prediction-Statistic Optimization |

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

This paper belongs to **Family I (Prediction-Statistic Optimization, predictive likelihood minimization)**. The core training signal is token-level perplexity: starting from a LLaMA 2 checkpoint, continual pretraining is performed on long-context corpora, directly minimizing the standard language-modeling loss (next-token-prediction NLL). The whole pretraining stage uses no external annotation, no human preference, and no external verifier; the objective is entirely the model's own intrinsic predictive likelihood over the observed text. The instruction-tuning stage likewise uses no human-annotated long data; instead, the model itself (LLaMA 2 Chat) is used to self-instruct long QA data and validate them by self-critique — the signal still comes from inside the model. The power-law relation between validation loss and context length, $L(c) = (\alpha/c)^\beta + \gamma$, further confirms that the core supervision signal is token-level perplexity.

---

## 2. Problem Addressed

- At the time, open-source long-context LLMs were evaluated mainly on perplexity and synthetic tasks, and could not maintain strong performance on both short- and long-context downstream tasks simultaneously.
- LLaMA 2's original RoPE positional encoding imposes too-strong attention decay on distant tokens, so the model fails to use context beyond 4,000–6,000 tokens.
- Collecting human-annotated long-context instruction data is prohibitively expensive, and existing open-source instruction datasets are mostly short — blocking alignment for long-context chat models.

---

## 3. Method

### 3.1 Continual Pretraining

Starting from a LLaMA 2 checkpoint, perform continual pretraining with longer training sequences over an extra 400B tokens (100,000 steps). Key design choices:

1. **Positional-encoding modification (RoPE ABF):** raise the RoPE base frequency $b$ from 10,000 to 500,000, lowering the rotation rate and reducing attention-score decay on distant tokens. This outperforms Position Interpolation (PI) and xPos ABF.
2. **Data mix:** augment LLaMA 2's original pretraining data with new long-text data and upsample the long-text proportion. Experiments show data **quality** matters more than the length distribution — even after removing most long text, the model still retains most of the performance gain.
3. **Sequence length:** 32,768-token sequences for 7B/13B; 16,384-token sequences for 34B/70B.
4. **Optimization details:** FlashAttention; learning rate $2\times10^{-5}$ for 7B/13B with cosine schedule and 2,000 warm-up steps; $1\times10^{-5}$ for 34B/70B.

### 3.2 Instruction tuning (no human-annotated long data)

1. Build on the LLaMA 2 Chat RLHF dataset (short text).
2. Use LLaMA 2 Chat itself to **self-instruct** long-context QA data: randomly select long passages from the pretraining corpus, prompt the model to generate a question–answer pair, then have the same model **self-critique** the answer quality.
3. Pack short instruction data into 16,384-token sequences; handle long instruction data separately (right padding).
4. Key trick: compute the loss not only on the output tokens but also as **LM loss on the long input prompt**, stabilizing learning under long/short imbalance and substantially improving downstream performance.

### 3.3 Training curriculum

Experiments show that continual pretraining (first 4k, then switch to 32k) saves about 40% of FLOPs with almost no performance loss; the model adapts within a few thousand steps after the length switch.

---

## 4. Datasets

### Pretraining data
- LLaMA 2's original pretraining data (mostly short text).
- Additional long-text data (specific sources not disclosed; includes Books, CommonCrawl, Wikipedia, etc.).
- Total: an extra 400B tokens.

### Evaluation data

| Task type | Dataset | Setup |
|---|---|---|
| Short-context general | MMLU, GSM8K, HumanEval, MATH, NaturalQuestions, TriviaQA, PIQA, SIQA, HellaSwag, WinoGrande, ARC, OpenBookQA, CommonsenseQA | Few-shot |
| Long-context QA / summarization | NarrativeQA (0-shot), Qasper (2-shot), QuALITY (2-shot), QMSum (1-shot) | Prompts truncated to 16,384 or 32,768 tokens |
| Long-context aggregate | ZeroSCROLLS (10 sub-tasks: GR, SS, QM, SQAL, Qspr, Nrtv, QALT, MuSQ, SpDg, BkSS) | Zero-shot |
| Long-context aggregate | L-Eval (6 long-text tasks) | — |
| Human evaluation | Multi-turn conversation, multi-document search QA (2,352 examples) | 3 annotators |
| Safety | TruthfulQA, ToxiGen, BOLD | Few-shot |

---

## 5. Evaluation metrics and main results

### Metrics
- **Language modeling:** validation perplexity (Books, CommonCrawl, Wikipedia).
- **Short-context:** Pass@1 (HumanEval), Top-1 accuracy (GSM8K, MMLU, etc.), 5-shot accuracy (NaturalQuestions, TriviaQA).
- **Long-context:** F1 (NarrativeQA, Qasper), EM (QuALITY), ROUGE-geo (QMSum), per-sub-task metrics for ZeroSCROLLS.
- **Human evaluation:** Win/Tie/Loss rate.
- **Safety:** TruthfulQA (truthful+informative %), ToxiGen (toxic %), BOLD (sentiment score).

### Main results

**Short-context tasks (Table 1):** LLaMA 2 Long matches or exceeds LLaMA 2 on virtually all short-context benchmarks; the 70B model gains substantially on Coding (39.9 vs 37.4), Math (41.3 vs 35.2), and MMLU (71.7 vs 68.9).

**Long-context tasks (Table 3):** LLaMA 2 Long 70B substantially leads open-source models on all long-context tasks (NarrativeQA 30.9, Qasper 35.7, QuALITY 79.7, QMSum 16.5), and performance grows monotonically with context length.

**ZeroSCROLLS (Table 4):** LLaMA 2 Long Chat 70B averages 37.7 — surpassing gpt-3.5-turbo-16k (36.7) on 7 of 10 sub-tasks — without any human-annotated long-context data.

**Human evaluation (Figure 3):**
- vs. MPT-30B-chat: 53.3% win.
- vs. GPT-3.5-turbo-16k: 35.8% win vs 32.8% loss.
- vs. Claude-2-100k: 38.9% win vs 31.7% loss.
- vs. GPT-4: 25.0% win vs 45.0% loss.

**Scaling law:** validation loss and context length follow a power-law-plus-constant relation; larger models (larger $\beta$) use long context more effectively.

**Key ablation findings:**
- RoPE ABF is the best positional-encoding choice — the only variant that retains performance on the 32,768-token FIRST-SENTENCE-RETRIEVAL task.
- Data quality > length distribution: even after removing most long text, the long-context gain is largely preserved; merely upsampling existing long text yields no clear extra benefit.
- Continual pretraining that switches from 4k to 32k (at 20–40% training progress) saves about 40% FLOPs with virtually no performance loss.
- Adding LM loss on input prompts and the self-instruct data are the two most important factors in the instruction-tuning stage.
