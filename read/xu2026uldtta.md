# ULDTTA — Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs

> **Added to survey on:** 2026-03-11

**Method:** ULDTTA | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token
**Authors:** Longhuan Xu, Cunjian Chen, Feng Yin
**Preprint:** February 11, 2026 | **arXiv:** 2602.09719

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

ULDTTA belongs to **Family I — Prediction-Statistic Optimization (local state / adapter shaping)**.

- **No external supervision signal:** at test time the method uses only the prompt's own negative log-likelihood (i.e. next-token-prediction loss on the prompt tokens) as the objective, with no reliance on gold answers, external verifiers, human annotation, or external AI labels.
- **Driven by intrinsic statistics:** the optimization signal comes entirely from the model's internal statistics on the prompt (prompt perplexity / conditional log-probability) — a legitimate intrinsic-statistic signal.
- **Layer-wise adapter shaping:** a lightweight hypernetwork (ScaleNet) dynamically predicts per-layer, per-step LoRA learning-rate multipliers based on the prompt's internal representations (mean-pooled first-layer and last-layer hidden states), providing fine-grained shaping of the Q/V-projection adapters at every transformer layer — a local adapter-modulation mechanism based on the model's own internal state.
- **Sample-specific, adapt-and-reset protocol:** each prompt undergoes a few gradient steps of LoRA adaptation and is then reset — no external data or labels are needed.

---

## 2. Problem Addressed

LLMs at deployment face two main problems — **distribution shift** and **instance-level objective mismatch**:

1. **Deployment distribution shift:** real-world prompts differ substantially in style, length, and domain vocabulary from the pretraining/fine-tuning stage, degrading performance.
2. **Global parameters are suboptimal for a single sample:** the global parameters from standard training are an average compromise across all samples and may not be optimal for any specific prompt — there is room for per-instance specialization.
3. **Naive TTA is unstable:** existing unsupervised sample-specific TTA (fixed learning rate, a few prompt gradient steps) is extremely fragile — too small a learning rate is ineffective; too large causes catastrophic parameter drift; gradient magnitudes and update sensitivities vary drastically across layers, so a single global learning rate cannot balance them.

ULDTTA aims to make unsupervised TTA both stable and effective within a few gradient steps via **learned, layer-wise, step-wise dynamic learning-rate control**.

---

## 3. Method

### 3.1 Overall framework

ULDTTA follows an **adapt-and-reset** protocol: for each test prompt x, initialize fresh LoRA parameters (B=0, A random), run K gradient updates on the prompt's negative log-likelihood (K_max=5), then generate the answer y with the adapted model, and discard the LoRA state.

The core innovation is replacing a fixed learning rate with a lightweight **hypernetwork ScaleNet** that predicts, per layer (separately for Q and V projections) and per step, a **non-negative learning-rate multiplier** applied multiplicatively to a base learning rate.

### 3.2 ScaleNet architecture

- **Input:** a fixed-length feature h(x) extracted from the prompt's forward pass — the concatenation of mean-pooled first-layer and last-layer hidden-state sequences, of dimension 2d. Also takes embeddings of the current TTA step k and the total number of steps K.
- **Architecture:** two-layer MLP (hidden size 128), with a non-negative activation (exp(a) for a≤0, 1+a+0.5a² for a>0) and an optional safety clamp.
- **Output:** one scalar multiplier per layer for each of Q and V, $s_l^{(k)}$ — a total of 2L per-step scalers.

### 3.3 Layer-wise dynamic update rule

For LoRA parameter $\phi_l$ at layer l, the k-th step update is:

> $\phi_l^{(k+1)} = \phi_l^{(k)} - \eta \cdot s_l^{(k)} \cdot \nabla_{\phi_l}(-\log P(x; \Phi^{(k)}))$

where $\eta$ is the base learning rate ($10^{-2}$) and $s_l^{(k)}$ is the multiplier output by ScaleNet.

### 3.4 Training

- ScaleNet is trained by **unrolling** the K-step TTA process: for each (x, y) pair in the training set, first run K steps of unsupervised TTA (using only x's loss), then compute the adapted model's loss on the gold answer y as the supervision loss for updating ScaleNet's parameters $\psi$.
- Use a **first-order approximation**: ignore the second-order dependence of TTA gradients on $\psi$ (treat the prompt gradient as a constant in $\psi$), avoiding expensive Hessians while remaining effective.
- During training, sample K uniformly from {0,1,...,K_max} to support multiple TTA schedules.
- Train with AdamW, learning rate $10^{-4}$, ~30k samples.

---

## 4. Datasets

### Training data
- About **30k training samples** per dataset–model pair (mostly without repeats).

### Evaluation datasets (~300 test samples each)

| Dataset | Type | Description |
|---|---|---|
| **XSum** | Summarization | Single-sentence summaries of BBC news articles. |
| **SQuAD** | Reading Comprehension | Reading comprehension over Wikipedia paragraphs. |
| **NQ-Open** | Open-domain QA | Short-answer prediction on real user queries. |
| **AdaptEval** | Comprehensive TTA benchmark | DomainBench (Geography, Agriculture, Medicine, Finance) + InstructionBench (Alpaca-GPT4, Dolly, InstructionWild) + ReasoningBench (GSM8K, MetaMath, LogiQA). |

### Evaluation models
- **Llama family:** Llama-3.2-3B, Llama-3.2-3B-Instruct, Llama-3.3-70B-Instruct.
- **Qwen family:** Qwen3-4B, Qwen3-4B-Instruct, Qwen3-32B.

---

## 5. Evaluation metrics and main results

### Metrics
- **NLL (Negative Log-Likelihood):** the average NLL per answer token — lower is better. The core metric, directly aligned with the TTA objective.
- **ROUGE-Lsum:** lexical overlap between generated text and reference answer — higher is better; reflects actual generation quality.

### Main results

**NLL results (medium-scale models, 4 datasets):**
- The naive fixed-rate TTA baseline shows small initial improvements but then NLL rises sharply — destructive drift.
- ULDTTA (layer-wise ScaleNet) **consistently reduces NLL** across all dataset–model pairs and stays stable as TTA steps grow.
- Layer-wise control beats step-wise-only ablation, confirming that different transformer layers genuinely need different update magnitudes.

**Large-model AdaptEval NLL (Table 1):**
- Llama-3.3-70B-Instruct: 5-step layer-wise TTA reaches NLL 1.7048, substantially better than No TTA (2.2114) and the fixed baseline (11.4970).
- Qwen3-32B: 5-step layer-wise TTA reaches NLL 1.8889, beating No TTA (2.1805) and the fixed baseline (2.0925).
- Gains on large models are even larger than on medium models, indicating good **scaling properties**.

**ROUGE-Lsum results (Table 2):**
- Clear, stable improvements on XSum and SQuAD (e.g. Qwen4B on XSum: No TTA 0.1700 → 5-step layer-wise 0.2247).
- Smaller and less stable improvements on NQ-Open and AdaptEval — these are more open-ended, require deeper reasoning, and prompt-only adaptation cannot fully optimize them.

**ScaleNet visualization analysis:**
- The learned learning-rate multipliers display rich structure across layers and between Q/V; adjacent layers or Q vs. V at the same layer can differ by orders of magnitude.
- Update magnitudes are largest at the first step and decay rapidly, indicating most of the benefit comes from the initial update.
- Across different schedule lengths, the scale at the same step is consistent — good schedule consistency.
