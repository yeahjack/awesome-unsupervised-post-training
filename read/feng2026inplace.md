# In-Place Test-Time Training

> **Added to survey on:** 2026-03-11

| Attribute | Value |
|---|---|
| Method | In-Place TTT |
| Title | In-Place Test-Time Training |
| Carrier | Direct Opt. |
| Regime | test-time |
| Level | Token |
| Venue | ICLR 2026 |
| Authors | Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai |
| Affiliations | ByteDance Seed; State Key Lab of General AI, Peking University |

| When to Adapt | Within-Sequence Adaptation within the current instance |
|---|---|
| Trigger Unit | within-sequence chunk / token block |
| Persistence | fast weights or inference state persist across chunks, reset at document boundary |
| Inference Coupling | interleaved adapt-and-infer within the same sequence |
| Input Visibility | Online |
| Update Persistence | Non-Cumulative |
| Reset Boundary | Sequence Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Within-Sequence Adaptation |
| Visibility Scope | Current Sequence Prefix Only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Within-Sequence Adaptation`; `Visibility Scope=Current Sequence Prefix Only`.
- **Two-axis coding:** `Input Visibility=Online`; `Update Persistence=Non-Cumulative`; `Reset Boundary=Sequence Boundary`.

- **When the update is triggered:** updates fire continuously chunk by chunk within the same sequence, not between samples.
- **Whose sample it serves:** the current chunk's update directly serves the inference of subsequent chunks; adaptation and inference are interleaved within the same sequence.
- **Whether parameters/state accumulate:** fast weights / inference state persist across chunks but are reset at the document boundary.
- **Reset boundary:** so this is neither Offline Corpus UPT nor test-time-instance adapt-then-reset, but a within-sequence online update.

## 1. UPT Assignment Rationale

This paper belongs to **Family I: Prediction-Statistic Optimization (predictive likelihood minimization)**.

- **Optimization signal entirely internal:** In-Place TTT performs online optimization on the autoregressive input stream at test time, with the language-modeling loss (next-token prediction loss) as the sole supervision signal — no external annotation, verifier, or human feedback.
- **Direct optimization of model parameters:** the framework treats the MLP block's final projection matrix (W_down) as the fast weights, updating it in place via gradient descent at inference time — a direct-optimization carrier.
- **Token-level adaptation:** chunk-by-chunk updates of fast weights, essentially online adaptation at the token level; once a chunk is processed, parameters are updated to serve subsequent tokens.
- **No external signal:** the training target V = Conv1D(X_0) W_target is derived entirely from the input token embeddings — direct use of intrinsic statistics.

---

## 2. Problem Addressed

LLMs follow a "train then deploy" static paradigm with frozen weights at inference, unable to dynamically adapt to new information in the input stream. Test-Time Training (TTT) offers a way to update some parameters (fast weights) at inference time, but faces three obstacles in the LLM ecosystem:

1. **Architecture incompatibility:** existing TTT methods typically introduce brand-new specialized layers that replace attention, requiring training from scratch and preventing direct use of existing large pretrained models.
2. **Low compute efficiency:** classic TTT's per-token update mechanism is inherently serial and cannot exploit GPU/TPU parallelism.
3. **Mismatched objective:** prior TTT generally uses a reconstruction objective (reconstruct the current token), which does not directly align with the LLM's core objective — predicting the next token — possibly storing information in the fast weights that is not useful for prediction.

---

## 3. Method

### 3.1 Overall framework: in-place reuse of the MLP block

Key insight: do not introduce new layers; instead, reuse the ubiquitous final projection matrix W_down of the Transformer's gated MLP block as fast weights. W_up and W_gate stay frozen (slow weights); only W_down is updated online at inference. This is a "drop-in" design that does not change the model architecture and works on any pretrained LLM.

### 3.2 Chunk-wise efficient updates

Split the input sequence into non-overlapping chunks of size C. For each chunk:

1. **Apply:** compute the chunk's output O_[i] = Z_[i] (W_down^(i))^T using the current fast weights.
2. **Update:** with Z_[i] as the key and V_[i] as the value, perform one gradient step to update the fast weights: W_down^(i+1) = W_down^(i) − eta · grad L(...).

This chunk-wise strategy is naturally parallel and supports chunk sizes 512–1024, running efficiently on GPUs/TPUs.

### 3.3 LM-Aligned objective

Unlike traditional TTT, which uses a reconstruction target (V_t = E_{x_t}, reconstruct the current token embedding), this paper proposes an LM-Aligned target:

- Target V = Conv1D(X_0) W_target, where the Conv1D introduces future-token information (it can be set to see only the next token) and W_target is a trainable projection matrix.
- Loss: negative cosine similarity, L(a, b) = -<a, b>_F.
- Theoretical guarantee (Theorem 1): the LM-Aligned target increases the logit of the correct next token (expected gain ≥ lambda_lr · c_norm² · c_align) while leaving other token logits almost unchanged; in contrast, the reconstruction target produces negligible changes to the correct-token logit.

### 3.4 Context parallelism

The update rule is associative, supporting a context-parallel implementation: (i) compute each chunk's intermediate activations and delta-W in parallel; (ii) aggregate updates via a prefix sum; (iii) compute each chunk's output in parallel. In practice, causal padding preserves causality and fast weights reset at document boundaries.

---

## 4. Datasets

### Drop-in enhancement experiments (Section 4.1)
- **Training data:** continual-training curriculum of ~20B tokens (32k context) + ~15B tokens (128k context).
- **Evaluation:** RULER benchmark (4k–256k context lengths).

### Pre-training-from-scratch experiments (Section 4.2)
- **Training data:** 500M and 1.5B models trained on 32k-context sequences; 4B model trained on 120B tokens at 8k context.
- **Evaluation:**
  - Sliding-Window Perplexity: Pile validation set, Proof-Pile-2.
  - Commonsense reasoning: HellaSwag, ARC-E, ARC-C, MMLU, PIQA.
  - Long context: RULER (4k, 8k, 16k).

---

## 5. Evaluation metrics and main results

### Metrics
- **RULER score:** long-context aggregate, reported as average accuracy (%).
- **Sliding-Window Perplexity:** perplexity computed on a fixed final block, measuring long-context utilization.
- **Commonsense accuracy:** HellaSwag, ARC-E, ARC-C, MMLU, PIQA.

### Main results

**Drop-in enhancement (Qwen3-4B-Base, RULER)**

| Context length | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|
| Baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| **In-Place TTT** | **96.1** | **95.6** | **92.7** | **89.3** | **78.7** | **77.0** | **43.9** |

- Substantial improvements at long context (64k, 128k): +4.4, +2.2; gains also at 256k (extrapolation).
- Equally effective on LLaMA-3.1-8B and Qwen3-14B-Base (64k: +2.1, +2.7).

**Pre-training from scratch (Sliding-Window Perplexity)**
- On 500M and 1.5B models, In-Place TTT achieves the lowest perplexity at every context length (2k–32k), consistently outperforming SWA, GLA, DeltaNet, LaCT, and other competitors.

**4B model commonsense and long context**

| Architecture | HellaSwag | ARC-E | ARC-C | MMLU | PIQA | RULER-4k | RULER-8k | RULER-16k |
|---|---|---|---|---|---|---|---|---|
| Full Attn. (Baseline) | 55.67 | 64.52 | 33.19 | 36.43 | 72.63 | 45.77 | 38.09 | 6.58 |
| Full Attn. + I.P. TTT | **55.85** | **64.98** | **32.34** | **37.42** | **73.29** | **49.98** | **43.82** | **19.99** |
| SWA (Baseline) | 54.92 | 64.18 | 32.85 | 36.06 | 72.58 | 14.77 | 9.91 | 5.07 |
| SWA + I.P. TTT | 55.24 | 64.60 | **33.70** | 36.48 | 72.03 | **28.33** | **26.80** | **7.57** |

- Long-context evaluation gains are dramatic (RULER-16k: 6.58 → 19.99); commonsense reasoning improves slightly as well.

### Key ablation findings
- **State size:** enabling more TTT layers (larger fast weights) yields consistent improvements.
- **Chunk size:** C=512 and C=1024 are optimal; too small or too large is suboptimal.
- **LM-Aligned objective:** both the Conv1D and the W_target projection are indispensable; Conv1D is especially crucial for long context, W_target for short context.
- **Efficiency:** the extra throughput and memory overhead introduced by In-Place TTT is negligible.
