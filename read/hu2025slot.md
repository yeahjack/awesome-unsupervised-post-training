# SLOT: Sample-specific Language Model Optimization at Test-time

> **Added to survey on:** 2026-03-11

**Method:** SLOT | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token
**Paper:** arXiv 2505.12392 (2025)
**Authors:** Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi
**Affiliations:** Westlake University, University of Washington, USTC

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

SLOT belongs to **Family I — Prediction-Statistic Optimization** (local-state / adapter shaping based on intrinsic LM statistics).

Core justification:
- **Optimization signal entirely from intrinsic statistics:** SLOT uses only the negative log-likelihood (cross-entropy loss) of the input prompt itself as the objective; it relies on no external labels, no external verifier, and no external AI feedback.
- **Update target is local state, not a pseudo-label:** SLOT optimizes a sample-specific parameter vector δ ∈ R^{1×d} added to the last-layer hidden features; each sample is optimized independently, and δ is discarded after inference, producing no new training data or pseudo-labels.
- **Direct token-level optimization:** gradient descent directly minimizes the token-level NLL on the prompt — a direct-optimization paradigm.

---

## 2. Problem Addressed

After being trained on general corpora, LLMs often fail to fully understand and follow complex, atypical prompts (e.g. special-format instructions, complex reasoning requirements), leading to degraded generations. Existing Test-Time Adaptation (TTA) methods face the following problems:

1. **High compute cost:** per-sample full-parameter updates of large LLMs are prohibitively expensive.
2. **Hard to design self-supervised signals:** for pure language tasks, it is difficult to design an effective self-supervised target per sample.
3. **Catastrophic forgetting and error accumulation:** TTA methods on LLMs are prone to both.

SLOT aims to perform a lightweight per-prompt adaptation at test time with minimal compute, so the model better understands and responds to that specific instruction.

---

## 3. Method

SLOT's core idea: treat the input prompt as a "special training sample" and perform a few optimization steps at inference time so the model better "learns" the prompt's content.

### The pipeline has two stages:

**Prompt Stage (optimization):**
1. Introduce a sample-specific parameter vector δ ∈ R^{1×d}, initialized to zero.
2. Broadcast-add δ to the last-layer hidden features H: H' = H + δ.
3. Compute logits with the modified H': logits = W_LM · H'.
4. With the cross-entropy loss on the prompt (i.e. NLL) as the objective, use AdamW to gradient-descent on δ for T steps (typically T=3).
5. Key efficiency design: since δ acts only on the last layer, all preceding layers' hidden features can be cached; only the last linear head needs forward and backward — overhead is minimal.

**Generation Stage:**
1. With the optimized δ_opt, add δ_opt to the last-layer hidden feature of each newly generated token during autoregressive decoding.
2. No further optimization is done; the δ obtained in the prompt stage is reused directly.

### Logit Modulation Vector (LMV) analysis
The optimized δ is equivalent to a fixed additive offset W_LM · δ on the logits. Experiments show that the LMV substantially boosts logits of reasoning-related tokens (e.g. "think", "reasoning") while suppressing number tokens, function words (e.g. "should", "will"), and end-of-text tokens, encouraging the model to engage in deeper reasoning.

### Hyperparameters
- Optimization steps T = 3.
- Learning rate η = 0.01.
- AdamW weight decay = 1×10⁻⁸, epsilon = 1×10⁻⁵.
- δ initialized to zero.

---

## 4. Datasets

Evaluation covers diverse task types:

| Dataset | Type | Description |
|--------|------|------|
| **AIME24** | Math competition | American Invitational Mathematics Examination 2024. |
| **Math500** | Math | MATH subset, covering K-12 to competition level. |
| **GPQA Diamond** | Graduate-level reasoning | Google-Proof Q&A — deep domain knowledge and complex reasoning (STEM). |
| **GSM8K** | Math | Grade-school math word problems, multi-step reasoning. |
| **HumanEval** | Code generation | Python code generation from docstrings. |
| **C-Eval** | Comprehensive (Chinese) | Chinese multi-discipline benchmark (STEM, social sciences, humanities, others, hard subset). |

Models tested include the Qwen family (Qwen-7B, Qwen2.5-Math/14B/32B), the Llama family (Llama-3.1-8B/70B-Instruct), and the DeepSeek-R1-Distill family (Qwen-1.5B/7B/14B/32B, Llama-8B/70B).

---

## 5. Evaluation metrics and main results

**Metric:** answer accuracy.

### Main results (T=3 optimization steps)

**Qwen-7B (general model):**

| Benchmark | Baseline | +SLOT | Gain |
|-----------|----------|-------|------|
| C-Eval (STEM) | 52.79 | 56.98 | +4.19 |
| C-Eval (Hard) | 36.22 | 44.77 | +8.55 |
| C-Eval (AVG) | 62.64 | 63.69 | +1.05 |
| GSM8K | 51.2 | 54.2 | +3.0 |
| HumanEval | 29.9 | 31.7 | +1.8 |

**Qwen2.5 series (math/reasoning models, AIME24 / Math500 / GPQA Diamond):**
- Qwen2.5-Math-1.5B: AIME24 6.67→10.00 (+3.33).
- Qwen2.5-Math-7B: AIME24 13.33→20.00 (+6.67), Math500 57.60→58.80 (+1.20), GPQA 25.76→32.83 (+7.07).
- Qwen2.5-32B: AIME24 13.33→+10.00, GPQA 36.36→42.93 (+6.57).

**DeepSeek-R1-Distill series (reasoning models):**
- DeepSeek-R1-Distill-Qwen-7B: AIME24 50.00→56.67 (+6.67), Math500 93.40→93.80 (+0.40).
- DeepSeek-R1-Distill-Qwen-32B: AIME24 70.00→80.00 (+10.00).
- **DeepSeek-R1-Distill-Llama-70B: AIME24 63.33→73.33 (+10.00), GPQA Diamond 65.66→68.69 (+3.03)** — the latter achieves SOTA among 70B open-source models.

**Inference-time overhead (GSM8K, Qwen2.5-7B, V100):**
- Baseline: 161.49s → T=3: 167.07s (only ~3.5% more).
- T=5: 174.32s (~7.9% more).

### Ablation findings (DeepSeek-R1-Distill-Qwen-1.5B on AIME24)
- SLOT is insensitive to hyperparameters; most configurations beat the baseline (26.67%).
- The best settings (T=4, η=0.05) and (T=5, η=0.05) reach 40.00%, a 13.33-point gain over baseline.
