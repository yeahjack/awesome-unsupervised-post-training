# SECL: Self-Calibrating Language Models via Test-Time Discriminative Distillation

> **Added to Survey:** 2026-04-16

**Paper:** Self-Calibrating Language Models via Test-Time Discriminative Distillation
**Authors:** Jan Strich et al.
**arXiv:** 2604.09624
**Venue:** arXiv 2026
**Code:** anonymous 4open.science link in the arXiv abstract

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| SECL | Direct Opt. / LoRA | test-time | Sequence / calibration |

| When to Adapt | Streaming Continual Adaptation (calibration) with shift-triggered LoRA bursts |
|---|---|
| Trigger Unit | input question stream; update bursts only when the entropy-based shift detector fires |
| Persistence | LoRA confidence updates accumulate across the shifted stream; per-question reset hurts |
| Inference Coupling | infer normally until shift; then distill internal discriminative confidence into the model |
| Input Visibility | Online |
| Update Persistence | Cumulative |
| Reset Boundary | Domain / stream boundary |
| Evidence Status | paper-explicit |
| Timing Regime | Streaming Continual Adaptation |
| Visibility Scope | Streaming prefix only |

---

## When-to-Adapt audit
- **Auxiliary taxonomy:** `Timing Regime=Streaming Continual Adaptation`; `Visibility Scope=Streaming prefix only`.
- **Two-axis encoding:** `Input Visibility=Online`; `Update Persistence=Cumulative`; `Reset Boundary=Domain / stream boundary`.
- **When does the update fire:** the paper uses an entropy-based change detector on the input stream; only when distribution shift is detected does the system enter a calibration burst, instead of updating per question.
- **Serving the current sample or downstream samples:** the current burst's update serves later same-domain inputs; the ablation shows per-question reset clearly worsens ECE / AUROC, so accumulated updates are part of the design.
- **Whether parameters / state accumulate:** LoRA updates accumulate within the stream; training only happens on 6%–26% of the question stream.
- **Reset boundary:** the right boundary is the domain / stream boundary, not the sample boundary.

## 1. UPT Assignment Rationale
SECL is a strict UPT candidate. It meets three criteria:

- **Real update:** at test time it uses LoRA to update confidence / calibration representations.
- **No external labels:** the arXiv abstract explicitly states the test-time training pipeline uses label-free self-supervision and requires no labeled data or human supervision.
- **Internal signal source:** the supervision comes from the model's probability of the `True` token when asked "Is this answer correct?", i.e., `P(True)` / `NormPTrue`; this internal discriminative confidence is then distilled back into the model's verbalized confidence.

A finer-grained family placement is **Family IV: internal evaluator bootstrapping**, because the core signal is not raw NLL / entropy but the model's internal correctness-evaluator channel. It also intersects Family I, because the optimization target is the calibration / confidence representation rather than producing external pseudo-labels.

## 2. Problem Addressed

LLMs often have over-confident verbalized confidence—they keep emitting high confidence on questions they are likely to get wrong. Conventional calibration usually needs a labeled validation set, fails under distribution shift, or has high inference cost. SECL's setting: at test time, when the input stream shifts, use the model's internal signal to do label-free calibration updates.

## 3. Method

SECL has three key components:

- **Shift detector:** an entropy-based change detector judges whether the input distribution has shifted.
- **Internal discriminative signal:** ask the model to judge whether its answer is correct and read `P(True)`, then normalize.
- **LoRA calibration burst:** when verbalized confidence disagrees with `NormPTrue`, run a lightweight LoRA update with a directional loss to reduce the gap between verbalized confidence and internal discriminative confidence.

The paper emphasizes that burst-level updates are more stable than per-question updates; per-question updates damage the confidence representation.

## 4. Datasets

Experiments cover four domains / datasets, including GSM8K-style math QA and question streams from other domains. Models span three families and four small language models.

## 5. Evaluation metrics and main results
The main metrics are Expected Calibration Error (ECE), Brier score, AUROC, and task accuracy. The abstract reports a 56%–78% ECE reduction by SECL, with lower cost than its distillation-source baseline; the paper text also reports up to ≈71% ECE reduction on Llama 3.2-3B.

## 6. Position in the UPT Survey
Recommended for inclusion in the survey's strict UPT table, or at minimum as a calibration-oriented test-time-adaptation representative of Family IV. It fills a direction the current taxonomy under-covers: rather than improving answer accuracy, it uses the model's internal self-evaluation signal for calibration post-training.
