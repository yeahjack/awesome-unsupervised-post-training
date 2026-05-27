# TTRV: Test-Time Reinforcement Learning for Vision Language Models

> **Added to survey on:** 2026-03-11

**Paper:** arXiv 2510.06783
**Authors:** Akshit Singh, Shyam Marjit, Wei Lin, Paul Gavrikov, Serena Yeung-Levy, Hilde Kuehne, Rogerio Feris, Sivan Doveh, James Glass, M. Jehanzeb Mirza
**Affiliations:** Independent, IISc Bangalore, JKU Linz, Stanford, Tübingen AI Center, MIT-IBM Watson AI Lab, MIT CSAIL

| When to Adapt | multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation |
|---|---|
| Trigger Unit | online primitive: encountered test sample; main evaluation: sampled target subset |
| Persistence | GRPO-updated weights accumulate within the adaptation run; no per-sample reset |
| Inference Coupling | online primitive: update during inference; main evaluation: adapt on sampled subset, then evaluate full target dataset |
| Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Cumulative |
| Reset Boundary | Multi-protocol: Evaluation Boundary + No Immediate Reset |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation |
| Visibility Scope | Multi-protocol: Few-sample target subset + Streaming prefix only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** this paper contains multiple protocol entries: `Timing Regime=Multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation`; `Visibility Scope=Multi-protocol: Few-sample target subset + Streaming prefix only`.
- **Two-axis coding:** `Input Visibility=Multi-protocol: Offline + Online`; `Update Persistence=Cumulative`; `Reset Boundary=Multi-protocol: Evaluation Boundary + No Immediate Reset`.

| Protocol Entry | Timing Regime | Visibility Scope | Input Visibility | Update Persistence | Reset Boundary | Note |
|---|---|---|---|---|---|---|
| TTRV / online GRPO primitive | Streaming Continual Adaptation | Streaming prefix only | Online | Cumulative | No Immediate Reset | Immediately do a GRPO update on the encountered test sample; weights carry forward into subsequent samples. |
| TTRV / 20-sample target-subset setting | Few-Sample Target Adaptation | Few-sample target subset | Offline | Cumulative | Evaluation Boundary | First sample 20 target samples for adaptation; then evaluate the full target dataset. |
| TTRV / 1-sample ablation | Few-Sample Target Adaptation | Few-sample target subset | Offline | Cumulative | Evaluation Boundary | Same as the main setting but the adaptation pool is reduced to a single sample. |

- **When the update is triggered:** the method is defined as "encounter a test sample, sample multiple times, construct a reward, do a GRPO update"; main experiments typically randomly sample 20 test samples for adaptation and then evaluate on the full dataset; an ablation also reports 1-sample adaptation.
- **Whose sample it serves:** the online primitive serves both the current and subsequent samples; updates in the 20-sample and 1-sample experimental settings mainly serve the subsequent evaluation on the full target dataset.
- **Whether parameters/state accumulate:** GRPO-updated weights are not reset after each sample; they accumulate within the sampled adaptation set.
- **Reset boundary:** a single paper-level classification therefore loses information: the framework primitive is closer to streaming cumulative, while the main-experiment presentation is closer to sampled-subset target-set calibration.

## 1. UPT Assignment Rationale

**Family II — Sample-Relation Supervision (population consensus)**

TTRV extends TTRL (Test-Time Reinforcement Learning) from LLMs to VLMs (Vision-Language Models). Its core supervision signal comes entirely from the model's own multi-sample outputs, with no reliance on external labels, verifiers, or human annotation:

- **Frequency-based reward (r₁):** sample N times on the same test sample, build the empirical frequency distribution p(ỹ_m), and set each answer's reward to its empirical probability. High-frequency answers receive higher rewards, but low-frequency answers still receive non-zero rewards — a soft, probabilistic population-consensus signal.
- **Diversity-control reward (r₂):** compute the negative Shannon entropy −H(P) = Σ p(ỹ_m) log p(ỹ_m) of the empirical distribution; penalizes overly dispersed outputs and encourages the model to converge to a consistent answer.
- **Combined reward:** R(ŷ_j) = r₁(ŷ_j) + α · r₂, with α controlling the convergence/diversity trade-off.

GRPO is used for test-time policy optimization. All reward signals are combinations of intrinsic statistics (empirical frequencies) and model-generated content (sampled answers), matching the UPT definition.

**Carrier:** Policy Optimization | **Regime:** Test-time | **Level:** Semantic

---

## 2. Problem Addressed

Existing RL-based fine-tuning for VLMs relies on labeled data and fixed train/test splits, and cannot dynamically adapt to new tasks or new domains at inference. Test-time training (TTT) methods do not need labels, but most target dual-encoder VLMs (such as CLIP), are constrained by the architecture (e.g. require class-level probability distributions), and are hard to apply directly to decoder-based VLMs.

The core problem TTRV solves: **how to apply RL optimization to decoder-based VLMs at test time with no labeled data, extracting self-supervised reward signals from unlabeled test data to lift visual-understanding ability.**

---

## 3. Method

### 3.1 GRPO recap

Given a prompt x, the VLM samples n answers {y_i}; each answer's advantage is

A_i = (r(x, y_i) − mean_j(r(x, y_j))) / std_j(r(x, y_j)).

The policy is updated with a clipped importance-weighted objective, plus KL regularization to constrain deviation from the reference policy.

### 3.2 Test-time distributional rewards

**Frequency-based reward:** sample N answers {ŷ₁, ..., ŷ_N} for test sample x; collect the unique-answer set U = {ỹ₁, ..., ỹ_M}; compute empirical probabilities p(ỹ_m) = (1/N) Σ 1{ŷ_j = ỹ_m}. Each answer's reward is

r₁(ŷ_j) = Σ_m p(ỹ_m) · 1{ŷ_j = ỹ_m},

i.e. answers are assigned their empirical frequency as reward. Unlike majority voting (which only takes the most-frequent answer), this is a soft probabilistic supervision that retains uncertainty modeling.

**Diversity-control reward:** compute the Shannon entropy H(P) = −Σ p(ỹ_m) log p(ỹ_m) of the empirical distribution and set r₂ = −H(P). Penalize dispersed outputs, encouraging the model to gradually concentrate on consistent predictions.

**Combined reward:** R(ŷ_j) = r₁(ŷ_j) + α · r₂, with α a hyperparameter.

### 3.3 Optimization

Optimization uses the standard autoregressive language-modeling objective, with the reward as a soft sample-level weighting. Parameters are updated by gradient ascent: θ ← θ + η ∇_θ E[R(y)]; GRPO replaces the raw reward with a relative advantage.

---

## 4. Datasets

### Image recognition (8 datasets)
- ImageNet, ImageNet-V2.
- ImageNet-Rendition (R), ImageNet-Sketch (S), ImageNet-Adversarial (A) (OOD variants).
- Food101, DTD (Describable Textures Dataset).
- Resisc45 (remote-sensing scene classification).

### Visual Question Answering (8 datasets)
- Math reasoning: MathVerse, MathVista.
- Science / everyday scenes: SEED, MME, RealWorldQA.
- Compositional / spatial reasoning: Capture, CRPE.
- Chart understanding: AI2D.

A total of **16 datasets**, covering both image classification and VQA.

---

## 5. Evaluation metrics and main results

### Metrics
- **Image Classification:** Top-1 Accuracy (%).
- **VQA:** Accuracy (%).

### Main results

**Image Classification (Table 1):**
- InternVL3-2B + TTRV: average 94.99% (base 62.03%, +32.95%); DTD +52.49%; ImageNet-R +30.88%.
- InternVL2.5-4B + TTRV: average 82.34% (base 70.47%, +11.88%).
- InternVL3-8B + TTRV: average 95.71% (base 66.74%, +28.97%); ImageNet 99.31% — **beats GPT-4o (98.30%)**.
- TTRV applied to InternVL-8B beats GPT-4o on image recognition by 2.3% on average.

**VQA (Table 2):**
- InternVL3-2B + TTRV: average 57.15% (base 47.47%, +9.69%); AI2D +28.07%.
- InternVL2.5-4B + TTRV: average 69.40% (base 66.37%, +3.03%).
- InternVL3-8B + TTRV: average 55.56% (base 38.05%, +17.50%).

**Key findings:**
- Substantial gains can be obtained with as few as 20 randomly sampled test samples per dataset.
- Single-sample TTRV (Table 6) still yields meaningful improvements (e.g. ImageNet-R +5.47%).
- Cross-dataset generalization (Table 3 / Figure 3): TTRV training on one dataset still substantially improves performance on completely different datasets (e.g. Food→DTD +52.03%, Resisc→DTD +52.33%).
- Ablation shows the frequency + diversity reward combination beats using either alone and also beats the majority-voting baseline.
- Generalizes to other model families: equally effective on Qwen2.5-VL-3B.
- Random rewards fail to produce comparable gains, indicating TTRV's reward design carries meaningful signal.
