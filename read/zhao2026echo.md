# ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning

> **Added to survey on:** 2026-03-11

> arXiv 2602.02150, Feb 2026
> Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo

| Attribute | Value |
|---|---|
| Method | ECHO |
| Carrier | Policy Optimization (GRPO-style) |
| Regime | Test-time |
| Level | Token |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | benchmark cohort / rollout group |
| Persistence | full parameter accumulate across test-time episodes; no per-sample reset |
| Inference Coupling | adapt within the cohort, then infer/evaluate with the updated model |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | protocol-inferred |
| Timing Regime (auxiliary taxonomy) | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** updates fire iteratively per rollout group / mini-batch on the target benchmark or its unlabeled cohort — not sample-local adapt-and-reset.
- **Whose sample it serves:** updates produced by the current sample or mini-batch mainly serve subsequent rounds within the same cohort and the final evaluation, not the current sample alone.
- **Whether parameters/state accumulate:** parameters accumulate throughout the test-time RL run; typically re-initialized only when switching to a new benchmark, model run, or independent experiment.
- **Reset boundary:** so it is closer to a cohort-level cumulative TTA than an arrival-by-arrival streaming-reset protocol.

## 1. UPT Assignment Rationale

ECHO belongs to **Family II: Sample-Relation Supervision (population consensus)**.

> **Reclassification note (2026-05-19):** previously classified under Family I (then Prediction-Statistic Optimization). After closer review of the reward construction, the dominant-artifact rule reassigns it to Family II: ECHO's reward proper is the majority-vote pseudo-label (same as TTRL); entropy and confidence serve only as token-level advantage shaping (i.e., $A^{\text{hyb}}_{i,t} = A^{\text{grp}}_{g,i}(1 + a S_{i,t})$), modulating gradient magnitude locally and not constituting an independent reward. The dominant artifact is therefore the majority-vote statistic across rollouts — same family as TTRL / RoiRL / SPINE.

Specifically:
- **Reward source:** the majority-voted pseudo-label (model-generated content) — the reward proper. The entropy-confidence hybrid signal serves only as an advantage-shaping term that modulates token-level gradient magnitudes — a secondary mechanism.
- **Signal type:** the reward itself is multi-rollout consensus (Family II); entropy $H_i$ and confidence $C_i$ enter only the advantage modulation.
- **Optimization carrier:** online policy updates via a clipped policy-gradient objective (GRPO-style) — a policy-optimization carrier.

---

## 2. Problem Addressed

Existing Test-Time Reinforcement Learning (TTRL) methods face two challenges:

1. **High-entropy rollout collapse:** entropy-based tree-search methods (e.g. ETMR) branch frequently at high-entropy nodes; the branching budget is consumed by a few high-entropy trajectories, the search tree degenerates into near-chain-like rollouts, and exploration coverage and pseudo-label diversity drop sharply.
2. **Early pseudo-label overfitting (self-reinforcing overfitting):** early-training pseudo-rewards are noisy and biased; the policy is easily pushed toward locally high-score solutions, forming a positive feedback loop — the output distribution sharpens fast (entropy drops), exploration dries up, and late-stage generalization degrades.

ECHO aims to mitigate both problems within a limited inference budget, improving test-time reasoning quality and generalization.

---

## 3. Method

ECHO has three core modules:

### 3.1 Entropy-Confidence Hybrid Tree-Structured Rollout

- Window-smoothed token entropy $\bar{H}_t$ and grouped confidence $C_t^G$ jointly decide **branch width** $B_t$: high entropy + low confidence → wider branching to encourage exploration; high entropy + high confidence → suppressed branching to avoid the high-entropy trap.
- **Online pruning** with three mechanisms:
  - *Low-confidence pruning:* terminate branching when the running minimum of grouped confidence $m_t < \tau_{\text{prune}}$.
  - *Tail-decline pruning:* terminate after several consecutive drops in tail confidence.
  - *Entropy-spike pruning:* terminate after several consecutive entropy spikes.
- Majority voting produces the pseudo-label $\hat{y}$; each trajectory receives a rule-based reward $R_i \in \{0, 1\}$ accordingly.

### 3.2 Confidence-Adaptive Clipping

- Dynamically tune the PPO-style clipping radius $\epsilon(o_i)$ based on trajectory-level tail confidence $C_{\text{tail}}(o_i)$:
  - High-confidence trajectory ⇒ narrower trust region (prevent early spurious high-reward trajectories from dominating updates).
  - Low-confidence trajectory ⇒ wider trust region (allow more exploratory updates).
- Formula: $\epsilon(o_i) = \epsilon_{\min} + (\epsilon_{\max} - \epsilon_{\min})\,\sigma\!\big(\kappa(1-C_{\text{tail}}(o_i))\big)$.

### 3.3 Entropy-Confidence Hybrid Advantage Shaping

- On top of GRPO's group-normalized trajectory-level advantage $A_{g,i}^{\text{grp}}$, add a token-level shaping signal:
  $$S_{i,t} = \alpha\, H_{i,t} + \beta\,(1 - C_{i,t})$$
- Final hybrid advantage: $A_{i,t}^{\text{hyb}} = A_{g,i}^{\text{grp}}\,(1 + a\, S_{i,t})$.
- Effect: tilt gradient signals toward uncertain yet decision-critical tokens, promoting learning on key reasoning steps while suppressing over-updates on high-entropy degenerate regions.

### 3.4 Overall objective

$$\mathcal{L}_{\text{ECHO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(r_{i,t}(\theta)\,A_{i,t}^{\text{hyb}},\;\text{clip}(r_{i,t}(\theta),\,1-\epsilon(o_i),\,1+\epsilon(o_i))\,A_{i,t}^{\text{hyb}}\big) - \beta_{\text{KL}}\,D_{\text{KL}}\big(\pi_\theta \| \pi_{\text{ref}}\big)\Big)\right]$$

---

## 4. Datasets

### Natural-language math reasoning (training / test sets)
| Dataset | Description |
|---|---|
| AIME2024 | American Invitational Mathematics Examination 2024. |
| AMC | American Mathematics Competition. |
| MATH-500 | 500-problem subset of the MATH dataset. |
| GPQA-Diamond | Graduate-level Q&A. |
| AIME2025 | American Invitational Mathematics Examination 2025. |

### Multimodal math reasoning
| Dataset | Description |
|---|---|
| Geometry3k | Geometry-reasoning training set. |
| GeoQA | Geometry Q&A training set. |
| MathVision | Visual math reasoning. |
| MathVerse | Visual math reasoning. |
| MathVista | Visual math reasoning. |
| LogicVista | Logical visual reasoning. |

---

## 5. Evaluation metrics and main results

### Metrics
- **pass@16:** accuracy when at least one of 16 rollouts is correct (natural-language reasoning).
- **pass@1:** single-generation accuracy (multimodal reasoning).

### Main results

**Natural-language reasoning (Table 1, pass@16):**
| Backbone | Training set | AIME2024 | AMC | MATH-500 | GPQA | AIME2025 | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | AIME2024 | **30.0** | **75.9** | **89.4** | **47.7** | **33.3** | **55.3** |
| Qwen2.5-7B | MATH-500 | **33.3** | **75.9** | **90.0** | **49.0** | **43.3** | **58.3** |
| Qwen3-8B | AIME2024 | **53.3** | **90.4** | **95.4** | **81.3** | **60.0** | **76.1** |
| Qwen3-8B | MATH-500 | **46.7** | **90.4** | **94.6** | **89.9** | **56.7** | **75.6** |

- Compared with the strongest baseline (INTUITOR), ECHO improves by about **3–4 points** on average across settings.
- On the hardest task (AIME2025), the improvement can reach **12.36%**.

**Multimodal reasoning (Table 2, pass@1):**
- On Qwen2.5-VL-7B and Qwen3-VL-8B, ECHO improves pass@1 over the MM-UPT baseline by **1.2–2.4 points** on average; on LogicVista, the relative improvement can reach **12.8%**.

### Ablation study (Table 3)
| Component | Effect |
|---|---|
| EC-Tree (entropy-confidence tree search) | Largest contribution; removal causes major drops. |
| CA-Clip (confidence-adaptive clipping) | Substantial impact on hard tasks (AIME2025: 54.3 → 30.1 when removed). |
| E-SC Adv (entropy-confidence advantage shaping) | Stable contribution; removal causes mild degradation on multiple benchmarks. |

### Key findings
1. ECHO effectively mitigates high-entropy rollout collapse: branching budget is distributed more evenly, and the number of effective branches rises sharply.
2. ECHO slows entropy's premature drop, suppressing self-reinforcing overfitting.
3. Under strict IID setup (training set = test set), ECHO still yields 4–9% average gain, showing the method is not sensitive to distribution shift.
