# SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

> **Added to survey on:** 2026-04-14

**Paper:** arXiv 2603.19880v1
**Authors:** Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan
**Date:** 2026-03-20

| Method | Carrier | Regime | Level |
|------|----------------|-------------|------------|
| SCRL | Policy Opt. | test-time | Semantic |

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

**Family II (Sample-Relation Supervision, population consensus with negative pseudo-labeling).**

SCRL is a strict UPT method for the following reasons:

- **Real online RL updates exist:** policy optimization with GRPO at test time.
- **Both positive and negative supervision come from the model's own rollout distribution:** positive labels come from selective positive pseudo-labeling; negative labels come from a negative set composed of low-frequency, high-uncertainty answers.
- **No external ground truth / verifier / tool / environment reward:** no human labels, no external judge, no tool execution, no environment feedback.
- **The core signal depends on the distributional structure of the rollout population:** whether it is the strict-consensus criterion for positive labels or the frequency + entropy rule for negative labels, both essentially rest on relations among the model's own outputs under the same query.

It is therefore a relatively direct extension of the TTRL family within Family II, rather than a jump to a tool-grounded or self-reward-model line.

---

## 2. Problem Addressed

SCRL focuses on a sharper problem: **weak consensus**.

- When the answer distribution is highly dispersed, majority voting still forcibly picks a "majority answer", but this majority is likely unreliable.
- In GRPO, such an erroneous pseudo-label gets amplified by group normalization, forming **label noise amplification**.
- The authors also point out that, in high-uncertainty settings, "finding the right answer" is hard, but "ruling out clearly unreliable answers" is often more reliable; TTRL therefore should not do only positive pseudo-labeling.

---

## 3. Method

### 3.1 Selective Positive Pseudo-Labeling

- Do not, as TTRL does by default, treat the modal answer as a positive label.
- Declare a positive pseudo-label only when:
  - The top answer's proportion exceeds a threshold `τpos`.
  - And it has a margin `τmarg` over the runner-up.
- Otherwise, **abstain** — give no positive label.

### 3.2 Entropy-Gated Negative Pseudo-Labeling

- For each answer, record:
  - Rollout proportion.
  - Trajectory-level uncertainty (aggregated from token entropy).
- If an answer is **low-frequency and high-uncertainty**, place it in the negative set `N-`.
- This lets the model still prune obviously bad trajectories via negative labels when no reliable positive label is available.

### 3.3 Dynamic Reward Shaping

- The reward combines:
  - The positive label.
  - The negative label.
  - An entropy-based penalty term.
- Strong consensus → strengthen positive reinforcement.
- Weak consensus → avoid wrong positive reinforcement while preserving exploration.

---

## 4. Datasets

Main experiments cover:

- **AIME25**
- **AMC**
- **MATH-500**
- **Minerva**
- **GPQA**

Additional instruct-model experiments:

- **AIME24**
- **MATH-500 / AMC**

Models include:

- Qwen2.5-3B.
- Qwen2.5-Math-7B.
- Llama-3.2-1B-Instruct.
- Llama-3.1-8B-Instruct.

---

## 5. Evaluation metrics and main results

Representative results:

- **Qwen2.5-3B, 32 candidates / 16 training samples**
  - AIME25: **2.6 → 8.4**.
  - AMC: **39.4 → 41.5**.
  - MATH-500: **66.9 → 68.2**.
- **Qwen2.5-Math-7B, 32 candidates / 16 training samples**
  - AIME25: **16.8 → 26.9**.
  - AMC: **65.7 → 66.9**.
  - Minerva: **14.5* → 41.6**.
- Under a **higher rollout budget (64 candidates / 32 training samples)**, SCRL retains its advantage, most clearly on AIME25 and Minerva.
- **Llama-3.1-8B-Instruct** average accuracy reaches **29.0%**, beating TTRL, ETMR, and RESTRAIN.

Ablations are also informative:

- Removing **Selective Positive Labeling** drops AIME25 noticeably — under weak consensus one should not force a positive label.
- Removing **Negative Labeling** drops it further — negative labels are not a dispensable add-on.
- Removing **Entropy Gate** or **Dynamic Reward** likewise regresses — uncertainty-aware filtering and reward calibration are both necessary components.
