# COMPASS: Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning

> **Added to survey on:** 2026-04-14

**Paper:** arXiv 2510.17923v4
**Authors:** Jingyu Xing, Chenwei Tang, Xinyu Liu, Deng Xiong, Shudong Huang, Wei Ju, Jiancheng Lv, Ziyue Qiao
**Date:** 2025-12-09

| Method | Carrier | Regime | Level |
|------|----------------|-------------|------------|
| COMPASS | Policy Opt. | test-time | Traj. |

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

**Family II (Sample-Relation Supervision, population consensus + path-quality shaping).**

COMPASS remains in the strict UPT scope for the following reasons:

- **Real test-time RL parameter updates exist:** the paper continues to optimize the policy with GRPO on unlabeled data.
- **The supervision signal comes entirely from inside the model:** the answer reward is computed from confidence-calibrated self-consistency over multiple samples; the path reward is derived from the entropy / decisiveness of the model's own token-probability distribution.
- **No external ground truth / verifier / tool / environment reward:** no human labels, no external AI labels, no external judge, no tool execution, no environment feedback.
- **The dominant signal is still relational supervision across multiple rollouts:** although DPR introduces a path-level dense reward, DCAR's core is still constructing a pseudo-label from the population and calibrating its credibility. Overall the method aligns more with Family II than Family I.

---

## 2. Problem Addressed

The authors argue that TTRL's majority voting has at least two flaws:

- **Unreliable consensus:** wrong but frequent answers get treated as pseudo-labels, leading to error self-reinforcement.
- **Only the destination is rewarded, not the journey:** looking only at the final answer wastes much of the available information from the reasoning process.

COMPASS's design goal is to address both issues simultaneously:

- Use **DCAR** to refine the answer-level pseudo-label and make the outcome reward more credible.
- Use **DPR** to provide dense supervision for the reasoning path, instead of looking only at the final answer.

---

## 3. Method

### 3.1 DCAR: Dual-Calibration Answer Reward

DCAR first does **confidence-calibrated self-consistency**:

- For each trajectory, compute fluctuations in the top-1 vs. top-2 token-probability gap to obtain a trajectory confidence.
- Replace naive majority voting with confidence-weighted voting to produce the pseudo-label.

Then it does **credibility calibration**:

- `CGeneral`: the highest confidence among responses that support the current pseudo-label.
- `CElite`: the highest confidence among all responses.
- Use `CGeneral / CElite` as a credibility coefficient for the pseudo-label, used to scale the answer reward.

This turns the original 0/1 sparse answer reward into a continuous reward modulated by consensus quality.

### 3.2 DPR: Decisive Path Reward

- At each generation position, compute:
  - **Decisiveness:** the gap between top-1 and top-2 token probabilities.
  - **Uncertainty:** the entropy of the token distribution.
- Weight decisiveness by entropy, so that the model learns to choose tokens more decisively at the key "high-uncertainty but must-decide" nodes.

This provides a **process-centric dense reward** for the reasoning path.

### 3.3 Final reward

- The final reward is `R = Ranswer + Rpath`.
- Both answer-level and path-level intrinsic signals jointly drive the GRPO update.

---

## 4. Datasets

The paper's benchmarks span math and general reasoning:

- **AIME 2024**
- **AMC**
- **MATH-500**
- **GPQA-Diamond**

Models include:

- Llama-3.2-1B-Instruct.
- Qwen2.5-Math-1.5B.
- Qwen2.5-7B.

---

## 5. Evaluation metrics and main results

The main metric is **pass@1**.

Representative results:

- **Qwen2.5-Math-1.5B:**
  - AIME 2024: **15.8 → 18.3**.
  - AMC: **47.4 → 48.6**.
  - MATH-500: **72.4 → 73.1**.
  - GPQA: **26.1 → 29.3**.
- **Qwen2.5-7B** (compared at fewer training epochs):
  - AIME 2024: **20.0 → 23.5**.
  - AMC: **76.6 → 76.9**.
  - GPQA: **31.1 → 31.7**.
- On **Llama-3.2-1B-Instruct**, gains are not stable across all datasets — in particular, AIME drops from **6.7** to **3.5**. The authors attribute this to small models lacking foundational knowledge: the high-entropy positions emphasized by DPR look more like "genuine confusion" than "valuable reasoning forks".

Ablation:

- Removing **credibility calibration** drops performance.
- Further removing **DPR** drops it more.
- Further removing **confidence calibration** regresses to a form closer to TTRL.

This indicates COMPASS's contributions come both from a more stable consensus and from directly optimizing path quality.
