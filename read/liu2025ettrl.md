# ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning Via Entropy Mechanism

> **Added to survey on:** 2026-03-11

> **Method:** ETTRL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv: 2508.11356 | Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, ShaoGuo Liu (Kuaishou Technology, Beihang University, Northwestern Polytechnical University)

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

ETTRL belongs to **Family II (Sample-Relation Supervision, population consensus)**. The method refines TTRL: the core reward signal comes from majority-voting over the model's own rollouts (consensus-based pseudo-label), with no reliance on external ground truth, human labels, or external verifiers. Its two core components (ETMR and EAR) both center on entropy — an intrinsic statistic — to balance exploration and exploitation; all supervision signals come from internal consistency relations among the model population.

---

## 2. Problem Addressed

TTRL (Test-Time Reinforcement Learning) lets an LLM self-optimize on unlabeled test data via majority-voted pseudo-labels, but it has two key drawbacks:

1. **Too-high inference cost:** TTRL needs dozens to hundreds of parallel rollouts to obtain reliable pseudo-labels — token consumption is enormous.
2. **Early-training estimation bias (overconfidence):** early in training, pseudo-label accuracy is very low (below 10% on AIME); the few "lucky" correct samples get disproportionately large advantages, pushing the model into local optima early and blocking further exploration.

ETTRL aims to simultaneously improve the exploration–exploitation balance on two dimensions: rollout efficiency and reward-advantage estimation.

---

## 3. Method

ETTRL contains two complementary components:

### 3.1 Entropy-fork Tree Majority Rollout (ETMR)

- **Core idea:** use token-level entropy to identify the key branching points of reasoning (high-entropy tokens, typically logical-transition words like "but" or "however"), branching only at these positions while reusing low-entropy tokens.
- **Procedure:**
  - Generate M independent trees; each tree starts from one full sampling.
  - Compute each token's Shannon entropy and pick the Top-N highest-entropy positions as forking points.
  - At each forking point, produce B new branches; each branch independently generates to a complete response.
  - All leaf responses are aggregated by majority voting to produce a pseudo-label.
- **Efficiency:** the total number of rollouts is $R_{tree} = M(1 + B \times N)$. In a typical setting ($N=3, B=2$), token consumption is only about **60%** of fully parallel sampling, while sampling diversity is preserved or improved.

### 3.2 Entropy-based Advantage Reshaping (EAR)

For the overconfidence in early-training advantage estimation, two strategies are proposed:

1. **Adv-Clip:** clip GRPO's advantage to $[-\beta, +\beta]$ to directly suppress extreme gradient updates and stabilize early training.
2. **Adv-Res (main method):** scale the advantage based on response-level relative entropy:
   $$Y_i = 1 + \frac{\text{avg}(H_{resp}(o_i)) - H_{resp}(o_i)}{\text{avg}(H_{resp}(o_i))}$$
   $$\hat{A}^{res}_{i,t} = Y_i \cdot \hat{A}_{i,t}$$
   - High-entropy (low-confidence) responses are down-weighted; low-entropy (high-confidence) responses have their advantage moderately amplified.
   - Compared with Adv-Clip, Adv-Res provides finer-grained soft regularization and is better across all models and datasets.

---

## 4. Datasets

- **AIME 2024**: American Invitational Mathematics Examination problems (30 problems); 80 training episodes.
- **AMC** (American Mathematics Competition): competition math problems; 30 training episodes.
- **MATH-500**: math reasoning benchmark; 10 training episodes.

All datasets are used at test time **without ground-truth labels**; rewards rely entirely on majority-voted pseudo-labels.

---

## 5. Evaluation metrics and main results

**Metric:** Pass@1 (greedy decoding).

### ETMR results (Table 1, vs. TTRL at similar rollout count)

| Model | Method | AIME 2024 | AMC | MATH-500 | Avg |
|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| | ETMR | 21.0 | 50.8 | 76.9 | 49.6 |
| | $\Delta$ | +32.9% | +3.9% | +5.3% | +8.1% |
| Qwen2.5-Base-3B | TTRL | 7.9 | 40.7 | 72.2 | 40.3 |
| | ETMR | 9.2 | 41.7 | 71.7 | 40.9 |
| | $\Delta$ | +16.5% | +2.5% | -0.7% | +1.5% |
| Llama-3.1-8B | TTRL | 10.0 | 32.3 | 63.7 | 35.3 |
| | ETMR | 16.9 | 35.4 | 59.5 | 37.3 |
| | $\Delta$ | **+69.0%** | +9.6% | -6.6% | +5.7% |

With only 60% of tokens, ETMR delivers substantial AIME 2024 gains (a 68% relative gain on Llama-3.1-8B).

### EAR results (Table 2, advantage-shaping methods)

| Model | Method | AIME 2024 | AMC | MATH-500 | Avg |
|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| | Adv-Res | 19.6 | 51.0 | 77.3 | 49.3 |
| | $\Delta$ | +24.1% | +4.3% | +5.9% | +7.4% |
| Qwen2.5-Base-3B | TTRL | 7.9 | 40.7 | 72.2 | 40.3 |
| | Adv-Res | 13.1 | 41.4 | 72.4 | 42.3 |
| | $\Delta$ | **+65.8%** | +3.2% | +0.3% | +5.0% |
| Llama-3.1-8B | TTRL | 10.0 | 32.3 | 63.7 | 35.3 |
| | Adv-Res | 13.5 | 36.4 | 61.3 | 37.1 |
| | $\Delta$ | +35.0% | +12.7% | -0.8% | +5.1% |

**Key findings:**
- Both ETMR and EAR consistently surpass the vanilla TTRL baseline across all models and datasets.
- The gains are most pronounced on the hardest benchmark (AIME 2024) — entropy-based methods are especially effective at exploring complex reasoning.
- Adv-Res is always better than Adv-Clip, showing fine-grained entropy-based soft regularization outperforms hard clipping.
- Non-math-specialized models (e.g. Llama-3.1-8B, Qwen2.5-Base-3B) benefit more, owing to higher epistemic uncertainty.
