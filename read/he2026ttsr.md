# TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement

> **Added to survey on:** 2026-03-11

**Paper:** TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement
**Authors:** Haoyang He, Zihua Rong, Liangjie Zhao, Yunjia Zhao, Lan Yang, Honggang Zhang (Beijing University of Posts and Telecommunications; Institute of Computing Technology, CAS; Southwestern University of Finance and Economics)
**ArXiv:** 2603.03297
**Date:** 2026-02-06

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| TTSR | Policy Opt. | test-time | Semantic |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | original test set plus synthesized variants |
| Persistence | full parameter accumulate across curriculum cycles |
| Inference Coupling | adapt on the evolving cohort, then re-infer in later cycles |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** updates run on the original test set plus its synthesized variants, on a curriculum-cycle schedule.
- **Serving the current sample or future ones:** updates in the current cycle mainly serve subsequent cycles and the subsequent evaluation rather than resetting after each sample.
- **Whether parameters/state accumulate:** parameters accumulate across curriculum cycles; the cohort evolves through reflection, reassignment, or resampling.
- **Reset boundary:** thus this is self-curriculum test-time adaptation rather than plain static test-cohort RL.

## 1. UPT Assignment Rationale

**Family III — Self-Generated Target Bootstrapping (reasoning / curriculum synthesis)**

TTSR is unsupervised post-training (UPT) because at test time it depends on no external ground-truth labels or stronger teacher model. The framework uses **a single pretrained model**, alternating between Student and Teacher functional roles for autonomous evolution.

Concretely, TTSR's generated internal artifacts include:
1. **Pseudo-target:** the Student samples multiple reasoning trajectories on a test question and majority-votes to a consensus outcome $\hat{y}(x)$ as the pseudo-correct reference, providing the GRPO reward signal.
2. **Reflection summary:** the Teacher observes Student failure trajectories, analyzes process-level deficiencies, and summarizes recurring reasoning weaknesses as structured JSON (containing reasoning_weakness, trigger_conditions, failure_signature).
3. **Targeted variant questions:** the Teacher synthesizes new training questions from the reflection summary; these preserve the original problem's core reasoning structure but deliberately expose the Student's known weaknesses, forming an adaptive curriculum.

These artifacts are entirely model-generated, with no human annotation or external model guidance. Variant questions become synthetic training targets injected into the Student's training set, driving test-time parameter updates — consistent with "Direct optimization → reasoning / curriculum synthesis" subclass.

---

## 2. Problem Addressed

TTSR targets two core challenges of Test-Time Training (TTT):

1. **Hard test problems make pseudo-labels unreliable:** on challenging reasoning tasks, test problems usually lie at the capability frontier. Model-generated pseudo-labels / reward signals are noisy and unstable, leading to inefficient or degenerative parameter updates. Existing methods (e.g., TTRL) apply RL directly to original hard problems; when pass rate is very low, the reward signal is nearly uninformative.

2. **No mechanism targeting specific reasoning weaknesses:** existing TTT and self-play methods focus on task-level diversity or difficulty scaling and ignore fine-grained capability deficiencies surfaced on specific instances. All reasoning errors are treated as undifferentiated noise — specific weaknesses cannot be efficiently corrected.

3. **Dependence on stronger external models:** some TTT methods depend on stronger teacher models for data or guidance, weakening fully autonomous self-evolution and limiting applicability where strong teachers are unavailable.

TTSR proposes: via the Teacher role, perform **trace-level reflection** on failed trajectories, diagnose specific weaknesses, and synthesize **moderately difficult targeted variant questions** (near the capability frontier), shifting the learning signal from unreliable hard problems to a learnable regime for stable test-time self-evolution.

---

## 3. Method

TTSR is a **teacher-reflective test-time self-evolving training** framework on top of GRPO and TTT, using a single pretrained model $\pi_\theta$ to alternately play Student and Teacher roles at test time (shared parameters, different functional behaviors).

### 3.1 Student: problem solving and test-time adaptation

The Student is the online solver and iteratively updates its policy at test time.

**Training-set construction:** in the $t$-th test-time iteration, the training set is:
$$D_t = X_{\text{test}} \cup X_{\text{var}}^{(t-1)}$$
where $X_{\text{test}}$ is the original test set and $X_{\text{var}}^{(t-1)}$ is the Teacher-synthesized variant questions from the previous round.

**Majority Voting Reward:** for each training question $x \in D_t$, the Student samples $G$ reasoning trajectories:
$$\{y_i\}_{i=1}^G \sim \pi_{\theta_t}(\cdot|x)$$
Majority voting yields the consensus outcome $\hat{y}(x)$, and pseudo-correctness reward is assigned:
$$R_S(y_i|x) = \mathbb{I}[y_i = \hat{y}(x)]$$

Define the empirical pseudo-correctness score:
$$s_t(x) = \frac{1}{G}\sum_{i=1}^G R_S(y_i|x)$$
This measures Student prediction stability under stochastic sampling. The Student is optimized with GRPO based on pseudo-correctness rewards.

### 3.2 Teacher: Reflection-Guided Curriculum Synthesis

The Teacher does not solve problems but observes Student reasoning failures and synthesizes an adaptive curriculum.

**Reflection on Reasoning Steps:** in iteration $t$, the Teacher collects trajectories with zero pseudo-correctness reward from the previous round and randomly samples up to $M$ failed instances. Each instance is a tuple $(x, y, \hat{y})$ (question, reasoning trajectory, pseudo-correct reference). The Teacher analyzes the reasoning steps in $y$ and produces a reflection summary characterizing incorrect reasoning patterns and missing/insufficient steps. Reflection focuses on **process-level deficiencies** rather than final-answer correctness.

**Reflection-Guided Question Synthesis:** based on the failed set $F^{(t-1)} = \{(x_k, y_k, \hat{y}_k)\}_{k=1}^m$, build a natural-language synthesis prompt $p_t$ and synthesize variant questions:
$$X_{\text{var}}^{(t)} = \{x'_j\}_{j=1}^M \sim \pi_{\theta_t}(\cdot \mid p_t, F^{(t-1)})$$
Synthesized questions retain the original problem's core reasoning structure, selectively modifying conditions or constraints to expose the reflected reasoning weaknesses. Group sampling encourages diversity.

### 3.3 Difficulty Reward

Use an entropy-based capability-frontier reward to evaluate variant difficulty:
$$R_{\text{diff}}(x') = \frac{H(\text{Bern}(s_t(x')))}{log\,2} = -\frac{s_t(x') \log s_t(x') + (1-s_t(x')) \log(1-s_t(x'))}{\log 2}$$
Reward maximizes at $s_t(x') \approx 0.5$ — the Student's maximal uncertainty on the problem, at the capability frontier.

### 3.4 Similarity Penalty Reward

To encourage exploration and reduce redundancy, introduce a group-level similarity penalty. Let $X' = \{x'_1, \ldots, x'_M\}$ be the current synthesized variants and $x$ the original test question; extended set $Z = X' \cup \{x\}$:
$$R_{\text{sim}}(x'_i, x) = \frac{1}{|Z|-1}\sum_{z \in Z \setminus \{x'_i\}} \max(0, \text{sim}(x'_i, z) - \tau)$$
where $\tau$ is a tolerance threshold for textual overlap. Similarity is based on sequence-based matching (contiguous overlapping spans), normalized as $\text{sim}(S_1, S_2) = 2M/T$.

### 3.5 Teacher Reward

The Teacher's composite reward is:
$$R_T(x'_i) = \max(0, R_{\text{diff}}(x'_i) - \lambda R_{\text{sim}}(X', x))$$
balancing difficulty (close to capability frontier) and diversity (avoid redundancy). It also enforces a format constraint: only questions correctly wrapped in `<question>` tags participate in reward.

### 3.6 Continual Self-Evolving Loop

Student and Teacher alternate to form a continual self-evolving loop: each round, the Teacher reflects on the previous Student failures → synthesizes targeted variants → the Student trains on original test questions + variants → updates the policy → next round reflects again. This co-evolution ensures the synthesized curriculum stays aligned with the Student's evolving capability.

### 3.7 Teacher Prompt Design

The Teacher uses two-stage prompting:
1. **Weakness extraction prompt:** input = original question + failed reasoning trace; output = structured JSON (reasoning_weakness, trigger_conditions, failure_signature, localization_summary).
2. **Question synthesis prompt:** input = original question + failed trace + weakness JSON; follow a 5-step flow (Anchor Structure → Error-Hitting Strategy → Generate Question → Hit Rationale → Self-Test Filter) to produce a targeted variant question.

---

## 4. Datasets

| Domain | Dataset | Description |
|------|--------|------|
| Mathematical Reasoning | AMC23 | Competition-style math, demanding precise reasoning and case analysis |
| Mathematical Reasoning | MATH-500 | Diverse math across algebra, geometry, number theory, combinatorics |
| Mathematical Reasoning | Minerva Olympiad | Advanced olympiad problems requiring long reasoning chains and non-trivial insights |
| Mathematical Reasoning | OlympiadBench | Olympiad-level bilingual multimodal science problems |
| Mathematical Reasoning | AIME 2024 | High-difficulty math competition; a single-step error can break the entire problem |
| Mathematical Reasoning | AIME 2025 | Same as above, 2025 edition |
| General Reasoning | GPQA-Diamond | Graduate-level science QA requiring deep reasoning |
| General Reasoning | MMLU-Pro | Broad multi-domain reasoning evaluation |

---

## 5. Evaluation metrics and main results
### Metrics

- **Mean@32:** for AIME 2024/2025, average accuracy over many sampled reasoning trajectories
- **Greedy Decoding (Pass@1):** for other benchmarks, greedy-decoding accuracy

### Main results

**Table 1: cross-model, cross-benchmark evaluation**

| Method | AMC23 | MATH500 | Minerva | Olympiad | AIME24 | AIME25 | GPQA-D | MMLU-Pro | Δ (avg) |
|--------|-------|---------|---------|----------|--------|--------|--------|----------|---------|
| **Qwen3-4B-Base** | | | | | | | | | |
| Base | 45.3 | 72.1 | 32.4 | 40.2 | 12.4 | 5.8 | 25.7 | 52.0 | – |
| R-Zero | 54.1 | 76.8 | 40.7 | 44.0 | 18.2 | 9.1 | 28.4 | 55.0 | +5.1 |
| TTRL | 55.8 | 79.1 | 43.6 | 46.0 | 17.6 | 9.7 | 29.1 | 56.0 | +6.4 |
| **TTSR** | **61.0** | **82.4** | **53.0** | **45.3** | **25.6** | **20.1** | **34.2** | **60.8** | **+12.1** |
| **Qwen3-8B-Base** | | | | | | | | | |
| Base | 51.4 | 77.9 | 39.6 | 41.2 | 15.9 | 9.8 | 33.1 | 58.6 | – |
| R-Zero | 58.7 | 82.1 | 47.8 | 48.6 | 22.6 | 13.4 | 36.7 | 61.5 | +5.5 |
| TTRL | 61.9 | 84.3 | 50.2 | 50.8 | 26.1 | 15.7 | 38.4 | 62.8 | +7.8 |
| **TTSR** | **66.4** | **87.5** | **54.9** | **55.2** | **30.8** | **19.1** | **42.6** | **66.7** | **+12.0** |
| **OctoThinker-8B-Hybrid-Base** | | | | | | | | | |
| Base | 28.4 | 45.6 | 15.2 | 16.8 | 7.1 | 3.9 | 15.2 | 26.8 | – |
| R-Zero | 35.7 | 54.1 | 22.9 | 25.4 | 11.6 | 6.8 | 19.7 | 31.4 | +6.1 |
| TTRL | 38.9 | 57.3 | 26.4 | 28.7 | 13.9 | 8.2 | 21.5 | 33.2 | +8.6 |
| **TTSR** | **46.8** | **64.9** | **34.1** | **36.8** | **19.7** | **12.4** | **27.9** | **39.8** | **+15.4** |

**Ablation (Qwen3-8B)**

| Settings | MATH500 | AIME25 | Olympiad | GPQA-D |
|----------|---------|--------|----------|--------|
| TTSR (Full) | 87.5 | 19.1 | 55.2 | 42.6 |
| w/o Reflection-Guided Synthesis | 85.0 (↓2.5) | 14.2 (↓4.9) | 49.7 (↓5.8) | 38.3 (↓4.3) |
| w/o Teacher Test-Time Update | 82.7 (↓4.8) | 13.3 (↓5.8) | 49.4 (↓6.1) | 37.9 (↓4.7) |
| w/o Similarity Penalty | 85.9 (↓1.6) | 16.3 (↓2.8) | 52.9 (↓2.6) | 40.1 (↓2.5) |

### Key findings

1. **TTSR consistently beats baselines across all models and benchmarks:** average gains of +12.0 to +15.4 points, clearly above TTRL (+6.4 to +8.6) and R-Zero (+5.1 to +6.1).
2. **Gains are largest on hardest tasks:** on AIME 2024/2025 and other tasks requiring deep multi-step reasoning, TTSR improves by more than 10 points (e.g., Qwen3-4B on AIME25 from 5.8 to 20.1) — reflection-guided synthesis is especially effective for hard problems.
3. **Strong cross-domain transfer:** trained on AIME25, TTSR transfers to GPQA-D with +7.2 (33.1 → 40.3), while TTRL gains only +1.3 (to 34.4); reverse from GPQA-D to AIME25 also yields +4.3, indicating TTSR's test-time updates capture reusable reasoning refinements.
4. **Ablations confirm each component's contribution:** dropping Reflection-Guided Synthesis loses 5.8 on Olympiad; dropping Teacher Test-Time Update (i.e., Teacher–Student co-evolution) causes the largest overall drops (AIME25 −5.8, Olympiad −6.1); dropping the similarity penalty causes smaller but consistent drops.
5. **Largest gains on reasoning-oriented inductive-bias models:** OctoThinker-8B-Hybrid-Base sees +15.4 — the highest of the three — indicating synergy between structured reflection and reasoning-oriented architectures.
6. **Compact and efficient training config:** Batch Size = 16, Student Rollout G = 8, Teacher Variants M = 4, Iterations T = 20, KL Coef = 0.001, LR = 3e-7, MaxLen = 4096 — uniform across all benchmarks and models.
