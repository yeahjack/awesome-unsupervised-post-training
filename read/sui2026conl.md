# CoNL: Conversation for Non-verifiable Learning

> **Added to Survey:** 2026-03-11

**Paper:** Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation
**Authors:** Yuan Sui, Bryan Hooi (NUS)
**ArXiv:** 2601.21464
**Date:** 2026-01-29

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CoNL | Policy Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch with self-generated responses / judgments |
| Persistence | full parameter accumulate across self-rewarding iterations |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | paper-explicit |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit
- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When does the update fire:** Updates fire inside a pre-deployment self-rewarding / judge-bootstrapping loop, typically generating responses first and then judgments or preference pairs.
- **Serving the current sample or downstream samples:** Each batch's responses / judgments primarily serve the next training round and the final deployed model, not the immediate inference of a single test sample.
- **Whether parameters / state accumulate:** Parameters accumulate across multiple self-rewarding iterations; even when role switching (actor / judge / meta-judge) is used, it stays in the same offline training loop.
- **Reset boundary:** The adaptation timing is therefore offline iterative bootstrapping, not deployment-time TTA.

## 1. UPT Assignment Rationale
**Family IV — Internal Evaluator Bootstrapping (evaluator-driven RL)**

CoNL unifies generation, evaluation, and meta-evaluation under a single policy via a multi-agent self-play protocol. Its core mechanism is a **conversation-driven diagnostic reward**: multiple agents that share the same policy propose, evaluate, and revise solutions across four structured rounds; whether each critique helps another agent improve is quantified as a reward signal that drives policy-gradient updates. This fits the "Internal Evaluator Bootstrapping → evaluator-driven RL" paradigm exactly—evaluation verdicts (pairwise rankings + critiques) produced by the model itself are aggregated into quality scores via a Bradley-Terry model, then converted into three rewards that drive policy optimization, with no external judge or ground truth.

---

## 2. Problem Addressed

Existing LLM training methods face fundamental challenges on **non-verifiable tasks** (creative writing, open-ended dialogue, ethical reasoning):

1. **No ground-truth labels:** SFT and RLVR rely on verifiable signals and break down where there is no objective answer.
2. **LLM-as-Judge evaluation ceiling:** a single judge model's bias (verbosity bias, position bias) directly caps the training ceiling. Empirically self-rewarding models exhibit response length jumping from 1k to 2.5k characters, indicating that the internal judge prefers longer rather than better answers.
3. **No meta-evaluation:** existing methods assume evaluation ability improves automatically with generation training, or rely on a static external judge. Without any mechanism to **evaluate the evaluator itself**, the system slips into an echo chamber.
4. **Circular reasoning in self-improvement:** Self-Taught Evaluators, Meta-Rewarding LMs, and similar methods use the model's own judgment to both generate and evaluate synthetic data, possibly amplifying rather than correcting bias.

CoNL proposes a third paradigm—**peer supervision**—measuring evaluation quality through critique-driven improvement, avoiding the circularity of self-training and the domain limits of verifiable RL.

---

## 3. Method

### 3.1 The CoNL Protocol: Four Rounds of Conversation

Given a query $q$, instantiate $N$ agents from the same policy $\pi_\theta$, each assigned a different persona (Rigorous Formalist, Creative Pattern-Finder, Adversarial Skeptic, … 7 roles in total, looped if $N > 7$); shared parameters but distinct conversational roles.

**Round 0 — Initial Proposals:** each agent $i$ independently produces an initial solution $s_i^{\text{init}}$.

**Round 1 — Initial Evaluation & Critique:** after observing all initial solutions, each agent emits:
- **Blind pairwise ranking** $\mathcal{R}_i^{\text{init}}$: agents cannot see each other's rankings, ensuring independent judgment. Format: "Agent X > Agent Y".
- **Critiques** $\{c_{i \to k}\}_{k \in \mathcal{T}_i}$: detailed textual critiques toward specific agents, calling out logical errors, missed cases, and so on.

**Round 2 — Revision:** every agent receives all critiques targeting it and produces a revised solution $s_i^{\text{rev}}$: it can adopt valid feedback to fix errors or defend against invalid critiques.

**Round 3 — Final Verdict:** after observing all revised solutions, each agent emits a final pairwise ranking $\mathcal{R}_i^{\text{final}}$.

**Memory buffering:** because multi-round, multi-agent conversations may exceed the 32k context window, a memory-buffering module compresses dialogue history while preserving key decisions, reasoning, and constraints.

### 3.2 Score Aggregation: Bradley-Terry

Use the Bradley-Terry (BT) model to aggregate possibly conflicting pairwise comparisons into latent quality scores:

$$P(\text{Agent } a \succ \text{Agent } b \mid V_a, V_b) = \frac{\exp(V_a)}{\exp(V_a) + \exp(V_b)}$$

Computed at two time points:
- $V_k^{\text{init}}$: aggregated from initial rankings, the pre-conversation evaluation.
- $V_k^{\text{final}}$: aggregated from final rankings, the post-conversation evaluation.

**Core insight:** $\Delta V_k = V_k^{\text{final}} - V_k^{\text{init}}$ measures whether agent $k$'s solution improved after revision. If agent $i$ critiqued agent $k$ and $\Delta V_k > 0$, the critique correctly identified a real issue.

### 3.3 Three Rewards

**Diagnostic reward ($r_{\text{diag}}$):** the central novelty—measuring whether a critique helps others improve:

$$r_{\text{diag}}(i) = \sum_{k \in \mathcal{T}_i} \max(0, V_k^{\text{final}} - V_k^{\text{init}})$$

The $\max(0, \cdot)$ ensures only diagnostic critiques (those that correctly identify problems and lead to improvement) earn positive reward.

**Solution quality ($r_{\text{sol}}$):** rewards solutions that the group rates highly:

$$r_{\text{sol}}(i) = V_i^{\text{final}}$$

**Majority alignment ($r_{\text{meta}}$):** measures alignment of the agent's pairwise judgments with the majority view:

$$r_{\text{meta}}(i) = \frac{1}{|\mathcal{R}_i^{\text{final}}|} \sum_{(a,b) \in \mathcal{R}_i^{\text{final}}} \mathbb{I}[\text{Pref}_i(a,b) = \text{Maj}(a,b)]$$

**Composite reward:**

$$r_{\text{total}}(i) = w_1 \cdot r_{\text{sol}}(i) + w_2 \cdot r_{\text{diag}}(i) + w_3 \cdot r_{\text{meta}}(i)$$

Default weights $w_1 = 1.0$, $w_2 = 2.0$, $w_3 = 1.0$. The diagnostic reward weighs the most, emphasizing meta-evaluation learning.

### 3.4 Token-Level Credit Assignment

Different conversation segments earn different rewards (rather than uniform):

| Round | Content | Reward | Reason |
|-------|---------|--------|--------|
| 0 | Initial solution $s_i^{\text{init}}$ | $r_{\text{sol}}$ | solution quality |
| 1 | Initial ranking $\mathcal{R}_i^{\text{init}}$ | **0 (masked)** | prevents gaming the baseline $V^{\text{init}}$ |
| 1 | Critiques $\{c_{i \to k}\}$ | $r_{\text{diag}}$ | diagnostic effectiveness |
| 2 | Revised solution $s_i^{\text{rev}}$ | $r_{\text{sol}}$ | solution quality |
| 3 | Final ranking $\mathcal{R}_i^{\text{final}}$ | $r_{\text{meta}}$ | majority alignment |

**Key design choice:** initial-ranking tokens get zero reward to stop agents from strategically lowballing peers to depress $V^{\text{init}}$ and inflate $\Delta V$.

### 3.5 Policy Training

Implement importance-sampling policy gradient via the Tinker API. Train policy $\pi_\theta$ on samples from the behavioral policy $\pi_{\theta_{\text{old}}}$:

$$\mathcal{L}_{\text{IS}}(\theta) = \mathbb{E}_{x \sim \pi_{\theta_{\text{old}}}} \left[ \frac{p_\theta(x)}{p_{\theta_{\text{old}}}(x)} A(x) \right]$$

Use generalized advantage estimation ($\lambda = 0.95$), learning rate $3 \times 10^{-5}$. Ground-truth labels are **never** used in the reward computation.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Math | DeepMath-103K | hard math problems (Level 5–9) across Algebra, Calculus, Number Theory, Geometry, Probability, Discrete Math; 3,500 problems sampled (Level 6–10) |
| Math competition | AIME 2024 | 30 problems covering number theory, algebra, geometry, combinatorics, probability; integer answers 000–999 |
| Math competition | AIME 2025 | 30 problems, same format as AIME 2024 |
| Science | GPQA Diamond | 198 PhD-level multiple-choice questions (physics, chemistry, biology), 4-way |
| Science | FrontierScience | 100 expert-level scientific reasoning questions in Olympic format (physics, chemistry, biology) |
| Programming | USACO Gold & Platinum | 84 competitive programming problems (63 Gold + 21 Platinum); must pass all test cases |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **Pass@1:** select the agent with the highest final consensus score $V^{\text{final}}$ and check whether its answer is correct.
- **Pass@K:** check whether any of the top-K agents (ranked by $V^{\text{final}}$) has the correct solution.
- **Rank-$\rho$:** Spearman rank correlation between the $V^{\text{final}}$ ranking and ground-truth correctness (1/0 labels). $\rho = 1$ denotes perfect evaluation; $\rho \approx 0$ denotes random.

### Main Results (Table 2 — Pass@1)

**Qwen3-8B:**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 60.0 | 60.0 | 67.0 | 70.5 | 43.0 | 10.2 |
| Self-Consistency | 61.8 | 61.2 | 68.2 | 75.4 | 43.1 | 10.3 |
| Self-Refine | 63.2 | 62.8 | 69.5 | 72.2 | 45.5 | 11.8 |
| Multi-Agent Debate | 64.5 | 64.2 | 70.8 | 71.0 | 44.2 | 11.0 |
| Self-Rewarding (Single-Turn)* | 68.5 | 69.0 | 73.5 | 77.5 | 50.5 | 15.5 |
| Self-Rewarding (Multi-Agent)* | 69.8 | 70.2 | 74.8 | 78.8 | 52.0 | 16.8 |
| **CoNL (Ours)*** | **76.5** | **73.5** | **79.2** | **87.1** | **55.7** | **19.5** |

**Qwen3-4B-Instruct:**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 50.0 | 54.0 | 45.0 | 78.5 | 18.0 | 6.0 |
| Self-Rewarding (Multi-Agent)* | 58.9 | 62.5 | 52.8 | 85.2 | 24.8 | 10.2 |
| **CoNL (Ours)** | **63.5** | **67.4** | **55.2** | **84.9** | **27.5** | **13.4** |

**Llama-3.1-8B:**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 13.0 | 7.0 | 23.0 | 45.0 | 1.9 | 3.0 |
| Self-Rewarding (Multi-Agent)* | 20.8 | 13.8 | 31.5 | 54.2 | 7.5 | 7.2 |
| **CoNL (Ours)*** | **23.5** | **16.2** | **34.0** | **57.5** | **10.2** | 7.0 |

**Llama-3.2-3B:**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 11.5 | 6.5 | 22.0 | 42.0 | 1.8 | 2.8 |
| Self-Rewarding (Multi-Agent)* | 17.5 | 11.5 | 28.2 | 49.8 | 5.8 | 6.2 |
| **CoNL (Ours)*** | **19.8** | **14.0** | **30.5** | **53.5** | **8.2** | **8.0** |

### Ablation Study (Table 3 — Qwen3-8B)

**Effect of reward components:**

| Variant | DeepMath Pass@1 | DeepMath Rank-ρ | AIME25 Pass@1 | AIME25 Rank-ρ |
|---------|-----------------|-----------------|---------------|---------------|
| CoNL (Full) | **87.1** | **0.78** | **73.5** | **0.65** |
| w/o Diagnostic ($w_2=0$) | 83.5 | 0.55 | 68.2 | 0.44 |
| w/o Consensus ($w_3=0$) | 85.2 | 0.68 | 70.5 | 0.56 |
| w/o Solution Quality ($w_1=0$) | 84.8 | 0.70 | 69.8 | 0.58 |
| w/o Blind Ranking | 82.5 | 0.45 | 67.5 | 0.39 |

**Effect of agent count:**

| N agents | DeepMath Pass@1 | DeepMath Rank-ρ | AIME25 Pass@1 | AIME25 Rank-ρ |
|----------|-----------------|-----------------|---------------|---------------|
| N=2 | 84.5 | 0.62 | 69.5 | 0.51 |
| N=3 | 85.9 | 0.71 | 71.8 | 0.60 |
| N=4 | **87.1** | **0.78** | **73.5** | 0.65 |
| N=5 | 87.0 | 0.76 | 73.2 | **0.66** |
| N=8 | 87.4 | 0.75 | 72.9 | 0.63 |

### Critique-Quality Analysis (Table 4)

| Benchmark | Initial state | Outcome type | Rate |
|-----------|---------------|--------------|------|
| DeepMath | Incorrect (×) | Correction (× → ✓) | **82.4%** |
| DeepMath | Correct (✓) | Harm (✓ → ×) | **3.1%** |
| AIME 2025 | Incorrect (×) | Correction (× → ✓) | **41.2%** |
| AIME 2025 | Correct (✓) | Harm (✓ → ×) | **9.4%** |

### Key Findings

1. **CoNL clearly beats self-rewarding baselines:** on Qwen3-8B, CoNL beats SRT-M by 6.7 pp (AIME24: 76.5 vs. 69.8) and 8.3 pp (DeepMath: 87.1 vs. 78.8) with lower training variance.
2. **Diagnostic reward is the most critical component:** removing $r_{\text{diag}}$ drops DeepMath Pass@1 by 3.6 pp (87.1 → 83.5) and Rank-$\rho$ from 0.78 to 0.55, confirming its central role in meta-evaluation learning.
3. **Blind ranking is essential for evaluation quality:** removing it drops DeepMath Rank-$\rho$ from 0.78 to 0.45 (a 42% relative drop), because agents can game baseline rankings.
4. **Superior training stability:** entropy, solution length, and accuracy all stay stable during training, closely matching ground-truth RL; SRT's majority-voting signal yields unstable convergence.
5. **High critique safety:** on DeepMath, only 3.1% of correct solutions are damaged by misleading critiques (Harm rate); the model learns a conservative strategy of revising only when critiques are well-grounded.
6. **N=4 agents is the sweet spot:** performance climbs from N=2 to N=4; from N=5 onward gains diminish or slightly drop (N=8 on AIME25 is 72.9 vs. 73.5 at N=4); too many agents introduce coordination overhead.
7. **Cross-model generalization:** CoNL is best or near-best on Qwen3-8B, Qwen3-4B, Llama-3.1-8B, and Llama-3.2-3B, demonstrating wide applicability.
8. **Adversarial-revision mechanism prevents false-critique reward:** when a critique is wrong, the critiqued agent can defend; its score stays unchanged ($V^{\text{final}} \approx V^{\text{init}}$) and the critiquing agent earns zero reward.
