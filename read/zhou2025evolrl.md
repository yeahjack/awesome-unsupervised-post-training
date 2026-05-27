# EVOL-RL: Evolving Language Models without Labels

> **Added to survey on:** 2026-03-11

**Paper:** Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation
**arXiv:** 2509.15194
**Method:** EVOL-RL | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / evolutionary round |
| Persistence | full parameter accumulate across rounds |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | note-explicit |
| Timing Regime (auxiliary taxonomy) | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates fire during pre-deployment evolution / coevolution rounds; the basic unit is a prompt batch plus one round of group selection or variation.
- **Whose sample it serves:** the variants and selection results from the current round mainly serve the next evolution round and the final deployed model, not the immediate inference of the current sample.
- **Whether parameters/state accumulate:** parameters accumulate across multiple evolution rounds; no per-sample reset.
- **Reset boundary:** the temporal structure of these methods is therefore offline evolutionary post-training.

## 1. UPT Assignment Rationale

EVOL-RL belongs to **Family II (Sample-Relation Supervision)**, sub-class population consensus via majority vote / distributional agreement.

Core reason: EVOL-RL's supervision signal comes entirely from relational structure inside the model population, with no reliance on external labels, verifiers, or human feedback. Specifically, for each prompt the model samples a set of responses; majority voting over them provides a pseudo-label (the selection signal), and each response's semantic novelty relative to the rest of the group provides a variation signal. Both signals come from distributional agreement and semantic dissimilarity within the same batch of sampled responses — relational / population-level internal supervision. The final reward is fed into GRPO for label-free policy optimization. The whole pipeline uses no ground-truth answers and no external judges.

---

## 2. Problem Addressed

The core challenge of existing label-free self-improvement methods — **entropy collapse**:

- **RLVR** depends on an external verifier and applies only to domains with automatic verification (math, code); cannot generalize to open-domain reasoning.
- **TTRL** and other majority-only methods need no labels but reward only responses that match the majority answer, leading to over-confident models and loss of solution diversity, manifested as policy entropy approaching zero, response length shrinking, and pass@n (n>1) dropping continuously.
- Existing entropy regularization or clip-high tricks are not enough to fundamentally solve the "majority trap".
- A label-free training framework is needed that simultaneously preserves **correctness anchoring** and **reasoning diversity**.

EVOL-RL borrows from biological evolution (selection + variation), balancing exploitation and exploration in a label-free setting and preventing entropy collapse.

---

## 3. Method

EVOL-RL uses GRPO as the optimization algorithm and designs a reward function that combines selection and variation:

### 3.1 Optimization framework (GRPO)

For each prompt $\mathbf{q}$, the policy samples $G$ responses $\{o_1, \dots, o_G\}$. Each response gets a scalar reward $r_i$, z-score normalized within the group to obtain the advantage $\hat{A}_i$, then updated with a PPO-style clipped surrogate objective plus a KL penalty for stability.

### 3.2 Reward design: selection + variation

Each response is scored along three dimensions:

1. **Validity:** the response must give a numeric answer in `\boxed{}`; otherwise reward = -1.
2. **Majority (Selection):** for valid responses, assign a binary label $y_i \in \{+1, -1\}$ depending on whether the final answer matches the majority-voted answer.
3. **Novelty (Variation):** compute embeddings of the reasoning portion of each response and build a cosine-similarity matrix. For response $o_i$, compute the average similarity $\bar{s}_i$ and the maximum similarity $m_i$ **within the same group**; the novelty score is

$$u_i = 1 - (\alpha\,\bar{s}_i + (1-\alpha)\,m_i), \quad \alpha = 0.5$$

Min-max normalize it within the majority and minority groups separately to obtain $\tilde{u}_i$.

### 3.3 Final reward mapping

Map the majority label and novelty score to **non-overlapping reward bands**:

$$r_i = \begin{cases} -1, & \text{if invalid} \\ 0.5 + 0.5\,\tilde{u}_i \in [0.5, 1], & \text{if } y_i = +1 \\ -1 + 0.5\,\tilde{u}_i \in [-1, -0.5], & \text{if } y_i = -1 \end{cases}$$

This guarantees that **any majority solution has a higher reward than any minority solution** (correctness first), while novelty makes fine distinctions inside each group (encourages diversity).

### 3.4 Auxiliary mechanisms

- **Asymmetric clipping:** $\epsilon_{\text{high}} > \epsilon_{\text{low}}$, letting rare, novel, correct solutions with high advantage receive fuller gradient updates.
- **Token-level entropy regularizer:** $\mathcal{L}_{\text{ent}} = -\lambda_{\text{ent}} \mathbb{E}[\frac{1}{|o|}\sum_t H(\pi_\theta(\cdot | o_{<t}, x))]$, preserving token-level diversity early in generation.
- Total objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \mathcal{L}_{\text{ent}}$.

### 3.5 Evolutionary analogy

- **Majority vote = selection pressure:** anchors the correct direction, preventing drift.
- **Novelty reward = variation pressure:** rewards semantically unique reasoning paths, preventing collapse.
- **Entropy regularizer = high mutation rate:** sustains the supply of diverse candidates.
- **Asymmetric clipping = preserve favorable mutations:** prevents rare high-value solutions from being clipped out.

Training dynamics show two phases: (1) early on the majority signal dominates and entropy drops quickly (TTRL-like); (2) past the "evolving point", the novelty + entropy mechanisms kick in — entropy rebounds, response lengths grow, out-of-domain accuracy keeps improving, while TTRL stays permanently in the low-entropy state.

---

## 4. Datasets

### Training sets
- **MATH-TRAIN:** large-scale standard math training set (questions only; no labels).
- **MATH-500:** smaller 500-problem subset.
- **AIME24:** competition-level math problems (30 problems).

### Evaluation sets
- **MATH-500** (in-domain).
- **AIME24** (competition math).
- **AIME25** (out-of-domain competition math).
- **AMC** (American Mathematics Competition).
- **GPQA-Diamond** (non-math reasoning, cross-domain generalization).
- **MMLU-Pro**, **SuperGPQA**, **BBEH** (broader reasoning benchmarks, evaluated only on the 8B model).

### Models
- **Qwen3-4B-Base**.
- **Qwen3-8B-Base**.
- (Supplementary experiments on OctoThinker-8B-Hybrid-Base in the appendix.)

---

## 5. Evaluation metrics and main results

### Metrics
- **pass@1:** single-sample accuracy.
- **pass@16:** probability that at least one of 16 samples is correct (measures reasoning diversity).
- All averaged over 32 rollouts.

### Main results (Table 1)

**Qwen3-4B-Base, trained on MATH-TRAIN:**

| Benchmark | TTRL pass@1/pass@16 | EVOL-RL pass@1/pass@16 | Delta |
|-----------|---------------------|------------------------|-------|
| MATH | 75.4/86.9 | 80.0/93.3 | +4.6/+6.4 |
| AIME24 | 12.1/23.2 | 20.7/47.6 | +8.6/+24.4 |
| AIME25 | 6.8/28.6 | 17.5/39.9 | +10.7/+11.3 |
| GPQA | 36.5/81.4 | 37.2/88.7 | +0.7/+7.3 |

**Qwen3-8B-Base, trained on MATH-TRAIN:**

| Benchmark | TTRL pass@1/pass@16 | EVOL-RL pass@1/pass@16 | Delta |
|-----------|---------------------|------------------------|-------|
| AIME24 | 16.7/37.6 | 26.0/51.7 | +9.3/+14.1 |
| AIME25 | 15.6/35.9 | 21.6/43.1 | +6.0/+7.2 |

### Key findings

1. **pass@1 and pass@16 improve simultaneously:** EVOL-RL consistently beats TTRL across all configurations; pass@16 gains are particularly large (+24.4 pp on AIME24 with the 4B model), indicating multi-path exploration benefits from preserved diversity.
2. **Consistency across model sizes and training-data scales:** 4B and 8B models, large MATH-TRAIN, small MATH-500, and AIME24 all show effects.
3. **Robustness across difficulty levels:** the 4B model trained on MATH-500 reaches AIME24/25 performance close to that of training directly on AIME24 — the learned reasoning is transferable rather than overfitted.
4. **Non-math-domain generalization:** TTRL's pass@16 on GPQA drops below the base model, while EVOL-RL stably recovers and improves it (+7 to +15 pp over TTRL); on MMLU-Pro / SuperGPQA / BBEH, EVOL-RL beats both TTRL and the base model on pass@1 and pass@4.
5. **Ablation:** the three components (novelty reward, entropy regularizer, asymmetric clipping) work synergistically; the full configuration is best. Removing the novelty reward hurts most on simple datasets; removing entropy/clipping hurts most on hard ones.
6. **Components transfer to supervised RLVR:** adding the three exploration components to standard labeled GRPO (RLVR) raises pass@16 by 7%–12% on AIME24/25.
