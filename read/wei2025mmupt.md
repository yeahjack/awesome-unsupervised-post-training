# MM-UPT: First SFT, Second RL, Third UPT

> **Added to survey on:** 2026-03-11

**Paper:** First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training
**arXiv:** 2505.22453
**Venue:** NeurIPS 2025
**Authors:** Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun
**Affiliations:** Shanghai Jiao Tong University, Zhongguancun Academy, Shanghai Innovation Institute, Lehigh University

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | unlabeled training dataset / per-sample rollout group |
| Persistence | full parameter accumulate across dataset episodes |
| Inference Coupling | adapt on the unlabeled cohort, then evaluate on held-out benchmarks |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** updates fire episode-by-episode on the unlabeled training set before deployment / evaluation — not on an arrival-by-arrival test stream.
- **Whose sample it serves:** the current sample's rollouts and rewards primarily serve subsequent steps within the same training cohort and the benchmark evaluation after training.
- **Whether parameters/state accumulate:** parameters accumulate across the entire unlabeled cohort; no per-sample reset.
- **Reset boundary:** so it is closer to offline unlabeled post-training followed by batch evaluation, not online sample-by-sample TTA.

## 1. UPT Assignment Rationale

**Family II — Sample-Relation Supervision (population consensus → majority vote)**

MM-UPT belongs to Family II for the following reasons:

- **Self-rewarding mechanism:** for the same unlabeled question, the model samples G answers and picks the most frequent one as the pseudo-label $y^*$ via **majority voting**, then assigns each answer a binary reward (1 if matching $y^*$, else 0).
- **Population-consensus signal:** the reward comes entirely from the consensus relation across the model's own sampled answers (relational); it does not depend on external ground truth, verifiers, or human/AI annotation.
- **GRPO policy optimization:** feed the majority-vote binary reward into GRPO for online RL, updating the MLLM policy until the model converges to high-confidence, consistent answers.
- **Data self-generation:** further, the model is allowed to generate new visual reasoning training data through two strategies — in-context synthesizing and direct synthesizing — with pseudo-labels still produced by majority voting.
- **Carrier:** Policy Optimization (GRPO) | **Regime:** training-time | **Level:** Semantic.

---

## 2. Problem Addressed

Post-training of MLLMs has traditionally relied on SFT and RL, both of which require substantial annotated data (ground-truth answers, human preference labels, external reward models). As task complexity grows, annotation cost becomes unsustainable.

MM-UPT proposes a three-stage post-training paradigm: **SFT → RL → UPT (Unsupervised Post-Training)**, where UPT is the third stage that aims to continually improve multimodal reasoning using only the model's own outputs, with no external supervision. The core challenge: how to build a reliable reward signal for GRPO without ground truth.

---

## 3. Method

### 3.1 Problem formulation

Given a trained MLLM $\pi_\theta$ and an unlabeled multimodal dataset $Q = \{(I_i, q_i)\}_{i=1}^N$ (images + questions, no answers), the goal is to improve performance using only the model's own outputs.

### 3.2 Training method — Majority-Vote GRPO

1. **Sampling:** for each sample $(I, q)$, use the old policy $\pi_{\theta_{old}}$ to sample $G$ answers $O = \{o_i\}_{i=1}^G$.
2. **Answer extraction:** use a rule-based answer extractor $E(\cdot)$ to extract the answer $\hat{Y} = \{\hat{y}_i\}_{i=1}^G$ from each answer.
3. **Majority voting:** pick the most frequent answer as the pseudo-label:
   $$y^* = \arg\max_{y \in \hat{Y}} \sum_{i=1}^G \mathbb{1}[y = \hat{y}_i]$$
4. **Binary reward assignment:** $r_i = 1$ if $\hat{y}_i = y^*$; else $r_i = 0$.
5. **Advantage estimation:** $\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$.
6. **GRPO optimization:** update policy parameters with the standard GRPO objective (clipped ratio + KL constraint).

### 3.3 Synthetic data — data self-generation

To further improve scalability, two strategies let the model generate new training samples:

- **In-context synthesizing:** given an original triplet (image, question, answer) as an in-context example, ask the model to produce a semantically related but different new question — similar to data rewriting.
- **Direct synthesizing:** give the model only the image with no reference question and let it freely produce new questions. More diverse but may hallucinate.

The synthesized data still get pseudo-labels via majority voting and are then used for MM-UPT training.

### 3.4 Implementation details

- Implemented in **EasyR1**, trained for 15 episodes.
- AdamW optimizer, learning rate $1 \times 10^{-6}$, weight decay $1 \times 10^{-2}$, gradient clipping 1.0.
- KL constraint $\beta = 0.01$; rollout temperature 0.7; group size $G = 10$.
- The vision tower is not frozen and participates in training.

---

## 4. Datasets

### Training data (unlabeled, questions and images only)

| Dataset | Type | Notes |
|-------|------|------|
| **Geometry3K** | Geometry problems | Multiple choice, geometric figures. |
| **GeoQA** | Geometry problems | Geometric figures + multiple choice. |
| **MMR1** | Mixed math reasoning | Includes geometry, charts, and various visual math problems. |

### Evaluation benchmarks

| Benchmark | Notes |
|-----------|------|
| **MathVision** | Multimodal math reasoning. |
| **MathVerse** | Multimodal math reasoning. |
| **MathVista** | Multimodal math reasoning, multi-type. |
| **We-Math** | Multimodal math reasoning, multi-level knowledge. |

---

## 5. Evaluation metrics and main results

**Metrics:** Accuracy (%); pass@1 and pass@10.

### Scenario 1: unsupervised training on standard datasets (GT labels removed)

Main results for MM-UPT trained on MMR1 with the **Qwen2.5-VL-7B** backbone:

| Method | MathVision | MathVerse | MathVista | We-Math | Avg |
|------|-----------|-----------|-----------|---------|-----|
| Qwen2.5-VL-7B (base) | 24.87 | 43.83 | 66.30 | 62.87 | 49.47 |
| + GRPO (supervised, MMR1) | 29.01 | 45.03 | 71.40 | 67.24 | 53.17 |
| + MM-UPT (MMR1) | **26.15** | **44.87** | **72.90** | **68.74** | **53.17** |
| + SRLM (MMR1) | 25.33 | 45.08 | 67.00 | 64.66 | 50.52 |
| + LMSI (MMR1) | 24.83 | 43.76 | 64.90 | 66.38 | 49.97 |
| + Genixer (MMR1) | 23.68 | 43.30 | 65.50 | 64.66 | 49.29 |
| + STIC (MMR1) | 23.78 | 42.72 | 66.10 | 63.74 | 49.09 |

**Key findings:**
- MM-UPT is the best among all unsupervised baselines, averaging +3.7 (49.47→53.17).
- MM-UPT even matches supervised methods (GRPO, rejection SFT).
- Effective on different backbones: Qwen2.5-VL-3B (+7.4%), MM-Eureka-7B (+1.3%), ThinkLite-VL-7B (+2.8%).

### Scenario 2: unsupervised training on synthetic data

| Data source | Generation strategy | Avg | Gain |
|--------|-----------|-----|------|
| Geo3K | Original Questions | 51.23 | +3.6% |
| Geo3K | In-Context Synthesizing | 51.00 | +3.1% |
| Geo3K | Direct Synthesizing | 52.32 | **+5.8%** |
| GeoQA | Direct Synthesizing | 52.67 | **+6.5%** |
| MMR1 | Original Questions | 53.17 | +7.5% |
| MMR1 | In-Context Synthesizing | 52.94 | +7.0% |

**Key findings:** training on synthetic data is comparable to human-written questions; Direct Synthesizing even surpasses original questions in some settings.

### Trade-offs

- **pass@1 improves but pass@10 drops:** MM-UPT lifts single-shot accuracy but reduces response diversity (the model tends to converge on high-consensus modes) — an inherent trade-off of majority-voting rewards.
- **Failure case:** on hard datasets where the model's prior knowledge is insufficient (e.g. ThinkLite-11K), majority voting amplifies errors and performance drops (49.47→44.11).
