# R-Zero: Self-Evolving Reasoning LLM from Zero Data

> **Added to survey on:** 2026-03-11

**Paper:** R-Zero: Self-Evolving Reasoning LLM from Zero Data
**Authors:** Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, Dong Yu (Tencent AI Seattle Lab, Washington University in St. Louis, University of Maryland College Park, UT Dallas)
**ArXiv:** 2508.05004
**Date:** ICLR 2026 (Feb 13, 2026)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| R-Zero | Policy Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | self-generated data batch / iteration round |
| Persistence | full parameter accumulate across synthesis / refinement rounds |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | note-explicit |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates run in an offline pre-deployment bootstrapping loop, typically a round-based schedule of "generate data / score / filter / retrain".
- **Serving the current sample or future ones:** synthetic samples or pseudo-targets produced in the current round mainly serve the next training round and the final deployed model, not the immediate inference of a particular test sample.
- **Whether parameters/state accumulate:** parameters accumulate continuously across bootstrapping rounds, with no sample-level reset.
- **Reset boundary:** the `When to Adapt` of such methods is centered on offline iterative bootstrapping rather than online test-time adaptation.

## 1. UPT Assignment Rationale

**Family III — Self-Generated Target Bootstrapping (reasoning / curriculum synthesis)**

R-Zero is a fully autonomous self-evolving framework that **relies on no external human-annotated data or pre-existing task set**; all training signals come from the model itself. Specifically:

- The **Challenger** model uses GRPO to learn to generate math problems near the Solver's capability frontier (difficulty-calibrated question synthesis), forming an **adaptive curriculum**.
- The **Solver** is trained with GRPO on Challenger-generated synthetic problems using **majority-vote pseudo-labels**; labels come from the model's own consistency voting and require no external oracle.
- Throughout the co-evolutionary loop (Challenger generates → Solver labels and filters → Solver trains → next Challenger training round), **both curriculum content and pseudo-labels are model-internal synthesized artifacts**, matching the definition of self-generated-target bootstrapping.

This belongs to the reasoning / plan / curriculum-synthesis subclass of Direct optimization: the Challenger self-generates frontier-difficulty problems; the Solver trains via GRPO on majority-vote pseudo-labels; the entire co-evolutionary curriculum and labels come from the model.

---

## 2. Problem Addressed

Existing self-evolving LLM methods face two core bottlenecks:

1. **Data-dependency bottleneck:** most RLVR (Reinforcement Learning with Verifiable Rewards) methods need large amounts of human-annotated tasks and labels as supervision, which limits cost and scalability and especially hinders AI systems aiming beyond human intelligence.
2. **Limits of label-free methods:** label-free RL methods (using confidence-, consistency-, or entropy-based rewards) avoid label dependence but still require a **pre-existing unlabeled problem corpus**, limiting scalability for truly self-evolving scenarios.
3. **Limits of self-play methods:** existing self-challenging methods (e.g., Coder-Tester self-play for code) rely on external code-execution verification oracles, hard to guarantee quality and correctness on open reasoning domains like math.

R-Zero aims to build a **fully autonomous reasoning-LLM training framework starting from zero data**, requiring no human-annotated tasks, labels, or external verifiers.

---

## 3. Method

### 3.1 Overall framework

R-Zero initializes two independent models from a base LLM: a **Challenger** $Q_\theta$ and a **Solver** $S_\phi$, alternately optimized in an iterative co-evolution loop. Each iteration has three phases:

### 3.2 Phase 1: Challenger Training

The Challenger is an autoregressive LM trained with GRPO to generate challenging math problems. The core is a reward function precisely capturing "good problem" properties:

**Uncertainty Reward:**

For a Challenger-generated problem $x$, sample $m$ answers $\{y_1, ..., y_m\}$ from the current frozen Solver $S_\phi$, derive a pseudo-label $\tilde{y}(x)$ via majority vote, and compute the Solver's empirical accuracy:

$$\hat{p}(x; S_\phi) = \frac{1}{m} \sum_{j=1}^{m} \mathbb{1}\{y_j = \tilde{y}(x)\}$$

The uncertainty reward is:

$$r_{\text{uncertainty}}(x; \phi) = 1 - 2\left|\hat{p}(x; S_\phi) - \frac{1}{2}\right|$$

It maximizes when Solver accuracy is near 50% — incentivizing the Challenger to generate problems on which the Solver is **maximally uncertain**, neither too easy nor too hard (theoretically, $\hat{p} = 0.5$ maximizes the KL-divergence lower bound on learning potential; see Appendix F).

**Repetition Penalty:**

To encourage in-batch diversity, compute pairwise distance from BLEU score $d_{ij} = 1 - \text{BLEU}(x_i, x_j)$, agglomeratively cluster problems with distance below threshold $\tau_{\text{BLEU}} = 0.5$, and penalize proportionally to relative cluster size:

$$r_{\text{rep}}(x_i) = \lambda \frac{|C_k|}{B}$$

with $B$ = batch size, $\lambda = 1$.

**Format Check Penalty:** generated problems must contain `<question>` and `</question>` tags; otherwise reward = 0.

**Composite Reward:**

$$r_i = \max\left(0, r_{\text{uncertainty}}(x_i; \phi) - r_{\text{rep}}(x_i)\right)$$

The Challenger policy is updated with GRPO using this reward.

### 3.3 Phase 2: Solver Dataset Construction

Sample $N = 8000$ candidate problems from the updated Challenger; for each, sample $m = 10$ Solver answers, derive pseudo-label $\tilde{y}_i$ via majority vote, and compute empirical accuracy $\hat{p}_i$.

**Difficulty-based filtering:** keep only problems with $|\hat{p}_i - \frac{1}{2}| \leq \delta$ ($\delta = 0.25$), i.e., majority-vote agreement in 3–7. This filtering simultaneously:
- **calibrates the curriculum:** removes problems that are too easy or too hard
- **acts as implicit quality control:** low-consistency problems are usually ambiguous or have unreliable pseudo-labels

### 3.4 Phase 3: Solver Training

Train the Solver on the filtered dataset with GRPO using a simple binary verifiable reward:

$$r_j = \begin{cases} 1, & \text{if } x_j \text{ matches pseudo-label } \tilde{y}_i \\ 0, & \text{otherwise} \end{cases}$$

### 3.5 Training hyperparameters

| Component | Parameter | Value |
|------|------|-----|
| Solver | Global Batch Size | 128 |
| Solver | Learning Rate | $1 \times 10^{-6}$ |
| Solver | Weight Decay | $1 \times 10^{-2}$ |
| Solver | KL Penalty $\lambda_{KL}$ | $1 \times 10^{-2}$ |
| Solver | Max Steps | 15 |
| Solver | Number of Rollouts | 5 |
| Challenger | Global Batch Size | 128 |
| Challenger | Learning Rate | $1 \times 10^{-6}$ |
| Challenger | Max Steps | 5 |
| Challenger | Number of Rollouts | 4 |
| Shared | Rollout Temperature | 1.0 |
| Shared | Rollout Top-p | 0.99 |
| Shared | Precision | BFloat16 + FlashAttention2 |

---

## 4. Datasets

### Training data

| Domain | Dataset | Description |
|------|--------|------|
| Math | Self-generated (Zero Data) | Each round, the Challenger generates N=8000 candidate problems; a filtered subset is used for Solver training; no external training data |
| Math (synergy experiment) | math12k (hiyouga) | Used to compare synergy of R-Zero and supervised fine-tuning |

### Evaluation benchmarks

| Domain | Dataset | Description |
|------|--------|------|
| Math reasoning | AMC | American Mathematics Competition, report mean@32 |
| Math reasoning | Minerva | Math-reasoning benchmark, greedy-decoding accuracy |
| Math reasoning | MATH-500 | Hendrycks et al. math problems, greedy-decoding accuracy |
| Math reasoning | GSM8K | Grade-school math word problems, greedy-decoding accuracy |
| Math reasoning | OlympiadBench | Olympiad-level bilingual multimodal science problems, greedy-decoding accuracy |
| Math reasoning | AIME-2024 | AIME 2024, report mean@32 |
| Math reasoning | AIME-2025 | AIME 2025, report mean@32 |
| General reasoning | MMLU-Pro | Enhanced Massive Multitask Language Understanding, exact-match accuracy |
| General reasoning | SuperGPQA | Graduate-level reasoning (285 subjects), exact-match accuracy |
| General reasoning | BBEH | BIG-Bench Extra Hard, more challenging complex reasoning, exact-match accuracy |

---

## 5. Evaluation metrics and main results
### Metrics

- **Math benchmarks:** AMC and AIME use **mean@32** (average over 32 samples); other math benchmarks use **greedy-decoding accuracy**
- **General benchmarks:** **Exact Match (EM) accuracy**, greedy decoding
- Math answer correctness is judged by **GPT-4o as judge** (temperature = 0.1)

### Main results

#### Math reasoning (Table 1, Step 45)

| Model | Avg | AMC | Minerva | MATH | GSM8K | Olympiad | AIME25 | AIME24 |
|------|-----|-----|---------|------|-------|----------|--------|--------|
| Qwen3-4B Base | 42.57 | 45.70 | 38.24 | 68.20 | 87.79 | 41.04 | 10.30 | 6.70 |
| Qwen3-4B + Absolute Zero | 46.42 | 52.45 | 41.96 | 76.20 | 89.34 | 42.56 | 10.20 | 12.20 |
| **Qwen3-4B + R-Zero** | **49.93** | **57.27** | **52.94** | **79.60** | **92.12** | **44.59** | 9.60 | **13.40** |
| Qwen3-8B Base | 48.64 | 51.95 | 50.00 | 78.00 | 89.08 | 44.74 | 12.10 | 14.60 |
| **Qwen3-8B + R-Zero** | **53.72** | **61.67** | **60.66** | **82.00** | **94.09** | **48.89** | **13.30** | **15.40** |
| OctoThinker-3B Base | 26.64 | 17.19 | 24.26 | 55.00 | 73.69 | 16.15 | 0.21 | 0.00 |
| **OctoThinker-3B + R-Zero** | **29.32** | **27.03** | **27.57** | 54.20 | **74.98** | **18.22** | **3.23** | 0.00 |
| OctoThinker-8B Base | 36.41 | 32.11 | 41.91 | 65.20 | 86.96 | 26.52 | 1.56 | 0.62 |
| **OctoThinker-8B + R-Zero** | **38.52** | **34.03** | **48.22** | **68.80** | **87.19** | **27.56** | 0.42 | **3.44** |

#### General reasoning (Table 2)

| Model | OverallAvg | SuperGPQA | MMLU-Pro | BBEH |
|------|-----------|-----------|----------|------|
| Qwen3-4B Base | 26.34 | 20.88 | 50.58 | 7.57 |
| **Qwen3-4B + R-Zero** | **31.15** | **27.55** | **55.47** | **10.42** |
| Qwen3-8B Base | 31.98 | 28.33 | 58.97 | 8.63 |
| **Qwen3-8B + R-Zero** | **34.50** | **31.38** | **61.53** | **10.60** |
| OctoThinker-3B Base | 7.47 | 10.09 | 10.87 | 1.46 |
| **OctoThinker-3B + R-Zero** | **11.12** | **12.44** | **16.71** | 4.20 |
| OctoThinker-8B Base | 11.70 | 13.26 | 20.21 | 1.64 |
| **OctoThinker-8B + R-Zero** | **23.00** | **19.82** | **40.92** | **8.25** |

#### Synergy with supervised fine-tuning (Table 4, Qwen3-4B)

| Training data | AMC | Minerva | MATH | GSM8K | Olympiad | AIME25 | AIME24 | SuperGPQA | MMLU-Pro | BBEH |
|----------|-----|---------|------|-------|----------|--------|--------|-----------|----------|------|
| Human only | 57.97 | 55.15 | 80.8 | 92.04 | 48 | 9.58 | 10.31 | 29.49 | 57.03 | 9.71 |
| R-Zero only | 57.27 | 52.94 | 79.6 | 92.12 | 44.59 | 9.6 | 13.4 | 27.55 | 55.47 | 10.42 |
| R-Zero + Human | 57.9 | 53.2 | 81.2 | 92.2 | 47.7 | 10.32 | 14.67 | 29.4 | 58.2 | 11.8 |

Using R-Zero as mid-training and then fine-tuning on human-annotated data yields Qwen3-4B **+2.35** average and Qwen3-8B **+3.69** (Figure 4), beating either single data source.

### Key findings

1. **Model-agnostic effectiveness:** R-Zero works across Qwen3 (0.6B/1.7B/4B/8B) and OctoThinker (3B/8B) architectures and sizes, with math-reasoning average gains of +2.68 to +7.36 points.

2. **Cross-domain transfer of reasoning:** trained only on math problems, R-Zero also markedly improves general reasoning benchmarks (e.g., Qwen3-4B general avg +4.81, OctoThinker-8B +11.30), indicating the method strengthens underlying reasoning rather than domain-specific knowledge.

3. **Key role of Challenger RL training:** the RL-trained Challenger clearly beats the untrained base Challenger. For Qwen3-4B, the first iteration alone brings +3.7 points (vs. +2.44 for base Challenger).

4. **Iteration scaling and performance collapse:** models keep improving early but **eventually degrade**. Larger models collapse later: 0.6B degrades after Step 15; 4B only after Step 45 — revealing inherent instability in self-evolving frameworks.

5. **Pseudo-label quality drop:** as iterations proceed, problems become harder (Step 15 Solver accuracy 59.0% → Step 45 47.0%), and pseudo-label accuracy systematically drops from 79.0% to 63.0% — a potential bottleneck.

6. **Necessity of separate models** (Table 6): the shared-parameter Single-R-Zero starts degrading after the first iteration (peak = 47.31), while two-model R-Zero keeps improving for three rounds (peak = 49.12). Shared parameters cause lower pseudo-label accuracy (63.4% vs. 71.0% at Step 15), possibly due to model-internal bias-induced overconfidence.

7. **Ablation** (Table 3, Qwen3-4B): removing Repetition Penalty drops Math AVG from 49.07 to 45.76 (−3.31); removing Task Filtering drops General AVG from 31.15 to 26.69 (−4.46) — diversity and curriculum calibration are critical.

8. **Domain-general training** (Table 8): removing math-specific prompt constraints further improves the 8B model on both math and general benchmarks (e.g., AMC 65.53 vs. 61.67), indicating extensibility to broader reasoning tasks.

---

## Citation

```bibtex
@inproceedings{huang2026rzero,
  title={R-Zero: Self-Evolving Reasoning LLM from Zero Data},
  author={Huang, Chengsong and Yu, Wenhao and Wang, Xiaoyang and Zhang, Hongming and Li, Zongxia and Li, Ruosen and Huang, Jiaxin and Mi, Haitao and Yu, Dong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
