# Self-Rewarding LM: Self-Rewarding Language Models

> **Added to Survey:** 2026-03-11

**Paper:** Self-Rewarding Language Models
**Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta & NYU)
**ArXiv:** arXiv:2401.10020
**Date:** 2024-01 (v3: 2025-03)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Self-Rewarding LM | Pref. Opt. | training-time | Semantic |

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

- **When does the update fire:** Updates fire inside a pre-deployment self-rewarding / judge-bootstrapping loop, which typically first generates responses and then generates judgments or preference pairs.
- **Serving the current sample or downstream samples:** The responses / judgments generated in the current batch primarily serve the next training round and the final deployed model, not the immediate inference of a single test sample.
- **Whether parameters / state accumulate:** Parameters accumulate across multiple self-rewarding iterations; even when the paper switches between actor / judge / meta-judge roles, it still belongs to the same offline training loop.
- **Reset boundary:** Therefore the adaptation timing is offline iterative bootstrapping, not deployment-time TTA.

## 1. UPT Assignment Rationale
**Family IV — Internal Evaluator Bootstrapping (self-rewarding)**

This paper is the foundational work of the **self-rewarding** sub-family within "Internal Evaluator Bootstrapping". Its core idea: instead of relying on an external, fixed reward model or human preference data to provide training signal, **let the same LLM simultaneously play both the instruction-following model and the reward model**. The model uses LLM-as-a-Judge prompting to score its own responses, builds chosen–rejected preference pairs, and bootstraps itself continuously inside an Iterative DPO framework.

Why this counts as unsupervised post-training (UPT):
- **No extra human annotation:** After the initial seed data, all new training data (prompt, response, reward) are generated and evaluated by the model itself.
- **Self-bootstrapping iterative improvement:** In each iteration the model improves not only its instruction-following ability but also its reward-modeling ability, forming a virtuous circle.
- **Breaking the human-feedback ceiling:** Standard RLHF is constrained by the scale and quality of human preference data and by the capability ceiling of a frozen reward model; Self-Rewarding pierces this ceiling by folding the reward model into the iterative training loop.

---

## 2. Problem Addressed

Standard LLM-alignment pipelines (RLHF / DPO) face two core bottlenecks:

1. **Human-feedback bottleneck:** The quality and scale of training signal are capped by the cost of human preference annotation and by annotator ability. Reaching superhuman agents requires superhuman feedback, which human annotation cannot deliver.
2. **Frozen reward-model bottleneck:** The standard RLHF pipeline trains an independent reward model and then freezes it; it no longer updates while the LLM trains. This pins the reward model's ceiling at the start of training, preventing it from co-evolving with the LLM.

The proposed remedy: **let the model itself be the reward model**, and use iterative training so that reward-modeling ability improves in lock-step with instruction-following ability, achieving self-alignment that can potentially break past the human-feedback ceiling.

---

## 3. Method

### 3.1 Overall Framework

Self-Rewarding Language Models endow the model with two abilities at once:
1. **Instruction Following:** generate high-quality answers conditioned on user prompts.
2. **Self-Instruction Creation:** generate new extended training samples and self-evaluate their quality.

The whole framework is an **iterative self-alignment** pipeline; every iteration contains a Self-Instruction Creation stage and an Instruction Following Training stage.

### 3.2 Initialization

The system needs two kinds of seed data:

- **IFT (Instruction Fine-Tuning) data:** human-annotated (instruction prompt, response) pairs used for SFT. Experiments use 3,200 high-quality English samples (rank 0) from Open Assistant.
- **EFT (Evaluation Fine-Tuning) data:** (evaluation instruction prompt, evaluation result response) pairs that train the model's LLM-as-a-Judge ability. Each item contains chain-of-thought justification and a final score (out of 5). Experiments build 1,630 train / 541 eval samples from Open Assistant.

Merging IFT and EFT data and training yields the initial model $M_1$.

### 3.3 Self-Instruction Creation

In every iteration the current model generates new training data:

1. **Generate new prompts:** with few-shot prompting (sampling 6 items from seed IFT + 2 model-generated items as demonstrations), apply Self-Instruct to generate new instruction prompts $x_i$.
2. **Generate candidate responses:** for each prompt $x_i$, sample $N=4$ candidate responses $\{y_i^1, \dots, y_i^N\}$ (temperature $T=0.7$, $p=0.9$).
3. **Self-evaluate:** use the same model's LLM-as-a-Judge ability to score each candidate $r_i^n \in [0, 5]$, averaging three evaluations per response.

### 3.4 LLM-as-a-Judge Prompt Design

Uses an **additive 5-point scoring system**, awarding cumulative points against five progressive criteria:
- **1 point:** the answer is relevant and provides some information.
- **2 points:** the answer covers the main parts of the user's question.
- **3 points:** the answer addresses the basic question elements and is useful.
- **4 points:** clearly written from an AI-Assistant viewpoint, well organized, comprehensive, helpful.
- **5 points:** a perfect AI-Assistant answer, no extraneous information, exhibits expert knowledge.

This additive prompt clearly beats multiple-choice-style prompts (e.g., Li et al. 2024): on the SFT Baseline, pairwise accuracy is 65.1% vs. only 26.6%.

### 3.5 Instruction Following Training (Iterative DPO)

Build **preference pairs** from the self-evaluation:
- Among the $N$ candidate responses for each prompt, pick the **highest-scored** as winning response $y^w$ and the **lowest-scored** as losing response $y^l$ (discard the pair if scores tie).
- Train with **DPO (Direct Preference Optimization)** on the preference pairs.

### 3.6 Model Sequence

$$M_0 \xrightarrow{\text{SFT on IFT+EFT}} M_1 \xrightarrow{\text{DPO on AIFT}(M_1)} M_2 \xrightarrow{\text{DPO on AIFT}(M_2)} M_3$$

- $M_0$: base pretrained LLM (Llama 2 70B).
- $M_1$: SFT-trained on IFT + EFT seed data.
- $M_2$: initialized from $M_1$, DPO-trained on $\text{AIFT}(M_1)$ data (3,964 preference pairs).
- $M_3$: initialized from $M_2$, DPO-trained on $\text{AIFT}(M_2)$ data (6,942 preference pairs).

Key property: in every round the reward model is the model itself, so the **reward model co-evolves with iterations**, in contrast to standard pipelines that use a fixed external reward model.

### 3.7 Training Hyperparameters

- **SFT:** learning rate 5.5e−6 (cosine decay to 1.1e−6), batch size 16, dropout 0.1.
- **DPO:** learning rate 1e−6 (decay to 1e−7), batch size 16, dropout 0.1, $\beta=0.1$.
- Save a checkpoint every 200 steps; early-stop on a 253-item validation set.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Seed IFT data | Open Assistant | 3,200 high-quality English first-turn samples (rank 0), for SFT |
| Seed EFT data | Open Assistant (multi-rank responses) | 1,630 train + 541 eval LLM-as-a-Judge samples |
| Synthetic preference data | AIFT($M_1$) | 3,964 preference pairs (model self-generated + self-evaluated) |
| Synthetic preference data | AIFT($M_2$) | 6,942 preference pairs (model self-generated + self-evaluated) |
| **Evaluation: Instruction Following** | | |
| Head-to-Head | IFT test data | 256 multi-source test prompts, GPT-4 pairwise evaluation |
| Leaderboard | AlpacaEval 2.0 | 805 prompts, GPT-4 win-rate vs. GPT-4 Turbo |
| Multi-turn | MT-Bench | multi-turn dialogue challenges (math, coding, roleplay, writing, …), GPT-4 scoring (out of 10) |
| NLP Benchmarks | ARC-Easy / ARC-Challenge / HellaSwag / SIQA / PIQA / GSM8K / MMLU / OBQA / NQ | general NLP capability-retention evaluation |
| **Evaluation: Reward Modeling** | Open Assistant (held-out) | on average 2.85 ranked responses per instruction, evaluating agreement with human rankings |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

**Instruction Following:**
- Head-to-Head win rate (GPT-4 judged, 256 prompts, scored in both orders, disagreements counted as ties).
- AlpacaEval 2.0 win rate (vs. GPT-4 Turbo, 805 prompts).
- MT-Bench score (out of 10, GPT-4 judged).
- Human evaluation (author-led blind eval, 50 prompts × 3 annotators, majority vote).
- NLP benchmark accuracy (9 standard NLP datasets).

**Reward Modeling:**
- Pairwise accuracy (pairwise consistency between model-induced ranking and human ranking).
- 5-best % (fraction of model-perfect-score (5) responses that humans also rank top).
- Exact-match % (fraction of full rankings identical to human ranking).
- Spearman correlation.
- Kendall's $\tau$ correlation.

### Main Results

#### Instruction-Following Ability Improves Across Iterations

**Head-to-Head vs. SFT Baseline (GPT-4 judged):**

| Comparison | Left Win | Tie | Right Win |
|------------|----------|-----|-----------|
| $M_1$ vs. SFT Baseline | 30.5% | 38.7% | 30.9% |
| $M_2$ vs. SFT Baseline | 49.2% | 36.3% | 14.5% |
| $M_3$ vs. SFT Baseline | **62.5%** | 27.7% | 9.8% |
| $M_2$ vs. $M_1$ | 55.5% | 32.8% | 11.7% |
| $M_3$ vs. $M_2$ | 47.7% | 39.8% | 12.5% |
| $M_3$ vs. $M_1$ | 68.8% | 22.7% | 8.6% |

**AlpacaEval 2.0 win rate (vs. GPT-4 Turbo):**

| Model | Win Rate |
|-------|----------|
| $M_1$ (Iteration 1) | 9.94% |
| $M_2$ (Iteration 2) | 15.38% |
| $M_3$ (Iteration 3) | **20.44%** |
| GPT-4 0314 | 22.07% |
| Claude 2 | 17.19% |
| Gemini Pro | 16.85% |
| GPT-4 0613 | 15.76% |
| LLaMA2 Chat 70B | 13.87% |

$M_3$ surpasses Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%).

**MT-Bench results (out of 10):**

| Model | Overall | Math/Code/Reasoning | Humanities/STEM/Roleplay/Writing |
|-------|---------|---------------------|----------------------------------|
| SFT Baseline | 6.85 | 3.93 | 8.60 |
| $M_1$ | 6.78 | 3.83 | 8.55 |
| $M_2$ | 7.01 | 4.05 | 8.79 |
| $M_3$ | **7.25** | **4.17** | **9.10** |

Among the fine-grained categories, Writing (8.83→9.58), Roleplay (8.15→8.73), Extraction (6.90→7.80), and STEM (9.18→9.45) show the largest gains.

**Human evaluation (50 prompts, 3 annotators per pair):**

| Comparison | Self-Rewarding Win | Tie | SFT Baseline Win |
|------------|--------------------|-----|------------------|
| $M_1$ vs. SFT | 28.0% | 26.0% | 46.0% |
| $M_2$ vs. SFT | 56.0% | 24.0% | 20.0% |
| $M_3$ vs. SFT | **66.0%** | 16.0% | 18.0% |

Human evaluation results agree with the GPT-4 automatic evaluation, confirming the effectiveness of iterative training.

#### Reward-Modeling Ability Improves Across Iterations

| Metric | SFT Baseline | $M_1$ (Iter 1) | $M_2$ (Iter 2) | $M_3$ (Iter 3) |
|--------|--------------|----------------|----------------|----------------|
| Pairwise Acc. ↑ | 65.1% | 78.7% | 80.4% | **81.7%** |
| 5-best % ↑ | 39.6% | 41.5% | **44.3%** | 43.2% |
| Exact Match % ↑ | 10.1% | 13.1% | **14.3%** | **14.3%** |
| Spearman Corr. ↑ | 0.253 | 0.279 | 0.331 | **0.349** |
| Kendall's $\tau$ ↑ | 0.233 | 0.253 | 0.315 | **0.324** |

Key finding: even though iterations add no further EFT data and the self-generated samples do not resemble LLM-as-a-Judge training samples, reward-modeling ability keeps improving.

#### NLP Benchmark Retention

| Model | ARC-C ↑ | HellaSwag ↑ | GSM8K ↑ | MMLU ↑ | NQ ↑ |
|-------|---------|-------------|---------|--------|------|
| Llama 2 | 57.40 | 85.30 | 56.80 | 68.90 | 25.30 |
| SFT Baseline | 55.97 | 85.17 | 50.72 | 69.76 | 34.35 |
| $M_1$ | 57.51 | 84.99 | 60.27 | 69.34 | 35.48 |
| $M_2$ | 54.51 | 84.27 | 59.29 | 69.31 | 33.07 |
| $M_3$ | 53.13 | 83.29 | 57.70 | 69.37 | 31.86 |

NLP-benchmark performance remains largely stable with mild drops (an "alignment tax"), but no serious degradation.

### Key Findings

1. **Two-axis joint improvement:** Self-Rewarding iterative training lifts both instruction-following and reward-modeling ability, forming a **virtuous circle**—a better reward model yields higher-quality preference data, which then trains an even better LLM.
2. **Importance of EFT data:** Adding EFT seed data is essential to the self-rewarding loop. Without EFT the model struggles to emit a stable scoring format, scores collapse toward 4, and usable training pairs become extremely sparse (only 541 / 429 pairs vs. the regular 3,964 / 6,942).
3. **Additive prompt advantage:** the additive 5-point scoring prompt dominates the multiple-choice variant (pairwise accuracy 65.1% vs. 26.6%).
4. **Preference optimization > positive-sample SFT:** adding only high-score positives to SFT is nearly useless (29% vs. 30% wins), while DPO preference pairs are clearly effective—preference signal is essential for the bootstrap loop.
5. **Uneven capability gains:** the largest gains land on writing / roleplay / extraction and other creative or synthetic tasks; math / coding / reasoning improve less—limited by the low share of such tasks in seed data.
6. **Response length growth:** $M_1$ averages 1,092 tokens → $M_2$ 1,552 tokens → $M_3$ 2,552 tokens; the model trends toward longer answers, which may partly drive the performance gain (and is a potential length bias).
7. **Possibility of surpassing the human-data ceiling:** because the reward model keeps improving during iterations, in principle one can reach performance beyond what an LLM / reward model trained only on the original human seed data could deliver, opening the door to self-improvement beyond human feedback.
