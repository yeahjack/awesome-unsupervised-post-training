# Meta-Rewarding: Meta-Rewarding Language Models

> **Added to Survey:** 2026-03-11

**Paper:** Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
**Authors:** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR & UC Berkeley & NYU)
**ArXiv:** arXiv:2407.19594
**Date:** 2024-07

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Meta-Rewarding | Pref. Opt. | training-time | Semantic |

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
- **Whether parameters / state accumulate:** Parameters accumulate across multiple self-rewarding iterations; even when role switching (actor / judge / meta-judge) is used, it remains in the same offline training loop.
- **Reset boundary:** The adaptation timing is therefore offline iterative bootstrapping, not deployment-time TTA.

## 1. UPT Assignment Rationale
**Family IV — Internal Evaluator Bootstrapping (meta-judge self-rewarding)**

This work is a direct extension of Self-Rewarding Language Models (Yuan et al., 2024c) and belongs to the **meta-judge self-rewarding** sub-family within "Internal Evaluator Bootstrapping". In the Self-Rewarding framework the same LLM plays both actor (generating responses) and judge (evaluating responses), and bootstraps itself through iterative DPO. However, the Self-Rewarding objective optimizes only actor response quality and **ignores improvement of judge ability**—without an improving judge, iterative training quickly saturates or starts exhibiting reward hacking.

Meta-Rewarding's central novelty is introducing a **third role—meta-judge**—that lets the model evaluate the quality of its own judgments and thereby build judge preference pairs to explicitly train judge ability. As a result every iteration improves both actor and judge simultaneously, forming a stronger self-bootstrapping loop. No extra human annotation is needed; all training signals (response, judgment, meta-judgment) are model-self-generated.

---

## 2. Problem Addressed

The Self-Rewarding framework has a key bottleneck:

1. **Judge saturation:** Self-Rewarding's DPO only optimizes the actor to generate better responses; judge ability is never explicitly trained or improved. As iterations progress, judge accuracy stalls and fails to provide useful discrimination signal for an ever-better actor, so training quickly saturates.
2. **Reward-hacking risk:** if judge ability does not improve, the actor may learn to "game" a static judge (e.g., by producing verbose responses for higher scores) rather than truly improving response quality.
3. **Length bias:** judges (and meta-judges) tend to prefer longer responses / judgments, so response length keeps inflating across iterations (length explosion), hurting length-controlled win rate.

Meta-Rewarding's remedies:
- introduce a **meta-judge** role to evaluate the quality of one's own judgments, build judge preference data, and DPO-train it jointly with actor preference data;
- add a **length-control mechanism** that prefers shorter-but-high-quality responses and judgments when selecting preference pairs.

---

## 3. Method

### 3.1 Overall Framework

In Meta-Rewarding's iterative pipeline, every iteration the model plays three roles:
1. **Actor:** generate a response from the prompt.
2. **Judge:** use LLM-as-a-Judge prompting to score the response pointwise (5-point scale).
3. **Meta-Judge:** use LLM-as-a-Meta-Judge prompting to pairwise-compare two judgments and pick the better one.

Each iteration produces two kinds of preference data:
- **Actor preference pairs:** chosen / rejected responses by judge score → train the actor.
- **Judge preference pairs:** chosen / rejected judgments by meta-judge comparison → train the judge.

The two preference sets are merged and used for joint DPO training, producing the next iteration's model.

### 3.2 Actor Preference Dataset Creation

#### Step 1: Sample responses
For a given prompt set, sample $K = 7$ responses per prompt with the current model (temperature 0.8, top-p 0.95), about 35,000 responses per round, deduplicated.

#### Step 2: Judge scoring
For each response, generate $N = 11$ judgments (using the pointwise 5-point rubric prompt); the judge outputs chain-of-thought reasoning and a final 1–5 score. Parse the score with regex, discard malformed judgments, and average the valid ones.

#### Step 3: Preference data selection with length control
Naively taking the highest-score $S_{\max}$ and lowest-score $S_{\min}$ responses as chosen / rejected causes length explosion (judges prefer long responses).

Meta-Rewarding introduces a **quality-tier parameter $\rho \in [0,1]$** to trade quality for length:
- **Chosen response $y_c$:** among responses in the top tier $[(1-\rho)S_{\max} + \rho S_{\min}, S_{\max}]$, pick the **shortest** response.
- **Rejected response $y_r$:** among responses in the bottom tier $[S_{\min}, (1-\rho)S_{\min} + \rho S_{\max}]$, pick the **longest** response.
- $\rho = 0$ degenerates to pure score selection (no length control).

### 3.3 Judge Preference Dataset Creation

This is the core innovation over Self-Rewarding.

#### Step 1: Response selection (pick the most uncertain response)
For every prompt's responses, compute the **variance** of its $N$ judgment scores and pick the highest-variance response $y$—the judge's most uncertain response—for judge training (ties broken randomly).

#### Step 2: Pairwise meta-judge evaluations
For the chosen response $y$, take its $N$ judgments $\{j_1, ..., j_N\}$ and pairwise-compare every distinct judgment pair $(j_m, j_n)$ with the Meta-Judge prompt (Figure 2). The Meta-Judge prompt contains:
- the original prompt $x$;
- the response $y$;
- two judgments (Judgment A and Judgment B);
- the scoring rubric.

The model outputs chain-of-thought reasoning + a winner selection.

**Positional-bias mitigation:** to dampen the meta-judge's tendency to favor the first-position judgment, evaluate every judgment pair twice (positions swapped). Use a weighted Elo score:

$$\text{win\_score}(j) = w_1 \cdot \text{win}_{1\text{st}}(j) + w_2 \cdot \text{win}_{2\text{nd}}(j)$$

with $w_1 < w_2$, giving more weight to second-position wins to compensate for the first-position advantage.

#### Step 3: Preference-pair selection
Pick the highest-Elo judgment as chosen $j_c$ and the lowest as rejected $j_r$.

**Judge length-bias filter:** the meta-judge also has length bias (prefers wordier judgments), so add a filter that drops pairs whose chosen judgment exceeds a length threshold.

### 3.4 Training Pipeline

Using Llama-3-8B-Instruct as the seed model:

1. **SFT on EFT:** first SFT on the Evaluation Fine-Tuning (EFT) set (ranked human responses from OpenAssistant) to initialize the model's LLM-as-a-Judge ability.

2. **Iteration 1:** generate actor + judge preference pairs from the SFT model → DPO from SFT → $M_1$.
   - $\rho = 0$ (no length control); filter chosen responses longer than 2500 characters.
   - Judge data: filter chosen judgments longer than 1100.

3. **Iteration 2:** generate actor + judge pairs from $M_1$ → DPO-train $M_1$ → $M_2$.
   - $\rho = 0.32$; judge-data threshold 1000.

4. **Iteration 3:** generate **actor-only** preference pairs from $M_2$ → DPO-train $M_2$ → $M_3$.
   - $\rho = 0.32$ (actor-only; no more judge training).

5. **Iteration 4:** generate **actor-only** preference pairs from $M_3$ → DPO-train $M_3$ → $M_4$.
   - $\rho = 0.4$.

Key observation: **only the first two iterations use the meta-judge for judge training**, while the last two run actor-only. The reason is that the meta-judge develops severe score bias and positional bias in later iterations (Table 5), and judge-training quality degrades accordingly.

### 3.5 DPO Training Details
- Learning rate $5 \times 10^{-6}$, $\beta = 0.1$, global batch size 32.
- 10 epochs, cosine learning-rate schedule.
- Sample 5,000 prompts per round from 20,000 seed prompts.
- $K = 7$ responses per prompt, $N = 11$ judgments per response.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Training prompts | 20K prompts generated by Llama-2-70B-Chat | 8-shot generation; 5,000 sampled per round |
| SFT | EFT (Evaluation Fine-Tuning) dataset | ranked human responses from OpenAssistant; initializes LLM-as-a-Judge ability |
| Actor evaluation | AlpacaEval 2 (805 prompts) | LC win rate vs. GPT-4-Turbo |
| Actor evaluation | Arena-Hard | challenging questions, highest correlation with Chatbot Arena |
| Actor evaluation | MT-Bench (8 categories) | multi-turn dialogue evaluation |
| Judge evaluation | OpenAssistant test set (190 samples, 580 responses) | Spearman correlation and agreement with human ranking |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

**Actor evaluation:**
- **AlpacaEval 2 LC win rate:** length-controlled win rate vs. GPT-4-Turbo (primary metric).
- **AlpacaEval 2 win rate:** raw win rate.
- **Arena-Hard score:** win rate vs. GPT-4 (95% CI).
- **MT-Bench score:** Turn 1 / Turn 2 scores.

**Judge evaluation:**
- **GPT-4 Chosen Pairs Agreement:** judge–GPT-4 preference agreement on human data.
- **Self-Chosen Pairs Agreement:** judge–GPT-4 agreement on self-generated data.
- **Agreement without Ties:** agreement after removing ties.
- **Human Agreement:** judge–human agreement and Spearman correlation against OpenAssistant human annotation.

### Main Results

#### AlpacaEval 2 (Table 1)
| Model | LC Win Rate | Win Rate | Length |
|-------|-------------|----------|--------|
| Llama-3-8B-Instruct (Seed) | 22.92% | 22.57% | 1899 |
| SFT on EFT | 25.47% | 25.10% | 1943 |
| Self-Rewarding + LC Iter 1 | 26.93% | 27.12% | 1983 |
| Self-Rewarding + LC Iter 2 | 30.38% | 29.77% | 1940 |
| Self-Rewarding + LC Iter 3 | 34.87% | 34.59% | 1967 |
| Self-Rewarding + LC Iter 4 | 35.49% | 35.37% | 2005 |
| **Meta-Rewarding Iter 1** | 27.85% | 27.62% | 1949 |
| **Meta-Rewarding Iter 2** | 32.66% | 33.29% | 2001 |
| **Meta-Rewarding Iter 3** | 35.45% | 37.24% | 2064 |
| **Meta-Rewarding Iter 4** | **39.44%** | **39.45%** | 2003 |

#### Arena-Hard (Table 2)
| Model | Score | 95% CI | Length |
|-------|-------|--------|--------|
| Llama-3-8B-Instruct (Seed) | 20.6% | (−2.0, 1.8) | 2485 |
| SFT on EFT | 24.2% | (−2.0, 1.8) | 2444 |
| Self-Rewarding + LC Iter 4 | 27.3% | (−2.0, 2.2) | 2448 |
| **Meta-Rewarding Iter 4** | **29.1%** | (−2.3, 2.1) | 2422 |

#### MT-Bench (Table 6, Appendix)
| Model | Score | Turn 1 | Turn 2 | Length |
|-------|-------|--------|--------|--------|
| Llama-3-8B-Instruct | 8.116 | 8.319 | 7.911 | 1568 |
| Self-Rewarding + LC Iter 4 | 8.028 | 8.381 | 7.675 | 1539 |
| **Meta-Rewarding Iter 3** | **8.341** | **8.731** | **7.950** | 1596 |
| Meta-Rewarding Iter 4 | 8.288 | 8.738 | 7.838 | 1592 |

#### Judge Agreement with GPT-4 (Table 3)
| Model | GPT-4 Chosen | GPT-4 w/o Tie | Self-Chosen | Self w/o Tie |
|-------|--------------|---------------|-------------|--------------|
| SFT on EFT | 51.48% | 51.79% | 61.66% | 73.51% |
| Self-Rewarding Iter 4 | 52.97% | 53.12% | 64.44% | 78.42% |
| **Meta-Rewarding Iter 3** | **58.63%** | **61.24%** | 63.43% | 76.80% |
| **Meta-Rewarding Iter 4** | 57.44% | 59.54% | **64.50%** | **79.33%** |

#### Judge Agreement with Human (Table 7, Appendix)
| Model | Agreement | Agree w/o Tie | Spearman corr. |
|-------|-----------|---------------|----------------|
| SFT on EFT | 63.20% | 64.59% | 0.321 |
| Self-Rewarding Iter 2 | 64.14% | 67.17% | 0.347 |
| **Meta-Rewarding Iter 2** | **66.64%** | **68.33%** | **0.382** |

### Key Findings

1. **Meta-Rewarding clearly beats Self-Rewarding:** on AlpacaEval 2, Meta-Rewarding reaches **39.44% LC win rate** after 4 iterations, while Self-Rewarding + LC under the same setting only reaches 35.49% (+3.95%); it even surpasses GPT-4-0314 and approaches Claude Opus level. Notable result for an 8B-parameter model with no extra human data.

2. **Meta-Rewarding also surpasses methods that use external reward models:** SPPO (which uses a reward model trained on large-scale human + GPT-4 data) reaches 38.77% LC win rate on AlpacaEval 2, still below Meta-Rewarding's 39.44%—even though the latter is purely self-improvement and does not rely on any external reward model.

3. **Judge ability really does improve:** Meta-Rewarding's GPT-4 Chosen Pairs Agreement w/o Tie rises from SFT's 51.79% to 61.24% (Iter 3); Self-Rewarding only reaches 55.90%. On human annotation, Meta-Rewarding Iter 2's Spearman correlation hits 0.382, clearly above Self-Rewarding's 0.347.

4. **Length control prevents length explosion:** response length stays stable across iterations (≈1949–2064 characters) without significant inflation. Ablation: with $\rho = 0$ (no length control), length inflates to 2212 characters and LC win rate drops.

5. **The meta-judge degrades in later iterations:** Table 5 shows that the Iteration-2 meta-judge develops severe score bias (97.68% pick the high-score judgment) and positional bias (68.11%), driving the judge-score distribution to cluster near 5 (mean rises from 4.1 to 4.7+); this is what limits further judge training in later iterations.

6. **Multi-turn ability is preserved:** despite training only on single-turn data, MT-Bench Turn 2 score in the final Meta-Rewarding iteration (7.838) drops only 0.073 vs. the seed (7.911), while Turn 1 rises from 8.319 to 8.738.

7. **An external reward model is not always better:** swapping self-judging for Starling-RM-34B as an external reward model gives only 24.63% LC win rate at iteration 1, below Meta-Rewarding's 27.85%—likely because of severer length bias in the external RM.

8. **17/18 AlpacaEval categories improve:** fine-grained analysis shows Meta-Rewarding gains most on Science, Gaming, Literature and other knowledge-/reasoning-heavy categories; Travel and Mathematics show the smallest gains.
