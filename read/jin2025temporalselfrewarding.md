# Temporal SRLM: Temporal Self-Rewarding Language Models

> **Added to Survey:** 2026-03-11

**Paper:** Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future
**Authors:** Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang
**Institutions:** Peking University, Tsinghua University, National University of Singapore, Southeast University, NC State University, University of Michigan, Beijing Institute of Technology, Central South University, Westlake University
**ArXiv:** 2025
**Date:** 2025

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Temporal SRLM | Pref. Opt. | training-time | Semantic |

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
**Family IV — Internal Evaluator Bootstrapping (temporal self-rewarding)**

Temporal Self-Rewarding sits in the preference-optimization branch of the Internal Evaluator Bootstrapping family, specifically as an improvement on the self-rewarding paradigm. Its core mechanism is the dual role of **generator** and **evaluator** (LLM-as-a-Judge): in iterative training the model generates responses itself, scores them with itself, builds preference pairs, and runs DPO. Its key novelty is **temporal decoupling**: anchor rejected responses with the past model ($M_0$) and lift chosen responses with a future model ($M_f$), keeping the self-rewarding signal alive without external reward models or human annotation. Conceptually it extends the self-rewarding framework along the time dimension by replacing same-model quality differences with past–future quality differences, preventing the bootstrap loop from collapsing as chosen and rejected drift together.

---

## 2. Problem Addressed

### Core problem: chosen–rejected quality-gap collapse in Self-Rewarding

In standard Self-Rewarding LMs, the same model generates chosen and rejected responses each round and scores them with itself. As iterations push capability up, a chain of issues emerges:

1. **Quality convergence:** as $M_i$ gets stronger, the gap between its best output (chosen $y_w$) and worst output (rejected $y_l$) shrinks. Empirically the chosen–rejected score gap on Llama 3.1-8B shrinks **9×** over 4 iterations.
2. **Representational collapse:** in latent space, the chosen and rejected hidden representations $h_w$ and $h_l$ converge ($\|h_w - h_l\| \to 0$) and cosine similarity keeps rising.
3. **Gradient vanishing:** Theorem 1 shows the DPO Directional Guidance term $\|\nabla_\theta \log \pi(y_w|x) - \nabla_\theta \log \pi(y_l|x)\| \le C \cdot \|h_w - h_l\|$. As representations converge this term vanishes, so $\hat r \to 0$, the Adaptive Weighting $(1 - \sigma(\hat r)) \to 0.5$, and the overall DPO gradient vanishes—training stalls.

---

## 3. Method

### 3.1 Overview

Temporal Self-Rewarding decouples chosen- and rejected-response generation across time steps via two phases:

- **Phase 1 — Anchored Rejection:** anchor rejected responses with the past model (initial $M_0$).
- **Phase 2 — Future-Guided Chosen:** lift chosen responses with a temporarily trained future model ($M_f$).

### 3.2 SFT Initialization

From the pretrained backbone $M_b$, supervised fine-tuning establishes the dual ability:

$$M_0 = \text{SFT}(M_b, \text{IFT} \cup \text{EFT})$$

IFT (Instruction Fine-Tuning) data trains response generation; EFT (Evaluation Fine-Tuning) data trains quality assessment (with judge explanations).

### 3.3 Phase 1: Anchored Rejection

For each prompt $p \in p_i$:

1. **Two-model generation:** sample $K=7$ responses each with the current $M_i$ and the initial $M_0$:
   - $r_i = \{r_i^1, \dots, r_i^K\}$ from $M_i$,
   - $r_0 = \{r_0^1, \dots, r_0^K\}$ from $M_0$.
2. **Score with current model:** $M_i$ scores all responses, yielding $s_i$ and $s_0$.
3. **Pick chosen:** the highest-scoring $M_i$ response: chosen $\leftarrow r_i^{\arg\max s_i}$.
4. **Pick rejected:** the lowest-scoring response from $M_0$ if $\min(s_0) < \min(s_i)$, else from $M_i$: rejected $\leftarrow r_0^{\arg\min s_0}$.
5. **Validity filter:** only add to $D_1$ if $s_{\text{chosen}} > s_{\text{rejected}}$.
6. **Train future model:**

$$M_f = \text{DPO}(M_i, D_1)$$

### 3.4 Phase 2: Future-Guided Chosen

For each prompt $p \in p_i$:

1. **Future-model generation:** sample $K$ responses with $M_f$: $r_f = \{r_f^1, \dots, r_f^K\}$.
2. **Score with current model:** $M_i$ scores them: $s_f$.
3. **Upgrade chosen:** if $\max(s_f) > \max(s_i)$, replace chosen with $r_f^{\arg\max s_f}$ (from the future model); otherwise keep $r_i^{\arg\max s_i}$.
4. **Reuse rejected:** keep the same prompt's rejected from Phase 1.
5. **Validity filter:** only add to $D_2$ if $s_{\text{chosen}} > s_{\text{rejected}}$.
6. **Train next-iteration model:**

$$M_{i+1} = \text{DPO}(M_i, D_2)$$

### 3.5 Theoretical Analysis

**Theorem 1 (bound on Directional Guidance):** let $\pi_\theta$ be a model that generates response $y$ via latent representation $h$. For any chosen–rejected pair $(y_w, y_l)$, the norm of the DPO directional-guidance term is bounded:

$$\|\nabla_\theta \log \pi_\theta^y(y_w|x) - \nabla_\theta \log \pi_\theta^y(y_l|x)\| \le C_{h_w, h_l} \cdot \|h_w - h_l\|$$

where $C_{h_w, h_l}$ is a finite constant in $h_w, h_l$ (the supremum of the gradient-function Jacobian on the segment $\{\lambda h_w + (1-\lambda) h_l\}$).

**Implication:** in vanilla Self-Rewarding, $\|h_w - h_l\| \to 0$ (representational collapse) and the theorem directly explains DPO-gradient vanishing. Temporal Self-Rewarding artificially keeps $\|h_w - h_l\|$ large by anchoring $y_l$ to $M_0$ (always lower-quality representations) and lifting $y_w$ to $M_f$ (higher-quality representations), keeping the gradient signal stable.

### 3.6 Iteration Efficiency

Temporal Self-Rewarding needs only **2 iterations** (iter0 + iter1) to surpass standard Self-Rewarding's **4 iterations** (iter0–iter3), even though each round trains an extra temporary future model $M_f$. Total compute is comparable to 5 iterations of Self-Rewarding (a fair comparison).

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Dialogue / instruction following | Open Assistant (oasst1 + oasst2) | English samples after dropping null-score, ≈5,000; for IFT seed |
| Multi-domain evaluation | UltraFeedback | scored multi-response data; drop top-2,000 highest-variance, take top-25,000 by score, merge into IFT |
| IFT seed | merged set | random 5,000 question–answer pairs from the merged Open Assistant + UltraFeedback |
| EFT seed | UltraFeedback high-variance subset | top-2,000 highest-variance, scored by GPT-4o; keep only the 1,871 whose model ranking matches the human ranking; includes judge explanations |
| Iterative-optimization data | remainder of merged set | the 20,000 remaining (after 5,000 IFT) split evenly into 5 chunks (4,000 each), one per round |
| Evaluation | AlpacaEval 2.0 | instruction following; pairwise vs. GPT-4 Preview |
| Evaluation | Arena-Hard-v0.1 | hard-prompt evaluation; pairwise vs. GPT-4-0314 |
| Evaluation | MT-Bench | multi-turn dialogue; GPT-4o direct scoring |
| OOD evaluation | ARC-Challenge, GSM8K, TruthfulQA, HumanEval | scientific reasoning, math reasoning, factual QA, code generation |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

| Metric | Benchmark | Description |
|--------|-----------|-------------|
| LC Win Rate (%) | AlpacaEval 2.0 | length-controlled win rate |
| Win Rate (%) | AlpacaEval 2.0 | raw win rate |
| Score (%) | Arena-Hard-v0.1 | win rate vs. GPT-4-0314 |
| 1st / 2nd / Avg | MT-Bench | turn-1 / turn-2 / average dialogue-quality scores |
| Accuracy | ARC, GSM8K, TruthfulQA, HumanEval | accuracy on each NLP benchmark |

### Main Results

#### Llama 3.1-8B main experiments (Table 1)

| Method | Best Iter | AlpacaEval LC Win | AlpacaEval Win | Arena-Hard | MT-Bench Avg |
|--------|-----------|-------------------|----------------|------------|--------------|
| SFT model | – | 8.73% | 5.96% | 6.3% | 4.81 |
| Rejection Sampling (best) | iter1 | 9.58% | 7.33% | 6.6% | 5.08 |
| SPIN (best) | iter3 | 8.93% | 6.83% | 4.7% | 5.11 |
| SPIN-Fair (best) | iter3 | 9.82% | 7.20% | 4.7% | 5.09 |
| Self-Rewarding (best) | iter3 | **19.92%** | **19.69%** | 9.4% | 5.74 |
| **Temporal SR (best)** | **iter1** | **27.94%†** | **29.44%†** | **14.6%†** | **5.89†** |

Temporal SR is best across all metrics and clearly beats 4-iter Self-Rewarding with only 2 iterations.

#### Cross-model-family generalization (Table 3)

| Model | Method | AlpacaEval LC Win | AlpacaEval Win | Arena-Hard | MT-Bench |
|-------|--------|-------------------|----------------|------------|----------|
| Llama 3.2-3B | SR best | 3.37% | 3.42% | 2.3% | 4.03 |
| Llama 3.2-3B | **TSR best** | **4.79%** | **8.20%** | **2.9%** | **4.32** |
| Llama 3.1-8B | SR best | 19.92% | 19.69% | 8.8% | 5.66 |
| Llama 3.1-8B | **TSR best** | **27.94%** | **29.44%** | **14.6%** | **5.89** |
| Llama 3.1-70B | SR best | 35.57% | 32.91% | 38.9% | 6.93 |
| Llama 3.1-70B | **TSR best** | **38.70%** | **33.66%** | **40.1%** | **6.98** |
| Qwen 2.5-7B | SR best | 21.53% | 18.14% | 21.5% | 6.09 |
| Qwen 2.5-7B | **TSR best** | **34.01%** | **35.90%** | **34.4%** | **6.29** |
| Mistral 7B | SR best | 25.48% | 27.58% | 12.8% | 5.68 |
| Mistral 7B | **TSR best** | **32.11%** | **35.16%** | **15.7%** | **5.76** |

#### Out-of-distribution NLP evaluation (Table 4, Llama 3.1-8B)

| Method | ARC | GSM8K | TruthfulQA | HumanEval |
|--------|-----|-------|------------|-----------|
| SFT | 0.531 | 0.530 | 0.505 | 0.220 |
| SR iter3 | 0.538 | 0.550 | 0.518 | 0.238 |
| **TSR iter1** | **0.549** | **0.563** | **0.544** | **0.262** |

### Key Findings

1. **Quantitative evidence of score-gap collapse:** in vanilla Self-Rewarding, Llama 3.1-8B's chosen–rejected score gap shrinks 9× over 4 iterations while the cosine similarity between chosen and rejected last-layer features keeps rising—confirming representational collapse.
2. **Temporal decoupling preserves the learning signal:** Temporal SR keeps the chosen–rejected quality gap stable across iterations and avoids the sharp decline seen in standard Self-Rewarding.
3. **Past component contributes more than the future component:** the ablation (Table 2) shows that using only the Past component (Anchored Rejection without Future-Guided Chosen) already beats the Self-Rewarding baseline on every metric. As iterations progress, scores of generated responses skew high, so anchoring rejected ones to the past sharpens chosen–rejected contrast more effectively. The future model adds complementary but secondary gains.
4. **Robustness to the judge model:** with Self-Judge, AutoJ-6B, or AutoJ-13B as judge, Temporal SR consistently beats Self-Rewarding—the advantage does not hinge on a specific judge. With a stronger external judge (AutoJ-13B), both methods improve markedly.
5. **Generalization across model families and scales:** consistent gains on Llama (3B / 8B / 70B), Qwen 2.5-7B, and Mistral 7B, with especially large relative gains on Qwen and Mistral (Qwen: 21.53% → 34.01% LC Win).
6. **Fewer iterations, better performance:** Temporal SR reaches a level that 4-iter Self-Rewarding cannot, with the compute of just 2 iterations (plus one auxiliary future-model training).
7. **OOD generalization:** even though training mainly targets instruction following, Temporal SR also lifts reasoning (ARC, GSM8K), factual QA (TruthfulQA), and code generation (HumanEval), suggesting the preference-learning improvement transfers broadly.
