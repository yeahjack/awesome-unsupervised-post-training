# CREAM: Consistency Regularized Self-Rewarding Language Models

> **Added to Survey:** 2026-03-11

**Paper:** CREAM: Consistency Regularized Self-Rewarding Language Models
**Authors:** Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, Huaxiu Yao
**Affiliations:** University of North Carolina at Chapel Hill, Nanyang Technological University, National University of Singapore, Microsoft Research
**ArXiv:** 2412.05321
**Venue:** ICLR 2025
**Code:** https://github.com/Raibows/CREAM

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CREAM | Pref. Opt. | training-time | Semantic |

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
**Family IV — Internal Evaluator Bootstrapping → Preference Optimization → Self-Rewarding**

CREAM belongs to the Internal Evaluator Bootstrapping family; the model **simultaneously plays both the policy model and the reward model**, with no external annotation or external reward model. Specifically, it uses DPO's intrinsic reward (i.e. $\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$) to score and rank multiple self-generated responses, form preference pairs, and then run DPO for preference optimization, iterating the whole process. This is a textbook self-rewarding paradigm. On top of it CREAM adds **consistency regularization** that exploits ranking consistency across iterations to ease self-bias and overconfidence, but the core data-generation and optimization loop remains self-rewarding preference optimization.

---

## 2. Problem Addressed

Self-Rewarding Language Models (SRLMs) have several pain points:

1. **Rewarding-bias accumulation:** the same LLM acting as both policy and reward model lets reward bias accumulate over rounds, gradually eroding preference-data quality.
2. **Overconfident preference labeling:** when two responses are close in quality, SRLM is still forced into a preference judgment, producing noisy labels that hurt downstream DPO.
3. **Degradation on small (7B) models:** empirically SRLM may collapse on 7B-scale LLMs after several rounds (especially Llama-2); standard LLM-as-a-Judge prompting is infeasible at that scale.
4. **No reward-reliability signal:** the vanilla SRLM framework has no mechanism to measure the trustworthiness of preference labeling.

---

## 3. Method

### 3.1 Generalized Iterative Preference Fine-tuning Framework

CREAM first proposes a unified iterative-preference-fine-tuning framework. The optimization objective is:

$$L(\theta, z) = L_{\text{SFT}}(\theta; D_S) + \mathbb{E}_{x \sim D_U; y, y' \sim \pi_{\theta_t}(\cdot|x)} [L_{\text{DPO}}(\theta; y, y', x, z)]$$

where $z(y, y', x) \in \{0, 1\}$ is the preference-label function. The framework uses **two-step alternating optimization**:

- **Step 1 (Preference-labeling):** fix $\theta = \theta_t$ and use the intrinsic reward to set preference labels:
  $$z_{t+1}(y, y', x) = \mathbf{1}[\log \pi_{\theta_t}(y|x) - \log \pi_{\text{ref}}(y|x) \geq \log \pi_{\theta_t}(y'|x) - \log \pi_{\text{ref}}(y'|x)]$$
- **Step 2 (Learning):** fix $z_{t+1}$ and minimize the DPO loss to update $\theta_{t+1}$.

**Key design choice:** instead of LLM-as-a-Judge prompting (unreliable for 7B models), use DPO's **intrinsic reward** $r_\theta(x,y) \propto \log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$ for scoring and ranking.

### 3.2 Consistency Regularization

Core insight: when two responses are close in quality, different reward models often disagree on their ranking. Use this **cross-iteration ranking disagreement** as a confidence signal.

Add a regularizer to the objective:

$$L(\theta, z) = L_{\text{SFT}}(\theta; D_S) + \mathbb{E}[L_{\text{DPO}}(\theta; y, y', x, z) + \lambda L_{\text{Reg}}(\theta; y, y', x)]$$

The expectation of $L_{\text{Reg}}$ is equivalent to $2 \text{KL}(u(\cdot) \| P_\theta(\cdot))$, regularizing the preference distribution of close-quality response pairs toward uniform.

**Theorem 3.3** proves the regularization is equivalent to **soft-labeled DPO**:

$$L(\theta, z) = C_\lambda L_{\text{DPO}}(\pi_\theta, D_{\text{DPO}}) + (1 - C_\lambda) L_{\text{DPO}}(\pi_\theta, D_{\text{RDPO}})$$

where $C_\lambda = (1+\lambda)/(1+2\lambda)$ and $D_{\text{RDPO}}$ is the preference-flipped dataset. In implementation this equals label smoothing.

### 3.3 Adaptive Consistency-Rate Estimation

Use **Kendall's Tau coefficient** to measure ranking consistency between the current model $\theta_t$ and the previous-round model $\theta_{t-1}$ on the same response set:

$$\tau_j = \frac{2}{N(N-1)} \sum_{1 \leq i < i' \leq N} [\mathbf{1}[(J_{ij} - J_{i'j})(K_{ij} - K_{i'j}) > 0] - \mathbf{1}[(J_{ij} - J_{i'j})(K_{ij} - K_{i'j}) < 0]]$$

The consistency rate is $C = |D_U|^{-1} \sum_j (\tau_j + 1)/2$, where $J$ is the current-model ranking and $K$ is the previous-round ranking.

**Lemma 3.4** ties Kendall τ to the regularization parameter $\lambda$: $\mathbb{E}[\tau_j] = 1 - 2\lambda$, so $C_\lambda \approx (1 + \tau_j)/2$.

### 3.4 Full Algorithm (Algorithm 1)

1. **SFT stage:** fine-tune the initial model on seed SFT data $D_S$: $\theta_0 \to \theta_1$.
2. **Iterative preference training** (T rounds):
   - **Response sampling:** for each prompt $x_j \in D_U$, sample $N=5$ responses with $\pi_{\theta_t}$.
   - **Self-rewarding:** score with intrinsic reward $r_{ij} = \log \pi_{\theta_t}(y_{ij}|x_i) - \log \pi_{\theta_0}(y_{ij}|x_i)$ and obtain ranking $J$.
   - **Consistency estimation:** score the same set with $\theta_{t-1}$ to obtain ranking $K$; compute Kendall τ and consistency rate $C$.
   - **Preference-data construction:** pair the top- and bottom-scored responses to form $D_{\text{DPO}}$ and flip them to form $D_{\text{RDPO}}$.
   - **Consistency-regularized training:** update $\theta_{t+1}$ by minimizing $C \cdot L_{\text{DPO}}(\pi_{\theta_t}, D_{\text{DPO}}) + (1-C) \cdot L_{\text{DPO}}(\pi_{\theta_t}, D_{\text{RDPO}})$.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Instruction following | OpenAssistant | ≈3.4K human-annotated samples as seed SFT data $D_S$ |
| Scientific reasoning | ARC-Easy | AI2 Reasoning Challenge, easy science QA |
| Scientific reasoning | ARC-Challenge | AI2 Reasoning Challenge, hard science QA |
| Commonsense reasoning | OpenBookQA | open-book QA |
| Social commonsense | SIQA (Social IQa) | social-interaction reasoning |
| Math reasoning | GSM8K | grade-school math word problems |
| Reward evaluation | RewardBench | pairwise-ranking accuracy for reward models |

**Unlabeled prompt set $D_U$:** mixes OpenAssistant prompts with the train-split prompts of the five downstream tasks above (prompt only), totaling ≈21K prompts, evenly split across iterations.

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **Exact-match accuracy:** exact-match accuracy on downstream tasks.
- **Ranking consistency:** Consistency Rate $C$, Kendall τ, Spearman correlation, TopOrder (stability of best / worst response ranks).
- **Ranking accuracy:** pairwise-ranking accuracy on RewardBench and curated preference data.
- **Alignment-Arena win rate:** head-to-head win rate judged by GPT-4o.

### Main Results

**Downstream-task performance (Llama-3, M3 iteration):**

| Method | ARC-Easy | ARC-Challenge | OpenBookQA | SIQA | GSM8K | Average |
|--------|----------|---------------|------------|------|-------|---------|
| Initial (M0) | 86.29 | 80.37 | 86.00 | 68.58 | 78.01 | 79.85 |
| SFT (M1) | 86.78 | 80.14 | 86.40 | 69.50 | 78.39 | 80.24 |
| SRLM (M3) | 87.17 | 81.23 | 87.30 | 70.37 | 77.48 | 80.71 |
| Oracle (M3) | 89.31 | 81.31 | 90.20 | 73.75 | 76.04 | 82.12 |
| **CREAM (M3)** | **89.52** | **83.36** | **90.20** | **72.06** | **81.73** | **83.37** |

**Downstream-task performance (Llama-2, M3 iteration):**

| Method | ARC-Easy | ARC-Challenge | OpenBookQA | SIQA | GSM8K | Average |
|--------|----------|---------------|------------|------|-------|---------|
| Initial (M0) | 61.07 | 48.98 | 62.20 | 50.36 | 23.65 | 49.25 |
| SRLM (M3) | 46.55 | 34.47 | 49.20 | 48.06 | 22.14 | 40.08 |
| **CREAM (M3)** | **62.08** | **48.81** | **64.60** | **51.22** | **25.85** | **50.51** |

**Ranking consistency (Llama-3, M3 vs M2):**

| Method | Consistency C↑ | Kendall τ↑ | Spearman↑ | TopOrder↑ |
|--------|----------------|------------|-----------|-----------|
| SRLM | 0.46±0.19 | −0.08±0.38 | 0.50±0.22 | 0.12±0.33 |
| CREAM | 0.92±0.09 | 0.84±0.19 | 0.95±0.07 | 0.59±0.49 |

**Alignment Arena (GPT-4o judged):**
- CREAM M3 vs SRLM M3: Win 34% / Tie 45% / Lose 21%.
- CREAM M3 vs Oracle M3: Win 11% / Tie 64% / Lose 25%.

### Key Findings

1. **Vanilla SRLM struggles on 7B models:** especially Llama-2 collapses badly across iterations (average drops from 49.35 to 40.08), indicating severe bias accumulation in small-model self-rewarding.
2. **CREAM clearly beats SRLM:** on Llama-3, CREAM M3 (83.37) even surpasses Oracle (82.12), which uses an external reward model; on Llama-2, CREAM is the only self-rewarding variant that keeps improving across iterations.
3. **CREAM keeps improving across iterations:** CREAM gains in every round (M1→M2→M3); on Llama-3 all five tasks improve from M2→M3, whereas SRLM frequently regresses from M2→M3.
4. **Ranking consistency rises sharply:** CREAM's Kendall τ jumps from SRLM's −0.08 to 0.84 (M3 vs M2), showing the regularization effectively stabilizes reward ranking.
5. **DPO rewarding beats prompt rewarding:** for 7B models, intrinsic-reward-based DPO ranking clearly beats LLM-as-a-Judge prompting (the latter starts regressing already at M1→M2).
6. **Adaptive consistency rate beats manual tuning:** the Kendall-τ-driven $C$ outperforms the CREAM w/o RC variant that hand-searches a fixed value, and avoids hyperparameter-search cost.
7. **Kendall τ is the best consistency metric:** compared with Spearman and TopOrder, Kendall τ wins on most datasets and is theoretically more robust.
