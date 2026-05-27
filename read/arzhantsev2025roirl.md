# RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative RL

> **Added to survey on:** 2026-03-11

**Paper:** RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning
**Authors:** Aleksei Arzhantsev (Criteo AI Lab, École Polytechnique), Otmane Sakhi (Criteo AI Lab), Flavian Vasile (Criteo AI Lab)
**ArXiv:** 2510.02891
**Date:** 2025-10 (NeurIPS 2025 Workshop: Efficient Reasoning)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RoiRL | Policy Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / offline self-evaluation batch |
| Persistence | full parameter accumulate across offline RL iterations |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates fire in an offline iterative RL pipeline; the core is to first collect self-evaluated data and then perform an offline policy improvement.
- **Whose sample it serves:** the current batch's update mainly serves the next offline iteration and the final deployed model, not the current sample's immediate inference.
- **Whether parameters/state accumulate:** parameters accumulate across multiple rounds of offline RL; no per-sample reset.
- **Reset boundary:** so it is offline iterative evaluator-driven training, not online TTA.

## 1. UPT Assignment Rationale
**Family II — Sample-Relation Supervision (population consensus)**

> **Reclassification note (2026-05-19):** previously placed in Family IV. Re-examined under the dominant-artifact rule: RoiRL's reward formula $r=\mathbf{1}[y=\mathrm{maj}]$ is identical in structure to TTRL's; the dominant artifact is the majority-vote statistic across rollouts, not the explicit output of an evaluator/scorer. The paper's "self-estimated reward / evaluator-driven RL" framing is a narrative choice; at the reward-formula level, there is no formal difference from TTRL. To keep family boundaries operate on the reward formula, RoiRL is now placed in Family II.

RoiRL uses **majority vote** as a self-generated reward signal (no ground-truth labels) and improves the LLM's reasoning ability via an offline iterative RL loop. The model itself produces several candidate solutions; the majority vote estimates correctness and serves as the reward $r=\mathbf{1}[y=\mathrm{maj}]$; weighted log-likelihood objectives then drive policy optimization. This reward formula is directly homologous to consensus-driven methods such as TTRL; it is not Family IV (CoNL, RLME, Meta-TTRL), where the reward comes from an explicitly constructed judge/introspector rather than directly from an agreement statistic.

---

## 2. Problem Addressed

1. **The bottleneck of RL training relying on ground-truth rewards:** traditional RL methods (e.g. GRPO) need correctness labels as the reward signal; large-scale, high-quality annotations are expensive and often unavailable.
2. **TTRL's compute cost:** Test-Time Reinforcement Learning (TTRL) replaces ground truth with majority votes, but its online RL setup must keep a reference model and repeatedly sample long CoTs and compute logits during training, saturating GPU memory and limiting scalability.
3. **TTRL's instability:** TTRL is highly sensitive to hyperparameter choices, and online updates introduce training instability.
4. **Core goal:** under a fully label-free condition, achieve self-improvement of LLM reasoning with a **simple, stable, efficient** offline method that can theoretically target the same optimal policy as TTRL.

---

## 3. Method

### 3.1 Problem formulation

Given a base LLM $\pi_0 = \pi_{\theta_0}$ and a prompt dataset $P_n = \{x_i\}_{i \in [n]}$ (**unlabeled**), the goal is to improve the model's reasoning in a self-supervised way.

TTRL's KL-regularized objective is:

$$\max_\theta \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_\theta(\cdot|x_i)} [\tilde{r}_k(y, x_i, \theta)] - \beta \text{KL}(\pi_\theta, \pi_0 | x_i)$$

where the majority-vote reward is $\tilde{r}_k(y, x_i, \theta) = \mathbf{1}[y = \tilde{y}_i^k(\theta)]$ and $\tilde{y}_i^k(\theta) = \text{maj}_{\ell \in [k]}(y_i^\ell)$. Note that the reward is **non-stationary** (it depends on the current policy), which makes the problem different from a simple majority-vote distillation.

### 3.2 The RoiRL algorithm

RoiRL (Reasoning with offline iterative Reinforcement Learning) alternates two steps at each iteration $m \geq 1$:

**Step 1 — Generation (offline data collection):** with the current policy $\pi_{m-1}$, sample $k$ candidate solutions $\{c_i^\ell, y_i^\ell\}_{\ell \in [k]}$ for each prompt $x_i$, score them with the majority-vote reward, and assemble the offline dataset:

$$D_{m-1} = \{x_i, \{c_i^\ell, y_i^\ell, \tilde{r}_k(y_i^\ell, x_i, \theta_{m-1})\}_{\ell \in [k]}\}_{i \in [n]}$$

**Step 2 — Offline update (weighted log-likelihood optimization):**

$$\theta_m = \arg\max_\theta \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_{m-1}(\cdot|x_i)} [g_m(\tilde{r}_k(y, x_i, \theta_{m-1})) \log \pi_\theta(c, y | x_i)]$$

where $g_m: \mathbb{R} \to \mathbb{R}$ is an increasing reward transform.

### 3.3 Reward-transform variants

The paper explores two reward transforms:

- **Dense reward $g_\beta$:** $g_\beta(r) = \exp(r / \beta)$ — emulates KL-regularized behavior; trained candidates are weighted exponentially.
- **Sparse reward $g_I$:** $g_I(r) = r$ (the identity) — equivalent to performing supervised fine-tuning only on candidates that match the majority vote.

### 3.4 Theoretical guarantee

**Proposition 3.1:** for any $\beta > 0$, there exist choices of reward transforms $(g_m)_{m \in \mathbb{N}}$ such that TTRL's Equation (1) and RoiRL's Algorithm 1 **share the same solution**.

Concretely, the analytic solution at iteration $m$ is:

$$\pi_m(c, y | x_i) \propto \prod_{j=1}^{m} g_j(\tilde{r}_k(y, x_i, \theta_{j-1})) \cdot \pi_0(c, y | x_i)$$

Choosing $g_j = g_\beta = \exp(r/\beta)$ recovers the closed-form solution of KL-regularized RL.

### 3.5 RoiRL's compute advantages

Compared with TTRL, RoiRL has the following compute advantages:

1. **No reference model needed:** $\pi_0$ does not have to be kept in memory, lowering memory cost.
2. **Better batching:** generation and training are strictly separated; questions can be batched together at generation time (which online RL cannot do).
3. **No need to store logits:** the reward does not depend on logits, so generation can use a larger batch size.
4. **Sparse reward speedup:** $g_I$ can substantially speed up training in early iterations when majority answers are sparse.

---

## 4. Datasets

| Domain | Dataset | Notes |
|------|--------|------|
| Math reasoning | MATH500 Train | 400 problems, **unlabeled** for training; ground truth used only for evaluation. |
| Math reasoning | MATH500 Test | 100 problems for evaluation. |
| Math reasoning | AMC | Competition math; evaluates generalization. |
| Math reasoning | AIME 2024 | High-difficulty competition math; evaluates generalization. |

**Base models (three different scales/architectures):**
- Qwen2.5-Math-1.5B.
- Phi-4-mini-reasoning-4B.
- Llama-3.2-3B-Instruct.

**Training setup:**
- $k = 10$ candidates per prompt.
- 3 epochs per round, up to 15 rounds (early stopping when maj₁₀ accuracy fails to improve for 5 consecutive rounds).
- TTRL uses GRPO with $\beta = 0.1$.
- RoiRL $g_\beta$ uses $\beta = 0.1$.

---

## 5. Evaluation metrics and main results

### Metrics

- **maj₁ (greedy decoding):** single-sample accuracy.
- **maj₁₀:** majority-vote accuracy over 10 samples.
- **maj₁₂₈:** majority-vote accuracy over 128 samples (a strong baseline).

### Main results

**Table 1: training without labels on MATH500 Train; evaluation on multiple datasets**

#### Qwen2.5-Math-1.5B

| Method | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.244 | 0.239 | 0.170 | 0.036 |
| TTRL | 0.307 | 0.298 | 0.214 | 0.026 |
| **RoiRL $g_I$** | **0.686** | **0.587** | **0.337** | **0.083** |
| RoiRL $g_\beta$ | 0.670 | 0.604 | 0.340 | 0.070 |

| Method | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.572 | 0.520 | 0.445 | 0.100 |
| TTRL | 0.625 | 0.560 | 0.469 | 0.066 |
| **RoiRL $g_I$** | **0.712** | **0.690** ⋆ | **0.518** ⋆ | 0.133 |
| RoiRL $g_\beta$ | 0.685 | 0.650 | 0.469 | **0.200** |

(⋆ marks results that exceed the base model's maj₁₂₈ performance: 0.717/0.680/0.506/0.233.)

#### Phi-4-mini-reasoning-4B

| Method | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.210 | 0.160 | 0.071 | 0.000 |
| TTRL | 0.272 | 0.225 | 0.090 | 0.000 |
| **RoiRL $g_I$** | **0.660** ⋆ | **0.511** | **0.246** | **0.016** ⋆ |

| Method | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.420 | 0.350 | 0.157 | 0.000 |
| TTRL | 0.483 | 0.460 | 0.193 | 0.000 |
| **RoiRL $g_I$** | **0.720** ⋆ | **0.680** ⋆ | **0.421** ⋆ | **0.067** ⋆ |

#### Llama-3.2-3B-Instruct

| Method | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.256 | 0.295 | 0.141 | 0.050 |
| TTRL | 0.361 | 0.394 | 0.159 | 0.043 |
| RoiRL $g_I$ | 0.395 | 0.376 | 0.198 | 0.060 |
| **RoiRL $g_\beta$** | **0.487** | 0.256 | 0.090 | 0.020 |

| Method | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.495 | 0.480 | 0.253 | 0.033 |
| TTRL | 0.510 | 0.490 | 0.313 | 0.167 |
| RoiRL $g_I$ | 0.508 | 0.520 | 0.313 | **0.200** ⋆ |
| RoiRL $g_\beta$ | 0.508 | **0.530** ⋆ | 0.229 | 0.100 |

### Training-speed comparison

| Method | Time per round | Relative speed |
|------|--------------|---------|
| RoiRL $g_I$ (sparse reward) | 6552.5s | **2.6× faster** |
| RoiRL $g_\beta$ (dense reward) | 8883.5s | 1.9× faster |
| TTRL | 17019.25s | 1× (baseline) |

(Tested on a single NVIDIA A100 80GB.)

### Key findings

1. **RoiRL beats TTRL in most settings:** across three base models and three evaluation sets, RoiRL consistently surpasses TTRL while training **2.5× faster or more**.

2. **$g_I$ (sparse / identity reward) is best overall:** although in theory $g_\beta$ targets TTRL's KL-regularized objective, the simpler $g_I$ (equivalent to SFT on majority-matching candidates) performs better in experiments, hinting that **KL regularization may not be optimal**.

3. **RoiRL is genuine self-improvement, not majority-vote distillation:**
   - The trained model's maj₁ can exceed the base model's maj₁₀.
   - The trained model's maj₁₀ can exceed the base model's **maj₁₂₈** (results marked ⋆).
   - This indicates the model learns more than majority-vote knowledge distillation — it gains **more general reasoning improvement**.

4. **Fast entropy decay aids convergence:** during RoiRL training, entropy drops quickly to near zero, while TTRL's entropy stays high. This explains RoiRL's faster convergence but also suggests more regularization (e.g. lower learning rate or alternative reward functions) may be needed to avoid premature collapse.

5. **Clever use of non-stationary rewards:** the majority-vote reward varies with the policy ($\tilde{r}_k$ depends on the current $\theta$), making each iteration's reward signal dynamically updated — avoiding the limits of pure distillation.

6. **Fully self-supervised loop without external labels:** the entire pipeline uses only unlabeled prompts; the reward is produced entirely by the model's own majority vote; ground-truth labels are used only for evaluation — true unsupervised reasoning improvement.
