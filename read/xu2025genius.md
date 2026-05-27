# GENIUS: A Generalizable and Purely Unsupervised Self-Training Framework for Advanced Reasoning

> **Added to survey on:** 2026-03-11

> **Method:** GENIUS | **Carrier:** Direct Opt. | **Regime:** Training-time | **Level:** Traj.
>
> arXiv 2504.08672 — Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu (Shanghai AI Lab, Xi'an Jiaotong University, HKU, PKU, HKUST)

| Property | Value |
|---|---|
| Method | GENIUS |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Traj. |

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

GENIUS belongs to **Family III — Self-Generated Target Bootstrapping**, subclass **reasoning / plan / curriculum synthesis**.

Core rationale: GENIUS is a **fully unsupervised** self-training framework that does not rely on any external ground-truth answer, reward model, or human annotation. Its pipeline: (1) for each unsupervised general query, the model uses **stepwise foresight re-sampling** to progressively search for better reasoning sequences, simulating future steps via rollouts and computing the foresight score (averaged log-probability) as a proxy for step value; (2) based on the foresight-score distribution, it performs sampling (exploration) and re-sampling (exploitation) to construct step-level preference pairs; (3) it then optimizes the policy with the self-generated preference pairs using an advantage-calibrated optimization (ACO) loss. The training targets (positive/negative trajectory pairs and their advantages) are entirely synthesized by the model itself — a textbook case of self-generated-target bootstrapping.

---

## 2. Problem Addressed

- **Scalability bottleneck of external supervision:** current post-training techniques (SFT, RL with outcome supervision, RL with reward model) all rely on annotated data or external reward models. Annotated data is largely confined to specific domains (math, code) and is expensive to collect; reward models require large-scale human labeling and are prone to reward hacking.
- **Limitations of existing self-rewarding methods:** response-level self-reward (e.g., sequence confidence, self-consistency) lacks fine-grained step-level guidance; step-level sampling methods (e.g., auto-regressive step-by-step sampling) are inherently **short-sighted** and cannot anticipate the global reasoning goal; MCTS-style search has global view but the backtracking cost is prohibitive.
- **Goal: improve general reasoning of LLMs using only unsupervised generic queries** without any external signal (Fig. 1(c)), unlocking the reasoning-scaling potential of massive generic data.

---

## 3. Method

### 3.1 Overall framework

GENIUS only needs a set of unsupervised NL queries plus the policy LLM $\pi_\theta$. The core pipeline has two objectives:

1. **Synthesize & Reward** (§2.2): generate high-quality preference pairs via stepwise foresight re-sampling.
2. **Optimize** (§2.3): robustly train the model using an advantage-calibrated optimization (ACO) loss.

> **Corresponding to Figure 2:** the framework runs a K-step sampling-and-rewarding loop (Steps 1–4) for each query, collects high-quality response sequences as training data (Step 5), and trains the model with ACO loss (Step 6).

### 3.2 Stepwise Foresight Re-Sampling

Uses beam search with beam size $M$. At each timestamp $k$:

#### (a) Step rollouts with foresight

Keep $M$ preceding paths $a_{<k}$ from timestamp $k-1$; for each beam, roll out $N$ candidate steps $a_k$, yielding $M \times N$ candidates. For each candidate, perform **foresight** — simulate future steps $a'_{>k}$ to obtain the complete response $T = (a_{<k}, a_k, a'_{>k})$, and use the averaged log-probability of the remaining steps as the foresight score $f_k$:

$$a'_{>k}, f_k \sim \pi_\theta(\cdot | a_{<k}; a_k)$$

Normalize the $M \times N$ scores into a distribution $F_k$:

$$F_k(i) = \frac{\exp(f_k^{(i)} / \tau)}{\sum_j \exp(f_k^{(j)} / \tau)}$$

where $\tau$ is the temperature.

#### (b) Re-sampling for exploration

Sample $M$ steps from $F_k$ to continue the beam search:

$$\{a_k^{(m)}\}_{m=1}^M \sim \text{Categorical}(F_k)$$

Each selected step's Q value equals its foresight score: $Q_k^{(m)} := f_k^{(m)}$.

#### (c) Re-sampling for exploitation

At each timestamp $k$, also build preference pairs via re-sampling:
- **Positive:** the response with the highest foresight score $T_k^w$ (score $f_k^w$).
- **Negative:** re-sample from $F_k \setminus f_k^w$ (the distribution with the positive removed) to obtain $T_k^l$ (score $f_k^l$).

#### (d) Advantage computation

Because steps come from different beams, foresight scores alone are insufficient — an advantage value is introduced:

$$A_k^w = f_k^w - Q_{k-1}^w, \quad A_k^l = f_k^l - Q_{k-1}^l$$

Each step produces a five-tuple training sample: $(x, T_k^w, A_k^w, T_k^l, A_k^l)$.

### 3.3 Advantage-Calibrated Optimization (ACO)

#### Self-Reward formulation

Following DPO, use the policy LLM as an implicit reward model with self-reward:

$$\phi(x, T) = \beta \log \frac{\pi_\theta(T|x)}{\pi_{\text{ref}}(T|x)}$$

#### ACO loss

In the unsupervised setting, preference pairs sampled by foresight score inevitably contain noise. For robustness, ACO adds a relaxation term $w(x, A)$ to the negative sample's self-reward:

$$\phi^l(x, T^l) = \beta \cdot w(x, A) \cdot \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}$$

$$w(x, A) = \text{clip}\left(\exp\left(-\frac{A^l - A^w}{\alpha}\right), 1\right)$$

- When $A^l - A^w \leq 0$ (negative is genuinely worse — **Normal Region**): $w = 1$, full penalty.
- When $A^l - A^w > 0$ (negative is actually better — **Calibration Region**): $w < 1$, reduced penalty.
- Hyperparameter $\alpha$ controls the decay rate.

> **Corresponding to Figure 3:** visualizes the calibration function $w(x, A)$ as a function of $A^l - A^w$, with different decay rates for $\alpha \in \{1, 2, 4, 8, \infty\}$.

Final ACO loss:

$$\mathcal{L}_{\text{ACO}} = -\mathbb{E}_{(x, T^w, T^l) \sim \mathcal{D}} \log \sigma\left(\beta \log \frac{\pi_\theta(T^w|x)}{\pi_{\text{ref}}(T^w|x)} - \beta \cdot \text{clip}\left(\exp\left(-\frac{A^l - A^w}{\alpha}\right), 1\right) \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}\right)$$

### 3.4 Sampling hyperparameters

- Beam size $M = 2$, candidates per beam $N = 4$, steps $K = 4$
- Total training pairs: Magpie 100K, OpenHermes2.5 128K
- Inference acceleration: vLLM engine

---

## 4. Datasets

### Training data

| Corpus | Source | Scale | Description |
|------|------|------|------|
| **Magpie** | Xu et al., 2024c | 25K queries | General NL queries, unlabeled |
| **OpenHermes-2.5** | Teknium, 2023 | 32K queries | General NL queries, unlabeled |

### Evaluation benchmarks

| Benchmark | Domain | Type |
|-----------|------|------|
| **GSM8K** | Math reasoning | Grade-school word problems |
| **MATH** | Math reasoning | High-school competition math |
| **GPQA** | Math reasoning | Graduate-level Q&A |
| **ReClor** | Logical reasoning | Reading comprehension + logic |
| **LogiQA** | Logical reasoning | Machine reading comprehension |
| **StrategyQA** | General reasoning | Implicit strategy reasoning |
| **ARC-Challenge** | General reasoning | AI2 reasoning challenge |
| **AlpacaEval** | General (subjective) | Instruction following |
| **WildBench** | General (subjective) | Real user tasks |
| **Arena-Hard** | General (subjective) | Human preference alignment |
| **WikiBench** | General (objective) | Community-driven AI eval |
| **MMLU** | General (objective) | Multi-task language understanding |
| **MMLU-Pro** | General (objective) | Enhanced MMLU |
| **AIME 2024** | Competition math | American Math Invitational |

---

## 5. Evaluation metrics and main results
### 5.1 Main experiment (Table 1: average of 7 reasoning benchmarks)

Base model: **LLaMA3.1-8B-Instruct**; CoT baseline averages 49.65%.

| Method | Supervised | GSM8K | MATH | ReClor | LogiQA | StrategyQA | GPQA | ARC-c | **Avg.** |
|------|------|-------|------|--------|--------|------------|------|-------|----------|
| LLaMA3.1-8B (CoT) | — | 70.28 | 30.52 | 49.40 | 33.33 | 58.91 | 26.56 | 78.33 | 49.65 |
| **Magpie 25K** | | | | | | | | | |
| SFT | Yes | 71.72 | 26.27 | 52.80 | 37.78 | 57.34 | 26.79 | 74.06 | 49.54 |
| SPIN | Yes | 74.91 | 31.49 | 57.40 | 40.09 | 71.35 | 29.91 | 83.96 | 55.59 |
| STaR | No | 72.86 | 29.32 | 46.40 | 35.94 | 33.36 | 20.31 | 67.24 | 43.63 |
| CoH | No | 74.37 | 32.29 | 56.20 | 38.56 | 69.08 | 28.13 | 82.51 | 54.45 |
| Self-Rewarding | No | 76.04 | 30.19 | 55.80 | 37.94 | 70.48 | 28.35 | 82.17 | 54.42 |
| ScPO | No | 71.11 | 30.99 | 55.00 | 40.40 | 59.87 | 28.57 | 78.92 | 52.12 |
| **GENIUS** | **No** | **78.32** | **34.64** | **58.80** | **40.86** | **72.53** | **30.35** | **84.04** | **57.08** |
| **OpenHermes 32K** | | | | | | | | | |
| SFT | Yes | 63.68 | 21.64 | 45.00 | 29.03 | 48.47 | 23.44 | 69.37 | 42.95 |
| SPIN | Yes | 63.61 | 24.74 | 54.00 | 35.33 | 59.00 | 28.57 | 71.76 | 48.14 |
| STaR | No | 75.51 | 29.47 | 43.60 | 34.87 | 19.34 | 22.99 | 68.43 | 42.03 |
| CoH | No | 74.29 | 31.22 | 54.80 | 38.40 | 69.91 | 29.69 | 81.48 | 54.26 |
| Self-Rewarding | No | 73.92 | 29.99 | 56.00 | 39.78 | 67.55 | 30.13 | 81.66 | 54.15 |
| ScPO | No | 73.54 | 31.27 | 54.80 | 41.01 | 58.65 | 28.79 | 79.52 | 52.51 |
| **GENIUS** | **No** | **75.82** | **34.42** | **57.60** | **41.63** | **70.79** | **34.82** | **83.19** | **56.90** |

**Key findings:**
- GENIUS improves LLaMA3.1-8B by **+7.43%** average on Magpie 25K, achieving SOTA among all unsupervised baselines (leading Self-Rewarding by >2%).
- On hard tasks like MATH, GENIUS leads Self-Rewarding by **>4%**.
- RL-based self-training methods generally beat SFT-style ones (SFT, STaR).

### 5.2 General-domain stability (Table 2)

| Benchmark | LLaMA3.1-8B | GENIUS (Magpie) | GENIUS (OpenHermes) |
|-----------|-------------|-----------------|---------------------|
| AlpacaEval | 24.60 | 26.96 | 25.47 |
| WildBench | -1.11 | 2.68 | 1.44 |
| Arena-Hard | 30.31 | **50.00** | **50.00** |
| WikiBench | 27.65 | 28.75 | 27.00 |
| MMLU | 71.14 | 71.86 | 72.21 |
| MMLU-Pro | 48.62 | 48.44 | 49.19 |

GENIUS maintains or slightly improves general benchmarks; Arena-Hard jumps by ~20 points, showing greatly improved human-preference alignment while avoiding catastrophic forgetting.

### 5.3 Cross-model generalization (Figure 4)

| Base model | GENIUS Avg. gain | Best baseline gain |
|----------|------------------|---------------|
| Qwen2.5-3B-Instruct | **+3.52%** | CoH +1.83% |
| Qwen2.5-7B-Instruct | **+2.16%** | ScPO +0.81% |

GENIUS also leads on the Qwen2.5 series, with smaller gains than on LLaMA3.1 (the authors hypothesize that Qwen2.5-Instruct has already received heavy post-training).

### 5.4 Competition-level tasks (Figure 5: AIME 2024)

On both LLaMA3.1-8B-Instruct and Qwen2.5-7B-Instruct, GENIUS improves Pass@1 by **+6.67%**, validating scalability to very hard scenarios.

### 5.5 Ablations

#### Sampling-strategy ablation (Table 3)

| Variant | Magpie Avg. | $\Delta$ | OpenHermes Avg. | $\Delta$ |
|---------|------------|----------|-----------------|----------|
| GENIUS (full) | 57.08 | — | 56.90 | — |
| w/o foresight | 53.91 | -3.17 | 53.65 | -3.25 |
| w/o sampling (greedy) | 52.98 | -4.10 | 53.80 | -3.10 |

- Removing foresight drops ~3%, confirming look-ahead simulation mitigates auto-regressive short-sightedness.
- Replacing sampling with greedy selection also degrades performance, confirming re-sampling balances exploration and exploitation.

#### Optimization ablation (Table 4)

| Loss | Magpie Avg. | OpenHermes Avg. |
|------|------------|-----------------|
| **ACO** | **57.08** | **56.90** |
| DPO | 55.51 | 55.73 |
| SimPO | 50.42 | 50.87 |
| IPO | 52.31 | 52.20 |
| ROPO | 55.30 | 55.25 |
| SFT | 44.63 | 49.70 |

ACO beats every other loss, with significant leads over DPO (+1.2–1.6%) and the robustness-oriented ROPO (+1.6–1.8%); SFT is the worst.

### 5.6 Post-training scaling law (Figure 6)

> **Corresponding to Figure 6:** plots scaling curves for different methods on LLaMA3.1-8B-Instruct vs. training steps. GENIUS's curve rises smoothly and is far from saturation, whereas CoH, Self-Rewarding, and ScPO plateau or decline with more steps.

This suggests GENIUS's self-training on general data has strong scalability potential and could further improve reasoning ability with more data and compute.
