# DiSCTT: Consensus-Guided Self-Curriculum for Efficient Test-Time Adaptation in Reasoning

> **Added to survey on:** 2026-03-11

> **Paper metadata**
> - Authors: Mohammad Mahdi Moradi, Sudhir Mudur (Concordia University)
> - Date: March 6, 2026 (arXiv: 2603.05357)

| Property | Value |
|---|---|
| Method | DiSCTT (Difficulty-aware Consensus-Guided Self-Curriculum Test-Time Adaptation) |
| Carrier | Policy Opt. |
| Regime | Test-time |
| Level | Semantic |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | original test set plus synthesized variants |
| Persistence | full parameter accumulate across curriculum cycles |
| Inference Coupling | adapt on the evolving cohort, then re-infer in later cycles |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** updates run on the original test set plus its synthesized variants, on a curriculum-cycle schedule.
- **Serving the current sample or future ones:** updates in the current cycle mainly serve subsequent cycles and the subsequent evaluation rather than resetting after each sample.
- **Whether parameters/state accumulate:** parameters accumulate across curriculum cycles; the cohort evolves through reflection, reassignment, or resampling.
- **Reset boundary:** thus this is self-curriculum test-time adaptation rather than plain static test-cohort RL.

## 1. UPT Assignment Rationale

DiSCTT belongs to **Family III — Self-Generated Target Bootstrapping**, subclass **reasoning / plan / curriculum synthesis**. The core idea: at test time, use the consensus among the model's own multi-sample outputs to construct a **self-evolving curriculum**, dynamically splitting test-time inputs into "easy" and "hard" buckets, then applying SFT (with the majority-agreed solution as pseudo-label) and RL (with a consensus-regularized reward driving exploration) accordingly.

Satisfies the key UPT conditions:
- **No external supervision:** all training signals come from the model's own outputs (majority voting as pseudo-label / reward); no ground-truth labels or external verifier.
- **Self-generated-target driven:** the dominant artifact is a consensus-based synthetic curriculum that decides which samples receive SFT consolidation vs. RL exploration, and the curriculum evolves as the model's ability evolves.
- **Policy-optimization carrier:** updates the model via SFT and GRPO (Group Relative Policy Optimization) alternately.

---

## 2. Problem Addressed

Existing test-time adaptation methods have core flaws:

1. **Difficulty blindness:** TTRL, EVOL-RL, etc., apply a single optimization objective uniformly to all test-time inputs (uniform RL or uniform SFT), ignoring the inherent heterogeneity of reasoning problems. Easy problems get unnecessary high-variance RL updates; hard problems lack enough exploration.
2. **Training instability:** uniformly applying RL without labels introduces noisy gradients and causes performance fluctuations or even collapse (see Figure 2: AMC performance of the RL-only strategy).
3. **Compute waste:** applying costly RL updates uniformly (including easy samples the model already solves with high confidence) creates a lot of redundant computation.
4. **Token-level uncertainty doesn't fit reasoning:** traditional token-level entropy / confidence cannot accurately measure epistemic uncertainty in multi-step reasoning; errors usually only surface at the trajectory level.

Core insight of DiSCTT: **effective test-time adaptation should allocate different learning objectives based on instance-level epistemic uncertainty**, not one-size-fits-all.

---

## 3. Method

DiSCTT has three core modules: **Consensus-Based Difficulty Estimation**, **Dynamic Self-Curriculum Training**, and **Stabilized Label-Free RL with Structured Reward**.

### 3.1 Consensus-Based Difficulty Estimation

For each input $x_j$, sample $M$ independent reasoning completions $\{y_{j,1}, \ldots, y_{j,M}\}$ from the current policy $\pi_\theta$, each containing a reasoning trajectory $r_{j,i}$ and final answer $a_{j,i}$. Define the empirical agreement ratio:

$$c_j = \frac{1}{M} \max_a \sum_{i=1}^{M} \mathbf{1}[a_{j,i} = a]$$

Higher $c_j$ means stronger consistency on the problem (lower epistemic uncertainty). With fixed threshold $\rho$, split the dataset:

- $\mathcal{D}_{\text{easy}} = \{x_j \mid c_j \ge \rho\}$ (high consensus, easy subset)
- $\mathcal{D}_{\text{hard}} = \{x_j \mid c_j < \rho\}$ (low consensus, hard subset)

**Key:** this split is **temporary and policy-dependent**. Every $K$ training steps, the model re-samples, recomputes the agreement ratio, and re-splits, forming a **self-evolving curriculum** — problems can migrate between easy and hard, reflecting evolving ability.

### 3.2 Dynamic Self-Curriculum Training

Training alternates between SFT and RL stages:

- **SFT stage** (on $\mathcal{D}_{\text{easy}}$): supervised fine-tuning with the majority-agreed completion $y_j^*$ as pseudo-label:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x_j, y_j^*) \sim \mathcal{D}_{\text{easy}}} \left[ \log \pi_\theta(y_j^* \mid x_j) \right]$$
  Purpose: low-variance consolidation of correct reasoning patterns on high-confidence problems.

- **RL stage** (on $\mathcal{D}_{\text{hard}}$): GRPO policy optimization, encouraging structured exploration on low-consensus hard problems.

Each curriculum cycle contains 2 SFT epochs + 10 RL epochs, after which consensus is recomputed and the curriculum split is updated.

> **Figure 3** (DiSCTT framework overview): a closed-loop pipeline. On the left, the model samples multiple reasoning trajectories per input and performs Consensus Assessment; based on the threshold, inputs are routed to two paths — high-consensus inputs go through the SFT Update path (supervised fine-tuning with pseudo-labels), low-consensus inputs go through the RL Update path (RL training for policy exploration). On the right, a decision diamond "Step % K == 0?" triggers difficulty re-assessment every K steps, re-splitting easy/hard subsets and forming a self-evolving curriculum loop.

### 3.3 Reinforcement Learning Objective (structured reward)

For each completion $y_i = (r_i, a_i)$ in $\mathcal{D}_{\text{hard}}$, the reward is the product of three components:

$$R(y_i) = \mathbf{1}[a_i = a_{\text{maj}}(x)] \cdot (\alpha + \beta \cdot \text{JSD}_{\text{nov}}(r_i)) \cdot (\varepsilon + (1-\varepsilon) \cdot g_{\text{rel}}(r_i))$$

Logical order of the three components:

1. **Correctness gate:** $\mathbf{1}[a_i = a_{\text{maj}}(x)]$ — only trajectories whose final answer matches the majority answer get nonzero reward. The majority answer is a self-consistency-induced pseudo-label, no ground truth required.

2. **Population-relative novelty:** uses Jensen-Shannon divergence to measure deviation between the current reasoning trajectory's token-level distribution and that of the majority-correct trajectory population:
$$\text{JSD}_{\text{nov}}(r_i) = \frac{1}{|T_i|} \sum_{t \in T_i} \text{JS}\left(p_\theta(\cdot|x, r_{i,<t}) \| \bar{p}_{\text{maj}}(\cdot|x, r_{<t})\right)$$
  where $\bar{p}_{\text{maj}}$ is the mean predictive distribution of all majority-correct trajectories. JSD is bounded and symmetric, avoiding KL instability. This encourages the model to explore novel reasoning strategies different from the mainstream while still giving the correct answer.

3. **Relevance-aware semantic gating:** uses a pretrained sentence encoder (BAAI/bge-base-en-v1.5) to compute semantic similarity between intermediate reasoning steps and the input prompt:
$$g_{\text{rel}}(r_i) = \frac{1}{n} \sum_{j=1}^n \text{clip}\left(\cos(e(s_{i,j}), e(x)), 0, 1\right)$$
  Modulates reward via the affine factor $\varepsilon + (1-\varepsilon) \cdot g_{\text{rel}}(r_i)$; lowering novelty reward when reasoning steps drift from the input semantics prevents spurious novelty (off-topic deviation); $\varepsilon > 0$ ensures the factor never zeroes out (avoiding dead gradients).

### 3.4 Overall algorithm (Algorithm 1)

```
Require: dataset D, policy πθ, # samples M, update interval K, consensus threshold ρ
1: initialize θ ← θ₀
2: for training step t = 1, 2, ... do
3:   if t mod K = 0 then
4:     for each input xⱼ ∈ D:
5:       sample M reasoning completions, compute consensus score cⱼ
6:     Deasy = {xⱼ : cⱼ ≥ ρ}, Dhard = D \ Deasy
7:   end if
8:   SFT: supervised fine-tuning on Deasy with the majority-agreed pseudo-label
9:   RL (GRPO): policy optimization on Dhard with the structured reward
10: end for
```

---

## 4. Datasets

The paper evaluates on **6 reasoning benchmarks** covering math reasoning, multi-hop QA, and knowledge reasoning:

| Dataset | Type | Description |
|---|---|---|
| **AMC** | Math reasoning | American Mathematics Competition problems |
| **MATH-500** | Math reasoning | 500-problem subset of MATH |
| **AIME-2024** | Math reasoning | AIME 2024 problems (very hard) |
| **GPQA** | Knowledge reasoning | Graduate-level science QA |
| **HotpotQA** | Multi-hop QA | Multi-hop reasoning QA dataset |
| **MMLU** | General knowledge reasoning | Massive Multitask Language Understanding |

Additional datasets for OOD-generalization evaluation: **ARC-Challenge**, **HumanEval**.

Experiments cover **6 models of different sizes and families**:
- Qwen-2.5-0.5B-Instruct, LLaMA-3.2-1B-Instruct, Qwen-3-1.7B-Base, LLaMA-3.2-3B-Instruct, Qwen-3-4B-Base, Qwen-2.5-7B-Instruct

---

## 5. Evaluation metrics and main results
### Metrics
- **Mean Accuracy ± Standard Deviation** (averaged over multiple independent runs)
- **Training FLOPs** and **Wall-Clock Time** (compute efficiency)
- **OOD Accuracy** (cross-domain generalization)

### 5.1 Main results (Table 1)

DiSCTT **consistently surpasses** Base, TTRL, and EVOL-RL across all 6 benchmarks and all model sizes with lower variance:

| Model | Method | AMC | MATH-500 | AIME-2024 | GPQA | HotpotQA | MMLU | Avg. |
|---|---|---|---|---|---|---|---|---|
| Qwen-2.5-7B-Instruct | Base | 39.6 | 58.8 | 10.7 | 27.3 | 51.2 | 76.2 | 43.9 |
| | TTRL | 51.1 | 74.2 | 20.3 | 28.8 | 62.1 | 76.4 | 52.1 |
| | EVOL-RL | 55.0 | 73.4 | 26.3 | 29.3 | 66.1 | 77.9 | 54.6 |
| | **DiSCTT** | **59.5** | **82.2** | **29.6** | **34.9** | **73.7** | **83.3** | **60.6** |
| Qwen-3-4B-Base | Base | 38.6 | 51.4 | 10.0 | 27.3 | 39.4 | 69.5 | 39.3 |
| | TTRL | 46.6 | 60.6 | 17.1 | 28.8 | 61.5 | 72.4 | 47.8 |
| | EVOL-RL | 51.8 | 64.6 | 20.4 | 30.3 | 66.3 | 74.5 | 51.3 |
| | **DiSCTT** | **57.0** | **75.2** | **23.7** | **38.4** | **72.9** | **81.3** | **58.1** |

DiSCTT averages about **+5 to +7 pts** over EVOL-RL, reaching 60.6% average on Qwen-2.5-7B-Instruct (vs. EVOL-RL 54.6%).

### 5.2 Curriculum-routing distribution (Table 2)

Different datasets naturally yield different SFT/RL ratios, confirming the adaptive nature of difficulty-aware routing:

| Dataset | SFT (%) | RL (%) |
|---|---|---|
| MMLU | 67.1 | 32.9 |
| HotpotQA | 58.3 | 41.7 |
| MATH-500 | 47.0 | 53.0 |
| AMC | 25.0 | 75.0 |
| GPQA | 28.8 | 71.2 |
| AIME-2024 | 3.3 | 96.7 |

Harder datasets (e.g., AIME-2024) route almost everything to the RL path, while easier ones (e.g., MMLU) consolidate most via SFT.

### 5.3 Compute efficiency (Table 3)

DiSCTT achieves higher accuracy while drastically reducing compute cost (**FLOPs reduced by 30–50%, wall-clock time by 40–49%**):

| Model | Dataset | DiSCTT FLOPs | TTRL FLOPs | Cost Ratio |
|---|---|---|---|---|
| LLaMA-3.2-1B | MMLU | 47.08×10¹⁸ | 86.44×10¹⁸ | 0.544 (↓45.6%) |
| Qwen-3-4B | MATH-500 | 22.46×10¹⁸ | 32.78×10¹⁸ | 0.683 (↓31.7%) |
| Qwen-2.5-7B | AMC | 8.50×10¹⁸ | 14.05×10¹⁸ | 0.573 (↓43.7%) |

Wall-clock time: e.g., LLaMA-3.2-1B on MMLU drops from TTRL's 241 hours to DiSCTT's 124 hours.

### 5.4 Difficulty-level analysis (Figure 5)

On MATH-500, by 5 difficulty levels, three training strategies are compared:
- **SFT-only:** improves Levels 1–3 but barely improves Levels 4–5; saturates or even degrades later.
- **RL-only (GRPO):** improves all levels but converges slowly with limited gains on hard levels.
- **DiSCTT:** **faster and stronger** improvement across all levels, especially on Levels 4–5 well above either single-strategy baseline — confirming curriculum-routing effectiveness.

### 5.5 Reward ablation (Table 4)

Incrementally adding reward components on Qwen-3-4B-Base, MATH-500:

| Config | Accuracy |
|---|---|
| Base model | 51.4% |
| + Correctness gate | 62.8% |
| + Population-relative novelty | 73.4% |
| + Relevance-aware semantic gating | **75.0%** |

Each component brings substantial gain, validating the reward design.

### 5.6 OOD generalization (Figure 4)

- Qwen-3-1.7B-Base trained on AMC, OOD results: ARC-Challenge +10.29, HumanEval +7.88, HotpotQA +29.14.
- LLaMA-3.2-3B-Instruct trained on MMLU (math excluded): ARC-Challenge +16.97, GPQA +17.68.

DiSCTT avoids catastrophic forgetting and **substantially improves** cross-domain generalization, confirming difficulty-aware routing prevents overfitting to test-time data.
