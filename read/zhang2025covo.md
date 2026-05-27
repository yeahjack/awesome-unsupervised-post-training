# CoVo: Consistent Paths Lead to Truth — Self-Rewarding RL for LLM Reasoning

> **Added to survey on:** 2026-03-11

> **Method:** CoVo | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Traj.
>
> arXiv 2506.08745, Jun 2025
> Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao
> Zhejiang University, Alibaba Cloud Computing, Nanyang Technological University

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / rollout group |
| Persistence | full parameter accumulate across RL steps |
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

- **When the update is triggered:** updates fire during an offline RL stage before deployment; the basic unit is a group of rollouts per prompt batch.
- **Whose sample it serves:** the current rollout group's update serves subsequent training steps and the final deployed model, not the immediate inference of any single sample.
- **Whether parameters/state accumulate:** parameters accumulate across RL training; no per-sample reset.
- **Reset boundary:** it is an offline RL-style UPT schedule, not test-time arrival-by-arrival adaptation.

## 1. UPT Assignment Rationale

**Family II — Sample-Relation Supervision (trajectory-level consistency)**

CoVo relies on no external ground truth, human labels, or external reward model. The core signal comes from **relational comparisons** among multiple reasoning trajectories sampled from the same prompt:

- **Consistency:** how many intermediate reasoning states in a trajectory point — in model-likelihood terms — to its own final answer (rather than to other trajectories' answers).
- **Volatility:** how late in the trajectory the last intermediate state appears that deviates from the trajectory's own final answer; reflects the stability of the reasoning path.

Both features are extracted from a distance matrix produced by multiple trajectories — the distance matrix records the token-level log-probability distance from each intermediate state to every candidate final answer. The reward signal is essentially an **intrinsic statistic of relations between trajectories** (correct trajectories: high consistency + low volatility; wrong trajectories: low consistency + high volatility) — sample-relation supervision. CoVo also adds a curiosity reward (based on KL divergence between token probabilities and the uniform distribution) to encourage exploration of uncertain regions, again without external labels.

---

## 2. Problem Addressed

1. **External-supervision dependence:** current RL-for-reasoning methods rely heavily on ground-truth verifiable answers or pretrained reward models; in open-ended reasoning, label acquisition is expensive and reward models suffer from distributional mismatch.
2. **Existing self-rewarding methods focus only on final answers:** TTRL (majority voting), EMPO (answer-cluster probability), and similar methods only use the final-answer distribution, ignoring the rich information in intermediate reasoning states and being prone to reward hacking (e.g. always outputting "0" can game majority voting on math problems).
3. **Diversity collapse during training:** as RL training proceeds, sampling diversity drops and the self-rewarding signal degrades; an extra mechanism is needed to maintain exploration.

---

## 3. Method

### 3.1 Distance matrix and trajectory-pattern observations

Given a prompt x, sample N reasoning trajectories. Each trajectory consists of T intermediate reasoning states plus a final answer. Define the distance from state s_i to answer y as the negative average log-probability:

$$d(s_i, y) = -\frac{1}{|y|}\sum_{j=1}^{|y|}\log \pi_\theta(y[j] \mid s_i, y[:j])$$

Build a distance matrix **D** (T × K), where K is the number of distinct candidate answers. Empirical observation: intermediate states of correct trajectories quickly and stably point to their own final answer (high consistency, low volatility), while wrong trajectories show fluctuation and delayed convergence.

### 3.2 Consistency and volatility

- **Consistency:** $Con(\tau) = \frac{1}{T}\sum_{i=0}^{T-1}\mathbb{I}(\mathbf{D}[i,0] = \min_{0\le k<K}\mathbf{D}[i,k])$ — the fraction of intermediate states pointing to the trajectory's own answer.
- **Volatility:** $Vol(\tau) = \frac{1}{T}\max\{i \in [0,T-1] \mid \mathbf{D}[i,0] \neq \min_{0\le k<K}\mathbf{D}[i,k]\}$ — the relative position of the last state that deviates from the trajectory's own answer.

### 3.3 Intrinsic reward (vectorial aggregation)

Group trajectories with the same final answer (G trajectories per group); encode each trajectory as a 2D vector:

$$\mathbf{v}_i = Con(\tau_i) \cdot [\cos(Vol(\tau_i)),\; \sin(Vol(\tau_i))]$$

Sum the vectors within the group and take its magnitude as the group reward:

$$r_{\text{int}}^V = \frac{1}{G}\|V_{\text{group}}\|$$

Vectorial aggregation is more outlier-robust than linear aggregation $r_{\text{int}}^L = \frac{1}{G}\sum(Con - Vol)$ while preserving monotonicity.

### 3.4 Curiosity reward

To prevent diversity collapse, a curiosity reward encourages exploration of low-probability reasoning paths:

$$r_{\text{cur}} = d(s_i, s_{i+1}) - p_{\text{KL}}$$

where $p_{\text{KL}} = \ln[KL(P_{i+1}, \mathcal{U}) + 1]$ penalizes extremely low-probability tokens to prevent the curiosity reward from blowing up.

### 3.5 Total reward and optimization

$$r_{\text{covo}} = r_{\text{int}} + r_{\text{cur}}$$

The Reinforce++ algorithm is used for policy optimization (clipped surrogate objective + normalized advantage).

### 3.6 Theoretical analysis

- Prove that a majority-voting reward causes model collapse (Proposition 1).
- CoVo's optimization objective is equivalent to variational inference over latent reasoning trajectories (Proposition 2).
- Convergence upper bound $T' \lesssim \mathcal{O}(\pi_{\theta(0)}(y^\gamma|x)^{-1})$ (Proposition 3).

---

## 4. Datasets

### Training data
- **Open-Reasoner-Zero** training set (instructions only; no labels used).

### Evaluation data
| Category | Datasets |
|------|--------|
| Mathematical reasoning | GSM8K, MATH-500, Olympiad Bench, AMC-23 |
| Commonsense reasoning | MMLU-Pro, CommonsenseQA |
| Science reasoning | GPQA |

### Models
- Llama3.2-3B-Instruct.
- Qwen2.5-3B-Instruct.
- Qwen2.5-7B-Instruct.

---

## 5. Evaluation metrics and main results

### Metrics
- **Pass@1 accuracy** (sampling temperature = 0); Math-Verify is used to judge correctness.
- **Reasoning Diversity:** encoded with Sentence Transformers and visualized with UMAP.

### Main results (selections from Table 1)

**Llama3.2-3B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 51.8 | 79.2 | 25.0 | 18.9 | 36.8 | 31.5 | 74.2 |
| TTRL (self-rewarding) | 51.0 | 79.5 | 25.0 | 19.6 | 36.4 | 31.7 | 74.1 |
| **CoVo** | **51.2** | **79.6** | **25.0** | **19.6** | **37.2** | **32.1** | **74.4** |

**Qwen2.5-3B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 67.4 | 88.7 | 45.0 | 29.4 | 43.2 | 31.3 | 75.5 |
| TTRL (self-rewarding) | 67.8 | 88.6 | 45.0 | 29.5 | 43.2 | 31.3 | 75.5 |
| **CoVo** | **68.2** | **88.7** | **47.5** | **29.6** | **43.5** | **31.7** | **75.7** |

**Qwen2.5-7B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 78.2 | 92.3 | 57.5 | 41.3 | 57.1 | 36.6 | 82.9 |
| Reinforce++ (supervised) | 78.2 | 92.6 | 60.0 | 40.9 | 57.2 | 36.6 | 82.5 |
| **CoVo** | **78.4** | **92.5** | **60.0** | **40.8** | **57.0** | **36.8** | **82.8** |

### Key findings

1. **Comparable to or even surpasses supervised RL:** CoVo, using no external labels, matches or exceeds GRPO, RLOO, and Reinforce++ — methods that use ground-truth reward — on most benchmarks.
2. **Cross-domain generalization:** training data come only from math, yet CoVo achieves comparable gains on commonsense (MMLU-Pro, CommonsenseQA) and science (GPQA) benchmarks.
3. **Better reasoning diversity:** compared with GRPO's concentrated reasoning-path distribution, CoVo's reasoning paths are more spread out and diverse (Figure 3).
4. **Reward stability:** CoVo's reward accuracy remains high throughout training, unlike majority-voting methods, which degrade due to reward hacking as training proceeds (Figure 5).
5. **Ablation:** the combination of vectorial aggregation $r_{\text{int}}^V$ + curiosity reward $r_{\text{cur}}$ performs best on most tasks; vectorial aggregation is more robust than linear aggregation on math and science reasoning.
