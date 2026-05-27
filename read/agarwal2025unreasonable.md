# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

> **Added to survey on:** 2026-03-11

**Paper information:** Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng (UIUC), arXiv: 2505.15134, May 2025.

| When to Adapt | multi-protocol: Offline Corpus UPT (EM-FT, EM-RL) + No-Update Inference (EM-INF, adjacent) |
|---|---|
| Trigger Unit | mixed: whole corpus / prompt batch / inference state |
| Persistence | mixed: offline parameter accumulation + prompt-local inference state |
| Inference Coupling | mixed |
| Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Multi-protocol: Cumulative + Non-Cumulative |
| Reset Boundary | Multi-protocol: Deployment Boundary + Token / Sequence Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Multi-protocol: Offline Corpus UPT + No-Update Inference (adjacent) |
| Visibility Scope | Multi-protocol: Pre-deployment corpus + Current Sequence Prefix Only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** this paper contains multiple protocol entries: `Timing Regime=Multi-protocol: Offline Corpus UPT + No-Update Inference (adjacent)`; `Visibility Scope=Multi-protocol: Pre-deployment corpus + Current Sequence Prefix Only`.
- **Two-axis coding:** `Input Visibility=Multi-protocol: Offline + Online`; `Update Persistence=Multi-protocol: Cumulative + Non-Cumulative`; `Reset Boundary=Multi-protocol: Deployment Boundary + Token / Sequence Boundary`.

| Protocol Entry | Timing Regime | Visibility Scope | Input Visibility | Update Persistence | Reset Boundary | Note |
|---|---|---|---|---|---|---|
| EM-FT | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | Directly minimize entropy in the offline training stage, producing a new deployed model. |
| EM-RL(seq) | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | Offline sequence-level RL variant; updates persist across samples into the deployment stage. |
| EM-RL(tok) | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | Offline token-level RL variant; updates persist across samples into the deployment stage. |
| EM-INF *(adjacent)* | No-Update Inference (adjacent) | Current Sequence Prefix Only | Online | Non-Cumulative | Token / Sequence Boundary | Logit-level entropy descent during inference; per the paper's figure (`No-Update Inference (adjacent)` leaf, fails B1) it is routed to adjacent rather than strict UPT. |

- **When the update is triggered:** this note covers a method family rather than a single protocol — EM-FT, EM-RL(seq), and EM-RL(tok) happen in the offline pre-deployment stage, while EM-INF runs at inference time.
- **Whose sample it serves:** correspondingly, "whose sample the current update serves" splits into two: the offline variants serve the subsequent deployed model, while EM-INF serves the current prompt / current inference state.
- **Whether parameters/state accumulate:** persistence is also mixed — the offline variants accumulate parameters, while EM-INF performs prompt-local logit / hidden-state optimization.
- **Reset boundary:** so this note adopts a protocol-level split: the offline variants are bounded at the deployment boundary, and EM-INF at the token / sequence boundary.

## 1. UPT Assignment Rationale

The training-time variants (EM-FT, EM-RL(seq), EM-RL(tok)) are strict UPT in **Family I: Prediction-Statistic Optimization**. EM-INF is the inference-time variant; the paper's figure routes it to the adjacent **No-Update Inference (adjacent)** branch because it does not modify parameters, adapters, memories, or persistent local state (fails B1).

Core reason: every entropy-minimization (EM) variant directly minimizes the Shannon entropy of the model's own predictive distribution, with no reliance on external labels, external verifiers, or human feedback. The sole optimization signal comes from intrinsic statistics of the model's outputs — token-level or trajectory-level entropy estimates. This fits the "prediction-statistic" definition: the objective is derived directly from the model's own prediction statistics.

Specifically:
- **EM-FT** (strict UPT, Family I): Direct Opt., training-time, token level — direct gradient descent on token-level entropy.
- **EM-RL(seq)** (strict UPT, Family I): Policy Opt., training-time, trajectory level — uses negative trajectory entropy as intrinsic reward, optimized by policy gradient.
- **EM-RL(tok)** (strict UPT, Family I): Policy Opt., training-time, token level — uses the cumulative negative token entropy as reward.
- **EM-INF** (*adjacent*, fails B1): State Opt., inference-time, token level — gradient descent on logits at inference to lower entropy; no persistent parameter, adapter, memory, or state update remains after the sequence ends.

All variants require no external ground truth; the signal type is intrinsic statistics.

---

## 2. Problem Addressed

The paper studies a core question: **can we improve an LLM's reasoning ability by minimizing only the entropy of the model's own outputs, without using any annotated data?**

Modern LLMs acquire substantial reasoning ability during pretraining, but this ability is often under-elicited. Traditional post-training methods (SFT, RLHF, GRPO) depend on labeled data or external verifiers. This paper explores an extremely minimal alternative: leverage the correlation between model confidence and correctness, reinforcing high-confidence outputs to lift performance.

Specific motivations:
- Traditional RL methods (GRPO, RLOO) require output verification, but many tasks (code generation, scientific programming) are hard to answer-extract and verify.
- Self-consistency relies on majority voting and is inapplicable for complex tasks.
- Iterative refinement is constrained by context length.
- A general post-training / inference method is needed that makes no task assumptions and requires no labels.

---

## 3. Method

### Basic definitions

Let $\pi_\theta$ be the autoregressive LLM policy. Entropy is estimated in two ways:

- **Trajectory-level entropy:** $\hat{\mathcal{H}}_{\text{traj}}(\pi_\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \pi_\theta(\mathbf{y}^i)$ — based on the log-probability of the full sequence.
- **Token-level entropy:** $\hat{\mathcal{H}}_{\text{tok}}(\pi_\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{|y^i|} \mathcal{H}(\pi_\theta(\cdot | \mathbf{y}_{<t}^i))$ — Shannon entropy summed token by token.

### 3.1 EM-FT: Direct Entropy Minimization Fine-tuning

- **Optimization:** Direct optimization (gradient descent).
- **Regime:** Training-time.
- **Level:** Token.

EM-FT mimics the SFT pipeline but uses no labeled data. Given input prompts, sample $N$ trajectories from the model itself and directly minimize the token-level entropy $\hat{\mathcal{H}}_{\text{tok}}$. The gradient is $\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{|y^i|} \nabla_\theta \mathcal{H}(\pi_\theta(\cdot | \mathbf{y}_{<t}^i))$.

Key finding: even $N=1$ (a single sample) yields an average +8% on math and code tasks over the base model, and on Minerva and LeetCode it surpasses GRPO and RLOO trained with 60K annotated examples.

### 3.2 EM-RL(seq): Trajectory-level Entropy as Intrinsic Reward

- **Optimization:** Policy optimization (REINFORCE).
- **Regime:** Training-time.
- **Level:** Trajectory.

Use negative trajectory-level entropy as the reward: $r_{\text{traj}}(\mathbf{y}) = \sum_{t=1}^{|\mathbf{y}|} \log \pi_\theta(y_t | y_{<t}) = \log \pi_\theta(\mathbf{y})$.

This reward favors high-overall-probability trajectories without caring whether each token is certain; it suits shorter reasoning tasks (simple math problems) with a limited number of reasoning paths. The RLOO baseline reduces gradient variance, plus a KL regularizer ($\beta=0.001$) to prevent drifting too far from the base model.

### 3.3 EM-RL(tok): Cumulative Negative Token Entropy as Reward

- **Optimization:** Policy optimization (REINFORCE).
- **Regime:** Training-time.
- **Level:** Token.

Use the cumulative negative token-level entropy as the reward: $r_{\text{tok}}(\mathbf{y}) = -\sum_{t=1}^{|\mathbf{y}|} \mathcal{H}(\pi_\theta(\cdot | y_{<t}))$.

This reward favors trajectories that are more confident at every generation step; it suits complex tasks that need long chain-of-thought reasoning and prevents the model from "getting lost" in long reasoning. Compared with EM-RL(seq), EM-RL(tok) does better on long-reasoning tasks such as LiveCode and AMC.

### 3.4 EM-INF: Inference-time Logit Optimization

- **Optimization:** State optimization (gradient descent on logits).
- **Regime:** Inference-time.
- **Level:** Token.

EM-INF performs no parameter update. At each generation step $t$, treat the model's last-layer logit vector $z_t \in \mathbb{R}^{|\mathcal{V}|}$ as a free parameter and gradient-descend to minimize the entropy of the softmax distribution:

$$\mathcal{L}_{\text{EM-INF}} = \max\left(-\sum_{j \in \mathcal{V}} \sigma(z_t)_j \log \sigma(z_t)_j, \delta\right)$$

where $\delta \in (0.1, 0.5)$ is a lower bound on entropy to prevent over-optimization that degenerates into greedy decoding. Empirically 5–15 gradient steps suffice. The optimization touches only $|V|$ parameters (~150K) — negligible relative to the 7B model parameters.

Key advantages:
- No training data or parameter update needed.
- Computational cost $\mathcal{O}(n)$ ($n$ = sequence length), whereas self-consistency and iterative refinement need $\mathcal{O}(Nn)$.
- Applicable to any task; no answer extraction or output verification required.
- Unlike temperature scaling, logit optimization can change the relative order of non-top logits, which is more effective in high-entropy settings.

---

## 4. Datasets

### Training data (for EM-FT and EM-RL)
- **Math:** 35K prompts randomly sampled from Numina Math.
- **Code:** 25K prompts from the Eurus-2 coding split.
- Training data contains only prompts, **no labels or answers**.

### Evaluation benchmarks

**Math:**
- MATH-500.
- AMC (AMC competition problems).
- AIME 2024.
- Minerva Math.
- Olympiad Bench (Olymp.).

**Coding:**
- LeetCode (LeetC).
- LiveCodeBench-v2 (LiveC).

**Scientific programming** (EM-INF only):
- SciCode.
- UGPhysics.

### Base models
- Qwen2.5-Math-7B / Qwen2.5-7B-Instruct (math training).
- Eurus-2-7B-SFT (code training).
- Qwen2.5-32B-Instruct (EM-INF experiments).
- Llama-3.1-8B-Instruct (ablation).

---

## 5. Evaluation metrics and main results

### Metrics
- Math: accuracy (extract the answer, compare with ground truth).
- Code: pass rate.
- Scientific programming (SciCode): accuracy on sub-problems and on main problems.
- Computational efficiency: FLOPs (training $6PD$, inference $2PD$; $P$ = parameter count, $D$ = number of tokens).

### EM-FT and EM-RL main results (Table 2, Qwen2.5-7B)

| Method | Math | AMC | AIME | Minerva | Olymp. | Avg.(Math) | LeetC | LiveC | Avg.(Code) |
|------|------|-----|------|---------|--------|------------|-------|-------|------------|
| Base model | 43.8 | 31.3 | 15.6 | 14.7 | 19.0 | 24.9 | 26.1 | 18.4 | 22.3 |
| w/ RLOO N=4 (60K labels) | 73.0 | 57.8 | 23.3 | 31.2 | 34.2 | 43.9 | 28.3 | 26.7 | 27.5 |
| w/ GRPO N=4 (60K labels) | 71.8 | 56.6 | 21.1 | 25.0 | 35.9 | 42.1 | 25.0 | 25.8 | 25.4 |
| **EM-FT N=1** (no labels) | 67.2 | 51.8 | 14.4 | **33.3** | 34.4 | 40.2 | 28.3 | 17.2 | 22.8 |
| **EM-RL-seq N=4** (no labels) | 67.2 | **53.0** | **21.1** | **30.9** | 35.6 | **41.6** | *31.1* | 21.7 | 26.4 |
| **EM-RL-tok N=4** (no labels) | **70.8** | **57.8** | 18.9 | **30.9** | 35.9 | **42.9** | 29.5 | 24.5 | 27.0 |

Key findings:
- **EM-FT:** average +8% over the base model without any labels; surpasses GRPO/RLOO on Minerva and LeetCode.
- **EM-RL:** average +11% over the base model without labels; surpasses labeled GRPO/RLOO by ~4.5% on AMC, Minerva, and LeetCode.
- **EM-RL-tok** is slightly better than EM-RL-seq on math overall, while EM-RL-seq is stronger on coding.

### EM-INF main results (Table 3)

On Qwen2.5-7B-Instruct, EM-INF gains an average of ~3% on math, beating self-consistency ($N=4$) and iterative refinement at the cost of a single trajectory.

### SciCode scientific programming (Table 4)

- Qwen2.5-7B + EM-INF: main-problem accuracy rises from 0.0% to 1.5%; sub-problem from 11.5% to 16.7%.
- Qwen2.5-32B + EM-INF: main-problem 10.7%, beating GPT-4o (9.2%), Claude 3.5 Sonnet (12.3%), Gemini 1.5 Pro (7.7%), and Adaptive Temperature (7.6%).
- Efficiency: EM-INF is ~3× faster than self-consistency (Figure 1).

### Limitations (Table 5)

- On Llama-3.1-8B the gains are markedly smaller than on Qwen2.5, indicating EM's effectiveness depends on the base model's pretraining quality.
- Almost ineffective on individualistic-value-reasoning (IndVal) tasks, because model confidence is not a reliable proxy for quality there.
- EM's premise is "model confidence correlates with correctness"; when the base model is too weak or the task is far from the pretraining distribution, this premise fails.
