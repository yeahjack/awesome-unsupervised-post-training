# RENT: Reinforcement Learning via Entropy Minimization

> **Added to survey on:** 2026-03-11

**Paper:** Maximizing Confidence Alone Improves Reasoning
**Authors:** Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, Deepak Pathak (Carnegie Mellon University)
**arXiv:** 2505.22660
**Method:** RENT | **Carrier:** Policy Opt. (GRPO) | **Regime:** training-time | **Level:** Token

| When to Adapt | Few-Sample Target Adaptation before held-out inference |
|---|---|
| Trigger Unit | small unlabeled task dataset / rollout batch |
| Persistence | full parameter accumulate across short RL runs |
| Inference Coupling | adapt first on the task cohort, then evaluate on downstream benchmarks |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Few-Sample Target Adaptation |
| Visibility Scope | Few-sample target subset |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Few-Sample Target Adaptation`; `Visibility Scope=Few-sample target subset`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.

- **When the update is triggered:** the update is triggered on a very small unlabeled task cohort, typically running only a small number of RL / optimization steps.
- **Whose sample it serves:** this batch of updates mainly serves a subsequent evaluation on a larger benchmark set, not the immediate inference of any individual prompt.
- **Whether parameters/state accumulate:** parameters accumulate continuously during the short training run and are reset only at the end of one experiment.
- **Reset boundary:** so it sits between offline adaptation and test-time micro-adaptation, but the primary coupling is still "adapt first, then evaluate as a whole".

## 1. UPT Assignment Rationale

RENT belongs to **Family I (Prediction-Statistic Optimization)**; specific reasons:

- **Source of the intrinsic reward:** RENT uses the negative average entropy of the model's own token-prediction distribution as the reward, i.e. $\mathcal{R}(y_{\text{pred}}) = -H(\pi(x)) = \frac{1}{T}\sum_{t=1}^{T}\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$. This signal comes entirely from the model's intrinsic predictive statistics, with no reliance on external ground truth, external verifiers, or human annotation.
- **Carrier is policy optimization:** GRPO is used to drive the policy update with the entropy reward — a training-time RL optimization.
- **Token-level signal:** the reward is computed as entropy at each token's prediction distribution and then averaged — token-level intrinsic statistics.
- **No external supervision:** $y_{\text{target}}$ never enters the reward computation, satisfying the UPT definition.

---

## 2. Problem Addressed

Current RL-based methods for improving LLM reasoning rely heavily on external supervision signals (e.g. a correctness reward derived from a ground-truth answer); in many real-world scenarios, external annotation is scarce or unavailable. RENT poses a core question: **can we improve an LLM's reasoning ability using only the model's own confidence as the reward, in a fully unsupervised manner?**

Specifically:
- Reward engineering is the core difficulty of RL; external reward design is costly and domain-specific.
- In open-ended, long-form free-response settings, alternatives such as majority voting are not viable.
- The model's token entropy is positively correlated with answer accuracy, especially toward the end of the response (near the final answer), where the correlation is strongest.

---

## 3. Method

### Core idea
RENT uses the negative entropy of the model's token-prediction distribution as the RL reward, encouraging the model to produce a high-confidence chain of thought and answer.

### Steps
1. **Entropy reward definition:** for a given prompt $x$, the model generates the response $y_{\text{pred}} = (y_{\text{pred},1}, \dots, y_{\text{pred},T})$. The entropy at token $t$ is $H(p_t) = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$. The reward for the full response is the negative average of all token entropies:
$$\mathcal{R}(y_{\text{pred}}) = \frac{1}{T}\sum_{t=1}^{T}\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$$
2. **Policy optimization:** GRPO (Group Relative Policy Optimization) is used. GRPO evaluates the current policy's reward improvement against a group of reference policies and is well suited to noisy or unsupervised rewards. The learning objective is:
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y_{\text{pred}} \sim \pi(x)}[\mathcal{R}(y_{\text{pred}})]\right]$$
3. **Token-selection analysis:** the paper investigates which tokens' entropy is most effective. Experiments show that the entropy of the response's last tokens ("last chunk" strategy) correlates best with accuracy, suggesting the model relies more on its own confidence as it approaches the final answer. The default implementation, however, averages over all tokens.
4. **Training details:** Adam optimizer, learning rate $1 \times 10^{-6}$; trained independently on each dataset; for some benchmarks the same dataset is used for both training and evaluation (since RENT does not rely on ground truth).

### Comparison with format reward
The paper verifies that RENT is more than just learning the correct format — compared with a format-only reward (a binary signal for whether the output follows \boxed{} etc.), RENT consistently beats the format-only baseline on most benchmarks.

---

## 4. Datasets

| Dataset | Description | Scale |
|--------|------|------|
| **GSM8K** | Grade-school math word problems | Train ~7473, test ~1319 |
| **MATH500** | 500-problem subset of the MATH test set (sampled by OpenAI), covering seven competition categories | 500 |
| **AMC** | 2022 and 2023 AMC12 problems (reformulated as integer answers) | 83 |
| **AIME24** | Both versions of the 2024 AIME | 30 |
| **GPQA** | PhD-level multiple-choice biology, physics, chemistry questions ("Google-proof") | 448 |

Note: aside from GSM8K, training and evaluation use the same dataset for the other benchmarks (since RENT uses no ground truth, no overfitting in the traditional sense occurs).

---

## 5. Evaluation metrics and main results

### Metrics
- **Accuracy:** answer-match accuracy per benchmark (string matching).
- **Standard deviation:** reported on GSM8K (5 samples), MATH500 (5), AMC (32), AIME (64), GPQA (10).

### Main results

**Cross-model gains (Figure 2 / Table 1):** on six models — Mistral-7B, Llama3.1-8B, Qwen2.5-1.5B, Qwen2.5-Math-1.5B, Qwen2.5-7B, Qwen2.5-Math-7B — RENT beats the baseline on essentially every benchmark:

| Model (w/ RENT) | GSM8K | MATH500 | AMC | AIME | GPQA |
|---|---|---|---|---|---|
| Mistral-7B | 0.492 | 0.168 | 0.068 | 0.033 | 0.267 |
| Llama3.1-8B | 0.859 | 0.548 | 0.339 | 0.082 | 0.332 |
| Qwen2.5-1.5B | 0.748 | 0.597 | 0.298 | 0.072 | 0.267 |
| Qwen2.5-Math-1.5B | 0.863 | 0.810 | 0.504 | 0.145 | 0.285 |
| Qwen2.5-7B | 0.911 | 0.823 | 0.518 | 0.270 | 0.365 |
| Qwen2.5-Math-7B | 0.967 | 0.882 | 0.591 | 0.167 | 0.400 |

**Comparison with concurrent work (Table 2, Qwen2.5-7B-Instruct):**

| Method | GSM8K | MATH500 | AMC | AIME | GPQA | Average |
|---|---|---|---|---|---|---|
| TTRL | **0.933** | 0.822 | 0.521 | 0.172 | 0.346 | 0.559 |
| Intuitor (forward KL) | 0.929 | 0.783 | **0.525** | 0.200 | 0.337 | 0.555 |
| Spurious Rewards | 0.910 | 0.774 | 0.459 | 0.156 | 0.342 | 0.528 |
| **RENT (Ours)** | 0.911 | **0.823** | 0.518 | **0.270** | **0.365** | **0.577** |

RENT has the highest average (0.577) and a particularly large lead on the hardest benchmark, AIME.

### Key findings
- **Confidence is strongly positively correlated with accuracy:** during training, model confidence and accuracy rise together (Figure 3).
- **Late-token entropy matters more:** the entropy–accuracy correlation of the "last chunk" strategy is much higher than that of the "first chunk" (Figure 4).
- **Not just format learning:** RENT consistently beats a format-only reward, showing that the model's reasoning ability actually improves.
- **Limitations:** risk of overconfidence; unsupervised methods cannot match methods with ground truth; calibration mistakes can lead to catastrophic errors.
