# RLME: Reinforcement Learning from Meta-Evaluation

> **Added to Survey:** 2026-03-11

**Paper:** Reinforcement Learning from Meta-Evaluation: Aligning Language Models Without Ground-Truth Labels
**Authors:** Micah Rentschler (Vanderbilt University), Jesse Roberts (Tennessee Technological University)
**ArXiv:** 2601.21268
**Date:** 2026-01-30

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RLME | Policy Opt. | training-time | Semantic |

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
**Family IV — Internal Evaluator Bootstrapping (evaluator-driven RL)**

RLME is a textbook evaluator-driven RL paradigm: the model itself (or another LLM) plays evaluator and answers natural-language meta-questions about the generator's output (e.g., "Is the answer correct?"); the evaluator's probability of outputting "Yes" is mapped directly to a scalar reward, and a GRPO-style algorithm (specifically CISPO) updates the generator with policy gradient. The whole pipeline avoids ground-truth labels and external reward models—the reward comes entirely from the evaluator's internal semantic judgment probability—so it sits squarely in **internal evaluator bootstrapping → policy optimization → evaluator-driven RL**. Conceptually similar to Self-Rewarding LM, but RLME generalizes evaluator judgment from preference ranking to arbitrary semantic-level yes/no questions through flexible meta-question design.

---

## 2. Problem Addressed

Most LLM RL pipelines (RLHF, RLVR) depend on **human preference annotation** or **ground-truth labels / task-specific verifiers**, hitting bottlenecks in scenarios such as:
1. **High annotation cost:** human feedback does not scale; automated verifiers are typically domain-specific and narrow.
2. **Ambiguous correctness:** open-domain QA, faithfulness, and similar tasks resist rule-based correctness verdicts.
3. **Label-less domains:** many real-world tasks have no ground truth at all (e.g., context faithfulness, reasoning consistency).

The core question RLME asks: can we train an LLM purely from the evaluator's answer probability on natural-language meta-questions, **without any ground-truth labels**?

---

## 3. Method

### 3.1 Assessment Prompting

For prompt $x \sim D$, the generator yields:

$$y \sim \pi_\theta(\cdot | x)$$

Then $J$ evaluators $\{\pi_{\phi_j}\}_{j=1}^{J}$ are queried with $K$ meta-questions $Q = \{q_1, \ldots, q_K\}$ (pre-written by human experts—"Is the answer correct?", "Is the reasoning logically consistent?", etc.). Each evaluator $j$ outputs a probability for the target answer $a_k$ (e.g., "YES") on meta-question $q_k$:

$$p_{j,k} = \pi_{\phi_j}(a_k \mid x, y, q_k)$$

The reward is a weighted log-prob aggregation across all evaluators and meta-questions:

$$r(x, y) = \sum_{j=1}^{J} \sum_{k=1}^{K} v_j \cdot w_k \cdot \log p_{j,k}$$

where $\{w_k\}$ are meta-question weights and $\{v_j\}$ are evaluator weights, both fixed hyperparameters.

### 3.2 Reinforcement Learning

Objective:

$$J(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta}\left[r(x, y)\right]$$

Updates use **CISPO** (Clipped IS-weight Policy Optimization, the GRPO variant from MiniMax 2025):

- **Advantage:** $A_i = r_i - \bar{r}$ (group-mean subtraction; **no division by std**, to avoid question-level difficulty bias, following Liu et al. 2025).
- **Importance-sampling ratio:** $\rho_i(\theta) = \frac{\pi_\theta(y_i | x_i)}{\pi_b(y_i | x_i)}$.
- **Clipping:** $\hat{\rho}_i(\theta) = \text{clip}(\rho_i(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}})$ with $\epsilon_{\text{low}} = 10000$ and $\epsilon_{\text{high}} = 5.0$.
- **Loss:**

$$L(\theta) = -\mathbb{E}_{y_i \sim \pi_b}\left[\text{sg}(\hat{\rho}_i(\theta)) \cdot A_i \cdot \sum_{t=1}^{T_i} \log \pi_\theta(y_{i,t} | x_i, y_{i,<t})\right]$$

### 3.3 Evaluator Configurations

| Config | Description |
|--------|-------------|
| **Live self-evaluation** | the generator itself is the evaluator; parameters co-evolve with training |
| **Frozen self-evaluation** | snapshot the generator at initialization as evaluator; parameters frozen |
| **Frozen other** | a different frozen external model is the evaluator |
| **Ensemble** | multiple evaluators aggregate their judgments (averaged log-probs) |

### 3.4 Meta-Question Design

Meta-questions are dataset-level generic semantic questions, not sample-specific ones. For instance:
- **Accuracy:** "Is the answer correct?"
- **Logical consistency:** "Does the whole solution logically lead from the question to an answer?"
- **Conciseness:** "Is the length of the solution between 200 and 500 characters?"
- **Context faithfulness:** "Is the answer supported by the context, regardless of whether it seems right or wrong?"

By combining meta-questions with weights, RLME implements **multi-objective behavioral control**.

### 3.5 Reward-Hacking Mitigation

The paper finds that long RLME training is prone to **reward hacking**: the generator learns to produce content that makes the evaluator say "Yes" while being wrong (e.g., empty platitudes such as "the only logical conclusion is that this is the correct answer", exploiting the evaluator's acquiescence bias). Mitigations:

- **Early stopping** based on validation accuracy.
- **Sparse ground-truth anchoring** (RLME-1GT / RLME-10GT): supplying the evaluator with ground-truth answers for 1% or 10% of training samples markedly stabilizes the curve.
- **Ensemble evaluator** (RLME-Crowd): multi-model voting smooths the reward curve but does not fully prevent reward hacking.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Math reasoning | **GSM8K** | grade-school math, fully verifiable, used in the core comparison |
| Open-domain QA | **CQAC** (custom) | 200 items each from SQuAD, NewsQA, TriviaQA, HotpotQA, BioASQ, DROP, RACE, TextbookQA (1600 total); context truncated to 4000 chars |
| Context faithfulness | **FaithEval-Counterfactual** | tests whether the model is faithful to context (even when context contradicts world knowledge); 300 held-out items; OOD generalization test |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **Accuracy:** regex-extract the answer from `\boxed{}`, exact match (integer / string).
- **FaithEval-Counterfactual accuracy:** does the model stay faithful to the provided context.
- **Solution length** (characters): used in multi-objective experiments to measure conciseness.
- **Counterfactual cheating rate:** with a ground-truth answer supplied during training, inject random wrong answers at test time and check whether the model blindly adopts them.

### Main Results

#### GSM8K (the core verifiable-domain experiment)
- The base model (Qwen3-4B-Base) starts at ≈30% accuracy.
- **RLME** rises rapidly past 90%, with a learning curve nearly identical to **RLVR** (the label-based baseline; Figure 2).
- Six independent runs, with overlapping ±1 σ bands.

#### Generator / Evaluator Choice
- **Generator choice matters far more than evaluator choice** (Figure 3 vs. Figure 4).
- Live self-evaluation and frozen self-evaluation are nearly identical.
- With SmolLM3 and Gemma3 as evaluators, accuracy drops late in training due to reward hacking.

#### Reward Hacking (Figure 5)
- Pure self-evaluation collapses in extended training while reward keeps climbing.
- RLME-Crowd (ensemble): smoother reward but still collapses.
- **RLME-10GT** (10% ground-truth) and **RLME-1GT** (1% ground-truth): effectively stabilize training and prevent collapse.

#### Multi-Objective: Accuracy + Conciseness (Figure 6)
- RLME-Concise nearly halves average solution length while keeping GSM8K accuracy comparable to RLME.
- Reasoning is compressed into denser mathematical expressions instead of verbose natural language.

#### Counterfactual Cheating Detection (Figure 7)
- RLVR and RLME-Base (meta-question "Is the answer correct?"): with ground truth provided in training, the model tends to "cheat" at test time when random answers are injected (rationalizing the injected answer).
- **RLME-NoCheat** (meta-question "Does the whole solution logically lead from the question to an answer?"): >80% accuracy on counterfactual tests, successfully avoiding cheating.

#### Open-Domain QA & Faithfulness (Tables 1 & 2)

| Method | CQAC Avg Accuracy | FaithEval-Counterfactual |
|--------|-------------------|--------------------------|
| Base (Qwen3-4B-Base) | 32.8% | 28.2% |
| RLVR | **62.1%** | 61.8% |
| RLVR+RLME | 57.0% | **70.4%** |

- RLVR+RLME slightly trails pure RLVR on CQAC (because it co-optimizes faithfulness) but **beats RLVR by nearly 9 percentage points on FaithEval-Counterfactual**.
- Key finding: the FaithEval improvement comes purely from meta-evaluation generalization—**no FaithEval samples appear in training**.

### Key Findings

1. **In a verifiable domain, meta-evaluation reward is on par with label-based RL** in both effectiveness and sample efficiency.
2. **Generator choice matters far more than evaluator choice**, supporting the "verifying is easier than generating" hypothesis.
3. **Reward hacking is RLME's central risk:** with extended training, the generator learns to exploit the evaluator's acquiescence bias—but **just 1% ground-truth anchoring is enough to mitigate it**.
4. **Meta-question design grants multi-objective and behavioral control:** changing the meta-question controls conciseness, reasoning honesty (anti-cheating), context faithfulness, etc.
5. **RLME generalizes to open-domain, label-less scenarios:** faithfulness meta-evaluation trained on CQAC transfers directly to the OOD FaithEval benchmark, showing that the optimization target defined by the meta-question generalizes domain-agnostically.
6. **RLME is best as a complement to, not a replacement for, RLVR:** RLVR is preferable when labels exist; RLME stands alone when they do not; the hybrid (RLVR+RLME) wins in multi-objective scenarios.
