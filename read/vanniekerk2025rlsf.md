# RLSF: Post-Training LLMs via Reinforcement Learning from Self-Feedback

> **Added to survey on:** 2026-03-11

**Paper:** Post-Training Large Language Models via Reinforcement Learning from Self-Feedback
**Authors:** Carel van Niekerk, Renato Vukovic, Benjamin Matthias Ruppik, Hsien-chin Lin, Milica Gašić (Heinrich Heine Universität, Düsseldorf, Germany)
**ArXiv:** 2507.19131
**Date:** 2025-07

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RLSF | Pref. Opt. | training-time | Traj. |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / synthetic preference pair batch |
| Persistence | full parameter accumulate across epochs or iterations |
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

- **When the update is triggered:** updates happen during the pre-deployment preference-optimization stage; the basic unit is model-generated chosen/rejected pairs.
- **Serving the current sample or future ones:** updates on the current pair batch serve subsequent training and the final deployed model, not the immediate inference of the current sample.
- **Whether parameters/state accumulate:** parameters accumulate across training epochs / iterations with no sample-level reset.
- **Reset boundary:** this is offline pair-based post-training, not online TTA.

## 1. UPT Assignment Rationale
**Family III — Self-Generated Target Bootstrapping (internally generated preference pairs)**

RLSF depends on no external human annotation, gold answer, or external reward model. It uses the **model's own confidence (the probability disparity on the answer span)** to rank multiple CoT traces of the same prompt, forming synthetic preference pairs, then performs preference optimization via PPO or DPO. The whole process is a pure self-supervised intrinsic-reward route — a textbook "internally generated preference pairs → preference optimization" paradigm.

---

## 2. Problem Addressed

LLMs face two core issues on reasoning-heavy tasks:
1. **Mis-calibration:** RLHF-trained models often become overconfident — output confidence diverges from actual accuracy, especially on complex reasoning.
2. **Unstable reasoning:** LLMs are unreliable on multi-step logical reasoning (math, commonsense); CoT prompting depends on prompt engineering and yields inconsistent results.

Existing methods either rely on human annotation (RLHF) or on stronger external LLM feedback (RLAIF) — expensive and limited in scalability. RLSF proposes using the model's own uncertainty as an intrinsic reward signal — mimicking how humans learn from confidence in the absence of external feedback — to improve both calibration and reasoning accuracy in a data-efficient way.

---

## 3. Method

Core idea: **in a well-calibrated model, answer confidence correlates positively with reasoning quality and accuracy** → rank CoT traces by confidence to form preference data → RL fine-tune.

### 3.1 Chain-of-Thought decoding to generate candidates

Given input query $q$, at the **first decoding step** sample the top-$K$ probability tokens $w_k$ ($k=1,...,K$), then greedily auto-regressively decode from each $w_k$ to obtain a full hypothesis $h_k$. This yields $K$ different reasoning traces per prompt.

### 3.2 Answer-span identification and confidence computation

Append "So the answer is" to each hypothesis $h_k$ to elicit answer tokens $g'_k$, then locate the answer span $g_k$ in the original hypothesis via string matching.

Confidence is measured by **probability disparity**:

$$c = \frac{1}{M} \sum_{i=0}^{M-1} \left[ \max_w \pi_\theta(w|q \oplus h_{m+i}) - \max_{w \neq \arg\max \pi_\theta} \pi_\theta(w|q \oplus h_{m+i}) \right]$$

i.e., averaged across each answer-span token position, the gap between the highest-probability and second-highest-probability token. Higher disparity → higher model certainty about the answer. Compared with raw token probability, disparity is a more reliable proxy for true confidence.

### 3.3 Building the preference dataset

Rank the $K$ hypotheses by confidence score $c_k$: high-confidence → preferred, low-confidence → rejected, forming preference dataset $D$. Raw reward values are linearly scaled to $[-1, 1]$ for training stability.

### 3.4 Policy optimization

Two variants:
- **RLSF with PPO:** train a Bradley-Terry reward model $R_\phi$ on $D$ (a scalar regression head on top of the base model, LoRA fine-tuned), then optimize the policy $\pi_\theta$ with PPO.
- **RLSF with DPO:** directly use $D$ to supervised-fine-tune the policy with DPO loss, bypassing explicit reward-model training.

Experiments show **PPO consistently beats DPO**; the authors argue RL is better suited to intrinsic-motivation signals.

### 3.5 Implementation details
- $K=10$ candidates during CoT decoding
- Reward model: LoRA fine-tune base model + scalar regression head
- Training uses the TRL (Transformer Reinforcement Learning) library
- PPO hyperparameters: lr 5e-5, epochs 5, temperature 0.7, KL coefficient β=0.05, clipping ε=0.2, γ=0.98, GAE λ=0.95
- DPO hyperparameters: lr 5e-5, epochs 5, label smoothing 0.01, β=0.2
- Inference latency is unaffected after RLSF (no CoT decoding needed)

---

## 4. Datasets

| Domain | Dataset | Description |
|------|--------|------|
| Math reasoning | MultiArith | Multi-step arithmetic word problems |
| Math reasoning | GSM8K | Grade-school math word problems requiring multi-step calculation and reasoning |
| Multi-choice QA | CommonsenseQA | Commonsense reasoning, constrained answer format |
| Multi-choice QA | ARC Easy | Grade-school science reasoning |
| Reward-model eval | RewardBench (Math Reasoning subset) | Evaluates reward-model ranking of preferred/rejected responses |
| Bias eval | XSTest | Detects LLM exaggerated safety behavior |
| Bias eval | AlpacaEval | Uses GPT-4o as zero-shot annotator to compare model outputs |

---

## 5. Evaluation metrics and main results
### Metrics

- **Answer Accuracy:** exact match with ground truth
- **Expected Calibration Error (ECE):** alignment between token-level confidence and actual correctness (lower is better)
- **Reward Model Accuracy:** fraction of preferred/rejected pairs correctly ordered by the reward model on RewardBench

### Main results

#### Reward-model evaluation (RewardBench Math Reasoning)

| Reward Model | Data needed | Accuracy↑ |
|---|---|---|
| URM (LLaMa 3.1 8B) | Prompt + Preference | 97.00 |
| QRM (LLaMa 3.1 8B) | Prompt + Preference | 96.80 |
| QWEN 2.5 7B AfD | Prompt + Answer | 89.29 |
| Gemma 2 2B AfD | Prompt + Answer | 73.12 |
| **QWEN 2.5 7B RLSF** | **Prompt only** | **76.13** |
| **Gemma 2 2B RLSF** | **Prompt only** | **81.43** |

RLSF is the only method that derives a reward model from prompts alone (no answers or preference annotations), with competitive performance.

#### Math reasoning (Table 2)

**Gemma 2 2B:**

| Setting | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| Base (Greedy) | 98.12 | 7.43 | 85.57 | 12.24 |
| Base (CoT K=10) | 99.01 | 4.12 | 89.18 | 10.94 |
| RLHF (PPO) + URM | 97.83 | 12.73 | 82.43 | 17.83 |
| **RLSF (DPO)** | 96.13 | 10.52 | 84.74 | 17.43 |
| **RLSF (PPO)** | **98.83** | **7.81** | **88.14** | **12.54** |

**QWEN 2.5 7B:**

| Setting | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| Base (Greedy) | 79.44 | 20.75 | 53.67 | 42.86 |
| Base (CoT K=10) | 77.77 | 22.22 | 58.52 | 41.46 |
| RLHF (PPO) + URM | 74.13 | 23.24 | 46.82 | 51.24 |
| **RLSF (DPO)** | 71.62 | 25.12 | 50.73 | 38.15 |
| **RLSF (PPO)** | **78.06** | **18.35** | **58.62** | **41.92** |

#### Multi-choice QA

**ARC Easy (Gemma 2 2B):**

| Setting | Accuracy↑ | ECE↓ |
|---|---|---|
| Base (Greedy) | 96.96 | 16.12 |
| Base (CoT K=10) | 96.28 | 3.03 |
| RLSF (DPO) | 97.05 | 18.83 |
| **RLSF (PPO)** | **97.04** | **5.12** |

**CommonsenseQA (Phi-2):**

| Setting | Accuracy↑ | ECE↓ |
|---|---|---|
| Base (Greedy) | 54.46 | 25.12 |
| Base (CoT K=10) | 58.91 | 23.11 |
| RLSF (DPO) | 59.79 | 30.91 |
| **RLSF (PPO)** | **61.13** | **19.64** |

#### Bias evaluation (XSTest, AlpacaEval)

| Training dataset | Evaluation dataset | Preference(%)↑ |
|---|---|---|
| GSM8K + MultiArith | XSTest | 50.73 |
| CommonsenseQA | XSTest | 51.82 |
| XSTest | XSTest | 63.24 |

RLSF trained on reasoning tasks does not amplify the base model's safety bias.

#### Discount-factor γ ablation (Gemma 2 2B, RLSF+PPO)

| Variant | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| DPO | 96.13 | 10.52 | 84.74 | 17.43 |
| PPO (γ=1.0) | 98.52 | 8.12 | 87.13 | 12.49 |
| PPO (γ=0.98) | 98.83 | 7.81 | 88.14 | 12.54 |

γ=0.98 slightly beats γ=1.0, suggesting modest discounting helps optimization.

### Key findings

1. **PPO consistently beats DPO:** across all tasks and models, RLSF(PPO) beats RLSF(DPO) on both accuracy and ECE — RL is better at exploiting intrinsic-motivation signals; DPO lacks reward discounting.
2. **Simultaneous improvement of calibration and accuracy:** RLSF(PPO) raises accuracy while lowering ECE, whereas RLHF(PPO)+URM actually degrades calibration (Gemma 2 ECE rises from 7.43 to 12.73).
3. **Competitive reward model from prompts only:** RLSF reaches 81.43 (Gemma 2) on RewardBench with prompts alone, and the result is robust to the choice of base LLM.
4. **Does not amplify bias:** RLSF trained on reasoning tasks performs comparably to the base model on XSTest safety tests (~50–52%), introducing no extra safety-related bias.
5. **Reward model learns meaningful token-level reward:** correct answers receive high reward, intermediate reasoning steps are also rewarded, lengthy explanations score lower, and bare answers without reasoning score low — the model learns to distinguish correctness from informative justification.
6. **No inference overhead:** although training-data construction requires CoT decoding ($K$× inference cost), the trained model has the same inference cost as the original.
