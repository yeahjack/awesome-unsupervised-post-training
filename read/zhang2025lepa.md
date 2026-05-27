# Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving

> **Added to Survey:** 2026-03-11

> **Paper info:** published as a conference paper at ICLR 2025. Authors are from Tsinghua University, Moonshot AI, and Washington University in St. Louis.

| Attribute | Value |
|-----------|-------|
| Method | LEPA (LEarning to Plan before Answering) |
| Carrier | Direct Opt. (SFT, RL-compatible) |
| Regime | Training-time |
| Level | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | self-generated data batch / iteration round |
| Persistence | full parameter accumulate across synthesis / refinement rounds |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | PDF-confirmed correct-answer / verifier caveat |
| Strict UPT Status | Not strict UPT; verifier-assisted self-training adjacent |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit
- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When does the update fire:** updates fire in an offline pre-deployment self-bootstrap loop, typically a round-based "generate data / score / filter / retrain" schedule.
- **Serving the current sample or downstream samples:** synthetic samples or pseudo-targets in the current round primarily serve the next training round and the eventual deployed model.
- **Whether parameters / state accumulate:** parameters accumulate across rounds; the paper does not perform sample-level resets.
- **Reset boundary:** the `When to Adapt` essence is offline iterative bootstrapping, not online test-time adaptation.

## 1. UPT Assignment Rationale
> **PDF re-audit update (2026-04-21): recommend moving LEPA out of the strict UPT main table and into the verifier / correct-answer-assisted self-training adjacent group.**

LEPA's central artifact is indeed the model's self-generated anticipatory plan; however, the PDF reveals that data generation and filtering rely on an external correctness signal, so it does not satisfy the survey's current strict UPT boundary.

- **Binary correctness function:** PDF §2.1 formalizes LEPA with a problem set $D_{\text{prompt}}$ and a binary scoring function $f_{\text{cor}}(x_i,y_i)$, which judges whether a solution is correct and decides whether the sample enters the training set.
- **Correct-answer-assisted reflection:** PDF §2.1 says that when the initial solution is wrong, the LLM receives the problem, prior plan, incorrect solution, and "correct answer (if accessible)" for self-reflection; the appendix prompt also includes "The desired correct final answer is: [Correct Answer]".
- **Dataset / verifier use:** the experiments use the dataset-creator-supplied function for Hendrycks MATH to evaluate solution correctness; the LEPA+REINFORCE variant rewards by final-answer correctness.
- **Strict UPT conflict:** although the plan is self-generated, training-sample acceptance, reflection, and reward labeling all rely on external correct answers / verifiers. It is closer to the STaR / ReST family of answer-verifier-assisted self-training than to no-external-ground-truth UPT.

**Suggested positioning:** `verifier-assisted / correct-answer-assisted self-generated-target adjacent`. It can stand as an important neighbor of Family III to discuss "plan as synthetic target", but it should not represent no-external-supervision on the strict UPT main table.

---

## 2. Problem Addressed

Existing self-training methods (STaR, ReST, ReST EM) only let the LLM generate step-by-step solutions during data generation, and the training objective only maximizes the log-likelihood of those solutions. Two core deficiencies follow:

1. **No transferable high-level meta-knowledge:** step-by-step solutions are problem-specific; the model only learns "how to solve this problem", not "the general strategy for solving this kind of problem", limiting generalization—especially on hard benchmarks like Hendrycks MATH.
2. **Self-reflection prone to false positives:** STaR and similar methods reference the correct answer to fix solutions, and the model tends to "cheat"—rewriting only the final answer without correcting reasoning, producing rationale-wrong-but-final-answer-right false positives.

LEPA's central question: **what should self-generated synthetic data contain?** The paper argues that, beyond the solution, it should contain an abstract, transferable high-level plan (anticipatory plan) so that the model gains general problem-solving strategies.

---

## 3. Method (with Figure / Table Descriptions)

### 3.1 Overall Framework

LEPA is an iterative self-training algorithm; each iteration has two stages:

- **Data Generation Phase:** generate high-quality (problem, plan, solution) triples.
- **Model Optimization Phase:** SFT-fine-tune the LLM on the generated data.

#### Figure 1 (paper p. 2)
Shows a didactic example contrasting baseline (ReST) and LEPA on a combinatorial-math problem about meerkats on guard duty:
- **(b) ReST** generates the solution directly; the reasoning errs (45/2 = 22.5 then somehow yields 9).
- **(c) LEPA** first generates an anticipatory plan (general combinatorial steps: identify item count → compute combinations → compute combinations per element → subtract guard-duty count), then solves step by step from the plan and arrives at the correct 36.

#### Figure 2 (paper p. 3)
Compares baseline and LEPA data-generation flows:
- **(a) Baseline:** prompt the LLM to generate step-by-step solutions; no high-level abstraction.
- **(b) LEPA:** generate an anticipatory plan first, then a solution; if the solution is wrong, optimize the plan via self-reflection and retry.

### 3.2 Data Generation Phase

Given an initial model $\theta_0$, problem set $D_{\text{prompt}} = \{x_i\}_{i=0}^{N-1}$, and a binary scoring function $f_{\text{cor}}(x_i, y_i)$, in iteration $t$:

1. **Plan generation:** prompt the LLM to produce an anticipatory plan $p_i^{t,0}$ for problem $x_i$. The prompt stresses that the plan should be "general meta-knowledge applicable to similar problems" with no problem-specific computation.
2. **Solution generation:** based on the plan and problem, the LLM generates a solution $y_i^{t,0}$.
3. **Correctness check:**
   - If $f_{\text{cor}}(x_i, y_i^{t,0}) = 1$, add $(x_i, p_i^{t,0}, y_i^{t,0})$ to the training set $D_{\text{train}}^t$.
   - Otherwise, enter the **self-reflection** loop (up to $l$ rounds):
     - The LLM receives the problem, prior plan, incorrect solution, and the correct answer (if accessible), analyzes the failure, and produces a new plan $p_i^{t,j}$.
     - The reflection prompt also forbids the new plan from including the correct answer or problem-specific information (**avoiding information bypassing**).
     - Generate a new solution from the new plan; iterate until correct or until the trial limit is hit.

#### Algorithm 1 (paper p. 5)
Provides full pseudocode, clearly showing the three-level nesting outer iteration $t$ → inner problem $i$ → self-reflection trial $j$.

### 3.3 Model Optimization Phase

After obtaining the training set $D_{\text{train}}^t$, LEPA formats the data as a **two-round conversation**:
- **Round 1:** the user gives the problem and asks the LLM for a plan → the assistant outputs the plan $p_i^t$.
- **Round 2:** the user asks the LLM to solve based on the plan → the assistant outputs the solution $y_i^t$.

The training objective minimizes negative log-likelihood:

$$L_{\text{SFT}}(\theta_t, D_{\text{train}}^t) = -\mathbb{E}_{(x_i, p_i^t, y_i^t) \sim D_{\text{train}}^t} [\log p_{\theta_t}(p_i^t, y_i^t | x_i)]$$

The default is SFT, but LEPA is also compatible with DPO, PPO, and REINFORCE.

### 3.4 Benefits of the Anticipatory Plan

Three angles:

1. **Reducing cognitive workload:** the plan acts as a blueprint that frames the high-level structure, preventing the LLM from getting lost in details.
2. **Learning generalizable high-level meta-knowledge:** the plan contains no problem-specific information and transfers to structurally similar problems.
3. **Avoiding information bypassing:** the plan forbids embedding the correct answer, isolating the answer-leakage channel and preventing false-positive solutions.

#### Figure 4 (paper p. 8)
A radical-simplification case study:
- The **initial plan** is too generic ("identify the math object and apply formulas to simplify"), giving insufficient guidance and producing sign errors in the solution.
- After **self-reflection**, the model finds the issue ("did not check signs") and generates a more concrete new plan ("simplify each radical with attention to signs; combine like terms; verify the result").
- Guided by the **new plan**, the model correctly simplifies $\sqrt{15 - 6\sqrt{6}} + \sqrt{15 + 6\sqrt{6}} = 6$.

### 3.5 Prompt Design

Appendix A (Figure 5) gives the full prompt templates:
- **Plan-generation prompt:** asks for "general knowledge applicable to similar problems", with no question-specific information, capped at 1024 tokens.
- **Solution-generation prompt:** asks the model to solve step by step from its own plan, indicating how each step is influenced by the plan.
- **Self-reflection prompt:** takes problem, original plan, incorrect solution, and correct answer; asks for an analysis of the failure and a new plan design.
- **New-plan-generation prompt:** outputs the new plan based on the reflection; also forbids embedding the correct answer.

---

## 4. Datasets

### Training / Evaluation Benchmarks

| Benchmark | Task type | Description |
|-----------|-----------|-------------|
| **Hendrycks MATH** | math reasoning | hard math problems; correctness function provided officially |
| **Hellaswag** | sentence-completion reasoning | tests commonsense reasoning and sentence understanding |
| **BoolQ** | passage understanding & reasoning | yes/no reading-comprehension questions |
| **PIQA** | physical commonsense | intuitive reasoning about the physical world |
| **CSQA** | commonsense QA | additional evaluation benchmark |
| **MMLU** | multi-task language understanding | additional evaluation benchmark |
| **MMLU-Pro (Math)** | OOD generalization test | trained on Hendrycks MATH, tested on MMLU-Pro Math |

### Base Models
- **Llama 3 8B Instruct** (main).
- **Llama 3.1 8B Instruct** (additional).

### Hyperparameters
- Up to 5 trials per round of data generation (LEPA: 1 initial + up to 4 self-reflections; ReST / ReST EM: rejection sampling 5×; STaR: up to 4 revisions).
- Sampling temperature 0.5 (data generation), 0.0005 (test).
- Learning rate 3e-7, 1 SFT epoch per round.

---

## 5. Evaluation metrics and main results
### 5.1 Main Results (Table 1)

Test accuracy on four reasoning benchmarks (base model Llama 3 8B Instruct):

| Method | Hellaswag | Hendrycks MATH | BoolQ | PIQA | Average |
|--------|-----------|----------------|-------|------|---------|
| CoT (zero-shot) | 60.8% | 19.5% | 77.3% | 67.0% | 56.1% |
| Plan+CoT (LEPA prompt, untrained) | 56.1% | 22.1% | 80.8% | 75.7% | 58.7% |
| ReST | 86.3% | 28.2% | 84.5% | 81.4% | 70.1% |
| ReST EM | 86.4% | 27.2% | 86.3% | 83.5% | 70.8% |
| STaR | 85.7% | 25.9% | 85.8% | 84.2% | 70.4% |
| **LEPA** | **91.2% (+4.8%)** | **30.2% (+2.0%)** | **88.4% (+2.1%)** | **85.9% (+1.7%)** | **73.9% (+3.1%)** |

**Key findings:**
- LEPA beats every baseline on all four benchmarks, with an average gain of +3.1%.
- The LEPA prompt alone (no training) outperforms zero-shot CoT on 3/4 benchmarks, showing that the plan is intrinsically valuable—but it underperforms on Hellaswag, suggesting the initial LLM is not yet calibrated to produce high-quality plans.
- STaR clearly drops on MATH; its false-positive-solution problem is especially harmful in complex reasoning.

### 5.2 Learning Curves (Figure 3)

Figure 3 plots test accuracy vs. iteration on the four benchmarks:
- On Hellaswag, LEPA trails for the first 10 rounds (the initial Plan+CoT prompt is worse than CoT) but gradually pulls ahead with training, showing self-training "wakes up" the LLM's plan-using ability.
- On the other three benchmarks, LEPA leads from the start and converges higher.

### 5.3 Ablation Studies

#### Necessity of the Anticipatory Plan (Table 2)

| Variant | MATH | BoolQ | PIQA |
|---------|------|-------|------|
| ReST EM (baseline) | 27.2% | 86.3% | 84.2% |
| Without Plan | 24.3% | 84.8% | 84.5% |
| Without Self-Reflection | 28.8% | 86.9% | 84.8% |
| **LEPA (full)** | **30.2%** | **88.4%** | **85.9%** |

- Removing the plan drops performance sharply (MATH 24.3% vs. 30.2%) and even falls below the ReST EM baseline.
- Removing self-reflection (using rejection sampling instead) also drops performance, showing that language-feedback-based plan optimization beats simple rejection sampling.

#### How to Spend Inference Compute (Table 3)

On Hendrycks MATH, comparing different ways to use extra inference compute:

| Method | Avg Tokens | Accuracy |
|--------|-----------|----------|
| STaR | 175.1 | 25.9% |
| ReST | 477.8 | 28.2% |
| **LEPA** | **826.4** | **30.2%** |
| Silence Tokens | 869.3 | 28.3% |
| Correction | 979.4 | 27.8% |
| Long Solution | 1409.7 | 25.4% |

LEPA is the only method that successfully turns extra inference compute into a real gain over the baseline. Silence tokens roughly equals ReST, while Correction and Long Solution even fall below ReST—useless token inflation cannot translate compute into performance.

#### RL Combination

LEPA+REINFORCE reaches 30.6% on MATH (vanilla LEPA 30.2%), confirming that LEPA composes with RL.

### 5.4 OOD Generalization (Table 4)

In the Hendrycks-MATH-train, MMLU-Pro-Math-test OOD setting, LEPA reaches 38.9%, clearly above STaR (35.8%) and ReST EM (35.3%), confirming that anticipatory plans really help the model learn transferable meta-knowledge.

### 5.5 Other LLMs and Benchmarks

- **Llama 3.1 8B Instruct** on MATH: LEPA 49.6% vs. ReST EM 46.9% (Table 5).
- **CSQA / MMLU** (Table 6): LEPA consistently beats every baseline.
- **Simple-Eval re-evaluation** (Table 7): LEPA 33.7% vs. ReST EM 31.4%; the conclusion holds.
