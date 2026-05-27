# T³RL: Tool Verification for Test-Time Reinforcement Learning

> **Added to Survey:** 2026-03-11

> **Method:** T³RL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv 2603.02203 — Ruotong Liao, Nikolai Röhrich, Xiaohan Wang, Yuhui Zhang, Yasaman Samadzadeh, Volker Tresp, Serena Yeung-Levy (LMU Munich, Stanford University)

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| Trigger Unit | benchmark cohort / rollout group + tool calls |
| Persistence | full parameter accumulate across test-time episodes; no per-sample reset |
| Inference Coupling | adapt within the cohort, then infer / evaluate with the updated model |
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

- **When does the update fire:** Updates fire on the target benchmark cohort by rollout group, but pseudo-label generation includes an extra layer of tool verification.
- **Serving the current sample or downstream samples:** The current group's update primarily serves the same cohort's later rounds and the eventual evaluation, not just one sample followed by a reset.
- **Whether parameters / state accumulate:** Parameters accumulate throughout the test-time run with no per-sample reset.
- **Reset boundary:** The temporal structure is therefore still close to cohort-level cumulative TTA; the evidence chain just adds one extra layer (a tool call) compared to strict UPT.

## 1. UPT Assignment Rationale
T³RL belongs to the **adjacent paradigm — Tool-Augmented Test-Time Self-Evolution**, a direct extension of strict UPT TTRL (Family II: Sample-Relation Supervision / population consensus).

Core mechanism: T³RL builds on TTRL's majority-vote consensus reward and adds **external tool verification** (a code interpreter) to correct the voting weights. Specifically, an LLM verifier rewrites each rollout as Python code; a code interpreter executes it and judges whether the rollout's answer is correct; verified rollouts get an extra weight ($\omega$ times) in weighted majority voting, producing more reliable pseudo-labels and reward signals.

**Relation to strict UPT:** T³RL's majority-vote aggregation and GRPO training fully inherit TTRL's relational-internal-supervision framework, sitting in strict UPT territory. But tool verification (code executor) introduces an **external execution environment as evidence source**, crossing the strict UPT "no external ground truth / verifier" hard boundary. Unlike RLVR-style ground-truth verifiers, the external tool used here (Python interpreter) is **generic and off-the-shelf**, requiring no task-specific ground-truth labels. So T³RL sits in the **frontier hybrid** zone between strict UPT and verifier-grounded RL.

---

## 2. Problem Addressed

- **False-popular mode collapse:** TTRL constructs pseudo-labels via majority voting, but when the model has systematic internal biases the most-frequent answer can be wrong. Such spurious consensus is reinforced in the online-RL positive-feedback loop, making the model increasingly confident in wrong answers.
- **Fragility of unverified consensus:** pure self-consistency rewards cannot tell "correct consensus" from "incorrect consensus" and lack external evidence for cross-checking.
- T³RL proposes giving rollouts grounded evidence via test-time tool verification (code execution), so verification-aware weighted voting replaces naive majority voting and stabilizes self-evolution.

---

## 3. Method

### 3.1 Overall Framework

On top of TTRL, T³RL adds three components:

1. **Verifier (LLM verifier):** given a question $x$ and each rollout $y_i$, the Verifier rewrites the rollout's reasoning into lightweight Python code and decides whether the execution result matches the rollout's extracted candidate answer.
2. **Verification tool (code interpreter):** runs the Python program produced by the Verifier and returns the execution result $a_i$ as evidence.
3. **Verification weight:** verified rollouts receive an $\omega$-fold vote in majority voting ($\omega \ge 1$); unverified rollouts keep weight 1.

### 3.2 Core Equations

**Candidate-answer extraction:**
$$\hat{a}_i = \text{Extract}(y_i)$$

**Tool verification:**
$$(a_i, v_i) = V(x, y_i), \quad v_i = \mathbf{1}[a_i = \hat{a}_i]$$
where $a_i$ is the tool's execution result and $v_i \in \{0, 1\}$ is the verification flag.

**Verification-aware weighted voting:**
$$w_i = (1 - v_i) \cdot 1 + v_i \cdot \omega$$
$$\tilde{y}^* = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^{N} w_i \cdot \mathbf{1}[a_i = a]$$

**Reward computation:**
$$r_i^v = \mathbf{1}[a_i = \tilde{y}^*]$$

### 3.3 Key Design Details

- **ω hyperparameter:** $\omega = 1$ degenerates to standard TTRL (pure majority voting); $\omega \to \infty$ approximates hard filtering of all unverified rollouts. Empirically $\omega = 5$ works best.
- **Verifier-capacity threshold:** weak verifiers (e.g., 0.5B) inject noise rather than reliable evidence and hurt performance. A 1B-level verifier is the minimum.
- **Vote-then-sample:** following TTRL, sample 64 rollouts for voting, then down-sample 32 for GRPO training.
- **Independent recomputation by verifier:** the system prompt explicitly instructs the verifier "do not assume the reasoning trace is correct" but to recompute from the original question.

### 3.4 Hyperparameters

- Learning rate: cosine schedule, peak $5 \times 10^{-7}$; AdamW.
- Rollout temperature: 0.6.
- Verifier temperature: 0.6; max generation length 1,024 tokens.
- Max generation length: 2,560 tokens.
- Verification weight $\omega$: 5.
- Episodes: MATH-500 (10), AMC (30), AIME 2024 (80).
- Hardware: 8 × NVIDIA A100 80GB GPUs.

---

## 4. Datasets

| Dataset | Domain | Notes |
|---------|--------|-------|
| **MATH-500** | math reasoning | a 500-item subset of the MATH test set spanning 5 difficulty levels (L1–L5) |
| **AMC** | math competition | American Mathematics Competition |
| **AIME 2024** | math competition | American Invitational Mathematics Examination, hardest |

All datasets are used **without labels**—only the questions are provided; ground-truth answers are not used.

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **pass@1:** average accuracy over 16 sampled answers at non-zero temperature (Qwen-Math experiments use greedy decoding).
- **Relative improvement over TTRL:** percentage gain of T³RL over TTRL.

### Main Results

**Math-specialized model (Qwen-2.5-Math-1.5B):**

| Method | AIME 2024 | AMC | MATH-500 | Avg |
|--------|-----------|-----|----------|-----|
| Baseline | 7.7 | 28.6 | 32.7 | 23.0 |
| w/ TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| w/ T³RL | **20.8** | **50.9** | **74.6** | **48.8** |
| Rel. over TTRL | +31.6% | +4.1% | +2.2% | +6.3% |

**Vanilla model (Qwen-2.5-1.5B):**

| Method | AIME 2024 | AMC | MATH-500 | Avg |
|--------|-----------|-----|----------|-----|
| Baseline | 0.2 | 0.6 | 7.7 | 2.8 |
| w/ TTRL | 3.5 | 28.6 | 63.2 | 31.8 |
| w/ T³RL | **4.1** | **30.7** | **65.0** | **33.3** |

**Instruct models (Llama-3.2-1B-Instruct, Llama-3-3B-Instruct):** consistent gains across the board.

### Key Findings

1. **Harder benchmarks gain more:** the largest relative gain is on AIME 2024 (the hardest, +31.6%) and the smallest on MATH-500 (the easiest, +2.2%). This matches intuition: on easy problems majority voting is already accurate, so tool verification adds little.
2. **MATH-500 difficulty stratification:** at L5 (hardest) T³RL gains +4.3% over TTRL; at L1 (easiest) only +0.2%.
3. **Compute efficiency:** T³RL with N=16 rollouts beats TTRL@64—verification raises the per-rollout information quality.
4. **Robustness:** T³RL's cross-run standard deviation is lower (best-accuracy std 2.638 → 1.890), with more stable training.
5. **A stronger verifier helps further:** a 7B verifier outperforms a 1.5B verifier on every benchmark.
