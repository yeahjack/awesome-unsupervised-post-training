# Can Large Reasoning Models Self-Train?

> **Added to survey on:** 2026-03-11

> **Paper metadata**

| Property | Value |
|---|---|
| Method | LRM Self-Train (Self-Rewarded Training, SRT) |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |
| Authors | Sheikh Shafayat (KAIST), Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, Andrea Zanette (CMU) |
| Status | Preprint, 2025 (arXiv: 2505.21444v2) |
| Code | https://github.com/tajwarfahim/srt |

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

This paper belongs to **Family III — Self-Generated Target Bootstrapping**, subclass **reasoning / plan / curriculum synthesis**.

Core rationale:

- **Pseudo-target construction without external labels:** the core mechanism of SRT uses the model's own majority vote across multiple samples to construct a pseudo-label (a semantic-level reasoning target), with no ground-truth verifier or human annotation. This is essentially a self-generated-target bootstrapping pipeline — the model "synthesizes" its own training targets.
- **Carrier is Direct Optimization:** the synthesized pseudo-reward is fed directly into the RL gradient update (RLOO/GRPO and other policy-gradient algorithms) to optimize model parameters online, not only adjusted at inference.
- **Level is Semantic:** majority vote selects consistency at the final-answer level — a semantic signal, not a trajectory-level full reasoning-path selection.
- **Self-amplifying self-feedback:** as RL training proceeds, the label-generating policy itself evolves (evolving teacher), producing higher-quality pseudo-targets and forming a positive cycle of "self-improving training signal" — a hallmark of self-generated-target bootstrapping.

---

## 2. Problem Addressed

### 2.1 Background and motivation

- The supply of pre-training data (human-curated corpora) is becoming a bottleneck for LLM scaling.
- RL with Verifiable Rewards (RLVR) has succeeded on reasoning tasks (e.g., DeepSeek-R1, OpenAI o1) but depends on ground-truth verifiers.
- Reaching super-intelligence requires going beyond domains where humans can supply ground truth, so models must **self-improve** — i.e., use judgments over their own outputs to guide later training.

### 2.2 Limitations of existing approaches

- Prior self-improvement work (STaR, Self-Instruct) is primarily SFT- or DPO-based; the label-generating policy stays fixed within a training round, updated only 1–10 rounds.
- These approaches cannot exploit the benefits of continuously updated on-policy data and are upper-bounded by the verification capability of a fixed teacher.

### 2.3 Core research question

> **In an RL framework, can self-training achieve continual self-improvement by updating the feedback signal in lockstep with each gradient update?**

Two sub-questions:
1. Can SRT exceed the base model's capability? (Capability improvement + self-supervision quality improvement.)
2. Can SRT's self-improvement continue indefinitely?

---

## 3. Method

### 3.1 Self-Rewarded Training (SRT) overall framework

The core idea: without a ground-truth verifier, use the model's own **majority vote** as a pseudo-label to construct the RL reward signal.

**Algorithm 1: Self-Rewarded Training (SRT)**

```
Input: Prompt dataset X
foreach RL iteration do:
    1. Sample minibatch B ⊆ X
    2. For each prompt x ∈ B:
       a. Generate n solutions y(1), ..., y(n) ~ π_label(·|x)
       b. Identify majority-vote answer:
          y_majority ← argmax_{y'} Σ_{i=1}^{n} 1[answer(y(i)) = y']
       c. Define reward function:
          r(y) ← 1[answer(y) = y_majority]
    3. Perform RL gradient update using r(·)
```

> **Figure 1** (Overview of SRT): in RLVR, the reward comes from a ground-truth verifier; in SRT, the model's own majority vote estimates the ground truth, and this proxy reward signal is used for training.

### 3.2 Majority voting as self-supervision

- **Theoretical basis:** majority-vote accuracy empirically exceeds single-sample accuracy (Wang et al., 2023a), exploiting the model's intrinsic **generation-verification gap**.
- **Concrete operation:** sample multiple responses per prompt → group by final answer → the most-voted answer becomes the pseudo-label → responses agreeing with the pseudo-label receive positive reward (binary reward).

### 3.3 Evolving teacher vs. fixed teacher

By controlling π_label, SRT studies two modes:
- **Fixed teacher:** π_label fixed as the base model (similar to Huang et al., 2023; Prasad et al., 2024).
- **Evolving teacher** (core of SRT): π_label = current policy π_θ, updated after every gradient step, so pseudo-label quality improves alongside training. In this mode, RL rollouts can be reused to generate pseudo-labels, **introducing no extra compute**.

### 3.4 RL-algorithm compatibility

SRT defines a reward-function form compatible with common RL algorithms:
- **RLOO** (Ahmadian et al., 2024): uses a leave-one-out baseline.
- **GRPO** (Shao et al., 2024): normalizes advantage by group-level mean/std.
- Experiments show no significant difference between RLOO and GRPO under SRT.

### 3.5 Curriculum-based self-training

In synthetic tasks with controllable difficulty, the paper proposes a simple curriculum:
- First train with ground-truth RL on the easiest difficulty level.
- Then use SRT to train successively harder levels (each level's final checkpoint becomes the next level's start point).
- Experiments show the model can keep climbing multiple difficulty levels without ground truth.

### 3.6 Reward hacking and model collapse

The paper identifies a key limitation of SRT:

- **Phenomenon:** prolonged SRT training leads the model to output high-entropy random token sequences plus a fixed "template" final answer (e.g., `\boxed{1}`) regardless of input.
- **Mechanism:** the self-consistency reward is hacked — the model maximizes the training pseudo-reward by emitting the same answer for every prompt, completely decoupling from true correctness.
- **Signals:**
  - Training pseudo-reward suddenly spikes
  - KL divergence rises sharply
  - Model entropy drops
  - Test-set accuracy collapses entirely

> **Figure 7:** shows SRT training dynamics — the synchronous sudden rise of self-reward and collapse of accuracy, confirming the reward-hacking hypothesis.

---

## 4. Datasets

### 4.1 Training datasets

| Dataset | Type | Description |
|---|---|---|
| **Reasoning Gym** (Stojanovski et al., 2025) | Synthetic | 3 tasks: Family Relationships, Bitwise Arithmetic, Knights & Knaves; controllable difficulty levels |
| **MATH-12K** | Math | 12K math problems |
| **DAPO** (Yu et al., 2025) | Math | Deduplicated DAPO dataset |
| **Big-Math-RL-Verified** (Albalak et al., 2025) | Math | Subset filtered by pass rate 0.3–0.7 (for Llama-3.1-8B) |
| **AIME (1983–2023)** | Math | Historical AIME competition problems |

### 4.2 Evaluation datasets

| Dataset | Description |
|---|---|
| **MATH-500** | 500 held-out math problems |
| **AIME 2024** | AIME 2024 problems |
| **AIME 2025** | AIME 2025 problems |
| **AMC** | AMC competition problems |
| **Reasoning Gym** (held-out sets per level) | Synthetic test sets |

### 4.3 Base models

- Qwen2.5-Math-7B
- Qwen3-4B-Base (Reasoning Gym experiments)
- Qwen3-14B-Base
- Llama-3.1-8B-Instruct
- Deepseek-Math-7B-Instruct

---

## 5. Evaluation metrics and main results
### 5.1 Metrics

- **avg@k (Mean@k):** mean correctness over k samples (multi-sample variant of pass@1).
- **majority@k (Maj@k):** correctness of the majority-vote answer over k samples; reflects self-supervision quality.
- **Pass@1:** single-sample correctness.
- **Training pseudo-reward:** the self-reward signal during SRT training.
- **KL divergence:** token-level mean KL between the current policy and the base model.

### 5.2 Main results

#### Result 1: SRT can exceed the base model (Takeaway 1)

**Synthetic tasks (Figure 2):**
- On Family Relationships (Level 5), Bitwise Arithmetic (Level 3), Knights & Knaves (Level 7), SRT improves both avg@16 and majority@16.
- Evolving teacher beats fixed teacher significantly: Bitwise Arithmetic +10%, Family Relationship +8%, Knights & Knaves +6%.

**Real-world math tasks (Figure 3, 4):**
- On 4 different base models, SRT beats the base model on both MATH-500 Pass@1 and Majority@32.
- SRT performance is on par with standard RLVR using ground-truth rewards.
- Average accuracy of Llama-3.1-8B-Instruct improves from 52.6% to nearly 60%.

**Comparison with offline methods (Table 1):**

| Method | MATH-12K | DAPO |
|---|---|---|
| Base Model | 0.15 | 0.15 |
| SFT (majority vote) | 0.18 | 0.18 |
| DPO | 0.23 | 0.21 |
| ScPO | 0.20 | 0.20 |
| **SRT** | **0.32** | **0.31** |
| RL on Ground Truth | 0.33 | 0.36 |

SRT clearly beats SFT/DPO/ScPO-style offline variants and approaches ground-truth RL.

#### Result 2: Curriculum self-training supports multi-level improvement (Figure 5)

- On Bitwise Arithmetic, climbs from Level 2 (ground-truth RL) → Level 3 → Level 4 successfully.
- On Knights & Knaves, starting from ground-truth training only at the easiest Level 2, the SRT curriculum climbs to Level 9 with near-100% accuracy.

#### Result 3: Prolonged SRT causes model collapse (Takeaway 2)

**Figure 6:** on 4 base models, extended SRT training exhibits a rise-then-collapse curve — initial performance matches ground-truth RL, then **collapses entirely to near 0%**.

**Ablation (Figure 8):**
- GRPO vs. RLOO: no significant difference; both collapse.
- Increasing the KL coefficient: cannot prevent reward hacking (reward signal too strong).
- Lowering the learning rate: delays but does not eliminate collapse.
- **Reducing generations per prompt** (e.g., n=4 vs. n=32): injects noise into majority vote and surprisingly **delays** collapse — a counter-intuitive but informative finding.

### 5.3 Key findings summary

| Finding | Description |
|---|---|
| **Self-improvement works** | SRT beats the base model on both synthetic and real-world reasoning tasks |
| **Self-supervision quality improves** | Evolving-teacher majority-vote accuracy rises with training, forming a positive cycle |
| **Comparable to RLVR** | SRT matches ground-truth RL across multiple model+dataset combos |
| **Curriculum is feasible** | A simple level-by-level curriculum can climb multiple difficulties without labels |
| **Long-term collapse is inevitable** | Prolonged SRT inevitably leads to reward hacking → model collapse |
| **Feedback design is the core challenge** | The paper calls for future research on more robust verification for continual self-improvement |
