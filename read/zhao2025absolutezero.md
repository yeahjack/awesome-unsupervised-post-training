# Absolute Zero: Reinforced Self-play Reasoning with Zero Data

> **Added to Survey:** 2026-03-11

> **Audit Update (2026-03-11): Retained in the purple Tool-as-Verifier branch under the survey's `whole-pipeline` rule.**
>
> Caveat: the main training is indeed `#data = 0`, but the seeding stage uses one handcrafted zero triplet plus a proposer prompt template as a minimal starting scaffold. The survey treats this as a `small seed`, not as a large-GT-dataset dependency.

> **Method:** AZR (Absolute Zero Reasoner) | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Semantic
>
> arXiv 2505.03335 — Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang (Tsinghua University, BIGAI, Penn State)

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | self-play episode / synthesized task batch |
| Persistence | full parameter accumulate across self-play rounds |
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

- **When does the update fire:** updates fire in pre-deployment self-play / self-evolution rounds, not driven online by real deployed samples.
- **Serving the current sample or downstream samples:** tasks and trajectories produced in the current self-play episode primarily serve later training rounds and the eventual deployed model.
- **Whether parameters / state accumulate:** parameters accumulate across self-play rounds with no per-sample reset.
- **Reset boundary:** so this is an offline self-evolution schedule, not test-time cumulative adaptation.

## 1. UPT Assignment Rationale
Absolute Zero belongs to **Tool-Augmented Adjacent Extensions**, sub-direction **code executor as unified verifier**.

Core mechanism: a single model plays both Proposer (generating reasoning tasks) and Solver (solving them), forming a self-play loop. A **Python code executor** is the only external tool, validating both the Proposer's generated tasks (program integrity, safety, determinism) and the Solver's answers. The main training uses **no external data, ground truth, or human annotation**; only the seeding stage uses one handcrafted zero triplet and a prompt template as a minimal starting scaffold.

Difference from strict UPT: rewards do not come from internal model statistics (entropy, consensus, etc.) but from **deterministic feedback of the external code-execution environment**. However, the tool is generic and off-the-shelf, requiring no task-specific annotation.

---

## 2. Problem Addressed

- Conventional RLVR (e.g., DeepSeek-R1-Zero) does not annotate the reasoning process but still depends on human-curated QA datasets.
- High-quality human data is becoming scarce, capping long-term scalability.
- In a hypothetical superintelligence scenario, human-provided tasks may not deliver effective learning to a superintelligent system.

Absolute Zero asks: can a model both create and solve tasks, relying only on a code executor for grounded feedback, without any external data?

---

## 3. Method

### 3.1 Overall Framework

AZR builds three reasoning modes over (Program $P$, Input $I$, Output $O$) triplets:

| Mode | Given | Inferred | Meaning |
|------|-------|----------|---------|
| **Deduction** | $P, I$ | $O=?$ | given program and input, infer the output |
| **Abduction** | $P, O$ | $I=?$ | given program and output, infer the input |
| **Induction** | $\{I_n, O_n\}$ | $P=?$ | from input–output examples, induce a program |

### 3.2 Self-Play Loop

1. **Proposer** $\pi_\theta^{\text{propose}}$: generate task $\tau$ conditioned $z$ on the buffer's historical triplets.
2. **Code-executor verification:**
   - program integrity: run the program, check syntax and that it returns output;
   - program safety: detect dangerous library use;
   - determinism check: run independently $j$ times to ensure consistent output.
3. **Solver** $\pi_\theta^{\text{solve}}$: take the verified task $(x, y^*)$ and produce an answer.
4. **Dual reward:** Proposer earns the learnability reward; Solver earns the accuracy reward.
5. **Joint policy optimization:** both roles update jointly.

### 3.3 Core Equations

**Joint objective:**
$$J(\theta) := \max_\theta \mathbb{E}_{z \sim p(z)} \mathbb{E}_{(x,y^*) \sim f_e(\cdot|\tau), \tau \sim \pi_\theta^{\text{propose}}(\cdot|z)} \left[ \lambda r_e^{\text{propose}}(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|x)} [r_e^{\text{solve}}(y, y^*)] \right]$$

**Proposer reward (learnability):**
$$r^{\text{propose}} = \begin{cases} 0, & \text{if } \bar{r}^{\text{solve}} = 0 \text{ (unsolvable)} \\ 1 - \bar{r}^{\text{solve}}, & \text{otherwise} \end{cases}$$

with $\bar{r}^{\text{solve}} = \frac{1}{G}\sum_{i=1}^{G} r^{\text{solve}}(o_i)$ the Monte-Carlo rollout success rate. Tasks that are too hard ($\bar r=0$) or too easy ($\bar r=1$) earn low rewards; medium-difficulty tasks earn the highest—an adaptive curriculum.

**Solver reward:** $r^{\text{solve}} = \mathbb{I}(y = y^*)$.

**Task-Relative REINFORCE++ (TRR++):**
$$A^{\text{norm}}_{\text{task,role}} = \frac{r - \mu_{\text{task,role}}}{\sigma_{\text{task,role}}}$$

separate baselines per the 6 task–role configurations (3 task types × 2 roles).

### 3.4 Answer-Verification Details

- **Deduction:** direct match $o^\pi = o^*$.
- **Abduction:** verify $p(i^\pi) = p(i^*)$ (run the program to check that the input produces the same output).
- **Induction:** verify $\{p^\pi(i_n^*) = o_n^*\}_N$ (the induced program is correct on every test case).

---

## 4. Training Setup

| Item | Detail |
|------|--------|
| Base models | Qwen2.5-7B, Qwen2.5-7B-Coder (main experiments); 3B, 14B, Llama-3.1-8B |
| Training algorithm | Task-Relative REINFORCE++ (TRR++) (in-house) |
| Batch size | 64 × 6 = 384 (2 roles × 3 task types) |
| Learning rate | 1e-6 (constant), AdamW |
| External data | **0** (zero data) |
| Output format | `<think>` / `<answer>` (DeepSeek R1 format) |

---

## 5. Core Results

| Model | External data | Code Avg | Math Avg | Overall |
|-------|---------------|----------|----------|---------|
| Qwen2.5-7B (base) | – | 52.0 | 27.5 | 39.8 |
| AZR-Base-7B | **0** | 55.2 | 38.4 | **46.8** |
| AZR-Coder-7B | **0** | 61.6 | 39.1 | **50.4** |
| ORZ (best curated baseline) | 57k | 55.6 | 41.6 | 48.6 |

- With zero data, beats baselines that use tens of thousands of human samples.
- Scale effect: 3B +5.7, 7B +10.2, 14B +13.2.
- Strong cross-domain transfer: code training delivers +15.2 on math.

---

## 6. Position in the UPT Survey
**Tool-as-Verifier prototypicality:** Absolute Zero is the purest "tool as verifier" form—the code executor simultaneously plays:
(a) the task-factory inspector (validating the Proposer's tasks),
(b) the answer judge (validating the Solver's answers),
(c) an indirect curriculum-design signal (modulating task difficulty via the learnability reward).

**Complementary to T³RL:** T³RL uses the code interpreter at test time to augment majority-vote pseudo-labels, while Absolute Zero uses the code executor at training time to drive zero-data self-play. They represent tool-as-verifier at inference-stage vs. training-stage respectively.

**Relation to strict UPT:** AZR's self-play structure inherits features from self-generated-target bootstrapping (self-generated tasks) and sample-relation supervision (the learnability reward is based on the Solver-group's performance), but the ground truth comes entirely from the code executor rather than the model's internals; so it sits at the boundary between UPT and environment-grounded RL.
