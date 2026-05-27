# DEBATE, TRAIN, EVOLVE: Self-Evolution of Language Model Reasoning

> **Added to survey on:** 2026-03-11

> **Authors**: Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang
> **Affiliation**: Virginia Tech, Department of Computer Science
> **Link**: arXiv:2505.15734v2 [cs.CL] 30 Sep 2025
> **Code**: https://github.com/ctrl-gaurav/Debate-Train-Evolve

| Property | Value |
|---|---|
| Method | DTE |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Traj. |

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

DTE belongs to **Family III — Self-Generated Target Bootstrapping**, subclass **reasoning / plan / curriculum synthesis**.

- **No external ground truth:** DTE's design principle is ground-truth-free training; signals come from model-generated multi-agent debate traces, with no human annotation or external teacher model.
- **Self-generated target construction:** through a multi-agent debate (where every agent is the same model) that converges to a consensus answer, debate traces yield a consolidated rationale that is used as the synthetic training trajectory.
- **Direct optimization via GRPO:** training uses Group Relative Policy Optimization (GRPO) to optimize the policy of a single model with a shaped reward built from debate consensus and format/length terms — no value network. Carrier is Direct Opt., Level is Traj. (full reasoning trajectory is the optimization unit).
- **Self-evolving bootstrapping loop:** the evolved model replaces the original agent in the next debate round, forming iterative self-evolution — consistent with self-generated-target bootstrapping.

---

## 2. Problem Addressed

### 2.1 Core question

> **Can the reasoning insights of multi-agent debate (MAD) be distilled into a single model so that one forward pass approaches or exceeds the multi-agent ensemble?**

### 2.2 Limitations of existing approaches

1. **Data saturation:** LLM reasoning gains have long relied on scaling training data, but the supply is approaching a bottleneck.
2. **Confirmation bias in single-model self-evolution:** existing self-refine / self-instruct methods rely on a single model or teacher-student feedback, prone to confirmation bias with limited reasoning diversity.
3. **High inference cost of MAD:** multi-agent debate boosts accuracy but requires running multiple model instances per query, with prohibitive compute and latency for large-scale deployment.
4. **Debate quality decay on small models:** standard MAD prompting exhibits two flaws on smaller models:
   - **Sycophancy:** agents abandon correct answers and adopt confident but wrong peer answers — sycophancy rate >28% on small models.
   - **Verbosity bias:** agents prefer longer rationales regardless of logical validity.

---

## 3. Method

DTE consists of three stages: **DEBATE → TRAIN → EVOLVE**, forming an iterative self-evolution loop.

### 3.1 Overall framework (Figure 1)

> **Figure 1:** three panels.
> - **Left (Debate):** multiple agents debate around a query until they reach consensus (green ✓) or expose wrong paths (red ✗).
> - **Center (Train):** strip pure debate elements (greetings, repetitions) and retain high-quality reasoning traces and consensus answers for GRPO fine-tuning of a single policy.
> - **Right (Evolve):** the evolved agent replaces its earlier version; subsequent inference uses one forward pass but beats the original agent committee on math, science, and commonsense benchmarks.

### 3.2 REFLECT-CRITIQUE-REFINE (RCR) prompting strategy

To address sycophancy and verbosity in standard MAD, DTE introduces a three-stage RCR prompt:

1. **Reflect:** each agent $a_i$ must generate a self-critique $c_i^{\text{self}}$ of its own reasoning $r_i^{(t-1)}$, identifying possible errors.
2. **Critique:** the agent then evaluates exactly two peer rationales, producing critiques $\{c_i^j\}_{j \in P_i}$ with $|P_i| = 2$. The fixed critique quota prevents runaway verbosity.
3. **Refine:** the agent updates its answer $(y_i^{(t)}, r_i^{(t)})$ under the key constraint that if $y_i^{(t)} \neq y_i^{(t-1)}$, $r_i^{(t)}$ must contain at least one novel reasoning step not appearing in any previous agent answer. This forces "think" before "copy" and effectively reduces sycophancy.

**Termination:** debate ends when all agents agree (consensus) or the maximum round $T$ is reached; the final answer is taken by majority vote.

**Effect of RCR (Figure 3):**
> **Figure 3:** bars compare 8 models on GSM8K, GSM-Plus, ARC-Challenge across three settings (single model, Original MAD@3, RCR-MAD@3). RCR improves GSM-Plus by +3.7 pts on average, reduces the mean sycophancy rate from 0.28 to 0.13, and shrinks the verbosity gap by 43%.

### 3.3 Training stage: GRPO with debate-derived rewards

#### Debate-trace extraction and reward design

The debate yields a consolidated rationale $R$ comprising (i) reasoning steps that gain agent consensus and (ii) steps introducing novel symbolic manipulation. Each training sample is $(q, y^*, R)$.

Shaped reward:

$$r(q, y) = w_{\text{ans}} \cdot \mathbb{1}[y = y^*] + w_{\text{fmt}} \cdot f_{\text{format}}(y) + w_{\text{len}} \cdot \exp(-|y| / \tau)$$

with $(w_{\text{ans}}, w_{\text{fmt}}, w_{\text{len}}) = (2.0, 0.5, 0.5)$, $\tau = 120$. The three terms reward answer correctness (exact string match), format compliance (XML template), and brevity.

#### Group Relative Policy Optimization (GRPO)

For each query $q$, sample $G$ responses $\{o_1, \ldots, o_G\}$ from the current policy $\pi_{\theta_{\text{old}}}$, then compute group-relative advantage:

$$\hat{A}_{i,t} = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

with $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$ and $\sigma_r$ the group std. The benefit: no value network needed; advantage is estimated by intra-group comparison, saving memory.

GRPO loss:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\ell_{\text{clip}}(i, t) - \beta \cdot D_{\text{KL}}^{(i,t)}\right]$$

with clipping threshold $\epsilon = 0.2$, KL coefficient $\beta = 0.02$. Clipping prevents overly large policy updates; the KL penalty anchors the policy to the reference model to preserve language coherence and avoid catastrophic forgetting.

#### Training hyperparameters

- Optimizer: AdamW, $\eta = 5 \times 10^{-6}$, weight decay $\lambda = 0.1$
- LoRA: rank $r = 128$, dropout $p = 0.05$, applied to attention and MLP projection matrices
- Group size $G = 8$
- Each evolution epoch processes ~8K debate traces (~2M tokens)
- Hardware: A100-80GB GPU

### 3.4 Evolution through iterative training

Full DTE pipeline (Algorithm 2):

1. **Debate generation:** sample query batch $Q_k$, use RCR-prompted MAD to generate debate traces, build training set $D_k = \{(q, y^*, R)\}$.
2. **Policy update:** fine-tune $\pi_{\theta_{k-1}}$ on $D_k$ via GRPO to get $\pi_{\theta_k}$.
3. **Agent replacement:** replace the older version in the debate ensemble with the evolved policy.
4. **Termination:** validation plateau or maximum iterations.

**Temperature annealing for small models:** for models <3B params, use $T = 0.7$ at Round 1, linearly decayed to $T = 0.3$ in later rounds — eases KL drift and forgetting. Dropping $T$ from 0.7 to 0.3 cuts KL drift by 1/3 and recovers up to 76% of the lost performance.

### 3.5 Agent configuration

Debate phase samples once per agent per query:
- Temperature: $T = 1.0$ (exploratory) or $T = 0.0$ (deterministic)
- Mixed team: 1 exploratory + 2 deterministic agents

---

## 4. Datasets

### Training data
- **GSM8K** (Cobbe et al., 2021): grade-school math word problems
- **GSM-Plus** (Li et al., 2024): adversarial GSM8K variant, harder

### Evaluation benchmarks (7 in total)

| Dataset | Domain | Description |
|---|---|---|
| GSM8K | Math | Grade-school word problems |
| GSM-Plus | Math | Adversarial math, harder |
| MATH | Math | Competition math (Hendrycks et al., 2021) |
| ARC-Easy | Science reasoning | Easier multi-choice science |
| ARC-Challenge | Science reasoning | Harder multi-choice science (Clark et al., 2018) |
| GPQA Main | STEM | Graduate-level STEM questions (Rein et al., 2024) |
| CommonsenseQA | Commonsense | Commonsense QA (Talmor et al., 2019) |

### Experimental models
- **Qwen-2.5** family: 0.5B, 1.5B, 3B, 7B, 14B
- **Llama** family: Llama-3.2-3B, Llama-3.1-8B
- **Others** (for RCR study): Mistral-7B, Phi-mini, GPT-4o, GPT-4o-mini

---

## 5. Evaluation metrics and main results
### Metrics
- **Exact match accuracy**: GSM-style datasets use normalized exact string match
- **MC-QA accuracy**: multi-choice datasets use option-match accuracy
- **Sycophancy rate**: fraction of debates where an agent switches from a correct to a wrong answer
- **[incorrect → correct] rate**: fraction where the answer flips from wrong to correct

### 5.1 Main results (Table 1)

After one DTE round:

| Model | GSM-Plus original | GSM-Plus DTE-evolved | Δ |
|---|---|---|---|
| Qwen-2.5-1.5B | 42.00 | 55.92 | **+13.92** |
| Qwen-2.5-3B | 61.75 | 69.50 | +7.75 |
| Qwen-2.5-7B | 68.62 | 74.71 | +6.09 |
| Qwen-2.5-14B | 71.79 | 78.88 | +7.09 |
| Llama-3.2-3B | 45.67 | 53.79 | +8.12 |
| Llama-3.1-8B | 55.62 | 66.17 | **+10.55** |

**Key findings:**
- **GSM-Plus average gain 8.92%**, all models improved.
- The evolved single model beats the 3-agent MAD by +2.38 pts on average — single-model inference beats the ensemble.
- The smallest model (Qwen-1.5B) gains most (+13.92 pts) due to the largest headroom and diverse traces from debate.
- On ARC-Challenge, larger models benefit most (Llama-8B +8.88 pts); small models stay within ±1 pt.

### 5.2 Cross-domain generalization (Table 2)

Trained on GSM8K, tested on unseen datasets:

| Model | GSM-Plus Δ | ARC-Easy Δ | ARC-Ch. Δ | CSQA Δ |
|---|---|---|---|---|
| Qwen-2.5-7B | +1.01 | +1.73 | +4.50 | +3.40 |
| Qwen-2.5-14B | +1.67 | +2.53 | +3.42 | +1.33 |
| Llama-3.1-8B | +8.13 | +3.91 | +6.74 | +1.10 |

**Key findings:**
- Average cross-domain gain +5.8 pts — DTE learns general reasoning heuristics (numeric decomposition, unit tracking) rather than dataset-specific patterns.
- ≥7B models do not lose more than 0.2 pts on any transfer task.
- Negative transfer only occurs on the smallest model (Qwen-1.5B).

### 5.3 Iterative evolution analysis (Figure 2)

> **Figure 2:** line chart of accuracy across Original → Round 1 → Round 2 for 5 models on GSM8K and GSM-Plus. Round 1 helps almost all models; Round 2 yields little additional gain or slight degradation.

- **Round 1 captures nearly all available gain.**
- Round-2 mean forgetting $F_{\text{gt2}} = \max_{t<2}(\text{Acc}_t - \text{Acc}_2)$: 0.92 pts for ≥7B, 1.6 pts for <3B.
- Small models are more susceptible to catastrophic forgetting.

### 5.4 Key ablations

#### (1) Number of agents (Figure 4)
> **Figure 4:** four curves (Qwen 1.5B/3B/7B/14B) show accuracy vs. number of agents from 1 to 7 on GSM8K, GSM-Plus, ARC-Easy, ARC-Challenge.
- **3 agents capture 85–95% of the maximum gain.**
- Harder tasks (GSM-Plus) prefer slightly more agents (4–5).
- Easy tasks (ARC-Easy) saturate at 2 agents.

#### (2) Optimization comparison (Table 3)

| Model | Original | SFT | DPO | GRPO |
|---|---|---|---|---|
| Qwen-2.5-1.5B | 42.00 | 47.31 | 51.34 | **55.92** |
| Qwen-2.5-3B | 61.75 | 58.33 | 64.32 | **69.50** |
| Qwen-2.5-7B | 68.62 | 67.89 | 69.88 | **74.71** |

GRPO consistently beats SFT and DPO across model sizes; its KL divergence < 0.24 stays well below DPO's 0.43, simultaneously raising reward and constraining policy drift.

#### (3) Data-selection strategy (Table 4)

| Strategy | Qwen-1.5B | Qwen-3B | Qwen-7B |
|---|---|---|---|
| Random-2K | 44.82 | 58.10 | 69.71 |
| Debate-Only | 51.61 | 62.70 | 72.53 |
| All-Traces | **55.92** | **69.50** | **74.71** |

Using all traces works best — All-Traces beats Debate-Only by +4.43 pts and Random-2K by +9.17 pts on average. Small models particularly benefit from "round-0" (uncontested) easy examples.

#### (4) Training steps (Figure 5)
> **Figure 5:** GSM-Plus accuracy vs. GRPO training steps (2K–10K). All models saturate around 8K steps; 8K→10K adds only +0.32 pts on average.

#### (5) Temperature annealing and catastrophic forgetting (Figure 6)
> **Figure 6:** Qwen-1.5B accuracy at Round 1 and Round 2 across sampling temperatures (1.0, 0.7, 0.4, 0.0).
- At $T = 1.0$, Round 2 loses 2.0 pts (GSM8K) with clear catastrophic forgetting.
- At $T = 0.4$, Round 2 stays within 0.9 pts of Round 1.
- KL divergence: $T = 1.0 \to 0.37$; $T = 0.4 \to 0.19$; $T = 0.0 \to 0.11$.

#### (6) Agent diversity
- When individual agent accuracy is comparable, cross-family mixtures (e.g., Qwen + Llama) beat homogeneous teams — architectural diversity yields complementary reasoning paths.
- When mixing strong and weak models, debate gravitates to the strong model — the weak agent neither helps nor hurts much.

### 5.5 Limitations

1. Iterative fine-tuning still causes catastrophic forgetting on models <3B params; not fully eliminated.
2. The framework assumes initial debate traces have reasonable quality; weak initial agents may degrade results.
3. Mainly validated on structured reasoning tasks (math, commonsense); effects on open-ended generation or dialogue remain to be studied.
4. Although more deployment-efficient than MAD, training cost is still higher than standard single-model fine-tuning.
