# MACA: Self-Improvement of Language Models by Post-Training on Multi-Agent Debate

> **Added to survey on:** 2026-03-11

**Paper:** Self-Improvement of Language Models by Post-Training on Multi-Agent Debate
**Authors:** Ankur Samanta, Akshayaa Magesh, Runzhe Wu, Ayush Jain, Youliang Yu, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani (Meta AI, Columbia University, Cornell Tech, Meta Superintelligence Labs)
**ArXiv:** 2025 (February 2, 2026)
**Code:** https://github.com/facebookresearch/maca
**Citation:** `samanta2025maca`

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| MACA | Pref. Opt. | training-time | Traj. |

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

MACA's core supervision signal comes entirely from the model's own multi-agent debate outputs. Specifically, multiple clones of the same model engage in multi-round debate exchanging chains-of-thought, and the consensus answer $\hat{a}(x)$ is decided by majority voting. Each agent's final-round trajectory is then partitioned into a **consensus-supporting** set $G^+(x)$ and a **dissenting** set $G^-(x)$ based on whether its final answer matches the majority, yielding preference pairs.

These preference pairs require no external ground-truth labels or human annotation — they are preference targets the model synthesizes autonomously via internal deliberation. Preference learning (MV-DPO / MV-KTO) discriminates majority from minority traces over full reasoning trajectories, so the model internalizes the subtle differences between stable consensus reasoning and deviant reasoning. Hence it belongs to the "Preference optimization → internally generated preference pairs" subclass of Family III.

---

## 2. Problem Addressed

- **Inconsistency in LLM reasoning:** under exploratory high-temperature sampling, LLMs produce contradictory answers to the same question, lacking an intrinsic mechanism to align diverse reasoning paths to a consistent answer.
- **Limits of inference-time methods:** self-consistency prompting and multi-agent debate boost accuracy only at inference via multi-sample / multi-agent coordination — no parameter updates, no persistent capability gain, and extra inference compute.
- **Single-round majority-vote signal is insufficient:** prior work (TTRL, ScPO) uses only single-round independent-sample majority vote for RL training; when small-model self-consistency is poor, noisy aggregation amplifies errors.
- **Ground-truth annotation does not scale:** external labels are costly and cannot scale to new domains or large data.

MACA uses multi-agent debate to produce richer consensus signals than single-round majority vote, and uses RL post-training to internalize them — simultaneously improving single-agent reasoning quality, multi-agent collaboration, and self-consistency.

---

## 3. Method

### 3.1 Formalizing self-consistency

Given prompt $x$, a model $\pi_\theta$ samples at temperature $\tau$, producing answer distribution $P_{\theta,\tau}(a|x)$. Define majority probability:

$$S^+_{\theta,\tau}(x) = P_{\theta,\tau}(a^*_{\theta,\tau}(x)|x)$$

with $a^* = \arg\max_a P_{\theta,\tau}(a|x)$ the majority answer. A self-consistent model maintains high $S^+$ even at high temperature.

In practice, estimate sampling consistency via $t$ independent samples:

$$s^{\theta,\tau}_t(x) = \frac{1}{t}\sum_{i=1}^t \mathbf{1}[a_i(x) = \hat{a}(x)]$$

For multi-agent debate, define a similar agreement metric $d^{\theta,\tau}_M(x)$.

### 3.2 Multi-agent debate procedure

1. **Clone:** replicate the same base LM into $M=3$ agents
2. **Initial round:** each agent independently produces an initial reasoning trajectory $y_m^{(1)} \sim \pi_{\theta_m}(\cdot|x)$
3. **Debate rounds** ($R=2$, i.e., 1 deliberation round): each agent sees the previous-round reasoning of all other agents and produces an updated answer conditioned on $x_m^{(r)} = [x; \{y_j^{(r-1)}\}_{j\neq m}]$
4. **Consensus extraction:** take final-round answers $a_m = A(y_m^{(R)})$ and decide $\hat{a}(x)$ via majority vote

### 3.3 Preference-data construction

Group final-round trajectories by majority consensus:
- **Consensus-supporting:** $G^+(x) = \{y \in Y(x) : A(y) = \hat{a}(x)\}$ (majority traces, treated as preferred)
- **Dissenting:** $G^-(x) = \{y \in Y(x) : A(y) \neq \hat{a}(x)\}$ (minority traces, treated as not preferred)

Form post-training dataset: $D_{\text{post}} = \{(x, \hat{a}(x), G^+(x), G^-(x))\}_{x \in D}$

### 3.4 Four post-training objectives

**MV-SFT:** directly imitate consensus-supporting trajectories
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{y^+ \in G^+(x)}[\log \pi_\theta(y^+|x)]$$

**MV-GRPO:** online sampling + consensus-based reward $r_x(y) = \mathbf{1}[A(y) = \hat{a}(x)]$, with group-normalized advantage
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{y \sim \pi_\theta}\left[\tilde{A}_x(y) \sum_t \log \pi_\theta(y_t|x, y_{<t})\right] + \lambda \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**MV-DPO** (best method): standard DPO on debate-derived preference pairs
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{(y^+, y^-) \in G^+ \times G^-}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \beta \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)\right]$$

**MV-KTO:** unpaired form handling positives/negatives separately, suited to majority-dominant imbalances

### 3.5 Key design choices

- **4-bit quantization + QLoRA:** trains 3 concurrent agents on a single multi-GPU node
- **Token limit = 256:** tests the model in a tight reasoning budget (gains transfer to 512-token tests)
- **Temperature $\tau=1.0$:** high-temperature exploratory sampling
- **Peer-context conditioning:** during DPO/KTO training, peer agents' chains-of-thought are part of the prompt context, teaching the model to make effective use of the debate context
- **Iterative training:** supports multiple debate→post-training iterations; most gains land in the first iteration (It-0→It-1), but subsequent iterations still add 1–3pp

### 3.6 Relation to TTRL/ScPO

TTRL and ScPO are special cases of MACA — multi-agent debate parameters degenerate to single-round majority vote:
- TTRL ≈ single-round R0 MV-GRPO
- ScPO ≈ single-round R0 MV-DPO (without the weighted loss)

MACA's advantage is using multi-round deliberation traces as conditioning context, so the model learns not just the correct aggregate output but internalizes the deliberative process that produces consensus.

---

## 4. Datasets

| Domain | Dataset | Description |
|------|--------|------|
| Math reasoning | MATH | 12,500 high-school math problems (algebra, geometry, combinatorics, number theory); multi-step reasoning |
| Math reasoning | GSM8K | 8,500 grade-school math word problems; arithmetic and logic |
| Math reasoning | MathQA | 37,000+ quantitative reasoning QA; NL → math expression |
| Math reasoning | SVAMP | Arithmetic word problems (rewritten to avoid artifacts); test only |
| Math reasoning | AMC 2023 | American Mathematics Competition; test only (40 problems) |
| Science reasoning | GPQA | 448 graduate-level physics/chemistry/biology multi-choice; test only |
| Commonsense reasoning | CommonsenseQA (CSQA) | 12,247 commonsense multi-choice; test only |

**Train/test splits:** MATH, GSM8K, MathQA each have 1500/500/500 train/valid/test splits; SVAMP (300), GPQA (448), CSQA (500), AMC (40) are OOD test only.

---

## 5. Evaluation metrics and main results
### 5.1 Metrics

- **Single-agent accuracy:** single-agent greedy decoding or Pass@t / MV@t ($t$ trajectories with majority vote)
- **Multi-agent debate accuracy:** majority vote of 3 agents' final-round answers after 2 debate rounds
- **Self-consistency** $s_t^{\theta,\tau}(x)$: fraction of $t$ samples matching the majority answer ($\tau=1.0$, $t=20$)
- **Agreement metric** $d_M^{\theta,\tau}(x)$: agreement among agents in multi-agent debate

### 5.2 Main results

#### Multi-agent debate setting (Table 1)

| Model | Dataset | Base Debate | MV-SFT | MV-GRPO | MV-KTO | MV-DPO | Best Δ |
|------|--------|------------|---------|----------|---------|---------|--------|
| Qwen-2B | MATH | 32.40 | 37.07 | 39.00 | **46.47** | 42.60 | **+14.07** |
| Qwen-2B | GSM8K | 49.60 | 50.53 | 54.13 | **63.07** | 58.47 | **+13.47** |
| Qwen-2B | MathQA | 24.20 | 26.27 | 29.93 | **32.60** | 28.33 | +9.13 |
| Llama-3B | MATH | 37.80 | 35.33 | 48.33 | **52.93** | 51.93 | **+15.27** |
| Llama-3B | MathQA | 21.60 | 40.07 | 48.73 | 64.00 | 63.13 | **+42.73** |
| Llama-8B | MATH | 32.80 | 34.13 | 45.93 | 53.93 | **59.67** | **+26.87** |
| Llama-8B | MathQA | 44.60 | 44.13 | 57.27 | 62.00 | **69.27** | **+24.67** |

#### Single-agent zero-shot (Table 2)

| Model | Dataset | Base | MV-SFT | MV-GRPO | MV-KTO | MV-DPO | Best Δ |
|------|--------|------|---------|----------|---------|---------|--------|
| Qwen-2B | MATH | 7.67 | 11.51 | 18.09 | 20.18 | **23.49** | **+15.82** |
| Qwen-2B | GSM8K | 23.00 | 24.84 | 34.40 | **45.13** | 43.87 | **+22.71** |
| Llama-3B | MathQA | 23.87 | 23.44 | 30.09 | 42.84 | **45.00** | **+21.13** |
| Llama-8B | MATH | 22.93 | 23.16 | 29.66 | 39.42 | **46.00** | **+23.07** |
| Llama-8B | GSM8K | 57.93 | 42.09 | 62.45 | 72.36 | **77.36** | **+19.43** |

#### OOD generalization (Table 3, MV-DPO, joint "All" training)

| Model | Train set | SVAMP | GPQA | CSQA |
|------|--------|-------|------|------|
| Qwen-2B | All | **+27.7** | **+16.3** | **+59.6** |
| Llama-3B | All | +7.1 | **+10.7** | +11.0 |

#### Debate vs. Ground-Truth (Table 4, Llama-8B)

Comparing debate-majority-vote (DMV) labels vs. ground-truth (GT) labels: DMV **matches or exceeds** GT in most settings, e.g., MV-DPO single-agent MATH: GT=45.13 vs. DMV=46.40; multi-agent GSM8K: GT=81.60 vs. DMV=83.00.

#### Debate RL vs. Single-round RL (Table 5)

| Method | Qwen-2B MATH | Llama-3B MATH | Phi-4B MATH | Llama-8B MATH |
|------|-------------|---------------|-------------|---------------|
| TTRL | +18.0±2.9 | +5.3±5.7 | +6.1±2.1 | +7.5±0.2 |
| ScPO | +2.3±1.1 | +3.4±0.2 | +0.1±0.5 | +3.7±0.6 |
| **MV-DPO (MACA)** | **+16.7±0.4** | **+12.5±0.7** | **+6.9±0.2** | **+17.1±0.8** |

MACA MV-DPO beats ScPO on all 8 model-dataset combos and beats TTRL on 6/8 (the other 2 within std).

### 5.3 Self-consistency gains

- Self-consistency curves stay up to **27.6 points** above baseline (GSM8K)
- Self-consistency correlates strongly with accuracy ($r > 0.86$ across all reasoning conditions)
- Post-trained responses are **22–36% shorter** than the base model — preference learning implicitly serves as a format reward

### 5.4 Multi-agent agreement improvements

Post-training (Qwen-2B on GSM8K):
- Full agreement (3/3) jumps from **13.4% → 43.4%** (3×)
- Unparseable responses drop from **11% → 0.6%**
- No-consensus (0/3) drops from 45.6% to 19.8%

### 5.5 Key findings

1. **Preference learning > scalar reward > imitation learning:** MV-DPO/MV-KTO > MV-GRPO > MV-SFT; contrastive comparisons over full reasoning trajectories solve credit assignment better. MV-KTO is best on small models (≤3B); MV-DPO is best on larger models (4–8B).
2. **Debate signal > single-round majority vote:** multi-round debate produces richer consensus signal via deliberative exchange than independent sampling; post-training on debate context teaches the model to leverage peer context.
3. **Self-supervised ≈ ground-truth supervision:** debate-derived majority-vote labels match or exceed oracle ground-truth labels in nearly all settings — no external annotation needed.
4. **Improvements come from reasoning quality, not formatting:** decomposition analysis shows **69–100%** of the gain comes from better reasoning rather than avoiding truncation or formatting improvements.
5. **Strong OOD generalization:** training on math data generalizes to science reasoning (GPQA +16.3%) and commonsense reasoning (CSQA +59.6%).
6. **4-bit training transfers to full precision:** gains on 4-bit quantized models transfer directly to full-precision models.
7. **Compute efficiency:** MACA MV-DPO uses 0.73–1.68 GPU hours — comparable to ScPO but far better performing, well below TTRL (2.2–7.7 GPU hours) and more stable.
8. **Iterative training adds diminishing but real gains:** It-1 → It-2/It-3 typically adds 1–3pp; 23/24 settings reach their best at later iterations.
