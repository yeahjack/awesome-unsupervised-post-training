# Intuitor: Learning to Reason without External Rewards

> **Added to survey on:** 2026-03-11

**Paper:** Learning to Reason without External Rewards (arXiv: 2505.19590)
**Authors:** Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, Dawn Song (UC Berkeley, Yale)
**Venue:** ICLR 2026
**Method:** Intuitor | **Carrier:** Policy Opt. (GRPO) | **Regime:** Training-time | **Level:** Token

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / rollout group |
| Persistence | full parameter accumulate across RL steps |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates fire during an offline RL stage before deployment; the basic unit is a group of rollouts per prompt batch.
- **Whose sample it serves:** the current rollout group's update serves subsequent training steps and the final deployed model, not the immediate inference of any single sample.
- **Whether parameters/state accumulate:** parameters accumulate across RL training; no per-sample reset.
- **Reset boundary:** it is an offline RL-style UPT schedule, not test-time arrival-by-arrival adaptation.

## 1. UPT Assignment Rationale

Intuitor belongs to **Family II (Sample-Relation Supervision)**, specifically the self-certainty sub-class under trajectory-level consistency.

The core intrinsic signal is **self-certainty**: at each token position in the sequence, compute the forward KL divergence from the uniform distribution $U$ to the model's next-token distribution $p_{\pi_\theta}$ (i.e. $\text{KL}(U \| p_{\pi_\theta})$), then average over all tokens to obtain a sequence-level intrinsic reward. Higher self-certainty means the model is more "confident" in its own outputs. This signal comes entirely from the model's own token distribution, with no reliance on ground truth, verifiers, or human annotation — fits the UPT definition.

Key features:
- The reward signal is defined at the token level and aggregated to the sequence level for use.
- Forward KL (mode-seeking) is used; compared with entropy (mode-covering), it is less biased toward long sequences.
- Online annotator: self-certainty is computed online by the current policy and evolves with it, preventing reward hacking.

---

## 2. Problem Addressed

Existing RL methods for LLM reasoning face two bottlenecks:

1. **RLHF** depends on large amounts of human-annotated preference data — costly and biased.
2. **RLVR** (e.g. exact-answer matching used by DeepSeek-R1) requires domain-specific gold-standard answers or verifiers, limiting applicability to open-ended tasks.

Intuitor's core question: **can an LLM improve its reasoning ability using only intrinsic signals (no external reward, no ground truth)?** To this end, the authors propose the Reinforcement Learning from Internal Feedback (RLIF) paradigm.

---

## 3. Method

### 3.1 RLIF framework

Optimization objective:

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_\theta(q)} \left[ u(q, o) - \beta \text{KL}[\pi_\theta(o|q) \| \pi_{\text{ref}}(o|q)] \right]$$

where $u(q,o)$ is the intrinsic signal (no external reward) and $\beta$ controls KL regularization strength.

### 3.2 Self-certainty definition

$$\text{Self-certainty}(o|q) := \frac{1}{|o|} \sum_{i=1}^{|o|} \text{KL}(U \| p_{\pi_\theta}(\cdot|q, o_{<i})) = -\frac{1}{|o| \cdot |\mathcal{V}|} \sum_{i=1}^{|o|} \sum_{j=1}^{|\mathcal{V}|} \log(|\mathcal{V}| \cdot p_{\pi_\theta}(j|q, o_{<i}))$$

- $\mathcal{V}$: vocabulary; $|o|$: sequence length.
- Higher values indicate more confident outputs.

### 3.3 Integration with GRPO

Built on GRPO: for each query $q$, sample $G$ candidate outputs and replace the external verifiable reward with self-certainty:

$$u_i = \text{Self-certainty}(o_i|q), \quad \hat{A}_{i,t} = \frac{u_i - \text{mean}(\{u_1, \ldots, u_G\})}{\text{std}(\{u_1, \ldots, u_G\})}$$

The normalized advantage is used for GRPO's clipped policy-gradient update. The whole pipeline requires no external supervision.

### 3.4 Online vs. offline annotator

Online self-certainty (computed by the current policy) is used rather than offline (computed by a fixed base model). Experiments show that the offline version is easy to exploit (the model learns to append a solved problem after the answer to inflate self-certainty); the online version, with the reward signal evolving alongside the policy, effectively prevents reward hacking.

---

## 4. Datasets

### Training data
- **MATH dataset** training set: 7,500 math problems (main experiments).
- **Codeforces** code-generation dataset: 3,200 problems (Intuitor-Code variant).

### Evaluation benchmarks
- **Math reasoning:** GSM8K, MATH500.
- **Code generation:** CRUXEval-O, LiveCodeBench v6 (LCB).
- **Instruction following:** AlpacaEval 2.0 (length-controlled win rate, GPT-4.1 judge).
- **Knowledge:** MMLU-Pro.

### Models
- Qwen2.5-1.5B, Qwen2.5-3B (main experiments).
- Qwen2.5-7B/14B, Qwen3-14B, Llama-3.2, OLMo-2 (ablation / extension experiments).

---

## 5. Evaluation metrics and main results

### Metrics
- Math: accuracy (greedy decoding).
- Code: pass rate.
- Instruction following: AlpacaEval length-controlled win rate.
- Knowledge: MMLU-Pro accuracy.

### Main results (Qwen2.5-3B, trained on MATH)

| Model | GSM8K | MATH500 | LCB | CRUX | MMLU-Pro | AlpacaEval |
|-------|-------|---------|-----|------|----------|------------|
| Base | 0.673 | 0.544 | 0.093 | 0.236 | 0.377 | 3.72 |
| + GRPO | 0.826 | 0.636 | 0.085 | 0.341 | 0.403 | 6.91 |
| + Intuitor | 0.792 | 0.612 | 0.153 | 0.416 | 0.379 | 7.10 |

### Key findings

1. **Comparable in-domain performance:** Intuitor's performance on GSM8K/MATH500 is close to GRPO (which uses gold answers) — slightly lower, but the gap is small.
2. **Better out-of-domain generalization:** Intuitor significantly beats GRPO on code generation (LCB: 0.153 vs 0.085, +65% relative; CRUX: 0.416 vs 0.341, +76% relative).
3. **Faster early learning:** after only 10 training steps, Intuitor already surpasses GRPO on GSM8K and MATH (Table 2).
4. **Emergent structured reasoning:** Intuitor-trained models spontaneously produce R1-style long-chain reasoning, generating pre-code reasoning automatically before JSON-format outputs.
5. **Better instruction following:** length-controlled win rate on AlpacaEval surpasses GRPO (7.10 vs 6.91).
6. **Reward-hacking resistant:** online self-certainty effectively prevents reward exploitation; training is stable.
7. **Cross-architecture robust:** effective on Llama-3.2, OLMo-2, and other model families.
