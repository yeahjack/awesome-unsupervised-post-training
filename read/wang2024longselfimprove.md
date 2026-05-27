# Long Self-Improve: Large Language Models Can Self-Improve in Long-context Reasoning

> **Added to survey on:** 2026-03-11

**Paper:** Large Language Models Can Self-Improve in Long-context Reasoning
**Authors:** Siheng Li, Cheng Yang, Zesen Cheng, Lemao Liu, Mo Yu, Yujiu Yang, Wai Lam (CUHK, Peking University, Tsinghua University, Tencent)
**ArXiv:** 2411.08147
**Date:** 2024-11

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Long Self-Improve | Direct Opt. | training-time | Semantic |

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

**Family III — Self-Generated Target Bootstrapping (reasoning / curriculum synthesis)**

SEALONG belongs to the self-generated-target bootstrapping branch of UPT for the following reasons:

1. **No external annotation or expert model:** the whole pipeline uses no human-annotated answers and no outputs from stronger external models like GPT-4o. All training signals come from the model's own multi-sample outputs.
2. **Self-generation + internal-criterion filtering:** for each long-context question, the model samples N reasoning trajectories with temperature sampling, then scores each output via Minimum Bayes Risk (MBR) decoding — an internal criterion based on semantic consistency among outputs. Highly consistent outputs are more likely to be correct (low-consistency ones are more likely hallucinations).
3. **Filtered results become learning targets:** the top-scoring MBR output is used as the SFT target, or high/low pairs form preference pairs for ORPO preference optimization — a textbook "self-generated synthetic target → direct optimization" paradigm.
4. **Reasoning synthesis:** the generated target is a complete reasoning trajectory (stepwise reasoning, information location, answer derivation) — reasoning / plan / curriculum synthesis.

---

## 2. Problem Addressed

**Insufficient long-context reasoning and existing improvements all rely on external supervision.**

Concretely:
- LLMs already excel at long-context retrieval (e.g., needle-in-a-haystack) but underperform on long-context tasks requiring **multi-hop reasoning across documents**, with notable degradation.
- Existing long-context-reasoning improvements rely on two kinds of external supervision: (1) human expert annotation (expensive and unscalable); (2) advanced-model (e.g., GPT-4o) synthesized data (capped by the existing model's ability and unable to push the ceiling).
- The paper asks: **Can an LLM self-improve long-context reasoning without any external supervision?**

Two key motivations are observed:
1. **Importance of prompting strategy:** switching from default prompting to plan-and-solve prompting raises Llama-3.1-8B-Instruct on 2WikiMQA from 66.0 to 82.0, showing strong latent potential that is under-elicited.
2. **Oracle far above greedy search:** sampling 128 outputs per question yields >90% oracle accuracy (at least one correct), far above greedy search — indicating multi-sample sets contain many correct reasoning trajectories.

---

## 3. Method

SEALONG (**S**elf-improving method for r**EA**soning over **LONG** contexts) has two stages: self-supervision creation and fine-tuning.

### 3.1 Stage 1: Self-supervision creation

#### 3.1.1 Multi-output sampling

For each query-context pair, use **plan-and-solve prompting** and temperature = 0.7 to sample $N$ reasoning trajectories; default $N = 32$.

Prompt template:
> {context}
> {input}
> Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step-by-step.

#### 3.1.2 MBR scoring

Core idea: **correct reasoning trajectories are usually more semantically consistent** (similar planning steps, referencing the same context), while wrong trajectories are more likely hallucinations and inconsistent with the rest.

Use the Minimum Bayes Risk (MBR) framework to score each output. The score of $y$ is its expected utility under the model distribution:

$$s(y) = \mathbb{E}_{y^* \sim \pi_\theta(y|x)}[u(y, y^*)]$$

Monte Carlo approximation via $N$ sampled outputs:

$$s(y) \approx \frac{1}{N} \sum_{i=1}^{N} u(y, y_i^*)$$

**Utility $u(y, y^*)$** uses sentence-embedding similarity:

$$u(y, y^*) = \text{Sim}(\text{Emb}(y), \text{Emb}(y^*))$$

Embeddings are computed with a lightweight RoBERTa-based model (experiments use **jina-embeddings-v3**); similarity is the inner product.

The highest-scoring output is the **MBR decoding output** ($y_w$, preferred); a random low-score output is the less-preferred output ($y_l$).

#### 3.1.3 Scoring-method comparison (Tab. 7)

| Method | HotpotQA | MuSiQue | 2WikiMQA |
|------|----------|---------|----------|
| Greedy Search | 64.0 | 49.5 | 82.0 |
| Random | 61.0 | 50.5 | 79.5 |
| Reference-free Self-evaluation | 64.0 | 51.5 | 83.0 |
| MBR-ROUGE | 66.5 | 53.5 | 85.0 |
| MBR-BERTScore | 67.5 | 50.0 | 86.5 |
| Reference-based Self-evaluation | 63.5 | 51.5 | 84.5 |
| **MBR-Sentence Embedding** | **67.5** | **56.0** | **88.0** |

MBR-based methods clearly beat reference-free self-evaluation — current LLMs have limited self-evaluation ability in long contexts. Sentence embeddings inject richer semantic information and perform best.

### 3.2 Stage 2: Fine-tuning

Two fine-tuning strategies are provided:

#### 3.2.1 Supervised Fine-tuning (SFT)

Target the MBR decoding output, minimizing negative log-likelihood:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|} \log \pi_\theta(y|x) = -\frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i | x, y_{<i})$$

#### 3.2.2 Preference Optimization (ORPO)

Use **ORPO (Monolithic Odds Ratio Preference Optimization)** with the odds-ratio loss:

$$\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w | x)}{\text{odds}_\theta(y_l | x)} \right)$$

where:

$$\text{odds}_\theta(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$$

The final objective combines SFT and OR losses:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \beta \cdot \mathcal{L}_{\text{OR}}$$

with default $\beta = 0.1$. ORPO needs no reference model, suited for efficient long-context training.

### 3.3 Data-synthesis details

- **Source:** questions from the MuSiQue training set (multi-hop questions), without using annotated answers
- **Context construction:** mix relevant Wikipedia documents with randomly sampled irrelevant ones, shuffle and concatenate; context length randomly set in 4K–31K tokens
- **Default: 2048 synthesized training samples**
- $N = 32$ outputs per sample

### 3.4 Training details

- Sequence parallelization (parallel size = 8)
- QLoRA: rank = 128, alpha = 128, dropout = 0.05; target modules: all attention and feedforward linear layers
- 1 training epoch
- Batch size = 8, learning rate = 5e-5, max sequence length = 32K
- Hardware: 8 × H100 GPUs

---

## 4. Datasets

### Training data

| Domain | Dataset | Description |
|------|--------|------|
| Multi-hop QA | MuSiQue (train) | Questions only, no annotated answers; each question links to multiple Wikipedia documents, mixed with irrelevant ones to form long context (4K–31K tokens) |

### Evaluation data

| Domain | Dataset | Description |
|------|--------|------|
| Single-doc QA | Qasper | 200 examples, max 21,110 tokens, avg 4,921 tokens |
| Single-doc QA | MultiFieldQA-En | 150 examples, max 14,947 tokens, avg 6,888 tokens |
| Multi-doc QA | HotpotQA | 200 examples, max 16,322 tokens, avg 12,779 tokens |
| Multi-doc QA | MuSiQue | 200 examples, max 16,335 tokens, avg 15,542 tokens |
| Multi-doc QA | 2WikiMultihopQA | 200 examples, max 16,319 tokens, avg 7,096 tokens |
| Short-context | OpenLLM Leaderboard | MMLU, GSM8K, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA |

---

## 5. Evaluation metrics and main results
### Metric

- **Substring Exact Match (SubEM):** checks whether the model output contains the golden answer as a substring

### Main results

#### Long-context reasoning (Tab. 2)

| Model | Qasper | MultiField | HotpotQA | MuSiQue | 2WikiMQA | **Avg.** |
|-------|--------|------------|----------|---------|----------|----------|
| Qwen-2.5-7B-Instruct | 21.0 | 28.0 | 70.5 | 48.0 | 77.5 | 49.0 |
| + SEALONG | 26.0 | 29.3 | 72.5 | 51.5 | 79.5 | **51.8** |
| Qwen-2.5-14B-Instruct | 21.0 | 32.0 | 73.0 | 52.0 | 83.0 | 52.2 |
| + SEALONG | 24.0 | 30.0 | 75.0 | 57.0 | 87.5 | **54.7** |
| Llama-3.1-8B-Instruct | 29.0 | 29.3 | 64.0 | 49.5 | 82.0 | 50.8 |
| + SEALONG | 32.5 | 31.3 | 68.0 | 58.5 | 84.5 | **55.0** |
| Qwen-2.5-32B-Instruct | 24.5 | 26.0 | 72.0 | 55.0 | 88.0 | 53.1 |
| Qwen-2.5-72B-Instruct | 27.0 | 28.7 | 74.5 | 58.5 | 89.0 | 55.5 |
| Llama-3.1-70B-Instruct | 30.0 | 33.3 | 74.0 | 68.5 | 85.5 | 58.3 |
| GPT-4o | 21.5 | 28.0 | 74.5 | 64.0 | 84.0 | 54.4 |

**Key findings:**
- Llama-3.1-8B + SEALONG (55.0) **beats GPT-4o** (54.4), absolute +4.2
- Qwen-2.5-14B + SEALONG (54.7) **beats Qwen-2.5-32B** (53.1) — the 14B model, after self-improvement, exceeds a same-family model with double the params
- Qwen-2.5-7B + SEALONG (51.8) approaches Qwen-2.5-14B base (52.2)
- Although training data uses only MuSiQue, all other tasks (Qasper, MultiFieldQA-En, HotpotQA, 2WikiMQA) improve — strong **generalization**

#### Comparison with existing datasets (Tab. 5, Llama-3.1-8B-Instruct)

| Category | Dataset | Avg. |
|----------|--------|------|
| Baseline | Llama-3.1-8B-Instruct | 50.8 |
| SFT | TULU-V2-mix | 37.0 |
| SFT | WildChat | 36.5 |
| SFT | LongAlpaca | 35.6 |
| SFT | LongAlign | 48.7 |
| SFT | LongMIT | 41.7 |
| SFT | LongReward-SFT | 47.4 |
| SFT | GPT-4o-MuSiQue | 50.9 |
| **SFT** | **SEALONG-SFT** | **52.4** |
| PO | UltraFeedback | 35.1 |
| PO | LongReward-Preference | 50.9 |
| **PO** | **SEALONG (ORPO)** | **55.0** |

Most external datasets actually **hurt** Llama-3.1-8B (because the model already has strong long-context capability and low-quality synthetic data damages it). SEALONG not only avoids harm but produces significant gains. Even versus GPT-4o-annotated MuSiQue answers, SEALONG's self-supervised data is better (55.0 vs. 50.9).

#### Short-context performance (Tab. 8, OpenLLM Leaderboard)

| Model | Long-Ctx Avg. | MMLU | GSM8K | ARC | HellaSwag | WinoGrande | TruthfulQA | Short Avg. |
|-------|---------------|------|-------|-----|-----------|------------|------------|------------|
| Qwen-2.5-7B | 49.0 | 74.2 | 82.4 | 67.1 | 81.5 | 74.7 | 64.7 | 74.1 |
| + SEALONG | 51.8 | 74.1 | 83.2 | 66.5 | 81.3 | 74.4 | 64.8 | **74.1** |
| Llama-3.1-8B | 50.8 | 68.3 | 77.7 | 60.2 | 80.1 | 77.4 | 54.1 | 69.6 |
| + SEALONG | 55.0 | 68.4 | 77.8 | 60.3 | 79.9 | 77.3 | 53.8 | **69.6** |

SEALONG substantially improves long-context performance while **almost not affecting** short-context performance (short avg. is unchanged).

### Key findings

1. **Prompting strategy matters:** plan-and-solve prompting beats default prompting in long-context reasoning (Llama-3.1-8B on 2WikiMQA: 66.0 → 82.0), showing under-elicited model potential.
2. **MBR > Self-evaluation:** MBR scoring based on multi-output consensus beats LLM direct self-evaluation, since current LLMs have limited self-evaluation in long contexts. Sentence-embedding similarity is the best utility metric.
3. **Data efficient:** only ~1K training samples reach near-saturation performance; more data adds little. SEALONG **elicits existing capability** rather than teaching new skills.
4. **Effect of $N$:** going from $N=8$ to $N=32$ consistently improves performance (better MBR estimation); $N > 32$ has diminishing returns — the scoring method's ability to pick high-quality outputs from a larger pool plateaus.
5. **ORPO beats SFT:** preference optimization (SEALONG-ORPO: 55.0) clearly beats SFT (SEALONG-SFT: 52.4); contrastive learning is more effective.
6. **Strong generalization:** synthesized training data come only from MuSiQue but improvements transfer consistently to Qasper, MultiFieldQA-En, HotpotQA, and other cross-domain tasks.
7. **Output length nearly unchanged:** SEALONG is not "guessing right by writing more tokens" — average output token count is roughly stable (Llama-3.1-8B: 289 → 295).
8. **Limitations:** (a) MBR scoring still has a notable gap with oracle — room for better scoring; (b) only validated on ≤14B models and ≤32K context; (c) training data only from MuSiQue's multi-hop QA, not covering harder problem types like full-context reasoning.
