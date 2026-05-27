# ScPO: Self-Consistency Preference Optimization

> **Added to survey on:** 2026-03-11

**Paper:** Self-Consistency Preference Optimization
**Authors:** Archiki Prasad (UNC Chapel Hill), Weizhe Yuan (Meta FAIR / NYU), Richard Yuanzhe Pang (Meta FAIR), Jing Xu (Meta FAIR), Maryam Fazel-Zarandi (Meta FAIR), Mohit Bansal (UNC Chapel Hill), Sainbayar Sukhbaatar (Meta FAIR / NYU), Jason Weston (Meta FAIR), Jane Yu (Meta FAIR)
**Venue:** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning)
**Citation:** prasad2025scpo

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| ScPO | Pref. Opt. | training-time | Semantic |

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

**Family III — Self-Generated Target Bootstrapping → Preference Optimization → Internally Generated Preference Pairs**

ScPO is unsupervised post-training (UPT) for the following reasons:

1. **No external annotation:** the pipeline does not rely on gold answers, human annotation, or external reward models. ScPO uses only unlabeled queries (answer-free math/logic problems) as seed data; all supervision comes from the model itself.
2. **Internally generated preference pairs:** ScPO uses self-consistency (majority voting) as a proxy — sample multiple answers per question and rank them by agreement. The most-consistent response becomes the chosen sample; the least-consistent becomes the rejected sample, forming preference pairs. These pairs come not from human preference or external RM scores, but entirely from the statistical agreement of model-population outputs.
3. **DPO-style training:** the constructed preference pairs are then used in a weighted DPO + NLL loss for preference optimization — a textbook preference-optimization carrier.
4. **Iterative bootstrapping:** through multiple iterations (M₀ → M₁ → M₂), the model becomes increasingly consistent and accurate, and additional new questions are generated each round — the self-generated-target bootstrapping paradigm.

The ScPO artifact is therefore: **construct preference pairs from majority vs. minority answers, then do DPO-style post-training**.

---

## 2. Problem Addressed

- **Self-alignment is weak on reasoning:** existing self-alignment methods (e.g., Self-Rewarding LM) let the model evaluate its own generations, but Huang et al. (2024) show LLMs **cannot accurately self-evaluate** on complex reasoning, breaking self-evaluation.
- **External reward models suffer OOD problems:** using an off-the-shelf RM (e.g., ArmoRM) to score answers and form preference pairs avoids gold labels but performs poorly OOD (e.g., ZebraLogic), is noisy, and can even rank correct answers below wrong ones.
- **Gold labels are expensive:** human annotation of complex multi-step reasoning is very costly, especially process-level supervision (e.g., PRM800K) — limiting scalability.
- **Self-consistency was only for inference:** traditional self-consistency (Wang et al., 2023) only boosts answer accuracy at inference via multi-sampling and majority vote, with extra compute cost and no parameter update — no persistent capability gain.

ScPO's core idea: **turn self-consistency from an inference-time technique into a training-time supervision signal**, using agreement to form preference pairs for iterative preference-optimization training.

---

## 3. Method

### 3.1 Overall pipeline

ScPO is an iterative training framework; each iteration has three steps:

1. **Query generation:** use the current model with few-shot prompting to generate new math/logic problems (questions only, no answers) to expand the training set.
2. **Build self-consistency preference pairs:** for each question, sample $k$ answers, rank by final-answer frequency (vote), and pick chosen and rejected.
3. **Train with weighted ScPO loss:** instance-level weighted DPO + NLL loss for preference optimization.

### 3.2 Initialization

- **Initial model $M_0$:** a pretrained base LLM (e.g., Llama-3 Base 8B); instruction tuning not required.
- **Seed query set:** a small batch of high-quality unlabeled queries (e.g., GSM8K/MATH train questions with answers removed).

### 3.3 Query generation

Use Llama-3 Instruct 8B with few-shot prompting on seed questions to generate new problems of similar difficulty:
- GSM8K / MATH: randomly pick 4 train questions as in-context examples; prompt the model to generate a new solvable math problem.
- ZebraLogic: one-shot prompt to replace attributes (names, drinks) of an existing puzzle to create variants.
- **Key design:** only valid questions are required; correct answers are not — invalid problems get filtered downstream by the vote filter.

### 3.4 Building self-consistency preference pairs

For each question $x$ in the training set, sample $k$ answers $\bar{y}_x = \{y_1, y_2, \dots, y_k\}$ from the current model $M_t$ with temperature sampling.

**Vote function:** extract the final answer from each response (via $\text{ans}(\cdot)$), and compute its relative frequency:
$$V(y) = \sum_{m=1}^{k} \mathbf{1}(\text{ans}(y_m) = \text{ans}(y))$$

**Preference-pair construction:**
$$D_t^{\text{pairs}} = \{(x, y^+, y^-) \mid x \in D_t, \; y^+ = \arg\max_{y \in \bar{y}_x} V(y), \; y^- = \arg\min_{y \in \bar{y}_x} V(y), \; V(y^+) \geq \tau\}$$

- Chosen $y^+$: a response with the most-voted answer (multiple responses may share the same final answer; pick one randomly).
- Rejected $y^-$: a response with the least-voted answer.
- **Filtering threshold $\tau$:** keep only instances where the chosen vote is $\geq \tau$. At $M_0$, $\tau = 0.5k$ (GSM8K/MATH); raised to $0.7k$ (GSM8K) and $0.6k$ (MATH) at $M_1$ as the model gets more consistent. For ZebraLogic: $M_1$ sets $\tau=2$ (at least 2 exactly matching votes); $M_2$ sets $\tau=0.5k$.

### 3.5 ScPO loss function

$$\mathcal{L}_{\text{ScPO}}(y^+, y^- \mid x) = \underbrace{-w(x) \log \sigma\!\left(\beta \log \frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta \log \frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} + \underbrace{\frac{\alpha \, w(x)}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- $w(x) = (V(y^+) - V(y^-)) / k$: instance-level weight reflecting pair quality / model confidence. Larger vote margin → higher weight → more reliable pair. $w(x) \in [0, 1]$.
- $\beta = 0.5$: DPO temperature.
- $\alpha = 1$: NLL regularization coefficient.
- Reference model: the model $M_t$ at the start of the current iteration.
- The loss form is similar to IRPO (Pang et al., 2024)'s DPO + NLL, but the key differences are: (a) unsupervised setting; (b) instance-level weight $w(x)$.

### 3.6 Iterative training

Two iterations ($T=2$):
- $M_0$: seed LLM (pretrained model)
- $M_1$: use $M_0$ to generate $D_0^{\text{pairs}}$ (seed + new questions), train with $\mathcal{L}_{\text{ScPO}}$
- $M_2$: use $M_1$ to generate $D_1^{\text{pairs}}$ (seed + new questions), train with $\mathcal{L}_{\text{ScPO}}$

Each iteration makes the model more consistent and accurate, enabling (a) valid preference pairs from more questions (previously unresolvable for $M_0$ now obtain majority answers); (b) higher-quality data for the next iteration.

### 3.7 Semi-supervised variant

When some data has gold labels:
- Labeled questions $x_{\text{gold}}$: sample $k$ answers; chosen = correct, rejected = wrong, weight $w(x_{\text{gold}}) = 1$.
- Unlabeled questions: standard ScPO self-consistency pair construction and weighting.
- Special case: with all labels, the loss reduces to IRPO loss.

### 3.8 Key hyperparameters

| Parameter | Setting |
|------|------|
| Samples $k$ | GSM8K/MATH: 8; ZebraLogic: 16 |
| Sampling temperature | 0.7 (response and new question generation); 1.2 (sampling rejected, encouraging diversity) |
| top-p | 0.9 |
| Training epochs | 10 |
| Learning rate | 5e-6 (cosine schedule) |
| Batch size | 16 (effective) |
| $\beta$ (DPO) | 0.5 |
| $\alpha$ (NLL) | 1 |
| Iterations $T$ | 2 |

---

## 4. Datasets

| Domain | Dataset | Description |
|------|--------|------|
| Grade-school math reasoning | GSM8K | 7.5K/1.3K (train/test) grade-school math word problems; train split into 6.7K/0.8K (train/dev) |
| High-school competition math | MATH | 7.5K/5K (train/test) challenging high-school competitions; train split into 6.7K/0.8K/5K (train/dev/test) |
| Logical reasoning | ZebraLogic | 1K Einstein-style logic-grid puzzles; test only, no train/dev; each puzzle is an $n \times m$ constraint-satisfaction table |

**Training-data generation statistics:**
- GSM8K: $M_1$ uses 5.3K seed queries; $M_2$ uses 1.4K seed + 5.1K generated.
- MATH: $M_1$ uses 0.6K seed + 1.2K generated; $M_2$ uses 1.2K seed + 2.5K generated (MATH is hard; only ~1/4 seed queries yield a clear majority answer).
- ZebraLogic: $M_1$ uses 0.4K seed + 2.0K generated; $M_2$ uses 0.5K seed + 2.2K generated.

---

## 5. Evaluation metrics and main results
### Metrics

- **GSM8K / MATH:** final-answer exact-match accuracy (greedy decoding + 8-way self-consistency (SC) decoding).
- **ZebraLogic:**
  - Puzzle accuracy (overall / easy / hard): exact match of the entire $n \times m$ table
  - Cell accuracy: per-cell match

### Main results

#### GSM8K (Llama-3 Base 8B)

| Method | Iter | Train data (K) | Greedy Acc. | SC Acc. |
|------|------|-------------|-------------|---------|
| Seed model (zero-shot CoT) | $M_0$ | - | 41.17% | 51.80% |
| IRPO_RM | $M_1$ | 5.5K seed | 48.67% | 69.98% |
| LMSI | $M_2$ | 1.1K+5.2K | 56.71% | 62.55% |
| **ScPO_Unsup.** | $M_1$ | 5.3K seed | **61.03%** | **71.49%** |
| **ScPO_Unsup.** | $M_2$ | 1.4K+5.1K | **63.91%** | 71.11% |
| IRPO_Gold (gold-label upper bound) | $M_2$ | 5.7K† | 64.29% | 72.56% |
| **ScPO_Semi-Sup.** | $M_2$ | 5.7K†+4.5K | **66.64%** | **74.75%** |

**Key findings:**
- ScPO_Unsup. after one iteration beats the seed model by **+22.74%** (greedy) and IRPO_RM by **+12.36%**.
- Two ScPO_Unsup. iterations close within **<1%** of gold-label IRPO_Gold (greedy: 63.91% vs. 64.29%).
- Semi-supervised ScPO further improves IRPO_Gold by **+2.35%** (greedy).

#### MATH (Llama-3 Base 8B)

| Method | Iter | Greedy Acc. | SC Acc. |
|------|------|-------------|---------|
| Seed model | $M_0$ | 14.46% | 18.20% |
| IRPO_RM | $M_1$ | 18.06% | 24.20% |
| LMSI | $M_2$ | 16.96% | 20.20% |
| **ScPO_Unsup.** | $M_1$ | 17.36% | **25.70%** |
| **ScPO_Unsup.** | $M_2$ | **19.72%** | 24.58% |
| IRPO_Gold | $M_2$ | 20.32% | 26.88% |
| **ScPO_Semi-Sup.** | $M_2$ | **20.48%** | 26.92% |

**Key findings:**
- Two ScPO_Unsup. iterations beat the seed model by **+5.26%** (greedy) and IRPO_RM by **+1.64%**.
- Within <1% of IRPO_Gold (greedy: 19.72% vs. 20.32%).
- MATH is harder than GSM8K; only ~1/4 seed questions yield a clear majority answer, so generated problems matter more.

#### ZebraLogic (Llama-3 Instruct 8B)

| Method | Puzzle Acc. (Overall) | Easy | Hard | Cell Acc. |
|------|-----------------------|------|------|-----------|
| Llama-3 Instruct 70B | 17.2% | 52.1% | 3.6% | 42.9% |
| Gemma-2 27B IT | 16.3% | 50.7% | 2.9% | 41.2% |
| Claude-3 Haiku | 14.3% | 47.9% | 1.2% | 37.9% |
| Llama-3 Instruct 8B ($M_0$) | 11.6% | 40.0% | 0.4% | 39.1% |
| IRPO_RM $M_1$ | 11.3% | 37.9% | 1.0% | 42.1% |
| LMSI $M_2$ | 16.8% | 53.6% | 2.5% | 46.9% |
| **ScPO_Unsup. $M_1$** | 17.0% | 54.3% | 2.5% | 47.6% |
| **ScPO_Unsup. $M_2$** | **18.1%** | **58.2%** | 2.5% | 45.2% |

**Key findings:**
- After two ScPO rounds, the 8B model **beats Llama-3 70B (+0.9%)**, **Gemma-2 27B (+1.8%)**, **Claude-3 Haiku (+3.8%)**.
- Leaderboard rank goes from #38 to #30.
- IRPO_RM is virtually useless on ZebraLogic (puzzle acc. drops by 0.3%), because ArmoRM is highly OOD on ZebraLogic — 40.5% of pairs have incorrect orderings.
- ScPO after one iteration beats IRPO_RM by **+5.7%** (puzzle acc.) and **+5.5%** (cell acc.).

#### Llama-3.1 Base 8B validation

| Method | GSM8K Greedy | MATH Greedy |
|------|-------------|-------------|
| Seed model | 43.14% | 15.70% |
| ScPO_Unsup. $M_2$ | 64.22% (+21.08%) | 23.20% (+7.50%) |
| ScPO_Semi-Sup. $M_2$ | 68.46% (+25.32%) | 24.36% (+8.66%) |

Same trend as on Llama-3 — confirms robustness.

### Key findings

1. **Weighted loss matters:** vs. unweighted loss ($w(x)=1$), the weighted ScPO loss improves +2.5% on GSM8K $M_1$ and +1.44% on MATH $M_1$ (greedy) — dynamically weighting by vote margin effectively uses pair-quality information.

2. **Model consistency rises with iterations:** majority-vote share ($V(y^+)/k$) grows steadily after each iteration (GSM8K: ~50% → ~60% → ~70%) — due to (i) accuracy improvement, (ii) preference optimization reducing output diversity, and (iii) ScPO distilling SC distribution into the single-sample distribution.

3. **Self-consistency outperforms RM as quality signal:** Figure 3 shows ArmoRM has higher incorrect-ordering rates (19.1%, 32.4%, 40.5%) than self-consistency (7.8%, 11.8%, 16.0%) on all three datasets — especially on OOD ZebraLogic where SC has 12.3% more correct orderings.

4. **Filter-threshold $\tau$ trade-off:** on MATH, $\tau$ from 0.1k → 0.5k raises the accuracy margin from 18% → 57% and model performance from 15.44% → 17.36% (quality > quantity); at $\tau=0.7k$ data drops below 700 pairs and performance falls to 14.76% (under-fit). $\tau=0.5k$ is the best balance.

5. **SC decoding gains shrink after training:** $M_2$'s SC accuracy is sometimes slightly below $M_1$'s — after ScPO the model is highly consistent, leaving little additional complementarity for 8-way SC. ScPO essentially "distills" inference-time SC into training.

6. **Transduction further helps:** a third ScPO round on train queries yields little (<1%), but building preference pairs from test-set questions yields an extra +1.44% (GSM8K greedy) — ScPO can adapt to the target distribution via transductive learning.
