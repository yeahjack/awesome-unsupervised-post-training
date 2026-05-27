# CSR: Calibrated Self-Rewarding Vision Language Models

> **Added to Survey:** 2026-03-11

**Paper:** Calibrated Self-Rewarding Vision Language Models
**Authors:** Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, Huaxiu Yao (UNC-Chapel Hill, UChicago, UMD, Rutgers, HKUST, PolyU, NTU, NUS)
**ArXiv:** 2405.14622
**Date:** 2024-05 (NeurIPS 2024)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CSR | Pref. Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch with self-generated responses / judgments |
| Persistence | full parameter accumulate across self-rewarding iterations |
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

- **When does the update fire:** Updates fire inside a pre-deployment self-rewarding / judge-bootstrapping loop, typically generating responses first and then judgments or preference pairs.
- **Serving the current sample or downstream samples:** Each batch's responses / judgments primarily serve the next training round and the final deployed model, not the immediate inference of a single test sample.
- **Whether parameters / state accumulate:** Parameters accumulate across multiple self-rewarding iterations; even with role switching (actor / judge / meta-judge), it stays in the same offline training loop.
- **Reset boundary:** The adaptation timing is therefore offline iterative bootstrapping, not deployment-time TTA.

## 1. UPT Assignment Rationale
**Adjacent — External-encoder-assisted self-rewarding (fails B4); shadow placement: Family IV — Internal Evaluator Bootstrapping (self-rewarding)**

> **Status update (paper-aligned):** Per paper §6.2 and the figure tree, CSR is **routed to adjacent**, not strict UPT. The boundary check that fails is **B4** (any judge / scorer / reward model used in the update must derive from the same model lineage): the calibrated reward sums (i) the LVLM's own language-decoder cumulative probability with (ii) a **CLIP-derived visual-relevance score**, where CLIP is a pretrained frozen encoder external to the LVLM lineage. The CLIP term injects supervision from a non-same-lineage scorer, so the system as a whole fails B4 and is listed in the adjacent table rather than counted in Family IV. We retain a *shadow* Family IV placement only because the dominant driver of preference-pair construction is the internal judge; the formal classification is adjacent.

CSR extends the self-rewarding-language-models paradigm from text-only LLMs to vision-language models (LVLMs). Its core idea fits internal evaluator bootstrapping: **the model itself plays both response generator and reward judge**, with no external human annotation or extra model (e.g., GPT-4) providing the preference signal.

Why this is a "near-miss" for strict UPT but ultimately adjacent:
1. **No human annotation:** preference data come entirely from the model itself; sentence-level beam search produces candidate responses, and the model's own language-decoder cumulative probability is used as the instruction-following reward.
2. **Internal evaluator (dominant term):** the model evaluates the quality of its own outputs—a calibrated reward score (mixing language score and CLIP-based visual-relevance score) ranks candidate responses and selects preferred / dispreferred pairs.
3. **Iterative bootstrapping:** multiple rounds of DPO fine-tuning continually improve the model's preference-judgment ability and generation quality—the canonical self-improvement loop.
4. **CLIP term (the B4 violation):** the visual-relevance term in the reward uses pretrained CLIP, an external frozen encoder outside the LVLM's same-model-lineage. Per paper §6.2, this single term is enough to route the full CSR system to adjacent (Table~\ref{tab:adjacent-methods}), even though the rest of the loop is internal.

The special contribution is using **visual constraints to calibrate the self-rewarding process**: because LVLMs exhibit modality misalignment (preferring text knowledge and ignoring visual input), porting LLM self-rewarding directly to LVLMs would have self-generated preferences that also ignore the image. CSR adds an image-response relevance score to the reward to correct this bias.

---

## 2. Problem Addressed

LVLMs' **hallucination problem**: text descriptions that look fluent but contradict the visual information in the input image. The root cause is **modality misalignment**—the model leans on the text knowledge embedded in pretraining and ignores the actual visual input.

Limitations of existing solutions:
- **Reliance on external models (e.g., GPT-4) or human annotation** to generate preference data—expensive and prone to extra biases.
- Externally generated preference data **cannot capture the target LVLM's intrinsic preferences**, so the curated preferences are easily distinguishable by the model and lose efficacy.
- Direct porting of LLM self-rewarding to LVLM is also problematic: LVLMs suffer modality misalignment in both response generation and preference modeling, and the self-generated preferences may also ignore visual information.

---

## 3. Method

CSR is an **iterative preference-optimization framework** that alternates: (1) candidate-response generation and (2) preference curation & fine-tuning.

### 3.1 Step-Level Reward Modeling and Calibration

CSR's reward design satisfies two principles:
- **Vision-Constrained Reward:** integrate image-relevance information into the reward to address the LVLM's tendency to ignore image input during preference generation.
- **Step-Wise Reward:** assign rewards at each generation step (sentence level) instead of giving the whole response a single reward, providing finer guidance.

#### (a) Self-generated instruction-following score $R_T(s)$

Use the LVLM's language decoder to compute the sentence-level cumulative probability:

$$R_T(s) = \prod_{o=1}^{N_o} P(r_o \mid x, r_1, r_2, \ldots, r_{o-1})$$

where $N_o$ is the number of tokens in sentence $s$ and $r_o$ is the $o$-th token. A higher score means the response better matches the instruction-following ability.

#### (b) Image-response relevance score $R_I(s)$

Use the CLIP score to measure the relevance between a generated sentence and the input image:

$$R_I(s) = \max(100 \times \cos(F_I(x_v), F_T(s)), 0)$$

where $F_I(x_v)$ and $F_T(s)$ are CLIP's visual and textual embeddings. CLIP's vision encoder is kept consistent with the target LVLM's vision encoder.

#### (c) Calibrated reward score $R(s)$

$$R(s) = \lambda \cdot R_I(s) + (1 - \lambda) \cdot R_T(s)$$

where $\lambda$ trades off image-response relevance and language instruction-following. Experiments use $\lambda = 0.9$ (CLIP-score weight 0.9, language-score weight 0.1)—**a heavy lean toward visual calibration**.

### 3.2 Iterative Fine-Tuning

#### Step 1: Step-level candidate-response generation

Use **sentence-level beam search** to generate candidate responses sentence by sentence:
1. Sample multiple candidate sentences in parallel, with "." as the sub-sentence delimiter.
2. For each sentence $s$, compute the calibrated reward $R(s)$.
3. Keep the top-k and bottom-k sentences for the next beam-search round.
4. Repeat until the response is complete.

Hyperparameters: `num_beams=5`, `num_token_beams=5`, `num_beam_group=5` (group beam search for diversity), `diversity_penalty=3.0`, `max_length=1024`, `max_new_tokens=74`.

#### Step 2: Preference curation and optimization

For each input prompt, take the response with the **highest** and **lowest** cumulative calibrated reward as the preferred ($y_w$) and dispreferred ($y_l$) response, forming the preference pair.

Fine-tune with **DPO**; the loss at iteration $t$:

$$L_t = -\mathbb{E}_{(x, y_{w,t}, y_{l,t}) \sim D} \left[ \log \sigma \left( \alpha \log \frac{\pi_\theta(y_{w,t} \mid x)}{\pi_{\theta_{t-1}}(y_{w,t} \mid x)} - \alpha \log \frac{\pi_\theta(y_{l,t} \mid x)}{\pi_{\theta_{t-1}}(y_{l,t} \mid x)} \right) \right]$$

where $\pi_{\theta_{t-1}}$ is the previous-iteration fine-tuned model serving as the reference model.

### 3.3 Iteration

Full pipeline (Algorithm 1):
1. Input dataset $D = \{x^{(i)}\}_{i=1}^N$, reference model $\pi_{\text{ref}}$, iteration count $T$.
2. In each iteration: run sentence-level beam search per input → compute $R_T(s)$, $R_I(s)$, $R(s)$ → choose top / bottom responses → update the model with DPO.
3. The updated model becomes both the seed and reference model for the next round.

### 3.4 Theoretical Analysis

The paper provides a theoretical framework explaining the effect of the image-response relevance score. Core conclusion (Theorem 5.1): when the model tends to prioritize textual information over visual input (i.e., $\|\beta^{*\top} V_1^\top \beta^*\| \ll \|\beta^{*\top} V_2^\top \beta^*\|$), there exists $\lambda < 1$ for which CSR's loss ($\lambda < 1$) is strictly smaller than the loss without the image-response relevance score ($\lambda = 1$). Intuitively, CSR up-weights the image signal to compensate for the model's neglect of visual input.

---

## 4. Datasets

| Domain | Dataset | Notes |
|--------|---------|-------|
| Training data | LLaVA-150k (subset) | randomly sample ≈**13,000** image-prompt pairs from the detailed-description and complex-reasoning subsets to build preference data |
| Comprehensive evaluation | MME | Perception (MME_P) + Cognition (MME_C), 14 sub-tasks |
| Comprehensive evaluation | SEED-Bench | 19K multiple-choice, 12 dimensions |
| Comprehensive evaluation | LLaVA-Bench (In-the-Wild) | 24 images, 60 questions |
| Comprehensive evaluation | MMBench | CircularEval + ChatGPT |
| Comprehensive evaluation | MM-Vet | 6 core abilities × 16 integration tasks |
| General VQA | ScienceQA (SQA^I) | ≈21K multiple-choice science questions |
| General VQA | VizWiz | 31K+ visual QA (blind-photographer scenarios) |
| General VQA | GQA | scene-graph-based, 22M questions |
| Hallucination evaluation | POPE | binary classification (Yes / No) on hallucinated objects |
| Hallucination evaluation | CHAIR (CHAIR_S / CHAIR_I) | object hallucination on 500 COCO val images, sentence and instance level |

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **MME_P / MME_C:** perception / cognition composite scores.
- **SEED:** generative-comprehension accuracy.
- **LLaVA^W:** multi-scenario visual-reasoning score.
- **MMBench (MMB):** multimodal composite evaluation.
- **MM-Vet:** multi-ability integration evaluation.
- **SQA^I:** ScienceQA image-subset accuracy.
- **VizWiz:** VQA accuracy.
- **GQA:** visual-reasoning accuracy.
- **POPE (acc / f1):** hallucination binary classification.
- **CHAIR_S / CHAIR_I:** object-hallucination rate (lower is better).

### Main Results

#### Table 1: CSR vs. baselines on LLaVA-1.5 (after 3 iterations)

**LLaVA-1.5-7B:**

| Method | MME_P | MME_C | SEED | LLaVA^W | MMB | MM-Vet | SQA^I | VizWiz | GQA | POPE | CHAIR_S | CHAIR_I |
|--------|-------|-------|------|---------|-----|--------|-------|--------|-----|------|---------|---------|
| LLaVA-1.5-7B (base) | 1510.7 | 348.2 | 58.6 | 63.4 | 64.3 | 30.5 | 66.8 | 50.0 | 62.0 | 85.90 | 48.8 | 14.9 |
| +VLFeedback | 1432.7 | 321.8 | 59.3 | 62.1 | 64.0 | 31.2 | 66.2 | 52.6 | 63.2 | 83.72 | 40.3 | 13.2 |
| +POVID | 1452.8 | 325.3 | 60.2 | 68.7 | 64.9 | 31.8 | 68.8 | 53.6 | 61.7 | 86.90 | 35.2 | 8.3 |
| +RLHF-V | 1489.2 | 349.4 | 60.1 | 65.4 | 63.6 | 30.9 | 67.1 | 54.2 | 62.1 | 86.20 | 29.7 | 7.5 |
| +Self-rewarding | 1505.6 | 362.5 | 60.0 | 61.2 | 64.5 | 31.4 | 69.6 | 53.9 | 61.7 | 86.88 | 24.0 | 6.7 |
| **+CSR (Ours)** | **1524.2** | **367.9** | **60.3** | **71.1** | **65.4** | **33.9** | **70.7** | **54.1** | **62.3** | **87.01** | **21.0** | **6.0** |

**LLaVA-1.5-13B:**

| Method | MME_P | MME_C | SEED | LLaVA^W | MMB | MM-Vet | SQA^I | VizWiz | GQA | POPE | CHAIR_S | CHAIR_I |
|--------|-------|-------|------|---------|-----|--------|-------|--------|-----|------|---------|---------|
| LLaVA-1.5-13B (base) | 1531.3 | 295.4 | 61.6 | 70.7 | 67.7 | 35.4 | 71.6 | 53.6 | 63.3 | 85.90 | 48.3 | 14.1 |
| +Self-rewarding | 1529.0 | 300.1 | 62.8 | 65.6 | 64.5 | 35.3 | 74.3 | 56.1 | 63.2 | 86.58 | 37.0 | 8.8 |
| **+CSR (Ours)** | **1530.6** | **303.9** | **62.9** | **74.7** | **68.8** | **37.8** | **75.1** | **56.8** | **63.7** | **87.30** | **28.0** | **7.3** |

#### Iterative gains (LLaVA-1.5-7B hallucination)

| Iteration | POPE acc | POPE f1 | CHAIR_S↓ | CHAIR_I↓ |
|-----------|----------|---------|----------|----------|
| Base | 85.90 | 84.29 | 48.8 | 14.9 |
| +CSR iter-1 | 86.94 | 85.80 | 26.6 | 7.2 |
| +CSR iter-2 | 86.82 | 85.62 | 23.0 | 6.1 |
| +CSR iter-3 | 87.01 | 85.93 | 21.0 | 6.0 |
| +CSR iter-4 | 87.05 | 85.95 | 19.0 | 5.9 |
| +CSR iter-5 | 87.16 | 85.98 | 18.3 | 5.4 |

#### Ablation (Table 2): Average performance score (100-point scale)

| Method | 7B | 13B |
|--------|-----|-----|
| Base | 66.61 | 68.08 |
| Only $R_T$ (language only) | 68.46 | 68.12 |
| Only $R_I$ (visual only) | 67.49 | 69.23 |
| **CSR (combined)** | **72.39** | **71.95** |

#### λ ablation (LLaVA-1.5-7B, 3 iterations)

| λ | LLaVA^W | CHAIR_S↓ | CHAIR_I↓ |
|---|---------|----------|----------|
| 0.1 | 66.7 | 40.8 | 10.2 |
| 0.5 | 68.2 | 28.2 | 6.7 |
| 0.9 | 71.1 | 21.0 | 6.0 |

#### Vila 7B compatibility (3 CSR iterations)

| Method | MME_P | SEED | LLaVA^W | MM-Vet | VizWiz | POPE | CHAIR_S↓ |
|--------|-------|------|---------|--------|--------|------|----------|
| Vila 7B (base) | 1533.0 | 61.1 | 69.7 | 34.9 | 57.8 | 85.50 | 31.0 |
| +CSR iter-3 | 1542.2 | 63.4 | 74.3 | 39.8 | 62.7 | 87.31 | 28.0 |

### Key Findings

1. **CSR keeps improving across iterations:** after 3 rounds the 7B model gains ≈**7.62%** averaged over benchmarks; the 13B model gains ≈**5.25%**. The biggest gains land on LLaVA^W (+8.9%) and CHAIR (+49.50%).
2. **Drastically reduced hallucination:** LLaVA-1.5-7B's CHAIR_S drops from 48.8 to 21.0 (−57%), CHAIR_I from 14.9 to 6.0 (−60%).
3. **Beats external-preference-data baselines:** CSR outperforms approaches relying on GPT-4 (VLFeedback / Silkie), human annotation (LLaVA-RLHF), and external models (POVID), showing that adaptive self-rewarding better captures the target LVLM's inherent preferences.
4. **Visual calibration is critical:** $\lambda=0.9$ is best, indicating that heavily favoring the CLIP visual score effectively corrects the model's text bias. Using only $R_T$ or only $R_I$ is worse than the combination (CSR on 7B reaches 72.39 vs. 68.46 for $R_T$-only and 67.49 for $R_I$-only).
5. **Attention reallocation:** attention-map analysis shows CSR boosts attention scores on visual tokens, easing over-reliance on contextual text.
6. **Cross-model generalization:** also effective on Vila 7B; after 3 iterations the average score rises by 3.37%, with VizWiz (+8.48%) and MM-Vet (+14.0%) showing especially large gains.
7. **Reward score correlates with performance:** chosen reward rises 0.4885 → 0.5066, rejected reward 0.4551 → 0.4799, and average performance climbs 66.61 → 72.24 across iterations—the model keeps producing higher-quality preference data.
