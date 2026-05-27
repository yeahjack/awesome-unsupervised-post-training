# Self-Training Large Language Models with Confident Reasoning

> **Added to survey on:** 2026-03-11

**Paper:** Self-Training Large Language Models with Confident Reasoning
**arXiv:** 2505.17454
**Authors:** Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn
**Affiliations:** POSTECH, KAIST

---

| Property | Value |
|---|---|
| Method | Confident ST (CORE-PO) |
| Carrier | Direct Opt. (online DPO) |
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

**Family III — Self-Generated Target Bootstrapping, subclass: rationale / critique / latent-thought self-training**

Confident ST (CORE-PO) belongs to Family III for the following reasons:

- **Self-confidence-driven pseudo-target selection:** the model samples $N$ outputs $\{(r_i, a_i)\}_{i=1}^N$ for each unlabeled question and uses its own reasoning-level confidence $C_\theta(r|x)$ (measured via P(True)) to identify high-quality reasoning paths, feeding them back as pseudo-targets.
- **Self-generated-target bootstrapping at the core:** the training signal comes entirely from model-generated reasoning paths and self-assessed confidence scores, with no external ground-truth labels, external verifier, or human annotation. Preference pairs are built from the self-assessed confidence, with high-confidence reasoning paths treated as preferred outputs.
- **Rationale self-training subclass:** unlike methods that look only at answer-level confidence (e.g., SC-PO), the key contribution is assessing the quality of the reasoning path itself (not just the final answer confidence) — a self-training paradigm centered on the quality of rationales.
- **Carrier — Direct Opt. (DPO):** uses online Direct Preference Optimization to inject the high-confidence-reasoning preference into model parameters.
- **Level — Traj.:** the optimization target is the full reasoning–answer trajectory $s = [r, a]$, and confidence is computed at the level of an entire reasoning path (monolithic P(True)) or per-statement averages (statement-wise P(True)).

---

## 2. Problem Addressed

Existing confidence-based self-training methods (e.g., Huang et al., 2023; Prasad et al., 2024; Zhang et al., 2024b) suffer from one core flaw: they **rely only on answer-level confidence** (estimated via majority voting) and ignore the quality of the reasoning path itself.

Specific issues include:

1. **The "wrong reasoning → correct answer" coincidence:** LLMs often produce wrong reasoning that happens to yield the correct answer (especially in multiple choice), giving high answer-level confidence. As Figure 1 shows, the model may use flawed reasoning (e.g., "(b)–(d) are boiling points") yet land on the correct answer; training reinforces that wrong reasoning pattern.
2. **Reasoning quality degradation:** once the model prefers such "accidentally correct" paths, its reasoning ability actually drops — producing systematic errors on new questions (e.g., labeling 32°C as a boiling point).
3. **Annotation cost limits:** high-quality human-annotated reasoning paths are expensive; a self-training alternative is needed.

The core insight: **we need to move from answer-level confidence to reasoning-level confidence** to more accurately identify high-quality reasoning paths for self-training.

---

## 3. Method

### 3.1 Core motivation and observations

On GPQA (Figure 2), the authors find:

- **Observation 1:** reasoning paths with high answer-level confidence are often wrong, even when the final answer is right; there is a clear gap between answer accuracy and reasoning accuracy.
- **Observation 2:** outputs with high reasoning-level confidence $C_\theta(r|x)$ tend to have fewer reasoning errors and simultaneously high answer-level accuracy.

These observations motivate introducing reasoning-level confidence into self-training.

### 3.2 CORE-PO pipeline

CORE-PO (**CO**nfidence **RE**asoning — **P**olicy **O**ptimization) is shown in Figure 3 and Algorithm 1:

**Step 1 — Multi-output sampling:** given question $x$, the LLM $M_\theta$ generates $N=5$ outputs $\{(r_i, a_i)\}_{i=1}^N$ with $T=1.0$, $\text{top-p}=0.9$.

**Step 2 — Reasoning-level confidence estimation:** uses **P(True)** (Kadavath et al., 2022) to estimate confidence of each reasoning path. Two variants:

- **Monolithic P(True):** evaluates the entire reasoning path in one shot — asks the LLM "Is the selected reasoning correct?" and takes the probability of "True" as $C_\theta(r|x)$. The prompt also includes $M=4$ randomly generated reasoning paths as comparative context.
- **Statement-wise P(True):** splits the reasoning path into statements $r = [r_1, \ldots, r_T]$ and evaluates each statement, then averages:
  $$C_\theta(r|x) = \frac{1}{T} \sum_{t=1}^{T} C_\theta(r_t | x, r_1, \ldots, r_{t-1})$$

Also estimates answer-level confidence $C_\theta(a|x, r)$ (P(True) of the answer conditioned on reasoning and question). The composite confidence is:
$$C_\theta(a, r|x) = C_\theta(a|x, r) \cdot C_\theta(r|x)$$

**Step 3 — Preference-pair construction and DPO training:** for the multiple outputs per question, sort by composite confidence and form preference pairs $(s_w, s_l)$, with $s_w$ high-confidence and $s_l$ low-confidence. Use online DPO:
$$\mathcal{L} = \log \sigma \left( \beta \log \frac{M_\theta(s_l|x)}{M_{\text{ref}}(s_l|x)} - \beta \log \frac{M_\theta(s_w|x)}{M_{\text{ref}}(s_w|x)} \right)$$

with fixed reference model $M_{\text{ref}}$, $\beta = 0.1$, pushing the model toward higher likelihood for high-confidence reasoning paths.

### 3.3 Implementation details

- **Base models:** Llama3.1-8B-Instruct and Qwen2.5-7B-Instruct
- **LoRA adapter:** rank = 128, $\alpha = 256$
- **Training sampling:** $N=5$ outputs per question, $T=1.0$, $\text{top-p}=0.9$
- **Inference sampling (inference-time scaling):** 8 samples, $T=0.7$, $\text{top-p}=0.9$
- **Default implementation uses monolithic P(True)** for reasoning-level confidence
- **Gradient clipping:** max norm = 1.0
- **Checkpoint selection:** save every 200 steps; pick the best on ARC-Challenge validation
- **Hardware:** 4× NVIDIA A100 SXM4 80GB
- **Training time:** 2–4 days
- **Framework:** transformers + trl + accelerate

---

## 4. Datasets

### 4.1 In-distribution training/evaluation datasets

| Dataset | Type | Train | Test | Metric |
|---|---|---|---|---|
| **GSM8K** | Multi-step arithmetic reasoning | 7.4K | 1.3K | Numeric answer accuracy |
| **ARC-Challenge** | Multi-choice science commonsense | 1.1K (+0.3K val) | 1.1K | Choice accuracy |
| **GPQA** | Graduate-level multi-choice science | 420 (main) | 509 (extended) | Choice accuracy |
| **MATH** | High-school math competition | 7.5K | 0.7K (Level-5) | Numeric answer accuracy |

### 4.2 Out-of-distribution evaluation datasets

| Dataset | Type | Test | Description |
|---|---|---|---|
| **CRUXEval (CRUXout)** | Code understanding and execution | 0.8K | Predict Python function output |
| **Game of 24** | Arithmetic reasoning | 1.3K | Reach 24 from four numbers |

For all training datasets, **questions only — no ground-truth labels** are used; the training signal comes entirely from the model's own confidence assessments.

---

## 5. Evaluation metrics and main results
### 5.1 Evaluation protocols

- **Greedy decoding:** evaluates the model's single greedy output accuracy
- **Inference-time scaling:** samples 8 outputs and uses each method's self-assessment to pick the best
  - SR-PO → linguistic self-assessment
  - SC-PO → majority voting (SC)
  - CORE-PO → P(True|r, a) confidence

### 5.2 Main results — Llama3.1-8B-Instruct (Table 1)

| Method | Decoding | GSM8K | ARC-Challenge | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|---|
| Base (no fine-tuning) | Greedy | 84.2 | 84.5 | 32.4 | 22.6 |
| Base | P(True\|r,a) | 89.7 | 87.0 | 34.5 | 25.2 |
| SR-PO | Greedy | 85.2 | 86.2 | 34.3 | 19.8 |
| SC-PO | Greedy | 85.7 | 86.0 | 33.7 | 25.1 |
| SC-PO | SC | 89.7 | 87.5 | 34.5 | 29.4 |
| **CORE-PO** | **Greedy** | **86.8** | **87.5** | **35.5** | **24.6** |
| **CORE-PO** | **P(True\|r,a)** | **90.5** | **89.2** | **36.1** | **29.8** |

### 5.3 Main results — Qwen2.5-7B-Instruct (Table 2)

| Method | Decoding | GSM8K | ARC-Challenge | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|---|
| Base (no fine-tuning) | Greedy | 90.0 | 89.1 | 30.6 | 45.4 |
| SC-PO | Greedy | 91.0 | 91.0 | 34.3 | 49.6 |
| SC-PO | SC | 93.0 | 92.0 | 36.3 | 55.7 |
| **CORE-PO** | **Greedy** | **91.3** | **92.2** | **37.5** | **49.6** |
| **CORE-PO** | **P(True\|r,a)** | **93.5** | **92.8** | **38.5** | **55.8** |

### 5.4 Reasoning-level quality improvement (Table 3, Llama3.1-8B-Instruct)

| Method | Decoding | GSM8K Conf. | GSM8K Reason Acc. | ARC Conf. | ARC Reason Acc. |
|---|---|---|---|---|---|
| Base | Greedy | 0.89 | 84.2 | 0.84 | 79.2 |
| Base | P(True\|r,a) | 0.98 | 89.7 | 0.94 | 81.2 |
| CORE-PO | Greedy | 0.94 | 86.8 | 0.95 | 81.5 |
| CORE-PO | P(True\|r,a) | 0.99 | 90.4 | 0.99 | 84.9 |

CORE-PO improves not just answer accuracy but also **reasoning-level confidence and reasoning-level accuracy**, indicating the model genuinely learns to produce higher-quality reasoning.

### 5.5 Out-of-distribution generalization (Table 4, Llama3.1-8B-Instruct)

| Method | Decoding | CRUXout | Game of 24 |
|---|---|---|---|
| Base | Greedy | 34.8 | 7.2 |
| SC-PO | Greedy | 43.8 | 8.3 |
| SC-PO | SC | 50.0 | 11.9 |
| **CORE-PO** | **Greedy** | **47.1** | **18.8** |
| **CORE-PO** | **P(True\|r,a)** | **48.0** | **22.1** |

CORE-PO generalizes better OOD, with a particularly large lead on Game of 24 (22.1 vs. 11.9 SC-PO; +10.2 absolute).

### 5.6 Ablations

**Monolithic vs. Statement-wise P(True) (Table 5):**

| Variant | GSM8K | ARC | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|
| Base | 84.2 | 84.5 | 32.4 | 22.6 |
| Monolithic P(True) | 86.8 | 87.5 | **35.5** | 24.7 |
| Statement-wise P(True) | **88.5** | **88.0** | 34.1 | **25.3** |

Both variants consistently beat the base model, indicating the gain comes from the core philosophy of "preferring high reasoning-level confidence" rather than a specific confidence estimator.

**Adding ground-truth answers (Table 6, ARC-Challenge):**

| Reward signal | Decoding | Answer Acc. | Reason Acc. |
|---|---|---|---|
| Answer Acc. only | Greedy | 87.4 | 73.6 |
| Answer Acc. + Reason Conf. | Greedy | **88.3** | **81.9** |
| Answer Acc. + Reason Conf. | P(True\|r,a) | **90.1** | **85.6** |

Even in traditional fine-tuning with ground-truth answers, adding reasoning-level confidence boosts reasoning accuracy substantially (+8.3 absolute), showing the method is not limited to pure self-training.

### 5.7 Key figures

- **Figure 1:** illustrates the core flaw of existing confidence-based self-training. In a question about water's rigid lattice temperature, several reasoning paths reach the correct answer (a) 0°C, but one flawed path ("(b)–(d) are boiling points") is chosen as a training target due to high answer-level confidence — leading the self-trained model to mis-classify "32°C is the boiling point". CORE-PO avoids this via reasoning-level confidence.
- **Figure 2:** shows answer accuracy and reasoning accuracy as Top-N% confidence filtering varies. When ranked by reasoning-level confidence, reasoning accuracy aligns better with answer accuracy; ranking by answer-level confidence leaves a large gap.
- **Figure 3:** CORE-PO method overview: the LLM produces multiple (reasoning, answer) outputs → reasoning-level confidence $C_\theta(r|x)$ is estimated → preference pairs are built → DPO fine-tunes the model to prefer high-confidence reasoning.
- **Table 7:** qualitative comparison. The model fine-tuned only on answer accuracy produces long, option-by-option analyses with flawed statements; adding Reason Conf. yields concise, accurate reasoning.
