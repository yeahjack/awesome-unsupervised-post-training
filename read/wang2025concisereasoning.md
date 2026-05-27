# Self-Training Elicits Concise Reasoning in Large Language Models

> **Added to Survey:** 2026-03-11

**Paper:** Self-Training Elicits Concise Reasoning in Large Language Models
**arXiv:** 2502.20122v3
**Authors:** Tergel Munkhbat*, Namgyu Ho*, Seo Hyun Kim*, Yongjin Yang, Yujin Kim, Se-Young Yun
**Affiliations:** KAIST AI
**Code:** https://github.com/TergelMunkhbat/concise-reasoning

---

| Attribute | Value |
|-----------|-------|
| Method | Concise ST |
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
| Evidence Status | PDF-confirmed external correctness / exemplar caveat |
| Strict UPT Status | Not strict UPT; GT-filtered / external-exemplar adjacent |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit
- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When does the update fire:** updates fire inside an offline pre-deployment self-bootstrap loop, typically a round-based "generate data / score / filter / retrain" schedule.
- **Serving the current sample or downstream samples:** synthetic samples or pseudo-targets in the current round primarily serve the next training round and the eventual deployed model, not the immediate inference of a given test sample.
- **Whether parameters / state accumulate:** parameters keep accumulating across rounds; the paper does not perform sample-level resets.
- **Reset boundary:** so the `When to Adapt` essence is offline iterative bootstrapping, not online test-time adaptation.

## 1. UPT Assignment Rationale
> **PDF re-audit update (2026-04-21): recommend moving Concise ST out of the strict UPT main table and into the GT-filtered / external-exemplar self-training adjacent group.**

The earlier provisional placement under Family III was based on the surface mechanism that "training targets are produced by the model itself"; however, the PDF source shows that the full method does not satisfy the survey's no-external-ground-truth strict UPT boundary.

- **GT / verifier dependency:** PDF §3.1 states that Naive BoN generates $N$ reasoning paths per training question and selects the "shortest correct reasoning path"; a footnote also notes that questions with no correct reasoning paths are excluded. The "correct" judgment requires a benchmark answer / parser / correctness check, not a purely internal signal.
- **External-exemplar dependency:** PDF §3.2 explicitly lists three sources for few-shot exemplars, including human annotation, proprietary frontier LLMs (GPT-4o), and self-generated samples; the main experiment uses `FS-GPT4o-BoN` as the headline combined method.
- **External baselines are not the only issue:** even ignoring FS-Human / FS-GPT4o, BoN data filtering itself depends on an external "correct path" judgment; therefore it is not strict UPT but **GT-filtered self-training / concise-rationale distillation**.
- **Still useful as adjacent:** the paper's analysis of "latent concise reasoning ability already present in the model's output distribution" is valuable and can be discussed in related work on selecting self-generated rationales, but it should not represent no-external-ground-truth strict UPT.

**Suggested positioning:** `semi-supervised / GT-filtered adjacent`; if it remains in the main text, write it as "self-generated trajectories + external correctness filtering" rather than strict UPT `Self-Generated Target Bootstrapping`.

---

## 2. Problem Addressed

Chain-of-thought (CoT) reasoning has clearly improved LLM performance on complex tasks, but the produced chains often contain a great deal of redundant tokens—wordy explanations, repeated phrases, and unnecessary context restatements—directly inflating inference latency and compute cost.

The authors argue that existing solutions are limited:

1. **Zero-shot prompting is unreliable:** strategies such as "Be Concise" and "Fixed Budget" are very inconsistent across models (Table 1) and are essentially ineffective on math-specialized models. Most prompting methods shrink length while crashing accuracy (e.g., Fixed Budget shortens by 32.2% on average but drops accuracy by 10.1%).
2. **Fine-tuning on external data sacrifices accuracy:** fine-tuning on concise reasoning written by humans or GPT-4o shortens length sharply but causes severe accuracy degradation due to distribution shift.
3. **Existing self-training baselines (Rational Metareasoning) only achieve about 12% average length reduction.**

Core observation: by inspecting the model's output distribution (Figure 1 & Figure 2), the authors find that the LLM already has **latent concise-reasoning ability**—the output distribution contains many shorter-than-average yet correct reasoning paths. The challenge is to elicit this latent ability effectively.

---

## 3. Method

### 3.1 Core Idea

Use the concise reasoning paths already present in the LLM's output distribution: through self-training (self-generated data + SFT), internalize the concise style into the model and avoid extra inference-time overhead.

### 3.2 Naive Best-of-N Sampling (BoN)

For each training question, sample $N$ reasoning paths ($N=16$) and pick the shortest correct one as a fine-tuning sample.

- **Question-wise selection:** select the shortest path per question (rather than from a global pool), ensuring training-sample coverage even on hard questions.
- **Limitation:** BoN's length reduction scales log-linearly with $N$ (Figure 3); further reduction requires exponentially more samples, and sample efficiency is low.

### 3.3 Few-Shot Conditioned Sampling (FS)

Use few-shot prompting to shift the sampling distribution so that the model naturally produces shorter reasoning paths. Three exemplar sources are considered:

| Method | Exemplar source | Notes |
|--------|------------------|-------|
| **FS-Human** | hand-written by Wei et al. (2022b) | off-the-shelf, very concise |
| **FS-GPT4o** | GPT-4o-generated | uses the Hand Crafted 3 prompt; preserves accuracy well |
| **FS-Self** | model-self-generated | two-stage: sample 128 paths each on 128 questions, sort by length → keep correct → use GPT-4o for quality verification, choose 8 |

Key finding (Figure 3): single-pass 8-shot conditioning already exceeds the length reduction of BoN with $N=256$.

### 3.4 Few-Shot Conditioned BoN Sampling (FS-BoN)

Combine few-shot conditioning with BoN sampling: BoN-sample on top of the few-shot-conditioned distribution. The two improvements are largely independent and stack additively (Figure 3), producing maximum length reduction. The main experiment adopts FS-GPT4o-BoN as the final method.

### 3.5 Sample Augmentation

Few-shot prompting has limited adaptability: it may (1) suppress correct paths on hard questions or (2) inject extra steps on extremely simple questions. To fix this, for each question, merge FS / FS-BoN samples with naive-BoN samples ($N$ items) and pick the shortest correct path from the union.

This augmentation maintains length reduction while improving accuracy (Figure 4) by widening coverage on hard questions.

### 3.6 Training Details

- SFT with HuggingFace Trainer, generation with vLLM.
- Fine-tuning: batch size 16, 1 epoch, learning rate 1e-5, bfloat16.
- Generation: temperature $T=0.7$; max 512 tokens for GSM8K, 1024 tokens for MATH.
- Training cost is tiny: fine-tuning takes about 2 minutes on a single GPU (Table 5: generation ≈1.5 h vs. training ≈2.4 min).

---

## 4. Datasets

### Training and Evaluation Data

| Dataset | Train | Test | Difficulty | License |
|---------|-------|------|------------|---------|
| **GSM8K** | 7,473 | 1,319 | grade-school math word problems, multi-step reasoning | MIT |
| **MATH** | 7,500 | 500 (MATH-500) | advanced math, 5 difficulty levels (algebra → advanced calculus) | MIT |

### Extra Evaluation Domains (MMLU-Pro)

| Domain | Notes |
|--------|-------|
| Business | business reasoning |
| Chemistry | chemistry reasoning |
| Physics | physics reasoning |

### Models

**Five main models:**
- Llama-3.2-3B (general)
- Gemma-2-2B (general)
- Qwen2.5-3B (general)
- Qwen2.5-Math-1.5B (math-specialized)
- DeepSeekMath-7B (math-specialized)

**Scaling study:** Llama-3.2-{1B, 3B}, Llama-3.1-8B.

---

## 5. Evaluation metrics and main results
### Evaluation Metrics

- **Accuracy:** Python-based answer parsing on the final answer.
- **Length:** average output token count over all outputs (including incorrect paths).
- **Relative Accuracy / Relative Length:** percentages relative to a baseline prompt (Pang et al., 2024) for cross-method comparison.
- Evaluation uses **greedy decoding** for reproducibility.

### Main Results (Table 2, average over five models)

| Method | GSM8K Rel. Acc (%) | GSM8K Rel. Len (%) | MATH Rel. Acc (%) | MATH Rel. Len (%) |
|--------|--------------------|--------------------|--------------------|--------------------|
| Baseline | 100.00 | 100.00 | 100.00 | 100.00 |
| Be Concise | 99.85 | 88.46 | 102.71 | 92.66 |
| Hand Crafted 2 | 98.27 | 77.10 | 101.62 | 85.26 |
| Human CoT (External) | 83.82 | 54.95 | 75.61 | 53.14 |
| GPT4o CoT (External) | 97.65 | 67.60 | 90.52 | 87.21 |
| Naive BoN | 98.79 | 87.17 | 101.74 | 89.89 |
| Rational Metareasoning | 97.21 | 84.93 | 103.02 | 90.56 |
| **FS-Human (Ours)** | 98.06 | 67.96 | 99.69 | 87.78 |
| **FS-GPT4o (Ours)** | 99.94 | 73.15 | 101.87 | 87.58 |
| **FS-Self (Ours)** | 98.86 | 77.51 | 102.67 | 88.50 |
| **FS-GPT4o-BoN (Ours)** | 97.00 | **64.25** | 102.56 | **76.30** |
| FS-GPT4o-BoN Budget-Matched | 97.44 | 67.15 | 101.58 | 80.43 |

### Key Findings

1. **FS-GPT4o-BoN delivers ≈30% average token reduction** (GSM8K: 35.75%, MATH: 23.70%) while keeping average accuracy essentially unchanged—**2.4×** the previous fine-tuning baseline (Rational Metareasoning, ~12% reduction).

2. **Self-training clearly beats external-data fine-tuning (Figure 4):** fine-tuning on GPT-4o external data shortens more but lands below the Pareto curve (severe accuracy loss); self-training methods sit on the Pareto frontier.

3. **Few-shot conditioning works on math-specialized models too:** zero-shot prompting is essentially useless on Qwen2.5-Math and DeepSeekMath (gray-shaded in Table 1), but FS methods still deliver substantial reductions on these models (Tables 13, 14).

4. **Adaptive token reduction (Figure 5):** the model adapts the reduction by question difficulty—Level 1-2 problems get 20%–40% shorter, Level 5 problems shorten less—showing the method removes real redundancy rather than essential reasoning steps.

5. **Scaling consistency (Figure 6):** on Llama 1B / 3B / 8B, the token-reduction rate grows with model size while accuracy stays stable; the 8B model shows the largest reduction with the best accuracy retention under FS-GPT4o-BoN.

6. **Training length transfers linearly to test length (Figure 7):** the relative length of the fine-tuning data is strongly linearly correlated with the model's relative test-output length, indicating SFT effectively conveys length reduction to the model.

7. **Cross-task generalization (Table 9):** training on one dataset still yields 10–12% length reduction on another OOD dataset with no more than 1.5 percentage points of accuracy loss.

8. **Broader-domain effectiveness (Table 10):** on the MMLU-Pro Business / Chemistry / Physics domains, FS-GPT4o-BoN simultaneously improves accuracy (avg. +16.51%) and shortens length (avg. 26.82%), with length efficiency more than 3× that of naive BoN.

### Real Efficiency Gains

| Metric | Improvement range |
|--------|-------------------|
| Wall-clock latency reduction | 15.38% – 52.94% (Table 7) |
| Peak-memory reduction | 2.50% – 6.26% (Table 8) |
| Fine-tuning overhead | only ≈2.4 min / single GPU (vs. ≈1.5 h for generation) |

### Qualitative Example (Table 3)

A GSM8K problem on Llama-3.1-8B:

- **Original output (Zero-Shot):** "To find the total number of bolts needed, we need to calculate the amount of white fiber first, since it's half the amount of blue fiber. Step 1: Determine the amount of blue fiber needed…" (a long step-by-step restatement of the question).
- **FS-GPT4o-BoN output:** "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2/2 = 1 bolt. Add the blue and white fiber together: 2 + 1 = 3 bolts. The answer is 3." (only the key computation steps, redundant explanation removed).
