# CYCLE-INSTRUCT: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency

> **Added to survey on:** 2026-03-11

- **Method**: CYCLE-INSTRUCT
- **Carrier**: Direct Opt.
- **Regime**: Training-time
- **Level**: Semantic
- **Venue**: EMNLP 2025

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

CYCLE-INSTRUCT belongs to **Family III (Self-Generated Target Bootstrapping, knowledge/instruction self-curation)**. The method does not rely on any human-annotated seed data, external teacher model, or external verifier. Its core idea is a dual self-training loop combined with cycle consistency to automatically generate and filter instruction–response pairs from raw unlabeled text. Two models (the answer generator and the question generator) are trained only on each other's pseudo-labels and on the reconstruction error against the original text; all supervisory signals come from the intrinsic consistency between model-generated content and the original text — matching the UPT definition that "model-generated content" and "intrinsic statistics" are legitimate signal sources.

---

## 2. Problem Addressed

Traditional instruction tuning relies on large amounts of human-annotated data or strong external teacher models (e.g., GPT-4) to generate synthetic data. Even instruction back-translation methods still require a carefully curated human seed set to bootstrap the generation process. This **seed dependency** introduces three core bottlenecks:

1. **High seed-collection cost**: building a high-quality, diverse seed set requires significant human effort;
2. **Data waste and distribution shift**: style and topic biases of a small seed set propagate into the synthetic data, harming diversity, and treating all unlabeled paragraphs uniformly as answers wastes data;
3. **Limited scalability**: in privacy-sensitive settings or specialized domains, acquiring seed data or invoking external models is often infeasible.

CYCLE-INSTRUCT aims at **fully seed-free** instruction tuning, generating high-quality instruction-following data starting solely from raw unlabeled text.

---

## 3. Method

### 3.1 Data Segmentation & Reformat

Split raw documents into paragraphs and use a simple heuristic (presence of "?" → question; otherwise → answer) to partition them into two sets:
- $\mathcal{D}_Q^{\text{raw}}$: potential question passages
- $\mathcal{D}_A^{\text{raw}}$: potential answer passages

A fixed rewriting prompt then converts raw paragraphs into standard instruction format (question → self-contained question) and response format (answer → coherent answer paragraph), yielding $\mathcal{D}_Q$ and $\mathcal{D}_A$.

### 3.2 Cycle Training Procedure

Two transformer models are instantiated from the same base model (e.g., LLaMA-3.1-8B):
- **Forward model** $\mathcal{M}_{Q \to A}$: generates a response given an instruction
- **Backward model** $\mathcal{M}_{A \to Q}$: generates an instruction given a response

Training iterates in 4-step cycles:

1. **Step 1 — Pseudo-Answer Generation**: use $\mathcal{M}_{Q \to A}$ to produce a pseudo-answer $\hat{a}_i$ for each question $q_i$ in $\mathcal{D}_Q$;
2. **Step 2 — Backward Model Training**: train $\mathcal{M}_{A \to Q}$ on $(q_i, \hat{a}_i)$ pairs, minimizing the negative log-likelihood of reconstructing the original question from the pseudo-answer;
3. **Step 3 — Pseudo-Instruction Generation**: use $\mathcal{M}_{A \to Q}$ to produce a pseudo-instruction $\hat{q}_j$ for each answer $a_j$ in $\mathcal{D}_A$;
4. **Step 4 — Forward Model Training**: train $\mathcal{M}_{Q \to A}$ on $(\hat{q}_j, a_j)$ pairs, reconstructing the original answer from the pseudo-instruction.

The core training signal comes from the **cycle-consistency loss**: the ground-truth end of each sample is always the original text itself, and the model must reconstruct the original text conditioned on the pseudo-label produced by the other model, providing mutual supervision between the two models.

### 3.3 Cycle-Consistency Filtering (optional)

The final dataset $\mathcal{D}_{\text{final}}$ merges pseudo-labeled pairs from both directions. Quality can be further improved through cycle-consistency filtering:

1. Pass the pseudo end of each pair into the opposite model for **one-step reconstruction**;
2. Compute the **embedding distance** between the original label and the reconstructed label (using a LLaMA-3 inference encoder);
3. Apply **semantic clustering** to the gold side (k-means, k=200);
4. Within each cluster, perform **k-center greedy pruning** to remove the top 5% samples with the largest embedding distance.

After filtering, $\mathcal{D}_{\text{cycle}} \approx 0.95 \cdot |\mathcal{D}_{\text{final}}|$.

---

## 4. Datasets

Experiments cover four data tracks:

| Track | Dataset | Pairs Used | Unlab. Q | Unlab. A |
|-------|--------|-----------|----------|----------|
| Track 1: General Instructions | Alpaca-GPT4 | 20,000 | 10,000 | 10,000 |
| Track 1: General Instructions | Dolly-15k | 15,000 | 7,500 | 7,500 |
| Track 2: Domain-Specific | Medical-Alpaca | 20,000 | 10,000 | 10,000 |
| Track 3: Dialogue Logs | OASST-1 | 40,000 | 15,126 | 24,874 |
| Track 4: Plain Text | WikiHow-4w | 40,000 | 5,178 | 34,822 |

- Track 1 uses existing instruction datasets, randomly removing one side of a pair to simulate the unlabeled setting;
- Track 2 uses Medical-Alpaca (medical domain), again removing one side to simulate missing annotation;
- Track 3 uses OASST-1 dialogue logs, identifying question/answer turns via the "?" heuristic;
- Track 4 uses WikiHow articles, extracting potential Q–A paragraphs from narrative text.

---

## 5. Evaluation metrics and main results
### Metrics

- **Standard instruction metrics**: MMLU (accuracy %), BBH, CRASS, DROP — using the InstructEval framework
- **Medical track**: 8 medical sub-fields of MMLU (CK, CB, CC, CM, HB, HC, MG, PM)
- **Open-ended quality**: AlpacaEval win-rate (pairwise comparison with the ALL-SFT baseline)
- **Synthetic data quality**: GPT-4o Mini alignment score on synthetic QA pairs (0–10 scale)

### Main results

All experiments use **LLaMA-3.1-8B-Base + LoRA** fine-tuning.

1. **Comprehensively surpasses back-translation baselines**: across all four tracks, CYCLE-INST and CYCLE-FILT (0% human annotation) significantly outperform all seed-based back-translation variants (Random-k%, Cluster-k% with 5–20% seed data).

2. **Approaches or exceeds fully supervised methods**:
   - **Alpaca-GPT4** (Table 1): Cycle-Filt Avg 54.17 vs. ALL-SFT (100% gold) 54.99 — a tiny gap;
   - **Dolly-15k** (Table 1): Cycle-Filt Avg 51.00 vs. ALL-SFT 53.45;
   - **Medical-Alpaca** (Table 2): Cycle-Filt Avg **0.636** beats ALL-SFT 0.626;
   - **OASST-1** (Table 4): Cycle-Filt Avg **51.98** beats ALL-SFT 51.69;
   - **WikiHow** (Table 4): Cycle-Filt Avg **51.07** beats ALL-SFT 50.59.

3. **Effectiveness of cycle-consistency filtering**: CYCLE-FILT systematically outperforms the unfiltered CYCLE-INST across all benchmarks, confirming that reconstruction verification can effectively remove noisy pseudo-pairs.

4. **AlpacaEval**: on OASST-1 and WikiHow, Cycle-Filt clearly beats the ALL-SFT win-rate; on Alpaca and Dolly it is also competitive.

5. **GPT-4o Mini QA alignment scores**: alignment quality of CYCLE-INST synthetic pairs is higher than all seed-based methods across datasets (Alpaca 9.46, Dolly 9.90, MedAlpaca 9.96, WikiHow 9.43).
