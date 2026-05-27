# QueST: Query-Conditioned Test-Time Self-Training

**Paper:** Query-Conditioned Test-Time Self-Training for Large Language Models
**Authors:** Chaehee Song, Minseok Seo, Yeeun Seong, Doyi Kim, Changick Kim
**arXiv:** 2605.13369
**Venue:** arXiv 2026
**Code:** https://chssong.github.io/Query-Conditioned-TTST/

| Method | Carrier | Regime | Level |
|---|---|---|---|
| QueST | LoRA / SFT | test-time | Semantic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.13369.pdf` |
| Extracted text | `0524_new_collection/texts/2605.13369.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against query-conditioned auxiliary problem generation, LoRA/SFT test-time updates, and no-external-data claims |
| BibTeX | `0524_new_collection/bib/2605.13369.bib` |
| Suggested taxonomy | strict UPT candidate; Family III |

## 1. UPT Assignment Rationale

QueST is a clear update-bearing test-time adaptation method. Given a user query, it first generates query-conditioned auxiliary problem-solution pairs, then performs a small LoRA SFT update at test time, and finally uses the adapted model to answer the original query. It does not retrieve external data or use ground-truth labels or external verifiers.

The dominant training object is the model-derived auxiliary problem-solution pairs from the current query, so it should be placed in **Family III: Self-Generated Target Bootstrapping**.

## 2. Boundary check

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | LoRA parameters are initialized and a test-time SFT update is performed per query |
| B2 internal signal | Pass | Supervision comes from query-conditioned generated problem-solution pairs |
| B3 no external supervision | Pass | No external data; benchmark answers are used only for evaluation |
| B4 internal judge/scorer | Pass | No judge / reward model is used |

## 3. Method

QueST pipeline:

1. Input the original query $q$;
2. Generate $N$ auxiliary problem-solution pairs $D(q)$ structurally related to $q$;
3. Initialize LoRA parameters;
4. Perform $T$ steps of test-time SFT on the model with $D(q)$;
5. Use the adapted model to answer $q$;
6. Do not accumulate to the next query; per-query adaptation by default.

Difference from TLM / TTT-NN: TLM only optimizes input perplexity; TTT-NN depends on an external retrieval corpus; QueST's supervision is generated directly from the current query.

## 4. Datasets

Evaluation covers seven math-reasoning benchmarks:

- AMC
- Minerva
- MATH-500
- GSM8K
- OlympiadBench
- AIME 2024
- AIME 2025

Plus GPQA-Diamond for science reasoning.

## 5. Evaluation metrics and main results
The main metric is accuracy / pass@1, compared with test-time scaling, TENT, TLM, and other baselines. The paper reports broad improvements on math and GPQA-Diamond and uses fewer test-time tokens than sampling-style test-time scaling.

Ablations show that query conditioning, LoRA adaptation, and self-generated QA all contribute together; self-QA without query conditioning has limited effect.

## 6. Position in the UPT Survey
Recommended as a new strict UPT candidate in Family III. It strengthens the survey's "Test-Time Instance Adaptation / per-query self-generated-target bootstrapping" slot: relative to `TTT-NN`'s nearest-neighbor external data and `TLM`'s prediction-statistic loss, QueST explicitly constructs self-generated targets.

Suggested main-table short label: `QueST`.
Suggested family: `Self-Generated Target Bootstrapping`.
Suggested timing: `Test-Time Instance Adaptation`.
Suggested caveat: by default, LoRA updates reset at the query boundary, not cumulative deployment learning.
