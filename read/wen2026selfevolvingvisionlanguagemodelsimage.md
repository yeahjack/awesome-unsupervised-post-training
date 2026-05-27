# EvoQuality: Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking

**Paper:** Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking  
**Authors:** Wen Wen, Tianwu Zhi, Kanglong Fan, Yang Li, Xinge Peng, Yabin Zhang, Yiting Liao, Junlin Li, Li Zhang  
**arXiv:** 2509.25787  
**Venue:** ICLR 2026  
**Code:** not found in the extracted text.

| Method | Carrier | Regime | Level |
|---|---|---|---|
| EvoQuality | VLM + pairwise ranking pseudo-labels + GRPO | training-time self-evolution on unlabeled image pairs | majority-voted pairwise preference, fidelity reward |

| Collection status | Value |
|---|---|
| Duplicate check | Not found as a standalone note in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or the current collection. |
| PDF | `0524_new_collection/pdfs/2509.25787.pdf` |
| Extracted text | `0524_new_collection/texts/2509.25787.txt` |
| Basis for this note | Complete downloaded PDF + extracted full text; source audit checked on 2026-05-24 against the pairwise majority-vote pseudo-ranking loop and the explicit discard of KONIQ ground-truth quality labels. |
| BibTeX | `0524_new_collection/bib/2509.25787.bib` |
| Suggested taxonomy | strict UPT candidate; Family II with a task-specific IQA caveat. |

## 1. UPT Assignment Rationale

EvoQuality is a VLM's self-supervised post-training on the Image Quality Assessment (IQA) task. It does not use ground-truth quality scores; instead it lets the VLM compare image pairs multiple times, obtains pseudo-ranking labels via pairwise majority voting, then turns these pseudo-rankings into a fidelity reward and updates the model with GRPO.

The method fits **Family II: Sample-Relation Supervision**: the supervision signal is not an external label per sample but the model's own relative judgments and majority agreement over multiple image pairs. It also has a Family I (Prediction-Statistic) flavor, since the task input itself is images and the reward is derived from the same model's relational comparisons.

## 2. Boundary checks

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | The online stage fine-tunes the VLM policy with GRPO. |
| B2 internal signal | Pass | Pseudo-ranking labels come from the same VLM's repeated pairwise voting on image pairs. |
| B3 no external supervision | Pass | The paper explicitly discards KONIQ quality scores and distortion labels. |
| B4 internal judge/scorer | Pass | No external reward model or stronger teacher is used. |

## 3. Method

EvoQuality has two stages — offline and online:

- Offline pseudo-ranking: for each image pair, the VLM generates multiple comparisons, and majority voting yields the pseudo preference of which image has higher quality.
- Online GRPO: insert the pseudo preference into a fidelity reward; train the model to produce more consistent quality judgments.
- Iterative evolution: the trained model can be used for the next round of pseudo-label generation, progressively improving IQA ability.

The key design is to use ranking rather than absolute scores as the self-supervision target, avoiding direct regression on noisy pseudo-scores.

## 4. Datasets

Training uses only the KONIQ image pool and explicitly discards ground-truth quality scores, distortion types, and severity labels. Evaluation covers multiple IQA benchmarks, comparing zero-shot VLMs, supervised VLM-IQA, and EvoQuality variants using standard IQA metrics such as PLCC / SRCC.

## 5. Evaluation metrics and main results

The paper reports that EvoQuality substantially improves Qwen2.5-VL-7B's zero-shot IQA performance on multiple IQA benchmarks; the abstract reports an average +31.8% in PLCC. On several benchmarks the results match or exceed supervised VLM-based IQA models. Ablations show pairwise ranking + majority vote is more stable than direct pseudo-score regression.

## 6. Position in the UPT Survey

Recommended as a new strict UPT candidate, with a note that it is a domain-specific perceptual self-evolution. It complements existing MLLM/VLM self-evolution entries by covering the IQA / perceptual-ranking scenario.
