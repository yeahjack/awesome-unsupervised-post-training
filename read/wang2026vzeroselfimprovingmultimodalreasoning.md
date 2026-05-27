# V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation

**Paper:** V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation
**Authors:** Han Wang, Yi Yang, Jingyuan Hu, Minfeng Zhu, Wei Chen
**arXiv:** 2601.10094
**Venue:** arXiv 2026
**Code:** https://github.com/SatonoDia/V-Zero

| Method | Carrier | Regime | Level |
|---|---|---|---|
| V-Zero | Questioner/Solver VLM co-evolution + GRPO/RLVR | training-time self-improvement from unlabeled images | self-generated questions and majority-vote pseudo-labels |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or current `0524_new_collection` |
| Screening source | Chrome MCP arXiv search/page text used only for candidate discovery |
| PDF | `0524_new_collection/pdfs/2601.10094.pdf` |
| Extracted text | `0524_new_collection/texts/2601.10094.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit checked on 2026-05-24 against Questioner/Solver loop, dual-track reasoning reward, majority pseudo-labeling, raw-image-only training claim, GRPO updates, baselines, and appendix data details |
| BibTeX | `0524_new_collection/bib/2601.10094.bib` |
| Suggested taxonomy | strict UPT candidate; Family III (primary, paper §F3 listing) with auxiliary Family II consensus signal |

## 1. UPT Assignment Rationale

V-Zero fits the current survey's strict UPT MLLM / VLM scope well. It performs post-training on raw unlabeled images and splits one base VLM into Questioner and Solver roles. The Questioner generates challenging multiple-choice questions from images; the Solver samples multiple reasoning answers, obtains a pseudo-label via majority voting, and uses the pseudo-label to train itself.

It satisfies the key strict UPT criteria: no human annotations, no external reward model, no external verifier; training signals come from Questioner/Solver interaction, majority vote, confidence/difficulty filtering, and a dual-track reasoning reward.

## 2. Boundary check

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Both Questioner and Solver are iteratively trained with GRPO / RLVR |
| B2 internal signal | Pass | Questioner reward comes from the relation between the intuitive answer and the Solver's majority reasoning answer; Solver pseudo-label comes from majority vote over its own samples |
| B3 no external supervision | Pass with image-source caveat | Training uses OpenVLThinker image pools but only the raw unlabeled images, not their annotations |
| B4 internal judge/scorer | Pass | No external judge; Qwen3-VL-32B is used only for question-difficulty analysis, not training reward |

## 3. Method

V-Zero's closed loop has Questioner training and Solver training:

1. The Questioner reads a raw image and produces a visual description, multiple-choice question, and a fast intuitive answer;
2. The frozen Solver samples multiple CoT reasoning responses for the question;
3. Majority vote over the multiple responses yields the reasoning-track pseudo-label, and the vote proportion becomes the confidence;
4. The Questioner is updated with the dual-track reasoning reward: when intuitive and reasoning answers disagree and the reasoning majority is high-confidence, the question is valuable;
5. The optimized Questioner generates training questions for raw images;
6. The Solver samples answers for these questions; majority-vote pseudo-labels and difficulty-guided filtering form the training set;
7. The Solver updates via RLVR / GRPO with a binary-accuracy reward against the pseudo-label.

This loop matches Family III/II well: Family III in the self-generated questions / synthetic training tasks; Family II in the majority-pseudo-label and group-level agreement signal.

## 4. Datasets

Training data come from image pools of OpenVLThinker-GRPO-medium and OpenVLThinker-GRPO-hard, about 9K images covering geometry, tables, spatial reasoning. The paper emphasizes that after pre-filtering, only raw, unlabeled image data are used; the appendix says about 4,000 images are used in the end.

Models are Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct. Training is built on verl / vLLM; Questioner group size is 4, Solver group size is 5, and the majority-vote sample size is 10.

## 5. Evaluation metrics and main results
Evaluation uses VLMEvalKit, covering general vision-centric tasks and visual mathematical reasoning, including MMMU, MMStar, MathVista, LogicVista. The paper compares against the base model, human-annotated supervised GRPO, and OpenVLThinker-7B baselines.

Main result: V-Zero stably improves Qwen2.5-VL-7B-Instruct without any human annotation. The abstract reports visual mathematical reasoning +1.7 and general vision-centric +2.6. The main text further shows V-Zero can exceed the supervised-GRPO baseline on the same image dataset — the dynamic self-generated question and pseudo-label loop produces training signals better adapted to the current model than static annotated training sets.

## 6. Position in the UPT Survey
Suggested short label: `V-Zero`.

Suggested family: strict UPT **Family III (primary) with Family II consensus signal**:

V-Zero trains a Questioner/Solver VLM loop on unlabeled images, where questions are self-generated and Solver pseudo-labels are produced by majority voting over its own sampled reasoning answers.

It can sit alongside `VisPlay`, `RISE`, `EvoLMM`, but V-Zero's "zero annotation" framing is the most direct, making it a clean MLLM strict UPT extension.
