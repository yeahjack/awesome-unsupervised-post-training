# When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning

**Paper:** When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning
**Authors:** Zhengxian Wu, Kai Shi, Chuanrui Zhang, Zirui Liao, Jun Yang, Ni Yang, Qiuying Peng, Luyuan Zhang, Hangrui Xu, Tianhuang Su, Zhenyu Yang, Haonan Lu, Haoqian Wang
**arXiv:** 2603.21289
**Venue:** arXiv 2026
**Code:** https://github.com/OPPO-Mente-Lab/LLM-Self-Judge

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| When Models Judge Themselves | Actor + frozen same-lineage Judge + GRPO | training-time multimodal UPT | self-consistency with internal judge modulation |

| Collection status | Value |
|-------------------|-------|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2603.21289.pdf` |
| Extracted text | `0524_new_collection/texts/2603.21289.txt` |
| Basis for this note | full downloaded PDF + extracted text; source-audit re-checked on 2026-05-24 against Actor / Judge initialization, frozen-Judge modulation, self-consistency rewards, GRPO updates, and the no-human-answers / no-external-reward-model claims |
| BibTeX | `0524_new_collection/bib/2603.21289.bib` |
| Suggested taxonomy | strict UPT candidate; Family IV / II hybrid |

## 1. UPT Assignment Rationale
This is a strict UPT multimodal UPT candidate. It targets multimodal reasoning, initializing both Actor and Judge from the same multimodal model; the Judge is a structurally identical frozen copy of the Actor policy, no external reward model. Training data are used as unlabeled multimodal inputs—no human-annotated answers. The Actor samples multiple reasoning trajectories per input; a self-consistency distributional prior comes first, then a frozen Judge applies a bounded-score modulation to adjust trajectory weights, and finally GRPO updates the Actor.

Under the current taxonomy it is a hybrid of **Family IV: Internal Evaluator Bootstrapping** and **Family II: Sample-Relation Supervision**. Family II shows up in same-input multi-trajectory self-consistency; Family IV shows up in the same-lineage frozen Judge modulating candidate-trajectory quality.

## 2. Boundary Audit

| Check | Judgment | Evidence |
|-------|----------|----------|
| B1 update-bearing | Pass | the Actor uses GRPO for unsupervised post-training |
| B2 internal signal | Pass | the reward prior comes from Actor self-consistency; Judge modulation comes from a same-lineage frozen copy |
| B3 no external supervision | Pass with dataset caveat | training uses raw images / RL-stage QA inputs as the unlabeled training set, no human-annotated answers |
| B4 internal judge / scorer | Pass | the Judge is initialized from the same base Actor and frozen; not an external reward model |

## 3. Method

The method has two modules, Actor and Judge. The Actor samples multiple reasoning trajectories per multimodal input. A first reward distribution comes from answer aggregation and self-consistency: trajectories that match the group consensus get a higher prior.

A frozen Judge is then introduced. The Judge has the same structure as the Actor and is initialized from a copy of the current Actor policy, then kept frozen during training. The Judge gives a score per trajectory, but this score is not the final reward; it goes through a bounded calibration function and becomes a modulation signal that continuously re-weights the self-consistency distribution. This avoids the incomparability of raw Judge scores across inputs and prevents a noisy Judge from dominating training.

Finally, the modulated scores are modeled as a group-level distribution and converted into relative advantages used by GRPO to update the Actor.

## 4. Datasets

Training uses inputs from several multimodal datasets. Some datasets contribute only raw images, with no human-annotated QA pairs or metadata; in other settings the RL-stage QA split is used as an unlabeled training set, i.e., the question and image are used as input but the answer is not used as supervision.

Evaluation covers several multimodal mathematical-reasoning benchmarks—MathVision, MathVerse, DynaMath, WeMath, LogicVista, and so on—and extends to non-math vision-centric benchmarks.

The main model is Qwen2.5-VL-7B-Instruct; both Actor and Judge are initialized from it, and the Judge is frozen.

## 5. Evaluation metrics and main results
The paper reports steady gains on several multimodal mathematical-reasoning benchmarks, beating majority-vote and self-consistency baselines. Ablations show that Judge-only is worse than the combination of self-consistency + Judge modulation; distributional reward modeling mitigates mode collapse and noisy-score effects.

Compared with supervised GRPO, the method matches or surpasses labeled-training pass@10 in some settings, with markedly lower training time. The authors also analyze the relation between self-consistency and Judge preference: after training, top-1 agreement between the two grows but does not saturate, suggesting the two internal signals are complementary.

## 6. Position in the UPT Survey
Recommended as a new multimodal Family IV candidate. Inside the MLLM / VLM landscape it is a very direct internal-evaluator-bootstrapping case: the judge is not an external model but a same-lineage frozen copy, and the judge score only applies bounded modulation.

Recommended short labels: `Self-Judge MLLM` or `Models Judge Themselves`.
Recommended family: `Internal Evaluator Bootstrapping`, secondary `Sample-Relation Supervision`.
Main-table caveat: training inputs come from public multimodal QA / image datasets, but answers / annotations are not used as training supervision.
