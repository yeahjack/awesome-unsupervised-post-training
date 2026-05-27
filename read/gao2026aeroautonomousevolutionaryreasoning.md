# AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback

**Paper:** AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback
**Authors:** Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu
**arXiv:** 2602.03084
**Venue:** arXiv 2026
**Code:** https://github.com/mira-ai-lab/AERO

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| AERO | LoRA + KTO | iterative self-evolution | self-generated tasks, answers, and internal criticism |

| Collection status | Value |
|-------------------|-------|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2602.03084.pdf` |
| Extracted text | `0524_new_collection/texts/2602.03084.txt` |
| Basis for this note | full downloaded PDF + extracted text; source-audit re-checked on 2026-05-24 against Generator / Solver / Refiner roles, entropy-based ZPD, ICC, KTO / LoRA updates, and external-verifier caveats |
| BibTeX | `0524_new_collection/bib/2602.03084.bib` |
| Suggested taxonomy | strict UPT candidate; Family IV (primary) with auxiliary Family III aspects (paper §F4 places AERO under Family IV) |

## 1. UPT Assignment Rationale
AERO is a strict UPT candidate. It tries to keep the same LLM cycling among self-questioning, self-answering, and self-criticism—generating tasks, answers, and reflections—then constructs preference data and updates the model parameters with KTO, all without any external annotated data or external verifier.

Under the current taxonomy it is a hybrid of **Family IV: Internal Evaluator Bootstrapping** and **Family III: Self-Generated Target Bootstrapping**. Family III shows up in the model self-generating training tasks and answers; Family IV shows up in ICC-based logical verification / self-criticism producing truth proxies and preference signals. If the main table allows only one row, list it primarily under Family IV, because the key novelty is replacing the external verifier with an endogenous critic.

## 2. Boundary Audit

| Check | Judgment | Evidence |
|-------|----------|----------|
| B1 update-bearing | Pass | the outer loop updates parameters via KTO / LoRA |
| B2 internal signal | Pass | tasks, answers, criticism, and pseudo-labels are all produced by role-switching of the same LLM |
| B3 no external supervision | Pass with analysis caveat | training claims no external data, human labels, or verifiers; DeepSeek-R1 is only used as a proxy reference for ICC-reliability analysis |
| B4 internal judge / scorer | Pass | the verifier is replaced by ICC and self-criticism; no external judge is called during training |

## 3. Method

AERO has an inner and an outer loop.

In the inner loop, the same LLM activates three roles via prompting:

- Generator: generates candidate reasoning tasks;
- Solver: samples multiple reasoning paths per task;
- Refiner / Critic: re-examines reasoning paths in a counterfactual setting.

The paper uses normalized Shannon entropy over answer clusters to decide whether a task lies in the Zone of Mastery, the Zone of Proximal Development, or the Zone of Chaos. Only tasks in the ZPD are kept—neither too easy nor purely random.

The core verification mechanism is **Independent Counterfactual Correction (ICC)**. For top answer clusters, the Refiner is asked to solve again under the counterfactual hypothesis "the prior answer might be wrong"; if the independent correction paths converge, the result is taken as a high-confidence truth proxy, otherwise it is dropped. This mechanism replaces an external verifier with internal logical consistency.

The outer loop turns the inner-loop data into three preference datasets (generator / solver / refiner) and runs staggered KTO. The staggered design avoids cross-degradation when generator, solver, and critic update simultaneously.

## 4. Datasets

AERO's training data is fully self-generated. The paper stresses that it is data-free / verifier-free self-evolution, with no human-annotated training set.

Evaluation covers nine reasoning benchmarks spanning general, mathematical, and physical reasoning. Headline numbers report Qwen3-4B-Base and Qwen3-8B-Base, with extensions to Llama-3.2, Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, etc.

Caveat: in the ICC pseudo-label-reliability analysis, the paper uses DeepSeek-R1 responses as proxy reference labels purely to quantify ICC precision; this is not a training signal.

## 5. Evaluation metrics and main results
The paper reports an average +4.57% on Qwen3-4B-Base and +5.10% on Qwen3-8B-Base, beating several self-evolution baselines on the nine benchmarks. Ablations show that removing self-criticism, ZPD, or ICC each lowers average performance, supporting the joint role of "task-difficulty placement + internal logical verification + staged updates".

The ICC-reliability analysis shows that, compared with majority voting, ICC pseudo-label precision is higher. This is useful for the survey, because it shows an internal evaluator does not have to be a direct self-judge—counterfactual consistency can build a stronger endogenous verification.

## 6. Position in the UPT Survey
Recommended as a frontier strict UPT candidate. It is closer than ordinary self-training to a complete self-evolution loop of "internal task factory + internal verifier + preference optimization".

Recommended short label: `AERO`.
Recommended family: primary `Internal Evaluator Bootstrapping`, secondary `Self-Generated Target Bootstrapping`.
Main-table caveat: DeepSeek-R1 only appears in the reliability analysis; when listing AERO, note whether the training signal is fully endogenous.
