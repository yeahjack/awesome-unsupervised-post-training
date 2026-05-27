# Multi-Reward RLIF: Two is Better than One

**Paper:** Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework
**Authors:** Shourov Joarder, Diganta Sikdar, Ahsan Habib Akash, Binod Bhattarai, Prashnna Gyawali
**arXiv:** 2605.22620
**Venue:** arXiv 2026
**Code:** announced as forthcoming

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Multi-Reward RLIF | Policy Opt. / GRPO | training-time RLIF | Semantic + token |

| Collection status | Value |
|-------------------|-------|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.22620.pdf` |
| Extracted text | `0524_new_collection/texts/2605.22620.txt` |
| Basis for this note | full downloaded PDF + extracted text; source-audit re-checked on 2026-05-24 against RLIF, cluster voting, self-certainty, KL-Cov, and no-external-supervision claims |
| BibTeX | `0524_new_collection/bib/2605.22620.bib` |
| Suggested taxonomy | strict UPT candidate; Family II dominant |

## 1. UPT Assignment Rationale
This paper is a very direct strict UPT candidate. It starts from a post-trained LLM, runs GRPO updates with unlabeled training prompts, and uses no human labels, no gold answers, no external verifier, and no external reward model. The paper explicitly positions the method as RLIF: the reward signal comes from the model's own outputs.

Two core training signals:

- answer-level cluster-voting reward: extract answers across multiple rollouts of the same prompt, cluster by equivalent answers, and use cluster size as the agreement reward;
- completion-level self-certainty reward: use the self-certainty of the token-wise predictive distribution as a confidence reward.

Under the survey's dominant-update-object rule, the dominant object is the agreement relation across multiple rollouts / answers, so the recommended placement is **Family II: Sample-Relation Supervision**. Self-certainty is a Family-I-style auxiliary signal but is neither the only nor the dominant update target.

## 2. Boundary Audit

| Check | Judgment | Evidence |
|-------|----------|----------|
| B1 update-bearing | Pass | uses GRPO full fine-tuning to update model parameters |
| B2 internal signal | Pass | both cluster voting and self-certainty come from the same model's rollouts / distribution |
| B3 no external supervision | Pass | the paper repeatedly stresses no use of gold-standard solutions or human annotations |
| B4 internal judge / scorer | Pass | no external judge; scoring comes from cluster relations and the model's probability |

## 3. Method

The method targets the issue that a single internal reward easily collapses. It splits the RLIF reward into two complementary channels:

1. Build answer clusters from rollout final answers and reward by cluster size.
2. Compute a self-certainty reward per completion.
3. Use GDPO-style per-channel normalization to prevent one reward's scale from suppressing the other.
4. Apply KL-Cov regularization specifically targeting the high-covariance tokens that drive entropy collapse.
5. Update the policy with a GRPO-style objective.

The paper's technical focus is not on proposing a brand-new UPT signal source but on showing how multiple reward channels under internal feedback complement each other, with KL-Cov preventing entropy collapse.

## 4. Datasets

Training uses NuminaMath as unlabeled prompts. Evaluation covers:

- GSM8K
- MATH-500
- MMLU-Pro
- AIME 2024 / AIME 2025
- LiveCodeBench v6
- CRUXEval-O

Code tasks are used for OOD generalization; no code dataset is used in training.

## 5. Evaluation metrics and main results
The main metric is greedy pass@1 / accuracy. The paper reports that Multi-Reward RLIF generally beats single-reward RLIF baselines on math and code benchmarks (e.g., Intuitor / self-certainty-only and cluster-only). Compared to supervised GRPO with ground-truth rewards, it gets close on some benchmarks but still trails.

Important diagnostic results:

- self-certainty-only suffers token-mode collapse;
- cluster-only suffers answer-cluster collapse;
- multi-reward delays but cannot fully eliminate collapse;
- Multi-Reward + KL-Cov is more stable under long training horizons.

## 6. Position in the UPT Survey
Recommended as a strict UPT addition for the 2026 freshness batch, placed near Family II's TTRL / EMPO / CoVo / RoiRL. It also supports the discussion on open problems of single-internal-reward proxy over-optimization and entropy collapse.

Recommended short label for the main table: `Multi-Reward RLIF`.
Recommended family: `Sample-Relation Supervision`.
Recommended timing: `Offline Corpus UPT` or training-time offline RLIF; if the main table keeps only the coarse regime, write `training-time`.
