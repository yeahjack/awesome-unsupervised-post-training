# Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism

**Paper:** Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism  
**Authors:** Kaixuan Du, Meng Cao, Hang Zhang, Yukun Wang, Xiangzhou Huang, Ni Li  
**arXiv:** 2603.16223  
**Venue:** arXiv 2026  
**Code:** not found in the extracted text.

| Method | Carrier | Regime | Level |
|---|---|---|---|
| DCRL / Dual Consensus | GRPO + temporary unlearning | training-time and test-time label-free RLVR | two-stage internal consensus |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files. |
| PDF | `0524_new_collection/pdfs/2603.16223.pdf` |
| Extracted text | `0524_new_collection/texts/2603.16223.txt` |
| Basis for this note | Complete downloaded PDF + extracted full text; source audit rechecked on 2026-05-24 against anchor/explorer temporary unlearning, harmonic dual consensus, GRPO update, and the "no external models / supervision" claims. |
| BibTeX | `0524_new_collection/bib/2603.16223.bib` |
| Suggested taxonomy | strict UPT candidate; Family II. |

## 1. UPT Assignment Rationale

Dual Consensus / DCRL is a strict UPT candidate. To address the spurious-majority problem of label-free RLVR methods like TTRL / self-reward, it proposes a two-stage consensus: the same model first acts as an anchor producing dominant responses, then via temporary unlearning becomes an explorer producing more dispersed auxiliary signals; the final pseudo-label is selected by the harmonic mean of the anchor and explorer signal sets. The entire pipeline uses no external models or supervision.

Under the current taxonomy, it should be placed in **Family II: Sample-Relation Supervision**. The core training signal comes from the relational structure of multiple rollouts under the same input, but it is stronger than simple majority voting because the anchor/explorer dual distributions reduce the dominance of incorrect majority answers.

## 2. Boundary checks

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Updates the policy with GRPO; also supports test-time adaptation. |
| B2 internal signal | Pass | Both anchor and explorer come from the current model; the explorer is the same-lineage model after temporary unlearning. |
| B3 no external supervision | Pass with prompt-source caveat | Training uses DAPO-Math-14K-style prompts but no ground-truth labels; the paper emphasizes no external models. |
| B4 internal judge/scorer | Pass | No external judge; the pseudo-label comes from harmonic election over internal rollouts. |

## 3. Method

DCRL addresses the failure mode of majority voting: when the model forms a spurious dominant answer on a hard problem, simple majority treats that wrong answer as the pseudo-label, reinforcing the wrong pattern.

The method has three steps:

1. Anchor rollout: the current policy samples `G` trajectories and obtains an anchor majority answer.
2. Unlearn then explore: copy the anchor model and take one temporary unlearning gradient step on the anchor trajectories to obtain an explorer model. This explorer depresses anchor-high-confidence tokens, exploring different answer modes.
3. Harmonic election: compute a harmonic mean score over the two answer distributions of the anchor and explorer to pick the pseudo-label supported by both the dominant mode and the exploratory distribution.

The reward design retains part of the anchor majority reward, avoiding too-noisy exploration signals during early training; adaptive sampling decides whether anchor/explorer samples participate in the policy gradient based on the agreement rate between the anchor and the final pseudo-label.

## 4. Datasets

Main training uses large-scale math prompts in the DAPO-Math-14K style; the paper notes the dataset is Qwen3-32B rephrased from the original source code, but training uses no ground-truth labels and calls no external models.

Evaluation covers eight benchmarks: six math tasks such as MATH-500, AIME, AMC, Minerva, and OlympiadBench, plus two multi-task benchmarks such as MMLU-Pro and GPQA. The paper also tests test-time adaptation, continuing unsupervised training on unseen benchmarks.

## 5. Evaluation metrics and main results

After training on DAPO-Math-14K, DCRL surpasses label-free baselines including RENT and TTRL / majority vote, and approaches supervised GRPO with gold labels. The paper also shows DCRL's reward signal is more stable than majority voting and mitigates spurious-majority bias.

Ablations show that the harmonic election, the temporary-unlearning explorer, adaptive sampling, and the conservative reward design all contribute. The authors also acknowledge that if both the anchor and the explorer converge on a systematic erroneous prior, Dual Consensus can still fail.

## 6. Position in the UPT Survey

Recommended as a new Family II candidate. It is in the same lineage as TTRL / ETTRL / DDRL / Co-rewarding, but the innovation is clearer: replacing single-rollout-majority with an anchor–explorer dual-distribution consensus to avoid wrong majorities.

Suggested short label: `Dual Consensus` or `DCRL`.  
Suggested family: `Sample-Relation Supervision`.  
Main-table caveat: training prompts come from an external math dataset, but the reward / pseudo-label does not use gold answers.
