# CSRS: Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling

**Paper:** Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling  
**Authors:** Yunyao Yu, Zhengxian Wu, Zhuohong Chen, Hangrui Xu, Zirui Liao, Xiangwen Deng, Zhifang Liu, Senyuan Shi, Haoqian Wang  
**arXiv:** 2604.03647  
**Venue:** arXiv 2026  
**Code:** https://github.com/yyy195/CSRS

| Method | Carrier | Regime | Level |
|---|---|---|---|
| CSRS | Qwen2.5-VL + GRPO | unsupervised MLLM self-evolution | softened frequency reward over sampled reasoning sets |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, or current collection. |
| PDF | `0524_new_collection/pdfs/2604.03647.pdf` |
| Extracted text | `0524_new_collection/texts/2604.03647.txt` |
| Basis for this note | Complete downloaded PDF + extracted full text; source audit rechecked on 2026-05-24 against CSRS retracing, softened frequency rewards, GRPO self-evolution, MLLM benchmarks, and ground-truth-label caveats. |
| BibTeX | `0524_new_collection/bib/2604.03647.bib` |
| Suggested taxonomy | strict UPT candidate; Family II. |

## 1. UPT Assignment Rationale

CSRS is a stabilization method for unsupervised MLLM self-evolution. It does not use human answers as the reward; instead, it constructs a continuous reward from the same model's multiple rollouts, re-inference sets, and answer frequencies, replacing the 0/1 hard reward of traditional majority voting.

Under the taxonomy, it best fits **Family II: Sample-Relation Supervision**. Its supervision signal is not an external label per sample but the frequency relations and consistency changes between the same model's maternal trajectories and retracing re-inference trajectories.

## 2. Boundary checks

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Uses GRPO to post-train Qwen2.5-VL-7B. |
| B2 internal signal | Pass | Reward comes from frequency / variance over model-generated answer sets. |
| B3 no external supervision | Pass with data-source caveat | Uses Geometry3K / GeoQA / MMR1 prompt-image pairs; the training reward does not depend on gold answers. |
| B4 internal judge/scorer | Pass | No external verifier; aligned with MM-UPT as unsupervised self-evolution. |

## 3. Method

CSRS comprises three components:

- Retracing Re-inference Mechanism: re-infer from anchor points of the initial reasoning trajectory to expand long-tail reasoning paths.
- Softened Frequency Reward: turn the frequency changes of answers in the maternal set and the re-inference set into a continuous reward, avoiding majority voting's premature reinforcement of initial bias.
- Visual Semantic Perturbation: perturb images to force the model to rely on stable mathematical logic rather than superficial visual patterns.

This design turns hard pseudo-labels into a continuous relational reward, so that low-frequency but stable latent-correct paths can also receive training signal.

## 4. Datasets

Training uses Geometry3K, GeoQA, and MMR1; evaluation is performed on MathVision, MathVerse, MathVista, and We-Math. The backbone is Qwen2.5-VL-7B, compared with unsupervised self-evolution baselines such as MM-UPT.

## 5. Evaluation metrics and main results

CSRS achieves stable gains over MM-UPT on the four multimodal mathematical reasoning benchmarks. The paper emphasizes that it slows down the trend of model-collapse caused by the too-rapid rise of the high-confidence sample proportion, and through retracing / softened reward, retains more reasoning diversity.

## 6. Position in the UPT Survey

Recommended as a strict UPT MLLM candidate, with short label `CSRS`. It can complement the failure-mode discussion of MM-UPT / majority-vote methods: rather than proposing a new data source, it improves the granularity and stability of self-consistency rewards.
