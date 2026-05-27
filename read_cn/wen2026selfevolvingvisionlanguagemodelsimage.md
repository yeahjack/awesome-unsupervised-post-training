# EvoQuality: Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking

**Paper:** Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking  
**Authors:** Wen Wen, Tianwu Zhi, Kanglong Fan, Yang Li, Xinge Peng, Yabin Zhang, Yiting Liao, Junlin Li, Li Zhang  
**arXiv:** 2509.25787  
**Venue:** ICLR 2026  
**Code:** not found in extracted text

| Method | Carrier | Regime | Level |
|---|---|---|---|
| EvoQuality | VLM + pairwise ranking pseudo-labels + GRPO | training-time self-evolution on unlabeled image pairs | majority-voted pairwise preference, fidelity reward |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or current collection as a standalone note |
| PDF | `0524_new_collection/pdfs/2509.25787.pdf` |
| Extracted text | `0524_new_collection/texts/2509.25787.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit checked on 2026-05-24 against the pairwise majority-vote pseudo-ranking loop and the explicit discard of KONIQ ground-truth quality labels |
| BibTeX | `0524_new_collection/bib/2509.25787.bib` |
| Suggested taxonomy | strict UPT candidate; Family II with task-specific IQA caveat |

## 1. UPT 归属理由

EvoQuality 是 VLM 在 Image Quality Assessment (IQA) 任务上的 self-supervised post-training。它不使用 ground-truth quality scores，而是让 VLM 对 image pairs 多次比较，通过 pairwise majority voting 得到 pseudo-ranking labels，再把这些 pseudo-rankings 转化为 fidelity reward，用 GRPO 更新模型。

该方法属于 **Family II: Sample-Relation Supervision**：监督信号不是单个样本的外部标签，而是模型自己对多个 image pairs 的相对判断和多数一致性。它也有 Family I (Prediction-Statistic) 的味道，因为任务输入本身是图像，reward 由同一模型的比较关系派生。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | online stage 用 GRPO fine-tune VLM policy |
| B2 internal signal | Pass | pseudo-ranking labels 来自同一 VLM 对 image pairs 的多次 pairwise voting |
| B3 no external supervision | Pass | 论文明确丢弃 KONIQ quality scores 和 distortion labels |
| B4 internal judge/scorer | Pass | 不使用外部 reward model 或 stronger teacher |

## 3. 方法介绍

EvoQuality 分为 offline 与 online 两个阶段：

- Offline pseudo-ranking：对每个 image pair，VLM 生成多次比较结果，并通过 majority voting 得到哪个图像质量更高的 pseudo preference。
- Online GRPO：将 pseudo preference 放入 fidelity reward，训练模型给出更一致的 quality judgment。
- Iterative evolution：训练后的模型可以继续用于下一轮 pseudo-label generation，从而逐步改进 IQA 能力。

关键设计是用 ranking 而不是 absolute score 作为自监督目标，避免直接回归嘈杂的 pseudo-score。

## 4. 数据集

训练只使用 KONIQ 图像池，并显式丢弃 ground-truth quality scores、distortion type 和 severity labels。评估覆盖多个 IQA benchmarks，用 PLCC / SRCC 等 IQA 常用指标比较 zero-shot VLM、supervised VLM-IQA 和 EvoQuality 变体。

## 5. 评估指标与主要结果

论文报告 EvoQuality 在多种 IQA benchmark 上显著提升 Qwen2.5-VL-7B 的 zero-shot IQA 表现，摘要中给出 PLCC 平均提升 31.8%。其结果在若干 benchmark 上达到或超过 supervised VLM-based IQA models。消融显示 pairwise ranking + majority vote 比直接 pseudo-score 回归更稳。

## 6. UPT Survey 定位

推荐作为 strict UPT 新增候选，但应注明它是 domain-specific perceptual self-evolution。它补充了现有 MLLM/VLM self-evolution 条目中对 IQA / perceptual ranking 场景的覆盖。
