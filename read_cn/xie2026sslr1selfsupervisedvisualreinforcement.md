# SSL-R1: Self-Supervised Visual Reinforcement Post-Training for Multimodal Large Language Models

**Paper:** SSL-R1: Self-Supervised Visual Reinforcement Post-Training for Multimodal Large Language Models  
**Authors:** Jiahao Xie, Alessio Tonioni, Nathalie Rauschmayr, Federico Tombari, Bernt Schiele  
**arXiv:** 2604.20705  
**Venue:** arXiv 2026  
**Project:** https://github.com/Jiahao000/SSL-R1

| Method | Carrier | Regime | Level |
|---|---|---|---|
| SSL-R1 | MLLM + GRPO | self-supervised visual RL post-training | prediction-statistic visual puzzle rewards |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, or current collection |
| PDF | `0524_new_collection/pdfs/2604.20705.pdf` |
| Extracted text | `0524_new_collection/texts/2604.20705.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against self-supervised visual puzzles, deterministic image-derived rewards, GRPO post-training, and no-human/external-model supervision claims |
| BibTeX | `0524_new_collection/bib/2604.20705.bib` |
| Suggested taxonomy | strict UPT candidate; Family I |

## 1. UPT 归属理由

SSL-R1 是 MLLM post-training 中很清晰的 prediction-statistic intrinsic optimization。它把视觉 self-supervised pretext tasks 转换为 RLVR-style puzzles，reward 从输入图像的确定性变换中自动产生，不需要 human annotations、external model supervision 或外部 reward model。

按当前 taxonomy，推荐归入 **Family I: Prediction-Statistic Optimization**。它的监督信号来自输入自身的几何/视觉结构，而不是 answer labels、teacher model 或 verifier。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 使用 GRPO 对 Qwen2.5-VL / MiMo-VL 做 post-training |
| B2 internal signal | Pass | reward 由 rotation、similarity、inpainting、patch ordering、geometric correspondence 等自监督任务自动生成 |
| B3 no external supervision | Pass | 训练使用 COCO raw images；不使用人工标注或外部模型监督 |
| B4 internal judge/scorer | Pass | reward 是规则化的 self-supervised target，不是 external judge |

## 3. 方法介绍

SSL-R1 设计五类 visual puzzles：

- Rotation prediction；
- Visual similarity；
- Region inpainting；
- Patch ordering；
- Geometric correspondence。

对于单值答案任务，reward 是 exact-match binary reward；对于序列答案任务，reward 是部分正确比例；另有格式 reward 要求 `<think>` 和 boxed answer。训练算法是 GRPO，去掉 value model 和 reference model，只依靠这些 intrinsic visual rewards。

## 4. 数据集

训练使用 COCO 的 118K raw images，并从这些图像自动构造 591K self-supervised QA pairs。评估覆盖 13 个 vision-centric multimodal benchmarks，包括 fine-grained perception、spatial understanding、compositional understanding 等。

## 5. 评估指标与主要结果

SSL-R1 在 Qwen2.5-VL-3B/7B 和 MiMo-VL-7B 上均带来提升。Ablation 显示不同 self-supervised tasks 提升不同能力，multi-task post-training 能进一步改善平均表现。论文还与 Visual Jigsaw、Jigsaw-R1 等并行 self-supervised RL post-training 方法比较，声称 SSL-R1 在多任务覆盖和迁移性上更强。

## 6. UPT Survey 定位

推荐作为 multimodal strict UPT candidate，短标签 `SSL-R1`。它是 Family I 的强例子：视觉输入本身提供可验证任务，且直接发生模型参数更新。
