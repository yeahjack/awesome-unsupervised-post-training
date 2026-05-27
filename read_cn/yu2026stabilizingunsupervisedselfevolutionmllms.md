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
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, or current collection |
| PDF | `0524_new_collection/pdfs/2604.03647.pdf` |
| Extracted text | `0524_new_collection/texts/2604.03647.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against CSRS retracing, softened frequency rewards, GRPO self-evolution, MLLM benchmarks, and ground-truth-label caveats |
| BibTeX | `0524_new_collection/bib/2604.03647.bib` |
| Suggested taxonomy | strict UPT candidate; Family II |

## 1. UPT 归属理由

CSRS 是一个针对 MLLM unsupervised self-evolution 的稳定化方法。它不使用人工答案作为 reward，而是从同一模型的 multiple rollouts、re-inference sets 和 answer frequency 中构造连续 reward，替代传统 majority voting 的 0/1 hard reward。

按 taxonomy，它最适合放入 **Family II: Sample-Relation Supervision**。其监督信号不是单条样本的外部标签，而是同一模型在 maternal trajectories 与 retracing re-inference trajectories 之间的频率关系和一致性变化。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 使用 GRPO 对 Qwen2.5-VL-7B post-train |
| B2 internal signal | Pass | reward 来自 model-generated answer sets 的 frequency / variance |
| B3 no external supervision | Pass with data-source caveat | 使用 Geometry3K / GeoQA / MMR1 的 prompt-image pairs；训练 reward 不依赖 gold answers |
| B4 internal judge/scorer | Pass | 无外部 verifier；和 MM-UPT 对齐为 unsupervised self-evolution |

## 3. 方法介绍

CSRS 由三个组件组成：

- Retracing Re-inference Mechanism：从初始 reasoning trajectory 的 anchor point 重新推理，扩展 long-tail reasoning paths。
- Softened Frequency Reward：把 answer 在 maternal set 与 re-inference set 中的频率变化转化为连续 reward，避免 majority voting 过早强化初始偏见。
- Visual Semantic Perturbation：对图像加入扰动，迫使模型依赖稳定的数学逻辑，而不是表层视觉 pattern。

这种设计把 hard pseudo-label 变成连续的 relational reward，使低频但稳定的潜在正确路径也能获得训练信号。

## 4. 数据集

训练侧使用 Geometry3K、GeoQA、MMR1，评估在 MathVision、MathVerse、MathVista、We-Math 上进行。Backbone 是 Qwen2.5-VL-7B，并与 MM-UPT 等 unsupervised self-evolution baseline 对比。

## 5. 评估指标与主要结果

CSRS 在四个 multimodal mathematical reasoning benchmarks 上相对 MM-UPT 有稳定提升。论文强调它能降低高置信样本比例过快上升的 model-collapse 趋势，并通过 retracing / softened reward 保留更多 reasoning diversity。

## 6. UPT Survey 定位

推荐作为 strict UPT MLLM candidate，短标签 `CSRS`。它可补充 MM-UPT / majority-vote 类方法的失败模式讨论：不是提出全新 data source，而是改进 self-consistency reward 的粒度与稳定性。
