# SUDER: Self-Improving Unified Large Multimodal Models with Dual Self-Rewards

**Paper:** SUDER: Self-Improving Unified Large Multimodal Models for Understanding and Generation with Dual Self-Rewards
**Authors:** Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan
**arXiv:** 2506.07963
**Venue:** arXiv 2025
**Code:** not found in extracted text

| Method | Carrier | Regime | Level |
|---|---|---|---|
| SUDER | unified LMM + dual likelihood self-reward + SimPO/GRPO | training-time self-supervised multimodal post-training | reverse-task likelihood between understanding and generation |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or current `0524_new_collection` |
| Screening source | Chrome MCP arXiv search/page text used only for candidate discovery |
| PDF | `0524_new_collection/pdfs/2506.07963.pdf` |
| Extracted text | `0524_new_collection/texts/2506.07963.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit checked on 2026-05-24 against dual self-reward equations, reverse likelihood computation, SimPO/GRPO optimization, training data, external-supervision claims, and CLIP comparison |
| BibTeX | `0524_new_collection/bib/2506.07963.bib` |
| Suggested taxonomy | strict UPT candidate; Family I / IV bridge for dual likelihood self-reward in unified multimodal models |

## 1. UPT 归属理由

SUDER 是一个比较干净的 strict UPT candidate。它面向 unified LMMs，同时优化 visual understanding 与 visual generation。核心信号不是 human preference、external reward model 或 ground-truth answer，而是模型自身 understanding/generation 两个方向的 **dual likelihood**：生成多个候选输出后，把输入输出反转，用同一模型计算原始输入在候选输出条件下的 likelihood，作为 self-reward。

按当前 taxonomy，它适合放入 **Family I / IV bridge**：

- Family I：reward 是模型自身条件 likelihood / prediction statistic；
- Family IV：understanding 与 generation 互为内部 evaluator，为对方提供 self-reward；
- multimodal scope：直接补强 UMM / MLLM generation-understanding alignment 方向。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 使用 online SimPO 和 GRPO 对 Janus-Pro-1B/7B 做 post-training |
| B2 internal signal | Pass | reward 来自同一 unified LMM 的 reverse-task likelihood |
| B3 no external supervision | Pass with data-source caveat | 使用 T2I-CompBench prompts、JourneyDB / COCO images，但不使用配对 image-text annotation 作为训练监督 |
| B4 internal judge/scorer | Pass | 没有 external reward model；CLIP 只作为对照实验，不是 SUDER 训练信号 |

## 3. 方法介绍

SUDER 利用 understanding 和 generation 的双向关系：

1. 对 visual understanding，模型给定 image 采样多个 textual descriptions；
2. 将每个 description 反过来作为条件，计算生成原始 image tokens 的 likelihood；
3. likelihood 越高，说明该 description 越能解释图像；
4. 对 text-to-image generation，模型给定 text prompt 采样多个 images；
5. 将每个 image 反过来作为条件，计算生成原始 text prompt 的 likelihood；
6. likelihood 越高，说明生成图像越符合文本语义。

这些 reverse likelihoods 构成 Dual Self-Reward (DSR)。SUDER 可用 SimPO 选取高低 reward 样本做 preference optimization，也可用 GRPO 做 group-relative policy optimization。论文还比较了 CLIP reward，结论是 DSR 比 CLIP 更能同时提升 generation 和 understanding。

## 4. 数据集

训练数据包括：

- text-to-image generation：T2I-CompBench training set，约 5,600 text prompts；
- visual understanding：从 JourneyDB 与 COCO118K / COCO Caption 派生数据中随机采样约 2,800 images。

论文明确说明 image 和 text training data 是 non-parallel，没有使用 annotated image-text pairs 来训练模型。实验主干是 Janus-Pro-1B 和 Janus-Pro-7B。

## 5. 评估指标与主要结果

生成评估使用 T2I-CompBench 和 GenEval；理解评估使用 HallusionBench、LLaVABench、POPE、MMBench、SEEDBench-IMG。SUDER 在 Janus-Pro-7B 上明显提升 T2I-CompBench，论文报告平均提升约 11.68%，GenEval 提升约 5%。理解任务上，LLaVABench 也有明显提升。

Ablation 显示，单独优化 generation 或 understanding 都不如统一优化稳定；SimPO 略优于 GRPO，但两者都可基于 DSR 工作。与 CLIP reward 的比较说明，外部 CLIP reward 可能改善 generation，却不能同样改善 understanding；DSR 更符合 unified LMM 的双向自我提升目标。

## 6. UPT Survey 定位

推荐短标签：`SUDER`。

可作为 strict UPT multimodal row，建议写入 Family I / IV bridge：

SUDER computes self-rewards by reversing understanding/generation input-output pairs and scoring the original input under the same unified LMM, enabling self-supervised SimPO/GRPO post-training without external reward models or paired annotations.

它与 `GvU`、`SRUM` 类方法有相邻关系，但比依赖外部 grounding / segmentation tools 的方法更适合 strict UPT，因为主要 reward 完全来自模型自身 reverse likelihood。
