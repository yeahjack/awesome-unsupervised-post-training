# When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning

**Paper:** When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning  
**Authors:** Zhengxian Wu, Kai Shi, Chuanrui Zhang, Zirui Liao, Jun Yang, Ni Yang, Qiuying Peng, Luyuan Zhang, Hangrui Xu, Tianhuang Su, Zhenyu Yang, Haonan Lu, Haoqian Wang  
**arXiv:** 2603.21289  
**Venue:** arXiv 2026  
**Code:** https://github.com/OPPO-Mente-Lab/LLM-Self-Judge

| Method | Carrier | Regime | Level |
|---|---|---|---|
| When Models Judge Themselves | Actor + frozen same-lineage Judge + GRPO | training-time multimodal UPT | self-consistency with internal judge modulation |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2603.21289.pdf` |
| Extracted text | `0524_new_collection/texts/2603.21289.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against Actor/Judge initialization, frozen Judge modulation, self-consistency rewards, GRPO updates, and no-human-answers/external-reward-model claims |
| BibTeX | `0524_new_collection/bib/2603.21289.bib` |
| Suggested taxonomy | strict UPT candidate; Family IV / II hybrid |

## 1. UPT 归属理由

这篇是 strict UPT multimodal UPT candidate。它面向 multimodal reasoning，用同一个 multimodal model 初始化 Actor 和 Judge；Judge 是 Actor policy 的结构相同 frozen copy，不使用 external reward model。训练数据作为 unlabeled multimodal inputs 使用，不用 human-annotated answers。Actor 对同一输入采样多条 reasoning trajectories，先用 self-consistency 得到 distributional prior，再用 frozen Judge 的 bounded score modulation 调整 trajectory weights，最后用 GRPO 更新 Actor。

按当前 taxonomy，它是 **Family IV: Internal Evaluator Bootstrapping** 与 **Family II: Sample-Relation Supervision** 的 hybrid。Family II 来自同输入多轨迹 self-consistency；Family IV 来自 same-lineage frozen Judge 对候选 trajectory 的质量调制。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Actor 使用 GRPO 做 unsupervised post-training |
| B2 internal signal | Pass | reward prior 来自 Actor self-consistency，Judge modulation 来自 same-lineage frozen copy |
| B3 no external supervision | Pass with dataset caveat | 使用 raw images / RL-stage QA inputs as unlabeled training set，不用 human-annotated answers |
| B4 internal judge/scorer | Pass | Judge 由同一 base Actor 初始化并冻结，不是外部 reward model |

## 3. 方法介绍

方法包含 Actor 和 Judge 两个模块。Actor 对同一 multimodal input 采样多条 reasoning trajectories。首先根据答案聚合和 self-consistency 得到初始 reward distribution：更符合 group consensus 的轨迹得到更高 prior。

然后引入 frozen Judge。Judge 与 Actor 结构相同，由当前 Actor policy copy 初始化，并在训练过程中保持 frozen。Judge 对每个 trajectory 给出 score，但这个 score 不直接作为最终 reward，而是经过 bounded calibration function 变成 modulation signal，用于连续地重加权 self-consistency distribution。这样可以避免 Judge raw scores 在不同输入之间不可比，也避免 noisy judge 直接主导训练。

最终，modulated scores 被建模成 group-level distribution，并转成 relative advantages，用 GRPO 更新 Actor。

## 4. 数据集

训练使用多种 multimodal datasets 的 inputs。论文说明部分数据集只使用 raw images，不使用 human-annotated QA pairs 或 metadata；另一些设置中使用 RL-stage QA split 作为 unlabeled training set，即使用问题和图像输入，但不使用答案监督。

评估包括多个 multimodal mathematical reasoning benchmarks，例如 MathVision、MathVerse、DynaMath、WeMath、LogicVista 等，并扩展到非数学、vision-centric benchmarks。

主要模型是 Qwen2.5-VL-7B-Instruct；Actor 和 Judge 都从该模型初始化，Judge frozen。

## 5. 评估指标与主要结果

论文报告该方法在多个 multimodal mathematical reasoning benchmarks 上稳定提升，并优于 majority-vote / self-consistency baselines。Ablation 显示，Judge-only 不如 self-consistency + Judge modulation 的组合；distributional reward modeling 能缓解 mode collapse 和 noisy score 影响。

与 supervised GRPO 相比，该方法在部分 pass@10 设置下接近或超过有标签训练，并且训练时间明显更低。作者还分析了 self-consistency 与 Judge preference 的关系，发现训练后二者 top-1 agreement 增加但不饱和，说明两类内部信号互补。

## 6. UPT Survey 定位

推荐作为 multimodal Family IV 新增候选。它是当前 MLLM/VLM 范围内很直接的 internal evaluator bootstrapping case：judge 不是外部模型，而是 same-lineage frozen copy，并且 judge score 只做 bounded modulation。

推荐短标签：`Self-Judge MLLM` 或 `Models Judge Themselves`。  
推荐 family：`Internal Evaluator Bootstrapping`，副标 `Sample-Relation Supervision`。  
主表 caveat：训练 inputs 来自公开 multimodal QA/images 数据集，但 answers / annotations 不作为训练监督。
