# V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation

**Paper:** V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation
**Authors:** Han Wang, Yi Yang, Jingyuan Hu, Minfeng Zhu, Wei Chen
**arXiv:** 2601.10094
**Venue:** arXiv 2026
**Code:** https://github.com/SatonoDia/V-Zero

| Method | Carrier | Regime | Level |
|---|---|---|---|
| V-Zero | Questioner/Solver VLM co-evolution + GRPO/RLVR | training-time self-improvement from unlabeled images | self-generated questions and majority-vote pseudo-labels |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or current `0524_new_collection` |
| Screening source | Chrome MCP arXiv search/page text used only for candidate discovery |
| PDF | `0524_new_collection/pdfs/2601.10094.pdf` |
| Extracted text | `0524_new_collection/texts/2601.10094.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit checked on 2026-05-24 against Questioner/Solver loop, dual-track reasoning reward, majority pseudo-labeling, raw-image-only training claim, GRPO updates, baselines, and appendix data details |
| BibTeX | `0524_new_collection/bib/2601.10094.bib` |
| Suggested taxonomy | strict UPT candidate; Family III / II hybrid for unlabeled-image VLM co-evolution |

## 1. UPT 归属理由

V-Zero 是很符合当前 survey 口径的 strict UPT MLLM / VLM candidate。它用 raw unlabeled images 进行 post-training，并把同一个基础 VLM 拆成 Questioner 和 Solver 两个角色。Questioner 根据图像生成有挑战的 multiple-choice questions；Solver 采样多条 reasoning answers，通过 majority voting 得到 pseudo-label，再用这些 pseudo-label 训练自身。

它满足 strict UPT 的关键点：没有 human annotations、没有 external reward model、没有外部 verifier；训练信号来自 Questioner/Solver 交互、majority vote、confidence / difficulty filtering，以及 dual-track reasoning reward。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Questioner 和 Solver 都通过 GRPO / RLVR 迭代训练 |
| B2 internal signal | Pass | Questioner reward 来自 intuitive answer 与 Solver majority reasoning answer 的关系；Solver pseudo-label 来自自身采样 majority vote |
| B3 no external supervision | Pass with image-source caveat | 训练使用 OpenVLThinker image pools，但只使用 raw unlabeled images，不使用其 annotations |
| B4 internal judge/scorer | Pass | 没有外部 judge；Qwen3-VL-32B 只用于 question-difficulty analysis，不是训练 reward |

## 3. 方法介绍

V-Zero 的闭环分为 Questioner training 与 Solver training：

1. Questioner 读取 raw image，生成 visual description、multiple-choice question 和 fast intuitive answer；
2. Frozen Solver 对该 question 采样多个 CoT reasoning responses；
3. 多个 responses 的 majority vote 形成 reasoning-track pseudo-label，vote proportion 形成 confidence；
4. Questioner 用 dual-track reasoning reward 更新：直觉答案和推理答案分歧且推理多数高置信时，说明问题有价值；
5. 优化后的 Questioner 为 raw images 生成训练 questions；
6. Solver 对这些 questions 采样答案，用 majority-vote pseudo-label 和 difficulty-guided filtering 形成训练集；
7. Solver 用 binary accuracy reward 对 pseudo-label 做 RLVR / GRPO 更新。

这个 loop 与当前 taxonomy 中的 Family III/II 很吻合：Family III 体现在 self-generated questions / synthetic training tasks；Family II 体现在 majority pseudo-label 和 group-level agreement signal。

## 4. 数据集

训练数据来自 OpenVLThinker-GRPO-medium 和 OpenVLThinker-GRPO-hard 的 image pools，约 9K images，覆盖 geometry、tables、spatial reasoning 等。论文强调初筛后只使用 raw, unlabeled image data，附录写到最终使用约 4,000 images。

模型是 Qwen2.5-VL-3B-Instruct 和 Qwen2.5-VL-7B-Instruct。训练基于 verl / vLLM；Questioner group size 为 4，Solver group size 为 5，majority vote sample size 为 10。

## 5. 评估指标与主要结果

评估使用 VLMEvalKit，覆盖 general vision-centric tasks 和 visual mathematical reasoning，包括 MMMU、MMStar、MathVista、LogicVista 等。论文比较 base model、human-annotated supervised GRPO、OpenVLThinker-7B 等 baseline。

主要结果是：V-Zero 在 Qwen2.5-VL-7B-Instruct 上不使用任何人工标注也能稳定提升。摘要报告 visual mathematical reasoning +1.7、general vision-centric +2.6。正文还显示 V-Zero 可超过同 image dataset 上的 supervised GRPO baseline，说明动态自生成问题与 pseudo-label loop 比静态标注训练集更能产生适配当前模型的学习信号。

## 6. UPT Survey 定位

推荐短标签：`V-Zero`。

建议归为 strict UPT **Family III / II hybrid**：

V-Zero trains a Questioner/Solver VLM loop on unlabeled images, where questions are self-generated and Solver pseudo-labels are produced by majority voting over its own sampled reasoning answers.

它可以和 `VisPlay`、`RISE`、`EvoLMM` 放在一起，但 V-Zero 的 "zero annotation" 口径更直接，适合作为 MLLM strict UPT 扩充项。
