# Latent-GRPO: Silence the Judge via Latent Geometric Clustering

**Paper:** Silence the Judge: Reinforcement Learning with Self-Verifier via Latent Geometric Clustering
**Authors:** Nonghai Zhang, Weitao Ma, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu
**arXiv:** 2601.08427
**Venue:** arXiv 2026
**Code:** will be released soon, according to paper text

| Method | Carrier | Regime | Level |
|---|---|---|---|
| Latent-GRPO | GRPO + Iterative Robust Centroid Estimation | training-time verifier-free reasoning RL | terminal hidden-state geometry |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, or current `0524_new_collection` |
| Screening source | Chrome MCP arXiv page used only for candidate discovery |
| PDF | `0524_new_collection/pdfs/2601.08427.pdf` |
| Extracted text | `0524_new_collection/texts/2601.08427.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit checked on 2026-05-24 against IRCE rewards, terminal hidden states, GPT-4o-only validation labels, training datasets, GRPO updates, and external verifier baselines |
| BibTeX | `0524_new_collection/bib/2601.08427.bib` |
| Suggested taxonomy | strict UPT candidate; Family I representation-statistic intrinsic optimization |

## 1. UPT 归属理由

Latent-GRPO 是一个比较干净的 strict UPT candidate。它的核心是不用 rule-based verifier、ground-truth reward 或 LLM-as-judge，而是从同一 GRPO group 内每条 trajectory 的 terminal-token hidden state 计算 latent geometric reward。训练信号来自模型 latent space 的几何聚类结构，不是外部标签。

按当前 taxonomy，它应放入 **Family I: Prediction-Statistic Optimization**。与 entropy、confidence、stable rank 等 output/representation statistics 类似，Latent-GRPO 的主导 update object 是模型自己的 latent representation geometry。它不是 Family II，因为 reward 并不直接来自 answer majority；也不是 Family III，因为没有构造 synthetic target；也不是 Family IV，因为没有单独 evaluator model。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Latent reward 被接入 GRPO pipeline 训练 Qwen3 系列模型 |
| B2 internal signal | Pass | reward 来自 terminal-token last hidden states 的组内 centroid distance |
| B3 no external supervision | Pass with prompt-source caveat | 训练使用 GSM8K、MATH、Open-Platypus prompts；reward 本身不使用答案标签 |
| B4 internal judge/scorer | Pass | 没有 external verifier；GPT-4o / rule labels 只用于验证 latent score 或 baseline 比较 |

## 3. 方法介绍

论文的观察是：正确 reasoning trajectories 的 terminal hidden states 会在 latent space 中形成 dense cluster，而错误 trajectories 更像 outliers。基于这个现象，作者提出 Iterative Robust Centroid Estimation (IRCE)：

1. 对每个 prompt 采样 GRPO group；
2. 提取每条 response terminal token 的 last hidden state；
3. 做 spherical projection，降低 magnitude fluctuation；
4. 迭代估计 "truth centroid"；
5. 用每条 trajectory 到 centroid 的距离作为 dense intrinsic reward；
6. 组内标准化后接入 GRPO 更新 policy。

论文强调 GPT-4o labels 只在分析章节用于验证 latent reward 与外部评分的一致性，不是训练框架所需。

## 4. 数据集

训练/评估覆盖三类 reasoning datasets：

- GSM8K：grade-school math reasoning；
- MATH：competition-level math；
- Open-Platypus：physics、logic、math 等多样 instruction-following reasoning。

附录还列出 benchmark：MMLU、MATH-500、AIME 2024/2025、AMC23、GPQA-Diamond 等，用于泛化分析。主实验模型是 Qwen3-0.6B、1.7B、4B，并补充 Llama3.2-3B 泛化。

## 5. 评估指标与主要结果

主要指标是 accuracy 和 training time per epoch。论文比较三种 reward paradigm：GPT-4o LLM-as-Judge、rule-based ground-truth verification、Latent-GRPO。结果显示 Latent-GRPO 在 GSM8K、MATH、Open-Platypus 上通常达到或超过 external verifier baseline，并有约 2x training speedup。

在 GSM8K 上，Qwen3-0.6B / 1.7B / 4B 的 Latent-GRPO accuracy 分别约为 61.25%、73.88%、82.34%；在 MATH 和 Open-Platypus 上也有类似优势。Ablation 显示 IRCE 优于 mean pooling、K-Means、eigen centrality 等 latent reward computation。

## 6. UPT Survey 定位

建议作为 Family I 新增候选，短标签 `Latent-GRPO` 或 `Silence the Judge`。它可以和 `SR-GRPO`、`VIGOR`、`PowerFlow` 放在一起，补强 Family I 中 "representation / geometry-based intrinsic reward" 子类。

主表可写为：Latent-GRPO derives dense GRPO rewards from terminal hidden-state clustering, replacing external verifiers with internal latent geometry.
