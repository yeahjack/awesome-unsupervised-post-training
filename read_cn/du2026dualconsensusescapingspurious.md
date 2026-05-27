# Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism

**Paper:** Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism  
**Authors:** Kaixuan Du, Meng Cao, Hang Zhang, Yukun Wang, Xiangzhou Huang, Ni Li  
**arXiv:** 2603.16223  
**Venue:** arXiv 2026  
**Code:** not found in extracted text

| Method | Carrier | Regime | Level |
|---|---|---|---|
| DCRL / Dual Consensus | GRPO + temporary unlearning | training-time and test-time label-free RLVR | two-stage internal consensus |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2603.16223.pdf` |
| Extracted text | `0524_new_collection/texts/2603.16223.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against anchor/explorer temporary unlearning, harmonic dual consensus, GRPO update, and no-external-model/supervision claims |
| BibTeX | `0524_new_collection/bib/2603.16223.bib` |
| Suggested taxonomy | strict UPT candidate; Family II |

## 1. UPT 归属理由

Dual Consensus / DCRL 是 strict UPT candidate。它针对 TTRL / self-reward 这类 label-free RLVR 的 spurious majority 问题，提出两阶段 consensus：同一个模型先作为 anchor 产生 dominant responses，再通过 temporary unlearning 变成 explorer，产生更分散的 auxiliary signals；最终 pseudo-label 由 anchor 与 explorer 两组信号的 harmonic mean 选择。整个过程不使用 external models 或 supervision。

按当前 taxonomy，它应归入 **Family II: Sample-Relation Supervision**。核心训练信号来自同一 input 下多 rollouts 的关系结构，但比 simple majority vote 更强，因为它通过 anchor/explorer 双分布减少错误多数答案的支配。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 用 GRPO 更新 policy；还支持 test-time adaptation |
| B2 internal signal | Pass | anchor/explorer 都来自当前模型，explorer 是 temporary unlearning 后的同源模型 |
| B3 no external supervision | Pass with prompt-source caveat | 训练用 DAPO-Math-14K style prompts，但不使用 ground-truth labels；论文强调 no external models |
| B4 internal judge/scorer | Pass | 没有外部 judge；pseudo-label 来自 harmonic election over internal rollouts |

## 3. 方法介绍

DCRL 解决 majority vote 的失败模式：当模型在难题上形成 spurious dominant answer，simple majority 会把错误答案当作 pseudo-label，强化错误模式。

方法包含三步：

1. Anchor rollout：当前 policy 采样 `G` 条 trajectories，得到 anchor majority answer。
2. Unlearn then explore：复制 anchor model，对 anchor trajectories 做一个 temporary unlearning gradient step，得到 explorer model。这个 explorer 会压低 anchor 高置信 token，从而探索不同答案模式。
3. Harmonic election：对 anchor 和 explorer 两组答案分布计算 harmonic mean score，选出同时被 dominant mode 和 exploratory distribution 支持的 pseudo-label。

reward 设计保留了 anchor majority 的部分 reward，避免 early training 中探索信号太嘈杂；adaptive sampling 根据 anchor 与最终 pseudo-label 的一致率调节 anchor/explorer samples 是否参与 policy gradient。

## 4. 数据集

主要训练使用 DAPO-Math-14K 风格的大规模数学 prompts，论文说明使用 Qwen3-32B rephrased dataset from original source code，但训练过程中不使用 ground-truth labels，也不调用 external models。

评估包括八个 benchmarks：六个数学任务，例如 MATH-500、AIME、AMC、Minerva、OlympiadBench，以及两个 multi-task benchmarks，例如 MMLU-Pro、GPQA。论文还测试 test-time adaptation，在 unseen benchmarks 上持续 unsupervised training。

## 5. 评估指标与主要结果

DCRL 在 DAPO-Math-14K 上训练后，超过 label-free baselines，包括 RENT、TTRL / majority vote 等，并接近 supervised GRPO with gold labels。论文还显示 DCRL 的 reward signal 比 majority vote 更稳定，能缓解 spurious majority bias。

Ablation 表明 harmonic election、temporary unlearning/explorer、adaptive sampling 和 conservative reward design 都有贡献。作者也承认如果 anchor 和 explorer 都收敛到系统性错误先验，Dual Consensus 仍可能失效。

## 6. UPT Survey 定位

推荐作为 Family II 新增候选。它和 TTRL / ETTRL / DDRL / Co-rewarding 一脉相承，但创新点更清楚：把单一 rollouts majority 改成 anchor-explorer 双分布 consensus，以避免错误多数。

推荐短标签：`Dual Consensus` 或 `DCRL`。  
推荐 family：`Sample-Relation Supervision`。  
主表 caveat：训练 prompts 来自外部数学数据集，但 reward / pseudo-label 不使用 gold answers。
