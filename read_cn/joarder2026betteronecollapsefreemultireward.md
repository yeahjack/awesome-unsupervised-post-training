# Multi-Reward RLIF: Two is better than one

**Paper:** Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework  
**Authors:** Shourov Joarder, Diganta Sikdar, Ahsan Habib Akash, Binod Bhattarai, Prashnna Gyawali  
**arXiv:** 2605.22620  
**Venue:** arXiv 2026  
**Code:** announced as forthcoming

| Method | Carrier | Regime | Level |
|---|---|---|---|
| Multi-Reward RLIF | Policy Opt. / GRPO | training-time RLIF | Semantic + token |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.22620.pdf` |
| Extracted text | `0524_new_collection/texts/2605.22620.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against RLIF, cluster-voting, self-certainty, KL-Cov, and no-external-supervision claims |
| BibTeX | `0524_new_collection/bib/2605.22620.bib` |
| Suggested taxonomy | strict UPT candidate; Family II dominant |

## 1. UPT 归属理由

这篇论文是一个很直接的 strict UPT candidate。它从 post-trained LLM 出发，用 unlabeled training prompts 进行 GRPO 更新，不依赖 human labels、gold answers、external verifier 或 external reward model。论文明确把方法定位为 RLIF：reward signal 来自模型自身输出。

核心训练信号有两个：

- answer-level cluster voting reward：对同一 prompt 的多个 rollout 提取答案并按等价答案聚类，用 cluster size 作为 agreement reward；
- completion-level self-certainty reward：用 token-wise predictive distribution 的 self-certainty 作为 confidence reward。

按当前 survey 的 dominant update-object rule，主导对象是多个 rollouts / answers 之间的 agreement relation，因此建议归入 **Family II: Sample-Relation Supervision**。self-certainty 是 Family I 风格的辅助信号，但不是唯一或主导更新对象。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 使用 GRPO full fine-tuning 更新模型参数 |
| B2 internal signal | Pass | cluster voting 与 self-certainty 都来自同一模型 rollout / distribution |
| B3 no external supervision | Pass | 论文反复强调不使用 gold-standard solutions 或 human annotations |
| B4 internal judge/scorer | Pass | 没有外部 judge；scoring 来自 cluster relation 与模型概率 |

## 3. 方法介绍

方法针对单一 internal reward 容易 collapse 的问题。它把 RLIF reward 分成两个互补通道：

1. 从 rollout final answers 中构造 answer clusters，按 cluster size 给 reward；
2. 对每个 completion 计算 self-certainty reward；
3. 用 GDPO-style per-channel normalization 避免一个 reward 的尺度压制另一个；
4. 用 KL-Cov regularization 专门约束导致 entropy collapse 的 high-covariance tokens；
5. 用 GRPO-style objective 进行策略更新。

论文的技术重点不在提出全新 UPT 来源，而在说明同属 internal feedback 的多个 reward channel 如何互补，并用 KL-Cov 防止 entropy collapse。

## 4. 数据集

训练使用 NuminaMath 作为 unlabeled prompts。评估覆盖：

- GSM8K
- MATH-500
- MMLU-Pro
- AIME 2024 / AIME 2025
- LiveCodeBench v6
- CRUXEval-O

代码任务用于 OOD generalization，训练并未使用 code datasets。

## 5. 评估指标与主要结果

主要指标是 greedy pass@1 / accuracy。论文报告 Multi-Reward RLIF 在数学和代码 benchmark 上通常优于 single-reward RLIF baselines，例如 Intuitor/self-certainty-only 和 cluster-only。相对 supervised GRPO with ground-truth reward，它在部分 benchmark 上接近但仍有差距。

重要 diagnostic 结果：

- self-certainty-only 会出现 token-mode collapse；
- cluster-only 会出现 answer-cluster collapse；
- multi-reward 能延迟但不能完全消除 collapse；
- Multi-Reward + KL-Cov 在长训练 horizon 下更稳定。

## 6. UPT Survey 定位

建议作为 2026 freshness batch 的 strict UPT 新增候选，放在 Family II 的 TTRL / EMPO / CoVo / RoiRL 相邻位置。它也能支持 discussion 里关于 single internal reward proxy over-optimization 和 entropy collapse 的 open problem。

推荐主表短标签：`Multi-Reward RLIF`。  
推荐 family：`Sample-Relation Supervision`。  
推荐 timing：`Offline Corpus UPT` 或 training-time offline RLIF；若主表只保留 coarse regime，可写 `training-time`。
