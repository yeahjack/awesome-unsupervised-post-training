# G-Zero: Self-Play for Open-Ended Generation from Zero Data

**Paper:** G-Zero: Self-Play for Open-Ended Generation from Zero Data  
**Authors:** Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, Jiaxin Huang  
**arXiv:** 2605.09959  
**Venue:** arXiv 2026  
**Code:** https://github.com/Chengsong-Huang/G-Zero

| Method | Carrier | Regime | Level |
|---|---|---|---|
| G-Zero | GRPO + DPO | training-time self-play | Semantic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.09959.pdf` |
| Extracted text | `0524_new_collection/texts/2605.09959.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against Hint-delta, Proposer GRPO, Generator DPO, open-ended verifier-free setting, and external-judge caveats |
| BibTeX | `0524_new_collection/bib/2605.09959.bib` |
| Suggested taxonomy | strict UPT candidate; Family III |

## 1. UPT 归属理由

G-Zero 明确面向 open-ended / unverifiable domains，目标是不用 external verifier、LLM-as-a-judge 或 ground-truth answers 进行 self-evolution。它的训练循环包含两个模型角色：Proposer 生成 question-hint pairs，Generator 生成 hint-free 与 hint-conditioned responses，并通过 DPO internalize hint-guided improvements。

虽然 Proposer 的 reward 是 Hint-\(\delta\) 这种分布差异信号，但 Generator 真正训练时优化的是由自生成 hint 构造出来的 preference pairs：hint-conditioned response 为 chosen，unassisted response 为 rejected。按 dominant update-object rule，建议归入 **Family III: Self-Generated Target Bootstrapping**，子类可视为 self-generated hints / preference pairs。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Proposer 经 GRPO 更新，Generator 经 DPO 更新 |
| B2 internal signal | Pass | Hint-\(\delta\) 由 Generator 自身 log-probability shift 计算，preference pair 来自 Generator dual responses |
| B3 no external supervision | Pass | 论文强调 no external verifiers or judges |
| B4 internal judge/scorer | Pass with role caveat | 没有 judge；但有 Proposer/Generator 双模型角色，需要确认是否同 lineage / same base setup |

## 3. 方法介绍

G-Zero 的核心是 **Hint-\(\delta\)**：

- Generator 先对 query \(q\) 产生 unassisted response；
- Proposer 产生 hint \(h\)；
- Hint-\(\delta\) 衡量加入 hint 后 Generator 对原始 unassisted response 的 predictive distribution shift；
- Proposer 用这个 intrinsic reward 学会生成能暴露 Generator blind spots 的 query-hint pairs；
- Generator 对同一 query 分别生成 hint-conditioned 与 unassisted response，并用 DPO 学习偏好 hint-conditioned response。

论文还对 DPO 数据做 \(\delta\)-filter，避免 high-\(\delta\) pairs 造成 answer leakage 或过大 KL shift。

## 4. 数据集

论文强调 zero-data / open-ended self-play，但评估覆盖两类：

- open-ended tasks: AlpacaEval、IFEval、chat / instruction following；
- verifiable transfer tasks: AIME 2024/2025、math/code 等。

附录 prompt 显示 Proposer 可生成 analysis、argument、text、dataset description、product 等多种开放任务类型。

## 5. 评估指标与主要结果

论文报告 G-Zero 经过 self-play iterations 后，在 open-ended tasks 和 math transfer 上都有提升，例如 AlpacaEval 与 AIME 2025 的增益。更关键的是它声称 improvement 来自 open-ended non-verifiable tasks 中 internalized logical depth，而不是 domain-specific memorization。

## 6. UPT Survey 定位

建议作为 Family III 的新增候选，和 `R-Zero`、`TTCS`、`DiSCTT`、`LongPO` 等 self-generated target / preference / curriculum 方法相邻。它的贡献是把 zero-data self-play 从 verifiable reasoning 扩展到 open-ended generation，并避免 LLM-as-a-judge。

推荐主表短标签：`G-Zero`。  
推荐 family：`Self-Generated Target Bootstrapping`。  
推荐 timing：`Offline Corpus UPT` / training-time self-play。  
需要进一步核查：Proposer 与 Generator 是否始终属于 same-model-lineage；如果使用 distinct pretrained bases，应在主文档中加 caveat。
