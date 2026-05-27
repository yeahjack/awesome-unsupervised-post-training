# QueST: Query-Conditioned Test-Time Self-Training

**Paper:** Query-Conditioned Test-Time Self-Training for Large Language Models  
**Authors:** Chaehee Song, Minseok Seo, Yeeun Seong, Doyi Kim, Changick Kim  
**arXiv:** 2605.13369  
**Venue:** arXiv 2026  
**Code:** https://chssong.github.io/Query-Conditioned-TTST/

| Method | Carrier | Regime | Level |
|---|---|---|---|
| QueST | LoRA / SFT | test-time | Semantic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.13369.pdf` |
| Extracted text | `0524_new_collection/texts/2605.13369.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against query-conditioned auxiliary problem generation, LoRA/SFT test-time updates, and no-external-data claims |
| BibTeX | `0524_new_collection/bib/2605.13369.bib` |
| Suggested taxonomy | strict UPT candidate; Family III |

## 1. UPT 归属理由

QueST 是非常清晰的 update-bearing test-time adaptation 方法。给定一个 user query，方法先生成 query-conditioned auxiliary problem-solution pairs，然后在 test time 用 LoRA 做少量 SFT 更新，最后用 adapted model 回答原 query。它不检索 external data，也不使用 ground-truth label 或 external verifier。

主导训练对象是模型从当前 query 派生出的 auxiliary problem-solution pairs，因此建议归入 **Family III: Self-Generated Target Bootstrapping**。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 每个 query 都初始化 LoRA 参数并执行 test-time SFT update |
| B2 internal signal | Pass | supervision 来自 query-conditioned generated problem-solution pairs |
| B3 no external supervision | Pass | 不依赖 external data；benchmark answers 只用于评估 |
| B4 internal judge/scorer | Pass | 不使用 judge / reward model |

## 3. 方法介绍

QueST 的流程：

1. 输入原 query \(q\)；
2. 生成 \(N\) 个与 \(q\) 结构相关的 auxiliary problem-solution pairs \(D(q)\)；
3. 初始化 LoRA parameters；
4. 用 \(D(q)\) 对模型做 \(T\) 步 test-time SFT；
5. 用 adapted model 回答原 query；
6. 不累积到下一 query，默认是 per-query adaptation。

它和 TLM / TTT-NN 的区别是：TLM 只优化 input perplexity，TTT-NN 依赖 external retrieval corpus，而 QueST 的 supervision 直接从当前 query 生成。

## 4. 数据集

评估覆盖七个数学推理 benchmark：

- AMC
- Minerva
- MATH-500
- GSM8K
- OlympiadBench
- AIME 2024
- AIME 2025

另有 GPQA-Diamond 科学推理评估。

## 5. 评估指标与主要结果

主要指标是 accuracy / pass@1，并与 test-time scaling、TENT、TLM 等 baselines 比较。论文报告 QueST 在数学和 GPQA-Diamond 上普遍提升，且相比采样式 test-time scaling 使用更少 test-time tokens。

Ablation 说明 query conditioning、LoRA adaptation 和 self-generated QA 三者共同作用；没有 query conditioning 的 self-QA 效果有限。

## 6. UPT Survey 定位

建议作为 Family III 的新 strict UPT candidate。它可以补强 survey 的 “Test-Time Instance Adaptation / per-query self-generated-target bootstrapping” 位置：相比 `TTT-NN` 的 nearest-neighbor external data 和 `TLM` 的 prediction-statistic loss，QueST 明确构造 self-generated targets。

推荐主表短标签：`QueST`。  
推荐 family：`Self-Generated Target Bootstrapping`。  
推荐 timing：`Test-Time Instance Adaptation`。  
推荐 caveat：默认 LoRA update 在 query 边界 reset，不是 cumulative deployment learning。
