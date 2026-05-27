# AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback

**Paper:** AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback  
**Authors:** Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu  
**arXiv:** 2602.03084  
**Venue:** arXiv 2026  
**Code:** https://github.com/mira-ai-lab/AERO

| Method | Carrier | Regime | Level |
|---|---|---|---|
| AERO | LoRA + KTO | iterative self-evolution | self-generated tasks, answers, and internal criticism |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2602.03084.pdf` |
| Extracted text | `0524_new_collection/texts/2602.03084.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against Generator/Solver/Refiner roles, entropy-based ZPD, ICC, KTO/LoRA updates, and external-verifier caveats |
| BibTeX | `0524_new_collection/bib/2602.03084.bib` |
| Suggested taxonomy | strict UPT candidate; Family IV / III hybrid |

## 1. UPT 归属理由

AERO 是 strict UPT candidate。它试图在没有 external annotated data 和 external verifier 的情况下，让同一个 LLM 在 self-questioning、self-answering、self-criticism 三个角色之间循环，生成任务、回答、反思并构造偏好数据，然后用 KTO 更新模型参数。

按当前 taxonomy，它是 **Family IV: Internal Evaluator Bootstrapping** 与 **Family III: Self-Generated Target Bootstrapping** 的 hybrid。Family III 体现在模型自己生成训练任务与答案；Family IV 体现在 ICC-based logical verification / self-criticism 被用来产生 truth proxy 和 preference signal。若主表只能放一栏，建议主归 Family IV，因为它的关键创新是用 endogenous critic 替代 external verifier。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 外层用 KTO / LoRA 更新模型参数 |
| B2 internal signal | Pass | task、answer、criticism、pseudo-label 都由同一 LLM 的角色化循环生成 |
| B3 no external supervision | Pass with analysis caveat | 训练过程声称不依赖外部数据、human labels 或 verifier；DeepSeek-R1 只用于 ICC 可靠性分析的 proxy reference |
| B4 internal judge/scorer | Pass | verifier 被替换为 ICC 和 self-criticism，不调用外部 judge during training |

## 3. 方法介绍

AERO 包含内外两层循环。

内层循环中，同一个 LLM 通过 prompt 激活三个角色：

- Generator：生成候选 reasoning tasks；
- Solver：对任务采样多个 reasoning paths；
- Refiner / Critic：在 counterfactual setting 下重新检查推理路径。

论文用 normalized Shannon entropy over answer clusters 判断任务位于 Zone of Mastery、Zone of Proximal Development 或 Zone of Chaos。只保留位于 ZPD 的任务，因为这些任务既不是太容易，也不是完全随机。

核心 verification 机制是 Independent Counterfactual Correction (ICC)。对 top answer clusters，Refiner 被要求在“先前答案可能是错的”的反事实假设下独立求解；如果独立 correction paths 收敛，就把结果作为 high-confidence truth proxy，否则丢弃。这个机制用 internal logical consistency 取代外部 verifier。

外层循环把内层产生的数据构造成 generator / solver / refiner 三类 preference datasets，并用 KTO 做 staggered optimization。staggered design 的目的在于避免 generator、solver、critic 同步更新时互相拉坏。

## 4. 数据集

AERO 的训练数据由模型自己生成。论文强调它是 data-free / verifier-free self-evolution，不依赖人工标注训练集。

评估覆盖九个 reasoning benchmarks，横跨 general reasoning、mathematical reasoning 和 physical reasoning。主要报告 Qwen3-4B-Base、Qwen3-8B-Base，也扩展到 Llama-3.2、Qwen2.5-7B-Instruct、Qwen2.5-32B-Instruct 等模型。

需要注意：论文在 ICC pseudo-label reliability 分析中用 DeepSeek-R1 responses 作为 proxy reference labels，只是为了量化 ICC 精度，不是训练信号。

## 5. 评估指标与主要结果

论文报告 AERO 在 Qwen3-4B-Base 上平均提升 4.57%，在 Qwen3-8B-Base 上平均提升 5.10%，并在九个 benchmark 上超过若干 self-evolution baselines。Ablation 显示移除 self-criticism、ZPD 或 ICC 都会降低平均性能，说明“任务难度定位 + 内部逻辑验证 + 分阶段更新”共同支撑效果。

ICC reliability 分析显示，相比 majority voting，ICC pseudo-label precision 更高。这个结果对 survey 有用，因为它说明 internal evaluator 不一定只是直接 self-judge，也可以通过反事实一致性构造更强的 endogenous verification。

## 6. UPT Survey 定位

推荐作为 frontier strict UPT candidate。它比常规 self-training 更接近“内部任务工厂 + 内部 verifier + 偏好优化”的完整 self-evolution loop。

推荐短标签：`AERO`。  
推荐 family：主归 `Internal Evaluator Bootstrapping`，副标 `Self-Generated Target Bootstrapping`。  
主表 caveat：DeepSeek-R1 只出现在 reliability analysis 中；纳入时要说明 training signal 是否完全 endogenous。
