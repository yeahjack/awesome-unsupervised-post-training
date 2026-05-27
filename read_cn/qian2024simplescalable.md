# Simple and Scalable Strategies to Continually Pre-train Large Language Models

> **加入 Survey 时间：** 2026-03-11

**方法名：** Simple CPT
**Carrier：** Direct Opt. | **Regime：** training-time | **Level：** Token
**发表：** Transactions on Machine Learning Research (06/2024)
**作者：** Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, Irina Rish
**arXiv：** 2403.08763

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | unlabeled corpus / continual training batch |
| 参数/状态持久性 Persistence | full parameter accumulate across corpus stages; no sample-level reset |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新在 deployment 前的 continued pretraining / CPT 阶段按 corpus 与 training batch 持续触发，不由单个测试样本触发。
- **服务当前样本还是后续样本：** 当前 batch 的更新服务后续训练批次与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整个 corpus stream / stage 序列中持续累积；若有阶段切换，也只是训练阶段切换，不是 sample-level reset。
- **reset 边界：** 因此它回答的是 deployment 前的 adaptation schedule，而不是 deployment 期间的 online TTA。

## 1. UPT 归属理由

本文属于 **Family I: Prediction-Statistic Optimization (predictive likelihood minimization)**，理由如下：

- 方法的核心目标始终是 **causal language modeling (CLM) loss**，即在纯文本 corpus 上的 next-token-prediction negative log-likelihood。不引入任何外部标注、外部 verifier、人类偏好或外部 AI 标签。
- 训练信号完全来自数据本身的 **intrinsic statistics** —— token 序列的条件概率分布。这是最原始的 prediction-statistic 形态的 UPT：只是在新数据上继续最小化 LM loss。
- Replay 机制仅 replay 旧 corpus 的原始 tokens，不涉及 model-generated content、不涉及偏好判断，仍然是对原始观测的 predictive likelihood minimization。
- Learning-rate re-warming / re-decaying 纯属优化层面的 schedule，不改变目标的性质。

本文因此是最典型的 Family I 代表：**continual pre-training，在新的纯文本 corpus 上持续最小化 LM loss。**

---

## 2. 解决的问题

大型语言模型 (LLM) 通常在数千亿 token 上从头预训练。当出现新数据时，从头重训计算代价高昂。**Continual pre-training (CPT)** 是更高效的替代，但面临两个核心挑战：

1. **Catastrophic forgetting：** 在新数据上继续训练会降低旧数据上的性能。
2. **Poor adaptation：** 简单地在新数据上继续训练（特别是学习率接近 0 时），模型对新数据适应不足。

这两个问题在大规模（200B+ tokens）与不同强度的分布漂移下（弱：英语→英语；强：英语→德语）尤为突出。本文要回答：**当应用简单可扩展的 continual-learning 技术时，CPT 训练得到的模型与在全部数据上从头训练得到的模型之间，性能差距有多大？**

---

## 3. 方法介绍

本文提出三种简单可扩展策略的组合：

### 3.1 Learning Rate (LR) Re-warming

大多数开源 LLM 用 cosine decay schedule，训练结束时 LR 已很小 (eta_min)。在此低 LR 上继续训练导致适应不足。本文提出 **re-warming**：切换到新数据集时把 LR 重新升回高值 (eta_max)，通常恢复至原 pre-training 的 eta_max（如 3e-4）。

### 3.2 Learning Rate (LR) Re-decaying

Re-warming 之后，再用 cosine 把 LR 从 eta_max 退火回 eta_min = 0.1 * eta_max。Re-warming 与 re-decaying 的组合对最大化对新数据的适应至关重要。实验发现：
- eta_max 越高，遗忘越强，但适应也越强。
- eta_max 越低，遗忘越弱，但适应也越弱。
- 线性 warmup 持续时间（0%–2%）对最终性能影响很小；默认 1%。

### 3.3 Compute-equivalent Replay

在新数据上训练时混入少量旧数据（replay）。为公平比较，本文采用 **compute-equivalent replay**：replay 消耗的 token 预算从新数据 token 预算中扣除，保持总 compute 不变。例如 5% replay 表示每个 batch 含 5% 旧数据集 D_0 的样本与 95% 新数据集 D_1 的样本。

关键发现：
- 即使极少量 replay（1%）也能显著缓解遗忘。
- 弱分布漂移下 5% replay 足够；强漂移下需 25% replay。
- Replay 对新数据适应几乎无影响。

### 3.4 Infinite Learning Rate Schedules

作为补充，本文还提出 **infinite LR schedules** 作为 cosine decay 的替代。这些 schedule 在 pre-training 阶段之后转入恒定的高 LR，避免 re-warming 引发的优化困难，且不绑定固定的 token 预算。在需要多次切换数据集（N > 2）时尤其有用。

---

## 4. 数据集

本文使用三个大规模 pre-training 数据集：

| Dataset | Scale | 用途 |
|--------|------|------|
| **Pile** | ~330B tokens（使用 300B 子集） | D_0: 初始 pre-training 数据（英语）|
| **SlimPajama** | 606B tokens（降采样至 ~299B） | D_1: 弱漂移目标（英语→英语）|
| **German Common Crawl** (Oscar) | ~195B tokens | D_1: 强漂移目标（英语→德语）|

SlimPajama 300B 子集的 domain 组成：

| Domain | Size | Sampling % |
|--------|------|-----------|
| Common Crawl | 155.89B | 52.09 |
| C4 | 79.87B | 26.69 |
| GitHub | 15.63B | 5.22 |
| Book | 12.58B | 4.20 |
| Wikipedia | 11.96B | 4.00 |
| Arxiv | 13.25B | 4.43 |
| Stack Exchange | 10.09B | 3.37 |

实验设置包括：
- **两数据集，弱漂移：** Pile (300B) → SlimPajama (300B)
- **两数据集，强漂移：** Pile (300B) → German (200B)
- **三数据集，无漂移：** SlimPajama 的三个 100B 切分
- **Domain incremental：** SlimPajama 按 domain 逐个训练

---

## 5. 评估指标与主要结果

### 指标

- **Validation loss** 在 D_0 (Pile) 与 D_1 (SlimPajama / German) 上：最终 validation loss（最后 100 iterations 的平均）。
- **English LM 评测 benchmark（zero-/few-shot accuracy）：**
  - Commonsense Reasoning (0-shot)：HellaSwag, Winogrande, PIQA, OpenBookQA, ARC-Easy, ARC-Challenge
  - World Knowledge (5-shot)：NaturalQuestions, TriviaQA
  - Reading Comprehension (0-shot)：BoolQ
  - Math：MathQA
  - Aggregated：MMLU (5-shot)
- **German benchmark：** HellaSwag-DE, ARC-Challenge-DE, TriviaQA-DE, MMLU-DE

### 主要结果

**405M 模型，弱漂移 (Pile→SlimPajama)：**
- CPT + LR re-warming + re-decaying + 5% replay 取得与在 Pile ∪ SP (600B) 上从头训练的 baseline 几乎相同的平均 validation loss（2.37 vs 2.35）与下游评测 accuracy。
- 仅约一半的 compute。

**405M 模型，强漂移 (Pile→German)：**
- CPT + LR re-warming + re-decaying + 25% replay 取得与 Pile ∪ German (500B) baseline 相同的平均 validation loss（1.75 vs 1.75）。
- 英语评测 accuracy：32.48 vs 32.43；German HellaSwag：31.04 vs 30.45。

**10B 模型，弱漂移 (Pile→SlimPajama)：**
- CPT + 5% replay 再次追平从头训练的 baseline，确认方法可扩展到更大模型规模。

**Replay 的影响（405M，Table 2）：**
- 弱漂移：无 replay AVG loss = 2.47，5% replay = 2.37，50% replay = 2.35（≈ union baseline 2.35）。
- 强漂移：无 replay AVG loss = 2.34，25% replay = 1.75，50% replay = 1.73（≈ union baseline 1.75）。
- Replay 显著降低遗忘，对适应几乎无影响。

**核心结论：** LR re-warming、LR re-decaying 与少量 replay 的简单组合，足以让 continual pre-trained 的 LLM 在 validation loss 与下游评测上都追平在全部数据上从头训练的模型，同时节省大量 compute。
