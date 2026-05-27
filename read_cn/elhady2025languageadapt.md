# LangAdapt CPT — Emergent Abilities of Large Language Models under Continued Pre-Training for Language Adaptation

> **加入 Survey 时间：** 2026-03-11

- **Method：** LangAdapt CPT
- **Carrier：** Direct Opt.
- **Regime：** Training-time
- **Level：** Token
- **作者：** Ahmed Elhady, Eneko Agirre, Mikel Artetxe
- **机构：** HiTZ Center, University of the Basque Country (UPV/EHU); Reka AI
- **发表：** ACL 2025 (Long Papers)

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

本文属于 **Family I — Prediction-Statistic Optimization（predictive likelihood minimization）**。

核心理由：
- 训练信号完全来自 **language modeling objective（LM loss / next-token prediction）**，即 predictive likelihood minimization；不依赖任何外部标注、verifier 或人工 / AI 偏好标签。
- 内部 artifact 是 **predictive likelihood**：模型直接在目标语言 corpus 上最小化 cross-entropy loss，其 perplexity（cross-entropy 的指数）即 intrinsic statistic。
- 整个 continued-pretraining (CPT) 过程仅对权重做标准 LM-loss 更新，即在 intrinsic statistics 上的 direct optimization。
- 提出的两种替代方案（curriculum learning 与 EMA）只调整训练 schedule 与参数平均，不引入任何外部监督信号。

---

## 2. 解决的问题

现有 LLM（如 Llama 2）以英语为中心，在低资源语言上性能显著较弱。CPT 是把 LLM 适应到新语言的主流途径；实际操作中常把英语数据混入目标语言 corpus，但 **英语数据在 CPT 过程中究竟扮演什么角色，尚无系统研究。**

本文回答以下关键问题：
1. **混入英语数据是否必要？** 作者发现"无英语 CPT"变体在目标语言验证 perplexity 上与"有英语 CPT"持平，但在下游任务 accuracy 上显著较差。
2. **差距的根因是什么？** 无英语 CPT 在训练最初几步即发生 catastrophic forgetting，表现为 in-context-learning (ICL) 能力急剧丢失与参数大幅 shift。
3. **能否用其他技术替代英语数据？** 作者探索 curriculum learning 与 EMA（exponential moving average）来缓解参数漂移，从而减少甚至去除对英语数据的依赖。

---

## 3. 方法介绍

### 3.1 基础 CPT 流程

从已有的英语 LLM 出发，以标准 next-token-prediction loss 为目标，在目标语言 corpus 上做全参数 continued pretraining。

两个常见变体：
- **CPT (lang)：** 仅使用目标语言数据。
- **CPT (lang+en)：** 目标语言数据 + 20% 英语数据混合。

### 3.2 Copain Benchmark

作者提出 **Copain (Contextual Pattern Inference)** —— 一个跨语言的 ICL 评测 benchmark，共 7 个任务（每任务 150 示例，总 1050 测试项），用于把 ICL 能力与目标语言知识解耦：
- 在整数列表中识别 min / max / median。
- 识别奇数 / 偶数位置上的数字。
- 识别字母序最前 / 最后的字符。

输入为数字或字符列表，无自然语言指令；模型必须从 few-shot demonstrations 中推断任务规则。

### 3.3 关键发现

1. **Perplexity 并不能反映全貌：** 两个 CPT 变体的验证 perplexity 几乎相同，但下游与 Copain 表现差距巨大。
2. **ICL 的 catastrophic forgetting：** 无英语 CPT 在最初几步 ICL 即崩溃至接近 0（Copain accuracy 从约 44 降至约 0），之后仅部分、缓慢恢复。
3. **参数 shift：** 无英语变体在前 10 步内的 L2 参数距离已超过有英语变体整个训练过程累计的变化；step 100 时 L2 距离是后者的 7 倍，step 1000 时是 15 倍。
4. **关键期：** 分布漂移的损害集中在 CPT 的前 1k 步。

### 3.4 替代方案

#### Curriculum Learning
- 仅在训练的前 10% 步数（前 1k 步）混入英语；之后切换为纯目标语言训练。
- 取得与全程混入英语相当的效果，验证关键期在训练开端。

#### EMA of Model Parameters
- 每 η 步对当前参数与 η 步前参数做加权平均：θ_t = α·θ_{t-η} + (1-α)·θ'_t，α = 0.92。
- 完全不需要英语数据；通过约束参数 shift 缓解 catastrophic forgetting。
- Basque 与 Indonesian 用 η=1；Arabic 用 η=10。
- 在验证 perplexity 与下游 accuracy 上与有英语 CPT 持平，但 Copain 结果有波动且对超参数敏感。

---

## 4. 数据集

### 训练数据
| 语言 | Corpus | Tokens |
|------|------|-----------|
| Basque (eu) | Latxa corpus (Etxaniz et al., 2024) | 4.7B tokens |
| Arabic (ar) | CulturaX (Nguyen et al., 2023), random sample | ~4.5–4.7B tokens |
| Indonesian (id) | CulturaX (Nguyen et al., 2023), random sample | ~4.5–4.7B tokens |
| English (en) | The Pile (Gao et al., 2020), 500k document sample | 占 CPT 总量的 20% |

### 评测 benchmark
| 语言 | Benchmark | 类型 |
|------|-----------|------|
| Basque | EusTrivia, EusProficiency, EusExams, EusReading | 多选（原生 Basque）|
| Basque | MGSM-eu (Baucells et al., 2025) | 生成式数学（从英语翻译）|
| Arabic | ArabicMMLU (Koto et al., 2024) | 多选，5 子任务 |
| Indonesian | IndoMMLU (Koto et al., 2023) | 多选，5 子任务 |
| 所有语言 | Copain (本文提出) | 跨语言 ICL，7 任务 |

---

## 5. 评估指标与主要结果

### 指标
- **Validation Perplexity (PPL)：** 目标语言验证集上的 perplexity。
- **Downstream Accuracy (Dwn)：** 目标语言多选 benchmark 上的平均 accuracy（5-shot；EusReading 1-shot）。
- **Copain Accuracy (Cop)：** 跨语言 ICL benchmark 上的 exact-match accuracy（3-shot）。

### 主要结果（以 Llama 2 7B 为基础模型）

| Configuration | PPL (eu) | Dwn (eu) | Cop (eu) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 23.64 | 27.43 | 44.67 |
| + CPT (eu+en) | **3.35** | **34.14** | 43.43 |
| + CPT (eu) | 3.58 | 28.89 | 20.12 |

| Configuration | PPL (ar) | Dwn (ar) | Cop (ar) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 4.36 | 32.45 | 44.67 |
| + CPT (ar+en) | 2.09 | **34.34** | 32.60 |
| + CPT (ar) | 2.12 | 32.67 | 23.80 |

| Configuration | PPL (id) | Dwn (id) | Cop (id) |
|------|----------|----------|----------|
| Llama 2 (7B) original | 6.27 | 26.65 | 44.67 |
| + CPT (id+en) | 3.25 | **30.79** | 30.79 |
| + CPT (id) | **3.05** | 26.92 | 27.34 |

### 核心结论

1. **英语数据不影响 perplexity 但显著提升下游性能：** 两个 CPT 变体 perplexity 相当，但有英语的版本在所有语言、所有模型上下游 accuracy 都更高，其中 Basque 差距最大（34.14 vs 28.89）。
2. **基础模型越弱，英语数据越关键：** Llama 2 13B 在 Basque 上下游差距达 7 个点（42.52 vs 35.20），而 Llama 3.1 8B 与 Gemma 2 9B 由于多语言能力更强，差距较小。
3. **Curriculum learning 有效：** 仅在前 10% 步混入英语即可达到与全程混入相当的效果（Basque: Dwn 35.12 vs 34.14）。
4. **EMA 可完全替代英语数据：** 无任何英语时，EMA 在 perplexity 上最佳，下游 accuracy 与有英语版本相当（Arabic: Dwn 33.36 vs 34.34），但 Copain 对超参数 η 敏感。
5. **Emergent abilities 突然出现：** 下游 accuracy 改进呈突现式（Basque 有英语版本在 step 2k–4k 之间提升 8 个点），挑战了"perplexity 相近即下游性能相近"的先验假设。
