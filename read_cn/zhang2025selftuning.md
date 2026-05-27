# Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching

> **加入 Survey 时间：** 2026-03-11

> **Method:** Self-Tuning | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | document corpus / self-teaching task batch |
| 参数/状态持久性 Persistence | full parameter accumulate across Stage 1–3 |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | note-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新围绕文档语料与 self-teaching 任务分阶段触发，核心是 Stage 1→Stage 2→Stage 3 的离线 schedule。
- **服务当前样本还是后续样本：** 当前阶段的更新主要服务后续阶段与最终部署模型，而不是服务某个测试样本的即时推理。
- **参数/状态是否累积：** 参数在三阶段间持续累积，论文没有 sample-level reset。
- **reset 边界：** 因此它更像 staged document adaptation，而不是一般意义上的 online TTA。

## 1. UPT 归属理由

Self-Tuning 属于 **Family III — Self-Generated Target Bootstrapping (knowledge/instruction self-curation)**。

核心理由：Self-Tuning 的关键组件 Self-Teaching strategy 完全基于模型自身从原始文档中以 self-supervised 方式生成知识密集型任务（memorization、comprehension、self-reflection 三类子任务），无需外部标注、外部 verifier 或人工标签。具体地：

- **Memorization**：直接对原始文档进行 next-token prediction，属于 intrinsic signal。
- **Comprehension**：自动生成 summarization（以文档标题为 ground truth）、gist identification（用 Spacy 提取实体作为答案）、NLI（随机采样句子并替换实体构造正/负例）——所有 target 均从文档内容自动派生，不依赖外部模型或人工。
- **Self-Reflection**：自动构造 Teaching、Flashcards、Fill-in-the-Blank、Multi-choice QA、Sentence Completion 等闭卷生成任务，target 均由文档内容自动生成。

模型自己为自己创建 knowledge/instruction targets，再将其回馈用于 self-teaching，完全符合 Self-Generated Target Bootstrapping 的定义。

---

## 2. 解决的问题

LLM 因一次性预训练的知识截止问题，难以持续获取新知识。现有方法（如 continued pre-training、standard instruction-tuning）虽然能将新文档注入模型参数，但在知识 **提取（extraction）** 阶段表现不佳——模型难以在推理时有效调用已注入的知识。Self-Tuning 旨在解决：

1. **知识吸收不充分**：单纯 continued pre-training 只做表面记忆，缺乏深层理解。
2. **知识提取困难**：注入的知识在 QA 场景下无法被有效检索和利用。
3. **知识遗忘**：学习新知识后，旧知识容易被灾难性遗忘。

---

## 3. 方法介绍

Self-Tuning 受 Feynman Technique 启发，包含 **三个训练阶段** 和一个核心 **Self-Teaching 策略**。

### 三阶段训练流程

**Stage 1：Learn How to Effectively Absorb Knowledge from Raw Documents**
- 使用训练文档 $D_{\text{train}}^{\text{Doc}}$、训练 QA 数据 $D_{\text{train}}^{\text{QA}}$ 以及 Self-Teaching 生成的任务 $D_{\text{train}}^{\text{Self}}$ 联合训练。
- 目标：让模型学会如何从原始文档中高效吸收知识，同时具备 QA 能力。
- 训练目标：$L_\theta^{\text{Stage1}} = L_\theta(D_{\text{train}}^{\text{Doc}}) + L_\theta(D_{\text{train}}^{\text{Self}}) + L_\theta(D_{\text{train}}^{\text{QA}})$

**Stage 2：Learn New Knowledge while Reviewing QA Skills**
- 在未见过的测试文档 $D_{\text{test}}^{\text{Doc}}$ 上继续预训练，同时混合训练 QA 数据以保持问答能力。
- 目标：将 Stage 1 学到的知识吸收策略迁移到新文档上。
- 训练目标：$L_\theta^{\text{Stage2}} = L_\theta(D_{\text{test}}^{\text{Doc}}) + L_\theta(D_{\text{train}}^{\text{QA}})$

**Stage 3：Continually Learn**
- 仅在测试文档 $D_{\text{test}}^{\text{Doc}}$ 上继续训练，确保彻底吸收新知识。
- 训练目标：$L_\theta^{\text{Stage3}} = L_\theta(D_{\text{test}}^{\text{Doc}})$

### Self-Teaching 策略（核心）

从三个维度以 self-supervised 方式为文档生成知识密集型任务：

**Memorization（记忆）**：
- 对原始文档执行 next-token prediction。

**Comprehension（理解）**：
- *Summarization*：prompt 为 `Write a title:`，以文档标题为 target。
- *Gist Identification*：prompt 为 `Highlight the key information within the article:`，以 Spacy 提取的实体为 target。
- *NLI*：随机采样文档句子为正例，替换实体构造负例，模型判断 Yes/No/Impossible。

**Self-Reflection（自省）**：
- *Teaching*：prompt 为 `Tell me about {topic}`，以文档内容为 target（闭卷生成）。
- *Flashcards*：prompt 为 `Generate a concrete description about {topic} based on the following keywords:`，以文档为 target。
- *Fill-in-the-Blank*：随机替换实体为空白，模型填空。
- *Multi-choice QA*：替换实体为 `–`，提供四个选项（含正确实体和三个干扰项）。
- *Sentence Completion*：截断文档句子，模型补全后续短语。

所有任务均不依赖任何外部 mining pattern，可适用于任意原始文本。

---

## 4. 数据集

### 训练/评估数据：Wiki-Newpages-2023-QA

从 Wikipedia NewPages 收集 2023 年 9 月至 10 月新发布的文章（共 4,257 篇），确保与 LLM 预训练数据零重叠。构建三个子数据集：

| 数据集 | 域 | 训练文档数 | 训练 QA 数 | 测试 QA 数 |
|---|---|---|---|---|
| **Wiki-Bio** | 单域（传记） | 1,263 | 6,136 QA + 1,136 Doc | 663 QA + 127 Doc |
| **Wiki-Multi** | 多域（新闻、体育等） | 2,104 | 10,004 QA + 1,823 Doc | 1,502 QA + 281 Doc |
| **Wiki-Film** | 单域（电影），仅测试 | — | — | 955 QA + 169 Doc |

- Wiki-Bio 和 Wiki-Multi 用于 single-domain 和 multi-domain 评估。
- Wiki-Film 仅作为测试集，用于 cross-domain 评估（训练在 Wiki-Bio 上）。
- QA pairs 由 handcrafted prompts 使用 GPT-4 生成，覆盖文档中所有事实信息（open-ended generation + NLI）。

### Knowledge Retention 评估数据
- **Natural Questions (NQ)**：评估旧知识提取能力（EM, F1）。
- **CommonsenseQA (CSQA)**：评估常识推理保持能力（Accuracy）。

---

## 5. 评估指标与主要结果

### 评估指标

| 任务 | 指标 |
|---|---|
| **Memorization** | Perplexity (PPL, ↓) |
| **Extraction**（开放式生成） | Accuracy, Exact Match (EM), F1, Recall, Rouge-L |
| **Reasoning**（NLI） | Accuracy |
| **Knowledge Retention (Extraction)** | EM, F1（在 NQ 上） |
| **Knowledge Retention (Reasoning)** | Accuracy（在 CSQA 上） |

### 主要结果（LLaMA2-7B，5-shot，Wiki-Bio single-domain）

| Method | PPL ↓ | Extraction EM | Extraction F1 | Reasoning Acc. |
|---|---|---|---|---|
| Cont. Pre-training | 7.28 | 3.62 | 15.96 | 53.40 |
| Standard Ins.-tuning | 6.83 | 5.13 | 19.15 | 51.84 |
| PIT | 2.08 | 11.61 | 27.15 | 57.58 |
| **Self-Tuning** | **1.11** | **31.52** | **50.83** | **66.01** |

### 关键发现

1. **知识获取全面领先**：Self-Tuning 将 PPL 降至接近 1，EM 相比 PIT 提升约 20 个百分点，Reasoning 准确率达 66.01%。
2. **跨域泛化能力强**：在 cross-domain（Wiki-Bio → Wiki-Film）设置下，Self-Tuning 同样取得最佳表现（EM 16.44, Reasoning 66.34）。
3. **知识保持优秀**：在 NQ 和 CSQA 上，Self-Tuning 不仅不损害旧知识，反而略有提升（NQ F1: 25.67, CSQA: 66.01），未出现灾难性遗忘。
4. **多模型泛化**：在 Qwen2-7B、Mistral-7B-v0.1 上同样表现最佳，且在 WebNews-2023 语料上也有效。
5. **非过拟合**：训练动态分析表明 Self-Tuning 在 5 个 epoch 内即超越 open-book baseline，25 epoch 达到峰值，长期训练中知识保持仅下降 2-3% EM。
