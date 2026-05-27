# KBAlign: Efficient Self Adaptation on Specific Textual Knowledge Bases

> **加入 Survey 时间：** 2026-03-11

> **Method:** KBAlign | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | self-generated data batch / iteration round |
| 参数/状态持久性 Persistence | full parameter accumulate across synthesis / refinement rounds |
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

- **更新何时触发：** 更新在 deployment 前的离线自举循环里触发，通常是“生成数据 / 评分 / 筛选 / 再训练”的 round-based schedule。
- **服务当前样本还是后续样本：** 当前 round 产生的合成样本或伪目标主要服务下一轮训练与最终部署模型，而不是服务某个测试样本的即时推理。
- **参数/状态是否累积：** 参数在多轮自举过程中持续累积，论文通常不会做 sample-level reset。
- **reset 边界：** 因此这类方法的 `When to Adapt` 核心是 offline iterative bootstrapping，而不是 online test-time adaptation。

## 1. UPT 归属理由

**Family III — Self-Generated Target Bootstrapping（knowledge/instruction self-curation）**

KBAlign 完全不依赖外部标注信号（无 human labels、无外部 AI 标注）。其核心流程为：给定一个特定领域的文本知识库 (KB)，模型 **自行围绕 KB 文本生成 instruction-response 形式的训练数据**（self-annotation），随后用这些自生成数据对模型进行 fine-tuning。此外，模型还通过 self-verify tuning 迭代地自我检查并纠正错误，所有监督信号均来自模型自身生成的内容，符合 self-generated-target bootstrapping 的定义。

---

## 2. 解决的问题

在特定领域的小规模文本知识库（如法律、生物医学文档）上，现有 RAG 系统面临以下困难：

- **Vanilla unsupervised training**（直接语言建模）效果有限，无法充分对齐模型与领域知识。
- **Supervised fine-tuning**（如用 GPT-4 标注训练数据）成本高昂，且在保密性和计算资源受限的场景下不可行。
- 模型在 RAG 推理时常因 retriever 返回的上下文不够精确而回答失败。

KBAlign 旨在以 **极低成本、无需外部监督** 的方式，让 LLM 高效适配特定领域的文本知识库，提升 knowledge-based QA 的准确率。

---

## 3. 方法介绍

KBAlign 包含三个核心模块：

### 3.1 Self Annotation（自标注）

模型以 KB 原始文本为来源，按照多粒度策略自动生成 Q&A 训练数据：

- **Short-dependency Annotation**：将 KB 切分为不超过 1,024 词的固定长度 chunk，每个 chunk 作为 golden context $C_g$，模型基于它生成问题 $Q$，再通过 retriever 补充相关上下文 $C_R$，基于 $C_g + C_R$ 标注答案 $A$。适用于需要精确局部事实知识的任务。
- **Long-dependency Annotation**：将 KB 切分为 < 256 词的短段，将同一层级目录或高 embedding 相似度的多个段落拼接为 $C_g$，要求模型生成涉及多段信息的 multi-hop 问题及长篇综合答案。适用于需要全局信息整合的任务。

### 3.2 Iterative Tuning（迭代调优）

- **Initial Tuning**：用自标注的 $\langle Q, A \rangle$ 数据 fine-tune 模型得到 $M_1$；训练时随机拼接 golden context 或 retrieved context 以弥合训练与推理的 gap。
- **Self-Verify Tuning**：将标注数据分为 k 部分，用第 1 部分训练后的模型对第 2 部分进行 RAG 推理得到预测 $P_2$，模型自行比较 $P_2$ 与 ground-truth $A_2$ 并生成纠错理由 $V_2$。下一阶段用 $\langle Q_2, P_2 \rangle \to V_2$ 作为训练数据继续调优。实验中采用 25% verify + 75% QA 的混合比例，迭代 2-3 次。

### 3.3 Targeted Inference（目标推理）

- **Query Expansion (QE)**：模型先生成一个初步预测 $P$（利用已记忆的领域知识），将 $Q+P$ 作为扩展 query 送入 retriever，获取更精准的检索结果。
- **Self Verification**：模型利用在 iterative tuning 中习得的验证能力，对多次采样结果进行置信度判断和自选择。

---

## 4. 数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| **LooGLE** | 长文本 QA | 长文本材料构成 KB，评估模型对特定知识的记忆能力 |
| **ASQA** | 长篇回答 QA | 评估知识回忆与信息组织能力（仅用 test set，不使用 training data） |
| **JEC-QA** | 中文法律选择题 | 评估专业领域学习与指令遵循能力（单选 + 多选） |
| **BioASQ** | 生物医学 QA | 评估生物医学知识检索与推理能力 |
| **WebASQ** | 通用 KBQA benchmark | 用于验证方法的泛化性 |

KB 规模从 0.41M 到 21M tokens 不等。

---

## 5. 评估指标与主要结果

### 评估指标

- **Rule metrics**：F1 score、Match score（长篇回答中关键元素的 recall）；JEC-QA 用选项准确率。
- **Similarity metrics**：BERT score、BLEU、ROUGE、text2vec cosine similarity。
- **Intelligent metrics**：GPT-4o 语义判断评分（LLM score）。

### 主要结果（Table 1）

| 模型 / 方法 | LooGLE F1 | ASQA Match | JEC-QA Total | BioASQ F1 |
|---|---|---|---|---|
| Vanilla RAG (MiniCPM-2B) | 30.92 | 11.91 | 25.69 | 29.27 |
| **KBAlign (MiniCPM-2B)** | **54.09** | **15.68** | **28.91** | **61.38** |
| Δ | +23.17 | +3.77 | +3.22 | +32.11 |
| Vanilla RAG (LLaMA-3.1-8B) | 40.46 | 20.21 | 23.73 | 27.96 |
| **KBAlign (LLaMA-3.1-8B)** | **62.07** | **25.23** | **23.83** | **70.97** |
| Δ | +21.61 | +5.02 | +0.10 | +43.01 |

关键发现：

- KBAlign 在 LooGLE 上带来超过 20% F1 提升，自适配后的 2B 模型甚至超过 GPT-4o 的表现。
- 在 BioASQ 上提升尤为显著（MiniCPM-2B: +32.11 F1, LLaMA-3.1-8B: +43.01 F1）。
- KBAlign 能保留 GPT-4 supervised adaptation 约 **90%** 的性能增益，但完全依赖自标注，无需外部大模型。
- 简单语言建模 (LM) 的效果有限，证明了 self-annotation 构造 instruction 数据的必要性。
- Self-verify iterative tuning 加速收敛，推荐至少 3 次迭代。
- 时间成本远低于直接 LM 训练（整个 KBAlign 流程 < 480 min vs. LM 480 min on A100）。
