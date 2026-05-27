# Large Language Models Can Self-Improve (LMSI)

> **加入 Survey 时间：** 2026-03-11

> Huang et al., EMNLP 2023

| 属性 | 值 |
|---|---|
| Method | Self-Improve |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |

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

**Family III — Self-Generated Target Bootstrapping (rationale/critique/latent-thought self-training)**

LMSI 完全不依赖外部 ground truth 标签。其核心流程为：利用 LLM 自身的 Chain-of-Thought (CoT) 推理能力，通过多路径采样 (multiple path decoding) 生成多条推理链，再以 self-consistency (majority voting) 筛选出高置信度的答案和对应推理路径，将这些 model-generated 的推理-答案对作为新的学习目标 (synthetic targets) 对模型进行 fine-tune。整个过程中没有使用任何外部标注、外部验证器或人类反馈，信号完全来自模型自身生成的内容 (model-generated content) 和内在统计量 (self-consistency 的投票置信度)。

---

## 2. 解决的问题

- LLM 的 fine-tuning 通常依赖大量高质量标注数据，在低资源场景或缺乏标注的领域中受到极大限制。
- 现有方法 (如 FLAN, InstructGPT, Minerva) 均需要大规模 supervised 数据集。
- 本文探索：能否仅用无标签的问题集 (question-only dataset)，让 LLM 通过自身推理能力进行 self-training，从而在不依赖 ground truth 的情况下提升推理性能？

---

## 3. 方法介绍

LMSI 包含以下关键步骤：

### 3.1 生成与筛选多条推理路径
- 给定一个 question-only 训练集 $D^{\text{train}} = \{x_i\}$，使用 few-shot CoT prompting 对每个问题采样 $m=32$ 条推理路径 (sampling temperature $T=0.7$)。
- 对每个问题通过 majority voting (self-consistency) 选出最一致的答案 $\tilde{y}_i$，保留所有得出该答案的推理路径作为 self-training 数据。
- 高置信度的答案更可能正确；低置信度的错误答案因支持路径少，对训练噪声影响有限。

### 3.2 Mixed-format 训练数据增强
为防止模型过拟合到单一 prompting 格式，每条推理路径被增强为 4 种格式：
1. **Format 1**: CoT prompting examples + Question → CoT reasoning path
2. **Format 2**: Standard prompting examples + Question → Direct answer
3. **Format 3**: Question + "Let's think step by step" → CoT reasoning (zero-shot)
4. **Format 4**: Question → Direct answer (zero-shot)

### 3.3 自生成问题与 Prompt (低资源扩展)
- **Question generation**: 在训练问题有限时，随机拼接已有问题作为 prompt，让 LLM 生成新问题，再通过 self-consistency 筛选高置信度问题。
- **Prompt generation**: 在无人类 CoT exemplars 时，使用 "Let's think step by step" (Kojima et al., 2022) 让模型自生成 CoT 示例，作为 few-shot prompting 的 exemplars。

### 3.4 Fine-tuning
- 在 self-generated 推理-答案对上 fine-tune 预训练 LLM。
- 训练 10k steps，learning rate $5\text{e-}5$，batch size 32。
- Fine-tune 后推理时使用 $T=1.2$ (高于训练前的 $T=0.7$)，因为 self-improvement 后模型输出分布的 entropy 降低。

---

## 4. 数据集

| 任务类型 | 数据集 | 说明 |
|---|---|---|
| Arithmetic reasoning | **GSM8K** | 小学数学应用题 |
| Arithmetic reasoning | **DROP** | 阅读理解 (数值推理)，分为 football / non-football 子集 |
| Commonsense reasoning | **OpenBookQA** | 多选题 |
| Commonsense reasoning | **ARC-c** (AI2 Reasoning Challenge) | 多选题 (Challenge subset) |
| NLI | **ANLI-A2, ANLI-A3** | 对抗性自然语言推断 |
| OOD 评估 | **AQUA, SVAMP, StrategyQA, ANLI-A1, RTE, MNLI-M/MM** | 泛化能力测试 |

注意：训练过程仅使用问题 (无 ground truth labels)。

---

## 5. 评估指标与主要结果

**评估指标**: Accuracy (所有任务均报告准确率)，使用三种推理方式评估：Standard Prompting、CoT-Prompting、Self-Consistency。

### In-domain 结果 (Table 3, PaLM 540B)

| 数据集 | w/o LMSI (Self-Consistency) | w/ LMSI (Self-Consistency) | 提升 |
|---|---|---|---|
| GSM8K | 74.4% | **82.1%** | +7.7 |
| DROP | 78.2% | **83.0%** | +4.8 |
| ARC-c | 88.7% | **89.8%** | +1.1 |
| OpenBookQA | 90.0% | **94.4%** | +4.4 |
| ANLI-A2 | 64.5% | **66.5%** | +2.0 |
| ANLI-A3 | 63.4% | **67.9%** | +4.5 |

### Out-of-Domain 泛化 (Table 4, 多任务联合训练后)
- 在 6 个 OOD 任务 (AQUA, SVAMP, StrategyQA, ANLI-A1, RTE, MNLI) 上均取得提升，表明 LMSI 增强了模型的通用推理能力而非仅拟合训练分布。

### 关键消融实验
- **Mixed format 重要性 (Table 5)**: 四种格式联合训练 (73.5% CoT on GSM8K) 优于仅用 CoT 格式 (69.4%) 或仅用 direct answer 格式 (61.6%)。
- **自生成问题 (Table 6)**: 使用自生成问题仍能提升性能 (GSM8K CoT: 66.2%)，但不如使用真实训练集问题 (73.5%)。
- **Zero-shot 自生成 prompt (Fig. 3)**: 在 GSM8K 上达到 74.2% (zero-shot SOTA)，无需人工 CoT exemplars。
- **蒸馏到小模型 (Table 7)**: LMSI 540B 生成的训练数据用于 fine-tune 8B 和 62B 模型，62B 蒸馏后 (57.4%) 超越原始 540B 预训练模型 (56.5%)。
- **采样路径数 (Fig. 5)**: $m=15$ 即可获得较好效果；LMSI 后仅 5 条 self-consistency 路径即超过未 LMSI 时 32 条路径的表现。
