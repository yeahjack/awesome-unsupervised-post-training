# ULDTTA — Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs

> **加入 Survey 时间：** 2026-03-11

**Method:** ULDTTA | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token
**Authors:** Longhuan Xu, Cunjian Chen, Feng Yin
**Preprint:** February 11, 2026 | **arXiv:** 2602.09719

| When to Adapt | Test-Time Instance Adaptation with reset at instance boundary |
|---|---|
| 触发单位 Trigger Unit | arriving sample / prompt |
| 参数/状态持久性 Persistence | sample-local parameter, adapter, or state update; reset after inference |
| 与推理关系 Inference Coupling | adapt on the current sample for the current sample |
| 输入可见性 Input Visibility | Online |
| Update Persistence | Non-Cumulative |
| 重置边界 Reset Boundary | Sample Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Test-Time Instance Adaptation |
| 可见数据范围 Visibility Scope | Current Instance Only |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Test-Time Instance Adaptation`；`Visibility Scope=Current Instance Only`。
- **两轴编码：** `Input Visibility=Online`；`Update Persistence=Non-Cumulative`；`Reset Boundary=Sample Boundary`。

- **更新何时触发：** 更新由单个 arriving sample / prompt 触发，直接围绕当前样本做少量优化。
- **服务当前样本还是后续样本：** 当前样本产生的更新主要就是服务当前样本本身，而不是服务后续样本。
- **参数/状态是否累积：** 参数、adapter 或局部状态在单样本内短暂存在，完成推理后即 reset / 丢弃。
- **reset 边界：** 因此这类方法是最典型的 test-time-instance adapt-and-reset protocol。

## 1. UPT 归属理由

ULDTTA 属于 **Family I — Prediction-Statistic Optimization (local state/adapter shaping)**。

- **无外部监督信号：** 该方法在 test time 仅使用 prompt 本身的 negative log-likelihood（即 next-token prediction loss on prompt tokens）作为优化目标，不依赖 gold answer、外部 verifier、人类标注或外部 AI 标签。
- **Intrinsic statistics 驱动：** 优化信号完全来自模型自身对 prompt 的内部统计量（prompt perplexity / conditional log-probability），属于 intrinsic statistics 这一合法信号类型。
- **Layer-wise adapter shaping：** 通过轻量级 hypernetwork（ScaleNet）根据 prompt 的内部表征（first-layer 和 last-layer hidden state 的 mean pooling）动态预测每层、每步的 LoRA learning-rate multiplier，对 transformer 各层的 Q/V projection adapter 进行细粒度 shaping。这是一种基于模型内部状态的 local adapter 调节机制。
- **Sample-specific, adapt-and-reset protocol：** 每个 prompt 独立进行少量梯度步的 LoRA 适配后即重置，完全不需要外部数据或标签。

---

## 2. 解决的问题

大语言模型在部署时面临 **分布偏移（distribution shift）** 和 **实例级目标失配（instance-level objective mismatch）** 两大问题：

1. **部署分布偏移：** 实际使用中 prompt 的风格、长度、领域术语等与预训练/微调阶段差异显著，导致模型表现下降。
2. **全局参数对单样本次优：** 标准训练产生的全局最优参数是对所有样本的平均折中，对于特定 prompt 未必最优，存在 per-instance specialization 的空间。
3. **朴素 TTA 不稳定：** 已有的 unsupervised sample-specific TTA（固定学习率、对 prompt 做少量梯度步）极度脆弱——学习率过小则更新无效，过大则参数灾难性偏移（destructive drift），且不同层的梯度幅度和更新灵敏度差异巨大，单一全局学习率无法兼顾。

ULDTTA 旨在通过 **学习化的 layer-wise、step-wise 动态学习率控制**，使 unsupervised TTA 在少量梯度步内既稳定又有效。

---

## 3. 方法介绍

### 3.1 整体框架

ULDTTA 采用 **adapt-and-reset** 协议：对每个 test prompt x，初始化一组新的 LoRA 参数（B=0, A 随机），在 prompt 的 negative log-likelihood 上运行 K 步梯度更新（K_max=5），然后用适配后的模型生成 answer y，最后丢弃 LoRA 状态。

核心创新在于用一个轻量级 **hypernetwork ScaleNet** 替代固定学习率，为每一层（分别对 Q 和 V projection）、每一步预测一个 **non-negative learning-rate multiplier**，对 base learning rate 进行乘性缩放。

### 3.2 ScaleNet 结构

- **输入：** 从 prompt 的 forward pass 中提取固定长度特征 h(x)，为 first-layer 和 last-layer hidden state 序列的 mean pooling 拼接，维度 2d。同时输入当前 TTA step k 和总步数 K 的 embedding。
- **架构：** 两层 MLP（hidden size 128），输出经过 non-negative activation（a<=0 时用 exp(a)，a>0 时用 1+a+0.5a^2）并可选 safety clamp。
- **输出：** 每层 Q/V 各一个 scalar multiplier s_l^(k)，共 2L 个 per-step scaler。

### 3.3 逐层动态更新规则

对第 l 层的 LoRA 参数 phi_l，第 k 步更新为：

> phi_l^(k+1) = phi_l^(k) - eta * s_l^(k) * grad_{phi_l}(-log P(x; Phi^(k)))

其中 eta 为 base learning rate（10^-2），s_l^(k) 为 ScaleNet 输出的 multiplier。

### 3.4 训练方法

- ScaleNet 通过 **unrolling** K 步 TTA 过程来训练：在训练集上对每个 (x,y) 对，先执行 K 步 unsupervised TTA（仅用 x 的 loss），然后计算适配后模型在 gold answer y 上的 loss 作为 supervision loss 来更新 ScaleNet 参数 psi。
- 采用 **first-order approximation**：忽略 TTA 梯度对 psi 的二阶依赖（即将 prompt gradient 视为关于 psi 的常数），避免计算昂贵的 Hessian，同时保持训练有效性。
- 训练时随机采样 K in {0,1,...,K_max} 以支持多种 TTA schedule。
- 训练使用 AdamW，学习率 10^-4，约 30k 样本。

---

## 4. 数据集

### 训练数据
- 每个 dataset-model pair 使用约 **30k 训练样本**（mostly without repeats）

### 评估数据集（每个约 300 test samples）

| 数据集 | 类型 | 说明 |
|---|---|---|
| **XSum** | Summarization | BBC 新闻文章的单句摘要 |
| **SQuAD** | Reading Comprehension | 基于 Wikipedia 段落的阅读理解 |
| **NQ-Open** | Open-domain QA | 真实用户查询的短答案预测 |
| **AdaptEval** | Comprehensive TTA Benchmark | 包含 DomainBench（Geography, Agriculture, Medicine, Finance）、InstructionBench（Alpaca-GPT4, Dolly, InstructionWild）、ReasoningBench（GSM8K, MetaMath, LogiQA） |

### 评估模型
- **Llama 系列：** Llama-3.2-3B, Llama-3.2-3B-Instruct, Llama-3.3-70B-Instruct
- **Qwen 系列：** Qwen3-4B, Qwen3-4B-Instruct, Qwen3-32B

---

## 5. 评估指标与主要结果

### 评估指标
- **NLL (Negative Log-Likelihood)**：每个 answer token 的平均 NLL，越低越好。为与 TTA 目标直接对齐的核心指标。
- **ROUGE-Lsum**：生成文本与参考答案的词汇重叠度，越高越好。衡量实际生成质量。

### 主要结果

**NLL 结果（中等规模模型，4 个数据集）：**
- 朴素 fixed-rate TTA baseline 在初始步有微弱改善，但随后 NLL 急剧上升，出现破坏性漂移。
- ULDTTA（layer-wise ScaleNet）在所有 dataset-model pair 上 **持续降低 NLL**，且随 TTA 步数增加保持稳定。
- Layer-wise 控制优于 step-wise only ablation，表明不同 transformer 层确实需要不同的更新幅度。

**大模型 AdaptEval NLL（Table 1）：**
- Llama-3.3-70B-Instruct：5 步 layer-wise TTA 达 NLL 1.7048，显著优于 No TTA（2.2114）和 fixed baseline（11.4970）。
- Qwen3-32B：5 步 layer-wise TTA 达 NLL 1.8889，优于 No TTA（2.1805）和 fixed baseline（2.0925）。
- 大模型上的增益甚至大于中等规模模型，说明方法具有良好的 **scaling 特性**。

**ROUGE-Lsum 结果（Table 2）：**
- 在 XSum 和 SQuAD 上改善明显且稳定（如 Qwen4B on XSum: No TTA 0.1700 -> 5-step layer-wise 0.2247）。
- 在 NQ-Open 和 AdaptEval 上改善较小且不太稳定，因为这些任务更开放，需更深层推理，prompt-only adaptation 难以充分优化。

**ScaleNet 可视化分析：**
- 学习到的 learning-rate multiplier 在层间和 Q/V 间呈现丰富结构，相邻层或同层 Q vs. V 可差数个量级。
- 更新幅度在第一步最大，后续步骤快速衰减，表明大部分收益来自初始更新。
- 不同 schedule 长度下同一步的 scale 保持一致，具有良好的 schedule consistency。
