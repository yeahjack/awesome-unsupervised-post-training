# TLM: Test-Time Learning for Large Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Test-Time Learning for Large Language Models
**arXiv:** 2505.20633
**Method:** TLM | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token

| When to Adapt | multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation |
|---|---|
| 触发单位 Trigger Unit | offline: all test samples; online: arriving test sample / batch |
| 参数/状态持久性 Persistence | LoRA accumulates within each offline or online run; online setting has no per-sample reset |
| 与推理关系 Inference Coupling | offline: adapt-before-test; online: interleaved infer-and-adapt |
| 输入可见性 Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Multi-protocol: Evaluation Boundary + No Immediate Reset |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation |
| 可见数据范围 Visibility Scope | Multi-protocol: Full target cohort + Streaming prefix only |

---

## When to Adapt 审计

- **辅助 taxonomy：** 这篇论文包含多个 protocol entry：`Timing Regime=Multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation`；`Visibility Scope=Multi-protocol: Full target cohort + Streaming prefix only`。
- **两轴编码：** `Input Visibility=Multi-protocol: Offline + Online`；`Update Persistence=Cumulative`；`Reset Boundary=Multi-protocol: Evaluation Boundary + No Immediate Reset`。

| Protocol Entry | Timing Regime | Visibility Scope | 输入可见性 Input Visibility | Update Persistence | Reset Boundary | 说明 Note |
|---|---|---|---|---|---|---|
| TLM / offline full-test-set setting | Full-Cohort Transductive Adaptation | Full target cohort | Offline | Cumulative | Evaluation Boundary | 先看完整 test set，再统一更新，然后执行整套评估 |
| TLM / online stream setting | Streaming Continual Adaptation | Streaming prefix only | Online | Cumulative | No Immediate Reset | 样本或 batch 顺序到达，更新持续累积到后续 stream items |

- **更新何时触发：** 论文实验明确包含 offline 与 online 两种设置。Offline setting 中所有 test data 一次性处理，先用全部测试样本更新参数，再开始测试；Online setting 中 test data 顺序到达，模型在单个样本或 batch 后更新。
- **服务当前样本还是后续样本：** Offline 版本服务整套 full test set 的后续评估；Online 版本中当前样本 / batch 的更新主要服务后续 stream items。
- **参数/状态是否累积：** 两种设置都通过 LoRA 更新；online 版本不做 per-sample reset，而是在 stream 内持续累积。
- **reset 边界：** Offline 版本的边界是 evaluation boundary；online 版本的边界是 stream / run boundary，而不是 sample boundary。

## 1. UPT 归属理由

TLM 属于 **Family I (Prediction-Statistic Optimization)**，具体为 predictive likelihood minimization 子类。其核心机制是在 test time 对未标注的测试样本持续最小化 input perplexity（即 negative log-likelihood），通过 LoRA 进行真实的参数更新。整个过程不依赖任何外部标注、外部验证器或外部 AI 标签：

- **信号来源：** 完全来自模型自身对输入 token 序列的 intrinsic statistics（input perplexity / NLL）。
- **优化目标：** $\min_{\Theta} S(x)\mathcal{P}(x;\Theta)$，其中 $\mathcal{P}(x;\Theta)$ 是 input perplexity，$S(x)$ 是基于 perplexity 的样本选择权重。
- **参数更新：** 通过 LoRA 对模型参数进行轻量级梯度更新，属于 direct optimization。
- **无外部监督：** 不需要知识库、训练数据、标注数据或外部模型，仅使用测试流中的 unlabeled test data。

---

## 2. 解决的问题

LLM 在部署时面临 **distribution shift** 问题，具体表现为：

1. **Vertical Domain Shift：** 测试数据包含领域专有术语（如医学、法律、金融），模型未在这些领域上充分训练，导致性能下降。
2. **Non-Specific Distributional Shift：** 用户意图变化、语言多样性（方言、俚语等）导致测试分布偏离训练分布。

现有方法的局限性：
- **Fine-tuning** 需要大量标注数据，在动态环境中不实际。
- **RAG** 依赖外部知识库质量，且引入额外检索延迟。
- **Test-Time Adaptation (TTA)** 方法（如 Tent、EATA）使用 entropy minimization，忽略了 LLM 的 autoregressive 特性，在 LLM 上效果不佳。
- **Test-Time Training (TTT)** 假设可以访问训练数据或知识库，实际场景中往往不可用。

---

## 3. 方法介绍

TLM 由三个核心组件构成：

### 3.1 Input Perplexity Minimization

核心观察（**Observation 1**）：LLM 的 input perplexity $\mathcal{P}(x;\Theta)$ 和 output perplexity $\mathcal{P}(y|x;\Theta)$ 趋势一致。因此，通过最小化 input perplexity 可以间接降低 output perplexity，提升生成质量。

理论依据：通过一阶 Taylor 展开分析，当 cross-gradient term $\langle \nabla_x, \nabla_y \rangle \geq 0$（实验中 98.75% 的 batch 满足此条件）时，最小化 input perplexity 可以保证 output log-probability 不下降。

### 3.2 Sample Efficient Learning Strategy

核心观察（**Observation 2**）：高 perplexity 样本对模型更新贡献更大，低 perplexity 样本可能导致 overfitting。

设计了 active sample selection score：

$$S(x) = \lambda \cdot e^{|\log \mathcal{P}(x;\Theta) - \log \mathcal{P}_0|} \cdot \mathbb{I}_{\{\mathcal{P}(x;\Theta) > \mathcal{P}_0\}}(\mathbf{x})$$

其中 $\mathcal{P}_0$ 为预设阈值（实验中取 $e^3$），$\lambda$ 为缩放系数。该策略过滤低 perplexity 样本，对高 perplexity 样本赋予更高权重，无需额外 gradient 计算。

### 3.3 Lightweight Parameter Updates via LoRA

核心观察（**Observation 3**）：LoRA 比 full-parameter updates 更能防止 catastrophic forgetting。

最终优化目标：

$$\min_{\Delta\Theta} S(x)\mathcal{P}(x; \Theta + \Delta\Theta)$$

其中 $\Delta\Theta = \mathcal{B}\mathcal{A}$，$\mathcal{B}$ 初始化为零，$\mathcal{A}$ 为随机高斯初始化，仅更新 $\Delta\Theta$。

### 算法流程

1. 初始化 LoRA 参数 $\Delta\Theta$，添加到预训练 LLM。
2. 对每个 batch：计算预测 $\bar{y}$，计算样本选择分数 $S(x)$，用加权 perplexity loss 更新 LoRA 参数。
3. 输出所有测试样本的答案。

---

## 4. 数据集

论文构建了综合 benchmark **AdaptEval**，包含三类数据集：

### DomainBench（垂直领域知识）
- **Geography** — 地理领域知识问答
- **Agriculture** — 农业领域知识问答
- **Medicine** — 医学领域知识问答
- **Finance** — 金融领域知识问答

### InstructionBench（指令遵循）
- **Alpaca-GPT4** — 通用指令遵循
- **Dolly** — 通用指令遵循
- **InstructionWild** — 通用指令遵循

### ReasoningBench（推理能力）
- **GSM8K** — 数学推理
- **MetaMath** — 数学推理
- **Logiqa** — 逻辑推理

### 测试模型
- Llama3.2-3B-Instruct
- Llama3-8B-Instruct
- Llama2-13B-Chat
- Qwen2.5-7B-Instruct

---

## 5. 评估指标与主要结果

### 评估指标
- **ROUGE-Lsum (R-Lsum)：** 用于 DomainBench 和 InstructionBench。
- **Exact Match (EM)：** 用于 ReasoningBench。

### 主要结果

**DomainBench（Table 2）：** TLM 在所有四个领域数据集上一致超越原始 LLM 和所有 baseline（Tent、EATA、COME），至少提升 20%。例如：
- Llama3.2-3B-Instruct Geography: 0.2395 → **0.2893**（+20.79%）
- Qwen2.5-7B-Instruct Agriculture: 0.1203 → **0.1652**（+37.32%）

**InstructionBench（Table 2）：** TLM 在所有指令遵循数据集上取得最佳表现。例如：
- Llama3.2-3B-Instruct Alpaca-GPT4: 0.3564 → **0.3883**（+13.91%）

**ReasoningBench（Table 3）：** TLM 在所有推理数据集上显著优于原始 LLM 和 baseline。例如：
- Llama3-8B-Instruct GSM8K: 0.7610 → **0.8074**（+6.10%）

### Ablation 结果
- **Input perplexity minimization 有效性：** 仅使用 perplexity minimization（无 SEL）在 Medicine 数据集上相对提升 83.9%。
- **Sample Efficient Learning 有效性：** SEL 带来约 2% 的额外性能提升，同时减少约 5% 的训练数据量。
- **Online 设置：** 每 100 个样本更新一次参数，backward 次数减少 69.7%（5000 → 1514），同时保持性能优势。
- **量化模型（NF4）：** 在 4-bit 量化 Llama3-8B-Instruct 上，TLM 在四个 DomainBench 数据集上至少提升 25%。
