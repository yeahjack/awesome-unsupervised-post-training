# SLOT: Sample-specific Language Model Optimization at Test-time

> **加入 Survey 时间：** 2026-03-11

**Method:** SLOT | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token
**Paper:** arXiv 2505.12392 (2025)
**Authors:** Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi
**Affiliations:** Westlake University, University of Washington, USTC

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

SLOT 属于 **Family I — Prediction-Statistic Optimization**（基于内禀 LM 统计量的局部状态/适配器塑形）。

核心依据：
- **优化信号完全来自内禀统计量**：SLOT 仅使用输入 prompt 自身的 negative log-likelihood（cross-entropy loss）作为优化目标，不依赖任何外部标签、外部验证器或外部 AI 反馈。
- **更新对象是局部状态而非伪标签**：SLOT 优化的是一个 sample-specific 的参数向量 δ ∈ R^{1×d}，加到最后一层 hidden features 上；每个样本独立优化，推理结束后丢弃，不生成新的训练数据或伪标签。
- **Token 级别的直接优化**：通过梯度下降直接最小化 prompt 上的 token-level NLL，属于 direct optimization 范式。

---

## 2. 解决的问题

大语言模型在通用语料上训练后，面对复杂、非典型的 prompt（如特殊格式指令、复杂推理要求）时，往往无法充分理解和遵循指令，导致生成质量下降。现有的 Test-Time Adaptation (TTA) 方法存在以下问题：

1. **计算开销大**：对大规模 LLM 进行逐样本全参数更新代价过高。
2. **自监督信号设计困难**：在纯语言任务中，很难为每个样本设计有效的自监督目标。
3. **灾难性遗忘和误差累积**：TTA 方法在 LLM 上容易引发这些问题。

SLOT 旨在以极低的计算开销，在 test-time 针对每个具体 prompt 进行轻量级适配，使模型更好地理解和响应该特定指令。

---

## 3. 方法介绍

SLOT 的核心思想：将输入 prompt 本身视为一个"特殊训练样本"，在推理阶段进行少量优化步骤，使模型更好地"学会"该 prompt 的内容。

### 整体流程分两个阶段：

**Prompt Stage（优化阶段）：**
1. 引入一个 sample-specific 参数向量 δ ∈ R^{1×d}，初始化为零向量。
2. 将 δ 广播加到最后一层 hidden features H 上：H' = H + δ。
3. 用修改后的 H' 计算 logits：logits = W_LM · H'。
4. 以 prompt 上的 cross-entropy loss（即 NLL）为目标，通过 AdamW 优化器对 δ 进行 T 步（通常 T=3）梯度下降。
5. 关键效率设计：由于 δ 仅作用于最后一层，可以缓存前面所有层的 hidden features，只需对最后一层线性头做前向和反向传播，开销极小。

**Generation Stage（生成阶段）：**
1. 使用优化后的 δ_opt，在自回归生成每个新 token 时，将 δ_opt 加到当前 token 的最后一层 hidden feature 上。
2. 不再进行额外优化，直接复用 prompt stage 得到的 δ。

### Logit Modulation Vector (LMV) 分析：
优化后的 δ 等效于在 logits 上施加一个固定的加法偏移 W_LM · δ。实验发现，LMV 显著增强了与推理相关的 token（如 "think"、"reasoning"）的 logit，同时抑制了数字 token 和功能词（如 "should"、"will"），以及 end-of-text token，从而鼓励模型进行更深入的推理。

### 超参数设置：
- 优化步数 T = 3
- 学习率 η = 0.01
- AdamW weight decay = 1×10⁻⁸，epsilon = 1×10⁻⁵
- δ 零初始化

---

## 4. 数据集

评估涵盖多种任务类型：

| 数据集 | 类型 | 说明 |
|--------|------|------|
| **AIME24** | 数学竞赛 | American Invitational Mathematics Examination 2024 |
| **Math500** | 数学 | MATH 数据集子集，覆盖 K-12 到竞赛级别 |
| **GPQA Diamond** | 研究生级推理 | Google-Proof Q&A，需深度领域知识和复杂推理（STEM） |
| **GSM8K** | 数学 | 小学数学应用题，需多步推理 |
| **HumanEval** | 代码生成 | 基于 docstring 的 Python 代码生成 |
| **C-Eval** | 综合（中文） | 中文多学科评测（STEM、社会科学、人文、其他、Hard 子集） |

测试模型包括 Qwen 系列（Qwen-7B、Qwen2.5-Math/14B/32B）、Llama 系列（Llama-3.1-8B/70B-Instruct）、DeepSeek-R1-Distill 系列（Qwen-1.5B/7B/14B/32B、Llama-8B/70B）。

---

## 5. 评估指标与主要结果

**评估指标：** Answer accuracy（答案准确率）。

### 主要结果（T=3 步优化）：

**Qwen-7B（通用模型）：**

| Benchmark | Baseline | +SLOT | 提升 |
|-----------|----------|-------|------|
| C-Eval (STEM) | 52.79 | 56.98 | +4.19 |
| C-Eval (Hard) | 36.22 | 44.77 | +8.55 |
| C-Eval (AVG) | 62.64 | 63.69 | +1.05 |
| GSM8K | 51.2 | 54.2 | +3.0 |
| HumanEval | 29.9 | 31.7 | +1.8 |

**Qwen2.5 系列（数学/推理模型，AIME24 / Math500 / GPQA Diamond）：**
- Qwen2.5-Math-1.5B: AIME24 6.67→10.00 (+3.33)
- Qwen2.5-Math-7B: AIME24 13.33→20.00 (+6.67), Math500 57.60→58.80 (+1.20), GPQA 25.76→32.83 (+7.07)
- Qwen2.5-32B: AIME24 13.33→+10.00, GPQA 36.36→42.93 (+6.57)

**DeepSeek-R1-Distill 系列（推理模型）：**
- DeepSeek-R1-Distill-Qwen-7B: AIME24 50.00→56.67 (+6.67), Math500 93.40→93.80 (+0.40)
- DeepSeek-R1-Distill-Qwen-32B: AIME24 70.00→80.00 (+10.00)
- **DeepSeek-R1-Distill-Llama-70B: AIME24 63.33→73.33 (+10.00), GPQA Diamond 65.66→68.69 (+3.03)**，后者在 70B 开源模型中达到 SOTA

**推理时间开销（GSM8K, Qwen2.5-7B, V100）：**
- Baseline: 161.49s → T=3: 167.07s（仅增加约 3.5%）
- T=5: 174.32s（增加约 7.9%）

### Ablation 结果（DeepSeek-R1-Distill-Qwen-1.5B on AIME24）：
- SLOT 对超参数不敏感，大部分配置均优于 baseline（26.67%）
- 最优配置 (T=4, η=0.05) 和 (T=5, η=0.05) 达到 40.00%，相比 baseline 提升 13.33 个百分点
