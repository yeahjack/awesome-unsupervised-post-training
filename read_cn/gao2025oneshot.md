# One-shot Entropy Minimization

> **加入 Survey 时间：** 2026-03-11

> **Method:** One-shot EM | **Title:** One-shot Entropy Minimization
> **Carrier:** Direct Opt. | **Regime:** test-time | **Level:** Token
> **arXiv:** 2505.20282 | **Authors:** Zitian Gao, Lynx Chen, Haoming Luo, Joey Zhou, Bryan Dai (Ubiquant)

| When to Adapt | Few-Sample Target Adaptation before held-out inference |
|---|---|
| 触发单位 Trigger Unit | one unlabeled prompt or tiny prompt pool |
| 参数/状态持久性 Persistence | full parameter update persists across subsequent evaluation until manual reset |
| 与推理关系 Inference Coupling | adapt first on the selected prompt(s), then evaluate on downstream benchmarks |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Evaluation Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Few-Sample Target Adaptation |
| 可见数据范围 Visibility Scope | Few-sample target subset |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Few-Sample Target Adaptation`；`Visibility Scope=Few-sample target subset`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Evaluation Boundary`。

- **更新何时触发：** 更新由极小的 prompt pool 触发，核心设定是先对一条或极少数无标签 prompt 做短程微适配。
- **服务当前样本还是后续样本：** 这些更新随后服务更广泛的 benchmark 评估，而不是只服务那条 prompt 本身。
- **参数/状态是否累积：** 参数在短程优化后持续保留到后续评估结束，论文没有采用 per-sample reset。
- **reset 边界：** 因此它更像 micro-dataset adapt-then-evaluate，而不是标准的 streaming TTA。

## 1. UPT 归属理由

本文属于 **Family I — Prediction-Statistic Optimization (entropy minimization)**。

核心理由：
- 训练过程完全**无监督**，不需要任何 ground-truth label、外部 verifier 或 reward model。
- 优化目标是模型自身生成 token 的 **conditional entropy**，属于 intrinsic statistics。
- 仅使用**单条无标注 prompt** 进行极少步（约 10 步）梯度更新，直接最小化生成序列的 token-level entropy。
- 信号完全来源于模型自身的预测分布（intrinsic statistics），不依赖任何外部反馈。

---

## 2. 解决的问题

传统 post-training 方法（如 RL）需要大量高质量标注数据和精心设计的 rule-based reward，准备成本高昂且容易出现 reward hacking。本文探索一个极端问题：**能否仅用一条无标注数据和极少训练步数，在不借助任何外部监督的情况下提升 LLM 的推理能力？**

作者发现 LLM 生成过程本质上是随机的（stochastic），而正确答案通常对应更低的 token entropy。基于此，通过直接最小化 entropy 可以让模型"锁定"高概率的正确推理路径，释放预训练模型的潜在能力。

---

## 3. 方法介绍

### 3.1 Entropy Minimization 算法

给定预训练模型 $p_\theta$ 和输入 prompt $x$，模型自回归生成响应 $y = (y_1, \dots, y_T)$。对每个生成位置 $t$ 计算 conditional entropy：

$$H_t = -\sum_{v \in \mathcal{V}} p_\theta(v \mid y_{<t}, x) \log p_\theta(v \mid y_{<t}, x)$$

仅对 **prompt 之后的生成 token** 计算 entropy（排除 prompt 部分和 PAD token），EM loss 为：

$$\mathcal{L}_{\text{EM}}(x; \theta) = \frac{1}{|\mathcal{I}|} \sum_{t \in \mathcal{I}} H_t$$

该 loss 完全可微，梯度形式类似 entropy-regularized RL 中的 score-function estimator，但无需外部 reward 或 value baseline。

### 3.2 数据选择策略

采用**基于 variance 的数据筛选**：对候选 prompt 池中的每条数据采样 $k$ 次，计算 pass@k 的方差：

$$x^* = \arg\max_{x \in \mathcal{D}} \text{Var}_{\text{pass@k}}(x)$$

高 variance 意味着模型对该 prompt 的行为最不确定（有时对有时错），此类"entropy-sensitive"样本能提供最大的梯度信号。

### 3.3 训练配置

- 仅使用 **1 条无标注数据**（从 NuminaMath 数据集中选取）
- 训练 **10 步**即可收敛（超过 10 步出现 over-confidence 导致性能下降）
- Learning rate: $2 \times 10^{-5}$，temperature: 0.5，batch size: 64
- 基于 Acclerate 框架实现

### 3.4 关键发现

- **Logits shift**：EM 训练使 logits 分布产生**右移**（skewness 增大），将概率集中到高概率 token 上，与 RL 的左移方向相反。这使得 greedy decoding 在 EM 后特别有效。
- **Over-confidence 效应**：EM loss 持续下降但性能在约 10 步后开始下降，说明 EM 本质上是 **distribution shaping tool** 而非传统学习方法。
- **Temperature 趋势反转**：EM 模型在**低推理温度**下表现最好（与 RL 模型偏好高温相反），因为 EM 已将概率集中到正确 token。
- **EM → RL 优于 RL → EM**：EM 先做再接 RL 效果好（EM 增强推理后 RL 进一步优化），反之 RL 后再做 EM 会破坏 RL 学到的分布。

---

## 4. 数据集

### 训练数据
- **NuminaMath**：从中选取 1 条（1-shot）或少量样本（multi-shot），仅使用 prompt 部分，不使用 label。

### 评估 Benchmark
| Benchmark | 类型 |
|---|---|
| MATH500 | 数学推理 |
| Minerva Math | 数学推理 |
| OlympiadBench | 数学竞赛 |
| AMC23 | 数学竞赛 |
| KK | 逻辑推理 |
| MBPP | 代码生成 |

---

## 5. 评估指标与主要结果

### 评估指标
- 各 benchmark 的 accuracy，使用 **avg@8**（8 次采样取平均）以降低随机性。
- 所有实验使用 16 个不同 random seed 重复以确保结论可靠。

### 主要结果（Qwen2.5-Math-7B 基座）

| 模型 | MATH500 | Minerva Math | Olympiad Bench | AMC23 | KK | MBPP | Avg. |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B (base) | 53.0 | 11.0 | 17.2 | 44.1 | 1.0 | 48.9 | 29.2 |
| **+ EM 1-shot** | **78.8** | **35.3** | **39.7** | **70.3** | **17.4** | **65.1** | **51.1** |
| 提升 | +25.8 | +24.3 | +22.5 | +26.2 | +16.4 | +16.2 | +21.9 |

- 仅用 1 条数据和 10 步训练，EM 平均提升 **+21.9 分**，在多个 benchmark 上接近甚至超越需要数千条数据和数百步训练的 RL 方法（如 RLVR、OpenReasoner-Zero、SimpleRL-Zoo）。
- 在多种基座模型上均有效：LLaMA-3.1-8B、Qwen2.5-7B、Qwen2.5-7B-Instruct、SimpleRL-Zoo、Qwen2.5-Math-7B。
- 1-shot EM 通常优于 multi-shot EM，因为单样本训练更稳定，prompt/output 长度波动更小，loss 收敛更平滑。
- 但对已经过充分 RL 训练的模型（如 SimpleRL-Zoo），EM 可能导致轻微性能下降（49.3 → 44.5），说明 EM 对已高度优化的分布可能产生负面影响。
