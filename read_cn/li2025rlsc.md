# RLSC: Reinforcement Learning via Self-Confidence

> **加入 Survey 时间：** 2026-03-11

**Paper:** Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models
**arXiv:** 2506.06395
**Method:** RLSC | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Token

| When to Adapt | Few-Sample Target Adaptation before held-out inference |
|---|---|
| 触发单位 Trigger Unit | small unlabeled task dataset / rollout batch |
| 参数/状态持久性 Persistence | full parameter accumulate across short RL runs |
| 与推理关系 Inference Coupling | adapt first on the task cohort, then evaluate on downstream benchmarks |
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

- **更新何时触发：** 更新先在一个很小的无标签任务 cohort 上触发，通常只跑少量 RL / optimization steps。
- **服务当前样本还是后续样本：** 这批更新主要服务随后在更大 benchmark 集上的评估，而不是服务当前单个 prompt 的即时推理。
- **参数/状态是否累积：** 参数在短程训练中持续累积，直到一次实验 run 结束才整体重置。
- **reset 边界：** 因此它处在 offline adaptation 与 test-time micro-adaptation 之间，但主耦合关系仍是“先适配，再整体评估”。

## 1. UPT 归属理由

RLSC 属于 **Family I (Prediction-Statistic Optimization)**，子类为 intrinsic predictive statistics via policy optimization。

核心理由：RLSC 直接利用模型自身的 average log-probability / self-confidence 作为 reward 信号，完全不依赖任何外部标签、人类反馈、外部 reward model 或 verifier。其优化目标为最大化模型对自身采样输出的置信度（即 mode sharpening），数学形式为 $F(p_\theta) = \mathbb{E}_{y \sim p_\theta(\cdot|x)}[p_\theta(y|x)]$，等价于最大化输出分布中两个独立采样一致的概率。这一目标完全由模型内在的预测统计量驱动，通过 policy optimization 实现，属于典型的 intrinsic statistics 信号。

---

## 2. 解决的问题

现有 RL post-training 方法存在以下局限：

- **RLHF** 需要大量人工标注和 preference model，成本高昂
- **RLVR (Reinforcement Learning with Verifiable Rewards)** 仍需要 ground-truth 标签来计算 reward
- **TTRL (Test-Time Reinforcement Learning)** 通过 majority voting 生成 pseudo-label，但需要每个问题采样 64 个回复，计算开销大，且需要从回答中提取答案的额外预处理

RLSC 旨在提供一种**零标签、低样本、低计算成本**的 RL fine-tuning 方法，仅利用模型自身的置信度信号即可提升推理能力。

---

## 3. 方法介绍

### 3.1 从 Majority Voting 到 Mode Sharpening

RLSC 的核心洞察：majority voting 本质上是在选择输出分布的 mode（众数），隐式地将概率质量集中到最可能的回答上。RLSC 将这一过程形式化为一个可微分的自监督优化目标——**mode sharpening**。

定义 self-confidence objective：

$$F(p_\theta) = \mathbb{E}_{y \sim p_\theta(\cdot|x)}[p_\theta(y|x)] = \sum_y p_\theta(y|x)^2$$

当分布退化为 delta function（模型完全确信某一回答）时，该目标达到最大值。

### 3.2 Self-Confidence Loss

通过 log-trick 求梯度，得到训练 loss：

$$\mathcal{L}_1 = -\sum_y p_{\text{old}}(y|x) \cdot \log p_\theta(y|x)$$

其中 $p_{\text{old}}$ 是冻结的旧模型副本，用于采样和加权（梯度不回传）。该 loss 鼓励模型对旧模型赋予高置信度的回复给予更高的 log-probability。

平滑变体加入常数 $\alpha > 0$：

$$\mathcal{L}_2 = -\sum_y (p_{\text{old}}(y|x) + \alpha) \cdot \log p_\theta(y|x)$$

经验发现 $\alpha = 0.1$ 可以改善收敛和泛化。

### 3.3 训练流程

1. 对每个问题，用 base model 以 temperature=0.5 生成 16 个候选回复
2. 对每个 (prompt + answer) 对，计算 token-level log-probabilities
3. 应用 assistant mask 仅保留回答部分的 token
4. 计算 masked log-prob 之和得到回复的 log-likelihood
5. 根据 self-confidence loss 进行反向传播更新参数

训练仅需 **10 或 20 步**，使用 8 张 NVIDIA A100 GPU (80GB)，AdamW optimizer，learning rate $1 \times 10^{-5}$，生成长度限制为 3072 tokens。

---

## 4. 数据集

**训练数据：**
- **AIME2024** 训练集（NuminaMath 中的 competition math 题目）——仅用题目（questions only），不使用任何标签

**评估 Benchmarks：**
- AIME2024
- MATH500
- AMC23
- GSM8K
- Minerva Math
- Olympiadbench
- MMLU Stem
- GPQADiamond

---

## 5. 评估指标与主要结果

**评估指标：** Accuracy（正确回答数 / 总样本数）、Pass@1

### 主要结果（Qwen2.5-Math-7B）

| Benchmark | Baseline | RLSC | 提升 |
|---|---|---|---|
| AIME2024 | 13.3 | 26.7 | **+13.4** |
| MATH500 | 51.4 | 72.6 | **+21.2** |
| AMC23 | 45.0 | 54.7 | **+9.7** |
| GSM8K | 84.3 | 86.3 | +2.0 |
| Olympiadbench | 15.1 | 35.9 | **+20.8** |
| Minerva Math | 10.7 | 32.4 | **+21.7** |
| MMLU Stem | 52.3 | 57.6 | +5.3 |

### 主要结果（Qwen2.5-Math-1.5B）

| Benchmark | Baseline | RLSC | 提升 |
|---|---|---|---|
| AIME2024 | 3.3 | 6.7 | +3.4 |
| MATH500 | 35.6 | 62.4 | **+26.8** |
| AMC23 | 34.7 | 46.2 | **+11.5** |
| Minerva Math | 11.4 | 26.1 | **+14.7** |
| MMLU Stem | 34.1 | 48.6 | **+14.5** |

### 关键发现

- 在 7B 规模上，Minerva Math 提升最为显著（+21.7%），AIME2024 和 Olympiadbench 也有大幅提升
- 1.5B 规模上 MATH500 提升最大（+26.8%）
- **涌现行为：** RLSC fine-tuning 后模型倾向于生成更短、更自信的回答，无需 "Let's think step by step" 等提示即可进行简洁推理
- 仅用 16 个采样 + 10-20 步训练即可获得显著提升，资源需求极低
