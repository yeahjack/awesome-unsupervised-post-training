# T³RL: Tool Verification for Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-03-11

> **Method:** T³RL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv 2603.02203 — Ruotong Liao, Nikolai Röhrich, Xiaohan Wang, Yuhui Zhang, Yasaman Samadzadeh, Volker Tresp, Serena Yeung-Levy (LMU Munich, Stanford University)

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | benchmark cohort / rollout group + tool calls |
| 参数/状态持久性 Persistence | full parameter accumulate across test-time episodes; no per-sample reset |
| 与推理关系 Inference Coupling | adapt within the cohort, then infer/evaluate with the updated model |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Evaluation Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Full-Cohort Transductive Adaptation |
| 可见数据范围 Visibility Scope | Full target cohort |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Full-Cohort Transductive Adaptation`；`Visibility Scope=Full target cohort`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Evaluation Boundary`。

- **更新何时触发：** 更新在目标 benchmark cohort 上按 rollout group 触发，但 pseudo-label 生成中额外插入了 tool verification。
- **服务当前样本还是后续样本：** 当前 group 的更新主要服务同一 cohort 后续轮次与最终评估，而不是服务单个样本后立刻 reset。
- **参数/状态是否累积：** 参数在整个 test-time run 中持续累积，不做 per-sample reset。
- **reset 边界：** 因此它在时间结构上仍接近 cohort-level cumulative TTA，只是证据链比 strict UPT 多了一层 tool call。

## 1. UPT 归属理由

T³RL 属于 **Adjacent Paradigm — Tool-Augmented Test-Time Self-Evolution**，是 strict UPT TTRL（Family II: Sample-Relation Supervision / population consensus）的直接扩展。

核心机制：T³RL 在 TTRL 的 majority-vote consensus reward 基础上，引入 **外部工具验证**（code interpreter）来修正 voting weight。具体来说，一个 LLM verifier 将每条 rollout 转写为 Python 代码，由 code interpreter 执行后判断其正确性，verified rollouts 在 weighted majority voting 中获得更高投票权重（ω 倍），从而产生更可靠的 pseudo-label 与 reward 信号。

**与 strict UPT 的关系：** T³RL 的 majority-vote 聚合与 GRPO 训练部分完全继承 TTRL 的 sample-relation supervision 框架，属于 strict UPT 机制。但 tool verification（code executor）引入了 **外部执行环境作为 evidence source**，超越了 strict UPT 的 "no external ground truth / verifier" 硬边界。然而，与传统 RLVR 中的 ground-truth verifier 不同，T³RL 使用的外部工具（Python interpreter）是 **通用且现成可用的**，不需要针对具体任务收集 ground-truth labels。因此 T³RL 处于 strict UPT 与 verifier-grounded RL 的 **frontier hybrid** 位置。

---

## 2. 解决的问题

- **False-popular mode collapse**：TTRL 使用 majority voting 构造 pseudo-label，但当模型内部推理存在系统性偏差时，频率最高的答案可能是错误的。这种 spurious consensus 会在 online RL 的正反馈循环中被强化，导致模型越来越自信地输出错误答案。
- **Unverified consensus 的脆弱性**：纯 self-consistency reward 无法区分"正确的一致"与"错误的一致"，缺乏外部 evidence 进行交叉验证。
- T³RL 提出：通过 test-time tool verification（代码执行）为 rollouts 提供 grounded evidence，使 verification-aware weighted voting 替代 naive majority voting，从而稳定 self-evolution。

---

## 3. 方法介绍

### 3.1 总体框架

T³RL 在 TTRL 基础上增加三个组件：

1. **Verifier（LLM 验证器）**：给定问题 $x$ 与每条 rollout $y_i$，Verifier 将 rollout 中的推理过程转写为轻量 Python 代码，并在代码执行后判断结果是否与 rollout 提取的候选答案一致。
2. **Verification Tool（代码解释器）**：执行 Verifier 生成的 Python 程序，返回执行结果 $a_i$ 作为 evidence。
3. **Verification Weight（验证权重）**：将验证通过的 rollout 在 majority voting 中赋予 $\omega$ 倍投票权重（$\omega \geq 1$），未验证的 rollout 保持单位权重。

### 3.2 核心公式

**候选答案提取：**
$$\hat{a}_i = \text{Extract}(y_i)$$

**工具验证：**
$$(a_i, v_i) = V(x, y_i), \quad v_i = \mathbf{1}[a_i = \hat{a}_i]$$
其中 $a_i$ 是 tool 执行结果，$v_i \in \{0, 1\}$ 表示是否通过验证。

**Verification-aware weighted voting：**
$$w_i = (1 - v_i) \cdot 1 + v_i \cdot \omega$$
$$\tilde{y}^* = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^{N} w_i \cdot \mathbf{1}[a_i = a]$$

**Reward 计算：**
$$r_i^v = \mathbf{1}[a_i = \tilde{y}^*]$$

### 3.3 关键设计细节

- **ω 超参数**：$\omega = 1$ 退化为标准 TTRL（纯 majority voting），$\omega \to \infty$ 近似硬过滤所有未验证 rollout。实验中 $\omega = 5$ 效果最佳。
- **Verifier 容量门槛**：弱 verifier（如 0.5B）会注入噪声而非提供可靠 evidence，导致性能下降。最低需要 1B 级别 verifier。
- **Vote-then-sample**：继承 TTRL 策略，先采 64 条用于 voting，下采样 32 条用于 GRPO 训练。
- **Verifier 独立重算**：系统 prompt 明确要求 verifier "不要假设推理 trace 正确"，而是从原始问题独立重新计算。

### 3.4 超参数

- Learning rate: cosine schedule, peak $5 \times 10^{-7}$; AdamW
- Rollout temperature: 0.6
- Verifier temperature: 0.6，最大生成长度 1,024 tokens
- 最大生成长度: 2,560 tokens
- Verification weight $\omega$: 5
- Episodes: MATH-500 (10), AMC (30), AIME 2024 (80)
- 硬件: 8 × NVIDIA A100 80GB GPUs

---

## 4. 数据集

| 数据集 | 领域 | 说明 |
|--------|------|------|
| **MATH-500** | 数学推理 | MATH 测试集 500 题子集，5 个难度级别（L1-L5） |
| **AMC** | 数学竞赛 | American Mathematics Competition |
| **AIME 2024** | 数学竞赛 | American Invitational Mathematics Examination，最高难度 |

所有数据集均以 **无标签** 方式使用——仅提供问题，不使用 ground-truth answer。

---

## 5. 评估指标与主要结果

### 评估指标

- **pass@1**：非零温度采样 16 个回答取平均正确率（Qwen-Math 实验使用 greedy decoding）。
- **Relative improvement over TTRL**：T³RL 相对 TTRL 的提升百分比。

### 主要结果

**Math-specialized model (Qwen-2.5-Math-1.5B)：**

| Method | AIME 2024 | AMC | MATH-500 | Avg |
|--------|-----------|-----|----------|-----|
| Baseline | 7.7 | 28.6 | 32.7 | 23.0 |
| w/ TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| w/ T³RL | **20.8** | **50.9** | **74.6** | **48.8** |
| Rel. over TTRL | +31.6% | +4.1% | +2.2% | +6.3% |

**Vanilla model (Qwen-2.5-1.5B)：**

| Method | AIME 2024 | AMC | MATH-500 | Avg |
|--------|-----------|-----|----------|-----|
| Baseline | 0.2 | 0.6 | 7.7 | 2.8 |
| w/ TTRL | 3.5 | 28.6 | 63.2 | 31.8 |
| w/ T³RL | **4.1** | **30.7** | **65.0** | **33.3** |

**Instruct models (Llama-3.2-1B-Instruct, Llama-3-3B-Instruct)：** 均获得一致提升。

### 关键发现

1. **越难的 benchmark 获益越大**：AIME 2024（最难）上的相对提升最大（+31.6%），MATH-500（最简单）上提升较小（+2.2%）。这与直觉一致：简单题上 majority voting 已经很准，tool verification 的边际收益有限。
2. **MATH-500 难度分层分析**：L5（最难）上 T³RL 相对 TTRL 提升 4.3%，L1（最简单）仅 0.2%。
3. **计算效率**：T³RL 用 N=16 rollouts 即可超越 TTRL@64，说明 verification 提高了每条 rollout 的信息质量。
4. **Robustness**：T³RL 跨运行的标准差更低（best accuracy std: 2.638 → 1.890），训练更稳定。
5. **更强的 Verifier 进一步提升**：7B verifier 比 1.5B verifier 在所有 benchmark 上更好。
