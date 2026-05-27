# DARE: Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-03-11

**Paper:** Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning
**Authors:** Bodong Du, Xuanqi Huang, Xiaomeng Li (HKUST)
**ArXiv:** 2601.21804
**Date:** 2026-01-30

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| DARE | Policy Opt. | Test-time | Traj. |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | benchmark cohort / rollout group |
| 参数/状态持久性 Persistence | full parameter accumulate across test-time episodes; no per-sample reset |
| 与推理关系 Inference Coupling | adapt within the cohort, then infer/evaluate with the updated model |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Evaluation Boundary |
| 证据状态 Evidence Status | protocol-inferred |
| 辅助分类 Timing Regime | Full-Cohort Transductive Adaptation |
| 可见数据范围 Visibility Scope | Full target cohort |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Full-Cohort Transductive Adaptation`；`Visibility Scope=Full target cohort`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Evaluation Boundary`。

- **更新何时触发：** 更新在目标 benchmark 或其无标签 cohort 上按 rollout group / mini-batch 迭代触发，不是 sample-local adapt-and-reset。
- **服务当前样本还是后续样本：** 当前样本或小批次产生的更新主要服务同一 cohort 后续轮次与最终评估，而不是只服务当前样本本身。
- **参数/状态是否累积：** 参数在整个 test-time RL run 中持续累积；通常只在切换到新的 benchmark、模型 run 或独立实验时才重新初始化。
- **reset 边界：** 因此它更接近 cohort-level cumulative TTA，而不是 arrival-by-arrival streaming reset 协议。

## 1. UPT 归属理由

**Family II — Sample-Relation Supervision (population consensus)**

DARE 的核心 reward signal 完全来源于模型自身多次 rollout 的 **经验分布统计量**（empirical distribution），不依赖任何外部 ground truth、verifier 或人工标注。具体而言：

- 给定一个 test query，policy 生成 M 条 rollout，统计各候选答案的 **empirical frequency** 与 **trace-level uncertainty**（token-level entropy 均值）。
- Reward 基于答案在 rollout population 中的 **分布关系**（distributional relations）计算：高频且低不确定性的答案获得高 reward，低频但低不确定性的答案通过 exploration bonus 获得额外激励。
- 整个信号构造过程是 **population-level relational**：每条 rollout 的 reward 取决于它与同一 query 下所有其他 rollout 的分布关系，而非独立的外部评判。

因此 DARE 属于 Family II 中利用群体内部统计关系（consensus / distributional structure）构建监督信号的方法。

---

## 2. 解决的问题

现有 Test-Time Reinforcement Learning (TTRL) 方法普遍采用 **Majority Voting (MV)** 作为 reward estimator：对多条 rollout 取众数答案作为 pseudo-label，然后二值化赋 reward。论文指出 MV 存在两个根本缺陷：

1. **信息损失 (Information Collapse)**：MV 将完整的 rollout 分布压缩为单一众数，丢弃所有非众数但正确的 rollout 信息。论文通过 Theorem 2.1 证明 MV reward 的互信息严格小于原始 reward 的互信息。
2. **系统性偏差 (Latent-Conditioned Bias)**：当 rollout 之间存在正相关（由隐变量 Z 驱动）时，MV 估计的是 latent-conditional mode 而非 marginal expected reward，导致系统性过估计频繁答案的正确性，引发 **confirmation collapse**——错误答案一旦占多数就持续自我强化。

DARE 的目标是用 **分布感知 (distribution-aware)** 的 reward estimation 替代 point-level consensus，从而提供更 informative、更 robust 的学习信号，改善 test-time 自适应的收敛稳定性和最终性能。

---

## 3. 方法介绍

DARE 框架包含五个步骤（对应 Figure 2 的 (a)–(e)）：

### 3.1 Rollout Sampling & Uncertainty-Aware Distribution

给定 test query q，policy $\pi_\theta$ 生成 M 条 rollout $\{\tau_1, \dots, \tau_M\}$，每条 rollout 产生一个最终答案 $\hat{y}_i$。

- **Empirical frequency**：$n(\hat{y}) = \sum_{k=1}^{M} \mathbf{1}[\hat{y}_k = \hat{y}]$
- **Trace-level uncertainty**：对每条 rollout 计算 token-level entropy 的平均值：$u(\hat{y}) = \frac{1}{n(\hat{y})} \sum_{k: \hat{y}_k = \hat{y}} \frac{1}{|\tau_k|} \sum_{i \in \tau_k} \sum_j -P_i(j) \log P_i(j)$
- **Uncertainty-aware empirical distribution**：将 frequency 除以 uncertainty 后归一化：$\hat{p}(\hat{y}) = \frac{n(\hat{y})/(u(\hat{y}) + \epsilon)}{\sum_{\hat{y}'} n(\hat{y}')/(u(\hat{y}') + \epsilon)}$

这样高频且低不确定性的答案获得更高概率，减少对频繁但不可靠答案的偏好。

### 3.2 Distribution-Based Reward & Exploration Bonus

- **Base reward**：直接使用 uncertainty-aware 分布概率作为 reward：$r_{\text{dis}}(y_i) = \hat{p}(y_i)$
- **Exploration bonus**：鼓励模型探索低频但低不确定性的 rollout：$b(y_i) = \left(1 - \frac{n(y_i)}{M}\right) \cdot (1 - u(y_i))$

  该 bonus 在答案频率低且 uncertainty 低时最大，避免放大 noisy rollout。
- **Combined reward**：$r(y_i) = r_{\text{dis}}(y_i) + \alpha \, b(y_i)$，其中 $\alpha \in [0, 1]$ 控制探索强度。

### 3.3 Distribution Support Pruning

移除 empirical probability 低于阈值 $\tau$ 的 rollout，对剩余 rollout 重新归一化：

$$\tilde{p}(y_i) = \frac{\hat{p}(y_i) \, \mathbf{1}[\hat{p}(y_i) \geq \tau]}{\sum_{k=1}^{M} \hat{p}(y_k) \, \mathbf{1}[\hat{p}(y_k) \geq \tau]}$$

剪枝后重新计算所有 distribution-dependent 统计量（$\tilde{n}$, $\tilde{u}$, $\tilde{b}$），最终 reward 为：$r(y_i) = \tilde{p}(y_i) + \alpha \, \tilde{b}(y_i)$。

此步骤去除退化低质量 rollout，降低 reward variance，稳定优化过程。

### 3.4 Test-Time Policy Optimization

使用 GRPO 算法，以上述 refined rollout-level reward 在 test time 更新 policy。整个过程无需外部标签，仅依赖模型自身生成的分布信息。

---

## 4. 数据集

论文在三个推理领域的五个 benchmark 上评估：

| 领域 | 数据集 | 说明 |
|------|--------|------|
| General Reasoning | **MMLU-Pro** | 多任务语言理解（更高难度版本） |
| Mathematical Reasoning | **MATH-500** | 数学推理 500 题子集 |
| Mathematical Reasoning | **AIME 2024** | 美国邀请赛数学竞赛（高难度） |
| Mathematical Reasoning | **AMC** | 美国数学竞赛 |
| Scientific Reasoning | **GPQA** | 研究生水平科学问答 |

**Backbone models**：Qwen2.5-Math-1.5B 和 Qwen3-1.7B。

OOD 泛化实验中，在一个 benchmark 上 adapt 后在其他 benchmark 上评估。

---

## 5. 评估指标与主要结果

### 评估指标

- **Pass@1**（stochastic decoding，temperature=1.0，top-p=0.95）
- 最大生成长度 3,072 tokens

### 主要结果（Table 1）

**Qwen2.5-Math-1.5B：**

| 方法 | MMLU-Pro | MATH-500 | AIME 2024 | AMC | GPQA | Avg |
|------|----------|----------|-----------|-----|------|-----|
| TTRL | 35.6 | 73.0 | 15.8 | 47.3 | 26.1 | 41.5 |
| **DARE** | **38.9** | **73.6** | **19.8** | **50.2** | **28.5** | **44.2** |

**Qwen3-1.7B：**

| 方法 | MMLU-Pro | MATH-500 | AIME 2024 | AMC | GPQA | Avg |
|------|----------|----------|-----------|-----|------|-----|
| TTRL | 46.9 | 78.2 | 24.0 | 52.9 | 31.2 | 48.6 |
| **DARE** | **48.8** | **79.6** | **26.3** | **55.7** | **32.7** | **50.6** |

### 关键发现

1. **全面超越所有 baseline**：DARE 在两个 backbone、五个 benchmark 上均取得最佳平均性能，相比 TTRL 平均提升 2.0-2.7 分。在 AIME 2024 上相对提升 25.3%，AMC 上提升 5.3%。
2. **Ablation 分析**（Table 2）：三个组件逐步叠加，distribution reward 贡献最大（如 AIME 2024 从 7.7 提升至 16.6），exploration bonus 和 pruning 提供互补增益，完整 DARE 效果最优。
3. **OOD 泛化**：在一个 benchmark 上 adapt 后，DARE 在未见 benchmark 上持续优于 TTRL，通常提升 2-5 个百分点，表明 distribution-aware reward 学到的行为更具迁移性。
4. **Rollout correlation 鲁棒性**：当 rollout 相关性增大时，TTRL 性能急剧下降，而 DARE 退化更缓慢（Figure 5），验证了其对 confirmation collapse 的缓解作用。
5. **收敛效率**：DARE 达到相同 accuracy threshold 所需的更新步数比 TTRL 少约 12-26 步（Figure 6），说明 distribution-aware reward 提供了更高效的学习信号。
