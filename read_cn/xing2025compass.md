# COMPASS: Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-04-14

**论文：** arXiv 2510.17923v4
**作者：** Jingyu Xing, Chenwei Tang, Xinyu Liu, Deng Xiong, Shudong Huang, Wei Ju, Jiancheng Lv, Ziyue Qiao
**日期：** 2025-12-09

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| COMPASS | Policy Opt. | test-time | Traj. |

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

**Family II（Sample-Relation Supervision，population consensus + path-quality shaping）**。

COMPASS 仍在 strict UPT 范围内，理由如下：

- **有真实的 test-time RL 参数更新**：论文用 GRPO 在无标签数据上继续优化 policy。
- **监督信号完全来自模型内部**：answer reward 来自多次采样后的 confidence-calibrated self-consistency；path reward 来自模型自身 token probability 分布的 entropy / decisiveness。
- **没有 external ground truth / verifier / tool / environment reward**：没有人类标签、外部 AI label、外部 judge、工具执行或环境反馈。
- **主导信号仍是多 rollout 间的关系型监督**：虽然 DPR 引入了 path-level dense reward，但 DCAR 的核心仍是从群体内部构造 pseudo-label 并校准其可信度，因此整体更贴近 Family II，而非单纯 Family I。

---

## 2. 解决的问题

作者认为 TTRL 的 majority voting 至少有两类缺陷：

- **consensus 不可靠**：错误但高频的答案会被当成 pseudo-label，导致错误自强化。
- **只奖励终点，不奖励路径**：只看 final answer 会丢掉 reasoning process 的大量可用信息。

COMPASS 的设计目标是同时解决这两个问题：

- 用 **DCAR** 改造 answer-level pseudo-label，使 outcome reward 更可信；
- 用 **DPR** 对 reasoning path 提供 dense supervision，而不是只看最终答案。

---

## 3. 方法介绍

### 3.1 DCAR：双重校准答案奖励（Dual-Calibration Answer Reward）

DCAR 先做 **confidence-calibrated self-consistency**：

- 对每条 trajectory 计算 top-1 与 top-2 token probability gap 的波动，得到 trajectory confidence；
- 用 confidence-weighted voting 代替 naive majority voting，得到 pseudo-label。

然后再做 **credibility calibration**：

- `CGeneral`：支持当前 pseudo-label 的 response 里，最高的 confidence；
- `CElite`：所有 response 中最高的 confidence；
- 用 `CGeneral / CElite` 作为 pseudo-label 的可信度系数，去缩放 answer reward。

这样就把原来 0/1 的稀疏 answer reward，变成受 consensus 质量调节的连续 reward。

### 3.2 DPR：决断路径奖励（Decisive Path Reward）

- 对每个生成位置计算：
  - **decisiveness**：top-1 与 top-2 token prob 的差；
  - **uncertainty**：token distribution entropy。
- 用 entropy 对 decisiveness 做加权，让模型在“高不确定但又必须做决定”的关键节点上学会更 decisively 地选择 token。

这相当于为 reasoning path 引入了 **process-centric dense reward**。

### 3.3 Final Reward

- 最终 reward 为 `R = Ranswer + Rpath`。
- answer-level 与 path-level 两类内在信号一起驱动 GRPO 更新。

---

## 4. 数据集

论文使用的 benchmark 覆盖数学与一般推理：

- **AIME 2024**
- **AMC**
- **MATH-500**
- **GPQA-Diamond**

模型包括：

- Llama-3.2-1B-Instruct
- Qwen2.5-Math-1.5B
- Qwen2.5-7B

---

## 5. 评估指标与主要结果

主要指标为 **pass@1**。

代表性结果如下：

- **Qwen2.5-Math-1.5B**：
  - AIME 2024：**15.8 -> 18.3**
  - AMC：**47.4 -> 48.6**
  - MATH-500：**72.4 -> 73.1**
  - GPQA：**26.1 -> 29.3**
- **Qwen2.5-7B**（作者在较少训练 epoch 下比较）：
  - AIME 2024：**20.0 -> 23.5**
  - AMC：**76.6 -> 76.9**
  - GPQA：**31.1 -> 31.7**
- **Llama-3.2-1B-Instruct** 上并非所有数据集都稳定增益，尤其在 AIME 上从 **6.7** 降到 **3.5**，作者将其解释为：小模型基础知识不足时，DPR 强调的高熵位置可能更像“真正困惑”而不是“有价值的 reasoning fork”。

消融实验（ablation）结果显示：

- 去掉 **credibility calibration** 性能下降；
- 再去掉 **DPR** 继续下降；
- 再去掉 **confidence calibration** 会退化到更接近 TTRL 的形式。

这说明 COMPASS 的贡献既来自更稳的 consensus，也来自对 path quality 的直接优化。
