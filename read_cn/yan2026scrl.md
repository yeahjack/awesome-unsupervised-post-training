# SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time

> **加入 Survey 时间：** 2026-04-14

**论文：** arXiv 2603.19880v1
**作者：** Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan
**日期：** 2026-03-20

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| SCRL | Policy Opt. | test-time | Semantic |

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

**Family II（Sample-Relation Supervision，population consensus with negative pseudo-labeling）**。

SCRL 属于 strict UPT，原因如下：

- **存在真实的 online RL 更新**：在 test-time 用 GRPO 进行 policy optimization。
- **正负监督都来自模型自身 rollout 分布**：正标签来自 selective positive pseudo-labeling；负标签来自低频且高不确定答案构成的 negative set。
- **没有 external ground truth / verifier / tool / environment reward**：没有人工标签、外部 judge、工具执行或环境反馈。
- **核心信号依赖 rollout population 的分布结构**：无论是正标签的严格 consensus 判据，还是负标签的 frequency + entropy 规则，本质都建立在同一 query 下多条模型输出之间的关系上。

因此，它是 TTRL 系列在 Family II 内部的一次较直接扩展，而不是跳到 tool-grounded 或 self-reward model 路线。

---

## 2. 解决的问题

SCRL 聚焦于一个更尖锐的问题：**weak consensus**。

- 当答案分布高度分散时，majority voting 仍会硬选一个“多数答案”，但这个多数很可能并不可靠。
- 在 GRPO 里，这样的错误 pseudo-label 会被 group normalization 放大，形成 **label noise amplification**。
- 同时，作者指出：在高不确定场景里，“找出正确答案”很难，但“排除明显不可信答案”往往更可靠，因此 TTRL 不该只做 positive pseudo-labeling。

---

## 3. 方法介绍

### 3.1 选择性正伪标签（Selective Positive Pseudo-Labeling）

- 不再像 TTRL 那样默认把众数答案当正标签。
- 只有当：
  - top answer 占比超过阈值 `τpos`；
  - 且与第二名拉开 margin `τmarg`；
  才会声明正 pseudo-label。
- 若不满足条件，则 **abstain**，即不给正标签。

### 3.2 熵门控负伪标签（Entropy-Gated Negative Pseudo-Labeling）

- 对每个 answer 统计：
  - rollout proportion；
  - trajectory-level uncertainty（由 token entropy 聚合）。
- 若某 answer **低频且高不确定**，则将其纳入 negative set `N-`。
- 这让模型在没有可信正标签时，仍能通过负标签去 pruning 明显差的 trajectory。

### 3.3 动态奖励塑形（Dynamic Reward Shaping）

- reward 同时结合：
  - positive label；
  - negative label；
  - 基于 entropy 的惩罚项。
- 强 consensus 时加强正向强化；
- 弱 consensus 时避免错误正强化，同时保留探索。

---

## 4. 数据集

主实验覆盖：

- **AIME25**
- **AMC**
- **MATH-500**
- **Minerva**
- **GPQA**

另有 instruct model 实验：

- **AIME24**
- **MATH-500 / AMC**

模型包括：

- Qwen2.5-3B
- Qwen2.5-Math-7B
- Llama-3.2-1B-Instruct
- Llama-3.1-8B-Instruct

---

## 5. 评估指标与主要结果

代表性结果：

- **Qwen2.5-3B，32 个 candidate / 16 个 train samples**
  - AIME25：**2.6 -> 8.4**
  - AMC：**39.4 -> 41.5**
  - MATH-500：**66.9 -> 68.2**
- **Qwen2.5-Math-7B，32 个 candidate / 16 个 train samples**
  - AIME25：**16.8 -> 26.9**
  - AMC：**65.7 -> 66.9**
  - Minerva：**14.5* -> 41.6**
- **更高 rollout budget（64 个 candidate / 32 个 train samples）** 下，SCRL 仍然保持优势，尤其在 AIME25 和 Minerva 上最明显。
- **Llama-3.1-8B-Instruct** 平均准确率达到 **29.0%**，高于 TTRL、ETMR 和 RESTRAIN。

消融实验（ablation）也很关键：

- 去掉 **Selective Positive Labeling**，在 AIME25 上明显下降，说明“弱 consensus 时不该硬给正标签”。
- 去掉 **Negative Labeling** 会进一步下降，说明负标签不是可有可无的附加项。
- 去掉 **Entropy Gate** 或 **Dynamic Reward** 同样退化，说明 uncertainty-aware filtering 和 reward calibration 都是必要组件。
