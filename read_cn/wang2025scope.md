# SCOPE: Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-04-14

**论文：** arXiv 2512.15146v2
**作者：** Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang
**日期：** 2025-12-18

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| SCOPE | Policy Opt. | test-time | Traj. |

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

**Family II（Sample-Relation Supervision，population consensus / subgroup consensus）**。

SCOPE 仍然属于 strict UPT，理由是：

- **存在真实 update-bearing optimization**：方法在 test time 用 GRPO 更新 policy 参数，不只是做搜索或 reranking。
- **监督信号来自模型自身 rollout 与 token probability**：pseudo-label 来自同一 query 下多条模型生成答案，外加模型自身解码分布导出的 step-wise confidence。
- **没有 external ground truth / verifier / tool / environment reward**：没有人工标签、外部 reward model、代码执行器、环境回报，也没有外部 teacher。
- **核心不是直接最小化单条序列的不确定性，而是重构群体内部监督**：它把 TTRL 的单一 majority voting 改造成 confidence-weighted consensus + subgroup-local consensus，因此更接近 Family II 的 sample-relation supervision，而不是 Family I 的 prediction-statistic intrinsic optimization。

---

## 2. 解决的问题

SCOPE 主要针对 TTRL 类方法的两个问题：

- **confirmation bias**：多数投票把所有 rollout 视为同权，错误但高频的答案会被错误强化。
- **reward 过于稀疏**：整组 rollout 只对应一个全局 consensus label，导致监督信号过粗、探索不足。

论文的核心思路是：一方面用 step-wise confidence 区分“高质量高置信 reasoning path”和“只是碰巧高频的答案”；另一方面把整组 rollout 切成多个 subgroup，为不同 subgroup 生成局部 consensus，从而提高 reward 密度和探索多样性。

---

## 3. 方法介绍

### 3.1 逐步置信度（Step-wise Confidence）

- 对每个 token 计算 top-k token probability 的平均负对数，得到 token confidence。
- 按 reasoning step 聚合，再对整条 response 取平均，形成 **average step confidence**。
- 用该 confidence 取代 naive vote count，在 candidate answers 上做 **confidence-weighted voting**，生成全局 pseudo-label。

### 3.2 子组特定奖励（Subgroup-specific Reward）

- 将同一 query 下的全部生成结果划分为若干 subgroup。
- 对每个 subgroup，不是直接在本 subgroup 内投票，而是从全局 rollout pool 做 bootstrap sampling，再用 confidence-weighted voting 得到该 subgroup 的局部 consensus。
- 每条 rollout 的 reward 由它是否匹配本 subgroup 的局部 consensus 决定，因此比 TTRL 的单一全局 label 更细粒度。

### 3.3 子组大小自动选择（Automatic Subgroup Size Selection）

- 对候选 subgroup size 同时衡量：
  - **quality rate**：输出与 subgroup consensus 的一致程度；
  - **exploration rate**：不同 subgroup 中出现不同 consensus 的比例。
- 用 Pareto front + trade-off distance 自动选出 subgroup size，平衡 correctness 与 diversity。

### 3.4 Optimization

- 用 GRPO 在 test time 更新 policy。
- 因此它不是单纯的 reward redesign，而是完整的 test-time RL 自举闭环。

---

## 4. 数据集

- **AIME 2024**
- **AIME 2025**
- **AMC**
- **MATH-500**

评估模型覆盖：

- Qwen2.5-Math-1.5B
- Qwen3-1.7B
- Llama-3.1-8B-Instruct
- Qwen3-8B

---

## 5. 评估指标与主要结果

论文主表明示 SCOPE 在各模型上整体都优于 TTRL 与多种无监督 RL baseline。

几个代表性结果：

- **Qwen2.5-Math-1.5B**：平均分从 **36.95** 提升到 **41.36**；AIME 2024 从 **16.48** 提升到 **22.50**。
- **Qwen3-1.7B**：平均分从 **41.91** 提升到 **44.02**。
- **Qwen3-8B**：平均分从 **58.21** 提升到 **62.20**；AIME 2024 从 **47.13** 提升到 **52.70**；AIME 2025 从 **27.40** 提升到 **31.00**。
- **Llama-3.1-8B-Instruct**：平均分从 **26.38** 提升到 **28.18**，尽管在较易的 MATH-500 上有轻微回落，但在更难竞赛题上增益更明显。

消融实验（ablation）也支持其两个核心部件：

- 去掉 **confidence weighting** 会明显退化，说明 naive majority voting 仍是瓶颈。
- 去掉 **subgroup partition** 也会明显退化，说明更细粒度 reward 对探索确实重要。
