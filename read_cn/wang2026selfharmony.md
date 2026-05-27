# Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-03-11

> **Method:** Self-Harmony | **Carrier:** Policy Opt. | **Regime:** Test-time | **Level:** Semantic
>
> Ru Wang, Wei Huang, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo (U of Tokyo, RIKEN, ISM, Google Research Australia)
>
> Published as a conference paper at ICLR 2026. arXiv: 2511.01191v2

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

Self-Harmony 完全不依赖外部 ground-truth 标签、外部 reward model 或人工标注。其核心监督信号来源于模型自身在两种"视角"（原始问题与模型自生成的 paraphrase 问题）下的 rollout 答案频率分布之间的关系：

- **Paraphrase consistency（改写一致性）：** 同一模型分别对原始问题和自生成的改写问题进行 rollout，正确答案应在两种视角下都稳定出现，而错误答案则往往只在某一种表述下频繁出现。
- **Self-supervision（自监督）：** 模型自身同时扮演 Solver（解题器）和 Reframer（改写器）两个角色，通过 prompt 切换实现，无需任何外部模型。
- **Self-play（自博弈）：** Solver 和 Reframer 通过合作式 self-play 相互促进——Solver 学习对齐 harmonic mean pseudo-label，Reframer 学习生成能暴露 Solver 偏差的改写。

最终的 pseudo-label 通过 **harmonic mean score (HMS)** 聚合两个视角的答案频率来选取，这是一种 relational intrinsic reward：它奖励在两个分布中都一致出现的答案，惩罚仅在单一视角下流行的 spurious 答案。整个过程完全是 population-level 的关系型内部监督。

---

## 2. 解决的问题

**Test-time reinforcement learning (TTRL) 中 pseudo-label 选择的脆弱性问题。**

- 在 TTRL 设定下，模型需要在推理时仅利用无标签测试集进行自适应，pseudo-label 的质量直接决定训练稳定性和最终性能。
- 目前主流方法使用 **majority voting** 作为 pseudo-label 选择策略。然而，当模型存在系统性推理偏差时（即 p(Correct | x) < p(Wrong | x)），majority voting 会放大错误——随着采样增多，正确答案的恢复概率趋近于零（"majority-vote trap"）。
- 现有替代方案（如使用外部 reward model 或强大 LLM 做验证）违背了 test-time 自包含的原则。
- Self-Harmony 提出了一种完全无监督的解决方案：利用 paraphrase 一致性和 harmonic mean 聚合，构建比 majority voting 更鲁棒的 pseudo-label。

---

## 3. 方法介绍

### 3.1 核心直觉

正确答案应在语义等价但表述不同的问题上保持稳定，而错误答案往往依赖于特定的表面表述（style-sensitive）。因此，pseudo-label 不应仅基于单一视角的流行度，而应基于跨视角的不变性。

### 3.2 理论基础：从 Majority Voting 到 Harmonic Mean

- 提出 **View-Invariance Assumption**：对于语义等价的查询 x 和 x'，正确标签 C 的生成概率近似不变（p(A=C|x) ≈ p(A=C|x')），而错误标签的概率则随视角变化。
- 定义 **View-Invariant Infomax objective**：J_λ(a) = I(Z_a; A) − λI(Z_a; X)，最大化答案信息的同时惩罚视角依赖性。
- **Theorem 3.2** 证明：当 λ=2 时，该目标的二阶近似最优解恰好等价于 harmonic mean：

  y* = arg max_a  2p₀(a)p₁(a) / (p₀(a) + p₁(a))

  其中 p₀(a) 和 p₁(a) 分别是原始问题和改写问题下答案 a 的经验频率。

### 3.3 Self-Harmony 框架

**单模型双角色：**
- **Solver (π_θ)：** 对原始问题 x 生成 N 个 rollout 答案 {y_i}
- **Reframer (ρ_θ)：** 将原始问题改写为语义等价的 paraphrase x'（通过 prompt 切换，共享参数）
- Solver 再对改写问题 x' 生成 N 个 rollout 答案 {y'_i}

**Pseudo-label 选择：** 计算每个候选答案 a 在两个视角下的经验频率 p̂₀(a) 和 p̂₁(a)，通过 HMS 选取 pseudo-label：y* = arg max_a HMS(p̂₀(a), p̂₁(a))

**实际实现优化：** 将 "reframe → solve" 融合为单次生成动作（模型在一次生成中先改写再解题），将三步流程压缩为两次模型调用。

### 3.4 Reward 设计

- **Solver reward：** R_solve(y) = I[y = y*]（答案是否匹配 pseudo-label）
- **Fused reframe-and-solve reward：** R_fused(y') = (1 − w_f · R^penalty_format(y')) · (1 − w_d · R^penalty_div(y', y)) · I[y' = y*]
  - Format Penalty：惩罚结构违规的改写
  - Diversity Penalty：基于 Jensen-Shannon divergence 惩罚与原始答案过于相似的改写
  - 仅在答案正确时才给予 reward（correctness 作为 success gate）

**策略优化：** 使用 GRPO 目标函数，分别对原始分支和改写分支计算 reward 并联合优化。

---

## 4. 数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| **MATH500** | 数学推理 | Hendrycks et al., 2021；500 道数学题 |
| **GSM8K** | 数学推理 | Cobbe et al., 2021；小学数学应用题 |
| **AIME 2024** | 数学竞赛 | 高难度数学竞赛题 |
| **AMC** | 数学竞赛 | 美国数学竞赛题 |
| **GPQA-Diamond** | 多学科 | Rein et al., 2023；研究生水平科学问答 |
| **MMLU-Pro** | 多任务 | Wang et al., 2024；多领域多任务推理 |

所有实验均在 label-free 设定下进行：每个数据集独立训练一个模型，训练过程中从不使用 ground-truth 标签生成 pseudo-label 或计算 correctness 信号。

---

## 5. 评估指标与主要结果

### 评估指标
- **pass@1 accuracy**（16 rollouts 的平均 pass@1）
- 训练稳定性（zero training failure）
- Pseudo-label 质量（准确率、F1-score、Spearman correlation）

### 主要结果（Table 1）

**五个基座模型：** Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct

**Baselines：** Intuitor, Rent, Majority-Voting (TTRL), Co-Reward

**关键数字：**

| 模型 + 数据集 | Before RL | Self-Harmony | 提升幅度 |
|---|---|---|---|
| Qwen3-1.7B + MATH500 | 42.70 | **69.60** | +26.9 |
| Qwen3-4B + MATH500 | 60.20 | **78.50** | +18.3 |
| Qwen3-8B + MATH500 | 66.80 | **80.00** | +13.2 |
| Llama-3.1-8B + GSM8K | 95.60 | **91.59** | — |
| Llama-3.1-8B + MATH500 | 41.46 | **50.40** | +8.94 |

- 在 30 个配置（5 模型 × 6 benchmark）中，Self-Harmony **排名第一 28 次，第二 2 次**。
- 所有实验 **零训练失败**（zero training failure），展现了前所未有的鲁棒性；而部分 baseline 在峰值后性能显著退化。
- Pseudo-label 准确率始终最高（约 80–85% on MATH500 Level 3），显著优于 Co-Reward 和 TTRL。

### Ablation（Table 2, MATH500）

| 变体 | Qwen3-4B | Qwen3-8B |
|---|---|---|
| Self-Harmony (Full) | 78.50 | 79.80 |
| w/o Format Reward | 78.40 | 77.46 |
| w/o Diversity Reward | 78.20 | 78.90 |
| Cross Selection 替代 HMS | 76.50 | 78.40 |
| Majority Voting 替代 HMS | 77.30 | 79.00 |

所有组件（HMS、Format Reward、Diversity Reward）均贡献于最终性能，验证了框架设计的必要性。
