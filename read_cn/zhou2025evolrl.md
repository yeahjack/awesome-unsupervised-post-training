# EVOL-RL: Evolving Language Models without Labels

> **加入 Survey 时间：** 2026-03-11

**Paper:** Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation
**arXiv:** 2509.15194
**Method:** EVOL-RL | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / evolutionary round |
| 参数/状态持久性 Persistence | full parameter accumulate across rounds |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | note-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新在 deployment 前的 evolution / coevolution rounds 中触发，基本单位是 prompt batch 加一轮群体选择或变异。
- **服务当前样本还是后续样本：** 当前轮次产生的变体与选择结果主要服务下一轮演化与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在多轮演化中持续累积，不做 per-sample reset。
- **reset 边界：** 因此这类方法的时间结构是 offline evolutionary post-training。

## 1. UPT 归属理由

EVOL-RL 属于 **Family II (Sample-Relation Supervision)**，子类为 population consensus via majority-vote / distributional agreement。

核心理由：EVOL-RL 的监督信号完全来自模型群体内部的关系结构，不依赖任何外部标签、verifier 或人类反馈。具体地，对于每个 prompt，模型采样一组回复并通过 majority voting 确定伪标签（selection 信号），同时计算每条回复相对于同组其他回复的语义新颖度（variation 信号）。两种信号均源自同一批采样回复之间的 distributional agreement 与 semantic dissimilarity，属于 relational / population-level 的内部监督。最终 reward 被送入 GRPO 进行 label-free 的 policy optimization。整个流程不使用任何 ground-truth 答案或外部评判。

---

## 2. 解决的问题

现有无标签自我改进方法面临的核心挑战——**entropy collapse（熵坍缩）**：

- **RLVR** 依赖外部 verifier，仅适用于数学、代码等可自动验证的领域，无法推广到开放域推理
- **TTRL** 等 majority-only 方法虽然无需标签，但仅奖励与多数答案一致的回复，导致模型过度自信、解法多样性丧失，表现为 policy entropy 趋近于零、response length 缩短、pass@n (n>1) 持续下降
- 现有的 entropy regularization 或 clip-high 等通用策略不足以根本解决 "majority trap"
- 需要一种能同时保持**正确性锚定**和**推理多样性**的 label-free 训练框架

EVOL-RL 借鉴生物进化原理（selection + variation），在无标签条件下平衡 exploitation 与 exploration，防止 entropy collapse。

---

## 3. 方法介绍

EVOL-RL 以 GRPO 为优化算法，设计了一套将 selection（选择）与 variation（变异）相结合的 reward 函数：

### 3.1 优化框架 (GRPO)

对每个 prompt $\mathbf{q}$，policy 采样 $G$ 条回复 $\{o_1, \dots, o_G\}$。每条回复获得标量 reward $r_i$，在组内做 z-score 归一化得到 advantage $\hat{A}_i$，然后通过 PPO-style clipped surrogate objective 更新策略，附加 KL penalty 保证稳定性。

### 3.2 Reward 设计：Selection + Variation

每条回复按三个维度打分：

1. **Validity（有效性）**：回复必须在 `\boxed{}` 中给出数值答案，否则 reward = -1
2. **Majority (Selection)**：对有效回复，根据最终答案是否与 majority-voted 答案一致赋予二值标签 $y_i \in \{+1, -1\}$
3. **Novelty (Variation)**：计算每条回复推理部分的 embedding，构造 cosine similarity 矩阵。对回复 $o_i$，在**同组内**计算平均相似度 $\bar{s}_i$ 和最大相似度 $m_i$，novelty score 为：
$$u_i = 1 - (\alpha\,\bar{s}_i + (1-\alpha)\,m_i), \quad \alpha = 0.5$$
之后在 majority / minority 组内分别做 min-max 归一化得到 $\tilde{u}_i$。

### 3.3 Final Reward Mapping

将 majority label 与 novelty score 映射到**不重叠的 reward band**：

$$r_i = \begin{cases} -1, & \text{if invalid} \\ 0.5 + 0.5\,\tilde{u}_i \in [0.5, 1], & \text{if } y_i = +1 \\ -1 + 0.5\,\tilde{u}_i \in [-1, -0.5], & \text{if } y_i = -1 \end{cases}$$

这保证**任何 majority 解的 reward 都高于任何 minority 解**（正确性优先），而 novelty 在各组内部做精细区分（鼓励多样性）。

### 3.4 辅助机制

- **Asymmetric clipping**：$\epsilon_{\text{high}} > \epsilon_{\text{low}}$，让高 advantage 的稀有、新颖且正确的解获得更充分的梯度更新
- **Token-level entropy regularizer**：$\mathcal{L}_{\text{ent}} = -\lambda_{\text{ent}} \mathbb{E}[\frac{1}{|o|}\sum_t H(\pi_\theta(\cdot | o_{<t}, x))]$，维持生成初期的 token-level 多样性
- 总目标：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \mathcal{L}_{\text{ent}}$

### 3.5 进化类比

- **Majority vote = Selection pressure**：锚定正确方向，防止 drift
- **Novelty reward = Variation pressure**：奖励语义独特的推理路径，防止 collapse
- **Entropy regularizer = 高突变率**：维持多样候选解的供给
- **Asymmetric clipping = 保留有利突变**：不让稀有高价值解被 clip 掉

训练动态呈现两阶段：(1) 初始阶段 majority signal 主导，entropy 快速下降（与 TTRL 类似）；(2) "evolving point" 后 novelty + entropy 机制介入，entropy 回升、response length 增长、out-of-domain accuracy 持续提升，TTRL 则永久停留在低 entropy 状态。

---

## 4. 数据集

### 训练集
- **MATH-TRAIN**：大规模标准数学训练集（仅使用题目，不使用标签）
- **MATH-500**：较小的 500 题子集
- **AIME24**：竞赛级数学题（30 题）

### 评估集
- **MATH-500**（in-domain）
- **AIME24**（竞赛数学）
- **AIME25**（out-of-domain 竞赛数学）
- **AMC**（American Mathematics Competition）
- **GPQA-Diamond**（非数学推理，跨域泛化）
- **MMLU-Pro**、**SuperGPQA**、**BBEH**（更广泛的推理 benchmark，仅 8B 模型评估）

### 模型
- **Qwen3-4B-Base**
- **Qwen3-8B-Base**
- （附录中还有 OctoThinker-8B-Hybrid-Base 的补充实验）

---

## 5. 评估指标与主要结果

### 评估指标
- **pass@1**：单次采样准确率
- **pass@16**：16 次采样中至少一次正确的概率（衡量推理多样性）
- 均基于 32 次 rollout 取平均

### 主要结果 (Table 1)

**Qwen3-4B-Base，训练于 MATH-TRAIN：**

| Benchmark | TTRL pass@1/pass@16 | EVOL-RL pass@1/pass@16 | Delta |
|-----------|---------------------|------------------------|-------|
| MATH | 75.4/86.9 | 80.0/93.3 | +4.6/+6.4 |
| AIME24 | 12.1/23.2 | 20.7/47.6 | +8.6/+24.4 |
| AIME25 | 6.8/28.6 | 17.5/39.9 | +10.7/+11.3 |
| GPQA | 36.5/81.4 | 37.2/88.7 | +0.7/+7.3 |

**Qwen3-8B-Base，训练于 MATH-TRAIN：**

| Benchmark | TTRL pass@1/pass@16 | EVOL-RL pass@1/pass@16 | Delta |
|-----------|---------------------|------------------------|-------|
| AIME24 | 16.7/37.6 | 26.0/51.7 | +9.3/+14.1 |
| AIME25 | 15.6/35.9 | 21.6/43.1 | +6.0/+7.2 |

### 关键发现

1. **pass@1 和 pass@16 同时提升**：EVOL-RL 在所有实验配置中均一致优于 TTRL baseline，pass@16 增益尤为显著（AIME24 上 4B 模型 +24.4 pp），表明多样性保持带来的多路径探索收益
2. **跨模型规模和训练数据规模的一致性**：4B/8B 模型、大规模 MATH-TRAIN / 小规模 MATH-500 / AIME24 均见效
3. **强跨难度鲁棒性**：在 MATH-500 上训练的 4B 模型，其 AIME24/25 性能几乎与直接在 AIME24 上训练相当，说明学到的是可迁移的推理能力而非过拟合
4. **非数学域泛化**：在 GPQA 上 TTRL 导致 pass@16 相对 base model 下降，而 EVOL-RL 稳定恢复并提升（+7 到 +15 pp over TTRL）；在 MMLU-Pro/SuperGPQA/BBEH 上 EVOL-RL 的 pass@1 和 pass@4 均优于 TTRL 和 base model
5. **Ablation**：三个组件（novelty reward、entropy regularizer、asymmetric clipping）协同作用，完整配置最优；移除 novelty reward 在简单数据集上影响最大，移除 entropy/clipping 在困难数据集上影响最大
6. **组件可迁移至有监督 RLVR**：将 EVOL-RL 的三个 exploration 组件加入标准有标签 GRPO (RLVR)，pass@16 在 AIME24/25 上提升 7%-12%
