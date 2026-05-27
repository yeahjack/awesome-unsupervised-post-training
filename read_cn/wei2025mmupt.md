# MM-UPT: First SFT, Second RL, Third UPT

> **加入 Survey 时间：** 2026-03-11

**Paper:** First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training
**arXiv:** 2505.22453
**Venue:** NeurIPS 2025
**Authors:** Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun
**Affiliations:** Shanghai Jiao Tong University, Zhongguancun Academy, Shanghai Innovation Institute, Lehigh University

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | unlabeled training dataset / per-sample rollout group |
| 参数/状态持久性 Persistence | full parameter accumulate across dataset episodes |
| 与推理关系 Inference Coupling | adapt on the unlabeled cohort, then evaluate on held-out benchmarks |
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

- **更新何时触发：** 更新在 deployment / evaluation 前的无标签训练集上按 episode 持续触发，而不是 arrival-by-arrival 的 test stream。
- **服务当前样本还是后续样本：** 当前样本的 rollout 与 reward 主要服务同一训练 cohort 的后续 step，以及训练完成后的 benchmark 评估。
- **参数/状态是否累积：** 参数在整个无标签 cohort 上持续累积，不做 per-sample reset。
- **reset 边界：** 因此它更接近 offline unlabeled post-training，再做 batch evaluation，而不是在线逐样本 TTA。

## 1. UPT 归属理由

**Family II — Sample-Relation Supervision (population consensus → majority-vote)**

MM-UPT 属于 Family II，理由如下：

- **自奖励机制 (self-rewarding)：** 模型对同一无标注问题采样 G 个回答，通过 **majority voting** 选出出现频率最高的答案作为 pseudo-label $y^*$，再据此给每个回答赋予 binary reward（与 $y^*$ 一致为 1，否则为 0）。
- **Population consensus 信号：** 奖励完全来自模型自身多个采样回答之间的群体共识关系（relational），不依赖任何外部 ground truth、verifier 或人工/AI 标注。
- **GRPO 策略优化：** 将上述 majority-vote binary reward 输入 GRPO 算法进行 online RL，更新 MLLM 策略，使模型收敛到高置信度、一致性的回答。
- **数据自生成 (data self-generation)：** 进一步通过 in-context synthesizing 和 direct synthesizing 两种策略让 MLLM 自行生成新的视觉推理训练数据，伪标签同样由 majority voting 产生。
- **Carrier:** Policy Optimization (GRPO) | **Regime:** training-time | **Level:** Semantic

---

## 2. 解决的问题

MLLM 的 post-training 传统上依赖 SFT 和 RL，两者都需要大量标注数据（ground truth 答案、人类偏好标注、外部 reward model）。随着任务复杂度增长，标注成本不可持续。

MM-UPT 提出将 post-training 划分为三阶段范式：**SFT → RL → UPT (Unsupervised Post-Training)**，其中 UPT 是第三阶段，旨在无任何外部监督的条件下，利用模型自身输出持续提升多模态推理能力。核心挑战在于：在没有 ground truth 的情况下如何为 GRPO 构建可靠的 reward signal。

---

## 3. 方法介绍

### 3.1 Problem Formulation

给定已训练的 MLLM $\pi_\theta$ 和无标注多模态数据集 $Q = \{(I_i, q_i)\}_{i=1}^N$（图像 + 问题，无答案），目标是仅利用模型自身输出提升性能。

### 3.2 Training Method — Majority-Vote GRPO

1. **采样：** 对每个样本 $(I, q)$，用旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个回答 $O = \{o_i\}_{i=1}^G$。
2. **答案提取：** 使用 rule-based answer extractor $E(\cdot)$ 从每个回答中提取答案 $\hat{Y} = \{\hat{y}_i\}_{i=1}^G$。
3. **Majority voting：** 选出出现最多的答案作为 pseudo-label：
   $$y^* = \arg\max_{y \in \hat{Y}} \sum_{i=1}^G \mathbb{1}[y = \hat{y}_i]$$
4. **Binary reward 赋值：** $r_i = 1$ if $\hat{y}_i = y^*$，否则 $r_i = 0$。
5. **Advantage 估计：** $\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$
6. **GRPO 优化：** 用标准 GRPO 目标（含 clipped ratio 和 KL divergence 约束）更新策略参数。

### 3.3 Synthetic Data — 数据自生成

为进一步提升可扩展性，设计两种策略让模型自行生成新训练样本：

- **In-Context Synthesizing：** 给定原始 triplet（图像、问题、答案）作为 in-context example，要求模型生成语义相关但不同的新问题。类似数据改写。
- **Direct Synthesizing：** 只给模型图像，不给参考问题，让其自由生成新问题。多样性更高但可能产生 hallucination。

生成的合成数据同样通过 majority voting 获取 pseudo-label 后用于 MM-UPT 训练。

### 3.4 实现细节

- 基于 **EasyR1** 框架实现，训练 15 个 episode。
- AdamW optimizer，learning rate $1 \times 10^{-6}$，weight decay $1 \times 10^{-2}$，gradient clipping 1.0。
- KL divergence 约束 $\beta = 0.01$，rollout temperature 0.7，group size $G = 10$。
- Vision tower 不冻结，参与训练。

---

## 4. 数据集

### 训练数据（无标注，仅用问题和图像）

| 数据集 | 类型 | 说明 |
|-------|------|------|
| **Geometry3K** | 几何题 | 多选题，几何图形 |
| **GeoQA** | 几何题 | 几何图形 + 多选 |
| **MMR1** | 混合数学推理 | 包含几何、图表等多种视觉数学题 |

### 评估 Benchmark

| Benchmark | 说明 |
|-----------|------|
| **MathVision** | 多模态数学推理 |
| **MathVerse** | 多模态数学推理 |
| **MathVista** | 多模态数学推理，多类型 |
| **We-Math** | 多模态数学推理，多知识层级 |

---

## 5. 评估指标与主要结果

**评估指标：** Accuracy (%)，pass@1 和 pass@10。

### Scenario 1: 标准数据集上的无监督训练（去除 GT 标签）

基于 **Qwen2.5-VL-7B** backbone，在 MMR1 数据集上训练的 MM-UPT 主要结果：

| 方法 | MathVision | MathVerse | MathVista | We-Math | Avg |
|------|-----------|-----------|-----------|---------|-----|
| Qwen2.5-VL-7B (base) | 24.87 | 43.83 | 66.30 | 62.87 | 49.47 |
| + GRPO (supervised, MMR1) | 29.01 | 45.03 | 71.40 | 67.24 | 53.17 |
| + MM-UPT (MMR1) | **26.15** | **44.87** | **72.90** | **68.74** | **53.17** |
| + SRLM (MMR1) | 25.33 | 45.08 | 67.00 | 64.66 | 50.52 |
| + LMSI (MMR1) | 24.83 | 43.76 | 64.90 | 66.38 | 49.97 |
| + Genixer (MMR1) | 23.68 | 43.30 | 65.50 | 64.66 | 49.29 |
| + STIC (MMR1) | 23.78 | 42.72 | 66.10 | 63.74 | 49.09 |

**关键发现：**
- MM-UPT 在所有无监督 baseline 中表现最优，平均提升 +3.7 (49.47→53.17)。
- MM-UPT 甚至与有监督方法（GRPO、rejection SFT）性能持平。
- 在不同 backbone 上均有效：Qwen2.5-VL-3B (+7.4%)、MM-Eureka-7B (+1.3%)、ThinkLite-VL-7B (+2.8%)。

### Scenario 2: 合成数据上的无监督训练

| 数据源 | 数据生成策略 | Avg | 提升 |
|--------|-----------|-----|------|
| Geo3K | Original Questions | 51.23 | +3.6% |
| Geo3K | In-Context Synthesizing | 51.00 | +3.1% |
| Geo3K | Direct Synthesizing | 52.32 | **+5.8%** |
| GeoQA | Direct Synthesizing | 52.67 | **+6.5%** |
| MMR1 | Original Questions | 53.17 | +7.5% |
| MMR1 | In-Context Synthesizing | 52.94 | +7.0% |

**关键发现：** 合成数据训练的效果可与人工撰写问题媲美，Direct Synthesizing 在部分设置下甚至超越原始问题。

### Trade-offs

- **pass@1 提升但 pass@10 下降：** MM-UPT 提升单次准确率，但降低回答多样性（模型倾向于收敛到高共识模式），这是 majority voting reward 的固有 trade-off。
- **失效场景：** 在模型先验知识不足的困难数据集（如 ThinkLite-11K）上，majority voting 会放大错误，导致性能下降（49.47→44.11）。
