# RENT: Reinforcement Learning via Entropy Minimization

> **加入 Survey 时间：** 2026-03-11

**Paper:** Maximizing Confidence Alone Improves Reasoning
**Authors:** Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, Deepak Pathak (Carnegie Mellon University)
**arXiv:** 2505.22660
**Method:** RENT | **Carrier:** Policy Opt. (GRPO) | **Regime:** training-time | **Level:** Token

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

RENT 属于 **Family I (Prediction-Statistic Optimization)**，具体归属理由如下：

- **Intrinsic reward 来源：** RENT 使用模型自身 token 预测分布的 negative average entropy 作为 reward signal，即 $\mathcal{R}(y_{\text{pred}}) = -H(\pi(x)) = \frac{1}{T}\sum_{t=1}^{T}\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$。该信号完全来自模型的 intrinsic predictive statistics，不依赖任何外部 ground-truth、外部 verifier 或人类标注。
- **Carrier 为 policy optimization：** 采用 GRPO 算法，以 entropy reward 驱动策略更新，属于 training-time 的 RL 优化。
- **Token-level signal：** Reward 在每个 token 的预测分布上计算 entropy，再取平均，属于 token-level 的 intrinsic statistics。
- **无外部监督：** 训练过程中 $y_{\text{target}}$ 完全不参与 reward 计算，满足 UPT 定义。

---

## 2. 解决的问题

当前基于 RL 的 LLM 推理能力提升方法高度依赖外部监督信号（如 ground-truth answer 的 correctness reward），而在许多实际场景中，外部标注稀缺或不可得。RENT 提出一个核心问题：**能否仅利用模型自身的 confidence（置信度）作为 reward，在完全无监督的情况下提升 LLM 的推理能力？**

具体而言，论文指出：
- Reward engineering 是 RL 的核心难题，外部 reward 设计成本高且领域特异性强。
- 在开放式、长文本的自由回答场景中，majority voting 等替代方案不可行。
- 模型的 token entropy 与 answer accuracy 之间存在正相关，尤其在 response 末尾（接近最终答案的部分）相关性最强。

---

## 3. 方法介绍

### 核心思想
RENT 将模型 token 预测分布的 negative entropy 作为 RL reward，鼓励模型生成高置信度的 chain-of-thought 及答案。

### 具体步骤
1. **Entropy Reward 定义：** 对给定 prompt $x$，模型生成 response $y_{\text{pred}} = (y_{\text{pred},1}, \dots, y_{\text{pred},T})$。每个 token $t$ 处的 entropy 为 $H(p_t) = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$。整条 response 的 reward 为所有 token entropy 的负平均值：
$$\mathcal{R}(y_{\text{pred}}) = \frac{1}{T}\sum_{t=1}^{T}\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$$
2. **Policy Optimization：** 采用 GRPO (Group Relative Policy Optimization) 算法。GRPO 评估当前策略相对于一组 reference policies 的 reward 改进，适合 noisy 或 unsupervised reward 场景。学习目标为：
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y_{\text{pred}} \sim \pi(x)}[\mathcal{R}(y_{\text{pred}})]\right]$$
3. **Token Selection 策略分析：** 论文研究了对哪些 token 计算 entropy 最有效。实验表明 response 末尾 token（"last chunk"策略）的 entropy 与 accuracy 相关性最高，说明模型在接近最终答案时更依赖自身 confidence。不过默认实现使用全 token 平均。
4. **训练细节：** 使用 Adam optimizer，学习率 $1 \times 10^{-6}$；在每个 dataset 上独立训练；部分 benchmark 用同一数据集同时训练和评估（因为不依赖 ground-truth）。

### 与格式 reward 的对比
论文验证 RENT 不仅是学会了正确格式——与仅使用 format reward（binary reward 判断是否遵循 \boxed{} 等格式）相比，RENT 在多数 benchmark 上一致优于 format-only baseline。

---

## 4. 数据集

| 数据集 | 描述 | 规模 |
|--------|------|------|
| **GSM8K** | 小学数学应用题 | 训练集 ~7473 题，测试集 ~1319 题 |
| **MATH500** | MATH 测试集的 500 题子集（OpenAI 采样），涵盖七类竞赛数学 | 500 题 |
| **AMC** | 2022 和 2023 AMC12 考题（改为整数作答格式） | 83 题 |
| **AIME24** | 2024 AIME 两个版本的题目 | 30 题 |
| **GPQA** | PhD 级别的生物、物理、化学多选题（"Google-proof"） | 448 题 |

说明：除 GSM8K 外，其余 benchmark 训练和评估使用相同数据集（因为 RENT 不使用 ground-truth，不存在传统意义上的过拟合）。

---

## 5. 评估指标与主要结果

### 评估指标
- **Accuracy（准确率）：** 各 benchmark 上的答案匹配准确率（string matching）。
- **Standard deviation：** 在 GSM8K (5 samples)、MATH500 (5)、AMC (32)、AIME (64)、GPQA (10) 上报告样本标准差。

### 主要结果

**跨模型提升（Figure 2 / Table 1）：** 在 Mistral-7B, Llama3.1-8B, Qwen2.5-1.5B, Qwen2.5-Math-1.5B, Qwen2.5-7B, Qwen2.5-Math-7B 六个模型上，RENT 在几乎所有 benchmark 上均优于 baseline：

| 模型 (w/ RENT) | GSM8K | MATH500 | AMC | AIME | GPQA |
|---|---|---|---|---|---|
| Mistral-7B | 0.492 | 0.168 | 0.068 | 0.033 | 0.267 |
| Llama3.1-8B | 0.859 | 0.548 | 0.339 | 0.082 | 0.332 |
| Qwen2.5-1.5B | 0.748 | 0.597 | 0.298 | 0.072 | 0.267 |
| Qwen2.5-Math-1.5B | 0.863 | 0.810 | 0.504 | 0.145 | 0.285 |
| Qwen2.5-7B | 0.911 | 0.823 | 0.518 | 0.270 | 0.365 |
| Qwen2.5-Math-7B | 0.967 | 0.882 | 0.591 | 0.167 | 0.400 |

**与并行工作对比（Table 2, Qwen2.5-7B-Instruct）：**

| 方法 | GSM8K | MATH500 | AMC | AIME | GPQA | Average |
|---|---|---|---|---|---|---|
| TTRL | **0.933** | 0.822 | 0.521 | 0.172 | 0.346 | 0.559 |
| Intuitor (forward KL) | 0.929 | 0.783 | **0.525** | 0.200 | 0.337 | 0.555 |
| Spurious Rewards | 0.910 | 0.774 | 0.459 | 0.156 | 0.342 | 0.528 |
| **RENT (Ours)** | 0.911 | **0.823** | 0.518 | **0.270** | **0.365** | **0.577** |

RENT 在平均分上最高（0.577），尤其在最难的 AIME 上大幅领先。

### 关键发现
- **Confidence 与 accuracy 高度正相关：** 训练过程中，模型 confidence 和 accuracy 同步提升（Figure 3）。
- **末尾 token entropy 更重要：** "last chunk" 策略的 entropy-accuracy 相关性远高于 "first chunk"（Figure 4）。
- **不只是格式学习：** RENT 一致优于 format-only reward，说明模型确实学到了推理能力的提升。
- **局限性：** 存在 overconfidence 风险；无监督方法无法匹敌有 ground-truth 的方法；calibration 失误可能导致灾难性错误。
