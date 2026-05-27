# TTRL: Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-03-11

> **Method:** TTRL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv 2504.16084 — Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui 等 (Tsinghua University, Shanghai AI Lab)

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

TTRL 属于 **Family II (Sample-Relation Supervision)**，子类为 population consensus / majority-vote。

核心机制：在 test time 对同一问题进行多次采样，通过 majority voting 从模型自身的多个输出中聚合出 consensus answer 作为 pseudo-label，再以该 pseudo-label 构造 rule-based reward（匹配=1，不匹配=0），驱动 online RL 训练。整个过程不依赖任何外部 ground-truth label、人工标注或外部 verifier，奖励信号完全来自模型群体输出之间的**关系型内部监督**。

---

## 2. 解决的问题

- 传统 RL for reasoning（如 DeepSeek-R1、GRPO）依赖 labeled data 提供 reward signal，标注成本高且不可扩展。
- Test-Time Scaling (TTS) 方法（如 majority voting、best-of-N）仅在推理时利用多次采样提升表现，但不更新模型参数，无法持久改进模型能力。
- 现实场景中，越来越多的难题缺乏标注答案，模型需要在 **无标签测试数据** 上自我进化。

TTRL 将 TTS 与 Test-Time Training (TTT) 结合，提出在推理阶段直接用 RL 更新模型参数，使模型在无标签数据上实现自我提升。

---

## 3. 方法介绍

### 3.1 总体框架

TTRL 在 test time 执行以下流程：

1. **Label Estimation（标签估计）**：给定问题 $x$，从当前 policy $\pi_\theta$ 中采样 $N$ 个候选输出 $\{y_1, y_2, \dots, y_N\}$，提取各自答案后通过 majority voting 得到 consensus answer $y^*$。
2. **Reward Calculation（奖励计算）**：对每个采样输出 $\hat{y}_i$，使用 rule-based reward：
   $$R(\hat{y}_i, y^*) = \begin{cases} 1, & \text{if } \hat{y}_i = y^* \\ 0, & \text{otherwise} \end{cases}$$
3. **Policy Optimization（策略优化）**：以上述 reward 驱动 online RL（默认使用 GRPO），通过梯度上升最大化期望奖励：
   $$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(y, y^*)]$$

### 3.2 关键设计细节

- **Vote-then-sample 策略**：先采样 64 个输出用于 majority voting 估计标签，再下采样 32 个用于 RL 训练，降低计算开销。
- **Online learning**：模型在训练过程中持续更新，使得 majority voting 的标签质量随模型能力提升而提高，形成正向自强化循环。
- **"Lucky Hit" 现象**：即使 majority vote 估计的标签不正确，由于错误预测通常分散在不同答案上，大部分 reward 仍然正确（实验中 label accuracy 仅 37% 时 reward accuracy 仍达 92%）。
- **兼容多种 RL 算法**：除 GRPO 外，还兼容 PPO 和 PRIME 等算法，训练曲线高度一致。

### 3.3 超参数

- Learning rate: cosine schedule，peak $5 \times 10^{-7}$；AdamW optimizer
- Rollout temperature: 0.6（Qwen2.5-Math 和 LRMs 用 1.0）
- 最大生成长度: LRMs 32,768 tokens，其他模型 3,072 tokens
- Episodes 数量: MATH-500 为 10，AMC 为 30，AIME 2024 为 80
- 硬件: 8 x NVIDIA A100 80GB GPUs

---

## 4. 数据集

| 数据集 | 领域 | 说明 |
|--------|------|------|
| **AIME 2024** | 数学竞赛 | American Invitational Mathematics Examination，高难度 |
| **AMC** | 数学竞赛 | American Mathematics Competition |
| **MATH-500** | 数学推理 | MATH 测试集的 500 题子集，包含 5 个难度级别 |
| **GPQA-Diamond** | 科学问答 | Graduate-Level Google-Proof Question Answering 高质量子集 |

所有数据集均以 **无标签** 方式使用——仅提供问题，不使用 ground-truth answer。

---

## 5. 评估指标与主要结果

### 评估指标

- **pass@1**：主实验使用 non-zero temperature sampling 生成 16 个回答（32k context 下为 4 个），取平均正确率。
- **maj@n**：majority voting 准确率（n=16 或 n=64）。
- **avg@n**：n 次采样的平均准确率。

### 主要结果

**Math Base Models (Qwen2.5-Math 系列)：**

| 模型 | AIME 2024 | AMC | MATH-500 | GPQA | 平均提升 |
|------|-----------|-----|----------|------|---------|
| Qwen2.5-Math-1.5B | 7.7 → 15.8 | 28.6 → 48.9 | 32.7 → 73.0 | 24.9 → 26.1 | +74.4% |
| Qwen2.5-Math-7B | 12.9 → 40.2 | 35.6 → 68.1 | 46.7 → 83.4 | 29.1 → 27.7 | +76.5% |
| **Qwen2.5-Math-7B AIME** | **+211.6%** | | | | |

**Vanilla Base / Instruct Models：**

| 模型 | AIME 2024 | AMC | MATH-500 | GPQA | 平均提升 |
|------|-----------|-----|----------|------|---------|
| Qwen2.5-7B | +194.9% | +62.6% | +33.1% | +5.7% | +43.7% |
| Qwen2.5-32B | +203.8% | +81.9% | +49.1% | +13.6% | +57.7% |
| LLaMA3.1-8B-Instruct | +117.4% | +38.6% | +31.1% | +10.7% | +30.6% |

**跨模型家族泛化（6 个额外模型）**：LLaMA、Mistral、DeepSeek 家族均获得一致提升。

### 关键发现

1. **超越 maj@n 上界**：TTRL 训练后模型的 avg@64 持续超过原始模型的 maj@64，说明模型通过自强化循环"自举"超越了初始监督信号的上限。
2. **逼近 RL (leakage) 上界**：TTRL 在 MATH-500 上的性能曲线接近直接使用 ground-truth label 做 RL 的 leakage 上界。
3. **OOD 泛化良好**：在某一 benchmark 上训练后，在其他 benchmark 上评估 pass@1 均获提升，表明非过拟合。
4. **模型规模 scaling**：1.5B → 7B → 32B 性能单调提升。
5. **失败场景**：当模型先验知识不足（如 AIME 高难度题）或超参数不当（temperature 过高、batch size 过大）时，TTRL 可能失效。
