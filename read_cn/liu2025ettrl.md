# ETTRL: Balancing Exploration and Exploitation in LLM Test-Time Reinforcement Learning Via Entropy Mechanism

> **加入 Survey 时间：** 2026-03-11

> **Method:** ETTRL | **Carrier:** Policy Opt. | **Regime:** test-time | **Level:** Semantic
>
> arXiv: 2508.11356 | Jia Liu, ChangYi He, YingQiao Lin, MingMin Yang, FeiYang Shen, ShaoGuo Liu (Kuaishou Technology, Beihang University, Northwestern Polytechnical University)

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

ETTRL 属于 **Family II (Sample-Relation Supervision, population consensus)**。该方法在 TTRL 的基础上进行改进，核心 reward signal 来自模型自身多次 rollout 的 majority voting 结果（consensus-based pseudo-label），不依赖任何外部 ground truth、human label 或外部 verifier。其两个核心组件（ETMR 和 EAR）均围绕 entropy 这一 intrinsic statistic 来平衡 exploration 与 exploitation，所有监督信号均由模型群体内部的一致性关系产生。

---

## 2. 解决的问题

TTRL（Test-Time Reinforcement Learning）允许 LLM 在无标注测试数据上通过 majority voting 产生 pseudo-label 进行自我优化，但存在两个关键缺陷：

1. **推理开销过高**：TTRL 需要数十到上百次并行 rollout 才能获得可靠的 pseudo-label，token 消耗极大。
2. **早期估计偏差（overconfidence）**：训练初期 pseudo-label 准确率很低（如 AIME 上低于 10%），少数"幸运"正确样本获得不成比例的大 advantage，导致模型过早陷入局部最优，阻碍进一步探索。

ETTRL 旨在从 rollout 效率和 reward advantage 估计两个维度同时改善 exploration-exploitation 的平衡。

---

## 3. 方法介绍

ETTRL 包含两个互补组件：

### 3.1 Entropy-fork Tree Majority Rollout (ETMR)

- **核心思想**：利用 token-level entropy 识别推理过程中的关键分叉点（high-entropy tokens，通常对应 "but"、"however" 等逻辑转折词），仅在这些位置进行分支采样，复用 low-entropy tokens。
- **具体流程**：
  - 生成 M 棵独立的 tree，每棵 tree 从一次完整采样开始。
  - 计算每个 token 的 Shannon entropy，选取 entropy 最高的 Top-N 个位置作为 forking points。
  - 在每个 forking point 产生 B 个新分支，各分支独立生成至完整 response。
  - 所有叶节点通过 majority voting 产生 pseudo-label。
- **效率提升**：总 rollout 数为 $R_{tree} = M(1 + B \times N)$。在典型配置 $N=3, B=2$ 下，token 消耗仅为完全并行采样的约 **60%**，同时保持甚至提高采样多样性。

### 3.2 Entropy-based Advantage Reshaping (EAR)

针对早期训练中 advantage 估计的 overconfidence 问题，提出两种策略：

1. **Adv-Clip**：将 GRPO 的 advantage 裁剪到 $[-\beta, +\beta]$ 范围内，直接抑制极端梯度更新，稳定早期训练。
2. **Adv-Res（主要方法）**：基于 response-level 相对 entropy 对 advantage 进行缩放：
   $$Y_i = 1 + \frac{\text{avg}(H_{resp}(o_i)) - H_{resp}(o_i)}{\text{avg}(H_{resp}(o_i))}$$
   $$\hat{A}^{res}_{i,t} = Y_i \cdot \hat{A}_{i,t}$$
   - 高 entropy（低置信度）的 response 被 down-weight，低 entropy（高置信度）的 response 的 advantage 被适度放大。
   - 相比 Adv-Clip，Adv-Res 提供更细粒度的 soft regularization，在所有模型和数据集上表现更优。

---

## 4. 数据集

- **AIME 2024**：美国数学邀请赛竞赛题（30 题），训练 80 episodes
- **AMC**（American Mathematics Competition）：竞赛数学题，训练 30 episodes
- **MATH-500**：数学推理 benchmark，训练 10 episodes

所有数据集在 test-time 使用，**无 ground-truth label**，完全依赖 majority voting pseudo-label。

---

## 5. 评估指标与主要结果

**评估指标**：Pass@1（greedy decoding）

### ETMR 结果（Table 1，与 TTRL 在相近 rollout 数下对比）

| Model | Method | AIME 2024 | AMC | MATH-500 | Avg |
|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| | ETMR | 21.0 | 50.8 | 76.9 | 49.6 |
| | $\Delta$ | +32.9% | +3.9% | +5.3% | +8.1% |
| Qwen2.5-Base-3B | TTRL | 7.9 | 40.7 | 72.2 | 40.3 |
| | ETMR | 9.2 | 41.7 | 71.7 | 40.9 |
| | $\Delta$ | +16.5% | +2.5% | -0.7% | +1.5% |
| Llama-3.1-8B | TTRL | 10.0 | 32.3 | 63.7 | 35.3 |
| | ETMR | 16.9 | 35.4 | 59.5 | 37.3 |
| | $\Delta$ | **+69.0%** | +9.6% | -6.6% | +5.7% |

ETMR 在仅消耗 60% token 的情况下，AIME 2024 上取得显著提升（Llama-3.1-8B 上相对提升 68%）。

### EAR 结果（Table 2，advantage shaping 方法对比）

| Model | Method | AIME 2024 | AMC | MATH-500 | Avg |
|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | TTRL | 15.8 | 48.9 | 73.0 | 45.9 |
| | Adv-Res | 19.6 | 51.0 | 77.3 | 49.3 |
| | $\Delta$ | +24.1% | +4.3% | +5.9% | +7.4% |
| Qwen2.5-Base-3B | TTRL | 7.9 | 40.7 | 72.2 | 40.3 |
| | Adv-Res | 13.1 | 41.4 | 72.4 | 42.3 |
| | $\Delta$ | **+65.8%** | +3.2% | +0.3% | +5.0% |
| Llama-3.1-8B | TTRL | 10.0 | 32.3 | 63.7 | 35.3 |
| | Adv-Res | 13.5 | 36.4 | 61.3 | 37.1 |
| | $\Delta$ | +35.0% | +12.7% | -0.8% | +5.1% |

**关键发现**：
- ETMR 和 EAR 均在所有模型和数据集上一致超过 vanilla TTRL baseline。
- 在较难的 benchmark（AIME 2024）上改善最为显著，说明 entropy-based 方法对复杂推理任务的探索尤其有效。
- Adv-Res 始终优于 Adv-Clip，表明细粒度的 entropy-based soft regularization 比硬裁剪更有效。
- 非数学专用模型（如 Llama-3.1-8B、Qwen2.5-Base-3B）受益更大，因其 epistemic uncertainty 更高。
