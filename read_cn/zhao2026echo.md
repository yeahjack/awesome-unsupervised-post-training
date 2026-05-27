# ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning

> **加入 Survey 时间：** 2026-03-11

> arXiv 2602.02150, Feb 2026
> Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo

| 属性 | 值 |
|---|---|
| Method | ECHO |
| Carrier | Policy Optimization (GRPO-style) |
| Regime | Test-time |
| Level | Token |

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

ECHO 属于 **Family II: Sample-Relation Supervision (population consensus)**.

> **Reclassification note (2026-05-19):** 此前归类为 Family I（Prediction-Statistic Optimization）。在更细致地审查 reward 构造后，按 dominant-artifact rule 重新归入 Family II：ECHO 的 reward 本体是与 TTRL 同源的 majority-vote pseudo-label，entropy 与 confidence 仅作为 token-level advantage shaping (i.e., $A^{\text{hyb}}_{i,t} = A^{\text{grp}}_{g,i}(1 + a S_{i,t})$)，对梯度幅度做局部调制，不构成独立 reward。因此其 dominant artifact 是多 rollout 间的多数票统计，与 TTRL / RoiRL / SPINE 同族。

具体而言：
- **Reward 来源**：majority voting 产生的 pseudo-label（模型自生成内容）—— 即 reward 本体。Entropy-confidence hybrid 信号仅作为 advantage shaping 项调制 token-level 梯度幅度，是次级机制。
- **信号类型**：reward 本身是 multi-rollout consensus（Family II）；entropy $H_i$ 与 confidence $C_i$ 仅在 advantage modulation 中出现。
- **优化载体**：通过 clipped policy-gradient objective（类 GRPO）进行在线 policy 更新，属于 policy optimization carrier。

---

## 2. 解决的问题

现有 Test-Time Reinforcement Learning (TTRL) 方法面临两大挑战：

1. **High-entropy rollout collapse**：基于 entropy 的 tree-search 方法（如 ETMR）在高 entropy 节点频繁分支，导致 branching budget 被少数高 entropy 轨迹消耗殆尽，搜索树退化为近似 chain-like rollout，探索覆盖率和 pseudo-label 多样性严重下降。
2. **Early pseudo-label overfitting（自我强化过拟合）**：训练早期的 pseudo-reward 噪声大且有偏，容易将 policy 推向局部高分解，形成正反馈循环——输出分布快速锐化（entropy 下降）、exploration 枯竭、后期泛化性能退化。

ECHO 的目标是在有限 inference budget 下，同时缓解以上两个问题，提升 test-time 推理质量和泛化能力。

---

## 3. 方法介绍

ECHO 包含三个核心模块：

### 3.1 Entropy-Confidence Hybrid Tree-Structured Rollout

- 使用 window-smoothed token entropy $\bar{H}_t$ 和 grouped confidence $C_t^G$ 联合决定 **branch width** $B_t$：高 entropy + 低 confidence 时增大分支以鼓励探索；高 entropy + 高 confidence 时抑制过度分支以避免 high-entropy trap。
- **Online pruning** 三机制：
  - *Low-confidence pruning*：当 running minimum of grouped confidence $m_t < \tau_{\text{prune}}$ 时终止分支。
  - *Tail-decline pruning*：连续多步 tail confidence 下降时终止。
  - *Entropy-spike pruning*：连续多步 entropy 突增时终止。
- Majority voting 产生 pseudo-label $\hat{y}$，据此对每条轨迹赋予 rule-based reward $R_i \in \{0, 1\}$。

### 3.2 Confidence-Adaptive Clipping

- 基于 trajectory-level tail confidence $C_{\text{tail}}(o_i)$ 动态调节 PPO-style clipping radius $\epsilon(o_i)$：
  - 高 confidence 轨迹 $\Rightarrow$ 更窄的 trust region（防止早期 spurious high-reward 轨迹主导更新）。
  - 低 confidence 轨迹 $\Rightarrow$ 更宽的 trust region（允许更多探索性更新）。
- 公式：$\epsilon(o_i) = \epsilon_{\min} + (\epsilon_{\max} - \epsilon_{\min})\,\sigma\!\big(\kappa(1-C_{\text{tail}}(o_i))\big)$。

### 3.3 Entropy-Confidence Hybrid Advantage Shaping

- 在 GRPO 的 group-normalized trajectory-level advantage $A_{g,i}^{\text{grp}}$ 基础上，引入 token-level shaping signal：
  $$S_{i,t} = \alpha\, H_{i,t} + \beta\,(1 - C_{i,t})$$
- 最终 hybrid advantage：$A_{i,t}^{\text{hyb}} = A_{g,i}^{\text{grp}}\,(1 + a\, S_{i,t})$。
- 效果：将梯度信号向 uncertain yet decision-critical tokens 倾斜，促进对关键推理步骤的学习，同时抑制对 high-entropy degenerate 区域的过度更新。

### 3.4 Overall Objective

$$\mathcal{L}_{\text{ECHO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(r_{i,t}(\theta)\,A_{i,t}^{\text{hyb}},\;\text{clip}(r_{i,t}(\theta),\,1-\epsilon(o_i),\,1+\epsilon(o_i))\,A_{i,t}^{\text{hyb}}\big) - \beta_{\text{KL}}\,D_{\text{KL}}\big(\pi_\theta \| \pi_{\text{ref}}\big)\Big)\right]$$

---

## 4. 数据集

### 自然语言数学推理（训练集 / 测试集）
| 数据集 | 说明 |
|---|---|
| AIME2024 | 美国数学邀请赛 2024 |
| AMC | 美国数学竞赛 |
| MATH-500 | MATH 数据集 500 题子集 |
| GPQA-Diamond | 研究生级别 Q&A |
| AIME2025 | 美国数学邀请赛 2025 |

### 多模态数学推理
| 数据集 | 说明 |
|---|---|
| Geometry3k | 几何推理训练集 |
| GeoQA | 几何 Q&A 训练集 |
| MathVision | 视觉数学推理 |
| MathVerse | 视觉数学推理 |
| MathVista | 视觉数学推理 |
| LogicVista | 逻辑视觉推理 |

---

## 5. 评估指标与主要结果

### 评估指标
- **pass@16**：16 条 rollout 中至少一条答对的准确率（自然语言推理）。
- **pass@1**：单次生成准确率（多模态推理）。

### 主要结果

**自然语言推理 (Table 1, pass@16)**：
| Backbone | 训练集 | AIME2024 | AMC | MATH-500 | GPQA | AIME2025 | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | AIME2024 | **30.0** | **75.9** | **89.4** | **47.7** | **33.3** | **55.3** |
| Qwen2.5-7B | MATH-500 | **33.3** | **75.9** | **90.0** | **49.0** | **43.3** | **58.3** |
| Qwen3-8B | AIME2024 | **53.3** | **90.4** | **95.4** | **81.3** | **60.0** | **76.1** |
| Qwen3-8B | MATH-500 | **46.7** | **90.4** | **94.6** | **89.9** | **56.7** | **75.6** |

- 相较最强 baseline（INTUITOR），ECHO 在各设置下平均提升约 **3--4 个百分点**。
- 在最难任务（AIME2025）上提升可达 **12.36%**。

**多模态推理 (Table 2, pass@1)**：
- 在 Qwen2.5-VL-7B 和 Qwen3-VL-8B 上，ECHO 较 MM-UPT baseline 平均 pass@1 提升约 **1.2--2.4 个百分点**，LogicVista 上相对提升可达 **12.8%**。

### Ablation Study (Table 3)
| 组件 | 作用 |
|---|---|
| EC-Tree（entropy-confidence tree search） | 贡献最大，移除后性能大幅下降 |
| CA-Clip（confidence-adaptive clipping） | 对困难任务（如 AIME2025）影响显著（移除后从 54.3 降至 30.1） |
| E-SC Adv（entropy-confidence advantage shaping） | 稳定贡献，移除后多 benchmark 轻度退化 |

### 关键发现
1. ECHO 有效缓解 high-entropy rollout collapse：branching budget 分配更均匀，有效分支数显著增加。
2. ECHO 减缓 entropy 过早下降，抑制 self-reinforcing overfitting。
3. 在 strict IID 设置（训练集=测试集）下仍有 4--9% 的平均提升，表明方法对分布偏移不敏感。
