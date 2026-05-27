# SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization

> **加入 Survey 时间：** 2026-03-11

> arXiv 2511.17938, Nov 2025
> Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, Jianfei Cai
> Monash University, Imperial College London

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| SPINE | Policy Opt. | test-time | Token |

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

> **Reclassification note (2026-05-19):** 此前归类为 Family I（Prediction-Statistic Optimization）。在更细致地审查 reward 构造后，按 dominant-artifact rule 重新归入 Family II：SPINE 的 reward 本体是 self-consistency majority voting (`r=1[y=maj]`)，与 TTRL / RoiRL / ECHO 同源；entropy band 仅用于 token-level selection（决定哪些 ``forking tokens'' 接受梯度），是 selection mask 而非 reward 本身。因此其 dominant artifact 是多 rollout 间的多数票统计。

SPINE 与 TTRL / RoiRL 的差异不在 reward 信号本身，而在 **gradient routing**：

- **Reward 来源**：majority voting pseudo-reward —— 即 reward 本体，与 TTRL 同源。
- **创新点（次级机制）**：Token 级别选择性更新——仅对高 entropy 的 "forking tokens"（推理分支点）执行梯度更新，低 entropy 的 "flowing tokens" 保持不变。Entropy-band regularization 进一步约束 forking-token entropy 在合理范围。
- **Policy optimization 载体**：GRPO 风格的策略优化目标。
- **无外部标签**：完全不依赖 ground-truth 标签、外部 reward model 或人类标注。

---

## 2. 解决的问题

标准 Test-Time Reinforcement Learning (TTRL) 使用 self-consistency 多数投票作为 pseudo-reward，对所有 token 统一进行策略更新。然而实践中这种方法会出现 **characteristic collapse**：

1. **多数投票奖励饱和**：majority-vote reward 快速增长至 1.0，但 Pass@1 准确率反而下降。
2. **响应长度急剧缩短**：模型趋向于生成短的、一致但错误的回答。
3. **推理多样性丧失**：策略过度优化 pseudo-consensus，而非真正的正确性。

作者将这些问题归因于 **对所有 token 统一更新的根本缺陷**：CoT 中绝大多数 token 是低 entropy 的 "flowing tokens"（约 80%），仅约 20% 的高 entropy "forking tokens" 真正决定推理分支方向。对低 entropy token 的梯度更新是冗余甚至有害的。

---

## 3. 方法介绍

SPINE（**S**elective **P**olicy **I**mprovements at **N**odes of **E**ntropy）包含两个核心模块：

### 3.1 Self-consistency Reward 与 GRPO 目标

- 对每个输入 x，采样 N 条响应 {y_i}，通过多数投票得到共识答案 y*。
- 每条响应获得 rule-based reward r_i = r(y_i, y*)（如精确匹配或代码单元测试）。
- 可选地使用 leave-one-out 变体 y*_{-i} 来减轻 self-inclusion bias。
- 采用 GRPO 计算 group-wise normalized advantage：A_hat_i = (r_i - mean) / (std + epsilon)。
- 使用标准 clipped PPO surrogate objective 进行策略更新。

### 3.2 Forking Token 选择

- 计算每个 token 位置 t 的 predictive entropy：H_t = -sum pi(v|s_t) log pi(v|s_t)。
- 选取 entropy 最高的 **top 20%** token 作为 forking tokens，用二值 mask m_t 标记。
- 梯度更新仅作用于 forking tokens，flowing tokens 的梯度被停止（stop gradient）。
- 同时对 forking tokens 施加 **masked KL divergence** 约束（以 pre-adaptation 模型为 anchor），防止在高不确定性位置出现策略漂移。

### 3.3 Entropy-Band Regularization（分位数带通正则化）

- 对每个样本 i 的 forking tokens 集合 S_i，计算两个分位数阈值：
  - H_low = Quantile_10%（下界）
  - H_high = Quantile_50%（上界，即中位数）
- 使用 hinge loss 约束 forking tokens 的 entropy：
  - l_high = max(0, H_t - H_high)：抑制 entropy 过高（减少噪声监督）
  - l_low = max(0, H_low - H_t)：防止 entropy 过低（维持探索能力）
- 分位数阈值按样本自适应计算，无需任务特定调参。

### 3.4 总体目标函数

L = L_core + R_band

其中 L_core = -E[m_t * l_PPO] + lambda_KL * l_KL^fork，R_band 为 entropy band 正则项。

### Test-time 更新流程

对每个 unlabeled 输入：(i) 采样 N=8 条响应；(ii) 多数投票生成 pseudo-label 并计算 reward；(iii) 计算 advantage；(iv) 计算 token entropy 并选择 forking tokens（top 20%），计算分位数阈值；(v) 最小化总目标 L 更新参数 theta。在 test split 上迭代少量轮次。

---

## 4. 数据集

论文跨 **10 个 benchmark**、4 个任务家族进行评估：

| 任务类型 | 数据集 |
|---------|--------|
| Multimodal VQA | MathVision, SLAKE, MedXpertQA-MM |
| 数学推理 | AIME 2025, AMC, MATH-500 |
| 通用/专家 QA | GPQA, MMLU |
| 医学 QA | MedQA (USMLE), PubMedQA |

**基座模型**：
- Qwen2.5-VL-3B-Instruct（多模态）
- Qwen3-1.7B（通用文本）
- Qwen2.5-Math-1.5B（数学专用）

所有实验基于 4x NVIDIA A100 80GB，使用 EasyR1 框架。

---

## 5. 评估指标与主要结果

**评估指标**：Pass@1 accuracy（greedy decoding, temperature=0），经标准归一化、大小写折叠、LaTeX 规范化、SymPy 代数等价检验后进行匹配。

### 主要结果

**Multimodal VQA（Qwen2.5-VL-3B-Instruct）**：

| 方法 | MathVision | SLAKE | MedXpertQA-MM | MedQA | PubMedQA | Avg |
|------|-----------|-------|--------------|-------|----------|-----|
| No adaptation | 19.65 | 26.17 | 17.17 | 30.40 | 68.00 | 32.28 |
| TTRL | 22.73 | 30.00 | 22.61 | 51.88 | 71.50 | 39.74 |
| **SPINE** | **27.26** | **38.66** | **23.92** | **55.40** | **76.20** | **44.29** |
| Delta vs TTRL | +4.5 | +8.7 | +1.3 | +3.5 | +4.7 | +4.6 |

**数学推理（Qwen2.5-Math-1.5B）**：

| 方法 | AIME 2025 | AMC | MATH-500 | GPQA | Avg |
|------|----------|-----|----------|------|-----|
| No adaptation | 10.00 | 28.91 | 30.20 | 4.06 | 18.29 |
| TTRL | 16.67 | 49.88 | 66.42 | 25.38 | 39.59 |
| **SPINE** | **20.00** | **59.03** | **77.00** | **30.96** | **46.75** |
| Delta vs TTRL | +3.3 | +9.2 | +10.6 | +5.6 | +7.2 |

**数学推理（Qwen3-1.7B）**：

| 方法 | AIME 2025 | AMC | MATH-500 | GPQA | MMLU | Avg |
|------|----------|-----|----------|------|------|-----|
| TTRL | 26.67 | 53.01 | 79.86 | 29.94 | 71.19 | 52.13 |
| **SPINE** | **36.67** | **61.46** | **81.40** | **36.04** | **72.66** | **57.65** |
| Delta | +10.0 | +8.5 | +1.5 | +6.1 | +1.5 | +5.5 |

### 关键发现

1. **SPINE 在所有 10 个 benchmark 上均一致超越 TTRL**，平均提升 4.6-7.2 个百分点。
2. **训练动态更稳定**：SPINE 避免了 TTRL 的 reward 饱和、响应缩短和 entropy 失控问题。
3. **跨任务泛化良好**：在 AIME 2025 上适配后，AMC、MATH-500、GPQA 均获得正向迁移（Table 3，平均从 32.31 提升至 46.96）。
4. **Ablation 表明两个模块互补**：Forking Token 选择贡献 +3.2（从 27.56 到 30.76），Entropy-Band 额外贡献 +3.2（到 33.99），合计比 base 提升 +15.7。
5. **对 SFT-based 方法（LMSI, SEALONG）具有明显优势**，后者在某些 benchmark 上甚至低于 no-adaptation baseline。
