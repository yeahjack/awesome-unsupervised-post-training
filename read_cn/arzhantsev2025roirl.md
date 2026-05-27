# RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative RL

> **加入 Survey 时间：** 2026-03-11

**Paper:** RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning
**Authors:** Aleksei Arzhantsev (Criteo AI Lab, Ecole Polytechnique), Otmane Sakhi (Criteo AI Lab), Flavian Vasile (Criteo AI Lab)
**ArXiv:** 2510.02891
**Date:** 2025-10 (NeurIPS 2025 Workshop: Efficient Reasoning)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RoiRL | Policy Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / offline self-evaluation batch |
| 参数/状态持久性 Persistence | full parameter accumulate across offline RL iterations |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新在离线 iterative RL pipeline 中触发，核心是先收集 self-evaluated data，再进行 offline policy improvement。
- **服务当前样本还是后续样本：** 当前 batch 的更新主要服务下一轮离线迭代与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在多轮 offline RL 中持续累积，不做 per-sample reset。
- **reset 边界：** 因此它属于 offline iterative evaluator-driven training，而不是 online TTA。

## 1. UPT 归属理由
**Family II — Sample-Relation Supervision (population consensus)**

> **Reclassification note (2026-05-19):** 此前归类为 Family IV。后按 dominant-artifact rule 重新审视：RoiRL 的 reward 公式 $r=\mathbf{1}[y=\mathrm{maj}]$ 与 \texttt{TTRL} 同源，主导 artifact 是多 rollout 间的多数票统计，而非显式 evaluator / scorer 输出。原文叙事化为 ``self-estimated reward / evaluator-driven RL'' 是 framing 选择，与 TTRL 在 reward formula 层面无形式差异。为保持 family 边界对 reward formula 起作用，RoiRL 现归入 Family II。

RoiRL 使用 **majority vote** 作为自生成的 reward 信号（无需 ground-truth labels），通过 offline iterative RL 循环优化 LLM 的 reasoning 能力。模型自身生成多个 candidate solutions，以 majority vote 估计 correctness 作为 reward $r=\mathbf{1}[y=\mathrm{maj}]$，再通过 weighted log-likelihood objectives 做 policy optimization。该 reward formula 与 \texttt{TTRL} 等 consensus-driven 方法直接同源；与 Family IV 的 \texttt{CoNL}、\texttt{RLME}、\texttt{Meta-TTRL} 不同（后者的 reward 来自显式构造的 judge / introspector，而非直接来自 agreement statistic）。

---

## 2. 解决的问题

1. **RL 训练依赖 ground-truth reward 的瓶颈**：传统 RL 方法（如 GRPO）需要 correctness labels 作为 reward signal，但大规模高质量标注成本高昂且常不可用。
2. **TTRL 的计算开销问题**：Test-Time Reinforcement Learning (TTRL) 虽然用 majority vote 替代了 ground-truth，但其 online RL 方式需要维护 reference model、在训练中反复采样长 CoT 并计算 logits，导致 GPU 内存饱和、扩展困难。
3. **TTRL 的不稳定性**：TTRL 对 hyperparameter 选择高度敏感，online 更新引入训练不稳定性。
4. **核心目标**：在完全无标签的条件下，以一种 **简单、稳定、高效** 的 offline 方法实现 LLM reasoning 的 self-improvement，同时在理论上能够 target 与 TTRL 相同的最优策略。

---

## 3. 方法介绍

### 3.1 问题形式化

给定 base LLM $\pi_0 = \pi_{\theta_0}$ 和 prompt dataset $P_n = \{x_i\}_{i \in [n]}$（**无标签**），目标是通过 self-supervised 方式提升模型 reasoning 能力。

TTRL 的 KL-regularized objective 为：

$$\max_\theta \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_\theta(\cdot|x_i)} [\tilde{r}_k(y, x_i, \theta)] - \beta \text{KL}(\pi_\theta, \pi_0 | x_i)$$

其中 majority vote reward 定义为 $\tilde{r}_k(y, x_i, \theta) = \mathbf{1}[y = \tilde{y}_i^k(\theta)]$，$\tilde{y}_i^k(\theta) = \text{maj}_{\ell \in [k]}(y_i^\ell)$。注意 reward 是 **non-stationary** 的（依赖当前 policy），这使得问题不同于简单的 majority vote distillation。

### 3.2 RoiRL 算法

RoiRL (Reasoning with offline iterative Reinforcement Learning) 在每个 iteration $m \geq 1$ 交替两步：

**Step 1 — Generation（离线数据收集）：** 用当前 policy $\pi_{m-1}$ 为每个 prompt $x_i$ 采样 $k$ 个 candidate solutions $\{c_i^\ell, y_i^\ell\}_{\ell \in [k]}$，用 majority vote reward 打分，构建 offline dataset：

$$D_{m-1} = \{x_i, \{c_i^\ell, y_i^\ell, \tilde{r}_k(y_i^\ell, x_i, \theta_{m-1})\}_{\ell \in [k]}\}_{i \in [n]}$$

**Step 2 — Offline Update（加权 log-likelihood 优化）：**

$$\theta_m = \arg\max_\theta \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_{m-1}(\cdot|x_i)} [g_m(\tilde{r}_k(y, x_i, \theta_{m-1})) \log \pi_\theta(c, y | x_i)]$$

其中 $g_m: \mathbb{R} \to \mathbb{R}$ 为 increasing reward transform。

### 3.3 Reward Transform 变体

论文探索了两种 reward transform：

- **Dense reward $g_\beta$**：$g_\beta(r) = \exp(r / \beta)$，模拟 KL-regularized 行为，trained candidates 按 exponential 加权
- **Sparse reward $g_I$**：$g_I(r) = r$（identity function），等价于只对 majority vote 一致的 correct candidates 做 supervised fine-tuning

### 3.4 理论保证

**Proposition 3.1：** 对任意 $\beta > 0$，存在 reward transforms $(g_m)_{m \in \mathbb{N}}$ 的选择，使得 TTRL 的 Equation (1) 与 RoiRL 的 Algorithm 1 **具有相同的解**。

具体地，analytical solution 在第 $m$ 次 iteration 为：

$$\pi_m(c, y | x_i) \propto \prod_{j=1}^{m} g_j(\tilde{r}_k(y, x_i, \theta_{j-1})) \cdot \pi_0(c, y | x_i)$$

当选择 $g_j = g_\beta = \exp(r/\beta)$ 时，这等价于 KL-regularized RL 的闭式解。

### 3.5 RoiRL 的计算优势

相比 TTRL，RoiRL 有以下计算优势：

1. **无需 reference model**：不用在内存中维护 $\pi_0$，降低内存开销
2. **更好的 batching**：严格分离 generation 和 training 阶段，generation 时可对多个 questions 一起 batch（online RL 做不到）
3. **无需存储 logits**：reward 不依赖 logits，生成阶段可用更大 batch size
4. **Sparse reward 加速**：$g_I$ 在早期阶段（majority answers sparse 时）可显著加速训练

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | MATH500 Train | 400 道题，**无标签**用于训练，ground-truth 仅用于评估 |
| 数学推理 | MATH500 Test | 100 道题，评估用 |
| 数学推理 | AMC | 竞赛数学，评估 generalization |
| 数学推理 | AIME 2024 | 高难度竞赛数学，评估 generalization |

**Base Models（三个不同规模/架构）：**
- Qwen2.5-Math-1.5B
- Phi-4-mini-reasoning-4B
- Llama-3.2-3B-Instruct

**训练设置：**
- 每个 prompt 生成 $k = 10$ candidates
- 每 round 训练 3 epochs，最多 15 rounds（early stopping if maj₁₀ accuracy 连续 5 rounds 无提升）
- TTRL 使用 GRPO，$\beta = 0.1$
- RoiRL $g_\beta$ 使用 $\beta = 0.1$

---

## 5. 评估指标与主要结果

### 评估指标

- **maj₁ (greedy decoding)**：单次采样准确率
- **maj₁₀**：10 次采样 majority vote 准确率
- **maj₁₂₈**：128 次采样 majority vote 准确率（强baseline）

### 主要结果

**Table 1：在 MATH500 Train 上无标签训练，多个数据集上评估**

#### Qwen2.5-Math-1.5B

| 方法 | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.244 | 0.239 | 0.170 | 0.036 |
| TTRL | 0.307 | 0.298 | 0.214 | 0.026 |
| **RoiRL $g_I$** | **0.686** | **0.587** | **0.337** | **0.083** |
| RoiRL $g_\beta$ | 0.670 | 0.604 | 0.340 | 0.070 |

| 方法 | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.572 | 0.520 | 0.445 | 0.100 |
| TTRL | 0.625 | 0.560 | 0.469 | 0.066 |
| **RoiRL $g_I$** | **0.712** | **0.690** ⋆ | **0.518** ⋆ | 0.133 |
| RoiRL $g_\beta$ | 0.685 | 0.650 | 0.469 | **0.200** |

（⋆ 表示超过 base model 的 maj₁₂₈ 性能：0.717/0.680/0.506/0.233）

#### Phi-4-mini-reasoning-4B

| 方法 | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.210 | 0.160 | 0.071 | 0.000 |
| TTRL | 0.272 | 0.225 | 0.090 | 0.000 |
| **RoiRL $g_I$** | **0.660** ⋆ | **0.511** | **0.246** | **0.016** ⋆ |

| 方法 | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.420 | 0.350 | 0.157 | 0.000 |
| TTRL | 0.483 | 0.460 | 0.193 | 0.000 |
| **RoiRL $g_I$** | **0.720** ⋆ | **0.680** ⋆ | **0.421** ⋆ | **0.067** ⋆ |

#### Llama-3.2-3B-Instruct

| 方法 | MATH500 Train (maj₁) | MATH500 Test (maj₁) | AMC (maj₁) | AIME (maj₁) |
|------|----------------------|---------------------|------------|-------------|
| Base π₀ | 0.256 | 0.295 | 0.141 | 0.050 |
| TTRL | 0.361 | 0.394 | 0.159 | 0.043 |
| RoiRL $g_I$ | 0.395 | 0.376 | 0.198 | 0.060 |
| **RoiRL $g_\beta$** | **0.487** | 0.256 | 0.090 | 0.020 |

| 方法 | MATH500 Train (maj₁₀) | MATH500 Test (maj₁₀) | AMC (maj₁₀) | AIME (maj₁₀) |
|------|------------------------|----------------------|-------------|--------------|
| Base π₀ | 0.495 | 0.480 | 0.253 | 0.033 |
| TTRL | 0.510 | 0.490 | 0.313 | 0.167 |
| RoiRL $g_I$ | 0.508 | 0.520 | 0.313 | **0.200** ⋆ |
| RoiRL $g_\beta$ | 0.508 | **0.530** ⋆ | 0.229 | 0.100 |

### 训练速度对比

| 方法 | 每 round 耗时 | 相对速度 |
|------|--------------|---------|
| RoiRL $g_I$（sparse reward） | 6552.5s | **2.6× faster** |
| RoiRL $g_\beta$（dense reward） | 8883.5s | 1.9× faster |
| TTRL | 17019.25s | 1× (baseline) |

（Single NVIDIA A100 80GB VRAM 上测试）

### 关键发现

1. **RoiRL 在绝大多数设置中优于 TTRL**：在三个 base model、三个评估集上，RoiRL 一致性地超过 TTRL，同时训练速度达到 **2.5× 以上加速**。

2. **$g_I$（sparse/identity reward）整体效果最优**：尽管理论上 $g_\beta$ 与 TTRL target 相同的 KL-regularized objective，但实验中更简单的 $g_I$（等价于对 majority vote 一致的 candidates 做 SFT）反而表现更好，暗示 **KL regularization 可能不是最优选择**。

3. **RoiRL 是真正的 self-improvement，非 majority vote distillation**：
   - 训练后模型的 maj₁ 可超过 base model 的 maj₁₀ 性能
   - 训练后模型的 maj₁₀ 可超过 base model 的 **maj₁₂₈** 性能（标 ⋆ 的结果）
   - 这说明模型学到的不仅仅是 majority vote 的知识蒸馏，而是 **更general 的 reasoning 能力提升**

4. **Entropy 快速衰减助力收敛**：RoiRL 训练过程中 entropy 迅速降至接近零，而 TTRL 的 entropy 始终较高。这解释了 RoiRL 更快的收敛速度，但也暗示可能需要更多 regularization（如降低学习率或使用 alternative reward functions）来避免过早坍缩。

5. **Non-stationary reward 的巧妙利用**：majority vote reward 随 policy 变化而变化（$\tilde{r}_k$ 依赖当前 $\theta$），这使得每一轮 iteration 的 reward signal 都是动态更新的，避免了单纯 distillation 的局限性。

6. **无需外部标签的完全 self-supervised 循环**：整个流程仅需 unlabeled prompts，reward 完全由模型自身 majority vote 产生，Ground-truth labels 仅用于评估，实现了真正的无监督 reasoning improvement。
