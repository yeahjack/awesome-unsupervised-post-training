# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

> **加入 Survey 时间：** 2026-03-11

> 论文: arXiv 2511.16672
> 方法: EvoLMM | Carrier: Policy Opt. | Regime: training-time | Level: Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / rollout group |
| 参数/状态持久性 Persistence | full parameter accumulate across RL steps |
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

- **更新何时触发：** 更新在 deployment 前的离线 RL 阶段触发，基本单位是 prompt batch 下的一组 rollouts。
- **服务当前样本还是后续样本：** 当前 rollout group 的更新服务后续训练 step 与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整段 RL 训练中持续累积，不做 per-sample reset。
- **reset 边界：** 因此它属于 offline RL-style UPT schedule，而不是 test-time arrival-by-arrival adaptation。

## 1. UPT 归属理由

**Family II — Sample-Relation Supervision (trajectory-level consistency)**

EvoLMM 从单一 backbone LMM 中实例化两个协作角色：**Proposer** 自主生成基于图像的视觉问题（image-grounded questions），**Solver** 对同一问题采样多条回答 rollout。训练信号完全来自 Solver 多条回答之间的 **inter-rollout internal consistency**（即经验答案分布的 agreement 程度），而非任何外部 ground truth、人工标注或外部 reward model。

具体而言：
- Solver reward 是基于多采样答案一致性的 **continuous self-consistency reward**，与答案 agreement 平滑地成比例缩放，替代了离散 majority-vote reward。
- Proposer reward 是基于 Solver 答案熵的 **entropy-based band-pass function**，鼓励 Proposer 生成中等难度的问题（既非过于简单也非无法回答）。
- 整个训练过程仅使用 raw images，不使用任何 QA 标注、metadata 或外部 verifier。

这将 sample-relation supervision 从纯文本 LLM 扩展到 VLM 领域，信号来源是模型自身多条 trajectory 之间的关系性一致性。

---

## 2. 解决的问题

现有 LMM 自改进方法存在两大依赖：(i) 依赖人工标注数据或 metadata；(ii) 依赖外部定义的 reward model 或 evaluator（如 knowledge-distilled reward networks）。这些依赖限制了自主性和可扩展性。

此外，在多模态场景中直接采用 discrete majority-vote reward（如 SQLM）效果不佳——早期训练阶段 Solver 输出高度不稳定，导致大量 zero-reward 更新和不稳定的优化过程。

EvoLMM 旨在在 **完全无监督** 的条件下（仅使用无标注图像），通过内部一致性信号实现 LMM 推理能力的自我提升。

---

## 3. 方法介绍

### 整体框架

从单一预训练 LMM（如 Qwen2.5-VL-7B）出发，通过 LoRA adapter 实例化两个轻量级角色：

- **Proposer** $\pi_\phi(q|x)$：给定 unlabeled image $x$，生成 visually grounded question $q$。
- **Solver** $\pi_\theta(y|x,q)$：对 question $q$ 采样 $N$ 条独立回答 $y_{1:N}$，计算经验答案分布 $p(a|x,q)$。

### Continuous Self-Consistency Reward（Solver 端）

对每条采样回答 $y_i$，reward 定义为：

$$r_i^{\text{sol}} = \big(p(y_i | x, q)\big)^\gamma \cdot \Big(1 - \lambda_{\text{len}} \cdot \max\{0,\, (w_i - \tau)/\tau\}\Big)$$

其中 $p(y_i|x,q)$ 是该回答在经验答案分布中的 agreement score，$\gamma \in (0,1]$ 控制 reward softness（较小值放大中间概率差异），$w_i$ 为回答中 `<answer>` 标签前的 token 数，$\tau$ 为目标简洁度阈值。此 continuous reward 与答案一致性平滑缩放，即使只有部分一致（如 2/5 一致）也能产生有意义的正梯度，避免了 discrete reward 的稀疏和不稳定问题。

### Entropy-Based Continuous Proposer Reward

计算 Solver 答案分布的 consensus entropy $H(x,q)$，Proposer reward 为 Gaussian band-pass function：

$$r^{\text{prop}} = \exp\!\left(-\frac{(H(x,q) - \mu_H)^2}{2\sigma_H^2}\right)$$

当 Solver 的不确定度处于中等范围时（既非过于确定/trivial，也非过于不确定/unsolvable），Proposer 获得最大 reward。这引导 Proposer 在 Solver 决策边界附近提问，形成 **adaptive curriculum**：随着 Solver 能力提升，Proposer 必须生成更难的问题以保持在 entropy band 内。

### 训练优化

两个角色均使用 REINFORCE policy gradient 优化，辅以：
- **Exponential moving-average baseline** $b_A$ 用于方差缩减。
- **Token-level KL regularization**（对 frozen reference model），通过 dynamic KL controller 自适应调整 $\beta_A$。
- Solver 每步更新；Proposer 每 $K=5$ 步更新一次。

---

## 4. 数据集

### 训练数据
仅使用 **raw images**（无 QA 对、无 metadata）。从 6 个数据集各采样约 1,000 张图像，总计约 6k 张：
- ChartQA、AI2D、InfographicVQA、PlotQA、ChartX、Geometry3K
- 涵盖图表、科学图解、几何图形等多样化视觉内容。

### 评估数据
8 个多模态数学与视觉推理 benchmark（使用官方 evaluation split）：
- **ChartQA**、**MathVista**、**MathVision**、**MathVerse**、**InfoGraphic-VQA**$_{\text{val}}$、**AI2D**、**ScienceQA**、**MMMU**$_{\text{val}}$

---

## 5. 评估指标与主要结果

### 评估指标
所有 benchmark 均采用其标准 accuracy 协议，使用 lmms-eval framework 统一推理。

### 主要结果（Table 1, Qwen2.5-VL-7B）

| Benchmark | Baseline | EvoLMM (Ours) | $\Delta$ |
|---|---|---|---|
| ChartQA | 84.00 | **86.70** | +2.7% |
| MathVista | 68.46 | **70.52** | +2.06% |
| MathVision | 23.91 | **24.81** | +0.9% |
| MathVerse | 43.78 | **44.88** | +1.1% |
| InfoGraphic-VQA | 80.44 | **81.06** | +0.62% |
| AI2D | 82.61 | **83.41** | +0.8% |
| ScienceQA | 88.30 | **89.50** | +1.2% |
| MMMU | 51.11 | **52.01** | +0.9% |

### 关键发现

1. **Continuous reward 显著优于 discrete reward**：Discrete majority-vote adaptation 仅带来 +0.3–0.6% 的微弱提升，甚至偶尔退化；continuous self-consistency reward 在所有 benchmark 上稳定提升 +2–3%。
2. **跨 backbone 泛化**（Table 3）：在 InternVL3-8B、Gemma3-12B-It、Llama-3.2-11B-Vision-Instruct 上应用相同框架，均获得约 +2–3% 的一致性提升，证明方法与 architecture 无关。
3. **模型规模 scaling**（Table 4）：72B 模型获得更强的绝对增益（如 ChartQA 88.20→91.04，MathVista 73.93→76.44）。
4. **LoRA 最优**（Table 2）：LoRA 在自演化设置中优于 QLoRA 和 full fine-tuning，后者因缺乏外部监督易过拟合并退化。
5. **Emergent curriculum**：训练过程中 Proposer 自动从生成简单问题转向中等难度问题，形成隐式课程学习。
