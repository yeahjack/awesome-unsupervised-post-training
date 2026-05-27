# DiSCTT: Consensus-Guided Self-Curriculum for Efficient Test-Time Adaptation in Reasoning

> **加入 Survey 时间：** 2026-03-11

> **论文元信息**
> - 作者: Mohammad Mahdi Moradi, Sudhir Mudur (Concordia University)
> - 时间: 2026年3月6日 (arXiv: 2603.05357)

| 属性 | 值 |
|---|---|
| Method | DiSCTT (Difficulty-aware Consensus-Guided Self-Curriculum Test-Time Adaptation) |
| Carrier | Policy Opt. |
| Regime | Test-time |
| Level | Semantic |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | original test set plus synthesized variants |
| 参数/状态持久性 Persistence | full parameter accumulate across curriculum cycles |
| 与推理关系 Inference Coupling | adapt on the evolving cohort, then re-infer in later cycles |
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

- **更新何时触发：** 更新在原始 test set 与其派生出来的 synthetic variants 上按 curriculum cycle 触发。
- **服务当前样本还是后续样本：** 当前 cycle 的更新主要服务后续 cycles 与后续评估，不是对单个样本 adapt 完就 reset。
- **参数/状态是否累积：** 参数在多个 curriculum cycles 中持续累积；cohort 会随反思、重分配或重采样而演化。
- **reset 边界：** 因此它属于 self-curriculum test-time adaptation，而不是简单的 static test-cohort RL。

## 1. UPT 归属理由

DiSCTT 属于 **Family III — Self-Generated Target Bootstrapping** 中的 **reasoning / plan / curriculum synthesis** 子类。其核心思想是在 test-time 阶段，利用模型自身的多次 sampling 结果之间的 consensus（共识度）来构建一个 **self-evolving curriculum**，动态地将 test-time 输入分为"简单"与"困难"两类，并分别施加 SFT（以 majority-agreed solution 作为 pseudo-label）和 RL（以 consensus-regularized reward 驱动探索）两种优化目标。

满足 UPT 的关键条件：
- **无需外部监督**: 所有训练信号均来自模型自身的生成结果（majority voting 作为 pseudo-label / reward），完全不依赖 ground-truth label 或外部 verifier。
- **合成目标驱动**: 主导产出物（dominant artifact）是一个基于 consensus 的合成 curriculum——它决定了哪些样本接受 SFT 巩固、哪些样本接受 RL 探索，且该 curriculum 随模型能力演化而周期性更新。
- **Policy Optimization 载体**: 通过 SFT 和 GRPO（Group Relative Policy Optimization）两种策略优化手段交替更新模型参数。

---

## 2. 解决的问题

现有 test-time adaptation 方法存在以下核心缺陷：

1. **Difficulty blindness（难度盲目性）**: TTRL、EVOL-RL 等方法对所有 test-time 输入统一施加单一优化目标（uniform RL 或 uniform SFT），忽略了推理问题的内在异质性。简单问题被施加不必要的高方差 RL 更新，困难问题又得不到足够的探索。
2. **训练不稳定**: 在无标签条件下统一施加 RL 会引入噪声梯度，导致性能波动甚至 performance collapse（见论文 Figure 2 中 RL-only 策略的 AMC 表现）。
3. **计算浪费**: 将高成本的 RL 更新均匀地应用于所有样本（包括模型已经能高置信度解决的简单样本），造成大量冗余计算。
4. **Token-level uncertainty 不适用于 reasoning**: 传统的 token-level entropy 或 confidence score 无法准确衡量多步推理的 epistemic uncertainty；错误往往只在 trajectory 层面才能暴露。

DiSCTT 的核心洞察：**有效的 test-time adaptation 需要根据 instance-level epistemic uncertainty 分配不同的学习目标**，而非"一刀切"。

---

## 3. 方法介绍

DiSCTT 包含三个核心模块：**Consensus-Based Difficulty Estimation**、**Dynamic Self-Curriculum Training**、以及 **Stabilized Label-Free RL with Structured Reward**。

### 3.1 Consensus-Based Difficulty Estimation（基于共识的难度估计）

对每个输入 $x_j$，使用当前策略 $\pi_\theta$ 独立采样 $M$ 条推理完成 $\{y_{j,1}, \ldots, y_{j,M}\}$，其中每条完成包含推理轨迹 $r_{j,i}$ 和最终答案 $a_{j,i}$。定义 empirical agreement ratio：

$$c_j = \frac{1}{M} \max_a \sum_{i=1}^{M} \mathbf{1}[a_{j,i} = a]$$

$c_j$ 越高说明模型对该问题的推理结果一致性越强（低 epistemic uncertainty）。使用固定阈值 $\rho$ 将数据集划分为：

- $\mathcal{D}_{\text{easy}} = \{x_j \mid c_j \ge \rho\}$（高共识，简单子集）
- $\mathcal{D}_{\text{hard}} = \{x_j \mid c_j < \rho\}$（低共识，困难子集）

**关键**: 这一划分是 **临时且策略依赖** 的。每隔 $K$ 步训练后，模型重新采样、重新计算 agreement ratio 并重新划分数据，形成 **self-evolving curriculum**——问题可以在 easy 和 hard 之间迁移，反映模型持续演化的能力。

### 3.2 Dynamic Self-Curriculum Training（动态自课程训练）

训练在 SFT 和 RL 两个阶段之间交替进行：

- **SFT 阶段**（针对 $\mathcal{D}_{\text{easy}}$）：使用 majority-agreed completion $y_j^*$ 作为 pseudo-label 进行监督微调：
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x_j, y_j^*) \sim \mathcal{D}_{\text{easy}}} \left[ \log \pi_\theta(y_j^* \mid x_j) \right]$$
  目的：以低方差方式巩固（consolidate）模型对高置信度问题的正确推理模式。

- **RL 阶段**（针对 $\mathcal{D}_{\text{hard}}$）：使用 GRPO 进行策略优化，鼓励模型在低共识困难问题上进行有结构的探索。

每个 curriculum cycle 包含 2 个 SFT epoch + 10 个 RL epoch，之后重新计算 consensus 并更新 curriculum 划分。

> **Figure 3 描述**（DiSCTT 整体框架图）：图示展示了一个闭环流程。左侧，模型对输入采样多条推理轨迹并进行 Consensus Assessment（共识评估），根据阈值将输入分流至两条路径——高共识输入经由 SFT Update 路径（使用 pseudo-label 进行监督微调），低共识输入经由 RL Update 路径（使用 RL Training 进行策略探索）。右侧有一个决策菱形"Step % K == 0?"，每隔 K 步触发 difficulty re-assessment，重新划分 easy/hard 子集，形成自我演化的 curriculum 循环。

### 3.3 Reinforcement Learning Objective（RL 目标：结构化 Reward）

对于 $\mathcal{D}_{\text{hard}}$ 中的每条完成 $y_i = (r_i, a_i)$，reward 由三个乘性组件构成：

$$R(y_i) = \mathbf{1}[a_i = a_{\text{maj}}(x)] \cdot (\alpha + \beta \cdot \text{JSD}_{\text{nov}}(r_i)) \cdot (\varepsilon + (1-\varepsilon) \cdot g_{\text{rel}}(r_i))$$

三个组件的逻辑顺序：

1. **Correctness Gate（正确性门控）**: $\mathbf{1}[a_i = a_{\text{maj}}(x)]$——只有最终答案与 majority answer 一致的轨迹才有非零 reward。majority answer 作为 self-consistency 诱导的 pseudo-label，无需 ground-truth。

2. **Population-Relative Novelty（群体相对新颖度）**: 通过 Jensen-Shannon Divergence 衡量当前推理轨迹与 majority-correct 轨迹群体的 token-level distribution 之间的偏离度：
$$\text{JSD}_{\text{nov}}(r_i) = \frac{1}{|T_i|} \sum_{t \in T_i} \text{JS}\left(p_\theta(\cdot|x, r_{i,<t}) \| \bar{p}_{\text{maj}}(\cdot|x, r_{<t})\right)$$
  其中 $\bar{p}_{\text{maj}}$ 是所有 majority-correct 轨迹的 predictive distribution 均值。JSD 有界且对称，避免了 KL divergence 的不稳定性。这鼓励模型在给出正确答案的同时，探索与主流推理路径不同的新推理策略。

3. **Relevance-Aware Semantic Gating（语义相关性门控）**: 使用预训练 sentence encoder（BAAI/bge-base-en-v1.5）计算推理过程中间步骤与输入 prompt 之间的语义相似度：
$$g_{\text{rel}}(r_i) = \frac{1}{n} \sum_{j=1}^n \text{clip}\left(\cos(e(s_{i,j}), e(x)), 0, 1\right)$$
  通过 affine factor $\varepsilon + (1-\varepsilon) \cdot g_{\text{rel}}(r_i)$ 调制 reward，当推理步骤偏离输入语义时降低 novelty reward，防止 spurious novelty（无关内容的偏离），同时 $\varepsilon > 0$ 确保不会完全归零导致 dead gradient。

### 3.4 算法整体流程（Algorithm 1）

```
Require: 数据集 D, 策略 πθ, 采样数 M, 更新间隔 K, 共识阈值 ρ
1: 初始化 θ ← θ₀
2: for training step t = 1, 2, ... do
3:   if t mod K = 0 then
4:     对每个输入 xⱼ ∈ D:
5:       采样 M 条推理完成, 计算 consensus score cⱼ
6:     Deasy = {xⱼ : cⱼ ≥ ρ}, Dhard = D \ Deasy
7:   end if
8:   SFT: 在 Deasy 上用 majority-agreed pseudo-label 做监督微调
9:   RL (GRPO): 在 Dhard 上用结构化 reward 做策略优化
10: end for
```

---

## 4. 数据集

论文在 **6 个推理 benchmark** 上进行评估，涵盖数学推理、多跳问答和知识推理：

| 数据集 | 类型 | 描述 |
|---|---|---|
| **AMC** | 数学推理 | 美国数学竞赛题 |
| **MATH-500** | 数学推理 | MATH 数据集的 500 题子集 |
| **AIME-2024** | 数学推理 | 美国数学邀请赛 2024 题目（高难度） |
| **GPQA** | 知识推理 | 研究生级别科学问答 |
| **HotpotQA** | 多跳问答 | 多跳推理问答数据集 |
| **MMLU** | 综合知识推理 | 大规模多任务语言理解 |

用于 OOD 泛化评估的额外数据集：**ARC-Challenge**、**HumanEval**。

实验涵盖 **6 个不同规模和家族的模型**：
- Qwen-2.5-0.5B-Instruct, LLaMA-3.2-1B-Instruct, Qwen-3-1.7B-Base, LLaMA-3.2-3B-Instruct, Qwen-3-4B-Base, Qwen-2.5-7B-Instruct

---

## 5. 评估指标与主要结果

### 评估指标
- **Mean Accuracy ± Standard Deviation**（多次独立运行取平均）
- **Training FLOPs** 和 **Wall-Clock Time**（计算效率）
- **OOD Accuracy**（跨域泛化能力）

### 5.1 主结果（Table 1）

DiSCTT 在所有 6 个 benchmark 和所有模型规模上均 **一致超越** Base model、TTRL 和 EVOL-RL，且方差更低：

| 模型 | 方法 | AMC | MATH-500 | AIME-2024 | GPQA | HotpotQA | MMLU | Avg. |
|---|---|---|---|---|---|---|---|---|
| Qwen-2.5-7B-Instruct | Base | 39.6 | 58.8 | 10.7 | 27.3 | 51.2 | 76.2 | 43.9 |
| | TTRL | 51.1 | 74.2 | 20.3 | 28.8 | 62.1 | 76.4 | 52.1 |
| | EVOL-RL | 55.0 | 73.4 | 26.3 | 29.3 | 66.1 | 77.9 | 54.6 |
| | **DiSCTT** | **59.5** | **82.2** | **29.6** | **34.9** | **73.7** | **83.3** | **60.6** |
| Qwen-3-4B-Base | Base | 38.6 | 51.4 | 10.0 | 27.3 | 39.4 | 69.5 | 39.3 |
| | TTRL | 46.6 | 60.6 | 17.1 | 28.8 | 61.5 | 72.4 | 47.8 |
| | EVOL-RL | 51.8 | 64.6 | 20.4 | 30.3 | 66.3 | 74.5 | 51.3 |
| | **DiSCTT** | **57.0** | **75.2** | **23.7** | **38.4** | **72.9** | **81.3** | **58.1** |

DiSCTT 相较 EVOL-RL 的平均提升约 **+5~7 个百分点**，在 Qwen-2.5-7B-Instruct 上平均达到 60.6%（vs. EVOL-RL 54.6%）。

### 5.2 Curriculum Routing 分布（Table 2）

不同数据集自然呈现不同的 SFT/RL 分配比例，验证了 difficulty-aware routing 的自适应性：

| 数据集 | SFT (%) | RL (%) |
|---|---|---|
| MMLU | 67.1 | 32.9 |
| HotpotQA | 58.3 | 41.7 |
| MATH-500 | 47.0 | 53.0 |
| AMC | 25.0 | 75.0 |
| GPQA | 28.8 | 71.2 |
| AIME-2024 | 3.3 | 96.7 |

越难的数据集（如 AIME-2024）几乎全部送入 RL 路径，而较简单的数据集（如 MMLU）大部分通过 SFT 巩固。

### 5.3 计算效率（Table 3）

DiSCTT 在保持更高精度的同时，大幅降低计算成本（**FLOPs 减少 30%~50%，wall-clock time 减少 40%~49%**）：

| 模型 | 数据集 | DiSCTT FLOPs | TTRL FLOPs | Cost Ratio |
|---|---|---|---|---|
| LLaMA-3.2-1B | MMLU | 47.08×10¹⁸ | 86.44×10¹⁸ | 0.544 (↓45.6%) |
| Qwen-3-4B | MATH-500 | 22.46×10¹⁸ | 32.78×10¹⁸ | 0.683 (↓31.7%) |
| Qwen-2.5-7B | AMC | 8.50×10¹⁸ | 14.05×10¹⁸ | 0.573 (↓43.7%) |

Wall-clock time 方面，例如 LLaMA-3.2-1B 在 MMLU 上从 TTRL 的 241 小时降至 DiSCTT 的 124 小时。

### 5.4 Difficulty-Level 分析（Figure 5）

在 MATH-500 上按 5 个难度级别比较三种训练策略：
- **SFT-only**: 对 Level 1-3 有改善但 Level 4-5 几乎无提升，后期出现 saturation 甚至退化。
- **RL-only (GRPO)**: 对所有级别均有提升但收敛慢，对高难度级别增益有限。
- **DiSCTT**: 所有难度级别均获得 **更快更强** 的提升，尤其在 Level 4-5 上表现显著优于两个单一策略 baseline，验证了 curriculum routing 的有效性。

### 5.5 Reward 消融（Table 4）

在 Qwen-3-4B-Base 上对 MATH-500 逐步添加 reward 组件：

| 配置 | Accuracy |
|---|---|
| Base model | 51.4% |
| + Correctness gate | 62.8% |
| + Population-relative novelty | 73.4% |
| + Relevance-aware semantic gating | **75.0%** |

每个组件都带来显著增益，验证了 reward 结构设计的合理性。

### 5.6 OOD 泛化（Figure 4）

- Qwen-3-1.7B-Base 在 AMC 上训练后，OOD 表现：ARC-Challenge +10.29, HumanEval +7.88, HotpotQA +29.14。
- LLaMA-3.2-3B-Instruct 在 MMLU（排除数学）上训练后：ARC-Challenge +16.97, GPQA +17.68。

DiSCTT 不仅不会导致灾难性遗忘，反而能 **显著提升** 跨域泛化能力，证明 difficulty-aware routing 有效防止了对 test-time 数据的过拟合。
