# Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models

> **加入 Survey 时间：** 2026-03-11

> arXiv: 2508.00410v2, Oct 2025
> Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, Bo Han
> HKBU / SJTU / Shanghai Innovation Institute

| Method | Carrier | Regime | Level |
|---|---|---|---|
| Co-rewarding | Policy Opt. | training-time | Traj. |

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

**Family II — Sample-Relation Supervision (trajectory-level consistency → contrastive agreement)**

Co-rewarding 完全不依赖外部 ground-truth label、人类标注或外部 reward model。其奖励信号来源于模型自身生成的 trajectory 之间的关系型内部监督：

- **Co-rewarding-I (data side)**：对同一问题生成语义等价的 paraphrase 版本，分别采样多条 rollout，通过 majority voting 互相生成 pseudo-label，再以 contrastive agreement（原始问题输出与 rephrased 问题输出之间的交叉一致性）作为奖励信号。这是典型的 trajectory-level contrastive agreement。
- **Co-rewarding-II (model side)**：引入 EMA (exponential moving average) 更新的 teacher 模型，teacher 的 rollout 通过 majority voting 产生 pseudo-label 来监督 student (online policy)。teacher 与 student 在时间维度上解耦，形成 temporal invariance 的 relational supervision。

两种实例化方式均通过模型内部多条 trajectory 之间的一致性/不一致性来构造奖励，无需任何外部验证器，完全符合 UPT 定义。

---

## 2. 解决的问题

现有 self-rewarding RL 方法（如 Self-Certainty、Entropy minimization、Majority-Voting）存在严重的 **training collapse** 问题：

- **单视角自洽幻觉 (self-consistent illusion)**：奖励信号完全由当前 policy 从单一视角生成，模型容易找到 trivial solution 来 hack reward（如 Entropy 方法中模型将概率集中在少量 token 上产生重复输出；Majority-Voting 中模型收敛到一致但错误的答案）。
- **Reward hacking**：训练初期性能快速上升，但很快 collapse，无法持续提升。
- **对 ground-truth label 的依赖**：传统 RLVR 依赖高质量外部标签，限制了 scaling。

Co-rewarding 的核心思想是从 self-supervised learning（如 SimCLR, BYOL, DINO）借鉴 **complementary supervision**（互补监督）理念，通过引入"另一个视角"的监督信号来打破单视角反馈回路，从而实现稳定的 self-supervised RL 训练。

---

## 3. 方法介绍

### 3.1 核心哲学：超越单视角的 invariance

真正的推理能力应当在不同数据视角和时间演化过程中保持不变 (invariance)。Co-rewarding 将 self-supervised RL 的基础从"可疑的单视角反馈"转移到"跨视角不变性"上。

### 3.2 两种实例化

#### Co-rewarding-I：Data Side（数据侧 analogy-invariance）

1. 对每个原始问题 x，利用 LLM（Qwen3-32B）生成语义等价的 rephrased 版本 x'
2. 分别对 x 和 x' 采样 G 条 rollout
3. 对 x 的 rollout 通过 majority voting 得到 pseudo-label y_v，对 x' 的 rollout 同理得到 y'_v
4. **交叉监督**：用 y'_v 作为 x 的 rollout 的奖励标准，用 y_v 作为 x' 的 rollout 的奖励标准
5. 基于 GRPO 框架计算 advantage 并优化

核心目标函数：J = J_original(θ) + J_rephrased(θ)，两部分通过交叉 pseudo-label 互相监督。

#### Co-rewarding-II：Model Side（模型侧 temporal invariance）

1. 维护一个 EMA 更新的 reference teacher 模型 π_ref
2. Teacher 对问题采样 rollout，通过 majority voting 产生 pseudo-label
3. 用 teacher 的 pseudo-label 作为 student (online policy) 的奖励信号
4. Teacher 通过 EMA 动态更新：π_ref^(k) ← α^(k) · π_ref^(k-1) + (1 - α^(k)) · π_θ_old^(k)
5. α 从 α_start=0.99 按 cosine annealing 调度到 α_end=0.9999（初期快速更新，后期缓慢更新）

这种设计类似 self-distillation（BYOL/DINO 风格），slowly-updated teacher 监督 fast-moving student，打破 on-policy 反馈回路。

### 3.3 实现细节

- 基于 GRPO 优化框架，使用 VeRL 实现
- 4 × H100-80GB GPU
- Batch size 128，每个问题采样 G = G̃ = 8 条 rollout
- AdamW optimizer，学习率 3×10⁻⁶

---

## 4. 数据集

### 训练集
- **MATH** (Hendrycks et al., 2021)：7,500 道数学题
- **DAPO-Math-17k** (Yu et al., 2025)：约 14.1k 道数学题（英文版 DAPO-14k）
- **OpenRS** (Dang & Ngo, 2025)：7,000 道数学题

### 评估 Benchmark
- **数学推理**：MATH500, GSM8K, AMC
- **代码生成**：LiveCodeBench (release_v6), CRUX
- **指令遵循**：IFEval
- **多任务能力**：MMLU-Pro

### 基座模型
- Qwen2.5 系列：3B, 7B
- Qwen3 系列：1.7B-Base, 4B-Base, 8B-Base
- Llama3 系列：Llama-3.2-3B-Instruct

---

## 5. 评估指标与主要结果

### 评估指标
- **Pass@1**：主要指标，衡量一次采样即正确的概率

### 主要结果

**对比 self-rewarding baselines 的优势**：
- Co-rewarding-I 在三个数学 benchmark 上相对最佳 baseline 平均提升 **+3.46%**（Table 1, MATH 训练集）
- Co-rewarding-II 在 DAPO-14k 训练集上平均相对提升 **+7.29%**（Table 2）
- 平均超越所有 self-rewarding baselines **+3.31%**

**超越 GT-Reward 的表现**：
- 在 GSM8K 上，两种 Co-rewarding 均超越使用 ground-truth label 的 RLVR，合计相对提升 +2.94%（Table 1）
- Co-rewarding-II + Qwen3-8B-Base 在 GSM8K 上达到 **94.01% Pass@1**，显著高于 GT-Reward

**训练稳定性**：
- 三种 self-rewarding baselines（Self-Certainty, Entropy, Majority-Voting）均出现 training collapse
- Co-rewarding-I 在 MATH 数据集上稳定，但在 DAPO-14k 上仍有 collapse（因 DAPO-14k 问题缺少丰富背景描述，rephrasing 效果有限）
- **Co-rewarding-II 在所有数据集上一致保持稳定**，无 collapse

**泛化能力**：
- 仅在数学数据上训练，但在代码 benchmark（LiveCodeBench, CRUX）上也有提升
- 在 MMLU-Pro 14 个类别中 12 个超越 self-rewarding baselines
- IFEval 指令遵循能力得到保持

### Ablation 结果（Table 3）
- Co-rewarding-I：去掉 cross-supervision（仅用 original 或仅用 rephrased）性能均下降，证明交叉监督是关键
- Co-rewarding-II：去掉 EMA 更新的 reference teacher 导致明显退化，证明 teacher 动态更新对 pseudo-label 质量至关重要
