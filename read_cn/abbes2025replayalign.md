# ReplayAlign CPT — Revisiting Replay and Gradient Alignment for Continual Pretraining of Large Language Models

> **加入 Survey 时间：** 2026-03-11

> **论文**: Revisiting Replay and Gradient Alignment for Continual Pretraining of Large Language Models
> **作者**: Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Thérien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish
> **机构**: Université de Montréal, Mila, IBM Research, Fujitsu Research, Polytechnique Montréal
> **发表**: 4th Conference on Lifelong Learning Agents (CoLLAs), 2025
> **ArXiv**: 2508.01908
> **Method**: ReplayAlign CPT | **Carrier**: Direct Opt. | **Regime**: training-time | **Level**: Token

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | unlabeled corpus / continual training batch |
| 参数/状态持久性 Persistence | full parameter accumulate across corpus stages; no sample-level reset |
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

- **更新何时触发：** 更新在 deployment 前的 continued pretraining / CPT 阶段按 corpus 与 training batch 持续触发，不由单个测试样本触发。
- **服务当前样本还是后续样本：** 当前 batch 的更新服务后续训练批次与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整个 corpus stream / stage 序列中持续累积；若有阶段切换，也只是训练阶段切换，不是 sample-level reset。
- **reset 边界：** 因此它回答的是 deployment 前的 adaptation schedule，而不是 deployment 期间的 online TTA。

## 1. UPT 归属理由

本文属于 **Family I (Prediction-Statistic Optimization)**，具体为 **predictive likelihood minimization** 子类。

- **核心训练目标**：标准的 language modeling cross-entropy loss（即 next-token prediction），不依赖任何外部标注、人类偏好或外部 verifier。
- **Replay 机制**：从磁盘上的无限 replay buffer 中按比例 α 采样旧数据与新数据混合，直接优化 LM loss。这是一种纯粹基于 intrinsic statistics（模型自身的 predictive likelihood）的优化策略。
- **Gradient Alignment（Reptile meta-update）**：通过 Reptile 一阶 meta-learning 进行周期性参数插值（每 k=500 步），正则化目标为最大化新旧 batch 梯度的 dot product，促进 transfer、减少 interference。该正则化信号完全来自模型自身的梯度统计量，无需外部监督。
- **无外部信号**：整个方法不使用 ground truth labels、human feedback、external AI labels 或 external verifier，完全符合 UPT 定义。

---

## 2. 解决的问题

**Continual pre-training (CPT) 中的灾难性遗忘与 stability-plasticity 困境。**

- LLM 需要持续在新数据/新领域上更新，但从头重新 pre-train 成本极高。Continual pre-training 是高效替代方案，但引入新数据后模型容易遗忘旧知识（catastrophic forgetting）。
- 现有 continual learning 文献中，experience replay 和 gradient alignment 方法虽然在小模型上效果良好，但**尚未在现代 LLM 规模下被系统评估**。
- 关键问题：在固定 compute budget 下，增加 replay 比例、增大模型、还是添加 gradient alignment，哪种策略更高效？

---

## 3. 方法介绍

### 3.1 Experience Replay

- 维护一个基于磁盘的"无限" replay buffer，存储所有已见过的训练数据。
- 每个 training step，batch 中 α 比例的 sample 从 replay buffer 随机采样，其余来自当前新数据流 p(x|t)。
- 实验中测试 α ∈ {0, 0.25, 0.5}（0%、25%、50% replay）。
- 磁盘实现使用异步 prefetching 和 caching，兼容 Megatron/NeoX 框架，避免 GPU/RAM 瓶颈。

### 3.2 Meta-Experience Replay (MER)

- 基于 Riemer et al. (2019a) 的方法，引入 **Reptile 一阶 meta-learning** 实现 gradient alignment。
- Reptile 近似优化目标：

  $$\arg\min_{\theta_t} \mathbb{E}_{B_1,...,B_k}\left[2\sum_{i=1}^{k}L(B_i) - \sum_{j=1}^{i-1}\beta\frac{\partial L(B_i)}{\partial \theta_t}\cdot\frac{\partial L(B_j)}{\partial \theta_t}\right]$$

  第二项促进不同 batch 之间梯度的对齐（最大化 dot product），从而促进正向 transfer、减少 interference。

- **实现**：每 k=500 步执行一次参数插值 θ_t ← θ_{t-k} + ε(θ_t − θ_{t-k})，其中 ε=0.1。
- 计算开销极小：每 500 个 batch 仅增加约 3 倍模型大小的 FLOPs（相对于梯度更新成本可忽略不计）。

### 3.3 完整方法：Replay + Reptile (MER)

- 将 replay buffer 与 Reptile meta-update 结合：replay 的 batch 序列中同时包含 α% 旧数据，Reptile 在此基础上周期性对齐梯度。
- 两者协同作用：replay 提供旧数据的直接优化信号，gradient alignment 促进模型学到在新旧数据上都泛化的参数。

---

## 4. 数据集

### 训练数据（Sequential CPT，每个 task 100B tokens）

| Task | 数据集 | 语言 | Tokens |
|------|--------|------|--------|
| A | DCLM-Baseline（DataComp-LM，CommonCrawl 子集） | English | 100B |
| B | OSCAR（CommonCrawl 提取） | French | 100B |
| C | OSCAR | German | 100B |
| D（扩展） | Aloui et al. (2024) | Arabic | 100B |
| E（扩展） | CommonCrawl-derived corpus (Hattori, 2024) | Japanese | 100B |

- 3-task 设置：English → French → German
- 5-task 扩展设置：English → French → German → Arabic → Japanese

### 模型

- **Spectra LLM suite**（Llama 架构族）：99M, 560M, 1B, 6B 四种规模
- AdamW optimizer，cosine learning rate schedule，357 步 warmup，batch size 4096
- 每阶段训练 1 epoch，阶段间无 gradient/optimizer 重置

---

## 5. 评估指标与主要结果

### 评估指标

- **Forgetting Score**：当前 validation loss 相对于历史最佳 validation loss 的增量（越低越好）
- **Retained Loss**：训练结束后所有 task 的平均 validation cross-entropy loss（衡量 stability）
- **Learned Loss**：每个 task 训练完成后立即在该 task 上的 validation loss（衡量 plasticity）
- **Downstream Performance**：HellaSwag、PiQA、PubMedQA 等 benchmark 上的 zero-shot 表现

### 主要结果

**Q1: Experience Replay 的效果**
- Replay 显著降低遗忘：560M 模型无 replay 的最终 validation loss ≈ 3.3，50% replay 稳定在 ≈ 3.0。
- 560M + 50% replay 的效果接近 1B 无 replay 模型，表明 replay 在 compute 效率上优于单纯增大模型。

**Q2: Gradient Alignment 与 Replay 的协同效应**
- 50% replay + Reptile 在所有模型规模上一致取得最低 average forgetting score。
- 560M 模型：25% replay + Reptile 的 downstream 平均分（67.5）超过 560M joint training baseline（67.33），也超过纯 replay（66.4）和纯 Reptile（65.4）。
- 6B 模型效果最明显：joint training baseline 平均 72.6，25% replay + Reptile 达 76.8，50% replay + Reptile 达 77.1，**超越 joint training**。
- 5-task 扩展实验中，50% replay + Reptile 甚至实现**负遗忘**（backward transfer），表明方法在更长任务序列中依然有效。

**Q3: Gradient Alignment 促进泛化**
- Reptile 的引入使 task-specific validation loss 曲线更接近 joint training baseline（尤其在 French/German 上），表明 gradient alignment 有效促进跨 task 泛化。

**Q4: Computational Scaling**
- Stability 和 plasticity 均呈 inverse power law 随 FLOPs/token 扩展。
- 25% replay 仅增加 1.33× FLOPs/token，效率优于 50% replay（2× FLOPs/token）。
- Reptile 的增益几乎"免费"——在 stability 和 plasticity 的 scaling curve 上均带来一致提升且不增加显著计算开销。
- 25% replay + Reptile 的 scaling 趋势甚至优于 25% replay without Reptile，且差距随模型增大而扩大。
