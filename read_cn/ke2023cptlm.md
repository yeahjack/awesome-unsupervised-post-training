# CPT-LM: Continual Pre-Training of Language Models

> **加入 Survey 时间：** 2026-03-11

- **taxonomy 中方法名：** CPT-LM
- **完整论文标题：** Continual Pre-Training of Language Models
- **作者：** Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, Bing Liu
- **发表：** ICLR 2023
- **Carrier：** Direct Opt. | **Regime：** training-time | **Level：** Token
- **Dominant artifact：** 直接在无标签文本上最小化 LM loss；训练信号为 token NLL / perplexity。

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

本文属于 **无监督后训练 (UPT)**，归入 **Family I: Prediction-Statistic Optimization**，具体为 **predictive likelihood minimization** 子类。理由：

- **无外部标签：** 整个 continual DAP-training 阶段仅使用无标签领域 corpus；不依赖人工标签、外部 verifier 或外部 AI 标签。
- **训练信号为 token-level NLL：** 核心训练目标是 Masked Language Model (MLM) loss，即在无标签文本上最小化 token-level negative log-likelihood。这是 language model 最原始的 intrinsic-statistic 信号 —— 直接从数据本身导出的 predictive likelihood。
- **辅助信号也是无监督的：** 额外的 contrastive loss 比较模型自身在不同 dropout mask 下的表示（full knowledge vs. previously learned knowledge），是对 model-generated content 的 self-comparison，未引入外部信号。
- **重要性计算使用 intrinsic proxy：** 用于识别通用知识重要性的 proxy KL-divergence loss 也仅利用模型自身在不同 dropout mask 下输出的差异（robustness）；不需要 pre-training 数据，也无需外部标注。

综上，每一项训练信号都来自 intrinsic statistics（token NLL / perplexity）与 model-generated content（表示的 self-comparison），完全符合 UPT 定义与 Family I 的 predictive likelihood minimization 范式。

---

## 2. 解决的问题

核心问题是 **continual domain-adaptive pre-training (continual DAP-training)**：如何让一个预训练 language model 在一系列无标签领域 corpus 上持续适应，且既不遗忘旧领域、也不损失知识迁移能力？具体上方法需要同时：

1. **克服 catastrophic forgetting (CF)：** 学习新领域时，不遗忘 (a) 旧领域知识和 (b) 原始 pretraining 的通用知识。
2. **实现 knowledge transfer (KT)：** 包括 forward transfer（旧领域知识助力新领域学习）与 backward transfer（学习新领域反过来提升旧领域性能）。
3. **不需要 domain-ID：** end-task fine-tuning 时无需知道样本的领域归属。

现有 continual-learning 方法（parameter-isolation、regularization、replay）都不适合：parameter-isolation 方法隔离子网络，阻止 end-task 利用全部知识；regularization 方法（如 EWC）以学习能力换取防遗忘；replay 方法需存储大量旧数据。

---

## 3. 方法介绍

提出的方法称为 **DAS (Continual DA-pre-training of LMs with Soft-masking)**，分两个主要阶段：

### 3.1 初始化：计算 unit 对通用知识的重要性

- 在 continual learning 开始前，计算 Transformer 中每个 unit（attention head 与 neuron）相对于通用知识的重要性 $I_l^{(0)}$。
- 由于无法访问原始 pre-training 数据，提出 **proxy KL-divergence loss**：将当前领域数据的一个子集以不同 dropout mask 两次送入 LM，计算两次输出的 KL divergence。梯度较大的 unit 对模型 robustness 更重要，即承载更多通用知识。

### 3.2 Continual Learning：Soft-masking + Contrastive Loss

对每个新领域 $t$，进行两步：

**(a) Domain Training：**

- **Soft-masking：** 使用累积重要性 $I_l^{(\leq t-1)}$（按 element-wise max 聚合），对梯度做 soft-mask：
  $$\hat{\nabla}_l = (1 - I_l^{(\leq t-1)}) \otimes \nabla_l$$
  重要 unit 的梯度被抑制（但不完全阻断）以防遗忘；不重要 unit 自由更新，促进 knowledge transfer。Soft-mask 仅作用于 backward pass，前向计算保留完整，使 end-task 仍可使用全网络。

- **Contrastive loss：** 对比 full-knowledge 表示 ($o^{\text{full}}$) 与 previously-learned-knowledge 表示 ($o^{\text{prev}}$，通过 importance-weighted masking 得到)，促使新领域学习与既有知识互补的表示：
  $$\mathcal{L}_{\text{DAP-train}} = \mathcal{L}_{\text{MLM}} + \lambda \mathcal{L}_{\text{contrast}}$$

**(b) Importance Computation：**

- 当前领域训练完成后，在当前领域数据上通过 gradient-based importance (Eq. 3) 计算 per-domain unit 重要性 $I_l^{(t)}$，供下一轮累积。

### 关键特征

- Soft-mask 值在 $[0,1]$ 连续（非二值），提供细粒度控制。
- 子网络不被隔离；所有知识在同一完整 LM 内累积。
- 不需要 replay memory，不需要 domain-ID。

---

## 4. 数据集

### DAP-training 用的 6 个无标签领域 corpus

| 来源 | Dataset | Size |
|--------|---------|------|
| Reviews | Yelp Restaurant | 758MB |
| Reviews | Amazon Phone | 724MB |
| Reviews | Amazon Camera | 319MB |
| Academic Papers | ACL Papers | 867MB |
| Academic Papers | AI Papers | 507MB |
| Academic Papers | PubMed Papers | 989MB |

### End-task 评测用的 6 个有监督分类数据集

| Dataset | Task | #Training | #Testing | #Classes |
|---------|------|-----------|----------|----------|
| Restaurant | Aspect Sentiment Classification (ASC) | 3,452 | 1,120 | 3 |
| Phone | Aspect Sentiment Classification (ASC) | 239 | 553 | 2 |
| Camera | Aspect Sentiment Classification (ASC) | 230 | 626 | 2 |
| ACL | Citation Intent Classification | 1,520 | 421 | 6 |
| AI | Relation Classification | 2,260 | 2,388 | 7 |
| PubMed | Chemical-protein Interaction Prediction (CHEMPORT) | 2,667 | 7,398 | 13 |

基础模型为 **RoBERTa-base**。Proxy KL-divergence 初始化阶段额外用 Wiki 数据作为 $D_0$ 的替身。

---

## 5. 评估指标与主要结果

### 指标

- **Macro-F1 (MF1)** 与 **Accuracy (Acc)** 在 6 个 end-task 分类数据集上（continual DAP-training 完成后，在每个领域上分别 fine-tune 与测试）。
- **Forgetting Rate (Forget R.)** 衡量 catastrophic forgetting 程度：$\text{Forget R.} = \frac{1}{t-1}\sum_{k=1}^{t-1}(A_{k,k} - A_{t,k})$，其中 $A_{k,k}$ 为刚训练完领域 $k$ 后该领域的 end-task accuracy，$A_{t,k}$ 为训练完所有领域后领域 $k$ 的 accuracy。负值表示 positive transfer。

### 主要结果（Table 2，5 个 random seed 平均）

| Model | Avg MF1 | Avg Acc | Forget R. (MF1) | Forget R. (Acc) |
|-------|---------|---------|-----------------|-----------------|
| RoBERTa (无 DAP) | 77.25 | — | — | — |
| Pool (所有领域合并) | 80.63 | 90.83 | — | — |
| NCL (朴素 continual) | 80.70 | 76.66 | 1.14 | 1.05 |
| DEMIX | 74.70 | — | 0.15 | — |
| EWC | 74.84 | — | 0.02 | -0.01 |
| DER++ | 75.51 | — | 2.36 | 1.53 |
| **DAS (本文)** | **81.91** | **77.93** | **-1.09** | **-0.60** |

### 关键发现

1. **DAS 在所有 baseline 中平均性能最高**（MF1 81.91），同时获得最强的 knowledge transfer（forgetting rate 为负 -1.09，即 positive transfer）。
2. **DAS 同时处理 CF 与 KT：** 防遗忘聚焦的方法（KD、EWC、DER++）牺牲学习能力；transfer 聚焦的方法（BCL、CLASSIC、DEMIX）仍会遗忘。DAS 两者兼得。
3. **Soft-masking 优于 parameter-isolation：** HAT 等二值 mask / 子网络方法表现较差，因为 end-task 无法利用完整网络的知识。
4. **Proxy KL-divergence 有效：** 相比用 Wiki 数据 + MLM loss 估算通用知识重要性，proxy KL-divergence 更优 —— 它测量 robustness，与领域无关，更能反映通用知识。
5. **消融确认每个组件都有贡献：** 去掉 initialization、soft-masking 或 contrastive learning 均会降低性能。
