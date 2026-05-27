# Temporal SRLM: Temporal Self-Rewarding Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future
**Authors:** Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang
**Institutions:** Peking University, Tsinghua University, National University of Singapore, Southeast University, NC State University, University of Michigan, Beijing Institute of Technology, Central South University, Westlake University
**ArXiv:** 2025
**Date:** 2025

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Temporal SRLM | Pref. Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch with self-generated responses / judgments |
| 参数/状态持久性 Persistence | full parameter accumulate across self-rewarding iterations |
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

- **更新何时触发：** 更新在 deployment 前的 self-rewarding / judge bootstrapping 循环里触发，通常先生成 responses，再生成 judgments 或 preference pairs。
- **服务当前样本还是后续样本：** 当前批次生成的 response / judgment 主要服务下一轮训练与最终部署模型，而不是服务某个测试样本的即时推理。
- **参数/状态是否累积：** 参数在多轮 self-rewarding 迭代中持续累积；若论文使用 actor / judge / meta-judge 角色切换，也仍属同一离线训练闭环。
- **reset 边界：** 因此它的 adaptation 时机是 offline iterative bootstrapping，而非 deployment-time TTA。

## 1. UPT 归属理由

**Family IV — Internal Evaluator Bootstrapping (temporal self-rewarding)**

Temporal Self-Rewarding 属于 Internal Evaluator Bootstrapping 家族中的 preference optimization 分支，具体归类为 self-rewarding 范式的改进。其核心机制是：模型同时担任 **生成者** 和 **评估者** 的双重角色（LLM-as-a-Judge），在迭代训练中自行生成 responses 并通过自身打分构建 preference pairs，然后利用 DPO 进行 preference optimization。该方法的关键创新在于引入 **temporal decoupling**，即通过 past model（M₀）锚定 rejected responses、通过 future model（Mf）提升 chosen responses，从而在不依赖外部 reward model 或人工标注的前提下，维持 self-rewarding 学习信号的有效性。这本质上是对 self-rewarding 框架的时间维度扩展，通过 past–future 的质量差异代替 same-model 的质量差异，确保 bootstrapping 循环不会因为 chosen 和 rejected 趋同而坍缩。

---

## 2. 解决的问题

### 核心问题：Self-Rewarding 中 Chosen-Rejected 质量差距的坍缩

在标准 Self-Rewarding Language Models 中，模型在每轮迭代中同时生成 chosen 和 rejected responses，并由同一模型打分。随着训练迭代进行，模型能力不断提升，导致以下连锁问题：

1. **质量趋同（Quality Convergence）**：随着模型 M_i 能力增强，其最优输出（chosen y_w）和最差输出（rejected y_l）的质量差距不断缩小。实验观察到 Llama 3.1-8B 上 chosen-rejected 的 score gap 在 4 轮迭代中缩小了 **9 倍**。

2. **表征坍缩（Representational Collapse）**：在 latent space 中，chosen 和 rejected 的 hidden representations h_w 和 h_l 逐渐趋近（‖h_w − h_l‖ → 0），cosine similarity 持续增大。

3. **梯度消失（Gradient Vanishing）**：根据 Theorem 1，DPO 梯度中的 Directional Guidance 项 ‖∇_θ log π(y_w|x) − ∇_θ log π(y_l|x)‖ ≤ C · ‖h_w − h_l‖。当表征趋同时，该项趋近于 0，导致 reward signal r̂ → 0，Adaptive Weighting 项 (1 − σ(r̂)) → 0.5，整体 DPO 梯度消失，训练陷入停滞。

---

## 3. 方法介绍

### 3.1 概览

Temporal Self-Rewarding 提出通过 **temporal decoupling** 将 chosen 和 rejected responses 的生成解耦到不同时间步的模型上，包含两个关键阶段：

- **Phase 1: Anchored Rejection** — 用 past model（初始模型 M₀）锚定 rejected responses
- **Phase 2: Future-Guided Chosen** — 用临时训练的 future model（Mf）提升 chosen responses

### 3.2 SFT Model 初始化

从预训练基座模型 M_b 出发，通过 supervised fine-tuning 建立双重能力：

$$M_0 = \text{SFT}(M_b, \text{IFT} \cup \text{EFT})$$

其中 IFT（Instruction Fine-Tuning）数据训练 response generation 能力，EFT（Evaluation Fine-Tuning）数据训练 quality assessment 能力（包含 judge explanations）。

### 3.3 Phase 1: Anchored Rejection（锚定拒绝）

对于每个 prompt p ∈ p_i：

1. **双模型生成**：用当前模型 M_i 和初始模型 M₀ 各生成 K=7 个 responses：
   - r_i = {r_i^1, ..., r_i^K}（来自 M_i）
   - r_0 = {r_0^1, ..., r_0^K}（来自 M₀）

2. **当前模型打分**：M_i 对所有 responses 打分：
   - s_i = {s_i^1, ..., s_i^K}（M_i 自身的 responses）
   - s_0 = {s_0^1, ..., s_0^K}（M₀ 的 responses）

3. **Chosen 选取**：从 M_i 中选最高分 response：chosen ← r_i^{argmax s_i}

4. **Rejected 选取**：优先从 M₀ 中选最低分 response（如果 min(s_0) < min(s_i)），否则从 M_i 中选最低分：rejected ← r_0^{argmin s_0}

5. **有效性过滤**：仅当 s_chosen > s_rejected 时加入数据集 D₁

6. **训练 Future Model**：

$$M_f = \text{DPO}(M_i, D_1)$$

### 3.4 Phase 2: Future-Guided Chosen（未来引导选择）

对于每个 prompt p ∈ p_i：

1. **Future 模型生成**：用 M_f 生成 K 个 responses：r_f = {r_f^1, ..., r_f^K}

2. **当前模型打分**：M_i 对 M_f 的 responses 打分：s_f = {s_f^1, ..., s_f^K}

3. **Chosen 升级**：如果 max(s_f) > max(s_i)，则用 M_f 的最佳 response 替换 chosen：
   - chosen ← r_f^{argmax s_f}（来自 future model）
   - 否则保留 r_i^{argmax s_i}（来自当前 model）

4. **Rejected 复用**：使用 Phase 1 中同一 prompt 的 rejected response

5. **有效性过滤**：仅当 s_chosen > s_rejected 时加入数据集 D₂

6. **训练下一轮模型**：

$$M_{i+1} = \text{DPO}(M_i, D_2)$$

### 3.5 理论分析

**Theorem 1 (Directional Guidance 的界)**：设 π_θ 为一个通过 latent representation h 生成 response y 的模型。对于任意 chosen-rejected pair (y_w, y_l)，DPO directional guidance 项的范数有界：

$$\|\nabla_\theta \log \pi_\theta^y(y_w|x) - \nabla_\theta \log \pi_\theta^y(y_l|x)\| \leq C_{h_w, h_l} \cdot \|h_w - h_l\|$$

其中 $C_{h_w, h_l}$ 是关于 h_w 和 h_l 的有限常数（由梯度函数的 Jacobian 在线段 {λh_w + (1−λ)h_l} 上的 supremum 决定）。

**推论**：在标准 Self-Rewarding 中，‖h_w − h_l‖ → 0（表征趋同），该定理直接证明了 DPO 梯度消失。Temporal Self-Rewarding 通过将 y_l 锚定到 M₀（表征质量始终较低）、将 y_w 提升到 M_f（表征质量更高），人为维持 ‖h_w − h_l‖ 的大小，保证梯度信号稳定。

### 3.6 迭代效率

Temporal Self-Rewarding 仅需 **2 轮迭代**（iter0 + iter1）即可超越标准 Self-Rewarding 的 **4 轮迭代**（iter0–iter3），尽管每轮额外训练一个 temporary future model M_f。总计算量与 Self-Rewarding 的 5 轮迭代相当（公平比较）。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 对话 / 指令跟随 | Open Assistant (oasst1 + oasst2) | 提取英文样本，去除 null-score 后约 5,000 条；用于 IFT seed data |
| 多领域评估 | UltraFeedback | 带评分的多 response 数据；去除 2,000 最高方差条目后，取 top 25,000 高分样本合并入 IFT |
| IFT Seed Data | 合并集 | 从合并后的 Open Assistant + UltraFeedback 中随机采样 5,000 条 question-answer pairs |
| EFT Seed Data | UltraFeedback 高方差子集 | 取 2,000 最高方差样本，由 GPT-4o 评分，仅保留模型打分顺序与人工排名一致的 1,871 条；包含 judge explanations |
| 迭代优化数据 | 混合集剩余部分 | 排除 5,000 IFT 样本后的 20,000 条，均分为 5 部分（每部分 4,000 条），每轮迭代使用一部分 |
| 评估 | AlpacaEval 2.0 | 指令跟随评估；pairwise comparison with GPT-4 Preview |
| 评估 | Arena-Hard-v0.1 | 高难度对话评估；pairwise comparison with GPT-4-0314 |
| 评估 | MT-Bench | 多轮对话评估；GPT-4o direct scoring |
| OOD 评估 | ARC-Challenge, GSM8K, TruthfulQA, HumanEval | 科学推理、数学推理、事实问答、代码生成 |

---

## 5. 评估指标与主要结果

### 评估指标

| 指标 | Benchmark | 说明 |
|------|-----------|------|
| LC Win Rate (%) | AlpacaEval 2.0 | Length-controlled win rate，控制响应长度偏差 |
| Win Rate (%) | AlpacaEval 2.0 | 直接胜率 |
| Score (%) | Arena-Hard-v0.1 | 对 GPT-4-0314 的相对胜率得分 |
| 1st / 2nd / Avg | MT-Bench | 第一轮 / 第二轮 / 平均对话质量评分 |
| Accuracy | ARC, GSM8K, TruthfulQA, HumanEval | 各 NLP benchmark 上的准确率 |

### 主要结果

#### Llama 3.1-8B 主实验 (Table 1)

| 方法 | 最佳迭代 | AlpacaEval LC Win | AlpacaEval Win | Arena-Hard Score | MT-Bench Avg |
|------|----------|-------------------|----------------|------------------|--------------|
| SFT Model | - | 8.73% | 5.96% | 6.3% | 4.81 |
| Rejection Sampling (best) | iter1 | 9.58% | 7.33% | 6.6% | 5.08 |
| SPIN (best) | iter3 | 8.93% | 6.83% | 4.7% | 5.11 |
| SPIN-Fair (best) | iter3 | 9.82% | 7.20% | 4.7% | 5.09 |
| Self-Rewarding (best) | iter3 | **19.92%** | **19.69%** | 9.4% | 5.74 |
| **Temporal SR (best)** | **iter1** | **27.94%†** | **29.44%†** | **14.6%†** | **5.89†** |

Temporal SR 在所有指标上均为最优，且仅用 2 轮迭代（iter0 + iter1）便显著超越 Self-Rewarding 的 4 轮迭代。

#### 跨模型家族泛化 (Table 3)

| 模型 | 方法 | AlpacaEval LC Win | AlpacaEval Win | Arena-Hard | MT-Bench |
|------|------|-------------------|----------------|------------|----------|
| Llama 3.2-3B | SR best | 3.37% | 3.42% | 2.3% | 4.03 |
| Llama 3.2-3B | **TSR best** | **4.79%** | **8.20%** | **2.9%** | **4.32** |
| Llama 3.1-8B | SR best | 19.92% | 19.69% | 8.8% | 5.66 |
| Llama 3.1-8B | **TSR best** | **27.94%** | **29.44%** | **14.6%** | **5.89** |
| Llama 3.1-70B | SR best | 35.57% | 32.91% | 38.9% | 6.93 |
| Llama 3.1-70B | **TSR best** | **38.70%** | **33.66%** | **40.1%** | **6.98** |
| Qwen 2.5-7B | SR best | 21.53% | 18.14% | 21.5% | 6.09 |
| Qwen 2.5-7B | **TSR best** | **34.01%** | **35.90%** | **34.4%** | **6.29** |
| Mistral 7B | SR best | 25.48% | 27.58% | 12.8% | 5.68 |
| Mistral 7B | **TSR best** | **32.11%** | **35.16%** | **15.7%** | **5.76** |

#### Out-of-Distribution NLP 评估 (Table 4, Llama 3.1-8B)

| 方法 | ARC | GSM8K | TruthfulQA | HumanEval |
|------|-----|-------|------------|-----------|
| SFT | 0.531 | 0.530 | 0.505 | 0.220 |
| SR iter3 | 0.538 | 0.550 | 0.518 | 0.238 |
| **TSR iter1** | **0.549** | **0.563** | **0.544** | **0.262** |

### 关键发现

1. **Score Gap 坍缩的定量证据**：在标准 Self-Rewarding 中，Llama 3.1-8B 的 chosen-rejected score gap 在 4 轮迭代中缩小了 9 倍，同时 chosen-rejected pairs 的 cosine similarity（基于最后一层 features）持续上升，验证了表征坍缩现象。

2. **Temporal Decoupling 有效维持学习信号**：Temporal SR 在迭代过程中保持了 chosen 和 rejected 之间稳定的质量差距，score difference 不出现标准 Self-Rewarding 中的急剧下降。

3. **Past Model 贡献 > Future Model**：消融实验（Table 2）显示，仅使用 Past component（即 Anchored Rejection without Future-Guided Chosen）已能在所有指标上显著超越 Self-Rewarding baseline。原因在于随着模型迭代优化，生成的 responses 分数普遍较高，对 rejected 进行 past anchoring 能更有效地放大 chosen-rejected 对比度。Future model 提供互补但相对次要的增益。

4. **Judge Model 鲁棒性**：无论使用 Self-Judge、AutoJ-6B 还是 AutoJ-13B 作为 judge，Temporal SR 均一致优于 Self-Rewarding，表明方法的优势不依赖于特定的 judge 选择。使用更强的外部 judge（AutoJ-13B）时，两种方法均有显著提升。

5. **跨模型家族和规模的泛化**：在 Llama (3B/8B/70B)、Qwen 2.5-7B、Mistral 7B 上均持续有效，且在 Qwen 和 Mistral 上的相对提升尤为显著（Qwen: 21.53% → 34.01% LC Win）。

6. **更少迭代、更优性能**：Temporal SR 以 2 轮迭代的计算量（加上额外的 future model 训练），达到标准 Self-Rewarding 4 轮迭代无法达到的性能水平。

7. **OOD 泛化**：即使训练主要针对 instruction-following，Temporal SR 在 reasoning (ARC, GSM8K)、factual QA (TruthfulQA)、code generation (HumanEval) 等 OOD 任务上也实现了显著提升，表明 preference learning 的改进具有广泛的迁移效应。
