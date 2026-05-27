# ScPO: Self-Consistency Preference Optimization

> **加入 Survey 时间：** 2026-03-11

**Paper:** Self-Consistency Preference Optimization
**Authors:** Archiki Prasad (UNC Chapel Hill), Weizhe Yuan (Meta FAIR / NYU), Richard Yuanzhe Pang (Meta FAIR), Jing Xu (Meta FAIR), Maryam Fazel-Zarandi (Meta FAIR), Mohit Bansal (UNC Chapel Hill), Sainbayar Sukhbaatar (Meta FAIR / NYU), Jason Weston (Meta FAIR), Jane Yu (Meta FAIR)
**Venue:** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning)
**Citation:** prasad2025scpo

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| ScPO | Pref. Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / synthetic preference pair batch |
| 参数/状态持久性 Persistence | full parameter accumulate across epochs or iterations |
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

- **更新何时触发：** 更新在 deployment 前的 preference optimization 阶段触发，基本单位是模型自生成的 chosen / rejected pairs。
- **服务当前样本还是后续样本：** 当前 pair batch 的更新服务后续训练与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在训练 epoch / iteration 中持续累积，不做 sample-level reset。
- **reset 边界：** 因此它更像 offline pair-based post-training，而不是 online TTA。

## 1. UPT 归属理由

**Family III — Self-Generated Target Bootstrapping → Preference Optimization → Internally Generated Preference Pairs**

ScPO 属于无监督后训练（UPT），原因如下：

1. **无需外部标注**：整个训练流程不依赖 gold answers、人工标注或外部 reward model。ScPO 仅使用 unlabeled queries（无答案的数学/逻辑问题）作为种子数据，所有监督信号均来自模型自身。
2. **内部生成 preference pairs**：ScPO 利用 self-consistency（多数投票）作为代理指标，把模型针对同一问题采样出的多个回答按一致性程度排序——多数答案（most consistent response）作为 chosen，少数答案（least consistent response）作为 rejected，从而构造出 preference pairs。这些 pairs 并非来自人工偏好或外部 RM 打分，而是完全由模型群体输出之间的统计一致性决定。
3. **DPO-style 训练**：构造出 preference pairs 后，使用加权 DPO + NLL 损失函数进行偏好优化训练，属于典型的 preference optimization carrier。
4. **迭代自举**：通过多轮迭代（M₀ → M₁ → M₂），模型逐步变得更 consistent、更 accurate，每轮还自动生成新问题扩充训练集，体现了 self-generated-target bootstrapping 的核心范式。

因此，ScPO 的 artifact 是：**把多数答案与少数答案构造成 preference pairs，再做 DPO-style post-training**。

---

## 2. 解决的问题

- **Self-alignment 在推理任务上效果不佳**：现有 self-alignment 方法（如 Self-Rewarding LM）让模型自己评估自身生成的回答质量，但 Huang et al. (2024) 证明 LLM 在复杂推理任务上**无法准确自评**（self-correct），导致 self-evaluation 失效。
- **外部 Reward Model 存在 OOD 问题**：使用 off-the-shelf RM（如 ArmoRM）对回答打分构造 preference pairs，虽然不需要 gold labels，但 RM 在 out-of-distribution 任务上表现差（如 ZebraLogic），噪声大，甚至会把正确答案排在错误答案后面。
- **Gold labels 获取成本高**：复杂多步推理问题的人工标注极其昂贵，尤其是 process-level supervision（如 PRM800K），严重限制了可扩展性。
- **Self-consistency 仅用于推理时**：传统 self-consistency (Wang et al., 2023) 只在 inference time 通过多次采样和多数投票提升回答准确率，但需要额外计算开销，且不更新模型参数，无法持久改善模型能力。

ScPO 的核心想法：**将 self-consistency 从 inference-time technique 转化为 training-time supervision signal**，用一致性程度构造 preference pairs 进行迭代偏好优化训练。

---

## 3. 方法介绍

### 3.1 总体流程

ScPO 是一个迭代训练框架，每轮迭代包含三个步骤：

1. **生成新问题（Query Generation）**：利用当前模型通过 few-shot prompting 生成新的数学/逻辑问题（仅生成问题，不需要生成答案），扩充训练集。
2. **构建 Self-Consistency Preference Pairs**：对每个问题采样 $k$ 个回答，按 final answer 的出现频率（vote）排序，选出 chosen 和 rejected。
3. **用加权 ScPO Loss 训练**：使用 instance-level weighted DPO + NLL loss 进行偏好优化。

### 3.2 初始化

- **初始模型 $M_0$**：预训练 base LLM（如 Llama-3 Base 8B），不要求 instruction-tuned。
- **种子查询集**：少量高质量的 unlabeled queries（如 GSM8K/MATH 的 train set 问题，去掉答案）。

### 3.3 生成新问题

利用 Llama-3 Instruct 8B 对种子问题进行 few-shot prompting，生成类似难度的新问题：
- GSM8K / MATH：从训练集中随机选 4 个问题作为 in-context examples，提示模型生成一个新的可解数学问题。
- ZebraLogic：以 one-shot 方式提示模型替换现有 puzzle 的属性（如人名、饮品名）生成变体。
- **关键设计**：只要求生成有效问题，不要求同时生成正确答案——即使部分问题不可解，也能通过后续 vote filter 过滤掉。

### 3.4 构建 Self-Consistency Preference Pairs

对训练集中的每个问题 $x$，用当前模型 $M_t$ 以温度采样生成 $k$ 个回答 $\bar{y}_x = \{y_1, y_2, \dots, y_k\}$。

**Vote 函数**：提取每个回答 $y$ 的 final answer（通过 $\text{ans}(\cdot)$），计算其相对频率：
$$V(y) = \sum_{m=1}^{k} \mathbf{1}(\text{ans}(y_m) = \text{ans}(y))$$

**Preference pair 构造**：
$$D_t^{\text{pairs}} = \{(x, y^+, y^-) \mid x \in D_t, \; y^+ = \arg\max_{y \in \bar{y}_x} V(y), \; y^- = \arg\min_{y \in \bar{y}_x} V(y), \; V(y^+) \geq \tau\}$$

- Chosen $y^+$：投票最多的答案对应的某个 response（多个 response 可能指向同一 final answer，随机选一）。
- Rejected $y^-$：投票最少的答案对应的 response。
- **过滤阈值 $\tau$**：只保留 chosen 的投票 $\geq \tau$ 的实例。$M_0$ 阶段 $\tau = 0.5k$（GSM8K/MATH），$M_1$ 阶段提高到 $0.7k$（GSM8K）和 $0.6k$（MATH），因为模型变得更 consistent 后需要更严格的筛选。ZebraLogic 中 $M_1$ 设 $\tau=2$（至少 2 票完全匹配），$M_2$ 设 $\tau=0.5k$。

### 3.5 ScPO Loss Function

$$\mathcal{L}_{\text{ScPO}}(y^+, y^- \mid x) = \underbrace{-w(x) \log \sigma\!\left(\beta \log \frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta \log \frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} + \underbrace{\frac{\alpha \, w(x)}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- $w(x) = (V(y^+) - V(y^-)) / k$：instance-level weight，反映 preference pair 的质量/模型对该 pair 的置信度。vote margin 大 → 权重高 → 更可靠的 pair。$w(x) \in [0, 1]$。
- $\beta = 0.5$：DPO temperature。
- $\alpha = 1$：NLL 正则化系数。
- Reference model：当前迭代初始化时的模型 $M_t$。
- 该 loss 形式类似 IRPO (Pang et al., 2024) 的 DPO + NLL，但关键区别在于：(a) 无监督设定；(b) 使用 instance-level weight $w(x)$。

### 3.6 迭代训练

采用 $T=2$ 轮迭代：
- $M_0$：种子 LLM（预训练模型）
- $M_1$：用 $M_0$ 生成 $D_0^{\text{pairs}}$（含种子问题 + 新问题），用 $\mathcal{L}_{\text{ScPO}}$ 训练
- $M_2$：用 $M_1$ 生成 $D_1^{\text{pairs}}$（含种子问题 + 新问题），用 $\mathcal{L}_{\text{ScPO}}$ 训练

每轮迭代中模型变得更 consistent 和 accurate，从而能：(a) 从更多问题中构造有效 preference pairs（之前 $M_0$ 无法获得多数答案的问题现在可以）；(b) 为下一轮提供更高质量的训练数据。

### 3.7 Semi-Supervised 变体

当部分数据有 gold labels 时：
- 有标注问题 $x_{\text{gold}}$：采样 $k$ 个回答，chosen 为正确答案、rejected 为错误答案，权重 $w(x_{\text{gold}}) = 1$。
- 无标注问题：按照标准 ScPO 自一致性方式构造 pair 和计算权重。
- 特殊情况：当所有数据都有标签时，loss 退化为 IRPO loss。

### 3.8 关键超参数

| 参数 | 设置 |
|------|------|
| 采样数 $k$ | GSM8K/MATH: 8；ZebraLogic: 16 |
| 采样温度 | 0.7（生成回答和新问题）；1.2（采样 rejected，鼓励多样性） |
| top-p | 0.9 |
| 训练 epochs | 10 |
| 学习率 | 5e-6（cosine scheduling） |
| batch size | 16（effective） |
| $\beta$ (DPO) | 0.5 |
| $\alpha$ (NLL) | 1 |
| 迭代次数 $T$ | 2 |

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 小学数学推理 | GSM8K | 7.5K/1.3K（train/test）grade school math word problems；实验中 train 拆分为 6.7K/0.8K（train/dev） |
| 高中竞赛数学 | MATH | 7.5K/5K（train/test）challenging high-school math competitions；实验中 train 拆分为 6.7K/0.8K/5K（train/dev/test） |
| 逻辑推理 | ZebraLogic | 1K 逻辑网格谜题（Einstein's puzzles），仅有 test set，无 train/dev set；每个 puzzle 是 $n \times m$ 表的约束满足问题 |

**训练数据生成统计**：
- GSM8K：$M_1$ 使用 5.3K seed queries；$M_2$ 使用 1.4K seed + 5.1K generated queries
- MATH：$M_1$ 使用 0.6K seed + 1.2K generated queries；$M_2$ 使用 1.2K seed + 2.5K generated queries（由于 MATH 难度高，仅约 1/4 的 seed queries 能获得清晰多数答案）
- ZebraLogic：$M_1$ 使用 0.4K seed + 2.0K generated；$M_2$ 使用 0.5K seed + 2.2K generated

---

## 5. 评估指标与主要结果

### 评估指标

- **GSM8K / MATH**：final answer exact match accuracy（greedy decoding + 8-way self-consistency (SC) decoding）
- **ZebraLogic**：
  - Puzzle accuracy（整体 / easy / hard）：完全正确匹配整张 $n \times m$ 表
  - Cell accuracy：匹配表中每个单元格

### 主要结果

#### GSM8K（Llama-3 Base 8B）

| 方法 | 迭代 | 训练数据 (K) | Greedy Acc. | SC Acc. |
|------|------|-------------|-------------|---------|
| Seed model (zero-shot CoT) | $M_0$ | - | 41.17% | 51.80% |
| IRPO_RM | $M_1$ | 5.5K seed | 48.67% | 69.98% |
| LMSI | $M_2$ | 1.1K+5.2K | 56.71% | 62.55% |
| **ScPO_Unsup.** | $M_1$ | 5.3K seed | **61.03%** | **71.49%** |
| **ScPO_Unsup.** | $M_2$ | 1.4K+5.1K | **63.91%** | 71.11% |
| IRPO_Gold（有标签上界） | $M_2$ | 5.7K† | 64.29% | 72.56% |
| **ScPO_Semi-Sup.** | $M_2$ | 5.7K†+4.5K | **66.64%** | **74.75%** |

**核心发现**：
- ScPO_Unsup. 单轮迭代即超越种子模型 **+22.74%**（greedy）、超越 IRPO_RM **+12.36%**
- 两轮 ScPO_Unsup. 与有 gold labels 的 IRPO_Gold 性能差距 **< 1%**（greedy: 63.91% vs 64.29%）
- Semi-supervised ScPO 在有 gold labels 基础上进一步提升 IRPO_Gold **+2.35%**（greedy）

#### MATH（Llama-3 Base 8B）

| 方法 | 迭代 | Greedy Acc. | SC Acc. |
|------|------|-------------|---------|
| Seed model | $M_0$ | 14.46% | 18.20% |
| IRPO_RM | $M_1$ | 18.06% | 24.20% |
| LMSI | $M_2$ | 16.96% | 20.20% |
| **ScPO_Unsup.** | $M_1$ | 17.36% | **25.70%** |
| **ScPO_Unsup.** | $M_2$ | **19.72%** | 24.58% |
| IRPO_Gold | $M_2$ | 20.32% | 26.88% |
| **ScPO_Semi-Sup.** | $M_2$ | **20.48%** | 26.92% |

**核心发现**：
- 两轮 ScPO_Unsup. 超越种子模型 **+5.26%**（greedy），超越 IRPO_RM **+1.64%**
- 与 IRPO_Gold 差距 < 1%（greedy: 19.72% vs 20.32%）
- MATH 比 GSM8K 更难，仅约 1/4 的种子问题能获得清晰多数答案，因此需要更依赖 generated problems 来补充训练数据

#### ZebraLogic（Llama-3 Instruct 8B）

| 方法 | Puzzle Acc. (Overall) | Easy | Hard | Cell Acc. |
|------|-----------------------|------|------|-----------|
| Llama-3 Instruct 70B | 17.2% | 52.1% | 3.6% | 42.9% |
| Gemma-2 27B IT | 16.3% | 50.7% | 2.9% | 41.2% |
| Claude-3 Haiku | 14.3% | 47.9% | 1.2% | 37.9% |
| Llama-3 Instruct 8B ($M_0$) | 11.6% | 40.0% | 0.4% | 39.1% |
| IRPO_RM $M_1$ | 11.3% | 37.9% | 1.0% | 42.1% |
| LMSI $M_2$ | 16.8% | 53.6% | 2.5% | 46.9% |
| **ScPO_Unsup. $M_1$** | 17.0% | 54.3% | 2.5% | 47.6% |
| **ScPO_Unsup. $M_2$** | **18.1%** | **58.2%** | 2.5% | 45.2% |

**核心发现**：
- 8B 模型经过两轮 ScPO 训练后，**超越 Llama-3 70B (+0.9%)**、**Gemma-2 27B (+1.8%)**、**Claude-3 Haiku (+3.8%)**
- 在 leaderboard 上从第 38 名上升至第 30 名
- IRPO_RM 在 ZebraLogic 上几乎无效（puzzle acc. 反而下降 0.3%），因为 ArmoRM 对 ZebraLogic 高度 OOD，40.5% 的 pairs 存在错误排序（incorrect orderings）
- ScPO 单轮迭代即超越 IRPO_RM **+5.7%**（puzzle acc.）和 **+5.5%**（cell acc.）

#### Llama-3.1 Base 8B 验证实验

| 方法 | GSM8K Greedy | MATH Greedy |
|------|-------------|-------------|
| Seed model | 43.14% | 15.70% |
| ScPO_Unsup. $M_2$ | 64.22% (+21.08%) | 23.20% (+7.50%) |
| ScPO_Semi-Sup. $M_2$ | 68.46% (+25.32%) | 24.36% (+8.66%) |

趋势与 Llama-3 一致，验证了方法的鲁棒性。

### 关键发现

1. **Weighted loss 至关重要**：与 unweighted loss ($w(x)=1$) 相比，ScPO 的加权 loss 在 GSM8K $M_1$ 上提升 +2.5%，MATH $M_1$ 上提升 +1.44%（greedy），说明根据 vote margin 动态调整权重能有效利用 pair 质量信息。

2. **模型一致性随迭代递增**：majority vote share（$V(y^+)/k$）在每轮迭代后稳步增长（GSM8K: ~50% → ~60% → ~70%），原因包括：(i) 模型准确率提升；(ii) preference optimization 降低模型输出多样性；(iii) ScPO 将 SC distribution 蒸馏进模型的 single-sample distribution。

3. **Self-consistency 优于 RM 作为质量指标**：在 Figure 3 的分析中，ArmoRM 在所有三个数据集上的 incorrect orderings 比例（19.1%, 32.4%, 40.5%）均高于 self-consistency（7.8%, 11.8%, 16.0%）。尤其在 OOD 的 ZebraLogic 上，self-consistency 比 ArmoRM 多 12.3% 的 correct orderings。

4. **过滤阈值 $\tau$ 的 trade-off**：在 MATH 上，$\tau$ 从 0.1k → 0.5k，accuracy margin 从 18% → 57%，模型性能从 15.44% → 17.36%（quality > quantity）；但 $\tau=0.7k$ 时数据量骤降至 < 700 pairs，性能反降至 14.76%（数据不足导致欠拟合）。$\tau=0.5k$ 是最优平衡点。

5. **SC decoding 在训练后收益递减**：$M_2$ 的 SC accuracy 有时略低于 $M_1$，因为模型经 ScPO 训练后已变得高度 consistent，8-way SC 带来的额外互补性减少。ScPO 本质上在训练时"蒸馏"了 inference-time SC 的效果。

6. **Transduction 进一步提升**：第三轮 ScPO 在 train queries 上收益很小（< 1%），但用 test set 问题构建 preference pairs 进行训练可额外提升 +1.44%（GSM8K greedy），说明 ScPO 可通过 transductive learning 适应目标数据分布。
