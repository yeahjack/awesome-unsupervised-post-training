# CYCLE-INSTRUCT: Fully Seed-Free Instruction Tuning via Dual Self-Training and Cycle Consistency

> **加入 Survey 时间：** 2026-03-11

- **Method**: CYCLE-INSTRUCT
- **Carrier**: Direct Opt.
- **Regime**: Training-time
- **Level**: Semantic
- **Venue**: EMNLP 2025

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | self-generated data batch / iteration round |
| 参数/状态持久性 Persistence | full parameter accumulate across synthesis / refinement rounds |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | note-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新在 deployment 前的离线自举循环里触发，通常是“生成数据 / 评分 / 筛选 / 再训练”的 round-based schedule。
- **服务当前样本还是后续样本：** 当前 round 产生的合成样本或伪目标主要服务下一轮训练与最终部署模型，而不是服务某个测试样本的即时推理。
- **参数/状态是否累积：** 参数在多轮自举过程中持续累积，论文通常不会做 sample-level reset。
- **reset 边界：** 因此这类方法的 `When to Adapt` 核心是 offline iterative bootstrapping，而不是 online test-time adaptation。

## 1. UPT 归属理由

CYCLE-INSTRUCT 属于 **Family III (Self-Generated Target Bootstrapping, knowledge/instruction self-curation)**。该方法完全不依赖任何人工标注的 seed data、外部 teacher model 或外部验证器。其核心思路是利用 dual self-training loop 和 cycle consistency，从原始无标签文本中自动生成并过滤 instruction-response 配对数据。两个模型（answer generator 和 question generator）仅以彼此的 pseudo-label 和原始文本的 reconstruction error 作为训练信号，所有监督信号均来自模型自身生成内容与原始文本的内在一致性，符合 UPT 中 "model-generated content" 和 "intrinsic statistics" 作为合法信号源的定义。

---

## 2. 解决的问题

传统 instruction tuning 依赖大量人工标注数据或强大的外部 teacher model（如 GPT-4）生成合成数据。即便是 instruction back-translation 方法，也仍需一个人工精心策划的 seed set 来引导生成过程。这种 **seed dependency** 带来三个核心瓶颈：

1. **Seed 收集成本高**：构建高质量、多样化的 seed set 需要大量人力；
2. **数据浪费与分布偏移**：小规模 seed set 的风格和主题偏差会传递到合成数据中，导致多样性不足，且将所有未标注段落一律视为 answer 会造成数据浪费；
3. **可扩展性受限**：在隐私敏感场景或特定领域中，获取 seed data 或调用外部模型常常不可行。

CYCLE-INSTRUCT 旨在实现 **完全无 seed** 的 instruction tuning，仅从原始未标注文本出发即可生成高质量 instruction-following 数据。

---

## 3. 方法介绍

### 3.1 Data Segmentation & Reformat

将原始文档按段落切分，利用简单的启发式规则（包含 "?" 即为 question，否则为 answer）将段落分为两个集合：
- $\mathcal{D}_Q^{\text{raw}}$：potential question passages
- $\mathcal{D}_A^{\text{raw}}$：potential answer passages

随后用固定的 rewriting prompt 将原始段落分别改写为标准的 instruction 格式（question → self-contained question）和 response 格式（answer → coherent answer paragraph），得到 $\mathcal{D}_Q$ 和 $\mathcal{D}_A$。

### 3.2 Cycle Training Procedure

基于同一 base model（如 LLaMA-3.1-8B）实例化两个 transformer 模型：
- **Forward model** $\mathcal{M}_{Q \to A}$：给定 instruction 生成 response
- **Backward model** $\mathcal{M}_{A \to Q}$：给定 response 生成 instruction

训练以四步为一个 cycle 迭代进行：

1. **Step 1 — Pseudo-Answer Generation**：用 $\mathcal{M}_{Q \to A}$ 为 $\mathcal{D}_Q$ 中每个 question $q_i$ 生成 pseudo-answer $\hat{a}_i$；
2. **Step 2 — Backward Model Training**：用 $(q_i, \hat{a}_i)$ 对训练 $\mathcal{M}_{A \to Q}$，目标是从 pseudo-answer 重构出原始 question（最小化负对数似然）；
3. **Step 3 — Pseudo-Instruction Generation**：用 $\mathcal{M}_{A \to Q}$ 为 $\mathcal{D}_A$ 中每个 answer $a_j$ 生成 pseudo-instruction $\hat{q}_j$；
4. **Step 4 — Forward Model Training**：用 $(\hat{q}_j, a_j)$ 对训练 $\mathcal{M}_{Q \to A}$，目标是从 pseudo-instruction 重构出原始 answer。

核心训练信号来源于 **cycle consistency loss**：每个样本的 ground-truth 一端始终是原始文本本身，模型需要在对方生成的 pseudo-label 条件下重构出原始文本，从而实现两个模型的互相监督。

### 3.3 Cycle-Consistency Filtering（可选）

最终数据集 $\mathcal{D}_{\text{final}}$ 合并两个方向的 pseudo-labeled pairs。可进一步通过 cycle-consistency filtering 进行质量筛选：

1. 将每个 pair 的 pseudo 端传入对向模型做 **one-step reconstruction**；
2. 计算原始标签与重构标签之间的 **embedding distance**（使用 LLaMA-3 inference encoder）；
3. 对 gold side 做 **semantic clustering**（k-means, k=200）；
4. 在每个 cluster 内用 **k-center greedy pruning** 去除 embedding distance 最大的 top 5% 样本。

过滤后数据集 $\mathcal{D}_{\text{cycle}} \approx 0.95 \cdot |\mathcal{D}_{\text{final}}|$。

---

## 4. 数据集

实验覆盖四个 data track：

| Track | 数据集 | Pairs Used | Unlab. Q | Unlab. A |
|-------|--------|-----------|----------|----------|
| Track 1: General Instructions | Alpaca-GPT4 | 20,000 | 10,000 | 10,000 |
| Track 1: General Instructions | Dolly-15k | 15,000 | 7,500 | 7,500 |
| Track 2: Domain-Specific | Medical-Alpaca | 20,000 | 10,000 | 10,000 |
| Track 3: Dialogue Logs | OASST-1 | 40,000 | 15,126 | 24,874 |
| Track 4: Plain Text | WikiHow-4w | 40,000 | 5,178 | 34,822 |

- Track 1 使用已有 instruction 数据集，随机移除配对的一侧来模拟 unlabeled 场景；
- Track 2 使用 Medical-Alpaca（医疗领域），同样移除一侧模拟缺少标注的场景；
- Track 3 使用 OASST-1 对话日志，按 "?" 启发式识别 question/answer turns；
- Track 4 使用 WikiHow 文章，从叙述性文本中提取潜在的 Q-A 段落。

---

## 5. 评估指标与主要结果

### 评估指标

- **Standard instruction metrics**：MMLU (accuracy %)、BBH、CRASS、DROP — 使用 InstructEval 框架
- **Medical track**：MMLU 的 8 个医学子领域（CK, CB, CC, CM, HB, HC, MG, PM）
- **Open-ended quality**：AlpacaEval win-rate（与 ALL-SFT baseline 的 pairwise comparison）
- **Synthetic data quality**：GPT-4o Mini 对合成 QA pair 的 alignment 评分（0-10 scale）

### 主要结果

所有实验基于 **LLaMA-3.1-8B-Base + LoRA** 微调。

1. **全面超越 back-translation baselines**：在所有四个 track 上，CYCLE-INST 和 CYCLE-FILT（0% 人工标注）均显著优于所有 seed-based back-translation 变体（Random-k%, Cluster-k%，使用 5-20% 的 seed data）。

2. **接近甚至超越全监督方法**：
   - **Alpaca-GPT4**（Table 1）：Cycle-Filt Avg 54.17 vs ALL-SFT (100% gold) 54.99，差距极小；
   - **Dolly-15k**（Table 1）：Cycle-Filt Avg 51.00 vs ALL-SFT 53.45；
   - **Medical-Alpaca**（Table 2）：Cycle-Filt Avg **0.636** 超过 ALL-SFT 0.626；
   - **OASST-1**（Table 4）：Cycle-Filt Avg **51.98** 超过 ALL-SFT 51.69；
   - **WikiHow**（Table 4）：Cycle-Filt Avg **51.07** 超过 ALL-SFT 50.59。

3. **Cycle-consistency filtering 的有效性**：CYCLE-FILT 在所有 benchmark 上系统性地优于未过滤的 CYCLE-INST，验证了 reconstruction verification 可有效去除噪声 pseudo-pairs。

4. **AlpacaEval 结果**：在 OASST-1 和 WikiHow 上，Cycle-Filt 的胜率明显超过 ALL-SFT；在 Alpaca 和 Dolly 上也有竞争力。

5. **GPT-4o Mini QA alignment 评分**：CYCLE-INST 在所有数据集上的合成 pair 对齐质量均高于 seed-based 方法（Alpaca 9.46, Dolly 9.90, MedAlpaca 9.96, WikiHow 9.43）。
