# R-Zero: Self-Evolving Reasoning LLM from Zero Data

> **加入 Survey 时间：** 2026-03-11

**Paper:** R-Zero: Self-Evolving Reasoning LLM from Zero Data
**Authors:** Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, Dong Yu (Tencent AI Seattle Lab, Washington University in St. Louis, University of Maryland College Park, UT Dallas)
**ArXiv:** 2508.05004
**Date:** ICLR 2026 (Feb 13, 2026)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| R-Zero | Policy Opt. | training-time | Semantic |

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

**Family III — Self-Generated Target Bootstrapping (reasoning / curriculum synthesis)**

R-Zero 是一个完全自主的 self-evolving 框架，**不依赖任何外部人工标注数据或预先存在的任务集**，所有训练信号均由模型自身生成。具体而言：

- **Challenger** 模型通过 GRPO 学习生成处于 Solver 能力边界上的数学问题（difficulty-calibrated question synthesis），这些问题构成了一个 **自适应课程（adaptive curriculum）**。
- **Solver** 在 Challenger 生成的合成问题上使用 **majority-vote pseudo-labels** 进行 GRPO 训练，标签完全来自模型自身的多次采样一致性投票，不需要外部 oracle。
- 整个 co-evolutionary 循环（Challenger 生成 → Solver 标注并过滤 → Solver 训练 → 下一轮 Challenger 训练）中的**课程内容和伪标签均为模型内部合成产物（internal artifacts）**，符合 Self-Generated Target Bootstrapping 的定义。

该方法属于 Direct optimization 子类中的 reasoning / plan / curriculum synthesis：Challenger 自生成边界难度问题，Solver 在 majority-vote pseudo-labels 上用 GRPO 训练；整个 co-evolutionary 课程与标签均来自模型内部。

---

## 2. 解决的问题

现有 self-evolving LLM 方法面临两个核心瓶颈：

1. **数据依赖瓶颈**：大多数 RLVR（Reinforcement Learning with Verifiable Rewards）方法需要大量人工标注的任务和标签作为监督信号，这在成本和可扩展性上存在根本限制，尤其阻碍了超越人类智能的 AI 系统发展。
2. **Label-free 方法的局限**：虽然 label-free RL 方法（如基于 confidence、consistency、entropy 的奖励信号）可以消除对标签的依赖，但它们仍然需要一个**预先存在的未标注问题集（unlabeled problem corpus）**，限制了在真正 self-evolving 场景下的可扩展性。
3. **Self-play 方法的局限**：现有 self-challenging 方法（如代码领域的 Coder-Tester self-play）依赖于外部代码执行器等 verification oracle，在缺乏可验证环境的开放推理领域（如数学推理）难以保证自生成数据的质量和正确性。

R-Zero 旨在构建一个**从零数据出发的完全自主推理 LLM 训练框架**，不需要任何人工标注任务、标签或外部验证器。

---

## 3. 方法介绍

### 3.1 总体框架

R-Zero 从一个 base LLM 初始化出两个独立模型：**Challenger** $Q_\theta$ 和 **Solver** $S_\phi$，二者通过迭代 co-evolution 循环进行交替优化。每轮迭代包含三个阶段：

### 3.2 Phase 1: Challenger Training（问题生成器训练）

Challenger 是一个 autoregressive LM，使用 GRPO 算法训练以生成具有挑战性的数学问题。核心在于设计精确捕捉"好问题"特性的 reward function：

**Uncertainty Reward（不确定性奖励）：**

对于 Challenger 生成的问题 $x$，从当前 frozen Solver $S_\phi$ 采样 $m$ 个回答 $\{y_1, ..., y_m\}$，通过 majority vote 得到 pseudo-label $\tilde{y}(x)$，计算 Solver 的经验准确率：

$$\hat{p}(x; S_\phi) = \frac{1}{m} \sum_{j=1}^{m} \mathbb{1}\{y_j = \tilde{y}(x)\}$$

不确定性奖励定义为：

$$r_{\text{uncertainty}}(x; \phi) = 1 - 2\left|\hat{p}(x; S_\phi) - \frac{1}{2}\right|$$

该函数在 Solver 准确率接近 50% 时达到最大值，即激励 Challenger 生成 Solver **最大不确定的问题**——既不太简单也不太难（理论上，当 $\hat{p} = 0.5$ 时，KL 散度下界所衡量的学习潜力最大，见 Appendix F）。

**Repetition Penalty（重复惩罚）：**

为鼓励 batch 内问题多样性，基于 BLEU score 计算 pairwise distance $d_{ij} = 1 - \text{BLEU}(x_i, x_j)$，对距离低于阈值 $\tau_{\text{BLEU}} = 0.5$ 的问题进行 agglomerative clustering，惩罚与 cluster 相对大小成正比：

$$r_{\text{rep}}(x_i) = \lambda \frac{|C_k|}{B}$$

其中 $B$ 为 batch size，$\lambda = 1$。

**Format Check Penalty：** 要求生成问题必须包含 `<question>` 和 `</question>` 标签，不合格直接赋 reward = 0。

**Composite Reward：**

$$r_i = \max\left(0, r_{\text{uncertainty}}(x_i; \phi) - r_{\text{rep}}(x_i)\right)$$

使用该 reward 通过 GRPO 更新 Challenger 策略。

### 3.3 Phase 2: Solver Dataset Construction（训练数据构建）

从更新后的 Challenger 采样 $N = 8000$ 个候选问题，对每个问题从 Solver 采样 $m = 10$ 个回答，通过 majority vote 确定 pseudo-label $\tilde{y}_i$ 并计算经验准确率 $\hat{p}_i$。

**Difficulty-based Filtering：** 仅保留满足 $|\hat{p}_i - \frac{1}{2}| \leq \delta$（$\delta = 0.25$）的问题，即 majority vote 匹配数量在 3-7 之间。该过滤同时起到：
- **课程校准**：去除过易或过难的任务
- **隐式质量控制**：低一致性问题通常意味着问题本身模糊或 pseudo-label 不可靠

### 3.4 Phase 3: Solver Training（求解器训练）

在过滤后的数据集上使用 GRPO 训练 Solver，采用简单的 binary verifiable reward：

$$r_j = \begin{cases} 1, & \text{if } x_j \text{ matches pseudo-label } \tilde{y}_i \\ 0, & \text{otherwise} \end{cases}$$

### 3.5 训练超参数

| 组件 | 参数 | 值 |
|------|------|-----|
| Solver | Global Batch Size | 128 |
| Solver | Learning Rate | $1 \times 10^{-6}$ |
| Solver | Weight Decay | $1 \times 10^{-2}$ |
| Solver | KL Penalty $\lambda_{KL}$ | $1 \times 10^{-2}$ |
| Solver | Max Steps | 15 |
| Solver | Number of Rollouts | 5 |
| Challenger | Global Batch Size | 128 |
| Challenger | Learning Rate | $1 \times 10^{-6}$ |
| Challenger | Max Steps | 5 |
| Challenger | Number of Rollouts | 4 |
| 共用 | Rollout Temperature | 1.0 |
| 共用 | Rollout Top-p | 0.99 |
| 共用 | 精度 | BFloat16 + FlashAttention2 |

---

## 4. 数据集

### 训练数据

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学 | 自生成（Zero Data） | Challenger 每轮生成 N=8000 候选问题，经过 difficulty filtering 后约剩余子集用于 Solver 训练；不使用任何外部训练数据 |
| 数学（synergy 实验） | math12k (hiyouga) | 用于对比 R-Zero 与 supervised fine-tuning 的协同效果 |

### 评估基准

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | AMC | 美国数学竞赛题，report mean@32 |
| 数学推理 | Minerva | 数学推理基准，greedy decoding accuracy |
| 数学推理 | MATH-500 | Hendrycks et al. 数学问题集，greedy decoding accuracy |
| 数学推理 | GSM8K | 小学数学应用题，greedy decoding accuracy |
| 数学推理 | OlympiadBench | 奥林匹克竞赛级双语多模态科学问题，greedy decoding accuracy |
| 数学推理 | AIME-2024 | 美国数学邀请赛 2024，report mean@32 |
| 数学推理 | AIME-2025 | 美国数学邀请赛 2025，report mean@32 |
| 通用推理 | MMLU-Pro | 增强版多任务语言理解基准，Exact Match accuracy |
| 通用推理 | SuperGPQA | 研究生级推理基准（285 学科），Exact Match accuracy |
| 通用推理 | BBEH | BIG-Bench Extra Hard，更高难度的复杂推理基准，Exact Match accuracy |

---

## 5. 评估指标与主要结果

### 评估指标

- **数学基准**：AMC 和 AIME 使用 **mean@32**（32 次采样取平均），其他数学基准使用 **greedy decoding accuracy**
- **通用基准**：使用 **Exact Match (EM) accuracy**，greedy decoding
- 数学答案正确性判定使用 **GPT-4o 作为 judge**（temperature=0.1）

### 主要结果

#### 数学推理（Table 1，Step 45 结果）

| 模型 | Avg | AMC | Minerva | MATH | GSM8K | Olympiad | AIME25 | AIME24 |
|------|-----|-----|---------|------|-------|----------|--------|--------|
| Qwen3-4B Base | 42.57 | 45.70 | 38.24 | 68.20 | 87.79 | 41.04 | 10.30 | 6.70 |
| Qwen3-4B + Absolute Zero | 46.42 | 52.45 | 41.96 | 76.20 | 89.34 | 42.56 | 10.20 | 12.20 |
| **Qwen3-4B + R-Zero** | **49.93** | **57.27** | **52.94** | **79.60** | **92.12** | **44.59** | 9.60 | **13.40** |
| Qwen3-8B Base | 48.64 | 51.95 | 50.00 | 78.00 | 89.08 | 44.74 | 12.10 | 14.60 |
| **Qwen3-8B + R-Zero** | **53.72** | **61.67** | **60.66** | **82.00** | **94.09** | **48.89** | **13.30** | **15.40** |
| OctoThinker-3B Base | 26.64 | 17.19 | 24.26 | 55.00 | 73.69 | 16.15 | 0.21 | 0.00 |
| **OctoThinker-3B + R-Zero** | **29.32** | **27.03** | **27.57** | 54.20 | **74.98** | **18.22** | **3.23** | 0.00 |
| OctoThinker-8B Base | 36.41 | 32.11 | 41.91 | 65.20 | 86.96 | 26.52 | 1.56 | 0.62 |
| **OctoThinker-8B + R-Zero** | **38.52** | **34.03** | **48.22** | **68.80** | **87.19** | **27.56** | 0.42 | **3.44** |

#### 通用推理（Table 2）

| 模型 | OverallAvg | SuperGPQA | MMLU-Pro | BBEH |
|------|-----------|-----------|----------|------|
| Qwen3-4B Base | 26.34 | 20.88 | 50.58 | 7.57 |
| **Qwen3-4B + R-Zero** | **31.15** | **27.55** | **55.47** | **10.42** |
| Qwen3-8B Base | 31.98 | 28.33 | 58.97 | 8.63 |
| **Qwen3-8B + R-Zero** | **34.50** | **31.38** | **61.53** | **10.60** |
| OctoThinker-3B Base | 7.47 | 10.09 | 10.87 | 1.46 |
| **OctoThinker-3B + R-Zero** | **11.12** | **12.44** | **16.71** | 4.20 |
| OctoThinker-8B Base | 11.70 | 13.26 | 20.21 | 1.64 |
| **OctoThinker-8B + R-Zero** | **23.00** | **19.82** | **40.92** | **8.25** |

#### R-Zero 与 Supervised Fine-Tuning 的协同（Table 4，Qwen3-4B）

| 训练数据 | AMC | Minerva | MATH | GSM8K | Olympiad | AIME25 | AIME24 | SuperGPQA | MMLU-Pro | BBEH |
|----------|-----|---------|------|-------|----------|--------|--------|-----------|----------|------|
| Human only | 57.97 | 55.15 | 80.8 | 92.04 | 48 | 9.58 | 10.31 | 29.49 | 57.03 | 9.71 |
| R-Zero only | 57.27 | 52.94 | 79.6 | 92.12 | 44.59 | 9.6 | 13.4 | 27.55 | 55.47 | 10.42 |
| R-Zero + Human | 57.9 | 53.2 | 81.2 | 92.2 | 47.7 | 10.32 | 14.67 | 29.4 | 58.2 | 11.8 |

使用 R-Zero 作为 mid-training 后再在人工标注数据上 fine-tune，Qwen3-4B 获得 **+2.35** 的增益，Qwen3-8B 获得 **+3.69** 增益（Figure 4），优于单独使用任一数据源。

### 关键发现

1. **Model-agnostic 有效性**：R-Zero 在 Qwen3（0.6B/1.7B/4B/8B）和 OctoThinker（3B/8B）等不同架构和规模的模型上均有效，数学推理平均提升 +2.68 到 +7.36 points。

2. **推理能力的跨领域迁移**：仅通过数学问题训练，R-Zero 在通用推理基准上也显示出显著提升（如 Qwen3-4B 通用平均 +4.81，OctoThinker-8B +11.30），表明该方法增强了模型的底层推理能力而非领域特定知识。

3. **Challenger RL 训练的关键作用**：经过 RL 训练的 Challenger 显著优于未训练的 base Challenger。以 Qwen3-4B 为例，第一轮迭代即带来 +3.7 points 的提升（vs. base Challenger 的 +2.44）。

4. **Iteration Scaling 与性能崩溃**：模型在早期迭代中持续提升，但**最终会出现性能退化**。模型越大，崩溃越晚：0.6B 在 Step 15 后即退化，4B 在 Step 45 后才退化。这揭示了 self-evolving 框架的内在不稳定性。

5. **Pseudo-label 质量下降**：随着迭代推进，Challenger 生成的问题难度递增（Step 15 时 Solver 准确率 59.0% → Step 45 时 47.0%），但 pseudo-label 准确率从 79.0% 系统性下降到 63.0%，成为框架性能的潜在瓶颈。

6. **两模型分离的必要性**（Table 6）：使用共享参数的 Single-R-Zero 在第一轮迭代后即开始退化（peak=47.31），而双模型 R-Zero 持续改善三轮（peak=49.12）。共享参数导致 pseudo-label 准确率更低（63.4% vs. 71.0% at Step 15），可能源于模型内部偏差导致的 overconfidence。

7. **Ablation**（Table 3，Qwen3-4B）：移除 Repetition Penalty 使 Math AVG 从 49.07 降至 45.76（-3.31）；移除 Task Filtering 使 General AVG 从 31.15 降至 26.69（-4.46），表明多样性和课程校准对框架成功至关重要。

8. **Domain-general training**（Table 8）：去除 math-specific prompt 限制后，8B 模型在数学和通用基准上均进一步提升（如 AMC 65.53 vs. 61.67），说明方法可扩展到更广泛的推理任务。

---

## 引用

```bibtex
@inproceedings{huang2026rzero,
  title={R-Zero: Self-Evolving Reasoning LLM from Zero Data},
  author={Huang, Chengsong and Yu, Wenhao and Wang, Xiaoyang and Zhang, Hongming and Li, Zongxia and Li, Ruosen and Huang, Jiaxin and Mi, Haitao and Yu, Dong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
