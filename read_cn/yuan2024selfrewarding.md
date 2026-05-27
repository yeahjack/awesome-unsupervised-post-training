# Self-Rewarding LM: Self-Rewarding Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Self-Rewarding Language Models
**Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta & NYU)
**ArXiv:** arXiv:2401.10020
**Date:** 2024-01 (v3: 2025-03)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Self-Rewarding LM | Pref. Opt. | training-time | Semantic |

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

**Family IV — Internal Evaluator Bootstrapping (self-rewarding)**

本文是 "Internal Evaluator Bootstrapping" 家族中 **self-rewarding** 子类的奠基性工作。其核心思想是：不再依赖外部、固定的 reward model 或人类偏好数据来提供训练信号，而是让 **同一个 LLM 同时扮演 instruction following model 和 reward model 两个角色**。模型通过 LLM-as-a-Judge prompting 对自身生成的回答进行打分，构建 chosen–rejected preference pairs，并在 Iterative DPO 框架中持续自举训练。

这种方法属于无监督 post-training（UPT）的原因：
- **无需额外人类标注**：在初始 seed data 之后，所有新的训练数据（prompt、response、reward）均由模型自身生成和评估
- **自举式迭代改进**：每一轮迭代中，模型不仅提升 instruction following 能力，还同步提升 reward modeling 能力，形成良性循环（virtuous circle）
- **突破人类反馈瓶颈**：传统 RLHF 受限于人类 preference 数据的规模和质量，以及冻结 reward model 的能力上限；Self-Rewarding 通过将 reward model 融入迭代训练过程来突破此上限

---

## 2. 解决的问题

传统 LLM 对齐方法（RLHF / DPO）面临两个核心瓶颈：

1. **人类反馈瓶颈（Human Feedback Bottleneck）**：训练信号的质量和规模受限于人类偏好标注数据的获取成本和标注者能力上限。要实现 superhuman agents，需要 superhuman feedback，而人类标注无法提供。
2. **冻结 Reward Model 瓶颈（Frozen Reward Model）**：标准 RLHF 流程训练出一个独立的 reward model 后将其冻结，在 LLM 训练过程中不再更新。这意味着 reward model 的能力上限在训练开始时就已固定，无法随着 LLM 能力的提升而同步进化。

本文提出的解决思路是：**让模型自身成为 reward model**，通过迭代训练使 reward modeling 能力与 instruction following 能力同步提升，从而实现自我对齐（self-alignment），潜在地突破人类反馈的天花板。

---

## 3. 方法介绍

### 3.1 总体框架

Self-Rewarding Language Models 的核心设计是让模型同时具备两项能力：
1. **Instruction Following**：根据 user prompt 生成高质量回答
2. **Self-Instruction Creation**：生成新的扩展训练样本并自我评估其质量

整个框架是一个 **迭代式自对齐（Iterative Self-Alignment）** 流程，每轮迭代包括自指令创建（Self-Instruction Creation）和指令遵循训练（Instruction Following Training）两个阶段。

### 3.2 初始化（Initialization）

系统需要两类 seed data：

- **IFT (Instruction Fine-Tuning) 数据**：人工标注的 (instruction prompt, response) 对，用于 SFT 训练。实验中使用 Open Assistant 数据集的 3,200 个高质量英文样本（rank 0）。
- **EFT (Evaluation Fine-Tuning) 数据**：(evaluation instruction prompt, evaluation result response) 对，训练模型的 LLM-as-a-Judge 能力。每条包含 chain-of-thought 推理（justification）和最终评分（满分 5 分）。实验中从 Open Assistant 构造了 1,630 条训练 / 541 条评估样本。

将 IFT 与 EFT 数据合并训练得到初始模型 $M_1$。

### 3.3 Self-Instruction Creation

在每轮迭代中，利用当前模型生成新的训练数据：

1. **生成新 Prompt**：使用 few-shot prompting（从 seed IFT 中采样 6 条 + 模型生成 2 条作为 demonstration），采用 Self-Instruct 方法生成新的 instruction prompt $x_i$
2. **生成候选回答**：对每个 prompt $x_i$ 采样生成 $N=4$ 条候选回答 $\{y_i^1, \dots, y_i^N\}$（temperature $T=0.7$, $p=0.9$）
3. **自我评估**：使用同一模型的 LLM-as-a-Judge 能力对候选回答打分 $r_i^n \in [0, 5]$，每条回答评估 3 次取平均。

### 3.4 LLM-as-a-Judge Prompt 设计

采用 **additive 5-point scoring system**，基于五个递进标准累积评分：
- **1 分**：回答相关且提供了部分信息
- **2 分**：回答覆盖了用户问题的主要部分
- **3 分**：回答了基本问题要素且有用
- **4 分**：清晰地从 AI Assistant 视角回答，组织良好、全面、有帮助
- **5 分**：完美的 AI Assistant 回答，无多余信息，展现专家知识

该 additive prompt 相比 multiple-choice 式 prompt（如 Li et al. 2024）效果显著更好——在 SFT Baseline 上 pairwise accuracy 为 65.1% vs. 仅 26.6%。

### 3.5 Instruction Following Training（Iterative DPO）

利用自评估结果构建 **preference pairs**：
- 对每个 prompt 的 $N$ 条候选回答，选取 **最高分** 作为 winning response $y^w$，**最低分** 作为 losing response $y^l$（若分数相同则丢弃该对）
- 使用 **DPO (Direct Preference Optimization)** 在 preference pairs 上训练

### 3.6 模型序列

$$M_0 \xrightarrow{\text{SFT on IFT+EFT}} M_1 \xrightarrow{\text{DPO on AIFT}(M_1)} M_2 \xrightarrow{\text{DPO on AIFT}(M_2)} M_3$$

- $M_0$：Base pretrained LLM（Llama 2 70B）
- $M_1$：在 IFT+EFT seed data 上 SFT 训练
- $M_2$：初始化为 $M_1$，在 $\text{AIFT}(M_1)$ 数据上 DPO 训练（3,964 preference pairs）
- $M_3$：初始化为 $M_2$，在 $\text{AIFT}(M_2)$ 数据上 DPO 训练（6,942 preference pairs）

关键特性：每一轮的 reward model 都是模型自身，因此 **reward model 随迭代同步进化**，区别于传统方法中使用固定外部 reward model。

### 3.7 训练超参数

- **SFT**：learning rate 5.5e−6（cosine decay to 1.1e−6），batch size 16，dropout 0.1
- **DPO**：learning rate 1e−6（decay to 1e−7），batch size 16，dropout 0.1，$\beta=0.1$
- 每 200 steps 保存 checkpoint，在 253 条验证集上 early stopping

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| Seed IFT 数据 | Open Assistant | 3,200 条高质量英文 first-turn 样本（rank 0），用于 SFT |
| Seed EFT 数据 | Open Assistant (多排名回答) | 1,630 训练 + 541 评估的 LLM-as-a-Judge 样本 |
| 合成偏好数据 | AIFT($M_1$) | 3,964 preference pairs（模型自生成 + 自评估） |
| 合成偏好数据 | AIFT($M_2$) | 6,942 preference pairs（模型自生成 + 自评估） |
| **评估：Instruction Following** | | |
| Head-to-Head | IFT test data | 256 条多来源 test prompts，GPT-4 pairwise 评估 |
| Leaderboard | AlpacaEval 2.0 | 805 prompts，GPT-4 评估 win rate vs. GPT-4 Turbo |
| Multi-turn | MT-Bench | 多轮对话挑战任务（math、coding、roleplay、writing 等），GPT-4 评分（满分 10） |
| NLP Benchmarks | ARC-Easy / ARC-Challenge / HellaSwag / SIQA / PIQA / GSM8K / MMLU / OBQA / NQ | 通用 NLP 能力保持性评估 |
| **评估：Reward Modeling** | Open Assistant (held-out) | 每条指令平均 2.85 个排名回答，评估与人类排名的一致性 |

---

## 5. 评估指标与主要结果

### 评估指标

**Instruction Following 指标：**
- Head-to-Head Win Rate（GPT-4 评估，256 prompts，双序比较，不一致记为 tie）
- AlpacaEval 2.0 Win Rate（vs. GPT-4 Turbo，805 prompts）
- MT-Bench Score（满分 10，GPT-4 评分）
- Human Evaluation（作者盲评，50 prompts × 3 annotators，majority vote）
- NLP Benchmark Accuracy（9 个标准 NLP 数据集）

**Reward Modeling 指标：**
- Pairwise Accuracy（模型评分排序与人类排名的 pairwise 一致率）
- 5-best %（模型打满分 5 的回答中被人类排名最高的比例）
- Exact Match %（完整排序与人类排名完全一致的比例）
- Spearman Correlation
- Kendall's $\tau$ Correlation

### 主要结果

#### Instruction Following 能力逐迭代提升

**Head-to-Head vs. SFT Baseline（GPT-4 评估）：**

| 对比 | 左方 Win | Tie | 右方 Win |
|------|---------|-----|---------|
| $M_1$ vs. SFT Baseline | 30.5% | 38.7% | 30.9% |
| $M_2$ vs. SFT Baseline | 49.2% | 36.3% | 14.5% |
| $M_3$ vs. SFT Baseline | **62.5%** | 27.7% | 9.8% |
| $M_2$ vs. $M_1$ | 55.5% | 32.8% | 11.7% |
| $M_3$ vs. $M_2$ | 47.7% | 39.8% | 12.5% |
| $M_3$ vs. $M_1$ | 68.8% | 22.7% | 8.6% |

**AlpacaEval 2.0 Win Rate（vs. GPT-4 Turbo）：**

| 模型 | Win Rate |
|------|----------|
| $M_1$ (Iteration 1) | 9.94% |
| $M_2$ (Iteration 2) | 15.38% |
| $M_3$ (Iteration 3) | **20.44%** |
| GPT-4 0314 | 22.07% |
| Claude 2 | 17.19% |
| Gemini Pro | 16.85% |
| GPT-4 0613 | 15.76% |
| LLaMA2 Chat 70B | 13.87% |

$M_3$ 超越了 Claude 2（17.19%）、Gemini Pro（16.85%）和 GPT-4 0613（15.76%）。

**MT-Bench 结果（满分 10）：**

| 模型 | Overall | Math/Code/Reasoning | Humanities/STEM/Roleplay/Writing |
|------|---------|---------------------|----------------------------------|
| SFT Baseline | 6.85 | 3.93 | 8.60 |
| $M_1$ | 6.78 | 3.83 | 8.55 |
| $M_2$ | 7.01 | 4.05 | 8.79 |
| $M_3$ | **7.25** | **4.17** | **9.10** |

细分类别中，Writing（8.83→9.58）、Roleplay（8.15→8.73）、Extraction（6.90→7.80）、STEM（9.18→9.45）提升最为显著。

**Human Evaluation（50 prompts，3 annotators per pair）：**

| 对比 | Self-Rewarding Win | Tie | SFT Baseline Win |
|------|-------------------|-----|-----------------|
| $M_1$ vs. SFT | 28.0% | 26.0% | 46.0% |
| $M_2$ vs. SFT | 56.0% | 24.0% | 20.0% |
| $M_3$ vs. SFT | **66.0%** | 16.0% | 18.0% |

人类评估结果与 GPT-4 自动评估一致，验证了迭代训练的有效性。

#### Reward Modeling 能力逐迭代提升

| 指标 | SFT Baseline | $M_1$ (Iter 1) | $M_2$ (Iter 2) | $M_3$ (Iter 3) |
|------|-------------|----------------|----------------|----------------|
| Pairwise Acc. ↑ | 65.1% | 78.7% | 80.4% | **81.7%** |
| 5-best % ↑ | 39.6% | 41.5% | **44.3%** | 43.2% |
| Exact Match % ↑ | 10.1% | 13.1% | **14.3%** | **14.3%** |
| Spearman Corr. ↑ | 0.253 | 0.279 | 0.331 | **0.349** |
| Kendall's $\tau$ ↑ | 0.233 | 0.253 | 0.315 | **0.324** |

关键发现：尽管迭代训练中未添加额外 EFT 数据，且自生成的样本也不像 LLM-as-a-Judge 训练样本，reward modeling 能力仍持续提升。

#### NLP Benchmark 能力保持

| 模型 | ARC-C ↑ | HellaSwag ↑ | GSM8K ↑ | MMLU ↑ | NQ ↑ |
|------|---------|-------------|---------|--------|------|
| Llama 2 | 57.40 | 85.30 | 56.80 | 68.90 | 25.30 |
| SFT Baseline | 55.97 | 85.17 | 50.72 | 69.76 | 34.35 |
| $M_1$ | 57.51 | 84.99 | 60.27 | 69.34 | 35.48 |
| $M_2$ | 54.51 | 84.27 | 59.29 | 69.31 | 33.07 |
| $M_3$ | 53.13 | 83.29 | 57.70 | 69.37 | 31.86 |

NLP benchmark 性能大致保持稳定，存在轻微下降（"alignment tax"），但未出现严重退化。

### 关键发现

1. **双轴同步提升**：Self-Rewarding 迭代训练不仅提升了 instruction following 能力，还同步提升了 reward modeling 能力，形成 **良性循环（virtuous circle）**——更好的 reward model 产生更高质量的 preference data，进而训练出更好的 LLM。
2. **EFT 数据的重要性**：加入 EFT seed data 对 self-rewarding loop 至关重要。不使用 EFT 时模型难以稳定输出评分格式、评分易收敛到 4 分，有效训练样本极其稀少（仅 541 / 429 pairs vs. 正常的 3,964 / 6,942 pairs）。
3. **Additive Prompt 优势**：additive 5-point scoring prompt 远优于 multiple-choice 式 prompt（pairwise accuracy 65.1% vs. 26.6%）。
4. **Preference Optimization > 正样本增强**：仅添加高分正样本进行 SFT 几乎无效（29% vs 30% wins），而 DPO preference pairs 显著有效，说明 preference signal 对自举训练至关重要。
5. **能力提升分布不均**：在 writing、roleplay、extraction 等创造性/综合性任务上提升最大，在 math、coding、reasoning 等任务上提升较小——受限于 seed data 中此类任务比例偏低。
6. **回答长度增长**：$M_1$ 平均 1,092 tokens → $M_2$ 1,552 tokens → $M_3$ 2,552 tokens，模型倾向生成更长的回答，这可能是性能提升的一个因素（也是潜在的 length bias）。
7. **突破人类数据上限的可能性**：由于 reward model 在迭代中持续改进，理论上可以获得超越仅用原始人类 seed data 训练出的 reward model / LLM 的性能，为 self-improvement beyond human feedback 打开了大门。
