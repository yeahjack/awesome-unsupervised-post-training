# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning

> **加入 Survey 时间：** 2026-03-11

**论文信息**: Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng (UIUC), arXiv: 2505.15134, 2025年5月

| When to Adapt | multi-protocol: Offline Corpus UPT (EM-FT, EM-RL) + No-Update Inference (EM-INF, adjacent) |
|---|---|
| 触发单位 Trigger Unit | mixed: whole corpus / prompt batch / inference state |
| 参数/状态持久性 Persistence | mixed: offline parameter accumulation + prompt-local inference state |
| 与推理关系 Inference Coupling | mixed |
| 输入可见性 Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Multi-protocol: Cumulative + Non-Cumulative |
| 重置边界 Reset Boundary | Multi-protocol: Deployment Boundary + Token / Sequence Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Multi-protocol: Offline Corpus UPT + No-Update Inference (adjacent) |
| 可见数据范围 Visibility Scope | Multi-protocol: Pre-deployment corpus + Current Sequence Prefix Only |

---

## When to Adapt 审计

- **辅助 taxonomy：** 这篇论文包含多个 protocol entry：`Timing Regime=Multi-protocol: Offline Corpus UPT + No-Update Inference (adjacent)`；`Visibility Scope=Multi-protocol: Pre-deployment corpus + Current Sequence Prefix Only`。
- **两轴编码：** `Input Visibility=Multi-protocol: Offline + Online`；`Update Persistence=Multi-protocol: Cumulative + Non-Cumulative`；`Reset Boundary=Multi-protocol: Deployment Boundary + Token / Sequence Boundary`。

| Protocol Entry | Timing Regime | Visibility Scope | 输入可见性 Input Visibility | Update Persistence | Reset Boundary | 说明 Note |
|---|---|---|---|---|---|---|
| EM-FT | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | 离线训练阶段直接最小化 entropy，形成新的部署模型 |
| EM-RL(seq) | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | 离线序列级 RL 变体，更新跨样本保留到部署阶段 |
| EM-RL(tok) | Offline Corpus UPT | Pre-deployment corpus | Offline | Cumulative | Deployment Boundary | 离线 token 级 RL 变体，更新跨样本保留到部署阶段 |
| EM-INF *(adjacent)* | No-Update Inference (adjacent) | Current Sequence Prefix Only | Online | Non-Cumulative | Token / Sequence Boundary | 推理时对 logits 做 entropy 下降；按论文图（`No-Update Inference (adjacent)` 叶，未通过 B1）路由到 adjacent 而非 strict UPT |

- **更新何时触发：** 这份笔记覆盖的是一个方法族而不是单一协议：EM-FT、EM-RL(seq)、EM-RL(tok) 发生在 offline pre-deployment 阶段，而 EM-INF 发生在 inference-stage。
- **服务当前样本还是后续样本：** 因此"当前更新服务谁"也分成两类：offline 变体服务后续部署模型，EM-INF 则服务当前 prompt / 当前 inference state。
- **参数/状态是否累积：** 持久性同样是混合的：offline 变体做参数累积，EM-INF 则是 prompt-local 的 logit / hidden-state 优化。
- **reset 边界：** 因此本笔记采用 protocol-level split：offline 变体以 deployment boundary 为边界，EM-INF 以 token / sequence boundary 为边界。

## 1. UPT 归属理由

训练时变体（EM-FT、EM-RL(seq)、EM-RL(tok)）属于 strict UPT，归入 **Family I: Prediction-Statistic Optimization**。EM-INF 是推理时变体；论文图将其路由到相邻分支 **No-Update Inference (adjacent)**，因为它不修改参数、适配器、记忆或持久局部状态（未通过 B1）。

核心理由：Entropy minimization (EM) 的所有变体都直接最小化模型自身预测分布的 Shannon entropy，不依赖任何外部标签、外部验证器或人工反馈。唯一的优化信号来源于模型输出的 intrinsic statistics——即 token 级别或 trajectory 级别的 entropy 估计。这完全符合 "prediction-statistic" 的定义：优化目标直接从模型自身的预测统计量中导出。

具体而言：
- **EM-FT**（strict UPT，Family I）: Direct Opt.，training-time，Token 级别——直接对 token-level entropy 做梯度下降
- **EM-RL(seq)**（strict UPT，Family I）: Policy Opt.，training-time，Traj. 级别——以负 trajectory entropy 作为 intrinsic reward，通过 policy gradient 优化
- **EM-RL(tok)**（strict UPT，Family I）: Policy Opt.，training-time，Token 级别——以累计负 token entropy 作为 reward
- **EM-INF**（*adjacent*，未通过 B1）: State Opt.，inference-time，Token 级别——在推理时对 logits 做梯度下降以降低 entropy；序列结束后无任何持久参数、适配器、记忆或状态更新留存

所有变体均无需外部 ground truth，信号类型为 intrinsic statistics。

---

## 2. 解决的问题

本文研究一个核心问题：**仅通过最小化模型自身输出的 entropy，不使用任何标注数据，能否提升 LLM 的推理能力？**

现代 LLM 在预训练阶段已获得大量推理能力，但这些能力往往未被充分激发。传统 post-training 方法（如 SFT、RLHF、GRPO）依赖标注数据或外部验证器。本文探索一种极简的替代方案：利用模型置信度与正确性之间的相关性，通过强化模型的高置信输出来提升性能。

具体动机包括：
- 传统 RL 方法（如 GRPO、RLOO）需要 output verification，而很多任务（如代码生成、科学编程）难以进行答案提取和验证
- Self-consistency 依赖 majority voting，在复杂任务中不适用
- Iterative refinement 受限于 context length
- 需要一种不依赖任何任务假设、不需要标注数据的通用 post-training/inference 方法

---

## 3. 方法介绍

### 基础定义

设 $\pi_\theta$ 为自回归 LLM policy，entropy 有两种估计方式：

- **Trajectory-level entropy**: $\hat{\mathcal{H}}_{\text{traj}}(\pi_\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \pi_\theta(\mathbf{y}^i)$，基于完整序列的 log-probability
- **Token-level entropy**: $\hat{\mathcal{H}}_{\text{tok}}(\pi_\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{|y^i|} \mathcal{H}(\pi_\theta(\cdot | \mathbf{y}_{<t}^i))$，逐 token 计算 Shannon entropy 后累加

### 3.1 EM-FT: Direct Entropy Minimization Fine-tuning

- **优化方式**: Direct optimization（直接梯度下降）
- **阶段**: Training-time
- **粒度**: Token 级别

EM-FT 模仿 SFT 的流程，但不使用标注数据。给定 input prompts，从模型自身采样 $N$ 条 trajectory，直接最小化 token-level entropy $\hat{\mathcal{H}}_{\text{tok}}$。梯度为 $\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{|y^i|} \nabla_\theta \mathcal{H}(\pi_\theta(\cdot | \mathbf{y}_{<t}^i))$。

关键发现：仅用 $N=1$（单条采样）即可在数学和编码任务上平均提升 base model 约 8%，在 Minerva 和 LeetCode 上甚至超过需要 60K 标注样本的 GRPO 和 RLOO。

### 3.2 EM-RL(seq): Trajectory-level Entropy as Intrinsic Reward

- **优化方式**: Policy optimization (REINFORCE)
- **阶段**: Training-time
- **粒度**: Trajectory 级别

将负 trajectory-level entropy 作为 reward：$r_{\text{traj}}(\mathbf{y}) = \sum_{t=1}^{|\mathbf{y}|} \log \pi_\theta(y_t | y_{<t}) = \log \pi_\theta(\mathbf{y})$。

该 reward 偏好整体高概率的 trajectory，不关心每个 token 是否确定，适用于较短的推理任务（如简单数学题），推理路径数量有限的场景。使用 RLOO baseline 减少梯度方差，附加 KL regularizer（$\beta=0.001$）防止偏离 base model 过远。

### 3.3 EM-RL(tok): Cumulative Negative Token Entropy as Reward

- **优化方式**: Policy optimization (REINFORCE)
- **阶段**: Training-time
- **粒度**: Token 级别

将负 token-level entropy 的累积和作为 reward：$r_{\text{tok}}(\mathbf{y}) = -\sum_{t=1}^{|\mathbf{y}|} \mathcal{H}(\pi_\theta(\cdot | y_{<t}))$。

该 reward 偏好在每一步生成时都更确定的 trajectory，适用于需要长 chain-of-thought 推理的复杂任务，防止模型在推理过程中"迷失"。相比 EM-RL(seq)，EM-RL(tok) 在 LiveCode 和 AMC 等需要长推理的任务上表现更好。

### 3.4 EM-INF: Inference-time Logit Optimization

- **优化方式**: State optimization（对 logits 做梯度下降）
- **阶段**: Inference-time
- **粒度**: Token 级别

EM-INF 完全不更新模型参数。在每个生成步骤 $t$，将模型最后一层输出的 logit 向量 $z_t \in \mathbb{R}^{|\mathcal{V}|}$ 视为自由参数，通过梯度下降最小化 softmax 分布的 entropy：

$$\mathcal{L}_{\text{EM-INF}} = \max\left(-\sum_{j \in \mathcal{V}} \sigma(z_t)_j \log \sigma(z_t)_j, \delta\right)$$

其中 $\delta \in (0.1, 0.5)$ 为 entropy 下界阈值，防止过度优化导致退化为 greedy decoding。实验发现 5-15 步梯度下降即可。优化仅作用于 $|V|$ 个参数（约 150K），远小于模型参数量（7B），开销可忽略。

关键优势：
- 不需要训练数据或参数更新
- 计算复杂度 $\mathcal{O}(n)$（$n$ 为序列长度），而 self-consistency 和 iterative refinement 需要 $\mathcal{O}(Nn)$
- 可应用于任何任务，无需答案提取或 output verification
- 与 temperature scaling 不同，logit optimization 可以改变 non-top logits 的相对顺序，在高 entropy 场景下更有效

---

## 4. 数据集

### 训练数据（用于 EM-FT 和 EM-RL）
- **数学**: 从 Numina Math 随机采样 35K prompts
- **编码**: 从 Eurus-2 coding split 采样 25K prompts
- 训练数据仅包含 prompts，**不使用任何标签或答案**

### 评估 Benchmarks

**数学任务**:
- MATH-500
- AMC (AMC competition problems)
- AIME 2024
- Minerva Math
- Olympiad Bench (Olymp.)

**编码任务**:
- LeetCode (LeetC)
- LiveCodeBench-v2 (LiveC)

**科学编程**（仅用于 EM-INF）:
- SciCode
- UGPhysics

### 基座模型
- Qwen2.5-Math-7B / Qwen2.5-7B-Instruct（数学训练）
- Eurus-2-7B-SFT（编码训练）
- Qwen2.5-32B-Instruct（EM-INF 实验）
- Llama-3.1-8B-Instruct（消融实验）

---

## 5. 评估指标与主要结果

### 评估指标
- 数学任务：accuracy（答案提取后与 ground truth 对比）
- 编码任务：pass rate
- 科学编程（SciCode）：sub-problem 和 main problem 的 accuracy
- 计算效率：FLOPs（训练 $6PD$，推理 $2PD$，$P$ 为参数量，$D$ 为 token 数）

### EM-FT 与 EM-RL 主要结果（Table 2, Qwen2.5-7B）

| 方法 | Math | AMC | AIME | Minerva | Olymp. | Avg.(Math) | LeetC | LiveC | Avg.(Code) |
|------|------|-----|------|---------|--------|------------|-------|-------|------------|
| Base model | 43.8 | 31.3 | 15.6 | 14.7 | 19.0 | 24.9 | 26.1 | 18.4 | 22.3 |
| w/ RLOO N=4 (60K标注) | 73.0 | 57.8 | 23.3 | 31.2 | 34.2 | 43.9 | 28.3 | 26.7 | 27.5 |
| w/ GRPO N=4 (60K标注) | 71.8 | 56.6 | 21.1 | 25.0 | 35.9 | 42.1 | 25.0 | 25.8 | 25.4 |
| **EM-FT N=1** (无标注) | 67.2 | 51.8 | 14.4 | **33.3** | 34.4 | 40.2 | 28.3 | 17.2 | 22.8 |
| **EM-RL-seq N=4** (无标注) | 67.2 | **53.0** | **21.1** | **30.9** | 35.6 | **41.6** | *31.1* | 21.7 | 26.4 |
| **EM-RL-tok N=4** (无标注) | **70.8** | **57.8** | 18.9 | **30.9** | 35.9 | **42.9** | 29.5 | 24.5 | 27.0 |

关键发现：
- **EM-FT**：无标注数据下平均提升 base model 约 8%，在 Minerva 和 LeetCode 上超过 GRPO/RLOO
- **EM-RL**：无标注数据下平均提升 base model 约 11%，在 AMC、Minerva、LeetCode 上超过有标注的 GRPO/RLOO 约 4.5%
- **EM-RL-tok** 在整体数学任务上略优于 EM-RL-seq，而 EM-RL-seq 在编码任务上有优势

### EM-INF 主要结果（Table 3）

在 Qwen2.5-7B-Instruct 上，EM-INF 在数学任务平均提升约 3%，优于 self-consistency（$N=4$）和 iterative refinement，且仅需单条 trajectory 的计算量。

### SciCode 科学编程结果（Table 4）

- Qwen2.5-7B + EM-INF：main problem accuracy 从 0.0% 提升至 1.5%，sub-problem 从 11.5% 提升至 16.7%
- Qwen2.5-32B + EM-INF：main problem 10.7%，超过 GPT-4o（9.2%）、Claude 3.5 Sonnet（12.3%）、Gemini 1.5 Pro（7.7%），以及 Adaptive Temperature（7.6%）
- 计算效率：EM-INF 比 self-consistency 快约 3 倍（Figure 1）

### 局限性（Table 5）

- 在 Llama-3.1-8B 上效果明显弱于 Qwen2.5，说明 EM 的有效性依赖于 base model 的预训练质量
- 在 individualistic value reasoning (IndVal) 任务上几乎无效，因为模型置信度不是该类任务质量的可靠代理
- EM 的前提假设是"模型置信度与正确性相关"，当 base model 能力不足或任务偏离预训练分布时，该假设不成立
