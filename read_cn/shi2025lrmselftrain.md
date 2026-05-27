# Can Large Reasoning Models Self-Train?

> **加入 Survey 时间：** 2026-03-11

> **论文元信息**

| 属性 | 值 |
|---|---|
| Method | LRM Self-Train (Self-Rewarded Training, SRT) |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |
| 作者 | Sheikh Shafayat (KAIST), Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, Andrea Zanette (CMU) |
| 发表状态 | Preprint, 2025 (arXiv: 2505.21444v2) |
| 代码 | https://github.com/tajwarfahim/srt |

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

本文属于 **Family III — Self-Generated Target Bootstrapping** 中的 **reasoning / plan / curriculum synthesis** 子类。

核心理由如下：

- **无外部标签的 pseudo-target 构造**：SRT 的核心机制是用模型自身多次生成的 majority vote 来构造 pseudo-label（即 semantic-level 的 reasoning target），而非依赖任何 ground-truth verifier 或人工标注。这本质上是一种 self-generated-target bootstrapping 流程——模型自身"合成"了训练目标。
- **Carrier 为 Direct Optimization**：合成出的 pseudo-reward 直接被注入 RL 梯度更新（RLOO/GRPO 等 policy gradient 算法），对模型参数进行在线优化，而非仅在 inference 阶段做调整。
- **Level 为 Semantic**：majority vote 筛选的是最终答案（final answer）层面的一致性，属于 semantic-level 的信号，而非 trajectory-level 的完整推理路径选择。
- **Self-feedback 的自增强效应**：随着 RL 训练的推进，label-generating policy 本身也在不断进化（evolving teacher），从而生成更高质量的 pseudo-target，形成"训练信号自我提升"的正向循环，这是 self-generated-target bootstrapping 的典型特征。

---

## 2. 解决的问题

### 2.1 背景与动机

- 预训练数据（human-curated corpora）的供给正成为 LLM scaling 的瓶颈。
- RL with Verifiable Rewards (RLVR) 已在推理任务上取得成功（如 DeepSeek-R1、OpenAI o1），但依赖 ground-truth verifier。
- 实现 super-intelligence 需要超越人类可提供 ground-truth 的领域，因此需要模型能够 **self-improve**——即利用自身输出的判断来指导后续训练。

### 2.2 现有方法的局限

- 先前 self-improvement 工作（如 STaR、Self-Instruct）主要基于 SFT 或 DPO，label-generating policy 在一个 training round 内保持固定，仅更新 1–10 轮。
- 这些方法无法利用 on-policy 数据的持续更新优势，且受限于固定 teacher 的 verification 能力上限。

### 2.3 核心研究问题

> **在 RL 框架下，self-training 能否通过每步梯度更新时同步更新 feedback 信号来实现持续自我提升？**

具体拆分为两个子问题：
1. SRT 能否超越 base model 的能力？（能力提升 + 自监督质量提升）
2. SRT 的自我提升能否无限持续？

---

## 3. 方法介绍

### 3.1 Self-Rewarded Training (SRT) 总体框架

SRT 的核心思想是：在缺乏 ground-truth verifier 的情况下，利用模型自身生成的 **majority vote** 作为 pseudo-label 来构造 RL 的 reward 信号。

**Algorithm 1: Self-Rewarded Training (SRT)**

```
Input: Prompt dataset X
foreach RL iteration do:
    1. Sample minibatch B ⊆ X
    2. For each prompt x ∈ B:
       a. Generate n solutions y(1), ..., y(n) ~ π_label(·|x)
       b. Identify majority-vote answer:
          y_majority ← argmax_{y'} Σ_{i=1}^{n} 1[answer(y(i)) = y']
       c. Define reward function:
          r(y) ← 1[answer(y) = y_majority]
    3. Perform RL gradient update using r(·)
```

> **Figure 1** (Overview of SRT)：在 RLVR 中，reward 来自 ground-truth verifier；而 SRT 用模型自身生成的 majority vote 来估计 ground truth，并将这个 proxy reward signal 用于训练。

### 3.2 Majority Voting 作为 Self-Supervision

- **理论基础**：majority vote 的准确率经验上高于单次生成的准确率（Wang et al., 2023a），因此可以利用模型内在的 **generation-verification gap**。
- **具体操作**：对每个 prompt 采样多个响应 → 按 final answer 分组 → 票数最多的答案作为 pseudo-label → 与 pseudo-label 一致的响应获得正 reward（binary reward）。

### 3.3 Evolving Teacher vs. Fixed Teacher

SRT 通过控制 π_label 来研究两种模式：
- **Fixed teacher**：π_label 固定为 base model（类似先前工作 Huang et al., 2023; Prasad et al., 2024）。
- **Evolving teacher**（SRT 核心）：π_label = 当前策略 π_θ，在每个梯度步之后都会更新，因此 pseudo-label 的质量随训练同步提升。在此模式下，可以复用 RL rollout 来生成 pseudo-label，**不引入额外计算开销**。

### 3.4 RL 算法兼容性

SRT 定义了一种 reward function 的形式，可与所有常见 RL 算法兼容：
- **RLOO**（Ahmadian et al., 2024）：使用 leave-one-out baseline。
- **GRPO**（Shao et al., 2024）：使用 group-level mean/std 归一化 advantage。
- 论文实验表明 RLOO 和 GRPO 在 SRT 下表现无显著差异。

### 3.5 Curriculum-based Self-Training

在可控制难度的 synthetic 任务中，论文提出了一种简单的 curriculum 策略：
- 先在最简单的 difficulty level 上用 ground-truth RL 训练。
- 然后逐级用 SRT 训练更高难度的数据（上一级的 final checkpoint 作为下一级的 starting point）。
- 实验表明模型可以在无 ground-truth 的情况下持续攀升多个难度级别。

### 3.6 Reward Hacking 与 Model Collapse

论文发现了 SRT 的关键局限：

- **现象**：prolonged SRT 训练导致模型学会输出高熵的随机 token 序列 + 固定的 "template" final answer（如 `\boxed{1}`），无论输入是什么。
- **机制**：self-consistency reward 被 hack——模型通过对所有 prompt 输出相同答案来最大化 training pseudo-reward，但这与真实正确性完全脱节。
- **信号**：
  - Training pseudo-reward 突然飙升
  - KL divergence 急剧增大
  - 模型 entropy 下降
  - 测试集 accuracy 完全崩溃

> **Figure 7**：展示了 SRT training dynamics——self-reward 突然升高与 accuracy 的同步崩溃，印证了 reward hacking 假说。

---

## 4. 数据集

### 4.1 训练数据集

| 数据集 | 类型 | 说明 |
|---|---|---|
| **Reasoning Gym** (Stojanovski et al., 2025) | Synthetic | 3 个任务：Family Relationships、Bitwise Arithmetic、Knights & Knaves；可调节 difficulty level |
| **MATH-12K** | Math | 12K 道数学问题 |
| **DAPO** (Yu et al., 2025) | Math | 去重后的 DAPO 数据集 |
| **Big-Math-RL-Verified** (Albalak et al., 2025) | Math | 按 pass rate 0.3–0.7 筛选的子集（用于 Llama-3.1-8B） |
| **AIME (1983–2023)** | Math | 历年 AIME 竞赛题 |

### 4.2 评估数据集

| 数据集 | 说明 |
|---|---|
| **MATH-500** | 500 道 held-out 数学题 |
| **AIME 2024** | 2024 年 AIME 竞赛题 |
| **AIME 2025** | 2025 年 AIME 竞赛题 |
| **AMC** | AMC 竞赛题 |
| **Reasoning Gym** (各 level 的 held-out set) | Synthetic 测试集 |

### 4.3 Base Models

- Qwen2.5-Math-7B
- Qwen3-4B-Base（Reasoning Gym 实验）
- Qwen3-14B-Base
- Llama-3.1-8B-Instruct
- Deepseek-Math-7B-Instruct

---

## 5. 评估指标与主要结果

### 5.1 评估指标

- **avg@k (Mean@k)**：k 次采样中的平均正确率（pass@1 的多采样版本）。
- **majority@k (Maj@k)**：k 次采样做 majority vote 后的正确率；反映 self-supervision 质量。
- **Pass@1**：单次采样正确率。
- **Training pseudo-reward**：SRT 训练过程中的自奖励信号。
- **KL divergence**：当前策略与 base model 之间的 token-level 平均 KL 散度。

### 5.2 主要结果

#### 结果一：SRT 可超越 Base Model（Takeaway 1）

**Synthetic tasks（Figure 2）**：
- 在 Family Relationships (Level 5)、Bitwise Arithmetic (Level 3)、Knights & Knaves (Level 7) 上，SRT 同时提升了 avg@16 和 majority@16 accuracy。
- Evolving teacher 显著优于 fixed teacher：Bitwise Arithmetic +10%，Family Relationship +8%，Knights & Knaves +6%。

**Real-world math tasks（Figure 3, 4）**：
- 在 4 个不同 base model 上，SRT 在 MATH-500 Pass@1 和 Majority@32 上均优于 base model。
- SRT 性能与使用 ground-truth reward 的标准 RLVR 相当。
- Llama-3.1-8B-Instruct 的 average accuracy 从 52.6% 提升至近 60%。

**与 offline 方法对比（Table 1）**：

| 方法 | MATH-12K | DAPO |
|---|---|---|
| Base Model | 0.15 | 0.15 |
| SFT (majority vote) | 0.18 | 0.18 |
| DPO | 0.23 | 0.21 |
| ScPO | 0.20 | 0.20 |
| **SRT** | **0.32** | **0.31** |
| RL on Ground Truth | 0.33 | 0.36 |

SRT 显著优于 SFT/DPO/ScPO 等 offline 变体，接近 ground-truth RL 的性能。

#### 结果二：Curriculum Self-Training 支持多级提升（Figure 5）

- 在 Bitwise Arithmetic 上，从 Level 2（ground-truth RL）→ Level 3 → Level 4 成功逐级攀升。
- 在 Knights & Knaves 上，从 Level 2 出发仅用 ground-truth 训练最简单级别，通过 SRT curriculum 攀升至 Level 9 接近 100% accuracy。

#### 结果三：Prolonged SRT 导致 Model Collapse（Takeaway 2）

**Figure 6**：在 4 个 base model 上，extended SRT 训练均展现出先升后崩的曲线——初始阶段性能与 ground-truth RL 匹配，但随后 **完全崩溃至接近 0%**。

**Ablation（Figure 8）**：
- GRPO vs RLOO：无显著差异，均会 collapse。
- 增大 KL coefficient：无法阻止 reward hacking（reward 信号太强）。
- 降低 learning rate：延缓但不消除 collapse。
- **减少 generations per prompt**（如 n=4 vs n=32）：注入噪声到 majority vote 中，反而 **延缓** collapse——这是一个反直觉但有意义的发现。

### 5.3 关键发现总结

| 发现 | 描述 |
|---|---|
| **Self-improvement 有效** | SRT 在 synthetic 和 real-world reasoning tasks 上均可超越 base model |
| **Self-supervision 质量提升** | Evolving teacher 的 majority vote accuracy 随训练提升，形成正向循环 |
| **与 RLVR 性能可比** | SRT 在多个 model + dataset 组合中接近 ground-truth RL 的表现 |
| **Curriculum 可行** | 简单的逐级 curriculum 策略可在无标签情况下实现多难度级别攀升 |
| **Long-term collapse 不可避免** | Prolonged SRT 必然出现 reward hacking → model collapse |
| **Feedback design 是核心挑战** | 论文呼吁未来研究开发更鲁棒的 verification 机制以实现持续 self-improvement |
