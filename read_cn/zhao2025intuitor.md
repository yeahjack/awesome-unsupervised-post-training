# Intuitor: Learning to Reason without External Rewards

> **加入 Survey 时间：** 2026-03-11

**Paper:** Learning to Reason without External Rewards (arXiv: 2505.19590)
**Authors:** Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, Dawn Song (UC Berkeley, Yale)
**Venue:** ICLR 2026
**Method:** Intuitor | **Carrier:** Policy Opt. (GRPO) | **Regime:** Training-time | **Level:** Token

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

Intuitor 属于 **Family II (Sample-Relation Supervision)**，具体为 trajectory-level consistency 下的 self-certainty 子类。

核心内在信号为 **self-certainty**：对序列中每个 token 位置，计算 uniform 分布 $U$ 相对于模型 next-token 分布 $p_{\pi_\theta}$ 的 forward KL divergence（即 $\text{KL}(U \| p_{\pi_\theta})$），然后对序列所有 token 取平均，得到 sequence-level 的内在奖励。Self-certainty 值越高代表模型对自身输出越"确信"。该信号完全来自模型自身的 token 分布，不依赖任何外部 ground truth、verifier 或人工标注，符合 UPT 定义。

关键特征：
- 奖励信号在 token level 定义，聚合到 sequence level 使用
- 使用 forward KL（mode-seeking），相比 entropy（mode-covering）不易偏向长序列
- Online annotator：self-certainty 由当前策略模型在线计算，随策略共同演化，避免 reward hacking

---

## 2. 解决的问题

现有 LLM 推理能力强化学习方法面临两大瓶颈：

1. **RLHF** 依赖大量人工标注偏好数据，成本高且存在偏差
2. **RLVR**（如 DeepSeek-R1 使用的 exact answer matching）需要 domain-specific 的 gold-standard 答案或 verifier，限制了在开放域任务上的适用性

Intuitor 提出的核心问题是：**LLM 能否仅依赖自身内在信号（不借助外部奖励或 ground truth）来提升推理能力？** 作者为此提出了 Reinforcement Learning from Internal Feedback (RLIF) 范式。

---

## 3. 方法介绍

### 3.1 RLIF 框架

优化目标为：

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_\theta(q)} \left[ u(q, o) - \beta \text{KL}[\pi_\theta(o|q) \| \pi_{\text{ref}}(o|q)] \right]$$

其中 $u(q,o)$ 为内在信号（非外部奖励），$\beta$ 控制 KL 正则化强度。

### 3.2 Self-Certainty 定义

$$\text{Self-certainty}(o|q) := \frac{1}{|o|} \sum_{i=1}^{|o|} \text{KL}(U \| p_{\pi_\theta}(\cdot|q, o_{<i})) = -\frac{1}{|o| \cdot |\mathcal{V}|} \sum_{i=1}^{|o|} \sum_{j=1}^{|\mathcal{V}|} \log(|\mathcal{V}| \cdot p_{\pi_\theta}(j|q, o_{<i}))$$

- $\mathcal{V}$：词表，$|o|$：序列长度
- 值越高表示模型越 confident

### 3.3 与 GRPO 的集成

基于 GRPO 框架，对每个 query $q$ 采样 $G$ 个候选输出，用 self-certainty 替代外部 verifiable reward：

$$u_i = \text{Self-certainty}(o_i|q), \quad \hat{A}_{i,t} = \frac{u_i - \text{mean}(\{u_1, \ldots, u_G\})}{\text{std}(\{u_1, \ldots, u_G\})}$$

归一化后的 advantage 用于 GRPO 的 clipped policy gradient 更新。整个流程不需要任何外部监督。

### 3.4 Online vs. Offline Annotator

使用 online self-certainty（由当前策略模型计算）而非 offline（固定 base model 计算）。实验表明 offline 版本容易被 exploit（模型学会在答案后附加已解决的问题来膨胀 self-certainty），而 online 版本因奖励信号与策略共同演化，能有效防止 reward hacking。

---

## 4. 数据集

### 训练数据
- **MATH dataset** 训练集：7,500 道数学题（主实验）
- **Codeforces** 代码生成数据集：3,200 道题（Intuitor-Code 变体）

### 评估 Benchmark
- **数学推理:** GSM8K, MATH500
- **代码生成:** CRUXEval-O, LiveCodeBench v6 (LCB)
- **指令遵循:** AlpacaEval 2.0 (length-controlled win rate, GPT-4.1 judge)
- **知识:** MMLU-Pro

### 模型
- Qwen2.5-1.5B, Qwen2.5-3B（主实验）
- Qwen2.5-7B/14B, Qwen3-14B, Llama-3.2, OLMo-2（ablation / 扩展实验）

---

## 5. 评估指标与主要结果

### 评估指标
- 数学任务：accuracy（greedy decoding）
- 代码任务：pass rate
- 指令遵循：AlpacaEval length-controlled win rate
- 知识：MMLU-Pro accuracy

### 主要结果（Qwen2.5-3B, 训练于 MATH）

| Model | GSM8K | MATH500 | LCB | CRUX | MMLU-Pro | AlpacaEval |
|-------|-------|---------|-----|------|----------|------------|
| Base | 0.673 | 0.544 | 0.093 | 0.236 | 0.377 | 3.72 |
| + GRPO | 0.826 | 0.636 | 0.085 | 0.341 | 0.403 | 6.91 |
| + Intuitor | 0.792 | 0.612 | 0.153 | 0.416 | 0.379 | 7.10 |

### 关键发现

1. **In-domain 性能可比:** Intuitor 在 GSM8K/MATH500 上与 GRPO（使用 gold answer）性能接近，略低但差距不大
2. **Out-of-domain 泛化更优:** 在代码生成任务上显著优于 GRPO（LCB: 0.153 vs 0.085, 相对提升 65%；CRUX: 0.416 vs 0.341, 相对提升 76%）
3. **更快的初期学习:** 仅 10 步训练后，Intuitor 在 GSM8K 和 MATH 上均超过 GRPO（Table 2）
4. **涌现结构化推理:** Intuitor 训练的模型自发产生 R1 风格的长链推理，在 JSON 格式输出前自动生成 pre-code reasoning
5. **指令遵循提升:** 在 AlpacaEval 上的 length-controlled win rate 超过 GRPO（7.10 vs 6.91）
6. **抗 reward hacking:** Online self-certainty 有效防止奖励 exploit，训练稳定
7. **跨架构鲁棒:** 在 Llama-3.2、OLMo-2 等不同模型族上均有效
