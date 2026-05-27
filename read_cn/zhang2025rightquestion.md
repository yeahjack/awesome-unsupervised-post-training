# EMPO: Entropy-Minimized Policy Optimization

> **加入 Survey 时间：** 2026-03-11

**论文**: Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization
**作者**: Qingyang Zhang, Haitao Wu, Changqing Zhang (Tianjin University), Peilin Zhao (Tencent AI Lab), Yatao Bian (Tencent AI Lab & NUS)
**arXiv**: 2504.05812
**Method**: EMPO | **Carrier**: Policy Opt. | **Regime**: training-time | **Level**: Semantic

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

EMPO 属于 **Family II (Sample-Relation Supervision)**，子类为 semantic agreement（语义一致性），具体机制为 semantic entropy / meaning clustering。

核心逻辑：给定一个 unlabeled question $q$，模型采样一组输出 $\{o_1, \dots, o_G\}$，随后按照语义等价性（bidirectional entailment）将输出聚类为 $M$ 个 meaning cluster $\{c_1, \dots, c_M\}$。每个输出的 reward 等于其所属 cluster 的概率 $r_i = p(c_j | q) \approx |c_j| / G$，即 cluster 越大（语义一致性越高）则 reward 越高。训练目标是最小化输出分布上的 semantic entropy $H = -\sum_{c_j} p(c_j|q) \log p(c_j|q)$。

**不使用任何外部监督信号**：无 ground-truth answer、无 rule-based verifier、无外部 reward model、无人工标注。唯一的信号来源是模型自身输出之间的语义关系（relational signal），通过 meaning clustering 将多个采样输出的一致性转化为 reward。

---

## 2. 解决的问题

现有 LLM reasoning 训练范式（SFT + RL）严重依赖外部监督信号，包括：
- 人工标注的 reasoning traces
- 验证过的 golden answers（rule-based reward）
- 预训练的 reward model

这些需求导致：
1. **数据获取成本高**：高质量推理数据需要领域专家标注
2. **可扩展性差**：难以推广到缺乏标准答案的开放性推理任务（如 free-form natural reasoning）
3. **依赖特定任务格式**：rule-based reward 仅适用于有确定性答案的数学类任务

EMPO 提出的核心问题：**能否在完全无监督的条件下激励 LLM 的推理能力？** 即仅利用 unlabeled questions 就能提升模型在数学推理和自由形式自然推理任务上的性能。

---

## 3. 方法介绍

### 3.1 Semantic Entropy 作为无监督优化目标

Semantic entropy 是 Shannon entropy 在语义空间上的自然扩展。直觉：一个可靠的模型应当对同一问题给出语义一致的回答；语义熵越低，模型的输出越集中在少数 meaning cluster 上，即越"确信"。已有工作证实 semantic entropy 与模型准确率呈强负相关。

**Meaning clustering 步骤**：
- 对数学推理任务：用正则表达式提取 `\boxed{}` 中的最终答案，答案相同则归入同一 cluster
- 对 free-form 自然推理任务：使用 General-Verifier（1.5B 参数的 SLM）判断两个输出是否语义等价（bidirectional entailment），等价则合并

**Semantic entropy 计算**：
$$p(c_j | q) \approx |c_j| / G, \quad H = -\sum_{c_j \in \{c\}} p(c_j|q) \log p(c_j|q)$$

### 3.2 Entropy-Minimized Policy Optimization (EMPO)

基于 GRPO 框架，但用 semantic entropy 替代外部 reward：

1. 对每个 question $q$，从当前 policy $\pi_\theta$ 采样 $G$ 个输出
2. 将输出聚类为 meaning clusters
3. 每个输出的 reward = 其所属 cluster 的概率：$r_i = p(c_j | q)$，其中 $l(o_i) = c_j$
4. 计算 group-normalized advantage：$A_i = (r_i - \text{mean}) / \text{std}$
5. 优化 clipped policy gradient 目标（无 KL 约束，$\epsilon$ clip 保证稳定性）

### 3.3 Entropy Thresholding 防止 Reward Hacking

设置双阈值 $\delta_{low} < H < \delta_{high}$：
- **过滤高熵问题**：模型高度不确定，输出不可靠
- **过滤低熵问题**：模型已很确信，继续优化存在 overconfidence 风险且收益有限

最终目标函数：
$$\mathcal{J}_{\text{EMPO}} = \mathbb{E}\left[\frac{1}{|G|}\sum_{i=1}^{|G|} \min(A_i, \text{clip}(1-\epsilon, 1+\epsilon) A_i)\right], \quad \text{s.t. } \delta_{low} < H < \delta_{high}$$

---

## 4. 数据集

### 训练数据
| 任务类型 | 数据集 | 规模 |
|---------|--------|------|
| 数学推理 | NuminaMath-CoT（随机采样） | 20K prompts |
| 自然推理 | Natural Reasoning (Facebook) | ~18K prompts（过滤后） |

训练数据仅包含 **unlabeled questions**，不使用任何答案或 reasoning traces。

### 评估 Benchmarks
| 任务类型 | Benchmarks |
|---------|-----------|
| 数学推理 | MATH, Minerva Math, AMC23, OlympiadBench, AIME24 |
| 自然推理 | MMLU-Pro, GPQA |

---

## 5. 评估指标与主要结果

**评估指标**：pass@1 accuracy（greedy decoding, zero-shot）

### 数学推理结果 (Table 1)

| 模型 | 监督信号 | MATH | Minerva | Olympiad | AIME24 | AMC23 | Avg. |
|------|---------|------|---------|----------|--------|-------|------|
| Qwen2.5-Math-1.5B Base | None | 52.2 | 10.7 | 25.2 | 10.0 | 42.5 | 28.1 |
| Qwen2.5-Math-1.5B w/GRPO | {q, a} | 75.2 | 32.0 | 33.6 | 16.7 | 52.5 | 42.0 |
| **Qwen2.5-Math-1.5B w/EMPO** | **{q}** | **73.0** | **32.4** | **36.6** | **13.3** | **55.0** | **42.1** |
| Qwen2.5-Math-7B Base | None | 64.8 | 15.1 | 26.7 | 6.7 | 40.0 | 30.7 |
| Qwen2.5-Math-7B w/GRPO | {q, a} | 77.8 | 39.7 | 39.1 | 20.0 | 57.5 | 46.9 |
| **Qwen2.5-Math-7B w/EMPO** | **{q}** | **78.0** | **40.4** | **37.3** | **20.0** | **65.0** | **48.1** |

关键发现：
- EMPO（仅用 questions）在 7B 模型上 **平均超越 GRPO**（48.1 vs 46.9），GRPO 需要 golden answers
- 1.5B 模型上 EMPO 与 GRPO 持平（42.1 vs 42.0）
- 相比 Base model，EMPO 在 7B 上提升 +17.4 个百分点

### 自然推理结果 (Table 2)

| 模型 | MMLU-Pro Avg. | GPQA |
|------|--------------|------|
| Qwen2.5-7B Base | 32.1 | 23.5 |
| Qwen2.5-7B w/GRPO | 33.8 | - |
| **Qwen2.5-7B w/EMPO** | **34.6** | **28.8** |
| Qwen2.5-14B Base | 30.6 | - |
| **Qwen2.5-14B w/EMPO** | **41.6** | **35.3** |

- MMLU-Pro 上 7B 模型提升 32.1% → 50.1%（+18.0）
- GPQA 上 7B 模型提升 15.9% → 28.8%（+12.9）

### 训练动态

训练过程中 semantic entropy 持续下降，同时 semantic probability reward 和 accuracy reward 均持续上升，三者变化趋势高度一致，验证了 semantic entropy 作为无监督 reward 信号的有效性。

### 核心洞察

作者通过 pass@k 分析发现，EMPO 和 GRPO 主要提升了模型在少量采样时找到正确推理路径的效率（pass@1 提升显著），但当 $k$ 增大时，Base model 的 pass@k 逐渐追平甚至超过 RL-trained model。这表明 **RL post-training（包括无监督的 EMPO）本质上是在引导和优化模型对 pre-training 阶段已习得的推理能力的使用效率，而非教会模型全新的推理技能**。
