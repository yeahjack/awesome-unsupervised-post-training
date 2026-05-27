# Self-Training Large Language Models with Confident Reasoning

> **加入 Survey 时间：** 2026-03-11

**Paper:** Self-Training Large Language Models with Confident Reasoning
**arXiv:** 2505.17454
**Authors:** Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn
**Affiliations:** POSTECH, KAIST

---

| 属性 | 值 |
|---|---|
| Method | Confident ST (CORE-PO) |
| Carrier | Direct Opt. (online DPO) |
| Regime | Training-time |
| Level | Traj. |

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

**Family III — Self-Generated Target Bootstrapping，子类：rationale / critique / latent-thought self-training**

Confident ST (CORE-PO) 属于 Family III，理由如下：

- **Self-confidence 驱动的 pseudo-target 选择：** 模型对每个无标注问题采样 N 个输出 $\{(r_i, a_i)\}_{i=1}^N$，利用自身的 reasoning-level confidence $C_\theta(r|x)$（通过 P(True) 衡量）来识别高质量的 reasoning path，将其作为 pseudo-target 回灌模型。
- **Self-Generated Target Bootstrapping 本质：** 训练信号完全来自模型自身生成的 reasoning path 及其 self-assessed confidence score，不依赖外部 ground truth label、external verifier 或人工标注。模型通过自评估的 confidence 来构建 preference pair，高 confidence 的 reasoning path 被视为 preferred output。
- **Rationale self-training 子类：** 区别于仅关注 answer-level confidence 的方法（如 SC-PO），本文核心创新在于评估 reasoning path 本身的质量（而非仅看最终答案的 confidence），属于对 rationale（推理过程）质量进行 self-training 的范式。
- **Carrier — Direct Opt. (DPO)：** 使用 online Direct Preference Optimization 将 high-confidence reasoning path 偏好注入模型参数。
- **Level — Traj.：** 优化目标是完整的 reasoning–answer trajectory $s = [r, a]$，confidence 评估粒度也是整条 reasoning path（monolithic P(True)）或逐 statement 平均（statement-wise P(True)）。

---

## 2. 解决的问题

现有 confidence-based self-training 方法（如 Huang et al., 2023; Prasad et al., 2024; Zhang et al., 2024b）存在一个核心缺陷：**仅依赖 answer-level confidence**（通过 majority voting 估计），忽视了 reasoning path 本身的质量。

具体问题包括：

1. **错误推理 → 正确答案的 "巧合" 现象：** LLM 经常生成错误的 reasoning path，但恰好得到正确答案（尤其在多选题场景），导致 answer-level confidence 很高。如 Figure 1 所示，模型可能用错误的推理（如 "(b)-(d) are boiling points"）得到正确答案，训练后这种错误推理模式会被强化。
2. **推理质量退化：** 当模型学会偏好这些 "碰巧正确" 的推理路径后，其推理能力反而下降——在新问题上会产生系统性错误（如将 32°C 标为 boiling point）。
3. **标注成本限制：** 高质量的人工标注 reasoning path 成本高昂，需要 self-training 方案来替代。

本文的核心洞察是：**需要从 answer-level confidence 上升到 reasoning-level confidence**，以更准确地识别高质量的 reasoning path 用于 self-training。

---

## 3. 方法介绍

### 3.1 核心动机与观察

作者在 GPQA 数据集上进行了观察实验（Figure 2），得出两个关键发现：

- **Observation 1：** 具有高 answer-level confidence 的 reasoning path 往往是错误的，即使最终答案正确。answer accuracy 与 reasoning accuracy 之间存在明显 gap。
- **Observation 2：** 高 reasoning-level confidence $C_\theta(r|x)$ 的输出倾向于产生更少的推理错误，且同时伴随高 answer-level accuracy。

这两个观察支持了将 reasoning-level confidence 引入 self-training 的必要性。

### 3.2 CORE-PO 方法流程

CORE-PO（**CO**nfidence **RE**asoning — **P**olicy **O**ptimization）的整体流程如 Figure 3 和 Algorithm 1 所示：

**Step 1 — 多路输出采样：**
给定问题 $x$，LLM $M_\theta$ 生成 $N=5$ 个输出 $\{(r_i, a_i)\}_{i=1}^N$，每个输出包含 reasoning path $r_i$ 和 answer $a_i$，采样参数为 $T=1.0$、$\text{top-p}=0.9$。

**Step 2 — Reasoning-level confidence 估计：**
使用 **P(True)** 方法（Kadavath et al., 2022）估计每条 reasoning path 的 confidence。具体有两种实现方式：

- **Monolithic P(True)：** 一次性评估整条 reasoning path 的正确性，即向 LLM 提问 "Is the selected reasoning correct?"，取模型返回 "True" 的概率作为 $C_\theta(r|x)$。提问时还会附加 $M=4$ 条随机生成的 reasoning 作为对比 context。
- **Statement-wise P(True)：** 将 reasoning path 拆分为多个 statement $r = [r_1, \ldots, r_T]$，逐步评估每个 statement 的正确性，然后取平均：
  $$C_\theta(r|x) = \frac{1}{T} \sum_{t=1}^{T} C_\theta(r_t | x, r_1, \ldots, r_{t-1})$$

同时估计 answer-level confidence $C_\theta(a|x, r)$（基于 reasoning 和 question 的条件下答案是否正确的 P(True)），最终的综合 confidence 为：
$$C_\theta(a, r|x) = C_\theta(a|x, r) \cdot C_\theta(r|x)$$

**Step 3 — Preference pair 构建与 DPO 训练：**
对同一问题的多个输出，按综合 confidence 排序，构建 preference pair $(s_w, s_l)$，其中 $s_w$ 为高 confidence 输出，$s_l$ 为低 confidence 输出。使用 online DPO 优化目标：
$$\mathcal{L} = \log \sigma \left( \beta \log \frac{M_\theta(s_l|x)}{M_{\text{ref}}(s_l|x)} - \beta \log \frac{M_\theta(s_w|x)}{M_{\text{ref}}(s_w|x)} \right)$$

其中 $M_{\text{ref}}$ 为固定的 reference model（训练期间不更新），$\beta = 0.1$，优化方向为让模型对 high-confidence reasoning path 赋予更高 likelihood。

### 3.3 实现细节

- **Base models：** Llama3.1-8B-Instruct 和 Qwen2.5-7B-Instruct
- **LoRA 适配：** rank = 128，$\alpha = 256$
- **训练采样：** 每个问题生成 $N=5$ 个输出，$T=1.0$，$\text{top-p}=0.9$
- **推理采样（inference-time scaling）：** 随机采样 8 个输出，$T=0.7$，$\text{top-p}=0.9$
- **默认实现使用 monolithic P(True)** 估计 reasoning-level confidence
- **Gradient clipping：** max norm = 1.0
- **Checkpoint 选择：** 每 200 步保存，选择 ARC-Challenge validation 上最优模型
- **硬件：** 4× NVIDIA A100 SXM4 80GB
- **训练时长：** 2–4 天
- **框架：** transformers + trl + accelerate

---

## 4. 数据集

### 4.1 In-distribution 训练/评估数据集

| 数据集 | 类型 | 训练集大小 | 测试集大小 | 评估指标 |
|---|---|---|---|---|
| **GSM8K** | 多步算术推理 | 7.4K | 1.3K | 数值答案 accuracy |
| **ARC-Challenge** | 多选科学常识 | 1.1K (+ 0.3K val) | 1.1K | 选项 accuracy |
| **GPQA** | 研究生级多选科学推理 | 420 (main) | 509 (extended) | 选项 accuracy |
| **MATH** | 高中数学竞赛 | 7.5K | 0.7K (Level-5) | 数值答案 accuracy |

### 4.2 Out-of-distribution 评估数据集

| 数据集 | 类型 | 测试集大小 | 说明 |
|---|---|---|---|
| **CRUXEval (CRUXout)** | 代码理解与执行 | 0.8K | 预测 Python 函数输出 |
| **Game of 24** | 算术推理 | 1.3K | 用四则运算凑 24 点 |

所有训练数据集的 **question 均无需 ground truth label**，完全依靠模型自身的 confidence 评估构建训练信号。

---

## 5. 评估指标与主要结果

### 5.1 评估方式

- **Greedy decoding：** 直接评估模型单次 greedy 输出的 accuracy
- **Inference-time scaling：** 采样 8 个输出，用各方法对应的 self-assessment 策略选择最优输出
  - SR-PO → Linguistic self-assessment
  - SC-PO → Majority voting (SC)
  - CORE-PO → P(True|r, a) confidence

### 5.2 主要结果 — Llama3.1-8B-Instruct (Table 1)

| Method | Decoding | GSM8K | ARC-Challenge | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|---|
| Base (无 fine-tuning) | Greedy | 84.2 | 84.5 | 32.4 | 22.6 |
| Base | P(True\|r,a) | 89.7 | 87.0 | 34.5 | 25.2 |
| SR-PO | Greedy | 85.2 | 86.2 | 34.3 | 19.8 |
| SC-PO | Greedy | 85.7 | 86.0 | 33.7 | 25.1 |
| SC-PO | SC | 89.7 | 87.5 | 34.5 | 29.4 |
| **CORE-PO** | **Greedy** | **86.8** | **87.5** | **35.5** | **24.6** |
| **CORE-PO** | **P(True\|r,a)** | **90.5** | **89.2** | **36.1** | **29.8** |

### 5.3 主要结果 — Qwen2.5-7B-Instruct (Table 2)

| Method | Decoding | GSM8K | ARC-Challenge | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|---|
| Base (无 fine-tuning) | Greedy | 90.0 | 89.1 | 30.6 | 45.4 |
| SC-PO | Greedy | 91.0 | 91.0 | 34.3 | 49.6 |
| SC-PO | SC | 93.0 | 92.0 | 36.3 | 55.7 |
| **CORE-PO** | **Greedy** | **91.3** | **92.2** | **37.5** | **49.6** |
| **CORE-PO** | **P(True\|r,a)** | **93.5** | **92.8** | **38.5** | **55.8** |

### 5.4 Reasoning-level 质量提升 (Table 3, Llama3.1-8B-Instruct)

| Method | Decoding | GSM8K Conf. | GSM8K Reason Acc. | ARC Conf. | ARC Reason Acc. |
|---|---|---|---|---|---|
| Base | Greedy | 0.89 | 84.2 | 0.84 | 79.2 |
| Base | P(True\|r,a) | 0.98 | 89.7 | 0.94 | 81.2 |
| CORE-PO | Greedy | 0.94 | 86.8 | 0.95 | 81.5 |
| CORE-PO | P(True\|r,a) | 0.99 | 90.4 | 0.99 | 84.9 |

CORE-PO 不仅提升了 answer accuracy，还显著提升了 **reasoning-level confidence 和 reasoning-level accuracy**，说明模型确实学会了生成更高质量的推理路径。

### 5.5 Out-of-distribution 泛化 (Table 4, Llama3.1-8B-Instruct)

| Method | Decoding | CRUXout | Game of 24 |
|---|---|---|---|
| Base | Greedy | 34.8 | 7.2 |
| SC-PO | Greedy | 43.8 | 8.3 |
| SC-PO | SC | 50.0 | 11.9 |
| **CORE-PO** | **Greedy** | **47.1** | **18.8** |
| **CORE-PO** | **P(True\|r,a)** | **48.0** | **22.1** |

在 OOD 任务上，CORE-PO 展现出更好的泛化能力，尤其在 Game of 24 任务上大幅领先 SC-PO（22.1 vs 11.9，+10.2 绝对提升）。

### 5.6 Ablation 结果

**Monolithic vs. Statement-wise P(True) (Table 5)：**

| 变体 | GSM8K | ARC | GPQA_ext | MATH_lv5 |
|---|---|---|---|---|
| Base | 84.2 | 84.5 | 32.4 | 22.6 |
| Monolithic P(True) | 86.8 | 87.5 | **35.5** | 24.7 |
| Statement-wise P(True) | **88.5** | **88.0** | 34.1 | **25.3** |

两种方式各有优劣，但均一致优于 base model，说明性能提升源于 "偏好高 reasoning-level confidence" 的核心哲学，而非特定的 confidence 估计实现。

**加入 Ground-truth 答案的实验 (Table 6, ARC-Challenge)：**

| Reward Signal | Decoding | Answer Acc. | Reason Acc. |
|---|---|---|---|
| Answer Acc. only | Greedy | 87.4 | 73.6 |
| Answer Acc. + Reason Conf. | Greedy | **88.3** | **81.9** |
| Answer Acc. + Reason Conf. | P(True\|r,a) | **90.1** | **85.6** |

即使在有 ground-truth 答案的传统 fine-tuning 中，加入 reasoning-level confidence 也能显著提升 reasoning accuracy（+8.3 绝对提升），表明该方法的适用性不局限于纯 self-training 场景。

### 5.7 关键图表说明

- **Figure 1：** 展示了现有 confidence-based self-training 方法的核心缺陷。以 "水分子形成刚性晶格" 的温度估计题为例，多条推理路径虽然得到正确答案 (a) 0°C，但其中一条错误推理（"(b)-(d) are boiling points"）因 answer-level confidence 高而被选为训练目标，导致 self-trained model 在新问题上产生 "32°C is the boiling point" 的错误。CORE-PO 通过 reasoning-level confidence 避免选择此类错误推理。
- **Figure 2：** 展示 answer accuracy 和 reasoning accuracy 随着 Top-N% confidence 筛选的变化。按 reasoning-level confidence 排序时，reasoning accuracy 与 answer accuracy 更加一致；而按 answer-level confidence 排序时两者存在显著 gap。
- **Figure 3：** CORE-PO 方法概览流程图。LLM 生成多个 (reasoning, answer) 输出 → 对每条 reasoning path 估计 $C_\theta(r|x)$ → 构建 preference pair → 用 DPO fine-tune 模型偏好高 confidence reasoning。
- **Table 7：** 定性示例对比。仅用 Answer Acc. fine-tune 的模型会生成冗长的逐选项分析（且包含错误 statement），而加入 Reason Conf. 后模型生成简洁准确的推理。
