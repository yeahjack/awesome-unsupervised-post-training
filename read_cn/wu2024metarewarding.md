# Meta-Rewarding: Meta-Rewarding Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge
**Authors:** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar (Meta FAIR & UC Berkeley & NYU)
**ArXiv:** arXiv:2407.19594
**Date:** 2024-07

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Meta-Rewarding | Pref. Opt. | training-time | Semantic |

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

**Family IV — Internal Evaluator Bootstrapping (meta-judge self-rewarding)**

本文是 Self-Rewarding Language Models (Yuan et al., 2024c) 的直接扩展，属于 "Internal Evaluator Bootstrapping" 家族中 **meta-judge self-rewarding** 子类。在 Self-Rewarding 框架中，同一个 LLM 既充当 actor（生成回答）又充当 judge（评价回答），并通过迭代 DPO 训练持续自举改进。然而，Self-Rewarding 的学习目标仅优化 actor 的回答质量，而 **忽视了 judge 能力的提升**——如果 judge 不改进，迭代训练很快就会饱和甚至出现 reward hacking。

Meta-Rewarding 的核心创新在于引入 **第三个角色——meta-judge**（元裁判），让模型自行评判自己的 judgments 质量，从而构建 judge preference pairs 来显式训练 judge 能力。这使得模型在每轮迭代中同时提升 actor 和 judge 两方面的能力，形成更强的自举循环。整个过程无需额外人类标注，所有训练信号（response、judgment、meta-judgment）均由模型自身生成。

---

## 2. 解决的问题

Self-Rewarding 框架存在一个关键瓶颈：

1. **Judge 能力饱和（Judge Saturation）**：Self-Rewarding 的 DPO 训练仅优化 actor 生成更好的 response，但 judge 的能力并未被显式训练提升。随着迭代推进，judge 的评估准确度停滞，无法为越来越好的 actor 提供有效的区分信号，导致训练快速饱和。
2. **Reward Hacking 风险**：如果 judge 能力不提升，actor 可能学会 "欺骗" 静态的 judge（例如通过生成冗长回答来获取高分），而非真正提高回答质量。
3. **Length Bias（长度偏好）**：judge（包括 meta-judge）倾向于偏好更长的回答/判断，这会导致每轮迭代后回答长度不断膨胀（length explosion），降低 length-controlled win rate。

Meta-Rewarding 提出的解决方案：
- 引入 **meta-judge** 角色，让模型评判自己的 judgments 质量，构建 judge preference data，与 actor preference data 一起联合进行 DPO 训练
- 引入 **length-control 机制**，在 preference pair 选择阶段优先选择更短但高质量的 response 和 judgment

---

## 3. 方法介绍

### 3.1 总体框架

Meta-Rewarding 的迭代训练方案中，模型在每轮迭代中扮演三个角色：
1. **Actor（演员）**：根据 prompt 生成 response
2. **Judge（裁判）**：使用 LLM-as-a-Judge prompting 对 response 进行 pointwise 评分（5-point scale）
3. **Meta-Judge（元裁判）**：使用 LLM-as-a-Meta-Judge prompting 对两个 judgments 进行 pairwise 比较，选出更好的 judgment

每轮迭代生成两类 preference data：
- **Actor preference pairs**：基于 judge 评分选出的 chosen/rejected responses → 训练 actor 能力
- **Judge preference pairs**：基于 meta-judge 比较选出的 chosen/rejected judgments → 训练 judge 能力

两类 preference data 合并后通过 DPO 联合训练，产生下一轮迭代的模型。

### 3.2 Actor Preference Dataset 创建

#### Step 1: 采样 Response
对于给定的 prompt 集合，使用当前模型为每个 prompt 生成 $K = 7$ 个 response（temperature = 0.8, top-p = 0.95），每轮共生成约 35,000 个 responses，去除重复后保留。

#### Step 2: Judge 评分
对每个 response 生成 $N = 11$ 个 judgments（使用 pointwise 5-point rubric prompt），jduge 输出 chain-of-thought 推理和最终 1-5 分评分。通过正则表达式解析分数，丢弃格式不符的 judgments，最终得分为所有有效 judgment 分数的平均值。

#### Step 3: Preference Data Selection with Length-Control
传统方法直接选择最高分 $S_{\max}$ 和最低分 $S_{\min}$ 的 response 作为 chosen/rejected pair，但这会导致 length explosion（judge 偏好长回答）。

Meta-Rewarding 引入 **quality tier 参数 $\rho \in [0,1]$** 来平衡质量和长度：
- **Chosen response $y_c$**：在分数位于 top tier $[(1-\rho)S_{\max} + \rho S_{\min}, S_{\max}]$ 的 responses 中，选择 **最短的** response
- **Rejected response $y_r$**：在分数位于 bottom tier $[S_{\min}, (1-\rho)S_{\min} + \rho S_{\max}]$ 的 responses 中，选择 **最长的** response
- $\rho = 0$ 时退化为纯分数选择（无 length-control）

### 3.3 Judge Preference Dataset 创建

这是 Meta-Rewarding 相对于 Self-Rewarding 的核心创新。

#### Step 1: Response Selection（选择最不确定的 response）
对每个 prompt 的所有 response，计算其 $N$ 个 judgment 分数的 **方差**，选择方差最大的 response $y$——即 judge 最不确定的 response——用于 judge 训练（方差相同则随机）。

#### Step 2: Pairwise Meta-Judge Evaluations
对选中的 response $y$，取其 $N$ 个 judgments $\{j_1, ..., j_N\}$，对所有不同 judgment 对 $(j_m, j_n)$ 使用 Meta-Judge prompt（见 Figure 2）进行 pairwise 比较。Meta-Judge prompt 包含：
- 原始 prompt $x$
- Response $y$
- 两个 judgments（Judgment A 和 Judgment B）
- 评分 rubric

模型需要输出 chain-of-thought 推理 + 选择 winner。

**位置偏差缓解**：为缓解 positional bias（meta-judge 可能偏好第一位置的 judgment），每对 judgments 评估两次（交换位置）。使用加权 Elo scoring：

$$\text{win\_score}(j) = w_1 \cdot \text{win}_{1\text{st}}(j) + w_2 \cdot \text{win}_{2\text{nd}}(j)$$

其中 $w_1 < w_2$，给予在第二位置获胜更高权重，以补偿第一位置的天然优势。

#### Step 3: Preference Pair 选择
选择 Elo score 最高的 judgment 作为 chosen $j_c$，最低的作为 rejected $j_r$。

**Judge length bias 过滤**：meta-judge 同样存在 length bias（偏好冗长的 judgment），因此额外加入过滤步骤，丢弃 chosen judgment 长度超过阈值的 preference pairs。

### 3.4 训练流程

以 Llama-3-8B-Instruct 为 seed model：

1. **SFT on EFT**：先在 Evaluation Fine-Tuning (EFT) 数据集（来自 OpenAssistant 的 ranked human responses）上做 SFT，初始化模型的 LLM-as-a-Judge 能力

2. **Iteration 1**：从 SFT 模型生成 actor + judge preference pairs → DPO 训练（初始化 from SFT）→ 得到 $M_1$
   - $\rho = 0$（无 length control），过滤 chosen response 长度 > 2500 字符
   - Judge data: 过滤 chosen judgment 长度 > 1100

3. **Iteration 2**：从 $M_1$ 生成 actor + judge preference pairs → DPO 训练 $M_1$ → 得到 $M_2$
   - $\rho = 0.32$，judge data 阈值 1000

4. **Iteration 3**：从 $M_2$ 生成 **仅 actor** preference pairs → DPO 训练 $M_2$ → 得到 $M_3$
   - $\rho = 0.32$（仅做 actor 训练，不再做 judge 训练）

5. **Iteration 4**：从 $M_3$ 生成 **仅 actor** preference pairs → DPO 训练 $M_3$ → 得到 $M_4$
   - $\rho = 0.4$

关键观察：**只在前两轮迭代使用 meta-judge 进行 judge 训练**，后续两轮仅做 actor 训练。这是因为 meta-judge 在后续迭代中出现了严重的 score bias 和 positional bias（见 Table 5），导致 judge 训练质量下降。

### 3.5 DPO 训练细节
- Learning rate: $5 \times 10^{-6}$, $\beta = 0.1$, global batch size 32
- 10 epochs, cosine learning rate scheduling
- 每轮从 20,000 seed prompts 中采样 5,000 个 prompts
- 每个 prompt 生成 $K = 7$ responses, 每个 response 生成 $N = 11$ judgments

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 训练 Prompt | Llama-2-70B-Chat 生成的 20K prompts | 使用 8-shot prompt 生成，每轮采样 5,000 |
| SFT | EFT (Evaluation Fine-Tuning) 数据集 | 来自 OpenAssistant 的 ranked human responses，用于初始化 LLM-as-a-Judge 能力 |
| Actor 评估 | AlpacaEval 2 (805 prompts) | 与 GPT-4-Turbo 比较的 LC win rate |
| Actor 评估 | Arena-Hard | 复杂/困难问题，与 Chatbot Arena 相关性最高 |
| Actor 评估 | MT-Bench (8 categories) | 多轮对话能力评估 |
| Judge 评估 | OpenAssistant test set (190 samples, 580 responses) | 评估 judge 与人类标注的 Spearman correlation 和 agreement |

---

## 5. 评估指标与主要结果

### 评估指标

**Actor 评估：**
- **AlpacaEval 2 LC win rate**：与 GPT-4-Turbo 比较的 length-controlled win rate（主要指标）
- **AlpacaEval 2 win rate**：原始 win rate
- **Arena-Hard score**：与 GPT-4 比较的 win rate（95% CI）
- **MT-Bench score**：Turn 1/Turn 2 分数

**Judge 评估：**
- **GPT-4 Chosen Pairs Agreement**：judge 与 GPT-4 在人类数据上的偏好一致率
- **Self-Chosen Pairs Agreement**：judge 与 GPT-4 在自生成数据上的偏好一致率
- **Agreement without Ties**：去除 tie 后的一致率
- **Human Agreement**：judge 与 OpenAssistant 人类标注的 agreement 和 Spearman correlation

### 主要结果

#### AlpacaEval 2（Table 1）
| Model | LC Win Rate | Win Rate | Length |
|-------|------------|----------|--------|
| Llama-3-8B-Instruct (Seed) | 22.92% | 22.57% | 1899 |
| SFT on EFT | 25.47% | 25.10% | 1943 |
| Self-Rewarding + LC Iter 1 | 26.93% | 27.12% | 1983 |
| Self-Rewarding + LC Iter 2 | 30.38% | 29.77% | 1940 |
| Self-Rewarding + LC Iter 3 | 34.87% | 34.59% | 1967 |
| Self-Rewarding + LC Iter 4 | 35.49% | 35.37% | 2005 |
| **Meta-Rewarding Iter 1** | 27.85% | 27.62% | 1949 |
| **Meta-Rewarding Iter 2** | 32.66% | 33.29% | 2001 |
| **Meta-Rewarding Iter 3** | 35.45% | 37.24% | 2064 |
| **Meta-Rewarding Iter 4** | **39.44%** | **39.45%** | 2003 |

#### Arena-Hard（Table 2）
| Model | Score | 95% CI | Length |
|-------|-------|--------|--------|
| Llama-3-8B-Instruct (Seed) | 20.6% | (-2.0, 1.8) | 2485 |
| SFT on EFT | 24.2% | (-2.0, 1.8) | 2444 |
| Self-Rewarding + LC Iter 4 | 27.3% | (-2.0, 2.2) | 2448 |
| **Meta-Rewarding Iter 4** | **29.1%** | (-2.3, 2.1) | 2422 |

#### MT-Bench（Table 6, Appendix）
| Model | Score | Turn 1 | Turn 2 | Length |
|-------|-------|--------|--------|--------|
| Llama-3-8B-Instruct | 8.116 | 8.319 | 7.911 | 1568 |
| Self-Rewarding + LC Iter 4 | 8.028 | 8.381 | 7.675 | 1539 |
| **Meta-Rewarding Iter 3** | **8.341** | **8.731** | **7.950** | 1596 |
| Meta-Rewarding Iter 4 | 8.288 | 8.738 | 7.838 | 1592 |

#### Judge Agreement with GPT-4（Table 3）
| Model | GPT-4 Chosen | GPT-4 w/o Tie | Self-Chosen | Self w/o Tie |
|-------|-------------|---------------|-------------|--------------|
| SFT on EFT | 51.48% | 51.79% | 61.66% | 73.51% |
| Self-Rewarding Iter 4 | 52.97% | 53.12% | 64.44% | 78.42% |
| **Meta-Rewarding Iter 3** | **58.63%** | **61.24%** | 63.43% | 76.80% |
| **Meta-Rewarding Iter 4** | 57.44% | 59.54% | **64.50%** | **79.33%** |

#### Judge Agreement with Human（Table 7, Appendix）
| Model | Agreement | Agree w/o Tie | Spearman corr. |
|-------|-----------|--------------|----------------|
| SFT on EFT | 63.20% | 64.59% | 0.321 |
| Self-Rewarding Iter 2 | 64.14% | 67.17% | 0.347 |
| **Meta-Rewarding Iter 2** | **66.64%** | **68.33%** | **0.382** |

### 关键发现

1. **Meta-Rewarding 显著优于 Self-Rewarding**：在 AlpacaEval 2 上，Meta-Rewarding 4 轮迭代达到 **39.44% LC win rate**，而同等设置的 Self-Rewarding + LC 仅 35.49%（+3.95%），甚至超越了 GPT-4-0314，接近 Claude Opus 水平。这从一个仅 8B 参数、无额外人类数据的模型出发，是非常显著的成绩。

2. **Meta-Rewarding 也超越了使用外部 reward model 的方法**：SPPO（使用在大量人类 + GPT-4 数据上训练的 reward model）在 AlpacaEval 2 上达到 38.77% LC win rate，仍低于 Meta-Rewarding 的 39.44%——尽管后者完全是 self-improvement，不依赖外部 reward model。

3. **Judge 能力确实得到提升**：Meta-Rewarding 在 GPT-4 Chosen Pairs 上的 Agreement w/o Tie 从 SFT 的 51.79% 提升到 61.24%（Iter 3），而 Self-Rewarding 仅达到 55.90%。在人类标注上，Meta-Rewarding Iter 2 的 Spearman correlation 达到 0.382，显著高于 Self-Rewarding 的 0.347。

4. **Length-control 机制有效防止 length explosion**：回答长度在训练迭代中保持稳定（约 1949–2064 字符），未出现显著膨胀。对比实验显示 $\rho = 0$（无 length control）时 length 膨胀至 2212 字符且 LC win rate 下降。

5. **Meta-judge 在后续迭代出现 bias 退化**：Table 5 显示 Iteration 2 的 meta-judge 出现严重 score bias（97.68% 选择高分 judgment）和 positional bias（68.11%），导致 judge 分数分布高度集中于满分附近（均值从 4.1 升至 4.7+），这限制了后续迭代继续进行 judge 训练。

6. **不牺牲多轮对话能力**：尽管仅在 single-turn 数据上训练，MT-Bench Turn 2 score 在 Meta-Rewarding 最终迭代（7.838）相比 seed（7.911）仅下降 0.073，而 Turn 1 从 8.319 提升到 8.738。

7. **外部 Reward Model 并不总是更好**：使用 Starling-RM-34B 作为外部 reward model 替代 self-judging，第一轮 LC win rate 仅 24.63%，低于 Meta-Rewarding 的 27.85%，可能由于外部 RM 的 length bias 更严重。

8. **17/18 个 AlpacaEval 类别得到改进**：细粒度分析显示 Meta-Rewarding 在 Science、Gaming、Literature 等需要大量知识和推理的类别上改进显著，仅 Travel 和 Mathematics 改进较小。
