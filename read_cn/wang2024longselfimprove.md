# Long Self-Improve: Large Language Models Can Self-Improve in Long-context Reasoning

> **加入 Survey 时间：** 2026-03-11

**Paper:** Large Language Models Can Self-Improve in Long-context Reasoning
**Authors:** Siheng Li, Cheng Yang, Zesen Cheng, Lemao Liu, Mo Yu, Yujiu Yang, Wai Lam (CUHK, Peking University, Tsinghua University, Tencent)
**ArXiv:** 2411.08147
**Date:** 2024-11

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| Long Self-Improve | Direct Opt. | training-time | Semantic |

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

SEALONG 属于 Unsupervised Post-Training 中的 self-generated-target bootstrapping 分支，核心理由如下：

1. **无需外部标注或专家模型：** 整个流程不依赖人工标注的答案，也不依赖 GPT-4o 等更强外部模型的输出。所有训练信号均来源于模型自身的多次采样输出。
2. **自生成 + 内部准则筛选：** 对每个长上下文问题，模型以 temperature sampling 产生 N 条推理轨迹，然后通过 Minimum Bayes Risk (MBR) decoding——一种基于输出间 semantic consistency（语义一致性）的内部评估准则——为每条输出打分。高一致性的输出被认为更可能是正确的推理轨迹（低一致性的更可能是 hallucination）。
3. **筛选结果直接作为学习目标：** MBR 选出的最高分输出作为 SFT 的学习目标，或者高低分输出组成 preference pair 用于 ORPO preference optimization。这构成了典型的 "self-generated synthetic target → direct optimization" 范式。
4. **属于 reasoning synthesis：** 生成的目标是完整的推理轨迹（reasoning trajectory），包含分步推理过程、信息定位与答案推导，属于 reasoning / plan / curriculum synthesis 类别。

---

## 2. 解决的问题

**长上下文推理能力不足且现有改进方法依赖外部监督。**

具体而言：
- LLM 在长上下文检索（如 needle-in-a-haystack）上已近乎完美，但在需要**跨文档多步推理**的长上下文任务上仍表现不佳，存在显著性能下降。
- 现有改进长上下文推理的方法主要依赖两类外部监督：(1) 人类专家标注（昂贵且难以规模化）；(2) 高级模型（如 GPT-4o）合成数据（受限于已有模型能力，无法突破上限）。
- 论文提问：**LLM 能否在不借助任何外部监督的情况下，自我提升长上下文推理能力？**

论文发现了两个关键动机：
1. **Prompting 策略的重要性：** 从 default prompting 切换到 plan-and-solve prompting 后，Llama-3.1-8B-Instruct 在 2WikiMQA 上从 66.0 提升到 82.0，说明模型内部已具备较强潜力，只是未被充分激发。
2. **Oracle 性能远超 greedy search：** 对每个问题采样 128 个输出后，oracle sample（至少有一个正确）的准确率超过 90%，远超 greedy search，表明多次采样中蕴含大量正确推理轨迹。

---

## 3. 方法介绍

SEALONG（**S**elf-improving method for r**EA**soning over **LONG** contexts）分为两个阶段：自监督信号创建（Self-supervision）和微调（Fine-tuning）。

### 3.1 Stage 1: Self-supervision 创建

#### 3.1.1 多输出采样

对每个 query-context 对，使用 **plan-and-solve prompting** 策略，以 temperature = 0.7 采样 $N$ 条推理轨迹。默认 $N = 32$。

Prompt 模板：
> {context}
> {input}
> Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step-by-step.

#### 3.1.2 MBR Scoring

核心思想：**正确的推理轨迹通常在语义上更一致**（遵循相似的规划步骤、引用相同的上下文信息），而错误轨迹更可能是 hallucination，表现为与其他输出不一致。

使用 Minimum Bayes Risk (MBR) 框架为每条输出打分。输出 $y$ 的分数定义为其在模型分布下的期望效用：

$$s(y) = \mathbb{E}_{y^* \sim \pi_\theta(y|x)}[u(y, y^*)]$$

通过 Monte Carlo 方法用 $N$ 个采样输出近似：

$$s(y) \approx \frac{1}{N} \sum_{i=1}^{N} u(y, y_i^*)$$

**效用度量 $u(y, y^*)$** 采用 sentence embedding similarity：

$$u(y, y^*) = \text{Sim}(\text{Emb}(y), \text{Emb}(y^*))$$

使用轻量级 RoBERTa-based 模型（实验中使用 **jina-embeddings-v3**）计算 embedding，用内积衡量相似度。

分数最高的输出即为 **MBR decoding output**（$y_w$），作为 preferred output；随机选取一个低分输出作为 less-preferred output（$y_l$）。

#### 3.1.3 Scoring 方法对比（Tab. 7）

| 方法 | HotpotQA | MuSiQue | 2WikiMQA |
|------|----------|---------|----------|
| Greedy Search | 64.0 | 49.5 | 82.0 |
| Random | 61.0 | 50.5 | 79.5 |
| Reference-free Self-evaluation | 64.0 | 51.5 | 83.0 |
| MBR-ROUGE | 66.5 | 53.5 | 85.0 |
| MBR-BERTScore | 67.5 | 50.0 | 86.5 |
| Reference-based Self-evaluation | 63.5 | 51.5 | 84.5 |
| **MBR-Sentence Embedding** | **67.5** | **56.0** | **88.0** |

MBR-based 方法显著优于 reference-free self-evaluation，原因是当前 LLM 的 self-evaluation 能力在长上下文场景中尤为有限。Sentence embedding 进一步引入更丰富的语义信息，表现最佳。

### 3.2 Stage 2: Fine-tuning

提供两种微调策略：

#### 3.2.1 Supervised Fine-tuning (SFT)

以 MBR decoding output 为目标，最小化负对数似然：

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|} \log \pi_\theta(y|x) = -\frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta(y_i | x, y_{<i})$$

#### 3.2.2 Preference Optimization (ORPO)

使用 **ORPO (Monolithic Odds Ratio Preference Optimization)** 算法，引入 odds ratio loss：

$$\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w | x)}{\text{odds}_\theta(y_l | x)} \right)$$

其中：

$$\text{odds}_\theta(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$$

最终目标结合 SFT 和 OR losses：

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \beta \cdot \mathcal{L}_{\text{OR}}$$

默认 $\beta = 0.1$。ORPO 不需要 reference model，适合长上下文场景下的高效训练。

### 3.3 数据合成细节

- **来源：** MuSiQue 训练集的问题（multi-hop questions），不使用标注答案
- **上下文构造：** 将相关 Wikipedia 文档与随机采样的无关文档混合、打乱并拼接，上下文长度随机指定在 4K–31K tokens 之间
- **默认合成 2048 个训练样本**
- 每个样本采样 $N = 32$ 条输出

### 3.4 训练细节

- Sequence parallelization（parallel size = 8）
- QLoRA：rank = 128, alpha = 128, dropout = 0.05，目标模块为所有 attention 和 feedforward linear layers
- 训练 1 个 epoch
- Batch size = 8, learning rate = 5e-5, max sequence length = 32K
- 硬件：8 × H100 GPUs

---

## 4. 数据集

### 训练数据

| 领域 | 数据集 | 说明 |
|------|--------|------|
| Multi-hop QA | MuSiQue (训练集) | 仅使用问题，不使用标注答案；每个问题关联多篇 Wikipedia 文档，混入无关文档构造长上下文（4K–31K tokens）|

### 评估数据

| 领域 | 数据集 | 说明 |
|------|--------|------|
| Single-doc QA | Qasper | 200 examples, max 21,110 tokens, avg 4,921 tokens |
| Single-doc QA | MultiFieldQA-En | 150 examples, max 14,947 tokens, avg 6,888 tokens |
| Multi-doc QA | HotpotQA | 200 examples, max 16,322 tokens, avg 12,779 tokens |
| Multi-doc QA | MuSiQue | 200 examples, max 16,335 tokens, avg 15,542 tokens |
| Multi-doc QA | 2WikiMultihopQA | 200 examples, max 16,319 tokens, avg 7,096 tokens |
| Short-context | OpenLLM Leaderboard | MMLU, GSM8K, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA |

---

## 5. 评估指标与主要结果

### 评估指标

- **Substring Exact Match (SubEM)：** 检查模型输出是否包含 golden answer 子串

### 主要结果

#### 长上下文推理（Tab. 2）

| Model | Qasper | MultiField | HotpotQA | MuSiQue | 2WikiMQA | **Avg.** |
|-------|--------|------------|----------|---------|----------|----------|
| Qwen-2.5-7B-Instruct | 21.0 | 28.0 | 70.5 | 48.0 | 77.5 | 49.0 |
| + SEALONG | 26.0 | 29.3 | 72.5 | 51.5 | 79.5 | **51.8** |
| Qwen-2.5-14B-Instruct | 21.0 | 32.0 | 73.0 | 52.0 | 83.0 | 52.2 |
| + SEALONG | 24.0 | 30.0 | 75.0 | 57.0 | 87.5 | **54.7** |
| Llama-3.1-8B-Instruct | 29.0 | 29.3 | 64.0 | 49.5 | 82.0 | 50.8 |
| + SEALONG | 32.5 | 31.3 | 68.0 | 58.5 | 84.5 | **55.0** |
| Qwen-2.5-32B-Instruct | 24.5 | 26.0 | 72.0 | 55.0 | 88.0 | 53.1 |
| Qwen-2.5-72B-Instruct | 27.0 | 28.7 | 74.5 | 58.5 | 89.0 | 55.5 |
| Llama-3.1-70B-Instruct | 30.0 | 33.3 | 74.0 | 68.5 | 85.5 | 58.3 |
| GPT-4o | 21.5 | 28.0 | 74.5 | 64.0 | 84.0 | 54.4 |

**核心发现：**
- Llama-3.1-8B + SEALONG (55.0) **超越 GPT-4o** (54.4)，绝对提升 +4.2
- Qwen-2.5-14B + SEALONG (54.7) **超越 Qwen-2.5-32B** (53.1)，14B 模型 self-improve 后超过了参数量翻倍的同系列模型
- Qwen-2.5-7B + SEALONG (51.8) 接近 Qwen-2.5-14B base (52.2)
- 虽然仅使用 MuSiQue 训练数据，但在其他任务（Qasper、MultiFieldQA-En、HotpotQA、2WikiMQA）上均有提升，展现了良好的**泛化能力**

#### 与现有数据集对比（Tab. 5, Llama-3.1-8B-Instruct）

| 方法类别 | 数据集 | Avg. |
|----------|--------|------|
| Baseline | Llama-3.1-8B-Instruct | 50.8 |
| SFT | TULU-V2-mix | 37.0 |
| SFT | WildChat | 36.5 |
| SFT | LongAlpaca | 35.6 |
| SFT | LongAlign | 48.7 |
| SFT | LongMIT | 41.7 |
| SFT | LongReward-SFT | 47.4 |
| SFT | GPT-4o-MuSiQue | 50.9 |
| **SFT** | **SEALONG-SFT** | **52.4** |
| PO | UltraFeedback | 35.1 |
| PO | LongReward-Preference | 50.9 |
| **PO** | **SEALONG (ORPO)** | **55.0** |

大多数外部数据集反而**损害**了 Llama-3.1-8B 的性能（因为模型本身已具备较强长上下文能力，低质量合成数据反而有害）。SEALONG 不仅不损害，还实现了显著提升。即使与使用 GPT-4o 标注答案的 GPT-4o-MuSiQue 相比，SEALONG 的自监督数据也更优（55.0 vs 50.9）。

#### 短上下文性能（Tab. 8, OpenLLM Leaderboard）

| Model | Long-Ctx Avg. | MMLU | GSM8K | ARC | HellaSwag | WinoGrande | TruthfulQA | Short Avg. |
|-------|---------------|------|-------|-----|-----------|------------|------------|------------|
| Qwen-2.5-7B | 49.0 | 74.2 | 82.4 | 67.1 | 81.5 | 74.7 | 64.7 | 74.1 |
| + SEALONG | 51.8 | 74.1 | 83.2 | 66.5 | 81.3 | 74.4 | 64.8 | **74.1** |
| Llama-3.1-8B | 50.8 | 68.3 | 77.7 | 60.2 | 80.1 | 77.4 | 54.1 | 69.6 |
| + SEALONG | 55.0 | 68.4 | 77.8 | 60.3 | 79.9 | 77.3 | 53.8 | **69.6** |

SEALONG 在大幅提升长上下文性能的同时，**几乎不影响短上下文性能**（短上下文平均分完全不变）。

### 关键发现

1. **Prompting 策略至关重要：** Plan-and-solve prompting 在长上下文推理中远优于 default prompting（Llama-3.1-8B 在 2WikiMQA 上从 66.0 → 82.0），表明模型潜力被低估。
2. **MBR > Self-evaluation：** 基于多输出共识的 MBR 打分优于 LLM 直接自评估，因为当前 LLM 在长上下文下的 self-evaluation 能力有限。Sentence embedding similarity 是最佳效用度量。
3. **数据效率高：** 仅 1K 训练样本即可达到接近饱和的性能，更多数据带来的收益有限。这说明 SEALONG 是在**激发模型已有能力**，而非教授新技能。
4. **采样数量 $N$ 的影响：** $N$ 从 8 增加到 32 一致提升性能（MBR 估计更准确）；$N > 32$ 后收益递减，可能因为 scoring 方法在更大输出集合中选择高质量输出的能力达到瓶颈。
5. **ORPO 优于 SFT：** Preference optimization (SEALONG-ORPO: 55.0) 显著优于 SFT (SEALONG-SFT: 52.4)，对比式学习更有效。
6. **泛化能力强：** 仅在 MuSiQue 上合成训练数据，但在 Qasper、MultiFieldQA-En、HotpotQA 等跨领域任务上也有一致提升。
7. **输出长度无显著变化：** SEALONG 不是通过生成更多 token 来"碰运气"匹配答案——平均输出 token 数几乎不变（Llama-3.1-8B: 289→295）。
8. **局限性：** (a) MBR scoring 与 oracle 之间仍有显著 gap，scoring 方法仍有改进空间；(b) 仅在 ≤14B 模型、≤32K 上下文长度上验证；(c) 训练数据仅来自 MuSiQue 的 multi-hop QA，未覆盖 full-context reasoning 等更难的问题类型。
