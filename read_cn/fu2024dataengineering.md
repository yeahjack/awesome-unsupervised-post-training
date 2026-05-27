# Data Engineering for Scaling Language Models to 128K Context

> **加入 Survey 时间：** 2026-03-11

**Paper:** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng (2024)
**arXiv:** 2402.10171
**Method:** Data Eng 128K | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | unlabeled corpus / continual training batch |
| 参数/状态持久性 Persistence | full parameter accumulate across corpus stages; no sample-level reset |
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

- **更新何时触发：** 更新在 deployment 前的 continued pretraining / CPT 阶段按 corpus 与 training batch 持续触发，不由单个测试样本触发。
- **服务当前样本还是后续样本：** 当前 batch 的更新服务后续训练批次与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整个 corpus stream / stage 序列中持续累积；若有阶段切换，也只是训练阶段切换，不是 sample-level reset。
- **reset 边界：** 因此它回答的是 deployment 前的 adaptation schedule，而不是 deployment 期间的 online TTA。

## 1. UPT 归属理由

本文属于 **Family I — Prediction-Statistic Optimization (predictive likelihood minimization)**。

- 核心做法是在预训练好的 LLaMA-2 模型上进行 **continual pretraining**，优化目标仍然是标准的 **next-token prediction (language modeling loss)**。
- 全部训练数据来自 SlimPajama（LLaMA 预训练数据的开源复现），不引入任何外部标注、人类偏好、外部验证器或外部 AI 标签。
- 唯一的创新在于数据工程层面（长度上采样 + 领域平衡），训练信号完全来自文本自身的 intrinsic statistics（token-level predictive likelihood）。
- 因此符合 UPT 定义：无外部 ground truth，纯粹依赖模型内在的 LM loss 进行 post-training 优化。

---

## 2. 解决的问题

现有开源长上下文模型（LongChat、Together LLaMA-2、YaRN Mistral、LongLoRA）虽声称支持 32K-128K 上下文，但在 **Needle-in-a-Haystack** 精确检索测试中表现不佳，无法在长文档的任意位置准确检索信息。作者指出：

1. **长上下文能力并非必须通过大规模训练注入**——模型在短上下文（4K）预训练中已大致获得了在任意位置利用信息的能力，只需少量 continual pretraining 即可将该能力延伸到 128K。
2. **现有工作忽视了数据工程细节**：训练长度不够长（Together 仅 32K）、领域不平衡（YaRN 仅用 book 数据）、未做长度上采样（LongLoRA）。
3. 需要一种 **低成本、学术预算可行** 的数据配方，使开源模型在 Needle-in-a-Haystack 上逼近 GPT-4 Turbo 128K 的表现。

---

## 3. 方法介绍

### 3.1 数据配方：Per-source Length Upsampling

以 SlimPajama 为数据源（82% web / 15% C4 / 4.5% code / 4.5% Wikipedia / 4.5% books / 2.5% Arxiv / 2.0% StackExchange），作者对比了多种数据混合策略：

| 策略 | 描述 |
|---|---|
| **Cut at 4K** | 将所有文档截断为 4K chunk（标准做法，破坏长程依赖） |
| **Cut at 128K** | 保留自然长文档，不改变领域分布（但自然长文档不足） |
| **Global Upsampling** | 全局上采样长序列，会改变领域分布 |
| **Upsample Arxiv/Book/Github** | 仅上采样特定领域，同时改变领域和长度分布 |
| **Per-source Upsampling**（推荐） | 在每个领域内部分别上采样长文档，保持领域比例不变，仅改变长度分布 |

**Per-source upsampling** 的关键：在每个数据源内将长于 4K 的文档比例从约 30% 上采样至约 70%，同时保持 67% CC / 15% C4 / 4.5% Github 等原始领域配比不变。这样既增加了长序列训练数据，又避免了领域偏移导致的短上下文性能退化。

### 3.2 训练配置

- **Base model:** LLaMA-2 7B / 13B
- **训练上下文长度:** 7B 用 80K，13B 用 64K（受显存限制）
- **RoPE base 调整:** 按 Xiong et al. (2023) 的方法修改 positional encoding
- **数据量:** 1B-5B tokens（约 2000 optimization steps，batch size 4M tokens）
- **硬件:** 8 x 80G A100（7B 约 5 天，13B 约 10 天），成本仅为 Xiong et al. (400B tokens) 的约 1%
- **框架:** Huggingface Transformers + DeepSpeed Zero 3 + FlashAttention 2 + Gradient Checkpointing + CPU Offloading
- **Learning rate:** constant 2e-5

### 3.3 关键发现

- **数据量方面:** 500M tokens 即可解锁模型在 80K 范围内的精确检索能力；5B tokens 时模型可泛化至未见的 80K-128K 范围；10B tokens 时出现过拟合，泛化能力下降。
- **领域平衡方面:** 仅上采样单一领域（如 Book/Code）的改进无法迁移到其他领域，甚至可能伤害其他领域；per-source upsampling 是唯一在所有领域和长度范围内均有提升的策略。
- **Validation loss 不够用:** 两种数据配方可能产生非常接近的 validation loss，但在 Needle-in-a-Haystack 精确检索上表现截然不同（Figure 4）。

---

## 4. 数据集

### 训练数据

- **SlimPajama** (Soboleva et al., 2023)：LLaMA 预训练数据的开源复现，627B tokens，包含 CommonCrawl (67%)、C4 (15%)、Wikipedia (4.5%)、Github (4.5%)、Books (4.5%)、Arxiv (2.5%)、StackExchange (2.0%)。
- 经 per-source length upsampling 处理后打包为 80K / 64K chunks，总训练量 **5B tokens**。

### 评估数据

| Benchmark | 描述 |
|---|---|
| **Needle-in-a-Haystack** (Kamradt, 2023) | 在 1K-128K 长文档的任意位置插入一句话（needle），测试模型能否精确复述，主要评估指标 |
| **InfiniBench BookQA** (Zhang et al., 2023) | 128K 长度的 book-long question answering |
| **MMLU** (Hendrycks et al., 2020) | 短上下文通用能力基准，用于验证不退化 |
| **Per-domain validation loss** | 各领域在 0-4K 和 4K-128K 范围的 perplexity |

---

## 5. 评估指标与主要结果

### 主要指标

- **Needle-in-a-Haystack accuracy**：在不同文档长度 x needle 位置的网格上计算检索成功率
- **BookQA accuracy**：128K 长度书籍问答准确率
- **MMLU score**：短上下文通用能力

### 核心结果

| Model | Context | Needle Acc | MMLU |
|---|---|---|---|
| GPT-4-Turbo | 128K | 87.1 | 86.4 |
| Together LLaMA-2 7B | 32K | 27.9 | 44.8 |
| LongChat v1.5 7B | 32K | 18.0 | 42.3 |
| LongLoRA 7B | 100K | 70.0 | 37.9 |
| YaRN Mistral 7B | 128K | 57.4 | 59.4 |
| **Ours LLaMA-2 7B** | **80K** | **88.0** | **43.3** |
| LongLoRA 13B | 64K | 54.1 | 50.1 |
| **Ours LLaMA-2 13B** | **64K** | **90.0** | **52.4** |

| Model | BookQA (128K) |
|---|---|
| GPT-4-Turbo 128K | 37.4 |
| LongLoRA 7B 100K | 24.3 |
| Ours LLaMA-2 7B 80K | 27.4 |
| YaRN Mistral 7B 128K | 26.3 |
| **Ours LLaMA-2 13B 64K** | **31.1** |

### 关键结论

1. 在 Needle-in-a-Haystack 上，7B 模型达到 **88.0** 准确率（训练于 80K 上下文），接近 GPT-4 Turbo 的 87.1，大幅超越所有开源 baseline。
2. **短上下文能力不退化**：MMLU 分数与 base model 持平（7B: 43.3 vs base ~45; 13B: 52.4 vs base ~55）。
3. Per-source length upsampling 是唯一在所有领域（0-4K 和 4K-128K）均不显著退化的数据配方。
4. 仅需 **5B tokens / 5 天 / 8xA100**，成本为大规模方案的 ~1%，证明学术预算下即可实现 128K 长上下文建模。
