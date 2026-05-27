# E2-LLM: Efficient and Extreme-Length Extension of Large Language Models

> **加入 Survey 时间：** 2026-03-11

**Method:** E2-LLM | **Carrier:** Direct Opt. | **Regime:** training-time | **Level:** Token

> Findings of ACL 2024, pp. 4243–4253

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

E2-LLM 属于 **Family I（Prediction-Statistic Optimization, predictive likelihood minimization）**。其核心训练目标是标准的 next token prediction loss（语言模型似然），不依赖任何外部标注、外部验证器或人类/AI 偏好信号。训练过程中仅使用模型自身的 predictive likelihood 作为优化信号，通过在短序列上进行 fine-tuning 来实现长上下文能力的扩展。dominant artifact 是 predictive likelihood，属于 token-level 的 direct optimization。

---

## 2. 解决的问题

现有 LLM 长上下文扩展方法存在两大瓶颈：

1. **数据需求大**：需要收集与目标上下文长度匹配的长序列训练数据（如 32K/64K tokens），数据获取困难。
2. **GPU 显存开销高**：训练长序列需要的显存随序列长度线性甚至二次增长，例如扩展到 64K 时显存需求可达 68.8 GB 甚至 OOM。
3. **泛化能力差**：Position Interpolation (PI) 等方法需要为每个目标上下文长度单独训练，且在 out-of-distribution (OOD) 长度上表现不佳。

E2-LLM 旨在用**极短的训练序列**（如 4K tokens）通过**一次训练**实现对**任意长上下文**（16K/32K/64K 乃至 120K）的扩展。

---

## 3. 方法介绍

E2-LLM 基于 RoPE (Rotary Position Embedding) 提出了一种 **dual augmentation strategy**，在训练时对 scale 和 position offset 两个超参数进行数据增强，使模型在短序列训练中接触到多样的相对位置分布。

### 3.1 核心公式

标准 RoPE 的 position embedding 为 $\mathbf{f}(\mathbf{x}, m)$，E2-LLM 将其改写为：

$$\mathbf{f}'(\mathbf{x}, m; t, g) = \mathbf{f}\left(\mathbf{x}, \frac{m+t}{g}\right)$$

其中 $g$ 为 scale parameter，$t$ 为 position offset。

### 3.2 Scale augmentation（对 $g$ 的增强）

- 在每个训练 iteration $i$，从集合 $\mathcal{G} = \{1, 2, \ldots, G_{max}\}$ 中按均匀分布采样 $g_i$。
- 不同的 $g$ 值对应不同的插值上下文窗口大小（如 $g=8$ 对应 32K，$g=16$ 对应 64K）。
- 这使得模型在训练中覆盖多种 position density，避免对某一特定插值比例过拟合（类比 Runge phenomenon）。

### 3.3 Position offset augmentation（对 $t$ 的增强）

- 保留序列前 4 个 token 的 position offset 为 0（attention sink 机制）。
- 对其余 token，每个 iteration 从均匀分布采样 $t_i \in \{0, \ldots, T_{max}\}$。
- $T_{max}$ 设为当前插值上下文窗口与训练窗口之差。
- 这使模型接触到不同的绝对位置值，从而泛化到更大的相对位置差异。

### 3.4 训练与推理

- **训练**：在短序列（4K/8K tokens）上进行标准 next token prediction fine-tuning，每个 iteration 随机采样 $g_i$ 和 $t_i$ 来修改 RoPE。
- **推理**：不引入额外参数，仅需根据目标上下文长度设置对应的 scale parameter $g$（如扩展到 32K 设 $g=8$，扩展到 64K 设 $g=16$）。同一模型权重可支持不同上下文窗口。

---

## 4. 数据集

### 训练数据

| 数据集 | 用途 |
|--------|------|
| **Pile** (Gao et al., 2020) | Pretrain 数据，提供通用语料 |
| **ShareGPT** (Zheng et al., 2023) | Fine-tuning 数据，提升问答能力 |
| **Long summarization datasets** (Cohan et al., 2018) | Fine-tuning 数据，提升长文本处理能力 |

训练序列最大长度 $R$ 仅为 **4K tokens**（部分实验为 8K）。

### 评估数据

| 数据集 | 说明 |
|--------|------|
| **LongBench** (Bai et al., 2023) | 双语长上下文理解 benchmark，包含 single-doc QA、multi-doc QA、summarization、few-shot learning、synthetic tasks、code tasks |
| **Arxiv Proof-Pile** (Azerbayev et al., 2022) | ArXiv 数学论文数据集，用于评估长序列 perplexity（每篇不少于 64K tokens） |
| **Needle In A Haystack** | 压力测试，评估不同上下文长度和文档深度下的检索准确率 |

---

## 5. 评估指标与主要结果

### 评估指标

- **LongBench accuracy**（0-100 分，越高越好）：覆盖 single-doc QA、multi-doc QA、summarization、few-shot learning、synthetic、code 六类子任务，分英文 (EN)、中文 (ZH) 和总体 (All)。
- **Perplexity (PPL)**（越低越好）：在 Arxiv Proof-Pile 上评估不同上下文窗口下的困惑度。
- **Needle In A Haystack accuracy**：不同文档深度和上下文长度下的检索准确率。

### 主要结果

#### LongBench（Llama2-13B 系列）

| 模型 | EN | ZH | All |
|------|----|----|-----|
| PI-Llama2-13B-16K | 40.88 | 26.35 | 37.65 |
| **E2-LLM-Llama2-13B-16K** | **44.73** | **28.56** | **41.13** |
| **E2-LLM-Llama2-13B-32K** | **44.55** | **31.93** | **41.74** |
| GPT-3.5-Turbo-16K | 44.60 | 33.78 | 42.19 |

- E2-LLM-Llama2-13B-32K 与 GPT-3.5-Turbo-16K 总体性能相当（41.74 vs. 42.19）。
- E2-LLM-13B-16K 比相同设置的 PI-Llama2-13B-16K 平均高约 **9%**。

#### Arxiv Proof-Pile PPL（Llama2-13B, 训练窗口 4K）

| 方法 | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
|------|-------|-------|--------|--------|--------|
| E2-LLM-16K | 2.82 | 2.59 | 2.43 | - | - |
| E2-LLM-32K | 2.85 | 2.61 | 2.44 | 2.34 | - |
| E2-LLM-64K | 2.91 | 2.67 | 2.49 | 2.39 | 2.44 |

- 随着评估上下文窗口增大，PPL 持续下降，表明模型有效利用了长上下文信息。

#### 泛化能力

- 设置 $G_{max}=20$（支持最大 80K 插值窗口），在推理时将 scale 提升至 30/40/50，对应 120K/160K/200K 上下文。
- 在 **120K tokens 以内** PPL 保持在合理水平，展现了对训练未见 scale 的泛化能力。

#### Ablation 结果

| 变体 | EN | ZH | All |
|------|----|----|-----|
| **E2-LLM (full)** | **44.55** | **31.93** | **41.74** |
| E2-LLM (no offset) | 42.28 | 29.49 | 39.44 |
| E2-LLM (fixed scale) | 41.66 | 28.33 | 38.77 |

- Scale augmentation 和 position offset augmentation 均有独立贡献，二者结合效果最佳。
- Initial fixed tokens 数量设为 4 效果最优。
