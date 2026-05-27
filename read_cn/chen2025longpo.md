# LongPO: Long Context Self-Evolution of LLMs through Short-to-Long Preference Optimization

> **加入 Survey 时间：** 2026-03-11

**Paper:** LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization
**Authors:** Guanzheng Chen, Xin Li, Michael Qizhe Shieh, Lidong Bing
**Affiliations:** National University of Singapore, DAMO Academy (Alibaba Group), Hupan Lab, Shanda AI Research Institute
**ArXiv:** 2409.10164
**Published:** ICLR 2025
**Code:** https://github.com/DAMO-NLP-SG/LongPO

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| LongPO | Pref. Opt. | training-time | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / synthetic preference pair batch |
| 参数/状态持久性 Persistence | full parameter accumulate across epochs or iterations |
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

- **更新何时触发：** 更新在 deployment 前的 preference optimization 阶段触发，基本单位是模型自生成的 chosen / rejected pairs。
- **服务当前样本还是后续样本：** 当前 pair batch 的更新服务后续训练与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在训练 epoch / iteration 中持续累积，不做 sample-level reset。
- **reset 边界：** 因此它更像 offline pair-based post-training，而不是 online TTA。

## 1. UPT 归属理由
**Family III — Self-Generated Target Bootstrapping (internally generated preference pairs)**

LongPO 完全符合 unsupervised post-training 的定义：它**不依赖任何人工标注的长上下文数据**，而是利用模型自身在短上下文上的已有能力，自动生成 short-to-long preference data（由同一 instruction 在长上下文和压缩短上下文上分别生成的 response 组成 preference pair）。这些 preference pair 完全由模型内部产生（短上下文 response → chosen，长上下文 response → rejected），随后通过 DPO-style 的 preference optimization 目标进行训练。整个流程无需外部 reward model 或人工反馈，属于典型的 internally generated preference pairs 路线。

---

## 2. 解决的问题

LongPO 解决的核心问题是：**如何在无人工长上下文标注的情况下，将短上下文 LLM 的能力有效迁移到长上下文场景，同时保持短上下文性能不退化**。

具体而言，现有的长上下文对齐面临两大挑战：
1. **数据稀缺**：随着上下文长度增加，人工标注变得不切实际且不可靠；使用高级 LLM 合成数据缺乏可扩展性；简单拼接短上下文数据效果不佳。
2. **性能平衡**：长上下文对齐往往导致短上下文性能严重退化。例如 LLaMA-3.1 系列在对齐阶段仅使用 0.1% 的长上下文数据，限制了长上下文对齐的充分性。

论文发现，即使是 GPT-4 这样短上下文性能极强的模型，在长上下文任务上也表现不佳（GPT-4-128K 在 ∞Bench 上仅 34.81 分，甚至不如 LLaMA-3.1-8B）。这揭示了 LLM 的长上下文潜力尚未被充分释放。

---

## 3. 方法介绍

### 3.1 核心思想：Short-to-Long Preference

LongPO 的核心假设是：**短上下文预训练和对齐中习得的能力可以有效迁移到长上下文场景，无需外部指导**。

给定一个长输入 $x_L = [C_L; I_L]$（长文档 $C_L$ + 指令 $I_L$），以及从长文档中提取的与指令相关的短上下文 $C_S = F(C_L, I_L)$，模型分别生成两个 response：
- **Chosen response**：$y_S \sim \pi_S(y \mid x_S)$，基于短上下文 $x_S = [C_S; I_L]$ 生成（短上下文模型擅长处理，质量高）
- **Rejected response**：$y_L \sim \pi_S(y \mid x_L)$，基于长上下文 $x_L$ 生成（短上下文模型处理长上下文能力不足，质量低）

定义 short-to-long preference distribution（基于 Bradley-Terry 模型）：

$$p_{SL}(y_S \succ y_L \mid x_L) = \sigma(r(x_L, y_S) - r(x_L, y_L))$$

LongPO 的优化目标为：

$$\mathcal{L}_{\text{LongPO}}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x_S, x_L, y_S, y_L) \sim D_{SL}} [\sigma(r_\theta(x_L, y_S) - r_\theta(x_L, y_L))]$$

### 3.2 Short-to-Long Constraint

长上下文对齐常导致短上下文性能退化。传统 DPO/RLHF 的 KL 约束 $\beta D_{KL}[\pi_\theta(y|x_L) \| \pi_{ref}(y|x_L)]$ 使用的 reference model 本身就不擅长长上下文，导致约束不恰当。

LongPO 提出一种改进的 **short-to-long KL constraint**：

$$C' = \beta D_{KL}[\pi_\theta(y \mid x_L) \| \pi_S(y \mid x_S)]$$

即约束策略模型在长上下文上的输出分布不偏离短上下文参考模型在包含相同核心信息的短上下文上的输出分布。

由此推导出 LongPO 的精炼 reward function：

$$r_\theta^{\text{LongPO}}(x_L, y) = \beta \log \frac{\pi_\theta(y \mid x_L)}{\pi_S(y \mid x_S)} + \beta \log Z(x_L, x_S)$$

最终的 LongPO objective：

$$\mathcal{L}_{\text{LongPO}}(\pi_\theta; \pi_S) = -\mathbb{E}_{D_{SL}} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_S \mid x_L)}{\pi_S(y_S \mid x_S)} - \beta \log \frac{\pi_\theta(y_L \mid x_L)}{\pi_S(y_L \mid x_S)} \right)\right]$$

注意与标准 DPO 的关键区别：分母中的 reference model 是在**短上下文** $x_S$ 上条件化的 $\pi_S(\cdot|x_S)$，而非在长上下文 $x_L$ 上。

### 3.3 Self-Evolving 流程

**初始化**：仅需一个对齐好的短上下文 LLM $\pi_S$ 和长上下文 plain corpus。

**Preference Data 构建**（两步）：
1. **Instruction Generation**：对每个长文档 $C_L$，随机采样一个短 chunk $C_S$，用 $\pi_S$ 通过 Self-Instruct 生成 instruction pool，随机选取一条 $I_L$。
2. **Response Generation**：用 $I_L$ 分别在短上下文 $x_S$ 和长上下文 $x_L$ 上生成 chosen response $y_S$ 和 rejected response $y_L$。

**Multi-turn LongPO**：由于一个长文档包含多个 chunk，收集多组 instruction-response triples 组成 multi-turn 数据集，并聚合概率：

$$\mathcal{L}_{\text{LongPO}}^{MT} = -\mathbb{E}_{\hat{D}_{SL}} \left[\log \sigma \left(\beta \log \frac{\sum_{i=1}^n \pi_\theta(y_{Si} \mid x_L)}{\sum_{i=1}^n \pi_S(y_{Si} \mid x_{Si})} - \beta \log \frac{\sum_{i=1}^n \pi_\theta(y_{Li} \mid x_L)}{\sum_{i=1}^n \pi_S(y_{Li} \mid x_{Si})} \right)\right]$$

**最终训练目标**结合 NLL loss 以稳定训练：

$$\mathcal{L}_\theta = \lambda \cdot \mathcal{L}_{\text{LongPO}}^{MT}(\pi_\theta; \pi_S) + \mathcal{L}_{\text{NLL}}(\pi_\theta; S_L)$$

其中 $S_L = [x_L; \{I_{Li}; y_{Si}\}_{i=1}^n]$ 是长上下文 + chosen response 的完整序列。

**迭代扩展**：LongPO 采用迭代方式逐步扩展上下文长度（32K → 128K → 256K → 512K），每一轮训练后的模型作为下一轮的"短上下文 LLM"。

### 3.4 训练细节

- 优化器：Adam，学习率 5e-7
- $\beta = 0.1$（DPO margin），$\lambda = 0.01$（NLL 权重）
- RoPE $\theta = 1 \times 10^7$，batch size = 8
- 使用 Deepspeed-Ulysses 进行序列并行，Flash Attention 加速

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 训练语料 | Long-Data-Collection (Book & ArXiv) | 长上下文 plain corpus，筛选 64K–128K 长度文档 |
| 训练语料 | RedPajama (GitHub) | 长上下文 plain corpus |
| 训练数据量 (Mistral-7B) | 自生成 preference data | 45K (128K), 16K (256K), 2.5K (512K) multi-turn samples |
| 训练数据量 (Qwen2.5-7B) | 自生成 preference data | 32K samples (128K) |
| 长上下文评估 | ∞Bench | En.Sum, En.QA, En.MC；评估长度 >100K |
| 长上下文评估 | RULER | NIAH retrieval, VT multi-hop, QA；4K–128K 序列长度 |
| 长上下文评估 | LongBench-Chat (EN) | 10K–100K tokens instruction-following，GPT-4 评分 |
| 短上下文评估 | MMLU | 通用语言理解 |
| 短上下文评估 | ARC-C | 科学推理 |
| 短上下文评估 | Hellaswag | 常识推理 |
| 短上下文评估 | Winogrande | 共指消解 |
| 短上下文评估 | MT-Bench | 多轮指令跟随能力，GPT-4 评分 |

---

## 5. 评估指标与主要结果

### 评估指标

- **∞Bench**：En.Sum (摘要 ROUGE)、En.QA (F1)、En.MC (Accuracy)，三项平均 (AVG.)
- **RULER**：NIAH / VT / QA accuracy，三项平均 (AVG.)；排除了 Aggregation 任务
- **LongBench-Chat**：GPT-4-128K 评分 (1–10)
- **短上下文**：MMLU, ARC-C, Hellaswag, Winogrande (accuracy)；MT-Bench (GPT-4 评分 1–10)

### 主要结果

#### 长上下文性能（∞Bench）

| Model | Context | En.Sum | En.QA | En.MC | AVG. |
|-------|---------|--------|-------|-------|------|
| GPT-4-128K | 128K | 14.73 | 22.44 | 67.25 | **34.81** |
| LLaMA-3.1-8B | 128K | 28.06 | 30.47 | 58.08 | 38.87 |
| GLM-4-9B-Chat | 128K | 14.84 | 9.51 | 67.25 | 30.53 |
| GLM-4-9B-Chat-1M | 1M | 28.3 | 9.7 | 68.6 | 35.53 |
| Mistral-7B (base) | 32K | 22.13 | 4.93 | 14.41 | 13.82 |
| Mistral-7B-SFT | 128K | 23.44 | 13.45 | 53.21 | 30.03 |
| Mistral-7B-DPO | 128K | 15.21 | 10.34 | 48.14 | 25.56 |
| **Mistral-7B-LongPO (iter1)** | **128K** | **27.05** | **23.51** | **67.25** | **39.27** |
| Mistral-7B-LongPO (iter2) | 256K | 28.16 | 24.43 | 66.35 | 39.65 |
| Mistral-7B-LongPO (iter3) | 512K | 29.10 | 27.85 | 66.67 | 41.21 |
| Qwen2.5-7B (base) | 128K | 22.89 | 6.08 | 52.4 | 27.12 |
| **Qwen2.5-7B-LongPO (iter1)** | **128K** | **32.06** | **17.32** | **72.05** | **40.48** |

#### 长上下文性能（RULER）

| Model | NIAH | VT | QA | AVG. |
|-------|------|------|------|------|
| GPT-4-128K | 95.4 | 99.9 | 70.3 | 88.53 |
| Mistral-7B (base) | 72.60 | 74.40 | 52.2 | 66.4 |
| Mistral-7B-SFT | 88.73 | 79.64 | 51.08 | 73.15 |
| Mistral-7B-DPO | 74.25 | 72.36 | 50.24 | 65.62 |
| **Mistral-7B-LongPO (iter1)** | **96.88** | **96.49** | **64.81** | **86.06** |
| Mistral-7B-LongPO (iter3) | 97.28 | 97.48 | 64.92 | 86.56 |
| Qwen2.5-7B-LongPO (iter1) | 95.81 | 89.71 | 59.4 | 81.64 |

#### 短上下文性能保持（Margin 相对 base model）

| Model | MMLU | ARC-C | Hellaswag | Winogrande | MT-Bench |
|-------|------|-------|-----------|------------|----------|
| LongPO (iter1) | +0.84 | +1.02 | +0.08 | +0.40 | +0.32 |
| LongPO (iter2) | +0.13 | +0.10 | -0.06 | -0.21 | -0.25 |
| SFT | -14.04 | -14.80 | -9.73 | -10.77 | -1.03 |
| DPO | -25.74 | -9.07 | -14.11 | -12.15 | -1.11 |
| GLM-4-9B → 1M | -0.26 | -0.42 | -0.51 | -0.55 | -0.40 |
| LWM-7B → 1M | -1.12 | -1.02 | -2.40 | -19.90 | -22.00 |

### 关键发现

1. **LongPO 大幅优于 SFT 和 DPO**：在相同模型和数据上，LongPO 在长上下文任务上比 SFT 和 DPO 高出 **10–20+ 分**（∞Bench: 39.27 vs. SFT 30.03 / DPO 25.56），同时在 RULER 上也大幅领先（86.06 vs. 73.15 / 65.62）。

2. **短上下文性能完全保持**：LongPO 在 MMLU 上保持 59.99（base model 59.15），而 SFT 和 DPO 分别导致 **14–25 分**的短上下文性能退化。这归功于 short-to-long KL constraint 的有效性。

3. **超越 GPT-4-128K**：Mistral-7B-LongPO-128K 在 ∞Bench 上达到 **39.27 分**，超过 GPT-4-128K 的 **34.81 分**，且参数量小得多（7B vs. ~1.8T）。这表明即使是最先进的 LLM 在长上下文场景中也尚未充分发挥潜力。

4. **迭代扩展有效**：从 128K → 256K → 512K 的迭代训练持续提升长上下文性能（∞Bench: 39.27 → 39.65 → 41.21），同时对短上下文性能影响极小。

5. **En.QA 任务的显著优势**：在涉及复杂 book-length 自由问答的 En.QA 任务上，LongPO 超过 GLM-4-9B 和 GLM-4-9B-1M **10+ 分**，说明人工标注不足以覆盖复杂长上下文任务。

6. **Ablation 验证**：
   - **Short-to-long preference 有效**：LongPO 始终优于 SFT-Chosen 和 SFT-Rejected，表明显式的 preference learning 比隐式偏好更有效。
   - **Short-to-long constraint 关键**：移除该约束（退化为 DPO）导致长短上下文性能均大幅下降，尤其是短上下文能力在训练初期便迅速退化。
   - **NLL loss 重要**：移除 NLL loss 导致长上下文性能收敛变慢，表明 NLL 在无 continual pretraining 条件下对稳定长上下文训练至关重要。

7. **跨模型泛化**：在 Qwen2.5-7B-Instruct 上也验证了 LongPO 的有效性（∞Bench: 27.12 → 40.48，+13.36 分）。
