# Effective Long-Context Scaling of Foundation Models

> **加入 Survey 时间：** 2026-03-11

**论文信息**: Wenhan Xiong, Jingyu Liu, Igor Molybog et al. (GenAI, Meta), arXiv:2309.16039, 2023

| 属性 | 值 |
|---|---|
| Method | LongContext Scaling |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Token |
| Family | I — Prediction-Statistic Optimization |

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

本文属于 **Family I (Prediction-Statistic Optimization, predictive likelihood minimization)**。核心训练信号是 token-level perplexity：在长上下文语料上对 LLaMA 2 checkpoint 进行 continual pretraining，直接最小化标准 language modeling loss（next-token prediction 的负对数似然）。整个 pretraining 阶段不依赖任何外部标注、human preference 或 external verifier，优化目标完全是模型自身对观测文本的 intrinsic predictive likelihood。Instruction tuning 阶段同样不使用 human-annotated long data，而是利用模型自身 (LLaMA 2 Chat) 生成 self-instruct QA 数据并进行 self-critique 验证，信号仍来自模型内部。Validation loss 与 context length 呈 power-law 关系 $L(c) = (\alpha/c)^\beta + \gamma$，进一步证实核心监督信号就是 token-level perplexity。

---

## 2. 解决的问题

- 当时的开源长上下文 LLM 在评测上主要依赖 perplexity 和 synthetic task，无法在真实的 short-context 与 long-context downstream task 上同时保持强性能。
- LLaMA 2 原始 RoPE positional encoding 对远距离 token 施加过强的 attention decay，使模型在 4,000-6,000 token 之后无法有效利用上下文。
- 收集 human-annotated long-context instruction data 成本极高，且现有开源 instruction 数据集以短文本为主，阻碍了长上下文 chat 模型的对齐。

---

## 3. 方法介绍

### 3.1 Continual Pretraining

从 LLaMA 2 checkpoint 出发，使用更长的训练序列进行 continual pretraining，共消耗额外 400B tokens（100,000 steps）。关键设计：

1. **Positional Encoding 修改 (RoPE ABF)**：将 RoPE 的 base frequency $b$ 从 10,000 提升至 500,000，降低旋转角速度，减轻对远距离 token 的 attention score 衰减。该方法优于 Position Interpolation (PI) 和 xPos ABF。
2. **Data Mix**：在 LLaMA 2 原始预训练数据基础上加入新的长文本数据，并上采样长文本比例。实验表明数据**质量**比长度分布更关键——即使去除大部分长文本，模型仍能获得大部分性能提升。
3. **序列长度**：7B/13B 模型使用 32,768-token 序列；34B/70B 模型使用 16,384-token 序列。
4. **优化细节**：使用 FlashAttention；7B/13B 学习率 $2 \times 10^{-5}$，cosine schedule + 2,000 warm-up steps；34B/70B 学习率 $1 \times 10^{-5}$。

### 3.2 Instruction Tuning（无需 human-annotated long data）

1. 以 LLaMA 2 Chat 的 RLHF 数据集（短文本）为基础。
2. 利用 LLaMA 2 Chat 自身进行 **self-instruct** 生成长上下文 QA 数据：从预训练语料中随机选取长文档片段，prompt 模型生成 question-answer pair，再用模型自身做 **self-critique** 验证答案质量。
3. 短 instruction 数据拼接为 16,384-token 序列；长 instruction 数据单独处理（右侧 padding）。
4. 关键技巧：不仅在 output token 上计算 loss，还在 **long input prompt 上计算 LM loss**，稳定长短不平衡输入的学习，显著提升下游表现。

### 3.3 Training Curriculum

实验表明 continual pretraining（先 4k 再切到 32k）可节省约 40% FLOPs，性能几乎无损。模型能在切换序列长度后数千步内快速适应。

---

## 4. 数据集

### 预训练数据
- LLaMA 2 原始预训练数据（以短文本为主）
- 新增长文本数据（具体来源未公开，包含 Books、CommonCrawl、Wikipedia 等领域）
- 总量：额外 400B tokens

### 评测数据

| 任务类型 | 数据集 | 评测设置 |
|---|---|---|
| Short-context 常规 | MMLU, GSM8K, HumanEval, MATH, NaturalQuestions, TriviaQA, PIQA, SIQA, HellaSwag, WinoGrande, ARC, OpenBookQA, CommonsenseQA | Few-shot |
| Long-context QA/摘要 | NarrativeQA (0-shot), Qasper (2-shot), QuALITY (2-shot), QMSum (1-shot) | Prompt 截断至 16,384 或 32,768 tokens |
| Long-context 综合 | ZeroSCROLLS (10 个子任务：GR, SS, QM, SQAL, Qspr, Nrtv, QALT, MuSQ, SpDg, BkSS) | Zero-shot |
| Long-context 综合 | L-Eval (6 个长文本任务) | — |
| Human evaluation | Multi-turn conversation, multi-document search QA (共 2,352 examples) | 3 名标注员 |
| Safety | TruthfulQA, ToxiGen, BOLD | Few-shot |

---

## 5. 评估指标与主要结果

### 评估指标
- **Language modeling**: Validation perplexity（Books, CommonCrawl, Wikipedia）
- **Short-context**: Pass@1 (HumanEval), Top-1 accuracy (GSM8K, MMLU 等), 5-shot accuracy (NaturalQuestions, TriviaQA)
- **Long-context**: F1 (NarrativeQA, Qasper), EM (QuALITY), ROUGE-geo (QMSum), ZeroSCROLLS 各子任务指标
- **Human evaluation**: Win/Tie/Loss rate
- **Safety**: TruthfulQA (truthful+informative %), ToxiGen (toxic %), BOLD (sentiment score)

### 主要结果

**短上下文任务 (Table 1)**：LLaMA 2 Long 在几乎所有短上下文基准上与 LLaMA 2 持平或更优，70B 模型在 Coding (39.9 vs 37.4)、Math (41.3 vs 35.2)、MMLU (71.7 vs 68.9) 上均显著提升。

**长上下文任务 (Table 3)**：LLaMA 2 Long 70B 在所有长上下文任务上大幅领先开源模型（NarrativeQA 30.9, Qasper 35.7, QuALITY 79.7, QMSum 16.5），且随 context length 增加性能单调提升。

**ZeroSCROLLS (Table 4)**：LLaMA 2 Long Chat 70B 平均分 37.7，在 10 个子任务中 7 个超越 gpt-3.5-turbo-16k (36.7)，无需任何 human-annotated long context data。

**Human Evaluation (Figure 3)**：
- vs. MPT-30B-chat: 53.3% win
- vs. GPT-3.5-turbo-16k: 35.8% win vs 32.8% loss
- vs. Claude-2-100k: 38.9% win vs 31.7% loss
- vs. GPT-4: 25.0% win vs 45.0% loss

**Scaling Law**：Validation loss 与 context length 呈 power-law + constant 关系，更大模型（更大 $\beta$）能更有效利用长上下文。

**Ablation 关键发现**：
- RoPE ABF 是最优 positional encoding 方案，唯一能在 32,768-token FIRST-SENTENCE-RETRIEVAL 任务上保持性能的变体。
- 数据质量 > 长度分布：即使移除大部分长文本，长上下文性能仍保留大部分提升；单纯上采样已有长文本无明显额外收益。
- Continual pretraining 从 4k 切换到 32k（在 20%-40% 训练进度时）可节省约 40% FLOPs，性能几乎不变。
- Instruction tuning 中加入 LM loss on input prompts 和 self-instruct data 是两个最关键的提升因素。
