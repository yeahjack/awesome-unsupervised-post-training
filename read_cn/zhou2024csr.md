# CSR: Calibrated Self-Rewarding Vision Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Calibrated Self-Rewarding Vision Language Models
**Authors:** Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, Huaxiu Yao (UNC-Chapel Hill, UChicago, UMD, Rutgers, HKUST, PolyU, NTU, NUS)
**ArXiv:** 2405.14622
**Date:** 2024-05 (NeurIPS 2024)

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CSR | Pref. Opt. | training-time | Semantic |

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

**Adjacent — 外部 encoder 协助 self-rewarding（未通过 B4）；shadow placement: Family IV — Internal Evaluator Bootstrapping (self-rewarding)**

> **Status update（对齐论文）：** 按论文 §6.2 与 figure tree，CSR **被路由到 adjacent**，不属于 strict UPT。未通过的边界检查是 **B4**（更新中所用的任何 judge / scorer / reward model 必须来自同一模型谱系）：calibrated reward 是 (i) LVLM 自身 language decoder 的 cumulative probability 与 (ii) **CLIP-derived visual-relevance score** 的加和，其中 CLIP 是 LVLM 谱系之外的预训练 frozen encoder。CLIP 项注入了来自非同谱系 scorer 的监督，因此整个系统未通过 B4，被列入 adjacent 表而不算入 Family IV。我们仍保留 Family IV 的 *shadow* 归属，仅因为 preference-pair 构造的主导者是 internal judge；正式分类为 adjacent。

CSR 将 self-rewarding language models 的范式从纯文本 LLM 推广到视觉-语言模型 (LVLM)。其核心思想契合 internal evaluator bootstrapping：**模型自身同时充当 response generator 和 reward judge**，无需任何外部人工标注或额外模型（如 GPT-4）提供偏好信号。

为何这是 strict UPT 的 "近邻" 但最终归为 adjacent：
1. **无人工标注**：偏好数据完全由模型自身产生，通过 sentence-level beam search 生成候选 response，并利用自身 language decoder 的 cumulative probability 作为 instruction-following reward。
2. **Internal evaluator (主导项)**：模型自己评估自己的输出质量，通过 calibrated reward score（融合 language score 与 CLIP-based visual relevance score）对 candidate response 进行排序，选出 preferred/dispreferred pair。
3. **Iterative bootstrapping**：通过多轮迭代 DPO fine-tuning，模型不断改进自身的偏好判断能力和生成质量，呈现典型的 self-improvement 循环。
4. **CLIP 项（B4 违反所在）**：reward 中的 visual relevance 项使用预训练 CLIP，是 LVLM 同模型谱系之外的外部 frozen encoder。按论文 §6.2，仅此一项即足以把整个 CSR 系统路由到 adjacent（Table~\ref{tab:adjacent-methods}），即便闭环其余部分都是内部的。

其特殊贡献在于引入 **visual constraints 校准 self-rewarding 过程**：由于 LVLM 存在 modality misalignment（倾向于优先利用 text knowledge 而忽略 visual input），直接将 LLM self-rewarding 搬到 LVLM 会导致 self-generated preference 同样忽略图像信息。CSR 通过在 reward 中加入 image-response relevance score 来校准这一偏差。

---

## 2. 解决的问题

LVLM 的 **hallucination 问题**，即模型生成的文本描述在语言上看似合理，但与输入图像中的视觉信息相矛盾。根本原因是 **modality misalignment**——模型倾向于优先利用训练语料中的文本知识，而忽略实际的视觉输入。

现有解决方案的局限：
- **依赖外部模型（如 GPT-4）或人工标注** 来生成偏好数据，代价高昂且可能引入额外偏差。
- 外部生成的偏好数据 **无法捕捉目标 LVLM 自身的固有偏好**，导致 curated preference 容易被模型区分（easily distinguishable），从而降低 preference learning 的有效性。
- 直接将 LLM 的 self-rewarding 方法应用到 LVLM 上同样存在问题：LVLM 在 response generation 和 preference modeling 两个阶段都存在 modality misalignment，self-generated preference 可能同样忽略视觉信息。

---

## 3. 方法介绍

CSR 是一个 **iterative preference optimization framework**，交替执行两个阶段：(1) candidate response generation 和 (2) preference curation & fine-tuning。

### 3.1 Step-Level Reward Modeling and Calibration

CSR 的 reward 设计满足两个准则：
- **Vision-Constrained Reward**：在 reward 定义中融入 image-relevance 信息，解决 LVLM 在生成偏好时忽略图像输入的问题。
- **Step-Wise Reward**：不为整个 response 分配单一 reward，而是在每个生成步骤（sentence level）分配 reward，提供更细粒度的指导。

#### (a) Self-Generated Instruction-Following Score $R_T(s)$

利用 LVLM 的 language decoder 计算 sentence-level cumulative probability：

$$R_T(s) = \prod_{o=1}^{N_o} P(r_o \mid x, r_1, r_2, \ldots, r_{o-1})$$

其中 $N_o$ 是 sentence $s$ 中的 token 数量，$r_o$ 是第 $o$ 个 token。分数越高表示生成的 response 越符合 instruction-following 能力。

#### (b) Image-Response Relevance Score $R_I(s)$

利用 CLIP score 衡量生成 sentence 与输入 image 之间的相关性：

$$R_I(s) = \max(100 \times \cos(F_I(x_v), F_T(s)), 0)$$

其中 $F_I(x_v)$ 和 $F_T(s)$ 分别是 CLIP 的 visual embedding 和 textual embedding。CLIP 中的 vision encoder 与目标 LVLM 中的 vision encoder 保持一致。

#### (c) Calibrated Reward Score $R(s)$

$$R(s) = \lambda \cdot R_I(s) + (1 - \lambda) \cdot R_T(s)$$

其中 $\lambda$ 是平衡 image-response relevance 和 language instruction-following 的超参数。实验中设置 $\lambda = 0.9$（CLIP score 权重 0.9，language score 权重 0.1），即 **大幅侧重 visual calibration**。

### 3.2 Iterative Fine-Tuning

#### Step 1: Step-Level Candidate Response Generation

采用 **sentence-level beam search** 逐句生成候选 response：
1. 同时采样多个候选 sentence，以句号 "." 作为 sub-sentence 边界（delimiter）
2. 对每个 sentence $s$ 计算 calibrated reward score $R(s)$
3. 选取 reward 最高的 top-k 和最低的 bottom-k sentences 进入下一轮 beam search
4. 重复直到生成完整 response

超参设置：`num_beams=5`，`num_token_beams=5`，`num_beam_group=5`（group beam search 增强多样性），`diversity_penalty=3.0`，`max_length=1024`，`max_new_tokens=74`。

#### Step 2: Preference Curation and Optimization

对每个 input prompt，选取 cumulative calibrated reward score **最高**和**最低**的 response 分别作为 preferred ($y_w$) 和 dispreferred ($y_l$) response，构成 preference pair。

使用 **DPO** 进行 fine-tuning，每次迭代 $t$ 的损失函数：

$$L_t = -\mathbb{E}_{(x, y_{w,t}, y_{l,t}) \sim D} \left[ \log \sigma \left( \alpha \log \frac{\pi_\theta(y_{w,t} \mid x)}{\pi_{\theta_{t-1}}(y_{w,t} \mid x)} - \alpha \log \frac{\pi_\theta(y_{l,t} \mid x)}{\pi_{\theta_{t-1}}(y_{l,t} \mid x)} \right) \right]$$

其中 $\pi_{\theta_{t-1}}$ 是上一轮迭代 fine-tuned 的模型作为 reference model。

### 3.3 迭代过程

完整流程（Algorithm 1）：
1. 输入 dataset $D = \{x^{(i)}\}_{i=1}^N$、reference model $\pi_{\text{ref}}$、迭代次数 $T$
2. 每轮迭代：对每个 input 执行 sentence-level beam search → 计算 $R_T(s)$、$R_I(s)$、$R(s)$ → 选择 top/bottom response → 用 DPO 更新模型
3. 更新后的模型作为下一轮的 seed model 和 reference model

### 3.4 理论分析

论文提供了理论框架说明引入 image-response relevance score 的有效性。核心结论（Theorem 5.1）：当模型倾向于 prioritize textual information over visual input（即 $\|\beta^{*\top} V_1^\top \beta^*\| \ll \|\beta^{*\top} V_2^\top \beta^*\|$）时，存在 $\lambda < 1$，使得 CSR（$\lambda < 1$）的 loss 严格小于不使用 image-response relevance score（$\lambda = 1$）的 loss。直觉上，CSR 通过 up-weight image signal 来补偿模型对 visual input 的忽视。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 训练数据 | LLaVA-150k (子集) | 从 detailed description 和 complex reasoning 子类中随机采样约 **13,000** 个 image-prompt 对用于构建 preference data |
| 综合评估 | MME | Perception (MME_P) + Cognition (MME_C)，14 个子任务 |
| 综合评估 | SEED-Bench | 19K multiple-choice，12 个维度 |
| 综合评估 | LLaVA-Bench (In-the-Wild) | 24 images, 60 questions |
| 综合评估 | MMBench | CircularEval + ChatGPT |
| 综合评估 | MM-Vet | 6 核心能力 × 16 整合任务 |
| 通用 VQA | ScienceQA (SQA^I) | ~21K multiple-choice 科学问题 |
| 通用 VQA | VizWiz | 31K+ 视觉问答（盲人拍照场景） |
| 通用 VQA | GQA | Scene graph-based, 22M 问题 |
| 幻觉评估 | POPE | Binary classification (Yes/No) 判断 hallucinated objects |
| 幻觉评估 | CHAIR (CHAIR_S / CHAIR_I) | 500 COCO 验证集图片上的 object hallucination，分 sentence-level 和 instance-level |

---

## 5. 评估指标与主要结果

### 评估指标

- **MME_P / MME_C**：感知/认知综合分数
- **SEED**：generative comprehension 准确率
- **LLaVA^W**：visual reasoning 多场景评估分数
- **MMBench (MMB)**：多模态综合评估
- **MM-Vet**：多能力整合评估
- **SQA^I**：ScienceQA 图像子集准确率
- **VizWiz**：VQA 准确率
- **GQA**：visual reasoning 准确率
- **POPE (acc / f1)**：hallucination binary classification
- **CHAIR_S / CHAIR_I**：object hallucination 比率（越低越好）

### 主要结果

#### Table 1: CSR 在 LLaVA-1.5 上的全面对比（3 轮迭代后）

**LLaVA-1.5-7B 结果：**

| Method | MME_P | MME_C | SEED | LLaVA^W | MMB | MM-Vet | SQA^I | VizWiz | GQA | POPE | CHAIR_S | CHAIR_I |
|--------|-------|-------|------|---------|-----|--------|-------|--------|-----|------|---------|---------|
| LLaVA-1.5-7B (base) | 1510.7 | 348.2 | 58.6 | 63.4 | 64.3 | 30.5 | 66.8 | 50.0 | 62.0 | 85.90 | 48.8 | 14.9 |
| +VLFeedback | 1432.7 | 321.8 | 59.3 | 62.1 | 64.0 | 31.2 | 66.2 | 52.6 | 63.2 | 83.72 | 40.3 | 13.2 |
| +POVID | 1452.8 | 325.3 | 60.2 | 68.7 | 64.9 | 31.8 | 68.8 | 53.6 | 61.7 | 86.90 | 35.2 | 8.3 |
| +RLHF-V | 1489.2 | 349.4 | 60.1 | 65.4 | 63.6 | 30.9 | 67.1 | 54.2 | 62.1 | 86.20 | 29.7 | 7.5 |
| +Self-rewarding | 1505.6 | 362.5 | 60.0 | 61.2 | 64.5 | 31.4 | 69.6 | 53.9 | 61.7 | 86.88 | 24.0 | 6.7 |
| **+CSR (Ours)** | **1524.2** | **367.9** | **60.3** | **71.1** | **65.4** | **33.9** | **70.7** | **54.1** | **62.3** | **87.01** | **21.0** | **6.0** |

**LLaVA-1.5-13B 结果：**

| Method | MME_P | MME_C | SEED | LLaVA^W | MMB | MM-Vet | SQA^I | VizWiz | GQA | POPE | CHAIR_S | CHAIR_I |
|--------|-------|-------|------|---------|-----|--------|-------|--------|-----|------|---------|---------|
| LLaVA-1.5-13B (base) | 1531.3 | 295.4 | 61.6 | 70.7 | 67.7 | 35.4 | 71.6 | 53.6 | 63.3 | 85.90 | 48.3 | 14.1 |
| +Self-rewarding | 1529.0 | 300.1 | 62.8 | 65.6 | 64.5 | 35.3 | 74.3 | 56.1 | 63.2 | 86.58 | 37.0 | 8.8 |
| **+CSR (Ours)** | **1530.6** | **303.9** | **62.9** | **74.7** | **68.8** | **37.8** | **75.1** | **56.8** | **63.7** | **87.30** | **28.0** | **7.3** |

#### 迭代提升（LLaVA-1.5-7B Hallucination）

| Iteration | POPE acc | POPE f1 | CHAIR_S↓ | CHAIR_I↓ |
|-----------|----------|---------|----------|----------|
| Base | 85.90 | 84.29 | 48.8 | 14.9 |
| +CSR iter-1 | 86.94 | 85.80 | 26.6 | 7.2 |
| +CSR iter-2 | 86.82 | 85.62 | 23.0 | 6.1 |
| +CSR iter-3 | 87.01 | 85.93 | 21.0 | 6.0 |
| +CSR iter-4 | 87.05 | 85.95 | 19.0 | 5.9 |
| +CSR iter-5 | 87.16 | 85.98 | 18.3 | 5.4 |

#### Ablation Study (Table 2)：Average Performance Score (100-point scale)

| Method | 7B | 13B |
|--------|-----|-----|
| Base | 66.61 | 68.08 |
| Only $R_T$ (language only) | 68.46 | 68.12 |
| Only $R_I$ (visual only) | 67.49 | 69.23 |
| **CSR (combined)** | **72.39** | **71.95** |

#### λ 消融实验 (LLaVA-1.5-7B, 3 轮迭代)

| λ | LLaVA^W | CHAIR_S↓ | CHAIR_I↓ |
|---|---------|----------|----------|
| 0.1 | 66.7 | 40.8 | 10.2 |
| 0.5 | 68.2 | 28.2 | 6.7 |
| 0.9 | 71.1 | 21.0 | 6.0 |

#### Vila 7B 兼容性验证 (3 轮 CSR 迭代)

| Method | MME_P | SEED | LLaVA^W | MM-Vet | VizWiz | POPE | CHAIR_S↓ |
|--------|-------|------|---------|--------|--------|------|----------|
| Vila 7B (base) | 1533.0 | 61.1 | 69.7 | 34.9 | 57.8 | 85.50 | 31.0 |
| +CSR iter-3 | 1542.2 | 63.4 | 74.3 | 39.8 | 62.7 | 87.31 | 28.0 |

### 关键发现

1. **CSR 持续迭代改进**：7B 模型经 3 轮在线迭代后在所有 benchmark 上平均提升约 **7.62%**，13B 模型提升约 **5.25%**。改进在 LLaVA^W (+8.9%) 和 CHAIR (+49.50%) 上尤为显著。
2. **显著减少 hallucination**：LLaVA-1.5-7B 的 CHAIR_S 从 48.8 降至 21.0（下降 57%），CHAIR_I 从 14.9 降至 6.0（下降 60%）。
3. **超越外部偏好数据方法**：CSR 优于依赖 GPT-4（VLFeedback/Silkie）、人工标注（LLaVA-RLHF）和外部模型（POVID）的方法，说明自适应 self-rewarding 能更好捕捉 target LVLM 的 inherent preference。
4. **Visual calibration 至关重要**：$\lambda=0.9$ 时效果最佳，表明大幅侧重 CLIP visual score 的 calibration 有效修正了模型对文本的偏倚。仅用 $R_T$ 或 $R_I$ 的效果均不如两者联合（CSR 在 7B 上达 72.39 vs Only $R_T$ 的 68.46 和 Only $R_I$ 的 67.49）。
5. **Attention 重分配**：通过 attention map 分析，CSR 有效增加了 visual tokens 的 attention score，减轻了对 contextual text 的 over-reliance。
6. **跨模型泛化**：在 Vila 7B 上同样有效，3 轮迭代后整体提升 3.37%，在 VizWiz (+8.48%) 和 MM-Vet (+14.0%) 上提升尤为明显。
7. **Reward score 与性能正相关**：随迭代增加，chosen reward 从 0.4885 → 0.5066，rejected reward 从 0.4551 → 0.4799，平均性能从 66.61 → 72.24，表明模型持续生成更高质量的偏好数据。
