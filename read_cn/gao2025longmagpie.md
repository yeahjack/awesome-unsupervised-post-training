# LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions

> **加入 Survey 时间：** 2026-03-11

> **论文元信息**
> - **arXiv**: 2505.17134v2 [cs.CL] 3 Jun 2025
> - **作者**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu
> - **机构**: Institute of Information Engineering (CAS), University of Chinese Academy of Sciences, Xiaohongshu Inc, Tsinghua University

| 属性 | 值 |
|---|---|
| Method | LongMagpie |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |

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

LongMagpie 属于 **Family III — Self-Generated Target Bootstrapping** 中的 **reasoning / plan / curriculum synthesis** 子类。其核心思路是：利用 aligned long-context LLM 自身的 auto-regressive 能力，在仅给定文档（document）和 user turn 前置 special tokens 的情况下，自动生成与文档相关的 query，再由模型自身产生 response，从而构成完整的 instruction triplet (D, Q, R)。整个过程不依赖人工标注、seed instruction 或外部 teacher model，完全由模型自合成（self-synthesis）长上下文 instruction data，随后用这些数据对模型进行 SFT 对齐。这是一种典型的"模型生成训练目标数据、再用该数据优化自身"的 bootstrapping 范式，属于 training-time 阶段的 semantic-level 直接优化。

---

## 2. 解决的问题

高质量的 long-context instruction data 是对齐长上下文 LLM 的关键需求，但目前面临三大瓶颈：

1. **数据封闭**：Qwen、Llama 等模型虽然开源了权重，但对应的 long-context instruction dataset 仍然私有，形成 closed-data 壁垒。
2. **人工标注成本高**：长上下文数据的标注难度远超短上下文——标注者需要阅读数千 token 的文档后才能撰写 instruction，耗时且品质难以保证。
3. **现有合成方法局限**：基于预定义 template（Sanh et al., 2021）或 seed question（Self-Instruct, Wang et al., 2022）的方法在 diversity、scale 和 quality 上存在瓶颈；而 ChatQA、LongAlign 等方法虽尝试拓展 seed 多样性，但流程复杂且开销巨大（合成每条 instruction 平均需要 10–13× 更多 token）。

LongMagpie 旨在提供一种 **无需人工标注、无需 seed instruction、无需复杂 pipeline** 的可扩展长上下文 instruction data 自合成方案。

---

## 3. 方法介绍（含图表描述）

### 3.1 核心洞察：Auto-Regressive Document-Query Generation

LongMagpie 的核心观察是：经过 instruction tuning 的长上下文 LLM 在训练过程中已经内化了 document-query 的关系模式。因此，当模型接收到一段文档 $D$ 后紧跟 user turn 前置 token $T_{pre}$（如 `<|im_start|>user` 或 `<|start_header_id|>user`），模型会 auto-regressively 生成与该文档语义相关的 query $Q$：

$$p_M(Q \mid D, T_{pre}) = \prod_{i=1}^{k} p_M(q_i \mid D, T_{pre}, q_{<i})$$

这不同于传统 prompt engineering 或 instruction-following：模型并非被显式指令生成问题，而是利用其在 instruction tuning 中学习到的 document-query 关系模式自动完成。

### 3.2 LongMagpie Pipeline

> **Figure 1**（论文第 2 页）展示了 LongMagpie 的完整 pipeline 概览，分为两个阶段：
> - **Stage 1**：将 document 作为 system prompt，用 special user token 触发 query generation，随后模型生成 response。
> - **Stage 2**：将 query-response pair 与原始文档及从 corpus 中采样的额外文档组合，构造具有挑战性的 multi-document long-instruction data。

#### Step 1: Query and Answer Generation

- **Document Preparation**：从 FineWeb-Edu 等多样化语料库收集文档，涵盖 science、history、literature、technical 等领域，平均长度约 1.6k tokens。
- **Query Generation**：对每篇文档 $D$，构造输入 $X = D \oplus T_{pre}$（其中 $T_{pre}$ 为 user turn 前置 token），送入 aligned LLM 采样生成 query $Q$。通过不同 sampling parameters 对每篇文档生成多条 query，自然获得 complexity 多样化的 document-query pairs。
- **Response Generation**：对每个 $(D, Q)$ 对，构造标准 instruction prompt（拼接 document、query 及 assistant 前置 token），生成 response $R$，形成完整 instruction triplet $(D, Q, R)$。若同一模型同时承担 query 和 response 生成，则全过程可无人工介入地一步完成。
- **Query Filtering**：采用两种过滤策略应对模型偶尔延续文档而非生成 query 的问题：(1) **Rule-based filtering**：保留以问号结尾的 query；(2) **Length-based filtering**：丢弃超过 1.5k 字符的文本（通常为继续生成的描述性段落）。

#### Step 2: Multi-Document Extension

为提升 task diversity 和实际应用性，将 pipeline 扩展到 multi-document 场景：

- 随机采样 $x$ 篇文档 $\{D_1, \ldots, D_x\}$ 作为 negative documents（$x$ 从 $[0, n]$ 均匀采样，$n=0$ 退化为 single-document QA）。
- 使用特殊 separator token（如 `<|doc_sep|>`）拼接为 $D_{multi} = D_1 \oplus \text{<|doc\_sep|>} \oplus \cdots \oplus D_x$。
- 在 multi-document context 上按原 pipeline 生成 query 和 response，得到要求 cross-document reasoning 的 triplet $(D_{multi}, Q, R)$。

### 3.3 p-Mix: Balancing Long-Context and Short-Context Capabilities

仅用长上下文数据训练会导致 short-context task 性能退化。LongMagpie 提出 **p-Mix** 策略：

1. **Short-context 前置**：每条训练 sequence 开头以一条 short-context instruction 开始，模拟通用任务的 non-contextual 起始模式。
2. **概率混合**：之后以概率 $P_L$ 追加一条 LongMagpie 生成的 long-context instruction，以概率 $1 - P_L$ 追加另一条 short-context instruction，迭代直至接近最大序列长度 $L_{max}$。

> **Algorithm 1**（论文第 16 页）给出了完整的伪代码：`CONSTRUCTHYBRIDSAMPLE(DS, DL, PL, Lmax, sep)`——先从 short-context set $D_S$ 中采样一条放在序列开头，然后循环中以 $P_L$ 概率交替选取 long/short sample 拼接，直到超出 $L_{max}$。

p-Mix 有效防止模型 overfit 到长上下文模式，在保持长上下文强表现的同时维持短上下文竞争力。

---

## 4. 数据集

### 生成数据

- **Source model**: Qwen2.5-70B-Instruct
- **Document corpus**: FineWeb-Edu（1.3 trillion tokens 的教育类 web 内容子集）
- **数据规模**: 450k long-context instruction samples（消融实验中也测试了 190k）
- **文档平均长度**: ~1.6k tokens

### 对比数据集

**Long Instruction Datasets**:
- **ChatQA (ChatQA2)**: 组合多个数据源（含 LongAlpaca12k、GPT-4 samples from Open Orca），共 150 万条 synthetic instructions
- **LongAlign**: 通过 prompted LLMs 对长文档生成 QA

**Short Instruction Datasets**（拼接至目标长度用于 long-context fine-tuning）:
- **Tulu**: 基于 Llama 3.1 的 open-source collection
- **Magpie**: 使用 template prefix 的 self-synthesis 方法
- **UltraChat**: 150 万条 multi-turn dialogues

### 训练配置

- **Base model**: Llama-3-8B-NExtLong-512K-Base（经过 long-context continued pre-training）
- **Batch size**: 4M tokens, 250 steps, 共 1B tokens
- **Sequence length**: 64K
- **Optimizer**: AdamW ($\beta_1=0.9$, $\beta_2=0.95$), lr = 2e-5, cosine decay
- **技术**: FlashAttention-2, ZeRO, document masking, bfloat16
- **硬件**: 8× H100, ~10h training

---

## 5. 评估指标与主要结果

### 评估 Benchmarks

**Long-context Evaluation**:
- **HELMET**: 评估长上下文模型在多种 application-centered tasks 上的表现，context length 达 128k tokens
- **RULER**: 提供 fine-grained synthetic tasks，灵活控制 sequence length 和 complexity，识别 retrieval 之外的性能瓶颈
- **LongBench-v2**: 评估 8k–2M words 的极长上下文理解，503 个 expert-validated questions，6 个类别

**Short-context Evaluation**: HellaSwag, Lambada_OpenAI, ARC-Challenge, ARC-Easy, PIQA, WinoGrande, Logiqa

### 主要结果

> **Table 1**（论文第 6 页）汇总了 LongMagpie 与其他方法在长/短上下文 benchmark 上的对比。

| Dataset | HELMET | RULER | LongBench v2 | LongAVG | ShortAVG |
|---|---|---|---|---|---|
| **Short Instruction Data** | | | | | |
| Tulu | 61.93 | 87.92 | 28.4 | 59.42 | 63.90 |
| Magpie | 60.18 | 87.06 | 31.4 | 59.55 | 63.32 |
| UltraChat | 60.55 | 83.85 | 30.4 | 58.27 | 64.43 |
| **Long Instruction Data** | | | | | |
| ChatQA | 60.23 | 89.82 | 30.8 | 60.28 | 63.58 |
| LongAlign | 57.79 | 86.08 | 24.5 | 56.12 | 60.97 |
| **LongMagpie** | **62.10** | **91.17** | **34.4** | **62.56** | 62.37 |
| **p-Mix: Long + Short** | | | | | |
| ChatQA + UltraChat | 60.80 | 87.42 | 31.4 | 59.87 | 64.38 |
| LongAlign + UltraChat | 60.98 | 89.49 | 30.6 | 60.36 | 64.17 |
| **LongMagpie + UltraChat** | **62.11** | 89.70 | 33.0 | **61.60** | **64.10** |

**关键发现**：

1. **Long-context 领先**：仅用 LongMagpie 数据训练即在所有长上下文 benchmark 上取得最优，LongAVG 达 62.56，比 ChatQA 高 +2.28，比 LongAlign 高 +6.44。
2. **p-Mix 平衡长短**：LongMagpie + UltraChat（p-Mix）在长上下文上保持领先（LongAVG 61.60），同时 ShortAVG 达 64.10（仅比最高差 0.33），有效解决了长上下文训练导致的短任务退化问题。

### 消融实验

> **Table 2**: Multi-document 数量 $n$ 的影响——$n=10$ 时 LongAVG 最高（62.56），过多文档（$n>20$）会因任务难度超出模型学习能力而导致性能下降。

> **Table 3**: Mixing strategy 对比——p-Mix 在 LongAVG（61.60）和 ShortAVG（64.10）上均优于 Sequential Mix（60.75/61.89）和 Simple Mix（60.90/64.04），实现了最佳平衡。

> **Table 4**: 数据规模影响——从 190k 扩展到 450k samples，LongAVG 从 61.51 提升至 62.56（+1.05），证明更大规模的高质量数据持续带来收益。

> **Table 5**: Source model 大小影响——Qwen-2.5-70B 生成数据的 LongAVG（62.56）远优于 Qwen-2.5-7B（59.61），更大模型具有更强的 long-context 能力建模，转化为更高质量的合成数据。

> **Figure 2**（论文第 8 页）：(a) Reward model 打分分布显示 LongMagpie 数据质量显著高于 ChatQA 和 LongAlign；(b) Pairwise query similarity 分析表明 LongMagpie query 之间的相似度更低，反映更好的 diversity。

> **Figure 3**（论文第 9 页）：(a, b) t-SNE embedding 可视化显示 LongMagpie query 分布更加 dispersed（多样化）；(c) Long-context performance vs. token consumption 图表明 LongMagpie 以平均 1.6k tokens/instruction 的极低开销（ChatQA/LongAlign 为 10–13× 更多）实现了最优的长上下文性能，展现了卓越的 sample efficiency。

### p-Mix 参数消融

> **Table 7**（Appendix）：$N_S=1, P_L=0.4$ 为最优参数配置（LongAVG 61.60, ShortAVG 64.10）。$P_L$ 过高会过度倾斜至长上下文、损害短任务；$N_S$ 过大（如 30）则过多短 instruction 稀释了长上下文信号，导致 LongAVG 下降。
