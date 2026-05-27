# SyTTA — You Only Need 4 Extra Tokens: Synergistic Test-time Adaptation for LLMs

> **加入 Survey 时间：** 2026-03-11

**Method:** SyTTA | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | domain/task cohort |
| 参数/状态持久性 Persistence | adapter accumulates within the cohort; typically reinitialized for a new cohort |
| 与推理关系 Inference Coupling | adapt on the cohort first, then infer with the adapted adapter |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Cohort Boundary |
| 证据状态 Evidence Status | note-explicit / protocol-inferred |
| 辅助分类 Timing Regime | Full-Cohort Transductive Adaptation |
| 可见数据范围 Visibility Scope | Full target cohort |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Full-Cohort Transductive Adaptation`；`Visibility Scope=Full target cohort`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Cohort Boundary`。

- **更新何时触发：** 更新先在同一 domain / task 的无标签 cohort 上触发，再用适配后的 adapter 统一推理。
- **服务当前样本还是后续样本：** 当前 cohort 的更新主要服务这一 cohort 及其同分布样本，而不是某个单样本的即时推理。
- **参数/状态是否累积：** adapter 在 cohort 内持续累积；切换到新的 cohort / 分布时通常需要重新初始化，这一点多由协议隐含。
- **reset 边界：** 因此它更接近 cohort-level adapter adaptation，而不是 per-sample adapt-reset。

## 1. UPT 归属理由

SyTTA 属于 **Family I — Prediction-Statistic Optimization**（基于内在 LM 统计量的局部状态/adapter 塑形）。

- **信号来源完全内在：** SyTTA 的两个优化目标均来自模型自身的内在统计量，不依赖任何外部标签、人类反馈或外部验证器：
  - *Input Distribution Adaptation (IDA)*：最小化输入 token 的 perplexity（即 negative log-likelihood），这是模型自身对输入分布的内在度量。
  - *Output Confidence Shaping (OCS)*：最小化输出 next-token 分布的 predictive entropy，并通过 reverse KL 散度约束适配模型与 base model 的距离——全部基于模型自身的 logit 分布。
- **载体为轻量级 LoRA adapter：** 在 test-time 通过 LoRA（rank=8，仅作用于 q_proj 和 v_proj）进行参数更新，属于 direct optimization 范式。
- **Token 级操作：** 适配过程仅生成一个极短的 prefix（通常 k=4 个 token），在该 prefix 上计算 entropy 和 KL 信号，实现 token-level 的优化。
- **无外部监督：** 整个流程为 "question-only" 设定，仅接收未标注的领域问题，不访问 ground-truth 答案。

---

## 2. 解决的问题

LLM 在专业领域（金融、医学、农业等）部署时，面临 **distribution shift** 问题：领域特定的术语、语言模式和知识需求与预训练数据分布存在显著差异，导致模型在这些领域表现退化。

传统解决方案（SFT、RLHF、RAG）均需要高质量标注数据或外部知识库，在标签稀缺的领域中难以适用。SyTTA 的目标是在 **推理阶段（test-time）** 利用无标签的领域问题，通过自监督信号对模型进行轻量级适配，弥合训练分布与部署分布之间的差距。

核心挑战在于：
- 仅优化输入端 perplexity 不能直接控制生成质量
- 仅优化输出端 entropy 容易导致 degeneration 和 collapse
- 需要在极低的计算开销下（每个 query 仅增加 4 个 token）实现稳定有效的适配

---

## 3. 方法介绍

SyTTA 通过耦合两个互补的自监督信号，在共享的短 prefix 上下文中实现 test-time adaptation：

### 3.1 Input Distribution Adaptation (IDA)

最小化输入问题的 prompt perplexity（等价于 NLL）：

$$\mathcal{L}_{\text{IDA}}(\theta') = -\frac{1}{m}\sum_{i=1}^{m} \log p_{\theta'}(x_i \mid x_{<i})$$

- 采用 **gating 机制**：仅对 base model 下 NLL 超过阈值的困难样本进行优化
- 对选中的样本，loss 按其 NLL 成比例放大，聚焦于更难的输入

### 3.2 Output Confidence Shaping (OCS)

对模型生成的短 prefix $\bar{y}(x)$（长度 k，通常 k=4）进行输出端正则化：

- **Entropy 项**：最小化适配模型在 prefix 上的 next-token predictive entropy
  $$\mathcal{L}_{\text{ENT}}(\theta') = \sum_{t=1}^{k} H(p_{\theta'}(\cdot \mid x, \bar{y}_{<t}))$$
- **Reverse KL 项**：约束适配模型与 base model 参考分布的距离，防止 drift 和 collapse
  $$\mathcal{L}_{\text{KL}}(\theta') = \sum_{t=1}^{k} D_{\text{KL}}(p_{\theta'}(\cdot \mid x, \bar{y}_{<t}) \| \text{softmax}(z_t^{\text{ref}}(x)))$$
- 组合为 $\mathcal{L}_{\text{OCS}} = \mathcal{L}_{\text{ENT}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$

### 3.3 两种部署模式

- **Static-Ref**：prefix 和 base-model reference logits 由冻结的 base model 预计算并缓存，适配仅需一次 forward pass，计算开销最低（推荐的默认模式）
- **Dynamic-Ref**：prefix 由适配中的模型实时生成，reference logits 在线计算，适用于需要紧密跟踪解码轨迹的场景

### 3.4 Dynamic Importance Weighting (DIW)

动态平衡 IDA 和 OCS 两个目标的权重：

$$\mathcal{L}_{\text{total}}(\theta') = w_{\text{IDA}}^{(t)} \mathcal{L}_{\text{IDA}}(\theta') + w_{\text{OCS}}^{(t)} \mathcal{L}_{\text{OCS}}(\theta')$$

- 使用 EMA 跟踪总 loss 量级，计算各目标的相对贡献比
- 通过 clipping 机制（$\alpha_{\min}, \alpha_{\max}$）防止任一目标主导
- 权重归一化保持总梯度量级稳定

### 3.5 实现细节

- 使用 LoRA（rank=8）仅更新 q_proj 和 v_proj
- 学习率 $1 \times 10^{-5}$，cosine scheduler，1 epoch
- Greedy decoding，基于 vLLM 引擎加速
- Cohort-level adaptation：对一批同领域问题做一次自监督适配，然后冻结参数生成答案

---

## 4. 数据集

基于 **AdaptEval** benchmark suite 进行评估，包含两大类共 7 个数据集：

### DomainBench（领域知识）
| 数据集 | 领域 |
|--------|------|
| Agriculture (KisanVaani) | 农业 |
| GeoSignal | 地理信号 |
| GenMedGPT | 医学（GPT 生成的临床文本） |
| Wealth | 金融/财富管理 |

### InstructBench（指令跟随）
| 数据集 | 特点 |
|--------|------|
| Dolly | 开源指令微调数据 |
| Alpaca-GPT4 | GPT-4 生成的指令数据 |
| InstructionWild | 多样化真实指令 |

---

## 5. 评估指标与主要结果

### 评估指标

- **ROUGE-L_sum**（×100）：主要指标，衡量生成答案与参考答案之间的句子级重叠

### 主要结果（Table 2）

在 4 个 base model（LLaMA 3.2-3B、LLaMA 3.1-8B、Qwen 2.5-7B、Qwen 2.5-14B）上与 base model、Tent、EATA、TLM 对比：

| 关键发现 | 详情 |
|----------|------|
| **一致性提升** | SyTTA 在所有模型和数据集上均优于 base model 及先前 test-time adaptation 方法 |
| **最大增益** | 在 Agriculture + Qwen 2.5-7B 上，SyTTA（仅 4 extra tokens）将 ROUGE-L_sum 提升超过 **120%** |
| **Entropy-only 方法失败** | Tent 和 EATA 在自回归 LLM 上常常 collapse 至接近零分 |
| **TLM 不稳定** | 仅优化 perplexity 的 TLM 是较强基线，但在 Dolly 上出现 collapse，且很少达到最优 |
| **平均提升幅度** | DomainBench 提升 40-60%，InstructBench 提升 15-22% |
| **Static-Ref vs Dynamic-Ref** | Static-Ref 在多数情况下更稳定且性能相当或更优，推荐作为默认部署模式 |

### 关键分析结论

- **Prefix 长度**：k=4 优于 k=16，大部分有用的适配信号集中在最初几个高 entropy token 中
- **KL 正则化**：有效防止 adaptation drift，在 Dynamic-Ref 模式下增益更大（起 trust-region 约束作用）
- **Dynamic Importance Weighting**：在大多数模型上提升 ROUGE-L_sum，对 LLaMA 系列效果更显著
- **模型规模趋势**：LLaMA 系列中更大模型受益更多；Qwen 系列中 7B 优于 14B（因 Qwen 更强的 post-training 对齐留给 test-time adaptation 的空间更小）
