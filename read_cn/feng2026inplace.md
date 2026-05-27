# In-Place Test-Time Training

> **加入 Survey 时间：** 2026-03-11

| 属性 | 值 |
|---|---|
| Method | In-Place TTT |
| Title | In-Place Test-Time Training |
| Carrier | Direct Opt. |
| Regime | test-time |
| Level | Token |
| Venue | ICLR 2026 |
| 作者 | Guhao Feng, Shengjie Luo, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai |
| 机构 | ByteDance Seed; 北京大学通用人工智能国家重点实验室 |

| When to Adapt | Within-Sequence Adaptation within the current instance |
|---|---|
| 触发单位 Trigger Unit | within-sequence chunk / token block |
| 参数/状态持久性 Persistence | fast weights or inference state persist across chunks, reset at document boundary |
| 与推理关系 Inference Coupling | interleaved adapt-and-infer within the same sequence |
| 输入可见性 Input Visibility | Online |
| Update Persistence | Non-Cumulative |
| 重置边界 Reset Boundary | Sequence Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Within-Sequence Adaptation |
| 可见数据范围 Visibility Scope | Current Sequence Prefix Only |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Within-Sequence Adaptation`；`Visibility Scope=Current Sequence Prefix Only`。
- **两轴编码：** `Input Visibility=Online`；`Update Persistence=Non-Cumulative`；`Reset Boundary=Sequence Boundary`。

- **更新何时触发：** 更新在同一序列内部按 chunk / block 持续触发，而不是在样本之间才触发。
- **服务当前样本还是后续样本：** 当前 chunk 的更新直接服务后续 chunk 的推理，因此 adaptation 与 inference 在同一序列里交织进行。
- **参数/状态是否累积：** fast weights / inference state 会跨 chunk 持续存在，但到 document boundary 会 reset。
- **reset 边界：** 因此它既不是 Offline Corpus UPT，也不是 test-time-instance adapt-then-reset，而是 within-sequence online update。

## 1. UPT 归属理由

本文属于 **Family I: Prediction-Statistic Optimization (predictive likelihood minimization)**。

- **优化信号完全来自模型内部**：In-Place TTT 在 test-time 对自回归输入流进行在线优化，唯一的监督信号是 language modeling loss（next-token prediction loss），不依赖任何外部标注、验证器或人类反馈。
- **直接优化模型参数**：框架将 MLP block 的 final projection matrix (W_down) 作为 fast weights，在推理时通过梯度下降直接原地更新，属于 direct optimization carrier。
- **Token 级别的自适应**：以 chunk-wise 方式逐 chunk 更新 fast weights，本质上是 token-level 的在线适应，每个 chunk 处理后参数即被更新以服务后续 token。
- **无外部信号**：训练目标 V = Conv1D(X_0) W_target 完全由输入 token embeddings 自身导出，是 intrinsic statistics 的直接利用。

---

## 2. 解决的问题

现有 LLM 采用"先训练后部署"的静态范式，推理时权重固定，无法动态适应输入流中的新信息。Test-Time Training (TTT) 提供了一种在推理时更新部分参数（fast weights）的方案，但在 LLM 生态中面临三大障碍：

1. **架构不兼容**：现有 TTT 方法通常引入全新的专用层替换 attention，要求从头预训练，无法直接应用于已有的大规模预训练模型。
2. **计算效率低**：经典 TTT 的 per-token 逐 token 更新机制本质上是串行的，无法充分利用 GPU/TPU 的并行能力。
3. **目标函数不匹配**：先前 TTT 普遍使用 reconstruction objective（重建当前 token），与 LLM 的核心目标——预测下一个 token——并不直接对齐，可能导致 fast weights 存储的信息对预测无益。

---

## 3. 方法介绍

### 3.1 整体框架：原地复用 MLP Block

核心洞察：不引入新层，而是将现有 Transformer 中无处不在的 gated MLP block 的 final projection matrix W_down 复用为 fast weights。W_up 和 W_gate 保持冻结（slow weights），仅 W_down 在推理时在线更新。这是一种"drop-in"设计，不改变模型架构，可直接应用于任意预训练 LLM。

### 3.2 Chunk-wise 高效更新

将输入序列按 chunk size C 分为不重叠的 chunk。对每个 chunk 执行两步操作：

1. **Apply**：用当前 fast weights W_down^(i) 计算该 chunk 的输出 O_[i] = Z_[i] (W_down^(i))^T。
2. **Update**：用 Z_[i] 作为 key、V_[i] 作为 value，通过一步梯度下降更新 fast weights：W_down^(i+1) = W_down^(i) - eta * grad L(...)。

该 chunk-wise 策略天然适合并行计算，支持 chunk size 512-1024，在 GPU/TPU 上可高效执行。

### 3.3 LM-Aligned Objective

不同于传统 TTT 使用 reconstruction target（V_t = E_{x_t}，重建当前 token embedding），本文提出 LM-Aligned target：

- 目标 V = Conv1D(X_0) W_target，其中 Conv1D 引入未来 token 信息（可设为仅看下一个 token），W_target 是可训练的投影矩阵。
- 损失函数采用负余弦相似度：L(a, b) = -<a, b>_F。
- 理论证明（Theorem 1）：LM-Aligned target 能保证正确的下一个 token 的 logit 增大（期望增量 >= lambda_lr * c_norm^2 * c_align），而其他 token logit 几乎不变；相比之下，reconstruction target 对正确 token 的 logit 变化可忽略不计。

### 3.4 Context Parallelism

更新规则具有结合律（associative），支持 context-parallel 实现：(i) 并行计算各 chunk 的中间激活和 delta W；(ii) prefix sum 聚合更新；(iii) 并行计算各 chunk 的输出。实际实现中通过 causal padding 保证因果性，文档边界处重置 fast weights。

---

## 4. 数据集

### Drop-in Enhancement 实验（Section 4.1）
- **训练数据**：约 20B tokens（32k context）+ 约 15B tokens（128k context）的 continual training curriculum
- **评估**：RULER benchmark（4k-256k context lengths）

### Pre-training from Scratch 实验（Section 4.2）
- **训练数据**：500M 和 1.5B 模型在 32k context length 序列上训练；4B 模型用 120B tokens（8k context）训练
- **评估**：
  - Sliding Window Perplexity: Pile validation set, Proof-Pile-2
  - 常识推理: HellaSwag, ARC-E, ARC-C, MMLU, PIQA
  - 长上下文: RULER (4k, 8k, 16k)

---

## 5. 评估指标与主要结果

### 评估指标
- **RULER score**：长上下文综合评测，报告 average accuracy (%)
- **Sliding Window Perplexity**：在固定末尾 block 上计算 perplexity，衡量长上下文利用能力
- **常识推理 accuracy**：HellaSwag, ARC-E, ARC-C, MMLU, PIQA 的准确率

### 主要结果

**Drop-in Enhancement (Qwen3-4B-Base, RULER)**

| Context Length | 4k | 8k | 16k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|---|
| Baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| **In-Place TTT** | **96.1** | **95.6** | **92.7** | **89.3** | **78.7** | **77.0** | **43.9** |

- 在长上下文（64k, 128k）上提升显著（+4.4, +2.2），256k 外推也有提升。
- 扩展到 LLaMA-3.1-8B 和 Qwen3-14B-Base 同样有效（64k 分别 +2.1, +2.7）。

**Pre-training from Scratch (Sliding Window Perplexity)**
- 500M 和 1.5B 模型上，In-Place TTT 在所有 context length (2k-32k) 上均取得最低 perplexity，持续优于 SWA, GLA, DeltaNet, LaCT 等竞争方法。

**4B 模型常识推理与长上下文**

| Architecture | HellaSwag | ARC-E | ARC-C | MMLU | PIQA | RULER-4k | RULER-8k | RULER-16k |
|---|---|---|---|---|---|---|---|---|
| Full Attn. (Baseline) | 55.67 | 64.52 | 33.19 | 36.43 | 72.63 | 45.77 | 38.09 | 6.58 |
| Full Attn. + I.P. TTT | **55.85** | **64.98** | **32.34** | **37.42** | **73.29** | **49.98** | **43.82** | **19.99** |
| SWA (Baseline) | 54.92 | 64.18 | 32.85 | 36.06 | 72.58 | 14.77 | 9.91 | 5.07 |
| SWA + I.P. TTT | 55.24 | 64.60 | **33.70** | 36.48 | 72.03 | **28.33** | **26.80** | **7.57** |

- 长上下文评估提升极为显著（RULER-16k: 6.58 -> 19.99），常识推理也有小幅改善。

### Ablation 关键发现
- **State size**：启用更多 TTT 层（更大 fast weights）性能持续提升。
- **Chunk size**：C=512 和 C=1024 为最优，过小或过大均不理想。
- **LM-Aligned Objective**：Conv1D 和 W_target 投影均不可或缺；Conv1D 对长上下文尤为关键，W_target 对短上下文至关重要。
- **效率**：In-Place TTT 引入的额外 throughput 和 memory 开销可忽略不计。
