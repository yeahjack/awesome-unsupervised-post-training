# PonderTTT: When to Ponder

> **加入 Survey 时间：** 2026-04-16

**论文：** When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training
**作者：** Gi Hyeon Sim et al.
**arXiv：** 2601.00894
**发表：** arXiv 2026

| Method | Carrier | Regime | Level |
|------|----------------|-------------|------------|
| PonderTTT | Direct Opt. / TTT fast weights | test-time | Token / chunk |

| When to Adapt | Within-Sequence selective TTT updates gated by self-supervised reconstruction loss |
|---|---|
| 触发单位 Trigger Unit | current code chunk / token block |
| 参数/状态持久性 Persistence | TTT fast weights update only on selected chunks; threshold adapted by EMA |
| 与推理关系 Inference Coupling | decide whether to update, then re-forward chunk with updated fast weights |
| 输入可见性 Input Visibility | Online |
| Update Persistence | Non-Cumulative |
| 重置边界 Reset Boundary | Sequence / run boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Within-Sequence Adaptation |
| 可见数据范围 Visibility Scope | Current Sequence Prefix Only |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Within-Sequence Adaptation`；`Visibility Scope=Current Sequence Prefix Only`。
- **两轴编码：** `Input Visibility=Online`；`Update Persistence=Non-Cumulative`；`Reset Boundary=Sequence / run boundary`。
- **更新何时触发：** 对每个 chunk，计算 TTT layer 的 self-supervised reconstruction loss；仅当 loss 超过 gating threshold 时执行更新。
- **服务当前样本还是后续样本：** 当前 chunk 的更新服务同一序列内后续的 next-token prediction。
- **参数/状态是否累积：** fast weights 在当前 sequence / run 内累积，独立样本之间一般不保留长期状态。
- **reset 边界：** sequence / inference run 边界。

## 1. UPT 归属理由

PonderTTT 是 strict UPT 候选，归入 **Family I: Prediction-Statistic Optimization**。它使用 TTT layer 的 self-supervised reconstruction loss 决定何时更新，不需要 ground-truth label；论文明确说明信号是 inference-compatible 的。

## 2. 解决的问题

标准 TTT 对每个输入都做统一更新，代价高且并非总是必要。PonderTTT 旨在决定"何时 ponder"：只有当当前 fast weights 不能很好表征输入 chunk 时才触发更新。

## 3. 方法介绍

核心机制：

- TTT layer 维护一个 fast weight `W_t`。
- 对每个输入 chunk，计算 reconstruction loss `L_rec`。
- 一个 EMA 追踪目标更新率，threshold gate 决定是否执行更新。
- 若更新，对当前 chunk 做 self-supervised reconstruction 更新，然后重新 forward 以产出 next-token prediction。

## 4. 数据集

实验使用 The Stack v2 code language modeling 设置，覆盖 Python 训练与 OOD code 语言。

## 5. 评估指标与主要结果

主要指标为 language modeling loss / perplexity、relative FLOPs、与 oracle recovery。论文报告：在 50% 更新预算下，Reconstruction Gating 显著优于 random skip；在 OOD 语言上 loss 下降可达约 16%，方法达到 82–89% 的 Oracle Recovery。

## 6. UPT Survey 定位

建议放入 Family I，或作为 In-Place TTT / TTT-layer 路线的辅助代表。其独特之处不在于新的 reward，而在于把 *when-to-adapt* 决策本身变成一个 prediction-statistic 的 gating signal。
