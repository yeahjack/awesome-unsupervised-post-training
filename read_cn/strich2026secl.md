# SECL: Self-Calibrating Language Models via Test-Time Discriminative Distillation

> **加入 Survey 时间：** 2026-04-16

**Paper:** Self-Calibrating Language Models via Test-Time Discriminative Distillation  
**Authors:** Jan Strich et al.  
**arXiv:** 2604.09624  
**Venue:** arXiv 2026  
**Code:** anonymous 4open.science link in arXiv abstract

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| SECL | Direct Opt. / LoRA | test-time | Sequence / calibration |

| When to Adapt | Streaming Continual Adaptation (calibration) with shift-triggered LoRA bursts |
|---|---|
| 触发单位 Trigger Unit | input question stream; update bursts only when entropy-based shift detector fires |
| 参数/状态持久性 Persistence | LoRA confidence updates accumulate across the shifted stream; per-question reset hurts |
| 与推理关系 Inference Coupling | infer normally until shift; then distill internal discriminative confidence into the model |
| 输入可见性 Input Visibility | Online |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Domain / stream boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Streaming Continual Adaptation |
| 可见数据范围 Visibility Scope | Streaming prefix only |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Streaming Continual Adaptation`；`Visibility Scope=Streaming prefix only`。
- **两轴编码：** `Input Visibility=Online`；`Update Persistence=Cumulative`；`Reset Boundary=Domain / stream boundary`。
- **更新何时触发：** 论文使用 entropy-based change detector 监控输入流；只有检测到 distribution shift 时才进入 calibration burst，而不是对每个问题都更新。
- **服务当前样本还是后续样本：** 当前 burst 的更新服务后续同域输入；论文 ablation 显示 per-question reset 会显著恶化 ECE / AUROC，说明累积更新是方法设计的一部分。
- **参数/状态是否累积：** LoRA 更新在流内累积，训练只发生在 6--26% 的 question stream 上。
- **reset 边界：** 合理边界是 domain / stream boundary；不是 sample boundary。

## 1. UPT 归属理由

SECL 是 strict UPT 候选。它满足三点：

- **真实 update：** test time 使用 LoRA 更新 confidence / calibration 表征。
- **无外部标签：** arXiv 摘要明确称其 test-time training pipeline 使用 label-free self-supervision，不需要 labeled data 或 human supervision。
- **内部信号来源：** 监督信号来自模型被问 “Is this answer correct?” 时对 `True` token 的概率，即 `P(True)` / `NormPTrue`，再把这个内部 discriminative confidence 蒸馏回模型的 verbalized confidence。

更细的 family 归属建议为 **Family IV: internal evaluator bootstrapping**，因为核心信号不是 raw NLL / entropy，而是模型内部的 correctness evaluator channel。它也和 Family I 有交叉，因为优化对象是 calibration / confidence 表征，不生成外部 pseudo-label。

## 2. 解决的问题

LLM 常常 verbalized confidence 过高，即明明容易答错仍给出高置信度。传统 calibration 方法通常需要 labeled validation set，或在 distribution shift 下不稳，或推理成本高。SECL 的问题设定是：在 test-time 输入流发生 shift 时，如何用模型内部信号做 label-free calibration update。

## 3. 方法介绍

SECL 包含三个关键组件：

- **Shift detector：** 基于 entropy 的 change detector 判断输入分布是否变化。
- **Internal discriminative signal：** 让模型判断自己答案是否正确，并读取 `P(True)`，再进行 normalization。
- **LoRA calibration burst：** 当 verbalized confidence 与 `NormPTrue` 不一致时，用 directional loss 做轻量 LoRA 更新，降低 verbalized confidence 与内部 discriminative confidence 的差距。

论文强调 burst 级更新比单题更新稳定；单题更新会破坏 confidence representation。

## 4. 数据集

实验覆盖四类 domain / dataset，包括 GSM8K 等数学问答，以及其它不同领域的 question streams。模型覆盖三个 model families、四个 small language models。

## 5. 评估指标与主要结果

主要指标是 Expected Calibration Error (ECE)、Brier score、AUROC 与 task accuracy。摘要报告 SECL 将 ECE 降低 56--78%，并在成本上低于其蒸馏来源 baseline；正文还报告在 Llama 3.2-3B 上可达到约 71% 的 ECE reduction。

## 6. UPT Survey 定位

建议纳入本 survey 的 strict UPT 表或至少作为 Family IV 的 calibration-oriented test-time adaptation 代表。它补上了当前 taxonomy 中较少覆盖的方向：不是提升 answer accuracy，而是用内部 self-evaluation signal 做 calibration post-training。
