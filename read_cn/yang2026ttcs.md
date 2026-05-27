# TTCS: Test-Time Curriculum Synthesis for Self-Evolving

> **加入 Survey 时间：** 2026-04-16

**Paper:** TTCS: Test-Time Curriculum Synthesis for Self-Evolving  
**Authors:** Chengyi Yang et al.  
**arXiv:** 2601.22628  
**Venue:** arXiv 2026  
**Code:** https://github.com/XMUDeepLIT/TTCS

| 方法 | 载体 Carrier | 阶段 Regime | 层级 Level |
|------|----------------|-------------|------------|
| TTCS | Policy Opt. / curriculum synthesis | test-time | Semantic / synthetic target |

| When to Adapt | Full-Cohort Transductive Adaptation: test-time curriculum synthesis |
|---|---|
| 触发单位 Trigger Unit | test questions plus synthesizer-generated variants |
| 参数/状态持久性 Persistence | solver and synthesizer policies co-evolve across test-time training |
| 与推理关系 Inference Coupling | synthesize easier variants, update solver with self-consistency reward, then evaluate |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Evaluation Boundary |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Full-Cohort Transductive Adaptation |
| 可见数据范围 Visibility Scope | Full target cohort |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Full-Cohort Transductive Adaptation`；`Visibility Scope=Full target cohort`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Evaluation Boundary`。
- **更新何时触发：** 在 test questions 上构造 curriculum variants，并用 self-consistency rewards 迭代更新 synthesizer / solver。
- **服务当前样本还是后续样本：** 生成的 variants 和 solver updates 服务整个 test-time training run 与后续 evaluation。
- **参数/状态是否累积：** 两个 policies 在 co-evolution run 中持续累积更新。
- **reset 边界：** evaluation / benchmark boundary。

## 1. UPT 归属理由

TTCS 是 strict UPT，并且是本项目此前已在主文档引用但缺少 `read/TTCS.md` 的明确缺口。它只使用 test questions，合成 targeted variants，并使用 self-consistency rewards 更新 solver；不需要 ground-truth labels。

在 dominant-artifact 规则下，TTCS 更适合 **Family III: Self-Generated Target Bootstrapping**，因为 consensus / self-consistency 主要服务于生成 curriculum variants 与 pseudo-targets，而不只是直接作为最终 reward 本体。

## 2. 解决的问题

困难 test questions 往往太难，直接 majority pseudo-label 不可靠；同时 test set 太小会让 continuous online updates 不稳定。TTCS 通过合成能力边界附近的 question variants 来稳定 test-time self-evolution。

## 3. 方法介绍

TTCS 初始化两个同源 policies：

- **Question synthesizer：** 基于 test question 生成 progressively challenging / tractable variants，并通过 question quality reward 更新。
- **Reasoning solver：** 在原始 test questions 与 synthetic variants 上生成多次 responses，通过 self-consistency reward 得到 pseudo-label 并用 GRPO 更新。

Solver feedback 反过来指导 synthesizer 生成适合当前能力的题目，形成 co-evolving curriculum。

## 4. 数据集

论文在多个数学推理 benchmarks 上评估，包括 AIME24/25 等 challenging mathematical datasets，并测试 general-domain transfer。

## 5. 评估指标与主要结果

主要报告 mean@32、greedy decoding accuracy / pass@1 等指标。论文称 TTCS 在多个数学 benchmark 与 general-domain reasoning tasks 上持续提升不同 backbone 的 reasoning ability。

## 6. UPT Survey 定位

必须补入 `read/` 证据层。主 taxonomy 中若已引用 TTCS，则该笔记用于补齐证据；若后续更新主表，TTCS 应保持在 Family III。
