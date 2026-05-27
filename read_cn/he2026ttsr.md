# TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement

> **加入 Survey 时间：** 2026-03-11

**Paper:** TTSR: Test-Time Self-Reflection for Continual Reasoning Improvement
**Authors:** Haoyang He, Zihua Rong, Liangjie Zhao, Yunjia Zhao, Lan Yang, Honggang Zhang (Beijing University of Posts and Telecommunications; Institute of Computing Technology, CAS; Southwestern University of Finance and Economics)
**ArXiv:** 2603.03297
**Date:** 2026-02-06

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| TTSR | Policy Opt. | test-time | Semantic |

| When to Adapt | Full-Cohort Transductive Adaptation before target inference |
|---|---|
| 触发单位 Trigger Unit | original test set plus synthesized variants |
| 参数/状态持久性 Persistence | full parameter accumulate across curriculum cycles |
| 与推理关系 Inference Coupling | adapt on the evolving cohort, then re-infer in later cycles |
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

- **更新何时触发：** 更新在原始 test set 与其派生出来的 synthetic variants 上按 curriculum cycle 触发。
- **服务当前样本还是后续样本：** 当前 cycle 的更新主要服务后续 cycles 与后续评估，不是对单个样本 adapt 完就 reset。
- **参数/状态是否累积：** 参数在多个 curriculum cycles 中持续累积；cohort 会随反思、重分配或重采样而演化。
- **reset 边界：** 因此它属于 self-curriculum test-time adaptation，而不是简单的 static test-cohort RL。

## 1. UPT 归属理由

**Family III — Self-Generated Target Bootstrapping (reasoning / curriculum synthesis)**

TTSR 属于无监督后训练（UPT），因为它在 test time 完全不依赖任何外部 ground-truth 标签或更强的 teacher 模型的监督。整个框架仅使用**一个预训练模型**，通过在 Student 和 Teacher 两个功能角色之间交替运行来实现自主进化。

具体而言，TTSR 生成的内部 artifact 包括：
1. **Pseudo-target（伪目标）**：Student 对测试题采样多条推理轨迹后，通过 majority voting 得到 consensus outcome $\hat{y}(x)$，作为 pseudo-correct reference，为 GRPO 优化提供 reward signal。
2. **Reflection summary（反思摘要）**：Teacher 观察 Student 的失败推理轨迹，分析 process-level deficiencies，总结 recurring reasoning weaknesses，产出结构化的 JSON 描述（包含 reasoning_weakness、trigger_conditions、failure_signature）。
3. **Targeted variant questions（定向变体问题）**：Teacher 基于反思摘要合成新的训练问题，这些问题保留原题的核心推理结构，但刻意暴露 Student 的已知弱点，构成 adaptive curriculum。

这些 artifact 完全由模型自身生成，无需人类标注或外部模型指导。变体问题作为 synthetic training target 被注入 Student 的训练集，驱动 test-time 参数更新，符合 "Direct optimization → reasoning / curriculum synthesis" 的子类别定义。

---

## 2. 解决的问题

TTSR 针对 Test-Time Training (TTT) 的两个核心挑战：

1. **困难测试题导致 pseudo-label 不可靠**：在 challenging reasoning tasks 中，测试题通常处于模型能力的边界附近。模型自生成的 pseudo-labels 或 reward signals 噪声大、不稳定，导致低效甚至退化的参数更新。现有方法（如 TTRL）直接在原始困难题上做 RL，当模型 pass rate 极低时，reward signal 几乎无信息量。

2. **缺乏针对模型特定推理弱点的适配机制**：现有 test-time 方法和 self-play 方法主要关注 task-level diversity 或 difficulty scaling，忽视了模型在具体实例上暴露的 fine-grained capability deficiencies。所有推理错误被当作无差别的噪声处理，无法高效纠正具体的推理短板。

3. **对外部强模型的依赖**：部分 TTT 方法依赖更强的 teacher model 来生成数据或指导，削弱了 fully autonomous self-evolution 的目标，限制了在强 teacher 不可用场景中的适用性。

TTSR 提出：通过 Teacher 角色对失败推理轨迹进行 **trace-level 反思**，诊断具体弱点并合成 **难度适中的 targeted variant questions**（位于模型 capability frontier 附近），将学习信号从不可靠的困难题转移到 learnable regime，实现稳定的 test-time 自进化。

---

## 3. 方法介绍

TTSR 是一个基于 GRPO 和 TTT 的 **teacher-reflective test-time self-evolving training** 框架，使用单一预训练模型 $\pi_\theta$ 在 test time 交替扮演 Student 和 Teacher 两个角色（共享参数，不同功能行为）。

### 3.1 Student：问题求解与 Test-Time 适应

Student 作为 online solver，在 test time 迭代更新策略。

**训练集构建**：在第 $t$ 次 test-time iteration，训练集为：
$$D_t = X_{\text{test}} \cup X_{\text{var}}^{(t-1)}$$
其中 $X_{\text{test}}$ 为原始测试集，$X_{\text{var}}^{(t-1)}$ 为上一轮 Teacher 合成的 variant questions。

**Majority Voting Reward**：对每个训练题 $x \in D_t$，Student 采样 $G$ 条推理轨迹：
$$\{y_i\}_{i=1}^G \sim \pi_{\theta_t}(\cdot|x)$$
通过 majority voting 得到 consensus outcome $\hat{y}(x)$，并赋予 pseudo-correctness reward：
$$R_S(y_i|x) = \mathbb{I}[y_i = \hat{y}(x)]$$

定义 empirical pseudo-correctness score：
$$s_t(x) = \frac{1}{G}\sum_{i=1}^G R_S(y_i|x)$$
衡量 Student 在 stochastic sampling 下预测的稳定性。Student 使用 GRPO 基于 pseudo-correctness rewards 进行优化。

### 3.2 Teacher：Reflection-Guided Curriculum Synthesis

Teacher 不直接解题，而是观察 Student 的推理失败并合成 adaptive curriculum。

**Reflection on Reasoning Steps**：在第 $t$ 次 iteration，Teacher 收集上一轮 pseudo-correctness reward 为零的推理轨迹，随机采样至多 $M$ 个 failed instances。每个实例为 tuple $(x, y, \hat{y})$（题目、推理轨迹、pseudo-correct reference）。Teacher 分析 $y$ 中的推理步骤，产出 reflection summary，刻画 incorrect reasoning patterns 和 missing/insufficient steps。反思聚焦于 **process-level deficiencies** 而非 final answer correctness。

**Reflection-Guided Question Synthesis**：基于 failed set $F^{(t-1)} = \{(x_k, y_k, \hat{y}_k)\}_{k=1}^m$，构造 natural-language synthesis prompt $p_t$，合成 variant questions：
$$X_{\text{var}}^{(t)} = \{x'_j\}_{j=1}^M \sim \pi_{\theta_t}(\cdot \mid p_t, F^{(t-1)})$$
合成的问题保留原题核心推理结构，选择性修改条件或约束以暴露被反思到的推理弱点。使用 group sampling 鼓励多样性。

### 3.3 Difficulty Reward

使用 entropy-based capability-frontier reward 评估 variant question 的难度：
$$R_{\text{diff}}(x') = \frac{H(\text{Bern}(s_t(x')))}{log\,2} = -\frac{s_t(x') \log s_t(x') + (1-s_t(x')) \log(1-s_t(x'))}{\log 2}$$
当 $s_t(x') \approx 0.5$ 时 reward 最大，即 Student 在该问题上表现出最大不确定性，处于 capability frontier。

### 3.4 Similarity Penalty Reward

为鼓励探索并减少冗余，引入 group-level similarity penalty。设 $X' = \{x'_1, \ldots, x'_M\}$ 为当前合成的 variant questions，$x$ 为原始测试题，扩展集 $Z = X' \cup \{x\}$，定义：
$$R_{\text{sim}}(x'_i, x) = \frac{1}{|Z|-1}\sum_{z \in Z \setminus \{x'_i\}} \max(0, \text{sim}(x'_i, z) - \tau)$$
其中 $\tau$ 为 textual overlap 的容忍阈值。similarity 基于 sequence-based matching（contiguous overlapping spans），归一化公式为 $\text{sim}(S_1, S_2) = 2M/T$。

### 3.5 Teacher Reward

Teacher 的综合 reward 为：
$$R_T(x'_i) = \max(0, R_{\text{diff}}(x'_i) - \lambda R_{\text{sim}}(X', x))$$
平衡 difficulty（靠近 capability frontier）和 diversity（避免冗余）。此外，强制 format constraint：只有正确包裹在 `<question>` 标签内的问题才参与 reward 计算。

### 3.6 Continual Self-Evolving Loop

Student 和 Teacher 通过迭代交替构成 continual self-evolving loop：每一轮 Teacher 反思上一轮 Student 的失败 → 合成 targeted variants → Student 在原始测试题 + variants 上训练 → 更新策略 → 下一轮再次反思。这种 co-evolution 保证合成的 curriculum 始终与 Student 的 evolving capability 对齐。

### 3.7 Teacher Prompt 设计

Teacher 使用两阶段 prompt：
1. **Weakness extraction prompt**：输入 original question + failed reasoning trace，输出结构化 JSON（reasoning_weakness、trigger_conditions、failure_signature、localization_summary）。
2. **Question synthesis prompt**：输入 original question + failed trace + weakness JSON，按 5 步流程（Anchor Structure → Error-Hitting Strategy → Generate Question → Hit Rationale → Self-Test Filter）生成 targeted variant question。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| Mathematical Reasoning | AMC23 | 竞赛风格数学题，要求精确推理与 case analysis |
| Mathematical Reasoning | MATH-500 | 涵盖代数、几何、数论、组合的多样数学题 |
| Mathematical Reasoning | Minerva Olympiad | 高级奥赛题，强调长推理链和非平凡数学洞见 |
| Mathematical Reasoning | OlympiadBench | 奥林匹克级别双语多模态科学问题 |
| Mathematical Reasoning | AIME 2024 | 高难度数学竞赛，单步推理错误可导致全题失败 |
| Mathematical Reasoning | AIME 2025 | 同上，2025年版本 |
| General Reasoning | GPQA-Diamond | 研究生级别科学问答，需要深度推理 |
| General Reasoning | MMLU-Pro | 广泛多领域推理评测 |

---

## 5. 评估指标与主要结果

### 评估指标

- **Mean@32**：用于 AIME 2024/2025，多条采样推理轨迹下的平均准确率
- **Greedy Decoding (Pass@1)**：用于其他 benchmarks，贪心解码准确率

### 主要结果

**Table 1: 跨模型、跨 benchmark 综合评测**

| Method | AMC23 | MATH500 | Minerva | Olympiad | AIME24 | AIME25 | GPQA-D | MMLU-Pro | Δ (avg) |
|--------|-------|---------|---------|----------|--------|--------|--------|----------|---------|
| **Qwen3-4B-Base** | | | | | | | | | |
| Base | 45.3 | 72.1 | 32.4 | 40.2 | 12.4 | 5.8 | 25.7 | 52.0 | – |
| R-Zero | 54.1 | 76.8 | 40.7 | 44.0 | 18.2 | 9.1 | 28.4 | 55.0 | +5.1 |
| TTRL | 55.8 | 79.1 | 43.6 | 46.0 | 17.6 | 9.7 | 29.1 | 56.0 | +6.4 |
| **TTSR** | **61.0** | **82.4** | **53.0** | **45.3** | **25.6** | **20.1** | **34.2** | **60.8** | **+12.1** |
| **Qwen3-8B-Base** | | | | | | | | | |
| Base | 51.4 | 77.9 | 39.6 | 41.2 | 15.9 | 9.8 | 33.1 | 58.6 | – |
| R-Zero | 58.7 | 82.1 | 47.8 | 48.6 | 22.6 | 13.4 | 36.7 | 61.5 | +5.5 |
| TTRL | 61.9 | 84.3 | 50.2 | 50.8 | 26.1 | 15.7 | 38.4 | 62.8 | +7.8 |
| **TTSR** | **66.4** | **87.5** | **54.9** | **55.2** | **30.8** | **19.1** | **42.6** | **66.7** | **+12.0** |
| **OctoThinker-8B-Hybrid-Base** | | | | | | | | | |
| Base | 28.4 | 45.6 | 15.2 | 16.8 | 7.1 | 3.9 | 15.2 | 26.8 | – |
| R-Zero | 35.7 | 54.1 | 22.9 | 25.4 | 11.6 | 6.8 | 19.7 | 31.4 | +6.1 |
| TTRL | 38.9 | 57.3 | 26.4 | 28.7 | 13.9 | 8.2 | 21.5 | 33.2 | +8.6 |
| **TTSR** | **46.8** | **64.9** | **34.1** | **36.8** | **19.7** | **12.4** | **27.9** | **39.8** | **+15.4** |

**Ablation Study (Qwen3-8B)**

| Settings | MATH500 | AIME25 | Olympiad | GPQA-D |
|----------|---------|--------|----------|--------|
| TTSR (Full) | 87.5 | 19.1 | 55.2 | 42.6 |
| w/o Reflection-Guided Synthesis | 85.0 (↓2.5) | 14.2 (↓4.9) | 49.7 (↓5.8) | 38.3 (↓4.3) |
| w/o Teacher Test-Time Update | 82.7 (↓4.8) | 13.3 (↓5.8) | 49.4 (↓6.1) | 37.9 (↓4.7) |
| w/o Similarity Penalty | 85.9 (↓1.6) | 16.3 (↓2.8) | 52.9 (↓2.6) | 40.1 (↓2.5) |

### 关键发现

1. **TTSR 在所有模型和 benchmark 上一致超越 baselines**：平均提升 +12.0 到 +15.4 个百分点，显著优于 TTRL（+6.4 ~ +8.6）和 R-Zero（+5.1 ~ +6.1）。

2. **在高难度任务上提升尤为显著**：AIME 2024/2025 等需要 deep multi-step reasoning 的任务上，TTSR 的改进超过 10 个百分点（如 Qwen3-4B 在 AIME25 从 5.8→20.1），表明 reflection-guided synthesis 对难题推理特别有效。

3. **跨域泛化能力强**：在 AIME25 上 train 后迁移到 GPQA-D，TTSR 提升 +7.2（从 33.1 到 40.3），而 TTRL 仅 +1.3（到 34.4）；反向从 GPQA-D train 迁移到 AIME25 也有 +4.3 的提升，说明 TTSR 的 test-time updates 捕获了 reusable reasoning refinements。

4. **Ablation 证实各组件贡献**：去掉 Reflection-Guided Synthesis 导致 Olympiad 下降 5.8 个点；去掉 Teacher Test-Time Update（即 Teacher-Student co-evolution）造成最大整体下降（AIME25 下降 5.8，Olympiad 下降 6.1）；去掉 Similarity Penalty 导致较小但一致的下降。

5. **在具有 reasoning-oriented inductive bias 的模型上增益最大**：OctoThinker-8B-Hybrid-Base 上 TTSR 平均提升 +15.4，为三个模型中最高，表明 structured reflection 与推理导向的模型架构具有协同效应。

6. **训练配置简洁高效**：Batch Size=16, Student Rollout G=8, Teacher Variants M=4, Iterations T=20, KL Coef=0.001, LR=3e-7, MaxLen=4096，所有 benchmark 和模型使用统一配置。
