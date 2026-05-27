# CoNL: Conversation for Non-verifiable Learning

> **加入 Survey 时间：** 2026-03-11

**Paper:** Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation
**Authors:** Yuan Sui, Bryan Hooi (NUS)
**ArXiv:** 2601.21464
**Date:** 2026-01-29

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CoNL | Policy Opt. | training-time | Semantic |

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
**Family IV — Internal Evaluator Bootstrapping (evaluator-driven RL)**

CoNL 通过 multi-agent self-play 协议将 generation、evaluation 与 meta-evaluation 统一在同一个 policy 中。其核心机制是**对话驱动的 diagnostic reward**：多个共享同一 policy 的 agent 在四轮结构化对话中提出、评价、修改解答，critique 能否帮助他人改善解答被量化为 reward signal，用于 policy gradient 更新。这完全符合 "Internal Evaluator Bootstrapping → evaluator-driven RL" 的范式——模型自身产生的 evaluation verdict（pairwise ranking + critique）被 Bradley-Terry 模型聚合为 quality score，再转化为三类 reward 驱动 policy optimization，无需外部 judge 或 ground truth。

---

## 2. 解决的问题

现有 LLM 训练方法在 **non-verifiable tasks**（如创意写作、开放对话、伦理推理）上面临根本性挑战：

1. **缺乏 ground-truth labels**：SFT 和 RLVR 等方法依赖可验证信号，在无客观答案的领域失效。
2. **LLM-as-Judge 的 evaluation ceiling 问题**：单一 judge 模型的 bias（如 verbosity bias、position bias）直接限制训练上限。经验证据表明 self-rewarding 模型会出现 response 长度从 1k 跳到 2.5k 字符的现象，说明内部 judge 偏向更长而非更好的回答。
3. **缺乏 meta-evaluation**：现有方法假设 evaluation 能力会随 generation 训练自然改善，或依赖静态 external judge。没有机制去**评估 evaluator 本身**，系统陷入 echo chamber。
4. **Self-improvement 的 circular reasoning**：Self-Taught Evaluators、Meta-Rewarding LM 等方法用模型自身判断来生成和评价 synthetic data，可能强化而非纠正 bias。

CoNL 提出第三种范式——**peer-supervision**：通过 critique-driven improvement 来衡量 evaluation 质量，避免 self-training 的循环性和 verifiable RL 的领域限制。

---

## 3. 方法介绍

### 3.1 CoNL 协议：四轮对话

给定 query $q$，从同一 policy $\pi_\theta$ 实例化 $N$ 个 agent，每个 agent 被赋予不同 persona（如 Rigorous Formalist、Creative Pattern-Finder、Adversarial Skeptic 等7种角色，$N > 7$ 时循环使用），shared parameters 但 distinct conversational roles。

**Round 0 — Initial Proposals：** 每个 agent $i$ 独立生成 initial solution $s_i^{\text{init}}$。

**Round 1 — Initial Evaluation & Critique：** 观察所有 initial solutions 后，每个 agent 产出：
- **Blind pairwise ranking** $\mathcal{R}_i^{\text{init}}$：agent 之间看不到彼此的 ranking，确保独立判断。格式为 "Agent X > Agent Y"。
- **Critiques** $\{c_{i \to k}\}_{k \in \mathcal{T}_i}$：针对特定 agent 的详细文本批评，指出逻辑错误、遗漏情况等。

**Round 2 — Revision：** 每个 agent 收到所有针对自己的 critique，生成 revised solution $s_i^{\text{rev}}$：可采纳有效反馈修复错误，也可辩护抵抗无效批评。

**Round 3 — Final Verdict：** 观察所有 revised solutions 后，每个 agent 产出 final pairwise ranking $\mathcal{R}_i^{\text{final}}$。

**Memory Buffering：** 由于多轮多 agent 对话可能超出 32k context window，实现了 memory buffering module 压缩历史对话，保留关键决策、推理和约束信息。

### 3.2 分数聚合：Bradley-Terry 模型

使用 Bradley-Terry (BT) 模型将可能冲突的 pairwise comparisons 聚合为 latent quality scores：

$$P(\text{Agent } a \succ \text{Agent } b \mid V_a, V_b) = \frac{\exp(V_a)}{\exp(V_a) + \exp(V_b)}$$

在两个时间点计算：
- $V_k^{\text{init}}$：从 initial rankings 聚合，反映对话前评估
- $V_k^{\text{final}}$：从 final rankings 聚合，反映对话后评估

**核心洞察：** $\Delta V_k = V_k^{\text{final}} - V_k^{\text{init}}$ 度量 agent $k$ 的解答在 revision 后是否改善。若 agent $i$ critique 了 agent $k$ 且 $\Delta V_k > 0$，说明 critique 正确识别了实际问题。

### 3.3 三类 Reward

**Diagnostic Reward（$r_{\text{diag}}$）：** 核心创新——衡量 critique 是否帮助他人改善：

$$r_{\text{diag}}(i) = \sum_{k \in \mathcal{T}_i} \max(0, V_k^{\text{final}} - V_k^{\text{init}})$$

$\max(0, \cdot)$ 确保只有 diagnostic critiques（正确识别问题并促成改善）获得正 reward。

**Solution Quality（$r_{\text{sol}}$）：** 奖励被群体高评的解答：

$$r_{\text{sol}}(i) = V_i^{\text{final}}$$

**Majority Alignment（$r_{\text{meta}}$）：** 衡量 agent 的 pairwise 判断与多数意见的一致性：

$$r_{\text{meta}}(i) = \frac{1}{|\mathcal{R}_i^{\text{final}}|} \sum_{(a,b) \in \mathcal{R}_i^{\text{final}}} \mathbb{I}[\text{Pref}_i(a,b) = \text{Maj}(a,b)]$$

**Composite reward：**

$$r_{\text{total}}(i) = w_1 \cdot r_{\text{sol}}(i) + w_2 \cdot r_{\text{diag}}(i) + w_3 \cdot r_{\text{meta}}(i)$$

默认权重 $w_1 = 1.0$, $w_2 = 2.0$, $w_3 = 1.0$。Diagnostic reward 权重最高，强调 meta-evaluation 学习。

### 3.4 Token-Level Credit Assignment

不同 conversation segment 获得不同 reward（而非统一分配）：

| Round | 内容 | Reward | 原因 |
|-------|------|--------|------|
| 0 | Initial solution $s_i^{\text{init}}$ | $r_{\text{sol}}$ | 解答质量 |
| 1 | Initial ranking $\mathcal{R}_i^{\text{init}}$ | **0 (masked)** | 防止 gaming baseline $V^{\text{init}}$ |
| 1 | Critiques $\{c_{i \to k}\}$ | $r_{\text{diag}}$ | 诊断有效性 |
| 2 | Revised solution $s_i^{\text{rev}}$ | $r_{\text{sol}}$ | 解答质量 |
| 3 | Final ranking $\mathcal{R}_i^{\text{final}}$ | $r_{\text{meta}}$ | 多数对齐 |

**关键设计：** Initial ranking tokens 获得 zero reward，防止 agent 策略性地给 peers 打低分以人为拉低 $V^{\text{init}}$，从而虚增 $\Delta V$。

### 3.5 Policy 训练

使用 Tinker API 实现 importance sampling policy gradient。训练 policy $\pi_\theta$，采样来自 behavioral policy $\pi_{\theta_{\text{old}}}$：

$$\mathcal{L}_{\text{IS}}(\theta) = \mathbb{E}_{x \sim \pi_{\theta_{\text{old}}}} \left[ \frac{p_\theta(x)}{p_{\theta_{\text{old}}}(x)} A(x) \right]$$

使用 generalized advantage estimation（$\lambda = 0.95$），学习率 $3 \times 10^{-5}$。Ground-truth labels **从不**用于 reward computation。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学 | DeepMath-103K | 高难度数学题（Level 5-9），涵盖 Algebra、Calculus、Number Theory、Geometry、Probability、Discrete Math。随机采样 3,500 题（Level 6-10）。|
| 数学竞赛 | AIME 2024 | 30 道题，涵盖 number theory、algebra、geometry、combinatorics、probability。答案为 000-999 整数。 |
| 数学竞赛 | AIME 2025 | 30 道题，同 AIME 2024 格式。 |
| 科学 | GPQA Diamond | 198 道 PhD 级多选题（physics、chemistry、biology），4 选 1。 |
| 科学 | FrontierScience | 100 道（Olympic format）expert-level 科学推理题（physics、chemistry、biology）。 |
| 编程 | USACO Gold & Platinum | 84 道竞赛编程题（63 Gold + 21 Platinum），需通过所有 test cases。 |

---

## 5. 评估指标与主要结果

### 评估指标

- **Pass@1：** 选择 final consensus score $V^{\text{final}}$ 最高的 agent，检验其答案是否正确。
- **Pass@K：** 检查 top-K agents（按 $V^{\text{final}}$ 排序）中是否有正确解答。
- **Rank-$\rho$：** Spearman rank correlation，度量 $V^{\text{final}}$ ranking 与 ground-truth correctness（1/0 labels）的相关性。$\rho = 1$ 表示完美 evaluation，$\rho \approx 0$ 表示随机。

### 主要结果（Table 2 — Pass@1）

**Qwen3-8B：**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 60.0 | 60.0 | 67.0 | 70.5 | 43.0 | 10.2 |
| Self-Consistency | 61.8 | 61.2 | 68.2 | 75.4 | 43.1 | 10.3 |
| Self-Refine | 63.2 | 62.8 | 69.5 | 72.2 | 45.5 | 11.8 |
| Multi-Agent Debate | 64.5 | 64.2 | 70.8 | 71.0 | 44.2 | 11.0 |
| Self-Rewarding (Single-Turn)* | 68.5 | 69.0 | 73.5 | 77.5 | 50.5 | 15.5 |
| Self-Rewarding (Multi-Agent)* | 69.8 | 70.2 | 74.8 | 78.8 | 52.0 | 16.8 |
| **CoNL (Ours)*** | **76.5** | **73.5** | **79.2** | **87.1** | **55.7** | **19.5** |

**Qwen3-4B-Instruct：**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 50.0 | 54.0 | 45.0 | 78.5 | 18.0 | 6.0 |
| Self-Rewarding (Multi-Agent)* | 58.9 | 62.5 | 52.8 | 85.2 | 24.8 | 10.2 |
| **CoNL (Ours)** | **63.5** | **67.4** | **55.2** | **84.9** | **27.5** | **13.4** |

**Llama-3.1-8B：**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 13.0 | 7.0 | 23.0 | 45.0 | 1.9 | 3.0 |
| Self-Rewarding (Multi-Agent)* | 20.8 | 13.8 | 31.5 | 54.2 | 7.5 | 7.2 |
| **CoNL (Ours)*** | **23.5** | **16.2** | **34.0** | **57.5** | **10.2** | 7.0 |

**Llama-3.2-3B：**

| Method | AIME24 | AIME25 | GPQA | DeepMath | FrontierSci | USACO |
|--------|--------|--------|------|----------|-------------|-------|
| 0-Shot Inference | 11.5 | 6.5 | 22.0 | 42.0 | 1.8 | 2.8 |
| Self-Rewarding (Multi-Agent)* | 17.5 | 11.5 | 28.2 | 49.8 | 5.8 | 6.2 |
| **CoNL (Ours)*** | **19.8** | **14.0** | **30.5** | **53.5** | **8.2** | **8.0** |

### Ablation Study（Table 3 — Qwen3-8B）

**Reward 组件的影响：**

| Variant | DeepMath Pass@1 | DeepMath Rank-ρ | AIME25 Pass@1 | AIME25 Rank-ρ |
|---------|----------------|-----------------|---------------|---------------|
| CoNL (Full) | **87.1** | **0.78** | **73.5** | **0.65** |
| w/o Diagnostic ($w_2=0$) | 83.5 | 0.55 | 68.2 | 0.44 |
| w/o Consensus ($w_3=0$) | 85.2 | 0.68 | 70.5 | 0.56 |
| w/o Solution Quality ($w_1=0$) | 84.8 | 0.70 | 69.8 | 0.58 |
| w/o Blind Ranking | 82.5 | 0.45 | 67.5 | 0.39 |

**Agent 数量的影响：**

| N agents | DeepMath Pass@1 | DeepMath Rank-ρ | AIME25 Pass@1 | AIME25 Rank-ρ |
|----------|----------------|-----------------|---------------|---------------|
| N=2 | 84.5 | 0.62 | 69.5 | 0.51 |
| N=3 | 85.9 | 0.71 | 71.8 | 0.60 |
| N=4 | **87.1** | **0.78** | **73.5** | 0.65 |
| N=5 | 87.0 | 0.76 | 73.2 | **0.66** |
| N=8 | 87.4 | 0.75 | 72.9 | 0.63 |

### Critique 质量分析（Table 4）

| Benchmark | 初始状态 | 结果类型 | 比率 |
|-----------|---------|---------|------|
| DeepMath | Incorrect (×) | Correction (× → ✓) | **82.4%** |
| DeepMath | Correct (✓) | Harm (✓ → ×) | **3.1%** |
| AIME 2025 | Incorrect (×) | Correction (× → ✓) | **41.2%** |
| AIME 2025 | Correct (✓) | Harm (✓ → ×) | **9.4%** |

### 关键发现

1. **CoNL 显著优于 self-rewarding baselines：** 在 Qwen3-8B 上，CoNL 比 SRT-M 高 6.7 pp（AIME24：76.5 vs 69.8）和 8.3 pp（DeepMath：87.1 vs 78.8），同时训练方差更低。
2. **Diagnostic reward 是最关键组件：** 移除 $r_{\text{diag}}$ 导致 DeepMath Pass@1 下降 3.6 pp（87.1 → 83.5），Rank-$\rho$ 从 0.78 暴跌至 0.55，证实其为 meta-evaluation 学习的核心机制。
3. **Blind ranking 对 evaluation 质量至关重要：** 移除 blind ranking 使 DeepMath Rank-$\rho$ 从 0.78 骤降至 0.45（42% 相对下降），因为 agent 可以 gaming baseline rankings。
4. **训练稳定性优越：** CoNL 的 entropy、solution length、accuracy 三项指标在训练过程中均保持稳定，closely matching ground-truth RL；而 SRT 的 majority-voting signal 导致不稳定收敛。
5. **Critique 具有高安全性：** DeepMath 上仅 3.1% 的正确解答被误导性 critique 损害（Harm rate），模型学会了保守策略——只在 critique 有 logical grounding 时才修改答案。
6. **N=4 agents 为最优平衡点：** 性能从 N=2 到 N=4 显著提升，N=5 以上收益递减甚至略降（N=8 在 AIME25 为 72.9 vs N=4 的 73.5），过多 agent 引入 coordination overhead。
7. **跨模型泛化：** CoNL 在 Qwen3-8B、Qwen3-4B、Llama-3.1-8B、Llama-3.2-3B 四个模型上均取得最佳或接近最佳表现，验证方法的通用性。
8. **Adversarial revision 机制有效防止 false critique reward：** 当 critique 错误时，被 critique 的 agent 可以辩护，其分数保持不变（$V^{\text{final}} \approx V^{\text{init}}$），critiquing agent 获得 zero reward。
