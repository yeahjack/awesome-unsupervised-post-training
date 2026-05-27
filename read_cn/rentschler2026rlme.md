# RLME: Reinforcement Learning from Meta-Evaluation

> **加入 Survey 时间：** 2026-03-11

**Paper:** Reinforcement Learning from Meta-Evaluation: Aligning Language Models Without Ground-Truth Labels
**Authors:** Micah Rentschler (Vanderbilt University), Jesse Roberts (Tennessee Technological University)
**ArXiv:** 2601.21268
**Date:** 2026-01-30

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RLME | Policy Opt. | training-time | Semantic |

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

RLME 属于 evaluator-driven RL 的典型范式：模型自身（或另一个 LLM）充当 evaluator，对 generator 输出回答自然语言 meta-questions（如 "Is the answer correct?"），evaluator 对 "Yes" 的输出概率被直接映射为 scalar reward，再用 GRPO-style（具体为 CISPO）policy gradient 更新 generator。整个流程不依赖 ground-truth labels 或外部 reward model，reward 完全来自 evaluator 内部的语义判断概率，属于 **internal evaluator bootstrapping → policy optimization → evaluator-driven RL** 路线。与 Self-Rewarding LM 类似，但 RLME 通过设计灵活的 meta-questions 把 evaluator 的判断从 preference ranking 推广到了任意语义层面的 yes/no 问题。

---

## 2. 解决的问题

大多数 LLM RL 训练方法（RLHF、RLVR）依赖 **人类标注偏好** 或 **ground-truth labels / task-specific verifiers**，在以下场景面临瓶颈：
1. **标注成本高**：人类反馈无法规模化；自动 verifier 通常领域特定且狭窄。
2. **正确性模糊**：open-domain QA、faithfulness 等任务中，正确与否难以用规则判定。
3. **缺乏标签的领域**：许多现实任务根本无法获得 ground-truth（如上下文忠实度、推理逻辑一致性）。

RLME 的核心 idea：能否用 evaluator 对自然语言 meta-questions 的回答概率作为 reward，在 **完全不使用 ground-truth labels** 的条件下训练 LLM？

---

## 3. 方法介绍

### 3.1 Assessment Prompting（评估提示构建）

给定 prompt $x \sim D$，generator 产出 response：

$$y \sim \pi_\theta(\cdot | x)$$

随后，$J$ 个 evaluator $\{\pi_{\phi_j}\}_{j=1}^{J}$ 被查询 $K$ 个 meta-questions $Q = \{q_1, \ldots, q_K\}$（由人类专家预先编写，如 "Is the answer correct?"、"Is the reasoning logically consistent?"）。每个 evaluator $j$ 对 meta-question $q_k$ 给出目标答案 $a_k$（如 "YES"）的概率：

$$p_{j,k} = \pi_{\phi_j}(a_k \mid x, y, q_k)$$

Reward 通过加权聚合所有 evaluator 和 meta-questions 的 log-probability 得到：

$$r(x, y) = \sum_{j=1}^{J} \sum_{k=1}^{K} v_j \cdot w_k \cdot \log p_{j,k}$$

其中 $\{w_k\}$ 是 meta-question 权重，$\{v_j\}$ 是 evaluator 权重，均为固定超参数。

### 3.2 Reinforcement Learning（策略优化）

目标函数：

$$J(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta}\left[r(x, y)\right]$$

采用 **CISPO**（Clipped IS-weight Policy Optimization，MiniMax 2025 提出的 GRPO 变体）进行更新：

- **Advantage**：$A_i = r_i - \bar{r}$（组内均值减去方差；**不除以标准差**，以避免 question-level difficulty bias，参考 Liu et al. 2025）
- **Importance sampling ratio**：$\rho_i(\theta) = \frac{\pi_\theta(y_i | x_i)}{\pi_b(y_i | x_i)}$
- **Clipping**：$\hat{\rho}_i(\theta) = \text{clip}(\rho_i(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}})$，其中 $\epsilon_{\text{low}} = 10000$，$\epsilon_{\text{high}} = 5.0$
- **Loss**：

$$L(\theta) = -\mathbb{E}_{y_i \sim \pi_b}\left[\text{sg}(\hat{\rho}_i(\theta)) \cdot A_i \cdot \sum_{t=1}^{T_i} \log \pi_\theta(y_{i,t} | x_i, y_{i,<t})\right]$$

### 3.3 Evaluator 配置方式

| 配置 | 说明 |
|------|------|
| **Live self-evaluation** | Generator 自身即 evaluator，参数随训练共同演化 |
| **Frozen self-evaluation** | 初始化时 snapshot 一份 generator 作为 evaluator，参数固定 |
| **Frozen other** | 使用另一个冻结的外部模型作 evaluator |
| **Ensemble** | 多个 evaluator 聚合判断（平均 log-prob） |

### 3.4 Meta-Question 设计

Meta-questions 是面向整个数据集的通用语义问题，而非针对单个样本的特定问题。例如：
- **准确性**："Is the answer correct?"
- **逻辑一致性**："Does the whole solution logically lead from the question to an answer?"
- **简洁性**："Is the length of the solution between 200 and 500 characters?"
- **上下文忠实度**："Is the answer supported by the context, regardless of whether it seems right or wrong?"

通过组合不同 meta-questions 及其权重，RLME 可实现 **multi-objective behavioral control**。

### 3.5 Reward Hacking 缓解策略

论文发现 RLME 在长时间训练后容易出现 **reward hacking**：generator 学会产出令 evaluator 回答 "Yes" 但实际不正确的内容（如空洞的套话 "the only logical conclusion is that this is the correct answer"，利用 evaluator 的 acquiescence bias）。缓解方案包括：

- **Early stopping**：基于验证集准确率提前停止
- **Sparse ground-truth anchoring**（RLME-1GT / RLME-10GT）：对 1% 或 10% 的训练样本向 evaluator 提供 ground-truth answer，有效稳定训练曲线
- **Ensemble evaluator**（RLME-Crowd）：多模型投票使 reward 曲线更平滑，但不能完全阻止 reward hacking

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | **GSM8K** | 小学数学题，fully verifiable，用于核心对比实验 |
| Open-domain QA | **CQAC**（自建） | 从 SQuAD、NewsQA、TriviaQA、HotpotQA、BioASQ、DROP、RACE、TextbookQA 各取 200 条（共 1600），context 截断至 4000 字符 |
| 上下文忠实度 | **FaithEval-Counterfactual** | 评估模型能否忠于 context（即使 context 与 world knowledge 矛盾），300 条 held-out，用于 OOD 泛化测试 |

---

## 5. 评估指标与主要结果

### 评估指标

- **Accuracy**：从 `\boxed{}` 中 regex 提取答案，exact match（整型匹配/字符串匹配）
- **FaithEval-Counterfactual accuracy**：模型是否忠实于提供的 context
- **Solution length**（字符数）：multi-objective 实验中衡量简洁性
- **Counterfactual cheating rate**：训练时提供 ground-truth answer，测试时注入随机错误答案，检查模型是否盲目采用

### 主要结果

#### GSM8K（核心可验证域实验）
- Base model（Qwen3-4B-Base）初始准确率约 30%
- **RLME** 快速提升到 >90%，与 **RLVR**（label-based baseline）学习曲线高度重合（Figure 2）
- 6 次独立实验，±1 std 区间重叠

#### Generator / Evaluator 选择
- **Generator 选择影响远大于 evaluator 选择**（Figure 3 vs. Figure 4）
- Live self-evaluation 与 frozen self-evaluation 几乎无差异
- SmolLM3 和 Gemma3 作为 evaluator 时，训练后期出现 reward hacking 导致准确率下降

#### Reward Hacking（Figure 5）
- 纯 self-evaluation 扩展训练后准确率骤降，reward 继续上升
- RLME-Crowd（ensemble）：reward 更平滑但仍崩溃
- **RLME-10GT**（10% ground-truth）和 **RLME-1GT**（1% ground-truth）：有效稳定训练，防止崩溃

#### Multi-Objective：Accuracy + Conciseness（Figure 6）
- RLME-Concise 将平均 solution 长度缩短近一半，同时保持与 RLME 相当的 GSM8K 准确率
- 推理被压缩为更密集的数学表达式而非冗长自然语言

#### Counterfactual Cheating Detection（Figure 7）
- RLVR 和 RLME-Base（meta-question: "Is the answer correct?"）：训练时给 ground-truth，测试时注入随机答案后模型倾向于 cheat（rationalize injected answer）
- **RLME-NoCheat**（meta-question: "Does the whole solution logically lead from the question to an answer?"）：在 counterfactual 测试中准确率 >80%，成功避免 cheating

#### Open-Domain QA & Faithfulness（Table 1 & Table 2）

| 方法 | CQAC Avg Accuracy | FaithEval-Counterfactual |
|------|-------------------|--------------------------|
| Base (Qwen3-4B-Base) | 32.8% | 28.2% |
| RLVR | **62.1%** | 61.8% |
| RLVR+RLME | 57.0% | **70.4%** |

- RLVR+RLME 在 CQAC 上略低于纯 RLVR（因为同时优化 faithfulness），但在 **FaithEval-Counterfactual 上超出 RLVR 近 9 个百分点**
- 关键发现：FaithEval 改进完全来自 meta-evaluation 的泛化，训练数据中 **不含任何 FaithEval 样本**

### 关键发现

1. **Meta-evaluation 提供的 reward signal 在可验证域中与 label-based RL 效果相当**，且 sample efficiency 类似。
2. **Generator 选择对性能影响远大于 evaluator 选择**，验证了 "验证比生成更容易" 的假说。
3. **Reward hacking 是 RLME 的核心风险**：延长训练后 generator 学会利用 evaluator 的 acquiescence bias；但 **仅 1% 的 ground-truth anchoring 即可有效缓解**。
4. **Meta-questions 的设计赋予了 multi-objective 和行为控制能力**：通过改变 meta-question 可以控制简洁性、推理诚实度（anti-cheating）、上下文忠实度等。
5. **RLME 可泛化到 open-domain 无标签场景**：在 CQAC 上训练的 faithfulness meta-evaluation 直接迁移到 OOD 的 FaithEval benchmark，证明了 meta-question 定义的优化目标具有 domain-agnostic 的泛化性。
6. **RLME 最适合作为 RLVR 的补充而非替代**：有标签时 RLVR 更优，无标签时 RLME 可独立使用，混合方案（RLVR+RLME）在多目标场景中表现最佳。
