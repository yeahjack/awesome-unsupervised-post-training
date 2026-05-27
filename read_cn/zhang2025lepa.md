# Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving

> **加入 Survey 时间：** 2026-03-11

> **论文元信息**：Published as a conference paper at ICLR 2025。作者来自清华大学、Moonshot AI 和华盛顿大学圣路易斯分校。

| 属性 | 值 |
|---|---|
| Method | LEPA (LEarning to Plan before Answering) |
| Carrier | Direct Opt. (SFT，兼容 RL) |
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
| 证据状态 Evidence Status | PDF-confirmed correct-answer / verifier caveat |
| Strict UPT Status | Not strict UPT; verifier-assisted self-training adjacent |
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

> **PDF 原文复核更新（2026-04-21）：建议从 strict UPT 主表移出，改为 verifier / correct-answer assisted self-training adjacent。**

LEPA 的核心 artifact 确实是模型自生成的 anticipatory plan；但 PDF 原文显示，数据生成与筛选过程依赖外部 correctness signal，因此不满足本 survey 当前的 strict UPT 边界。

- **binary correctness function:** PDF §2.1 把 LEPA 建模为给定问题集 $D_{\text{prompt}}$ 和 binary scoring function $f_{\text{cor}}(x_i,y_i)$，该函数判断 solution 是否正确并决定样本是否进入训练集。
- **correct-answer assisted reflection:** PDF §2.1 说明，若初始 solution 错误，LLM 会收到 problem、previous plan、incorrect solution 和 “correct answer (if accessible)” 来做 self-reflection；Appendix prompt 也直接包含 “The desired correct final answer is: [Correct Answer]”。
- **dataset/verifier use:** PDF 实验部分说明 Hendrycks MATH 使用 dataset creators 提供的 function 评估 solution correctness；LEPA+REINFORCE 变体也按 final-answer correctness 给 reward。
- **strict UPT 冲突:** 虽然 plan 本身由模型生成，训练样本的接受、反思和 reward 标注都依赖 external correct answer / verifier。它更接近 STaR/ReST 系列的 answer-verifier assisted self-training，而不是 no-external-ground-truth UPT。

**建议定位：** `verifier-assisted / correct-answer-assisted self-generated-target adjacent`。可作为 Family III 的重要相邻工作讨论"plan as synthetic target"，但不应在 strict UPT 主表中作为无外部监督代表。

---

## 2. 解决的问题

现有 self-training 方法（如 STaR、ReST、ReST EM）在数据生成阶段仅让 LLM 生成 step-by-step solution，训练目标也仅是最大化对这些 solution 的 log-likelihood。这种方式存在两个核心不足：

1. **缺乏可迁移的高层 meta-knowledge**：step-by-step solution 是 problem-specific 的，模型仅学到了"如何解这道题"，而未学到"解这类题的通用策略"，导致泛化能力有限，尤其在 Hendrycks MATH 等高难度 benchmark 上表现不佳。
2. **Self-reflection 易产生 false-positive**：STaR 等方法在纠错时直接参考 correct answer 修改 solution，模型容易"作弊"——仅修改最终答案而不修正推理过程，产生 rationale 错误但 final answer 正确的 false-positive solution。

LEPA 的核心问题即：**self-training 所生成的 synthetic data 应当包含什么信息？** 论文认为，除了 solution 之外，还应包含抽象的、可迁移的高层 plan（anticipatory plan），以帮助模型在面对新问题时拥有通用的解题策略。

---

## 3. 方法介绍（含图表描述）

### 3.1 总体框架

LEPA 是一个迭代式 self-training 算法，每轮迭代包含两个阶段：

- **Data Generation Phase**：生成高质量的 (problem, plan, solution) 三元组
- **Model Optimization Phase**：使用 SFT 在生成的数据上微调 LLM

#### Figure 1（论文第 2 页）
展示了一个 didactic example，对比了 baseline（ReST）和 LEPA 的效果。给定一道关于 meerkat 站岗的组合数学题：
- **(b) ReST** 直接生成 solution，推理过程出错（错误计算 45/2=22.5 后又莫名得出 9）。
- **(c) LEPA** 先生成一个 anticipatory plan（列出组合数学的通用步骤：识别元素数量→计算组合数→计算每个元素的组合数→用总数减去值班次数），再据此 plan 逐步求解，最终正确得出答案 36。

#### Figure 2（论文第 3 页）
对比了 baseline 和 LEPA 的数据生成流程：
- **(a) Baseline**：直接 prompt LLM 生成 step-by-step solution，缺少高层抽象。
- **(b) LEPA**：先生成 anticipatory plan，再基于 plan 生成 solution；若 solution 错误，通过 self-reflection 优化 plan 后重试。

### 3.2 Data Generation Phase

给定初始模型 $\theta_0$、问题集 $D_{\text{prompt}} = \{x_i\}_{i=0}^{N-1}$、以及 binary scoring function $f_{\text{cor}}(x_i, y_i)$，在第 $t$ 轮迭代中：

1. **Plan Generation**：prompt LLM 针对问题 $x_i$ 生成 anticipatory plan $p_i^{t,0}$。Prompt 中强调 plan 应为"适用于同类问题的通用 meta-knowledge"，不得包含 problem-specific 的具体计算。
2. **Solution Generation**：基于 plan 和 problem，LLM 生成 solution $y_i^{t,0}$。
3. **Correctness Check**：
   - 若 $f_{\text{cor}}(x_i, y_i^{t,0}) = 1$，则 $(x_i, p_i^{t,0}, y_i^{t,0})$ 加入训练集 $D_{\text{train}}^t$。
   - 否则进入 **Self-Reflection** 循环（最多 $l$ 次）：
     - LLM 接收 problem、prior plan、incorrect solution 和 correct answer（如可用），分析失败原因，生成新的 plan $p_i^{t,j}$。
     - Reflection prompt 同样强调新 plan 不得包含 correct answer 或 problem-specific 信息（**避免 information bypassing**）。
     - 基于新 plan 重新生成 solution，直到正确或达到最大试验次数。

#### Algorithm 1（论文第 5 页）
给出了 LEPA 的完整伪代码，清晰地展示了外循环（iteration $t$）→ 内循环（problem $i$）→ self-reflection 循环（trial $j$）的三层嵌套结构。

### 3.3 Model Optimization Phase

获得训练集 $D_{\text{train}}^t$ 后，LEPA 将数据格式化为 **two-round conversation**：
- **Round 1**：user 给出 problem 并要求 LLM 生成 plan → assistant 输出 plan $p_i^t$
- **Round 2**：user 要求 LLM 基于 plan 求解 → assistant 输出 solution $y_i^t$

训练目标为最小化 negative log-likelihood：

$$L_{\text{SFT}}(\theta_t, D_{\text{train}}^t) = -\mathbb{E}_{(x_i, p_i^t, y_i^t) \sim D_{\text{train}}^t} [\log p_{\theta_t}(p_i^t, y_i^t | x_i)]$$

虽然默认使用 SFT，但 LEPA 也兼容 DPO、PPO、REINFORCE 等 RL 算法。

### 3.4 Anticipatory Plan 的优势

论文从三个角度分析了 plan 的益处：

1. **Reducing Cognitive Workload**：plan 作为 blueprint 提纲挈领，避免 LLM 在细节中迷失方向。
2. **Learning Generalizable High-level Meta-knowledge**：plan 不涉及 problem-specific 信息，可迁移到结构相似的问题。
3. **Avoiding Information Bypassing**：plan 中禁止包含 correct answer，从而隔离了答案泄露通道，避免生成 false-positive solution。

#### Figure 4（论文第 8 页）
展示了一个关于根式化简的 case study：
- **初始 plan** 过于笼统（"分析数学对象类型，应用公式简化"），无法为模型提供足够指导，导致 solution 中符号错误。
- **Self-reflection** 后，模型发现问题在于"未注意符号正负"，生成了更具体的新 plan（"分别化简每个根式，注意符号；合并同类项；验证结果"）。
- **新 plan** 指导下，模型正确化简了 $\sqrt{15 - 6\sqrt{6}} + \sqrt{15 + 6\sqrt{6}} = 6$。

### 3.5 Prompt 设计

论文在 Appendix A（Figure 5）中给出了完整的 prompt 模板：
- **Plan generation prompt**：强调生成"适用于同类问题的通用知识"，不含 question-specific 信息，不超过 1024 tokens。
- **Solution generation prompt**：要求模型基于自己的 plan 逐步求解，每步说明 plan 如何影响推理。
- **Self-reflection prompt**：输入 problem、original plan、incorrect solution、correct answer，要求分析失败原因并设计新 plan。
- **New plan generation prompt**：基于 reflection 结果输出新 plan，同样禁止包含 correct answer。

---

## 4. 数据集

### 训练/评估 Benchmark

| Benchmark | 任务类型 | 描述 |
|---|---|---|
| **Hendrycks MATH** | 数学推理 | 高难度数学问题，使用官方提供的 correctness 函数评估 |
| **Hellaswag** | 句子补全推理 | 测试常识推理与句子理解 |
| **BoolQ** | 段落理解与推理 | Yes/No 形式的阅读理解问题 |
| **PIQA** | 物理常识推理 | 物理世界的 intuitive reasoning |
| **CSQA** | 常识问答 | 附加评估 benchmark |
| **MMLU** | 多任务语言理解 | 附加评估 benchmark |
| **MMLU-Pro (Math)** | OOD 泛化测试 | 在 Hendrycks MATH 上训练，在 MMLU-Pro Math 上测试 |

### 基础模型
- **Llama 3 8B Instruct**（主实验）
- **Llama 3.1 8B Instruct**（附加实验）

### 超参数
- 每轮数据生成最多 5 次 trial（LEPA: 1 次初始 + 最多 4 次 self-reflection；ReST/ReST EM: rejection sampling 5 次；STaR: 最多 4 次修改）
- 采样温度 0.5（数据生成）、0.0005（测试）
- 学习率 3e-7，每轮 SFT 训练 1 epoch

---

## 5. 评估指标与主要结果

### 5.1 主要结果（Table 1）

在四个 reasoning benchmark 上的 test accuracy（基础模型 Llama 3 8B Instruct）：

| 方法 | Hellaswag | Hendrycks MATH | BoolQ | PIQA | 平均 |
|---|---|---|---|---|---|
| CoT (zero-shot) | 60.8% | 19.5% | 77.3% | 67.0% | 56.1% |
| Plan+CoT (LEPA prompt, 不训练) | 56.1% | 22.1% | 80.8% | 75.7% | 58.7% |
| ReST | 86.3% | 28.2% | 84.5% | 81.4% | 70.1% |
| ReST EM | 86.4% | 27.2% | 86.3% | 83.5% | 70.8% |
| STaR | 85.7% | 25.9% | 85.8% | 84.2% | 70.4% |
| **LEPA** | **91.2% (+4.8%)** | **30.2% (+2.0%)** | **88.4% (+2.1%)** | **85.9% (+1.7%)** | **73.9% (+3.1%)** |

**关键发现**：
- LEPA 在所有四个 benchmark 上均超越所有 baseline，平均提升 3.1%。
- 仅使用 LEPA prompt（不训练）在 3/4 benchmark 上优于 zero-shot CoT，说明 plan 本身即有价值；但在 Hellaswag 上反而下降，表明初始 LLM 尚未校准好生成高质量 plan 的能力。
- STaR 在 MATH 上出现明显性能下降，因其 false-positive solution 问题在复杂推理任务上尤为严重。

### 5.2 学习曲线（Figure 3）

Figure 3 展示了四个 benchmark 上各算法随迭代轮次的 test accuracy 变化：
- 在 Hellaswag 上，LEPA 前 10 轮略落后（因 Plan+CoT prompt 初始效果不如 CoT），但随训练推进逐渐超越 baseline，说明 self-training 能够"唤醒"LLM 利用 plan 的能力。
- 在其他三个 benchmark 上，LEPA 从初始就取得更好的性能，并收敛于更高的准确率。

### 5.3 Ablation Studies

#### Anticipatory Plan 的必要性（Table 2）
| 变体 | MATH | BoolQ | PIQA |
|---|---|---|---|
| ReST EM (baseline) | 27.2% | 86.3% | 84.2% |
| Without Plan | 24.3% | 84.8% | 84.5% |
| Without Self-Reflection | 28.8% | 86.9% | 84.8% |
| **LEPA (完整)** | **30.2%** | **88.4%** | **85.9%** |

- 去掉 plan 后性能大幅下降（MATH 24.3% vs 30.2%），甚至低于 ReST EM baseline。
- 去掉 self-reflection（改用 rejection sampling）也导致性能下降，说明 self-reflection 通过语言反馈优化 plan 比简单拒绝采样更有效。

#### Inference Compute 的使用方式（Table 3）
在 Hendrycks MATH 上对比不同利用推理算力的方式：

| 方法 | Avg Tokens | Accuracy |
|---|---|---|
| STaR | 175.1 | 25.9% |
| ReST | 477.8 | 28.2% |
| **LEPA** | **826.4** | **30.2%** |
| Silence Tokens | 869.3 | 28.3% |
| Correction | 979.4 | 27.8% |
| Long Solution | 1409.7 | 25.4% |

LEPA 是唯一成功利用额外 inference compute 超越 baseline 的方法。Silence tokens 仅略等于 ReST，而 Correction 和 Long Solution 甚至低于 ReST，说明无意义的 token 膨胀无法有效利用算力。

#### 与 RL 结合
LEPA+REINFORCE 在 MATH 上达到 30.6%（原始 LEPA 30.2%），验证了 LEPA 与 RL 算法结合的可行性。

### 5.4 OOD 泛化（Table 4）
在 Hendrycks MATH 训练、MMLU-Pro Math 测试的 OOD 设置下，LEPA 达到 38.9%，显著优于 STaR (35.8%)、ReST EM (35.3%) 等 baseline，证明 anticipatory plan 确实帮助模型学到了可迁移的 meta-knowledge。

### 5.5 其他 LLM 和 Benchmark
- **Llama 3.1 8B Instruct** 在 MATH 上：LEPA 49.6% vs ReST EM 46.9%（Table 5）
- **CSQA / MMLU**（Table 6）：LEPA 均一致性地优于所有 baseline
- **Simple-Eval 重评估**（Table 7）：LEPA 33.7% vs ReST EM 31.4%，结论一致
