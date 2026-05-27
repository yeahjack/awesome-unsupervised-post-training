# Self-Training Elicits Concise Reasoning in Large Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** Self-Training Elicits Concise Reasoning in Large Language Models
**arXiv:** 2502.20122v3
**Authors:** Tergel Munkhbat*, Namgyu Ho*, Seo Hyun Kim*, Yongjin Yang, Yujin Kim, Se-Young Yun
**Affiliations:** KAIST AI
**Code:** https://github.com/TergelMunkhbat/concise-reasoning

---

| 属性 | 值 |
|---|---|
| Method | Concise ST |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Traj. |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | self-generated data batch / iteration round |
| 参数/状态持久性 Persistence | full parameter accumulate across synthesis / refinement rounds |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | PDF-confirmed external correctness / exemplar caveat |
| Strict UPT Status | Not strict UPT; GT-filtered / external-exemplar adjacent |
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

> **PDF 原文复核更新（2026-04-21）：建议从 strict UPT 主表移出，改为 GT-filtered / external-exemplar self-training adjacent。**

原先将 Concise ST 暂归入 Family III，是基于“训练 target 由模型自身生成”的表层机制；但 PDF 原文显示，完整方法并不满足当前 survey 的 no-external-ground-truth strict UPT 边界。

- **GT / verifier dependency:** PDF §3.1 明确说明，Naive BoN 会为每个 training question 生成 $N$ 条 reasoning paths，并选择 “shortest correct reasoning path”；脚注还说明没有任何 correct reasoning paths 的问题会被排除。这个 “correct” 判断需要 benchmark answer / parser / correctness check，不是纯 internal signal。
- **外部 exemplar dependency:** PDF §3.2 明确列出 three sources for few-shot exemplars，包括 human-annotation、proprietary frontier LLMs (GPT-4o) 和 self-generated samples；主实验采用 `FS-GPT4o-BoN` 作为最终组合方法。
- **外部对照不是唯一问题:** 即使不考虑 FS-Human / FS-GPT4o，BoN 数据筛选本身也依赖“正确路径”的外部判定；因此它不是 strict UPT，而是 **GT-filtered self-training / concise-rationale distillation**。
- **仍可作为 adjacent:** 这篇工作对“模型自身输出分布中存在 latent concise reasoning ability”的分析很有价值，可放在 related work 中讨论 self-generated rationale 的选择问题，但不应作为 no-external-ground-truth strict UPT 代表。

**建议定位：** `semi-supervised / GT-filtered adjacent`；若后续正文保留，应明确写成“self-generated trajectories + external correctness filtering”，而不是 strict UPT `Self-Generated Target Bootstrapping`。

---

## 2. 解决的问题

Chain-of-thought (CoT) reasoning 虽然显著提升了 LLM 的复杂任务推理能力，但当前模型生成的推理链往往包含大量冗余 token——冗长的解释、重复的措辞和不必要的上下文复述。这些冗余直接导致推理延迟增大和计算成本升高。

作者指出现有解决方案的局限性：

1. **Zero-shot prompting 不可靠：** 如 "Be Concise"、"Fixed Budget" 等 prompting 策略在不同模型间表现极不一致（Table 1），尤其对 math-specialized 模型几乎无效。多数 prompting 方法在缩短长度的同时会造成显著 accuracy 下降（如 Fixed Budget 平均缩短 32.2% 但 accuracy 下降 10.1%）。
2. **外部数据 fine-tuning 牺牲准确率：** 用 human 或 GPT-4o 生成的简洁推理做 fine-tuning 虽能大幅缩短长度，但因分布偏移会导致严重的 accuracy 退化。
3. **已有 self-training baseline（Rational Metareasoning）效果有限：** 仅实现约 12% 的平均长度缩减。

核心观察：通过分析模型输出分布（Figure 1 & Figure 2），作者发现 LLM 已经具备 **latent concise reasoning ability**——输出分布中存在大量比平均长度更短但仍然正确的推理路径。问题在于如何有效 elicit 这种潜在能力。

---

## 3. 方法介绍

### 3.1 核心思路

利用 LLM 输出分布中已存在的 concise reasoning path，通过 self-training（自生成数据 + SFT）将简洁推理风格内化到模型中，避免 inference-time 的额外开销。

### 3.2 Naive Best-of-N Sampling (BoN)

对训练集中每个问题生成 $N$ 条 reasoning path（$N=16$），选取其中最短的正确路径作为 fine-tuning 样本。

- **Question-wise selection：** 逐题选取最短路径（而非全局 pool 选最短），确保困难问题也有训练样本覆盖。
- **局限性：** BoN 的长度缩减与 $N$ 呈 log-linear 关系（Figure 3），进一步缩短需要指数级的生成成本，sample efficiency 较低。

### 3.3 Few-Shot Conditioned Sampling (FS)

利用 few-shot prompting 对采样分布进行偏移，使模型天然倾向于生成更短的推理路径。考虑三种 few-shot exemplar 来源：

| 方法 | Exemplar 来源 | 说明 |
|------|-------------|------|
| **FS-Human** | Wei et al. (2022b) 人工标注 | 现成可用，推理极为精简 |
| **FS-GPT4o** | GPT-4o 生成 | 使用 Hand Crafted 3 prompt 生成，accuracy 保持好 |
| **FS-Self** | 模型自生成 | 两阶段流程：对 128 题各采样 128 条，按长度排序 → 筛选正确 → GPT-4o 验证质量，选 8 条 |

关键发现（Figure 3）：8-shot conditioning 单次采样即可实现超过 BoN $N=256$ 的长度缩减效果。

### 3.4 Few-Shot Conditioned BoN Sampling (FS-BoN)

将 few-shot conditioning 和 BoN sampling 联合使用：在 few-shot conditioned 分布上再做 BoN 采样。两者的改进基本独立且可叠加（Figure 3），实现最大程度的长度缩减。实验采用 FS-GPT4o-BoN 作为最终方法。

### 3.5 Sample Augmentation

Few-shot prompting 的适应性有限：可能（1）抑制复杂问题的正确路径生成，（2）对极简单问题引入多余步骤。为此，对每个问题，将 FS / FS-BoN 生成的样本与 naive BoN 生成的 $N$ 条样本合并，从合并集中选取最短正确路径。

这一 augmentation 策略在保持长度缩减的同时提升 accuracy（Figure 4），因为扩大了对困难问题的覆盖。

### 3.6 训练细节

- 使用 HuggingFace Trainer 做 SFT，vLLM 做生成。
- Fine-tuning：batch size 16，1 epoch，learning rate 1e-5，bfloat16 precision。
- 生成：temperature $T=0.7$，GSM8K 最大 512 tokens，MATH 最大 1024 tokens。
- 训练成本极低：单 GPU 上 fine-tuning 仅需约 2 分钟（Table 5：生成 ~1.5h vs 训练 ~2.4min）。

---

## 4. 数据集

### 训练与评估数据

| 数据集 | Train | Test | 难度 | License |
|--------|-------|------|------|---------|
| **GSM8K** | 7,473 | 1,319 | 小学数学应用题，多步推理 | MIT |
| **MATH** | 7,500 | 500 (MATH-500) | 高级数学，5 级难度（代数→高阶微积分） | MIT |

### 额外评估域（MMLU-Pro）

| 领域 | 说明 |
|------|------|
| Business | 商业推理 |
| Chemistry | 化学推理 |
| Physics | 物理推理 |

### 模型

**五个主模型：**
- Llama-3.2-3B（通用）
- Gemma-2-2B（通用）
- Qwen2.5-3B（通用）
- Qwen2.5-Math-1.5B（数学专用）
- DeepSeekMath-7B（数学专用）

**Scaling 研究：** Llama-3.2-{1B, 3B}、Llama-3.1-8B

---

## 5. 评估指标与主要结果

### 评估指标

- **Accuracy：** Python-based answer parsing 评估最终答案正确率。
- **Length：** 所有输出（含错误路径）的平均 output token 数。
- **Relative Accuracy / Relative Length：** 相对于 baseline prompt (Pang et al., 2024) 的百分比，用于跨方法比较。
- 评估使用 **greedy decoding** 确保可复现性。

### 主要结果（Table 2，五模型平均）

| 方法 | GSM8K Rel. Acc (%) | GSM8K Rel. Len (%) | MATH Rel. Acc (%) | MATH Rel. Len (%) |
|------|-------------------|-------------------|-------------------|-------------------|
| Baseline | 100.00 | 100.00 | 100.00 | 100.00 |
| Be Concise | 99.85 | 88.46 | 102.71 | 92.66 |
| Hand Crafted 2 | 98.27 | 77.10 | 101.62 | 85.26 |
| Human CoT (External) | 83.82 | 54.95 | 75.61 | 53.14 |
| GPT4o CoT (External) | 97.65 | 67.60 | 90.52 | 87.21 |
| Naive BoN | 98.79 | 87.17 | 101.74 | 89.89 |
| Rational Metareasoning | 97.21 | 84.93 | 103.02 | 90.56 |
| **FS-Human (Ours)** | 98.06 | 67.96 | 99.69 | 87.78 |
| **FS-GPT4o (Ours)** | 99.94 | 73.15 | 101.87 | 87.58 |
| **FS-Self (Ours)** | 98.86 | 77.51 | 102.67 | 88.50 |
| **FS-GPT4o-BoN (Ours)** | 97.00 | **64.25** | 102.56 | **76.30** |
| FS-GPT4o-BoN Budget-Matched | 97.44 | 67.15 | 101.58 | 80.43 |

### 关键发现

1. **FS-GPT4o-BoN 实现平均约 30% 的 token 缩减**（GSM8K: 35.75%, MATH: 23.70%），同时保持平均 accuracy 基本不变——这是 prior fine-tuning baseline (Rational Metareasoning, ~12% 缩减) 的 **2.4 倍**改进。

2. **Self-training 显著优于 external data fine-tuning（Figure 4）：** 用 GPT-4o 外部数据 fine-tuning 虽然长度缩减更大但落在 Pareto 曲线之下（accuracy 损失严重），而 self-training 方法位于 Pareto 前沿。

3. **Few-shot conditioning 对 math-specialized 模型同样有效：** Zero-shot prompting 对 Qwen2.5-Math 和 DeepSeekMath 几乎无效（Table 1 中 gray 标注），但 FS 方法在这些模型上同样实现了显著缩减（Table 13, Table 14）。

4. **Adaptive token reduction（Figure 5）：** 模型根据问题难度自适应调整缩减幅度——简单问题（Level 1-2）缩减 20%–40%，困难问题（Level 5）缩减较少，说明方法消除的是真正的冗余而非必要推理步骤。

5. **Scaling consistency（Figure 6）：** 在 Llama 1B / 3B / 8B 上，token 缩减率随模型规模增大而增加，accuracy 保持稳定。8B 模型上 FS-GPT4o-BoN 实现最大缩减且 accuracy 保持最好。

6. **训练长度到测试长度的线性转移（Figure 7）：** Fine-tuning 数据的 relative length 与模型测试输出 relative length 呈强线性相关，表明 SFT 能有效将长度缩减传递给模型。

7. **Cross-task generalization（Table 9）：** 在一个数据集上训练后，在另一个 OOD 数据集上仍能实现 10-12% 的长度缩减，且 accuracy 下降不超过 1.5%。

8. **Broader domain effectiveness（Table 10）：** 在 MMLU-Pro 的 Business / Chemistry / Physics 领域上，FS-GPT4o-BoN 同时实现 accuracy 提升（平均 +16.51%）和长度缩减（平均 26.82%），比 naive BoN 的长度效率高 3 倍以上。

### 实际效率提升

| 指标 | 改善范围 |
|------|---------|
| Wall-clock latency 缩减 | 15.38% – 52.94%（Table 7） |
| Peak memory 缩减 | 2.50% – 6.26%（Table 8） |
| Fine-tuning 开销 | 仅需 ~2.4 min / 单 GPU（vs 生成 ~1.5h） |

### 定性示例（Table 3）

以 Llama-3.1-8B 在 GSM8K 上的一道题为例：

- **原始输出（Zero-Shot）：** "To find the total number of bolts needed, we need to calculate the amount of white fiber first, since it's half the amount of blue fiber. Step 1: Determine the amount of blue fiber needed..."（冗长的 step-by-step 复述问题）
- **FS-GPT4o-BoN 输出：** "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2/2 = 1 bolt. Add the blue and white fiber together: 2 + 1 = 3 bolts. The answer is 3."（仅保留关键计算步骤，去除冗余解释）
