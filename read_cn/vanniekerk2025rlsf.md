# RLSF: Post-Training LLMs via Reinforcement Learning from Self-Feedback

> **加入 Survey 时间：** 2026-03-11

**Paper:** Post-Training Large Language Models via Reinforcement Learning from Self-Feedback
**Authors:** Carel van Niekerk, Renato Vukovic, Benjamin Matthias Ruppik, Hsien-chin Lin, Milica Gašić (Heinrich Heine Universität, Düsseldorf, Germany)
**ArXiv:** 2507.19131
**Date:** 2025-07

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| RLSF | Pref. Opt. | training-time | Traj. |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / synthetic preference pair batch |
| 参数/状态持久性 Persistence | full parameter accumulate across epochs or iterations |
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

- **更新何时触发：** 更新在 deployment 前的 preference optimization 阶段触发，基本单位是模型自生成的 chosen / rejected pairs。
- **服务当前样本还是后续样本：** 当前 pair batch 的更新服务后续训练与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在训练 epoch / iteration 中持续累积，不做 sample-level reset。
- **reset 边界：** 因此它更像 offline pair-based post-training，而不是 online TTA。

## 1. UPT 归属理由
**Family III — Self-Generated Target Bootstrapping (internally generated preference pairs)**

RLSF 完全不依赖外部人类标注、gold answer 或外部 reward model。它利用**模型自身的 confidence（answer span 的 probability disparity）**对同一 prompt 的多条 CoT trace 进行排序，从而构造 synthetic preference pairs，再通过 PPO 或 DPO 进行 preference optimization。整个过程是纯 self-supervised 的 intrinsic reward 驱动，属于典型的"内部生成 preference pairs → preference optimization"范式。

---

## 2. 解决的问题

LLM 在推理密集型任务上存在两大核心问题：
1. **Calibration 失调**：经过 RLHF 后的模型常出现 overconfidence，即模型输出的置信度与实际准确率不匹配，尤其在复杂推理任务上表现严重。
2. **推理能力不稳定**：LLM 在需要多步逻辑推理的任务上（数学、常识推理）性能不可靠，CoT prompting 依赖 prompt 设计且效果不一致。

现有方法要么依赖人类标注（RLHF），要么依赖外部强 LLM 的反馈（RLAIF），成本高且可扩展性有限。RLSF 提出用模型自身 uncertainty 作为 intrinsic reward 信号，模仿人类在缺乏外部反馈时利用 confidence 进行学习的机制，以 data-efficient 的方式同时改善 calibration 和推理准确率。

---

## 3. 方法介绍

RLSF 的核心思路：**well-calibrated model 中，answer confidence 与 reasoning 质量和 accuracy 正相关** → 用 confidence 排序 CoT traces 构造 preference data → 进行 RL fine-tuning。

### 3.1 Chain-of-Thought Decoding 生成候选

给定 input query $q$，在**第一个 decoding step** 采样 top-$K$ 概率的 token $w_k$（$k=1,...,K$），然后从每个 $w_k$ 出发用 greedy auto-regressive decoding 生成完整 hypothesis $h_k$。这样为每个 prompt 生成 $K$ 条不同的 reasoning trace。

### 3.2 Answer Span 识别与 Confidence 计算

在每条 hypothesis $h_k$ 末尾添加 "So the answer is" 提示模型继续生成 answer tokens $g'_k$，然后在原始 hypothesis 中通过 string matching 定位 answer span $g_k$。

Confidence 使用 **probability disparity** 衡量：

$$c = \frac{1}{M} \sum_{i=0}^{M-1} \left[ \max_w \pi_\theta(w|q \oplus h_{m+i}) - \max_{w \neq \arg\max \pi_\theta} \pi_\theta(w|q \oplus h_{m+i}) \right]$$

即 answer span 中每个 token 位置上，最高概率 token 与次高概率 token 的概率差的平均值。Disparity 越高，模型对该 answer 越确定。相比单纯的 token probability，disparity 能更可靠地反映模型的真实 confidence。

### 3.3 构造 Preference Dataset

根据 confidence score $c_k$ 对 $K$ 条 hypothesis 进行排序，高 confidence → preferred，低 confidence → rejected，形成 preference dataset $D$。Raw reward values 线性缩放到 $[-1, 1]$ 区间以稳定训练。

### 3.4 Policy Optimization

提供两种优化方式：
- **RLSF with PPO**：用 preference dataset $D$ 训练 Bradley-Terry reward model $R_\phi$（在 base model 上附加 scalar regression head，用 LoRA fine-tune），然后用 PPO 优化 policy $\pi_\theta$。
- **RLSF with DPO**：直接用 preference dataset $D$ 以 DPO loss 对 policy 进行 supervised fine-tuning，绕过显式 reward model 训练。

实验表明 **PPO 一致优于 DPO**，作者认为 RL 更适合利用 intrinsic motivation 信号。

### 3.5 实现细节
- CoT decoding 时 $K=10$ 条候选
- Reward model 使用 LoRA fine-tune base model + scalar regression head
- 训练使用 TRL (Transformer Reinforcement Learning) 库
- PPO 超参：学习率 5e-5，epochs 5，温度 0.7，KL 系数 β=0.05，clipping ε=0.2，γ=0.98，GAE λ=0.95
- DPO 超参：学习率 5e-5，epochs 5，label smoothing 0.01，β=0.2
- RLSF 训练后的模型推理时间不受影响（无需 CoT decoding）

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | MultiArith | 多步算术应用题 |
| 数学推理 | GSM8K | 小学数学应用题，需要多步计算和推理 |
| 多选题 QA | CommonsenseQA | 常识推理，约束答案格式 |
| 多选题 QA | ARC Easy | 小学科学推理 |
| Reward Model 评估 | RewardBench (Math Reasoning subset) | 评估 reward model 对 preferred/rejected response 的排序能力 |
| Bias 评估 | XSTest | 检测 LLM 的 exaggerated safety behavior |
| Bias 评估 | AlpacaEval | 用 GPT-4o 做 zero-shot annotator 对比模型输出 |

---

## 5. 评估指标与主要结果

### 评估指标

- **Answer Accuracy**：模型回答与 ground truth 的精确匹配率
- **Expected Calibration Error (ECE)**：token 级别 confidence 与实际 correctness 的对齐程度（越低越好）
- **Reward Model Accuracy**：在 RewardBench 上 reward model 正确排序 preferred/rejected 对的比例

### 主要结果

#### Reward Model 评估 (RewardBench Math Reasoning)

| Reward Model | 所需数据 | Accuracy↑ |
|---|---|---|
| URM (LLaMa 3.1 8B) | Prompt + Preference | 97.00 |
| QRM (LLaMa 3.1 8B) | Prompt + Preference | 96.80 |
| QWEN 2.5 7B AfD | Prompt + Answer | 89.29 |
| Gemma 2 2B AfD | Prompt + Answer | 73.12 |
| **QWEN 2.5 7B RLSF** | **仅 Prompt** | **76.13** |
| **Gemma 2 2B RLSF** | **仅 Prompt** | **81.43** |

RLSF 是唯一仅依赖 prompt（不需要 answer 或 preference 标注）就能导出 reward model 的方法，且性能具有竞争力。

#### 数学推理 (Table 2)

**Gemma 2 2B：**

| 设置 | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| Base (Greedy) | 98.12 | 7.43 | 85.57 | 12.24 |
| Base (CoT K=10) | 99.01 | 4.12 | 89.18 | 10.94 |
| RLHF (PPO) + URM | 97.83 | 12.73 | 82.43 | 17.83 |
| **RLSF (DPO)** | 96.13 | 10.52 | 84.74 | 17.43 |
| **RLSF (PPO)** | **98.83** | **7.81** | **88.14** | **12.54** |

**QWEN 2.5 7B：**

| 设置 | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| Base (Greedy) | 79.44 | 20.75 | 53.67 | 42.86 |
| Base (CoT K=10) | 77.77 | 22.22 | 58.52 | 41.46 |
| RLHF (PPO) + URM | 74.13 | 23.24 | 46.82 | 51.24 |
| **RLSF (DPO)** | 71.62 | 25.12 | 50.73 | 38.15 |
| **RLSF (PPO)** | **78.06** | **18.35** | **58.62** | **41.92** |

#### 多选题 QA

**ARC Easy (Gemma 2 2B):**

| 设置 | Accuracy↑ | ECE↓ |
|---|---|---|
| Base (Greedy) | 96.96 | 16.12 |
| Base (CoT K=10) | 96.28 | 3.03 |
| RLSF (DPO) | 97.05 | 18.83 |
| **RLSF (PPO)** | **97.04** | **5.12** |

**CommonsenseQA (Phi-2):**

| 设置 | Accuracy↑ | ECE↓ |
|---|---|---|
| Base (Greedy) | 54.46 | 25.12 |
| Base (CoT K=10) | 58.91 | 23.11 |
| RLSF (DPO) | 59.79 | 30.91 |
| **RLSF (PPO)** | **61.13** | **19.64** |

#### Bias 评估 (XSTest, AlpacaEval)

| Training Dataset | Evaluation Dataset | Preference(%)↑ |
|---|---|---|
| GSM8K + MultiArith | XSTest | 50.73 |
| CommonsenseQA | XSTest | 51.82 |
| XSTest | XSTest | 63.24 |

在推理任务上训练的 RLSF 不会放大 base model 的 safety bias。

#### Discount Factor γ 消融 (Gemma 2 2B, RLSF+PPO)

| Variant | MultiArith Acc↑ | MultiArith ECE↓ | GSM8K Acc↑ | GSM8K ECE↓ |
|---|---|---|---|---|
| DPO | 96.13 | 10.52 | 84.74 | 17.43 |
| PPO (γ=1.0) | 98.52 | 8.12 | 87.13 | 12.49 |
| PPO (γ=0.98) | 98.83 | 7.81 | 88.14 | 12.54 |

Discount factor γ=0.98 略优于 γ=1.0，说明适度的 discounting 有助于优化。

### 关键发现

1. **PPO 一致优于 DPO**：在所有任务和模型上，RLSF(PPO) 在 accuracy 和 ECE 上均优于 RLSF(DPO)，说明 RL 更适合利用 intrinsic motivation 信号，而 DPO 中缺少 reward discounting 机制。
2. **同时改善 calibration 和 accuracy**：RLSF(PPO) 在提升准确率的同时降低 ECE，而 RLHF(PPO)+URM 反而导致 calibration 退化（Gemma 2 ECE 从 7.43 升至 12.73）。
3. **仅用 prompt 即可导出 competitive reward model**：RLSF 在 RewardBench 上仅用 prompt 就达到 81.43 准确率（Gemma 2），且对 base LLM 选择不敏感。
4. **不放大 bias**：在推理任务上训练的 RLSF 模型在 XSTest 安全性测试中表现与 base model comparable（~50-52%），不会引入额外的 safety-related bias。
5. **Reward model 学到有意义的 token 级 reward**：正确答案获得高 reward，中间推理步骤也被奖励，冗长解释获得较低分数，无推理的裸答案得分不高，说明模型学会了区分 correctness 与 informative justification。
6. **推理成本不增加**：虽然构造训练数据时需要 CoT decoding（推理成本 $K$ 倍），但训练后模型的推理成本与原模型相同。
