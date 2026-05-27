# TTRV: Test-Time Reinforcement Learning for Vision Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** arXiv 2510.06783
**Authors:** Akshit Singh, Shyam Marjit, Wei Lin, Paul Gavrikov, Serena Yeung-Levy, Hilde Kuehne, Rogerio Feris, Sivan Doveh, James Glass, M. Jehanzeb Mirza
**Affiliations:** Independent, IISc Bangalore, JKU Linz, Stanford, Tübingen AI Center, MIT-IBM Watson AI Lab, MIT CSAIL

| When to Adapt | multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation |
|---|---|
| 触发单位 Trigger Unit | online primitive: encountered test sample; main evaluation: sampled target subset |
| 参数/状态持久性 Persistence | GRPO-updated weights accumulate within the adaptation run; no per-sample reset |
| 与推理关系 Inference Coupling | online primitive: update during inference; main evaluation: adapt on sampled subset, then evaluate full target dataset |
| 输入可见性 Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Multi-protocol: Evaluation Boundary + No Immediate Reset |
| 证据状态 Evidence Status | paper-explicit |
| 辅助分类 Timing Regime | Multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation |
| 可见数据范围 Visibility Scope | Multi-protocol: Few-sample target subset + Streaming prefix only |

---

## When to Adapt 审计

- **辅助 taxonomy：** 这篇论文包含多个 protocol entry：`Timing Regime=Multi-protocol: Few-Sample Target Adaptation + Streaming Continual Adaptation`；`Visibility Scope=Multi-protocol: Few-sample target subset + Streaming prefix only`。
- **两轴编码：** `Input Visibility=Multi-protocol: Offline + Online`；`Update Persistence=Cumulative`；`Reset Boundary=Multi-protocol: Evaluation Boundary + No Immediate Reset`。

| Protocol Entry | Timing Regime | Visibility Scope | 输入可见性 Input Visibility | Update Persistence | Reset Boundary | 说明 Note |
|---|---|---|---|---|---|---|
| TTRV / online GRPO primitive | Streaming Continual Adaptation | Streaming prefix only | Online | Cumulative | No Immediate Reset | 遇到 test sample 后即时做 GRPO 更新，权重继续带到后续样本 |
| TTRV / 20-sample target-subset setting | Few-Sample Target Adaptation | Few-sample target subset | Offline | Cumulative | Evaluation Boundary | 先抽取 20 个 target samples 做 adaptation，再评估 full target dataset |
| TTRV / 1-sample ablation | Few-Sample Target Adaptation | Few-sample target subset | Offline | Cumulative | Evaluation Boundary | 与主设定相同，但 adaptation pool 缩到 1 个样本 |

- **更新何时触发：** 方法机制按“遇到 test sample 后多次采样、构造 reward、GRPO 更新”定义；主实验通常随机抽取 20 个 test samples 做 adaptation，再在 full dataset 上评估；消融还报告 1-sample adaptation。
- **服务当前样本还是后续样本：** Online primitive 同时服务当前/后续样本；20-sample 与 1-sample 实验设置中的更新主要服务 full target dataset 的后续评估。
- **参数/状态是否累积：** GRPO 更新得到的权重不会在每个样本后 reset；在 sampled adaptation set 内持续累积。
- **reset 边界：** 因此 paper-level 单一分类会丢失信息：framework primitive 更接近 streaming cumulative，主实验呈现则更接近 sampled-subset target-set calibration。

## 1. UPT 归属理由

**Family II — Sample-Relation Supervision (population consensus)**

TTRV 将 TTRL（Test-Time Reinforcement Learning）的思路从 LLM 扩展到 VLM（Vision-Language Model）。其核心监督信号完全来自模型自身的多次采样输出，不依赖任何外部标签、verifier 或人工标注：

- **Frequency-based reward（r₁）**：对同一测试样本采样 N 次，统计经验频率分布 p(ỹ_m)，将每个回答的 reward 设为其经验概率值。高频回答获得更高奖励，但低频回答也保留非零奖励——这是一种 soft, probabilistic 的 population consensus 信号。
- **Diversity control reward（r₂）**：计算经验分布的负 Shannon entropy −H(P) = Σ p(ỹ_m) log p(ỹ_m)，惩罚输出过于分散，鼓励模型逐步收敛到一致答案。
- **Combined reward**：R(ŷ_j) = r₁(ŷ_j) + α · r₂，其中 α 控制 convergence 与 diversity 的 trade-off。

最终使用 GRPO 进行 test-time policy optimization。所有奖励信号均为 intrinsic statistics（经验频率）和 model-generated content（采样回答）的组合，符合 UPT 定义。

**Carrier:** Policy Optimization | **Regime:** Test-time | **Level:** Semantic

---

## 2. 解决的问题

现有 VLM 的 RL-based fine-tuning 方法依赖标注数据和固定的 train/test 划分，无法在推理时动态适应新任务或新领域。Test-time training (TTT) 方法虽不需要标签，但多数面向 dual-encoder VLM（如 CLIP），受限于架构约束（如需要 class-level probability distribution），难以直接应用于 decoder-based VLM。

TTRV 解决的核心问题：**如何在没有任何标注数据的情况下，在 test time 对 decoder-based VLM 进行 RL 优化，从未标注的测试数据中提取自监督奖励信号以提升视觉理解能力。**

---

## 3. 方法介绍

### 3.1 GRPO 回顾

给定 prompt x，VLM 采样 n 个回答 {y_i}，每个回答的 advantage 为：

A_i = (r(x, y_i) − mean_j(r(x, y_j))) / std_j(r(x, y_j))

通过 clipped importance-weighted objective 更新策略，同时用 KL regularization 约束偏离参考策略的程度。

### 3.2 Test-Time Distributional Rewards

**Frequency-Based Reward：** 对测试样本 x 采样 N 个回答 {ŷ₁, ..., ŷ_N}，统计唯一回答集合 U = {ỹ₁, ..., ỹ_M}，计算经验概率 p(ỹ_m) = (1/N) Σ 1{ŷ_j = ỹ_m}。每个回答的 reward 为：

r₁(ŷ_j) = Σ_m p(ỹ_m) · 1{ŷ_j = ỹ_m}

即回答被赋予其经验频率作为 reward。与 majority voting（只取最高频答案）不同，这是一种 soft probabilistic supervision，保留了对不确定性的建模。

**Diversity Control Reward：** 计算经验分布的 Shannon entropy H(P) = −Σ p(ỹ_m) log p(ỹ_m)，令 r₂ = −H(P)。惩罚分散输出，鼓励模型逐步集中到一致预测。

**Combined Reward：** R(ŷ_j) = r₁(ŷ_j) + α · r₂，α 为超参数。

### 3.3 Optimization

通过标准 autoregressive language modeling objective 进行优化，reward 作为 soft, sample-level weighting。参数通过 gradient ascent 更新：θ ← θ + η ∇_θ E[R(y)]，GRPO 将 raw reward 替换为 relative advantage。

---

## 4. 数据集

### Image Recognition（8 个数据集）
- ImageNet, ImageNet-V2
- ImageNet-Rendition (R), ImageNet-Sketch (S), ImageNet-Adversarial (A)（OOD variants）
- Food101, DTD (Describable Textures Dataset)
- Resisc45（遥感场景分类）

### Visual Question Answering（8 个数据集）
- 数学推理：MathVerse, MathVista
- 科学/日常场景：SEED, MME, RealWorldQA
- 组合推理/空间推理：Capture, CRPE
- 图表理解：AI2D

共 **16 个数据集**，涵盖 image classification 和 VQA 两大任务。

---

## 5. 评估指标与主要结果

### 评估指标
- **Image Classification：** Top-1 Accuracy (%)
- **VQA：** Accuracy (%)

### 主要结果

**Image Classification (Table 1)：**
- InternVL3-2B + TTRV：平均 94.99%（base 62.03%，+32.95%），在 DTD 上提升 +52.49%，ImageNet-R 上 +30.88%
- InternVL2.5-4B + TTRV：平均 82.34%（base 70.47%，+11.88%）
- InternVL3-8B + TTRV：平均 95.71%（base 66.74%，+28.97%），ImageNet 上达 99.31%，**超越 GPT-4o (98.30%)**
- TTRV applied to InternVL-8B 在 image recognition 上平均超过 GPT-4o 2.3%

**VQA (Table 2)：**
- InternVL3-2B + TTRV：平均 57.15%（base 47.47%，+9.69%），AI2D 上 +28.07%
- InternVL2.5-4B + TTRV：平均 69.40%（base 66.37%，+3.03%）
- InternVL3-8B + TTRV：平均 55.56%（base 38.05%，+17.50%）

**关键发现：**
- 仅用每个数据集随机采样的 20 个测试样本即可获得显著提升
- 单样本 TTRV（Table 6）仍可产生有意义的改进（如 ImageNet-R +5.47%）
- 跨数据集泛化（Table 3/Figure 3）：在一个数据集上 TTRV 训练后迁移到完全不同数据集仍有大幅提升（如 Food→DTD +52.03%, Resisc→DTD +52.33%）
- Ablation 表明 frequency + diversity reward 组合优于单独使用任一项，也优于 majority voting baseline
- 泛化到其他模型家族：在 Qwen2.5-VL-3B 上同样有效
- Random reward 无法产生类似增益，说明 TTRV 的 reward 设计包含有意义的信号
