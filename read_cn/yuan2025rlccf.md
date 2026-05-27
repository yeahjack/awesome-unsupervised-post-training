# RLCCF — Reinforcement Learning from Coevolutionary Collective Feedback

> **加入 Survey 时间：** 2026-03-11

**论文**: Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback
**作者**: Wenzhen Yuan, Shengji Tang, Weihao Lin, Jiacheng Ruan, Ganqu Cui, Bo Zhang, Tao Chen, Ting Liu, Yuzhuo Fu, Peng Ye, LEI BAI
**机构**: Shanghai Jiao Tong University, Shanghai AI Lab, CUHK, Fudan University
**arXiv**: 2508.12338 (Aug 2025)

| 属性 | 值 |
|---|---|
| Method | RLCCF |
| Carrier | Policy Opt. |
| Regime | training-time |
| Level | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | prompt batch / rollout group |
| 参数/状态持久性 Persistence | full parameter accumulate across RL steps |
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

- **更新何时触发：** 更新在 deployment 前的离线 RL 阶段触发，基本单位是 prompt batch 下的一组 rollouts。
- **服务当前样本还是后续样本：** 当前 rollout group 的更新服务后续训练 step 与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整段 RL 训练中持续累积，不做 per-sample reset。
- **reset 边界：** 因此它属于 offline RL-style UPT schedule，而不是 test-time arrival-by-arrival adaptation。

## 1. UPT 归属理由

RLCCF 属于 **Family II (Sample-Relation Supervision)**，子类为 population consensus / collective agreement。

核心理由：RLCCF 不依赖任何外部 ground truth、人类标注或外部 reward model。其监督信号完全来自多个模型群体内部的 **集体一致性 (Collective Consistency, CC)**。具体而言，多个异构 LLM 各自独立采样候选答案，然后通过 Self-Consistency (SC) 加权的 majority voting 产生 pseudo-label，作为所有模型共享的 reward signal。每个模型的投票权重由其自身输出的 SC score（即模型对自身最常见答案的频率）决定——这是一种纯粹的 **模型间关系型内部监督**：reward 来源于群体共识，而非任何外部验证器。随后各模型利用该 pseudo-label 通过 GRPO 进行 policy optimization，实现协同进化 (coevolution)。

---

## 2. 解决的问题

- **单模型 self-feedback 的局限性**：TTRL、EMPO、Intuitor 等方法依赖单一模型自身输出构建 reward，容易导致：
  - 对错误答案过度自信 (overconfidence in incorrect outputs)
  - Reward hacking
  - Training collapse（正反馈循环失控）
- **外部监督的可扩展性瓶颈**：RLHF 需要昂贵人类标注训练 reward model；RLVR 需要 rule-based verifier 或标注数据，限制了适用领域
- RLCCF 提出利用 **多模型群体的互补性与多样性** 来产生更鲁棒、更准确的 pseudo-label，同时避免外部监督

---

## 3. 方法介绍

RLCCF 框架包含两个核心阶段：

### 3.1 Pseudo-label Estimation（SC 加权 Majority Voting）

给定模型集合 M = {M_1, ..., M_N}，对每个 query q：

1. **独立采样**：每个模型 M_n 独立生成 K 个候选回答 {o_{n,k}}
2. **计算 SC score**：对每个模型计算 Self-Consistency 分数 SC_n，定义为其最常见答案的频率（反映该模型输出的内部一致性/可靠程度）
3. **SC 加权投票**：pseudo-label 由所有模型的 SC 加权投票决定：

   â = argmax_a Σ_{n=1}^{N} Σ_{k=1}^{K} SC_n · I[a = o_{n,k}]

   SC score 高的模型在投票中拥有更大权重，确保更可靠的模型对集体决策贡献更多。

### 3.2 Policy Optimization（基于 GRPO）

以 pseudo-label â 为共享监督信号，对每个模型独立进行 policy optimization：

- **Reward 定义**：r_{n,k} = 1 if o_{n,k} = â, else 0（二值 reward，答案是否与 pseudo-label 一致）
- **优化目标**：采用 GRPO 框架，每个模型 M_n 最大化：

  J_n(q) = (1/K) Σ_{k=1}^{K} Â_{n,k} - β · KL(π_{θ_n}(·|q) || π_{ref_n}(·|q))

  其中 Â_{n,k} 为 clipped advantage（类似 PPO），β 控制 KL 正则化

### 3.3 理论分析：为什么多模型协作有效？

- **Bias Reduction**：将每个模型输出建模为 X_{n,k} ~ N(GT + ε_n, σ_n²)，当模型数量 N 增大时，聚合输出的均值趋近 ground truth（E[X] → GT as N → ∞），因为不同模型的偏差相互抵消
- **Model Complementarity**：不同模型在不同领域有互补优势（如 math vs. code），协作训练使各模型在弱项上也能显著提升

---

## 4. 数据集

### 训练数据
- **MATH-700**：从 MATH 训练集的 7 个子类别中各随机抽取 100 道题，共 700 题（无标签，仅用题目）

### 评估 Benchmarks
| 数据集 | 描述 | 难度 |
|---|---|---|
| **AIME 2024** | 美国数学邀请赛 | 最高 |
| **OlympiadBench** | 奥林匹克级别多模态科学问题 | 高 |
| **AMC 2024** | 美国数学竞赛 | 中高 |
| **MATH-500** | MATH 测试子集 | 中 |

---

## 5. 评估指标与主要结果

### 评估指标
- **Average Accuracy**：每道题采样 32 个回答取平均准确率
- **Group Majority Vote Accuracy**：4 个模型各采样 32 个回答后汇总投票
- **Label Accuracy**：训练过程中 pseudo-label 的准确率
- **Reward Accuracy**：训练过程中 reward signal 的准确率

### 主要结果（Table 1）

| Model | Base AVG | RLCCF AVG | 相对提升 |
|---|---|---|---|
| Qwen2.5-7B | 31.92 | 40.60 | +27.20% |
| GLM-4-9B | 32.99 | 38.48 | +16.63% |
| InternLM3-8B-Instruct | 34.51 | 39.80 | +15.31% |
| LLaMA-3.1-8B-Instruct | 23.68 | 25.51 | +7.74% |
| Group Majority Vote | 48.70 | **50.90** | +4.51% |

### 关键发现

1. **全面优于 self-feedback baselines**：RLCCF 的平均表现超过 Intuitor (+38.99%)、EMPO (+2.82%)、TTRL (+2.45%)
2. **拓展集体能力边界**：Group Majority Vote 从 48.70% 提升至 50.90%，而单模型 self-feedback 方法大多无法提升甚至损害群体投票准确率
3. **训练稳定性**：相比 Intuitor 的训练 collapse 问题，RLCCF 提供稳定且持续的性能增益
4. **SC 加权投票优于简单投票**：SC weighted voting 在所有模型上均优于 simple majority voting（Table 2），Group 级别提升 +1.29%
5. **Collective Consistency 增强**：训练后模型群体的答案分布收敛到一致的正确峰值，而 TTRL 训练的模型虽提升一致性但峰值可能指向不同（错误）答案
