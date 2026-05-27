# Absolute Zero: Reinforced Self-play Reasoning with Zero Data

> **加入 Survey 时间：** 2026-03-11

> **Audit Update (2026-03-11): Retained in the purple Tool-as-Verifier branch under the survey's `whole-pipeline` rule.**
>
> 保留 caveat：论文主体训练确为 `#data = 0`，但 seeding 阶段存在一个 handcrafted zero triplet 与 proposer prompt template，作为最小启动 scaffold。当前 survey 将其视为 `small seed`，不视作 large GT dataset 依赖。

> **Method:** AZR (Absolute Zero Reasoner) | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Semantic
>
> arXiv 2505.03335 — Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang (清华大学, BIGAI, Penn State)

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | self-play episode / synthesized task batch |
| 参数/状态持久性 Persistence | full parameter accumulate across self-play rounds |
| 与推理关系 Inference Coupling | offline pre-deployment |
| 输入可见性 Input Visibility | Offline |
| Update Persistence | Cumulative |
| 重置边界 Reset Boundary | Deployment Boundary |
| 证据状态 Evidence Status | note-explicit |
| 辅助分类 Timing Regime | Offline Corpus UPT |
| 可见数据范围 Visibility Scope | Pre-deployment corpus |

---

## When to Adapt 审计

- **辅助 taxonomy：** `Timing Regime=Offline Corpus UPT`；`Visibility Scope=Pre-deployment corpus`。
- **两轴编码：** `Input Visibility=Offline`；`Update Persistence=Cumulative`；`Reset Boundary=Deployment Boundary`。

- **更新何时触发：** 更新在 deployment 前的 self-play / self-evolution 回合中触发，而不是由真实部署样本在线驱动。
- **服务当前样本还是后续样本：** 当前 self-play episode 产生的任务与轨迹主要服务后续训练轮次与最终部署模型。
- **参数/状态是否累积：** 参数在多轮 self-play 中持续累积，不做 per-sample reset。
- **reset 边界：** 因此它是 offline self-evolution schedule，而非 test-time cumulative adaptation。

## 1. UPT 归属理由

Absolute Zero 属于 **Tool-Augmented Adjacent Extensions**，子方向为 **code executor as unified verifier**。

核心机制：单一模型同时担任 Proposer（生成推理任务）和 Solver（解决任务）两个角色，形成自博弈闭环。**Python 代码执行器**作为唯一外部工具，既验证 Proposer 生成的任务的有效性（program integrity、safety、determinism），又验证 Solver 答案的正确性。主体训练过程 **不使用任何外部数据、ground truth 或人类标注**；仅在 seeding 阶段使用一个 handcrafted zero triplet 与 prompt template 作为最小启动 scaffold。

与 strict UPT 的区别：reward 不来自模型内部统计量（entropy、consensus 等），而来自 **外部代码执行环境的确定性反馈**。但该工具是通用的、现成的，不需要针对任务收集标注。

---

## 2. 解决的问题

- 传统 RLVR（如 DeepSeek-R1-Zero）虽然不标注推理过程，但仍依赖人工策划的 QA 数据集。
- 高质量人类数据稀缺性日益显现，long-term scalability 受限。
- 在超智能假设场景下，人类提供的任务可能无法对超智能系统产生有效学习。

Absolute Zero 提出：能否让模型既生成任务又解决任务，仅靠代码执行器提供 grounded feedback，完全无需外部数据？

---

## 3. 方法介绍

### 3.1 总体框架

AZR 在 (Program $P$, Input $I$, Output $O$) 三元组上构建三种推理模式：

| 推理模式 | 已知 | 待推断 | 含义 |
|----------|------|--------|------|
| **Deduction（演绎）** | $P, I$ | $O=?$ | 给定程序和输入，推断输出 |
| **Abduction（溯因）** | $P, O$ | $I=?$ | 给定程序和输出，推断输入 |
| **Induction（归纳）** | $\{I_n, O_n\}$ | $P=?$ | 从输入-输出示例归纳程序 |

### 3.2 自博弈循环

1. **Proposer** $\pi_\theta^{\text{propose}}$：基于 buffer 中历史 triplet 条件 $z$ 生成任务 $\tau$
2. **代码执行器验证**：
   - Program Integrity：执行程序检查语法正确性和能否正常返回输出
   - Program Safety：检测危险库使用
   - Determinism Check：独立运行 $j$ 次确保输出一致
3. **Solver** $\pi_\theta^{\text{solve}}$：接收验证后的任务 $(x, y^*)$ 并生成答案
4. **双重奖励计算**：Proposer 获 learnability reward，Solver 获 accuracy reward
5. **联合策略优化**：两角色联合更新

### 3.3 核心公式

**联合目标函数：**
$$J(\theta) := \max_\theta \mathbb{E}_{z \sim p(z)} \mathbb{E}_{(x,y^*) \sim f_e(\cdot|\tau), \tau \sim \pi_\theta^{\text{propose}}(\cdot|z)} \left[ \lambda r_e^{\text{propose}}(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|x)} [r_e^{\text{solve}}(y, y^*)] \right]$$

**Proposer reward（Learnability reward）：**
$$r^{\text{propose}} = \begin{cases} 0, & \text{if } \bar{r}^{\text{solve}} = 0 \text{（不可解）} \\ 1 - \bar{r}^{\text{solve}}, & \text{otherwise} \end{cases}$$

其中 $\bar{r}^{\text{solve}} = \frac{1}{G}\sum_{i=1}^{G} r^{\text{solve}}(o_i)$ 为 Monte Carlo rollout 平均成功率。
过难（=0）和过易（=1）的任务 reward 低，中等难度最高——形成自适应课程学习。

**Solver reward：** $r^{\text{solve}} = \mathbb{I}(y = y^*)$

**Task-Relative REINFORCE++ (TRR++)：**
$$A^{\text{norm}}_{\text{task,role}} = \frac{r - \mu_{\text{task,role}}}{\sigma_{\text{task,role}}}$$

对 6 种 task-role 配置（3 task types × 2 roles）分别计算 baseline。

### 3.4 Answer verification 细节

- **Deduction**：直接匹配输出 $o^\pi = o^*$
- **Abduction**：验证 $p(i^\pi) = p(i^*)$（运行程序检查输入是否产生同样输出）
- **Induction**：验证 $\{p^\pi(i_n^*) = o_n^*\}_N$（模型归纳的程序在所有测试用例上正确）

---

## 4. 训练配置

| 项目 | 细节 |
|------|------|
| Base models | Qwen2.5-7B, Qwen2.5-7B-Coder（主实验）；3B, 14B, Llama-3.1-8B |
| 训练算法 | Task-Relative REINFORCE++ (TRR++)（自研） |
| Batch size | 64 × 6 = 384（2 roles × 3 task types） |
| 学习率 | 1e-6（常数），AdamW |
| 外部数据 | **0**（零数据） |
| 输出格式 | `<think>` / `<answer>` (DeepSeek R1 格式) |

---

## 5. 核心结果

| 模型 | 外部数据 | Code Avg | Math Avg | Overall |
|------|----------|----------|----------|---------|
| Qwen2.5-7B (base) | - | 52.0 | 27.5 | 39.8 |
| AZR-Base-7B | **0** | 55.2 | 38.4 | **46.8** |
| AZR-Coder-7B | **0** | 61.6 | 39.1 | **50.4** |
| ORZ（最佳 curated baseline） | 57k | 55.6 | 41.6 | 48.6 |

- 零数据情况下超越使用数万条人工数据的基线
- 规模效应：3B +5.7, 7B +10.2, 14B +13.2
- 跨域迁移显著：code 训练带来 math +15.2 提升

---

## 6. UPT Survey 定位

**Tool-as-Verifier 典型性：** Absolute Zero 是最纯粹的"tool as verifier"形式——代码执行器同时扮演：
(a) 任务工厂的质检员（验证 Proposer 生成任务的有效性），
(b) 答案裁判（验证 Solver 答案的正确性），
(c) 课程设计师的间接信号源（通过 learnability reward 调节任务难度）。

**与 T³RL 的互补：** T³RL 在 test-time 使用 code interpreter 增强 majority-vote pseudo-labels，Absolute Zero 在 training-time 使用 code executor 驱动零数据自博弈。两者分别代表 tool-as-verifier 在推理阶段和训练阶段的应用。

**与 strict UPT 的关系：** AZR 的自博弈结构继承了 self-generated-target bootstrapping（自生成任务）和 sample-relation supervision（learnability reward 基于 Solver 群体表现计算）的特征，但 ground truth 完全来自代码执行器而非模型内部，因此处于 UPT 与 environment-grounded RL 的交界地带。
