# CREAM: Consistency Regularized Self-Rewarding Language Models

> **加入 Survey 时间：** 2026-03-11

**Paper:** CREAM: Consistency Regularized Self-Rewarding Language Models
**Authors:** Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, Huaxiu Yao
**Affiliations:** University of North Carolina at Chapel Hill, Nanyang Technological University, National University of Singapore, Microsoft Research
**ArXiv:** 2412.05321
**Venue:** ICLR 2025
**Code:** https://github.com/Raibows/CREAM

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| CREAM | Pref. Opt. | training-time | Semantic |

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

**Family IV — Internal Evaluator Bootstrapping → Preference Optimization → Self-Rewarding**

CREAM 属于 Internal Evaluator Bootstrapping 家族，核心在于模型**自身同时充当 policy model 和 reward model**，无需外部标注或外部奖励模型。具体而言，模型利用 DPO 的 intrinsic reward (即 $\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$) 对自生成的多个 response 进行打分与排序，形成 preference pairs 后再用 DPO 进行偏好优化，整个过程迭代进行。这属于典型的 self-rewarding 范式。CREAM 在此基础上引入了 **consistency regularization**，利用不同迭代间 reward ranking 的一致性来缓解 self-bias 和 overconfidence，但核心的数据生成与优化循环仍属于 self-rewarding preference optimization。

---

## 2. 解决的问题

Self-Rewarding Language Models (SRLMs) 存在以下关键问题：

1. **Rewarding bias 累积**：同一个 LLM 既做 policy 又做 reward model，在多轮迭代中 reward 的偏差会逐轮积累，导致 preference data 质量逐渐下降。
2. **Overconfident preference labeling**：当两个 response 质量相近时，SRLM 仍被强制做出偏好判断，产生噪声标注（noisy labeling），伤害后续 DPO 训练。
3. **小模型（7B）上效果退化**：实证发现 SRLM 在 7B 规模 LLM 上经过多轮迭代后性能可能显著下降（尤其 Llama-2），标准 LLM-as-a-Judge prompting 对小模型不可行。
4. **缺乏 reward 可靠性信号**：传统 SRLM 框架中没有任何机制来衡量 preference labeling 的可信度。

---

## 3. 方法介绍

### 3.1 Generalized Iterative Preference Fine-tuning Framework

CREAM 首先提出一个统一的迭代偏好微调框架。优化目标为：

$$L(\theta, z) = L_{\text{SFT}}(\theta; D_S) + \mathbb{E}_{x \sim D_U; y, y' \sim \pi_{\theta_t}(\cdot|x)} [L_{\text{DPO}}(\theta; y, y', x, z)]$$

其中 $z(y, y', x) \in \{0, 1\}$ 为 preference label function。该框架采用 **两步交替优化**：

- **Step 1 (Preference-labeling)**：固定 $\theta = \theta_t$，利用 intrinsic reward 确定偏好标签：
  $$z_{t+1}(y, y', x) = \mathbf{1}[\log \pi_{\theta_t}(y|x) - \log \pi_{\text{ref}}(y|x) \geq \log \pi_{\theta_t}(y'|x) - \log \pi_{\text{ref}}(y'|x)]$$
- **Step 2 (Learning)**：固定 $z_{t+1}$，最小化 DPO loss 更新参数 $\theta_{t+1}$。

**关键选择**：不使用 LLM-as-a-Judge prompting（对 7B 模型不可靠），而是使用 DPO 的 **intrinsic reward** $r_\theta(x,y) \propto \log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$ 进行打分排序。

### 3.2 Consistency Regularization

核心洞察：当两个 response 质量相近时，不同 reward model 的 ranking 结果往往不一致。利用这种**跨迭代的 ranking 不一致性**作为 confidence 信号。

在优化目标中加入正则化项：

$$L(\theta, z) = L_{\text{SFT}}(\theta; D_S) + \mathbb{E}[L_{\text{DPO}}(\theta; y, y', x, z) + \lambda L_{\text{Reg}}(\theta; y, y', x)]$$

其中 $L_{\text{Reg}}$ 的期望等价于 $2 \text{KL}(u(\cdot) \| P_\theta(\cdot))$，即将质量相近的 response pair 的偏好分布正则化为均匀分布。

**Theorem 3.3** 证明上述正则化等价于 **soft-labeled DPO**：

$$L(\theta, z) = C_\lambda L_{\text{DPO}}(\pi_\theta, D_{\text{DPO}}) + (1 - C_\lambda) L_{\text{DPO}}(\pi_\theta, D_{\text{RDPO}})$$

其中 $C_\lambda = (1+\lambda)/(1+2\lambda)$，$D_{\text{RDPO}}$ 是 preference 顺序翻转的数据集。这在实现上等同于 label smoothing。

### 3.3 Adaptive Consistency Rate Estimation

用 **Kendall's Tau 系数** 度量当前模型 $\theta_t$ 与上一轮模型 $\theta_{t-1}$ 对同一组 response 的 ranking 一致性：

$$\tau_j = \frac{2}{N(N-1)} \sum_{1 \leq i < i' \leq N} [\mathbf{1}[(J_{ij} - J_{i'j})(K_{ij} - K_{i'j}) > 0] - \mathbf{1}[(J_{ij} - J_{i'j})(K_{ij} - K_{i'j}) < 0]]$$

Consistency rate 计算为 $C = |D_U|^{-1} \sum_j (\tau_j + 1)/2$，其中 $J$ 为当前模型排名，$K$ 为上一轮模型排名。

**Lemma 3.4** 建立了 Kendall τ 与正则化参数 $\lambda$ 的理论联系：$\mathbb{E}[\tau_j] = 1 - 2\lambda$，从而 $C_\lambda \approx (1 + \tau_j)/2$。

### 3.4 完整算法流程 (Algorithm 1)

1. **SFT 阶段**：在种子 SFT 数据 $D_S$ 上微调初始模型 $\theta_0 \to \theta_1$。
2. **迭代偏好训练**（T 轮）：
   - **Response sampling**：对每个 prompt $x_j \in D_U$，用 $\pi_{\theta_t}$ 生成 $N=5$ 个 response。
   - **Self-rewarding**：用 intrinsic reward $r_{ij} = \log \pi_{\theta_t}(y_{ij}|x_i) - \log \pi_{\theta_0}(y_{ij}|x_i)$ 打分并排序得 rank $J$。
   - **Consistency estimation**：用上一轮模型 $\theta_{t-1}$ 同样打分排序得 rank $K$，计算 Kendall τ 及 consistency rate $C$。
   - **Preference data 构建**：取最高分与最低分 response 组成 $D_{\text{DPO}}$，翻转后组成 $D_{\text{RDPO}}$。
   - **Consistency-regularized training**：$\theta_{t+1}$ 通过最小化 $C \cdot L_{\text{DPO}}(\pi_{\theta_t}, D_{\text{DPO}}) + (1-C) \cdot L_{\text{DPO}}(\pi_{\theta_t}, D_{\text{RDPO}})$ 更新。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| Instruction Following | OpenAssistant | ~3.4K 人工标注样本作为种子 SFT 数据 $D_S$ |
| 科学推理 | ARC-Easy | AI2 Reasoning Challenge，简单科学问答 |
| 科学推理 | ARC-Challenge | AI2 Reasoning Challenge，困难科学问答 |
| 常识推理 | OpenBookQA | 开放知识库问答 |
| 社会常识 | SIQA (Social IQa) | 社会交互推理 |
| 数学推理 | GSM8K | 小学数学文字题 |
| Reward 评估 | RewardBench | 评估 reward model 的 pairwise ranking 准确率 |

**Unlabeled prompt set $D_U$**：混合 OpenAssistant prompts 与上述 5 个下游任务的 train split prompts（仅保留 prompt），共约 21K prompts，均分到各迭代中使用。

---

## 5. 评估指标与主要结果

### 评估指标

- **Exact Match Accuracy**：下游任务的精确匹配准确率
- **Ranking Consistency**：包括 Consistency Rate $C$、Kendall τ、Spearman 相关系数、TopOrder（最优/最差 response 排名稳定性）
- **Ranking Accuracy**：在 RewardBench 和 curated preference data 上的 pairwise ranking 准确率
- **Alignment Arena Win Rate**：由 GPT-4o 评判的 head-to-head 对比胜率

### 主要结果

**下游任务性能（Llama-3，M3 迭代）：**

| 方法 | ARC-Easy | ARC-Challenge | OpenBookQA | SIQA | GSM8K | Average |
|------|----------|---------------|------------|------|-------|---------|
| Initial (M0) | 86.29 | 80.37 | 86.00 | 68.58 | 78.01 | 79.85 |
| SFT (M1) | 86.78 | 80.14 | 86.40 | 69.50 | 78.39 | 80.24 |
| SRLM (M3) | 87.17 | 81.23 | 87.30 | 70.37 | 77.48 | 80.71 |
| Oracle (M3) | 89.31 | 81.31 | 90.20 | 73.75 | 76.04 | 82.12 |
| **CREAM (M3)** | **89.52** | **83.36** | **90.20** | **72.06** | **81.73** | **83.37** |

**下游任务性能（Llama-2，M3 迭代）：**

| 方法 | ARC-Easy | ARC-Challenge | OpenBookQA | SIQA | GSM8K | Average |
|------|----------|---------------|------------|------|-------|---------|
| Initial (M0) | 61.07 | 48.98 | 62.20 | 50.36 | 23.65 | 49.25 |
| SRLM (M3) | 46.55 | 34.47 | 49.20 | 48.06 | 22.14 | 40.08 |
| **CREAM (M3)** | **62.08** | **48.81** | **64.60** | **51.22** | **25.85** | **50.51** |

**Ranking Consistency（Llama-3, M3 vs M2）：**

| 方法 | Consistency C↑ | Kendall τ↑ | Spearman↑ | TopOrder↑ |
|------|----------------|------------|-----------|-----------|
| SRLM | 0.46±0.19 | −0.08±0.38 | 0.50±0.22 | 0.12±0.33 |
| CREAM | 0.92±0.09 | 0.84±0.19 | 0.95±0.07 | 0.59±0.49 |

**Alignment Arena（GPT-4o 评判）：**
- CREAM M3 vs SRLM M3：Win 34% / Tie 45% / Lose 21%
- CREAM M3 vs Oracle M3：Win 11% / Tie 64% / Lose 25%

### 关键发现

1. **标准 SRLM 在 7B 模型上表现不佳**：尤其 Llama-2 在迭代后性能大幅退化（Average 从 49.35 降至 40.08），表明小模型的 self-rewarding 存在严重 bias 累积问题。
2. **CREAM 显著优于 SRLM**：在 Llama-3 上 CREAM M3 (83.37) 甚至超过了使用外部 reward model 的 Oracle (82.12)；在 Llama-2 上 CREAM 是唯一能在迭代中持续提升的 self-rewarding 方法。
3. **CREAM 跨迭代持续改进**：CREAM 在每一轮迭代（M1→M2→M3）中均实现性能提升（尤其 Llama-3 上所有 5 个任务在 M2→M3 均↑），而 SRLM 在 M2→M3 经常退化。
4. **Ranking consistency 显著提高**：CREAM 的 Kendall τ 从 SRLM 的 −0.08 提升至 0.84（M3 vs M2），说明正则化有效稳定了 reward ranking。
5. **DPO rewarding 优于 prompt rewarding**：对 7B 模型，基于 intrinsic reward 的 DPO ranking 明显优于 LLM-as-a-Judge prompting（后者在 M1→M2 就开始退化）。
6. **Adaptive consistency rate 优于手动设置**：基于 Kendall τ 自动计算的 $C$ 优于手动搜索固定值的 CREAM w/o RC variant，且免去超参搜索开销。
7. **Kendall τ 是最佳 consistency metric**：对比 Spearman 和 TopOrder，Kendall τ 在多数数据集上取得更高性能，且理论上更鲁棒。
