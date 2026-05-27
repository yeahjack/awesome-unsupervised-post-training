# CoVo: Consistent Paths Lead to Truth — Self-Rewarding RL for LLM Reasoning

> **加入 Survey 时间：** 2026-03-11

> **Method:** CoVo | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Traj.
>
> arXiv 2506.08745, Jun 2025
> Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao
> Zhejiang University, Alibaba Cloud Computing, Nanyang Technological University

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

**Family II — Sample-Relation Supervision (trajectory-level consistency)**

CoVo 完全不依赖外部 ground truth、human labels 或 external reward model。其核心信号来源于模型自身对同一 prompt 采样得到的多条 reasoning trajectory 之间的 **关系型比较**：

- **Consistency**：衡量一条 trajectory 中有多少 intermediate reasoning state 在 model likelihood 意义上指向其自身最终答案（而非其他 trajectory 的答案）。
- **Volatility**：衡量 trajectory 中最后一个偏离自身最终答案的 intermediate state 出现在多晚的位置，反映推理路径的稳定程度。

这两个特征均从多条 trajectory 产生的 distance matrix 中提取——distance matrix 记录每个 intermediate state 到所有候选 final answer 的 token-level log-probability 距离。奖励信号的本质是 **trajectory 间关系的内在统计量**（正确轨迹 high consistency + low volatility vs. 错误轨迹 low consistency + high volatility），属于 sample-relation supervision。此外，CoVo 还引入 curiosity reward（基于 token probability 与 uniform distribution 的 KL divergence），鼓励模型探索不确定区域，同样不需要外部标签。

---

## 2. 解决的问题

1. **外部监督依赖**：现有 RL for reasoning 方法严重依赖 ground-truth verifiable answers 或 pretrained reward model，在开放式推理场景中标签获取成本高、reward model 存在 distributional mismatch。
2. **现有 self-rewarding 方法仅关注 final answer**：TTRL（majority voting）、EMPO（answer cluster probability）等方法只利用最终答案分布，忽略了 intermediate reasoning state 蕴含的丰富信息，容易被 reward hacking（如对所有数学题输出 "0" 即可骗过 majority voting）。
3. **训练过程中多样性坍缩**：随着 RL 训练推进，模型采样多样性下降，导致 self-rewarding 信号退化；需要额外机制维持探索。

---

## 3. 方法介绍

### 3.1 Distance Matrix 与 Trajectory Pattern 观察

给定 prompt x，采样 N 条 reasoning trajectory。每条 trajectory 由 T 个 intermediate reasoning state 和一个 final answer 组成。定义 state s_i 到 answer y 的距离为 negative average log-probability：

$$d(s_i, y) = -\frac{1}{|y|}\sum_{j=1}^{|y|}\log \pi_\theta(y[j] \mid s_i, y[:j])$$

构建 distance matrix **D** (T x K)，K 为不同候选答案数。实证发现：正确 trajectory 的 intermediate state 很早就稳定地指向自身 final answer（high consistency, low volatility），错误 trajectory 则表现出波动和延迟收敛。

### 3.2 Consistency 与 Volatility

- **Consistency**：$Con(\tau) = \frac{1}{T}\sum_{i=0}^{T-1}\mathbb{I}(\mathbf{D}[i,0] = \min_{0\le k<K}\mathbf{D}[i,k])$，即 intermediate state 中指向自身答案的比例。
- **Volatility**：$Vol(\tau) = \frac{1}{T}\max\{i \in [0,T-1] \mid \mathbf{D}[i,0] \neq \min_{0\le k<K}\mathbf{D}[i,k]\}$，即最后一个偏离自身答案的 state 位置占比。

### 3.3 Intrinsic Reward（Vectorial Aggregation）

将同一 final answer 的 trajectory 分为一组（共 G 条），每条 trajectory 编码为二维向量：

$$\mathbf{v}_i = Con(\tau_i) \cdot [\cos(Vol(\tau_i)),\; \sin(Vol(\tau_i))]$$

组内向量求和后取模长作为 group reward：

$$r_{\text{int}}^V = \frac{1}{G}\|V_{\text{group}}\|$$

Vectorial aggregation 相比线性聚合 $r_{\text{int}}^L = \frac{1}{G}\sum(Con - Vol)$ 对 outlier 更鲁棒，同时保持单调性。

### 3.4 Curiosity Reward

为防止多样性坍缩，引入 curiosity reward 鼓励模型探索低概率 reasoning path：

$$r_{\text{cur}} = d(s_i, s_{i+1}) - p_{\text{KL}}$$

其中 $p_{\text{KL}} = \ln[KL(P_{i+1}, \mathcal{U}) + 1]$ 用于惩罚极端低概率 token，防止 curiosity reward 过大。

### 3.5 总奖励与优化

$$r_{\text{covo}} = r_{\text{int}} + r_{\text{cur}}$$

使用 Reinforce++ 算法进行策略优化（clipped surrogate objective + normalized advantage）。

### 3.6 理论分析

- 证明 majority voting reward 会导致 model collapse（Proposition 1）。
- CoVo 的优化目标等价于对 latent reasoning trajectory 进行 variational inference（Proposition 2）。
- 给出收敛上界 $T' \lesssim \mathcal{O}(\pi_{\theta(0)}(y^\gamma|x)^{-1})$（Proposition 3）。

---

## 4. 数据集

### 训练数据
- **Open-Reasoner-Zero** 训练集（仅使用 instruction，不使用任何 label）。

### 评估数据
| 类别 | 数据集 |
|------|--------|
| Mathematical reasoning | GSM8K, MATH-500, Olympiad Bench, AMC-23 |
| Commonsense reasoning | MMLU-Pro, CommonsenseQA |
| Science reasoning | GPQA |

### 模型
- Llama3.2-3B-Instruct
- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct

---

## 5. 评估指标与主要结果

### 评估指标
- **Pass@1 accuracy**（sampling temperature = 0），使用 Math-Verify 判断正确性。
- **Reasoning Diversity**：使用 Sentence Transformers 编码 + UMAP 可视化。

### 主要结果（Table 1 精选）

**Llama3.2-3B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 51.8 | 79.2 | 25.0 | 18.9 | 36.8 | 31.5 | 74.2 |
| TTRL (self-rewarding) | 51.0 | 79.5 | 25.0 | 19.6 | 36.4 | 31.7 | 74.1 |
| **CoVo** | **51.2** | **79.6** | **25.0** | **19.6** | **37.2** | **32.1** | **74.4** |

**Qwen2.5-3B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 67.4 | 88.7 | 45.0 | 29.4 | 43.2 | 31.3 | 75.5 |
| TTRL (self-rewarding) | 67.8 | 88.6 | 45.0 | 29.5 | 43.2 | 31.3 | 75.5 |
| **CoVo** | **68.2** | **88.7** | **47.5** | **29.6** | **43.5** | **31.7** | **75.7** |

**Qwen2.5-7B-Instruct:**

| Method | MATH-500 | GSM8K | AMC-23 | Olympiad | MMLU-Pro | GPQA | CommonsenseQA |
|--------|----------|-------|--------|----------|----------|------|---------------|
| GRPO (supervised) | 78.2 | 92.3 | 57.5 | 41.3 | 57.1 | 36.6 | 82.9 |
| Reinforce++ (supervised) | 78.2 | 92.6 | 60.0 | 40.9 | 57.2 | 36.6 | 82.5 |
| **CoVo** | **78.4** | **92.5** | **60.0** | **40.8** | **57.0** | **36.8** | **82.8** |

### 关键发现

1. **可比甚至超越 supervised RL**：CoVo 在不使用任何外部标签的情况下，在大多数 benchmark 上达到或超过 GRPO、RLOO、Reinforce++ 等使用 ground-truth reward 的方法。
2. **跨领域泛化**：训练数据仅来自数学领域，但在 commonsense（MMLU-Pro, CommonsenseQA）和 science（GPQA）benchmark 上同样展现出可比的性能增益。
3. **推理多样性更优**：相比 GRPO 的集中式 reasoning path 分布，CoVo 生成的 reasoning path 更分散多样（Figure 3）。
4. **Reward 稳定性**：CoVo 的 reward accuracy 在训练过程中持续保持较高水平，不像 majority voting 方法随训练推进出现 reward hacking 导致的退化（Figure 5）。
5. **Ablation**：vectorial aggregation $r_{\text{int}}^V$ + curiosity reward $r_{\text{cur}}$ 的组合在大多数任务上表现最优；vectorial aggregation 比 linear aggregation 在数学和科学推理上更鲁棒。
