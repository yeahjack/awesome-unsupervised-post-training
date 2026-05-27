# GENIUS: A Generalizable and Purely Unsupervised Self-Training Framework for Advanced Reasoning

> **加入 Survey 时间：** 2026-03-11

> **Method:** GENIUS | **Carrier:** Direct Opt. | **Regime:** Training-time | **Level:** Traj.
>
> arXiv 2504.08672 — Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu (Shanghai AI Lab, Xi'an Jiaotong University, HKU, PKU, HKUST)

| 属性 | 值 |
|---|---|
| Method | GENIUS |
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
| 证据状态 Evidence Status | note-explicit |
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

GENIUS 属于 **Family III — Self-Generated Target Bootstrapping**，子类为 **reasoning / plan / curriculum synthesis**。

核心依据：GENIUS 是一个**完全无监督**的 self-training 框架，不依赖任何外部 ground-truth answer、reward model 或人工标注。其核心流程为：(1) 对每个 unsupervised general query，模型通过 **stepwise foresight re-sampling** 逐步搜索更优的 reasoning sequence，利用 rollout 模拟未来步骤并以 averaged log probability 计算 foresight score 作为 step value 的近似；(2) 基于 foresight score 分布进行 sampling（exploration）和 re-sampling（exploitation），构建 step-level preference pair；(3) 以自生成的 preference pair 通过 advantage-calibrated optimization (ACO) loss 优化策略模型。整个过程中的训练 target（即 positive/negative trajectory pair 及其 advantage）完全由模型自身合成，属于典型的 self-generated-target bootstrapping。

---

## 2. 解决的问题

- **外部监督的可扩展性瓶颈**：当前 post-training 技术（SFT、RL with outcome supervision、RL with reward model）依赖有标注数据或外部 reward model。有标注数据覆盖受限于特定领域（数学、代码），标注成本高；reward model 的训练需要大量人工标注且易诱发 reward hacking。
- **现有 self-rewarding 方法的局限性**：response-level self-reward（如 sequence confidence、self-consistency）缺乏细粒度 step-level 指导；step-level sampling 方法（如 auto-regressive 逐步采样）天然存在 **short-sighted** 问题，无法全局感知 reasoning 目标；MCTS 风格的搜索方法虽有全局视野但 backtracking 计算代价极高。
- **目标：仅使用通用 unsupervised query 提升 LLM 的通用推理能力**，不需要任何外部信号（Fig. 1(c)），从而释放海量通用数据的 reasoning scaling 潜力。

---

## 3. 方法介绍

### 3.1 总体框架

GENIUS 仅需一组 unsupervised NL query 作为输入，配合 policy LLM $\pi_\theta$ 完成 self-training。核心流程分两大目标：

1. **Synthesize & Reward**（§2.2）：通过 stepwise foresight re-sampling 生成高质量 preference pair。
2. **Optimize**（§2.3）：通过 advantage-calibrated optimization (ACO) loss 对模型进行 robust 训练。

> **对应论文 Figure 2**：整体框架图展示了 Genius 对每个 query 执行 K 步的 sampling 和 rewarding 循环（Step 1-4），收集高质量 response sequence 作为训练数据（Step 5），再用 ACO loss 训练模型（Step 6）。

### 3.2 Stepwise Foresight Re-Sampling

采用 beam search 策略，beam size 为 $M$，在每个 timestamp $k$ 执行以下操作：

#### (a) Step Rollouts with Foresight

在 timestamp $k-1$ 保留 $M$ 条 preceding path $a_{<k}$，对每条 beam rollout $N$ 个 candidate step $a_k$，共产生 $M \times N$ 个候选步骤。对每个候选步骤进行 **foresight**——模拟未来步骤 $a'_{>k}$，得到完整 response $T = (a_{<k}, a_k, a'_{>k})$，并以 remaining steps 的 averaged log probability 作为 foresight score $f_k$：

$$a'_{>k}, f_k \sim \pi_\theta(\cdot | a_{<k}; a_k)$$

将 $M \times N$ 个 foresight score 归一化为分布 $F_k$：

$$F_k(i) = \frac{\exp(f_k^{(i)} / \tau)}{\sum_j \exp(f_k^{(j)} / \tau)}$$

其中 $\tau$ 为 temperature 参数。

#### (b) Re-Sampling for Exploration

从分布 $F_k$ 中 **sampling** 选出 $M$ 个 step 用于 exploration（保持 beam 搜索继续）：

$$\{a_k^{(m)}\}_{m=1}^M \sim \text{Categorical}(F_k)$$

每个被选中步骤的 Q value 定义为其 foresight score：$Q_k^{(m)} := f_k^{(m)}$。

#### (c) Re-Sampling for Exploitation

在每个 timestamp $k$，还执行 **re-sampling** 构建 preference pair：
- **Positive**：选取 foresight score 最高的 response $T_k^w$，对应分数 $f_k^w$。
- **Negative**：从 $F_k / f_k^w$（去除 positive 后的分布）中 re-sample 得到 $T_k^l$，对应分数 $f_k^l$。

#### (d) Advantage Computation

由于各 step 来自不同 beam，仅用 foresight score 不足以评估 step 质量，因此引入 advantage value：

$$A_k^w = f_k^w - Q_{k-1}^w, \quad A_k^l = f_k^l - Q_{k-1}^l$$

最终每步产生一个五元组训练样本：$(x, T_k^w, A_k^w, T_k^l, A_k^l)$。

### 3.3 Advantage-Calibrated Optimization (ACO)

#### Self-Reward 建模

沿用 DPO 思路，以 policy LLM 作为 implicit reward model，self-reward 函数为：

$$\phi(x, T) = \beta \log \frac{\pi_\theta(T|x)}{\pi_{\text{ref}}(T|x)}$$

#### ACO Loss

无监督设定下基于 foresight score 采样的 preference pair 不可避免地包含噪声。为提升 robustness，ACO 对 negative sample 的 self-reward 添加 relaxation term $w(x, A)$：

$$\phi^l(x, T^l) = \beta \cdot w(x, A) \cdot \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}$$

$$w(x, A) = \text{clip}\left(\exp\left(-\frac{A^l - A^w}{\alpha}\right), 1\right)$$

- 当 $A^l - A^w \leq 0$（negative 确实较差，**Normal Region**）：$w = 1$，正常惩罚。
- 当 $A^l - A^w > 0$（negative 实际上更好，**Calibration Region**）：$w < 1$，减轻惩罚。
- 超参数 $\alpha$ 控制衰减速率。

> **对应论文 Figure 3**：可视化了 calibration function $w(x, A)$ 随 $A^l - A^w$ 变化的曲线，不同 $\alpha$ 值（1, 2, 4, 8, $\infty$）对应不同的衰减程度。

最终 ACO loss：

$$\mathcal{L}_{\text{ACO}} = -\mathbb{E}_{(x, T^w, T^l) \sim \mathcal{D}} \log \sigma\left(\beta \log \frac{\pi_\theta(T^w|x)}{\pi_{\text{ref}}(T^w|x)} - \beta \cdot \text{clip}\left(\exp\left(-\frac{A^l - A^w}{\alpha}\right), 1\right) \log \frac{\pi_\theta(T^l|x)}{\pi_{\text{ref}}(T^l|x)}\right)$$

### 3.4 Sampling 超参数

- Beam size $M = 2$，每 beam rollout 候选数 $N = 4$，步数 $K = 4$
- 总训练 pair 数：Magpie 100K，OpenHermes2.5 128K
- 推理加速：vLLM engine

---

## 4. 数据集

### 训练数据

| 语料 | 来源 | 规模 | 说明 |
|------|------|------|------|
| **Magpie** | Xu et al., 2024c | 25K queries | 通用 NL query，无标注 |
| **OpenHermes-2.5** | Teknium, 2023 | 32K queries | 通用 NL query，无标注 |

### 评估 Benchmark

| Benchmark | 领域 | 类型 |
|-----------|------|------|
| **GSM8K** | 数学推理 | 小学数学应用题 |
| **MATH** | 数学推理 | 高中竞赛数学 |
| **GPQA** | 数学推理 | 研究生水平 Q&A |
| **ReClor** | 逻辑推理 | 阅读理解+逻辑 |
| **LogiQA** | 逻辑推理 | 机器阅读理解 |
| **StrategyQA** | 通用推理 | 隐式策略推理 |
| **ARC-Challenge** | 通用推理 | AI2 推理挑战 |
| **AlpacaEval** | 通用（主观） | 指令遵循评估 |
| **WildBench** | 通用（主观） | 真实用户任务 |
| **Arena-Hard** | 通用（主观） | 人类偏好对齐 |
| **WikiBench** | 通用（客观） | 社区驱动 AI 评估 |
| **MMLU** | 通用（客观） | 多任务语言理解 |
| **MMLU-Pro** | 通用（客观） | 增强版 MMLU |
| **AIME 2024** | 竞赛级数学 | 美国数学邀请赛 |

---

## 5. 评估指标与主要结果

### 5.1 主实验（Table 1: 7 项推理 Benchmark 平均）

基座模型为 **LLaMA3.1-8B-Instruct**，CoT 基线平均 49.65%。

| 方法 | 监督 | GSM8K | MATH | ReClor | LogiQA | StrategyQA | GPQA | ARC-c | **Avg.** |
|------|------|-------|------|--------|--------|------------|------|-------|----------|
| LLaMA3.1-8B (CoT) | — | 70.28 | 30.52 | 49.40 | 33.33 | 58.91 | 26.56 | 78.33 | 49.65 |
| **Magpie 25K** | | | | | | | | | |
| SFT | 有 | 71.72 | 26.27 | 52.80 | 37.78 | 57.34 | 26.79 | 74.06 | 49.54 |
| SPIN | 有 | 74.91 | 31.49 | 57.40 | 40.09 | 71.35 | 29.91 | 83.96 | 55.59 |
| STaR | 无 | 72.86 | 29.32 | 46.40 | 35.94 | 33.36 | 20.31 | 67.24 | 43.63 |
| CoH | 无 | 74.37 | 32.29 | 56.20 | 38.56 | 69.08 | 28.13 | 82.51 | 54.45 |
| Self-Rewarding | 无 | 76.04 | 30.19 | 55.80 | 37.94 | 70.48 | 28.35 | 82.17 | 54.42 |
| ScPO | 无 | 71.11 | 30.99 | 55.00 | 40.40 | 59.87 | 28.57 | 78.92 | 52.12 |
| **GENIUS** | **无** | **78.32** | **34.64** | **58.80** | **40.86** | **72.53** | **30.35** | **84.04** | **57.08** |
| **OpenHermes 32K** | | | | | | | | | |
| SFT | 有 | 63.68 | 21.64 | 45.00 | 29.03 | 48.47 | 23.44 | 69.37 | 42.95 |
| SPIN | 有 | 63.61 | 24.74 | 54.00 | 35.33 | 59.00 | 28.57 | 71.76 | 48.14 |
| STaR | 无 | 75.51 | 29.47 | 43.60 | 34.87 | 19.34 | 22.99 | 68.43 | 42.03 |
| CoH | 无 | 74.29 | 31.22 | 54.80 | 38.40 | 69.91 | 29.69 | 81.48 | 54.26 |
| Self-Rewarding | 无 | 73.92 | 29.99 | 56.00 | 39.78 | 67.55 | 30.13 | 81.66 | 54.15 |
| ScPO | 无 | 73.54 | 31.27 | 54.80 | 41.01 | 58.65 | 28.79 | 79.52 | 52.51 |
| **GENIUS** | **无** | **75.82** | **34.42** | **57.60** | **41.63** | **70.79** | **34.82** | **83.19** | **56.90** |

**关键发现**：
- GENIUS 在 Magpie 25K 上将 LLaMA3.1-8B 的平均性能提升 **+7.43%**，在所有无监督基线中取得 SOTA，领先 Self-Rewarding >2%。
- 在 MATH 等高难度任务上，GENIUS 领先 Self-Rewarding **>4%**。
- 基于 RL 优化的 self-training 方法整体优于 SFT 类方法（SFT、STaR）。

### 5.2 通用领域稳定性（Table 2）

| Benchmark | LLaMA3.1-8B | GENIUS (Magpie) | GENIUS (OpenHermes) |
|-----------|-------------|-----------------|---------------------|
| AlpacaEval | 24.60 | 26.96 | 25.47 |
| WildBench | -1.11 | 2.68 | 1.44 |
| Arena-Hard | 30.31 | **50.00** | **50.00** |
| WikiBench | 27.65 | 28.75 | 27.00 |
| MMLU | 71.14 | 71.86 | 72.21 |
| MMLU-Pro | 48.62 | 48.44 | 49.19 |

GENIUS 在通用 benchmark 上保持甚至略有提升，Arena-Hard 跃升近 20 个百分点，表明与 human preference 的 alignment 显著增强，同时避免 catastrophic forgetting。

### 5.3 跨模型泛化（Figure 4）

| 基座模型 | GENIUS Avg. Gain | 最强基线 Gain |
|----------|------------------|---------------|
| Qwen2.5-3B-Instruct | **+3.52%** | CoH +1.83% |
| Qwen2.5-7B-Instruct | **+2.16%** | ScPO +0.81% |

GENIUS 在 Qwen2.5 系列上同样持续领先，但增益小于 LLaMA3.1（作者推测 Qwen2.5-Instruct 已经过充分 post-training）。

### 5.4 竞赛级任务（Figure 5: AIME 2024）

在 LLaMA3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 上，GENIUS 均提升 Pass@1 **+6.67%**，验证了框架在高难度场景下的可扩展性。

### 5.5 Ablation Studies

#### Sampling 策略消融（Table 3）

| Variant | Magpie Avg. | $\Delta$ | OpenHermes Avg. | $\Delta$ |
|---------|------------|----------|-----------------|----------|
| GENIUS (full) | 57.08 | — | 56.90 | — |
| w/o foresight | 53.91 | -3.17 | 53.65 | -3.25 |
| w/o sampling (greedy) | 52.98 | -4.10 | 53.80 | -3.10 |

- 去除 foresight 导致约 3% 下降，证明 look-ahead 模拟缓解了 auto-regressive 的 short-sightedness。
- 将 sampling 替换为 greedy selection 也显著下降，验证了 re-sampling 策略在 exploration-exploitation 之间的平衡作用。

#### Optimization 方法消融（Table 4）

| Loss | Magpie Avg. | OpenHermes Avg. |
|------|------------|-----------------|
| **ACO** | **57.08** | **56.90** |
| DPO | 55.51 | 55.73 |
| SimPO | 50.42 | 50.87 |
| IPO | 52.31 | 52.20 |
| ROPO | 55.30 | 55.25 |
| SFT | 44.63 | 49.70 |

ACO 在所有 loss function 中表现最佳，显著优于 DPO (+1.2~1.6%) 和 robustness 优化方法 ROPO (+1.6~1.8%)，SFT 表现最差。

### 5.6 Post-Training Scaling Law（Figure 6）

> **对应论文 Figure 6**：展示了不同方法在 LLaMA3.1-8B-Instruct 上随 training step 增加的 scaling curve。GENIUS 的性能提升曲线平滑上升且远未饱和，而 CoH、Self-Rewarding、ScPO 在更多训练步后趋于平稳甚至下降。

这表明 GENIUS 在通用数据上的 self-training 具有良好的 scalability 潜力，随着数据和计算规模增加可进一步提升 reasoning 能力。
