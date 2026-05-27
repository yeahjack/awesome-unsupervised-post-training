# DEBATE, TRAIN, EVOLVE: Self-Evolution of Language Model Reasoning

> **加入 Survey 时间：** 2026-03-11

> **作者**: Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang  
> **机构**: Virginia Tech, Department of Computer Science  
> **链接**: arXiv:2505.15734v2 [cs.CL] 30 Sep 2025  
> **代码**: https://github.com/ctrl-gaurav/Debate-Train-Evolve

| 属性 | 值 |
|---|---|
| Method | DTE |
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

DTE 属于 **Family III — Self-Generated Target Bootstrapping** 中的 **reasoning / plan / curriculum synthesis** 子类。理由如下：

- **无需外部 ground truth**：DTE 的核心设计原则是 ground-truth-free training，训练信号完全来自模型自身生成的 multi-agent debate traces，不依赖人类标注或外部教师模型。这是典型的 unsupervised post-training 特征。
- **Synthetic target 生成**：通过 multi-agent debate 机制，多个 agent（均由同一模型实例化）进行辩论并收敛到 consensus answer，构成 synthetic training target。Debate traces 中提取的 consolidated rationale 作为合成的推理轨迹（trajectory），直接用于训练。
- **Direct optimization via GRPO**：训练阶段使用 Group Relative Policy Optimization (GRPO) 直接优化单一模型的 policy，以 debate consensus 和 format/length reward 为 shaped reward，无需 value network。Carrier 为 Direct Opt.，Level 为 Traj.（整条推理轨迹作为优化单元）。
- **自我进化的 bootstrapping 循环**：Evolved model 替换原始 agent 参与下一轮 debate，形成 iterative self-evolution，与 self-generated-target bootstrapping 的核心机制一致。

---

## 2. 解决的问题

### 2.1 核心研究问题

> **能否将 multi-agent debate (MAD) 的推理洞见蒸馏到单一模型中，使其在单次 forward pass 下获得接近甚至优于多 agent 集合体的推理能力？**

### 2.2 现有方法的局限性

1. **数据饱和**：LLM 的推理能力提升长期依赖于更大规模的训练数据，但数据供给正趋于饱和（data bottleneck）。
2. **单模型 self-evolution 的确认偏差**：现有 self-refine、self-instruct 等方法主要依赖单一模型或 teacher-student 配置的反馈，容易产生 confirmation bias，推理多样性不足。
3. **MAD 的推理时高开销**：Multi-agent debate 虽然能显著提升推理准确率，但要求同时运行多个模型实例处理每个 query，计算开销和延迟极高，不适合大规模部署。
4. **小模型 debate 质量退化**：标准 MAD prompting 在较小模型上暴露两个严重缺陷：
   - **Sycophancy（谄媚）**：agent 放弃正确答案，转而采纳错误但表述自信的 peer 答案，小模型上 sycophancy rate 超过 28%。
   - **Verbosity bias（冗长偏差）**：agent 倾向于采纳更长的 rationale，而不论其逻辑有效性。

---

## 3. 方法介绍

DTE 框架由三个阶段组成：**DEBATE → TRAIN → EVOLVE**，形成 iterative self-evolution 循环。

### 3.1 整体框架（Figure 1）

> **Figure 1 描述**：论文的核心框架图，分为三个部分。
> - **左侧（Debate）**：多个 agent 围绕一个 query 进行辩论，直到收敛为 consensus（绿色 ✓）或暴露错误路径（红色 ✗）。
> - **中间（Train）**：去除纯粹的辩论元素（如寒暄、重复），保留 high-quality reasoning traces 和 consensus answer，用于 GRPO fine-tune 单一 policy。
> - **右侧（Evolve）**：进化后的 agent 替换其先前版本，后续推理仅需一次 forward pass，但在数学、科学和常识 benchmark 上超越了原来的 agent 委员会。

### 3.2 REFLECT-CRITIQUE-REFINE (RCR) Prompting Strategy

为解决标准 MAD prompting 的 sycophancy 和 verbosity 问题，DTE 引入了 RCR 三阶段 prompting 策略：

1. **Reflect（反思）**：每个 agent $a_i$ 必须对自身推理 $r_i^{(t-1)}$ 生成 self-critique $c_i^{\text{self}}$，识别潜在错误。
2. **Critique（批评）**：agent 随后评估恰好两个 peer 的 rationale，生成批评 $\{c_i^j\}_{j \in P_i}$，其中 $|P_i| = 2$。固定 critique 配额防止无限制的冗长输出。
3. **Refine（修正）**：agent 更新其回答 $(y_i^{(t)}, r_i^{(t)})$，关键约束是：若 $y_i^{(t)} \neq y_i^{(t-1)}$，则 $r_i^{(t)}$ 必须至少包含一个在之前所有 agent 回答中未出现过的 novel reasoning step。这一约束强制 agent 在"复制"前先"思考"，有效降低 sycophancy。

**Debate 终止条件**：当所有 agent 的答案一致（consensus）或达到最大轮数 $T$ 时终止，最终答案通过 majority vote 确定。

**RCR 效果**（对应 Figure 3）：
> **Figure 3 描述**：条形图对比了 8 个模型在 GSM8K、GSM-Plus、ARC-Challenge 上三种设定（single model、Original MAD@3、RCR-MAD@3）的表现。RCR 在 GSM-Plus 上平均提升 +3.7 pts，将平均 sycophancy rate 从 0.28 降至 0.13，verbosity gap 缩小 43%。

### 3.3 训练阶段：GRPO with Debate-Derived Rewards

#### Debate Trace 提取与 Reward 设计

从 debate 中提取 consolidated rationale $R$，包含 (i) 多个 agent 共识出现的推理步骤，(ii) 引入 novel symbolic manipulation 的步骤。每个训练样本为 $(q, y^*, R)$。

Shaped reward function：

$$r(q, y) = w_{\text{ans}} \cdot \mathbb{1}[y = y^*] + w_{\text{fmt}} \cdot f_{\text{format}}(y) + w_{\text{len}} \cdot \exp(-|y| / \tau)$$

其中 $(w_{\text{ans}}, w_{\text{fmt}}, w_{\text{len}}) = (2.0, 0.5, 0.5)$，$\tau = 120$。三项分别奖励：答案正确性（exact string match）、格式合规性（XML 模板结构）、长度简洁性。

#### Group Relative Policy Optimization (GRPO)

对每个 query $q$，从当前 policy $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个 response $\{o_1, \ldots, o_G\}$，计算 group-relative advantage：

$$\hat{A}_{i,t} = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

其中 $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$，$\sigma_r$ 为标准差。优势在于：无需 value network，通过组内比较估计 advantage，减少内存开销。

GRPO loss：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\ell_{\text{clip}}(i, t) - \beta \cdot D_{\text{KL}}^{(i,t)}\right]$$

其中 clipping threshold $\epsilon = 0.2$，KL coefficient $\beta = 0.02$。Clipping 防止单步过大的 policy 更新，KL penalty 将 policy 锚定于参考模型以维持语言连贯性、防止 catastrophic forgetting。

#### 训练超参数

- Optimizer：AdamW，$\eta = 5 \times 10^{-6}$，weight decay $\lambda = 0.1$
- LoRA：rank $r = 128$，dropout $p = 0.05$，应用于 attention 和 MLP projection matrices
- Group size $G = 8$
- 每个 evolution epoch 处理 ~8K debate traces（~2M tokens）
- 硬件：A100-80GB GPU

### 3.4 Evolution through Iterative Training

完整 DTE 流程（Algorithm 2）：

1. **Debate Generation**：采样 query batch $Q_k$，使用 RCR-prompted MAD 生成 debate traces，形成训练集 $D_k = \{(q, y^*, R)\}$。
2. **Policy Update**：在 $D_k$ 上用 GRPO fine-tune $\pi_{\theta_{k-1}}$ 得到 $\pi_{\theta_k}$。
3. **Agent Replacement**：用 evolved policy 替换 debate ensemble 中的旧版本。
4. **迭代终止**：验证集 performance 停滞或达到最大迭代次数。

**小模型温度退火**：对 < 3B 参数的模型，在 Round 1 使用 $T = 0.7$，后续轮次线性衰减到 $T = 0.3$，以缓解 KL divergence 增长和 catastrophic forgetting。实验表明，将温度从 0.7 降至 0.3 可将 KL drift 减少 1/3，恢复高达 76% 的性能损失。

### 3.5 Agent 配置

Debate 阶段每个 agent 对每个 query 采样一次：
- 温度设置：$T = 1.0$（exploratory）或 $T = 0.0$（deterministic）
- Mixed-team：1 个 exploratory agent + 2 个 deterministic agent

---

## 4. 数据集

### 训练数据
- **GSM8K** (Cobbe et al., 2021)：小学数学应用题
- **GSM-Plus** (Li et al., 2024)：GSM8K 的对抗性变体，难度更高

### 评估 Benchmark（共 7 个）

| 数据集 | 领域 | 说明 |
|---|---|---|
| GSM8K | 数学 | 小学数学应用题 |
| GSM-Plus | 数学 | 对抗性数学问题，更难 |
| MATH | 数学 | 竞赛级数学 (Hendrycks et al., 2021) |
| ARC-Easy | 科学推理 | 较易科学选择题 |
| ARC-Challenge | 科学推理 | 较难科学选择题 (Clark et al., 2018) |
| GPQA Main | STEM | 研究生级 STEM 问题 (Rein et al., 2024) |
| CommonsenseQA | 常识推理 | 常识问答 (Talmor et al., 2019) |

### 实验模型
- **Qwen-2.5** 系列：0.5B, 1.5B, 3B, 7B, 14B
- **Llama** 系列：Llama-3.2-3B, Llama-3.1-8B
- **其他**（RCR 研究用）：Mistral-7B, Phi-mini, GPT-4o, GPT-4o-mini

---

## 5. 评估指标与主要结果

### 评估指标
- **Exact match accuracy**：GSM 风格数据集使用标准化后的 exact string match
- **MC-QA accuracy**：多选题数据集使用选项匹配准确率
- **Sycophancy rate**：debate 中 agent 从正确答案切换到错误答案的比率
- **[incorrect → correct] rate**：debate 中答案从错误转为正确的比率

### 5.1 主要结果（Table 1）

一轮 DTE 后的性能：

| 模型 | GSM-Plus 原始 | GSM-Plus DTE 进化 | 提升 |
|---|---|---|---|
| Qwen-2.5-1.5B | 42.00 | 55.92 | **+13.92** |
| Qwen-2.5-3B | 61.75 | 69.50 | +7.75 |
| Qwen-2.5-7B | 68.62 | 74.71 | +6.09 |
| Qwen-2.5-14B | 71.79 | 78.88 | +7.09 |
| Llama-3.2-3B | 45.67 | 53.79 | +8.12 |
| Llama-3.1-8B | 55.62 | 66.17 | **+10.55** |

**关键发现**：
- **GSM-Plus 平均增益 8.92%**，所有模型均获提升。
- Evolved single model 比 3-agent MAD 平均高出 +2.38 pts，实现了以单模型推理超越多 agent 集合体。
- 小模型（Qwen-1.5B）提升最大（+13.92 pts），因其 headroom 最大且 debate 提供了多样化 traces。
- ARC-Challenge 上大模型受益显著（Llama-8B +8.88 pts），小模型变化在 ±1 pt 以内。

### 5.2 Cross-Domain Generalization（Table 2）

在 GSM8K 上训练后，测试未见过的数据集：

| 模型 | GSM-Plus Δ | ARC-Easy Δ | ARC-Ch. Δ | CSQA Δ |
|---|---|---|---|---|
| Qwen-2.5-7B | +1.01 | +1.73 | +4.50 | +3.40 |
| Qwen-2.5-14B | +1.67 | +2.53 | +3.42 | +1.33 |
| Llama-3.1-8B | +8.13 | +3.91 | +6.74 | +1.10 |

**关键发现**：
- 平均 cross-domain 增益 +5.8 pts，表明 DTE 学习到的是通用推理启发式（如 numeric decomposition、unit tracking），而非数据集特异性 pattern。
- ≥7B 模型在任何 transfer task 上都不会损失超过 0.2 pts。
- 负迁移（negative transfer）仅限于最小模型（Qwen-1.5B）。

### 5.3 迭代进化分析（Figure 2）

> **Figure 2 描述**：折线图展示 5 个模型在 GSM8K 和 GSM-Plus 上经过 Original → Round 1 → Round 2 的准确率变化。Round 1 几乎所有模型都获益，但 Round 2 改善甚微甚至下降。

- **Round 1 捕获了几乎所有可用增益**。
- Round 2 的 mean forgetting $F_{\text{gt2}} = \max_{t<2}(\text{Acc}_t - \text{Acc}_2)$：≥7B 模型为 0.92 pts，< 3B 模型为 1.6 pts。
- 小模型更易受 catastrophic forgetting 影响。

### 5.4 关键消融实验

#### (1) Agent 数量（Figure 4）
> **Figure 4 描述**：四条线（Qwen 1.5B/3B/7B/14B）展示 agent 数从 1 到 7 时在 GSM8K、GSM-Plus、ARC-Easy、ARC-Challenge 上的准确率变化。
- **3 个 agent 即可捕获 85-95% 的最大增益**。
- 更难的任务（GSM-Plus）最优 agent 数略多（4-5）。
- 简单任务（ARC-Easy）在 2 个 agent 即饱和。

#### (2) 优化方法比较（Table 3）

| 模型 | Original | SFT | DPO | GRPO |
|---|---|---|---|---|
| Qwen-2.5-1.5B | 42.00 | 47.31 | 51.34 | **55.92** |
| Qwen-2.5-3B | 61.75 | 58.33 | 64.32 | **69.50** |
| Qwen-2.5-7B | 68.62 | 67.89 | 69.88 | **74.71** |

GRPO 在所有模型尺寸上一致优于 SFT 和 DPO，且其 KL divergence < 0.24，远低于 DPO 的 0.43，在提升 reward 的同时有效约束 policy drift。

#### (3) 数据选择策略（Table 4）

| 策略 | Qwen-1.5B | Qwen-3B | Qwen-7B |
|---|---|---|---|
| Random-2K | 44.82 | 58.10 | 69.71 |
| Debate-Only | 51.61 | 62.70 | 72.53 |
| All-Traces | **55.92** | **69.50** | **74.71** |

使用全部 trace 效果最佳，All-Traces 比 Debate-Only 平均高 4.43 pts，比 Random-2K 高 9.17 pts。小模型从 "round-0"（无争议的）easy examples 中获益尤大。

#### (4) 训练步数（Figure 5）
> **Figure 5 描述**：GSM-Plus 准确率随 GRPO 训练步数（2K-10K）变化的曲线。所有模型在约 8K 步后饱和，8K→10K 的平均提升仅 +0.32 pts。

#### (5) 温度退火与 Catastrophic Forgetting（Figure 6）
> **Figure 6 描述**：Qwen-1.5B 在不同 sampling temperature（1.0, 0.7, 0.4, 0.0）下 Round 1 和 Round 2 的准确率。
- $T = 1.0$ 时 Round 2 损失 2.0 pts（GSM8K），出现明显 catastrophic forgetting。
- $T = 0.4$ 时 Round 2 准确率在 Round 1 的 0.9 pts 以内。
- KL divergence：$T = 1.0$ → 0.37；$T = 0.4$ → 0.19；$T = 0.0$ → 0.11。

#### (6) Agent 多样性
- 当 agent 个体精度相当时，cross-family mixtures（如 Qwen + Llama）优于同质团队，架构多样性能带来互补推理路径。
- 当混合强弱模型时，debate 结果倾向于强模型——弱 agent 既不帮助也不严重损害。

### 5.5 局限性

1. Iterative fine-tuning 在 < 3B 参数模型上会导致 catastrophic forgetting，尚不能完全消除。
2. 框架假设初始 debate traces 具有合理质量；若 agent 初始性能较弱，效果可能退化。
3. 主要验证于结构化推理任务（数学、常识），在开放式文本生成或对话任务上的效果有待研究。
4. 虽比传统 MAD 部署高效，但训练成本仍高于标准 single-model fine-tuning。
