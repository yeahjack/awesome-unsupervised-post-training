# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

> **加入 Survey 时间：** 2026-03-11

**Paper:** Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
**arXiv:** 2403.09629
**Venue:** arXiv 2024 (March 2024)
**Authors:** Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
**Affiliations:** Stanford University, Notbad AI Inc

| 属性 | 值 |
|---|---|
| Method | Quiet-STaR |
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

**Family III — Self-Generated Target Bootstrapping, 子类: rationale / critique / latent-thought self-training**

Quiet-STaR 属于 Unsupervised Post-Training (UPT)，具体归入 Family III 的 latent-thought self-training 子类，理由如下：

- **无 ground truth 监督：** 训练过程不使用任何标注数据集的标签（无 QA pair、无人工 rationale），仅在通用互联网文本语料（OpenWebMath、C4）上进行 continued pretraining。模型的学习信号完全来自对未来文本的预测能力（language modeling objective），无需外部 reward model 或人工偏好标注。
- **Latent thought self-training：** 模型在每个 token 位置自行生成隐式 rationale（latent thought），这些 thought 不出现在最终输出中（"quietly"），而是作为内部推理过程指导下一个 token 的预测。模型通过 REINFORCE 算法自我评估哪些 thought 有助于预测未来文本，并基于此自我训练——这正是 self-generated-target bootstrapping 的核心范式：模型自身生成训练目标（thought），再用这些目标来提升自身。
- **Carrier — Direct Opt.：** Quiet-STaR 直接优化语言模型参数（包括 LM 权重、`<|startofthought|>` / `<|endofthought|>` meta-token embedding、mixing head），不使用独立的 reward model 或 verifier，奖励信号直接来自语言建模损失的差值。
- **Level — Traj.：** 优化的单位是完整的 thought trajectory（一段多 token 的 rationale 序列），而非单个 token 或语义级别的判断。REINFORCE 以整个 thought 序列的质量（对后续文本预测的帮助程度）作为 reward。

与 STaR 相比，Quiet-STaR 从特定 QA 任务推广到任意文本（unsupervised），从显式 reasoning 推广到隐式 latent thought，使其成为一种通用的无监督自训练范式。

---

## 2. 解决的问题

### 核心动机

人在写作和对话时常常"先想再说"——推理（reasoning）隐含在几乎所有文本中，例如数学证明中未写出的推导步骤、对话中未显式表达的 theory of mind。然而，现有 reasoning 方法（如 chain-of-thought prompting、STaR）都局限于特定 QA 数据集，依赖精心策划的标注数据。

### 现有方法的局限

1. **STaR（Self-Taught Reasoner）：** 在 QA 数据集上训练模型生成 rationale，仅保留导致正确答案的 rationale 进行训练。但这严重依赖 ground truth 答案，且受限于特定任务类型。
2. **人工标注 reasoning trace：** 昂贵、不可扩展，且是 off-policy 的（标注者的推理分布与模型自身不匹配）。
3. **Chain-of-thought prompting：** 仅在推理时使用，不改变模型参数，且需要显式 prompt 触发。
4. **Pause tokens：** 单 token pause 提供的思考空间有限，Goyal et al. (2023) 发现 pause token fine-tuning 效果有限且额外 token 反而损害性能。

### Quiet-STaR 要解决的问题

**如何让语言模型在无任何 ground truth 的通用文本上学习生成隐式 rationale（latent thought），从而全面提升推理能力？** 具体需解决三个关键挑战：
1. 为每个 token 生成 continuation 的计算代价极高；
2. 模型初始不知道如何生成或使用 internal thought；
3. 评估 thought 质量不应仅依赖紧邻的下一个 token，而需考虑多个未来 token。

---

## 3. 方法介绍

Quiet-STaR 的核心思路是将 STaR 从特定 QA 任务推广到通用文本：在每个 token 后生成 rationale 来帮助预测后续文本，用 REINFORCE 奖励有用的 thought。算法分为三个主要步骤：**Think → Talk → Learn**。

### 3.1 总体框架（Algorithm 1）

**输入：** 预训练语言模型 $\theta_0$，训练步数 num_steps，序列长度 $l$，thought 长度 $t$，学习率 $\alpha$，batch size $b$，每个 token 的 thought 数量 $n_{\text{thoughts}}$，监督的 ground truth token 数量 $n_{\text{true}}$。

**输出：** 能够生成 rationale 以预测未来文本的语言模型 $\theta$。

每一步训练：
1. 采样长度为 $l$ 的 batch 序列 $X$
2. 对所有 token 位置并行生成 thought
3. 混合有/无 thought 的预测
4. 用 REINFORCE 更新参数

### 3.2 Step 1: Think — 并行 Rationale 生成

**问题：** 朴素地为每个 token 单独做 forward pass 生成 thought 计算不可行。

**解决方案 — Parallel Generation Algorithm：** 论文提出一种基于 attention mask 的并行采样算法（Figure 3）。核心观察是：语言模型的一次 inference pass 天然为所有输入 token 产生下一个 token 的分布。通过构造对角线 attention mask，使得每个 thought token 只能 attend to：
- 自身
- 同一 thought 内的前序 thought token
- 对应的前文文本 token

这样各位置的 thought 互不干扰，可以完全并行生成。每次 inference call 为所有文本 token 各生成一个额外 thought token，重复 $t$ 次即可生成长度为 $t$ 的 thought。

**Meta-tokens：** 引入可学习的特殊 token `<|startofthought|>` 和 `<|endofthought|>` 标记 thought 的边界：
- `<|startofthought|>` 使模型进入"思考模式"
- `<|endofthought|>` 通知模型思考结束、开始预测

初始化策略：两个 meta-token 的 embedding 初始化为 em dash "—" 的 embedding（文本中 em dash 常表示停顿或思考），利用模型已有知识。为加速收敛，对这两个 embedding 的梯度施加超参数权重（1e2）。

### 3.3 Step 2: Talk — Mixing Head

**问题：** 预训练模型初始时生成的 thought 是 out-of-distribution 的，直接使用会损害语言建模性能。

**解决方案 — Mixing（Residual）Head：** 引入一个浅层 MLP（三层，ReLU 激活），输入为 end-of-thought token 的 hidden state 与原始文本 token 的 hidden state 的拼接，输出一个标量权重 $w$，用于混合有/无 thought 的 next-token logits：

$$\log p^{\text{talk}}_j = w_j \cdot \log p^{\text{init}}_j + (1 - w_j) \cdot \log p^{\text{thought}}_j$$

这种设计在训练初期让模型主要依赖 base prediction（$w \approx 1$），随着 thought 质量提升逐步增加 thought 的贡献，平滑过渡。

### 3.4 Step 3: Learn — REINFORCE 优化

**Reward 定义：** 对每个位置 $j$ 的 thought $T_j$，reward 为该 thought 的混合预测概率与同位置所有 thought 平均概率的差值：

$$r_j = \log p^{\text{talk}}_{j:j+n_{\text{true}}}(X_{j+1:j+n_{\text{true}}+1}) - \overline{\log p^{\text{talk}}_{j:j+n_{\text{true}}}(X_{j+1:j+n_{\text{true}}+1})}$$

通过生成多个 rationale 并以均值为 baseline 来降低方差（受 TRICE 启发）。

**REINFORCE Loss：**

$$\nabla_\theta \mathcal{L}^{\text{REINFORCE}}_j = -r_j \cdot \mathbf{1}[r_j > 0] \cdot \nabla_\theta \log p_\theta(T_j | [X_{:j}; \texttt{<|startofthought|>}])$$

仅保留正 reward 的梯度（丢弃负 reward），以提升训练稳定性。

**Non-myopic Scoring & Teacher Forcing（Figure 4）：** 由于不是每个 token 都需要 thought，评估 thought 质量不应仅看紧邻的下一个 token。论文通过 teacher forcing 技术，将后续 $n_{\text{true}}$ 个 ground truth token 注入以计算多步预测概率，使 reward 依赖更长范围的语义内容而非单一 token。可视化如 Figure 4 所示：实线表示 LM 计算，虚线表示 teacher forcing 插入的 token，mixer 块表示 mixing head。

**总 Loss：**

$$\nabla_\theta \mathcal{L}_j = \nabla_\theta \mathcal{L}^{\text{NLL}}_j + \nabla_\theta \mathcal{L}^{\text{REINFORCE}}_j$$

其中 $\mathcal{L}^{\text{NLL}}_j$ 为标准 next-token prediction loss（通过 mixing head 后的概率），确保 base LM head 也接收语言建模信号。

### 3.5 图表描述

**Figure 1（算法总览）：** 展示 Quiet-STaR 在训练时对单个 thought 的三步流程。(1) Think：在所有 token 后并行生成 thought；(2) Talk：将有/无 thought 的 next-token 预测通过 mixing head 混合；(3) Learn：用 REINFORCE 增加帮助预测的 thought 的概率，丢弃损害预测的 thought。

**Figure 3（并行生成）：** 展示 attention mask 的结构。Base text token 为 $a, b, c, d$，各自生成的 thought token 通过对角 attention mask 实现并行推理，互不影响。

**Figure 4（Forward Pass & Teacher Forcing）：** 展示单次 forward pass 的完整流程。包含 thought 生成、end-of-thought token、teacher forcing 插入的 ground truth token，以及 mixing head 的输出。以预测 3 个 token ahead 为例。

### 3.6 实现细节

- **基础模型：** Mistral 7B (base)
- **Optimizer：** AdamW，warmup 20 steps，learning rate $1 \times 10^{-6}$，weight decay 0.001，batch size 8
- **Meta-token 梯度权重：** 1e2；policy 权重：1e6
- **训练时 temperature：** $T = 1$（采样）；推理时 thought 使用 greedy decoding
- **REINFORCE temperature：** $T = 3$（importance sampling）
- **序列长度：** 每个样本随机取 256 token span
- **硬件：** 单节点 8× H100 80GB GPU

---

## 4. 数据集

### 训练数据（无标注，通用文本语料）

| 数据集 | 领域 | 说明 |
|-------|------|------|
| **OpenWebMath** (Paster et al., 2023) | 数学/技术网页 | 偏技术性网页爬取，reasoning 密度较高，为主要训练集 |
| **C4** (Raffel et al., 2020) | 通用网页文本 | Colossal Clean Crawled Corpus，广泛使用的 LM 预训练语料，文本更多样 |

### 评估数据集（zero-shot，无 fine-tuning）

| 数据集 | 领域 | 说明 |
|-------|------|------|
| **GSM8K** (Cobbe et al., 2021) | 数学推理 | 小学数学应用题，评估 zero-shot direct reasoning 能力 |
| **CommonsenseQA** (Talmor et al., 2018) | 常识推理 | 多选题，评估常识知识推理能力 |

---

## 5. 评估指标与主要结果

**评估指标：** Zero-shot accuracy (%)，直接计算正确答案 token 的条件概率（对多选题计算 A–E 对应 logits 上的 accuracy），不进行任何 task-specific fine-tuning。

### 5.1 主要 Zero-shot 结果

| 设定 | GSM8K | CommonsenseQA |
|------|-------|---------------|
| Mistral 7B (baseline) | 5.9% | 36.3% |
| + Quiet-STaR (OpenWebMath, 最佳配置) | **10.9%** | **47.2%** |
| + Quiet-STaR (C4) | 8.1% | 42.6% |
| + 无 thought baseline（相同数据训练） | ~6% | ~36% |

**提升幅度：** GSM8K +5.0%，CommonsenseQA +10.9%（OpenWebMath 训练）。

### 5.2 Thought 长度对性能的影响（Figure 2）

论文系统评估了不同 thought token 数量和 ahead token 数量的组合：

| # Thought Tokens, # Ahead Tokens | GSM8K (最佳) | CommonsenseQA (最佳) |
|----------------------------------|-------------|---------------------|
| 8, 4 | ~8% | ~40% |
| 10, 4 | ~9% | ~42% |
| 12, 4 | ~9.5% | ~44% |
| 16, 8 | ~10% | ~45% |
| 24, 12 | **~10.9%** | **~47.2%** |

**关键发现：** 更长的 thought 和更多的 ahead token 一致地带来更好的性能，表明多 token rationale 比 single-token pause 更有效。

### 5.3 Quiet-STaR + Chain-of-Thought（Figure 5）

Quiet-STaR 与 zero-shot CoT 互补（"Let's think step by step." prompt），内部 rationale 辅助外部 CoT 生成更结构化的推理：

| 方法 | GSM8K (maj@8, $T=0.7$) |
|------|------------------------|
| Baseline (CoT) | 40.6% |
| Quiet-STaR + CoT | **47.7%** |

（在 128 个 GSM8K 测试样本上评估）

### 5.4 Ablation：多 Thought 和多 Token Ahead 的影响

- 使用多个 thought per token（2–4 个）比仅用 base 作为 baseline 效果更好（GSM8K ~+0.5%，CommonsenseQA ~+3%），但超过 2 个后边际收益递减（~0.1–0.3%）。
- 预测超过 1 个 token ahead 显著帮助（GSM8K +0.3%，CommonsenseQA +3.1%），但超过 2 个 ahead token 后收益不明显。定性观察发现更多 token-ahead 使 rationale 更连贯。

### 5.5 改进分布（Appendix Figure 7 & 8）

- 大多数 token 的预测改善幅度很小，但难预测 token 获得不成比例的大幅改善——符合直觉：大部分网页文本不需要深度推理，但 thought 对困难 token 帮助显著。
- 定性分析表明 thought 尤其有助于需要回忆相关信息的 token（如适用定理的名称、证明的下一步等）。

### 5.6 生成的 Thought 示例

论文展示了训练后模型生成的有用 thought 示例：
- **化学推理：** 在预测关于 magnesium nitride 合成的文本时，thought 回忆了"从 magnesium 出发生产 magnesium nitride"这一事实，帮助预测下一步涉及加热 magnesium。
- **数学证明：** 在证明 $A = B$ 需要证明 $A \subseteq B$ 和 $B \subseteq A$ 的文本中，thought 生成了"in some sense - to be the more difficult"，帮助预测后续 "trickiest for students"。
- **常识推理：** 在阅读 CommonsenseQA 问题时，thought 提前生成了选项结构，帮助预测问题的后续部分。
