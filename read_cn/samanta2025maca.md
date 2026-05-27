# MACA: Self-Improvement of Language Models by Post-Training on Multi-Agent Debate

> **加入 Survey 时间：** 2026-03-11

**Paper:** Self-Improvement of Language Models by Post-Training on Multi-Agent Debate
**Authors:** Ankur Samanta, Akshayaa Magesh, Runzhe Wu, Ayush Jain, Youliang Yu, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani (Meta AI, Columbia University, Cornell Tech, Meta Superintelligence Labs)
**ArXiv:** 2025 (February 2, 2026)
**Code:** https://github.com/facebookresearch/maca
**Citation:** `samanta2025maca`

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| MACA | Pref. Opt. | training-time | Traj. |

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

**Family III — Self-Generated Target Bootstrapping（internally generated preference pairs）**

MACA 的核心监督信号完全来自模型自身的 multi-agent debate 输出。具体而言，多个模型克隆体（clones）在多轮辩论（multi-round debate）中交换推理过程（chain-of-thought），最终通过 majority voting 确定 consensus answer $\hat{a}(x)$。根据各 agent 的 final-round 回答是否与 majority 一致，将完整推理轨迹（trajectories）划分为 **consensus-supporting** $G^+(x)$ 和 **dissenting** $G^-(x)$ 两组，从而构造出 preference pairs。

这些 preference pairs 不依赖任何外部 ground-truth label 或人工标注——它们是模型通过内部协商（deliberation）自主合成的偏好目标。preference learning（MV-DPO / MV-KTO）在完整推理轨迹层面学习区分 majority 与 minority traces，使模型内化稳定共识推理与偏离推理之间的细微差异。因此归入 Family III 的 "Preference optimization → internally generated preference pairs" 子类。

---

## 2. 解决的问题

- **LLM 推理不一致性**：语言模型在探索性采样（high temperature sampling）下对同一问题产生矛盾的解答，缺乏将多样推理路径对齐至一致答案的内在机制。
- **Inference-time 方法的局限**：Self-consistency prompting 和 multi-agent debate 只在推理时利用多次采样/多 agent 协作提升准确性，不更新模型参数，无法持久改善模型能力，且需额外推理计算。
- **Single-round majority vote 信号不足**：先前工作（TTRL、ScPO）仅使用单轮独立采样的 majority vote 信号做 RL 训练；当小模型 self-consistency 很差时，noisy aggregation 反而放大错误。
- **Ground-truth 标注不可扩展**：外部标签获取成本高，无法扩展到新领域或大规模数据。

MACA 通过 multi-agent debate 产生比 single-round majority vote 更丰富的共识信号，并用 RL post-training 将这些信号内化，同时提升单 agent 推理质量、multi-agent 协作能力和 self-consistency。

---

## 3. 方法介绍

### 3.1 Self-Consistency 的形式化

给定 prompt $x$，模型 $\pi_\theta$ 在温度 $\tau$ 下采样产生答案分布 $P_{\theta,\tau}(a|x)$。定义 majority probability：

$$S^+_{\theta,\tau}(x) = P_{\theta,\tau}(a^*_{\theta,\tau}(x)|x)$$

其中 $a^* = \arg\max_a P_{\theta,\tau}(a|x)$ 为 majority answer。self-consistent 的模型应在高温下仍保持较高的 $S^+$。

实际中通过 $t$ 次独立采样估计 sampling consistency：

$$s^{\theta,\tau}_t(x) = \frac{1}{t}\sum_{i=1}^t \mathbf{1}[a_i(x) = \hat{a}(x)]$$

对于 multi-agent debate，类似定义 agreement metric $d^{\theta,\tau}_M(x)$。

### 3.2 Multi-Agent Debate 过程

1. **Clone**：将同一 base LM 复制为 $M=3$ 个 agents
2. **Initial round**：每个 agent 独立生成初始推理轨迹 $y_m^{(1)} \sim \pi_{\theta_m}(\cdot|x)$
3. **Debate rounds**（$R=2$ 轮共 1 轮 deliberation）：每个 agent 看到所有其他 agents 上一轮的推理过程，以此为条件生成更新回答，即 $x_m^{(r)} = [x; \{y_j^{(r-1)}\}_{j\neq m}]$
4. **Consensus extraction**：提取最终轮答案 $a_m = A(y_m^{(R)})$，通过 majority vote 确定 $\hat{a}(x)$

### 3.3 Preference Data 构造

根据 majority consensus 将最终轮轨迹分组：
- **Consensus-supporting**：$G^+(x) = \{y \in Y(x) : A(y) = \hat{a}(x)\}$（majority traces，视为 preferred）
- **Dissenting**：$G^-(x) = \{y \in Y(x) : A(y) \neq \hat{a}(x)\}$（minority traces，视为 not preferred）

构造 post-training dataset：$D_{\text{post}} = \{(x, \hat{a}(x), G^+(x), G^-(x))\}_{x \in D}$

### 3.4 四种 Post-Training 目标

**MV-SFT**：直接模仿 consensus-supporting 轨迹
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{y^+ \in G^+(x)}[\log \pi_\theta(y^+|x)]$$

**MV-GRPO**：online sampling + consensus-based reward $r_x(y) = \mathbf{1}[A(y) = \hat{a}(x)]$，带 group-normalized advantage
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{y \sim \pi_\theta}\left[\tilde{A}_x(y) \sum_t \log \pi_\theta(y_t|x, y_{<t})\right] + \lambda \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**MV-DPO**（最佳方法）：使用 debate-derived preference pairs 做标准 DPO
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{x \sim D}\mathbb{E}_{(y^+, y^-) \in G^+ \times G^-}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \beta \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)\right]$$

**MV-KTO**：unpaired 形式，分别处理 positive/negative 样本，适合 majority trajectories 占主导的不平衡场景

### 3.5 关键设计选择

- **4-bit quantization + QLoRA**：在单节点多 GPU 集群上训练 3 个并发 agents
- **Token limit = 256**：测试模型在紧凑推理预算下的表现（改善可迁移到 512 token 测试）
- **Temperature $\tau=1.0$**：高温度探索性采样
- **Peer context conditioning**：DPO/KTO 训练时将 peer agents 的 chains-of-thought 作为 prompt context，教模型有效利用辩论上下文
- **Iterative training**：支持多轮 debate→post-training 迭代，虽然主要收益在第 1 轮迭代（It-0→It-1），后续迭代仍有 1-3pp 增量

### 3.6 与 TTRL/ScPO 的关系

TTRL 和 ScPO 是 MACA 的特殊情况——当将 multi-agent debate 参数退化为单轮 majority vote 时：
- TTRL ≈ single-round R0 MV-GRPO
- ScPO ≈ single-round R0 MV-DPO（不含 weighted loss）

MACA 的优势在于引入多轮 deliberation traces 作为 conditioning context，让模型学习的不仅是 aggregate 正确输出，而是内化产生共识的 deliberative process。

---

## 4. 数据集

| 领域 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | MATH | 12,500 高中数学题（代数、几何、组合、数论），多步推理 |
| 数学推理 | GSM8K | 8,500 小学数学应用题，算术与逻辑推理 |
| 数学推理 | MathQA | 37,000+ 量化推理 QA，自然语言→数学表达式 |
| 数学推理 | SVAMP | 算术应用题（重写防 artifact），仅用于测试 |
| 数学推理 | AMC 2023 | 美国数学竞赛题，仅用于测试（40 题） |
| 科学推理 | GPQA | 448 研究生级物理/化学/生物多选题，仅用于测试 |
| 常识推理 | CommonsenseQA (CSQA) | 12,247 常识多选题，仅用于测试 |

**训练/测试划分**：MATH、GSM8K、MathQA 各取 1500/500/500 train/valid/test split；SVAMP (300)、GPQA (448)、CSQA (500)、AMC (40) 仅用作 OOD 测试。

---

## 5. 评估指标与主要结果

### 5.1 评估指标

- **Single-agent accuracy**：单 agent greedy decoding 或 Pass@t / MV@t（采样 $t$ 个轨迹取 majority vote）
- **Multi-agent debate accuracy**：3 agents 经 2 轮辩论后 final-round majority vote
- **Self-consistency** $s_t^{\theta,\tau}(x)$：$t$ 次采样中与 majority answer 一致的比例（$\tau=1.0$，$t=20$）
- **Agreement metric** $d_M^{\theta,\tau}(x)$：multi-agent debate 中 agent 一致程度

### 5.2 主要结果

#### Multi-agent debate setting（Table 1）

| 模型 | 数据集 | Base Debate | MV-SFT | MV-GRPO | MV-KTO | MV-DPO | Best Δ |
|------|--------|------------|---------|----------|---------|---------|--------|
| Qwen-2B | MATH | 32.40 | 37.07 | 39.00 | **46.47** | 42.60 | **+14.07** |
| Qwen-2B | GSM8K | 49.60 | 50.53 | 54.13 | **63.07** | 58.47 | **+13.47** |
| Qwen-2B | MathQA | 24.20 | 26.27 | 29.93 | **32.60** | 28.33 | +9.13 |
| Llama-3B | MATH | 37.80 | 35.33 | 48.33 | **52.93** | 51.93 | **+15.27** |
| Llama-3B | MathQA | 21.60 | 40.07 | 48.73 | 64.00 | 63.13 | **+42.73** |
| Llama-8B | MATH | 32.80 | 34.13 | 45.93 | 53.93 | **59.67** | **+26.87** |
| Llama-8B | MathQA | 44.60 | 44.13 | 57.27 | 62.00 | **69.27** | **+24.67** |

#### Single-agent zero-shot（Table 2）

| 模型 | 数据集 | Base | MV-SFT | MV-GRPO | MV-KTO | MV-DPO | Best Δ |
|------|--------|------|---------|----------|---------|---------|--------|
| Qwen-2B | MATH | 7.67 | 11.51 | 18.09 | 20.18 | **23.49** | **+15.82** |
| Qwen-2B | GSM8K | 23.00 | 24.84 | 34.40 | **45.13** | 43.87 | **+22.71** |
| Llama-3B | MathQA | 23.87 | 23.44 | 30.09 | 42.84 | **45.00** | **+21.13** |
| Llama-8B | MATH | 22.93 | 23.16 | 29.66 | 39.42 | **46.00** | **+23.07** |
| Llama-8B | GSM8K | 57.93 | 42.09 | 62.45 | 72.36 | **77.36** | **+19.43** |

#### OOD 泛化（Table 3, MV-DPO, joint training "All"）

| 模型 | 训练集 | SVAMP | GPQA | CSQA |
|------|--------|-------|------|------|
| Qwen-2B | All | **+27.7** | **+16.3** | **+59.6** |
| Llama-3B | All | +7.1 | **+10.7** | +11.0 |

#### Debate vs Ground-Truth（Table 4, Llama-8B）

使用 debate majority vote (DMV) 标签与 ground-truth (GT) 标签训练效果对比：DMV 在多数 setting 下与 GT **持平或更优**，例如 MV-DPO single-agent MATH: GT=45.13 vs DMV=46.40；multi-agent GSM8K: GT=81.60 vs DMV=83.00。

#### Debate RL vs Single-round RL（Table 5）

| 方法 | Qwen-2B MATH | Llama-3B MATH | Phi-4B MATH | Llama-8B MATH |
|------|-------------|---------------|-------------|---------------|
| TTRL | +18.0±2.9 | +5.3±5.7 | +6.1±2.1 | +7.5±0.2 |
| ScPO | +2.3±1.1 | +3.4±0.2 | +0.1±0.5 | +3.7±0.6 |
| **MV-DPO (MACA)** | **+16.7±0.4** | **+12.5±0.7** | **+6.9±0.2** | **+17.1±0.8** |

MACA MV-DPO 在所有 8 个 model-dataset 组合中均优于 ScPO，在 6/8 中优于 TTRL（另 2 个在标准差内）。

### 5.3 Self-Consistency 提升

- Self-consistency 曲线稳定高出 baseline 最多 **27.6 个百分点**（GSM8K）
- Self-consistency 与 accuracy 强相关（$r > 0.86$，所有推理条件下）
- Post-training 后模型生成的回答比 base model **短 22-36%**，说明 preference learning 隐式充当 format reward

### 5.4 Multi-agent Agreement 改善

Post-training 后（Qwen-2B on GSM8K）：
- 全体一致（3/3 agreement）从 **13.4% → 43.4%**（三倍增长）
- 不可解析响应从 **11% → 0.6%**
- 无共识（0/3 agreement）从 45.6% → 19.8%

### 5.5 关键发现

1. **Preference learning > scalar reward > imitation learning**：MV-DPO/MV-KTO 优于 MV-GRPO 优于 MV-SFT；在完整推理轨迹上做 contrastive 比较，能更好解决 credit assignment 问题。MV-KTO 在小模型（≤3B）表现更佳，MV-DPO 在大模型（4-8B）更佳。
2. **Debate 信号 > single-round majority vote**：multi-round debate 通过 deliberative exchange 产生比独立采样更丰富的共识信号，post-training on debate context 让模型学会 peer context utilization。
3. **Self-supervised ≈ ground-truth supervision**：Debate-derived majority vote 标签在几乎所有 setting 下与 oracle ground-truth 标签训练效果持平甚至更优，无需任何外部标注。
4. **改善源于推理质量而非格式**：分解分析表明 **69-100%** 的改善来自 better reasoning 而非避免 truncation 或格式改善。
5. **强 OOD 泛化**：在数学数据上训练可泛化到科学推理（GPQA +16.3%）和常识推理（CSQA +59.6%）。
6. **4-bit 训练可迁移到 full-precision**：在 4-bit quantized 模型上的改善可直接迁移到 full-precision 模型。
7. **Compute efficiency**：MACA MV-DPO 需 0.73-1.68 GPU hours，计算量与 ScPO 相当但性能远超，远低于 TTRL（2.2-7.7 GPU hours）且结果更稳定。
8. **Iterative training 带来持续但递减的收益**：It-1 → It-2/It-3 通常额外提升 1-3pp，23/24 个设定中后续迭代达到最佳。
