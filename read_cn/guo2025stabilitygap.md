# Stability-Gap CPT

> **加入 Survey 时间：** 2026-03-11

**论文：** Efficient Domain Continual Pretraining by Mitigating the Stability Gap
**作者：** Yiduo Guo, Jie Fu, Huishuai Zhang, Dongyan Zhao (Peking University, HKUST)
**发表：** ACL 2025 (Long Paper)
**Method：** Stability-Gap CPT | **Carrier：** Direct Opt. | **Regime：** training-time | **Level：** Token

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| 触发单位 Trigger Unit | unlabeled corpus / continual training batch |
| 参数/状态持久性 Persistence | full parameter accumulate across corpus stages; no sample-level reset |
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

- **更新何时触发：** 更新在 deployment 前的 continued pretraining / CPT 阶段按 corpus 与 training batch 持续触发，不由单个测试样本触发。
- **服务当前样本还是后续样本：** 当前 batch 的更新服务后续训练批次与最终部署模型，而不是服务当前样本的即时推理。
- **参数/状态是否累积：** 参数在整个 corpus stream / stage 序列中持续累积；若有阶段切换，也只是训练阶段切换，不是 sample-level reset。
- **reset 边界：** 因此它回答的是 deployment 前的 adaptation schedule，而不是 deployment 期间的 online TTA。

## 1. UPT 归属理由

本文属于 **Family I (Prediction-Statistic Optimization)**：核心目标始终是 **token-level language-modeling loss（predictive likelihood minimization）**。尽管论文引入三种稳定化机制（多 epoch 子集训练、基于 domain-PPL 的数据选择、与 pretraining mixture rate 匹配），但它们都是为了在领域 corpus 上更高效地最小化 LM loss，没有任何一项依赖外部标注、外部 verifier 或人工偏好信号。用于数据选择的 KenLM perplexity 本身是 intrinsic statistic（模型自身的 n-gram 概率），完全符合 UPT 定义。

---

## 2. 解决的问题

Continual pretraining (CPT) 是把通用 LLM 适应到特定领域（医疗、法律等）的常用方式。然而作者观察到一个跨 model scale 与跨领域普遍存在的现象：**CPT 早期目标领域任务性能先下降，再逐步恢复，最终超过原始模型**，呈 V 形曲线。作者借用视觉 continual learning 中的 **stability gap** 概念，归因为 **训练开始时 plasticity gradient（学习新知识）压倒 stability gradient（保留旧知识）**，临时损害模型的通用能力（instruction-following、commonsense）。

这种 stability gap 浪费 compute budget —— 模型需要大量 tokens 才能从初始下降中恢复。本文目标：
1. 系统识别并解释 CPT 中的 stability gap；
2. 在 **固定 compute budget** 下提出高效策略以缓解 stability gap，使模型更快达到更高性能。

---

## 3. 方法介绍

作者提出三种互补的高效 CPT 策略：

### 策略 I：Multi-epoch Subset Training
不再在整个大 corpus 上训练 1 epoch，而是 **在更小子集上多 epoch 训练**。核心洞察：大 corpus 需要持续高 plasticity gradient 较长时间，而小子集多 epoch 训练能在第一个 epoch 之后缩短 plasticity gradient 的持续时间，让 stability gradient 上升、性能更快恢复。实验显示峰值性能出现在第 4 epoch 附近。

### 策略 II：Domain PPL-based Data Selection
用 **KenLM**（在医疗 Wikipedia corpus 上训练的 n-gram 模型）计算 RefinedWeb 中每个样本的 **domain perplexity (PPL)**，并选取 PPL 最低的子集作为领域 corpus。更低的 domain PPL 意味着与目标领域分布更匹配，能加速恢复并提升峰值性能。

### 策略 III：Pretraining Mixture Rate Matching
按原始 pretraining 数据的 **mixture rate**（各数据源比例）构造 CPT 数据混合。具体：先按 Llama mixture rate 采样 5B tokens，再把 CC 与 C4 部分（约 82%）替换为策略 II 用 KenLM 选出的医疗 tokens。每个 epoch 重新采样并替换，以降低分布漂移、稳定 instruction-following 能力。两种变体：
- **Rate-Fixed-Data-Fixed：** 采样一次；所有 epoch 用同一 corpus。
- **Rate-Fixed-Data-Dynamic：** 每个 epoch 独立采样，得到动态变化的训练 corpus。

三种策略可组合使用以联合缓解 stability gap。

---

## 4. 数据集

### Pretraining 数据
- **基础模型：** OpenLLaMA-3B-v2（在 RefinedWeb 上训练）、TinyLlama-1.1B、Llama-3-8B。
- **领域 corpus 来源：** RefinedWeb dataset，通过 KenLM（在医疗 Wikipedia corpus 上训练）按 domain PPL 排序；选取最低 PPL 子集。
- **Compute budget：** baseline 用 50B 医疗 tokens（1 epoch）；本文策略用 5B tokens × 4 epochs = **20B tokens**（仅为 baseline 的 40%）。
- **领域：** 主实验为 **医疗**；补充实验覆盖 **法律** 与一般的 continual-pretraining 场景。

### 评测数据
- **MMLU-Medical：** medical genetics, anatomy, clinical knowledge, professional medicine, college medicine。
- **PubMedQA** (Jin et al., 2019)。
- **MedMCQA** (Pal et al., 2022)。
- **MedQA-4-Option** (Jin et al., 2021a)。
- 评测框架：lm-evaluation-harness (Gao et al., 2023)。

### Llama-3-Physician 附加评测
- Classification (HOC), Relation Extraction (DDI-2013), BioNLI, Summarization (MIMIC-CXR)。

---

## 5. 评估指标与主要结果

### 指标
- **Zero-shot accuracy** 在医疗 benchmark 上。
- **Average medical performance：** 上述四个医疗 benchmark 的平均 accuracy。
- **Commonsense task performance：** 衡量通用能力的保留。
- **Medical perplexity (PPL)：** 医疗 Wikipedia corpus 上的 perplexity。

### 主要结果

**OpenLLaMA-3B 实验（Table 1）：**

| Method | Tokens | MMLU-Med | PubMedQA | MedMCQA | MedQA-4-Opt | Avg |
|---|---|---|---|---|---|---|
| OpenLLaMA-3B (原始) | - | 25.6 | 68.4 | 25.4 | 25.4 | 36.2 |
| Full token baseline | 50B | 26.1 | 70.4 | 26.1 | 27.1 | 37.4 |
| Replay 10B data | 50B | 29.3 | 71.0 | 30.4 | 27.6 | 39.5 |
| **本文策略** | **20B** | **30.0** | **71.2** | **34.0** | **27.8** | **40.7** |

- 仅用 **40% 的 compute budget（20B vs 50B tokens）**，medical 平均性能从 36.2% 升至 **40.7%**（+4.5%），击败所有 baseline。
- **无通用能力的 catastrophic forgetting：** commonsense 性能甚至有所提升。

**Llama-3-Physician-8B（Table 2，任务特定 fine-tuning）：**

| Model | MMLU-Med | PubMedQA | MedMCQA | MedQA-4-Opt | Avg |
|---|---|---|---|---|---|
| Llama-3-8B base | 47.2 | 52.1 | 38.2 | 35.5 | 43.3 |
| Llama-3-Physician-8B (本文) | **85.0** | **79.1** | **81.4** | 61.5 | **76.7** |
| GPT-3.5-turbo-finetuned | 70.5 | 71.4 | 61.8 | 63.3 | 66.7 |
| Llama-2-70B | - | 78.0 | 62.7 | 61.3 | 67.2 |

- Llama-3-Physician-8B 在同 scale 段（7–8B）开源模型中最佳。
- 平均性能接近 GPT-4；在 classification、relation extraction、NLI、summarization 任务上大幅超过 GPT-4。
- **7B 模型平均甚至击败多个 70B 模型。**

### 关键消融结论
- **多 epoch 训练** 加速恢复，峰值在 epoch 4。
- **KenLM 选出的子集** 恢复更快、峰值更高，优于随机子集。
- **数据 mixture rate 匹配** 在保留 medical 性能的同时提升 commonsense 性能。
- 学习率过高导致通用能力大幅下降；过低阻碍新知识获取；子集过大引入 stability gap；子集过小导致后期 overfitting。
