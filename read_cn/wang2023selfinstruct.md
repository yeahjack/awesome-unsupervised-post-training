# Self-Instruct: Aligning Language Models with Self-Generated Instructions

> **Added to survey on:** 2026-05-26（adjacent precursor）

## 元信息
- **Paper:** Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Authors**: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Venue:** ACL 2023 (Volume 1: Long Papers), Toronto, Canada
- **arXiv:** 2212.10560
- **ACL Anthology**: https://aclanthology.org/2023.acl-long.754/
- **Code:** https://github.com/yizhongw/self-instruct

## Audit Update / Caveat
本论文在本 UPT survey 中作为 **adjacent (B3 failed, seed-supervised bootstrapping) precursor** 处理，**不属于 strict UPT**。原因记录在 `adjacent_methods.csv`：`signal_source = human-written seed instructions`、`failed_check = B3`、`decision = precursor of self-generated target bootstrapping`。在 main.tex §F.3 与 §sec:boundary-protocol 中显式声明："Self-Instruct … bootstrap from human-written seeds, failing B3 at the seed stage. They are treated as precursors of self-generated target bootstrapping rather than strict UPT."

## 1. 与 UPT survey 的关系
- 它属于 self-generated target / instruction self-curation 路线（即 strict UPT Family III "Self-Generated Target Bootstrapping, knowledge / instruction self-curation"）的**直接思想前驱**：模型自生成 instruction，再用自生成数据进行 SFT。
- 它失败的 boundary check 是 **B3（no human / external supervision in the loop）**：bootstrap 阶段使用了 **175 条人工编写的 seed tasks**（每条含 1 个 instruction、1 个 input、1 个 output 示例，由作者团队手写），这些 seed 直接进入 in-context 提示去诱导 LLM 生成新 instruction，因此 seed 监督泄漏到了 self-training 数据流。
- 因此被路由到 adjacent，而非 strict UPT。它是 strict UPT 中 `CYCLE-INSTRUCT` (shen2025cycleinstruct) 等 **完全 seed-free** instruction tuning 方法的直接对照对象。

## 2. 方法核心
- **输入信号**：175 条人工编写的 seed task（task pool 初始化）+ 一个待 align 的强 base LM（论文主实验用 GPT-3 `davinci`，175B）。
- **输出**：约 **52K instruction-following 样本**（约 82K instruction–instance 对，覆盖 ~52K 条独立 instruction），随后用于 SFT 该 LM。
- **训练目标**：标准 instruction-tuning SFT（cross-entropy 拟合 model-generated 的 instruction → output）。
- **生成–筛选 pipeline**（4 步迭代）：
  1. **Instruction generation**：从 task pool 随机抽 8 条 seed（6 条人工 + 2 条 model-generated）作为 in-context demo，用 GPT-3 自生成新 instruction；
  2. **Classification task identification**：用 few-shot prompt 让模型判断该 instruction 是否为分类任务（影响后续 input/output 生成顺序）；
  3. **Instance generation**：input-first（generation 任务）或 output-first（分类任务）方式生成 (input, output)；
  4. **Filtering**：ROUGE-L 与已有 instruction 相似度 > 0.7 即丢弃；过滤包含图片、过短/过长或含特定关键词的样本；通过的加入 task pool 形成迭代。
- **关键点**：seed 来源是 **作者手写的 175 条**，每条包含完整 (instruction, input, output) 三元组；selection 流程纯启发式（ROUGE-L 去重 + 长度/关键词过滤），不依赖外部模型评分。

## 3. 数据集与规模
- **Seed**: 175 人工编写 task（覆盖 brainstorming、classification、generation、open QA、closed QA、rewriting、summarization 等多类）。
- **Generated pool (final)**: ~52K instructions, ~82K (instruction, input, output) instances。
- **Released artefact**: `GPT3SELF-INST`（GPT-3 davinci 在 self-instruct 数据上 SFT 后的模型）以及 52K 数据集。

## 4. 实验结果（简要）
- **SUPER-NaturalInstructions (SuperNI) 评测（119 tasks）**：原始 GPT-3 (davinci) → 加 self-instruct SFT 后，ROUGE-L 大幅提升，与在人工 SuperNI 训练数据上 fine-tune 的 `T0` / `Tk-Instruct` 等方法相比缩短了大约 ~5 个点的差距，且仅依赖自生成数据。
- **252 条作者编写的 user-oriented instructions 上的人工评测**：`GPT3SELF-INST` 大幅优于原始 GPT-3，与 `InstructGPT-001`（OpenAI 商业模型）总体表现相当或略接近。
- **Ablation**：数据多样性（ROUGE-L 去重阈值）对最终性能至关重要；规模与质量之间，规模仍占主导（生成的 52K 数据噪声不低，但量大胜过精）。

## 5. UPT Survey 定位
- **与 strict UPT Family III 的关系**：Self-Instruct 是 "model-generated instruction → self-train" 范式的奠基性工作，思想直接通向 Family III 中的 instruction / knowledge self-curation 路线（CYCLE-INSTRUCT、Self-Rewarding LM 的 IFT 阶段、LongMagpie 等）。
- **为什么不算 strict UPT**：bootstrap 起点不是 unlabeled corpus，而是 **175 条精心编写的人工 seed**——这些 seed 的覆盖度、风格与质量直接塑造了下游所有自生成数据的分布，因此 supervision signal 本质上来自人工，不满足 boundary check **B3**。
- **对后续工作的影响**：
  1. Stanford **Alpaca** 直接复用 Self-Instruct pipeline；
  2. **Vicuna**, **WizardLM (Evol-Instruct)**, **Orca**, **Baize** 等几乎所有早期 open-source instruction-tuning 工作均沿用或扩展此 seed-then-self-generate 框架；
  3. CYCLE-INSTRUCT、Magpie 等 strict UPT 方法将其作为 "需要打破 seed 依赖" 的**对照**，证明 instruction tuning 可在不依赖人工 seed 的情况下进行。
- **一句话定性**：Self-Instruct 在思想上是 self-generated target bootstrapping 的开山之作，但在 boundary protocol 上因其人工 seed 依赖被划入 adjacent precursor，与 strict UPT 仅一步之遥。
