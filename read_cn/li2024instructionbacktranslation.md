# Self-Alignment with Instruction Backtranslation (Humpback)

> **Added to survey on:** 2026-05-26（adjacent precursor）

## 元信息
- **Paper:** Self-Alignment with Instruction Backtranslation
- **Authors**: Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis
- **Venue:** arXiv 2308.06259 (Aug 2023)；ICLR 2024
- **arXiv:** 2308.06259
- **Project name**: 模型称为 **Humpback**

## Audit Update / Caveat
本论文在本 UPT survey 中作为 **adjacent (B3 failed, seed-supervised bootstrapping) precursor** 处理，**不属于 strict UPT**。`adjacent_methods.csv` 记录：`signal_source = human-written seed pairs`、`failed_check = B3`、`decision = precursor of self-generated target bootstrapping`。main.tex §F.3："instruction-backtranslation pipelines bootstrap from human-written seeds, failing B3 at the seed stage."

## 1. 与 UPT survey 的关系
- 它属于 self-generated instruction (instruction self-curation) 类工作的 precursor，与 strict UPT Family III 思想同源：把无标注 web 文本视为 "answer"，反向**生成 instruction**，得到自合成的 (instruction, response) 对。
- 它失败的 boundary check 是 **B3**：起始的 backward model（"`M_yx`"）需要在一个 **3,200 对人工 (instruction, output) 种子** 上做 SFT 才能产生可用的反向标注能力；这一步 seed supervision 后续传递给所有 self-curated 数据。
- 因此被路由到 adjacent。它是后续完全 seed-free instruction-tuning（如 CYCLE-INSTRUCT）所要打破的范式之一。

## 2. 方法核心
- **输入**：
  1. 一个小 **seed set**：3,200 条 (instruction, output) 对，来自 OpenAssistant (OASST1) 人工标注的对话首轮；
  2. 一个大规模 **unlabeled web corpus**：从 ClueWeb 抽取 502K 条候选英文文本作为 "potential responses"；
  3. base LM：LLaMA-1 65B。
- **输出**：自筛选后的 high-quality 合成 (instruction, response) 对（论文 A5 阶段约 51K 条）；用于 SFT 得到 **Humpback** 模型。
- **训练目标**：standard SFT（cross-entropy）。
- **Pipeline（自标注 + 自筛选两阶段）**：
  1. **Self-augment（backtranslation）**：在 3,200 seed 上 SFT 得到反向模型 `M_yx`（output → instruction）。把它套到 502K 无标注 web 文本上，为每条文本生成一个候选 instruction，得到 `A_0` 集合；
  2. **Self-curate（迭代质量评分）**：在同样的 3,200 seed 上 SFT 一个 forward 评分模型 `M_0`（也即直接 instruction-tuning 的 LLaMA），用 5-point Likert prompt 让它对 `A_0` 中每条 (instruction, response) 打分；保留分数 ≥ 5 的高质量子集 `A_1^{(5)}`；
  3. **Iterate**：用 seed ∪ `A_1^{(5)}` 再训一个更强的 forward model `M_1`，再次打分得到 `A_2^{(5)}`，如此迭代两轮。最终训练数据为 seed + `A_2^{(5)}`。
- **关键点**：
  - **Seed 性质**：3,200 条 OASST1 人工对话，是真实 human-written 监督；这是失败 B3 的根本原因；
  - **Self-curate 评分器**也是从 seed SFT 出来的，因此筛选信号也间接来自 seed；
  - **Web text** 本身无标注，但被当作 "高质量 response"，本质上把 seed 风格映射到 web 语料。

## 3. 数据集与规模
- **Seed**: 3,200 (instruction, output) 对（OASST1 高质量 turn）。
- **Unlabeled corpus**: 502K ClueWeb 段落（启发式过滤后）。
- **Final A5 (score ≥ 5) curated set**: ~51K 条。
- **Base model**: LLaMA-1 65B（也报告 7B / 33B 结果）。

## 4. 实验结果（简要）
- **AlpacaEval (vs. text-davinci-003)**：Humpback 65B win-rate **83.7%**，超过 LIMA 65B、Guanaco 65B、Vicuna 13B 等同期开源 instruction-tuned 模型。
- **Scaling 行为**：随着 seed-augmented self-curated 数据规模增大，AlpacaEval 胜率单调上升；而仅扩大 seed 规模或仅扩大 self-augmented (未筛选) 数据规模都更早饱和——说明 **self-curation 评分** 是关键。
- **Ablation**：
  - 不做 self-curation（直接 SFT 在 A_0 上）→ 性能明显下降；
  - 不迭代（仅一轮）→ 性能也下降。

## 5. UPT Survey 定位
- **与 strict UPT Family III 的关系**：Instruction Backtranslation 把 "self-generated instruction over unlabeled web text" 这一思路工程化，是 CYCLE-INSTRUCT、Magpie 等 strict UPT 方法的**直接技术前驱**——后者把它变成完全 seed-free（CYCLE-INSTRUCT 用同一模型双头 + cycle consistency 替代了 seed-trained backward model）。
- **为什么不算 strict UPT**：
  1. 反向模型 `M_yx` 必须先在 3,200 条人工 OASST1 对上 SFT，否则反向标注能力不存在；
  2. self-curate 评分器同样源于 seed-SFT；因此整个 pipeline 的两个关键 "self-" 组件都依赖人工 seed，违反 B3。
- **对后续 self-generated target 类方法的影响**：
  1. 验证了 "用大规模 unlabeled web text 当作 response，再让模型反向生成 instruction" 的可行性；
  2. 提出的 **self-curation via Likert-scale self-rating** 是后续 self-rewarding / self-judging 类方法（Self-Rewarding LM, Meta-Rewarding 等）的早期形式；
  3. 后续完全 seed-free 工作（CYCLE-INSTRUCT 等）的 motivation 段落几乎都把 Humpback 作为 "依赖 seed" 的对照范例。
- **一句话定性**：Humpback 在思路上预示了 strict UPT 中的 instruction self-curation 与 self-rating，但因 backward model 与 self-rater 都需要人工 seed SFT，被划入 adjacent precursor。
