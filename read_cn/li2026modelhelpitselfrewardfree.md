# SePT: A Model Can Help Itself

**Paper:** A Model Can Help Itself: Reward-Free Self-Training for LLM Reasoning  
**Authors:** Mengqi Li, Lei Zhao, Anthony Man-Cho So, Ruoyu Sun, Xiao Li  
**arXiv:** 2510.18814  
**Venue:** arXiv 2026  
**Code:** https://github.com/ElementQi/SePT

| Method | Carrier | Regime | Level |
|---|---|---|---|
| SePT | online SFT on self-generated traces | training-time reward-free self-training | self-generated reasoning trajectories |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2510.18814.pdf` |
| Extracted text | `0524_new_collection/texts/2510.18814.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against reward-free online self-training, self-generated traces, SFT/NLL updates, temperature dynamics, and prompt-source caveats |
| BibTeX | `0524_new_collection/bib/2510.18814.bib` |
| Suggested taxonomy | strict UPT candidate; Family III |

## 1. UPT 归属理由

SePT 是 strict UPT candidate。它提出 reward-free online self-training：模型从当前 policy 对无答案 prompts 自采样 reasoning traces，然后直接用标准 SFT / NLL objective 训练这些 self-generated traces；更新后的模型再继续生成下一轮训练数据。整个过程不使用 rewards、verifiers 或 teacher-provided solution traces。

按当前 taxonomy，它应归入 **Family III: Self-Generated Target Bootstrapping**。它的训练 target 是模型自己的 sampled trajectories，而不是 entropy/gradient/statistic reward，也不是 sample-relation consensus。与普通 self-training 相比，SePT 的重点是在线 refresh：每轮用更新后的模型重新采样，避免固定离线自生成数据的弱增益。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 对模型参数执行标准 SFT / NLL 更新 |
| B2 internal signal | Pass | target traces 由当前模型在 prompts 上自生成 |
| B3 no external supervision | Pass with prompt-source caveat | 使用 DeepScaleR / OpenThoughts-Math question sets，但不使用答案、reward、verifier 或 teacher solutions |
| B4 internal judge/scorer | Not applicable | 方法没有 judge/scorer；只做 reward-free self-training |

## 3. 方法介绍

SePT 的算法很简单：

1. 从训练 question set 中采样 prompts；
2. 用当前模型在 sampling temperature `tau_s` 下生成 `G` 个 outputs，默认 `G=1`；
3. 把 `(q, o)` 加入本轮 self-training batch；
4. 在 training temperature `tau_t` 下最小化 standard NLL / SFT loss；
5. 用更新后的模型进入下一轮 self-generation。

论文把一轮 SePT 解释为 forward-KL projection：学生模型被拉向当前低温 self-teacher distribution。关键经验设置是 `tau_s < tau_t`，低温采样能保持局部 token ordering 并放大 margin。与 RLVR/GRPO 不同，SePT 没有 reward-weighted policy optimization，也没有 external verifier。

## 4. 数据集

主要训练 question set 是 DeepScaleR (DSR)，另一个训练 question set 是 OpenThoughts-Math (OTM)。论文重点是使用 questions/prompts，而不是使用答案作为监督。

评估覆盖六个数学推理 benchmark：

- MATH-500
- AMC-23
- Minerva Math
- OlympiadBench
- AIME-24
- AIME-25

作者还做了 DSR / OTM 与评测集的 contamination audit。DSR contamination 明显高于 OTM，尤其来自 MATH-500 和 OlympiadBench；OTM 的污染更低。这一点纳入 survey 时需要作为 prompt-source caveat 写清楚。

## 5. 评估指标与主要结果

在 Qwen2.5-Math-7B 上，SePT 把六个 math benchmarks 平均 Pass@1/8/32 从 22.7 / 47.3 / 61.0 提高到 39.5 / 57.7 / 67.9，AVG 从 43.7 提高到 55.0。在 Qwen2.5-7B 上，AVG 从 44.1 提高到 50.7。对 instruction-tuned models 的提升较小，论文明确指出 SePT 的有效性与模型族和初始分布有关。

与 GRPO 相比，SePT 性能通常较弱，但计算量低很多；在 OTM question set 上与 GRPO 的差距缩小。与 EM-FT 相比，SePT 更强，因为它训练整条 self-generated trajectory，而不是只在 visited prefixes 上做 token-level entropy minimization。

离线版本 SePT (Offline) 的提升明显弱于在线 refresh 版本，说明 update-bearing self-generated data loop 是方法核心。

## 6. UPT Survey 定位

推荐作为 Family III 新增候选。它是一个很干净的 reward-free self-generated target baseline，能够和 Self-Tuning、Self-Improve、G-Zero 等方法形成清晰对照：没有 verifier，没有 reward model，也没有偏好优化，只用自生成轨迹做在线 SFT。

推荐短标签：`SePT`。  
推荐 family：`Self-Generated Target Bootstrapping`。  
主表 caveat：训练 question sets 来源于 DSR / OTM，且 DSR 有 contamination 风险；但训练信号本身不使用答案、verifier 或 teacher traces。
