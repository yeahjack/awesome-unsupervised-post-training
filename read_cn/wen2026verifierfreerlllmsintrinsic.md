# VIGOR: Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward

**Paper:** Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward  
**Authors:** Xuexiang Wen, Hang Yu, Linchao Zhu, Gaoang Wang  
**arXiv:** 2605.09920  
**Venue:** arXiv 2026; Chrome/arXiv result page marks Findings of ACL 2026  
**Code:** https://github.com/ZJUSCL/VIGOR

| Method | Carrier | Regime | Level |
|---|---|---|---|
| VIGOR | GRPO | training-time verifier-free RL | parameter-space gradient statistic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files |
| PDF | `0524_new_collection/pdfs/2605.09920.pdf` |
| Extracted text | `0524_new_collection/texts/2605.09920.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against gradient-norm intrinsic reward, length correction, rank shaping, GRPO updates, and discarded-reference/label-free training claims |
| BibTeX | `0524_new_collection/bib/2605.09920.bib` |
| Suggested taxonomy | strict UPT candidate; Family I |

## 1. UPT 归属理由

VIGOR 是 strict UPT candidate。它不使用 gold-answer verifier、human labels、external reward model 或 teacher labels，而是把当前 policy 自身的 teacher-forced NLL gradient norm 作为 intrinsic reward。对同一个 prompt 采样一组 completions 后，VIGOR 偏好那些在当前参数附近产生更小 `L2` gradient norm 的 completion，并把这个组内排序信号接入 GRPO 更新。

按当前 taxonomy，它最适合放入 **Family I: Prediction-Statistic Optimization**。它的主信号不是样本之间的 majority consensus，也不是 self-generated target distillation，而是从当前模型参数空间直接读出的 optimization statistic。与 entropy minimization、stable-rank reward、self-certainty reward 相比，VIGOR 把 prediction-statistic signal 扩展到 gradient geometry。

## 2. 边界检查

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | 使用 GRPO 对 Qwen2.5-3B-Base / Qwen2.5-7B-Base 做 post-training |
| B2 internal signal | Pass | reward 由当前 policy 的 NLL gradient norm 计算，论文表格说明 VIGOR reward external dependency 为 none |
| B3 no external supervision | Pass | MATH / CodeContests 训练时丢弃 reference solutions / labels，只保留 prompts/problems |
| B4 internal judge/scorer | Pass | 没有独立 judge；scorer 是同一 policy 的参数梯度统计 |

## 3. 方法介绍

给定 prompt `x` 和 completion `y`，VIGOR 先计算 token-averaged negative log-likelihood loss `l_mean(x,y)`，再取当前参数下的梯度范数 `||grad_theta l_mean(x,y)||_2`。直觉是：如果一个 completion 与模型当前已经学到的局部结构更一致，它会诱发更小的参数更新；错误、突兀或不稳定 completion 更可能产生较大的 gradient norm。

论文还处理了两个实现问题：

- 长度偏置：raw gradient norm 与 token 数有关，因此使用 `sqrt(T)` length correction。
- reward 稳定性：不直接用跨 prompt 的绝对数值，而是在同一 prompt 的 completion group 内排序并归一化到 rank-shaped reward。

最终训练流程是：对每个 prompt 采样 `G=8` 个 completions，计算 length-corrected gradient norm，组内排序得到 reward，然后用 GRPO objective 更新 policy。reward computation 中的 gradient norm 被 detach，只作为 scalar reward，不通过二阶路径反传。

## 4. 数据集

训练数据包括：

- MATH training set：使用 7,500 个 math problems，丢弃答案 / reference solutions。
- CodeContests：使用前 3,200 个训练 problems，作为 code 任务 sanity check，丢弃 reference solutions。

评估包括 MATH500、GSM8K、AMC、LiveCodeBench-v6、CRUX、MMLU-Pro 和 IFEval。主要模型是 Qwen2.5-3B-Base 与 Qwen2.5-7B-Base。GT-Reward baseline 使用 exact-match verifier 和 reference answers，仅作为外部监督对照，不是 VIGOR 自身信号。

## 5. 评估指标与主要结果

论文报告，在 MATH 上 post-train 后，VIGOR 在 Qwen2.5-7B 上的 math average 明显提升，并相对 INTUITOR 这类 self-certainty reward 方法更稳定。CodeContests 训练也能迁移到 math/code 多项评测，但作者指出 gradient-norm signal 对 code correctness 的覆盖并不完美。

消融显示，`sqrt(T)` length correction 与 rank shaping 都是关键组件。reward reliability 分析中，同一 prompt 内被 gradient norm 排得更好的 completions 往往有更高 correctness，说明这个 intrinsic parameter-space signal 与 task performance 存在可用相关性。

## 6. UPT Survey 定位

推荐作为 Family I 新增候选。它能补足当前 Family I 中“概率/熵/表示统计”之外的 parameter-gradient 方向：同样不需要外部 verifier，但信号来自更新几何而非 output distribution 本身。

推荐短标签：`VIGOR`。  
推荐 family：`Prediction-Statistic Optimization`。  
主表 caveat：训练 prompts 来自 MATH / CodeContests，但 VIGOR 自身丢弃 labels/reference solutions；GT-Reward 只是 baseline。
