# VIGOR: Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward

**Paper:** Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward  
**Authors:** Xuexiang Wen, Hang Yu, Linchao Zhu, Gaoang Wang  
**arXiv:** 2605.09920  
**Venue:** arXiv 2026; the Chrome/arXiv result page marks Findings of ACL 2026.  
**Code:** https://github.com/ZJUSCL/VIGOR

| Method | Carrier | Regime | Level |
|---|---|---|---|
| VIGOR | GRPO | training-time verifier-free RL | parameter-space gradient statistic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, `acl-style-files/custom.bib`, or main taxonomy files. |
| PDF | `0524_new_collection/pdfs/2605.09920.pdf` |
| Extracted text | `0524_new_collection/texts/2605.09920.txt` |
| Basis for this note | Complete downloaded PDF + extracted full text; source audit rechecked on 2026-05-24 against the gradient-norm intrinsic reward, length correction, rank shaping, GRPO updates, and the "discarded references / label-free training" claims. |
| BibTeX | `0524_new_collection/bib/2605.09920.bib` |
| Suggested taxonomy | strict UPT candidate; Family I. |

## 1. UPT Assignment Rationale

VIGOR is a strict UPT candidate. It uses no gold-answer verifier, no human labels, no external reward model, and no teacher labels; instead, it takes the current policy's teacher-forced NLL gradient norm as the intrinsic reward. After sampling a group of completions for the same prompt, VIGOR prefers completions whose `L2` gradient norm at the current parameters is smaller, and pipes this intra-group ranking signal into the GRPO update.

Under the current taxonomy, the best fit is **Family I: Prediction-Statistic Optimization**. Its primary signal is neither majority consensus across samples nor self-generated-target distillation; it is an optimization statistic read directly from the current model's parameter space. Compared with entropy minimization, stable-rank reward, and self-certainty reward, VIGOR extends the prediction-statistic signal to gradient geometry.

## 2. Boundary checks

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Uses GRPO to post-train Qwen2.5-3B-Base / Qwen2.5-7B-Base. |
| B2 internal signal | Pass | Reward is computed from the current policy's NLL gradient norm; the paper's table marks VIGOR's reward external dependency as none. |
| B3 no external supervision | Pass | When training on MATH / CodeContests, reference solutions / labels are discarded; only prompts/problems are kept. |
| B4 internal judge/scorer | Pass | No separate judge; the scorer is the gradient-statistic of the same policy's parameters. |

## 3. Method

Given a prompt `x` and a completion `y`, VIGOR first computes the token-averaged negative log-likelihood loss `l_mean(x,y)`, then takes the gradient norm at the current parameters: `||grad_theta l_mean(x,y)||_2`. The intuition: if a completion is more consistent with the local structure the model has already learned, it induces a smaller parameter update; wrong, abrupt, or unstable completions are more likely to produce larger gradient norms.

The paper also handles two implementation issues:

- Length bias: the raw gradient norm depends on token count, so a `sqrt(T)` length correction is applied.
- Reward stability: instead of using absolute values across prompts, completions within the same prompt's group are ranked and normalized into a rank-shaped reward.

The final training pipeline: for each prompt, sample `G=8` completions, compute length-corrected gradient norms, rank within the group to produce a reward, then update the policy with the GRPO objective. The gradient norm used in reward computation is detached and serves as a scalar reward only — no second-order path is backpropagated.

## 4. Datasets

Training data includes:

- MATH training set: 7,500 math problems; answers / reference solutions discarded.
- CodeContests: the first 3,200 training problems as a sanity check for code; reference solutions discarded.

Evaluation includes MATH500, GSM8K, AMC, LiveCodeBench-v6, CRUX, MMLU-Pro, and IFEval. The main models are Qwen2.5-3B-Base and Qwen2.5-7B-Base. The GT-Reward baseline uses an exact-match verifier and reference answers; it serves only as an external-supervision contrast and is not VIGOR's own signal.

## 5. Evaluation metrics and main results

The paper reports that after post-training on MATH, VIGOR substantially improves the math average on Qwen2.5-7B and is more stable than self-certainty-reward methods such as INTUITOR. Training on CodeContests also transfers to multiple math/code evaluations, although the authors note that the gradient-norm signal does not fully cover code correctness.

Ablations show that the `sqrt(T)` length correction and rank shaping are both key components. The reward-reliability analysis shows that within a prompt, completions ranked higher by gradient norm tend to have higher correctness, indicating a usable correlation between this intrinsic parameter-space signal and task performance.

## 6. Position in the UPT Survey

Recommended as a new Family I candidate. It complements Family I's existing "probability / entropy / representation statistics" with a parameter-gradient direction: still no external verifier, but the signal comes from update geometry rather than the output distribution itself.

Suggested short label: `VIGOR`.  
Suggested family: `Prediction-Statistic Optimization`.  
Main-table caveat: training prompts come from MATH / CodeContests, but VIGOR itself discards labels / reference solutions; GT-Reward is only a baseline.
