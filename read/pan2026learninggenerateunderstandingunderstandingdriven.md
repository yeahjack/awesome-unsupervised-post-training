# GvU: Learning to Generate via Understanding

**Paper:** Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models
**Authors:** Jiadong Pan, Liang Li, Yuxin Peng, Yu-Ming Tang, Shuohuan Wang, Yu Sun, Hua Wu, Qingming Huang, Haifeng Wang
**arXiv:** 2603.06043
**Venue:** CVPR 2026 / arXiv 2026
**Code:** https://matrix0721.github.io/gvu.github.io/

| Method | Carrier | Regime | Level |
|--------|---------|--------|-------|
| GvU | unified multimodal model + token-level intrinsic text-image reward + GRPO/LoRA | training-time self-supervised UMM generation post-training | internal understanding-to-generation alignment |

| Collection status | Value |
|-------------------|-------|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, or current `0524_new_collection` |
| Screening source | Chrome MCP arXiv page used only for candidate discovery |
| PDF | `0524_new_collection/pdfs/2603.06043.pdf` |
| Extracted text | `0524_new_collection/texts/2603.06043.txt` |
| Basis for this note | full downloaded PDF + extracted text; source-audit checked on 2026-05-24 against self-generation pipeline, token-level intrinsic reward, text-only prompt training set, GRPO / LoRA update, and absence of external reward models / image datasets |
| BibTeX | `0524_new_collection/bib/2603.06043.bib` |
| Suggested taxonomy | strict UPT candidate; multimodal Family I / IV bridge |

## 1. UPT Assignment Rationale
GvU is a strong strict UPT candidate that helps extend the MLLM / UMM coverage. It studies the generation–understanding gap of unified multimodal models (UMMs): the model can understand image details but cannot reliably satisfy complex prompts when generating images. The method uses no external reward model or human preference; instead, the model's own understanding branch evaluates whether its generated image supports the original text prompt.

Under the current taxonomy, GvU is a **Family I / IV bridge**:

- Family I: the reward is an prediction-statistic / model-intrinsic text-image alignment probability, drawn from the model's own token-likelihood over the generated image and prompt;
- Family IV: the understanding branch acts as an internal evaluator that guides the generation branch's training.

It has genuine update-bearing training: the paper post-trains a UMM with GRPO and LoRA in a self-supervised RL framework, so B1 passes.

## 2. Boundary Audit

| Check | Judgment | Evidence |
|-------|----------|----------|
| B1 update-bearing | Pass | uses a GRPO-based RL framework and LoRA to train X-Omni / weak base |
| B2 internal signal | Pass | reward comes from the UMM understanding branch's probability of the prompt tokens conditioned on the generated image |
| B3 no external supervision | Pass with prompt-source caveat | training uses only a text-only prompt dataset; no external image resources or human preferences |
| B4 internal judge / scorer | Pass | the evaluator is the same UMM's understanding branch, not an external CLIP / GPT / reward model |

## 3. Method

GvU's self-generation pipeline:

1. Given a text-prompt dataset $D_T$;
2. The UMM generation branch produces image tokens conditioned on the prompt and decodes them into an image via the diffusion head;
3. The UMM understanding branch reads the generated image and the original prompt;
4. Compute the likelihood of the original prompt tokens under the image-conditioned understanding branch;
5. That likelihood is the token-level model-intrinsic reward;
6. For each prompt, generate a group of images and run GRPO group-relative policy updates.

The key point is that the reward does not come from CLIPScore, ImageReward, human preferences, or GPT-4V, but from the same UMM's internal understanding ability. It converts the UMM's understanding capability into a generation training signal.

## 4. Datasets

The training set is 50,000 text-only prompts describing objects, positional relationships, quantities, attributes, and so on. The paper stresses that the self-generation process needs no external image resources.

Evaluation covers:

- T2I generation: GenEval, DPG-Bench, GenEval++;
- visual understanding: POPE, GQA, MMB, SEED, DocVQA, OCRB;
- fine-grained understanding: MMT-Bench sub-tasks.

The main model is X-Omni regular / weak base, trained with the TRL framework and LoRA.

## 5. Evaluation metrics and main results
GvU improves text-image alignment on GenEval, DPG-Bench, and GenEval++. The paper reports GenEval++ rising from 0.282 to 0.404, about a 43.3% improvement; DPG-Bench overall reaches 85.68. As RL steps grow, GenEval / DPG-Bench / GenEval++ keep climbing, suggesting the intrinsic reward can drive progressive self-improvement.

Visual-understanding evaluation also improves modestly, indicating that gains in the generation branch reciprocally strengthen fine-grained visual understanding, although the authors acknowledge the mutual-reinforcement effect remains limited.

## 6. Position in the UPT Survey
Recommended as a multimodal strict UPT candidate with short label `GvU`. Place it in the main table at the Family I / IV bridge:

- Family I label: `prediction-statistic / model-intrinsic text-image alignment reward`;
- Family IV label: `understanding branch as internal evaluator`;
- Regime: training-time UMM post-training;
- Caveat: the training prompts themselves form a text-only prompt dataset, so it is not strictly zero-data; but the reward depends on no external ground truth / verifier / external AI labels.

It is valuable for the survey because it pushes no-external-ground-truth UPT from LLM reasoning into unified multimodal generation.
