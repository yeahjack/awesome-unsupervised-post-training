# SUDER: Self-Improving Unified Large Multimodal Models with Dual Self-Rewards

**Paper:** SUDER: Self-Improving Unified Large Multimodal Models for Understanding and Generation with Dual Self-Rewards
**Authors:** Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan
**arXiv:** 2506.07963
**Venue:** arXiv 2025
**Code:** not found in the extracted text.

| Method | Carrier | Regime | Level |
|---|---|---|---|
| SUDER | unified LMM + dual likelihood self-reward + SimPO/GRPO | training-time self-supervised multimodal post-training | reverse-task likelihood between understanding and generation |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, `download_manifest.csv`, main taxonomy files, `acl-style-files`, or current `0524_new_collection`. |
| Screening source | Chrome MCP arXiv search/page text used only for candidate discovery. |
| PDF | `0524_new_collection/pdfs/2506.07963.pdf` |
| Extracted text | `0524_new_collection/texts/2506.07963.txt` |
| Basis for this note | Complete downloaded PDF + extracted full text; source audit performed on 2026-05-24 against dual self-reward equations, reverse likelihood computation, SimPO/GRPO optimization, training data, external-supervision claims, and CLIP comparison. |
| BibTeX | `0524_new_collection/bib/2506.07963.bib` |
| Suggested taxonomy | strict UPT candidate; Family I / IV bridge for dual likelihood self-reward in unified multimodal models. |

## 1. UPT Assignment Rationale

SUDER is a fairly clean strict UPT candidate. It targets unified LMMs, jointly optimizing visual understanding and visual generation. The core signal is not human preference, an external reward model, or a ground-truth answer; it is the model's own **dual likelihood** in the understanding/generation directions: after sampling multiple candidate outputs, the input and output are swapped, and the same model computes the likelihood of the original input given the candidate output as the self-reward.

Under the current taxonomy, it fits the **Family I / IV bridge**:

- Family I: the reward is the model's own conditional likelihood / prediction statistic.
- Family IV: understanding and generation serve as each other's internal evaluator, providing self-reward.
- Multimodal scope: directly strengthens the UMM / MLLM generation–understanding alignment direction.

## 2. Boundary checks

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | Uses online SimPO and GRPO to post-train Janus-Pro-1B/7B. |
| B2 internal signal | Pass | The reward comes from the same unified LMM's reverse-task likelihood. |
| B3 no external supervision | Pass with data-source caveat | Uses T2I-CompBench prompts and JourneyDB / COCO images, but no paired image-text annotation is used as a training signal. |
| B4 internal judge/scorer | Pass | No external reward model; CLIP appears only as a contrast experiment, not as SUDER's training signal. |

## 3. Method

SUDER exploits the bidirectional relationship between understanding and generation:

1. For visual understanding, the model samples multiple textual descriptions given an image.
2. Each description is then used as a condition, and the likelihood of generating the original image tokens is computed.
3. A higher likelihood means the description better explains the image.
4. For text-to-image generation, the model samples multiple images given a text prompt.
5. Each image is then used as a condition, and the likelihood of generating the original text prompt is computed.
6. A higher likelihood means the generated image better matches the text semantics.

These reverse likelihoods form the Dual Self-Reward (DSR). SUDER can use SimPO to pick high-/low-reward samples for preference optimization, or use GRPO for group-relative policy optimization. The paper also compares with a CLIP reward, concluding that DSR improves both generation and understanding while CLIP cannot.

## 4. Datasets

Training data includes:

- Text-to-image generation: T2I-CompBench training set, about 5,600 text prompts.
- Visual understanding: about 2,800 images randomly sampled from JourneyDB and COCO118K / COCO Caption derivatives.

The paper explicitly states that the image and text training data are non-parallel; annotated image-text pairs are not used. The main backbones are Janus-Pro-1B and Janus-Pro-7B.

## 5. Evaluation metrics and main results

Generation is evaluated on T2I-CompBench and GenEval; understanding is evaluated on HallusionBench, LLaVABench, POPE, MMBench, SEEDBench-IMG. SUDER substantially improves T2I-CompBench on Janus-Pro-7B; the paper reports an average gain of about 11.68% on T2I-CompBench and about 5% on GenEval. On understanding tasks, LLaVABench also improves notably.

Ablations show that optimizing generation or understanding alone is less stable than joint optimization; SimPO is slightly better than GRPO, but both work with DSR. The comparison with CLIP reward shows that an external CLIP reward may improve generation but cannot equally improve understanding; DSR fits the bidirectional self-improvement goal of a unified LMM.

## 6. Position in the UPT Survey

Suggested short label: `SUDER`.

Can serve as a strict UPT multimodal row; we recommend listing it as a Family I / IV bridge:

SUDER computes self-rewards by reversing understanding/generation input-output pairs and scoring the original input under the same unified LMM, enabling self-supervised SimPO/GRPO post-training without external reward models or paired annotations.

It is adjacent to methods like `GvU` and `SRUM` family, but more strictly Family-I-grounded than methods that rely on external grounding / segmentation tools, because its primary reward comes entirely from the model's own reverse likelihood.
