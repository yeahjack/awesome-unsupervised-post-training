# Self-Instruct: Aligning Language Models with Self-Generated Instructions

> **Added to survey on:** 2026-05-26 (adjacent precursor)

## Bibliographic Information
- **Paper:** Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Venue:** ACL 2023 (Volume 1: Long Papers), Toronto, Canada
- **arXiv:** 2212.10560
- **ACL Anthology:** https://aclanthology.org/2023.acl-long.754/
- **Code:** https://github.com/yizhongw/self-instruct

## Audit Update / Caveat
This paper is treated in the UPT survey as an **adjacent (B3-failed, seed-supervised bootstrapping) precursor**, **not strict UPT**. The reason is logged in `adjacent_methods.csv`: `signal_source = human-written seed instructions`, `failed_check = B3`, `decision = precursor of self-generated target bootstrapping`. main.tex §F.3 and §sec:boundary-protocol state explicitly: "Self-Instruct … bootstrap from human-written seeds, failing B3 at the seed stage. They are treated as precursors of self-generated target bootstrapping rather than strict UPT."

## 1. Relation to the UPT Survey

- It belongs to the self-generated target / instruction-self-curation lineage (i.e., strict UPT Family III "Self-Generated Target Bootstrapping, knowledge / instruction self-curation") and is its **direct conceptual precursor**: the model self-generates instructions, which are then used to SFT itself.
- The boundary check it fails is **B3 (no human / external supervision in the loop)**: at bootstrap, **175 hand-written seed tasks** (each with one instruction, one input, one output, written by the authors) are used; these seeds enter the in-context prompt that induces the LLM to generate new instructions, so the seed supervision leaks into the self-training data stream.
- Therefore it is routed to adjacent, not strict UPT. It serves as the direct counterpart to **fully seed-free** instruction-tuning methods inside strict UPT, such as `CYCLE-INSTRUCT` (shen2025cycleinstruct).

## 2. Core Method

- **Input signals:** 175 hand-written seed tasks (initialize the task pool) + a strong base LM to align (the main experiment uses GPT-3 `davinci`, 175B).
- **Output:** ≈**52K instruction-following samples** (≈82K instruction–instance pairs covering ≈52K unique instructions), then used for SFT.
- **Training objective:** standard instruction-tuning SFT (cross-entropy on the model-generated instruction → output).
- **Generation–filtering pipeline (4-step iteration):**
  1. **Instruction generation:** sample 8 seed instructions from the task pool (6 human-written + 2 model-generated) as in-context demonstrations and let GPT-3 generate new instructions.
  2. **Classification-task identification:** with a few-shot prompt, decide whether the instruction is a classification task (this affects the input / output generation order).
  3. **Instance generation:** input-first (for generation tasks) or output-first (for classification tasks) to produce (input, output).
  4. **Filtering:** drop if ROUGE-L similarity to existing instructions exceeds 0.7; filter out items containing images, too-short / too-long items, or items containing specific keywords; surviving items are added back to the task pool.
- **Key point:** the seeds are **the authors' hand-written 175 items**, each a complete (instruction, input, output) triple; the selection pipeline is purely heuristic (ROUGE-L de-duplication + length / keyword filtering), without external-model scoring.

## 3. Datasets and Scale

- **Seed:** 175 hand-written tasks (covering brainstorming, classification, generation, open QA, closed QA, rewriting, summarization, etc.).
- **Generated pool (final):** ≈52K instructions, ≈82K (instruction, input, output) instances.
- **Released artifact:** `GPT3SELF-INST` (GPT-3 davinci SFT-trained on the self-instruct data) and the 52K dataset.

## 4. Experimental Results (Brief)

- **SUPER-NaturalInstructions (SuperNI) evaluation, 119 tasks:** plain GPT-3 (davinci) → after self-instruct SFT, ROUGE-L jumps substantially, narrowing the gap to methods fine-tuned on human-curated SuperNI data (`T0`, `Tk-Instruct`) by roughly 5 points, while relying solely on self-generated data.
- **Human evaluation on 252 author-written user-oriented instructions:** `GPT3SELF-INST` clearly beats vanilla GPT-3 and is roughly on par with (or close to) `InstructGPT-001` (OpenAI's commercial model) overall.
- **Ablation:** data diversity (the ROUGE-L de-dup threshold) is critical; between scale and quality, scale dominates (the generated 52K is fairly noisy, but quantity beats refinement).

## 5. UPT Survey Positioning

- **Relation to strict UPT Family III:** Self-Instruct is the foundational work of the "model-generated instruction → self-train" paradigm; the idea leads directly into Family III's instruction / knowledge-self-curation lineage (CYCLE-INSTRUCT, the IFT stage of Self-Rewarding LM, LongMagpie, etc.).
- **Why it does not count as strict UPT:** the bootstrap starting point is not an unlabeled corpus but **175 carefully written human seeds**—their coverage, style, and quality directly shape the distribution of all downstream self-generated data, so the supervision signal is essentially human, failing boundary check **B3**.
- **Influence on later work:**
  1. Stanford **Alpaca** directly reuses the Self-Instruct pipeline.
  2. **Vicuna**, **WizardLM (Evol-Instruct)**, **Orca**, **Baize**, and nearly all early open-source instruction-tuning works follow or extend this seed-then-self-generate framework.
  3. CYCLE-INSTRUCT, Magpie, and other strict-UPT methods take it as the **counterpoint** they need to break the seed dependency, showing that instruction tuning can proceed without human seeds.
- **One-line characterization:** Self-Instruct is conceptually the founding work of self-generated-target bootstrapping; under the boundary protocol it is classified as adjacent precursor due to its human-seed dependency—just one step away from strict UPT.
