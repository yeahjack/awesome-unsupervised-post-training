# Self-Alignment with Instruction Backtranslation (Humpback)

> **Added to survey on:** 2026-05-26 (adjacent precursor)

## Bibliographic Information
- **Paper:** Self-Alignment with Instruction Backtranslation
- **Authors:** Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis
- **Venue:** arXiv 2308.06259 (Aug 2023); ICLR 2024
- **arXiv:** 2308.06259
- **Project name:** the model is called **Humpback**

## Audit Update / Caveat
This paper is treated in the UPT survey as an **adjacent (B3-failed, seed-supervised bootstrapping) precursor**, **not strict UPT**. `adjacent_methods.csv` records `signal_source = human-written seed pairs`, `failed_check = B3`, `decision = precursor of self-generated target bootstrapping`. main.tex §F.3: "instruction-backtranslation pipelines bootstrap from human-written seeds, failing B3 at the seed stage."

## 1. Relation to the UPT Survey

- It belongs to the self-generated-instruction (instruction-self-curation) line of work and shares its conceptual lineage with strict UPT Family III: treat unlabeled web text as the "answer" and **generate the instruction backwards**, yielding self-synthesized (instruction, response) pairs.
- The boundary check it fails is **B3**: the starting backward model ("`M_yx`") must be SFT-trained on **3,200 human (instruction, output) seeds** before it can produce usable backward annotations; this seed supervision then propagates to all self-curated data.
- Therefore it is routed to adjacent. It is one of the paradigms that fully seed-free instruction-tuning (e.g., CYCLE-INSTRUCT) sets out to break.

## 2. Core Method

- **Inputs:**
  1. a small **seed set:** 3,200 (instruction, output) pairs from OpenAssistant (OASST1) human-annotated first-turn conversations;
  2. a large-scale **unlabeled web corpus:** 502K candidate English passages extracted from ClueWeb as "potential responses";
  3. base LM: LLaMA-1 65B.
- **Output:** a high-quality self-curated (instruction, response) pair set (≈51K items at the paper's A5 stage); used to SFT the **Humpback** model.
- **Training objective:** standard SFT (cross-entropy).
- **Pipeline (self-augment + self-curate, two stages):**
  1. **Self-augment (backtranslation):** SFT a backward model `M_yx` (output → instruction) on the 3,200 seeds. Apply it to the 502K unlabeled web texts to generate one candidate instruction per passage, producing the `A_0` set.
  2. **Self-curate (iterative quality scoring):** SFT a forward scoring model `M_0` (essentially a directly instruction-tuned LLaMA) on the same 3,200 seeds; use a 5-point Likert prompt to let it score every (instruction, response) item in `A_0`; keep only items scored ≥ 5, the high-quality subset `A_1^{(5)}`.
  3. **Iterate:** train a stronger forward model `M_1` on seed ∪ `A_1^{(5)}`, score again to obtain `A_2^{(5)}`, and iterate twice. The final training data is seed + `A_2^{(5)}`.
- **Key points:**
  - **Seed nature:** 3,200 OASST1 human conversations, real human-written supervision—this is the root cause for failing B3.
  - **The self-curate scorer** is also SFT-trained from the seeds, so the filtering signal indirectly comes from the seeds.
  - **Web text** itself is unlabeled but is treated as a "high-quality response", essentially mapping seed style onto the web corpus.

## 3. Datasets and Scale

- **Seed:** 3,200 (instruction, output) pairs (high-quality OASST1 turns).
- **Unlabeled corpus:** 502K ClueWeb passages (after heuristic filtering).
- **Final A5 (score ≥ 5) curated set:** ≈51K items.
- **Base model:** LLaMA-1 65B (results also reported for 7B / 33B).

## 4. Experimental Results (Brief)

- **AlpacaEval (vs. text-davinci-003):** Humpback 65B reaches a win rate of **83.7%**, beating the same-period open-source instruction-tuned models LIMA 65B, Guanaco 65B, Vicuna 13B, etc.
- **Scaling behavior:** AlpacaEval win rate rises monotonically as the seed-augmented self-curated data grows; growing only the seed pool, or only the unfiltered self-augmented data, saturates earlier—showing that the **self-curation scoring** is the key.
- **Ablations:**
  - skip self-curation (SFT directly on A_0) → clear performance drop;
  - skip iteration (single round) → also a drop.

## 5. UPT Survey Positioning

- **Relation to strict UPT Family III:** Instruction Backtranslation engineers the idea of "self-generating instructions over unlabeled web text"; it is the **direct technical precursor** of CYCLE-INSTRUCT, Magpie, and similar strict-UPT methods—those later works make it fully seed-free (CYCLE-INSTRUCT replaces the seed-trained backward model with a single-model dual-head plus cycle-consistency).
- **Why it does not count as strict UPT:**
  1. the backward model `M_yx` must first be SFT-trained on 3,200 human OASST1 pairs, otherwise the backward-annotation ability does not exist;
  2. the self-curate scorer also stems from seed SFT; both of the pipeline's key "self-" components depend on human seeds, violating B3.
- **Influence on subsequent self-generated-target methods:**
  1. Demonstrates the feasibility of "treating large-scale unlabeled web text as response and letting the model generate the instruction backwards";
  2. The proposed **self-curation via Likert-scale self-rating** is an early form of later self-rewarding / self-judging methods (Self-Rewarding LM, Meta-Rewarding, etc.);
  3. Subsequent fully seed-free work (CYCLE-INSTRUCT and others) almost always cites Humpback as the "seed-dependent" counterpoint in the motivation paragraph.
- **One-line characterization:** Humpback conceptually anticipates the instruction-self-curation and self-rating themes within strict UPT, but is classified as adjacent precursor because both the backward model and the self-rater require human-seed SFT.
