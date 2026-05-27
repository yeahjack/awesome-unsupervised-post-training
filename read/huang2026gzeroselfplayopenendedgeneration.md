# G-Zero: Self-Play for Open-Ended Generation from Zero Data

**Paper:** G-Zero: Self-Play for Open-Ended Generation from Zero Data
**Authors:** Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, Jiaxin Huang
**arXiv:** 2605.09959
**Venue:** arXiv 2026
**Code:** https://github.com/Chengsong-Huang/G-Zero

| Method | Carrier | Regime | Level |
|---|---|---|---|
| G-Zero | GRPO + DPO | training-time self-play | Semantic |

| Collection status | Value |
|---|---|
| Duplicate check | Not found in existing `read/`, `references.bib`, or `download_manifest.csv` |
| PDF | `0524_new_collection/pdfs/2605.09959.pdf` |
| Extracted text | `0524_new_collection/texts/2605.09959.txt` |
| Basis for this note | complete downloaded PDF + extracted full text; source-audit rechecked on 2026-05-24 against Hint-delta, Proposer GRPO, Generator DPO, open-ended verifier-free setting, and external-judge caveats |
| BibTeX | `0524_new_collection/bib/2605.09959.bib` |
| Suggested taxonomy | strict UPT candidate; Family III |

## 1. UPT Assignment Rationale

G-Zero explicitly targets open-ended / unverifiable domains, aiming to self-evolve without external verifiers, LLM-as-a-judge, or ground-truth answers. Its training loop has two model roles: the Proposer generates question-hint pairs; the Generator generates hint-free and hint-conditioned responses, and internalizes hint-guided improvements via DPO.

Although the Proposer's reward is the Hint-$\delta$ distributional-shift signal, what the Generator is actually trained on are preference pairs constructed from self-generated hints: the hint-conditioned response is chosen and the unassisted response is rejected. By the dominant update-object rule, it is best placed in **Family III: Self-Generated Target Bootstrapping**, with the subclass viewable as self-generated hints / preference pairs.

## 2. Boundary check

| Check | Judgment | Evidence |
|---|---|---|
| B1 update-bearing | Pass | The Proposer is updated by GRPO, the Generator by DPO |
| B2 internal signal | Pass | Hint-$\delta$ is computed from the Generator's own log-probability shift; preference pairs come from the Generator's dual responses |
| B3 no external supervision | Pass | The paper stresses no external verifiers or judges |
| B4 internal judge/scorer | Pass with role caveat | There is no judge, but the dual Proposer/Generator roles need confirmation that they share the same lineage / base setup |

## 3. Method

G-Zero's core is **Hint-$\delta$**:

- The Generator first produces an unassisted response to a query $q$;
- The Proposer produces a hint $h$;
- Hint-$\delta$ measures the Generator's predictive-distribution shift on the original unassisted response after the hint is added;
- The Proposer uses this intrinsic reward to learn to produce query-hint pairs that expose the Generator's blind spots;
- For the same query, the Generator produces a hint-conditioned and an unassisted response, and DPO trains it to prefer the hint-conditioned response.

The paper also applies a $\delta$-filter to DPO data to avoid high-$\delta$ pairs that cause answer leakage or excessive KL shift.

## 4. Datasets

The paper emphasizes zero-data / open-ended self-play, but evaluation covers two categories:

- Open-ended tasks: AlpacaEval, IFEval, chat / instruction following;
- Verifiable transfer tasks: AIME 2024/2025, math/code, etc.

Appendix prompts show the Proposer can generate analysis, argument, text, dataset description, product, and other open-ended task types.

## 5. Evaluation metrics and main results
The paper reports that after self-play iterations, G-Zero improves both open-ended tasks and math transfer (e.g., AlpacaEval and AIME 2025 gains). The key claim is that improvement stems from internalized logical depth on open-ended non-verifiable tasks rather than domain-specific memorization.

## 6. Position in the UPT Survey
Recommended as a new Family III candidate, alongside `R-Zero`, `TTCS`, `DiSCTT`, `LongPO`, and other self-generated target / preference / curriculum methods. Its contribution is extending zero-data self-play from verifiable reasoning to open-ended generation while avoiding LLM-as-a-judge.

Suggested main-table short label: `G-Zero`.
Suggested family: `Self-Generated Target Bootstrapping`.
Suggested timing: `Offline Corpus UPT` / training-time self-play.
Needs further verification: whether the Proposer and Generator stay within the same model lineage; if they use distinct pretrained bases, a caveat should be added in the main document.
