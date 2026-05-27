# Ambiguous Case Decision Log

This file mirrors Appendix C (`app:reliability`) of the main paper. Each entry
records a borderline method, the two candidate categories, the rule that
resolved the assignment, and the page/equation in the source paper that
supports the decision.

## 1. Family II vs. Family III

### LRM Self-Train (`shi2025lrmselftrain`)
- **Candidates:** F2 (multi-sample-statistic reward) vs. F3 (synthetic-target SFT).
- **Rule applied:** update-object rule — gradient is computed against kept
  solutions, not the consensus statistic itself.
- **Decision:** **F3**.
- **Contrast:** TTRL, RoiRL, and methods whose gradient is taken directly
  against \(r=\mathbf{1}[y=\mathrm{maj}]\) remain in F2.
- **Note path:** `read/shi2025lrmselftrain.md`.

## 2. Family III vs. Family IV

### Confident ST (`wang2025confidentreasoning`)
- **Candidates:** F3 (confidence-as-selection-mask) vs. F4 (confidence-as-scalar-reward).
- **Rule applied:** update-object rule — score acts as a selection mask before SFT,
  so gradient is computed against the kept generations.
- **Decision:** **F3** in current paper; flagged as boundary-with-F4.
- **Note path:** `read/wang2025confidentreasoning.md`.

### RLSF (`vanniekerk2025rlsf`)
- **Candidates:** F3 vs. F4.
- **Rule applied:** same as above — confidence ranks chains-of-thought and
  produces preference pairs; DPO step is the update operator.
- **Decision:** **F3**; boundary-with-F4 noted.
- **Note path:** `read/vanniekerk2025rlsf.md`.

## 3. Strict UPT vs. Adjacent

### ECHO (`zhao2026echo`) and SPINE (`wu2025spine`)
- **Candidates:** strict F2 vs. adjacent (consensus + intrinsic term).
- **Rule applied:** consensus is the reward; intrinsic term is a within-family
  modulation (advantage shaping for ECHO, token-mask for SPINE).
- **Decision:** **strict F2**.
- **Note paths:** `read/zhao2026echo.md`, `read/wu2025spine.md`.

### EM-INF (`agarwal2025unreasonable`)
- **Candidates:** strict F1 vs. adjacent.
- **Rule applied:** B1 — no explicit update to parameters, adapters, memories,
  or persistent local state (inference-time entropy descent over logits or
  hidden states only).
- **Decision:** **adjacent (B1)**.

### Full CSR system (`zhou2024csr`)
- **Candidates:** strict F4 (vision self-judge) vs. adjacent.
- **Rule applied:** B4 — one term of the reward is a CLIP-derived visual
  relevance score from a frozen external encoder, not derived from the same
  model lineage.
- **Decision:** **adjacent (B4)**.

## 4. Family I/IV bridge cases

### SUDER (`hong2025suderselfimprovingunifiedlarge`) and GvU (`pan2026learninggenerateunderstandingunderstandingdriven`)
- **Bridge:** unified multimodal systems with both prediction-statistic and
  internal-evaluator signals.
- **Rule applied:** update-object rule routes each by the **primary**
  update-signal object that is differentiated against.
- **Decision:**
  - SUDER: reverse-task likelihood as a prediction-statistic reward → **F1**.
  - GvU: cross-branch evaluator-driven update → **F4**.
- **Marker:** `^\ddagger` in Tables 1 and 4 of main.tex.
