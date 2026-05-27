# TTCS: Test-Time Curriculum Synthesis for Self-Evolving

> **Added to survey on:** 2026-04-16

**Paper:** TTCS: Test-Time Curriculum Synthesis for Self-Evolving
**Authors:** Chengyi Yang et al.
**arXiv:** 2601.22628
**Venue:** arXiv 2026
**Code:** https://github.com/XMUDeepLIT/TTCS

| Method | Carrier | Regime | Level |
|------|----------------|-------------|------------|
| TTCS | Policy Opt. / curriculum synthesis | test-time | Semantic / synthetic target |

| When to Adapt | Full-Cohort Transductive Adaptation: test-time curriculum synthesis |
|---|---|
| Trigger Unit | test questions plus synthesizer-generated variants |
| Persistence | solver and synthesizer policies co-evolve across test-time training |
| Inference Coupling | synthesize easier variants, update solver with self-consistency reward, then evaluate |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Evaluation Boundary |
| Evidence Status | paper-explicit |
| Timing Regime | Full-Cohort Transductive Adaptation |
| Visibility Scope | Full target cohort |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Full-Cohort Transductive Adaptation`; `Visibility Scope=Full target cohort`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Evaluation Boundary`.
- **When the update is triggered:** curriculum variants are constructed on the test questions and the synthesizer / solver are iteratively updated with self-consistency rewards.
- **Serving the current sample or future ones:** generated variants and solver updates serve the entire test-time training run and the subsequent evaluation.
- **Whether parameters/state accumulate:** both policies accumulate updates throughout the co-evolution run.
- **Reset boundary:** evaluation / benchmark boundary.

## 1. UPT Assignment Rationale

TTCS is strict UPT, and it was a known gap previously cited in the main document but missing the `read/TTCS.md` note. It uses only test questions, synthesizes targeted variants, and updates the solver via self-consistency rewards; no ground-truth labels are required.

Under the dominant-artifact rule, TTCS fits **Family III: Self-Generated Target Bootstrapping** better, because consensus / self-consistency primarily serves to generate curriculum variants and pseudo-targets rather than acting as the final reward itself.

## 2. Problem Addressed

Hard test questions are often too difficult — direct majority pseudo-labels are unreliable — and a small test set makes continuous online updates unstable. TTCS synthesizes question variants around the capability frontier to stabilize test-time self-evolution.

## 3. Method

TTCS initializes two same-lineage policies:

- **Question synthesizer:** generates progressively challenging / tractable variants of the test question, updated by a question-quality reward.
- **Reasoning solver:** generates multiple responses on the original test questions and synthetic variants, obtains pseudo-labels via a self-consistency reward, and updates with GRPO.

Solver feedback then guides the synthesizer to produce questions suited to the current ability, forming a co-evolving curriculum.

## 4. Datasets

The paper evaluates on multiple math-reasoning benchmarks, including challenging math datasets like AIME24/25, and tests general-domain transfer.

## 5. Evaluation metrics and main results
Main reported metrics include mean@32 and greedy-decoding accuracy / pass@1. The paper claims TTCS consistently improves the reasoning ability of different backbones on multiple math benchmarks and general-domain reasoning tasks.

## 6. Position in the UPT Survey
Must be added to the `read/` evidence layer. If TTCS is already cited in the main taxonomy, this note fills in the evidence; if the main table is later updated, TTCS should remain in Family III.
