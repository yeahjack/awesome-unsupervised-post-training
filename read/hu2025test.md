# TLM: Test-Time Learning for Large Language Models

> **Added to survey on:** 2026-03-11

**Paper:** Test-Time Learning for Large Language Models
**arXiv:** 2505.20633
**Method:** TLM | **Carrier:** Direct Opt. | **Regime:** Test-time | **Level:** Token

| When to Adapt | multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation |
|---|---|
| Trigger Unit | offline: all test samples; online: arriving test sample / batch |
| Persistence | LoRA accumulates within each offline or online run; the online setting has no per-sample reset |
| Inference Coupling | offline: adapt-before-test; online: interleaved infer-and-adapt |
| Input Visibility | Multi-protocol: Offline + Online |
| Update Persistence | Cumulative |
| Reset Boundary | Multi-protocol: Evaluation Boundary + No Immediate Reset |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation |
| Visibility Scope | Multi-protocol: Full target cohort + Streaming prefix only |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** this paper contains multiple protocol entries: `Timing Regime=Multi-protocol: Full-Cohort Transductive Adaptation + Streaming Continual Adaptation`; `Visibility Scope=Multi-protocol: Full target cohort + Streaming prefix only`.
- **Two-axis coding:** `Input Visibility=Multi-protocol: Offline + Online`; `Update Persistence=Cumulative`; `Reset Boundary=Multi-protocol: Evaluation Boundary + No Immediate Reset`.

| Protocol Entry | Timing Regime | Visibility Scope | Input Visibility | Update Persistence | Reset Boundary | Note |
|---|---|---|---|---|---|---|
| TLM / offline full-test-set setting | Full-Cohort Transductive Adaptation | Full target cohort | Offline | Cumulative | Evaluation Boundary | First see the entire test set, then update once, then run the full evaluation. |
| TLM / online stream setting | Streaming Continual Adaptation | Streaming prefix only | Online | Cumulative | No Immediate Reset | Samples or batches arrive sequentially; updates accumulate forward into subsequent stream items. |

- **When the update is triggered:** the paper's experiments explicitly include both offline and online settings. In the offline setting, all test data are handled at once: parameters are updated using all test samples first, then evaluation begins. In the online setting, test data arrive sequentially and the model is updated after each sample or batch.
- **Whose sample it serves:** the offline version serves subsequent evaluation over the full test set; in the online version, updates from the current sample/batch primarily serve later stream items.
- **Whether parameters/state accumulate:** both settings update via LoRA; the online version does no per-sample reset but accumulates within the stream.
- **Reset boundary:** the offline version is bounded at the evaluation boundary; the online version at the stream/run boundary rather than the sample boundary.

## 1. UPT Assignment Rationale

TLM belongs to **Family I (Prediction-Statistic Optimization)**, specifically the predictive-likelihood-minimization sub-class. Its core mechanism is to continually minimize input perplexity (i.e. NLL) on unlabeled test samples at test time, performing real parameter updates via LoRA. The process relies on no external annotation, external verifier, or external AI label:

- **Signal source:** entirely the model's own intrinsic statistics on the input token sequence (input perplexity / NLL).
- **Objective:** $\min_{\Theta} S(x)\mathcal{P}(x;\Theta)$, where $\mathcal{P}(x;\Theta)$ is the input perplexity and $S(x)$ is a perplexity-based sample-selection weight.
- **Parameter updates:** lightweight gradient updates via LoRA — direct optimization.
- **No external supervision:** no knowledge base, no training data, no annotations, no external model; only unlabeled test data from the test stream.

---

## 2. Problem Addressed

LLMs at deployment face **distribution shift**, manifested in two forms:

1. **Vertical Domain Shift:** test data contain domain-specific terminology (medical, legal, financial) on which the model was not sufficiently trained, leading to drops in performance.
2. **Non-Specific Distributional Shift:** changing user intents and language diversity (dialects, slang, etc.) drive the test distribution away from the training distribution.

Existing methods have limitations:
- **Fine-tuning** requires substantial annotated data and is impractical in dynamic environments.
- **RAG** depends on the quality of an external knowledge base and adds retrieval latency.
- **Test-Time Adaptation (TTA)** methods (Tent, EATA, etc.) use entropy minimization, which ignores the LLM's autoregressive nature and is ineffective on LLMs.
- **Test-Time Training (TTT)** assumes access to training data or a knowledge base, which is often unavailable in real settings.

---

## 3. Method

TLM consists of three core components:

### 3.1 Input Perplexity Minimization

Key observation (**Observation 1**): the LLM's input perplexity $\mathcal{P}(x;\Theta)$ and output perplexity $\mathcal{P}(y|x;\Theta)$ trend together. Therefore minimizing input perplexity indirectly lowers output perplexity and improves generation quality.

Theoretical justification: a first-order Taylor analysis shows that when the cross-gradient term $\langle \nabla_x, \nabla_y \rangle \geq 0$ (which holds for 98.75% of batches in the experiments), minimizing input perplexity guarantees that the output log-probability does not decrease.

### 3.2 Sample Efficient Learning Strategy

Key observation (**Observation 2**): high-perplexity samples contribute more to the update, while low-perplexity samples can cause overfitting.

Design an active sample-selection score:

$$S(x) = \lambda \cdot e^{|\log \mathcal{P}(x;\Theta) - \log \mathcal{P}_0|} \cdot \mathbb{I}_{\{\mathcal{P}(x;\Theta) > \mathcal{P}_0\}}(\mathbf{x})$$

where $\mathcal{P}_0$ is a preset threshold (set to $e^3$ in experiments) and $\lambda$ is a scaling coefficient. This filters out low-perplexity samples and gives higher weight to high-perplexity samples — no extra gradient computation needed.

### 3.3 Lightweight Parameter Updates via LoRA

Key observation (**Observation 3**): LoRA prevents catastrophic forgetting better than full-parameter updates.

Final objective:

$$\min_{\Delta\Theta} S(x)\mathcal{P}(x; \Theta + \Delta\Theta)$$

where $\Delta\Theta = \mathcal{B}\mathcal{A}$, $\mathcal{B}$ is initialized to zero, $\mathcal{A}$ to a random Gaussian; only $\Delta\Theta$ is updated.

### Algorithm

1. Initialize LoRA parameters $\Delta\Theta$ and attach them to the pretrained LLM.
2. For each batch: compute the prediction $\bar{y}$, compute the sample-selection score $S(x)$, and update the LoRA parameters with the weighted perplexity loss.
3. Output answers for all test samples.

---

## 4. Datasets

The paper assembles a comprehensive benchmark **AdaptEval** with three categories:

### DomainBench (vertical-domain knowledge)
- **Geography** — geography-domain QA.
- **Agriculture** — agriculture-domain QA.
- **Medicine** — medical-domain QA.
- **Finance** — finance-domain QA.

### InstructionBench (instruction following)
- **Alpaca-GPT4** — general instruction following.
- **Dolly** — general instruction following.
- **InstructionWild** — general instruction following.

### ReasoningBench (reasoning)
- **GSM8K** — mathematical reasoning.
- **MetaMath** — mathematical reasoning.
- **Logiqa** — logical reasoning.

### Models tested
- Llama3.2-3B-Instruct.
- Llama3-8B-Instruct.
- Llama2-13B-Chat.
- Qwen2.5-7B-Instruct.

---

## 5. Evaluation metrics and main results

### Metrics
- **ROUGE-Lsum (R-Lsum):** for DomainBench and InstructionBench.
- **Exact Match (EM):** for ReasoningBench.

### Main results

**DomainBench (Table 2):** TLM consistently beats the base LLM and all baselines (Tent, EATA, COME) across all four domain datasets, with gains of at least 20%. For example:
- Llama3.2-3B-Instruct Geography: 0.2395 → **0.2893** (+20.79%).
- Qwen2.5-7B-Instruct Agriculture: 0.1203 → **0.1652** (+37.32%).

**InstructionBench (Table 2):** TLM is best on all instruction-following datasets. For example:
- Llama3.2-3B-Instruct Alpaca-GPT4: 0.3564 → **0.3883** (+13.91%).

**ReasoningBench (Table 3):** TLM substantially outperforms the base LLM and all baselines on all reasoning datasets. For example:
- Llama3-8B-Instruct GSM8K: 0.7610 → **0.8074** (+6.10%).

### Ablation findings
- **Effectiveness of input perplexity minimization:** perplexity minimization alone (without SEL) yields an 83.9% relative gain on Medicine.
- **Effectiveness of Sample Efficient Learning:** SEL adds about 2% more performance while reducing the training-data volume by about 5%.
- **Online setting:** updating parameters every 100 samples reduces backward passes by 69.7% (5000 → 1514) while preserving the performance gains.
- **Quantized models (NF4):** on 4-bit-quantized Llama3-8B-Instruct, TLM still improves all four DomainBench datasets by at least 25%.
