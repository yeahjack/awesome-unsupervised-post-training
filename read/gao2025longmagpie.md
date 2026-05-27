# LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions

> **Added to survey on:** 2026-03-11

> **Paper metadata**
> - **arXiv**: 2505.17134v2 [cs.CL] 3 Jun 2025
> - **Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu
> - **Affiliations**: Institute of Information Engineering (CAS), University of Chinese Academy of Sciences, Xiaohongshu Inc, Tsinghua University

| Property | Value |
|---|---|
| Method | LongMagpie |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Semantic |

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | self-generated data batch / iteration round |
| Persistence | full parameter accumulate across synthesis / refinement rounds |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | note-explicit |
| Timing Regime | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis encoding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates run in an offline pre-deployment bootstrapping loop, typically a round-based schedule of "generate data / score / filter / retrain".
- **Serving the current sample or future ones:** synthetic samples or pseudo-targets produced in the current round mainly serve the next training round and the final deployed model, not the immediate inference of a particular test sample.
- **Whether parameters/state accumulate:** parameters accumulate continuously across bootstrapping rounds, with no sample-level reset.
- **Reset boundary:** the `When to Adapt` of such methods is centered on offline iterative bootstrapping rather than online test-time adaptation.

## 1. UPT Assignment Rationale

LongMagpie belongs to **Family III — Self-Generated Target Bootstrapping**, subclass **reasoning / plan / curriculum synthesis**. Core idea: leverage the auto-regressive ability of an aligned long-context LLM to automatically generate document-relevant queries by feeding the document together with the user-turn prefix tokens, then have the model produce its own response — yielding a full instruction triplet (D, Q, R). The whole process relies on no human annotation, seed instruction, or external teacher model; the model self-synthesizes long-context instruction data, which is then used for SFT alignment. This is the textbook "model generates training-target data, then optimizes itself on it" bootstrapping paradigm — a training-time, semantic-level direct optimization.

---

## 2. Problem Addressed

High-quality long-context instruction data is essential for aligning long-context LLMs, but three bottlenecks remain:

1. **Closed data**: open-weight models (Qwen, Llama) keep their long-context instruction datasets private, forming a closed-data barrier.
2. **High annotation cost**: labeling long-context data is far harder than short-context — annotators must read thousands of tokens before writing an instruction, with high time cost and uneven quality.
3. **Limitations of existing synthesis**: template-based methods (Sanh et al., 2021) and seed-question methods (Self-Instruct, Wang et al., 2022) plateau in diversity, scale, and quality; methods like ChatQA and LongAlign expand seed diversity but require complex pipelines and large overhead (10–13× more tokens per instruction).

LongMagpie aims at a **human-annotation-free, seed-free, pipeline-light** scalable scheme for self-synthesizing long-context instruction data.

---

## 3. Method (with figure descriptions)

### 3.1 Core insight: auto-regressive document–query generation

LongMagpie's key observation: instruction-tuned long-context LLMs have already internalized the document-query relationship through training. So when a model receives a document $D$ followed by user-turn prefix tokens $T_{pre}$ (e.g., `<|im_start|>user` or `<|start_header_id|>user`), it auto-regressively produces a query $Q$ semantically relevant to that document:

$$p_M(Q \mid D, T_{pre}) = \prod_{i=1}^{k} p_M(q_i \mid D, T_{pre}, q_{<i})$$

This differs from traditional prompt engineering or instruction-following: the model is not explicitly told to generate a question; it leverages the document-query patterns learned during instruction tuning to do so automatically.

### 3.2 LongMagpie pipeline

> **Figure 1** (paper p.2) shows the full pipeline in two stages:
> - **Stage 1**: feed the document as system prompt, trigger query generation via special user tokens, then have the model produce the response.
> - **Stage 2**: combine the query-response pair with the original document plus additional documents sampled from the corpus to construct challenging multi-document long-instruction data.

#### Step 1: Query and Answer Generation

- **Document preparation**: gather documents from diverse corpora like FineWeb-Edu, covering science, history, literature, technical, with average length ~1.6k tokens.
- **Query generation**: for each document $D$, build input $X = D \oplus T_{pre}$ (with $T_{pre}$ the user-turn prefix tokens), feed into the aligned LLM, and sample to generate query $Q$. By varying sampling parameters, multiple queries per document naturally yield document-query pairs of diverse complexity.
- **Response generation**: for each $(D, Q)$ pair, build a standard instruction prompt (concatenating document, query, assistant prefix) and generate response $R$, completing the instruction triplet $(D, Q, R)$. If one model produces both query and response, the whole process is human-free.
- **Query filtering**: two strategies handle cases where the model continues the document instead of generating a query: (1) **rule-based filtering** keeps queries ending with "?"; (2) **length-based filtering** drops texts longer than 1.5k characters (typically continued descriptive paragraphs).

#### Step 2: Multi-document extension

To raise task diversity and practical utility, the pipeline extends to multi-document settings:

- Randomly sample $x$ documents $\{D_1, \ldots, D_x\}$ as negative documents ($x$ uniform in $[0, n]$; $n=0$ reduces to single-document QA).
- Concatenate with a special separator token (e.g., `<|doc_sep|>`): $D_{multi} = D_1 \oplus \text{<|doc\_sep|>} \oplus \cdots \oplus D_x$.
- Generate query and response on the multi-document context per the same pipeline, yielding triplets $(D_{multi}, Q, R)$ that demand cross-document reasoning.

### 3.3 p-Mix: balancing long-context and short-context capabilities

Training only on long-context data degrades short-context tasks. LongMagpie proposes **p-Mix**:

1. **Short-context prefix**: each training sequence begins with a short-context instruction, mimicking the non-contextual start of generic tasks.
2. **Probabilistic mixing**: thereafter, with probability $P_L$ append a LongMagpie-generated long-context instruction, with probability $1 - P_L$ append another short-context instruction; iterate until approaching the maximum sequence length $L_{max}$.

> **Algorithm 1** (paper p.16) gives the pseudocode: `CONSTRUCTHYBRIDSAMPLE(DS, DL, PL, Lmax, sep)` — start by sampling one item from the short-context set $D_S$, then alternately sample long/short with probability $P_L$ until exceeding $L_{max}$.

p-Mix prevents the model from overfitting to long-context patterns, keeping strong long-context performance while maintaining short-context competitiveness.

---

## 4. Datasets

### Generated data

- **Source model**: Qwen2.5-70B-Instruct
- **Document corpus**: FineWeb-Edu (a subset of educational web content, 1.3 trillion tokens)
- **Scale**: 450k long-context instruction samples (ablations also test 190k)
- **Avg. document length**: ~1.6k tokens

### Comparison datasets

**Long-instruction datasets**:
- **ChatQA (ChatQA2)**: combines multiple sources (LongAlpaca12k, GPT-4 samples from Open Orca, etc.), 1.5M synthetic instructions in total
- **LongAlign**: prompts LLMs to produce QA from long documents

**Short-instruction datasets** (concatenated to target length for long-context fine-tuning):
- **Tulu**: open-source collection based on Llama 3.1
- **Magpie**: self-synthesis using template prefixes
- **UltraChat**: 1.5M multi-turn dialogues

### Training configuration

- **Base model**: Llama-3-8B-NExtLong-512K-Base (long-context continued-pretrained)
- **Batch size**: 4M tokens, 250 steps, 1B tokens total
- **Sequence length**: 64K
- **Optimizer**: AdamW ($\beta_1=0.9$, $\beta_2=0.95$), lr = 2e-5, cosine decay
- **Techniques**: FlashAttention-2, ZeRO, document masking, bfloat16
- **Hardware**: 8× H100, ~10h training

---

## 5. Evaluation metrics and main results
### Evaluation benchmarks

**Long-context evaluation**:
- **HELMET**: assesses long-context models on application-centered tasks, up to 128k tokens.
- **RULER**: fine-grained synthetic tasks with flexible control over sequence length and complexity, identifying bottlenecks beyond retrieval.
- **LongBench-v2**: evaluates extremely long (8k–2M words) understanding, 503 expert-validated questions, 6 categories.

**Short-context evaluation**: HellaSwag, Lambada_OpenAI, ARC-Challenge, ARC-Easy, PIQA, WinoGrande, Logiqa.

### Main results

> **Table 1** (paper p.6) compares LongMagpie with other methods on long/short benchmarks.

| Dataset | HELMET | RULER | LongBench v2 | LongAVG | ShortAVG |
|---|---|---|---|---|---|
| **Short Instruction Data** | | | | | |
| Tulu | 61.93 | 87.92 | 28.4 | 59.42 | 63.90 |
| Magpie | 60.18 | 87.06 | 31.4 | 59.55 | 63.32 |
| UltraChat | 60.55 | 83.85 | 30.4 | 58.27 | 64.43 |
| **Long Instruction Data** | | | | | |
| ChatQA | 60.23 | 89.82 | 30.8 | 60.28 | 63.58 |
| LongAlign | 57.79 | 86.08 | 24.5 | 56.12 | 60.97 |
| **LongMagpie** | **62.10** | **91.17** | **34.4** | **62.56** | 62.37 |
| **p-Mix: Long + Short** | | | | | |
| ChatQA + UltraChat | 60.80 | 87.42 | 31.4 | 59.87 | 64.38 |
| LongAlign + UltraChat | 60.98 | 89.49 | 30.6 | 60.36 | 64.17 |
| **LongMagpie + UltraChat** | **62.11** | 89.70 | 33.0 | **61.60** | **64.10** |

**Key findings:**

1. **Long-context leader**: training only on LongMagpie data already achieves the best on all long-context benchmarks; LongAVG reaches 62.56, +2.28 over ChatQA and +6.44 over LongAlign.
2. **p-Mix balances long and short**: LongMagpie + UltraChat (p-Mix) keeps long-context leadership (LongAVG 61.60) while ShortAVG reaches 64.10 (within 0.33 of the highest), effectively resolving short-task degradation caused by long-context training.

### Ablations

> **Table 2**: effect of multi-document count $n$ — $n=10$ yields the highest LongAVG (62.56); too many documents ($n>20$) push task difficulty beyond model capacity and hurt performance.

> **Table 3**: mixing strategy comparison — p-Mix beats Sequential Mix (60.75/61.89) and Simple Mix (60.90/64.04) on both LongAVG (61.60) and ShortAVG (64.10), best balance.

> **Table 4**: scale effect — going from 190k to 450k samples raises LongAVG from 61.51 to 62.56 (+1.05), showing continued gains from larger high-quality data.

> **Table 5**: source-model size effect — Qwen-2.5-70B yields LongAVG 62.56, far above Qwen-2.5-7B's 59.61; stronger long-context modeling translates into higher-quality synthetic data.

> **Figure 2** (paper p.8): (a) reward-model score distribution shows LongMagpie data quality is markedly higher than ChatQA and LongAlign; (b) pairwise query similarity shows LongMagpie queries are more dissimilar to each other — better diversity.

> **Figure 3** (paper p.9): (a, b) t-SNE embedding visualizations show LongMagpie queries are more dispersed (diverse); (c) long-context performance vs. token consumption plot shows LongMagpie reaches the best long-context performance at an average 1.6k tokens/instruction — 10–13× cheaper than ChatQA/LongAlign — with outstanding sample efficiency.

### p-Mix parameter ablation

> **Table 7** (Appendix): $N_S=1, P_L=0.4$ is the best config (LongAVG 61.60, ShortAVG 64.10). Too-high $P_L$ tilts toward long context and hurts short tasks; too-large $N_S$ (e.g., 30) dilutes the long-context signal with too many short instructions, lowering LongAVG.
