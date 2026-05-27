# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

> **Added to survey on:** 2026-03-11

**Paper:** Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
**arXiv:** 2403.09629
**Venue:** arXiv 2024 (March 2024)
**Authors:** Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
**Affiliations:** Stanford University, Notbad AI Inc

| Property | Value |
|---|---|
| Method | Quiet-STaR |
| Carrier | Direct Opt. |
| Regime | Training-time |
| Level | Traj. |

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

**Family III — Self-Generated Target Bootstrapping, subclass: rationale / critique / latent-thought self-training**

Quiet-STaR belongs to Unsupervised Post-Training (UPT), specifically to the latent-thought self-training subclass of Family III, for the following reasons:

- **No ground-truth supervision:** training uses no labels from any annotated dataset (no QA pairs, no human rationales); it performs continued pretraining only on general internet text corpora (OpenWebMath, C4). The learning signal comes entirely from the language-modeling objective (the ability to predict future text), with no external reward model or human preference annotation.
- **Latent-thought self-training:** the model generates an implicit rationale (latent thought) at each token position; these thoughts do not appear in the final output ("quietly") but serve as internal reasoning to guide next-token prediction. The model self-evaluates which thoughts help future-text prediction via REINFORCE and self-trains on this signal — exactly the core paradigm of self-generated target bootstrapping: the model generates its own training targets (thoughts) and uses them to improve itself.
- **Carrier — Direct Opt.:** Quiet-STaR directly optimizes language-model parameters (LM weights, `<|startofthought|>` / `<|endofthought|>` meta-token embeddings, and the mixing head); there is no separate reward model or verifier; the reward signal comes directly from differences in language-modeling loss.
- **Level — Traj.:** the optimization unit is a full thought trajectory (a multi-token rationale sequence), not a single token or a semantic-level judgment. REINFORCE uses the quality of the entire thought sequence (its help to subsequent-text prediction) as reward.

Compared with STaR, Quiet-STaR generalizes from specific QA tasks to arbitrary text (unsupervised) and from explicit reasoning to implicit latent thought, making it a general unsupervised self-training paradigm.

---

## 2. Problem Addressed

### Core motivation

People often "think before they speak" in writing and conversation — reasoning is implicit in nearly all text, e.g., derivation steps not written down in math proofs, theory-of-mind not made explicit in dialogue. However, existing reasoning methods (e.g., chain-of-thought prompting, STaR) are confined to specific QA datasets and depend on carefully curated annotated data.

### Limitations of existing approaches

1. **STaR (Self-Taught Reasoner):** trains a model on QA datasets to generate rationales, keeping only those rationales that lead to the correct answer. This heavily depends on ground-truth answers and is limited to specific task types.
2. **Human-annotated reasoning traces:** expensive, unscalable, and off-policy (the annotator's reasoning distribution mismatches the model's own).
3. **Chain-of-thought prompting:** used only at inference, does not change model parameters, and requires an explicit triggering prompt.
4. **Pause tokens:** a single pause token provides limited thinking space; Goyal et al. (2023) found that pause-token fine-tuning has limited effect and extra tokens can even hurt performance.

### Problem Quiet-STaR aims to solve

**How can a language model learn to generate implicit rationales (latent thoughts) on general text without any ground truth, thereby comprehensively improving reasoning?** Three key challenges must be addressed:
1. Generating a continuation for every token is computationally expensive;
2. The model initially does not know how to generate or use internal thoughts;
3. Evaluating thought quality should not rely only on the immediately next token but on multiple future tokens.

---

## 3. Method

Quiet-STaR's core idea is to generalize STaR from specific QA tasks to general text: after each token, generate a rationale to help predict subsequent text, and use REINFORCE to reward useful thoughts. The algorithm has three main steps: **Think → Talk → Learn**.

### 3.1 Overall framework (Algorithm 1)

**Input:** a pretrained language model $\theta_0$, training steps num_steps, sequence length $l$, thought length $t$, learning rate $\alpha$, batch size $b$, number of thoughts per token $n_{\text{thoughts}}$, number of supervised ground-truth tokens $n_{\text{true}}$.

**Output:** a language model $\theta$ that can generate rationales to predict future text.

Each training step:
1. Sample a batch of length-$l$ sequences $X$;
2. Generate thoughts in parallel for all token positions;
3. Mix predictions with/without thought;
4. Update parameters via REINFORCE.

### 3.2 Step 1: Think — parallel rationale generation

**Problem:** naively performing a separate forward pass for each token's thought is infeasible.

**Solution — Parallel Generation Algorithm:** the paper proposes a parallel sampling algorithm based on an attention mask (Figure 3). The core observation: one inference pass of a language model naturally produces a next-token distribution for every input token. A diagonal attention mask is constructed so that each thought token can only attend to:
- itself;
- previous thought tokens in the same thought;
- the corresponding preceding text token.

This way thoughts at different positions do not interfere, so they can be generated fully in parallel. Each inference call generates an additional thought token for every text token; repeating $t$ times yields a length-$t$ thought.

**Meta-tokens:** introduce learnable special tokens `<|startofthought|>` and `<|endofthought|>` to mark thought boundaries:
- `<|startofthought|>` puts the model into "thinking mode";
- `<|endofthought|>` signals end of thinking and start of prediction.

Initialization strategy: both meta-token embeddings are initialized to the em-dash "—" embedding (em dashes often indicate pauses or thought in text), exploiting existing model knowledge. To accelerate convergence, the gradients on these two embeddings are scaled by a hyperparameter weight (1e2).

### 3.3 Step 2: Talk — Mixing Head

**Problem:** thoughts generated by the pretrained model are initially out-of-distribution and would damage language-modeling performance if used directly.

**Solution — Mixing (Residual) Head:** introduce a shallow MLP (three layers, ReLU) whose input is the concatenation of the end-of-thought token's hidden state and the original text token's hidden state, and whose output is a scalar weight $w$ used to mix the next-token logits with and without thought:

$$\log p^{\text{talk}}_j = w_j \cdot \log p^{\text{init}}_j + (1 - w_j) \cdot \log p^{\text{thought}}_j$$

This lets the model rely mainly on the base prediction at the start of training ($w \approx 1$), and gradually increase the thought contribution as thought quality improves — a smooth transition.

### 3.4 Step 3: Learn — REINFORCE optimization

**Reward definition:** for the thought $T_j$ at position $j$, the reward is the difference between this thought's mixed prediction probability and the average across all thoughts at the same position:

$$r_j = \log p^{\text{talk}}_{j:j+n_{\text{true}}}(X_{j+1:j+n_{\text{true}}+1}) - \overline{\log p^{\text{talk}}_{j:j+n_{\text{true}}}(X_{j+1:j+n_{\text{true}}+1})}$$

Generating multiple rationales and using their mean as a baseline reduces variance (inspired by TRICE).

**REINFORCE Loss:**

$$\nabla_\theta \mathcal{L}^{\text{REINFORCE}}_j = -r_j \cdot \mathbf{1}[r_j > 0] \cdot \nabla_\theta \log p_\theta(T_j | [X_{:j}; \texttt{<|startofthought|>}])$$

Only positive-reward gradients are kept (negative rewards are dropped) to improve training stability.

**Non-myopic scoring & teacher forcing (Figure 4):** since not every token needs a thought, thought quality should not be judged only on the immediately next token. The paper uses teacher forcing to inject the next $n_{\text{true}}$ ground-truth tokens to compute multi-step prediction probabilities, so reward depends on longer-range semantic content rather than a single token. Visualization in Figure 4: solid lines denote LM computation, dashed lines denote teacher-forced inserted tokens, mixer blocks denote the mixing head.

**Total Loss:**

$$\nabla_\theta \mathcal{L}_j = \nabla_\theta \mathcal{L}^{\text{NLL}}_j + \nabla_\theta \mathcal{L}^{\text{REINFORCE}}_j$$

where $\mathcal{L}^{\text{NLL}}_j$ is the standard next-token-prediction loss (through the mixing-head probability), ensuring the base LM head also receives a language-modeling signal.

### 3.5 Figure descriptions

**Figure 1 (algorithm overview):** illustrates the three-step pipeline of Quiet-STaR for a single thought during training. (1) Think: generate thoughts in parallel after all tokens; (2) Talk: mix next-token predictions with and without thought via the mixing head; (3) Learn: use REINFORCE to increase probabilities of thoughts that help prediction and drop those that hurt.

**Figure 3 (parallel generation):** shows the attention-mask structure. Base text tokens are $a, b, c, d$; each generates its own thought tokens, and the diagonal attention mask realizes parallel inference without interference.

**Figure 4 (forward pass & teacher forcing):** shows the full pipeline of a single forward pass, including thought generation, the end-of-thought token, teacher-forced ground-truth tokens, and the mixing-head output. The example predicts 3 tokens ahead.

### 3.6 Implementation details

- **Base model:** Mistral 7B (base)
- **Optimizer:** AdamW; 20-step warmup; learning rate $1 \times 10^{-6}$; weight decay 0.001; batch size 8
- **Meta-token gradient weight:** 1e2; policy weight: 1e6
- **Training temperature:** $T = 1$ (sampling); inference uses greedy decoding for thoughts
- **REINFORCE temperature:** $T = 3$ (importance sampling)
- **Sequence length:** each sample is a random 256-token span
- **Hardware:** a single node with 8× H100 80GB GPUs

---

## 4. Datasets

### Training data (unlabeled general text corpora)

| Dataset | Domain | Description |
|-------|------|------|
| **OpenWebMath** (Paster et al., 2023) | Math/technical web pages | Technical web crawl, high reasoning density; the main training set |
| **C4** (Raffel et al., 2020) | General web text | Colossal Clean Crawled Corpus, widely used for LM pretraining; more diverse text |

### Evaluation datasets (zero-shot, no fine-tuning)

| Dataset | Domain | Description |
|-------|------|------|
| **GSM8K** (Cobbe et al., 2021) | Math reasoning | Grade-school math word problems; evaluates zero-shot direct reasoning |
| **CommonsenseQA** (Talmor et al., 2018) | Commonsense reasoning | Multiple choice; evaluates commonsense knowledge reasoning |

---

## 5. Evaluation metrics and main results
**Metric:** Zero-shot accuracy (%), computed directly from the conditional probability of the correct answer token (for multiple choice, accuracy over A–E logits); no task-specific fine-tuning.

### 5.1 Main zero-shot results

| Setting | GSM8K | CommonsenseQA |
|------|-------|---------------|
| Mistral 7B (baseline) | 5.9% | 36.3% |
| + Quiet-STaR (OpenWebMath, best config) | **10.9%** | **47.2%** |
| + Quiet-STaR (C4) | 8.1% | 42.6% |
| + No-thought baseline (same data) | ~6% | ~36% |

**Gains:** GSM8K +5.0%; CommonsenseQA +10.9% (OpenWebMath training).

### 5.2 Effect of thought length (Figure 2)

The paper systematically evaluates combinations of thought-token and ahead-token counts:

| # Thought Tokens, # Ahead Tokens | GSM8K (best) | CommonsenseQA (best) |
|----------------------------------|-------------|---------------------|
| 8, 4 | ~8% | ~40% |
| 10, 4 | ~9% | ~42% |
| 12, 4 | ~9.5% | ~44% |
| 16, 8 | ~10% | ~45% |
| 24, 12 | **~10.9%** | **~47.2%** |

**Key finding:** longer thoughts and more ahead tokens consistently bring better performance, showing that multi-token rationales beat single-token pauses.

### 5.3 Quiet-STaR + Chain-of-Thought (Figure 5)

Quiet-STaR is complementary to zero-shot CoT (the "Let's think step by step." prompt); internal rationales help external CoT produce more structured reasoning:

| Method | GSM8K (maj@8, $T=0.7$) |
|------|------------------------|
| Baseline (CoT) | 40.6% |
| Quiet-STaR + CoT | **47.7%** |

(Evaluated on 128 GSM8K test samples.)

### 5.4 Ablation: effect of multiple thoughts and tokens ahead

- Using multiple thoughts per token (2–4) beats using just the base as a baseline (GSM8K ~+0.5%, CommonsenseQA ~+3%), but diminishing returns set in beyond 2 (~0.1–0.3%).
- Predicting more than 1 token ahead helps significantly (GSM8K +0.3%, CommonsenseQA +3.1%), but gains taper off beyond 2 ahead tokens. Qualitative observation: more ahead tokens make rationales more coherent.

### 5.5 Improvement distribution (Appendix Figure 7 & 8)

- Most tokens see small prediction improvements, but hard-to-predict tokens receive disproportionately large gains — matching intuition: most web text does not require deep reasoning, but thoughts help difficult tokens substantially.
- Qualitative analysis shows thoughts especially help tokens that require recalling relevant information (e.g., names of applicable theorems, next steps of proofs).

### 5.6 Generated thought examples

The paper shows useful thoughts produced by the trained model:
- **Chemistry reasoning:** when predicting text about synthesizing magnesium nitride, the thought recalls the fact "starting from magnesium to produce magnesium nitride", helping predict the next step about heating magnesium.
- **Math proof:** in text about proving $A = B$ by showing $A \subseteq B$ and $B \subseteq A$, the thought produces "in some sense — to be the more difficult", helping predict the subsequent "trickiest for students".
- **Commonsense reasoning:** when reading a CommonsenseQA question, the thought anticipates the option structure, helping predict the rest of the question.
