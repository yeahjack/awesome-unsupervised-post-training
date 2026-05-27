# EMPO: Entropy-Minimized Policy Optimization

> **Added to survey on:** 2026-03-11

**Paper:** Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization
**Authors:** Qingyang Zhang, Haitao Wu, Changqing Zhang (Tianjin University), Peilin Zhao (Tencent AI Lab), Yatao Bian (Tencent AI Lab & NUS)
**arXiv:** 2504.05812
**Method:** EMPO | **Carrier:** Policy Opt. | **Regime:** training-time | **Level:** Semantic

| When to Adapt | Offline Corpus UPT before deployment |
|---|---|
| Trigger Unit | prompt batch / rollout group |
| Persistence | full parameter accumulate across RL steps |
| Inference Coupling | offline pre-deployment |
| Input Visibility | Offline |
| Update Persistence | Cumulative |
| Reset Boundary | Deployment Boundary |
| Evidence Status | paper-explicit |
| Timing Regime (auxiliary taxonomy) | Offline Corpus UPT |
| Visibility Scope | Pre-deployment corpus |

---

## When-to-Adapt audit

- **Auxiliary taxonomy:** `Timing Regime=Offline Corpus UPT`; `Visibility Scope=Pre-deployment corpus`.
- **Two-axis coding:** `Input Visibility=Offline`; `Update Persistence=Cumulative`; `Reset Boundary=Deployment Boundary`.

- **When the update is triggered:** updates are triggered in an offline RL stage before deployment; the basic unit is a group of rollouts per prompt batch.
- **Whose sample it serves:** the current rollout group's update serves subsequent training steps and the final deployed model, not the immediate inference of any single sample.
- **Whether parameters/state accumulate:** parameters accumulate across the RL training run; no per-sample reset is performed.
- **Reset boundary:** so it is an offline RL-style UPT schedule rather than test-time arrival-by-arrival adaptation.

## 1. UPT Assignment Rationale

EMPO belongs to **Family II (Sample-Relation Supervision)**, sub-class semantic agreement, with the specific mechanism of semantic entropy / meaning clustering.

Core logic: given an unlabeled question $q$, the model samples a group of outputs $\{o_1, \dots, o_G\}$ and then clusters them into $M$ meaning clusters $\{c_1, \dots, c_M\}$ by semantic equivalence (bidirectional entailment). Each output's reward equals the probability of its cluster, $r_i = p(c_j | q) \approx |c_j| / G$ — the larger the cluster (the more semantically consistent), the higher the reward. The training objective is to minimize the semantic entropy on the output distribution: $H = -\sum_{c_j} p(c_j|q) \log p(c_j|q)$.

**No external supervision signal is used:** no ground-truth answer, no rule-based verifier, no external reward model, no human annotation. The sole signal source is the semantic relation among the model's own outputs (a relational signal), turned into a reward through meaning clustering.

---

## 2. Problem Addressed

Existing LLM reasoning training pipelines (SFT + RL) rely heavily on external supervision signals, including:
- human-annotated reasoning traces;
- verified golden answers (rule-based reward);
- pretrained reward models.

These requirements lead to:
1. **High data-acquisition cost:** high-quality reasoning data needs expert annotation.
2. **Poor scalability:** hard to generalize to open-ended reasoning tasks without standard answers (e.g. free-form natural reasoning).
3. **Task-format dependence:** rule-based rewards apply only to tasks with deterministic answers (math).

EMPO's core question: **can we incentivize LLM reasoning ability under fully unsupervised conditions?** That is, can we improve performance on math reasoning and free-form natural reasoning using only unlabeled questions?

---

## 3. Method

### 3.1 Semantic entropy as the unsupervised objective

Semantic entropy is a natural extension of Shannon entropy in semantic space. Intuition: a reliable model should give semantically consistent answers to the same question; lower semantic entropy means outputs concentrate on fewer meaning clusters, i.e. the model is more "confident". Prior work shows semantic entropy correlates strongly negatively with model accuracy.

**Meaning-clustering procedure:**
- For math reasoning: extract the final answer in `\boxed{}` with a regex; identical answers go to the same cluster.
- For free-form natural reasoning: use the General-Verifier (a 1.5B SLM) to judge whether two outputs are semantically equivalent (bidirectional entailment); merge if equivalent.

**Semantic-entropy computation:**
$$p(c_j | q) \approx |c_j| / G, \quad H = -\sum_{c_j \in \{c\}} p(c_j|q) \log p(c_j|q)$$

### 3.2 Entropy-Minimized Policy Optimization (EMPO)

Based on the GRPO framework, but replacing the external reward with semantic entropy:

1. For each question $q$, sample $G$ outputs from the current policy $\pi_\theta$.
2. Cluster the outputs into meaning clusters.
3. Each output's reward = the probability of its cluster: $r_i = p(c_j | q)$ with $l(o_i) = c_j$.
4. Compute the group-normalized advantage: $A_i = (r_i - \text{mean}) / \text{std}$.
5. Optimize the clipped policy-gradient objective (no KL constraint; $\epsilon$ clip preserves stability).

### 3.3 Entropy thresholding to prevent reward hacking

Set a double threshold $\delta_{low} < H < \delta_{high}$:
- **Filter out high-entropy questions:** the model is too uncertain; outputs are unreliable.
- **Filter out low-entropy questions:** the model is already confident; further optimization risks overconfidence and yields little benefit.

Final objective:
$$\mathcal{J}_{\text{EMPO}} = \mathbb{E}\left[\frac{1}{|G|}\sum_{i=1}^{|G|} \min(A_i, \text{clip}(1-\epsilon, 1+\epsilon) A_i)\right], \quad \text{s.t. } \delta_{low} < H < \delta_{high}$$

---

## 4. Datasets

### Training data
| Task type | Dataset | Scale |
|---------|--------|------|
| Math reasoning | NuminaMath-CoT (random subsample) | 20K prompts |
| Natural reasoning | Natural Reasoning (Facebook) | ~18K prompts (after filtering) |

Training data contains **only unlabeled questions**, with no answers or reasoning traces.

### Evaluation benchmarks
| Task type | Benchmarks |
|---------|-----------|
| Math reasoning | MATH, Minerva Math, AMC23, OlympiadBench, AIME24 |
| Natural reasoning | MMLU-Pro, GPQA |

---

## 5. Evaluation metrics and main results

**Metric:** pass@1 accuracy (greedy decoding, zero-shot).

### Math reasoning results (Table 1)

| Model | Supervision | MATH | Minerva | Olympiad | AIME24 | AMC23 | Avg. |
|------|---------|------|---------|----------|--------|-------|------|
| Qwen2.5-Math-1.5B Base | None | 52.2 | 10.7 | 25.2 | 10.0 | 42.5 | 28.1 |
| Qwen2.5-Math-1.5B w/GRPO | {q, a} | 75.2 | 32.0 | 33.6 | 16.7 | 52.5 | 42.0 |
| **Qwen2.5-Math-1.5B w/EMPO** | **{q}** | **73.0** | **32.4** | **36.6** | **13.3** | **55.0** | **42.1** |
| Qwen2.5-Math-7B Base | None | 64.8 | 15.1 | 26.7 | 6.7 | 40.0 | 30.7 |
| Qwen2.5-Math-7B w/GRPO | {q, a} | 77.8 | 39.7 | 39.1 | 20.0 | 57.5 | 46.9 |
| **Qwen2.5-Math-7B w/EMPO** | **{q}** | **78.0** | **40.4** | **37.3** | **20.0** | **65.0** | **48.1** |

Key findings:
- EMPO (questions only) **beats GRPO on average** at 7B (48.1 vs 46.9); GRPO requires golden answers.
- At 1.5B, EMPO matches GRPO (42.1 vs 42.0).
- Compared to the base model, EMPO yields +17.4 points at 7B.

### Natural reasoning results (Table 2)

| Model | MMLU-Pro Avg. | GPQA |
|------|--------------|------|
| Qwen2.5-7B Base | 32.1 | 23.5 |
| Qwen2.5-7B w/GRPO | 33.8 | - |
| **Qwen2.5-7B w/EMPO** | **34.6** | **28.8** |
| Qwen2.5-14B Base | 30.6 | - |
| **Qwen2.5-14B w/EMPO** | **41.6** | **35.3** |

- MMLU-Pro: 7B improves 32.1% → 50.1% (+18.0).
- GPQA: 7B improves 15.9% → 28.8% (+12.9).

### Training dynamics

During training, semantic entropy keeps decreasing while both semantic-probability reward and accuracy reward keep rising; the three curves are highly aligned, validating semantic entropy as an effective unsupervised reward signal.

### Core insight

Via pass@k analysis, the authors find that both EMPO and GRPO mainly improve the model's efficiency in finding the correct reasoning path under few samples (substantial pass@1 gain); but as $k$ grows, the base model's pass@k gradually catches up and may even surpass the RL-trained model. This indicates that **RL post-training (including unsupervised EMPO) essentially guides and optimizes the model's *usage efficiency* of reasoning ability already learned during pretraining, rather than teaching brand-new reasoning skills.**
