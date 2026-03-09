# CompRAG: Comparative Utilization Metrics for Retrieval-Augmented Generation

## Project Summary (v9 — March 2026)

### What This Is

A benchmark harness that evaluates RAG faithfulness across small language models (1.7B–14B parameters) and quantization levels (Q3_K_M through FP16), all running on a single NVIDIA V100 32GB. The goal is a publishable comparison matrix characterizing how model architecture, parameter count, and quantization affect RAG output quality, with specific focus on the balance between context utilization (trusting retrieved documents) and self-knowledge (relying on parametric memory).

The study uses a **three-pass evaluation protocol** that separates baseline knowledge, behavioral preference, and forced-context capability into independently measurable signals.

### v9 Changelog

- **Frontier API comparison fully removed.** No frontier generation runs. The 5-model reference line layer (GPT-5.4, Claude Opus 4.6, Gemini 3 Flash, DeepSeek V3.2, GLM-5) is cut. CompRAG's contribution is the quantization-faithfulness curve shape, not cost-performance positioning against commercial APIs. Frontier models introduced uncontrolled variables (RLHF intensity, unknown quantization, reasoning modes) that muddied the analysis without strengthening the core finding.
- **Datasets scoped to RGB only.** NQ (3.6k queries, general QA) and HaluEval (10k queries, hallucination detection) dropped. Neither participates in the three-pass protocol, and both add pass2-only volume without analytical depth. RGB's three subsets (counterfactual, noise robustness, negative rejection) provide the adversarial structure needed for CU/SK/Preference_Gap analysis. Runtime drops from ~8-10 days to ~9 hours.
- **500 queries per model.** 100 per RGB subset (counterfactual × 3 passes, noise_robustness × pass2, negative_rejection × pass2). Sufficient for bootstrap CIs at 1000 resamples.
- **Judge architecture finalized: Command R primary, Sonnet 4.6 spot-check.** Command R 35B Q4_K_M (local, on V100) is the primary bulk judge for all scoring. Claude Sonnet 4.6 (API) scores a 500-record validation sample stratified across the model matrix. DeepSeek test pass confirmed Command R produces sufficient agreement. The three-way frontier judge panel (Opus/GPT-5.4/Gemini) is replaced by this simpler, cheaper architecture. Cohen's κ between Command R and Sonnet 4.6 is reported per metric; κ ≥ 0.80 on the primary analysis metric (CU) validates the local judge, with raw agreement rates reported for metrics where κ is deflated by marginal distribution skew.
- **Hardware updated.** x99 mATX Chinese board with E5-2667 v4 replaces Dell T7820. V100 SXM2 32GB unchanged.
- **Priority claim validated.** Maestro et al. (2024, arxiv:2406.10251) compared FP16 vs INT4 on 2 models using task accuracy — no CU/faithfulness metrics, no multi-point quant curve, no architecture comparison. "Which Quantization Should I Use?" (arxiv:2601.14277) evaluates GGUF quants on general benchmarks, not RAG-specific faithfulness. CompRAG's systematic CU/faithfulness curves across 6 quant levels × 7 architectures with RAGChecker metrics appears novel.

### v8 Changelog

- RAGChecker metrics expanded to full 11-metric suite (Overall: Precision, Recall, F1; Retriever: Claim Recall, Context Precision; Generator: CU, SK, NS-I, NS-II, Hallucination, Faithfulness).
- Noise Sensitivity split into NS-I (relevant) and NS-II (irrelevant).
- RAGChecker measurement limitations added to Threats to Validity.

### v7 Changelog

Major simplification based on consolidated meta-review (three independent adversarial reviews) and subsequent design review:

1. **Logit Gap removed entirely.** All three reviewers identified it as the weakest component, with five independent measurement artifacts. The mechanistic "why" behind quantization-faithfulness shifts is deferred to a future study. This study characterizes *what* happens, not *why*.
2. **Simplified to three-pass protocol.** Pass 2' (instruction compliance without context) removed.
3. **CU_Preference ratio replaced with Preference_Gap delta.** Preference_Gap = Pass3_CU − Pass2_CU.
4. **E-waste/deprecated hardware narrative removed.** The framing is now resource-constrained RAG benchmarking.
5. **GPU layer offload sensitivity analysis removed.** All runs execute fully offloaded to GPU on a single V100 32GB.
6. **Model matrix restructured.** Qwen 2.5 7B promoted to primary. GLM-4 9B dropped.
7. **Pre-registered falsification restructured.** Two tiers: global falsification criterion plus per-model sub-predictions.
8. **Seed strategy revised.** Determinism pilot added. Local models bootstrapped over queries, not seeds, if greedy decoding is confirmed deterministic.

### Hypothesis

**There exists a quantization-dependent faithfulness curve whose shape varies by model architecture.** Aggressive quantization weakens a model's parametric memory, potentially reducing its confidence in internally stored knowledge and increasing reliance on retrieved context. This would produce higher Context Utilization (CU) scores — meaning quantization, typically framed as a necessary compromise, may in some cases be associated with improved RAG faithfulness. However, sufficient quantization eventually degrades the reasoning capacity needed to synthesize answers from context, creating a faithfulness-capability tradeoff.

The central empirical question is the shape of this curve across model architectures: is there a quantization sweet spot for RAG faithfulness, and does it shift predictably by model family? Prior research indicates degradation is non-linear and architecture-dependent — Llama 3.1 8B suffers disproportionate accuracy loss under quantization compared to Qwen 2.5 and Phi-4 architectures. CompRAG tests whether this asymmetry extends to RAG-specific faithfulness metrics, not just general benchmark scores.

**Pre-registered predictions (committed before first experimental run):**

- **Qwen 2.5 14B:** CU will increase monotonically from FP16 → Q3. Predicted pattern: Gullible (quantization erodes parametric confidence, model defers to context).
- **Llama 3.1 8B:** CU will increase from FP16 → Q5 but decrease from Q4 → Q3 as capability degradation outpaces parametric erosion. Predicted curve shape: inverted U (non-monotonic). Statistical criterion: the Q3 point's bootstrap CI upper bound must be below the peak point's bootstrap CI lower bound. If CIs overlap, the prediction is indeterminate.
- **Qwen 2.5 7B:** CU curve shape will qualitatively match Qwen 14B (Gullible), controlling for the size confound in the 14B-vs-8B Llama comparison.
- **Global falsification criterion:** If no primary model architecture demonstrates the Gullible pattern (CU monotonically increasing with quantization), the core hypothesis is refuted. Per-model predictions above are sub-predictions; their individual failure does not refute the overall thesis if other architectures show the predicted pattern.

**Three predicted outcome patterns:** Resilient (CU stays flat across quantization — context attention is quantization-robust), Stubborn (CU decreases with quantization — model defaults to parametric weights), or Gullible (CU increases with quantization — parametric confidence erodes and model defers to context). The Gullible pattern is consistent with the thesis. Classification uses data-driven thresholds based on bootstrap CI position relative to the FP16 baseline.

The headline finding, if observed: **quantization strategy and model architecture selection correlate more strongly with RAG faithfulness than model size past a surprisingly low threshold.**

### Related Work

CompRAG sits at the intersection of three established research areas:

**Knowledge conflict in LLMs.** Longpre et al. (2021), Neeman et al. (2023), and Xie et al. (2024) establish that models with less relevant parametric knowledge are more susceptible to contextual override, including from adversarial or incorrect context. CompRAG operationalizes this known phenomenon along the quantization axis specifically — testing whether weight degradation from quantization produces the same deference-to-context behavior as inherently weaker parametric knowledge.

**Adaptive retrieval.** FLARE (Jiang et al. 2023) uses iterative generation confidence to trigger retrieval. Self-RAG (Asai et al. 2023) trains models to emit special tokens deciding when to retrieve. SKR (Wang et al. 2023) explores similar gating. CompRAG does not propose an adaptive retrieval mechanism — it characterizes the relationship between quantization and faithfulness to inform model selection for fixed-retrieval RAG deployments.

**Quantization effects on model capability.** The LLM-QAT and GPTQ literature extensively documents how quantization affects general benchmark performance. Maestro et al. (2024, arxiv:2406.10251) is the closest prior work: they compared FP16 vs INT4 on two models (7B/8B) using task accuracy for RAG, but used only two quantization levels, no CU/faithfulness metrics, and no architecture comparison. "Which Quantization Should I Use?" (arxiv:2601.14277) evaluates GGUF quants on general benchmarks but not RAG-specific faithfulness. CompRAG extends this line of work with systematic CU/faithfulness curves across six quantization levels and seven architectures using claim-level RAGChecker metrics.

**Uncertainty estimation and selective prediction.** Hendrycks & Gimpel (2017) and Geifman & El-Yaniv (2017) establish logit margins and softmax entropy as uncertainty estimates. These techniques inform the design space for future mechanistic analysis of quantization-induced confidence shifts, but are not directly employed in the current study.

**CompRAG's specific contribution:** The first systematic measurement of how GGUF quantization levels shift RAG faithfulness metrics across model architectures, operationalized through a three-pass diagnostic protocol. Prior work examined quantization's effect on RAG at coarse granularity (Maestro et al.) and on general benchmarks (arxiv:2601.14277), but no study has mapped the quantization-faithfulness relationship at fine granularity across architectures using claim-level metrics.

### Experimental Design

**Controlled variable:** The retrieval pipeline is held constant across all runs. Same embedding model (all-MiniLM-L6-v2), same ChromaDB vector index, same top-k=5, same 300-word chunks. Only the generator model varies.

**Locked parameters:** Prompt template is identical across all models. Temperature 0.0, seed 42, max_tokens 512. No per-model tuning. Chat template wrappers vary by model family (ChatML, Llama, etc.) but the content of system/context/user messages is fixed.

**Single platform:** All local inference runs execute fully offloaded to GPU on one NVIDIA V100 SXM2 32GB. No CPU/GPU layer splitting, no partial offload. This eliminates floating-point divergence confounds from mixed-backend computation.

### Three-Pass Evaluation Protocol

Each query in the RGB counterfactual subset is evaluated under three prompting conditions that isolate distinct aspects of model behavior. This separates the question "does the model know the answer?" from "does it prefer context?" from "can it use context when forced?"

**Pass 1 — Baseline (No Retrieval).** The question is presented with no context. This measures raw Self-Knowledge (SK): what the model knows from its parametric memory alone. The prompt is minimal — just the question with the model's standard instruction wrapper. This establishes a per-query baseline: if the model doesn't know the answer without context, high CU in Passes 2–3 is expected and uninformative. If it *does* know the answer, then CU in later passes represents a genuine conflict between parametric and contextual signals.

```
<s>[INST] {question} [/INST]
```

**Pass 2 — Loose RAG (Behavioral Preference).** The question is presented with retrieved context using a neutral framing ("I have some background information that might be helpful"). This is the primary test condition — it measures whether the model *prefers* the context over its own weights when not explicitly instructed to do so. CU/SK metrics from RAGChecker are measured on this pass.

```
<s>[INST] I have some background information that might be helpful:
{context}

Based on that, {question} [/INST]
```

**Pass 3 — Strict RAG (Forced Capability).** The question is presented with an explicit instruction to answer only from context. This is an ablation: if the model fails here, it lacks the *capability* to attend to context regardless of preference. If it succeeds here but fails in Pass 2, the issue is behavioral (the model can read context but chooses not to).

```
<s>[INST] System: Answer using ONLY the provided context.
If the answer is not in the context, say "I do not know."

Context:
{context}

User: {question} [/INST]
```

**Analytical value:** The three-pass protocol transforms a single CU measurement into a diagnostic triple:

- A model that scores (Pass 1: correct, Pass 2: follows context, Pass 3: follows context) is genuinely deferring to retrieval despite knowing better — the strongest evidence for the confidence-erosion pattern.
- A model that scores (Pass 1: correct, Pass 2: ignores context, Pass 3: follows context) has the capability but not the natural inclination — a behavioral preference that quantization might shift.
- A model that scores (Pass 1: wrong, Pass 2: follows context, Pass 3: follows context) tells us nothing about CU/SK tradeoffs since there was no parametric knowledge to override.

**Preference_Gap metric:** Preference_Gap = Pass3_CU − Pass2_CU. This separates behavioral preference from capability. A gap of 0 means the model naturally defers as much as when forced. A large positive gap means capability without preference. A negative gap means the model uses context more in the loose prompt than the strict one (likely noise or an instruction-following artifact). Reported for all configurations including capability-degraded ones. Configurations where Pass3_CU drops below a minimum threshold are visually flagged (distinct markers, shaded region) but not excluded — the transition from "increasing CU" to "model can't parse context" is itself a finding, representing the right edge of the predicted inverted-U curve.

Note: All three passes are run on the RGB counterfactual subset. The noise robustness and negative rejection subsets use Pass 2 only, consistent with standard RAG benchmarking methodology.

### Model Matrix

All models in GGUF format via llama.cpp. All models are established architectures with published quantization baselines on general benchmarks.

| Model | Params | Quants | Role |
|-------|--------|--------|------|
| Qwen 2.5 14B Instruct | 14B | Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16 | Primary — strong RAG performer, quantization-resilient |
| Phi-4 14B | 14B | Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16 | Primary — reasoning-dense training, quantization-resilient, matched size with Qwen 14B |
| Llama 3.1 8B Instruct | 8B | Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16 | Primary — known quantization-fragile, predicted inverted-U |
| Qwen 2.5 7B Instruct | 7B | Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16 | Primary — size-controlled comparison with Llama 8B, architecture comparison with Qwen 14B |
| Mistral NeMo 12B Instruct | 12B | Q4_K_M, Q8_0 | Secondary |
| Gemma 2 9B Instruct | 9B | Q4_K_M, Q8_0 | Secondary |
| SmolLM2 1.7B Instruct | 1.7B | Q8_0, FP16 | Floor test — minimal parametric knowledge |

**Cross-model comparison rationale:**

- *Architecture effect at matched size:* Qwen 14B vs Phi-4 14B (both 14B, different architectures, both quantization-resilient). Qwen 7B vs Llama 8B vs Gemma 9B (7–9B range, three architectures, different quantization robustness profiles).
- *Size effect within architecture:* Qwen 14B vs Qwen 7B (same architecture family, different parameter count, full quantization resolution on both).
- *Quantization robustness:* Llama 8B (fragile) vs Qwen 7B (resilient) at near-matched size — the cleanest test of architecture-dependent quantization asymmetry for RAG.
- *Floor test:* SmolLM2 1.7B establishes whether sub-2B models produce meaningful CU/SK signal or collapse entirely.

### Evaluation

**Primary framework:** RAGChecker (pin version in requirements.txt and pre-registration file; report exact commit hash if git install). All 11 RAGChecker metrics are captured and stored per query. They are organized as follows:

*Overall metrics (3):* Precision (fraction of response claims entailed in ground truth), Recall (fraction of ground-truth claims entailed in response), F1 (harmonic mean). These provide holistic response quality assessment.

*Retriever metrics (2):* Claim Recall (fraction of ground-truth claims entailed in any retrieved chunk), Context Precision (fraction of retrieved chunks containing ≥1 ground-truth claim). These are constant across generator configurations since the retrieval pipeline is held fixed, but are recorded per query to verify retrieval consistency and enable per-query difficulty stratification.

*Generator metrics (6):* Context Utilization (CU), Self-Knowledge (SK), Relevant Noise Sensitivity (NS-I), Irrelevant Noise Sensitivity (NS-II), Hallucination, Faithfulness. CU and SK directly operationalize the hypothesis — CU measures how much the model uses retrieved context, SK measures how much it relies on parametric memory. NS-I (incorrect claims entailed in relevant chunks) and NS-II (incorrect claims entailed in irrelevant chunks) are reported separately; the RAGChecker paper demonstrates that NS-I >> NS-II across systems due to chunk-level trust, and this distinction is relevant for diagnosing whether quantization shifts the CU/NS tradeoff curve or just moves along it. Hallucination (incorrect claims not entailed in any chunk) and Faithfulness (fraction of all response claims entailed in any chunk) provide complementary error characterization.

*Primary analysis variables:* CU and SK (core hypothesis). NS-I, NS-II, Faithfulness, and Hallucination (secondary — characterize the CU/NS/Faithfulness trilemma identified in the RAGChecker paper as it interacts with quantization). Overall Precision/Recall/F1 (tertiary — sanity check that overall response quality tracks expectations).

**Derived metric:** Preference_Gap = Pass3_CU − Pass2_CU. Separates behavioral preference from capability. Reported for all configurations with visual flagging (not exclusion) of capability-degraded configurations.

**Judge model:** Command R 35B Q4_K_M (local, running on V100 via llama.cpp server on port 8081) is the primary bulk judge for all scoring runs. The judge is instrumentation, not an experimental variable — Command R shares no architecture or training data with any model in the test matrix, eliminating self-evaluation bias. The generation server (port 8080) must be stopped before starting the judge server — they cannot coexist on the V100.

**Judge validation:** Claude Sonnet 4.6 (API) scores a 500-record validation sample stratified across the model matrix as an appendix validation. Cohen's κ between Command R and Sonnet 4.6 is reported per metric. For the primary analysis metric (CU), κ ≥ 0.80 is required. For secondary metrics where both judges produce near-zero scores (e.g., hallucination, SK), κ is expected to be deflated by marginal distribution skew; raw agreement rates are reported alongside κ to contextualize these cases. DeepSeek test pass during development confirmed Command R produces sufficient judge quality.

**Dataset:** RGB only (noise robustness, negative rejection, counterfactual robustness). 500 queries per model — 100 per subset, with the counterfactual subset running all three passes and the other two subsets running pass 2 only.

**RGB counterfactual sub-analysis:** Within the RGB counterfactual subset, the Negative Rejection subset (where the model *should* reject the context) is analyzed separately from the Noise Robustness subset. This distinguishes the "faithful" failure mode (model blindly repeats malicious context) from the "robust" success mode (model correctly ignores irrelevant noise). High CU on Noise Robustness is good; high CU on Negative Rejection is a vulnerability. Reporting them together would mask this distinction.

**Seed strategy and variance:** A determinism pilot is run before main experiments: 10 seeds on one model/quant combo (Qwen 14B Q4_K_M), 50 queries. If outputs are bit-identical across seeds, seed-based replication is dropped for local models — bootstrap CIs are computed over queries only (~500 per model). Report "local greedy decoding verified deterministic on this hardware" and cite the pilot.

**Statistical rigor:** 500 queries per model, bootstrap resampling (1000×) for 95% confidence intervals.

### Hardware

Single platform: NVIDIA V100 SXM2 32GB (via SXM2-to-PCIe adapter) in an x99 mATX board with Intel Xeon E5-2667 v4, 16GB DDR4. Display GPU: GTX 1660 Super on x4 riser (not used for inference).

### Threats to Validity

**Faithfulness vs. gullibility confound.** High CU at aggressive quantization levels may indicate the model has lost its ability to fact-check retrieved context, not that it has become a better RAG system. A model that blindly repeats a malicious prompt scores high on CU but represents a failure mode, not a success. This is partially mitigated by the separate analysis of RGB Negative Rejection (where the model *should* reject context). The three-pass protocol further disambiguates: if a model scores high CU in Pass 2 but also correctly rejects adversarial context in Negative Rejection queries, the CU increase is consistent with appropriate deference rather than gullibility. However, this study cannot definitively distinguish the two mechanisms without interventional experiments (e.g., selective layer quantization or activation steering) that are outside the current scope.

**Capability degradation vs. preference shift.** Aggressive quantization degrades reading comprehension and instruction following independently of confidence effects. The Preference_Gap metric (Pass3_CU − Pass2_CU) partially addresses this by measuring the delta between forced and natural context utilization. Capability-degraded configurations (where Pass3_CU drops below a minimum threshold) are visually flagged but not excluded from analysis — the transition from functional to degraded is itself informative, as it represents the right edge of the predicted inverted-U curve.

**Chat template variation.** Chat template wrappers vary by model family (ChatML, Llama, Gemma, etc.) in ways that affect tokenization, attention patterns, and instruction-following behavior. This study holds prompt content constant but cannot control for template-induced differences in how models attend to system instructions versus user content. Architecture-level comparisons should be interpreted with this confound in mind; within-architecture quantization comparisons are unaffected.

**Embedding model and effective retrieval quality.** The retrieval pipeline (all-MiniLM-L6-v2, top-k=5, 300-word chunks) is held constant, but effective retrieval quality as perceived by the generator varies with model capability. A 14B model may synthesize information from marginally relevant chunks that a 1.7B model cannot. CU comparisons across model sizes should account for this: lower CU in smaller models may reflect inability to use the retrieved context, not preference against it. Within-model quantization comparisons are unaffected. Cross-architecture comparisons at matched size (Qwen 7B vs Llama 8B, Qwen 14B vs Phi-4 14B) partially control for this.

**RGB counterfactual dataset assumes shared parametric knowledge.** The "parametric" answer in RGB is the real-world ground truth. Whether a given model actually has this fact in its parameters depends on its training data, which varies by model family and is unknown. CU/SK metrics from RAGChecker are computed on all queries without a knowledge filter, meaning cross-model CU comparisons conflate parametric knowledge breadth with context-utilization preference. Within-model quantization comparisons are unaffected (same training data across quant levels).

**Judge model independence.** The primary judge (Command R 35B) shares no architecture or training data with any model in the experimental matrix, eliminating same-family evaluation bias. The validation judge (Claude Sonnet 4.6) is similarly outside the test matrix. A 500-record κ check stratified across the model matrix validates the primary judge, with per-metric reporting to account for marginal distribution skew in near-zero metrics. Remaining risk: all LLM judges share broad pretraining distributions, so subtle shared biases are possible but not systematically directional for any particular model family in the matrix.

**RAGChecker measurement limitations.** RAGChecker's claim-entailment checker (Llama3-70B via RefChecker) collapses Neutral and Contradiction entailment results into a single "not entailed" bucket. A claim that *contradicts* a retrieved chunk represents a different failure mode than one the chunk is silent about, but both are scored identically. This matters for quantized models specifically: degraded weight precision may produce imprecise paraphrasing (Neutral) rather than factual contradiction, and RAGChecker treats these identically. Additionally, all claims are weighted equally regardless of importance — a minor factual detail and a critical safety-relevant claim contribute equally to CU/SK/NS scores. These are upstream measurement limitations that apply equally across all configurations and do not differentially bias within-model quantization comparisons, but they limit the interpretive depth of cross-model comparisons. Finally, RAGChecker's SK metric conflates capability loss with behavioral preference: a model that has lost parametric knowledge through quantization and a model that has chosen to defer to context both score low on SK. The three-pass protocol partially disambiguates this via Preference_Gap and Pass 1 baseline knowledge measurement, but a full mechanistic complement (e.g., activation probing, LUMINA-style context-knowledge signals) would be needed to fully resolve the distinction. This is deferred to future work.

### Novel Contributions

1. **Quantization-faithfulness curves** — systematic CU/SK measurements across six quantization levels for multiple model architectures, characterizing whether and how quantization correlates with RAG faithfulness changes. First published measurement of this relationship for RAG-specific metrics. Prior work (Maestro et al. 2024) compared only two quantization levels on two models using task accuracy; CompRAG provides six-point curves across seven architectures using claim-level faithfulness metrics.
2. **Three-pass evaluation protocol** — separating baseline knowledge, behavioral preference, and forced-context capability into independently measurable signals per query, enabling diagnostic classification of model behavior rather than single-score ranking.
3. **Preference_Gap metric** — Pass3_CU − Pass2_CU, separating behavioral preference from capability degradation under quantization without the statistical degeneracy of ratio metrics.
4. **Architecture-dependent quantization asymmetry for RAG** — extending known general-benchmark degradation asymmetries (Llama vs Qwen/Phi) to RAG-specific faithfulness metrics, with size-controlled cross-architecture comparisons (Qwen 7B vs Llama 8B, Qwen 14B vs Phi-4 14B).

### Tech Stack

- **Inference (generation):** llama.cpp server (GGUF models, OpenAI-compatible HTTP API, port 8080)
- **Inference (judge):** llama.cpp server (Command R 35B Q4_K_M, port 8081) — cannot coexist with generation server on V100
- **Vector store:** ChromaDB (local, file-based)
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers, CPU)
- **Evaluation:** RAGChecker (version pinned)
- **Judge validation:** Claude Sonnet 4.6 API (500-record stratified validation sample)
- **Infrastructure:** Docker (multi-target: CPU/CUDA), GitHub Actions CI
- **Analysis:** Python (numpy, scipy, pandas, matplotlib, seaborn)

### Pre-Experiment Checklist

Before any GPU time is spent:

1. **Pre-registration file committed to git.** Contains: hypothesis, predictions with statistical criteria, global falsification criterion, model matrix, quant levels, dataset versions, RAGChecker version/commit hash, judge model specification, Preference_Gap definition, bootstrap parameters, determinism pilot protocol.
2. **Determinism pilot.** 10 seeds × 50 queries × 1 config (Qwen 14B Q4_K_M). Confirm bit-identical outputs or characterize inter-run variance.
3. **GGUF checksums recorded.** SHA256 of every GGUF file used, stored in pre-registration file.
4. **RAGChecker version pinned.** Exact version or commit hash in requirements.txt and pre-registration.
5. **RGB dataset version pinned.** Download date and source URL documented.
6. **Priority claim verified.** Searched r/LocalLLaMA, HuggingFace, arxiv. Closest: Maestro et al. (2024, arxiv:2406.10251) — FP16 vs INT4 on 2 models, task accuracy only. No multi-point quant curve, no CU/faithfulness metrics, no architecture comparison. CompRAG's contribution appears novel.
