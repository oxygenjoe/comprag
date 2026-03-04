# CompRAG: Comparative Utilization Metrics for Retrieval-Augmented Generation

## Project Summary (v5 — March 2026)

### What This Is

A benchmark harness that evaluates RAG faithfulness across small language models (1.7B–14B parameters), quantization levels (Q4_K_M through FP16), and simulated VRAM tiers — all running on a single NVIDIA V100 32GB. The goal is a publishable comparison matrix isolating how model architecture, parameter count, and quantization affect RAG output quality, with specific focus on the balance between context utilization (trusting retrieved documents) and self-knowledge (hallucinating from parametric memory).

The study includes a **Logit Gap** mechanistic analysis layer that explains *why* quantization affects faithfulness at the token probability level, a **three-pass evaluation protocol** that separates baseline knowledge, behavioral preference, and forced-context capability, and a **frontier API comparison** that anchors local model results against five major commercial models.

### Hypothesis

**There exists a quantization-dependent faithfulness curve with a non-obvious optimum.** Aggressive quantization weakens a model's parametric memory, reducing its confidence in internally stored knowledge and forcing greater reliance on retrieved context. This produces higher Context Utilization (CU) scores — meaning quantization, typically framed as a necessary compromise, may be actively beneficial for RAG faithfulness. However, sufficient quantization eventually degrades the reasoning capacity needed to synthesize multi-hop answers from context, creating a faithfulness-capability tradeoff.

The central empirical question is the shape of this curve across model architectures: where is the sweet spot, and does it shift predictably by model family? Prior research indicates degradation is non-linear and architecture-dependent — Llama 3.1 8B suffers disproportionate accuracy loss under quantization compared to Qwen 2.5 and Phi-4 architectures. CompRAG tests whether this asymmetry extends to RAG-specific faithfulness metrics, not just general benchmark scores.

**Mechanistic prediction:** If the hypothesis is correct, the Logit Gap — defined as `logit(contextual_token) - logit(parametric_token)` at the first answer token — should trend positive as quantization increases. Specifically, parametric token probability should degrade faster than contextual token probability, proving the mechanism is "confidence erosion" rather than "attention improvement." Three outcome curves are predicted: Resilient (gap stays flat — attention is quantization-robust), Stubborn (gap goes negative — model defaults to hallucinated weights), or Gullible (gap goes positive — parametric confidence erodes and model clings to context). The Gullible curve validates the thesis.

The headline finding, if confirmed: **quantization strategy and model architecture selection matter more for RAG faithfulness than model size or hardware cost past a surprisingly low threshold.**

### Experimental Design

**Controlled variable:** The retrieval pipeline is held constant across all runs. Same embedding model (all-MiniLM-L6-v2), same ChromaDB vector index, same top-k=5, same 300-word chunks. Only the generator model varies.

**Locked parameters:** Prompt template is identical across all models. Temperature 0.0, seed 42, max_tokens 512. No per-model tuning. Chat template wrappers vary by model family (ChatML, Llama, etc.) but the content of system/context/user messages is fixed.

**Simulated VRAM tiers:** Consumer VRAM constraints (0GB/CPU-only, 6GB, 8GB, 16GB, 32GB) are simulated on a single V100 32GB by constraining GPU layer offload via `--n-gpu-layers` in llama.cpp. All runs execute on identical silicon, eliminating confounds between VRAM capacity and other hardware variables (memory bandwidth, compute architecture, driver maturity) that would be present when comparing physically different GPUs.

Before each run, the harness calculates the model's VRAM footprint at the given quantization. If it exceeds the simulated tier's usable budget (accounting for driver/CUDA context overhead), the configuration is flagged as infeasible for that tier — equivalent to OOM on real hardware. However, the run continues with partial offload, producing both a practical feasibility matrix (what fits on a real 6GB card) and faithfulness data across offload levels. The OOM boundary is metadata, not a termination condition.

Simulated tier usable budgets (after overhead):

| Tier | Simulated Usable VRAM |
|------|----------------------|
| 0GB (CPU-only) | 0 GB (full offload) |
| 6GB  | ~5.2 GB              |
| 8GB  | ~7.0 GB              |
| 16GB | ~14.5 GB             |
| 32GB | ~30.0 GB             |

**Methodological validation — offload equivalence:** Partial CPU offload changes the numerical computation path (CUDA kernels vs CPU BLAS routines), introducing potential floating-point divergence. At temperature 0.0, near-tied logits could flip token selection, cascading through autoregressive generation. To validate that this doesn't systematically affect faithfulness metrics, a small validation set (50 queries, one model, one quantization) is run at full GPU, partial offload, and full CPU offload. If CU/SK scores fall within bootstrap confidence intervals of each other, the assumption that offload level doesn't affect faithfulness is empirically confirmed. Results reported in appendix.

**OOM as data:** The feasibility matrix — which model × quantization combinations fit at each simulated VRAM tier — is reported as a practical deployment guide alongside the faithfulness results. The failure mode taxonomy (OOM_LOAD, OOM_PROMPT, OOM_GENERATION, OOM_KV_PROGRESSIVE, TIMEOUT, ERROR_OTHER) classifies each infeasible configuration.

### Three-Pass Evaluation Protocol

Each query in the RGB counterfactual subset is evaluated under three prompting conditions that isolate distinct aspects of model behavior. This separates the question "does the model know the answer?" from "does it prefer context?" from "can it follow context?"

**Pass 1 — Baseline (No Retrieval).** The question is presented with no context. This measures raw Self-Knowledge (SK): what the model knows from its parametric memory alone. The prompt is minimal — just the question with the model's standard instruction wrapper. This establishes a per-query baseline: if the model doesn't know the answer without context, high CU in Passes 2–3 is expected and uninformative. If it *does* know the answer, then CU in later passes represents a genuine conflict between parametric and contextual signals.

```
<s>[INST] {question} [/INST]
```

**Pass 2 — Loose RAG (Behavioral Preference).** The question is presented with retrieved context using a neutral framing ("I have some background information that might be helpful"). This is the primary test condition — it measures whether the model *prefers* the context over its own weights when not explicitly instructed to do so. CU/SK metrics from RAGChecker and the Logit Gap are both measured on this pass.

```
<s>[INST] I have some background information that might be helpful:
{context}

Based on that, {question} [/INST]
```

**Pass 3 — Strict RAG (Forced Capability).** The question is presented with an explicit instruction to answer only from context. This is an ablation: if the model fails here, it lacks the *capability* to attend to context regardless of preference. If it succeeds here but fails in Pass 2, the issue is behavioral (the model can read context but chooses not to). The delta between Pass 2 and Pass 3 measures the gap between natural behavior and forced compliance.

```
<s>[INST] System: Answer using ONLY the provided context.
If the answer is not in the context, say "I do not know."

Context:
{context}

User: {question} [/INST]
```

**Analytical value:** The three-pass protocol transforms a single CU measurement into a diagnostic triple. A model that scores (Pass 1: correct, Pass 2: follows context, Pass 3: follows context) is genuinely deferring to retrieval despite knowing better — the strongest evidence for the gullibility thesis. A model that scores (Pass 1: correct, Pass 2: ignores context, Pass 3: follows context) has the capability but not the inclination — a behavioral issue that quantization might shift. A model that scores (Pass 1: wrong, Pass 2: follows context, Pass 3: follows context) tells us nothing about CU/SK tradeoffs since there was no parametric knowledge to override.

Note: Passes 1 and 3 are run on the RGB counterfactual subset only. The full dataset evaluation uses Pass 2 (Loose RAG) as the primary condition, consistent with standard RAG benchmarking methodology.

### Logit Gap Analysis

The Logit Gap is a mechanistic metric that measures the model's internal confidence conflict between its parametric memory and the provided context at the token probability level. Where CU/SK from RAGChecker measure the *outcome* (what the model said), the Logit Gap measures the *decision point* (how conflicted it was).

**Definition:** For a counterfactual query where the context provides answer C and the model's training data contains answer P:

```
Logit Gap = logit(C_token) - logit(P_token)
```

Where `C_token` and `P_token` are the first tokens of the contextual and parametric answers respectively. Positive gap means the model's probability distribution favors the contextual answer. Negative gap means it favors its parametric knowledge. Near-zero means high conflict.

**Implementation — white-box via llama-cpp-python:** The Logit Gap requires access to the full logit distribution, not just top-N probabilities. The llama.cpp HTTP server only exposes top-N token probabilities, which may not include a "suppressed" target token (e.g., the probability of "Biden" when the model strongly wants to say "Snoopy"). Direct model loading via the llama-cpp-python library with `logits_all=True` provides raw access to the complete logit vector at every position.

This creates a dual-path architecture: the standard evaluation pipeline uses the llama-server HTTP API for generation and RAGChecker scoring (Pass 2 and Pass 3 responses), while the Logit Gap module loads the model directly for single-token forward passes with full logit extraction. The model is loaded once per (model × quantization) combination and reused across all queries. KV cache is cleared between queries via standard cache eviction; full model reloading is unnecessary since transformer forward passes with cleared KV cache are stateless by construction.

**Scope:** Logit Gap is measured on the RGB counterfactual subset only, where ground-truth (parametric) and counterfactual (contextual) answer pairs are explicitly provided. It is measured on the Pass 2 (Loose RAG) prompt to capture natural behavioral preference rather than forced compliance.

**BPE tokenization handling:** Answer strings are tokenized with and without a leading space (e.g., both `"Biden"` and `" Biden"`) since BPE tokenizers treat these as distinct tokens. Only the first token of each answer is used — the first divergence token is where the model "decides" which path to take. Multi-token answers would conflate gap measurement with autoregressive generation dynamics.

**Aggregation:** Per-query gaps are aggregated to (model × quantization) level with bootstrap confidence intervals (1000 resamples). The aggregate is classified into curve types: Gullible (CI entirely above +0.5), Stubborn (CI entirely below −0.5), Resilient (mean near zero, CI spans zero), or Ambiguous.

**The thesis check:** Plot mean Logit Gap (y-axis) against quantization level (x-axis) per model family. If the curve trends upward (more positive) at lower quantization, the mechanism is confirmed: parametric confidence degrades faster than contextual attention, producing the CU increase observed in RAGChecker scores. This is the "why" behind the "what."

### Model Matrix

All models in GGUF format via llama.cpp.

| Model | Params | Quants | Role |
|-------|--------|--------|------|
| Qwen 2.5 14B Instruct | 14B | Q4_K_M, Q8_0, FP16 | Primary — strong RAG performer, quantization-resilient |
| Phi-4 14B | 14B | Q4_K_M, Q8_0, FP16 | Primary — reasoning-dense training, quantization-resilient |
| Mistral NeMo 12B Instruct | 12B | Q4_K_M, Q8_0 | Primary — low hallucination baseline |
| Llama 3.1 8B Instruct | 8B | Q4_K_M, Q8_0, FP16 | Control — known to degrade disproportionately under quantization |
| Qwen 2.5 7B Instruct | 7B | Q4_K_M, Q8_0 | Secondary |
| Gemma 2 9B Instruct | 9B | Q4_K_M, Q8_0 | Secondary |
| GLM-4 9B Chat | 9B | Q4_K_M | Secondary |
| SmolLM2 1.7B Instruct | 1.7B | Q8_0, FP16 | Floor test — minimal parametric knowledge |

### Frontier API Comparison (v5)

Five frontier API models are evaluated as ceiling/floor reference points. These are not part of the quantization-faithfulness curve (their precision and hardware are unknown/uncontrollable) but provide the comparison point every reviewer will ask for: "how do your $200-of-eBay-hardware results compare to just calling GPT-5?"

| Model | API | Model ID | Role |
|-------|-----|----------|------|
| GPT-5 | OpenAI | `gpt-5` | Default industry comparison — strongest overall benchmark performance |
| Claude Sonnet 4.5 | Anthropic | `claude-sonnet-4-5-20250929` | Strong reasoning, Sonnet over Opus for cost at volume |
| Gemini 2.5 Flash | Google | `gemini-2.5-flash` | Cheapest frontier-class API, high adoption |
| DeepSeek V3.2 | DeepSeek | `deepseek-chat` | Cheapest frontier API, open-weight sibling exists, China comparison point |
| GLM-5 | Zhipu AI (Z.ai) | `glm-5` | 744B MoE, lowest published hallucination rate, trained entirely on Huawei Ascend |

**What is measured:** Passes 2 and 3 (Loose RAG and Strict RAG) with identical prompts and retrieved context as local models. CU/SK via RAGChecker using the same judge model and scoring pipeline. Same ~500 RGB counterfactual queries, 3 seeds. Pass 1 (no retrieval) is also run — frontier models' vastly larger parametric knowledge means the baseline is almost always correct, making every counterfactual query a genuine conflict. This is useful context for interpreting CU/SK results.

**What is NOT measured:** Logit Gap — dead on arrival for API models. No API exposes full logit distributions. Claude provides no logprobs. Gemini provides limited top-N logprobs. The mechanistic layer of the study is local-model-only. Quantization axis — doesn't exist for API models. VRAM simulation — meaningless. These models sit outside the main experimental matrix.

**How results are presented:** Frontier models appear as horizontal reference lines on the quantization-faithfulness curves, not as data points within them. Each frontier model gets a single CU score and single SK score (with bootstrap CIs) plotted against the local model curves. This visual immediately answers "where does a Q4 quantized 14B model sit relative to GPT-5 on RAG faithfulness?"

**The narrative:** If a local Q4 model exceeds frontier CU scores (plausible — large models have strong parametric priors that resist context override), the headline writes itself: "a Qwen 2.5 14B at Q4 on $200 of surplus hardware achieves higher RAG faithfulness than GPT-5 because its weaker parametric memory creates less resistance to retrieved context." If frontier models dominate across all metrics, the contribution shifts to the cost-performance Pareto frontier.

**Methodological constraints for API models:**

*Temperature/determinism:* API models may not honor temperature=0.0 identically to llama.cpp. OpenAI uses `seed` for reproducibility but doesn't guarantee determinism. DeepSeek and GLM-5 support temperature=0.0. Variance across seeds is expected and reported via CIs.

*Model versioning:* API model behavior changes over time. All model version strings and run dates are pinned and documented. Local GGUF models are frozen artifacts; API models are moving targets. The comparison is a snapshot, not a permanent ranking.

*Reasoning mode:* All API calls use standard chat completions mode, not reasoning/thinking variants (o4, deepseek-reasoner, etc.). Chain-of-thought inflates token counts and changes generation dynamics in ways not comparable to local models. Non-thinking mode keeps it apples-to-apples.

**Cost estimate:** ~500 queries × 3 passes × 5 APIs × 3 seeds = ~22,500 API calls. At typical pricing this is ~$50-80 total. DeepSeek and GLM-5 pricing is substantially lower. Not a budget concern.

### Evaluation

**Primary framework:** RAGChecker — Context Utilization (CU), Self-Knowledge (SK), Noise Sensitivity (NS). These directly operationalize the hypothesis. CU measures how much the model uses retrieved context. SK measures how much it relies on parametric memory. The CU/SK ratio is the core signal.

**Mechanistic framework:** Logit Gap — token-level confidence measurement on RGB counterfactual subset (local models only). Provides causal evidence for the CU/SK observations by quantifying the internal probability conflict between parametric and contextual answers.

**Baseline framework:** RAGAS — Faithfulness, Answer Relevancy, Context Precision, Context Recall.

**Judge model:** Qwen 2.5 14B (fixed across all evaluations — local and API — for consistency).

**Datasets:** RGB (noise robustness, negative rejection, counterfactual robustness), Natural Questions (nq_open, general RAG accuracy), HaluEval (hallucination detection with human labels).

**RGB counterfactual sub-analysis:** Within the RGB counterfactual subset, the Negative Rejection subset (where the model should reject the context) is analyzed separately from the Noise Robustness subset. This distinguishes the "faithful" failure mode (model blindly repeats malicious context) from the "robust" success mode (model correctly ignores irrelevant noise). High CU on Noise Robustness is good; high CU on Negative Rejection is a vulnerability. Reporting them together would mask this distinction.

**Statistical rigor:** ~500 queries per dataset, 3 datasets, 3 seeds per configuration. Bootstrap resampling (1000×) for 95% confidence intervals.

### Optane Weight-Storage Tier Benchmark

An auxiliary experiment measuring whether model weight storage medium affects output quality or purely throughput. The Dell T7820 has 384GB Intel Optane DCPMM alongside 48GB DDR4, enabling a controlled A/B comparison: same model, same queries, weights loaded from Optane App Direct (via mmap) versus weights in DDR4.

**Hypothesis:** Weight storage tier is a pure throughput variable with zero impact on output quality. The forward pass reads weights as read-only tensors; whether those reads are served from DDR4 (~90 GB/s bandwidth) or Optane (~20 GB/s sequential read in App Direct mode) should produce bit-identical logits since the values are the same — only the latency to access them changes.

**If confirmed:** Optane DCPMM is validated as a viable weight-storage tier for quality-insensitive RAG deployments (e.g., batch processing where latency doesn't matter). At current used prices (~$3/GB vs ~$5/GB for DDR4), this has practical deployment implications for large model weight storage.

**If refuted:** A quality difference from storage tier alone would be a genuinely surprising finding, likely indicating either numerical instability in the memory controller path or a subtle interaction between weight access latency and KV cache management. Worth a standalone investigation.

**Metrics recorded:** Tokens per second, time to first token, CU/SK scores on a fixed query subset (50 queries, one model, one quantization), and bit-level output comparison (greedy decoding should be deterministic if weights are identical).

This is a single-row addition to the hardware matrix. It does not change the main experimental design. Results reported as an appendix.

### Hardware

Single platform: NVIDIA V100 SXM2 32GB (via SXM2-to-PCIe adapter) in a Dell Precision T7820 with dual Xeon Gold 6230 CPUs, 48GB DDR4, 384GB Optane DCPMM.

### Threats to Validity

**Faithfulness vs. gullibility confound.** High CU at aggressive quantization levels may indicate the model has lost its ability to fact-check retrieved context, not that it has become a better RAG system. A model that blindly repeats a malicious prompt scores high on CU but represents a failure mode, not a success. This is partially mitigated by the separate analysis of RGB Negative Rejection (where the model *should* reject context) and by the Logit Gap analysis (which distinguishes "lost confidence in weights" from "gained attention to context"). The three-pass protocol further disambiguates: if a model scores high CU in Pass 2 but also correctly rejects adversarial context in Negative Rejection queries, the CU increase reflects appropriate deference rather than gullibility.

**Floating-point divergence across offload levels.** CPU (BLAS) and GPU (CUDA) floating-point accumulation is not bit-exact. When splitting layers across backends via partial offload, non-associative floating-point addition can cause micro-divergences. At temperature 0.0 with near-tied logits, a backend switch could flip token selection and cascade through autoregressive generation. The offload equivalence validation (50-query subset, three offload levels, bootstrap CIs) tests whether this divergence is systematic. If CIs overlap, macro-metrics are treated as stable across offload levels. Micro-divergence is expected and acknowledged; the claim is stability of aggregate metrics, not bit-identical outputs.

**Bandwidth confound in simulated VRAM tiers.** Constraining `--n-gpu-layers` on a V100 gives the simulated tier the memory bandwidth of a datacenter card (~900 GB/s HBM2) rather than the consumer card it's simulating (~336 GB/s for a 1660 Super, ~512 GB/s for MI25). This study measures model capability (does it fit and produce faithful outputs?), not wall-clock latency. Any inference speed or latency conclusions drawn from simulated tiers would be invalid. The feasibility matrix answers "can it run?" and the faithfulness metrics answer "is it faithful?" — neither claims to answer "how fast?"

**Frontier API comparison is a snapshot.** API model behavior changes across versions and over time. Model IDs and run dates are pinned, but results represent a point-in-time comparison. Local GGUF models are frozen artifacts with reproducible checksums. API models are not. The frontier comparison provides context, not a permanent ranking.

### Novel Contributions

1. **Quantization-faithfulness curves** — systematic CU/SK measurements across quantization levels for multiple model architectures, testing whether quantization improves RAG faithfulness by weakening parametric memory. Nobody publishes this for RAG-specific metrics.
2. **Logit Gap as mechanistic evidence** — token-probability analysis showing *why* quantization affects faithfulness, not just *that* it does. Distinguishes "confidence erosion" from "attention improvement" as the underlying mechanism.
3. **Three-pass evaluation protocol** — separating baseline knowledge, behavioral preference, and forced-context capability into independently measurable signals per query, enabling diagnostic classification of model behavior rather than single-score ranking.
4. **Frontier API ceiling comparison (v5)** — anchoring local quantized model results against five major commercial APIs (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Flash, DeepSeek V3.2, GLM-5) using identical prompts, retrieval, and evaluation. Tests whether cheap quantized local models can match or exceed frontier RAG faithfulness due to weaker parametric priors.
5. **Simulated VRAM tier methodology** — isolating VRAM capacity from other hardware variables on a single GPU. OOM boundaries reported as a practical deployment feasibility matrix; faithfulness measured across all offload levels.
6. **Architecture-dependent quantization degradation for RAG** — extending known general-benchmark degradation asymmetries (Llama vs Qwen/Phi) to RAG-specific faithfulness metrics, identifying which model families maintain context utilization under quantization.
7. **Optane DCPMM as weight-storage tier benchmark** — first published comparison of model weight storage medium (Optane App Direct vs DDR4) effect on inference output quality, testing whether storage tier is a pure throughput variable.

### Tech Stack

- **Inference (generation):** llama.cpp server (GGUF models, OpenAI-compatible HTTP API)
- **Inference (logit gap):** llama-cpp-python direct binding (`logits_all=True` for raw logit access)
- **Inference (frontier):** OpenAI, Anthropic, Google, DeepSeek, Zhipu AI APIs (all via OpenAI-compatible clients)
- **Vector store:** ChromaDB (local, file-based)
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers, CPU)
- **Evaluation:** RAGChecker (primary), RAGAS (baseline)
- **Infrastructure:** Docker (multi-target: CPU/CUDA), GitHub Actions CI
- **Analysis:** Python (numpy, scipy, pandas, matplotlib, seaborn)
