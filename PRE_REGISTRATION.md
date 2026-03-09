# CompRAG Pre-Registration

**Committed before first experimental run.**
**Date: 2026-03-08**
**Git commit hash: (filled at commit time)**

---

## Hypothesis

There exists a quantization-dependent faithfulness curve whose shape varies by model architecture. Aggressive GGUF quantization weakens a model's parametric memory, potentially reducing its confidence in internally stored knowledge and increasing reliance on retrieved context, producing higher Context Utilization (CU) scores. However, sufficient quantization eventually degrades the reasoning capacity needed to synthesize answers from context, creating a faithfulness-capability tradeoff.

## Headline Prediction

Model architecture and GGUF quantization level correlate more strongly with RAG faithfulness than model size past a surprisingly low parameter threshold.

## Global Falsification Criterion

If no primary model architecture demonstrates the Gullible pattern (CU monotonically increasing with quantization level from FP16 → Q3_K_M), the core hypothesis is refuted.

## Per-Model Predictions

### Qwen 2.5 14B Instruct
- **Predicted pattern:** Gullible
- **Prediction:** CU increases monotonically from FP16 → Q3_K_M
- **Mechanism:** Quantization erodes parametric confidence; model defers to retrieved context

### Llama 3.1 8B Instruct
- **Predicted pattern:** Non-monotonic (inverted U)
- **Prediction:** CU increases from FP16 → Q5_K_M, decreases from Q4_K_M → Q3_K_M as capability degradation outpaces parametric erosion
- **Statistical criterion:** Q3_K_M bootstrap CI upper bound must fall below the peak quantization level's bootstrap CI lower bound. If CIs overlap, the prediction is indeterminate.

### Qwen 2.5 7B Instruct
- **Predicted pattern:** Gullible (matching Qwen 14B)
- **Prediction:** CU curve shape qualitatively matches Qwen 2.5 14B, controlling for the size confound in the 14B-vs-8B cross-architecture comparison

### Phi-4 14B
- **Predicted pattern:** Resilient
- **Prediction:** CU remains relatively flat across quantization levels from FP16 → Q3_K_M. Reasoning-dense training and known quantization resilience on general benchmarks preserves parametric confidence under weight degradation, maintaining consistent context utilization behavior.

## Three Predicted Outcome Patterns

- **Resilient:** CU stays flat across quantization levels — context attention is quantization-robust
- **Stubborn:** CU decreases with quantization — model defaults to parametric weights under degradation
- **Gullible:** CU increases with quantization — parametric confidence erodes and model defers to context

Classification uses data-driven thresholds based on bootstrap CI position relative to the FP16 baseline.

## Scope Boundaries (What Is NOT Claimed)

- No claim about quantization **method** differences (GPTQ, AWQ, SqueezeLLM, etc.) — all models use GGUF round-to-nearest quantization only
- No claim about adaptive retrieval or retrieval pipeline optimization
- No causal mechanism claim — observed correlations between quantization level and CU are characterized, not explained
- No claim that high CU is inherently desirable — the faithfulness/gullibility confound is acknowledged and partially addressed by separate analysis of Negative Rejection queries
- No claim about API model quantization or internals — frontier models are reference lines, not experimental variables

## Model Matrix

### Primary Models (6 quantization levels each: Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16)

| Model | Params | Role |
|-------|--------|------|
| Qwen 2.5 14B Instruct | 14B | Strong RAG performer, quantization-resilient |
| Phi-4 14B | 14B | Reasoning-dense training, matched size with Qwen 14B |
| Llama 3.1 8B Instruct | 8B | Known quantization-fragile, predicted inverted-U |
| Qwen 2.5 7B Instruct | 7B | Size-controlled comparison with Llama 8B |

### Secondary Models (2 quantization levels: Q4_K_M, Q8_0)

| Model | Params | Role |
|-------|--------|------|
| Mistral NeMo 12B Instruct | 12B | Additional architecture |
| Gemma 2 9B Instruct | 9B | Additional architecture |

### Floor Test (2 levels: Q8_0, FP16)

| Model | Params | Role |
|-------|--------|------|
| SmolLM2 1.7B Instruct | 1.7B | Minimal parametric knowledge baseline |

### Cross-Model Comparison Rationale

- **Architecture effect at matched size:** Qwen 14B vs Phi-4 14B; Qwen 7B vs Llama 8B vs Gemma 9B
- **Size effect within architecture:** Qwen 14B vs Qwen 7B
- **Quantization robustness:** Llama 8B (fragile) vs Qwen 7B (resilient) at near-matched size

## Frontier Reference Line

| Model | API | Model ID |
|-------|-----|----------|
| Claude Sonnet 4.6 | Anthropic | `claude-sonnet-4-6` |

Presented as a single horizontal reference line on quantization-faithfulness curves. Not an experimental variable. Evaluated using Pass 2 and Pass 3 only, same queries and retrieval pipeline as local models.

## Evaluation Protocol

### Three-Pass Protocol (RGB Counterfactual Subset Only)

- **Pass 1 — Baseline (No Retrieval):** Measures raw Self-Knowledge (SK)
- **Pass 2 — Loose RAG (Behavioral Preference):** Primary test condition. Neutral framing.
- **Pass 3 — Strict RAG (Forced Capability):** Explicit instruction to answer only from context.

### Metrics

- **Primary:** Context Utilization (CU), Self-Knowledge (SK) — via RAGChecker
- **Derived:** Preference_Gap = Pass3_CU − Pass2_CU
- **Supporting:** Hallucination, Faithfulness, Precision, Recall, F1 — via RAGChecker + RAGAS

### Query Counts

- Counterfactual: 100 queries × 3 passes = 300 inference calls per model config
- Noise Robustness: 100 queries × 1 pass = 100 inference calls per model config
- Negative Rejection: 100 queries × 1 pass = 100 inference calls per model config
- **Total: 300 unique queries, 500 inference calls per model configuration**

### Statistical Methods

- Bootstrap resampling: 1000 iterations, 95% confidence intervals
- Bootstrap computed over queries (not seeds) for local models if determinism pilot confirms bit-identical outputs
- For frontier API: 3 seeds per configuration; bootstrap reflects both query and seed variance

## Judge Model

- **Primary (bulk scoring):** Cohere Command R 35B Q4_K_M (local)
- **Validation (appendix spot-check):** Claude Sonnet 4.6 (Anthropic API), 100-record sample
- **Judge validation result:** Command R validated against DeepSeek at CU κ=0.853, overall CU+faithfulness κ>0.93 across pass types

## Locked Parameters

- Embedding model: all-MiniLM-L6-v2 (sentence-transformers, CPU)
- Vector store: ChromaDB, pre-built index (~22K chunks)
- Retrieval: top-k=5, 300-word chunks
- Temperature: 0.0
- Seed: 42
- max_tokens: 512
- GPU offload: full (--n-gpu-layers -1), no partial offload
- Hardware: NVIDIA V100 SXM2 32GB, single GPU

## Pinned Versions

- RAGChecker version: 0.1.9
- RAGAS version: 0.4.3
- llama.cpp version: b8182 (05728db18)
- RGB dataset download date: 2026-02-28
- RGB dataset source URL: https://github.com/chen700564/RGB

## GGUF Checksums (SHA256)

```
# Primary: Qwen 2.5 14B Instruct
2f68ac3ba018f7de7641229f19adafde5e59d02bbf5651fdbcc510bb9f3facca  Qwen2.5-14B-Instruct-Q3_K_M.gguf
e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008  Qwen2.5-14B-Instruct-Q4_K_M.gguf
48ad2dafedac636f62847f5338c356cd21f5cfa6b1e2c885360cb10c890b8cb2  Qwen2.5-14B-Instruct-Q5_K_M.gguf
18cd6b7d5feb00c57ff81ede8f2164ffd86be90dbee9c05bf09ded1ab179740d  Qwen2.5-14B-Instruct-Q6_K.gguf
23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf  Qwen2.5-14B-Instruct-Q8_0.gguf
0811262438ff013dd1421b7b4c6cbd526f0a392a5deec0f7ba7ba6722f49d83b  Qwen2.5-14B-Instruct-f16.gguf

# Primary: Phi-4 14B
8bf3e36c72aec8107ed9942bc9f1c89f4b7786f17c795e305b5993426293d14f  phi-4-Q3_K_M.gguf
009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9  phi-4-Q4_K_M.gguf
b4b1ecedddfdd25a9c44c10a77bb118bcbb6a9004234286c7d4a4510c907f073  phi-4-Q5_K_M.gguf
537248da5492a9a34b1d811307c6b9226d9539a81653aa9bb0d7ec6104169320  phi-4-Q6_K.gguf
68efc525888cc34fc2cb539c2f4409653d57994e35b3a5735d2a59b871871bf3  phi-4-Q8_0.gguf
f92820e6fec31c444795020c78e74b514f08455371bcf01977bc912c99bb4dba  phi-4-f16.gguf

# Primary: Llama 3.1 8B Instruct
6be122b5c8f2a33974953e58e0ffd2be505661acc6f4caf733e4bca130e89fea  Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf
7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
14e10feba0c82a55da198dcd69d137206ad22d116a809926d27fa5f2398c69c7  Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
33981adf6bae52c503fb5c24f72539010632f7ed290a56c1315a8cd50adca587  Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
139c255d857940bc945b2a3242fbbb2641b59191d79413bcad9b74fa1784c7b0  Meta-Llama-3.1-8B-Instruct-f16.gguf

# Primary: Qwen 2.5 7B Instruct
6738a2d4f9b280c55b2a19a7ab27334a75d2cafc6ef82a11a075f4b5613eb736  Qwen2.5-7B-Instruct-Q3_K_M.gguf
65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423  Qwen2.5-7B-Instruct-Q4_K_M.gguf
2e998d7e181c8756c5ffc55231b9ee1cdc9d3acec4245d6e27d32bd8e738c474  Qwen2.5-7B-Instruct-Q5_K_M.gguf
489138dfed4f04cd6dea56e5a8423e4aa05a0318cce2a4a72250fe1278e97cf8  Qwen2.5-7B-Instruct-Q6_K.gguf
9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b  Qwen2.5-7B-Instruct-Q8_0.gguf
863c978275bca3fdacdde06cbfc1a65f2bd65210bb7ecfd9ded3b24219d81b54  Qwen2.5-7B-Instruct-f16.gguf

# Secondary: Mistral NeMo 12B
7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46  Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
7985eab9b84d7d514253a53f372095d51f941f84a62cadaa0ccf6e1b3fbef494  Mistral-Nemo-Instruct-2407-Q8_0.gguf

# Secondary: Gemma 2 9B
13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe  gemma-2-9b-it-Q4_K_M.gguf
07ac9ec01217d7bf2976a3a0dbf564a91b6957f94ab2ad30671eb6100f64bc2b  gemma-2-9b-it-Q8_0.gguf

# Floor: SmolLM2 1.7B
0c6e8955788b1253f418c354a4bdc4cf507b8cfe49c48bb10c7c58ae713cfa2a  SmolLM2-1.7B-Instruct-Q8_0.gguf
cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852  SmolLM2-1.7B-Instruct-f16.gguf

# Judge: Command R 35B
9fd704afe04c123ce11f9f25709f6b233065853dc346ebc3b7eeff8edc4d20a0  c4ai-command-r-08-2024-Q4_K_M.gguf
```

## Determinism Pilot

- Config: Qwen 2.5 14B Instruct Q4_K_M
- Protocol: 10 seeds × 50 queries
- Expected result: bit-identical outputs under greedy decoding (temp=0.0, seed=42)
- If confirmed: seed-based replication dropped for local models; bootstrap over queries only
- If not confirmed: characterize inter-run variance, include seed variance in bootstrap CIs
- **Result:** CONFIRMED deterministic. 10 seeds × 50 queries, all bit-identical. Single seed (42) used for all runs, bootstrap over queries only.

## Preference_Gap Definition

Preference_Gap = Pass3_CU − Pass2_CU

- Gap ≈ 0: Model naturally defers as much as when forced
- Large positive gap: Capability without preference (model can use context but doesn't by default)
- Negative gap: Model uses context more in loose prompt than strict (likely noise or instruction-following artifact)
- Configurations where Pass3_CU drops below minimum threshold are visually flagged (distinct markers) but not excluded

## OOM/Error Handling

OOM events and errors are logged as metadata, not treated as failures. Six categories:
1. Model fails to load (exceeds VRAM)
2. KV cache OOM during generation
3. Generation timeout
4. Malformed output (unparseable response)
5. Server crash/restart
6. Scoring failure (judge model error)

These boundaries represent hardware-attributable constraints and are reported as data points.

## Priority Claim Verification

Search conducted before commit on: r/LocalLLaMA, HuggingFace, arxiv
Search terms: "quantization RAG faithfulness," "GGUF RAG evaluation," "quantization context utilization"
**Result:** No direct prior work found measuring CU/faithfulness across a multi-level GGUF quantization sweep. Closest: Maistro et al. (2024) "The Impact of Quantization on RAG" (arxiv:2406.10251) compared FP16 vs INT4 on 2 models (7B/8B) using task accuracy, not RAGChecker metrics. No multi-point quant curve, no CU measurement, no architecture comparison. Also: "Which Quantization Should I Use?" (arxiv:2601.14277) evaluates GGUF quants on general benchmarks but not RAG-specific faithfulness. CompRAG's contribution — systematic CU/faithfulness curves across 6 quant levels × 7 architectures — appears novel.
