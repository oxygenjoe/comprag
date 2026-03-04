# Methodology

## 1. Research Question & Thesis

**Research question:** Does model size inversely correlate with RAG faithfulness when retrieval quality is held constant?

**Thesis:** Smaller language models (1.7B--14B parameters) may produce more faithful retrieval-augmented generation outputs than larger models because they possess less parametric knowledge to contradict retrieved context. A model with weaker internal priors is more likely to defer to the retrieval pipeline, yielding higher context utilization, lower self-knowledge leakage, and stronger noise sensitivity scores. The hardware tier running the model affects throughput and latency but has minimal impact on groundedness metrics.

This study tests both claims by sweeping model size, quantization level, and hardware tier while holding the retrieval pipeline constant. The independent variables are the generator model, its quantization, and the hardware it runs on. The dependent variables are six quality metrics measuring faithfulness and groundedness, plus four performance metrics measuring throughput and resource consumption. If the thesis holds, smaller models will score higher on context_utilization and lower on self_knowledge across datasets, while hardware tier will show no statistically significant effect on quality metrics.

### 1.1 Experimental Design

The experiment follows a three-factor factorial design with constant retrieval:

- **Factor A (Model):** 9 levels, spanning 1.7B to 14B parameters across 6 architecture families
- **Factor B (Quantization):** Up to 3 levels per model (Q4_K_M, Q8_0, FP16), testing whether lossy weight compression degrades faithfulness
- **Factor C (Hardware Tier):** 6 levels (V100, MI25, 1660 Super, FPGA, CPU, Optane), testing whether inference substrate affects quality metrics
- **Constant:** The entire retrieval pipeline (embedding model, vector store, chunking, top-k) is identical across all runs

The design is not fully crossed: not every model-quantization-hardware combination is feasible (e.g., 14B FP16 cannot run on a 6GB GPU). The feasibility matrix in `config/models.yaml` encodes valid combinations. Analysis accounts for this unbalanced design by comparing within feasible subsets rather than assuming complete factorial coverage.

Each valid combination is evaluated against three datasets (RGB with 4 subsets, NQ, HaluEval), yielding 8 distinct evaluation contexts per combination. With a minimum of 3 runs per combination-dataset pair, the total number of individual inference runs exceeds 1,000 for the full matrix.

### 1.2 Null Hypotheses

The study tests the following null hypotheses:

- **H0_model:** There is no significant difference in `faithfulness`, `context_utilization`, or `self_knowledge` scores between models of different parameter counts when retrieval is held constant.
- **H0_quant:** There is no significant difference in quality metrics between quantization levels of the same model.
- **H0_hw:** There is no significant difference in quality metrics between hardware tiers running the same model at the same quantization.

Rejection of H0_model and H0_quant, combined with failure to reject H0_hw, would support the thesis.

## 2. Independent Variables

### 2.1 Model

Nine models span the 1.7B--14B parameter range, selected to cover multiple architecture families and training lineages. The model registry is defined in `config/models.yaml`.

| Model | Parameters | Context Length | Architecture Family | Priority |
|-------|-----------|----------------|-------------------|----------|
| Qwen 2.5 14B Instruct | 14B | 32768 | Qwen/Alibaba | HIGH |
| Phi-4 14B | 14B | 16384 | Phi/Microsoft | HIGH |
| Mistral NeMo 12B Instruct | 12B | 32768 | Mistral | HIGH |
| Llama 3.1 8B Instruct | 8B | 131072 | Llama/Meta | HIGH (baseline) |
| Qwen 2.5 7B Instruct | 7B | 32768 | Qwen/Alibaba | MEDIUM |
| Gemma 2 9B Instruct | 9B | 8192 | Gemma/Google | MEDIUM |
| GLM-4 9B Chat | 9B | 131072 | GLM/Zhipu | MEDIUM |
| SmolLM2 1.7B Instruct | 1.7B | 8192 | SmolLM/HuggingFace | LOW (floor test) |
| BitNet 1.58b (TBD) | 3B | TBD | BitNet/Microsoft | LOW (FPGA only) |

Llama 3.1 8B Instruct serves as the primary baseline due to its wide community adoption and extensive benchmark coverage. SmolLM2 1.7B provides the floor test -- if a 1.7B model can produce faithful RAG outputs, it strengthens the thesis that retrieval compensates for limited parametric knowledge. BitNet 1.58b is included as an extreme quantization test on dedicated FPGA hardware.

All models are served in GGUF format via llama.cpp's OpenAI-compatible server on port 8080 (`config/eval_config.yaml`, `generation.server_url`). GGUF files are sourced from HuggingFace (bartowski or official quants) and downloaded via `scripts/download_models.py`. The exact filename, HuggingFace repository, and file size for each quantization level are recorded in `config/models.yaml` under each model's `quantizations` block.

### 2.2 Quantization

Each model is tested at available quantization levels from `config/models.yaml`:

| Quantization | Effective Bits | Typical Compression vs FP16 | Notes |
|-------------|---------------|---------------------------|------|
| Q4_K_M | ~4.5 | ~4x | Primary eval quant; k-quant with mixed precision per tensor group. Fits most tiers. |
| Q8_0 | 8 | ~2x | Higher fidelity round-trip quantization. Fits V100 and MI25. |
| FP16 | 16 | 1x (baseline) | Full precision half-float. Only on V100 for 8B/14B models. |
| 1.58-bit | 1.58 | ~10x | BitNet ternary weights (-1, 0, 1). FPGA only. |

Not all models have all quantizations. The exact availability per model is encoded in `config/models.yaml` under each model's `quantizations` block. For example:

- Qwen 2.5 14B Instruct: Q4_K_M (8.9 GB), Q8_0 (15.7 GB), FP16 (29.5 GB)
- Phi-4 14B: Q4_K_M (8.4 GB), Q8_0 (14.8 GB), FP16 (27.8 GB)
- Mistral NeMo 12B Instruct: Q4_K_M (7.4 GB), Q8_0 (13.0 GB)
- Llama 3.1 8B Instruct: Q4_K_M (4.9 GB), Q8_0 (8.5 GB), FP16 (16.1 GB)
- Qwen 2.5 7B Instruct: Q4_K_M (4.7 GB), Q8_0 (8.1 GB)
- Gemma 2 9B Instruct: Q4_K_M (5.8 GB), Q8_0 (10.1 GB)
- GLM-4 9B Chat: Q4_K_M (5.5 GB)
- SmolLM2 1.7B Instruct: Q8_0 (1.8 GB), FP16 (3.4 GB)
- BitNet 1.58b: 1.58-bit native (~0.5 GB)

Quantization tests whether lossy weight compression degrades faithfulness -- a model that trusts retrieved context should be less affected by weight quantization than one relying on parametric recall. If quantization primarily degrades the model's internal knowledge while preserving its ability to read and repeat context, then quality metrics should remain stable across quantization levels for faithful models.

### 2.3 Hardware Tier

Six hardware tiers are defined in `config/hardware.yaml`, spanning GPU, CPU, Optane persistent memory, and FPGA:

| Tier ID | Hardware | Compute Type | VRAM/RAM | CPU | Software Stack | Status |
|---------|----------|-------------|----------|-----|---------------|--------|
| `v100` | NVIDIA V100 SXM2 32GB (Dell T7820) | GPU | 32GB HBM2 | Xeon Gold 6230 (20C/40T) | CUDA 12.x, llama.cpp (GGML_CUDA=1) | Incoming |
| `mi25` | AMD MI25 16GB | GPU | 16GB HBM | TBD | ROCm, llama.cpp (GGML_HIPBLAS=1) | Available |
| `1660s` | NVIDIA 1660 Super 6GB | GPU | 6GB GDDR6 | E5-2667 v4 (8C/16T) | CUDA 12.x, llama.cpp (GGML_CUDA=1) | Available |
| `fpga` | Inspur FPGA board + 16GB SODIMM | FPGA | 16GB DDR4 | N/A | TBD (BitNet runtime) | Planned |
| `cpu` | Intel E5-2667v4 + 32GB DDR4 | CPU | 32GB DDR4 | E5-2667 v4 (8C/16T) | llama.cpp CPU (AVX2) | Available |
| `optane` | E5-2667v4 + Optane DCPMM 384GB | CPU | 384GB Optane (App Direct) | E5-2667 v4 (8C/16T) | llama.cpp CPU (weights on Optane DAX) | Available |

### 2.4 Model-Hardware Feasibility Matrix

Not every model-quantization pair runs on every tier. The feasibility matrix in `config/models.yaml` encodes which combinations are valid:

| Model | V100 32GB | MI25 16GB | 1660S 6GB | FPGA | CPU 32GB | Optane 384GB |
|-------|-----------|-----------|-----------|------|----------|-------------|
| Qwen 2.5 14B (Q4/Q8/FP16) | All | Q4 (tight) | No | No | Q4 if patient | Q4 if patient |
| Phi-4 14B (Q4/Q8/FP16) | All | Q4 (tight) | No | No | Q4 if patient | Q4 if patient |
| Mistral NeMo 12B (Q4/Q8) | All | Q4/Q8 | No | No | Q4 if patient | Q4 if patient |
| Llama 3.1 8B (Q4/Q8/FP16) | All | Q4/Q8 | Q4 (tight) | No | Q4 primary | Q4 primary |
| Qwen 2.5 7B (Q4/Q8) | All | Q4/Q8 | Q4 (tight) | No | Q4 primary | Q4 primary |
| Gemma 2 9B (Q4/Q8) | All | Q4/Q8 | Q4 (tight) | No | Q4 primary | Q4 primary |
| GLM-4 9B (Q4) | Q4 | Q4 | Q4 (tight) | No | Q4 | Q4 |
| SmolLM2 1.7B (Q8/FP16) | All | All | All | No | All | All |
| BitNet 1.58b | No | No | No | 1.58-bit native | No | No |

Entries marked "tight" or "if patient" indicate combinations that are technically feasible but constrained by VRAM headroom (may OOM under load) or inference speed (<5 tok/s on CPU). The runner (`comprag/runner.py`) logs OOM errors and moves to the next combination rather than crashing the full evaluation matrix.

The hardware tier is expected to affect only performance metrics (tokens/sec, latency), not quality metrics. If hardware tier does affect quality, it would indicate non-determinism in the inference backend, thermal throttling causing numerical drift, or VRAM pressure forcing the model to truncate KV cache.

## 3. Controlled Variables

The following variables are held constant across all evaluation runs to isolate the effect of model, quantization, and hardware.

### 3.1 Retrieval Pipeline

Defined in `config/eval_config.yaml` under the `retrieval` block and implemented in `comprag/retriever.py`:

| Parameter | Value | Source |
|-----------|-------|--------|
| Vector store | ChromaDB (local, file-based) | `retrieval.vector_store` |
| Persist directory | `index/` | `retrieval.index_dir` |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | `retrieval.embedding_model` |
| Embedding dimension | 384 | `retrieval.embedding_dim` |
| Chunk size | 300 whitespace words | `retrieval.chunk_size` |
| Chunk overlap | 64 words | `retrieval.overlap` |
| Top-k | 5 chunks per query | `retrieval.top_k` |
| Reranking | None (phase 1) | N/A |

The embedding model is `all-MiniLM-L6-v2` from sentence-transformers, a small (22M parameter), fast model that runs on CPU. It produces 384-dimensional embeddings. The single source of truth for the embedding model name is `eval_config.yaml`; the retriever reads the model name from ChromaDB collection metadata at init, falling back to config if metadata is absent (`comprag/retriever.py`, `_get_default_embedding_model()`).

The chunking strategy uses whitespace word boundaries (not BPE tokens) with 300 words per chunk and 64-word overlap. This corresponds to approximately 400--500 BPE tokens per chunk depending on the tokenizer. The comment in `eval_config.yaml` explicitly documents this: `# Whitespace words, not BPE tokens. 300 words ~ 400-500 BPE tokens.`

Index integrity is enforced at query time: `validate_index()` in `comprag/retriever.py` raises `ValueError` if the collection's `embedding_model` or `chunk_size_words` metadata differs from `eval_config.yaml`. Collections use content-addressed names (e.g., `comprag_rgb_noise_robustness_300w_a3f7c2d1`) computed by `make_collection_name()` in `scripts/build_index.py` to prevent stale index reuse. The hash is derived from `sha256("{dataset}|{embedding_model}|{chunk_size}|{chunk_overlap}")[:8]`.

Per-dataset collections isolate different corpus sources:

| Collection Pattern | Dataset | Corpus Source |
|-------------------|---------|---------------|
| `comprag_rgb_noise_robustness_300w_<hash>` | RGB | Per-question passages from RGB dataset |
| `comprag_rgb_negative_rejection_300w_<hash>` | RGB | Per-question passages from RGB dataset |
| `comprag_rgb_information_integration_300w_<hash>` | RGB | Per-question passages from RGB dataset |
| `comprag_rgb_counterfactual_robustness_300w_<hash>` | RGB | Per-question passages from RGB dataset |
| `comprag_nq_wiki_300w_<hash>` | NQ | Wikipedia corpus (2018-12-20 dump) |
| `comprag_halueval_300w_<hash>` | HaluEval | Per-sample `knowledge` fields |

Collection names set to `"auto"` in `eval_config.yaml` are resolved at runtime by `resolve_collection_name()` in `comprag/retriever.py`.

### 3.2 Prompt Template

A single prompt template is used for all models, all hardware tiers, and all datasets. Stored in `config/prompt_template.txt` and referenced by `config/eval_config.yaml` at `generation.prompt_template`:

```
<|system|>
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer the question, say
"I cannot answer this question based on the provided context."
Do not use any knowledge from your training data.

<|context|>
{retrieved_chunks}

<|user|>
{query}

<|assistant|>
```

The `{retrieved_chunks}` and `{query}` placeholders are filled by the runner (`comprag/runner.py`, via `format_prompt()` in `comprag/generator.py`). The llama.cpp server handles chat template conversion per model (ChatML for Qwen, Llama template for Llama 3.1, etc.) -- only the message content is controlled, not the template wrapper. This means the system prompt, context, and user query are identical across models; only the special tokens wrapping them differ.

The prompt instructs the model to use ONLY the provided context and to explicitly refuse if context is insufficient. This instruction is critical for the negative rejection subset of RGB and for the self_knowledge metric: a model that follows this instruction will score low on self_knowledge (good) and high on negative_rejection_rate (good).

### 3.3 Generation Parameters

Locked in `config/eval_config.yaml` under the `generation` block. These parameters are enforced in `comprag/runner.py` via the `_LOCKED_GEN_PARAMS` constant:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Greedy decoding; minimizes stochastic variation between runs |
| `top_p` | 1.0 | No nucleus sampling (redundant with temp=0, but explicit for reproducibility) |
| `max_tokens` | 512 | Sufficient for factoid answers without truncation risk |
| `seed` | 42 | Deterministic sampling where supported by the llama.cpp backend |

These parameters are passed to the llama.cpp server via `generation.server_url` (`http://localhost:8080`) and `generation.endpoint` (`/v1/chat/completions`). The runner sends them in the OpenAI-compatible request payload. Stop sequences are model-appropriate and handled by the llama.cpp server based on the loaded model's tokenizer.

Temperature 0.0 produces greedy (argmax) decoding. Combined with a fixed seed, this should yield identical outputs across runs on the same hardware. However, llama.cpp does not guarantee bitwise reproducibility across runs due to floating-point non-associativity in parallel CUDA kernels (see Section 8.2). The minimum 3 runs per combination quantify this residual variance.

## 4. Dependent Variables

### 4.1 Quality Metrics

Six quality metrics measure faithfulness, groundedness, and retrieval-awareness. These are the primary dependent variables. Defined in `comprag/aggregator.py` as `QUALITY_METRICS`:

```python
QUALITY_METRICS = (
    "faithfulness",
    "context_utilization",
    "self_knowledge",
    "noise_sensitivity",
    "answer_relevancy",
    "negative_rejection_rate",
)
```

#### 4.1.1 faithfulness

- **Source framework:** RAGAS
- **Range:** [0, 1]
- **Definition:** The ratio of claims in the model's answer that are entailed by the retrieved context. RAGAS decomposes the answer into atomic claims using the judge LLM, then checks each claim against the context for entailment.
- **Interpretation:** Higher is better. A faithfulness of 1.0 means every claim in the answer is supported by the retrieved chunks. A model that hallucinates from parametric memory will score lower because those claims have no support in the context.
- **Thesis prediction:** Smaller models will score higher on faithfulness because they have fewer parametric claims to inject.

#### 4.1.2 context_utilization

- **Source framework:** RAGChecker
- **Range:** [0, 1]
- **Definition:** The fraction of retrieved context chunks that contributed information to the answer. Measured by RAGChecker via claim extraction: for each claim in the answer, RAGChecker traces it back to a source chunk. Chunks that were retrieved but never referenced receive zero utilization.
- **Interpretation:** Higher is better for RAG. A model that ignores retrieved context and answers from parametric memory will score low on context_utilization even if its answer is correct. This metric directly tests whether the model is actually using the retrieval pipeline.
- **Thesis prediction:** Smaller models will show higher context_utilization because they have less internal knowledge to substitute.

#### 4.1.3 self_knowledge

- **Source framework:** RAGChecker
- **Range:** [0, 1]
- **Definition:** The degree to which the model used parametric (training-time) memory instead of the provided context. RAGChecker computes this by identifying claims in the answer that do not trace back to any retrieved chunk -- these are assumed to originate from the model's parameters.
- **Interpretation:** Lower is better for RAG. A self_knowledge score of 0.0 means the model added no information beyond what was in the context. A score of 1.0 means the model ignored the context entirely and answered from memory. This is the most direct test of the thesis.
- **Thesis prediction:** Smaller models will score lower on self_knowledge.

#### 4.1.4 noise_sensitivity

- **Source framework:** RAGChecker
- **Range:** [0, 1]
- **Definition:** The performance degradation when irrelevant (noisy) chunks are included in the retrieved context. RAGChecker measures this by comparing answer quality with clean context vs. context contaminated with irrelevant documents. The RGB noise_robustness subset provides the test cases.
- **Interpretation:** Lower noise_sensitivity is better -- it means the model can filter out irrelevant information and still produce a correct answer. A model that is highly sensitive to noise will incorporate irrelevant claims into its answer.
- **Thesis prediction:** Ambiguous. Smaller models may be more susceptible to noise (less capacity to distinguish relevant from irrelevant), or they may be less susceptible (fewer parametric priors to conflict with noisy context).

#### 4.1.5 answer_relevancy

- **Source framework:** RAGAS
- **Range:** [0, 1]
- **Definition:** Whether the answer addresses the actual question asked. RAGAS computes this by generating synthetic questions from the answer and measuring their semantic similarity to the original query. If the answer is on-topic, the generated questions will be similar to the original.
- **Interpretation:** Higher is better. A model that produces technically faithful but off-topic answers will score low on answer_relevancy.
- **Thesis prediction:** No strong prediction. Answer relevancy is expected to be relatively stable across model sizes if the prompt template is effective.

#### 4.1.6 negative_rejection_rate

- **Source framework:** Composite (computed from RGB negative_rejection subset results)
- **Range:** [0, 1]
- **Definition:** The fraction of unanswerable questions that the model correctly refused to answer. A question is "unanswerable" when none of the retrieved chunks contain the answer. The prompt template instructs the model to say "I cannot answer this question based on the provided context" in such cases.
- **Interpretation:** Higher is better. A model with strong parametric knowledge may attempt to answer from memory instead of refusing, resulting in a lower negative_rejection_rate.
- **Thesis prediction:** Smaller models will show higher negative_rejection_rate because they have less parametric knowledge to fall back on.

### 4.2 Performance Metrics

Four performance metrics characterize inference cost. Defined in `comprag/aggregator.py` as `PERFORMANCE_METRICS`:

```python
PERFORMANCE_METRICS = (
    "tokens_per_second",
    "ttft_ms",
    "vram_usage_mb",
    "gpu_temp",
)
```

| Metric | Unit | Source | What It Measures |
|--------|------|--------|-----------------|
| `tokens_per_second` | tok/s | Runner `perf` block | Generation throughput (output tokens / wall clock time) |
| `ttft_ms` | milliseconds | Runner `perf` block | Time to first token (prompt processing + first decode step latency) |
| `vram_usage_mb` | MB | Runner `perf` block | Peak VRAM consumption during inference |
| `gpu_temp` | Celsius | Runner `perf` block | GPU temperature during inference (thermal throttle indicator) |

The aggregator handles aliased field names from the runner's v2 output schema via the `_PERF_ALIASES` dict in `comprag/aggregator.py`:

```python
_PERF_ALIASES: dict[str, str] = {
    "tokens_per_sec": "tokens_per_second",
    "time_to_first_token_ms": "ttft_ms",
    "vram_mb": "vram_usage_mb",
}
```

This ensures both v1 (flat field names) and v2 (nested `perf` block with shortened names) raw result formats contribute to the same aggregated metric.

## 5. Datasets

Three evaluation datasets are used, each targeting different aspects of RAG faithfulness. Dataset configuration is in `config/eval_config.yaml` under the `datasets` block.

### 5.1 RGB (Retrieval-Augmented Generation Benchmark)

**Source:** https://github.com/chen700564/RGB

RGB is the primary evaluation dataset with four subsets, each testing a distinct failure mode of RAG systems:

| Subset | What It Tests | Why It Matters for the Thesis |
|--------|--------------|------------------------------|
| `noise_robustness` | Can the model produce correct answers when irrelevant chunks are mixed with relevant ones? | Tests whether the model can selectively attend to relevant context |
| `negative_rejection` | Does the model correctly refuse to answer when no retrieved chunk contains the answer? | Directly tests parametric override -- a model with strong priors will attempt to answer from memory |
| `information_integration` | Can the model synthesize answers from information spread across multiple chunks? | Tests the model's ability to combine evidence from multiple retrieved passages |
| `counterfactual_robustness` | Does the model resist factually incorrect statements in retrieved context? | Tests whether the model blindly trusts context vs. cross-checking with internal knowledge |

RGB ships its own passages per question. These are indexed into per-subset ChromaDB collections (`comprag_rgb_noise_robustness_300w_<hash>`, etc.) so that retrieval tests the model's robustness to the provided passages, not retrieval quality. This is a critical design choice: RGB tests retrieval robustness, not retrieval quality. The retriever searches within per-sample passages, not a global corpus.

Normalization from RGB's native JSON format to CompRAG's unified JSONL schema is handled by `scripts/normalize_datasets.py`. Each entry maps `question` to `query`, `answer` to `ground_truth`, and preserves `passages` and `label` in the `metadata` field.

### 5.2 Natural Questions (NQ)

**Source:** HuggingFace `nq_open`

NQ provides factoid questions with human-labeled answers derived from Wikipedia. Unlike RGB, NQ does not provide passages -- retrieval runs against a global Wikipedia corpus indexed in `comprag_nq_wiki_300w_<hash>`. This tests end-to-end RAG accuracy including retrieval quality, making it complementary to RGB's controlled-passage design.

NQ entries include multiple acceptable answer strings. The normalization script (`scripts/normalize_datasets.py`) uses the first answer as `ground_truth` and preserves all answers in `metadata.all_answers`. The NQ corpus requires the 2018-12-20 Wikipedia dump that NQ was built against.

### 5.3 HaluEval

**Source:** https://github.com/RUCAIBox/HaluEval

HaluEval provides question-answer pairs with explicit hallucination labels. Each sample in the QA subset includes:

- `knowledge`: The ground-truth context document
- `question`: The query
- `answer`: The correct, non-hallucinated answer
- `hallucinated_answer`: A plausible but incorrect answer containing hallucinations

The `knowledge` field is indexed as the retrievable document in `comprag_halueval_300w_<hash>`. This dataset directly measures whether the model generates the faithful answer or drifts toward the hallucinated variant, providing binary ground truth for hallucination detection. The normalization script maps `question` to `query`, `answer` to `ground_truth`, and preserves `knowledge` and `hallucinated_answer` in `metadata`.

### 5.4 Normalization

All datasets are normalized to a unified JSONL schema before entering the pipeline (via `scripts/normalize_datasets.py`). One JSON object per line:

```json
{
  "sample_id": "rgb_noise_0042",
  "dataset": "rgb",
  "subset": "noise_robustness",
  "query": "What year was the Eiffel Tower completed?",
  "ground_truth": "The Eiffel Tower was completed in 1889.",
  "corpus_doc_ids": ["rgb_doc_0108", "rgb_doc_0109"],
  "metadata": {}
}
```

Field contracts:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `sample_id` | string | yes | Globally unique. Format: `{dataset}_{subset}_{index:04d}` |
| `dataset` | string | yes | One of: `rgb`, `nq`, `halueval` |
| `subset` | string | yes | Dataset-specific subset name |
| `query` | string | yes | The question to ask the model |
| `ground_truth` | string | yes | Reference answer for scoring |
| `corpus_doc_ids` | list[str] | no | Document IDs in the ChromaDB collection. Empty = search full index. |
| `metadata` | dict | no | Dataset-specific fields (RGB `label`, HaluEval `hallucination_label`, etc.) |

Normalized files are stored under `datasets/{dataset}/normalized/`.

### 5.5 Phase 2 Datasets (Future)

The following datasets are planned for phase 2 but not included in the initial evaluation:

- **MS MARCO:** For retrieval quality isolation (separating retrieval errors from generation errors)
- **RAGAS synthetic generation:** For domain-specific eval sets
- **CompRAG-Clinical:** Custom clinical lab informatics corpus for domain-specific RAG evaluation

## 6. Statistical Approach

Statistical methodology is implemented in `comprag/aggregator.py` and configured in `config/eval_config.yaml` under the `statistics` block.

### 6.1 Grouping

Results are grouped by the 5-tuple `(model, quantization, hardware_tier, dataset, eval_subset)`, defined as `GROUP_KEYS` in `comprag/aggregator.py`:

```python
GROUP_KEYS = ("model", "quantization", "hardware_tier", "dataset", "eval_subset")
```

The `_extract_group_key()` function handles both v1 (flat top-level fields) and v2 (nested `run_config` block) input schemas. V2 field mapping uses `_GROUP_KEY_MAP`:

```python
_GROUP_KEY_MAP: dict[str, str] = {
    "model": "model",
    "quant": "quantization",
    "hardware": "hardware_tier",
}
```

The function probes `run_config` for v2 keys first, then falls back to top-level fields for v1 compatibility. The `eval_subset` field is extracted from `subset` in v2 records.

### 6.2 Minimum Runs

Each combination requires a minimum of **3 runs** (`MIN_RUNS = 3` in `comprag/aggregator.py`, `statistics.min_runs: 3` in config). Groups with fewer runs emit a warning:

```
"Only {n} run(s) -- minimum 3 required for reliable CIs"
```

The 3-run minimum is a pragmatic floor for bootstrap resampling. With fewer than 3 data points, the BCa method may degenerate, and the resulting CIs lack statistical meaning. Groups with exactly 1 run produce degenerate CIs (low = high = mean, width = 0) and are flagged accordingly.

### 6.3 Bootstrap Resampling

Confidence intervals are computed via the `bootstrap_ci()` function in `comprag/aggregator.py`. Configuration from `config/eval_config.yaml`:

```yaml
statistics:
  min_runs: 3
  bootstrap_resamples: 1000
  confidence_level: 0.95
  ci_width_flag_threshold: 0.15
```

Implementation parameters:

| Parameter | Value | Source |
|-----------|-------|--------|
| Number of resamples | 1000 | `BOOTSTRAP_RESAMPLES = 1000` |
| Confidence level | 95% | `CI_LEVEL = 0.95` |
| Primary method | BCa (bias-corrected and accelerated) | `scipy.stats.bootstrap(method="BCa")` |
| Fallback method | Simple percentile bootstrap | Manual resampling with `numpy` |
| Random seed | 42 | `rng_seed=42` in `bootstrap_ci()` |
| RNG | `numpy.random.default_rng(42)` | Passed to `scipy.stats.bootstrap` as `random_state` |

The BCa (bias-corrected and accelerated) method is preferred over simple percentile bootstrap because it adjusts for both bias and skewness in the bootstrap distribution. This is particularly important for small sample sizes (n=3--10) where the bootstrap distribution may be asymmetric.

The implementation:

```python
result = scipy_stats.bootstrap(
    (values,),
    statistic=np.mean,
    n_resamples=n_resamples,
    confidence_level=confidence_level,
    random_state=rng,
    method="BCa",
)
ci_low = float(result.confidence_interval.low)
ci_high = float(result.confidence_interval.high)
```

If BCa fails (e.g., degenerate data, constant values, or numerical issues), the function falls back to simple percentile bootstrap:

```python
boot_means = np.empty(n_resamples)
for i in range(n_resamples):
    sample = rng.choice(values, size=n, replace=True)
    boot_means[i] = np.mean(sample)

alpha = 1.0 - confidence_level
ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
```

For groups with fewer than 2 data points, CI is degenerate (low = high = mean, width = 0) and no bootstrapping is performed.

### 6.4 CI Width Flagging

Any result where CI width exceeds 15% of the absolute mean is flagged for additional runs. From `comprag/aggregator.py`:

```python
CI_WIDTH_THRESHOLD = 0.15  # Flag if CI width > 15% of mean
```

The flagging logic in `bootstrap_ci()`:

```python
flagged = False
if abs(mean_val) > 1e-10:
    flagged = ci_width > CI_WIDTH_THRESHOLD * abs(mean_val)
```

The `1e-10` guard prevents division-by-zero when the mean is effectively zero. Flagged results appear in the aggregated output with `"flagged": true` and in the warnings section of Markdown summaries with the message:

```
"{metric}: CI width ({ci_width:.4f}) > 15% of mean ({mean:.4f}) -- needs more runs"
```

The 15% threshold is a pragmatic choice: tighter thresholds would require impractical run counts on slow hardware tiers (e.g., CPU at ~2-5 tok/s), while looser thresholds mask genuine variance in quality metrics.

### 6.5 Aggregation Output

The `aggregate_group()` function in `comprag/aggregator.py` computes bootstrap CIs for every metric in `ALL_METRICS = QUALITY_METRICS + PERFORMANCE_METRICS`. The output per group includes:

```python
{
    "n_runs": int,
    "warnings": list[str],
    "metrics": {
        "metric_name": {
            "mean": float,
            "ci_low": float,
            "ci_high": float,
            "ci_width": float,
            "n": int,
            "flagged": bool,
        }
    }
}
```

The `run_aggregation()` function writes three output formats:

- **CSV** (`results/aggregated/summary.csv`): Flat rows with `{metric}_mean`, `{metric}_ci_low`, `{metric}_ci_high`, `{metric}_flagged` columns
- **JSONL** (`results/aggregated/summary.jsonl`): One record per group with nested `metrics` (bare means) and `metrics_ci` (full CI detail) blocks
- **Markdown** (`results/aggregated/summary.md`): Human-readable table with quality and performance metrics in separate sections

### 6.6 Implementation Dependencies

All statistical computation uses `scipy.stats` and `numpy`, as specified in the project requirements. No custom bootstrap implementations are used for the primary BCa method. The percentile fallback exists only for robustness against edge cases where `scipy.stats.bootstrap` raises (e.g., constant data, single unique value).

## 7. Scoring Pipeline

The scoring pipeline is implemented in `comprag/evaluator.py` and uses a dual-framework, dual-server architecture.

### 7.1 Frameworks

Two evaluation frameworks score each result:

**RAGChecker (primary):** Computes `context_utilization` (CU), `self_knowledge` (SK), and `noise_sensitivity` (NS) -- the three metrics most directly testing the thesis. RAGChecker uses RefChecker internally for claim extraction and entailment checking. Configured via `eval_config.yaml` under `evaluation.ragchecker_metrics`:

```yaml
ragchecker_metrics:
  - context_utilization
  - self_knowledge
  - noise_sensitivity
```

**RAGAS (baseline):** Computes `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`. Provides industry-standard baseline scores for comparison with published benchmarks. Configured via `eval_config.yaml` under `evaluation.ragas_metrics`:

```yaml
ragas_metrics:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall
```

Both frameworks run with graceful degradation: if one fails, the other's results are still returned. Error messages are captured in `ragas_error` and `ragchecker_error` fields of the `EvalResult` dataclass in `comprag/evaluator.py`.

### 7.2 Judge Model

Both frameworks require an LLM judge for claim extraction and entailment. The judge model is **Qwen 2.5 14B Instruct**, configured in `config/eval_config.yaml` under the `judge` block:

```yaml
judge:
  model: "qwen2.5-14b-instruct"
  quant: "Q4_K_M"
  server_port: 8081
  context_length: 8192
  max_tokens: 1024
  temperature: 0.0
  self_judge_quant: "Q8_0"
```

Key design decisions:

- **Model choice:** Qwen 2.5 14B is the strongest model in the evaluation matrix that fits on the V100. The RAGChecker paper used Llama3-70B-Instruct, which requires 70B parameters -- far beyond what any CompRAG hardware tier can serve. This quality ceiling is acknowledged in Section 8.1.
- **Port separation:** The judge server runs on port 8081, separate from the generation server on port 8080.
- **Context length:** 8192 tokens. The judge processes individual claims (short) against retrieved chunks, not full documents. The full 32k context is unnecessary and would waste VRAM on KV cache.
- **Temperature:** 0.0 for deterministic scoring.
- **Fixed judge:** The judge model is fixed for ALL scoring runs. Changing the judge invalidates all prior scores. Results are only comparable when scored by the same judge.

### 7.3 Dual Server Architecture

The generation server (port 8080) and judge server (port 8081) cannot run simultaneously on the V100 due to VRAM constraints -- both are GPU-accelerated models that require significant VRAM. The scoring pipeline in `comprag/evaluator.py` enforces a strictly sequential workflow:

1. **Stop the generation server** on port 8080 (`stop_generation_server()`)
2. **Start the judge server** on port 8081 with the judge GGUF (`start_judge_server()`)
3. **Wait for `/health` 200** on port 8081 (up to 180s timeout for model loading)
4. **Initialize RAGChecker** with `extractor_name="openai/qwen2.5-14b-instruct"` and `checker_name="openai/qwen2.5-14b-instruct"`, both pointed at `http://localhost:8081/v1`
5. **Initialize RAGAS** via LangChain's `ChatOpenAI` pointed at the same judge server. This is the ONLY place LangChain enters the pipeline:
   ```python
   judge_llm = ChatOpenAI(
       base_url="http://localhost:8081/v1",
       api_key="not-needed",
       model="qwen2.5-14b-instruct",
       temperature=0.0,
       max_tokens=1024,
   )
   ```
6. **Convert raw JSONL** to RAGChecker input format via `to_ragchecker_format()`
7. **Score all samples**, collecting `EvalResult` per sample
8. **Stop the judge server** (`stop_judge_server()`)

### 7.4 RAGChecker Input Format Bridge

The runner's raw JSONL output must be converted to RAGChecker's expected format. The `to_ragchecker_format()` function in `comprag/evaluator.py` performs this mapping:

```python
def to_ragchecker_format(raw_results: list[dict]) -> dict:
    return {
        "results": [
            {
                "query_id": entry["sample_id"],
                "query": entry["query"],
                "gt_answer": entry["ground_truth"],
                "response": entry["response"],
                "retrieved_context": [
                    {"doc_id": chunk["doc_id"], "text": chunk["text"]}
                    for chunk in entry["retrieved_chunks"]
                ],
            }
            for entry in raw_results
        ]
    }
```

### 7.5 Self-Judge Quantization Swap

When the model being evaluated is Qwen 2.5 14B Instruct Q4_K_M (i.e., the same model and quantization as the judge), using the identical weights for both generation and scoring creates a circular evaluation. The `_resolve_judge_gguf()` function in `comprag/evaluator.py` detects this by comparing the eval model filename against the judge model name. When a match is detected, the judge quant is swapped to `Q8_0` (`judge.self_judge_quant` in config):

```python
if judge_stem in eval_stem:
    self_quant = judge_config.get("self_judge_quant")
    if self_quant:
        quant = self_quant
```

This ensures the judge always uses different weights than the model being scored, even when they share the same architecture. Q8_0 provides higher fidelity than Q4_K_M while still fitting on the V100 alongside KV cache. The quant swap is logged for traceability.

### 7.6 Raw Output Schema

The runner's raw JSONL output (one line per query) follows the v2 schema defined in `comprag/runner.py`. Each record includes:

```json
{
  "sample_id": "rgb_noise_0042",
  "dataset": "rgb",
  "subset": "noise_robustness",
  "query": "What year was the Eiffel Tower completed?",
  "ground_truth": "The Eiffel Tower was completed in 1889.",
  "response": "The Eiffel Tower was completed in 1889.",
  "retrieved_chunks": [
    {"doc_id": "rgb_doc_0108", "text": "...", "distance": 0.234, "rank": 1}
  ],
  "run_config": {
    "model": "llama-3.1-8b-instruct",
    "quant": "Q4_K_M",
    "hardware": "v100",
    "seed": 42
  },
  "perf": {
    "ttft_ms": 142,
    "total_tokens": 23,
    "tokens_per_sec": 48.2,
    "wall_clock_sec": 0.477,
    "vram_mb": 8432,
    "gpu_temp": 67
  },
  "timestamp": "2026-03-15T14:23:07Z",
  "error": null
}
```

On failure, `response` is `null` and `error` contains the error string (e.g., `"connection_refused: llama-server crashed (likely OOM mid-inference)"`). Failed queries are logged but excluded from quality metric aggregation. Performance metrics from failed queries may still be recorded if the failure occurred after the perf snapshot.

## 8. Threats to Validity

### 8.1 Judge Model Quality Ceiling

The judge model (Qwen 2.5 14B Q4_K_M) is the strongest model available on CompRAG hardware, but it is significantly weaker than the judges used in the original RAGChecker paper (Llama3-70B-Instruct). Seventy-billion-parameter models require 35+ GB in Q4 quantization -- far beyond what any single CompRAG hardware tier can serve. This creates a quality ceiling: the judge may fail to detect subtle faithfulness violations or incorrectly flag valid claims.

Mitigation: Results should be interpreted relative to each other within this benchmark, not compared directly to RAGChecker scores computed with stronger judges. The internal ranking of models (which model is most faithful) should be robust to judge quality as long as the judge's errors are not systematically correlated with specific generator models.

### 8.2 Temperature 0 Non-Determinism

Temperature 0.0 with seed 42 should produce deterministic outputs, but llama.cpp does not guarantee bitwise reproducibility across runs. Sources of non-determinism include:

- **Floating-point non-associativity** in parallel CUDA kernel execution
- **Batch size differences** affecting numerical accumulation order
- **CUDA kernel scheduling** variability
- **Memory allocation patterns** differing between runs

The minimum 3 runs per combination and bootstrap CIs (Section 6) quantify this residual variance. If CI widths are consistently narrow (<5% of mean), the non-determinism is negligible for the purposes of this study. If CI widths are wide, additional runs are triggered by the 15% flagging threshold.

### 8.3 Single Embedding Model

All retrieval uses `all-MiniLM-L6-v2` (384-dim). This model's retrieval quality sets a ceiling on the context quality available to the generator. If MiniLM-L6-v2 systematically retrieves poor context for certain query types, the generator will appear unfaithful regardless of its actual grounding capability.

Mitigation: RGB provides its own passages per question (bypassing the embedding model for 4 of 6 evaluation subsets). Only NQ relies on end-to-end retrieval against a global corpus. Phase 2 will add MS MARCO for retrieval quality isolation and potentially test additional embedding models.

### 8.4 Corpus Coverage Gaps

NQ requires a Wikipedia corpus for retrieval. The indexed corpus may not contain the exact passages needed for all NQ questions, especially if using a different Wikipedia dump than the 2018-12-20 version NQ was built against. Missing passages will cause retrieval failures that appear as model unfaithfulness.

Mitigation: HaluEval provides the ground-truth knowledge field as the retrievable document, guaranteeing perfect retrieval for that dataset. RGB provides per-question passages. Together, 5 of 6 evaluation subsets (4 RGB + HaluEval) have controlled retrieval, isolating corpus coverage issues to the NQ subset.

### 8.5 Hardware Thermal Throttling

Extended evaluation runs on consumer hardware (1660 Super, E5-2667v4) may trigger thermal throttling, reducing clock speeds mid-run. This affects performance metrics (tokens/sec, latency) and could introduce variance in quality metrics if throttling causes numerical differences in inference.

Mitigation: GPU temperature is logged as `gpu_temp` in the performance metrics (`comprag/aggregator.py`, `PERFORMANCE_METRICS`) to detect throttling post-hoc. Runs where `gpu_temp` exceeds the GPU's known throttle threshold (e.g., 83C for the 1660 Super) should be flagged during analysis. The V100 (primary eval tier) has datacenter-grade cooling and is unlikely to throttle.

### 8.6 Quantization as Confound

Quantization is both an independent variable and a potential confound. Lower quantization (Q4_K_M) reduces model capacity, which could independently affect faithfulness beyond the model-size effect. A Q4_K_M 14B model may behave like a smaller effective model, confounding the parameter-count analysis.

Mitigation: The study tests this explicitly by comparing the same model at different quantization levels (e.g., Qwen 2.5 14B at Q4_K_M vs Q8_0 vs FP16). If quantization significantly affects quality metrics, it must be controlled for when comparing across model sizes. The FP16 baselines on the V100 provide the unquantized reference point for each model.

### 8.7 Limited Judge Diversity

Using a single judge model means systematic biases in Qwen 2.5 14B's claim extraction or entailment checking will propagate to all scores. A judge that is lenient toward certain phrasing patterns will inflate faithfulness scores for models that produce similar phrasing. This is inherent to LLM-as-judge evaluation and cannot be fully mitigated without multiple independent judges (impractical given hardware constraints).

Mitigation: The self-judge quant swap (Section 7.5) prevents circular evaluation for the Qwen 2.5 14B eval runs. Cross-referencing RAGAS and RAGChecker scores provides a second opinion on each result, though both ultimately rely on the same judge LLM.

### 8.8 Prompt Template Sensitivity

Different model architectures may interpret the generic prompt template differently. Models trained with specific instruction formats (ChatML, Llama, Alpaca) may not optimally parse the `<|system|>`, `<|context|>`, `<|user|>`, `<|assistant|>` delimiters. The llama.cpp server applies model-specific chat templates, but the effectiveness of this conversion varies.

Mitigation: The prompt content is held constant (Section 3.2). Only the template wrapper changes per model, handled by llama.cpp. If a model consistently scores poorly across all metrics, prompt sensitivity should be investigated before attributing the results to model quality.

## Appendix A: Configuration File Reference

All configuration files are in the `config/` directory of the project root.

| File | Purpose | Key Contents |
|------|---------|-------------|
| `config/eval_config.yaml` | Master evaluation configuration | Dataset paths, retrieval params, generation params, judge config, statistical params, output paths |
| `config/models.yaml` | Model registry | 9 models with HF repos, GGUF filenames, quantization sizes, feasibility matrix |
| `config/hardware.yaml` | Hardware tier definitions | 6 tiers with specs, software stacks, status |
| `config/prompt_template.txt` | Locked prompt template | System/context/user/assistant blocks with placeholders |

## Appendix B: Code Module Reference

| Module | Purpose | Key Functions/Classes |
|--------|---------|----------------------|
| `comprag/runner.py` | Main eval loop | `EvalRunner`, `run_eval()`, signal handlers, context window safety check |
| `comprag/evaluator.py` | Scoring pipeline | `RAGASEvaluator`, `RAGCheckerEvaluator`, `CombinedEvaluator`, `to_ragchecker_format()` |
| `comprag/aggregator.py` | Statistical aggregation | `bootstrap_ci()`, `aggregate_all()`, `run_aggregation()`, CSV/JSONL/Markdown output |
| `comprag/retriever.py` | ChromaDB interface | `Retriever`, `validate_index()`, `resolve_collection_name()`, `make_collection_name()` |
| `comprag/generator.py` | llama.cpp server interface | `LlamaServer`, `format_prompt()`, `load_prompt_template()`, zombie process cleanup |
| `comprag/utils.py` | Shared utilities | `get_hardware_meta()`, `Timer`, `read_jsonl()`, `append_jsonl()`, `load_config()` |
| `scripts/build_index.py` | Index builder | Chunk documents, embed with sentence-transformers, persist to ChromaDB |
| `scripts/normalize_datasets.py` | Dataset normalization | Per-dataset transforms to unified JSONL schema |
| `scripts/download_datasets.py` | Dataset downloader | Fetch RGB, NQ, HaluEval from sources |
| `scripts/download_models.py` | Model downloader | Fetch GGUF files from HuggingFace |

## Appendix C: Reproducibility Checklist

To reproduce the results reported in this study:

1. **Environment:** Python 3.12, Ubuntu 24.04 (Dell T7820) or Linux Mint 22.3 (x99 staging)
2. **Dependencies:** `pip install -r requirements.txt` (includes exact version pins for scipy, numpy, chromadb, sentence-transformers, ragas, ragchecker, refchecker, spacy)
3. **Datasets:** Run `scripts/download_datasets.py` followed by `scripts/normalize_datasets.py`
4. **Models:** Download GGUF files per `config/models.yaml` via `scripts/download_models.py`
5. **Index:** Build vector index via `scripts/build_index.py --all`
6. **Evaluation:** Run the eval matrix via `comprag/runner.py` (see CLI arguments)
7. **Scoring:** Run the scoring pipeline via `comprag/evaluator.py`
8. **Aggregation:** Compute CIs via `comprag/aggregator.py --input results/raw/ --output results/aggregated/`

All random seeds are fixed at 42. The locked generation parameters (temp=0.0, top_p=1.0, max_tokens=512, seed=42) are enforced by the runner. The bootstrap CI computation uses a fixed RNG seed. Given identical hardware and software versions, results should be reproducible within the CI bounds documented in the aggregated output.
