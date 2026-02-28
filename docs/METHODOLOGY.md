# Methodology

## 1. Research Question & Thesis

**Research question:** Does model size inversely correlate with RAG faithfulness when retrieval quality is held constant?

**Thesis:** Smaller language models (1.7B--14B parameters) may produce more faithful retrieval-augmented generation outputs than larger models because they possess less parametric knowledge to contradict retrieved context. A model with weaker internal priors is more likely to defer to the retrieval pipeline, yielding higher context utilization, lower self-knowledge leakage, and stronger noise sensitivity scores. The hardware tier running the model affects throughput and latency but has minimal impact on groundedness metrics.

This study tests both claims by sweeping model size, quantization level, and hardware tier while holding the retrieval pipeline constant. The independent variables are the generator model, its quantization, and the hardware it runs on. The dependent variables are six quality metrics measuring faithfulness and groundedness, plus five performance metrics measuring throughput and resource consumption. If the thesis holds, smaller models will score higher on context_utilization and lower on self_knowledge across datasets, while hardware tier will show no statistically significant effect on quality metrics.

## 2. Independent Variables

### 2.1 Model

Nine models span the 1.7B--14B parameter range, selected to cover multiple architecture families and training lineages. The model registry is defined in `config/models.yaml`.

| Model | Parameters | Context Length | Priority |
|-------|-----------|----------------|----------|
| Qwen 2.5 14B Instruct | 14B | 32768 | HIGH |
| Phi-4 14B | 14B | 16384 | HIGH |
| Mistral NeMo 12B Instruct | 12B | 32768 | HIGH |
| Llama 3.1 8B Instruct | 8B | 131072 | HIGH (baseline) |
| Qwen 2.5 7B Instruct | 7B | 32768 | MEDIUM |
| Gemma 2 9B Instruct | 9B | 8192 | MEDIUM |
| GLM-4 9B Chat | 9B | 131072 | MEDIUM |
| SmolLM2 1.7B Instruct | 1.7B | 8192 | LOW (floor test) |
| BitNet 1.58b (TBD) | 3B | TBD | LOW (FPGA only) |

Llama 3.1 8B Instruct serves as the primary baseline due to its wide community adoption and extensive benchmark coverage. SmolLM2 1.7B provides the floor test — if a 1.7B model can produce faithful RAG outputs, it strengthens the thesis that retrieval compensates for limited parametric knowledge.

All models are served in GGUF format via llama.cpp's OpenAI-compatible server on port 8080 (`config/eval_config.yaml`, `generation.server_url`).

### 2.2 Quantization

Each model is tested at available quantization levels from `config/models.yaml`:

| Quantization | Bits | Typical Compression | Notes |
|-------------|------|--------------------|----|
| Q4_K_M | ~4.5 | ~4x vs FP16 | Primary eval quant, fits most tiers |
| Q8_0 | 8 | ~2x vs FP16 | Higher fidelity, fits V100 and MI25 |
| FP16 | 16 | 1x (baseline) | Only on V100 for 8B/14B models |

Not all models have all quantizations. The exact availability per model is encoded in `config/models.yaml` under each model's `quantizations` block. Quantization tests whether lossy weight compression degrades faithfulness — a model that trusts retrieved context should be less affected by weight quantization than one relying on parametric recall.

### 2.3 Hardware Tier

Six hardware tiers are defined in `config/hardware.yaml`, spanning GPU, CPU, Optane persistent memory, and FPGA:

| Tier ID | Hardware | Compute Type | VRAM/RAM | Status |
|---------|----------|-------------|----------|--------|
| `v100` | NVIDIA V100 SXM2 32GB (Dell T7820) | GPU | 32GB HBM2 | Incoming |
| `mi25` | AMD MI25 16GB | GPU | 16GB HBM | Available |
| `1660s` | NVIDIA 1660 Super 6GB | GPU | 6GB GDDR6 | Available |
| `fpga` | Inspur FPGA board + 16GB SODIMM | FPGA | 16GB DDR4 | Planned |
| `cpu` | Intel E5-2667v4 + 32GB DDR4 | CPU | 32GB DDR4 | Available |
| `optane` | E5-2667v4 + Optane DCPMM 384GB | CPU | 384GB Optane | Available |

Not every model-quantization pair runs on every tier. The feasibility matrix in `config/models.yaml` encodes which combinations are valid (e.g., 14B models cannot run on the 1660 Super's 6GB VRAM). The hardware tier is expected to affect only performance metrics (tokens/sec, latency), not quality metrics. If hardware tier does affect quality, it would indicate non-determinism in the inference backend or thermal throttling causing numerical drift.

## 3. Controlled Variables

The following variables are held constant across all evaluation runs to isolate the effect of model, quantization, and hardware.

### 3.1 Retrieval Pipeline

Defined in `config/eval_config.yaml` under the `retrieval` block and implemented in `cumrag/retriever.py`:

- **Vector store:** ChromaDB (local, file-based persistence in `index/`)
- **Embedding model:** `all-MiniLM-L6-v2` (384-dimensional, sentence-transformers). Single source of truth is `eval_config.yaml`; the retriever reads the model name from ChromaDB collection metadata at init, falling back to config if metadata is absent (`cumrag/retriever.py`, `_get_default_embedding_model()`).
- **Chunk size:** 300 whitespace words (~400--500 BPE tokens)
- **Chunk overlap:** 64 words
- **Top-k:** 5 chunks retrieved per query
- **No reranking** in phase 1

Index integrity is enforced at query time: `validate_index()` in `cumrag/retriever.py` raises `ValueError` if the collection's `embedding_model` or `chunk_size_words` metadata differs from `eval_config.yaml`. Collections use content-addressed names (e.g., `cumrag_rgb_noise_robustness_300w_a3f7c2d1`) computed by `make_collection_name()` to prevent stale index reuse.

### 3.2 Prompt Template

A single prompt template is used for all models, all hardware tiers, and all datasets. Stored in `config/prompt_template.txt`:

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

The `{retrieved_chunks}` and `{query}` placeholders are filled by the runner. The llama.cpp server handles chat template conversion per model (ChatML, Llama, etc.) — only the message content is controlled, not the template wrapper.

### 3.3 Generation Parameters

Locked in `config/eval_config.yaml` under the `generation` block:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Greedy decoding, minimizes stochastic variation |
| `top_p` | 1.0 | No nucleus sampling (redundant with temp 0, but explicit) |
| `max_tokens` | 512 | Sufficient for factoid answers without truncation risk |
| `seed` | 42 | Deterministic sampling where supported by backend |

These parameters are passed to the llama.cpp server via `generation.server_url` (`http://localhost:8080`) and `generation.endpoint` (`/v1/chat/completions`).

## 4. Dependent Variables

### 4.1 Quality Metrics

Six quality metrics measure faithfulness, groundedness, and retrieval-awareness. These are the primary dependent variables. Defined in `cumrag/aggregator.py` as `QUALITY_METRICS`:

| Metric | Source Framework | Range | What It Measures |
|--------|-----------------|-------|-----------------|
| `faithfulness` | RAGAS | [0, 1] | Ratio of claims in the answer that are supported by retrieved context |
| `context_utilization` | RAGChecker | [0, 1] | Fraction of retrieved context actually used in the answer |
| `self_knowledge` | RAGChecker | [0, 1] | Degree to which the model used parametric memory instead of context (lower is better for RAG) |
| `noise_sensitivity` | RAGChecker | [0, 1] | Performance degradation when irrelevant chunks are injected |
| `answer_relevancy` | RAGAS | [0, 1] | Whether the answer addresses the actual question |
| `negative_rejection_rate` | Composite | [0, 1] | Fraction of unanswerable questions correctly refused |

The thesis predicts that smaller models will show higher `context_utilization`, lower `self_knowledge`, and higher `faithfulness`. The `self_knowledge` metric is the most direct test: a model with strong parametric priors will score high on self_knowledge (bad for RAG), while a model that defers to context will score low (good for RAG).

### 4.2 Performance Metrics

Five performance metrics characterize inference cost. Defined in `cumrag/aggregator.py` as `PERFORMANCE_METRICS`:

| Metric | Unit | What It Measures |
|--------|------|-----------------|
| `tokens_per_second` | tok/s | Generation throughput |
| `ttft_ms` | milliseconds | Time to first token (latency) |
| `vram_usage_mb` | MB | Peak VRAM consumption |
| `gpu_temp` | Celsius | GPU temperature during inference |

The aggregator also handles aliased field names from the runner's v2 output schema (e.g., `tokens_per_sec` maps to `tokens_per_second`, `time_to_first_token_ms` maps to `ttft_ms`) via the `_PERF_ALIASES` dict in `cumrag/aggregator.py`.

## 5. Datasets

Three evaluation datasets are used, each targeting different aspects of RAG faithfulness. Dataset configuration is in `config/eval_config.yaml` under the `datasets` block.

### 5.1 RGB (Retrieval-Augmented Generation Benchmark)

**Source:** https://github.com/chen700564/RGB

RGB is the primary evaluation dataset with four subsets, each testing a distinct failure mode:

| Subset | What It Tests |
|--------|--------------|
| `noise_robustness` | Can the model produce correct answers when irrelevant chunks are mixed with relevant ones? |
| `negative_rejection` | Does the model correctly refuse to answer when no retrieved chunk contains the answer? |
| `information_integration` | Can the model synthesize answers from information spread across multiple chunks? |
| `counterfactual_robustness` | Does the model resist factually incorrect statements in retrieved context? |

RGB ships its own passages per question. These are indexed into per-subset ChromaDB collections (`cumrag_rgb_noise_robustness_300w_<hash>`, etc.) so that retrieval tests the model's robustness to the provided passages, not retrieval quality.

### 5.2 Natural Questions (NQ)

**Source:** HuggingFace `nq_open`

NQ provides factoid questions with human-labeled answers derived from Wikipedia. Unlike RGB, NQ does not provide passages — retrieval runs against a global Wikipedia corpus indexed in `cumrag_nq_wiki_300w_<hash>`. This tests end-to-end RAG accuracy including retrieval quality, making it complementary to RGB's controlled-passage design.

### 5.3 HaluEval

**Source:** https://github.com/RUCAIBox/HaluEval

HaluEval provides question-answer pairs with explicit hallucination labels. Each sample includes a `knowledge` field (the ground-truth context), a correct `answer`, and a `hallucinated_answer`. The `knowledge` field is indexed as the retrievable document in `cumrag_halueval_300w_<hash>`. This dataset directly measures whether the model generates the faithful answer or drifts toward the hallucinated variant, providing binary ground truth for hallucination detection.

### 5.4 Normalization

All datasets are normalized to a unified JSONL schema before entering the pipeline (via `scripts/normalize_datasets.py`). The schema includes `sample_id`, `dataset`, `subset`, `query`, `ground_truth`, `corpus_doc_ids`, and `metadata`. Normalized files are stored under `datasets/{dataset}/normalized/`.

## 6. Statistical Approach

Statistical methodology is implemented in `cumrag/aggregator.py` and configured in `config/eval_config.yaml` under the `statistics` block.

### 6.1 Grouping

Results are grouped by the 5-tuple `(model, quantization, hardware_tier, dataset, eval_subset)`, defined as `GROUP_KEYS` in `cumrag/aggregator.py`. The `_extract_group_key()` function handles both v1 (flat top-level fields) and v2 (nested `run_config` block) input schemas.

### 6.2 Minimum Runs

Each combination requires a minimum of **3 runs** (`MIN_RUNS = 3` in `cumrag/aggregator.py`, `statistics.min_runs: 3` in config). Groups with fewer runs emit a warning: `"Only {n} run(s) — minimum 3 required for reliable CIs"`.

### 6.3 Bootstrap Resampling

Confidence intervals are computed via the `bootstrap_ci()` function in `cumrag/aggregator.py`:

- **Number of resamples:** 1000 (`BOOTSTRAP_RESAMPLES = 1000`, `statistics.bootstrap_resamples: 1000`)
- **Confidence level:** 95% (`CI_LEVEL = 0.95`, `statistics.confidence_level: 0.95`)
- **Primary method:** BCa (bias-corrected and accelerated) via `scipy.stats.bootstrap()` with `method="BCa"`
- **Fallback:** Simple percentile bootstrap if BCa fails (e.g., degenerate data). The fallback computes `n_resamples` bootstrap means and takes the `alpha/2` and `1-alpha/2` percentiles.
- **Random seed:** Fixed at 42 for reproducibility (`rng_seed=42` in `bootstrap_ci()`)

For groups with fewer than 2 data points, CI is degenerate (low = high = mean, width = 0).

### 6.4 CI Width Flagging

Any result where CI width exceeds 15% of the absolute mean is flagged for additional runs (`CI_WIDTH_THRESHOLD = 0.15`, `statistics.ci_width_flag_threshold: 0.15`). The flagging logic in `bootstrap_ci()`:

```python
if abs(mean_val) > 1e-10:
    flagged = ci_width > CI_WIDTH_THRESHOLD * abs(mean_val)
```

Flagged results appear in the aggregated output with `"flagged": true` and in the warnings section of Markdown summaries. The 15% threshold is a pragmatic choice: tighter thresholds would require impractical run counts on slow hardware tiers, while looser thresholds mask genuine variance.

### 6.5 Implementation

All statistical computation uses `scipy.stats` and `numpy`, as specified in the project requirements. No custom bootstrap implementations are used for the primary BCa method. The percentile fallback exists only for robustness against edge cases where `scipy.stats.bootstrap` raises (e.g., constant data, single unique value).

## 7. Scoring Pipeline

The scoring pipeline is implemented in `cumrag/evaluator.py` and uses a dual-framework, dual-server architecture.

### 7.1 Frameworks

Two evaluation frameworks score each result:

- **RAGChecker (primary):** Computes `context_utilization` (CU), `self_knowledge` (SK), and `noise_sensitivity` (NS) — the three metrics most directly testing the thesis. RAGChecker uses RefChecker internally for claim extraction and entailment checking. Configured via `eval_config.yaml` under `evaluation.ragchecker_metrics`.
- **RAGAS (baseline):** Computes `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`. Provides industry-standard baseline scores for comparison with published benchmarks. Configured via `eval_config.yaml` under `evaluation.ragas_metrics`.

Both frameworks run with graceful degradation: if one fails, the other's results are still returned. Error messages are captured in `ragas_error` and `ragchecker_error` fields of the `EvalResult` dataclass.

### 7.2 Judge Model

Both frameworks require an LLM judge for claim extraction and entailment. The judge model is **Qwen 2.5 14B Instruct**, configured in `config/eval_config.yaml` under the `judge` block:

- **Model:** `qwen2.5-14b-instruct`
- **Default quantization:** `Q4_K_M`
- **Server port:** 8081 (separate from the generation server on 8080)
- **Context length:** 8192 (claims are short; the judge does not need the full 32k context)
- **Temperature:** 0.0
- **Max tokens:** 1024

The judge model is fixed for all scoring runs. Changing the judge invalidates all prior scores.

### 7.3 Dual Server Architecture

The generation server (port 8080) and judge server (port 8081) cannot run simultaneously on the V100 due to VRAM constraints. The scoring pipeline in `cumrag/evaluator.py` enforces a strictly sequential workflow:

1. **Stop the generation server** on port 8080 (`stop_generation_server()`)
2. **Start the judge server** on port 8081 (`start_judge_server()`)
3. **Wait for `/health` 200** on port 8081 (up to 180s timeout)
4. **Initialize RAGChecker** with `extractor_name="openai/qwen2.5-14b-instruct"` and `checker_name="openai/qwen2.5-14b-instruct"`, both pointed at `http://localhost:8081/v1`
5. **Initialize RAGAS** via LangChain's `ChatOpenAI` pointed at the same judge server — this is the only place LangChain enters the pipeline (`_get_judge_llm()`)
6. **Convert raw JSONL** to RAGChecker input format via `to_ragchecker_format()`
7. **Score all samples**, collecting `EvalResult` per sample
8. **Stop the judge server** (`stop_judge_server()`)

### 7.4 Self-Judge Quantization Swap

When the model being evaluated is Qwen 2.5 14B Instruct Q4_K_M (i.e., the same model as the judge), using the identical weights for both generation and scoring creates a circular evaluation. The `_resolve_judge_gguf()` function in `cumrag/evaluator.py` detects this by comparing the eval model filename against the judge model name. When a match is detected, the judge quant is swapped to `Q8_0` (`judge.self_judge_quant` in config):

```python
if judge_stem in eval_stem:
    self_quant = judge_config.get("self_judge_quant")
    if self_quant:
        quant = self_quant
```

This ensures the judge always uses different weights than the model being scored, even when they share the same architecture.

## 8. Threats to Validity

### 8.1 Judge Model Quality Ceiling

The judge model (Qwen 2.5 14B Q4_K_M) is the strongest model available on CUMRAG hardware, but it is weaker than the judges used in the original RAGChecker paper (Llama3-70B-Instruct). This creates a quality ceiling: the judge may fail to detect subtle faithfulness violations or incorrectly flag valid claims. Results should be interpreted relative to each other within this benchmark, not compared directly to RAGChecker scores computed with stronger judges. The self-judge quant swap (Section 7.4) mitigates but does not eliminate the circular evaluation risk for the Qwen 2.5 14B eval runs.

### 8.2 Temperature 0 Non-Determinism

Temperature 0.0 with seed 42 should produce deterministic outputs, but llama.cpp does not guarantee bitwise reproducibility across runs due to floating-point non-associativity in parallel computation, batch size differences, and CUDA kernel scheduling. The minimum 3 runs per combination and bootstrap CIs (Section 6) quantify this residual variance. If CI widths are consistently narrow (<5% of mean), the non-determinism is negligible for the purposes of this study.

### 8.3 Single Embedding Model

All retrieval uses `all-MiniLM-L6-v2` (384-dim). This model's retrieval quality sets a ceiling on the context quality available to the generator. If MiniLM-L6-v2 systematically retrieves poor context for certain query types, the generator will appear unfaithful regardless of its actual grounding capability. Phase 2 will add MS MARCO for retrieval quality isolation and potentially test additional embedding models.

### 8.4 Corpus Coverage Gaps

NQ requires a Wikipedia corpus for retrieval. The indexed corpus may not contain the exact passages needed for all NQ questions (especially if using a different Wikipedia dump than the 2018-12-20 version NQ was built against). Missing passages will cause retrieval failures that appear as model unfaithfulness. HaluEval mitigates this by providing the ground-truth knowledge field as the retrievable document.

### 8.5 Hardware Thermal Throttling

Extended evaluation runs on consumer hardware (1660 Super, E5-2667v4) may trigger thermal throttling, reducing clock speeds mid-run. This affects performance metrics (tokens/sec, latency) and could introduce variance in quality metrics if throttling causes numerical differences in inference. GPU temperature is logged as `gpu_temp` in the performance metrics (`cumrag/aggregator.py`, `PERFORMANCE_METRICS`) to detect throttling post-hoc. Runs where `gpu_temp` exceeds the GPU's throttle threshold should be flagged during analysis.

### 8.6 Quantization as Confound

Quantization is both an independent variable and a potential confound. Lower quantization (Q4_K_M) reduces model capacity, which could independently affect faithfulness beyond the model-size effect. The study tests this by comparing the same model at different quantization levels (e.g., Qwen 2.5 14B at Q4_K_M vs Q8_0 vs FP16). If quantization significantly affects quality metrics, it must be controlled for when comparing across model sizes.

### 8.7 Limited Judge Diversity

Using a single judge model means systematic biases in Qwen 2.5 14B's claim extraction or entailment checking will propagate to all scores. A judge that is lenient toward certain phrasing patterns will inflate faithfulness scores for models that produce similar phrasing. This is inherent to LLM-as-judge evaluation and cannot be fully mitigated without multiple independent judges (impractical given hardware constraints).
