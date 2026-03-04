# CUMRAG: Implementation Guide for Claude Code

## v2.0 — Combined Spec + Refactor Fixes + Gap Fills

---

## What This Is

A benchmark harness that evaluates RAG faithfulness and groundedness across multiple small language models, quantization levels, and hardware tiers — all running on surplus/consumer hardware. The goal is to produce a publishable comparison matrix showing how model size, quantization, and hardware affect RAG output quality, with special focus on faithfulness (does the model trust retrieved context?) vs parametric override (does it hallucinate from training data instead?).

### Thesis

Smaller, "dumber" models may produce more faithful RAG outputs because they have less parametric knowledge to contradict retrieved context. The hardware running the model matters less than the model's architecture for groundedness metrics. We prove this with rigorous benchmarks on eBay-grade hardware.

---

## Hardware Matrix

| Tier ID | Hardware | VRAM/RAM | Software Stack | Status |
|---------|----------|----------|---------------|--------|
| `v100` | NVIDIA V100 SXM2 32GB (in Dell T7820) | 32GB HBM2 | CUDA, llama.cpp | Incoming |
| `mi25` | AMD MI25 16GB | 16GB HBM | ROCm, llama.cpp | Available |
| `1660s` | NVIDIA 1660 Super 6GB | 6GB GDDR6 | CUDA, llama.cpp | Available |
| `fpga` | Inspur FPGA board + 16GB SODIMM | 16GB DDR4 | TBD (BitNet runtime) | Planned |
| `cpu` | Intel E5-2667v4 + 32GB DDR4 | 32GB DDR4 | llama.cpp CPU | Available |
| `optane` | E5-2667v4 + Optane DCPMM 384GB | 384GB Optane | llama.cpp CPU (weights on Optane) | Available |

---

## Model Matrix

Download all models in GGUF format from HuggingFace (bartowski or official quants).

| Model | Parameters | Quantizations to Test | Context Length | Priority |
|-------|-----------|----------------------|----------------|----------|
| Qwen 2.5 14B Instruct | 14B | Q4_K_M, Q8_0, FP16 | 32768 | HIGH |
| Phi-4 14B | 14B | Q4_K_M, Q8_0, FP16 | 16384 | HIGH |
| Mistral NeMo 12B Instruct | 12B | Q4_K_M, Q8_0 | 32768 | HIGH |
| Llama 3.1 8B Instruct | 8B | Q4_K_M, Q8_0, FP16 | 131072 | HIGH (baseline) |
| Qwen 2.5 7B Instruct | 7B | Q4_K_M, Q8_0 | 32768 | MEDIUM |
| Gemma 2 9B Instruct | 9B | Q4_K_M, Q8_0 | 8192 | MEDIUM |
| GLM-4 9B Chat | 9B | Q4_K_M | 131072 | MEDIUM |
| SmolLM2 1.7B Instruct | 1.7B | Q8_0, FP16 | 8192 | LOW (floor test) |
| BitNet 1.58b (TBD model) | 3B | 1.58-bit native | TBD | LOW (FPGA only) |

**Model-Hardware feasibility:**
- V100 32GB: All models, all quantizations including 14B FP16
- MI25 16GB: 7-8B Q4/Q8, 14B Q4 (tight)
- 1660 Super 6GB: 3B and under, 7B Q4 (very tight, may OOM)
- FPGA: BitNet only
- CPU/Optane: 7B Q4 primary, 14B Q4 if patient

Add `context_length` to each model entry in `config/models.yaml`. Default to 4096 if unspecified.

---

## Evaluation Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| RGB (Retrieval-Augmented Generation Benchmark) | Noise robustness, negative rejection, information integration, counterfactual robustness | https://github.com/chen700564/RGB |
| Natural Questions (NQ) | General RAG accuracy with human-labeled answers | Via HuggingFace: `nq_open` |
| HaluEval | Hallucination detection with human labels | https://github.com/RUCAIBox/HaluEval |

**Phase 2 (future):**
- MS MARCO for retrieval quality isolation
- RAGAS synthetic generation for domain-specific eval sets
- Custom clinical lab informatics corpus (CUMRAG-Clinical)

### Unified Internal Format

Every dataset must be normalized to this JSONL schema before entering the pipeline. One JSON object per line:

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
| `corpus_doc_ids` | list[str] | no | If dataset provides specific docs, list their IDs in the ChromaDB collection. If empty, retriever searches the full index. |
| `metadata` | dict | no | Dataset-specific fields (e.g., RGB's `label` for counterfactual, HaluEval's `hallucination_label`) |

### Per-Dataset Transforms

Create `scripts/normalize_datasets.py`. Run it once after `download_datasets.py`.

#### RGB (github.com/chen700564/RGB)

RGB ships as JSON files per subset. Each entry has `question`, `answer`, and a list of `passages`.

```python
def normalize_rgb(raw_entry: dict, subset: str, index: int) -> dict:
    return {
        "sample_id": f"rgb_{subset}_{index:04d}",
        "dataset": "rgb",
        "subset": subset,
        "query": raw_entry["question"],
        "ground_truth": raw_entry["answer"],
        "corpus_doc_ids": [],  # passages are injected into ChromaDB at index time
        "metadata": {
            "original_passages": raw_entry.get("passages", []),
            "label": raw_entry.get("label", None),  # counterfactual subset has these
        },
    }
```

**RGB-specific indexing note:** RGB provides its own passages per question. These MUST be indexed into ChromaDB with doc_ids that link back to the sample. The retriever then retrieves from these per-sample passages, NOT from a global corpus. This is critical — RGB tests retrieval robustness, not retrieval quality. Handle this by creating a separate ChromaDB collection per RGB subset, or by filtering on doc metadata at query time.

#### Natural Questions (nq_open via HuggingFace)

NQ provides `question` and `answer` (a list of acceptable answer strings). No passages — retrieval is against a Wikipedia corpus.

```python
def normalize_nq(raw_entry: dict, index: int) -> dict:
    answers = raw_entry["answer"]
    return {
        "sample_id": f"nq_test_{index:04d}",
        "dataset": "nq",
        "subset": "test",
        "query": raw_entry["question"],
        "ground_truth": answers[0],  # primary answer
        "corpus_doc_ids": [],  # retrieves from global Wikipedia index
        "metadata": {
            "all_answers": answers,
        },
    }
```

**NQ corpus note:** NQ requires a Wikipedia corpus for retrieval. Use the 2018-12-20 Wikipedia dump that NQ was built against. Download via `datasets` library or use a pre-chunked version. This is a ~6GB download. Index it into ChromaDB collection `comprag_nq_wiki`.

#### HaluEval (github.com/RUCAIBox/HaluEval)

HaluEval QA subset has `knowledge`, `question`, `answer`, `hallucinated_answer`, and a binary label.

```python
def normalize_halueval(raw_entry: dict, index: int) -> dict:
    return {
        "sample_id": f"halueval_qa_{index:04d}",
        "dataset": "halueval",
        "subset": "qa",
        "query": raw_entry["question"],
        "ground_truth": raw_entry["answer"],  # the NON-hallucinated answer
        "corpus_doc_ids": [],
        "metadata": {
            "knowledge": raw_entry["knowledge"],
            "hallucinated_answer": raw_entry["hallucinated_answer"],
        },
    }
```

**HaluEval indexing note:** HaluEval provides a `knowledge` field per sample which is the context that should be retrievable. Index these as documents in collection `comprag_halueval`.

### Normalized Output Layout

```
datasets/
├── rgb/
│   ├── raw/                          # original downloaded files
│   └── normalized/
│       ├── noise_robustness.jsonl
│       ├── negative_rejection.jsonl
│       ├── information_integration.jsonl
│       └── counterfactual_robustness.jsonl
├── nq/
│   ├── raw/
│   └── normalized/
│       └── test.jsonl
└── halueval/
    ├── raw/
    └── normalized/
        └── qa.jsonl
```

---

## Evaluation Frameworks

Install both. RAGAS for quick baseline scores, RAGCHECKER for the deep diagnostics.

### RAGAS (baseline)
- pip install ragas
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Docs: https://docs.ragas.io

### RAGCHECKER (primary)
- pip install ragchecker
- python -m spacy download en_core_web_sm
- GitHub: https://github.com/amazon-science/RAGChecker
- Metrics: Context Utilization (CU), Self-Knowledge (SK), Noise Sensitivity (NS)
- These are the core metrics that test our thesis
- Docs: Read the repo README before implementing

---

## Judge Model Architecture

RAGChecker uses RefChecker under the hood, which requires an LLM for claim extraction and entailment checking. RAGAS similarly requires an LLM judge. **This judge model must be separate from the model being tested.**

### Judge Model Selection

Use **Qwen 2.5 14B Instruct Q4_K_M** as the fixed judge model for all scoring runs.

- The RAGChecker paper used Llama3-70B-Instruct. 70B won't fit on any CUMRAG hardware tier. Qwen 2.5 14B is the strongest model in the matrix that fits on the V100.
- The judge model is **fixed for ALL scoring runs**. Never change it mid-benchmark. If you change it, re-score everything.
- When scoring Qwen 2.5 14B Q4_K_M outputs, use Qwen 2.5 14B **Q8_0** as the judge (different quant = different model behavior). Document this in results.

### Dual llama.cpp Server Architecture

RAGChecker/RefChecker accepts any OpenAI-compatible API endpoint via `extractor_name` and `checker_name` parameters with the `openai/` prefix and `api_base` override. llama.cpp's server exposes `/v1/chat/completions` which is OpenAI-compatible. Run a second llama.cpp instance as the judge on a different port.

### Config in `eval_config.yaml`

```yaml
judge:
  model: "qwen2.5-14b-instruct"
  quant: "Q4_K_M"
  server_port: 8081          # DIFFERENT from generation server (8080)
  context_length: 8192       # Judge doesn't need full 32k — claims are short
  max_tokens: 1024
  temperature: 0.0
  # When judge model == eval model, override:
  self_judge_quant: "Q8_0"   # Use different quant to avoid circular eval
```

### Scoring Pipeline Operational Sequence

The scoring pipeline (`evaluator.py`) must:

1. **Stop the generation server** (port 8080) if still running — you don't need it during scoring.
2. **Start the judge server** on port 8081 with the judge GGUF.
3. **Wait for `/health` 200** on port 8081.
4. Initialize RAGChecker:
   ```python
   from ragchecker import RAGResults, RAGChecker
   from ragchecker.metrics import all_metrics

   evaluator = RAGChecker(
       extractor_name="openai/qwen2.5-14b-instruct",
       checker_name="openai/qwen2.5-14b-instruct",
       extractor_api_base="http://localhost:8081/v1",
       checker_api_base="http://localhost:8081/v1",
       batch_size_extractor=8,    # tune based on VRAM headroom
       batch_size_checker=8,
   )
   ```
5. Load raw JSONL, convert to RAGChecker input format, score, write scored JSONL.
6. **Stop the judge server** after scoring completes.

### RAGChecker Input Format Bridge

Convert CUMRAG raw JSONL to RAGChecker's expected format in `evaluator.py`:

```python
def to_ragchecker_format(raw_results: list[dict]) -> dict:
    """Convert CUMRAG raw JSONL entries to RAGChecker checking_inputs format."""
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

### RAGAS Judge Wiring

RAGAS uses LangChain's LLM abstraction. Point it at the same judge server:

```python
from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",              # llama.cpp doesn't validate
    model="qwen2.5-14b-instruct",
    temperature=0.0,
    max_tokens=1024,
)
```

This is the ONLY place LangChain enters the pipeline.

### Gotcha: Port Conflicts

Both the generation server (8080) and judge server (8081) cannot run simultaneously on the V100 if both models are GPU-accelerated — you'll OOM. The pipeline is strictly sequential: generate first, score later.

---

## Retrieval Pipeline

Keep retrieval constant across all tests. The retrieval component is NOT the variable — the generator model is.

### Embedding Model
- Use `all-MiniLM-L6-v2` from sentence-transformers (small, fast, runs on CPU)
- Same embedding model for all tests, no exceptions
- **Single source of truth:** `eval_config.yaml` under `retrieval.embedding_model`. Do NOT hardcode in `build_index.py` or `retriever.py` — read from config.
- When writing ChromaDB collection metadata, include `{"embedding_model": model_name}`.
- On retriever init, first try to read embedding model from the ChromaDB collection's metadata (written at index time). Fall back to `eval_config.yaml` only if collection metadata is missing. Log a warning on mismatch.

### Vector Store
- ChromaDB (local, file-based, zero config)
- Pre-compute and persist the vector index once
- **Per-dataset collections** — RGB, NQ, and HaluEval have different corpus sources (see Index Versioning below)

### Chunking
- Fixed chunk size: **300 whitespace words** (~400-500 BPE tokens)
- Overlap: 64 words
- Comment in config: `# Whitespace words, not BPE tokens. 300 words ≈ 400-500 BPE tokens.`
- Document this in results — chunking strategy affects everything

### Retrieval Parameters
- Top-k: 5 chunks retrieved per query
- No reranking in phase 1 (add as variable in phase 2)

### Context Window Safety Check

In `runner.py`, before generation:
- Estimate prompt length: `estimated_tokens = len(formatted_prompt.split()) * 1.4`
- If `estimated_tokens > 0.9 * model_context_length`, log a warning: `"Estimated prompt ({est} tokens) may exceed model context ({ctx} tokens). Consider reducing top_k."`
- Do NOT auto-reduce top_k — just warn. The researcher decides.
- `model_context_length` is read from `models.yaml` (see Model Matrix above), default 4096 if unspecified.

---

## ChromaDB Index Versioning

### Content-Addressed Collection Names

Encode the indexing parameters into the collection name to prevent stale index reuse:

```python
import hashlib

def make_collection_name(
    dataset: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Generate a deterministic, versioned collection name."""
    params = f"{dataset}|{embedding_model}|{chunk_size}|{chunk_overlap}"
    param_hash = hashlib.sha256(params.encode()).hexdigest()[:8]
    return f"comprag_{dataset}_{chunk_size}w_{param_hash}"
    # Example: "comprag_rgb_noise_robustness_300w_a3f7c2d1"
```

### Collection Metadata

Store the full indexing config in ChromaDB collection-level metadata:

```python
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunk_size_words": 300,
        "chunk_overlap_words": 64,
        "dataset": "rgb",
        "subset": "noise_robustness",
        "created_at": datetime.utcnow().isoformat(),
        "comprag_version": "2.0",
    },
)
```

### Validation at Query Time

The retriever must validate metadata matches `eval_config.yaml` before running queries:

```python
def validate_index(collection, config: dict) -> None:
    """Raise if the index was built with different parameters than current config."""
    meta = collection.metadata
    expected_model = config["retrieval"]["embedding_model"]
    expected_chunk = config["retrieval"]["chunk_size"]

    if meta.get("embedding_model") != expected_model:
        raise ValueError(
            f"Index embedding model mismatch: {meta.get('embedding_model')} "
            f"vs config {expected_model}. Rebuild index."
        )
    if meta.get("chunk_size_words") != expected_chunk:
        raise ValueError(
            f"Index chunk size mismatch: {meta.get('chunk_size_words')} "
            f"vs config {expected_chunk}. Rebuild index."
        )
```

Add this check to the pre-flight checks in the runner.

### Per-Dataset Collections

```
comprag_rgb_noise_robustness_300w_<hash>
comprag_rgb_negative_rejection_300w_<hash>
comprag_rgb_information_integration_300w_<hash>
comprag_rgb_counterfactual_robustness_300w_<hash>
comprag_nq_wiki_300w_<hash>
comprag_halueval_300w_<hash>
```

Config in `eval_config.yaml`:

```yaml
retrieval:
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dim: 384
  chunk_size: 300              # Whitespace words, not BPE tokens. 300 words ≈ 400-500 BPE tokens.
  chunk_overlap: 64
  top_k: 5
  collections:
    rgb_noise_robustness: "auto"      # "auto" = computed from params
    rgb_negative_rejection: "auto"
    rgb_information_integration: "auto"
    rgb_counterfactual_robustness: "auto"
    nq_wiki: "auto"
    halueval: "auto"
  index_dir: "index/"                  # ChromaDB persist directory
```

`"auto"` means `build_index.py` computes the collection name from parameters. The runner resolves `"auto"` at startup.

---

## Prompt Template

**CRITICAL: This template must be identical across ALL models and ALL hardware tiers. Do not adjust per-model.**

```
<|system|>
You are a helpful assistant. Answer the user's question using ONLY the provided context. If the context does not contain enough information to answer the question, say "I cannot answer this question based on the provided context." Do not use any knowledge from your training data.

<|context|>
{retrieved_chunks}

<|user|>
{query}

<|assistant|>
```

**Generation parameters (locked across all runs):**
- Temperature: 0.0
- Top-p: 1.0
- Max tokens: 512
- Seed: 42 (where supported)
- Stop sequences: model-appropriate

**Note:** Some models use different chat templates (ChatML, Llama, etc.). The llama.cpp server handles template conversion. The CONTENT of the system/context/user message must remain identical. Only the template wrapper changes per model.

---

## Metrics to Record

For every (model × quantization × hardware) combination, record:

### Quality Metrics (from RAGCHECKER/RAGAS)
- `faithfulness`: Ratio of claims supported by retrieved context
- `context_utilization`: % of retrieved context actually used in answer
- `self_knowledge`: Degree model used parametric memory instead of context
- `noise_sensitivity`: Performance degradation with irrelevant chunks
- `answer_relevancy`: Did the answer address the actual question
- `negative_rejection_rate`: % of unanswerable questions correctly refused

### Performance Metrics
- `tokens_per_second`: Generation throughput
- `time_to_first_token`: Latency in ms
- `total_inference_time`: Wall clock per query in ms
- `vram_usage_mb`: Peak VRAM consumption
- `ram_usage_mb`: Peak system RAM consumption

### Hardware Metadata
- `hardware_tier`: Tier ID from hardware matrix
- `gpu_model`: Exact GPU string
- `driver_version`: CUDA/ROCm version
- `inference_framework`: llama.cpp version/commit hash
- `quantization`: Exact quant format string
- `model_file`: Exact GGUF filename with hash

---

## Output Format: Runner → Evaluator Contract

The raw JSONL output is the **only** interface between generation and scoring. The evaluator never needs to know about llama.cpp, ChromaDB internals, or hardware specifics — it just reads these JSONL files.

### Raw Output JSONL Schema

Each line in `results/raw/{run_id}.jsonl`:

```json
{
  "sample_id": "rgb_noise_0042",
  "dataset": "rgb",
  "subset": "noise_robustness",
  "query": "What year was the Eiffel Tower completed?",
  "ground_truth": "The Eiffel Tower was completed in 1889.",
  "response": "The Eiffel Tower was completed in 1889 after approximately two years of construction.",
  "retrieved_chunks": [
    {
      "doc_id": "rgb_doc_0108",
      "text": "The Eiffel Tower was completed on March 31, 1889...",
      "distance": 0.234,
      "rank": 1
    }
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

On failure:

```json
{
  "sample_id": "rgb_noise_0042",
  "response": null,
  "error": "connection_refused: llama-server crashed (likely OOM mid-inference)"
}
```

### Aggregated Output Format

Results stored as JSONL after scoring (one JSON object per eval run):

```json
{
  "timestamp": "2026-03-01T14:30:00Z",
  "model": "Qwen2.5-14B-Instruct",
  "quantization": "Q4_K_M",
  "hardware_tier": "v100",
  "dataset": "RGB",
  "eval_subset": "noise_robustness",
  "noise_ratio": 0.4,
  "metrics": {
    "faithfulness": 0.82,
    "context_utilization": 0.71,
    "self_knowledge": 0.15,
    "answer_relevancy": 0.88,
    "tokens_per_second": 45.2,
    "time_to_first_token_ms": 120,
    "vram_usage_mb": 9200
  },
  "hardware_meta": {
    "gpu": "NVIDIA Tesla V100-SXM2-32GB",
    "driver": "CUDA 12.4",
    "framework": "llama.cpp b4567",
    "os": "Ubuntu 24.04"
  }
}
```

---

## Statistical Requirements

- Run each (model × quant × hardware × dataset_subset) combination **minimum 3 times** with different random seeds
- Compute **mean and 95% confidence interval** via bootstrap resampling (1000 resamples)
- Report confidence intervals alongside all point estimates
- Any result with CI width > 15% of the mean gets flagged for additional runs
- Use `scipy.stats` or `numpy` for bootstrap — no custom implementations

---

## Zombie llama-server Process Prevention

If the Python runner crashes hard (kernel OOM kill, segfault), `llama-server` keeps running as an orphan, holding the port. All subsequent runs fail until manual cleanup.

### Required Changes in `generator.py`

1. Add `_check_port_available()` method to `LlamaServer`:
   - Before starting a new server, check if the target port is already in use.
   - If occupied, identify the PID bound to that port (`lsof -ti:{port}` or parsing `/proc/net/tcp`).
   - If the process is a `llama-server`, kill it and log a warning: `"Killed orphaned llama-server (PID {pid}) on port {port}"`.
   - If it's something else, raise an error — don't kill random processes.
   - Call this method at the top of `start()`.

2. Add an `atexit` handler in `LlamaServer.__init__`:
   ```python
   import atexit
   atexit.register(self._cleanup)
   ```
   Where `_cleanup()` calls `self.stop()` if the server is still running.

3. In `runner.py`, register signal handlers at the top of the main eval loop:
   ```python
   import signal
   signal.signal(signal.SIGTERM, _shutdown_handler)
   signal.signal(signal.SIGINT, _shutdown_handler)
   ```
   Where `_shutdown_handler` stops the server and exits cleanly. Note: `SIGSEGV` handler won't reliably work — the port check on startup is the real safety net for hard crashes.

**Do NOT change:** The `preexec_fn=os.setsid` in `_start_server` — it's correct for process group management. The `try/except` crash recovery in `runner.py` must be preserved.

---

## Filesystem Layout

```
comprag/
├── README.md
├── requirements.txt
├── setup.sh                          # One-shot environment setup
│
├── config/
│   ├── models.yaml                   # Model registry (names, HF paths, quant formats, context_length)
│   ├── hardware.yaml                 # Hardware tier definitions
│   ├── eval_config.yaml              # Dataset paths, retrieval params, generation params, judge config
│   └── prompt_template.txt           # The sacred prompt template
│
├── scripts/
│   ├── download_datasets.py          # Fetch RGB, NQ, HaluEval
│   ├── normalize_datasets.py         # Per-dataset transforms → normalized JSONL
│   ├── download_models.py            # Fetch GGUF files from HuggingFace
│   ├── build_index.py                # Chunk documents, embed, persist to ChromaDB (per-dataset collections)
│   └── profile_hardware.py           # Log GPU/CPU/RAM specs to hardware.yaml
│
├── comprag/
│   ├── __init__.py
│   ├── retriever.py                  # ChromaDB query interface + index validation
│   ├── generator.py                  # llama.cpp server interface (start/stop/query) + zombie prevention
│   ├── evaluator.py                  # RAGAS + RAGCHECKER scoring wrapper + judge server lifecycle
│   ├── runner.py                     # Main eval loop: retrieve → generate → log
│   ├── aggregator.py                 # Read JSONL results, bootstrap CIs, output tables
│   └── utils.py                      # Hardware logger, timing utils, JSONL I/O
│
├── datasets/
│   ├── rgb/
│   │   ├── raw/
│   │   └── normalized/
│   ├── nq/
│   │   ├── raw/
│   │   └── normalized/
│   └── halueval/
│       ├── raw/
│       └── normalized/
│
├── models/                           # GGUF model files (gitignored, large)
│   ├── qwen2.5-14b-instruct-q4_k_m.gguf
│   ├── qwen2.5-14b-instruct-q8_0.gguf
│   └── ...
│
├── index/                            # Persisted ChromaDB vector store (per-dataset collections)
│
├── results/
│   ├── raw/                          # JSONL files per run (runner output)
│   ├── scored/                       # JSONL files per run (evaluator output)
│   ├── aggregated/                   # Summary CSVs with CIs
│   └── figures/                      # Generated charts
│
└── docs/
    ├── METHODOLOGY.md                # Statistical methodology writeup
    ├── HARDWARE.md                   # Hardware descriptions, photos, costs, sourcing
    └── RESULTS.md                    # Auto-generated results summary
```

---

## Setup Instructions

### Step 1: System dependencies
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git cmake build-essential
```

### Step 2: Create project
```bash
mkdir -p ~/comprag && cd ~/comprag
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Python dependencies
```bash
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn
pip install chromadb sentence-transformers
pip install ragas langchain-openai
pip install ragchecker>=0.1.9 refchecker spacy
python -m spacy download en_core_web_sm
pip install huggingface_hub requests pyyaml tqdm
```

### Step 4: Build llama.cpp
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF  # CPU-only on x99, rebuild per-hardware later
cmake --build build --config Release -j$(nproc)
```

### Step 5: Download and normalize datasets
```bash
cd ~/comprag
python scripts/download_datasets.py
python scripts/normalize_datasets.py
```

### Step 6: Download models (start with smallest for testing)
```bash
python scripts/download_models.py --model llama-3.1-8b-instruct --quant Q4_K_M
```

### Step 7: Build vector index
```bash
python scripts/build_index.py --all
```

### Step 8: Smoke test — generation
```bash
python -m comprag.runner \
  --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
  --dataset rgb \
  --subset noise_robustness \
  --hardware-tier cpu \
  --num-queries 10 \
  --output results/raw/smoke_test.jsonl
```

### Step 9: Smoke test — scoring
```bash
python -m comprag.evaluator --input results/raw/smoke_test.jsonl
```

---

## What Claude Code Should Build

### Phase 1: Scaffold (now, on x99 box)
1. Create the directory structure (see filesystem layout)
2. Write dataset download scripts (RGB, NQ, HaluEval)
3. Write dataset normalization script (per-dataset transforms → unified JSONL)
4. Write model download script (HuggingFace GGUF downloads)
5. Set up ChromaDB vector store and indexing pipeline (content-addressed collections, per-dataset, metadata)
6. Build the eval harness runner script that:
   - Loads a specified model via llama.cpp server
   - Validates index metadata matches config (pre-flight)
   - Checks context window safety before generation
   - Runs queries from normalized JSONL
   - Records all metrics to raw JSONL (full schema per Runner → Evaluator Contract)
   - Handles the prompt template injection
   - Handles zombie server cleanup on start and on exit
7. Build the scoring pipeline that:
   - Starts the judge server (port 8081)
   - Converts raw JSONL to RAGChecker input format
   - Feeds outputs to RAGAS/RAGCHECKER
   - Handles judge model self-eval quant swap
8. Build the results aggregation script (reads JSONL, computes bootstrap CIs, outputs summary tables)
9. Write the standardized hardware profiling logger

### Phase 2: Run (when hardware arrives)
1. Run eval matrix on each hardware tier
2. Generate comparison tables and charts
3. Build README with results

### No ARES or LUMINA integration (Phase 2)
Phase 2 will add ARES (synthetic data + fine-tuned classifiers + PPI) and LUMINA (information processing rate, MMD-based context utilization). These require their own research paper deep-dives. Upload the relevant paper when implementing each module.

---

## Notes for Claude Code

- Do NOT use LangChain for the core pipeline. Use it only as a thin wrapper for RAGAS judge wiring. The retrieval and generation pipeline should be direct API calls to ChromaDB and llama.cpp server.
- The llama.cpp generation server exposes an OpenAI-compatible API at localhost:8080. The judge server runs on localhost:8081. Never run both GPU-accelerated simultaneously.
- All file I/O uses JSONL (one JSON per line), not monolithic JSON files.
- The harness must be hardware-agnostic. Hardware-specific config (CUDA vs ROCm vs CPU) is handled at llama.cpp build time and in hardware.yaml, not in the Python code.
- Fail loudly. If a model OOMs, log the failure with the exact error and move to the next combination. Don't crash the entire run.
- Every script should be runnable standalone AND importable as a module.
- Embedding model config: single source of truth is `eval_config.yaml`. Never hardcode.
- The `WhitespaceTokenizer` class is fine for chunking. The `hashlib.md5` idempotency logic and `overlap` parameter should be preserved as-is.
- `EMBEDDING_DIM = 384` can stay hardcoded — it's determined by the model.
