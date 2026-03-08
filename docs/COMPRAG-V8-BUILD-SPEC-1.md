# CompRAG v8 — Build Spec

## For Claude Code

The ground truth is `comprag-thesis-v8.md`. If this spec conflicts with the thesis, the thesis wins. If you're unsure, ask — don't invent.

**v8 change:** RAGChecker schema expanded to full 11-metric suite with NS-I/NS-II split. Judge model upgraded to Claude Opus 4.6. Local Command R fallback removed; replaced with frontier-vs-frontier judge agreement validation (Opus vs GPT-5.4 vs Gemini 3 Flash). Frontier models updated to current versions. Kimi K2.5 replaced with GLM-5. Reasoning mode constraint dropped — frontier models use provider defaults.

---

## Architecture Overview

CompRAG is a benchmark harness. It runs queries through a RAG pipeline under controlled conditions, scores the outputs, and computes statistics. That's it.

```
datasets (RGB, NQ, HaluEval)
    ↓
retriever (ChromaDB + MiniLM embeddings, held constant)
    ↓
generator (llama.cpp server for local, API clients for frontier)
    ↓
scorer (RAGChecker primary, RAGAS baseline)
    ↓
aggregator (bootstrap CIs, Preference_Gap)
    ↓
visualizer (matplotlib curves)
```

Every step reads from and writes to JSONL files. No in-memory pipeline coupling. Each step is independently re-runnable.

---

## Output Schema (write tests against this FIRST)

Every query produces one JSONL record. This is the central data contract.

```jsonc
{
  // Identity
  "run_id": "uuid4",
  "timestamp": "ISO8601",
  
  // Configuration
  "model": "qwen2.5-14b-instruct",
  "quantization": "Q4_K_M",          // or "API" for frontier models
  "source": "local",                  // "local" | "api"
  "provider": null,                   // null for local, "openai"|"anthropic"|"google"|"deepseek"|"zhipu" for api
  "dataset": "rgb",
  "subset": "counterfactual",         // dataset-specific subset
  "pass": "pass2_loose",              // "pass1_baseline" | "pass2_loose" | "pass3_strict"
  "seed": 42,
  "query_id": "rgb_cf_042",           // stable query identifier
  
  // Inputs
  "query": "the actual question text",
  "context_chunks": ["chunk1", "chunk2", ...],  // null for pass1
  "ground_truth": "expected answer",
  
  // Outputs
  "response": "model's generated answer",
  "generation_time_ms": 1234,
  
  // Scores (null until scoring step)
  "scores": {
    "ragchecker": {
      // Overall (3)
      "overall_precision": 0.78,
      "overall_recall": 0.65,
      "overall_f1": 0.71,
      // Retriever (2) — constant across generators, recorded for per-query stratification
      "claim_recall": 0.82,
      "context_precision": 0.60,
      // Generator (6) — primary analysis variables
      "context_utilization": 0.85,
      "self_knowledge": 0.12,
      "noise_sensitivity_relevant": 0.08,   // NS-I: incorrect claims in relevant chunks
      "noise_sensitivity_irrelevant": 0.02, // NS-II: incorrect claims in irrelevant chunks
      "hallucination": 0.05,
      "faithfulness": 0.93
    },
    "ragas": {
      "faithfulness": 0.90,
      "answer_relevancy": 0.88,
      "context_precision": 0.75,
      "context_recall": 0.82
    }
  }
}
```

Aggregated output (per model × quant × dataset × subset × pass):

```jsonc
{
  "model": "qwen2.5-14b-instruct",
  "quantization": "Q4_K_M",
  "source": "local",
  "dataset": "rgb",
  "subset": "counterfactual",
  "pass": "pass2_loose",
  "n_queries": 487,
  "metrics": {
    // Primary analysis (generator metrics)
    "cu": {"mean": 0.72, "ci_lo": 0.68, "ci_hi": 0.76, "std": 0.15},
    "sk": {"mean": 0.25, "ci_lo": 0.21, "ci_hi": 0.29, "std": 0.12},
    "ns_relevant": {"mean": 0.08, "ci_lo": 0.05, "ci_hi": 0.11, "std": 0.06},
    "ns_irrelevant": {"mean": 0.03, "ci_lo": 0.01, "ci_hi": 0.05, "std": 0.03},
    "hallucination": {"mean": 0.05, "ci_lo": 0.03, "ci_hi": 0.08, "std": 0.04},
    "faithfulness": {"mean": 0.93, "ci_lo": 0.90, "ci_hi": 0.96, "std": 0.05},
    // Overall metrics
    "overall_precision": {"mean": 0.78, "ci_lo": 0.74, "ci_hi": 0.82, "std": 0.10},
    "overall_recall": {"mean": 0.65, "ci_lo": 0.60, "ci_hi": 0.70, "std": 0.12},
    "overall_f1": {"mean": 0.71, "ci_lo": 0.66, "ci_hi": 0.75, "std": 0.11},
    // Retriever metrics (constant across generators, recorded for validation)
    "claim_recall": {"mean": 0.82, "ci_lo": 0.78, "ci_hi": 0.86, "std": 0.08},
    "context_precision": {"mean": 0.60, "ci_lo": 0.55, "ci_hi": 0.65, "std": 0.09},
    // Derived
    "preference_gap": {"mean": 0.13, "ci_lo": 0.09, "ci_hi": 0.17, "std": 0.10}
  },
  "capability_degraded": false   // true if pass3 CU below threshold
}
```

---

## Directory Structure

```
comprag/
├── comprag/
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   ├── server.py             # llama.cpp process manager
│   ├── retrieve.py           # ChromaDB retrieval
│   ├── generate.py           # query execution (local + frontier)
│   ├── score.py              # RAGChecker + RAGAS scoring
│   ├── aggregate.py          # bootstrap stats + Preference_Gap
│   └── visualize.py          # matplotlib figures
├── config/
│   ├── models.yaml           # local model registry
│   ├── frontier.yaml         # API model registry
│   ├── prompts.yaml          # all 3 pass templates in one file
│   └── eval.yaml             # retrieval params, scoring config, stats params
├── scripts/
│   ├── download_datasets.py
│   ├── build_index.py
│   ├── download_models.py
│   ├── determinism_pilot.py
│   ├── judge_agreement.py
│   └── generate_preregistration.py
├── tests/
│   ├── test_schema.py        # golden-file tests against output schema
│   ├── test_aggregate.py     # Preference_Gap, bootstrap CI
│   └── test_generate.py      # prompt construction, message formatting
├── requirements.txt
├── setup.sh
└── README.md
```

No `runner.py` mega-module. Each file does one thing.

---

## Module Specs

### `comprag/server.py` — llama.cpp process manager

~150 lines. Three responsibilities:

1. Start a llama.cpp server subprocess with locked params
2. Poll `/health` until ready (with timeout)
3. Kill it cleanly

```python
class LlamaCppServer:
    def __init__(self, model_path: str, port: int = 8080):
        self.proc = None
    
    def start(self, ctx_len: int = 4096) -> None:
        """Start llama-server with full GPU offload, temp=0, greedy."""
        # Args: --model, --port, --n-gpu-layers -1, --ctx-size, --seed 42
        # NO variable params. Everything locked.
    
    def wait_ready(self, timeout: float = 180.0) -> None:
        """Poll /health until 200 or timeout."""
    
    def stop(self) -> None:
        """SIGTERM, wait 5s, SIGKILL if needed."""
    
    def __enter__(self) / __exit__(self):
        """Context manager for clean lifecycle."""
```

That's it. No VRAM simulation, no tier routing, no restart logic beyond basic failure.

### `comprag/retrieve.py` — ChromaDB retrieval

~100 lines. Wraps ChromaDB with locked params.

```python
class Retriever:
    def __init__(self, index_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        # Load sentence-transformers model, connect to ChromaDB
    
    def query(self, text: str, collection: str, top_k: int = 5) -> list[str]:
        """Return top-k chunks as plain strings."""
```

Index building is a separate script (`scripts/build_index.py`), not part of the runtime.

### `comprag/generate.py` — query execution

~300 lines. Two code paths: local (llama.cpp HTTP) and frontier (API clients).

```python
def build_messages(query: str, context: list[str] | None, pass_name: str) -> list[dict]:
    """Construct messages array for chat completions API.
    
    pass1_baseline: [{"role": "user", "content": query}]
    pass2_loose:    [{"role": "user", "content": f"I have some background...\n{context}\n\nBased on that, {query}"}]
    pass3_strict:   [{"role": "system", "content": "Answer using ONLY..."}, 
                     {"role": "user", "content": f"Context:\n{context}\n\n{query}"}]
    
    NO manual chat template tokens. The server or API handles wrapping.
    """

def generate_local(messages: list[dict], server_url: str = "http://localhost:8080") -> tuple[str, int]:
    """POST to /v1/chat/completions. Returns (response_text, time_ms).
    Locked: temperature=0.0, max_tokens=512, seed=42."""

def generate_frontier(messages: list[dict], provider: str, model_id: str, seed: int = 42) -> tuple[str, int]:
    """Call frontier API. Returns (response_text, time_ms).
    
    Provider routing:
    - openai/deepseek/zhipu: openai SDK with base_url override
    - anthropic: anthropic SDK (different message format)
    - google: google-genai SDK or OpenAI-compat endpoint
    """
```

**Critical: Anthropic handling.** The Anthropic API is NOT OpenAI-compatible. It uses content blocks, doesn't support `seed`, and has different system message handling. Use the `anthropic` Python SDK. Document that Anthropic runs have no seed-based reproducibility — variance is captured in bootstrap CIs across the 3 seed runs (where "seed" just means "run index" for Anthropic).

**Critical: prompt construction.** `build_messages` returns a standard `messages` array. For local, this goes directly to llama.cpp's `/v1/chat/completions` which applies the model's chat template. For frontier APIs, same array adapted per SDK. The prompt *content* is identical across all backends — only the transport differs.

### `comprag/score.py` — evaluation

~200 lines. Thin wrapper around RAGChecker and RAGAS.

```python
def score_ragchecker(query: str, response: str, context: list[str], ground_truth: str,
                     judge_provider: str = "anthropic",
                     judge_model: str = "claude-opus-4-6") -> dict:
    """Returns {"overall_precision": float, "overall_recall": float, "overall_f1": float,
               "claim_recall": float, "context_precision": float,
               "context_utilization": float, "self_knowledge": float,
               "noise_sensitivity_relevant": float, "noise_sensitivity_irrelevant": float,
               "hallucination": float, "faithfulness": float}
    judge_provider/judge_model: selects which frontier API scores the claims.
    Primary: anthropic/claude-opus-4-6. Agreement check uses openai/gpt-5.4 and google/gemini-3-flash-preview."""

def score_ragas(query: str, response: str, context: list[str], ground_truth: str,
                judge_provider: str = "anthropic",
                judge_model: str = "claude-opus-4-6") -> dict:
    """Returns {"faithfulness": float, "answer_relevancy": float, 
                "context_precision": float, "context_recall": float}"""
```

Judge model: frontier API only (Claude Opus 4.6, primary). The `score.py` module accepts `judge_provider` and `judge_model` parameters to select which frontier API performs claim entailment. This enables the judge agreement script to score the same queries with multiple judges without code changes.

Pin RAGChecker version in requirements.txt. Record exact commit hash if installed from git.

### `comprag/aggregate.py` — statistics

~200 lines. Reads scored JSONL, groups, bootstraps.

```python
def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, 
                 confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper)."""

def compute_preference_gap(pass2_records: list, pass3_records: list) -> dict:
    """Per-query: pass3_cu - pass2_cu. Then bootstrap over queries.
    Returns {"mean": ..., "ci_lo": ..., "ci_hi": ..., "std": ...}"""

def aggregate_results(scored_dir: str) -> list[dict]:
    """Group by (model, quant, dataset, subset, pass). 
    Bootstrap all RAGChecker metrics (CU, SK, NS-I, NS-II, Hallucination, 
    Faithfulness, Overall P/R/F1, Claim Recall, Context Precision) 
    and Preference_Gap.
    Flag capability-degraded configs (pass3_cu below threshold).
    Write aggregated JSONL."""
```

Group keys: `(model, quantization, dataset, subset, pass)`. No hardware tier dimension.

### `comprag/visualize.py` — figures

~300 lines. Reads aggregated JSONL, produces matplotlib figures.

**Required figures:**

1. **CU vs Quantization** — X: quant level (Q3→FP16), Y: CU. One curve per model. Frontier models as horizontal dashed lines. Error bars from bootstrap CIs. Capability-degraded points get hollow markers.

2. **SK vs Quantization** — Same layout as above.

3. **Preference_Gap vs Quantization** — X: quant level, Y: Pass3_CU - Pass2_CU. One curve per model.

4. **Cross-architecture comparison** — Paired panels: Qwen 7B vs Llama 8B, Qwen 14B vs Phi-4 14B. Same CU curves, side by side.

5. **RGB Negative Rejection** — Separate panel. CU on negative rejection subset only (where high CU = bad). This is the gullibility check.

6. **SmolLM2 floor test** — Small standalone panel. Does sub-2B produce any CU/SK signal at all?

X-axis ordering for quant levels: Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16 (increasing precision left to right).

Frontier reference lines: one per API model, labeled, dashed, with shaded CI band.

### `comprag/__main__.py` — CLI

Subcommands:

```
comprag retrieve   --dataset rgb --subset counterfactual
comprag generate   --model qwen2.5-14b-instruct --quant Q4_K_M --dataset rgb --pass pass2_loose
comprag generate   --frontier --provider openai --model gpt-5.4 --dataset rgb --pass pass2_loose --seed 42
comprag score      --input results/raw/some_run.jsonl
comprag aggregate  --input-dir results/scored/ --output-dir results/aggregated/
comprag visualize  --input-dir results/aggregated/ --output-dir results/figures/
```

Each subcommand is a thin wrapper around the corresponding module. No god-object orchestrator.

---

## Config Files

### `config/models.yaml`

```yaml
models:
  qwen2.5-14b-instruct:
    display_name: "Qwen 2.5 14B Instruct"
    params: 14B
    role: primary
    hf_repo: "bartowski/Qwen2.5-14B-Instruct-GGUF"
    quants:
      Q3_K_M: "Qwen2.5-14B-Instruct-Q3_K_M.gguf"
      Q4_K_M: "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
      Q5_K_M: "Qwen2.5-14B-Instruct-Q5_K_M.gguf"
      Q6_K:   "Qwen2.5-14B-Instruct-Q6_K.gguf"
      Q8_0:   "Qwen2.5-14B-Instruct-Q8_0.gguf"
      FP16:   "qwen2.5-14b-instruct-fp16.gguf"  # from Qwen/Qwen2.5-14B-Instruct-GGUF

  phi-4-14b:
    display_name: "Phi-4 14B"
    params: 14B
    role: primary
    hf_repo: "bartowski/phi-4-GGUF"
    quants:
      Q3_K_M: "phi-4-Q3_K_M.gguf"
      Q4_K_M: "phi-4-Q4_K_M.gguf"
      Q5_K_M: "phi-4-Q5_K_M.gguf"
      Q6_K:   "phi-4-Q6_K.gguf"
      Q8_0:   "phi-4-Q8_0.gguf"
      FP16:   "phi-4-f16.gguf"

  llama-3.1-8b-instruct:
    display_name: "Llama 3.1 8B Instruct"
    params: 8B
    role: primary
    hf_repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    quants:
      Q3_K_M: "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf"
      Q4_K_M: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
      Q5_K_M: "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
      Q6_K:   "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
      Q8_0:   "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
      FP16:   "Meta-Llama-3.1-8B-Instruct-f16.gguf"

  qwen2.5-7b-instruct:
    display_name: "Qwen 2.5 7B Instruct"
    params: 7B
    role: primary
    hf_repo: "bartowski/Qwen2.5-7B-Instruct-GGUF"
    quants:
      Q3_K_M: "Qwen2.5-7B-Instruct-Q3_K_M.gguf"
      Q4_K_M: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
      Q5_K_M: "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
      Q6_K:   "Qwen2.5-7B-Instruct-Q6_K.gguf"
      Q8_0:   "Qwen2.5-7B-Instruct-Q8_0.gguf"
      FP16:   "Qwen2.5-7B-Instruct-f16.gguf"

  mistral-nemo-12b-instruct:
    display_name: "Mistral NeMo 12B Instruct"
    params: 12B
    role: secondary
    hf_repo: "bartowski/Mistral-Nemo-Instruct-2407-GGUF"
    quants:
      Q4_K_M: "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
      Q8_0:   "Mistral-Nemo-Instruct-2407-Q8_0.gguf"

  gemma-2-9b-instruct:
    display_name: "Gemma 2 9B Instruct"
    params: 9B
    role: secondary
    hf_repo: "bartowski/gemma-2-9b-it-GGUF"
    quants:
      Q4_K_M: "gemma-2-9b-it-Q4_K_M.gguf"
      Q8_0:   "gemma-2-9b-it-Q8_0.gguf"

  smollm2-1.7b-instruct:
    display_name: "SmolLM2 1.7B Instruct"
    params: 1.7B
    role: floor
    hf_repo: "bartowski/SmolLM2-1.7B-Instruct-GGUF"
    quants:
      Q8_0: "SmolLM2-1.7B-Instruct-Q8_0.gguf"
      FP16: "SmolLM2-1.7B-Instruct-f16.gguf"
```

### `config/frontier.yaml`

```yaml
frontier_models:
  gpt-5.4:
    provider: openai
    model_id: "gpt-5.4"
    display_name: "GPT-5.4"
    base_url: null  # default openai endpoint
  
  claude-opus-4-6:
    provider: anthropic
    model_id: "claude-opus-4-6"
    display_name: "Claude Opus 4.6"
    base_url: null  # uses anthropic SDK
  
  gemini-3-flash:
    provider: google
    model_id: "gemini-3-flash-preview"
    display_name: "Gemini 3 Flash"
    base_url: null
  
  deepseek-v3.2:
    provider: deepseek
    model_id: "deepseek-chat"
    display_name: "DeepSeek V3.2"
    base_url: "https://api.deepseek.com/v1"
  
  glm-5:
    provider: zhipu
    model_id: "glm-5"
    display_name: "GLM-5"
    base_url: "https://open.bigmodel.cn/api/paas/v4"

passes: [pass2_loose, pass3_strict]
seeds: [42, 43, 44]
```

### `config/prompts.yaml`

```yaml
pass1_baseline:
  # No system message. Just the query as a user message.
  system: null
  user: "{query}"
  include_context: false

pass2_loose:
  # Neutral framing. Context embedded in user message.
  system: null
  user: |
    I have some background information that might be helpful:
    {context}

    Based on that, {query}
  include_context: true

pass3_strict:
  # Explicit instruction via system message. Context in user message.
  system: "Answer using ONLY the provided context. If the answer is not in the context, say \"I do not know.\""
  user: |
    Context:
    {context}

    {query}
  include_context: true
```

### `config/eval.yaml`

```yaml
retrieval:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size_words: 300
  overlap_words: 64
  top_k: 5
  index_dir: "index/"

generation:
  # ALL LOCKED. No per-model overrides.
  temperature: 0.0
  max_tokens: 512
  seed: 42
  local_server_port: 8080

judge:
  primary:
    provider: "anthropic"
    model_id: "claude-opus-4-6"
    temperature: 0.0
    max_tokens: 1024
  agreement_judges:
    - provider: "openai"
      model_id: "gpt-5.4"
    - provider: "google"
      model_id: "gemini-3-flash-preview"

scoring:
  primary: ragchecker
  baseline: ragas

statistics:
  bootstrap_resamples: 1000
  confidence_level: 0.95
  capability_degradation_threshold: 0.30  # pass3 CU below this = degraded

datasets:
  rgb:
    source: "https://github.com/chen700564/RGB"
    subsets: [noise_robustness, negative_rejection, counterfactual]
    pass_policy:
      counterfactual: [pass1_baseline, pass2_loose, pass3_strict]
      default: [pass2_loose]
  nq:
    source: "huggingface:nq_open"
    subsets: [default]
    pass_policy:
      default: [pass2_loose]
  halueval:
    source: "https://github.com/RUCAIBox/HaluEval"
    subsets: [default]
    pass_policy:
      default: [pass2_loose]

judge_validation:
  sample_size: 100
  primary_arch_sample: 50
  secondary_arch_sample: 50
  agreement_threshold_kappa: 0.80
  judges:  # pairwise κ computed for all pairs
    - {provider: "anthropic", model_id: "claude-opus-4-6"}
    - {provider: "openai", model_id: "gpt-5.4"}
    - {provider: "google", model_id: "gemini-3-flash-preview"}
```

---

## Scripts

### `scripts/determinism_pilot.py`

Run 10 seeds × 50 queries on Qwen 14B Q4_K_M. Compare outputs.

If bit-identical: write `determinism_pilot_result.json` with `{"deterministic": true, "model": "...", "n_seeds": 10, "n_queries": 50}`. All subsequent local runs use seed=42 only, bootstrap over queries.

If not: write variance stats, keep 3-seed runs for local.

### `scripts/judge_agreement.py`

Score 100 outputs (50 from primary-architecture models, 50 from secondary-architecture models) with three frontier judges: Claude Opus 4.6, GPT-5.4, and Gemini 3 Flash. Compute pairwise Cohen's kappa for all three pairs, overall and per stratum. Output `judge_agreement.json`. If a cheaper judge hits κ ≥ 0.80 against Opus, document it as a validated cost-efficient alternative. Report divergence patterns regardless.

### `scripts/generate_preregistration.py`

Collect: GGUF SHA256s, RAGChecker version, RGB download date, hypothesis text, predictions, statistical criteria. Write `PRE_REGISTRATION.md`. This gets committed before any experimental run.

---

## Build Order for Claude Code

Give Claude Code these as sequential tasks. Each task should produce working, testable code before moving to the next.

### Task 1: Schema + Tests

"Create `tests/test_schema.py` with golden-file validation for the raw JSONL schema and aggregated output schema defined in the spec. Create `tests/test_aggregate.py` with unit tests for `bootstrap_ci` and `compute_preference_gap`. These tests define the contract — all subsequent code must pass them. Also create all config files from the spec."

### Task 2: Server + Retrieve

"Create `comprag/server.py` (llama.cpp process manager) and `comprag/retrieve.py` (ChromaDB wrapper). Keep them minimal — the specs define the full interface. Create `scripts/build_index.py` and `scripts/download_datasets.py`."

### Task 3: Generate

"Create `comprag/generate.py` with `build_messages`, `generate_local`, and `generate_frontier`. The `build_messages` function constructs messages arrays from `config/prompts.yaml` — no manual chat template tokens. Frontier provider routing: openai SDK for OpenAI/DeepSeek/Zhipu, anthropic SDK for Anthropic, google-genai for Google. Create `tests/test_generate.py` to verify message construction for all 3 passes."

### Task 4: Score

"Create `comprag/score.py` wrapping RAGChecker and RAGAS. Thin wrappers — these libraries do the heavy lifting. Primary judge is frontier API (Claude Opus 4.6 via anthropic SDK). `judge_provider` and `judge_model` parameters allow switching to other frontier APIs for the judge agreement check."

### Task 5: Aggregate

"Create `comprag/aggregate.py` with bootstrap CIs and Preference_Gap computation. Must pass the tests from Task 1."

### Task 6: Visualize

"Create `comprag/visualize.py` producing the 6 figure types defined in the spec. X-axis: quant level (Q3→FP16). Frontier models as horizontal reference lines. Capability-degraded points get hollow markers."

### Task 7: CLI + Glue

"Create `comprag/__main__.py` with subcommands: retrieve, generate, score, aggregate, visualize. Each subcommand is a thin wrapper. Create `scripts/determinism_pilot.py`, `scripts/judge_agreement.py`, `scripts/generate_preregistration.py`. Create `requirements.txt`, `setup.sh`, `README.md`."

### Task 8: Review

"Run all tests. Read every module end to end. Check that: no hardware tier references exist, no logit gap references exist, no manual chat template tokens exist in prompts, Preference_Gap is computed correctly, frontier models only run passes 2 and 3, the JSONL schema matches the spec exactly."

---

## Constraints for Claude Code

Paste these as system-level instructions:

1. **No file over 300 lines.** If it's getting long, you're over-abstracting. Split or simplify.
2. **No classes unless managing state** (server process, DB connection). Everything else is functions.
3. **No retry/backoff decorators, no custom exception hierarchies, no abstract base classes.** Use try/except at call sites.
4. **JSONL is the interface between steps.** Read JSONL in, write JSONL out. No intermediate pickle, no database, no shared state.
5. **Locked parameters are constants, not config.** `temperature=0.0`, `max_tokens=512`, `seed=42`, `n_gpu_layers=-1`, `top_k=5`. Don't thread them through function signatures.
6. **No hardware tier dimension.** There is one GPU. Everything runs on it fully offloaded.
7. **Type hints on all function signatures.** No `Any` unless truly necessary.
8. **Logging: `logging.getLogger(__name__)` with INFO default.** No custom logging framework.
9. **If you're writing a function longer than 40 lines, stop and reconsider.**
10. **The thesis is the spec. Don't add features it doesn't mention.**
