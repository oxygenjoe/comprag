# CompRAG

Benchmark harness measuring how GGUF quantization levels affect RAG faithfulness metrics across local LLMs. Runs queries through a controlled RAG pipeline, scores outputs with RAGChecker, and computes bootstrap confidence intervals.

## Quick Start

```bash
# Setup
chmod +x setup.sh
./setup.sh
source .venv/bin/activate

# Download datasets and build retrieval index
python scripts/download_datasets.py --dataset rgb
python scripts/build_index.py

# Download model GGUFs
python scripts/download_models.py --model qwen2.5-14b-instruct --quant Q4_K_M
```

## CLI Usage

CompRAG uses subcommands. All commands read/write JSONL files -- each step is independently re-runnable.

```bash
# Retrieve context chunks for a dataset
python -m comprag retrieve --dataset rgb --subset counterfactual

# Generate with local llama.cpp model
python -m comprag generate --model qwen2.5-14b-instruct --quant Q4_K_M \
    --dataset rgb --pass pass2_loose

# Score results with RAGChecker
python -m comprag score --input results/raw/some_run.jsonl

# Score with frontier judge (Sonnet 4.6 validation sample)
python -m comprag score --input results/raw/some_run.jsonl --judge-mode frontier

# Aggregate scored results (bootstrap CIs, Preference_Gap)
python -m comprag aggregate --input-dir results/scored/ --output-dir results/aggregated/

# Generate figures
python -m comprag visualize --input-dir results/aggregated/ --output-dir results/figures/
```

## Directory Structure

```
comprag/
  __init__.py          # Package init
  __main__.py          # CLI entry point with subcommands
  server.py            # llama.cpp process manager
  retrieve.py          # ChromaDB retrieval wrapper
  generate.py          # Local query execution
  score.py             # RAGChecker scoring
  aggregate.py         # Bootstrap stats, Preference_Gap
  visualize.py         # Matplotlib figures (6 plot types)
config/
  models.yaml          # Local model registry (GGUF repos, quant variants)
  judge.yaml           # Judge model config (Command R primary, Sonnet 4.6 validation)
  prompts.yaml         # Prompt templates for pass1/pass2/pass3
  eval.yaml            # Retrieval params, scoring config, stats params
scripts/
  download_datasets.py       # Download RGB
  build_index.py             # Build ChromaDB index from normalized JSONL
  download_models.py         # Download GGUF files from HuggingFace
  determinism_pilot.py       # Verify greedy decoding is seed-invariant
  judge_agreement.py         # Cohen's kappa: Command R vs Sonnet 4.6
  generate_preregistration.py  # Lock experimental design before runs
tests/
  test_schema.py       # Golden-file tests for JSONL output schema
  test_aggregate.py    # Bootstrap CI + Preference_Gap unit tests
  test_generate.py     # Prompt construction tests
datasets/              # Downloaded benchmark data (RGB)
index/                 # ChromaDB persistent storage
models/                # Downloaded GGUF model files
results/
  raw/                 # Generation output JSONL
  scored/              # Scored JSONL (RAGChecker)
  aggregated/          # Aggregated stats JSONL
  figures/             # Matplotlib output PNGs
```

## Config Files

- **models.yaml** -- Defines local models with HuggingFace repos and available quantization levels (Q3_K_M through FP16).
- **judge.yaml** -- Judge model config: Command R 35B Q4_K_M (primary, local) and Claude Sonnet 4.6 (validation, API).
- **prompts.yaml** -- Three prompt templates: pass1_baseline (no context), pass2_loose (context in user message), pass3_strict (system instruction to use only context).
- **eval.yaml** -- Retrieval parameters (embedding model, chunk size, top-k), generation locks (temp=0, max_tokens=512), judge validation config, bootstrap stats params, dataset definitions.

## Environment Variables

```bash
export ANTHROPIC_API_KEY="..."  # For Sonnet 4.6 judge validation only
```

## Requirements

- Python 3.10+
- CUDA toolkit 12.x (for llama.cpp GPU inference)
- llama.cpp built with `LLAMA_CUDA=1` and `llama-server` on PATH
- See `requirements.txt` for Python dependencies
