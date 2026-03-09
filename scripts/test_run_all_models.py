#!/usr/bin/env python3
"""Test run: 100 queries (pass2_loose, RGB counterfactual) across all downloaded models.

Starts/stops llama-server for each model/quant combo automatically.
Writes results to results/raw/<model>_<quant>_rgb_pass2_loose.jsonl.
Resume-safe: skips queries already in output file.
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comprag.generate import build_messages, generate_local
from comprag.retrieve import Retriever
from comprag.server import LlamaCppServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path("/mnt/data/comprag-models")
RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All downloaded model/quant combos with their logical names
MODELS = [
    # Primary: Qwen 2.5 14B (full sweep)
    ("qwen2.5-14b-instruct", "Q3_K_M", "Qwen2.5-14B-Instruct-Q3_K_M.gguf"),
    ("qwen2.5-14b-instruct", "Q4_K_M", "Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
    ("qwen2.5-14b-instruct", "Q5_K_M", "Qwen2.5-14B-Instruct-Q5_K_M.gguf"),
    ("qwen2.5-14b-instruct", "Q6_K", "Qwen2.5-14B-Instruct-Q6_K.gguf"),
    ("qwen2.5-14b-instruct", "Q8_0", "Qwen2.5-14B-Instruct-Q8_0.gguf"),
    ("qwen2.5-14b-instruct", "FP16", "Qwen2.5-14B-Instruct-f16.gguf"),
    # Primary: Phi-4 14B (4 quants)
    ("phi-4-14b", "Q3_K_M", "phi-4-Q3_K_M.gguf"),
    ("phi-4-14b", "Q4_K_M", "phi-4-Q4_K_M.gguf"),
    ("phi-4-14b", "Q5_K_M", "phi-4-Q5_K_M.gguf"),
    ("phi-4-14b", "Q8_0", "phi-4-Q8_0.gguf"),
    # Primary: Llama 3.1 8B (2 quants)
    ("llama-3.1-8b-instruct", "Q4_K_M", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("llama-3.1-8b-instruct", "Q8_0", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    # Primary: Qwen 2.5 7B (2 quants)
    ("qwen2.5-7b-instruct", "Q4_K_M", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ("qwen2.5-7b-instruct", "Q8_0", "Qwen2.5-7B-Instruct-Q8_0.gguf"),
    # Secondary: Mistral NeMo 12B
    ("mistral-nemo-12b-instruct", "Q4_K_M", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"),
    ("mistral-nemo-12b-instruct", "Q8_0", "Mistral-Nemo-Instruct-2407-Q8_0.gguf"),
    # Secondary: Gemma 2 9B
    ("gemma-2-9b-instruct", "Q4_K_M", "gemma-2-9b-it-Q4_K_M.gguf"),
    ("gemma-2-9b-instruct", "Q8_0", "gemma-2-9b-it-Q8_0.gguf"),
    # Floor: SmolLM2 1.7B
    ("smollm2-1.7b-instruct", "Q8_0", "SmolLM2-1.7B-Instruct-Q8_0.gguf"),
    ("smollm2-1.7b-instruct", "FP16", "SmolLM2-1.7B-Instruct-f16.gguf"),
]

DATASET = "rgb"
SUBSET = "counterfactual_robustness"
PASS = "pass2_loose"
LIMIT = 100
PORT = 5741


def load_queries() -> list[dict]:
    path = PROJECT_ROOT / "datasets" / DATASET / "normalized" / f"{SUBSET}.jsonl"
    with open(path) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    return queries[:LIMIT]


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    with open(output_path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["query_id"])
    return ids


def run_model(model_name: str, quant: str, gguf_file: str, queries: list[dict],
              retriever: Retriever) -> None:
    model_path = str(MODELS_DIR / gguf_file)
    output_path = RESULTS_DIR / f"{model_name}_{quant}_{DATASET}_{PASS}.jsonl"

    done_ids = load_done_ids(output_path)
    remaining = [q for q in queries if q.get("query_id", "") not in done_ids]

    if not remaining:
        logger.info("SKIP %s %s — all %d queries done", model_name, quant, len(queries))
        return

    logger.info("RUN %s %s — %d/%d remaining", model_name, quant, len(remaining), len(queries))
    run_id = str(uuid.uuid4())

    # 14B FP16 is 28GB — needs larger ctx for V100 32GB
    ctx_len = 2048 if "14b" in model_name.lower() and quant == "FP16" else 4096

    with LlamaCppServer(model_path, port=PORT) as srv:
        with open(output_path, "a") as out:
            for i, rec in enumerate(remaining):
                text = rec.get("query", rec.get("question", ""))
                context = retriever.query(text=text, collection=f"{DATASET}_{SUBSET}")
                messages = build_messages(text, context, PASS)

                try:
                    response, time_ms, response_model = generate_local(messages)
                except Exception as e:
                    logger.error("FAIL %s %s query %d: %s", model_name, quant, i, e)
                    continue

                record = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model_name,
                    "quantization": quant,
                    "source": "local",
                    "provider": None,
                    "dataset": DATASET,
                    "subset": SUBSET,
                    "pass": PASS,
                    "seed": 42,
                    "query_id": rec.get("query_id", rec.get("sample_id", "")),
                    "query": text,
                    "context_chunks": context,
                    "ground_truth": rec.get("ground_truth", rec.get("answer", "")),
                    "response": response,
                    "generation_time_ms": time_ms,
                    "response_model": response_model,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                logger.info("[%s %s] %d/%d (%.1fs) %s",
                            model_name, quant, i + 1, len(remaining),
                            time_ms / 1000, text[:50])


def main():
    logger.info("Loading queries: %s/%s (limit=%d)", DATASET, SUBSET, LIMIT)
    queries = load_queries()
    logger.info("Loaded %d queries", len(queries))

    index_dir = str(PROJECT_ROOT / "index")
    retriever = Retriever(index_dir=index_dir)

    total = len(MODELS)
    for idx, (model_name, quant, gguf_file) in enumerate(MODELS):
        logger.info("=== Model %d/%d: %s %s ===", idx + 1, total, model_name, quant)
        try:
            run_model(model_name, quant, gguf_file, queries, retriever)
        except Exception as e:
            logger.error("FAILED %s %s: %s", model_name, quant, e)
            continue

    logger.info("Done. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
