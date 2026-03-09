#!/usr/bin/env python3
"""Generate pass1_baseline and pass3_strict for Gemma 2 9B Q4/Q8 on RGB counterfactual.

Same 100 queries used in the judge agreement comparison.
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

MODELS = [
    ("gemma-2-9b-instruct", "Q4_K_M", "gemma-2-9b-it-Q4_K_M.gguf"),
    ("gemma-2-9b-instruct", "Q8_0", "gemma-2-9b-it-Q8_0.gguf"),
]

PASSES = ["pass1_baseline", "pass3_strict"]
DATASET = "rgb"
SUBSET = "counterfactual_robustness"
LIMIT = 500
PORT = 5741


def load_queries():
    path = PROJECT_ROOT / "datasets" / DATASET / "normalized" / f"{SUBSET}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()][:LIMIT]


def load_done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["query_id"] for line in f if line.strip()}


def run(model_name, quant, gguf, queries, retriever):
    model_path = str(MODELS_DIR / gguf)

    with LlamaCppServer(model_path, port=PORT) as srv:
        for pass_name in PASSES:
            output_path = RESULTS_DIR / f"{model_name}_{quant}_{DATASET}_{pass_name}.jsonl"
            done_ids = load_done_ids(output_path)
            remaining = [q for q in queries if q.get("query_id", "") not in done_ids]

            if not remaining:
                logger.info("SKIP %s %s %s — done", model_name, quant, pass_name)
                continue

            logger.info("RUN %s %s %s — %d queries", model_name, quant, pass_name, len(remaining))
            run_id = str(uuid.uuid4())
            need_context = pass_name != "pass1_baseline"

            with open(output_path, "a") as out:
                for i, rec in enumerate(remaining):
                    text = rec.get("query", rec.get("question", ""))
                    context = None
                    if need_context and retriever:
                        context = retriever.query(text=text, collection=f"{DATASET}_{SUBSET}")
                    messages = build_messages(text, context, pass_name)

                    try:
                        response, time_ms, response_model = generate_local(messages)
                    except Exception as e:
                        logger.error("FAIL %s %s %s query %d: %s", model_name, quant, pass_name, i, e)
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
                        "pass": pass_name,
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
                    logger.info("[%s %s %s] %d/%d (%.1fs)", model_name, quant, pass_name,
                                i + 1, len(remaining), time_ms / 1000)


def main():
    queries = load_queries()
    logger.info("Loaded %d queries", len(queries))
    retriever = Retriever(index_dir=str(PROJECT_ROOT / "index"))

    for model_name, quant, gguf in MODELS:
        logger.info("=== %s %s ===", model_name, quant)
        run(model_name, quant, gguf, queries, retriever)

    logger.info("Done.")


if __name__ == "__main__":
    main()
