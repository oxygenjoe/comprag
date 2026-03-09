#!/usr/bin/env python3
"""Production run: all 30 model/quant combos × 500 queries (RGB sampled).

Phase 1: Inference — iterate models, start/stop llama-server per combo.
Phase 2: Scoring — start Command R, score all raw results.

Resume-safe throughout. Run with nohup for the full ~27h.
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
from comprag.score import score_ragchecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/production_run.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR_1 = Path("/mnt/data/comprag-models")
MODELS_DIR_2 = PROJECT_ROOT / "models2"
RESULTS_RAW = PROJECT_ROOT / "results" / "raw_production"
RESULTS_SCORED = PROJECT_ROOT / "results" / "scored_production"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)
RESULTS_SCORED.mkdir(parents=True, exist_ok=True)

SAMPLED_DIR = PROJECT_ROOT / "datasets" / "rgb" / "sampled"

PORT = 5741
COMMANDR_GGUF = str(MODELS_DIR_1 / "c4ai-command-r-08-2024-Q4_K_M.gguf")

# All 30 model/quant combos
MODELS = [
    # Primary: Qwen 2.5 14B (6 quants) — all on /mnt/data
    ("qwen2.5-14b-instruct", "Q3_K_M", "Qwen2.5-14B-Instruct-Q3_K_M.gguf", 1),
    ("qwen2.5-14b-instruct", "Q4_K_M", "Qwen2.5-14B-Instruct-Q4_K_M.gguf", 1),
    ("qwen2.5-14b-instruct", "Q5_K_M", "Qwen2.5-14B-Instruct-Q5_K_M.gguf", 1),
    ("qwen2.5-14b-instruct", "Q6_K", "Qwen2.5-14B-Instruct-Q6_K.gguf", 1),
    ("qwen2.5-14b-instruct", "Q8_0", "Qwen2.5-14B-Instruct-Q8_0.gguf", 1),
    ("qwen2.5-14b-instruct", "FP16", "Qwen2.5-14B-Instruct-f16.gguf", 1),
    # Primary: Phi-4 14B (6 quants) — Q6_K and FP16 on models2
    ("phi-4-14b", "Q3_K_M", "phi-4-Q3_K_M.gguf", 1),
    ("phi-4-14b", "Q4_K_M", "phi-4-Q4_K_M.gguf", 1),
    ("phi-4-14b", "Q5_K_M", "phi-4-Q5_K_M.gguf", 1),
    ("phi-4-14b", "Q6_K", "phi-4-Q6_K.gguf", 2),
    ("phi-4-14b", "Q8_0", "phi-4-Q8_0.gguf", 1),
    ("phi-4-14b", "FP16", "phi-4-f16.gguf", 2),
    # Primary: Llama 3.1 8B (6 quants) — Q3, Q5, Q6, FP16 on models2
    ("llama-3.1-8b-instruct", "Q3_K_M", "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf", 2),
    ("llama-3.1-8b-instruct", "Q4_K_M", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 1),
    ("llama-3.1-8b-instruct", "Q5_K_M", "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", 2),
    ("llama-3.1-8b-instruct", "Q6_K", "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", 2),
    ("llama-3.1-8b-instruct", "Q8_0", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 1),
    ("llama-3.1-8b-instruct", "FP16", "Meta-Llama-3.1-8B-Instruct-f16.gguf", 2),
    # Primary: Qwen 2.5 7B (6 quants) — Q3, Q5, Q6, FP16 on models2
    ("qwen2.5-7b-instruct", "Q3_K_M", "Qwen2.5-7B-Instruct-Q3_K_M.gguf", 2),
    ("qwen2.5-7b-instruct", "Q4_K_M", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", 1),
    ("qwen2.5-7b-instruct", "Q5_K_M", "Qwen2.5-7B-Instruct-Q5_K_M.gguf", 2),
    ("qwen2.5-7b-instruct", "Q6_K", "Qwen2.5-7B-Instruct-Q6_K.gguf", 2),
    ("qwen2.5-7b-instruct", "Q8_0", "Qwen2.5-7B-Instruct-Q8_0.gguf", 1),
    ("qwen2.5-7b-instruct", "FP16", "Qwen2.5-7B-Instruct-f16.gguf", 2),
    # Secondary: Mistral NeMo 12B (2 quants)
    ("mistral-nemo-12b-instruct", "Q4_K_M", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", 1),
    ("mistral-nemo-12b-instruct", "Q8_0", "Mistral-Nemo-Instruct-2407-Q8_0.gguf", 1),
    # Secondary: Gemma 2 9B (2 quants)
    ("gemma-2-9b-instruct", "Q4_K_M", "gemma-2-9b-it-Q4_K_M.gguf", 1),
    ("gemma-2-9b-instruct", "Q8_0", "gemma-2-9b-it-Q8_0.gguf", 1),
    # Floor: SmolLM2 1.7B (2 quants)
    ("smollm2-1.7b-instruct", "Q8_0", "SmolLM2-1.7B-Instruct-Q8_0.gguf", 1),
    ("smollm2-1.7b-instruct", "FP16", "SmolLM2-1.7B-Instruct-f16.gguf", 1),
]

# Subsets and their pass policies
EVAL_PLAN = [
    ("counterfactual_robustness", ["pass1_baseline", "pass2_loose", "pass3_strict"]),
    ("noise_robustness", ["pass2_loose"]),
    ("negative_rejection", ["pass2_loose"]),
]


def resolve_model_path(gguf: str, disk: int) -> str:
    base = MODELS_DIR_1 if disk == 1 else MODELS_DIR_2
    return str(base / gguf)


def load_queries(subset: str) -> list[dict]:
    with open(SAMPLED_DIR / f"{subset}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["query_id"])
    return ids


def run_inference_model(model_name: str, quant: str, gguf: str, disk: int,
                        retriever: Retriever) -> list[Path]:
    """Run all subsets/passes for one model. Returns list of raw output paths."""
    model_path = resolve_model_path(gguf, disk)
    ctx_len = 2048 if "14b" in model_name.lower() and quant == "FP16" else 4096
    output_paths = []

    with LlamaCppServer(model_path, port=PORT) as srv:
        for subset, passes in EVAL_PLAN:
            queries = load_queries(subset)
            for pass_name in passes:
                output_path = RESULTS_RAW / f"{model_name}_{quant}_rgb_{subset}_{pass_name}.jsonl"
                output_paths.append(output_path)

                done_ids = load_done_ids(output_path)
                remaining = [q for q in queries if q.get("query_id", "") not in done_ids]
                if not remaining:
                    logger.info("SKIP %s %s %s %s — done", model_name, quant, subset, pass_name)
                    continue

                logger.info("GEN %s %s %s %s — %d queries",
                            model_name, quant, subset, pass_name, len(remaining))
                run_id = str(uuid.uuid4())
                need_context = pass_name != "pass1_baseline"

                with open(output_path, "a") as out:
                    for i, rec in enumerate(remaining):
                        text = rec.get("query", rec.get("question", ""))
                        context = None
                        if need_context:
                            rec_subset = rec.get("subset", subset)
                            context = retriever.query(
                                text=text, collection=f"rgb_{rec_subset}")
                        messages = build_messages(text, context, pass_name)
                        try:
                            response, time_ms, response_model = generate_local(messages)
                        except Exception as e:
                            logger.error("FAIL %s %s %s %s q%d: %s",
                                         model_name, quant, subset, pass_name, i, e)
                            continue

                        record = {
                            "run_id": run_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model_name,
                            "quantization": quant,
                            "source": "local",
                            "provider": None,
                            "dataset": "rgb",
                            "subset": subset,
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
                        if (i + 1) % 10 == 0 or i + 1 == len(remaining):
                            logger.info("[%s %s %s %s] %d/%d",
                                        model_name, quant, subset, pass_name,
                                        i + 1, len(remaining))
    return output_paths


def run_scoring(raw_paths: list[Path]) -> None:
    """Score all raw result files with Command R local judge."""
    import os
    os.environ["_LOCAL_JUDGE"] = "not-needed"

    with LlamaCppServer(COMMANDR_GGUF, port=PORT) as srv:
        for raw_path in sorted(raw_paths):
            if not raw_path.exists():
                continue
            scored_path = RESULTS_SCORED / raw_path.name
            done_ids = load_done_ids(scored_path)

            with open(raw_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            remaining = [r for r in records if r.get("query_id", "") not in done_ids]

            if not remaining:
                logger.info("SCORE SKIP %s — done", raw_path.name)
                continue

            logger.info("SCORE %s — %d/%d remaining", raw_path.name,
                        len(remaining), len(records))

            with open(scored_path, "a") as out:
                for i, rec in enumerate(remaining):
                    try:
                        rc = score_ragchecker(
                            rec["query"], rec["response"],
                            rec["context_chunks"] or [], rec["ground_truth"],
                            judge_provider="local",
                            judge_model="c4ai-command-r-08-2024",
                        )
                    except Exception as e:
                        logger.error("SCORE FAIL %s q%d: %s", raw_path.name, i, e)
                        rc = {k: 0.0 for k in [
                            "overall_precision", "overall_recall", "overall_f1",
                            "claim_recall", "context_precision", "context_utilization",
                            "noise_sensitivity_relevant", "noise_sensitivity_irrelevant",
                            "hallucination", "self_knowledge", "faithfulness"]}

                    rec["scores"] = {"ragchecker": rc, "ragas": None}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if (i + 1) % 10 == 0 or i + 1 == len(remaining):
                        logger.info("SCORED [%d/%d] %s", i + 1, len(remaining), raw_path.name)


def main():
    logger.info("=" * 60)
    logger.info("COMPRAG PRODUCTION RUN")
    logger.info("30 models × 500 queries = 15,000 inference + scoring")
    logger.info("=" * 60)

    # Phase 1: Inference
    logger.info("=== PHASE 1: INFERENCE ===")
    retriever = Retriever(index_dir=str(PROJECT_ROOT / "index"))
    all_raw_paths: list[Path] = []

    for idx, (model_name, quant, gguf, disk) in enumerate(MODELS):
        logger.info("=== Model %d/%d: %s %s ===", idx + 1, len(MODELS), model_name, quant)
        try:
            paths = run_inference_model(model_name, quant, gguf, disk, retriever)
            all_raw_paths.extend(paths)
        except Exception as e:
            logger.error("MODEL FAILED %s %s: %s", model_name, quant, e)
            continue

    logger.info("=== PHASE 1 COMPLETE: %d raw files ===", len(all_raw_paths))

    # Phase 2: Scoring with Command R
    logger.info("=== PHASE 2: SCORING (Command R local) ===")
    # Collect all raw files (including any from previous partial runs)
    all_raw_paths = sorted(RESULTS_RAW.glob("*.jsonl"))
    run_scoring(all_raw_paths)

    logger.info("=== PHASE 2 COMPLETE ===")

    # Phase 3: Aggregate + Visualize
    logger.info("=== PHASE 3: AGGREGATE + VISUALIZE ===")
    from comprag.aggregate import aggregate_results
    from comprag.visualize import generate_all_figures

    agg_dir = str(PROJECT_ROOT / "results" / "aggregated_production")
    fig_dir = str(PROJECT_ROOT / "results" / "figures_production")

    results = aggregate_results(str(RESULTS_SCORED), agg_dir)
    logger.info("Aggregated %d groups", len(results))

    paths = generate_all_figures(agg_dir, fig_dir)
    logger.info("Generated %d figures", len(paths))

    logger.info("=== PRODUCTION RUN COMPLETE ===")


if __name__ == "__main__":
    main()
