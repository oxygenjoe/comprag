"""CompRAG CLI entry point — thin wrappers around comprag modules."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dataset_queries(dataset: str, subset: str | None) -> list[dict[str, Any]]:
    """Load normalized JSONL queries from datasets/<dataset>/normalized/."""
    datasets_dir = PROJECT_ROOT / "datasets" / dataset / "normalized"
    if not datasets_dir.exists():
        logger.error("Dataset directory not found: %s", datasets_dir)
        sys.exit(1)
    records: list[dict[str, Any]] = []
    for jsonl_file in sorted(datasets_dir.glob("*.jsonl")):
        if subset and jsonl_file.stem != subset:
            continue
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d queries from %s (subset=%s)", len(records), dataset, subset)
    return records


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Run retrieval: query ChromaDB for each dataset sample."""
    from comprag.retrieve import Retriever

    logging.basicConfig(level=logging.INFO)
    index_dir = str(PROJECT_ROOT / "index")
    retriever = Retriever(index_dir=index_dir)
    queries = _load_dataset_queries(args.dataset, args.subset)
    collection = f"{args.dataset}_{args.subset}" if args.subset else args.dataset

    for query_rec in queries:
        text = query_rec.get("query", query_rec.get("question", ""))
        chunks = retriever.query(text=text, collection=collection)
        logger.info("query=%s chunks=%d", text[:60], len(chunks))


def _load_completed_query_ids(output_path: Path) -> set[str]:
    """Load query IDs already written to an output JSONL file for resume."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    qid = rec.get("query_id", "")
                    if qid:
                        done.add(qid)
                except json.JSONDecodeError:
                    continue
    return done


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation: local llama.cpp or frontier API, write raw JSONL."""
    logging.basicConfig(level=logging.INFO)
    pass_name: str = getattr(args, "pass")
    queries = _load_dataset_queries(args.dataset, getattr(args, "subset", None))
    run_id = str(uuid.uuid4())
    output_dir = PROJECT_ROOT / "results" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}_{args.dataset}_{pass_name}.jsonl"

    # Resume: skip queries already in output file.
    done_ids = _load_completed_query_ids(output_path)
    if done_ids:
        total_before = len(queries)
        queries = [q for q in queries if q.get("query_id", q.get("sample_id", q.get("id", ""))) not in done_ids]
        logger.info("Resuming: %d/%d queries already done, %d remaining", len(done_ids), total_before, len(queries))
    if not queries:
        logger.info("All queries already completed in %s", output_path)
        return

    if args.frontier:
        # Frontier models only run pass2_loose and pass3_strict (frontier.yaml contract)
        if pass_name not in ("pass2_loose", "pass3_strict"):
            logger.error(
                "Frontier models only support pass2_loose and pass3_strict, got '%s'",
                pass_name,
            )
            sys.exit(1)
        _run_generate_frontier(args, queries, pass_name, run_id, output_path)
    else:
        _run_generate_local(args, queries, pass_name, run_id, output_path)
    logger.info("Wrote %d records to %s", len(queries), output_path)


def _run_generate_local(
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    pass_name: str,
    run_id: str,
    output_path: Path,
) -> None:
    """Generate responses using local llama.cpp server."""
    from comprag.generate import build_messages, generate_local
    from comprag.retrieve import Retriever

    index_dir = str(PROJECT_ROOT / "index")
    need_context = pass_name != "pass1_baseline"
    retriever = Retriever(index_dir=index_dir) if need_context else None
    subset = getattr(args, "subset", None)
    collection = f"{args.dataset}_{subset}" if subset else args.dataset

    with open(output_path, "a") as out:
        for i, rec in enumerate(queries):
            text = rec.get("query", rec.get("question", ""))
            context = None
            if retriever and need_context:
                rec_subset = rec.get("subset", subset or "default")
                rec_collection = f"{args.dataset}_{rec_subset}" if rec_subset and rec_subset != "default" else collection
                context = retriever.query(text=text, collection=rec_collection)
            messages = build_messages(text, context, pass_name)
            response, time_ms, _model = generate_local(messages)
            record = _build_raw_record(
                run_id, args.model, args.quant or "unknown", "local", None,
                args.dataset, rec.get("subset", "default"), pass_name,
                args.seed, rec, text, context, response, time_ms,
            )
            out.write(json.dumps(record) + "\n")
            out.flush()
            logger.info("[%d/%d] %s", i + 1, len(queries), text[:60])


def _run_generate_frontier(
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    pass_name: str,
    run_id: str,
    output_path: Path,
) -> None:
    """Generate responses using frontier API."""
    from comprag.generate import build_messages, generate_frontier
    from comprag.retrieve import Retriever

    index_dir = str(PROJECT_ROOT / "index")
    need_context = pass_name != "pass1_baseline"
    retriever = Retriever(index_dir=index_dir) if need_context else None
    subset = getattr(args, "subset", None)
    collection = f"{args.dataset}_{subset}" if subset else args.dataset

    with open(output_path, "a") as out:
        for i, rec in enumerate(queries):
            text = rec.get("query", rec.get("question", ""))
            context = None
            if retriever and need_context:
                rec_subset = rec.get("subset", subset or "default")
                rec_collection = f"{args.dataset}_{rec_subset}" if rec_subset and rec_subset != "default" else collection
                context = retriever.query(text=text, collection=rec_collection)
            messages = build_messages(text, context, pass_name)
            response, time_ms, _model = generate_frontier(
                messages, args.provider, args.model, seed=args.seed,
            )
            record = _build_raw_record(
                run_id, args.model, "API", "api", args.provider,
                args.dataset, rec.get("subset", "default"), pass_name,
                args.seed, rec, text, context, response, time_ms,
            )
            out.write(json.dumps(record) + "\n")
            out.flush()
            logger.info("[%d/%d] %s", i + 1, len(queries), text[:60])


def _build_raw_record(
    run_id: str, model: str, quant: str, source: str, provider: str | None,
    dataset: str, subset: str, pass_name: str, seed: int,
    rec: dict, query: str, context: list[str] | None,
    response: str, time_ms: int,
) -> dict[str, Any]:
    """Construct a single raw JSONL record matching the spec schema."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "quantization": quant,
        "source": source,
        "provider": provider,
        "dataset": dataset,
        "subset": subset,
        "pass": pass_name,
        "seed": seed,
        "query_id": rec.get("query_id", rec.get("sample_id", rec.get("id", ""))),
        "query": query,
        "context_chunks": context,
        "ground_truth": rec.get("ground_truth", rec.get("answer", "")),
        "response": response,
        "generation_time_ms": time_ms,
        "scores": None,
    }


def cmd_score(args: argparse.Namespace) -> None:
    """Score each record in a raw JSONL file, write scored JSONL."""
    from comprag.score import score_ragchecker

    logging.basicConfig(level=logging.INFO)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_dir = PROJECT_ROOT / "results" / "scored"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    judge_provider = getattr(args, "judge_provider", "anthropic")
    judge_model = getattr(args, "judge_model", "claude-opus-4-6")
    records: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Resume: skip already-scored records.
    done_ids = _load_completed_query_ids(output_path)
    if done_ids:
        total_before = len(records)
        records = [r for r in records if r.get("query_id", "") not in done_ids]
        logger.info("Scoring resume: %d/%d already done, %d remaining", len(done_ids), total_before, len(records))
    if not records:
        logger.info("All records already scored in %s", output_path)
        return

    with open(output_path, "a") as out:
        for i, rec in enumerate(records):
            rc = score_ragchecker(
                rec["query"], rec["response"],
                rec["context_chunks"] or [], rec["ground_truth"],
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            rec["scores"] = {"ragchecker": rc, "ragas": None}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            logger.info("Scored [%d/%d] %s", i + 1, len(records), rec.get("query_id", "")[:60])

    logger.info("Scored %d records -> %s", len(records), output_path)


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Aggregate scored JSONL files with bootstrap statistics."""
    from comprag.aggregate import aggregate_results

    logging.basicConfig(level=logging.INFO)
    results = aggregate_results(args.input_dir)
    logger.info("Aggregated %d groups", len(results))


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate all figures from aggregated results."""
    from comprag.visualize import generate_all_figures

    logging.basicConfig(level=logging.INFO)
    paths = generate_all_figures(args.input_dir, args.output_dir)
    logger.info("Generated %d figures", len(paths))


def _add_retrieve_parser(sub: argparse._SubParsersAction) -> None:
    """Add retrieve subcommand."""
    p = sub.add_parser("retrieve", help="Run retrieval on a dataset")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. rgb, nq, halueval)")
    p.add_argument("--subset", type=str, default=None, help="Dataset subset (e.g. counterfactual)")
    p.set_defaults(func=cmd_retrieve)


def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
    """Add generate subcommand."""
    p = sub.add_parser("generate", help="Run generation on a dataset")
    p.add_argument("--model", type=str, required=True, help="Model name")
    p.add_argument("--quant", type=str, default=None, help="Quantization level (e.g. Q4_K_M)")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name")
    p.add_argument("--pass", type=str, required=True,
                   choices=["pass1_baseline", "pass2_loose", "pass3_strict"],
                   help="Evaluation pass")
    p.add_argument("--frontier", action="store_true", help="Use frontier API model")
    p.add_argument("--provider", type=str, default=None,
                   choices=["openai", "anthropic", "google", "deepseek", "zhipu"],
                   help="API provider (required with --frontier)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.set_defaults(func=cmd_generate)


def _add_score_parser(sub: argparse._SubParsersAction) -> None:
    """Add score subcommand."""
    p = sub.add_parser("score", help="Score a results JSONL file")
    p.add_argument("--input", type=str, required=True, help="Path to raw results JSONL")
    p.add_argument("--judge-provider", type=str, default="anthropic",
                   choices=["anthropic", "openai", "google", "deepseek"],
                   help="Frontier API provider for judge")
    p.add_argument("--judge-model", type=str, default="claude-opus-4-6",
                   help="Model ID for the judge")
    p.set_defaults(func=cmd_score)


def _add_aggregate_parser(sub: argparse._SubParsersAction) -> None:
    """Add aggregate subcommand."""
    p = sub.add_parser("aggregate", help="Aggregate scored results")
    p.add_argument("--input-dir", type=str, required=True, help="Directory of scored JSONL files")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.set_defaults(func=cmd_aggregate)


def _add_visualize_parser(sub: argparse._SubParsersAction) -> None:
    """Add visualize subcommand."""
    p = sub.add_parser("visualize", help="Generate figures from aggregated results")
    p.add_argument("--input-dir", type=str, required=True, help="Directory of aggregated results")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for figures")
    p.set_defaults(func=cmd_visualize)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(prog="comprag", description="CompRAG benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_retrieve_parser(sub)
    _add_generate_parser(sub)
    _add_score_parser(sub)
    _add_aggregate_parser(sub)
    _add_visualize_parser(sub)
    return parser


def main() -> None:
    """Parse arguments and dispatch to subcommand handler."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
