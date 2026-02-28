#!/usr/bin/env python3
"""Main eval loop for the CUMRAG benchmark harness.

Orchestrates the full pipeline: retrieve -> generate -> score -> log.
Handles server lifecycle, fault tolerance, multi-seed runs, and the
full (model x quant x hardware x dataset) evaluation matrix.

Importable as module:
    from cumrag.runner import EvalRunner, run_eval

CLI:
    python -m cumrag.runner \\
        --model models/llama-3.1-8b-instruct-q4_k_m.gguf \\
        --dataset rgb --subset noise_robustness \\
        --hardware-tier cpu --num-queries 10 \\
        --output results/raw/smoke_test.jsonl
"""

import argparse
import json
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from cumrag.evaluator import CombinedEvaluator, EvalSample
from cumrag.generator import LlamaServer, format_prompt, load_prompt_template
from cumrag.retriever import Retriever, resolve_collection_name, validate_index
from cumrag.utils import (
    Timer,
    append_jsonl,
    get_hardware_meta,
    get_logger,
    get_ram_usage_mb,
    get_resource_snapshot,
    get_vram_usage_mb,
    load_config,
    read_jsonl,
    set_seed,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = get_logger("cumrag.runner")

# Locked generation params per spec — no per-model overrides
_LOCKED_GEN_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}

_DEFAULT_SEEDS = [42, 43, 44]
_DEFAULT_TOP_K = 5
_SERVER_STARTUP_TIMEOUT = 180.0  # seconds
_SERVER_RESTART_ATTEMPTS = 1


# ---------------------------------------------------------------------------
# GPU temperature helper
# ---------------------------------------------------------------------------


def _get_gpu_temp() -> Optional[int]:
    """Read current GPU temperature via nvidia-smi.

    Returns:
        GPU temperature in Celsius as int, or None if unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = result.stdout.strip()
        if out:
            return int(out.split("\n")[0].strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# v2 error formatting helper
# ---------------------------------------------------------------------------


def _classify_error(exc: Exception) -> str:
    """Classify an exception into a v2 error string: 'error_type: description'.

    Args:
        exc: The exception to classify.

    Returns:
        Formatted error string like 'connection_refused: description'.
    """
    exc_type = type(exc).__name__
    msg = str(exc)

    if isinstance(exc, ConnectionError):
        return f"connection_refused: {msg}"
    if isinstance(exc, MemoryError):
        return f"oom: {msg}"
    if isinstance(exc, TimeoutError):
        return f"timeout: {msg}"
    if isinstance(exc, RuntimeError) and "OOM" in msg.upper():
        return f"oom: {msg}"
    if isinstance(exc, RuntimeError) and "crash" in msg.lower():
        return f"server_crash: {msg}"

    return f"{exc_type.lower()}: {msg}"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset_queries(
    dataset: str,
    eval_subset: Optional[str] = None,
    datasets_dir: Optional[Union[str, Path]] = None,
    num_queries: Optional[int] = None,
) -> list[dict]:
    """Load queries from a dataset directory.

    Expects JSONL files in the dataset directory. Each record should have
    at minimum a 'question' (or 'query') field and optionally 'ground_truth'
    (or 'answer', 'gt_answer').

    Args:
        dataset: Dataset name (rgb, nq, halueval).
        eval_subset: Optional subset filter (e.g. 'noise_robustness').
            If set, only loads records matching this subset.
        datasets_dir: Override datasets directory.
        num_queries: Limit number of queries loaded.

    Returns:
        List of query dicts with at least 'question' and 'ground_truth' keys.

    Raises:
        FileNotFoundError: If dataset directory or files not found.
    """
    base = Path(datasets_dir) if datasets_dir else _PROJECT_ROOT / "datasets"
    dataset_dir = base / dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Run 'python scripts/download_datasets.py --dataset {dataset}' first."
        )

    # Collect all JSONL files in the dataset directory
    jsonl_files = sorted(dataset_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        # Also try .json files
        json_files = sorted(dataset_dir.glob("**/*.json"))
        if json_files:
            # Load JSON files (may be arrays)
            queries = []
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    queries.extend(data)
                elif isinstance(data, dict):
                    # Could be a dict with a 'data' or 'results' key
                    for key in ("data", "results", "questions", "examples"):
                        if key in data and isinstance(data[key], list):
                            queries.extend(data[key])
                            break
                    else:
                        queries.append(data)
            if not queries:
                raise FileNotFoundError(
                    f"No usable data found in JSON files under {dataset_dir}"
                )
        else:
            raise FileNotFoundError(
                f"No JSONL or JSON files found in {dataset_dir}\n"
                f"Run 'python scripts/download_datasets.py --dataset {dataset}' first."
            )
    else:
        queries = []
        for jf in jsonl_files:
            for record in read_jsonl(jf):
                queries.append(record)

    # Normalize field names
    normalized = []
    for q in queries:
        record = {
            "question": q.get("question", q.get("query", q.get("input", ""))),
            "ground_truth": q.get(
                "ground_truth",
                q.get("answer", q.get("gt_answer", q.get("expected", ""))),
            ),
        }
        # Carry through any extra metadata
        for k, v in q.items():
            if k not in ("question", "query", "input", "ground_truth", "answer",
                         "gt_answer", "expected"):
                record[k] = v
        if record["question"]:
            normalized.append(record)

    # Filter by subset if specified
    if eval_subset:
        filtered = [
            q for q in normalized
            if q.get("subset", q.get("eval_subset", q.get("category", "")))
            == eval_subset
        ]
        if filtered:
            normalized = filtered
        else:
            logger.warning(
                "No queries matched subset '%s', using all %d queries",
                eval_subset, len(normalized),
            )

    # Handle ground_truth that's a list
    for q in normalized:
        gt = q["ground_truth"]
        if isinstance(gt, list):
            q["ground_truth"] = gt[0] if gt else ""

    # Limit query count
    if num_queries is not None and num_queries > 0:
        normalized = normalized[:num_queries]

    logger.info(
        "Loaded %d queries from dataset '%s' (subset=%s)",
        len(normalized), dataset, eval_subset,
    )
    return normalized


# ---------------------------------------------------------------------------
# Model name extraction from GGUF path
# ---------------------------------------------------------------------------


def _extract_model_info(model_path: str) -> tuple[str, str]:
    """Extract model name and quantization from a GGUF filename.

    Examples:
        'models/llama-3.1-8b-instruct-q4_k_m.gguf'
            -> ('Llama-3.1-8B-Instruct', 'Q4_K_M')
        'qwen2.5-14b-instruct-q8_0.gguf'
            -> ('Qwen2.5-14B-Instruct', 'Q8_0')

    Returns:
        Tuple of (model_name, quantization). Falls back to filename
        and 'unknown' if parsing fails.
    """
    stem = Path(model_path).stem  # strip .gguf

    # Common quantization suffixes
    quant_suffixes = [
        "q4_k_m", "q4_k_s", "q4_0", "q4_1",
        "q5_k_m", "q5_k_s", "q5_0", "q5_1",
        "q6_k", "q8_0", "q8_1",
        "fp16", "f16", "fp32", "f32",
        "iq4_xs", "iq4_nl",
    ]

    stem_lower = stem.lower()
    quant = "unknown"
    model_name = stem

    for qs in quant_suffixes:
        # Check for -quant or _quant at end
        for sep in ("-", "_"):
            suffix = f"{sep}{qs}"
            if stem_lower.endswith(suffix):
                quant = qs.upper()
                model_name = stem[: -len(suffix)]
                break
        if quant != "unknown":
            break

    # Clean up model name — capitalize parts
    parts = model_name.replace("_", "-").split("-")
    cleaned = []
    for p in parts:
        if p.lower() in ("instruct", "chat", "base"):
            cleaned.append(p.capitalize())
        elif p.upper() == p and len(p) <= 4:
            cleaned.append(p.upper())
        else:
            cleaned.append(p.capitalize() if not any(c.isupper() for c in p) else p)
    model_name = "-".join(cleaned) if cleaned else stem

    return model_name, quant


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Main evaluation runner — orchestrates retrieve -> generate -> score -> log.

    Args:
        config_path: Path to eval_config.yaml. Defaults to project config/.

    Attributes:
        config: Parsed eval_config.yaml dict.
        retriever: ChromaDB retriever instance (lazy-initialized per dataset).
        server: LlamaServer instance (manages lifecycle).
        evaluator: CombinedEvaluator for scoring.
        hardware_meta: Cached hardware metadata dict.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        # Load config
        try:
            if config_path:
                config_dir = Path(config_path)
                if config_dir.is_file():
                    config_dir = config_dir.parent
                self.config = load_config("eval_config", config_dir=config_dir)
            else:
                self.config = load_config("eval_config")
        except FileNotFoundError:
            logger.warning("eval_config.yaml not found, using built-in defaults")
            self.config = {}

        # Retrieval config
        retrieval_cfg = self.config.get("retrieval", {})
        self._top_k = retrieval_cfg.get("top_k", _DEFAULT_TOP_K)
        # v2 config uses index_dir; fall back to deprecated index_path
        self._index_dir = retrieval_cfg.get(
            "index_dir", retrieval_cfg.get("index_path", "index/")
        )

        # Generation config (locked per spec)
        gen_cfg = self.config.get("generation", {})
        self._server_url = gen_cfg.get("server_url", "http://localhost:8080")
        self._prompt_template_path = gen_cfg.get("prompt_template")

        # Evaluation config
        eval_cfg = self.config.get("evaluation", {})
        self._judge_model = eval_cfg.get("judge_model", "claude")
        self._enable_ragas = "ragas" in eval_cfg.get("frameworks", ["ragas", "ragchecker"])
        self._enable_ragchecker = "ragchecker" in eval_cfg.get(
            "frameworks", ["ragas", "ragchecker"]
        )

        # Statistics config
        stats_cfg = self.config.get("statistics", {})
        self._min_runs = stats_cfg.get("min_runs", 3)

        # Output config
        output_cfg = self.config.get("output", {})
        self._raw_results_dir = output_cfg.get("raw_results", "results/raw/")

        # Components (lazy-initialized)
        self._retrievers: dict[str, Retriever] = {}
        self._server: Optional[LlamaServer] = None
        self._evaluator: Optional[CombinedEvaluator] = None
        self._template: Optional[str] = None

        # Hardware metadata (cached once)
        self.hardware_meta = get_hardware_meta()
        logger.info("EvalRunner initialized (hardware: %s)", self.hardware_meta.get("gpu"))

    # --- Component accessors ---

    def _get_retriever(
        self, dataset: str, eval_subset: Optional[str] = None
    ) -> Retriever:
        """Get or create a Retriever for the specified dataset.

        Uses resolve_collection_name() from retriever.py to look up the
        collection name in eval_config, and passes config for validate_index()
        pre-flight check.
        """
        cache_key = f"{dataset}_{eval_subset}" if eval_subset else dataset
        if cache_key not in self._retrievers:
            # Try to resolve collection name via config
            collection_name = None
            if self.config and eval_subset:
                # Build the dataset key used in config collections map
                dataset_key = f"{dataset}_{eval_subset}"
                try:
                    collection_name = resolve_collection_name(
                        dataset_key, self.config
                    )
                    logger.info(
                        "Resolved collection name: %s -> %s",
                        dataset_key, collection_name,
                    )
                except KeyError:
                    logger.debug(
                        "No collection mapping for '%s', using default",
                        dataset_key,
                    )

            self._retrievers[cache_key] = Retriever(
                index_dir=self._index_dir,
                dataset=dataset,
                collection_name=collection_name,
                config=self.config if self.config else None,
            )
        return self._retrievers[cache_key]

    def _get_evaluator(self) -> CombinedEvaluator:
        """Get or create the CombinedEvaluator."""
        if self._evaluator is None:
            self._evaluator = CombinedEvaluator(
                judge_model=self._judge_model,
                enable_ragas=self._enable_ragas,
                enable_ragchecker=self._enable_ragchecker,
            )
        return self._evaluator

    def _get_template(self) -> str:
        """Load and cache the prompt template."""
        if self._template is None:
            self._template = load_prompt_template(self._prompt_template_path)
        return self._template

    # --- Model metadata ---

    def _get_model_context_length(self, model_name: str) -> int:
        """Look up model context length from models.yaml. Default 4096 if unknown."""
        try:
            models_config = load_config("models")
            for key, model_def in models_config.get("models", {}).items():
                if model_def.get("display_name") == model_name or key == model_name:
                    return model_def.get("context_length", 4096)
        except (FileNotFoundError, KeyError):
            pass
        return 4096

    # --- Server lifecycle ---

    def _start_server(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        port: int = 8080,
        **server_kwargs: Any,
    ) -> LlamaServer:
        """Start llama-server for a model. Returns the LlamaServer instance.

        Stops any existing server first.
        """
        self._stop_server()

        server = LlamaServer(port=port)
        server.load_generation_params()

        logger.info("Starting llama-server for model: %s", model_path)
        server.start(
            model_path,
            n_gpu_layers=n_gpu_layers,
            **server_kwargs,
        )
        server.wait_ready(timeout=_SERVER_STARTUP_TIMEOUT)
        logger.info("llama-server ready")

        self._server = server
        return server

    def _stop_server(self) -> None:
        """Stop the current llama-server if running."""
        if self._server is not None:
            logger.info("Stopping llama-server...")
            self._server.stop()
            self._server = None

    def _restart_server(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        port: int = 8080,
        **server_kwargs: Any,
    ) -> bool:
        """Attempt to restart the server after a crash.

        Returns True if restart succeeded, False otherwise.
        """
        logger.warning("Attempting server restart...")
        try:
            self._stop_server()
            time.sleep(2)  # Brief pause for cleanup
            self._start_server(
                model_path,
                n_gpu_layers=n_gpu_layers,
                port=port,
                **server_kwargs,
            )
            return True
        except Exception as e:
            logger.error("Server restart failed: %s", e)
            return False

    # --- Single query execution ---

    def run_single(
        self,
        query: str,
        ground_truth: str,
        dataset: str,
        eval_subset: Optional[str] = None,
        hardware_tier: str = "cpu",
        model_name: str = "unknown",
        quantization: str = "unknown",
        seed: int = 42,
        run_id: int = 1,
        query_id: Optional[str] = None,
        skip_eval: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Run one query through the full pipeline: retrieve -> generate -> score.

        Args:
            query: The question text.
            ground_truth: Expected/reference answer.
            dataset: Dataset name for retrieval collection.
            eval_subset: Evaluation subset name.
            hardware_tier: Hardware tier identifier.
            model_name: Model name for JSONL output.
            quantization: Quantization level string.
            seed: Random seed for this run.
            run_id: Run number (1-indexed).
            query_id: Unique query identifier. Auto-generated if not provided.
            skip_eval: If True, skip scoring (useful for perf-only runs).
            **kwargs: Extra fields to include in the output record.

        Returns:
            Dict matching the v2 spec JSONL output schema.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sample_id = query_id or f"{dataset}_{eval_subset or 'all'}_{str(uuid.uuid4())[:4]}"

        # v2 run_config block
        run_config: dict[str, Any] = {
            "model": model_name,
            "quant": quantization,
            "hardware": hardware_tier,
            "seed": seed,
        }

        # Build the v2 result record
        result_record: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": dataset,
            "subset": eval_subset or "",
            "query": query,
            "ground_truth": ground_truth,
            "response": None,
            "retrieved_chunks": [],
            "run_config": run_config,
            "perf": {},
            "metrics": {},
            "timestamp": timestamp,
            "error": None,
        }

        try:
            # 1. Retrieve top-k chunks
            retriever = self._get_retriever(dataset, eval_subset=eval_subset)
            with Timer() as t_retrieve:
                chunks = retriever.retrieve(query, top_k=self._top_k)

            # v2: structured retrieved_chunks with doc_id, text, distance, rank
            retrieved_chunks = []
            context_texts = []
            for rank_idx, chunk in enumerate(chunks, start=1):
                meta = chunk.get("metadata", {})
                doc_id = meta.get("source", meta.get("doc_id", f"unknown_{rank_idx}"))
                retrieved_chunks.append({
                    "doc_id": doc_id,
                    "text": chunk.get("text", ""),
                    "distance": chunk.get("distance", 0.0),
                    "rank": rank_idx,
                })
                context_texts.append(chunk.get("text", ""))
            result_record["retrieved_chunks"] = retrieved_chunks

            logger.debug(
                "Retrieved %d chunks in %.1fms for query: %s",
                len(chunks), t_retrieve.elapsed_ms, query[:80],
            )

            # 2. Format prompt using locked template
            template = self._get_template()
            prompt = format_prompt(template, query, chunks)

            # Context window safety check
            estimated_tokens = int(len(prompt.split()) * 1.4)
            model_ctx = self._get_model_context_length(model_name)
            if estimated_tokens > 0.9 * model_ctx:
                logger.warning(
                    "Context window risk: estimated prompt ~%d tokens exceeds "
                    "90%% of model context (%d tokens). Consider reducing top_k.",
                    estimated_tokens,
                    model_ctx,
                )

            # 3. Generate response with metrics
            if self._server is None:
                raise RuntimeError(
                    "llama-server not running. Call _start_server() first."
                )

            wall_start = time.monotonic()
            gen_result = self._server.generate_with_metrics(
                prompt,
                seed=seed,
                **_LOCKED_GEN_PARAMS,
            )
            wall_clock_sec = time.monotonic() - wall_start

            generated_answer = gen_result["text"]
            result_record["response"] = generated_answer

            # 4. Collect v2 perf block
            resource_snap = get_resource_snapshot()
            total_tokens = gen_result.get("completion_tokens", 0)
            tokens_per_sec = gen_result.get("tokens_per_second", 0.0)
            ttft_ms = gen_result.get("time_to_first_token_ms", 0.0)
            vram_mb = resource_snap.get("vram_usage_mb")
            gpu_temp = _get_gpu_temp()

            result_record["perf"] = {
                "ttft_ms": round(ttft_ms, 2),
                "total_tokens": total_tokens,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "wall_clock_sec": round(wall_clock_sec, 3),
                "vram_mb": vram_mb,
                "gpu_temp": gpu_temp,
            }

            # 5. Score response (if not skipped)
            eval_metrics: dict[str, Any] = {}
            if not skip_eval:
                try:
                    evaluator = self._get_evaluator()
                    eval_result = evaluator.evaluate(
                        question=query,
                        answer=generated_answer,
                        contexts=context_texts,
                        ground_truth=ground_truth,
                    )
                    eval_metrics = eval_result.to_spec_metrics()
                except Exception as eval_err:
                    logger.error(
                        "Scoring failed for query '%s': %s",
                        query[:60], eval_err,
                    )
                    eval_metrics = {
                        "faithfulness": None,
                        "context_utilization": None,
                        "self_knowledge": None,
                        "noise_sensitivity": None,
                        "answer_relevancy": None,
                        "negative_rejection_rate": None,
                    }

            result_record["metrics"] = eval_metrics

        except Exception as e:
            error_msg = _classify_error(e)
            logger.error("Query failed: %s\n%s", error_msg, traceback.format_exc())
            result_record["response"] = None
            result_record["error"] = error_msg
            result_record["perf"] = {
                "ttft_ms": None,
                "total_tokens": None,
                "tokens_per_sec": None,
                "wall_clock_sec": None,
                "vram_mb": None,
                "gpu_temp": None,
            }
            result_record["metrics"] = {
                "faithfulness": None,
                "context_utilization": None,
                "self_knowledge": None,
                "noise_sensitivity": None,
                "answer_relevancy": None,
                "negative_rejection_rate": None,
            }

        return result_record

    # --- Dataset-level evaluation ---

    def run_dataset(
        self,
        dataset: str,
        eval_subset: Optional[str] = None,
        model_path: Optional[str] = None,
        hardware_tier: str = "cpu",
        num_queries: Optional[int] = None,
        seeds: Optional[list[int]] = None,
        output_path: Optional[Union[str, Path]] = None,
        n_gpu_layers: int = -1,
        port: int = 8080,
        skip_eval: bool = False,
        server_kwargs: Optional[dict] = None,
    ) -> list[dict]:
        """Run full dataset evaluation with multiple seeds.

        Manages server lifecycle: starts before run, stops after.

        Args:
            dataset: Dataset name (rgb, nq, halueval).
            eval_subset: Optional subset filter.
            model_path: Path to GGUF model file. If None, assumes server
                is already running externally.
            hardware_tier: Hardware tier identifier.
            num_queries: Limit number of queries (None = all).
            seeds: List of random seeds for multi-run. Default [42, 43, 44].
            output_path: JSONL output file path. If None, auto-generated.
            n_gpu_layers: GPU layers to offload (-1=all, 0=CPU).
            port: Server port.
            skip_eval: Skip scoring.
            server_kwargs: Extra kwargs for server startup.

        Returns:
            List of result dicts (one per query per seed).
        """
        seeds = seeds or _DEFAULT_SEEDS
        server_kwargs = server_kwargs or {}

        # Extract model info
        if model_path:
            model_name, quantization = _extract_model_info(model_path)
        else:
            model_name, quantization = "external", "unknown"

        # Load dataset queries
        queries = load_dataset_queries(
            dataset=dataset,
            eval_subset=eval_subset,
            num_queries=num_queries,
        )

        if not queries:
            logger.error("No queries loaded for dataset '%s'", dataset)
            return []

        # Determine output path
        if output_path is None:
            output_dir = _PROJECT_ROOT / self._raw_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{dataset}_{hardware_tier}_{ts}.jsonl"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        total_runs = len(queries) * len(seeds)
        logger.info(
            "Starting eval: dataset=%s, subset=%s, model=%s (%s), "
            "queries=%d, seeds=%s, total_runs=%d, output=%s",
            dataset, eval_subset, model_name, quantization,
            len(queries), seeds, total_runs, output_path,
        )

        # Start server if model path provided
        if model_path:
            try:
                self._start_server(
                    model_path,
                    n_gpu_layers=n_gpu_layers,
                    port=port,
                    **server_kwargs,
                )
            except Exception as e:
                logger.error("Failed to start server: %s", e)
                # Log a single failure record in v2 format and return
                fail_record = {
                    "sample_id": f"{dataset}_{eval_subset or 'all'}_server_fail",
                    "dataset": dataset,
                    "subset": eval_subset or "",
                    "query": None,
                    "ground_truth": None,
                    "response": None,
                    "retrieved_chunks": [],
                    "run_config": {
                        "model": model_name,
                        "quant": quantization,
                        "hardware": hardware_tier,
                        "seed": seeds[0] if seeds else 42,
                    },
                    "perf": {
                        "ttft_ms": None,
                        "total_tokens": None,
                        "tokens_per_sec": None,
                        "wall_clock_sec": None,
                        "vram_mb": None,
                        "gpu_temp": None,
                    },
                    "metrics": {},
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "error": _classify_error(e),
                }
                append_jsonl(output_path, fail_record)
                return [fail_record]
        else:
            # External server — create a client-only LlamaServer
            self._server = LlamaServer(server_url=self._server_url)
            self._server.load_generation_params()

        # Register signal handlers to ensure server cleanup on SIGTERM/SIGINT
        _original_sigterm = signal.getsignal(signal.SIGTERM)
        _original_sigint = signal.getsignal(signal.SIGINT)

        def _shutdown_handler(signum, frame):
            signame = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
            logger.warning("Received %s — stopping server and exiting", signame)
            self._stop_server()
            sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, _shutdown_handler)
        signal.signal(signal.SIGINT, _shutdown_handler)

        # Progress tracking
        try:
            from tqdm import tqdm
            progress = tqdm(total=total_runs, desc="Eval", unit="query")
        except ImportError:
            progress = None

        all_results = []
        completed = 0
        errors = 0

        try:
            for seed_idx, seed in enumerate(seeds):
                set_seed(seed)
                run_id = seed_idx + 1

                for q_idx, query_data in enumerate(queries):
                    question = query_data["question"]
                    ground_truth = query_data["ground_truth"]
                    query_id = query_data.get("id", query_data.get("query_id", f"q{q_idx}"))

                    try:
                        result = self.run_single(
                            query=question,
                            ground_truth=ground_truth,
                            dataset=dataset,
                            eval_subset=eval_subset,
                            hardware_tier=hardware_tier,
                            model_name=model_name,
                            quantization=quantization,
                            seed=seed,
                            run_id=run_id,
                            query_id=str(query_id),
                            skip_eval=skip_eval,
                        )

                        append_jsonl(output_path, result)
                        all_results.append(result)
                        completed += 1

                        if result.get("error"):
                            errors += 1

                    except MemoryError as mem_err:
                        # OOM — log and continue
                        logger.error(
                            "OOM on query %d (seed=%d): %s",
                            q_idx, seed, question[:60],
                        )
                        oom_record = {
                            "sample_id": str(query_id),
                            "dataset": dataset,
                            "subset": eval_subset or "",
                            "query": question,
                            "ground_truth": ground_truth,
                            "response": None,
                            "retrieved_chunks": [],
                            "run_config": {
                                "model": model_name,
                                "quant": quantization,
                                "hardware": hardware_tier,
                                "seed": seed,
                            },
                            "perf": {
                                "ttft_ms": None,
                                "total_tokens": None,
                                "tokens_per_sec": None,
                                "wall_clock_sec": None,
                                "vram_mb": None,
                                "gpu_temp": None,
                            },
                            "metrics": {},
                            "timestamp": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "error": _classify_error(mem_err),
                        }
                        append_jsonl(output_path, oom_record)
                        all_results.append(oom_record)
                        errors += 1

                    except (ConnectionError, RuntimeError) as server_err:
                        # Server may have crashed — attempt restart
                        logger.error(
                            "Server error on query %d (seed=%d): %s",
                            q_idx, seed, server_err,
                        )
                        if model_path:
                            restarted = False
                            for attempt in range(_SERVER_RESTART_ATTEMPTS):
                                if self._restart_server(
                                    model_path,
                                    n_gpu_layers=n_gpu_layers,
                                    port=port,
                                    **server_kwargs,
                                ):
                                    restarted = True
                                    break

                            if not restarted:
                                logger.error(
                                    "Server restart failed after %d attempts. "
                                    "Stopping run.",
                                    _SERVER_RESTART_ATTEMPTS,
                                )
                                crash_record = {
                                    "sample_id": str(query_id),
                                    "dataset": dataset,
                                    "subset": eval_subset or "",
                                    "query": question,
                                    "ground_truth": ground_truth,
                                    "response": None,
                                    "retrieved_chunks": [],
                                    "run_config": {
                                        "model": model_name,
                                        "quant": quantization,
                                        "hardware": hardware_tier,
                                        "seed": seed,
                                    },
                                    "perf": {
                                        "ttft_ms": None,
                                        "total_tokens": None,
                                        "tokens_per_sec": None,
                                        "wall_clock_sec": None,
                                        "vram_mb": None,
                                        "gpu_temp": None,
                                    },
                                    "metrics": {},
                                    "timestamp": datetime.now(timezone.utc).strftime(
                                        "%Y-%m-%dT%H:%M:%SZ"
                                    ),
                                    "error": f"server_crash: restart failed after "
                                             f"{_SERVER_RESTART_ATTEMPTS} attempts: {server_err}",
                                }
                                append_jsonl(output_path, crash_record)
                                all_results.append(crash_record)
                                # Break out of both loops
                                if progress:
                                    progress.close()
                                return all_results
                        else:
                            # No model path — can't restart external server
                            logger.error(
                                "External server unreachable, stopping run"
                            )
                            break

                    if progress:
                        progress.update(1)
                        progress.set_postfix(
                            ok=completed - errors, err=errors, seed=seed,
                        )

        except KeyboardInterrupt:
            logger.warning("Run interrupted by user (Ctrl+C)")
        finally:
            if progress:
                progress.close()
            if model_path:
                self._stop_server()
            # Restore original signal handlers
            signal.signal(signal.SIGTERM, _original_sigterm)
            signal.signal(signal.SIGINT, _original_sigint)

        # Summary
        logger.info(
            "Eval complete: %d/%d queries processed, %d errors. Output: %s",
            completed, total_runs, errors, output_path,
        )

        return all_results

    # --- Matrix-level evaluation ---

    def run_matrix(
        self,
        combinations: list[dict],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, list[dict]]:
        """Run multiple (model x quant x hardware x dataset) combinations.

        Each combination dict should have keys:
            - model_path (str): Path to GGUF file
            - dataset (str): Dataset name
            - eval_subset (str, optional): Subset filter
            - hardware_tier (str): Hardware tier ID
            - num_queries (int, optional): Query limit
            - seeds (list[int], optional): Random seeds
            - n_gpu_layers (int, optional): GPU layers
            - port (int, optional): Server port

        Args:
            combinations: List of combination dicts.
            output_dir: Override output directory for all results.

        Returns:
            Dict mapping combination key to list of result dicts.
        """
        if output_dir:
            output_base = Path(output_dir)
        else:
            output_base = _PROJECT_ROOT / self._raw_results_dir
        output_base.mkdir(parents=True, exist_ok=True)

        all_results: dict[str, list[dict]] = {}
        total = len(combinations)

        for idx, combo in enumerate(combinations, start=1):
            model_path = combo.get("model_path", "")
            dataset = combo.get("dataset", "rgb")
            eval_subset = combo.get("eval_subset")
            hardware_tier = combo.get("hardware_tier", "cpu")
            num_queries = combo.get("num_queries")
            seeds = combo.get("seeds")
            n_gpu_layers = combo.get("n_gpu_layers", -1)
            port = combo.get("port", 8080)

            model_name, quant = _extract_model_info(model_path) if model_path else ("external", "unknown")
            combo_key = f"{model_name}_{quant}_{hardware_tier}_{dataset}"
            if eval_subset:
                combo_key += f"_{eval_subset}"

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_base / f"{combo_key}_{ts}.jsonl"

            logger.info(
                "=== Matrix run %d/%d: %s ===", idx, total, combo_key,
            )

            try:
                results = self.run_dataset(
                    dataset=dataset,
                    eval_subset=eval_subset,
                    model_path=model_path,
                    hardware_tier=hardware_tier,
                    num_queries=num_queries,
                    seeds=seeds,
                    output_path=output_path,
                    n_gpu_layers=n_gpu_layers,
                    port=port,
                )
                all_results[combo_key] = results
            except Exception as e:
                logger.error(
                    "Matrix combination '%s' failed: %s\n%s",
                    combo_key, e, traceback.format_exc(),
                )
                # Log the failure in v2 format
                fail_record = {
                    "sample_id": f"{combo_key}_matrix_fail",
                    "dataset": dataset,
                    "subset": eval_subset or "",
                    "query": None,
                    "ground_truth": None,
                    "response": None,
                    "retrieved_chunks": [],
                    "run_config": {
                        "model": model_name,
                        "quant": quant,
                        "hardware": hardware_tier,
                        "seed": seeds[0] if seeds else 42,
                    },
                    "perf": {
                        "ttft_ms": None,
                        "total_tokens": None,
                        "tokens_per_sec": None,
                        "wall_clock_sec": None,
                        "vram_mb": None,
                        "gpu_temp": None,
                    },
                    "metrics": {},
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "error": _classify_error(e),
                }
                append_jsonl(output_path, fail_record)
                all_results[combo_key] = [fail_record]

        logger.info(
            "Matrix evaluation complete: %d/%d combinations processed",
            len([v for v in all_results.values() if v]), total,
        )
        return all_results

    def cleanup(self) -> None:
        """Clean up resources — stop server, close connections."""
        self._stop_server()
        self._retrievers.clear()
        self._evaluator = None


# ---------------------------------------------------------------------------
# Convenience function for module-level import
# ---------------------------------------------------------------------------


def run_eval(
    model_path: str,
    dataset: str = "rgb",
    eval_subset: Optional[str] = None,
    hardware_tier: str = "cpu",
    num_queries: Optional[int] = None,
    seeds: Optional[list[int]] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    n_gpu_layers: int = -1,
    skip_eval: bool = False,
) -> list[dict]:
    """Run a complete evaluation. Convenience wrapper around EvalRunner.

    Args:
        model_path: Path to GGUF model file.
        dataset: Dataset name.
        eval_subset: Optional subset filter.
        hardware_tier: Hardware tier identifier.
        num_queries: Query limit.
        seeds: Random seeds.
        output_path: JSONL output file.
        config_path: Path to eval_config.yaml.
        n_gpu_layers: GPU layers to offload.
        skip_eval: Skip scoring.

    Returns:
        List of result dicts.
    """
    runner = EvalRunner(config_path=config_path)
    try:
        return runner.run_dataset(
            dataset=dataset,
            eval_subset=eval_subset,
            model_path=model_path,
            hardware_tier=hardware_tier,
            num_queries=num_queries,
            seeds=seeds,
            output_path=output_path,
            n_gpu_layers=n_gpu_layers,
            skip_eval=skip_eval,
        )
    finally:
        runner.cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "CUMRAG eval runner -- orchestrate retrieve -> generate -> score -> log.\n"
            "Runs the full evaluation pipeline for a model on a dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model models/llama-3.1-8b-instruct-q4_k_m.gguf "
            "--dataset rgb --subset noise_robustness --hardware-tier cpu "
            "--num-queries 10 --output results/raw/smoke_test.jsonl\n\n"
            "  %(prog)s --model models/qwen2.5-14b-instruct-q4_k_m.gguf "
            "--dataset nq --hardware-tier v100 --num-runs 5\n\n"
            "  %(prog)s --server-url http://localhost:8080 --dataset rgb "
            "--hardware-tier cpu --skip-eval"
        ),
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to GGUF model file. If not set, connects to --server-url.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["rgb", "nq", "halueval"],
        default="rgb",
        help="Evaluation dataset (default: rgb).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/category to filter on (e.g. noise_robustness).",
    )

    # Hardware
    parser.add_argument(
        "--hardware-tier",
        type=str,
        default="cpu",
        help="Hardware tier ID (default: cpu).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1=all, 0=CPU-only). Default: -1.",
    )

    # Run parameters
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to run (default: all).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs with different seeds (default: 3).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds (overrides --num-runs). "
             "E.g.: '42,43,44'",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file path. Auto-generated if not set.",
    )

    # Server
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Connect to an existing llama-server URL instead of starting one.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port when starting a local server (default: 8080).",
    )

    # Eval
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip scoring (RAGAS + RAGChecker). Only collect perf metrics.",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval_config.yaml (default: config/).",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Also write logs to this file.",
    )

    args = parser.parse_args(argv)

    if args.model is None and args.server_url is None:
        parser.error("Either --model or --server-url is required.")

    return args


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    import logging as _logging
    level = getattr(_logging, args.log_level)
    setup_logging(level=level, log_file=args.log_file)

    # Parse seeds
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            logger.error("Invalid --seeds format. Use comma-separated integers.")
            return 1
    else:
        # Generate seeds from num_runs
        seeds = list(range(42, 42 + args.num_runs))

    logger.info("CUMRAG Eval Runner starting")
    logger.info(
        "Config: model=%s, dataset=%s, subset=%s, tier=%s, "
        "queries=%s, seeds=%s",
        args.model or args.server_url,
        args.dataset, args.subset, args.hardware_tier,
        args.num_queries or "all", seeds,
    )

    runner = EvalRunner(config_path=args.config)

    # If server URL provided, override the server connection
    if args.server_url:
        runner._server = LlamaServer(server_url=args.server_url)
        runner._server.load_generation_params()
        model_path = None
    else:
        model_path = args.model

    try:
        results = runner.run_dataset(
            dataset=args.dataset,
            eval_subset=args.subset,
            model_path=model_path,
            hardware_tier=args.hardware_tier,
            num_queries=args.num_queries,
            seeds=seeds,
            output_path=args.output,
            n_gpu_layers=args.n_gpu_layers,
            port=args.port,
            skip_eval=args.skip_eval,
        )

        # Print summary
        total = len(results)
        errored = sum(1 for r in results if r.get("error"))
        print(f"\nEval Summary:")
        print(f"  Total queries: {total}")
        print(f"  Successful:    {total - errored}")
        print(f"  Errors:        {errored}")

        if results and not results[0].get("error"):
            # Print sample metrics from first result
            metrics = results[0].get("metrics", {})
            if metrics:
                print(f"\n  Sample metrics (first query):")
                for k, v in metrics.items():
                    if v is not None:
                        if isinstance(v, float):
                            print(f"    {k}: {v:.4f}")
                        else:
                            print(f"    {k}: {v}")

        return 0 if errored < total else 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
