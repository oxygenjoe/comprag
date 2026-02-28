"""RAGAS + RAGChecker evaluation wrapper for CUMRAG.

Provides scoring pipelines for RAG faithfulness, context utilization,
self-knowledge, noise sensitivity, and answer relevancy. Wraps both
RAGAS and RAGChecker frameworks with graceful degradation — if one
framework fails, the other's results are still returned.

Judge model runs locally via llama.cpp on port 8081 (separate from the
generation server on 8080). Both servers cannot run simultaneously on
V100 due to VRAM limits — the scoring pipeline is strictly sequential:
stop gen server -> start judge server -> score -> stop judge server.

Importable as module; CLI mode to evaluate a single JSONL file:
    python -m cumrag.evaluator --input results/raw/run.jsonl --output results/raw/scored.jsonl

Classes:
    RAGASEvaluator: RAGAS framework wrapper (faithfulness, answer_relevancy, etc.)
    RAGCheckerEvaluator: RAGChecker wrapper (context_utilization, self_knowledge, noise_sensitivity)
    CombinedEvaluator: Runs both, returns unified metrics dict
"""

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from cumrag.utils import (
    Timer,
    append_jsonl,
    get_logger,
    load_config,
    read_jsonl,
)

logger = get_logger("cumrag.evaluator")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"
_JUDGE_PORT = 8081
_GEN_PORT = 8080

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """Single evaluation sample with question, answer, contexts, and ground truth.

    Attributes:
        question: The user query.
        answer: The model-generated answer.
        contexts: List of retrieved context strings.
        ground_truth: The reference/gold answer (optional for some metrics).
        metadata: Optional extra fields carried through evaluation.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Unified evaluation result from one or both frameworks.

    All metric fields default to None (not computed / framework failed).
    """

    # RAGAS metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # RAGChecker metrics
    context_utilization: Optional[float] = None
    self_knowledge: Optional[float] = None
    noise_sensitivity: Optional[float] = None

    # Composite / derived
    negative_rejection_rate: Optional[float] = None

    # Error tracking
    ragas_error: Optional[str] = None
    ragchecker_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, omitting None values for clean output."""
        d = {}
        for k in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
            "self_knowledge",
            "noise_sensitivity",
            "negative_rejection_rate",
        ]:
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        # Include errors if present
        if self.ragas_error:
            d["ragas_error"] = self.ragas_error
        if self.ragchecker_error:
            d["ragchecker_error"] = self.ragchecker_error
        return d

    def to_spec_metrics(self) -> dict[str, Any]:
        """Return metrics dict matching the spec schema exactly.

        Always includes all six spec metrics, using None for unavailable.
        """
        return {
            "faithfulness": self.faithfulness,
            "context_utilization": self.context_utilization,
            "self_knowledge": self.self_knowledge,
            "noise_sensitivity": self.noise_sensitivity,
            "answer_relevancy": self.answer_relevancy,
            "negative_rejection_rate": self.negative_rejection_rate,
        }


# ---------------------------------------------------------------------------
# Judge model configuration — local llama.cpp on port 8081
# ---------------------------------------------------------------------------


def _load_judge_config() -> dict[str, Any]:
    """Load judge configuration from eval_config.yaml.

    Returns:
        Judge config dict with keys: model, quant, server_port,
        context_length, max_tokens, temperature, self_judge_quant.

    Raises:
        FileNotFoundError: If eval_config.yaml not found.
        KeyError: If 'judge' block missing from config.
    """
    config = load_config("eval_config")
    judge = config.get("judge")
    if not judge:
        raise KeyError(
            "Missing 'judge' block in eval_config.yaml. "
            "Required fields: model, quant, server_port."
        )
    return judge


def _resolve_judge_gguf(
    judge_config: dict[str, Any],
    eval_model_path: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the judge GGUF file path from config.

    Handles self-judge quant swap: when the judge model matches the eval
    model, use self_judge_quant (Q8_0) to avoid circular evaluation with
    the same weights.

    Args:
        judge_config: Judge config dict from eval_config.yaml.
        eval_model_path: Path to the generation model GGUF (for self-judge
            detection). If None, no self-judge swap is attempted.
        models_dir: Override models directory. Defaults to <project>/models/.

    Returns:
        Path to the judge GGUF file.

    Raises:
        FileNotFoundError: If the resolved GGUF file does not exist.
    """
    base = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
    model = judge_config["model"]
    quant = judge_config["quant"]

    # Self-judge detection: if judge model name matches eval model, swap quant
    if eval_model_path:
        eval_stem = Path(eval_model_path).stem.lower()
        judge_stem = model.lower()
        # Check if the eval model filename contains the judge model name
        if judge_stem in eval_stem:
            self_quant = judge_config.get("self_judge_quant")
            if self_quant:
                logger.info(
                    "Self-judge detected: judge model '%s' matches eval model '%s'. "
                    "Swapping quant %s -> %s",
                    model, eval_model_path, quant, self_quant,
                )
                quant = self_quant

    # Build GGUF filename: {Model}-{Quant}.gguf
    # Capitalize model parts to match HuggingFace naming convention
    # e.g. "qwen2.5-14b-instruct" -> "Qwen2.5-14B-Instruct"
    parts = model.split("-")
    capitalized = []
    for p in parts:
        if p.lower() in ("instruct", "chat", "base", "coder"):
            capitalized.append(p.capitalize())
        elif p.upper() == p and len(p) <= 4:
            capitalized.append(p.upper())
        elif p.lower().endswith("b") and p[:-1].replace(".", "").isdigit():
            # Size like "14b" -> "14B"
            capitalized.append(p[:-1] + "B")
        else:
            capitalized.append(
                p.capitalize() if not any(c.isupper() for c in p) else p
            )
    model_name = "-".join(capitalized)
    quant_upper = quant.upper().replace("-", "_")
    filename = f"{model_name}-{quant_upper}.gguf"

    gguf_path = base / filename
    if not gguf_path.exists():
        # Try case-insensitive search as fallback
        for f in base.iterdir():
            if f.name.lower() == filename.lower() and f.suffix == ".gguf":
                logger.info(
                    "Found judge GGUF via case-insensitive match: %s", f.name
                )
                return f
        raise FileNotFoundError(
            f"Judge GGUF not found: {gguf_path}\n"
            f"Expected filename: {filename}\n"
            f"Models dir: {base}\n"
            f"Available GGUFs: {[f.name for f in base.glob('*.gguf')]}"
        )

    return gguf_path


def _get_judge_llm(judge_config: dict[str, Any]) -> Any:
    """Instantiate the LLM judge for RAGAS evaluation via local llama.cpp.

    Uses LangChain's ChatOpenAI pointed at the local llama.cpp judge server
    on port 8081. This is the ONLY place LangChain enters the pipeline.

    Args:
        judge_config: Judge config dict from eval_config.yaml.

    Returns:
        A LangChain ChatOpenAI instance pointed at localhost:8081.
    """
    port = judge_config.get("server_port", _JUDGE_PORT)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai required for local judge LLM: "
            "pip install langchain-openai"
        )

    return ChatOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
        model=judge_config["model"],
        temperature=judge_config.get("temperature", 0.0),
        max_tokens=judge_config.get("max_tokens", 1024),
    )


def _get_judge_embeddings() -> Any:
    """Get embeddings model for RAGAS (uses same model as retrieval pipeline).

    Returns:
        A LangChain embeddings instance.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-huggingface or langchain-community required: "
                "pip install langchain-huggingface"
            )
    try:
        config = load_config("eval_config")
        model_name = config["retrieval"]["embedding_model"]
    except (FileNotFoundError, KeyError):
        model_name = "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


# ---------------------------------------------------------------------------
# Judge server lifecycle
# ---------------------------------------------------------------------------


def start_judge_server(
    judge_config: dict[str, Any],
    eval_model_path: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> "LlamaServer":
    """Start the llama.cpp judge server on port 8081.

    Resolves the judge GGUF from config, handles self-judge quant swap,
    starts the server, and waits for /health 200.

    Args:
        judge_config: Judge config dict from eval_config.yaml.
        eval_model_path: Path to generation model GGUF (for self-judge detection).
        models_dir: Override models directory.

    Returns:
        Running LlamaServer instance (caller must call stop()).

    Raises:
        FileNotFoundError: If judge GGUF or server binary not found.
        TimeoutError: If judge server doesn't become ready.
        RuntimeError: If server fails to start.
    """
    from cumrag.generator import LlamaServer

    gguf_path = _resolve_judge_gguf(
        judge_config,
        eval_model_path=eval_model_path,
        models_dir=models_dir,
    )
    port = judge_config.get("server_port", _JUDGE_PORT)
    ctx_len = judge_config.get("context_length", 8192)

    logger.info(
        "Starting judge server: model=%s port=%d ctx=%d",
        gguf_path.name, port, ctx_len,
    )

    server = LlamaServer(port=port)
    server.start(
        model_path=gguf_path,
        port=port,
        n_gpu_layers=-1,
        ctx_size=ctx_len,
    )

    logger.info("Waiting for judge server readiness...")
    server.wait_ready(timeout=180.0)
    logger.info("Judge server ready on port %d", port)

    return server


def stop_judge_server(server: "LlamaServer") -> None:
    """Stop the judge server cleanly.

    Args:
        server: LlamaServer instance to stop.
    """
    if server is not None:
        logger.info("Stopping judge server...")
        server.stop()
        logger.info("Judge server stopped")


def stop_generation_server(
    gen_server: Optional["LlamaServer"] = None,
) -> None:
    """Stop the generation server (port 8080) if running.

    Must be called before starting the judge server to free VRAM.
    Both servers cannot run simultaneously on V100.

    Args:
        gen_server: LlamaServer instance for generation. If None,
            attempts to kill any orphaned llama-server on port 8080.
    """
    if gen_server is not None:
        logger.info("Stopping generation server to free VRAM for judge...")
        gen_server.stop()
        logger.info("Generation server stopped")
    else:
        # Try to clean up orphaned process on gen port
        from cumrag.generator import LlamaServer
        temp = LlamaServer(port=_GEN_PORT)
        temp._check_port_available(_GEN_PORT)


# ---------------------------------------------------------------------------
# RAGChecker input format bridge
# ---------------------------------------------------------------------------


def to_ragchecker_format(raw_results: list[dict]) -> dict:
    """Convert raw evaluation results to RAGChecker's expected input format.

    Maps the CUMRAG result schema to RAGChecker's JSON structure:
      - sample_id -> query_id
      - query -> query
      - ground_truth -> gt_answer
      - response -> response
      - retrieved_chunks -> retrieved_context (with doc_id/text per chunk)

    Args:
        raw_results: List of dicts from the eval runner, each with:
            sample_id, query, ground_truth, response, retrieved_chunks.

    Returns:
        Dict with 'results' key containing the RAGChecker-formatted list.
    """
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
            if entry.get("response") is not None and entry.get("error") is None
        ]
    }


# ---------------------------------------------------------------------------
# RAGAS Evaluator
# ---------------------------------------------------------------------------


class RAGASEvaluator:
    """Wrapper around the RAGAS evaluation framework.

    Computes: faithfulness, answer_relevancy, context_precision, context_recall.
    Uses a local llama.cpp judge server on port 8081 for metric computation.

    Args:
        judge_config: Judge config dict from eval_config.yaml.
        metrics: Optional list of metric names to compute. Defaults to all four.

    Example::

        judge_config = _load_judge_config()
        evaluator = RAGASEvaluator(judge_config=judge_config)
        result = evaluator.evaluate(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["France is a country in Europe. Its capital is Paris."],
            ground_truth="Paris",
        )
        print(result)  # {"faithfulness": 1.0, "answer_relevancy": 0.95, ...}
    """

    # Map of metric name -> RAGAS metric class factory
    AVAILABLE_METRICS = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }

    def __init__(
        self,
        judge_config: dict[str, Any],
        metrics: Optional[list[str]] = None,
    ) -> None:
        self.judge_config = judge_config
        self.metric_names = metrics or list(self.AVAILABLE_METRICS)

        # Validate metric names
        invalid = set(self.metric_names) - self.AVAILABLE_METRICS
        if invalid:
            raise ValueError(
                f"Unknown RAGAS metrics: {invalid}. "
                f"Available: {self.AVAILABLE_METRICS}"
            )

        self._llm: Any = None
        self._embeddings: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of LLM judge and embeddings."""
        if self._initialized:
            return

        logger.info(
            "Initializing RAGAS evaluator with local judge on port %d",
            self.judge_config.get("server_port", _JUDGE_PORT),
        )
        self._llm = _get_judge_llm(self.judge_config)
        self._embeddings = _get_judge_embeddings()
        self._initialized = True
        logger.info("RAGAS evaluator initialized")

    def _build_metrics(self) -> list:
        """Build RAGAS metric instances for the configured metric names."""
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        return [metric_map[name] for name in self.metric_names]

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, Optional[float]]:
        """Evaluate a single QA sample with RAGAS.

        Args:
            question: The user query.
            answer: The model-generated answer.
            contexts: Retrieved context chunks.
            ground_truth: Reference answer (needed for context_recall/precision).

        Returns:
            Dict of metric_name -> score (float). Missing metrics map to None.
        """
        return self.evaluate_batch(
            [
                EvalSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
            ]
        )[0]

    def evaluate_batch(
        self, samples: list[EvalSample]
    ) -> list[dict[str, Optional[float]]]:
        """Evaluate a batch of QA samples with RAGAS.

        Args:
            samples: List of EvalSample instances.

        Returns:
            List of dicts, one per sample, with metric_name -> score.
        """
        self._ensure_initialized()

        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
        except ImportError as e:
            raise ImportError(
                f"RAGAS dependencies missing: {e}. Install with: pip install ragas datasets"
            ) from e

        # Build the HuggingFace Dataset in the format RAGAS expects
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
        dataset = Dataset.from_dict(data)

        metrics = self._build_metrics()

        logger.info("Running RAGAS evaluation on %d samples, metrics=%s",
                     len(samples), self.metric_names)

        with Timer() as t:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )

        logger.info("RAGAS evaluation completed in %.2fs", t.elapsed)

        # Extract per-sample results from the RAGAS result DataFrame
        results_df = result.to_pandas()

        batch_results = []
        for idx in range(len(samples)):
            sample_result: dict[str, Optional[float]] = {}
            for metric_name in self.metric_names:
                if metric_name in results_df.columns:
                    val = results_df.iloc[idx][metric_name]
                    # Handle NaN
                    if val is not None and val == val:  # NaN != NaN
                        sample_result[metric_name] = float(val)
                    else:
                        sample_result[metric_name] = None
                else:
                    sample_result[metric_name] = None
            batch_results.append(sample_result)

        return batch_results


# ---------------------------------------------------------------------------
# RAGChecker Evaluator
# ---------------------------------------------------------------------------


class RAGCheckerEvaluator:
    """Wrapper around the RAGChecker evaluation framework.

    Computes the core thesis metrics: context_utilization (CU),
    self_knowledge (SK), noise_sensitivity (NS).

    These metrics measure how faithfully a model uses retrieved context
    vs relying on parametric knowledge -- the central question of this project.

    Uses a local llama.cpp judge server via OpenAI-compatible API on port 8081.

    Args:
        judge_config: Judge config dict from eval_config.yaml.

    Example::

        judge_config = _load_judge_config()
        evaluator = RAGCheckerEvaluator(judge_config=judge_config)
        result = evaluator.evaluate(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["France is a country in Europe. Its capital is Paris."],
            ground_truth="Paris",
        )
        print(result)  # {"context_utilization": 0.85, "self_knowledge": 0.1, ...}
    """

    AVAILABLE_METRICS = {
        "context_utilization",
        "self_knowledge",
        "noise_sensitivity",
    }

    def __init__(self, judge_config: dict[str, Any]) -> None:
        self.judge_config = judge_config
        self._initialized = False
        self._checker: Any = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of RAGChecker with local llama.cpp endpoints."""
        if self._initialized:
            return

        port = self.judge_config.get("server_port", _JUDGE_PORT)
        model = self.judge_config["model"]
        api_base = f"http://localhost:{port}/v1"

        logger.info(
            "Initializing RAGChecker evaluator with local judge: "
            "model=%s api_base=%s",
            model, api_base,
        )

        try:
            from ragchecker import RAGResults, RAGChecker
            from ragchecker.metrics import all_metrics
        except ImportError:
            raise ImportError(
                "RAGChecker not installed. Install with: "
                "pip install ragchecker"
            )

        # Use openai/ prefix with api_base override for local llama.cpp
        extractor_name = f"openai/{model}"
        checker_name = f"openai/{model}"

        self._checker = RAGChecker(
            extractor_name=extractor_name,
            checker_name=checker_name,
            extractor_api_base=api_base,
            checker_api_base=api_base,
            batch_size_extractor=8,
            batch_size_checker=8,
        )

        self._initialized = True
        logger.info("RAGChecker evaluator initialized")

    def _build_rag_results(self, samples: list[EvalSample]) -> Any:
        """Build RAGChecker RAGResults object from samples.

        RAGChecker expects a specific JSON format with queries, ground truths,
        retrieved contexts, and generated responses.
        """
        from ragchecker import RAGResults

        results_data = []
        for i, sample in enumerate(samples):
            entry = {
                "query_id": str(i),
                "query": sample.question,
                "gt_answer": [sample.ground_truth] if sample.ground_truth else [],
                "response": sample.answer,
                "retrieved_context": [
                    {
                        "doc_id": f"doc_{i}_{j}",
                        "text": ctx,
                    }
                    for j, ctx in enumerate(sample.contexts)
                ],
            }
            results_data.append(entry)

        return RAGResults.from_dict(
            {
                "results": results_data,
            }
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, Optional[float]]:
        """Evaluate a single QA sample with RAGChecker.

        Args:
            question: The user query.
            answer: The model-generated answer.
            contexts: Retrieved context chunks.
            ground_truth: Reference answer.

        Returns:
            Dict with context_utilization, self_knowledge, noise_sensitivity.
        """
        return self.evaluate_batch(
            [
                EvalSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
            ]
        )[0]

    def evaluate_batch(
        self, samples: list[EvalSample]
    ) -> list[dict[str, Optional[float]]]:
        """Evaluate a batch of QA samples with RAGChecker.

        Args:
            samples: List of EvalSample instances.

        Returns:
            List of dicts with RAGChecker metrics per sample.
        """
        self._ensure_initialized()

        try:
            from ragchecker.metrics import all_metrics
        except ImportError:
            raise ImportError("RAGChecker metrics import failed")

        logger.info("Running RAGChecker evaluation on %d samples", len(samples))

        rag_results = self._build_rag_results(samples)

        with Timer() as t:
            self._checker.evaluate(rag_results, all_metrics)

        logger.info("RAGChecker evaluation completed in %.2fs", t.elapsed)

        # Extract per-sample metrics from RAGChecker results
        batch_results = []
        for i, sample_result in enumerate(rag_results.results):
            metrics: dict[str, Optional[float]] = {}

            # RAGChecker stores metrics on the result objects
            result_metrics = getattr(sample_result, "metrics", {})
            if isinstance(result_metrics, dict):
                # Map RAGChecker metric names to our spec names
                # RAGChecker uses: context_utilization, self_knowledge, noise_sensitivity_in_relevant
                metrics["context_utilization"] = _safe_float(
                    result_metrics.get("context_utilization")
                )
                metrics["self_knowledge"] = _safe_float(
                    result_metrics.get("self_knowledge")
                )
                metrics["noise_sensitivity"] = _safe_float(
                    result_metrics.get(
                        "noise_sensitivity_in_relevant",
                        result_metrics.get("noise_sensitivity"),
                    )
                )
            else:
                metrics["context_utilization"] = None
                metrics["self_knowledge"] = None
                metrics["noise_sensitivity"] = None

            batch_results.append(metrics)

        return batch_results


# ---------------------------------------------------------------------------
# Combined Evaluator
# ---------------------------------------------------------------------------


class CombinedEvaluator:
    """Runs both RAGAS and RAGChecker, returns unified metrics.

    Graceful degradation: if one framework fails, the other's results
    are still returned with error information for the failed framework.

    Manages the full judge server lifecycle: stop gen server -> start
    judge server on port 8081 -> score with both frameworks -> stop judge.

    Args:
        judge_config: Judge config dict from eval_config.yaml.
        ragas_metrics: Optional list of RAGAS metric names.
        enable_ragas: Whether to run RAGAS (default True).
        enable_ragchecker: Whether to run RAGChecker (default True).
        eval_model_path: Path to generation model GGUF (for self-judge detection).
        gen_server: Optional LlamaServer for generation (stopped before scoring).

    Example::

        judge_config = _load_judge_config()
        evaluator = CombinedEvaluator(judge_config=judge_config)
        results = evaluator.evaluate_batch(samples)
        for r in results:
            print(r.to_spec_metrics())
    """

    def __init__(
        self,
        judge_config: dict[str, Any],
        ragas_metrics: Optional[list[str]] = None,
        enable_ragas: bool = True,
        enable_ragchecker: bool = True,
        eval_model_path: Optional[str] = None,
        gen_server: Optional[Any] = None,
    ) -> None:
        self.judge_config = judge_config
        self.enable_ragas = enable_ragas
        self.enable_ragchecker = enable_ragchecker
        self.eval_model_path = eval_model_path
        self.gen_server = gen_server

        self._ragas: Optional[RAGASEvaluator] = None
        self._ragchecker: Optional[RAGCheckerEvaluator] = None

        if enable_ragas:
            self._ragas = RAGASEvaluator(
                judge_config=judge_config,
                metrics=ragas_metrics,
            )
        if enable_ragchecker:
            self._ragchecker = RAGCheckerEvaluator(judge_config=judge_config)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalResult:
        """Evaluate a single sample with both frameworks.

        Args:
            question: The user query.
            answer: The model-generated answer.
            contexts: Retrieved context chunks.
            ground_truth: Reference answer.

        Returns:
            EvalResult with metrics from both frameworks.
        """
        return self.evaluate_batch(
            [
                EvalSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
            ]
        )[0]

    def evaluate_batch(self, samples: list[EvalSample]) -> list[EvalResult]:
        """Evaluate a batch of samples with both frameworks.

        Manages the judge server lifecycle:
        1. Stop generation server (port 8080) to free VRAM
        2. Start judge server (port 8081) with judge GGUF
        3. Run RAGAS and RAGChecker scoring
        4. Stop judge server

        Each framework runs independently. If one fails, the other's results
        are still returned. Errors are logged and stored in EvalResult.

        Args:
            samples: List of EvalSample instances.

        Returns:
            List of EvalResult instances with unified metrics.
        """
        n = len(samples)
        logger.info("Combined evaluation starting on %d samples", n)

        # Initialize per-sample results
        results = [EvalResult() for _ in range(n)]

        # --- Judge server lifecycle ---
        judge_server = None
        try:
            # Step 1: Stop generation server to free VRAM
            stop_generation_server(self.gen_server)

            # Step 2: Start judge server on port 8081
            judge_server = start_judge_server(
                self.judge_config,
                eval_model_path=self.eval_model_path,
            )

            # Step 3: Run scoring

            # --- RAGAS ---
            ragas_results: Optional[list[dict]] = None
            if self._ragas is not None:
                try:
                    with Timer() as t:
                        ragas_results = self._ragas.evaluate_batch(samples)
                    logger.info("RAGAS batch completed in %.2fs", t.elapsed)
                except Exception as e:
                    error_msg = f"RAGAS failed: {type(e).__name__}: {e}"
                    logger.error(error_msg)
                    logger.debug("RAGAS traceback:\n%s", traceback.format_exc())
                    for r in results:
                        r.ragas_error = error_msg

            # --- RAGChecker ---
            ragchecker_results: Optional[list[dict]] = None
            if self._ragchecker is not None:
                try:
                    with Timer() as t:
                        ragchecker_results = self._ragchecker.evaluate_batch(samples)
                    logger.info("RAGChecker batch completed in %.2fs", t.elapsed)
                except Exception as e:
                    error_msg = f"RAGChecker failed: {type(e).__name__}: {e}"
                    logger.error(error_msg)
                    logger.debug("RAGChecker traceback:\n%s", traceback.format_exc())
                    for r in results:
                        r.ragchecker_error = error_msg

        finally:
            # Step 4: Always stop the judge server
            if judge_server is not None:
                stop_judge_server(judge_server)

        # --- Merge results ---
        for i in range(n):
            if ragas_results is not None and i < len(ragas_results):
                r = ragas_results[i]
                results[i].faithfulness = r.get("faithfulness")
                results[i].answer_relevancy = r.get("answer_relevancy")
                results[i].context_precision = r.get("context_precision")
                results[i].context_recall = r.get("context_recall")

            if ragchecker_results is not None and i < len(ragchecker_results):
                r = ragchecker_results[i]
                results[i].context_utilization = r.get("context_utilization")
                results[i].self_knowledge = r.get("self_knowledge")
                results[i].noise_sensitivity = r.get("noise_sensitivity")

            # Compute negative_rejection_rate from the answer if applicable
            results[i].negative_rejection_rate = _compute_negative_rejection(
                samples[i], results[i]
            )

        logger.info("Combined evaluation complete for %d samples", n)
        return results


# ---------------------------------------------------------------------------
# Negative rejection rate computation
# ---------------------------------------------------------------------------

_REJECTION_PHRASES = [
    "i cannot answer",
    "cannot answer this question",
    "based on the provided context",
    "not enough information",
    "the context does not contain",
    "i don't have enough information",
    "the provided context doesn't",
    "the provided context does not",
    "no relevant information",
    "unable to answer",
]


def _compute_negative_rejection(sample: EvalSample, result: EvalResult) -> Optional[float]:
    """Compute whether the model correctly rejected an unanswerable question.

    Returns 1.0 if the answer contains a rejection phrase (model refused),
    0.0 if it attempted to answer. Returns None if ground_truth doesn't
    indicate an unanswerable question (can't compute this metric).

    This is a per-sample binary indicator. The actual negative_rejection_rate
    is computed as the mean over all unanswerable samples in the batch during
    aggregation.
    """
    # Check if this is an unanswerable question
    gt_lower = sample.ground_truth.lower().strip()
    is_unanswerable = gt_lower in (
        "",
        "unanswerable",
        "no answer",
        "n/a",
        "none",
        "insufficient context",
    )

    if not is_unanswerable:
        return None

    # Check if the model's answer contains a rejection
    answer_lower = sample.answer.lower()
    for phrase in _REJECTION_PHRASES:
        if phrase in answer_lower:
            return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None for None/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def samples_from_jsonl(filepath: Union[str, Path]) -> list[EvalSample]:
    """Load EvalSample instances from a JSONL file.

    Expected JSONL format (one JSON object per line)::

        {
            "question": "...",
            "answer": "...",
            "contexts": ["...", "..."],
            "ground_truth": "..."   // optional
        }

    Also accepts the output format from the runner, where the question
    may be under 'query' and contexts under 'retrieved_contexts'.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of EvalSample instances.
    """
    samples = []
    for record in read_jsonl(filepath):
        question = record.get("question", record.get("query", ""))
        answer = record.get("answer", record.get("response", ""))
        contexts = record.get("contexts", record.get("retrieved_contexts", []))
        ground_truth = record.get("ground_truth", record.get("gt_answer", ""))

        if not question or not answer:
            logger.warning("Skipping record with missing question or answer: %s",
                           json.dumps(record, ensure_ascii=False)[:200])
            continue

        # Ensure contexts is a list of strings
        if isinstance(contexts, str):
            contexts = [contexts]

        # Handle ground_truth that might be a list
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else ""

        # Preserve all other fields as metadata
        meta_keys = set(record.keys()) - {
            "question", "query", "answer", "response",
            "contexts", "retrieved_contexts", "ground_truth", "gt_answer",
        }
        metadata = {k: record[k] for k in meta_keys}

        samples.append(
            EvalSample(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                metadata=metadata,
            )
        )

    logger.info("Loaded %d samples from %s", len(samples), filepath)
    return samples


# ---------------------------------------------------------------------------
# Batch evaluation with per-sample error handling
# ---------------------------------------------------------------------------


def evaluate_batch_safe(
    evaluator: CombinedEvaluator,
    samples: list[EvalSample],
    batch_size: int = 10,
) -> list[EvalResult]:
    """Evaluate samples in batches with per-sample error handling.

    If a batch fails, falls back to evaluating each sample individually.
    Individual sample failures are logged and return empty EvalResult with
    error info.

    Args:
        evaluator: CombinedEvaluator instance.
        samples: List of EvalSample instances.
        batch_size: Number of samples per batch.

    Returns:
        List of EvalResult instances (same length as samples).
    """
    all_results: list[EvalResult] = []
    total = len(samples)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = samples[batch_start:batch_end]
        batch_idx = f"[{batch_start+1}-{batch_end}/{total}]"

        logger.info("Evaluating batch %s", batch_idx)

        try:
            batch_results = evaluator.evaluate_batch(batch)
            all_results.extend(batch_results)
            logger.info("Batch %s succeeded", batch_idx)
        except Exception as e:
            logger.error(
                "Batch %s failed (%s), falling back to per-sample evaluation",
                batch_idx, e,
            )
            # Fall back to individual evaluation
            for i, sample in enumerate(batch):
                sample_idx = batch_start + i + 1
                try:
                    result = evaluator.evaluate(
                        question=sample.question,
                        answer=sample.answer,
                        contexts=sample.contexts,
                        ground_truth=sample.ground_truth,
                    )
                    all_results.append(result)
                except Exception as sample_e:
                    logger.error(
                        "Sample %d/%d failed: %s: %s",
                        sample_idx, total,
                        type(sample_e).__name__, sample_e,
                    )
                    error_result = EvalResult(
                        ragas_error=f"Sample error: {sample_e}",
                        ragchecker_error=f"Sample error: {sample_e}",
                    )
                    all_results.append(error_result)

    return all_results


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for evaluating a JSONL file.

    Usage::

        python -m cumrag.evaluator \\
            --input results/raw/run.jsonl \\
            --output results/raw/scored.jsonl \\
            --batch-size 10
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CUMRAG Evaluator -- score RAG outputs with RAGAS + RAGChecker",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file with question/answer/contexts records",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file with evaluation metrics appended",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Path to generation model GGUF (for self-judge quant swap detection)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for evaluation (default 10)",
    )
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="Only run RAGAS evaluation",
    )
    parser.add_argument(
        "--ragchecker-only",
        action="store_true",
        help="Only run RAGChecker evaluation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    from cumrag.utils import setup_logging
    setup_logging(level=getattr(logging, args.log_level))

    # Load judge config from eval_config.yaml
    try:
        judge_config = _load_judge_config()
    except (FileNotFoundError, KeyError) as e:
        logger.error("Failed to load judge config: %s", e)
        sys.exit(1)

    logger.info(
        "Using local judge: model=%s quant=%s port=%d",
        judge_config["model"],
        judge_config["quant"],
        judge_config.get("server_port", _JUDGE_PORT),
    )

    # Load samples
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    samples = samples_from_jsonl(input_path)
    if not samples:
        logger.error("No valid samples found in %s", input_path)
        sys.exit(1)

    # Configure evaluator
    enable_ragas = not args.ragchecker_only
    enable_ragchecker = not args.ragas_only

    evaluator = CombinedEvaluator(
        judge_config=judge_config,
        enable_ragas=enable_ragas,
        enable_ragchecker=enable_ragchecker,
        eval_model_path=args.eval_model,
    )

    # Run evaluation
    results = evaluate_batch_safe(
        evaluator=evaluator,
        samples=samples,
        batch_size=args.batch_size,
    )

    # Write output -- merge original records with eval metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-read input to preserve original fields
    input_records = list(read_jsonl(input_path))

    written = 0
    errors = 0
    for record, sample, result in zip(input_records, samples, results):
        # Merge evaluation metrics into the original record
        record["eval_metrics"] = result.to_dict()
        record["eval_spec_metrics"] = result.to_spec_metrics()
        record["eval_judge_model"] = judge_config["model"]
        record["eval_judge_quant"] = judge_config["quant"]

        if result.ragas_error or result.ragchecker_error:
            errors += 1

        append_jsonl(output_path, record)
        written += 1

    logger.info(
        "Evaluation complete: %d samples scored, %d with errors. Output: %s",
        written, errors, output_path,
    )

    # Print summary to stdout
    print(f"\nEvaluation Summary")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Samples: {written}")
    print(f"  Errors: {errors}")
    print(f"  Judge: {judge_config['model']} ({judge_config['quant']})")
    print(f"  Judge port: {judge_config.get('server_port', _JUDGE_PORT)}")
    print(f"  RAGAS: {'enabled' if enable_ragas else 'disabled'}")
    print(f"  RAGChecker: {'enabled' if enable_ragchecker else 'disabled'}")


if __name__ == "__main__":
    main()
