#!/usr/bin/env python3
"""Integration tests for the EvalRunner — dry-run with real ChromaDB retrieval.

These tests exercise the runner's pre-flight checks and retrieval path using
real ChromaDB indexes. The llama.cpp generation server is mocked (no GPU model
serving on x99), but everything else is real:
  - Config loading and collection name resolution
  - JSONL dataset loading and normalization
  - ChromaDB retrieval with real embedded data
  - Prompt template formatting with real retrieved chunks
  - v2 output schema structure validation
  - Error handling when the server is unreachable

Run:
    python -m pytest tests/test_runner_integration.py -v
    python -m pytest tests/test_runner_integration.py -v -m integration

Skip in fast CI:
    python -m pytest -m "not integration"
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comprag.generator import LlamaServer, format_prompt, load_prompt_template
from comprag.retriever import Retriever, resolve_collection_name
from comprag.runner import EvalRunner, load_dataset_queries
from comprag.utils import load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG = None


def _get_config() -> dict:
    """Cache config across tests (file I/O once)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config("eval_config")
    return _CONFIG


RGB_SUBSETS = [
    "rgb_noise_robustness",
    "rgb_negative_rejection",
    "rgb_information_integration",
    "rgb_counterfactual_robustness",
]

ALL_COLLECTION_KEYS = RGB_SUBSETS + ["halueval", "nq_wiki"]

SAMPLE_QUERY = "What is the capital of France?"


@pytest.fixture(scope="module")
def config() -> dict:
    return _get_config()


@pytest.fixture(scope="module")
def runner() -> EvalRunner:
    """Create an EvalRunner with real config. No server started."""
    return EvalRunner()


@pytest.fixture(scope="module")
def prompt_template() -> str:
    """Load the real prompt template."""
    return load_prompt_template()


# ---------------------------------------------------------------------------
# Test 1: Runner loads config and resolves collection names
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerLoadsConfig:
    """EvalRunner loads eval_config.yaml and resolves collection names."""

    def test_runner_has_config(self, runner):
        """EvalRunner.config is populated after init."""
        assert runner.config is not None
        assert isinstance(runner.config, dict)
        assert "retrieval" in runner.config
        assert "generation" in runner.config
        assert "datasets" in runner.config

    def test_runner_retrieval_config(self, runner):
        """Runner reads retrieval parameters from config."""
        assert runner._top_k == 5
        assert runner._index_dir is not None

    def test_runner_resolves_collection_names(self, config):
        """resolve_collection_name works for all configured collections."""
        for key in ALL_COLLECTION_KEYS:
            name = resolve_collection_name(key, config)
            assert isinstance(name, str)
            assert len(name) > 0
            assert name.startswith("comprag_"), (
                f"Collection name '{name}' for key '{key}' does not start with 'comprag_'"
            )

    def test_runner_hardware_meta(self, runner):
        """Runner captures hardware metadata on init."""
        assert runner.hardware_meta is not None
        assert isinstance(runner.hardware_meta, dict)


# ---------------------------------------------------------------------------
# Test 2: Runner loads normalized JSONL samples
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerLoadsSamples:
    """Runner loads normalized JSONL for each dataset/subset."""

    def test_load_rgb_noise_robustness(self):
        """Load RGB noise_robustness queries."""
        queries = load_dataset_queries(
            dataset="rgb", eval_subset="noise_robustness", num_queries=5
        )
        assert len(queries) > 0
        assert len(queries) <= 5
        for q in queries:
            assert "question" in q
            assert "ground_truth" in q
            assert isinstance(q["question"], str)
            assert len(q["question"]) > 0

    def test_load_nq_test(self):
        """Load NQ test queries."""
        queries = load_dataset_queries(
            dataset="nq", eval_subset="test", num_queries=5
        )
        assert len(queries) > 0
        for q in queries:
            assert "question" in q
            assert "ground_truth" in q

    def test_load_halueval(self):
        """Load HaluEval queries."""
        queries = load_dataset_queries(dataset="halueval", num_queries=5)
        assert len(queries) > 0
        for q in queries:
            assert "question" in q
            assert "ground_truth" in q

    def test_load_all_rgb_subsets(self):
        """Verify all four RGB subsets produce queries."""
        for subset_name in ["noise_robustness", "negative_rejection",
                            "information_integration", "counterfactual_robustness"]:
            queries = load_dataset_queries(
                dataset="rgb", eval_subset=subset_name, num_queries=2
            )
            assert len(queries) > 0, f"No queries for RGB subset '{subset_name}'"

    def test_num_queries_limit(self):
        """num_queries parameter limits the returned query count."""
        queries_all = load_dataset_queries(dataset="rgb", num_queries=None)
        queries_limited = load_dataset_queries(dataset="rgb", num_queries=3)
        assert len(queries_limited) == 3
        assert len(queries_all) >= len(queries_limited)


# ---------------------------------------------------------------------------
# Test 3: Runner retrieves context from real ChromaDB
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerRetrievesContext:
    """Runner retrieves context from real ChromaDB for a sample query."""

    def test_get_retriever_rgb(self, runner, config):
        """Runner creates a working Retriever for RGB noise_robustness."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        assert retriever is not None
        assert isinstance(retriever, Retriever)

    def test_retriever_returns_chunks(self, runner):
        """Retriever returns real chunks with text and metadata."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=5)

        assert len(chunks) > 0, "Expected at least one chunk"
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "distance" in chunk
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_retriever_caching(self, runner):
        """Runner caches retrievers — same object returned on second call."""
        r1 = runner._get_retriever("rgb", eval_subset="noise_robustness")
        r2 = runner._get_retriever("rgb", eval_subset="noise_robustness")
        assert r1 is r2, "Runner should cache Retriever instances"


# ---------------------------------------------------------------------------
# Test 4: Runner formats prompt with real retrieved chunks
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerFormatsPrompt:
    """Runner formats prompt with real retrieved chunks, verifies template structure."""

    def test_format_prompt_with_real_chunks(self, runner, prompt_template):
        """Format the prompt template with real retrieved chunks."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=3)
        assert len(chunks) > 0

        prompt = format_prompt(prompt_template, SAMPLE_QUERY, chunks)

        # Prompt should contain the query
        assert SAMPLE_QUERY in prompt

        # Prompt should contain chunk text
        for chunk in chunks:
            text_fragment = chunk["text"][:50]
            assert text_fragment in prompt, (
                f"Prompt missing chunk text fragment: '{text_fragment}'"
            )

        # Prompt should have template markers
        assert "<|system|>" in prompt
        assert "<|context|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt

    def test_format_prompt_numbered_chunks(self, runner, prompt_template):
        """Each chunk is numbered [1], [2], etc. in the formatted prompt."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=3)
        assert len(chunks) >= 2

        prompt = format_prompt(prompt_template, SAMPLE_QUERY, chunks)

        for i in range(1, len(chunks) + 1):
            assert f"[{i}]" in prompt, (
                f"Prompt missing chunk numbering [{i}]"
            )

    def test_format_prompt_no_raw_placeholders(self, runner, prompt_template):
        """After formatting, no raw {retrieved_chunks} or {query} remain."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=3)

        prompt = format_prompt(prompt_template, SAMPLE_QUERY, chunks)

        assert "{retrieved_chunks}" not in prompt, "Raw placeholder not replaced"
        assert "{query}" not in prompt, "Raw placeholder not replaced"


# ---------------------------------------------------------------------------
# Test 5: Context window safety check
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerContextWindowCheck:
    """Context window safety check works with real prompt lengths."""

    def test_context_window_no_crash_on_long_prompt(self, runner, prompt_template):
        """Even with many chunks, the context window check does not crash."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=5)

        prompt = format_prompt(prompt_template, SAMPLE_QUERY, chunks)

        # Simulate the context window check from run_single
        estimated_tokens = int(len(prompt.split()) * 1.4)
        model_ctx = runner._get_model_context_length("unknown-model")

        # Should not raise, just compute values
        assert isinstance(estimated_tokens, int)
        assert estimated_tokens > 0
        assert model_ctx > 0, "Default context length should be > 0"

    def test_estimated_tokens_reasonable(self, runner, prompt_template):
        """Estimated token count is in a reasonable range for top_k=5."""
        retriever = runner._get_retriever("rgb", eval_subset="noise_robustness")
        chunks = retriever.retrieve(SAMPLE_QUERY, top_k=5)

        prompt = format_prompt(prompt_template, SAMPLE_QUERY, chunks)
        estimated_tokens = int(len(prompt.split()) * 1.4)

        # With 5 chunks of real data, prompt should be non-trivial but not huge
        assert estimated_tokens > 50, (
            f"Estimated tokens ({estimated_tokens}) too low — prompt likely empty"
        )
        assert estimated_tokens < 50000, (
            f"Estimated tokens ({estimated_tokens}) unreasonably high"
        )


# ---------------------------------------------------------------------------
# Test 6: Runner handles missing collection gracefully
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerHandlesMissingCollection:
    """Runner fails gracefully when collection does not exist."""

    def test_missing_collection_raises(self):
        """Retriever raises ValueError for a non-existent collection."""
        with pytest.raises(ValueError, match="not found"):
            Retriever(
                index_dir="index",
                dataset="rgb",
                collection_name="comprag_nonexistent_collection_xyz_99999",
            )

    def test_runner_get_retriever_missing_subset(self, runner):
        """Runner propagates ValueError for a missing subset collection."""
        # Use a dataset key that does not exist in the config collections map.
        # The runner will fall back to a default collection name that
        # does not exist, causing a ValueError.
        with pytest.raises((ValueError, KeyError)):
            # Force fresh retriever creation by using a unique fake subset
            runner._get_retriever("rgb", eval_subset="totally_fake_subset_xyz")


# ---------------------------------------------------------------------------
# Test 7: Runner pre-flight index validation passes for all indexed datasets
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("collection_key", ALL_COLLECTION_KEYS)
def test_runner_preflight_validation(config, collection_key):
    """validate_index() succeeds for all indexed dataset collections."""
    import chromadb

    from comprag.retriever import validate_index

    collection_name = resolve_collection_name(collection_key, config)
    client = chromadb.PersistentClient(path="index")
    collection = client.get_collection(collection_name)

    # Should NOT raise ValueError — index metadata matches config
    validate_index(collection, config)


# ---------------------------------------------------------------------------
# Test 8: run_single with mocked LlamaServer — v2 JSONL output schema
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerSingleMocked:
    """run_single with mocked LlamaServer produces correct v2 JSONL output."""

    def _mock_server(self) -> MagicMock:
        """Create a mock LlamaServer that returns a fake generation result."""
        mock = MagicMock(spec=LlamaServer)
        mock.generate_with_metrics.return_value = {
            "text": "The capital of France is Paris.",
            "tokens_per_second": 25.0,
            "time_to_first_token_ms": 50.0,
            "total_inference_time_ms": 200.0,
            "prompt_tokens": 150,
            "completion_tokens": 8,
        }
        return mock

    def test_run_single_v2_schema(self, runner):
        """run_single returns a dict matching the v2 output schema."""
        runner._server = self._mock_server()
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            query_id="test_q0",
            skip_eval=True,
        )

        # v2 required top-level keys
        required_keys = [
            "sample_id", "dataset", "subset", "query", "ground_truth",
            "response", "retrieved_chunks", "run_config", "perf",
            "metrics", "timestamp", "error",
        ]
        for key in required_keys:
            assert key in result, f"Missing v2 key: {key}"

        # Verify field values
        assert result["dataset"] == "rgb"
        assert result["subset"] == "noise_robustness"
        assert result["query"] == SAMPLE_QUERY
        assert result["ground_truth"] == "Paris"
        assert result["response"] == "The capital of France is Paris."
        assert result["error"] is None

        # Verify run_config block
        rc = result["run_config"]
        assert rc["model"] == "test-model"
        assert rc["quant"] == "Q4_K_M"
        assert rc["hardware"] == "cpu"
        assert rc["seed"] == 42

        # Verify perf block has expected keys
        perf = result["perf"]
        for perf_key in ["ttft_ms", "total_tokens", "tokens_per_sec",
                         "wall_clock_sec", "vram_mb", "gpu_temp"]:
            assert perf_key in perf, f"Missing perf key: {perf_key}"

        # Verify retrieved_chunks is a list of dicts with v2 structure
        assert isinstance(result["retrieved_chunks"], list)
        assert len(result["retrieved_chunks"]) > 0
        for chunk in result["retrieved_chunks"]:
            assert "doc_id" in chunk
            assert "text" in chunk
            assert "distance" in chunk
            assert "rank" in chunk
            assert isinstance(chunk["rank"], int)
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_run_single_retrieved_chunks_are_real(self, runner):
        """Retrieved chunks in run_single output come from real ChromaDB."""
        runner._server = self._mock_server()
        runner._template = load_prompt_template()

        result = runner.run_single(
            query="Who discovered penicillin?",
            ground_truth="Alexander Fleming",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        # Should have retrieved chunks with non-empty text
        chunks = result["retrieved_chunks"]
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk["text"]) > 0, "Chunk text should not be empty"
            assert chunk["distance"] >= 0.0, "Distance should be non-negative"
            assert chunk["rank"] >= 1, "Rank should be >= 1"

        # Ranks should be sequential starting from 1
        ranks = [c["rank"] for c in chunks]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_run_single_jsonl_serializable(self, runner):
        """run_single output is JSON-serializable (can write to JSONL)."""
        runner._server = self._mock_server()
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        # Must not raise
        serialized = json.dumps(result, ensure_ascii=False)
        assert len(serialized) > 0

        # Round-trip: deserialize and verify structure preserved
        deserialized = json.loads(serialized)
        assert deserialized["query"] == SAMPLE_QUERY
        assert deserialized["response"] == "The capital of France is Paris."
        assert isinstance(deserialized["retrieved_chunks"], list)


# ---------------------------------------------------------------------------
# Test 9: Error handling — connection refused (no llama.cpp server running)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunnerErrorHandling:
    """Error handling when llama.cpp server is not running."""

    def test_run_single_connection_refused(self, runner):
        """When server raises RuntimeError (not running), result has error field."""
        # Set _server to None — simulates no server started
        runner._server = None
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        # v2 schema is still produced
        assert "sample_id" in result
        assert "dataset" in result
        assert "error" in result
        assert result["error"] is not None, "Error field should be set"
        assert result["response"] is None, "Response should be None on error"

        # Perf block should have None values on error
        perf = result["perf"]
        assert perf["ttft_ms"] is None
        assert perf["tokens_per_sec"] is None

    def test_run_single_connection_error_classified(self, runner):
        """ConnectionError is classified as 'connection_refused' in error field."""
        mock_server = MagicMock(spec=LlamaServer)
        mock_server.generate_with_metrics.side_effect = ConnectionError(
            "Failed to connect to llama-server after 3 attempts"
        )
        runner._server = mock_server
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        assert result["error"] is not None
        assert "connection_refused" in result["error"], (
            f"Expected 'connection_refused' in error, got: {result['error']}"
        )
        assert result["response"] is None

    def test_error_result_has_v2_structure(self, runner):
        """Even on error, the output record has all v2 required keys."""
        runner._server = None
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        required_keys = [
            "sample_id", "dataset", "subset", "query", "ground_truth",
            "response", "retrieved_chunks", "run_config", "perf",
            "metrics", "timestamp", "error",
        ]
        for key in required_keys:
            assert key in result, f"Error result missing v2 key: {key}"

        # Metrics should have None values for all metric keys
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        for mk in ["faithfulness", "context_utilization", "self_knowledge",
                    "noise_sensitivity", "answer_relevancy", "negative_rejection_rate"]:
            assert mk in metrics, f"Error metrics missing key: {mk}"
            assert metrics[mk] is None

    def test_error_result_jsonl_serializable(self, runner):
        """Error result records are JSON-serializable."""
        runner._server = None
        runner._template = load_prompt_template()

        result = runner.run_single(
            query=SAMPLE_QUERY,
            ground_truth="Paris",
            dataset="rgb",
            eval_subset="noise_robustness",
            hardware_tier="cpu",
            model_name="test-model",
            quantization="Q4_K_M",
            seed=42,
            run_id=1,
            skip_eval=True,
        )

        serialized = json.dumps(result, ensure_ascii=False)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["error"] is not None
