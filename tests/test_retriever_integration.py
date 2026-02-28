#!/usr/bin/env python3
"""Integration tests for the Retriever against real ChromaDB indexes.

These tests hit actual persisted ChromaDB collections (not mocks).
They require the index/ directory to be populated with all 6 collections:
  - 4 RGB subsets (noise_robustness, negative_rejection, information_integration, counterfactual_robustness)
  - 1 HaluEval collection
  - 1 NQ stub collection

Run:
    python -m pytest tests/test_retriever_integration.py -v
    python -m pytest tests/test_retriever_integration.py -v -m integration

Skip in fast CI:
    python -m pytest -m "not integration"
"""

import pytest

from cumrag.retriever import Retriever, resolve_collection_name, validate_index
from cumrag.utils import load_config

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
def rgb_noise_retriever(config) -> Retriever:
    """Retriever for the RGB noise_robustness collection."""
    name = resolve_collection_name("rgb_noise_robustness", config)
    return Retriever(
        index_dir="index",
        dataset="rgb",
        collection_name=name,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRetrieverRGBNoiseRobustness:
    """Test retrieval from the RGB noise_robustness collection."""

    def test_retrieve_rgb_noise_robustness(self, rgb_noise_retriever):
        """Query RGB noise_robustness collection, verify result structure."""
        results = rgb_noise_retriever.retrieve(SAMPLE_QUERY, top_k=5)

        assert len(results) > 0, "Expected at least one result"

        for r in results:
            # Each result must have text, metadata, distance
            assert "text" in r, "Result missing 'text' field"
            assert "metadata" in r, "Result missing 'metadata' field"
            assert "distance" in r, "Result missing 'distance' field"

            assert isinstance(r["text"], str)
            assert len(r["text"]) > 0, "Result text should be non-empty"
            assert isinstance(r["distance"], float)

            meta = r["metadata"]
            assert "source" in meta, "Metadata missing 'source' (doc_id)"
            assert "dataset" in meta, "Metadata missing 'dataset'"


@pytest.mark.integration
@pytest.mark.parametrize("subset_key", RGB_SUBSETS)
def test_retrieve_rgb_all_subsets(config, subset_key):
    """Parametrize across all 4 RGB subsets, verify each returns results."""
    collection_name = resolve_collection_name(subset_key, config)
    retriever = Retriever(
        index_dir="index",
        dataset="rgb",
        collection_name=collection_name,
    )

    results = retriever.retrieve(SAMPLE_QUERY, top_k=3)
    assert len(results) > 0, f"No results from {subset_key}"

    # Verify all results have the expected structure
    for r in results:
        assert "text" in r
        assert "metadata" in r
        assert "distance" in r


@pytest.mark.integration
def test_retrieve_halueval(config):
    """Query HaluEval collection, verify knowledge text is retrievable."""
    collection_name = resolve_collection_name("halueval", config)
    retriever = Retriever(
        index_dir="index",
        dataset="halueval",
        collection_name=collection_name,
    )

    results = retriever.retrieve("Tell me about artificial intelligence", top_k=5)
    assert len(results) > 0, "HaluEval collection returned no results"

    for r in results:
        assert isinstance(r["text"], str)
        assert len(r["text"]) > 0
        assert isinstance(r["distance"], float)
        assert r["distance"] >= 0.0


@pytest.mark.integration
def test_retrieve_nq_stub(config):
    """Query NQ stub collection, verify results returned."""
    collection_name = resolve_collection_name("nq_wiki", config)
    retriever = Retriever(
        index_dir="index",
        dataset="nq",
        collection_name=collection_name,
    )

    results = retriever.retrieve("Who was the first president of the United States?", top_k=5)
    assert len(results) > 0, "NQ stub collection returned no results"

    for r in results:
        assert "text" in r
        assert "metadata" in r
        assert "distance" in r
        assert isinstance(r["text"], str)


@pytest.mark.integration
@pytest.mark.parametrize("collection_key", ALL_COLLECTION_KEYS)
def test_index_validation_passes(config, collection_key):
    """Call validate_index() for all collections, verify no ValueError raised."""
    import chromadb

    collection_name = resolve_collection_name(collection_key, config)
    client = chromadb.PersistentClient(path="index")
    collection = client.get_collection(collection_name)

    # Should NOT raise ValueError
    validate_index(collection, config)


@pytest.mark.integration
def test_format_context_real_data(rgb_noise_retriever):
    """Retrieve real chunks, pass to format_context(), verify output."""
    results = rgb_noise_retriever.retrieve(SAMPLE_QUERY, top_k=3)
    assert len(results) > 0

    context = rgb_noise_retriever.format_context(results)

    assert isinstance(context, str)
    assert len(context) > 0, "format_context() returned empty string"
    # Context should contain text from at least one result
    assert any(
        r["text"].strip()[:50] in context for r in results if r["text"].strip()
    ), "format_context() output does not contain retrieved text"


@pytest.mark.integration
@pytest.mark.parametrize("top_k", [1, 3])
def test_retrieve_top_k(rgb_noise_retriever, top_k):
    """Verify requesting top_k=N returns exactly N results."""
    results = rgb_noise_retriever.retrieve(SAMPLE_QUERY, top_k=top_k)
    assert len(results) == top_k, (
        f"Expected {top_k} results, got {len(results)}"
    )


@pytest.mark.integration
def test_retrieve_result_ordering(rgb_noise_retriever):
    """Verify results are ordered by distance (ascending)."""
    results = rgb_noise_retriever.retrieve(SAMPLE_QUERY, top_k=5)
    assert len(results) >= 2, "Need at least 2 results to check ordering"

    distances = [r["distance"] for r in results]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1], (
            f"Results not sorted by distance: {distances[i]:.6f} > {distances[i + 1]:.6f} "
            f"at positions {i} and {i + 1}"
        )


@pytest.mark.integration
def test_retrieve_deterministic(rgb_noise_retriever):
    """Same query returns same results (deterministic retrieval)."""
    query = "What is quantum computing?"
    results_1 = rgb_noise_retriever.retrieve(query, top_k=3)
    results_2 = rgb_noise_retriever.retrieve(query, top_k=3)

    assert len(results_1) == len(results_2)

    for r1, r2 in zip(results_1, results_2):
        assert r1["text"] == r2["text"], "Non-deterministic: texts differ across runs"
        assert r1["distance"] == r2["distance"], "Non-deterministic: distances differ"
        assert r1["metadata"] == r2["metadata"], "Non-deterministic: metadata differs"
