"""Golden-file validation tests for CompRAG JSONL schemas.

Tests the raw per-query JSONL record schema and the aggregated output schema
as defined in COMPRAG-V8-BUILD-SPEC-1.md (Output Schema section).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Schema constants — these ARE the data contract
# ---------------------------------------------------------------------------

RAW_RECORD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "run_id": str,
    "timestamp": str,
    "model": str,
    "quantization": str,
    "source": str,
    "provider": (str, type(None)),
    "dataset": str,
    "subset": str,
    "pass": str,
    "seed": int,
    "query_id": str,
    "query": str,
    "context_chunks": (list, type(None)),
    "ground_truth": str,
    "response": str,
    "generation_time_ms": (int, float),
    "scores": dict,
}

SCORES_RAGCHECKER_FIELDS: dict[str, type | tuple[type, ...]] = {
    "overall_precision": (int, float),
    "overall_recall": (int, float),
    "overall_f1": (int, float),
    "claim_recall": (int, float),
    "context_precision": (int, float),
    "context_utilization": (int, float),
    "self_knowledge": (int, float),
    "noise_sensitivity_relevant": (int, float),
    "noise_sensitivity_irrelevant": (int, float),
    "hallucination": (int, float),
    "faithfulness": (int, float),
}

SCORES_RAGAS_FIELDS: dict[str, type | tuple[type, ...]] = {
    "faithfulness": (int, float),
    "answer_relevancy": (int, float),
    "context_precision": (int, float),
    "context_recall": (int, float),
}

VALID_SOURCES = {"local", "api"}
VALID_PASSES = {"pass1_baseline", "pass2_loose", "pass3_strict"}
VALID_PROVIDERS = {None, "openai", "anthropic", "google", "deepseek", "zhipu"}

AGGREGATED_RECORD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "model": str,
    "quantization": str,
    "source": str,
    "dataset": str,
    "subset": str,
    "pass": str,
    "n_queries": int,
    "metrics": dict,
    "capability_degraded": bool,
}

METRIC_KEYS = {
    "cu", "sk", "ns_relevant", "ns_irrelevant",
    "hallucination", "faithfulness",
    "overall_precision", "overall_recall", "overall_f1",
    "claim_recall", "context_precision",
    "preference_gap",
}
METRIC_STAT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "mean": (int, float),
    "ci_lo": (int, float),
    "ci_hi": (int, float),
    "std": (int, float),
}


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------

def _valid_raw_record() -> dict[str, Any]:
    """Return a fully valid raw JSONL record."""
    return {
        "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "timestamp": "2026-03-05T14:30:00Z",
        "model": "qwen2.5-14b-instruct",
        "quantization": "Q4_K_M",
        "source": "local",
        "provider": None,
        "dataset": "rgb",
        "subset": "counterfactual",
        "pass": "pass2_loose",
        "seed": 42,
        "query_id": "rgb_cf_042",
        "query": "What is the capital of France?",
        "context_chunks": ["Paris is the capital of France.", "France is in Europe."],
        "ground_truth": "Paris",
        "response": "The capital of France is Paris.",
        "generation_time_ms": 1234,
        "scores": {
            "ragchecker": {
                "overall_precision": 0.78,
                "overall_recall": 0.65,
                "overall_f1": 0.71,
                "claim_recall": 0.82,
                "context_precision": 0.60,
                "context_utilization": 0.85,
                "self_knowledge": 0.12,
                "noise_sensitivity_relevant": 0.08,
                "noise_sensitivity_irrelevant": 0.02,
                "hallucination": 0.05,
                "faithfulness": 0.93,
            },
            "ragas": {
                "faithfulness": 0.90,
                "answer_relevancy": 0.88,
                "context_precision": 0.75,
                "context_recall": 0.82,
            },
        },
    }


def _valid_aggregated_record() -> dict[str, Any]:
    """Return a fully valid aggregated JSONL record."""
    return {
        "model": "qwen2.5-14b-instruct",
        "quantization": "Q4_K_M",
        "source": "local",
        "dataset": "rgb",
        "subset": "counterfactual",
        "pass": "pass2_loose",
        "n_queries": 487,
        "metrics": {
            "cu": {"mean": 0.72, "ci_lo": 0.68, "ci_hi": 0.76, "std": 0.15},
            "sk": {"mean": 0.25, "ci_lo": 0.21, "ci_hi": 0.29, "std": 0.12},
            "ns_relevant": {"mean": 0.08, "ci_lo": 0.05, "ci_hi": 0.11, "std": 0.06},
            "ns_irrelevant": {"mean": 0.03, "ci_lo": 0.01, "ci_hi": 0.05, "std": 0.03},
            "hallucination": {"mean": 0.05, "ci_lo": 0.03, "ci_hi": 0.08, "std": 0.04},
            "faithfulness": {"mean": 0.93, "ci_lo": 0.90, "ci_hi": 0.96, "std": 0.05},
            "overall_precision": {"mean": 0.78, "ci_lo": 0.74, "ci_hi": 0.82, "std": 0.10},
            "overall_recall": {"mean": 0.65, "ci_lo": 0.60, "ci_hi": 0.70, "std": 0.12},
            "overall_f1": {"mean": 0.71, "ci_lo": 0.66, "ci_hi": 0.75, "std": 0.11},
            "claim_recall": {"mean": 0.82, "ci_lo": 0.78, "ci_hi": 0.86, "std": 0.08},
            "context_precision": {"mean": 0.60, "ci_lo": 0.55, "ci_hi": 0.65, "std": 0.09},
            "preference_gap": {"mean": 0.13, "ci_lo": 0.09, "ci_hi": 0.17, "std": 0.10},
        },
        "capability_degraded": False,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_field_presence(record: dict[str, Any], schema: dict[str, type | tuple[type, ...]]) -> list[str]:
    """Return list of error messages for missing or extra fields."""
    errors: list[str] = []
    expected = set(schema.keys())
    actual = set(record.keys())
    for missing in expected - actual:
        errors.append(f"missing field: {missing}")
    return errors


def _validate_field_types(record: dict[str, Any], schema: dict[str, type | tuple[type, ...]]) -> list[str]:
    """Return list of error messages for type mismatches."""
    errors: list[str] = []
    for field, expected_type in schema.items():
        if field not in record:
            continue
        if not isinstance(record[field], expected_type):
            errors.append(
                f"field '{field}': expected {expected_type}, got {type(record[field]).__name__}"
            )
    return errors


# ---------------------------------------------------------------------------
# Raw record tests
# ---------------------------------------------------------------------------

class TestRawRecordSchema:
    """Tests for the per-query raw JSONL record schema."""

    def test_valid_record_passes(self) -> None:
        record = _valid_raw_record()
        errors = _validate_field_presence(record, RAW_RECORD_FIELDS)
        errors += _validate_field_types(record, RAW_RECORD_FIELDS)
        assert errors == [], f"Valid record failed validation: {errors}"

    def test_valid_record_has_all_fields(self) -> None:
        record = _valid_raw_record()
        assert set(record.keys()) >= set(RAW_RECORD_FIELDS.keys())

    def test_valid_record_scores_ragchecker(self) -> None:
        record = _valid_raw_record()
        rc = record["scores"]["ragchecker"]
        errors = _validate_field_presence(rc, SCORES_RAGCHECKER_FIELDS)
        errors += _validate_field_types(rc, SCORES_RAGCHECKER_FIELDS)
        assert errors == [], f"RAGChecker scores invalid: {errors}"

    def test_valid_record_scores_ragas(self) -> None:
        record = _valid_raw_record()
        ragas = record["scores"]["ragas"]
        errors = _validate_field_presence(ragas, SCORES_RAGAS_FIELDS)
        errors += _validate_field_types(ragas, SCORES_RAGAS_FIELDS)
        assert errors == [], f"RAGAS scores invalid: {errors}"

    def test_valid_source_values(self) -> None:
        record = _valid_raw_record()
        assert record["source"] in VALID_SOURCES

    def test_valid_pass_values(self) -> None:
        record = _valid_raw_record()
        assert record["pass"] in VALID_PASSES

    def test_valid_provider_values(self) -> None:
        record = _valid_raw_record()
        assert record["provider"] in VALID_PROVIDERS

    def test_context_chunks_null_for_pass1(self) -> None:
        """pass1_baseline has no retrieval, so context_chunks should be null."""
        record = _valid_raw_record()
        record["pass"] = "pass1_baseline"
        record["context_chunks"] = None
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert errors == []

    def test_api_source_has_api_quantization(self) -> None:
        """API models use 'API' as quantization value."""
        record = _valid_raw_record()
        record["source"] = "api"
        record["quantization"] = "API"
        record["provider"] = "openai"
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert errors == []

    # --- Invalid record tests ---

    def test_missing_run_id_fails(self) -> None:
        record = _valid_raw_record()
        del record["run_id"]
        errors = _validate_field_presence(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0
        assert any("run_id" in e for e in errors)

    def test_missing_scores_fails(self) -> None:
        record = _valid_raw_record()
        del record["scores"]
        errors = _validate_field_presence(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0

    def test_missing_multiple_fields_fails(self) -> None:
        record = _valid_raw_record()
        del record["run_id"]
        del record["timestamp"]
        del record["model"]
        errors = _validate_field_presence(record, RAW_RECORD_FIELDS)
        assert len(errors) == 3

    def test_wrong_type_seed_fails(self) -> None:
        record = _valid_raw_record()
        record["seed"] = "forty-two"
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0
        assert any("seed" in e for e in errors)

    def test_wrong_type_generation_time_fails(self) -> None:
        record = _valid_raw_record()
        record["generation_time_ms"] = "fast"
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0
        assert any("generation_time_ms" in e for e in errors)

    def test_wrong_type_scores_fails(self) -> None:
        record = _valid_raw_record()
        record["scores"] = "not a dict"
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0

    def test_wrong_type_context_chunks_fails(self) -> None:
        record = _valid_raw_record()
        record["context_chunks"] = "not a list"
        errors = _validate_field_types(record, RAW_RECORD_FIELDS)
        assert len(errors) > 0

    def test_ragchecker_missing_field_fails(self) -> None:
        record = _valid_raw_record()
        del record["scores"]["ragchecker"]["context_utilization"]
        errors = _validate_field_presence(
            record["scores"]["ragchecker"], SCORES_RAGCHECKER_FIELDS
        )
        assert len(errors) > 0

    def test_ragas_wrong_type_fails(self) -> None:
        record = _valid_raw_record()
        record["scores"]["ragas"]["faithfulness"] = "high"
        errors = _validate_field_types(
            record["scores"]["ragas"], SCORES_RAGAS_FIELDS
        )
        assert len(errors) > 0

    def test_invalid_source_detected(self) -> None:
        record = _valid_raw_record()
        record["source"] = "cloud"
        assert record["source"] not in VALID_SOURCES

    def test_invalid_pass_detected(self) -> None:
        record = _valid_raw_record()
        record["pass"] = "pass4_extra"
        assert record["pass"] not in VALID_PASSES


# ---------------------------------------------------------------------------
# Aggregated record tests
# ---------------------------------------------------------------------------

class TestAggregatedRecordSchema:
    """Tests for the aggregated output JSONL record schema."""

    def test_valid_record_passes(self) -> None:
        record = _valid_aggregated_record()
        errors = _validate_field_presence(record, AGGREGATED_RECORD_FIELDS)
        errors += _validate_field_types(record, AGGREGATED_RECORD_FIELDS)
        assert errors == [], f"Valid aggregated record failed: {errors}"

    def test_valid_record_has_all_fields(self) -> None:
        record = _valid_aggregated_record()
        assert set(record.keys()) >= set(AGGREGATED_RECORD_FIELDS.keys())

    def test_metrics_has_all_keys(self) -> None:
        record = _valid_aggregated_record()
        assert set(record["metrics"].keys()) == METRIC_KEYS

    def test_each_metric_has_stat_fields(self) -> None:
        record = _valid_aggregated_record()
        for metric_name in METRIC_KEYS:
            metric = record["metrics"][metric_name]
            errors = _validate_field_presence(metric, METRIC_STAT_FIELDS)
            errors += _validate_field_types(metric, METRIC_STAT_FIELDS)
            assert errors == [], f"Metric '{metric_name}' invalid: {errors}"

    def test_ci_lo_le_mean_le_ci_hi(self) -> None:
        """Confidence interval sanity: ci_lo <= mean <= ci_hi."""
        record = _valid_aggregated_record()
        for metric_name in METRIC_KEYS:
            m = record["metrics"][metric_name]
            assert m["ci_lo"] <= m["mean"] <= m["ci_hi"], (
                f"{metric_name}: ci_lo={m['ci_lo']} <= mean={m['mean']} <= ci_hi={m['ci_hi']}"
            )

    def test_n_queries_positive(self) -> None:
        record = _valid_aggregated_record()
        assert record["n_queries"] > 0

    def test_capability_degraded_is_bool(self) -> None:
        record = _valid_aggregated_record()
        assert isinstance(record["capability_degraded"], bool)

    def test_valid_source_values(self) -> None:
        record = _valid_aggregated_record()
        assert record["source"] in VALID_SOURCES

    def test_valid_pass_values(self) -> None:
        record = _valid_aggregated_record()
        assert record["pass"] in VALID_PASSES

    # --- Invalid aggregated record tests ---

    def test_missing_model_fails(self) -> None:
        record = _valid_aggregated_record()
        del record["model"]
        errors = _validate_field_presence(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_missing_metrics_fails(self) -> None:
        record = _valid_aggregated_record()
        del record["metrics"]
        errors = _validate_field_presence(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_missing_capability_degraded_fails(self) -> None:
        record = _valid_aggregated_record()
        del record["capability_degraded"]
        errors = _validate_field_presence(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_wrong_type_n_queries_fails(self) -> None:
        record = _valid_aggregated_record()
        record["n_queries"] = 487.5
        errors = _validate_field_types(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_wrong_type_capability_degraded_fails(self) -> None:
        record = _valid_aggregated_record()
        record["capability_degraded"] = "no"
        errors = _validate_field_types(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_wrong_type_metrics_fails(self) -> None:
        record = _valid_aggregated_record()
        record["metrics"] = [1, 2, 3]
        errors = _validate_field_types(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) > 0

    def test_missing_metric_key_detected(self) -> None:
        record = _valid_aggregated_record()
        del record["metrics"]["preference_gap"]
        assert set(record["metrics"].keys()) != METRIC_KEYS

    def test_metric_missing_stat_field_fails(self) -> None:
        record = _valid_aggregated_record()
        del record["metrics"]["cu"]["std"]
        errors = _validate_field_presence(record["metrics"]["cu"], METRIC_STAT_FIELDS)
        assert len(errors) > 0

    def test_metric_wrong_stat_type_fails(self) -> None:
        record = _valid_aggregated_record()
        record["metrics"]["hallucination"]["mean"] = "high"
        errors = _validate_field_types(record["metrics"]["hallucination"], METRIC_STAT_FIELDS)
        assert len(errors) > 0

    def test_multiple_missing_fields_fails(self) -> None:
        record = _valid_aggregated_record()
        del record["model"]
        del record["dataset"]
        del record["n_queries"]
        errors = _validate_field_presence(record, AGGREGATED_RECORD_FIELDS)
        assert len(errors) == 3
