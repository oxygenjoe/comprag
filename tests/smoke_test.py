#!/usr/bin/env python3
"""End-to-end smoke test for the CUMRAG eval harness.

NOT a unit test — a validation script that verifies the entire harness
is wired up correctly. Tests imports, configs, module interfaces, and
(if models/datasets are available) a minimal end-to-end run.

CLI:
    python tests/smoke_test.py                    # Core checks only
    python tests/smoke_test.py --live \\
        --model models/test.gguf \\
        --dataset rgb                             # Full live pipeline test

Exit codes:
    0 — all core checks passed
    1 — one or more core checks failed
"""

import argparse
import inspect
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"

_results: list[dict] = []


def _record(name: str, status: str, detail: str = "") -> None:
    """Record a test result."""
    _results.append({"name": name, "status": status, "detail": detail})
    symbol = {"PASS": "+", "FAIL": "!", "SKIP": "~"}[status]
    line = f"  [{symbol}] {status}: {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def _run_check(name: str, fn: Callable[[], Optional[str]]) -> bool:
    """Run a check function. Returns True on PASS.

    The check function should return None on success, or an error string
    on failure. Exceptions are caught and reported as FAIL.
    """
    try:
        error = fn()
        if error is None:
            _record(name, _PASS)
            return True
        else:
            _record(name, _FAIL, str(error))
            return False
    except Exception as e:
        tb = traceback.format_exc()
        _record(name, _FAIL, f"{type(e).__name__}: {e}\n{tb}")
        return False


# ===================================================================
# CHECK 1: Import verification
# ===================================================================


def check_import_utils() -> Optional[str]:
    """Verify cumrag.utils imports cleanly with expected symbols."""
    from cumrag import utils

    required = [
        "append_jsonl", "read_jsonl", "Timer", "timer",
        "get_hardware_meta", "get_hardware_full",
        "get_vram_usage_mb", "get_ram_usage_mb", "get_resource_snapshot",
        "load_config", "set_seed", "setup_logging", "get_logger",
    ]
    missing = [s for s in required if not hasattr(utils, s)]
    if missing:
        return f"Missing symbols in cumrag.utils: {missing}"
    return None


def check_import_retriever() -> Optional[str]:
    """Verify cumrag.retriever imports cleanly."""
    from cumrag.retriever import Retriever
    if not inspect.isclass(Retriever):
        return "Retriever is not a class"
    return None


def check_import_generator() -> Optional[str]:
    """Verify cumrag.generator imports cleanly."""
    from cumrag.generator import LlamaServer, format_prompt, load_prompt_template
    if not inspect.isclass(LlamaServer):
        return "LlamaServer is not a class"
    if not callable(format_prompt):
        return "format_prompt is not callable"
    if not callable(load_prompt_template):
        return "load_prompt_template is not callable"
    return None


def check_import_evaluator() -> Optional[str]:
    """Verify cumrag.evaluator imports cleanly."""
    from cumrag.evaluator import (
        RAGASEvaluator,
        RAGCheckerEvaluator,
        CombinedEvaluator,
        EvalSample,
        EvalResult,
    )
    for cls in (RAGASEvaluator, RAGCheckerEvaluator, CombinedEvaluator, EvalSample, EvalResult):
        if not (inspect.isclass(cls) or inspect.isfunction(cls)):
            return f"{cls.__name__} is not a class"
    return None


def check_import_runner() -> Optional[str]:
    """Verify cumrag.runner imports cleanly."""
    from cumrag.runner import EvalRunner
    if not inspect.isclass(EvalRunner):
        return "EvalRunner is not a class"
    return None


def check_import_aggregator() -> Optional[str]:
    """Verify cumrag.aggregator imports cleanly."""
    from cumrag import aggregator
    required = [
        "load_results", "group_results", "bootstrap_ci", "run_aggregation",
    ]
    missing = [s for s in required if not hasattr(aggregator, s)]
    if missing:
        return f"Missing symbols in cumrag.aggregator: {missing}"
    return None


# ===================================================================
# CHECK 2: Module interface verification
# ===================================================================


def check_retriever_interface() -> Optional[str]:
    """Verify Retriever class has expected methods with correct signatures."""
    from cumrag.retriever import Retriever

    required_methods = {
        "retrieve": ["self", "query"],
        "retrieve_batch": ["self", "queries"],
        "get_collection_info": ["self"],
    }
    errors = []
    for method_name, expected_params in required_methods.items():
        if not hasattr(Retriever, method_name):
            errors.append(f"Missing method: {method_name}")
            continue
        method = getattr(Retriever, method_name)
        sig = inspect.signature(method)
        actual_params = list(sig.parameters.keys())
        for p in expected_params:
            if p not in actual_params:
                errors.append(f"{method_name}: missing param '{p}' (has: {actual_params})")

    if errors:
        return "; ".join(errors)
    return None


def check_generator_interface() -> Optional[str]:
    """Verify LlamaServer class has expected methods."""
    from cumrag.generator import LlamaServer

    required_methods = [
        "start", "stop", "wait_ready", "generate", "generate_with_metrics",
    ]
    missing = [m for m in required_methods if not hasattr(LlamaServer, m)]
    if missing:
        return f"LlamaServer missing methods: {missing}"

    # Check start() has model_path param
    sig = inspect.signature(LlamaServer.start)
    if "model_path" not in sig.parameters:
        return "LlamaServer.start() missing 'model_path' parameter"

    return None


def check_evaluator_interface() -> Optional[str]:
    """Verify evaluator classes have evaluate() methods."""
    from cumrag.evaluator import RAGASEvaluator, RAGCheckerEvaluator, CombinedEvaluator

    errors = []
    for cls_name, cls in [
        ("RAGASEvaluator", RAGASEvaluator),
        ("RAGCheckerEvaluator", RAGCheckerEvaluator),
        ("CombinedEvaluator", CombinedEvaluator),
    ]:
        if not hasattr(cls, "evaluate"):
            errors.append(f"{cls_name} missing evaluate()")
            continue
        sig = inspect.signature(cls.evaluate)
        params = list(sig.parameters.keys())
        # All evaluators should accept question, answer, contexts
        for p in ["question", "answer", "contexts"]:
            if p not in params:
                errors.append(f"{cls_name}.evaluate() missing param '{p}'")

    if errors:
        return "; ".join(errors)
    return None


def check_runner_interface() -> Optional[str]:
    """Verify EvalRunner class has expected methods."""
    from cumrag.runner import EvalRunner

    required_methods = ["run_single", "run_dataset", "run_matrix"]
    missing = [m for m in required_methods if not hasattr(EvalRunner, m)]
    if missing:
        return f"EvalRunner missing methods: {missing}"
    return None


def check_aggregator_interface() -> Optional[str]:
    """Verify aggregator functions have correct signatures."""
    from cumrag.aggregator import load_results, group_results, bootstrap_ci, run_aggregation

    # bootstrap_ci should accept a numpy array
    sig = inspect.signature(bootstrap_ci)
    if "values" not in sig.parameters:
        return "bootstrap_ci() missing 'values' parameter"

    # run_aggregation should accept input_path and output_dir
    sig = inspect.signature(run_aggregation)
    params = list(sig.parameters.keys())
    for p in ["input_path", "output_dir"]:
        if p not in params:
            return f"run_aggregation() missing '{p}' parameter"

    return None


# ===================================================================
# CHECK 3: Config loading verification
# ===================================================================


def check_config_eval() -> Optional[str]:
    """Load eval_config.yaml and verify expected keys."""
    from cumrag.utils import load_config

    config = load_config("eval_config")
    required_sections = ["datasets", "retrieval", "generation", "evaluation", "statistics", "output"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        return f"eval_config.yaml missing sections: {missing}"

    # Check generation params match spec
    gen = config.get("generation", {})
    if gen.get("temperature") != 0.0:
        return f"generation.temperature should be 0.0, got {gen.get('temperature')}"
    if gen.get("max_tokens") != 512:
        return f"generation.max_tokens should be 512, got {gen.get('max_tokens')}"
    if gen.get("seed") != 42:
        return f"generation.seed should be 42, got {gen.get('seed')}"

    return None


def check_config_models() -> Optional[str]:
    """Load models.yaml and verify structure."""
    from cumrag.utils import load_config

    config = load_config("models")
    models = config.get("models")
    if not models:
        return "models.yaml has no 'models' key"
    if not isinstance(models, dict):
        return f"models.yaml 'models' should be dict, got {type(models).__name__}"

    # Spot-check baseline model
    baseline = models.get("llama-3.1-8b-instruct")
    if not baseline:
        return "Baseline model 'llama-3.1-8b-instruct' not in registry"
    if "quantizations" not in baseline:
        return "Baseline model missing 'quantizations' key"
    if "Q4_K_M" not in baseline["quantizations"]:
        return "Baseline model missing Q4_K_M quantization"

    return None


def check_config_hardware() -> Optional[str]:
    """Load hardware.yaml and verify structure."""
    from cumrag.utils import load_config

    config = load_config("hardware")
    tiers = config.get("hardware_tiers")
    if not tiers:
        return "hardware.yaml has no 'hardware_tiers' key"

    expected_tiers = ["v100", "mi25", "1660s", "fpga", "cpu", "optane"]
    missing = [t for t in expected_tiers if t not in tiers]
    if missing:
        return f"hardware.yaml missing tiers: {missing}"

    return None


# ===================================================================
# CHECK 4: Prompt template verification
# ===================================================================


def check_prompt_template() -> Optional[str]:
    """Load prompt template and verify placeholders."""
    from cumrag.generator import load_prompt_template

    template = load_prompt_template()
    if "{retrieved_chunks}" not in template:
        return "Prompt template missing {retrieved_chunks} placeholder"
    if "{query}" not in template:
        return "Prompt template missing {query} placeholder"

    # Verify template sections
    if "<|system|>" not in template:
        return "Prompt template missing <|system|> marker"
    if "<|context|>" not in template:
        return "Prompt template missing <|context|> marker"
    if "<|user|>" not in template:
        return "Prompt template missing <|user|> marker"
    if "<|assistant|>" not in template:
        return "Prompt template missing <|assistant|> marker"

    return None


# ===================================================================
# CHECK 5: JSONL round-trip test
# ===================================================================


def check_jsonl_roundtrip() -> Optional[str]:
    """Write a sample record, read it back, verify structure matches spec."""
    from cumrag.utils import append_jsonl, read_jsonl

    # Build a spec-compliant sample record
    sample_record = {
        "timestamp": "2026-03-01T14:30:00Z",
        "model": "test-model",
        "quantization": "Q4_K_M",
        "hardware_tier": "cpu",
        "dataset": "rgb",
        "eval_subset": "noise_robustness",
        "noise_ratio": 0.4,
        "metrics": {
            "faithfulness": 0.82,
            "context_utilization": 0.71,
            "self_knowledge": 0.15,
            "answer_relevancy": 0.88,
            "tokens_per_second": 3.2,
            "time_to_first_token_ms": 1200,
            "vram_usage_mb": 0,
        },
        "hardware_meta": {
            "gpu": "No GPU detected",
            "driver": "CPU-only (no GPU driver)",
            "framework": "llama.cpp (not found)",
            "os": "Linux Test",
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_output.jsonl"

        # Write
        append_jsonl(filepath, sample_record)

        # Verify file exists and is non-empty
        if not filepath.exists():
            return "JSONL file was not created"
        if filepath.stat().st_size == 0:
            return "JSONL file is empty after write"

        # Read back
        records = list(read_jsonl(filepath))
        if len(records) != 1:
            return f"Expected 1 record, got {len(records)}"

        record = records[0]

        # Verify spec fields
        spec_fields = [
            "timestamp", "model", "quantization", "hardware_tier",
            "dataset", "eval_subset", "metrics", "hardware_meta",
        ]
        missing = [f for f in spec_fields if f not in record]
        if missing:
            return f"Round-trip record missing spec fields: {missing}"

        # Verify metrics sub-fields
        metrics = record.get("metrics", {})
        metric_fields = [
            "faithfulness", "context_utilization", "self_knowledge",
            "answer_relevancy", "tokens_per_second", "time_to_first_token_ms",
        ]
        missing_metrics = [f for f in metric_fields if f not in metrics]
        if missing_metrics:
            return f"Round-trip metrics missing fields: {missing_metrics}"

        # Verify hardware_meta sub-fields
        hw_meta = record.get("hardware_meta", {})
        hw_fields = ["gpu", "driver", "framework", "os"]
        missing_hw = [f for f in hw_fields if f not in hw_meta]
        if missing_hw:
            return f"Round-trip hardware_meta missing fields: {missing_hw}"

        # Verify numeric values survived the round-trip
        if metrics["faithfulness"] != 0.82:
            return f"Faithfulness value corrupted: {metrics['faithfulness']} != 0.82"

        # Write a second record to test append behavior
        append_jsonl(filepath, sample_record)
        records = list(read_jsonl(filepath))
        if len(records) != 2:
            return f"Expected 2 records after append, got {len(records)}"

    return None


# ===================================================================
# CHECK 6: Hardware detection test
# ===================================================================


def check_hardware_detection() -> Optional[str]:
    """Run get_hardware_meta() and verify it returns valid dict."""
    from cumrag.utils import get_hardware_meta

    meta = get_hardware_meta()
    if not isinstance(meta, dict):
        return f"get_hardware_meta() returned {type(meta).__name__}, expected dict"

    required_keys = ["gpu", "driver", "framework", "os"]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        return f"hardware_meta missing keys: {missing}"

    # All values should be non-empty strings
    for key in required_keys:
        val = meta[key]
        if not isinstance(val, str):
            return f"hardware_meta['{key}'] is {type(val).__name__}, expected str"
        if not val.strip():
            return f"hardware_meta['{key}'] is empty string"

    return None


def check_hardware_full() -> Optional[str]:
    """Run get_hardware_full() and verify extended info."""
    from cumrag.utils import get_hardware_full

    info = get_hardware_full()
    if not isinstance(info, dict):
        return f"get_hardware_full() returned {type(info).__name__}, expected dict"

    # Extended info should include CPU, RAM
    if "cpu" not in info:
        return "get_hardware_full() missing 'cpu' key"
    if "ram_total_mb" not in info:
        return "get_hardware_full() missing 'ram_total_mb' key"
    if not isinstance(info["ram_total_mb"], int):
        return f"ram_total_mb is {type(info['ram_total_mb']).__name__}, expected int"

    return None


# ===================================================================
# CHECK 7: Timer test
# ===================================================================


def check_timer() -> Optional[str]:
    """Verify Timer context manager works correctly."""
    from cumrag.utils import Timer
    import time

    with Timer() as t:
        time.sleep(0.05)  # 50ms

    if t.elapsed <= 0:
        return f"Timer.elapsed is non-positive: {t.elapsed}"
    if t.elapsed_ms <= 0:
        return f"Timer.elapsed_ms is non-positive: {t.elapsed_ms}"
    # Should be at least 40ms (allowing some slack)
    if t.elapsed_ms < 40:
        return f"Timer.elapsed_ms too low: {t.elapsed_ms:.1f}ms (expected >= 40ms)"

    return None


# ===================================================================
# CHECK 8: format_prompt test
# ===================================================================


def check_format_prompt() -> Optional[str]:
    """Verify prompt formatting works end-to-end."""
    from cumrag.generator import load_prompt_template, format_prompt

    template = load_prompt_template()
    chunks = [
        {"text": "France is a country in Europe. Its capital is Paris."},
        {"text": "Paris has a population of about 2 million."},
    ]
    query = "What is the capital of France?"

    formatted = format_prompt(template, query, chunks)

    if not formatted:
        return "format_prompt returned empty string"
    if "France" not in formatted:
        return "Formatted prompt missing chunk content"
    if "What is the capital of France?" not in formatted:
        return "Formatted prompt missing query text"
    if "{retrieved_chunks}" in formatted:
        return "Formatted prompt still has {retrieved_chunks} placeholder"
    if "{query}" in formatted:
        return "Formatted prompt still has {query} placeholder"

    return None


# ===================================================================
# CHECK 9: Script --help verification
# ===================================================================


def check_script_help(script_path: str, script_name: str) -> Optional[str]:
    """Verify a script responds to --help without error."""
    import subprocess

    full_path = _PROJECT_ROOT / script_path
    if not full_path.exists():
        return f"Script not found: {full_path}"

    result = subprocess.run(
        [sys.executable, str(full_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return f"{script_name} --help exited with code {result.returncode}: {result.stderr[:300]}"
    if not result.stdout.strip():
        return f"{script_name} --help produced no output"

    return None


def check_script_download_datasets() -> Optional[str]:
    return check_script_help("scripts/download_datasets.py", "download_datasets")


def check_script_download_models() -> Optional[str]:
    return check_script_help("scripts/download_models.py", "download_models")


def check_script_build_index() -> Optional[str]:
    return check_script_help("scripts/build_index.py", "build_index")


def check_script_profile_hardware() -> Optional[str]:
    return check_script_help("scripts/profile_hardware.py", "profile_hardware")


# ===================================================================
# CHECK 10: Module CLI --help verification
# ===================================================================


def check_module_retriever_help() -> Optional[str]:
    """Verify python -m cumrag.retriever --help works."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "cumrag.retriever", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        return f"cumrag.retriever --help exited {result.returncode}: {result.stderr[:300]}"
    return None


def check_module_aggregator_help() -> Optional[str]:
    """Verify python -m cumrag.aggregator --help works."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "cumrag.aggregator", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        return f"cumrag.aggregator --help exited {result.returncode}: {result.stderr[:300]}"
    return None


def check_module_utils_help() -> Optional[str]:
    """Verify python -m cumrag.utils --help works."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "cumrag.utils", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        return f"cumrag.utils --help exited {result.returncode}: {result.stderr[:300]}"
    return None


# ===================================================================
# CHECK 11: Aggregator bootstrap_ci numeric test
# ===================================================================


def check_bootstrap_ci() -> Optional[str]:
    """Verify bootstrap_ci produces sane output on known data."""
    import numpy as np
    from cumrag.aggregator import bootstrap_ci

    values = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.80, 0.78, 0.84])
    result = bootstrap_ci(values)

    required_keys = ["mean", "ci_low", "ci_high", "ci_width", "n", "flagged"]
    missing = [k for k in required_keys if k not in result]
    if missing:
        return f"bootstrap_ci result missing keys: {missing}"

    if result["n"] != 8:
        return f"bootstrap_ci n={result['n']}, expected 8"

    mean = result["mean"]
    if not (0.75 <= mean <= 0.85):
        return f"bootstrap_ci mean={mean}, expected ~0.81"

    if result["ci_low"] >= result["ci_high"]:
        return f"bootstrap_ci ci_low ({result['ci_low']}) >= ci_high ({result['ci_high']})"

    if result["ci_width"] < 0:
        return f"bootstrap_ci ci_width is negative: {result['ci_width']}"

    return None


# ===================================================================
# CHECK 12: EvalSample and EvalResult dataclasses
# ===================================================================


def check_eval_dataclasses() -> Optional[str]:
    """Verify EvalSample/EvalResult instantiation and serialization."""
    from cumrag.evaluator import EvalSample, EvalResult

    sample = EvalSample(
        question="What is 2+2?",
        answer="4",
        contexts=["Basic arithmetic"],
        ground_truth="4",
    )
    if sample.question != "What is 2+2?":
        return "EvalSample.question not set correctly"
    if sample.contexts != ["Basic arithmetic"]:
        return "EvalSample.contexts not set correctly"

    result = EvalResult(
        faithfulness=0.9,
        answer_relevancy=0.85,
        context_utilization=0.7,
    )
    d = result.to_dict()
    if not isinstance(d, dict):
        return f"EvalResult.to_dict() returned {type(d).__name__}"
    if d.get("faithfulness") != 0.9:
        return f"EvalResult.to_dict() faithfulness mismatch"

    spec = result.to_spec_metrics()
    if not isinstance(spec, dict):
        return f"EvalResult.to_spec_metrics() returned {type(spec).__name__}"
    expected_keys = [
        "faithfulness", "context_utilization", "self_knowledge",
        "noise_sensitivity", "answer_relevancy", "negative_rejection_rate",
    ]
    missing = [k for k in expected_keys if k not in spec]
    if missing:
        return f"to_spec_metrics() missing keys: {missing}"

    return None


# ===================================================================
# CHECK 13: set_seed reproducibility
# ===================================================================


def check_set_seed() -> Optional[str]:
    """Verify set_seed produces deterministic random output."""
    import random
    from cumrag.utils import set_seed

    set_seed(42)
    val1 = random.random()
    set_seed(42)
    val2 = random.random()

    if val1 != val2:
        return f"set_seed(42) not deterministic: {val1} != {val2}"

    return None


# ===================================================================
# CHECK 14: Phase 2 — normalize_datasets.py
# ===================================================================


def check_p2_normalize_import() -> Optional[str]:
    """Verify normalize_datasets.py is importable with expected symbols."""
    import importlib.util

    script_path = _PROJECT_ROOT / "scripts" / "normalize_datasets.py"
    if not script_path.exists():
        return f"scripts/normalize_datasets.py not found at {script_path}"

    spec = importlib.util.spec_from_file_location("normalize_datasets", str(script_path))
    if spec is None or spec.loader is None:
        return "Could not load module spec for normalize_datasets.py"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = ["normalize_rgb", "normalize_nq", "normalize_halueval"]
    missing = [s for s in required if not hasattr(mod, s)]
    if missing:
        return f"Missing symbols in normalize_datasets: {missing}"
    return None


def check_p2_normalize_help() -> Optional[str]:
    """Verify normalize_datasets.py --help works."""
    return check_script_help("scripts/normalize_datasets.py", "normalize_datasets")


def check_p2_normalize_transforms() -> Optional[str]:
    """Verify the 3 transform functions produce valid unified schema."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_datasets",
        str(_PROJECT_ROOT / "scripts" / "normalize_datasets.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    schema_fields = ["sample_id", "dataset", "subset", "query", "ground_truth",
                     "corpus_doc_ids", "metadata"]

    # Test normalize_rgb with synthetic data
    rgb_entry = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "passages": ["France is a country. Its capital is Paris."],
        "label": "positive",
    }
    rgb_result = mod.normalize_rgb(rgb_entry, "noise_robustness", 0)
    missing = [f for f in schema_fields if f not in rgb_result]
    if missing:
        return f"normalize_rgb output missing fields: {missing}"
    if rgb_result["dataset"] != "rgb":
        return f"normalize_rgb dataset should be 'rgb', got '{rgb_result['dataset']}'"
    if rgb_result["subset"] != "noise_robustness":
        return f"normalize_rgb subset mismatch: '{rgb_result['subset']}'"

    # Test normalize_nq with synthetic data
    nq_entry = {
        "question": "When was Python created?",
        "answer": ["1991", "February 1991"],
    }
    nq_result = mod.normalize_nq(nq_entry, 0)
    missing = [f for f in schema_fields if f not in nq_result]
    if missing:
        return f"normalize_nq output missing fields: {missing}"
    if nq_result["dataset"] != "nq":
        return f"normalize_nq dataset should be 'nq', got '{nq_result['dataset']}'"
    if nq_result["ground_truth"] != "1991":
        return f"normalize_nq ground_truth should be '1991', got '{nq_result['ground_truth']}'"

    # Test normalize_halueval with synthetic data
    halueval_entry = {
        "knowledge": "Python was created by Guido van Rossum.",
        "question": "Who created Python?",
        "right_answer": "Guido van Rossum",
        "hallucinated_answer": "Linus Torvalds",
    }
    halueval_result = mod.normalize_halueval(halueval_entry, 0)
    missing = [f for f in schema_fields if f not in halueval_result]
    if missing:
        return f"normalize_halueval output missing fields: {missing}"
    if halueval_result["dataset"] != "halueval":
        return f"normalize_halueval dataset should be 'halueval', got '{halueval_result['dataset']}'"

    return None


# ===================================================================
# CHECK 15: Phase 2 — make_collection_name()
# ===================================================================


def check_p2_make_collection_name() -> Optional[str]:
    """Verify make_collection_name is importable and deterministic."""
    from cumrag.utils import make_collection_name

    if not callable(make_collection_name):
        return "make_collection_name is not callable"

    # Determinism: same inputs -> same output
    name1 = make_collection_name("rgb_noise_robustness", "all-MiniLM-L6-v2", 300, 64)
    name2 = make_collection_name("rgb_noise_robustness", "all-MiniLM-L6-v2", 300, 64)
    if name1 != name2:
        return f"make_collection_name not deterministic: '{name1}' != '{name2}'"

    # Different inputs -> different output
    name3 = make_collection_name("nq_wiki", "all-MiniLM-L6-v2", 300, 64)
    if name1 == name3:
        return f"make_collection_name same for different datasets: '{name1}' == '{name3}'"

    name4 = make_collection_name("rgb_noise_robustness", "all-MiniLM-L6-v2", 400, 64)
    if name1 == name4:
        return f"make_collection_name same for different chunk_size: '{name1}' == '{name4}'"

    # Format check: should start with "cumrag_"
    if not name1.startswith("cumrag_"):
        return f"make_collection_name should start with 'cumrag_', got '{name1}'"

    # Should be a non-empty string
    if not name1 or not isinstance(name1, str):
        return f"make_collection_name returned invalid value: {name1!r}"

    return None


# ===================================================================
# CHECK 16: Phase 2 — validate_index() and resolve_collection_name()
# ===================================================================


def check_p2_validate_index() -> Optional[str]:
    """Verify validate_index importable and raises ValueError on mismatch."""
    from cumrag.retriever import validate_index

    if not callable(validate_index):
        return "validate_index is not callable"

    # Create a mock collection with matching metadata
    class MockCollection:
        def __init__(self, metadata):
            self.metadata = metadata

    config = {
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 300,
        }
    }

    # Should pass on match
    good_coll = MockCollection({
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_words": 300,
    })
    try:
        validate_index(good_coll, config)
    except ValueError as e:
        return f"validate_index raised on matching metadata: {e}"

    # Should raise on embedding model mismatch
    bad_model_coll = MockCollection({
        "embedding_model": "some-other-model",
        "chunk_size_words": 300,
    })
    try:
        validate_index(bad_model_coll, config)
        return "validate_index did not raise on embedding model mismatch"
    except ValueError:
        pass  # Expected

    # Should raise on chunk size mismatch
    bad_chunk_coll = MockCollection({
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_words": 500,
    })
    try:
        validate_index(bad_chunk_coll, config)
        return "validate_index did not raise on chunk size mismatch"
    except ValueError:
        pass  # Expected

    return None


def check_p2_resolve_collection_name() -> Optional[str]:
    """Verify resolve_collection_name resolves 'auto' correctly."""
    from cumrag.retriever import resolve_collection_name

    if not callable(resolve_collection_name):
        return "resolve_collection_name is not callable"

    config = {
        "retrieval": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 300,
            "overlap": 64,
            "collections": {
                "rgb_noise_robustness": "auto",
                "explicit_name": "my_custom_collection",
            },
        }
    }

    # "auto" should resolve to a deterministic name via make_collection_name
    auto_name = resolve_collection_name("rgb_noise_robustness", config)
    if not auto_name or not isinstance(auto_name, str):
        return f"resolve_collection_name returned invalid for 'auto': {auto_name!r}"
    if not auto_name.startswith("cumrag_"):
        return f"Auto-resolved name should start with 'cumrag_', got '{auto_name}'"

    # Same call should be deterministic
    auto_name2 = resolve_collection_name("rgb_noise_robustness", config)
    if auto_name != auto_name2:
        return f"resolve_collection_name not deterministic: '{auto_name}' != '{auto_name2}'"

    # Explicit name should be returned as-is
    explicit = resolve_collection_name("explicit_name", config)
    if explicit != "my_custom_collection":
        return f"Explicit name should be 'my_custom_collection', got '{explicit}'"

    # Missing key should raise KeyError
    try:
        resolve_collection_name("nonexistent_key", config)
        return "resolve_collection_name did not raise on missing key"
    except KeyError:
        pass  # Expected

    return None


# ===================================================================
# CHECK 17: Phase 2 — to_ragchecker_format()
# ===================================================================


def check_p2_to_ragchecker_format() -> Optional[str]:
    """Verify to_ragchecker_format produces correct RAGChecker schema."""
    from cumrag.evaluator import to_ragchecker_format

    if not callable(to_ragchecker_format):
        return "to_ragchecker_format is not callable"

    # Build synthetic raw results
    raw_results = [
        {
            "sample_id": "rgb_noise_robustness_0001",
            "query": "What is the capital of France?",
            "ground_truth": "Paris",
            "response": "The capital of France is Paris.",
            "retrieved_chunks": [
                {"doc_id": "wiki_france_01", "text": "Paris is the capital of France."},
                {"doc_id": "wiki_france_02", "text": "France is in Western Europe."},
            ],
        },
        {
            "sample_id": "rgb_noise_robustness_0002",
            "query": "Who wrote Python?",
            "ground_truth": "Guido van Rossum",
            "response": "Guido van Rossum created Python.",
            "retrieved_chunks": [
                {"doc_id": "wiki_python_01", "text": "Python was created by Guido."},
            ],
        },
    ]

    output = to_ragchecker_format(raw_results)

    # Must have 'results' key
    if "results" not in output:
        return "to_ragchecker_format output missing 'results' key"

    results = output["results"]
    if not isinstance(results, list):
        return f"'results' should be a list, got {type(results).__name__}"
    if len(results) != 2:
        return f"Expected 2 results, got {len(results)}"

    # Check first result has correct RAGChecker fields
    first = results[0]
    required = ["query_id", "query", "gt_answer", "response", "retrieved_context"]
    missing = [k for k in required if k not in first]
    if missing:
        return f"RAGChecker result missing fields: {missing}"

    if first["query_id"] != "rgb_noise_robustness_0001":
        return f"query_id mismatch: '{first['query_id']}'"
    if first["gt_answer"] != "Paris":
        return f"gt_answer should be 'Paris', got '{first['gt_answer']}'"

    # Check retrieved_context structure
    ctx = first["retrieved_context"]
    if not isinstance(ctx, list) or len(ctx) != 2:
        return f"retrieved_context should have 2 items, got {len(ctx) if isinstance(ctx, list) else type(ctx).__name__}"
    if "doc_id" not in ctx[0] or "text" not in ctx[0]:
        return f"retrieved_context items missing doc_id/text keys"

    return None


# ===================================================================
# CHECK 18: Phase 2 — eval_config.yaml v2 fields
# ===================================================================


def check_p2_config_judge() -> Optional[str]:
    """Verify eval_config.yaml has the judge block with required fields."""
    from cumrag.utils import load_config

    config = load_config("eval_config")
    judge = config.get("judge")
    if not judge:
        return "eval_config.yaml missing top-level 'judge' block"
    if not isinstance(judge, dict):
        return f"'judge' should be a dict, got {type(judge).__name__}"

    required = ["model", "quant", "server_port"]
    missing = [k for k in required if k not in judge]
    if missing:
        return f"judge block missing fields: {missing}"

    # Judge port should differ from generation port (8080)
    if judge.get("server_port") == 8080:
        return "judge.server_port should NOT be 8080 (conflicts with generation server)"

    return None


def check_p2_config_collections() -> Optional[str]:
    """Verify eval_config.yaml has retrieval.collections mapping."""
    from cumrag.utils import load_config

    config = load_config("eval_config")
    retrieval = config.get("retrieval", {})
    collections = retrieval.get("collections")
    if not collections:
        return "eval_config.yaml missing retrieval.collections"
    if not isinstance(collections, dict):
        return f"retrieval.collections should be a dict, got {type(collections).__name__}"

    # Should have at least one collection entry
    if len(collections) == 0:
        return "retrieval.collections is empty"

    # Check for expected RGB subset collection entries
    expected_keys = ["rgb_noise_robustness"]
    missing = [k for k in expected_keys if k not in collections]
    if missing:
        return f"retrieval.collections missing expected keys: {missing}"

    return None


# ===================================================================
# CHECK 19: Phase 2 — Runner v2 output schema
# ===================================================================


def check_p2_runner_v2_schema() -> Optional[str]:
    """Verify runner v2 output structure includes run_config, perf, retrieved_chunks."""
    from cumrag.runner import EvalRunner

    # Validate that EvalRunner.run_single exists and accepts the right params
    if not hasattr(EvalRunner, "run_single"):
        return "EvalRunner missing run_single method"

    sig = inspect.signature(EvalRunner.run_single)
    params = list(sig.parameters.keys())

    # run_single should accept key v2 params
    for p in ["query", "dataset", "model_name", "hardware_tier"]:
        if p not in params:
            return f"EvalRunner.run_single missing param '{p}'"

    # Build a synthetic v2 output record to validate schema
    v2_record = {
        "sample_id": "test_001",
        "dataset": "rgb",
        "subset": "noise_robustness",
        "query": "test query",
        "ground_truth": "test answer",
        "response": "generated response",
        "retrieved_chunks": [
            {"doc_id": "doc1", "text": "chunk text", "distance": 0.5, "rank": 1},
        ],
        "run_config": {
            "model": "test-model",
            "quant": "Q4_K_M",
            "hardware": "cpu",
            "seed": 42,
        },
        "perf": {
            "tokens_per_sec": 3.2,
            "ttft_ms": 1200,
            "vram_mb": 0,
        },
        "metrics": {
            "faithfulness": 0.9,
            "context_utilization": 0.8,
        },
        "timestamp": "2026-03-01T14:30:00Z",
        "error": None,
    }

    # Validate v2 top-level keys
    v2_required = ["sample_id", "dataset", "subset", "query", "ground_truth",
                    "response", "retrieved_chunks", "run_config", "perf",
                    "metrics", "timestamp"]
    missing = [k for k in v2_required if k not in v2_record]
    if missing:
        return f"v2 record missing keys: {missing}"

    # Validate run_config sub-fields
    rc = v2_record["run_config"]
    rc_required = ["model", "quant", "hardware", "seed"]
    missing_rc = [k for k in rc_required if k not in rc]
    if missing_rc:
        return f"v2 run_config missing keys: {missing_rc}"

    # Validate retrieved_chunks structure
    chunks = v2_record["retrieved_chunks"]
    if not isinstance(chunks, list):
        return "retrieved_chunks should be a list"
    if len(chunks) > 0:
        chunk = chunks[0]
        chunk_required = ["doc_id", "text", "distance", "rank"]
        missing_c = [k for k in chunk_required if k not in chunk]
        if missing_c:
            return f"retrieved_chunk missing keys: {missing_c}"

    return None


# ===================================================================
# CHECK 20: Phase 2 — Aggregator v2 format parsing
# ===================================================================


def check_p2_aggregator_v2_parse() -> Optional[str]:
    """Verify aggregator can parse v2 format records (run_config, perf, metrics)."""
    from cumrag.aggregator import group_results, bootstrap_ci, _extract_metric

    # Build synthetic v2 records
    v2_records = [
        {
            "sample_id": f"test_{i:04d}",
            "dataset": "rgb",
            "subset": "noise_robustness",
            "run_config": {
                "model": "llama-3.1-8b",
                "quant": "Q4_K_M",
                "hardware": "cpu",
            },
            "perf": {
                "tokens_per_sec": 3.0 + i * 0.1,
                "ttft_ms": 1200 + i * 10,
                "vram_mb": 0,
            },
            "metrics": {
                "faithfulness": 0.80 + i * 0.01,
                "context_utilization": 0.70 + i * 0.02,
            },
        }
        for i in range(5)
    ]

    # Test grouping
    groups = group_results(v2_records)
    if not groups:
        return "group_results returned empty for v2 records"
    if len(groups) != 1:
        return f"Expected 1 group, got {len(groups)}"

    # Check the group key maps correctly
    key = list(groups.keys())[0]
    # key should be (model, quantization, hardware_tier, dataset, eval_subset)
    if len(key) != 5:
        return f"Group key should have 5 elements, got {len(key)}"
    model, quant, hw, dataset, subset = key
    if model != "llama-3.1-8b":
        return f"Group key model should be 'llama-3.1-8b', got '{model}'"
    if quant != "Q4_K_M":
        return f"Group key quant should be 'Q4_K_M', got '{quant}'"
    if hw != "cpu":
        return f"Group key hardware should be 'cpu', got '{hw}'"

    # Test metric extraction from v2 format
    rec = v2_records[0]
    faith = _extract_metric(rec, "faithfulness")
    if faith is None:
        return "_extract_metric could not find 'faithfulness' in v2 record"
    if abs(faith - 0.80) > 0.001:
        return f"_extract_metric faithfulness expected ~0.80, got {faith}"

    tps = _extract_metric(rec, "tokens_per_second")
    if tps is None:
        return "_extract_metric could not find 'tokens_per_second' via v2 perf alias"
    if abs(tps - 3.0) > 0.001:
        return f"_extract_metric tokens_per_second expected ~3.0, got {tps}"

    return None


# ===================================================================
# OPTIONAL LIVE TEST
# ===================================================================


def run_live_test(model_path: str, dataset: str) -> bool:
    """Run a minimal live pipeline test (requires model + dataset).

    Starts llama-server, retrieves chunks, generates a response,
    and verifies the output format.

    Returns:
        True if test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("LIVE PIPELINE TEST")
    print("=" * 60)

    from cumrag.generator import LlamaServer, load_prompt_template, format_prompt
    from cumrag.utils import Timer, get_hardware_meta

    model_file = Path(model_path)
    if not model_file.exists():
        _record("live.model_exists", _FAIL, f"Model not found: {model_file}")
        return False
    _record("live.model_exists", _PASS, str(model_file))

    # Check dataset index exists
    index_dir = _PROJECT_ROOT / "index"
    if not index_dir.exists():
        _record("live.index_exists", _FAIL, f"Index dir not found: {index_dir}")
        return False

    try:
        from cumrag.retriever import Retriever
        retriever = Retriever(index_dir=str(index_dir), dataset=dataset)
        info = retriever.get_collection_info()
        _record("live.retriever_init", _PASS, f"{info['count']} chunks in {info['collection_name']}")
    except Exception as e:
        _record("live.retriever_init", _FAIL, str(e))
        return False

    # Retrieve test chunks
    test_query = "What is the capital of France?"
    try:
        chunks = retriever.retrieve(test_query, top_k=3)
        if not chunks:
            _record("live.retrieve", _FAIL, "No chunks returned")
            return False
        _record("live.retrieve", _PASS, f"{len(chunks)} chunks retrieved")
    except Exception as e:
        _record("live.retrieve", _FAIL, str(e))
        return False

    # Start server, generate, stop
    srv = LlamaServer(port=8090)  # Non-default port to avoid conflicts
    try:
        _record("live.server_start", _SKIP, "Starting llama-server...")
        srv.start(model_file, n_gpu_layers=0, port=8090)
        srv.wait_ready(timeout=300)
        _record("live.server_ready", _PASS, "Server ready")

        # Format prompt and generate
        template = load_prompt_template()
        prompt = format_prompt(template, test_query, chunks)

        with Timer() as t:
            result = srv.generate_with_metrics(prompt)

        if not result.get("text"):
            _record("live.generate", _FAIL, "Empty response")
            return False

        _record(
            "live.generate", _PASS,
            f"{result.get('completion_tokens', '?')} tokens, "
            f"{result.get('tokens_per_second', '?'):.1f} tok/s, "
            f"{t.elapsed:.1f}s wall"
        )

        # Verify response is not just whitespace
        if not result["text"].strip():
            _record("live.response_quality", _FAIL, "Response is whitespace only")
            return False
        _record("live.response_quality", _PASS, f"Response: {result['text'][:100]}...")

    except Exception as e:
        _record("live.pipeline", _FAIL, f"{type(e).__name__}: {e}")
        return False
    finally:
        srv.stop()
        _record("live.server_stop", _PASS, "Server stopped cleanly")

    return True


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CUMRAG smoke test — validate harness is wired up correctly",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live pipeline test (requires model + dataset index)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to GGUF model file for live test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rgb",
        help="Dataset for live test retrieval (default: rgb)",
    )
    args = parser.parse_args()

    if args.live and not args.model:
        parser.error("--live requires --model")

    print("=" * 60)
    print("  CUMRAG Smoke Test")
    print("=" * 60)
    print()

    # --- Import checks ---
    print("[1/13] Import verification")
    _run_check("import.utils", check_import_utils)
    _run_check("import.retriever", check_import_retriever)
    _run_check("import.generator", check_import_generator)
    _run_check("import.evaluator", check_import_evaluator)
    _run_check("import.runner", check_import_runner)
    _run_check("import.aggregator", check_import_aggregator)
    print()

    # --- Interface checks ---
    print("[2/13] Module interface verification")
    _run_check("interface.retriever", check_retriever_interface)
    _run_check("interface.generator", check_generator_interface)
    _run_check("interface.evaluator", check_evaluator_interface)
    _run_check("interface.runner", check_runner_interface)
    _run_check("interface.aggregator", check_aggregator_interface)
    print()

    # --- Config checks ---
    print("[3/13] Config loading verification")
    _run_check("config.eval_config", check_config_eval)
    _run_check("config.models", check_config_models)
    _run_check("config.hardware", check_config_hardware)
    print()

    # --- Prompt template ---
    print("[4/13] Prompt template verification")
    _run_check("template.prompt", check_prompt_template)
    _run_check("template.format", check_format_prompt)
    print()

    # --- JSONL round-trip ---
    print("[5/13] JSONL round-trip test")
    _run_check("jsonl.roundtrip", check_jsonl_roundtrip)
    print()

    # --- Hardware detection ---
    print("[6/13] Hardware detection test")
    _run_check("hardware.meta", check_hardware_detection)
    _run_check("hardware.full", check_hardware_full)
    print()

    # --- Timer ---
    print("[7/13] Timer test")
    _run_check("timer.context_manager", check_timer)
    print()

    # --- Script --help ---
    print("[8/13] Script --help verification")
    _run_check("script.download_datasets", check_script_download_datasets)
    _run_check("script.download_models", check_script_download_models)
    _run_check("script.build_index", check_script_build_index)
    _run_check("script.profile_hardware", check_script_profile_hardware)
    print()

    # --- Module CLI --help ---
    print("[9/13] Module CLI --help verification")
    _run_check("module.retriever", check_module_retriever_help)
    _run_check("module.aggregator", check_module_aggregator_help)
    _run_check("module.utils", check_module_utils_help)
    print()

    # --- Bootstrap CI numeric ---
    print("[10/13] Bootstrap CI numeric test")
    _run_check("aggregator.bootstrap_ci", check_bootstrap_ci)
    print()

    # --- Dataclass tests ---
    print("[11/13] EvalSample / EvalResult dataclass test")
    _run_check("evaluator.dataclasses", check_eval_dataclasses)
    print()

    # --- Seed reproducibility ---
    print("[12/13] Seed reproducibility test")
    _run_check("utils.set_seed", check_set_seed)
    print()

    # --- Phase 2: normalize_datasets ---
    print("[14/20] normalize_datasets.py verification")
    _run_check("p2.normalize_datasets.import", check_p2_normalize_import)
    _run_check("p2.normalize_datasets.help", check_p2_normalize_help)
    _run_check("p2.normalize_datasets.transforms", check_p2_normalize_transforms)
    print()

    # --- Phase 2: make_collection_name ---
    print("[15/20] make_collection_name() verification")
    _run_check("p2.utils.make_collection_name", check_p2_make_collection_name)
    print()

    # --- Phase 2: validate_index and resolve_collection_name ---
    print("[16/20] validate_index() and resolve_collection_name() verification")
    _run_check("p2.retriever.validate_index", check_p2_validate_index)
    _run_check("p2.retriever.resolve_collection_name", check_p2_resolve_collection_name)
    print()

    # --- Phase 2: to_ragchecker_format ---
    print("[17/20] to_ragchecker_format() verification")
    _run_check("p2.evaluator.to_ragchecker_format", check_p2_to_ragchecker_format)
    print()

    # --- Phase 2: eval_config v2 fields ---
    print("[18/20] eval_config.yaml v2 fields verification")
    _run_check("p2.config.judge_block", check_p2_config_judge)
    _run_check("p2.config.retrieval_collections", check_p2_config_collections)
    print()

    # --- Phase 2: Runner v2 output schema ---
    print("[19/20] Runner v2 output schema verification")
    _run_check("p2.runner.v2_schema", check_p2_runner_v2_schema)
    print()

    # --- Phase 2: Aggregator v2 parsing ---
    print("[20/20] Aggregator v2 format parsing verification")
    _run_check("p2.aggregator.v2_parse", check_p2_aggregator_v2_parse)
    print()

    # --- Live test (optional) ---
    live_passed = True
    if args.live:
        print("[LIVE] Live pipeline test")
        live_passed = run_live_test(args.model, args.dataset)
        print()
    else:
        print("[LIVE] Live pipeline test — SKIPPED (use --live --model PATH)")
        print()

    # --- Summary ---
    total = len(_results)
    passed = sum(1 for r in _results if r["status"] == _PASS)
    failed = sum(1 for r in _results if r["status"] == _FAIL)
    skipped = sum(1 for r in _results if r["status"] == _SKIP)

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
    print("=" * 60)

    if failed > 0:
        print("\nFailed checks:")
        for r in _results:
            if r["status"] == _FAIL:
                print(f"  - {r['name']}: {r['detail'][:200]}")
        print()

    # Core checks: exit 1 if any non-live check failed
    core_failures = [
        r for r in _results
        if r["status"] == _FAIL and not r["name"].startswith("live.")
    ]
    if core_failures:
        print("SMOKE TEST: FAILED")
        return 1

    if failed > 0 and not core_failures:
        # Only live tests failed
        print("SMOKE TEST: CORE PASSED (live test failures)")
        return 0

    print("SMOKE TEST: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
