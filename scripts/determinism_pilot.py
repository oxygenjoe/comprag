#!/usr/bin/env python3
"""Determinism pilot: verify llama.cpp greedy decoding is seed-invariant.

Runs N seeds x M queries against a local llama-server with temperature=0.
If all outputs are bit-identical across seeds, the experiment can use a
single seed (42) and bootstrap over queries only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comprag.server import LlamaCppServer, LlamaCppServerError

logger = logging.getLogger(__name__)

# Sample queries covering diverse topics for determinism testing.
PILOT_QUERIES = [
    "What is the capital of France?",
    "Explain photosynthesis in three sentences.",
    "What causes tides on Earth?",
    "Describe the structure of DNA.",
    "Who wrote Hamlet?",
    "What is the speed of light in vacuum?",
    "Explain the difference between TCP and UDP.",
    "What are the three laws of thermodynamics?",
    "Describe how a transistor works.",
    "What is the Pythagorean theorem?",
    "Explain continental drift theory.",
    "What is an enzyme?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Explain the concept of supply and demand.",
    "What is the greenhouse effect?",
    "Describe how vaccines work.",
    "What are prime numbers?",
    "Explain the theory of relativity briefly.",
    "What is the function of mitochondria?",
    "Describe the process of photosynthesis.",
    "What causes earthquakes?",
    "Explain how the internet works.",
    "What is the periodic table?",
    "Describe the structure of an atom.",
    "What is natural selection?",
    "Explain the concept of inflation in economics.",
    "What is a black hole?",
    "Describe how neural networks learn.",
    "What are antibiotics?",
    "Explain the Doppler effect.",
    "What is a genome?",
    "Describe the scientific method.",
    "What is cryptocurrency?",
    "Explain how solar panels work.",
    "What are tectonic plates?",
    "Describe the human circulatory system.",
    "What is an algorithm?",
    "Explain the Big Bang theory.",
    "What is climate change?",
    "Describe how a combustion engine works.",
    "What is the Heisenberg uncertainty principle?",
    "Explain the concept of pH.",
    "What is a supernova?",
    "Describe the process of nuclear fission.",
    "What is artificial intelligence?",
    "Explain how GPS works.",
    "What are stem cells?",
    "Describe the carbon cycle.",
    "What is quantum computing?",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Determinism pilot: test seed-invariance of greedy decoding.",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to GGUF model file.",
    )
    parser.add_argument(
        "--quant", default="Q4_K_M",
        help="Quantization label for output metadata (default: Q4_K_M).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds to test (default: 10).",
    )
    parser.add_argument(
        "--n-queries", type=int, default=50,
        help="Number of queries to run per seed (default: 50).",
    )
    parser.add_argument(
        "--output", default="determinism_pilot_result.json",
        help="Output JSON file path (default: determinism_pilot_result.json).",
    )
    parser.add_argument(
        "--port", type=int, default=5741,
        help="Port for llama-server (default: 5741).",
    )
    return parser.parse_args()


def send_query(
    query: str, seed: int, server_url: str,
) -> tuple[str, int]:
    """Send a single query to llama-server with a specific seed.

    Returns:
        Tuple of (response_text, time_ms).

    Raises:
        urllib.error.URLError: On connection failure.
    """
    payload = json.dumps({
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.0,
        "max_tokens": 512,
        "seed": seed,
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    time_ms = int((time.monotonic() - t0) * 1000)

    text = body["choices"][0]["message"]["content"]
    return text, time_ms


def run_seed(
    seed: int,
    queries: list[str],
    server_url: str,
) -> list[str]:
    """Run all queries for one seed, return list of response texts."""
    responses: list[str] = []
    for i, query in enumerate(queries):
        try:
            text, time_ms = send_query(query, seed, server_url)
            logger.info(
                "seed=%d query=%d/%d time=%dms",
                seed, i + 1, len(queries), time_ms,
            )
            responses.append(text)
        except (urllib.error.URLError, OSError) as e:
            logger.error("seed=%d query=%d failed: %s", seed, i + 1, e)
            responses.append(f"ERROR: {e}")
    return responses


def compute_variance_stats(
    all_responses: dict[int, list[str]],
    queries: list[str],
) -> list[dict[str, Any]]:
    """Compute per-query variance stats across seeds.

    Returns list of dicts with query index, query text, number of
    unique responses, and the distinct responses observed.
    """
    stats: list[dict[str, Any]] = []
    seeds = sorted(all_responses.keys())
    n_queries = len(queries)

    for qi in range(n_queries):
        responses_for_query = [all_responses[s][qi] for s in seeds]
        unique = list(set(responses_for_query))
        stats.append({
            "query_index": qi,
            "query": queries[qi],
            "n_unique_responses": len(unique),
            "unique_responses": unique[:5],  # cap for readability
        })
    return stats


def write_result(result: dict[str, Any], output_path: str) -> None:
    """Write result dict to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Result written to %s", path)


def main() -> None:
    """Entry point: run determinism pilot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    queries = PILOT_QUERIES[: args.n_queries]
    if len(queries) < args.n_queries:
        logger.warning(
            "Only %d built-in queries available, requested %d",
            len(queries), args.n_queries,
        )

    seeds = list(range(1, args.n_seeds + 1))
    model_label = Path(args.model).stem
    server_url = f"http://localhost:{args.port}"

    logger.info(
        "Determinism pilot: model=%s seeds=%d queries=%d",
        model_label, len(seeds), len(queries),
    )

    all_responses: dict[int, list[str]] = {}

    try:
        with LlamaCppServer(args.model, port=args.port) as srv:
            for seed in seeds:
                logger.info("--- Running seed %d/%d ---", seed, len(seeds))
                all_responses[seed] = run_seed(
                    seed, queries, srv.base_url,
                )
    except LlamaCppServerError as e:
        logger.error("Server failure: %s", e)
        write_result(
            {"error": str(e), "deterministic": None},
            args.output,
        )
        sys.exit(1)

    # Compare: check if all seeds produced identical outputs.
    reference = all_responses[seeds[0]]
    is_deterministic = all(
        all_responses[s] == reference for s in seeds[1:]
    )

    if is_deterministic:
        result: dict[str, Any] = {
            "deterministic": True,
            "model": model_label,
            "quant": args.quant,
            "n_seeds": len(seeds),
            "n_queries": len(queries),
        }
        logger.info("RESULT: deterministic=True")
    else:
        variance_stats = compute_variance_stats(all_responses, queries)
        n_divergent = sum(
            1 for s in variance_stats if s["n_unique_responses"] > 1
        )
        result = {
            "deterministic": False,
            "model": model_label,
            "quant": args.quant,
            "n_seeds": len(seeds),
            "n_queries": len(queries),
            "n_divergent_queries": n_divergent,
            "recommended_seeds": 3,
            "variance_stats": variance_stats,
        }
        logger.info(
            "RESULT: deterministic=False, %d/%d queries diverged",
            n_divergent, len(queries),
        )

    write_result(result, args.output)


if __name__ == "__main__":
    main()
