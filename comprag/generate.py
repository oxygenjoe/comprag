"""Query execution: message construction, local generation only.

Builds standard OpenAI-style messages arrays from prompt templates.
No manual chat template tokens — the server or API handles wrapping.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"

_prompts_cache: dict[str, Any] | None = None


def _load_prompts(path: Path = _DEFAULT_PROMPTS_PATH) -> dict[str, Any]:
    """Load and cache prompt templates from YAML config."""
    global _prompts_cache
    if _prompts_cache is not None:
        return _prompts_cache
    logger.debug("Loading prompts from %s", path)
    with open(path) as f:
        _prompts_cache = yaml.safe_load(f)
    return _prompts_cache


def _format_context(context: list[str]) -> str:
    """Join context chunks with double newlines."""
    return "\n\n".join(context)


def build_messages(
    query: str,
    context: list[str] | None,
    pass_name: str,
    prompts_path: Path = _DEFAULT_PROMPTS_PATH,
) -> list[dict]:
    """Construct messages array for chat completions API.

    Args:
        query: The user's question.
        context: Retrieved text chunks, or None for baseline pass.
        pass_name: One of 'pass1_baseline', 'pass2_loose', 'pass3_strict'.
        prompts_path: Path to prompts YAML config.

    Returns:
        Standard messages array: list of {"role": ..., "content": ...} dicts.

    Raises:
        KeyError: If pass_name not found in prompts config.
        ValueError: If context required but not provided.
    """
    prompts = _load_prompts(prompts_path)
    if pass_name not in prompts:
        raise KeyError(f"Unknown pass_name '{pass_name}'. Expected one of: {list(prompts.keys())}")

    template = prompts[pass_name]
    context_str = _format_context(context) if context else ""

    if template.get("include_context") and not context:
        raise ValueError(f"Pass '{pass_name}' requires context but none provided")

    messages: list[dict] = []

    if template.get("system"):
        messages.append({"role": "system", "content": template["system"]})

    user_content = template["user"].format(query=query, context=context_str)
    messages.append({"role": "user", "content": user_content})

    logger.debug("Built %d message(s) for %s", len(messages), pass_name)
    return messages


def generate_local(
    messages: list[dict],
    server_url: str = "http://localhost:5741",
) -> tuple[str, int, str]:
    """POST to /v1/chat/completions. Returns (response_text, time_ms, response_model).

    Locked: temperature=0.0, max_tokens=512, seed=42.

    Args:
        messages: OpenAI-style messages array.
        server_url: Base URL of the llama.cpp server.

    Returns:
        Tuple of (generated text, wall-clock milliseconds, model ID from response).
    """
    payload = json.dumps({
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "seed": 42,
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    time_ms = int((time.monotonic() - t0) * 1000)

    text = body["choices"][0]["message"]["content"]
    response_model = body.get("model", "")
    logger.debug("generate_local: %d tokens in %d ms, model=%s", len(text.split()), time_ms, response_model)
    return text, time_ms, response_model
