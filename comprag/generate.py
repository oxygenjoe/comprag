"""Query execution: message construction, local and frontier generation.

Builds standard OpenAI-style messages arrays from prompt templates.
No manual chat template tokens — the server or API handles wrapping.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import anthropic
import openai
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
    server_url: str = "http://localhost:8080",
) -> tuple[str, int]:
    """POST to /v1/chat/completions. Returns (response_text, time_ms).

    Locked: temperature=0.0, max_tokens=512, seed=42.

    Args:
        messages: OpenAI-style messages array.
        server_url: Base URL of the llama.cpp server.

    Returns:
        Tuple of (generated text, wall-clock milliseconds).
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
    logger.debug("generate_local: %d tokens in %d ms", len(text.split()), time_ms)
    return text, time_ms


# --- Provider-specific API key env var mapping ---
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
}

# --- Default base URLs for OpenAI-compatible providers ---
_PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,  # uses default openai endpoint
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


def _get_api_key(provider: str) -> str:
    """Resolve API key from environment variable for a given provider."""
    env_var = _PROVIDER_API_KEY_ENV[provider]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"Missing environment variable {env_var} for provider '{provider}'")
    return key


def _call_openai_compat(
    messages: list[dict], model_id: str, provider: str, seed: int,
) -> tuple[str, int]:
    """Call OpenAI-compatible API (openai/deepseek/zhipu/google). Returns (text, time_ms)."""
    api_key = _get_api_key(provider)
    base_url = _PROVIDER_BASE_URLS.get(provider)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        seed=seed,
    )
    time_ms = int((time.monotonic() - t0) * 1000)

    text = resp.choices[0].message.content
    logger.debug("generate_frontier[%s]: %d ms", provider, time_ms)
    return text, time_ms


def _extract_system_and_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Split system message from user/assistant messages for Anthropic format."""
    system: str | None = None
    non_system: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            non_system.append({"role": msg["role"], "content": msg["content"]})
    return system, non_system


def _call_anthropic(
    messages: list[dict], model_id: str,
) -> tuple[str, int]:
    """Call Anthropic API. Returns (text, time_ms).

    NOTE: Anthropic does not support seed-based reproducibility.
    Variance across runs is captured in bootstrap CIs.
    """
    api_key = _get_api_key("anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    system, user_messages = _extract_system_and_messages(messages)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": user_messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if system:
        kwargs["system"] = system

    t0 = time.monotonic()
    resp = client.messages.create(**kwargs)
    time_ms = int((time.monotonic() - t0) * 1000)

    text = resp.content[0].text
    logger.debug("generate_frontier[anthropic]: %d ms", time_ms)
    return text, time_ms


def generate_frontier(
    messages: list[dict], provider: str, model_id: str, seed: int = 42,
) -> tuple[str, int]:
    """Call frontier API. Returns (response_text, time_ms).

    Provider routing:
    - openai/deepseek/zhipu: openai SDK with base_url override
    - anthropic: anthropic SDK (no seed support; variance in bootstrap CIs)
    - google: openai SDK with Google's OpenAI-compat endpoint

    Locked: temperature=0.0, max_tokens=512. Seed passed where supported.
    API keys read from environment variables.
    """
    if provider == "anthropic":
        return _call_anthropic(messages, model_id)
    if provider in ("openai", "deepseek", "zhipu", "google"):
        return _call_openai_compat(messages, model_id, provider, seed)
    raise ValueError(f"Unknown provider '{provider}'. Expected one of: {list(_PROVIDER_API_KEY_ENV.keys())}")
