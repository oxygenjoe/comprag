"""Tests for comprag.generate — message construction and generation calls.

Covers:
- build_messages for all 3 passes (baseline, loose, strict)
- generate_local with mocked urllib
- generate_frontier with mocked Anthropic and OpenAI SDKs
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from comprag.generate import (
    _format_context,
    build_messages,
    generate_frontier,
    generate_local,
)

# Reset the module-level prompts cache before each test so
# tests that supply a custom prompts_path get a clean load.
_PROMPTS_YAML = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


@pytest.fixture(autouse=True)
def _clear_prompts_cache() -> None:
    """Reset the prompts cache between tests."""
    import comprag.generate as gen_mod
    gen_mod._prompts_cache = None


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_joins_with_double_newline(self) -> None:
        chunks = ["chunk A", "chunk B", "chunk C"]
        assert _format_context(chunks) == "chunk A\n\nchunk B\n\nchunk C"

    def test_single_chunk(self) -> None:
        assert _format_context(["only one"]) == "only one"

    def test_empty_list(self) -> None:
        assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# build_messages — pass1_baseline
# ---------------------------------------------------------------------------


class TestBuildMessagesPass1:
    def test_returns_single_user_message(self) -> None:
        msgs = build_messages("What is Python?", None, "pass1_baseline")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_user_content_is_query(self) -> None:
        query = "What is the capital of France?"
        msgs = build_messages(query, None, "pass1_baseline")
        assert msgs[0]["content"] == query

    def test_no_system_message(self) -> None:
        msgs = build_messages("test", None, "pass1_baseline")
        roles = [m["role"] for m in msgs]
        assert "system" not in roles

    def test_ignores_context_when_provided(self) -> None:
        """pass1_baseline should not embed context even if given."""
        msgs = build_messages("query", ["ctx1", "ctx2"], "pass1_baseline")
        assert len(msgs) == 1
        # Context strings should NOT appear in the message
        assert "ctx1" not in msgs[0]["content"]
        assert "ctx2" not in msgs[0]["content"]


# ---------------------------------------------------------------------------
# build_messages — pass2_loose
# ---------------------------------------------------------------------------


class TestBuildMessagesPass2:
    def test_returns_single_user_message(self) -> None:
        msgs = build_messages("q", ["c1"], "pass2_loose")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_no_system_message(self) -> None:
        msgs = build_messages("q", ["c1"], "pass2_loose")
        roles = [m["role"] for m in msgs]
        assert "system" not in roles

    def test_context_present_in_message(self) -> None:
        msgs = build_messages("q", ["chunk alpha", "chunk beta"], "pass2_loose")
        content = msgs[0]["content"]
        assert "chunk alpha" in content
        assert "chunk beta" in content

    def test_query_present_in_message(self) -> None:
        msgs = build_messages("What is X?", ["c"], "pass2_loose")
        assert "What is X?" in msgs[0]["content"]

    def test_raises_on_missing_context(self) -> None:
        with pytest.raises(ValueError, match="requires context"):
            build_messages("q", None, "pass2_loose")

    def test_raises_on_empty_context(self) -> None:
        with pytest.raises(ValueError, match="requires context"):
            build_messages("q", [], "pass2_loose")


# ---------------------------------------------------------------------------
# build_messages — pass3_strict
# ---------------------------------------------------------------------------


class TestBuildMessagesPass3:
    def test_returns_system_and_user(self) -> None:
        msgs = build_messages("q", ["c1"], "pass3_strict")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_says_answer_only(self) -> None:
        msgs = build_messages("q", ["c1"], "pass3_strict")
        assert "Answer using ONLY" in msgs[0]["content"]

    def test_context_in_user_message(self) -> None:
        msgs = build_messages("q", ["ctx_data"], "pass3_strict")
        assert "ctx_data" in msgs[1]["content"]

    def test_query_in_user_message(self) -> None:
        msgs = build_messages("my question?", ["c"], "pass3_strict")
        assert "my question?" in msgs[1]["content"]

    def test_raises_on_missing_context(self) -> None:
        with pytest.raises(ValueError, match="requires context"):
            build_messages("q", None, "pass3_strict")

    def test_raises_on_empty_context(self) -> None:
        with pytest.raises(ValueError, match="requires context"):
            build_messages("q", [], "pass3_strict")


# ---------------------------------------------------------------------------
# build_messages — error cases
# ---------------------------------------------------------------------------


class TestBuildMessagesErrors:
    def test_invalid_pass_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown pass_name"):
            build_messages("q", None, "pass4_nonexistent")

    def test_empty_string_pass_name_raises(self) -> None:
        with pytest.raises(KeyError):
            build_messages("q", None, "")


# ---------------------------------------------------------------------------
# generate_local — mocked urllib
# ---------------------------------------------------------------------------


def _make_urlopen_response(content: str) -> MagicMock:
    """Create a mock context-manager response for urllib.request.urlopen."""
    body = json.dumps({
        "choices": [{"message": {"content": content}}],
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestGenerateLocal:
    @patch("comprag.generate.urllib.request.urlopen")
    def test_returns_text_and_time(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response("hello world")
        text, time_ms, response_model = generate_local([{"role": "user", "content": "hi"}])
        assert text == "hello world"
        assert isinstance(time_ms, int)
        assert time_ms >= 0

    @patch("comprag.generate.urllib.request.urlopen")
    def test_payload_has_locked_params(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response("ok")
        messages = [{"role": "user", "content": "test"}]
        generate_local(messages, server_url="http://myhost:9999")

        # Inspect the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        req_obj = call_args[0][0]
        sent_payload = json.loads(req_obj.data.decode())

        assert sent_payload["temperature"] == 0.0
        assert sent_payload["max_tokens"] == 512
        assert sent_payload["seed"] == 42
        assert sent_payload["messages"] == messages

    @patch("comprag.generate.urllib.request.urlopen")
    def test_url_construction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response("ok")
        generate_local([{"role": "user", "content": "x"}], server_url="http://box:8080")

        req_obj = mock_urlopen.call_args[0][0]
        assert req_obj.full_url == "http://box:8080/v1/chat/completions"


# ---------------------------------------------------------------------------
# generate_frontier — Anthropic path (mocked SDK)
# ---------------------------------------------------------------------------


class TestGenerateFrontierAnthropic:
    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.anthropic.Anthropic")
    def test_returns_text_and_time(
        self, mock_anthropic_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="anthropic answer")]
        mock_resp.model = "claude-sonnet-4-20250514"
        mock_client.messages.create.return_value = mock_resp

        text, time_ms, response_model = generate_frontier(
            [{"role": "user", "content": "q"}], "anthropic", "claude-sonnet-4-20250514",
        )
        assert text == "anthropic answer"
        assert isinstance(time_ms, int)
        assert response_model == "claude-sonnet-4-20250514"

    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.anthropic.Anthropic")
    def test_system_extracted_from_messages(
        self, mock_anthropic_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        mock_client.messages.create.return_value = mock_resp

        messages = [
            {"role": "system", "content": "be strict"},
            {"role": "user", "content": "q"},
        ]
        generate_frontier(messages, "anthropic", "claude-sonnet-4-20250514")

        create_kwargs = mock_client.messages.create.call_args
        # system should be extracted to a kwarg, not in messages list
        assert create_kwargs.kwargs.get("system") == "be strict" or \
               create_kwargs[1].get("system") == "be strict"
        # The messages passed should NOT contain a system role
        passed_msgs = (
            create_kwargs.kwargs.get("messages") or create_kwargs[1].get("messages")
        )
        assert all(m["role"] != "system" for m in passed_msgs)

    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.anthropic.Anthropic")
    def test_no_seed_param(
        self, mock_anthropic_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        mock_client.messages.create.return_value = mock_resp

        generate_frontier(
            [{"role": "user", "content": "q"}], "anthropic", "claude-sonnet-4-20250514",
        )
        create_kwargs = mock_client.messages.create.call_args
        all_kwargs = {**dict(zip(
            mock_client.messages.create.call_args.args,
            range(100),
        ))}
        # seed should NOT be passed to anthropic
        kw = create_kwargs.kwargs if create_kwargs.kwargs else create_kwargs[1]
        assert "seed" not in kw


# ---------------------------------------------------------------------------
# generate_frontier — OpenAI-compat path (mocked SDK)
# ---------------------------------------------------------------------------


class TestGenerateFrontierOpenAI:
    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.openai.OpenAI")
    def test_returns_text_and_time(
        self, mock_openai_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="openai answer"))]
        mock_resp.model = "gpt-4o"
        mock_client.chat.completions.create.return_value = mock_resp

        text, time_ms, response_model = generate_frontier(
            [{"role": "user", "content": "q"}], "openai", "gpt-4o",
        )
        assert text == "openai answer"
        assert isinstance(time_ms, int)
        assert response_model == "gpt-4o"

    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.openai.OpenAI")
    def test_messages_passed_as_is(
        self, mock_openai_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_resp

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        generate_frontier(messages, "openai", "gpt-4o")

        create_kwargs = mock_client.chat.completions.create.call_args
        kw = create_kwargs.kwargs if create_kwargs.kwargs else create_kwargs[1]
        # Messages should be passed as-is (system message stays in the list)
        assert kw["messages"] == messages

    @patch("comprag.generate._get_api_key", return_value="sk-test-key")
    @patch("comprag.generate.openai.OpenAI")
    def test_seed_included(
        self, mock_openai_cls: MagicMock, mock_key: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_resp

        generate_frontier(
            [{"role": "user", "content": "q"}], "openai", "gpt-4o", seed=99,
        )
        create_kwargs = mock_client.chat.completions.create.call_args
        kw = create_kwargs.kwargs if create_kwargs.kwargs else create_kwargs[1]
        assert kw["seed"] == 99


# ---------------------------------------------------------------------------
# generate_frontier — unknown provider
# ---------------------------------------------------------------------------


class TestGenerateFrontierErrors:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            generate_frontier([{"role": "user", "content": "q"}], "fakeprovider", "m")
