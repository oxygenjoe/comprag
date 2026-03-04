"""llama.cpp server interface for the CompRAG eval harness.

Manages llama-server process lifecycle and provides OpenAI-compatible
API access for text generation. Supports server start/stop, health
polling, prompt template injection, and performance metrics collection.

Importable as module; also runnable as CLI via `python -m comprag.generator`.
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import requests

from comprag.utils import Timer, get_logger, load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SERVER_BINARY_CANDIDATES = [
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
    Path.home() / "llama.cpp" / "build" / "bin" / "server",
    Path("/usr/local/bin/llama-server"),
]

_DEFAULT_PORT = 8080
_DEFAULT_HOST = "127.0.0.1"
_HEALTH_POLL_INTERVAL = 1.0  # seconds between /health checks
_REQUEST_TIMEOUT = 300  # seconds per generation request
_MAX_CONNECT_RETRIES = 3
_CONNECT_RETRY_DELAY = 2.0  # seconds between connection retries

logger = get_logger("comprag.generator")


# ---------------------------------------------------------------------------
# Prompt template loader
# ---------------------------------------------------------------------------


def load_prompt_template(
    template_path: Optional[Union[str, Path]] = None,
) -> str:
    """Load the prompt template from disk.

    Args:
        template_path: Path to the template file. Defaults to
            config/prompt_template.txt relative to project root.

    Returns:
        Template string with {retrieved_chunks} and {query} placeholders.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    if template_path is None:
        template_path = _PROJECT_ROOT / "config" / "prompt_template.txt"
    else:
        template_path = Path(template_path)

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(
    template: str,
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """Inject retrieved chunks and query into the prompt template.

    Args:
        template: The prompt template with {retrieved_chunks} and {query}
            placeholders.
        query: The user's query string.
        retrieved_chunks: List of chunk dicts, each with at least a 'text'
            key. May also have 'metadata' with 'title', 'section', etc.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    # Format chunks as numbered text blocks
    chunk_texts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # Prefix with title/section if available
        prefix_parts = []
        if metadata.get("title"):
            prefix_parts.append(metadata["title"])
        if metadata.get("section"):
            prefix_parts.append(metadata["section"])

        if prefix_parts:
            header = " - ".join(prefix_parts)
            chunk_texts.append(f"[{i}] {header}:\n{text}")
        else:
            chunk_texts.append(f"[{i}] {text}")

    chunks_str = "\n\n".join(chunk_texts)

    # Guard: if template has no {retrieved_chunks} placeholder (e.g. pass1_baseline),
    # skip chunk injection entirely
    if "{retrieved_chunks}" in template:
        result = template.replace("{retrieved_chunks}", chunks_str)
    else:
        result = template

    return result.replace("{query}", query)


# ---------------------------------------------------------------------------
# Server binary discovery
# ---------------------------------------------------------------------------


def find_server_binary(
    binary_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Find the llama-server binary.

    Args:
        binary_path: Explicit path to the binary. If None, searches
            common installation locations.

    Returns:
        Path to the server binary.

    Raises:
        FileNotFoundError: If no server binary is found.
    """
    if binary_path is not None:
        p = Path(binary_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"llama-server binary not found at: {p}")

    for candidate in _DEFAULT_SERVER_BINARY_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "llama-server binary not found. Searched:\n"
        + "\n".join(f"  - {c}" for c in _DEFAULT_SERVER_BINARY_CANDIDATES)
        + "\nSet LLAMA_SERVER_PATH env var or pass binary_path explicitly."
    )


# ---------------------------------------------------------------------------
# LlamaServer — lifecycle + generation
# ---------------------------------------------------------------------------


class LlamaServer:
    """Manages a llama-server subprocess and provides generation API.

    Supports use as a context manager for automatic cleanup::

        with LlamaServer() as srv:
            srv.start("models/test.gguf")
            srv.wait_ready()
            response = srv.generate("Hello world")

    Or explicitly::

        srv = LlamaServer()
        srv.start("models/test.gguf")
        srv.wait_ready()
        ...
        srv.stop()

    Args:
        host: Server bind address. Default 127.0.0.1.
        port: Server port. Default 8080.
        binary_path: Explicit path to llama-server binary.
        server_url: Override the full server URL (e.g. for remote servers).
            If set, start/stop are no-ops — assumes server is externally managed.
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        binary_path: Optional[Union[str, Path]] = None,
        server_url: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.binary_path = binary_path
        self._process: Optional[subprocess.Popen] = None
        self._external = server_url is not None

        if server_url:
            self.base_url = server_url.rstrip("/")
        else:
            self.base_url = f"http://{host}:{port}"

        self._session = requests.Session()
        self._template: Optional[str] = None

        # Default generation params from spec — overridable per call
        self._default_params: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "seed": 42,
        }

        # Register atexit handler to clean up server on normal Python exit
        atexit.register(self._cleanup)

    def __enter__(self) -> "LlamaServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # --- Config loading ---

    def load_generation_params(
        self, config_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Load default generation params from eval_config.yaml.

        Args:
            config_path: Path to config directory. Defaults to project config/.
        """
        try:
            config = load_config("eval_config", config_dir=config_path)
            gen_config = config.get("generation", {})
            for key in ("temperature", "top_p", "max_tokens", "seed"):
                if key in gen_config:
                    self._default_params[key] = gen_config[key]
            logger.info(
                "Loaded generation params from config: %s", self._default_params
            )
        except FileNotFoundError:
            logger.warning(
                "eval_config.yaml not found, using built-in defaults: %s",
                self._default_params,
            )

    def load_template(
        self, template_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Load prompt template from disk. Called automatically on first format.

        Args:
            template_path: Path to template file.
        """
        self._template = load_prompt_template(template_path)
        logger.info("Loaded prompt template (%d chars)", len(self._template))

    # --- Cleanup and port management ---

    def _cleanup(self) -> None:
        """Atexit handler: stop the server if still running."""
        if self.is_running:
            logger.info("Atexit cleanup: stopping llama-server")
            self.stop()

    def _check_port_available(self, port: int) -> None:
        """Check if the target port is free; kill orphaned llama-server if found.

        Uses lsof to identify the process holding the port. Only kills it if
        it's a llama-server process — refuses to kill anything else.

        Args:
            port: Port number to check.

        Raises:
            RuntimeError: If the port is occupied by a non-llama-server process.
        """
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return  # Port is free

            pids = result.stdout.strip().split("\n")
            own_pid = os.getpid()
            for pid_str in pids:
                pid = int(pid_str.strip())
                # Skip our own process (lsof can report the runner
                # if it had an HTTP connection to a previous server)
                if pid == own_pid:
                    continue
                # Identify the process
                try:
                    cmdline_path = f"/proc/{pid}/cmdline"
                    with open(cmdline_path, "r") as f:
                        cmdline = f.read().replace("\x00", " ").strip()
                except (FileNotFoundError, PermissionError):
                    cmdline = ""

                if "llama-server" in cmdline or "llama_server" in cmdline:
                    logger.warning(
                        "Killed orphaned llama-server (PID %d) on port %d",
                        pid,
                        port,
                    )
                    try:
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(0.5)
                    except (ProcessLookupError, PermissionError):
                        pass
                else:
                    raise RuntimeError(
                        f"Port {port} is occupied by PID {pid} ({cmdline[:100]}), "
                        f"which is not a llama-server. Free the port manually."
                    )
        except FileNotFoundError:
            # lsof not installed — fall back to /proc/net/tcp check
            pass
        except subprocess.TimeoutExpired:
            pass

    # --- Server lifecycle ---

    def start(
        self,
        model_path: Union[str, Path],
        port: Optional[int] = None,
        n_gpu_layers: int = -1,
        **kwargs: Any,
    ) -> None:
        """Start the llama-server subprocess.

        Args:
            model_path: Path to the GGUF model file.
            port: Override port (updates self.port and base_url).
            n_gpu_layers: Number of layers to offload to GPU. -1 means all.
                Set to 0 for CPU-only.
            **kwargs: Additional llama-server CLI arguments as key=value.
                Underscores in keys are converted to hyphens.
                Example: ctx_size=4096 becomes --ctx-size 4096.

        Raises:
            FileNotFoundError: If model or server binary not found.
            RuntimeError: If server is already running.
        """
        if self._external:
            logger.info(
                "External server mode — skipping start (url=%s)", self.base_url
            )
            return

        # Check for orphaned llama-server on the target port
        target_port = port if port is not None else self.port
        self._check_port_available(target_port)

        if self._process is not None and self._process.poll() is None:
            raise RuntimeError(
                "Server already running (PID %d). Call stop() first.",
                self._process.pid,
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if port is not None:
            self.port = port
            self.base_url = f"http://{self.host}:{port}"

        binary = find_server_binary(
            self.binary_path or os.environ.get("LLAMA_SERVER_PATH")
        )

        cmd = [
            str(binary),
            "--model", str(model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(n_gpu_layers),
        ]

        # Add extra kwargs as CLI flags
        for key, value in kwargs.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        logger.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # New process group for clean kill
            )
        except OSError as e:
            raise RuntimeError(f"Failed to start llama-server: {e}") from e

        logger.info("llama-server started (PID %d)", self._process.pid)

    def stop(self) -> None:
        """Stop the llama-server subprocess cleanly.

        Sends SIGTERM first, waits up to 10 seconds, then SIGKILL if needed.
        No-op if server is externally managed or not running.
        """
        if self._external:
            logger.debug("External server mode — skipping stop")
            return

        if self._process is None:
            return

        if self._process.poll() is not None:
            logger.info(
                "Server already exited (PID %d, returncode %d)",
                self._process.pid,
                self._process.returncode,
            )
            self._process = None
            return

        pid = self._process.pid
        logger.info("Stopping llama-server (PID %d)...", pid)

        try:
            # Kill the entire process group
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self._process.wait(timeout=10)
            logger.info("Server stopped cleanly (PID %d)", pid)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Server did not exit after SIGTERM, sending SIGKILL (PID %d)", pid
            )
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self._process.wait(timeout=5)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                logger.error("Failed to kill server (PID %d)", pid)

        self._process = None

    @property
    def is_running(self) -> bool:
        """Check if the server subprocess is still running."""
        if self._external:
            return True  # Assume external server is always running
        return self._process is not None and self._process.poll() is None

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Poll the server's /health endpoint until it reports ready.

        Args:
            timeout: Maximum seconds to wait for server readiness.

        Raises:
            TimeoutError: If server doesn't become ready within timeout.
            RuntimeError: If server process exits before becoming ready.
        """
        url = f"{self.base_url}/health"
        deadline = time.monotonic() + timeout
        attempt = 0

        logger.info("Waiting for server readiness at %s (timeout=%.0fs)...", url, timeout)

        while time.monotonic() < deadline:
            # Check if process died
            if not self._external and self._process is not None:
                retcode = self._process.poll()
                if retcode is not None:
                    stderr_out = ""
                    if self._process.stderr:
                        try:
                            stderr_out = self._process.stderr.read().decode(
                                "utf-8", errors="replace"
                            )[-2000:]
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"llama-server exited with code {retcode} before "
                        f"becoming ready.\nStderr (last 2000 chars):\n{stderr_out}"
                    )

            try:
                resp = self._session.get(url, timeout=5)
                if resp.status_code == 200:
                    body = resp.json() if resp.text else {}
                    status = body.get("status", "ok")
                    if status == "ok" or status == "no slot available":
                        logger.info(
                            "Server ready (attempt %d, status=%s)", attempt + 1, status
                        )
                        return
                    # "loading model" status — keep waiting
                    logger.debug("Server status: %s", status)
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass
            except Exception as e:
                logger.debug("Health check error: %s", e)

            attempt += 1
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(
            f"llama-server did not become ready within {timeout:.0f}s "
            f"({attempt} health checks attempted)"
        )

    # --- Generation ---

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Parse the prompt template format into OpenAI chat messages.

        The template uses <|system|>, <|context|>, <|user|>, <|assistant|>
        markers. We combine system + context into the system message and
        user into the user message, which is what llama.cpp expects for
        the /v1/chat/completions endpoint.

        Args:
            prompt: Fully formatted prompt string (template already filled).

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        # Parse sections from the template markers
        system_content = ""
        user_content = ""

        # Split by markers and extract content
        sections = {}
        current_section = None
        lines = prompt.split("\n")

        for line in lines:
            stripped = line.strip()
            if stripped == "<|system|>":
                current_section = "system"
                sections["system"] = []
            elif stripped == "<|context|>":
                current_section = "context"
                sections["context"] = []
            elif stripped == "<|user|>":
                current_section = "user"
                sections["user"] = []
            elif stripped == "<|assistant|>":
                current_section = None  # Stop collecting
            elif current_section is not None:
                sections.setdefault(current_section, []).append(line)

        # Build system message: system instruction + context
        system_parts = []
        if "system" in sections:
            system_parts.append("\n".join(sections["system"]).strip())
        if "context" in sections:
            context_text = "\n".join(sections["context"]).strip()
            if context_text:
                system_parts.append(f"\nContext:\n{context_text}")

        system_content = "\n".join(system_parts).strip()
        user_content = "\n".join(sections.get("user", [])).strip()

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        else:
            # Fallback: if parsing failed, send the whole prompt as user message
            messages.append({"role": "user", "content": prompt})

        return messages

    def generate(
        self, prompt: str, timeout: Optional[float] = None, **params: Any
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            prompt: The fully formatted prompt (template already filled with
                query and chunks). If you have raw query + chunks, use
                generate_rag() instead.
            timeout: Per-request timeout in seconds. If not provided, uses
                the module-level _REQUEST_TIMEOUT (300s).
            **params: Override generation parameters. Supported keys:
                temperature, top_p, max_tokens, seed, stop.

        Returns:
            The generated response text.

        Raises:
            requests.HTTPError: On HTTP errors from the server.
            requests.ConnectionError: If server is unreachable.
            requests.Timeout: If request exceeds timeout.
            RuntimeError: On unexpected response format.
        """
        url = f"{self.base_url}/v1/chat/completions"
        request_timeout = timeout if timeout is not None else _REQUEST_TIMEOUT

        # Merge defaults with per-call overrides
        gen_params = {**self._default_params, **params}

        messages = self._build_messages(prompt)

        payload = {
            "messages": messages,
            "temperature": gen_params.get("temperature", 0.0),
            "top_p": gen_params.get("top_p", 1.0),
            "max_tokens": gen_params.get("max_tokens", 512),
            "seed": gen_params.get("seed", 42),
        }

        if "stop" in gen_params:
            payload["stop"] = gen_params["stop"]

        last_error = None
        for attempt in range(_MAX_CONNECT_RETRIES):
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    timeout=request_timeout,
                )
                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(
                        f"No choices in response: {json.dumps(data, indent=2)}"
                    )

                return choices[0]["message"]["content"]

            except requests.ConnectionError as e:
                last_error = e
                if attempt < _MAX_CONNECT_RETRIES - 1:
                    logger.warning(
                        "Connection error (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_CONNECT_RETRIES,
                        e,
                    )
                    time.sleep(_CONNECT_RETRY_DELAY)
                    continue
            except requests.HTTPError as e:
                logger.error("HTTP error from llama-server: %s", e)
                logger.error("Response body: %s", e.response.text[:2000] if e.response else "N/A")
                raise
            except requests.Timeout as e:
                logger.error(
                    "Request timed out after %ds: %s", request_timeout, e
                )
                raise

        raise ConnectionError(
            f"Failed to connect to llama-server after {_MAX_CONNECT_RETRIES} "
            f"attempts: {last_error}"
        )

    def generate_with_metrics(
        self, prompt: str, timeout: Optional[float] = None, **params: Any
    ) -> dict[str, Any]:
        """Send a completion request and return response with performance metrics.

        Args:
            prompt: The fully formatted prompt.
            timeout: Per-request timeout in seconds. If not provided, uses
                the module-level _REQUEST_TIMEOUT (300s).
            **params: Override generation parameters.

        Returns:
            Dict with keys:
                - text: Generated response text
                - tokens_per_second: Generation throughput
                - time_to_first_token_ms: Time to first token in ms
                - total_inference_time_ms: Total wall-clock time in ms
                - prompt_tokens: Number of prompt tokens (if reported)
                - completion_tokens: Number of completion tokens (if reported)

        Raises:
            Same as generate().
        """
        url = f"{self.base_url}/v1/chat/completions"
        request_timeout = timeout if timeout is not None else _REQUEST_TIMEOUT

        gen_params = {**self._default_params, **params}
        messages = self._build_messages(prompt)

        payload = {
            "messages": messages,
            "temperature": gen_params.get("temperature", 0.0),
            "top_p": gen_params.get("top_p", 1.0),
            "max_tokens": gen_params.get("max_tokens", 512),
            "seed": gen_params.get("seed", 42),
        }

        if "stop" in gen_params:
            payload["stop"] = gen_params["stop"]

        last_error = None
        for attempt in range(_MAX_CONNECT_RETRIES):
            try:
                with Timer() as t:
                    resp = self._session.post(
                        url,
                        json=payload,
                        timeout=request_timeout,
                    )
                    resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(
                        f"No choices in response: {json.dumps(data, indent=2)}"
                    )

                text = choices[0]["message"]["content"]

                # Extract token counts from response
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Compute throughput
                total_ms = t.elapsed_ms
                tokens_per_second = (
                    (completion_tokens / t.elapsed) if t.elapsed > 0 and completion_tokens > 0 else 0.0
                )

                # llama.cpp includes timings in response if available
                timings = data.get("timings", {})
                ttft_ms = timings.get("prompt_ms", 0.0)  # Time processing prompt ~ TTFT
                if not ttft_ms and timings.get("prompt_per_second", 0) > 0:
                    ttft_ms = (prompt_tokens / timings["prompt_per_second"]) * 1000.0

                # Prefer server-reported tokens_per_second if available
                if timings.get("predicted_per_second", 0) > 0:
                    tokens_per_second = timings["predicted_per_second"]

                return {
                    "text": text,
                    "tokens_per_second": round(tokens_per_second, 2),
                    "time_to_first_token_ms": round(ttft_ms, 2),
                    "total_inference_time_ms": round(total_ms, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            except requests.ConnectionError as e:
                last_error = e
                if attempt < _MAX_CONNECT_RETRIES - 1:
                    logger.warning(
                        "Connection error (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_CONNECT_RETRIES,
                        e,
                    )
                    time.sleep(_CONNECT_RETRY_DELAY)
                    continue
            except requests.HTTPError as e:
                logger.error("HTTP error from llama-server: %s", e)
                logger.error("Response body: %s", e.response.text[:2000] if e.response else "N/A")
                raise
            except requests.Timeout as e:
                logger.error(
                    "Request timed out after %ds: %s", request_timeout, e
                )
                raise

        raise ConnectionError(
            f"Failed to connect to llama-server after {_MAX_CONNECT_RETRIES} "
            f"attempts: {last_error}"
        )

    # --- RAG convenience methods ---

    def generate_rag(
        self,
        query: str,
        retrieved_chunks: list[dict],
        timeout: Optional[float] = None,
        **params: Any,
    ) -> str:
        """Format prompt from template + query + chunks, then generate.

        Args:
            query: The user's query.
            retrieved_chunks: List of chunk dicts with 'text' key.
            timeout: Per-request timeout in seconds.
            **params: Override generation parameters.

        Returns:
            Generated response text.
        """
        if self._template is None:
            self.load_template()

        prompt = format_prompt(self._template, query, retrieved_chunks)
        return self.generate(prompt, timeout=timeout, **params)

    def generate_rag_with_metrics(
        self,
        query: str,
        retrieved_chunks: list[dict],
        timeout: Optional[float] = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Format prompt from template + query + chunks, generate with metrics.

        Args:
            query: The user's query.
            retrieved_chunks: List of chunk dicts with 'text' key.
            timeout: Per-request timeout in seconds.
            **params: Override generation parameters.

        Returns:
            Dict with text and performance metrics (see generate_with_metrics).
        """
        if self._template is None:
            self.load_template()

        prompt = format_prompt(self._template, query, retrieved_chunks)
        return self.generate_with_metrics(prompt, timeout=timeout, **params)


# ---------------------------------------------------------------------------
# CLI mode — `python -m comprag.generator`
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI mode. Starts server, runs a test query, stops."""
    parser = argparse.ArgumentParser(
        description="CompRAG generator — llama.cpp server interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- 'serve' sub-command: start server and wait ---
    serve_parser = subparsers.add_parser(
        "serve", help="Start llama-server and keep it running"
    )
    serve_parser.add_argument(
        "--model", required=True, help="Path to GGUF model file"
    )
    serve_parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help="Server port"
    )
    serve_parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1=all, 0=CPU-only)",
    )
    serve_parser.add_argument(
        "--ctx-size", type=int, default=None, help="Context size"
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Startup timeout in seconds",
    )

    # --- 'query' sub-command: send a prompt to running server ---
    query_parser = subparsers.add_parser(
        "query", help="Send a prompt to a running llama-server"
    )
    query_parser.add_argument(
        "--prompt", required=True, help="Prompt text to send"
    )
    query_parser.add_argument(
        "--url",
        default=f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}",
        help="Server URL",
    )
    query_parser.add_argument(
        "--metrics", action="store_true", help="Print performance metrics"
    )

    # --- 'test' sub-command: start, query, stop ---
    test_parser = subparsers.add_parser(
        "test", help="Start server, run test query, stop server"
    )
    test_parser.add_argument(
        "--model", required=True, help="Path to GGUF model file"
    )
    test_parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help="Server port"
    )
    test_parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1=all, 0=CPU-only)",
    )
    test_parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Test prompt",
    )
    test_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Startup timeout in seconds",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        srv = LlamaServer(port=args.port)
        srv.load_generation_params()

        extra_kwargs = {}
        if args.ctx_size is not None:
            extra_kwargs["ctx_size"] = args.ctx_size

        srv.start(
            args.model,
            n_gpu_layers=args.n_gpu_layers,
            **extra_kwargs,
        )

        try:
            srv.wait_ready(timeout=args.timeout)
            print(f"Server ready at {srv.base_url}", file=sys.stderr)
            print("Press Ctrl+C to stop.", file=sys.stderr)
            # Block until interrupted
            while srv.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...", file=sys.stderr)
        finally:
            srv.stop()

    elif args.command == "query":
        srv = LlamaServer(server_url=args.url)
        srv.load_generation_params()

        if args.metrics:
            result = srv.generate_with_metrics(args.prompt)
            print(result["text"])
            print("\n--- Metrics ---", file=sys.stderr)
            print(
                f"Tokens/s:           {result['tokens_per_second']:.2f}",
                file=sys.stderr,
            )
            print(
                f"Time to first token: {result['time_to_first_token_ms']:.1f}ms",
                file=sys.stderr,
            )
            print(
                f"Total inference:     {result['total_inference_time_ms']:.1f}ms",
                file=sys.stderr,
            )
            print(
                f"Prompt tokens:       {result['prompt_tokens']}",
                file=sys.stderr,
            )
            print(
                f"Completion tokens:   {result['completion_tokens']}",
                file=sys.stderr,
            )
        else:
            print(srv.generate(args.prompt))

    elif args.command == "test":
        print(f"Starting llama-server with model: {args.model}", file=sys.stderr)

        with LlamaServer(port=args.port) as srv:
            srv.load_generation_params()
            srv.start(args.model, n_gpu_layers=args.n_gpu_layers)

            print("Waiting for server readiness...", file=sys.stderr)
            srv.wait_ready(timeout=args.timeout)
            print("Server ready.", file=sys.stderr)

            print(f"\nTest prompt: {args.prompt}", file=sys.stderr)
            result = srv.generate_with_metrics(args.prompt)

            print("\n=== Response ===")
            print(result["text"])
            print("\n=== Metrics ===")
            print(f"Tokens/s:            {result['tokens_per_second']:.2f}")
            print(
                f"Time to first token: {result['time_to_first_token_ms']:.1f}ms"
            )
            print(
                f"Total inference:     {result['total_inference_time_ms']:.1f}ms"
            )
            print(f"Prompt tokens:       {result['prompt_tokens']}")
            print(f"Completion tokens:   {result['completion_tokens']}")

        print("\nServer stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
