"""llama.cpp process manager.

Starts, monitors, and stops a llama-server subprocess with locked parameters.
"""

import logging
import signal
import subprocess
import time
from typing import Optional

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Locked parameters — not configurable.
N_GPU_LAYERS = -1
SEED = 42
TEMPERATURE = 0.0

# Server binary name.
LLAMA_SERVER_BIN = "llama-server"

# Health poll interval in seconds.
HEALTH_POLL_INTERVAL = 1.0


class LlamaCppServerError(Exception):
    """Raised when the llama.cpp server fails to start or respond."""


class LlamaCppServer:
    """Manages a llama-server subprocess lifecycle.

    Usage::

        with LlamaCppServer("/path/to/model.gguf", port=8080) as srv:
            # server is ready, POST to http://localhost:8080/v1/chat/completions
            ...
    """

    def __init__(self, model_path: str, port: int = 8080) -> None:
        self.model_path = model_path
        self.port = port
        self.proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        """Base URL for the running server."""
        return f"http://localhost:{self.port}"

    def start(self, ctx_len: int = 4096) -> None:
        """Start llama-server with full GPU offload, temp=0, greedy decoding.

        Args:
            ctx_len: Context window size in tokens.

        Raises:
            LlamaCppServerError: If the server process fails to start.
        """
        if self.proc is not None:
            raise LlamaCppServerError("Server already running")

        cmd = [
            LLAMA_SERVER_BIN,
            "--model", self.model_path,
            "--port", str(self.port),
            "--n-gpu-layers", str(N_GPU_LAYERS),
            "--ctx-size", str(ctx_len),
            "--seed", str(SEED),
        ]

        logger.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise LlamaCppServerError(
                f"'{LLAMA_SERVER_BIN}' not found. "
                "Ensure llama.cpp is built and llama-server is on PATH."
            )
        except OSError as e:
            raise LlamaCppServerError(f"Failed to start llama-server: {e}")

    def wait_ready(self, timeout: float = 180.0) -> None:
        """Poll /health until 200 or timeout.

        Args:
            timeout: Maximum seconds to wait for the server to become ready.

        Raises:
            LlamaCppServerError: If the server exits, or timeout is reached.
        """
        if self.proc is None:
            raise LlamaCppServerError("Server not started")

        health_url = f"{self.base_url}/health"
        deadline = time.monotonic() + timeout
        logger.info("Waiting for server health at %s (timeout=%.0fs)", health_url, timeout)

        while time.monotonic() < deadline:
            # Check if process has died.
            ret = self.proc.poll()
            if ret is not None:
                stderr_out = ""
                if self.proc.stderr:
                    stderr_out = self.proc.stderr.read().decode(errors="replace")
                self.proc = None
                raise LlamaCppServerError(
                    f"llama-server exited with code {ret}. stderr: {stderr_out[:500]}"
                )

            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("Server ready on port %d", self.port)
                        return
            except (urllib.error.URLError, OSError):
                pass

            time.sleep(HEALTH_POLL_INTERVAL)

        # Timed out — kill the process.
        self.stop()
        raise LlamaCppServerError(
            f"Server did not become ready within {timeout}s"
        )

    def stop(self) -> None:
        """Stop the server process. SIGTERM, wait 5s, SIGKILL if needed."""
        if self.proc is None:
            return

        pid = self.proc.pid
        logger.info("Stopping llama-server (pid=%d)", pid)

        try:
            self.proc.send_signal(signal.SIGTERM)
        except OSError:
            logger.warning("Failed to send SIGTERM to pid %d", pid)
            self.proc = None
            return

        try:
            self.proc.wait(timeout=5.0)
            logger.info("Server exited cleanly (pid=%d)", pid)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit after SIGTERM, sending SIGKILL (pid=%d)", pid)
            try:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                logger.error("Failed to kill server (pid=%d)", pid)

        self.proc = None

    def __enter__(self) -> "LlamaCppServer":
        """Start the server and wait for it to become ready."""
        self.start()
        self.wait_ready()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Stop the server on context exit."""
        self.stop()
