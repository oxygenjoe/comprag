"""Shared utilities for the CUMRAG eval harness.

Provides JSONL I/O, timing utilities, hardware metadata collection,
resource monitoring, YAML config loading, reproducibility seed setter,
content-addressed collection naming, and logging setup. Importable as
module; also runnable as CLI via `python -m comprag.utils` to print
hardware info.
"""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_log_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    name: str = "comprag",
) -> logging.Logger:
    """Configure project-wide logging.

    Args:
        level: Logging level (default INFO).
        log_file: Optional file path to also write logs to.
        name: Logger name (default 'comprag').

    Returns:
        Configured logger instance.
    """
    global _log_initialized

    logger = logging.getLogger(name)

    if _log_initialized:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(fh)

    _log_initialized = True
    return logger


def get_logger(name: str = "comprag") -> logging.Logger:
    """Get a child logger under the comprag namespace.

    If setup_logging() hasn't been called yet, initializes with defaults.
    """
    global _log_initialized
    if not _log_initialized:
        setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# JSONL I/O — append-safe, crash-safe (write + flush per record)
# ---------------------------------------------------------------------------


def append_jsonl(filepath: Union[str, Path], record: dict) -> None:
    """Append a single JSON record to a JSONL file.

    Writes one JSON object per line, immediately flushes to disk for
    crash safety. Creates parent directories if they don't exist.

    Args:
        filepath: Path to the JSONL file (created if absent).
        record: Dict to serialize as one JSON line.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(filepath: Union[str, Path]) -> Generator[dict, None, None]:
    """Read a JSONL file, yielding one dict per line.

    Skips blank lines. Raises on malformed JSON with line number context.

    Args:
        filepath: Path to the JSONL file.

    Yields:
        Parsed dict for each non-blank line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: On malformed JSON (with line number).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Malformed JSON at line {line_num} in {filepath}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that records wall-clock time in seconds.

    Usage::

        with Timer() as t:
            do_work()
        print(f"Took {t.elapsed:.3f}s ({t.elapsed_ms:.1f}ms)")

    Attributes:
        elapsed: Wall-clock seconds (float). Set after exiting context.
        elapsed_ms: Wall-clock milliseconds (float).
        start: Start timestamp (time.perf_counter).
        end: End timestamp (time.perf_counter).
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        self.elapsed_ms = self.elapsed * 1000.0


@contextmanager
def timer(label: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """Context manager that optionally logs wall-clock time.

    Args:
        label: Optional label for the log message.
        logger: Logger to use. If None and label is set, uses default.

    Yields:
        Timer instance with .elapsed and .elapsed_ms attributes.
    """
    t = Timer()
    t.start = time.perf_counter()
    try:
        yield t
    finally:
        t.end = time.perf_counter()
        t.elapsed = t.end - t.start
        t.elapsed_ms = t.elapsed * 1000.0
        if label is not None:
            log = logger or get_logger("comprag.timer")
            log.info("%s: %.3fs (%.1fms)", label, t.elapsed, t.elapsed_ms)


# ---------------------------------------------------------------------------
# Config loader (YAML)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_config(name: str, config_dir: Optional[Union[str, Path]] = None) -> dict:
    """Load a YAML config file from the config directory.

    Args:
        name: Config file name (e.g. 'models.yaml' or just 'models').
        config_dir: Override config directory (defaults to <project>/config/).

    Returns:
        Parsed YAML contents as a dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    base = Path(config_dir) if config_dir else _CONFIG_DIR

    # Allow passing with or without .yaml extension
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"

    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Content-addressed collection naming
# ---------------------------------------------------------------------------


def make_collection_name(
    dataset: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Generate a deterministic, versioned ChromaDB collection name.

    Encodes the indexing parameters into the collection name via a SHA256
    hash prefix. This prevents stale index reuse when any parameter changes.

    Args:
        dataset: Dataset identifier (e.g. "rgb_noise_robustness", "nq_wiki").
        embedding_model: Sentence-transformers model name (e.g. "all-MiniLM-L6-v2").
        chunk_size: Chunk size in whitespace words.
        chunk_overlap: Chunk overlap in whitespace words.

    Returns:
        Collection name in the format ``comprag_{dataset}_{chunk_size}w_{hash[:8]}``.

    Examples:
        >>> make_collection_name("rgb_noise_robustness", "all-MiniLM-L6-v2", 300, 64)
        'comprag_rgb_noise_robustness_300w_...'
    """
    params = f"{dataset}|{embedding_model}|{chunk_size}|{chunk_overlap}"
    param_hash = hashlib.sha256(params.encode()).hexdigest()[:8]
    return f"comprag_{dataset}_{chunk_size}w_{param_hash}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets Python's random module and, if available, numpy's random state.

    Args:
        seed: Seed value (default 42, matching spec).
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Hardware metadata collection
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], default: str = "unknown") -> str:
    """Run a subprocess command, return stripped stdout or default on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = result.stdout.strip()
        return out if out else default
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return default


def _get_gpu_info() -> str:
    """Detect GPU model string via nvidia-smi or rocm-smi."""
    # Try NVIDIA first
    gpu = _run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        default="",
    )
    if gpu:
        # nvidia-smi may return multiple lines for multi-GPU; take first
        return gpu.split("\n")[0].strip()

    # Try AMD ROCm
    rocm = _run_cmd(["rocm-smi", "--showproductname"], default="")
    if rocm:
        # Parse the product name from rocm-smi output
        for line in rocm.split("\n"):
            if "GPU" in line and ":" in line:
                return line.split(":", 1)[1].strip()
        return rocm.split("\n")[0].strip()

    return "No GPU detected"


def _get_driver_info() -> str:
    """Get CUDA or ROCm driver version string."""
    # Try CUDA version via nvcc
    nvcc_out = _run_cmd(["nvcc", "--version"], default="")
    if nvcc_out:
        for line in nvcc_out.split("\n"):
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.4, V12.4.131"
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].strip().rstrip(",").split(",")[0].strip()
                    return f"CUDA {version}"

    # Fallback: nvidia-smi driver version
    driver = _run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        default="",
    )
    if driver:
        return f"NVIDIA Driver {driver.split(chr(10))[0].strip()}"

    # Try ROCm
    rocm = _run_cmd(["rocm-smi", "--showdriverversion"], default="")
    if rocm:
        for line in rocm.split("\n"):
            if "driver" in line.lower() and ":" in line:
                return f"ROCm {line.split(':', 1)[1].strip()}"

    return "CPU-only (no GPU driver)"


def _get_framework_info() -> str:
    """Detect llama.cpp version by running the server binary with --version."""
    # Common llama.cpp binary locations
    candidates = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path.home() / "llama.cpp" / "build" / "bin" / "server",
        Path("/usr/local/bin/llama-server"),
    ]

    for binary in candidates:
        if binary.exists():
            version_out = _run_cmd([str(binary), "--version"], default="")
            if version_out:
                # Parse build info, typically "version: <number> (commit hash)"
                return f"llama.cpp {version_out.split(chr(10))[0].strip()}"
            # If --version doesn't work, at least confirm it exists
            return f"llama.cpp (at {binary})"

    return "llama.cpp (not found)"


def _get_cpu_info() -> str:
    """Get CPU model string."""
    # Try lscpu
    lscpu = _run_cmd(["lscpu"], default="")
    if lscpu:
        for line in lscpu.split("\n"):
            if "Model name" in line and ":" in line:
                return line.split(":", 1)[1].strip()

    # Fallback: /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, IOError):
        pass

    return platform.processor() or "unknown"


def _get_os_info() -> str:
    """Get OS description string."""
    # Try lsb_release first for distro name
    lsb = _run_cmd(["lsb_release", "-ds"], default="")
    if lsb and lsb != "unknown":
        return lsb.strip('"')

    # Fallback: /etc/os-release
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except (OSError, IOError):
        pass

    return f"{platform.system()} {platform.release()}"


def _get_ram_total_mb() -> int:
    """Get total system RAM in MB via /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value in kB
                    kb = int(line.split()[1])
                    return kb // 1024
    except (OSError, IOError, ValueError):
        pass
    return 0


def _get_vram_total_mb() -> int:
    """Get total GPU VRAM in MB via nvidia-smi."""
    vram = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        default="",
    )
    if vram:
        try:
            return int(vram.split("\n")[0].strip())
        except (ValueError, IndexError):
            pass
    return 0


def get_hardware_meta() -> dict:
    """Collect hardware metadata matching the spec's hardware_meta schema.

    Returns a dict with keys: gpu, driver, framework, os.
    These populate the hardware_meta field in result JSONL records.

    Returns:
        Dict matching::

            {
                "gpu": "NVIDIA Tesla V100-SXM2-32GB",
                "driver": "CUDA 12.4",
                "framework": "llama.cpp b4567",
                "os": "Ubuntu 24.04"
            }
    """
    return {
        "gpu": _get_gpu_info(),
        "driver": _get_driver_info(),
        "framework": _get_framework_info(),
        "os": _get_os_info(),
    }


def get_hardware_full() -> dict:
    """Extended hardware info including CPU, RAM, VRAM. For diagnostics."""
    meta = get_hardware_meta()
    meta["cpu"] = _get_cpu_info()
    meta["ram_total_mb"] = _get_ram_total_mb()
    meta["vram_total_mb"] = _get_vram_total_mb()
    return meta


# ---------------------------------------------------------------------------
# Resource monitor — current VRAM and RAM usage
# ---------------------------------------------------------------------------


def get_vram_usage_mb() -> Optional[int]:
    """Get current GPU VRAM usage in MB via nvidia-smi.

    Returns:
        VRAM used in MB, or None if nvidia-smi is unavailable.
    """
    used = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        default="",
    )
    if used:
        try:
            return int(used.split("\n")[0].strip())
        except (ValueError, IndexError):
            pass
    return None


def get_ram_usage_mb() -> Optional[int]:
    """Get current system RAM usage in MB via /proc/meminfo.

    Computes used = total - available (MemAvailable accounts for
    buffers/cache properly on modern kernels).

    Returns:
        RAM used in MB, or None if /proc/meminfo is unreadable.
    """
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem[key] = int(parts[1])  # kB

        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        if total > 0:
            return (total - available) // 1024
    except (OSError, IOError, ValueError):
        pass
    return None


def get_resource_snapshot() -> dict:
    """Get current resource usage snapshot.

    Returns:
        Dict with vram_usage_mb and ram_usage_mb (values may be None).
    """
    return {
        "vram_usage_mb": get_vram_usage_mb(),
        "ram_usage_mb": get_ram_usage_mb(),
    }


# ---------------------------------------------------------------------------
# CLI mode — `python -m comprag.utils`
# ---------------------------------------------------------------------------


def _print_hardware_info() -> None:
    """Print hardware info to stdout (CLI mode)."""
    print("=" * 60)
    print("CUMRAG Hardware Info")
    print("=" * 60)

    info = get_hardware_full()
    for key, value in info.items():
        label = key.replace("_", " ").title()
        print(f"  {label:.<30} {value}")

    print()
    print("Resource Usage:")
    resources = get_resource_snapshot()
    for key, value in resources.items():
        label = key.replace("_", " ").title()
        display = f"{value} MB" if value is not None else "N/A"
        print(f"  {label:.<30} {display}")

    print()
    print("hardware_meta (for JSONL):")
    meta = get_hardware_meta()
    print(json.dumps(meta, indent=2))


def main() -> None:
    """Entry point for CLI mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CUMRAG utilities — print hardware info and run diagnostics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output hardware_meta as a single JSON line (for piping)",
    )
    parser.add_argument(
        "--resources",
        action="store_true",
        help="Include current resource usage in JSON output",
    )
    args = parser.parse_args()

    if args.json:
        output = get_hardware_meta()
        if args.resources:
            output["resources"] = get_resource_snapshot()
        print(json.dumps(output, ensure_ascii=False))
    else:
        _print_hardware_info()


if __name__ == "__main__":
    main()
