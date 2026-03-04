#!/usr/bin/env python3
"""Hardware profiling script for CUMRAG eval harness.

Detects GPU/CPU/RAM/OS specs, matches detected hardware to a tier from
hardware.yaml, and outputs the hardware_meta dict matching the spec schema.
Optionally writes detected specs back to hardware.yaml.

CLI usage:
    python scripts/profile_hardware.py                  # Pretty-print profile
    python scripts/profile_hardware.py --json            # JSON output
    python scripts/profile_hardware.py --update-config   # Write to hardware.yaml

Importable:
    from scripts.profile_hardware import profile_hardware, match_tier
"""

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Ensure comprag package is importable when running as standalone script
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

from comprag.utils import (  # noqa: E402
    _get_cpu_info,
    _get_driver_info,
    _get_framework_info,
    _get_gpu_info,
    _get_os_info,
    _get_ram_total_mb,
    _get_vram_total_mb,
    _run_cmd,
    get_hardware_full,
    get_hardware_meta,
    get_logger,
)

logger = get_logger("comprag.profile_hardware")

# ---------------------------------------------------------------------------
# Extended detection helpers (beyond what utils.py provides)
# ---------------------------------------------------------------------------


def _get_cpu_cores() -> int:
    """Get physical CPU core count."""
    try:
        lscpu = _run_cmd(["lscpu"], default="")
        for line in lscpu.split("\n"):
            # Match "Core(s) per socket:" line
            if "Core(s) per socket" in line and ":" in line:
                cores_per_socket = int(line.split(":", 1)[1].strip())
                # Get socket count
                for line2 in lscpu.split("\n"):
                    if "Socket(s)" in line2 and ":" in line2:
                        sockets = int(line2.split(":", 1)[1].strip())
                        return cores_per_socket * sockets
                return cores_per_socket
    except (ValueError, IndexError):
        pass

    # Fallback: os.cpu_count() returns logical cores, divide by 2 for HT
    count = os.cpu_count()
    if count:
        return count
    return 0


def _get_cpu_threads() -> int:
    """Get logical thread count."""
    try:
        lscpu = _run_cmd(["lscpu"], default="")
        for line in lscpu.split("\n"):
            if "CPU(s):" in line and "On-line" not in line and "NUMA" not in line:
                return int(line.split(":", 1)[1].strip())
    except (ValueError, IndexError):
        pass

    count = os.cpu_count()
    return count if count else 0


def _get_nvidia_driver_version() -> str:
    """Get NVIDIA driver version string."""
    driver = _run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        default="",
    )
    if driver:
        return driver.split("\n")[0].strip()
    return "unknown"


def _get_cuda_version() -> str:
    """Get CUDA toolkit version."""
    nvcc_out = _run_cmd(["nvcc", "--version"], default="")
    if nvcc_out:
        for line in nvcc_out.split("\n"):
            if "release" in line.lower():
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].strip().rstrip(",").split(",")[0].strip()
                    return version
    # Fallback: nvidia-smi CUDA version
    smi_out = _run_cmd(["nvidia-smi"], default="")
    if smi_out:
        match = re.search(r"CUDA Version:\s*([\d.]+)", smi_out)
        if match:
            return match.group(1)
    return "unknown"


def _get_rocm_version() -> str:
    """Get ROCm version."""
    rocm = _run_cmd(["rocm-smi", "--showdriverversion"], default="")
    if rocm:
        for line in rocm.split("\n"):
            if "driver" in line.lower() and ":" in line:
                return line.split(":", 1)[1].strip()
    # Try rocminfo
    rocminfo = _run_cmd(["rocminfo"], default="")
    if rocminfo:
        for line in rocminfo.split("\n"):
            if "Runtime Version" in line and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def _get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_llama_cpp_version() -> str:
    """Get llama.cpp version with more detail than utils._get_framework_info()."""
    candidates = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path.home() / "llama.cpp" / "build" / "bin" / "server",
        Path("/usr/local/bin/llama-server"),
    ]

    for binary in candidates:
        if binary.exists():
            version_out = _run_cmd([str(binary), "--version"], default="")
            if version_out:
                return version_out.split("\n")[0].strip()

            # Try to get git commit from the llama.cpp repo
            llama_dir = binary.parent.parent.parent
            git_hash = _run_cmd(
                ["git", "-C", str(llama_dir), "rev-parse", "--short", "HEAD"],
                default="",
            )
            if git_hash:
                return f"git-{git_hash}"
            return f"found at {binary}"

    return "not installed"


def _get_kernel_version() -> str:
    """Get Linux kernel version."""
    return platform.release()


def _detect_gpu_framework() -> str:
    """Detect whether CUDA or ROCm is the active GPU framework."""
    # Check nvidia-smi
    nvidia = _run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        default="",
    )
    if nvidia:
        cuda_ver = _get_cuda_version()
        if cuda_ver != "unknown":
            return f"CUDA {cuda_ver}"
        return "CUDA (version unknown)"

    # Check rocm-smi
    rocm = _run_cmd(["rocm-smi", "--showproductname"], default="")
    if rocm:
        rocm_ver = _get_rocm_version()
        if rocm_ver != "unknown":
            return f"ROCm {rocm_ver}"
        return "ROCm (version unknown)"

    return "CPU-only"


# ---------------------------------------------------------------------------
# Full hardware profile
# ---------------------------------------------------------------------------


def profile_hardware() -> dict[str, Any]:
    """Collect comprehensive hardware profile.

    Returns a dict with all detected hardware specs, suitable for
    display, JSON serialization, or writing to hardware.yaml.

    Returns:
        Dict with keys:
            gpu_model, gpu_vram_mb, gpu_driver_version, gpu_framework,
            cpu_model, cpu_cores, cpu_threads, ram_total_mb,
            os_version, kernel_version, python_version,
            llama_cpp_version, hardware_meta
    """
    gpu_model = _get_gpu_info()
    vram_mb = _get_vram_total_mb()
    driver_info = _get_driver_info()
    framework = _detect_gpu_framework()
    cpu_model = _get_cpu_info()
    cpu_cores = _get_cpu_cores()
    cpu_threads = _get_cpu_threads()
    ram_mb = _get_ram_total_mb()
    os_version = _get_os_info()
    kernel = _get_kernel_version()
    python_ver = _get_python_version()
    llama_ver = _get_llama_cpp_version()
    framework_str = _get_framework_info()

    profile = {
        "gpu_model": gpu_model,
        "gpu_vram_mb": vram_mb,
        "gpu_driver_version": driver_info,
        "gpu_framework": framework,
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_total_mb": ram_mb,
        "os_version": os_version,
        "kernel_version": kernel,
        "python_version": python_ver,
        "llama_cpp_version": llama_ver,
        # The spec-compliant hardware_meta dict for JSONL records
        "hardware_meta": {
            "gpu": gpu_model,
            "driver": driver_info,
            "framework": framework_str,
            "os": os_version,
        },
    }

    return profile


# ---------------------------------------------------------------------------
# Tier matching
# ---------------------------------------------------------------------------


def _load_hardware_tiers(
    config_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load hardware tier definitions from hardware.yaml.

    Args:
        config_path: Override path to hardware.yaml.

    Returns:
        Dict mapping tier_id -> tier definition.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "hardware.yaml"

    if not config_path.exists():
        logger.warning("hardware.yaml not found at %s", config_path)
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("hardware_tiers", {})


def match_tier(
    profile: Optional[dict] = None,
    config_path: Optional[Path] = None,
) -> tuple[str, dict]:
    """Match detected hardware to a tier from hardware.yaml.

    Matching logic:
    1. GPU name substring match against tier display_name
    2. CPU name substring match
    3. Falls back to 'cpu' tier if no GPU detected

    Args:
        profile: Hardware profile dict (from profile_hardware()). If None,
                 auto-detects.
        config_path: Override path to hardware.yaml.

    Returns:
        Tuple of (tier_id, tier_definition). If no match, returns
        ('unknown', {}).
    """
    if profile is None:
        profile = profile_hardware()

    tiers = _load_hardware_tiers(config_path)
    if not tiers:
        return ("unknown", {})

    gpu = profile.get("gpu_model", "").lower()
    cpu = profile.get("cpu_model", "").lower()

    # GPU-based matching — check for specific GPU identifiers
    gpu_patterns = {
        "v100": ["v100"],
        "mi25": ["mi25", "vega"],
        "1660s": ["1660 super", "1660s"],
        "fpga": ["fpga", "inspur"],
    }

    for tier_id, patterns in gpu_patterns.items():
        if tier_id in tiers:
            for pattern in patterns:
                if pattern in gpu:
                    return (tier_id, tiers[tier_id])

    # CPU-based matching for non-GPU tiers
    # Check if Optane is present (check mount points)
    if "e5-2667" in cpu or "2667" in cpu:
        # Check for Optane DAX mounts
        optane_present = False
        try:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    if "pmem" in line or "optane" in line.lower():
                        optane_present = True
                        break
        except (OSError, IOError):
            pass

        if optane_present and "optane" in tiers:
            return ("optane", tiers["optane"])
        if "cpu" in tiers:
            return ("cpu", tiers["cpu"])

    # Generic GPU present but not matching known tier
    if gpu and "no gpu" not in gpu.lower():
        # Check display_name of each tier for GPU model substring
        for tier_id, tier_def in tiers.items():
            display = tier_def.get("display_name", "").lower()
            # Extract GPU model tokens from display name and check
            if any(token in gpu for token in display.split() if len(token) > 3):
                return (tier_id, tier_def)

    # Fallback to CPU tier
    if "cpu" in tiers:
        return ("cpu", tiers["cpu"])

    return ("unknown", {})


# ---------------------------------------------------------------------------
# Update hardware.yaml
# ---------------------------------------------------------------------------


def update_hardware_config(
    profile: dict,
    tier_id: str,
    config_path: Optional[Path] = None,
) -> Path:
    """Write detected hardware specs back to hardware.yaml.

    Updates the matched tier's fields with detected values. Does NOT
    overwrite the entire file — reads, modifies the specific tier, writes.

    Args:
        profile: Hardware profile from profile_hardware().
        tier_id: The matched tier ID to update.
        config_path: Override path to hardware.yaml.

    Returns:
        Path to the written config file.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "hardware.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tiers = data.setdefault("hardware_tiers", {})
    tier = tiers.setdefault(tier_id, {})

    # Update with detected values
    gpu_model = profile.get("gpu_model", "")
    vram_mb = profile.get("gpu_vram_mb", 0)
    ram_mb = profile.get("ram_total_mb", 0)
    cpu_model = profile.get("cpu_model", "")
    gpu_framework = profile.get("gpu_framework", "")

    if gpu_model and "no gpu" not in gpu_model.lower():
        vram_str = f"{vram_mb}MB" if vram_mb else tier.get("vram", "N/A")
        tier["vram"] = vram_str
    tier["ram"] = f"{ram_mb}MB ({ram_mb // 1024}GB)" if ram_mb else tier.get("ram", "unknown")
    tier["cpu"] = cpu_model if cpu_model else tier.get("cpu", "unknown")
    tier["status"] = "available"

    # Update software stack with detected framework
    if gpu_framework and gpu_framework != "CPU-only":
        llama_ver = profile.get("llama_cpp_version", "not installed")
        tier["software_stack"] = [
            gpu_framework,
            f"llama.cpp ({llama_ver})",
        ]
    else:
        llama_ver = profile.get("llama_cpp_version", "not installed")
        tier["software_stack"] = [
            f"llama.cpp CPU ({llama_ver})",
        ]

    # Add detected_at timestamp
    from datetime import datetime, timezone

    tier["detected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

    logger.info("Updated hardware.yaml tier '%s' at %s", tier_id, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _format_profile(profile: dict, tier_id: str, tier_def: dict) -> str:
    """Format hardware profile for human-readable display."""
    lines = []
    lines.append("=" * 64)
    lines.append("  CUMRAG Hardware Profile")
    lines.append("=" * 64)
    lines.append("")

    # GPU
    lines.append("  GPU")
    lines.append(f"    Model ............... {profile['gpu_model']}")
    vram = profile["gpu_vram_mb"]
    vram_str = f"{vram} MB ({vram / 1024:.1f} GB)" if vram else "N/A"
    lines.append(f"    VRAM ................ {vram_str}")
    lines.append(f"    Driver .............. {profile['gpu_driver_version']}")
    lines.append(f"    Framework ........... {profile['gpu_framework']}")
    lines.append("")

    # CPU
    lines.append("  CPU")
    lines.append(f"    Model ............... {profile['cpu_model']}")
    lines.append(f"    Cores ............... {profile['cpu_cores']}")
    lines.append(f"    Threads ............. {profile['cpu_threads']}")
    lines.append("")

    # Memory
    lines.append("  Memory")
    ram = profile["ram_total_mb"]
    ram_str = f"{ram} MB ({ram / 1024:.1f} GB)" if ram else "N/A"
    lines.append(f"    RAM Total ........... {ram_str}")
    lines.append("")

    # System
    lines.append("  System")
    lines.append(f"    OS .................. {profile['os_version']}")
    lines.append(f"    Kernel .............. {profile['kernel_version']}")
    lines.append(f"    Python .............. {profile['python_version']}")
    lines.append(f"    llama.cpp ........... {profile['llama_cpp_version']}")
    lines.append("")

    # Tier match
    lines.append("  Matched Tier")
    lines.append(f"    Tier ID ............. {tier_id}")
    if tier_def:
        lines.append(f"    Display Name ........ {tier_def.get('display_name', 'N/A')}")
        lines.append(f"    Status .............. {tier_def.get('status', 'N/A')}")
    else:
        lines.append("    (no matching tier found in hardware.yaml)")
    lines.append("")

    # Spec-compliant hardware_meta
    lines.append("  hardware_meta (for JSONL records):")
    meta = profile["hardware_meta"]
    lines.append(f"    {json.dumps(meta, indent=2).replace(chr(10), chr(10) + '    ')}")
    lines.append("")
    lines.append("=" * 64)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for hardware profiling."""
    parser = argparse.ArgumentParser(
        description="CUMRAG Hardware Profiler — detect GPU/CPU/RAM, match tier, output metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/profile_hardware.py                  # Pretty-print profile
  python scripts/profile_hardware.py --json            # JSON output (pipe-friendly)
  python scripts/profile_hardware.py --update-config   # Write detected specs to hardware.yaml
  python scripts/profile_hardware.py --json --update-config  # Both""",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (single line, suitable for piping)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write detected specs back to config/hardware.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override path to hardware.yaml (default: config/hardware.yaml)",
    )

    args = parser.parse_args()

    # Detect hardware
    profile = profile_hardware()
    tier_id, tier_def = match_tier(profile, config_path=args.config)
    profile["matched_tier"] = tier_id

    if args.update_config:
        config_path = update_hardware_config(profile, tier_id, config_path=args.config)
        if not args.json:
            print(f"Updated {config_path}")

    if args.json:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
    else:
        print(_format_profile(profile, tier_id, tier_def))


if __name__ == "__main__":
    main()
