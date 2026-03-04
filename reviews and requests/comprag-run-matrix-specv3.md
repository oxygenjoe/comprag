# CUMRAG Run Matrix Spec v3 — Claude Code Handoff

## Context

Final run matrix. Changes from v2:

- **MI25 removed entirely.** Deferred to side project. ROCm/gfx900 risk not worth blocking Phase 1.
- **Simulated VRAM tiers added on V100.** Use `CUDA_MEM_LIMIT` to simulate 16GB, 12GB, 8GB constraints on the physical 32GB V100. Produces a clean VRAM constraint gradient with zero hardware confounds.
- **All Phase 1 is now pure CUDA.** One software stack, one Dockerfile target, no driver debugging.

**Scope:** Config doc for Claude Code. Create `config/run_matrix.yaml` and update the runner to consume it.

---

## VRAM Budget Reference

| Model | FP16 | Q8_0 | Q4_K_M |
|-------|------|------|--------|
| Qwen 2.5 14B | ~28GB | ~15GB | ~9GB |
| Mistral NeMo 12B | ~24GB | ~13GB | ~7.5GB |
| Llama 3.1 8B | ~16GB | ~8.5GB | ~5GB |
| Qwen 2.5 7B | ~14GB | ~7.5GB | ~4.5GB |
| SmolLM2 1.7B | ~3.4GB | ~1.8GB | ~1.1GB |

KV cache overhead: ~0.5-1.5GB depending on model.

---

## Design Principles

1. **Maximize quant levels per model per tier.** The thesis is about quantization shifting CU/SK.

2. **V100 32GB is the reference.** Full quant sweep on all models. Superset of every other tier.

3. **Simulated VRAM tiers are valid for CU/SK, not for throughput.** Same silicon, same bandwidth — only VRAM cap changes. Throughput metrics from simulated tiers must be clearly marked as non-representative. Real hardware (1660S, M4000) provides throughput data.

4. **OOM at boundary (~85-100% VRAM) is data.** OOM at 140%+ is a hard skip.

5. **Timeouts are mandatory.** Prevents runner hangs on boundary checks.

6. **All Phase 1 is CUDA.** No ROCm, no CPU tier, no driver debugging.

---

## Simulated VRAM Tiers

The V100 32GB can simulate any VRAM cap via the `CUDA_MEM_LIMIT_0` environment variable (CUDA 12+). The runner sets this before launching llama-server.

```bash
# Example: simulate 16GB V100
CUDA_MEM_LIMIT_0=16384MB llama-server -m model.gguf ...
```

The CUDA runtime returns `cudaErrorMemoryAllocation` at the cap, so OOM boundaries behave identically to a physically constrained card. llama.cpp's partial offload behavior also triggers at the same thresholds.

**What's valid:** CU/SK metrics, OOM boundaries, failure modes, partial offload behavior.

**What's NOT valid:** Throughput (tok/s), latency (TTFT), bandwidth-dependent metrics. The simulated tier still has 900 GB/s HBM2 regardless of VRAM cap.

The runner must tag simulated tiers in the JSONL output so throughput data is clearly distinguished:

```json
{
  "hardware_tier": "v100_16gb_sim",
  "simulated": true,
  "physical_gpu": "V100-SXM2-32GB",
  "vram_limit_mb": 16384
}
```

---

## Phase 1 Run Matrix

### V100 32GB (Physical) — Reference Tier

Full quant sweeps. Superset of all other tiers. No VRAM constraint.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Qwen 2.5 14B | ✅ | ✅ | ✅ | FP16 ~28GB + KV is tight. Attempt. OOM = valid data. |
| Llama 3.1 8B | ✅ | ✅ | ✅ | Baseline/control. Expect worst quant degradation. |
| Mistral NeMo 12B | ✅ | ✅ | ✅ | Full sweep fits comfortably. |
| Qwen 2.5 7B | ✅ | ✅ | ✅ | Baseline for constrained tier comparisons. |
| SmolLM2 1.7B | ✅ | ✅ | ✅ | Floor test. |

**Combos:** 15

### V100 16GB (Simulated) — Mid-Constraint Tier

The interesting boundary: 14B models get squeezed, 7-8B models fit at all quants.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Qwen 2.5 14B | ✅ | ⚠️ boundary | ❌ skip | Q4 ~9GB fits. Q8 ~15GB + KV is boundary. FP16 ~28GB impossible. |
| Llama 3.1 8B | ✅ | ✅ | ⚠️ boundary | Q4+Q8 fit. FP16 ~16GB = exact cap — attempt. |
| Mistral NeMo 12B | ✅ | ⚠️ boundary | ❌ skip | Q4 ~7.5GB fits. Q8 ~13GB + KV is boundary. |
| Qwen 2.5 7B | ✅ | ✅ | ⚠️ boundary | Q4+Q8 fit. FP16 ~14GB + KV is boundary. |
| SmolLM2 1.7B | ✅ | ✅ | ✅ | All fit. |

**Combos:** 14 (9 expected OK, 5 boundary)

### V100 12GB (Simulated) — High-Constraint Tier

14B models are out. Tests how 7-8B models behave with increasing VRAM pressure.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Qwen 2.5 14B | ⚠️ boundary | ❌ skip | ❌ skip | Q4 ~9GB + KV is boundary at 12GB. |
| Llama 3.1 8B | ✅ | ⚠️ boundary | ❌ skip | Q4 fits. Q8 ~8.5GB + KV is boundary. |
| Mistral NeMo 12B | ✅ | ❌ skip | ❌ skip | Q4 ~7.5GB fits. Q8 ~13GB impossible. |
| Qwen 2.5 7B | ✅ | ✅ | ❌ skip | Q4+Q8 fit. FP16 ~14GB impossible. |
| SmolLM2 1.7B | ✅ | ✅ | ✅ | All fit. |

**Combos:** 10 (7 expected OK, 3 boundary)

### V100 8GB (Simulated) — Matches M4000 Physical VRAM

Direct comparison with M4000: same VRAM cap, vastly different bandwidth (900 vs 192 GB/s). If CU/SK is identical, bandwidth doesn't affect output quality. If tok/s differs (it will, massively), that isolates bandwidth's effect on throughput.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Llama 3.1 8B | ✅ | ⚠️ boundary | ❌ skip | Same as M4000 budget. |
| Qwen 2.5 7B | ✅ | ⚠️ boundary | ❌ skip | Same as M4000 budget. |
| SmolLM2 1.7B | ✅ | ✅ | ❌ skip | No FP16 — matching M4000 constraint for comparison. |

**Combos:** 6 (4 expected OK, 2 boundary)

### 1660 Super 6GB (Physical) — Consumer VRAM-Constrained

Headless Linux mandatory.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Llama 3.1 8B | ✅ | ❌ skip | ❌ skip | Q4 ~5GB + KV fits headless. Q8 at 8.5GB is 140% — hard skip. |
| Qwen 2.5 7B | ✅ | ❌ skip | ❌ skip | Q4 ~4.5GB fits. Q8 at 7.5GB is 125% — hard skip. |
| SmolLM2 1.7B | ✅ | ✅ | ✅ | All fit. Full quant sweep. |

**Combos:** 5

### M4000 8GB (Physical) — Maxwell Legacy

No tensor cores. No native FP16. Q4/Q8 only.

| Model | Q4_K_M | Q8_0 | FP16 | Notes |
|-------|--------|------|------|-------|
| Llama 3.1 8B | ✅ | ⚠️ boundary | ❌ skip | Q4 fits. Q8 ~8.5GB is boundary. |
| Qwen 2.5 7B | ✅ | ⚠️ boundary | ❌ skip | Q4 fits. Q8 ~7.5GB + KV is boundary. |
| SmolLM2 1.7B | ✅ | ✅ | ❌ skip | No FP16 per Maxwell limitation. |

**Combos:** 6 (4 expected OK, 2 boundary)

---

## Cross-Tier Comparison Matrix

The experimental design enables these comparisons:

### 1. VRAM Constraint Effect on CU/SK (same silicon)
V100 32GB → 16GB → 12GB → 8GB simulated. Same bandwidth, same compute. Any CU/SK delta is purely VRAM-attributable. Expected finding: CU/SK doesn't change above the "fits" threshold.

### 2. Hardware Effect on CU/SK (same VRAM)
V100 8GB sim vs M4000 8GB physical. Same VRAM budget, different bandwidth (900 vs 192 GB/s), different architecture (Volta vs Maxwell). Expected finding: CU/SK identical, tok/s dramatically different.

### 3. Quantization Effect on CU/SK (same hardware, same model)
Q4 vs Q8 vs FP16 on V100 32GB for each model. The core thesis test. Expected finding: Q4 has higher CU and lower SK than FP16 because quantization weakens parametric confidence.

### 4. Architecture Effect on CU/SK (same quant, same hardware)
Qwen 14B Q4 vs Llama 8B Q4 vs Mistral 12B Q4 on V100. Same quant, different architectures. Shows whether some architectures inherently defer to retrieval more.

---

## V100 Superset Verification

| Combo | 1660S | M4000 | V100 6GB sim | V100 8GB sim | V100 12GB sim | V100 16GB sim | V100 32GB |
|-------|-------|-------|-------------|-------------|--------------|--------------|-----------|
| Qwen 14B Q4 | — | — | — | — | ⚠️ | ✅ | ✅ |
| Qwen 14B Q8 | — | — | — | — | — | ⚠️ | ✅ |
| Qwen 14B FP16 | — | — | — | — | — | — | ✅ |
| Llama 8B Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Llama 8B Q8 | — | ⚠️ | — | ⚠️ | ⚠️ | ✅ | ✅ |
| Llama 8B FP16 | — | — | — | — | — | ⚠️ | ✅ |
| Mistral 12B Q4 | — | — | — | — | ✅ | ✅ | ✅ |
| Mistral 12B Q8 | — | — | — | — | — | ⚠️ | ✅ |
| Mistral 12B FP16 | — | — | — | — | — | — | ✅ |
| Qwen 7B Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen 7B Q8 | — | ⚠️ | — | ⚠️ | ✅ | ✅ | ✅ |
| Qwen 7B FP16 | — | — | — | — | — | ⚠️ | ✅ |
| SmolLM2 Q4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SmolLM2 Q8 | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| SmolLM2 FP16 | ✅ | — | ✅ | — | ✅ | ✅ | ✅ |

✅ Superset confirmed. Every combo on any tier has a V100 32GB counterpart.

---

## Phase 2 (Deferred)

| Item | Rationale |
|------|-----------|
| MI25 ROCm | Side project. "Can AI debug a dead architecture?" narrative. |
| Phi-4 14B on V100 | Architecture comparison if Qwen 14B results are interesting. |
| Gemma 2 9B, GLM-4 9B | Additional architectures. |
| CPU/Optane | Throughput floor + Optane weight storage A/B test. |
| FPGA/BitNet | Separate experiment. |
| Full dataset runs | num_queries=null backfill on V100 if reviewers ask. |

---

## Summary: Phase 1 Totals

| Tier | Combos | Expected OK | Boundary | Runs (×3 datasets ×3 seeds) |
|------|--------|-------------|----------|----------------------------|
| V100 32GB | 15 | 14 | 1 | 135 |
| V100 16GB sim | 14 | 9 | 5 | 126 |
| V100 12GB sim | 10 | 7 | 3 | 90 |
| V100 8GB sim | 6 | 4 | 2 | 54 |
| V100 6GB sim | 5 | 5 | 0 | 45 |
| 1660S 6GB | 5 | 5 | 0 | 45 |
| M4000 8GB | 6 | 4 | 2 | 54 |
| **Total** | **61** | **48** | **13** | **549** |

At 500 queries/dataset/run = 4,500 inferences per successful combo.

All V100 tiers (physical + simulated) run sequentially on one card. Simulated tiers add no hardware cost, only time. Since it's all the same physical GPU at ~40-80 tok/s average, the V100 block (45 combos) should complete in ~5-7 days total.

1660S and M4000 run in parallel on separate machines. ~4-7 days each.

**Total Phase 1 wall-clock time: ~7-10 days** (V100 is the bottleneck, other GPUs run in parallel).

---

## Task for Claude Code

### 1. Create `config/run_matrix.yaml`

```yaml
# CUMRAG Phase 1 Run Matrix v3
# All CUDA. MI25/ROCm deferred. Simulated VRAM tiers on V100.
# V100 32GB is superset of all tiers.

phase: 1

eval_config:
  timeout_load_s: 120
  timeout_gen_s: 30
  max_retries: 1

tiers:
  v100_32gb:
    description: "Reference Tier — V100 SXM2 32GB, no VRAM constraint"
    physical_gpu: V100-SXM2-32GB
    simulated: false
    models:
      - name: Qwen2.5-14B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: Mistral-NeMo-12B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]

  v100_16gb_sim:
    description: "V100 32GB constrained to 16GB via CUDA_MEM_LIMIT"
    physical_gpu: V100-SXM2-32GB
    simulated: true
    vram_limit_mb: 16384
    models:
      - name: Qwen2.5-14B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: Mistral-NeMo-12B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]

  v100_12gb_sim:
    description: "V100 32GB constrained to 12GB via CUDA_MEM_LIMIT"
    physical_gpu: V100-SXM2-32GB
    simulated: true
    vram_limit_mb: 12288
    models:
      - name: Qwen2.5-14B-Instruct
        quants: [Q4_K_M]
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: Mistral-NeMo-12B-Instruct
        quants: [Q4_K_M]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]

  v100_8gb_sim:
    description: "V100 32GB constrained to 8GB — matches M4000 physical VRAM"
    physical_gpu: V100-SXM2-32GB
    simulated: true
    vram_limit_mb: 8192
    models:
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0]

  v100_6gb_sim:
    description: "V100 32GB constrained to 6GB — matches 1660S physical VRAM"
    physical_gpu: V100-SXM2-32GB
    simulated: true
    vram_limit_mb: 6144
    models:
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]

  1660s:
    description: "Consumer Tier — 1660 Super 6GB. MUST run headless."
    physical_gpu: GTX-1660-Super-6GB
    simulated: false
    models:
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0, FP16]

  m4000:
    description: "Maxwell Legacy Tier — Quadro M4000 8GB. No FP16."
    physical_gpu: Quadro-M4000-8GB
    simulated: false
    models:
      - name: Llama-3.1-8B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: Qwen2.5-7B-Instruct
        quants: [Q4_K_M, Q8_0]
      - name: SmolLM2-1.7B-Instruct
        quants: [Q4_K_M, Q8_0]

datasets: [rgb, nq, halueval]
runs_per_combo: 3
num_queries: 500
```

### 2. Implement VRAM limit in runner.py

When a tier has `simulated: true` and `vram_limit_mb` set, the runner must set the environment variable before launching llama-server:

```python
import os

def apply_vram_limit(tier_config: dict):
    """Set CUDA_MEM_LIMIT for simulated VRAM tiers."""
    if tier_config.get('simulated') and tier_config.get('vram_limit_mb'):
        limit = tier_config['vram_limit_mb']
        os.environ['CUDA_MEM_LIMIT_0'] = f"{limit}MB"
        logger.info(f"VRAM limit set to {limit}MB (simulated tier)")
    else:
        # Clear any previous limit
        os.environ.pop('CUDA_MEM_LIMIT_0', None)
```

Call this BEFORE starting llama-server for each tier.

**Critical:** The env var must be set in the parent process environment before the subprocess launch. llama-server inherits it. Clear it between tier switches.

### 3. Tag simulated tiers in JSONL output

Add these fields to every result record:

```json
{
  "hardware_tier": "v100_16gb_sim",
  "hardware_meta": {
    "physical_gpu": "V100-SXM2-32GB",
    "simulated": true,
    "vram_limit_mb": 16384,
    "driver": "CUDA 12.4",
    "framework": "llama.cpp b4567",
    "os": "Ubuntu 24.04",
    "note": "Throughput metrics not representative — simulated VRAM constraint only"
  }
}
```

### 4. Add timeout support to runner.py

- **Load timeout:** Health check within `timeout_load_s` or kill + log `TIMEOUT_LOAD`.
- **Generation timeout:** Per-inference `timeout_gen_s` or kill request + log `TIMEOUT_GEN`. Continue to next query.
- **Max retries:** Retry failed loads `max_retries` times. Log failure mode and advance.

### 5. Add headless check for 1660S

```python
def check_headless():
    """Warn if a display server is eating VRAM."""
    display = os.environ.get('DISPLAY')
    wayland = os.environ.get('WAYLAND_DISPLAY')
    if display or wayland:
        logger.warning(
            "Display server detected (DISPLAY=%s, WAYLAND=%s). "
            "On 6GB GPUs, the compositor reserves 0.5-1.0GB VRAM. "
            "Run headless for accurate results.", display, wayland
        )
```

### 6. Add `--matrix` CLI flag

```bash
# Run all combos for a tier
python -m comprag.runner --matrix config/run_matrix.yaml --hardware-tier v100_32gb

# Run simulated tier
python -m comprag.runner --matrix config/run_matrix.yaml --hardware-tier v100_16gb_sim

# Per-model still works
python -m comprag.runner --model models/smollm2-1.7b-instruct-q4_k_m.gguf --hardware-tier m4000
```

### 7. Do NOT change

- models.yaml, hardware.yaml, eval_config.yaml
- Evaluation logic, metrics computation
- Prompt template, retrieval pipeline
- Existing JSONL schema fields (only ADD simulated/vram_limit fields)
- Existing failure modes (only ADD TIMEOUT_LOAD, TIMEOUT_GEN)
