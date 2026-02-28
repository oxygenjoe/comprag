# HARDWARE.md -- CUMRAG Hardware Reference

All hardware in this project is surplus/eBay-grade. The thesis is that rigorous retrieval
optimization on cheap iron beats throwing money at GPU compute. Every component listed here
was sourced from eBay, surplus liquidators, or local recyclers unless noted otherwise.

---

## Hardware Matrix

| Tier ID  | Hardware                                  | VRAM / RAM                      | Software Stack                          | Status    | Approx. Cost (eBay/Surplus) |
|----------|-------------------------------------------|---------------------------------|-----------------------------------------|-----------|-----------------------------|
| `v100`   | NVIDIA V100 SXM2 32GB in Dell T7820      | 32GB HBM2 / 48GB DDR4 + 384GB Optane | CUDA 12.x, llama.cpp (GGML_CUDA=1)     | Incoming  | ~$300-400 (V100 SXM2), ~$200-350 (T7820), ~$50-80 ea (Optane 128GB) |
| `mi25`   | AMD MI25 16GB                             | 16GB HBM / 32GB DDR4           | ROCm, llama.cpp (GGML_HIPBLAS=1)       | Available | ~$60-100 |
| `1660s`  | NVIDIA 1660 Super 6GB                     | 6GB GDDR6 / 32GB DDR4          | CUDA 12.x, llama.cpp (GGML_CUDA=1)     | Available | ~$80-120 |
| `fpga`   | Inspur FPGA board + 16GB SODIMM           | N/A / 16GB DDR4                 | TBD (BitNet runtime)                    | Planned   | ~$30-60 (surplus Inspur pull) |
| `cpu`    | Intel E5-2667v4 + 32GB DDR4              | N/A / 32GB DDR4                 | llama.cpp CPU (AVX2)                    | Available | ~$20-40 (CPU), ~$30-50 (board + RAM) |
| `optane` | E5-2667v4 + 3x128GB Optane DCPMM 384GB  | N/A / 384GB Optane (App Direct) | llama.cpp CPU (weights on Optane DAX)   | Available | ~$50-80 ea (Optane 128GB DCPMM) |

All prices reflect early-2026 eBay/surplus market. Datacenter pulls fluctuate -- these are
ballpark ranges from completed listings, not BIN aspirational pricing.

---

## Per-Tier Details

### Tier: `v100` -- NVIDIA V100 SXM2 32GB in Dell Precision T7820

The primary compute and inference tier. This is the only tier that can run all models at
all quantization levels, including 14B FP16.

**Host System: Dell Precision T7820**
- CPU: 1x Intel Xeon Gold 6230 (20C/40T, 2.1GHz base / 3.9GHz turbo, Cascade Lake, AVX-512)
- RAM: 3x 16GB DDR4-2666 RDIMM (48GB total, 6-channel capable but 3 DIMMs = 3-channel)
- Storage: 2x 500GB SSD (OS, RAID-1 optional), 1x 2TB HDD (raw corpus)
- PSU: 950W (mandatory -- V100 draws up to 300W alone)
- OS: Ubuntu 24.04 LTS

**GPU: NVIDIA Tesla V100 SXM2 32GB**
- 32GB HBM2 (900 GB/s bandwidth)
- 5120 CUDA cores, 640 Tensor cores
- PCIe Gen3 x16 via SXM2-to-PCIe adapter board
- TDP: 300W

**Cooling: Push-Pull P8 Max Configuration**
- Two P8 Max fans in push-pull configuration straddling the SXM2 adapter heatsink
- Entire assembly sealed with HVAC aluminum tape to force all airflow through the heatsink fins
- **NO fan mounted on top of the adapter** -- top fan creates turbulence that fights the push-pull flow
- Intake fan pushes ambient air through fins; exhaust fan pulls heated air out the rear of the chassis
- Target: keep junction temp under 83C at sustained load

**Persistent Memory: 3x 128GB Intel Optane DCPMM (384GB total)**
- App Direct mode (NOT Memory Mode -- we need explicit placement control)
- Mounted as DAX filesystems at `/mnt/optane0`, `/mnt/optane1`, `/mnt/optane2`
- Primary use: FAISS index (HNSW graph + IVF_PQ inverted lists), chunk text storage
- ~300ns random access latency (vs ~100us for NVMe -- 333x advantage for HNSW pointer-chasing)
- See [Optane tier](#tier-optane----optane-dcpmm-384gb-app-direct-dax) for configuration details

**Memory Hierarchy (V100 tier)**
| Layer | Capacity | Role |
|-------|----------|------|
| V100 VRAM | 32GB HBM2 | Model weights (~17GB Q4_K_M) + KV cache (~14GB) + FAISS coarse quantizer |
| DDR4 | 48GB | OS, FAISS query buffers, working memory |
| Optane | 384GB | Persistent FAISS index + chunk text, mmap'd via DAX |
| SSD | 2x 500GB | OS, model GGUFs, staging area |
| HDD | 2TB | Raw corpus (PDFs, EPUBs, repos, dumps) |

**Sourcing Notes**
- V100 SXM2 32GB: ~$300-400 on eBay (2026). These are datacenter pulls from decomissioned
  DGX-1 and HGX systems. Verify HBM2 health with `nvidia-smi -q -d MEMORY` after install.
  SXM2 cards require an adapter board (Supermicro or generic Chinese PCIe adapter, ~$30-50).
- Dell T7820: ~$200-350 barebones (no CPU/RAM/GPU). Dual-socket Xeon workstation with 950W PSU.
  Common in IT liquidation. Confirm BIOS version supports Cascade Lake before buying.
- Xeon Gold 6230: ~$30-60 (tray pull). 20C/40T with AVX-512. Cascade Lake is the sweet spot --
  Skylake-SP lacks some microcode fixes, Ice Lake is more expensive for marginal gains here.

---

### Tier: `mi25` -- AMD MI25 16GB

Secondary GPU tier for AMD/ROCm validation. The MI25 (Vega 10 based) has 16GB HBM at
roughly half the bandwidth of V100. Main purpose: verify llama.cpp ROCm builds produce
identical outputs to CUDA builds, and benchmark the performance delta.

**Specifications**
- GPU: AMD Radeon Instinct MI25
- Memory: 16GB HBM (High Bandwidth Memory, 1st gen)
- Architecture: Vega 10 (GCN 5.0)
- TDP: 300W
- Interface: PCIe Gen3 x16

**Software Requirements**
- ROCm 6.x (check MI25/Vega10 support -- AMD periodically drops older architectures)
- llama.cpp built with `-DGGML_HIPBLAS=ON`
- `HSA_OVERRIDE_GFX_VERSION=9.0.0` may be needed if ROCm doesn't auto-detect Vega correctly

**Feasibility**
- Comfortable: 7-8B Q4_K_M, 7-8B Q8_0
- Tight: 14B Q4_K_M (weights ~8.9GB + KV cache -- leaves ~5-6GB headroom, workable at short context)
- No: 14B Q8_0/FP16

**Sourcing Notes**
- MI25 16GB: ~$60-100 on eBay. These are datacenter pulls, often from cloud providers that
  decommissioned early Vega compute nodes. Passive cooling variants need a chassis with strong
  front-to-back airflow or an aftermarket blower. Active-cooled variants exist but are rarer.

---

### Tier: `1660s` -- NVIDIA 1660 Super 6GB

The "consumer GPU that could" tier. 6GB GDDR6 is extremely limiting for LLM inference --
this tier exists to test the lower boundary of viable RAG inference and to demonstrate
where small models (1.7B, 3B) are the only option.

**Specifications**
- GPU: NVIDIA GeForce GTX 1660 Super
- Memory: 6GB GDDR6 (192-bit, 336 GB/s)
- Architecture: Turing (TU116), no Tensor cores
- TDP: 125W
- Interface: PCIe Gen3 x16

**Host System** (current staging box, Chinese X99)
- CPU: Intel E5-2667 v4 (8C/16T, 3.2GHz base / 3.6GHz turbo, Broadwell-EP)
- RAM: 32GB DDR4
- OS: Linux Mint 22.3
- Network: Tailscale mesh (100.85.113.86)
- Management: Cockpit on port 9090, SSH on port 22

**Feasibility**
- Comfortable: SmolLM2 1.7B (all quants), BitNet 3B (if runtime supports CUDA)
- Very tight / OOM risk: 7-8B Q4_K_M (~4.7-4.9GB weights, leaves ~1GB for KV cache -- will OOM at any meaningful context length)
- No: anything 9B+

**Expected OOM Combinations**
| Model | Quant | Weights (GB) | Remaining VRAM | Verdict |
|-------|-------|-------------|----------------|---------|
| Llama 3.1 8B | Q4_K_M | 4.9 | ~1.1GB | OOM at >512 ctx tokens |
| Qwen 2.5 7B | Q4_K_M | 4.7 | ~1.3GB | OOM at >512 ctx tokens |
| Gemma 2 9B | Q4_K_M | 5.8 | ~0.2GB | OOM immediately |
| Any 12-14B | Any | 7.4+ | Negative | Impossible |

The 7-8B Q4 runs on this tier will likely crash mid-inference when the KV cache grows beyond
the remaining VRAM. Log these as OOM failures -- they are valid data points for the paper.

**Sourcing Notes**
- 1660 Super: ~$80-120 used. Extremely common on the secondhand market from gaming PC upgrades.
  Verify fan health (these are consumer coolers, not datacenter blowers).

---

### Tier: `cpu` -- Intel E5-2667v4 + 32GB DDR4

Pure CPU inference. No GPU offload. Painfully slow but useful as a baseline to quantify
the GPU advantage, and to demonstrate that RAG faithfulness metrics are hardware-independent
(the model's groundedness should not change based on inference speed).

**Specifications**
- CPU: Intel Xeon E5-2667 v4 (8C/16T, 3.2GHz base / 3.6GHz turbo, Broadwell-EP)
- Instruction set: AVX2 (no AVX-512 -- Broadwell predates it)
- RAM: 32GB DDR4-2400
- Expected throughput: ~2-5 tok/s for 7B Q4_K_M

**Feasibility**
- Primary: 7B Q4_K_M (weights ~4.7GB fit easily in 32GB RAM)
- Slow but functional: 14B Q4_K_M (weights ~8.9GB, fits in RAM, expect ~1-2 tok/s)
- All SmolLM2 1.7B variants

**Sourcing Notes**
- E5-2667 v4: ~$20-40 on eBay. Broadwell-EP Xeons are dirt cheap because they're two
  generations behind on instruction set (no AVX-512). The 3.2GHz base clock partially
  compensates for the lack of wider SIMD.
- Chinese X99 motherboard + DDR4: ~$30-50 as a combo. These are the AliExpress specials
  with varying BIOS quality. They work.

---

### Tier: `optane` -- Optane DCPMM 384GB (App Direct / DAX)

This is not a separate physical machine -- it is the V100 tier's Dell T7820 with weights
memory-mapped from Optane persistent memory instead of loaded into DDR4 or VRAM. The purpose
is to benchmark whether Optane's ~300ns random access latency is viable for LLM weight
access patterns (which are largely sequential/streaming, not random -- so Optane's strength
is actually less relevant here than for FAISS HNSW traversal).

For FAISS index serving, Optane is the primary storage tier and its latency advantage over
NVMe is the entire architectural thesis (see CLAUDE.md, FAISS Index Design).

**Specifications**
- Memory: 3x 128GB Intel Optane DCPMM 2666 (384GB total)
- Mode: App Direct (NOT Memory Mode)
- Access: DAX (Direct Access) filesystem mount -- bypasses page cache, provides direct
  load/store to persistent media
- Latency: ~300ns random read (vs ~100us NVMe, ~70ns DDR4)
- Bandwidth: ~6.6 GB/s read per DIMM (sequential), ~39.6 GB/s aggregate across 6 DIMMs
  (but only 3 populated in this config, so ~19.8 GB/s)

**Configuration**

```bash
# Step 1: Create App Direct goal (destroys existing namespaces)
sudo ipmctl create -goal PersistentMemoryType=AppDirect

# Step 2: Reboot
sudo reboot

# Step 3: Create DAX namespaces
sudo ndctl create-namespace --mode=fsdax --region=region0
sudo ndctl create-namespace --mode=fsdax --region=region1
sudo ndctl create-namespace --mode=fsdax --region=region2

# Step 4: Create filesystems
sudo mkfs.ext4 /dev/pmem0
sudo mkfs.ext4 /dev/pmem1
sudo mkfs.ext4 /dev/pmem2

# Step 5: Mount with DAX option
sudo mkdir -p /mnt/optane{0,1,2}
sudo mount -o dax /dev/pmem0 /mnt/optane0
sudo mount -o dax /dev/pmem1 /mnt/optane1
sudo mount -o dax /dev/pmem2 /mnt/optane2

# Step 6: Add to /etc/fstab for persistence
# /dev/pmem0  /mnt/optane0  ext4  dax,defaults  0  0
# /dev/pmem1  /mnt/optane1  ext4  dax,defaults  0  0
# /dev/pmem2  /mnt/optane2  ext4  dax,defaults  0  0
```

**Verification**

```bash
# Confirm App Direct mode
ipmctl show -memoryresources
# Should show "AppDirectCapacity" near 384GB, "MemoryCapacity" near 0

# Confirm DAX mount
mount | grep pmem
# Should show "dax" in mount options

# Confirm namespace health
ndctl list -N
```

**Feasibility**
- Primary: 7B Q4_K_M with weights mmap'd from Optane DAX mount
- Functional: 14B Q4_K_M (weights fit, throughput limited by Optane bandwidth)
- Real purpose: FAISS HNSW index serving (byte-addressable random access at 300ns)

**Sourcing Notes**
- Intel Optane DCPMM 128GB: ~$50-80 each on eBay (2026). These are server pulls from
  cloud providers that adopted Optane early and then moved on. Verify the DIMM is DCPMM
  (persistent memory), NOT Optane SSD. The DIMMs look like regular DDR4 but are physically
  keyed differently. Requires a Cascade Lake or Ice Lake Xeon platform with DCPMM support
  (Xeon Gold 6230 qualifies).

---

### Tier: `fpga` -- Inspur FPGA Board (BitNet Speculative)

Speculative/experimental tier. The hypothesis is that FPGA fabric can natively implement
BitNet's 1.58-bit ({-1, 0, 1}) ternary arithmetic without the overhead of GPU/CPU
general-purpose compute. This tier exists for future exploration only.

**Specifications**
- Board: Inspur FPGA accelerator (exact model TBD -- surplus datacenter pull)
- Memory: 16GB DDR4 SODIMM (on-board)
- Interface: PCIe
- Target runtime: BitNet 1.58-bit native inference

**Feasibility**
- Only: BitNet 1.58b ~3B parameter model (1.58-bit native, ~0.5GB)
- No standard GGUF models -- FPGA requires custom bitstream, not llama.cpp

**Sourcing Notes**
- Inspur FPGA boards: ~$30-60 on eBay/surplus. These appear in datacenter liquidation lots,
  often with no documentation. Reverse-engineering the PCIe interface and programming the
  bitstream is a non-trivial project. This tier is Phase 3+ and may never materialize --
  included for completeness.

---

## Model-Hardware Feasibility Matrix

Rows are models (with quant), columns are hardware tiers. Entries indicate whether the
combination fits in memory and is expected to produce results.

| Model | Quant | Size (GB) | `v100` | `mi25` | `1660s` | `cpu` | `optane` | `fpga` |
|-------|-------|-----------|--------|--------|---------|-------|----------|--------|
| Qwen 2.5 14B Instruct | Q4_K_M | 8.9 | YES | Tight | No | Slow | Slow | No |
| Qwen 2.5 14B Instruct | Q8_0 | 15.7 | YES | No | No | Slow | Slow | No |
| Qwen 2.5 14B Instruct | FP16 | 29.5 | YES | No | No | No | No | No |
| Phi-4 14B | Q4_K_M | 8.4 | YES | Tight | No | Slow | Slow | No |
| Phi-4 14B | Q8_0 | 14.8 | YES | No | No | Slow | Slow | No |
| Phi-4 14B | FP16 | 27.8 | YES | No | No | No | No | No |
| Mistral NeMo 12B | Q4_K_M | 7.4 | YES | YES | No | Slow | Slow | No |
| Mistral NeMo 12B | Q8_0 | 13.0 | YES | YES | No | Slow | Slow | No |
| Llama 3.1 8B Instruct | Q4_K_M | 4.9 | YES | YES | OOM risk | YES | YES | No |
| Llama 3.1 8B Instruct | Q8_0 | 8.5 | YES | YES | No | YES | YES | No |
| Llama 3.1 8B Instruct | FP16 | 16.1 | YES | No | No | No | No | No |
| Qwen 2.5 7B Instruct | Q4_K_M | 4.7 | YES | YES | OOM risk | YES | YES | No |
| Qwen 2.5 7B Instruct | Q8_0 | 8.1 | YES | YES | No | YES | YES | No |
| Gemma 2 9B Instruct | Q4_K_M | 5.8 | YES | YES | Tight | YES | YES | No |
| Gemma 2 9B Instruct | Q8_0 | 10.1 | YES | YES | No | YES | YES | No |
| GLM-4 9B Chat | Q4_K_M | 5.5 | YES | YES | Tight | YES | YES | No |
| SmolLM2 1.7B Instruct | Q8_0 | 1.8 | YES | YES | YES | YES | YES | No |
| SmolLM2 1.7B Instruct | FP16 | 3.4 | YES | YES | YES | YES | YES | No |
| BitNet 1.58b | 1.58-bit | 0.5 | No* | No* | No* | No* | No* | YES |

**Legend:**
- **YES** -- Fits comfortably, expected to complete inference
- **Tight** -- Fits but with minimal KV cache headroom; may OOM at longer context
- **OOM risk** -- Weights load but KV cache will likely cause OOM mid-inference
- **Slow** -- Fits in RAM, runs on CPU at ~1-5 tok/s; produces valid results
- **No** -- Does not fit or is architecturally incompatible
- **No*** -- BitNet requires a dedicated 1.58-bit runtime, not standard llama.cpp GGUF inference

---

## Build Instructions

### llama.cpp with CUDA (V100, 1660 Super)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# CUDA build
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="70;75" \
  -DCMAKE_BUILD_TYPE=Release

# 70 = V100 (Volta), 75 = 1660 Super (Turing)
# Omit 75 if building exclusively for V100

cmake --build build --config Release -j$(nproc)

# Verify CUDA backend loaded
./build/bin/llama-server --help 2>&1 | grep -i cuda
```

### llama.cpp with ROCm / HIP (MI25)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# ROCm/HIP build
cmake -B build \
  -DGGML_HIPBLAS=ON \
  -DAMDGPU_TARGETS="gfx900" \
  -DCMAKE_BUILD_TYPE=Release

# gfx900 = Vega 10 (MI25)
# If ROCm doesn't auto-detect, set: export HSA_OVERRIDE_GFX_VERSION=9.0.0

cmake --build build --config Release -j$(nproc)

# Verify HIP backend
./build/bin/llama-server --help 2>&1 | grep -i hip
```

### llama.cpp CPU-only (E5-2667v4, Optane)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# CPU-only build (AVX2 for Broadwell)
cmake -B build \
  -DGGML_CUDA=OFF \
  -DGGML_HIPBLAS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# For Optane tier, mmap weights from DAX mount:
./build/bin/llama-server \
  --model /mnt/optane0/models/llama-3.1-8b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 0
```

### Build Notes

- Always rebuild from clean (`rm -rf build/`) when switching backends (CUDA/HIP/CPU).
  CMake caches backend flags aggressively.
- The `-j$(nproc)` flag uses all available cores. On the 8C/16T E5-2667v4 this is fine.
  On the 20C/40T Xeon Gold 6230, compilation finishes in under 2 minutes.
- For V100 on the T7820: CUDA Toolkit 12.x from NVIDIA's official Ubuntu 24.04 repo.
  Do NOT use the distro-packaged CUDA -- it's usually outdated.

---

## Known Gotchas

### Dell T7820 Side Cover Interlock

The T7820 has a chassis intrusion switch. **The system will not POST with the side cover
removed.** This is a BIOS-level interlock, not a software setting. If you need to run with
the cover off (e.g., for thermal testing or cable routing), you must either:
- Tape the intrusion switch in the "closed" position
- Disable chassis intrusion in BIOS (if available -- some T7820 BIOS versions hide this)
- Just put the damn cover back on

This will bite you during initial V100 adapter installation when you're testing POST cycles
with the cover off.

### V100 SXM2 Cooling: Push-Pull P8 Max, NO Fan on Top

The V100 SXM2 adapter uses a custom heatsink. Cooling configuration:
- **Two P8 Max fans** in push-pull (one intake, one exhaust) straddling the heatsink
- **HVAC aluminum tape** sealing all gaps between fans and heatsink to prevent air bypass
- **Absolutely NO fan mounted on top of the adapter heatsink.** A top fan creates a
  pressure zone that fights the push-pull airflow, reducing cooling efficiency and creating
  turbulence. This was tested empirically -- removing the top fan dropped junction temps by 5-8C.

### PCIe Link Speed: BIOS Shows Only Auto/Gen1/Gen2

The T7820 BIOS PCIe Link Speed setting only lists Auto, Gen1, and Gen2. **There is no
explicit Gen3 option.** Setting it to "Auto" will negotiate Gen3 x16 with the V100 adapter.
Verify after boot:

```bash
# Check negotiated link speed
sudo lspci -vvv -s $(lspci | grep -i nvidia | awk '{print $1}') | grep -i "lnksta:"
# Should show: Speed 8GT/s (Gen3), Width x16
```

If it negotiates Gen1 or Gen2, reseat the adapter and check for bent pins on the PCIe slot.

### "Memory Map IO above 4GB" MUST Be Enabled

The V100 32GB has a 32GB BAR (Base Address Register) that must be mapped into the system's
physical address space. If "Memory Map IO above 4GB" (also called "Above 4G Decoding") is
disabled in BIOS, the system cannot address the full 32GB of HBM2. Symptoms:

- `nvidia-smi` shows 0MB or a fraction of expected VRAM
- CUDA programs crash with `CUDA_ERROR_OUT_OF_MEMORY` on small allocations
- The GPU may not appear at all in `lspci`

**Fix:** Enter BIOS setup, navigate to Advanced > PCI Configuration (exact path varies by
BIOS version), enable "Memory Map IO above 4GB" / "Above 4G Decoding". Save and reboot.

### Wiki Extraction Artifacts

The custom `wiki_extract.py` script leaves some minor artifacts in extracted text:
- Stray `]]` from incompletely stripped wikitext links
- `\n` converted to literal `nn` in some entries

These are cosmetic and do not meaningfully affect embedding quality. Not worth re-running
the 19.3M chunk extraction pipeline to fix.

### Python 3.12 and WikiExtractor

Python 3.12 breaks the `WikiExtractor` package's regex patterns (changes to `re` module
escape handling). This is why we wrote custom `wiki_extract.py` and `wiki_chunk.py` scripts
instead of depending on WikiExtractor. Do not attempt to use WikiExtractor on Python 3.12+.
