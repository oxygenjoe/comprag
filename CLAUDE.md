# CLAUDE.md — CompRAG

Benchmarking study measuring how GGUF quantization levels affect RAG faithfulness metrics across local LLMs. Master's thesis-level rigor. GitHub: github.com/oxygenjoe/comprag

## Environment
- Venv: `/home/jophes/comprag/.venv`
- Repo: `/home/jophes/comprag`

## GPU / Server Notes
- V100 PSU is janky — set power limit to 150W: `sudo nvidia-smi -i 1 -pl 150`
- CUDA device order ≠ nvidia-smi order. nvidia-smi GPU 0 = 1660 Super (dead), GPU 1 = V100. But CUDA sees V100 as device 0.
- To run llama.cpp on V100: `CUDA_VISIBLE_DEVICES=0 llama-server ...` (NOT `--main-gpu 1`)
- Do NOT use `--main-gpu` — use `CUDA_VISIBLE_DEVICES` to isolate the correct GPU
- Generation server: port 8080. Judge server: port 8081. Cannot coexist on V100.

## Locked Parameters
- temperature=0.0, seed=42 on ALL runs, no exceptions
- Embedding: all-MiniLM-L6-v2
- Primary models (4, full 6-quant sweep Q3→FP16): Qwen 2.5 14B, Phi-4 14B, Llama 3.1 8B, Qwen 2.5 7B
- Secondary models (2, Q4_K_M + Q8_0): Mistral NeMo 12B, Gemma 2 9B
- Floor model: SmolLM2 1.7B (Q8_0 + FP16)
- 500 queries per model (100 per RGB subset)

## Judge Architecture (v9)
- Primary judge: Command R 35B Q4_K_M (local, llama.cpp on port 8081)
- Validation judge: Claude Sonnet 4.6 API (100-record spot-check for appendix)
- No frontier generation — all generation is local via llama.cpp
- Judge validation: Cohen's κ between Command R and Sonnet 4.6 on 100 records
- Existing results: CU κ=0.884 (pass), faithfulness κ=0.772 (below 0.80 threshold)

## Pipeline
- Flat pipeline, JSONL interfaces between steps
- No module >300 lines
- Three-pass protocol: pass1 (no context), pass2 (loose context), pass3 (strict context)
- Only RGB counterfactual gets all 3 passes; other subsets get pass2_loose only
- Dataset: RGB only (counterfactual, noise_robustness, negative_rejection)
- Output dir: results/
- Key metric: Preference_Gap = Pass3_CU − Pass2_CU
- Pre-registered two-tier falsification structure

## Current Spec
- Thesis: docs/comprag-thesis-v9.md
- Build spec: docs/COMPRAG-V9-BUILD-SPEC.md
