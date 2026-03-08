# CLAUDE.md — CompRAG

Benchmarking study measuring how GGUF quantization levels affect RAG faithfulness metrics across local LLMs. Master's thesis-level rigor. GitHub: github.com/oxygenjoe/comprag

## Environment
- Venv: `/home/jophes/comprag/.venv`
- Repo: `/home/jophes/comprag`

## GPU / Server Notes
- V100 PSU is janky — set power limit to 150W: `sudo nvidia-smi -i 1 -pl 150`
- CUDA device order ≠ nvidia-smi order. nvidia-smi GPU 0 = 1660 Super, GPU 1 = V100. But CUDA sees V100 as device 0.
- To run llama.cpp on V100: `CUDA_VISIBLE_DEVICES=0 llama-server ...` (NOT `--main-gpu 1`)
- Do NOT use `--main-gpu` — use `CUDA_VISIBLE_DEVICES` to isolate the correct GPU

## Locked Parameters
- temperature=0.0, seed=42 on ALL runs, no exceptions
- Judge model: Claude Opus 4.6 (frontier API only, no local fallback)
- Judge agreement: pairwise kappa across Opus 4.6, GPT-5.4, Gemini 3 Flash
- Embedding: all-MiniLM-L6-v2
- Primary models (4, full 6-quant sweep Q3→FP16): Qwen 2.5 14B, Phi-4 14B, Llama 3.1 8B, Qwen 2.5 7B
- Secondary models (2, Q4_K_M + Q8_0): Mistral NeMo 12B, Gemma 2 9B
- Floor model: SmolLM2 1.7B (Q8_0 + FP16)
- Frontier comparison: GPT-5.4, Claude Opus 4.6, Gemini 3 Flash, DeepSeek V3.2, GLM-5 (ZhipuAI)

## Pipeline
- Flat pipeline, JSONL interfaces between steps
- No module >300 lines
- Three-pass protocol: pass1 (no context), pass2 (loose context), pass3 (strict context)
- Only RGB counterfactual gets all 3 passes; other subsets get pass2_loose only
- Dataset: RGB (noise_robustness, negative_rejection, counterfactual), NQ, HaluEval
- Output dir: results/
- Key metric: Preference_Gap = Pass3_CU − Pass2_CU
- Pre-registered two-tier falsification structure

## Frontier API Notes
- ZhipuAI (GLM-5): use `glm-5`, base_url `https://open.bigmodel.cn/api/paas/v4`, env var `ZHIPU_API_KEY`
- All frontier model IDs are preview/versioned — pin and record them.
