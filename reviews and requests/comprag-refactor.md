# CompRAG Refactoring Spec — Post-Review Fixes

## Context

Independent code review (Gemini) identified three edge cases in the Phase 1 scaffold that need remediation before benchmark runs begin. All three are integration bugs that would cause silent failures or incorrect results during long eval runs.

**Scope:** Targeted fixes only. Do not restructure, rename, or reorganize existing modules. Preserve all existing behavior except where explicitly changed below.

---

## Fix 1: Centralize Embedding Model Config

**Problem:** `build_index.py` and `retriever.py` both hardcode `"all-MiniLM-L6-v2"` independently. Changing one without the other causes silent retrieval failures.

**Changes required:**

1. In `eval_config.yaml`, the embedding model is already defined under `retrieval.embedding_model`. This is the single source of truth.

2. In `build_index.py`:
   - Remove the hardcoded `EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"` constant.
   - Read `retrieval.embedding_model` from `eval_config.yaml` via `load_config()`.
   - When writing ChromaDB collection metadata, include `{"embedding_model": model_name}` (verify this is already happening — if so, no change needed).

3. In `retriever.py`:
   - Remove the hardcoded `DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"` constant.
   - On init, first try to read the embedding model name from the ChromaDB collection's metadata (which `build_index.py` writes at index time).
   - Fall back to reading from `eval_config.yaml` only if collection metadata is missing.
   - Log a warning if the collection metadata embedding model doesn't match the config file.

**Do NOT change:** The `EMBEDDING_DIM = 384` constant can stay hardcoded — it's determined by the model and doesn't need to be configurable.

---

## Fix 2: Zombie llama-server Process Prevention

**Problem:** If the Python runner crashes hard (kernel OOM kill, segfault), `llama-server` keeps running as an orphan, holding port 8080. All subsequent runs fail until manual cleanup.

**Changes required:**

1. In `generator.py`, add a `_check_port_available()` method to `LlamaServer`:
   - Before starting a new server, check if the target port is already in use.
   - If occupied, attempt to identify the PID bound to that port (`lsof -ti:{port}` or parsing `/proc/net/tcp`).
   - If the process is a `llama-server`, kill it and log a warning: `"Killed orphaned llama-server (PID {pid}) on port {port}"`.
   - If it's something else, raise an error — don't kill random processes.
   - Call this method at the top of `start()`.

2. In `generator.py`, add an `atexit` handler in `LlamaServer.__init__`:
   ```python
   import atexit
   atexit.register(self._cleanup)
   ```
   Where `_cleanup()` calls `self.stop()` if the server is still running. This catches normal Python exits that skip the context manager.

3. In `runner.py`, register signal handlers at the top of the main eval loop:
   ```python
   import signal
   signal.signal(signal.SIGTERM, _shutdown_handler)
   signal.signal(signal.SIGINT, _shutdown_handler)
   ```
   Where `_shutdown_handler` stops the server and exits cleanly. Note: `SIGSEGV` handler won't reliably work — the port check on startup is the real safety net for hard crashes.

**Do NOT change:** The `preexec_fn=os.setsid` in `_start_server` — it's correct for process group management. The `try/except` crash recovery in `runner.py` must be preserved.

---

## Fix 3: Tokenizer Mismatch — Safer Chunk Sizing

**Problem:** `WhitespaceTokenizer` at 512 words produces chunks of ~650-800 BPE tokens. With `top_k=5`, total context can hit 4000+ tokens. Models with 2048 context windows will silently truncate, corrupting eval results.

**Changes required:**

1. In `build_index.py`:
   - Change the default `chunk_size` from 512 to 300 (whitespace words).
   - Update the constant comment to explain the reasoning: `# ~300 whitespace words ≈ 400-500 BPE tokens, safe for top_k=5 within 4096 ctx`.
   - The `--chunk-size` CLI flag stays — this just changes the default.

2. In `eval_config.yaml`:
   - Change `chunk_size: 512` to `chunk_size: 300`.
   - Add a comment: `# Whitespace words, not BPE tokens. 300 words ≈ 400-500 BPE tokens.`

3. In `runner.py`, add a context window safety check before generation:
   - Estimate prompt length: `estimated_tokens = len(formatted_prompt.split()) * 1.4` (rough whitespace-to-BPE ratio).
   - If `estimated_tokens > 0.9 * model_context_length`, log a warning: `"Estimated prompt ({est} tokens) may exceed model context ({ctx} tokens). Consider reducing top_k."`.
   - Do NOT auto-reduce top_k — just warn. The researcher decides.
   - `model_context_length` can be read from a new optional field in `models.yaml` (add `context_length: 32768` etc. to each model entry) or default to 4096 if unspecified.

4. In `config/models.yaml`:
   - Add `context_length` field to each model entry. Values:
     - Qwen 2.5 14B/7B: 32768
     - Phi-4 14B: 16384
     - Mistral NeMo 12B: 32768
     - Llama 3.1 8B: 131072
     - Gemma 2 9B: 8192
     - GLM-4 9B: 131072
     - SmolLM2 1.7B: 8192

**Do NOT change:** The `WhitespaceTokenizer` class itself — it's fine for chunking. The `hashlib.md5` idempotency logic. The `overlap` parameter.

---

## Rebuild Index After Fix 3

After applying Fix 3, the existing ChromaDB index (if any) must be rebuilt with the new chunk size:

```bash
python scripts/build_index.py --all --force
```

Document this in the commit message.

---

## Verification

After all three fixes:
1. Run `python tests/smoke_test.py` — all core checks must still pass.
2. Verify `retriever.py` reads embedding model from collection metadata.
3. Verify starting the runner twice in a row doesn't fail on port conflict.
4. Verify chunk sizes in the rebuilt index average ~300 words.
