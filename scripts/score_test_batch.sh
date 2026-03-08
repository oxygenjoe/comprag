#!/bin/bash
# Score all 20 test run result files with DeepSeek judge, then aggregate + visualize.
# Resume-safe: skips already-scored records.

set -e
export DEEPSEEK_API_KEY="sk-1d1225579ed14ad1ac81778a45b877d9"
cd /home/jophes/comprag
PYTHON=/home/jophes/comprag/.venv/bin/python3

# Score each raw file from the test run (model_quant pattern)
for f in results/raw/*-instruct_Q*_rgb_pass2_loose.jsonl \
         results/raw/phi-4-14b_Q*_rgb_pass2_loose.jsonl \
         results/raw/smollm2-*_rgb_pass2_loose.jsonl \
         results/raw/*-instruct_FP16_rgb_pass2_loose.jsonl \
         results/raw/phi-4-14b_FP16_rgb_pass2_loose.jsonl; do
    [ -f "$f" ] || continue
    echo "=== Scoring: $(basename $f) ==="
    $PYTHON -m comprag score \
        --input "$f" \
        --judge-provider deepseek \
        --judge-model deepseek-chat \
        || echo "FAILED: $f"
done

echo "=== Aggregating ==="
$PYTHON -m comprag aggregate \
    --input-dir results/scored/ \
    --output-dir results/aggregated_test2/

echo "=== Visualizing ==="
$PYTHON -m comprag visualize \
    --input-dir results/aggregated_test2/ \
    --output-dir results/figures_test2/

echo "=== Done ==="
