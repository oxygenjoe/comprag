## Quality Metrics (95% CI)

| model | quantization | hardware_tier | dataset | eval_subset | n | faithfulness | context_utilization | self_knowledge | noise_sensitivity | answer_relevancy | negative_rejection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-14B-Instruct | Q4_K_M | cpu | rgb | noise_robustness | 10 | N/A | N/A | N/A | N/A | N/A | N/A |
| llama-3.1-8b-instruct | Q4_K_M | v100 | halueval | qa | 30 | 0.715 [0.702, 0.727] | 0.646 [0.640, 0.654] | 0.615 [0.609, 0.622] | 0.303 [0.294, 0.311] | 0.683 [0.677, 0.690] | 0.571 [0.559, 0.585] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | nq | test | 30 | 0.716 [0.701, 0.728] | 0.658 [0.651, 0.668] | 0.612 [0.607, 0.619] | 0.302 [0.293, 0.311] | 0.679 [0.669, 0.686] | 0.570 [0.560, 0.580] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | rgb | negative_rejection | 30 | 0.725 [0.712, 0.738] | 0.651 [0.643, 0.660] | 0.605 [0.596, 0.614] | 0.296 [0.289, 0.305] | 0.677 [0.670, 0.684] | 0.579 [0.568, 0.590] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | rgb | noise_robustness | 30 | 0.727 [0.712, 0.738] | 0.648 [0.641, 0.655] | 0.612 [0.605, 0.619] | 0.304 [0.291, 0.318] | 0.673 [0.666, 0.679] | 0.568 [0.558, 0.578] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | halueval | qa | 30 | 0.668 [0.654, 0.680] | 0.613 [0.606, 0.620] | 0.581 [0.575, 0.587] | 0.298 [0.286, 0.308] | 0.648 [0.641, 0.657] | 0.536 [0.528, 0.545] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | nq | test | 30 | 0.682 [0.672, 0.692] | 0.611 [0.604, 0.617] | 0.570 [0.563, 0.577] | 0.296 [0.285, 0.307] | 0.643 [0.635, 0.650] | 0.543 [0.531, 0.555] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | rgb | negative_rejection | 30 | 0.684 [0.668, 0.695] | 0.613 [0.607, 0.619] | 0.578 [0.571, 0.586] | 0.299 [0.286, 0.314] | 0.644 [0.636, 0.652] | 0.545 [0.534, 0.556] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | rgb | noise_robustness | 30 | 0.677 [0.663, 0.691] | 0.609 [0.600, 0.616] | 0.575 [0.567, 0.585] | 0.289 [0.278, 0.301] | 0.645 [0.637, 0.653] | 0.555 [0.545, 0.565] |
| smollm2-1.7b-instruct | Q8_0 | v100 | halueval | qa | 30 | 0.455 [0.444, 0.466] | 0.403 [0.397, 0.410] | 0.379 [0.370, 0.386] | 0.303 [0.288, 0.315] | 0.430 [0.424, 0.436] | 0.359 [0.349, 0.369] |
| smollm2-1.7b-instruct | Q8_0 | v100 | nq | test | 30 | 0.444 [0.432, 0.457] | 0.404 [0.397, 0.411] | 0.387 [0.381, 0.395] | 0.292 [0.281, 0.303] | 0.427 [0.420, 0.434] | 0.361 [0.353, 0.373] |
| smollm2-1.7b-instruct | Q8_0 | v100 | rgb | negative_rejection | 30 | 0.447 [0.435, 0.462] | 0.402 [0.394, 0.408] | 0.380 [0.372, 0.389] | 0.307 [0.298, 0.317] | 0.428 [0.423, 0.434] | 0.367 [0.356, 0.379] |
| smollm2-1.7b-instruct | Q8_0 | v100 | rgb | noise_robustness | 30 | 0.455 [0.441, 0.469] | 0.405 [0.400, 0.413] | 0.379 [0.370, 0.385] | 0.295 [0.281, 0.309] | 0.429 [0.422, 0.435] | 0.364 [0.352, 0.377] |

## Performance Metrics (95% CI)

| model | quantization | hardware_tier | dataset | eval_subset | n | tokens_per_second | ttft_ms | vram_usage_mb | gpu_temp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-14B-Instruct | Q4_K_M | cpu | rgb | noise_robustness | 10 | 4.3 [4.0, 4.4] | 50820.5 [39285.9, 65988.2] **!** | 203.0 [nan, nan] | 27.6 [27.3, 27.9] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | halueval | qa | 30 | 32.2 [31.4, 33.4] | 52.8 [49.6, 56.6] | 6299.7 [6280.6, 6319.0] | 67.7 [66.2, 69.2] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | nq | test | 30 | 31.5 [30.8, 32.2] | 50.5 [47.0, 53.7] | 6293.5 [6272.7, 6315.6] | 69.5 [67.5, 71.1] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | rgb | negative_rejection | 30 | 32.5 [31.7, 33.3] | 49.8 [45.6, 54.2] **!** | 6304.4 [6284.2, 6325.1] | 66.7 [65.1, 68.6] |
| llama-3.1-8b-instruct | Q4_K_M | v100 | rgb | noise_robustness | 30 | 32.5 [31.9, 33.2] | 51.3 [48.2, 54.4] | 6294.0 [6274.9, 6312.2] | 67.1 [65.2, 68.6] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | halueval | qa | 30 | 34.5 [33.8, 35.3] | 48.7 [44.4, 52.8] **!** | 6013.1 [5991.6, 6036.7] | 67.5 [65.8, 69.1] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | nq | test | 30 | 34.2 [33.6, 34.8] | 47.0 [43.9, 49.9] | 6003.7 [5978.4, 6029.2] | 66.9 [65.1, 68.5] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | rgb | negative_rejection | 30 | 35.7 [35.1, 36.3] | 51.0 [48.1, 55.2] | 6005.6 [5982.6, 6025.0] | 65.3 [63.9, 66.8] |
| qwen2.5-7b-instruct | Q4_K_M | v100 | rgb | noise_robustness | 30 | 35.1 [34.4, 35.7] | 48.9 [45.4, 52.1] | 6008.7 [5988.9, 6030.3] | 67.7 [65.9, 69.3] |
| smollm2-1.7b-instruct | Q8_0 | v100 | halueval | qa | 30 | 54.8 [54.1, 55.4] | 49.8 [46.7, 52.6] | 2495.6 [2475.3, 2514.7] | 65.1 [64.0, 66.9] |
| smollm2-1.7b-instruct | Q8_0 | v100 | nq | test | 30 | 55.3 [54.4, 56.0] | 50.9 [47.7, 54.1] | 2485.6 [2467.1, 2506.4] | 68.6 [67.0, 70.2] |
| smollm2-1.7b-instruct | Q8_0 | v100 | rgb | negative_rejection | 30 | 54.4 [53.6, 55.3] | 49.4 [45.3, 53.5] **!** | 2498.0 [2479.6, 2517.9] | 68.2 [66.5, 70.0] |
| smollm2-1.7b-instruct | Q8_0 | v100 | rgb | noise_robustness | 30 | 54.9 [54.1, 55.8] | 51.7 [49.2, 54.4] | 2497.8 [2477.5, 2518.4] | 68.0 [66.3, 69.8] |

## Warnings

- **model=Qwen2.5-14B-Instruct | quantization=Q4_K_M | hardware_tier=cpu | dataset=rgb | eval_subset=noise_robustness**: ttft_ms: CI width (26702.2842) > 15% of mean (50820.4510) — needs more runs
- **model=llama-3.1-8b-instruct | quantization=Q4_K_M | hardware_tier=v100 | dataset=rgb | eval_subset=negative_rejection**: ttft_ms: CI width (8.6277) > 15% of mean (49.7780) — needs more runs
- **model=qwen2.5-7b-instruct | quantization=Q4_K_M | hardware_tier=v100 | dataset=halueval | eval_subset=qa**: ttft_ms: CI width (8.4358) > 15% of mean (48.7480) — needs more runs
- **model=smollm2-1.7b-instruct | quantization=Q8_0 | hardware_tier=v100 | dataset=rgb | eval_subset=negative_rejection**: ttft_ms: CI width (8.2343) > 15% of mean (49.4483) — needs more runs
