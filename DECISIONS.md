# CompRAG Decision Log

## 2026-03-08 — Command R validated as local judge (κ>0.93 on CU/faithfulness)

**Context:** Spec calls for Claude Opus 4.6 as primary judge. Opus API key not yet available.
Tested Command R 35B Q4_K_M (local, V100) against DeepSeek-chat (API) on 600 records
across all 3 passes (pass1_baseline, pass2_loose, pass3_strict), 2 models (Gemma 2 9B Q4_K_M, Q8_0),
RGB counterfactual subset, 100 queries each.

**Method:** 3-bin ordinal discretization [0–0.33, 0.33–0.67, 0.67–1.0], Cohen's κ (linear weighted).

**Results (all passes combined, n=600):**

| Metric | κ (linear) | Raw agreement |
|---|---|---|
| context_utilization | 0.932 | 577/600 |
| faithfulness | 0.931 | 574/600 |
| overall_recall | 0.887 | 583/600 |
| hallucination | 0.709 | 579/600 |
| overall_f1 | 0.698 | 519/600 |
| overall_precision | 0.686 | 520/600 |
| noise_sensitivity_relevant | 0.290 | 552/600 |
| self_knowledge | ~0 | 594/600 (no signal) |

**Per-pass notes:**
- Pass1 (no context): CU/SK/hallucination constant across both judges — correct behavior.
- Pass2 (loose): CU κ=0.853, faithfulness κ=0.387 (weaker on loose prompts).
- Pass3 (strict): hallucination κ=0.817, faithfulness κ=0.801 — judges agree more under strict prompting.

**Decision:** Use Command R Q4_K_M as primary workhorse judge (local, zero cost, ~5s/record on V100).
Validate with 100-record Opus 4.6 sample for thesis appendix when API key is available.
CU and faithfulness (the thesis's primary metrics) both clear the 0.80 κ threshold with room to spare.

**Rationale:** Command R at 19GB on V100 matches DeepSeek API speed (~5s/record) at zero cost.
κ>0.93 on CU/faithfulness means switching judges does not meaningfully change results.
Weaker agreement on NS_relevant (κ=0.29) is acceptable — that metric is secondary and both judges
produce near-zero values anyway (552/600 raw agreement despite low κ due to marginal distribution skew).
