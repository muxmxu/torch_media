# Task 02 — bss_eval_sources and speaker-verification metrics
id: 2026-10-24/task02
parent: 2026-10-24/progress01
status: done
owner: code_agent

## Objective
mir_eval 0.8.2 `bss_eval_sources` (SDR / SIR / SAR / permutation) and EER / minDCF defined from sklearn's ROC, in
`torchmedia::metric::audio`.

## Scope
In:
- `bss_eval_sources(reference_sources, estimated_sources, compute_permutation)` -> `bss_eval_sources_result`
  (progress01 D4), mir_eval's validation and empty-input result, the lstsq fallback for a singular Gram matrix.
- `equal_error_rate`, `min_dcf` (`min_dcf_option`), classes `EqualErrorRate`, `MinDCF` (progress01 D5).
Out:
- `bss_eval_images`, the framewise variants.

## Acceptance criteria
- [x] bss_eval within 1e-9 dB of mir_eval on the goldens (identical permutations); one-source SIR inf.
- [x] EER / minDCF bit-exact against gen_golden.py's independent `eer_np` / `min_dcf_np` (ties, exact crossing,
  perfect / inverted / fully tied scores, several costs).
