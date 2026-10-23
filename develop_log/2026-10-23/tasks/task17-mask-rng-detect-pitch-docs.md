# Task 17 — mask RNG, detect_pitch parity and validation, API-difference docs
id: 2026-10-23/task17
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
Same seeded masks and pitch values as torchaudio. Reject pitch-detection arguments that were UB. Document the
intentional API differences.

## Findings
- A2-05: mask_along_axis drew its uniforms in the spectrogram's dtype/device (float64 / MPS masks differed, int threw).
- A2-07: detect_pitch_frequency Hz 1 ulp off (correctly rounded division vs Python's reciprocal * int); NCCF norm via
  sqrt(sum) instead of vector_norm.
- A2-08: freq_low / frame_time 0 cast inf to int64 (UB); win_length 0 ended in a c10 median error.
- A2-10: TimeStretch hop 0, LFCC f_max 0, masking defaults, PitchShift window, float64 transforms undocumented.
- A2-12 (part): no torchaudio-seeded mask golden.

## Scope
In: `_functional.hpp` (mask_along_axis, `_compute_nccf`, detect_pitch_frequency), option comments in
`_transform_time.hpp`, `_transform_feature.hpp`, `_transform_augment.hpp`, tests, gen_golden.py (both).

## Deliverables
- `torch::rand({1})` in mask_along_axis. `torch::linalg_vector_norm` in the NCCF. `reciprocal(EPS + idx) *
  sample_rate`. An up-front argument check. win_length 1-2 kept as a documented extension.
- Tests `test_p4_mask_rng_cpu_float32` (float64 / int64 / MPS masks equal torchaudio's seed-0 masks),
  `test_p4_detect_pitch_parity_and_validation` (exact Hz goldens incl. a frame where the division rounding differs, NCCF
  goldens, five invalid arguments, win_length 1), transform `test_p4_specaugment_f64_seeded` (torchaudio sum/zero count).
- Review (2026-10-23): on those tones vector_norm and sqrt(sum(x * x)) give the same NCCF, so a sqrt(sum) mutant
  passed. Added exact NCCF goldens on a noise-like signal, where they differ by up to 5e-7 relative.

## Acceptance criteria
- [x] Red on 4eb755b: float64 mask columns differed, int64 threw, SpecAugment float64 sum/zeros differed,
  pf[0][0][5] 1 ulp off, sum off by 1.7e-4, sample_rate 0 raised a c10 error.
- [x] Green; A2 probe dp_* and ma_* rows exact; T_sa_f64_noiid exact.
