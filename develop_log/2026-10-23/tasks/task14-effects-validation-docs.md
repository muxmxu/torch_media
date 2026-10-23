# Task 14 — effects: validation, loudness rounding, intentional differences, header include
id: 2026-10-23/task14
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
Close the cheap low findings of the filtering/effects surface and pin them with tests.

## Findings
- A1-03: loudness gating step rounded half away from zero (11025 Hz: 1103 instead of Python's 1102).
- A1-06: phaser with mod_speed 40000 segfaulted, delay_ms 0 / 0.01 ran `% 0`.
- A1-07: flanger on 1-D input read `shape[SIZE_MAX]`.
- A1-08: compute_deltas silently replicated for any unknown mode.
- A1-09: add_noise had none of torchaudio's checks; float lengths mask wrong from 2^24 samples (float16: from 2048).
- A1-11 (part): Vol power gain 0 returned silence (torchaudio: math domain error).
- A1-12: dcshift/dither intentional differences undocumented, GPDF not pinned.
- A1-13: `_functional.hpp` used lfilter / treble_biquad without including `_functional_filtering.hpp`.

## Scope
In: `_functional.hpp` (loudness, compute_deltas, add_noise, include), `_functional_filtering.hpp` (phaser, flanger,
dcshift/dither comments), `_transform_effects.hpp` (Vol), `audio.hpp` comment, tests, gen_golden.py.
Out: the silent defaults of Vol/Vad/Loudness (A1-11 rest, API break) and A1-10.

## Deliverables
- `std::nearbyint` for gate_samples/step. phaser buffer-length check before the casts. flanger dim >= 2. compute_deltas
  rejects unknown modes. add_noise rank/length checks and an int64 mask on lengths' device. Vol power gain <= 0 raises.
- Comments: dcshift / GPDF / mono GPDF intentional differences.
- Tests `test_p4_loudness_11025` (torchaudio -12.05608081817627), `test_p4_dither_seeded_and_intentional_differences`
  (seeded RPDF/GPDF/GPDF+noise-shaping goldens; input unchanged), `test_p4_effect_validation` (float16 mask golden),
  transform Vol, and noexcept checks. The transform test includes `_transform_effects.hpp` first.
- Review (2026-10-23): the float16 mask check put its sample at index 2049, which a float16 arange also excludes
  (the comparison rounds lengths 2049 to 2048 as well), so a float arange mutant passed. The sample is now at index
  2048, the one index the two masks disagree on. add_noise energies now use `linalg_vector_norm(...).pow(2)` as
  torchaudio does (the sum of squares was 1 ulp off in float32), pinned by exact float32 goldens.

## Acceptance criteria
- [x] Red on 4eb755b: loudness off by 9.7e-3 LU, compute_deltas/add_noise checks failed, GPDF goldens off,
  phaser/flanger crash class, header-only TU did not compile (2 errors).
- [x] Green; A1 probe ld_11025 1.8e-7 of peak.

## Notes
- GPDF golden = torchaudio's draws on an unaliased copy (`_gpdf_unaliased` in gen_golden.py). torchaudio's own output
  differs at 3 elements here, because it re-reads the sample it accumulates into (progress04 D6).
