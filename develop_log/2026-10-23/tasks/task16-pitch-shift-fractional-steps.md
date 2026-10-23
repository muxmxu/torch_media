# Task 16 — pitch_shift: fractional steps, validated rates
id: 2026-10-23/task16
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
Accept torchaudio's fractional n_steps and validate the derived resample rate before any int cast, in both layers.

## Findings
- A2-06 (medium): `int n_steps` silently truncated 0.5 to 0 (no shift).
- A2-04 (low): the PitchShift constructor skipped pitch_shift's checks. The int casts of an inf/NaN/huge rate were UB.

## Scope
In: `_functional.hpp` (`_pitch_shift_orig_freq`, pitch_shift, `_stretch_waveform`), `_transform_time.hpp`
(pitch_shift_option, PitchShift), tests, gen_golden.py (both).
Out: per-dtype PitchShift kernel cache (A2-11, open).

## Deliverables
- `double n_steps` in both layers (source-compatible for int callers). `_pitch_shift_orig_freq` (bins_per_octave != 0,
  orig_freq in [1, INT_MAX], resample args) shared by pitch_shift and the PitchShift constructor. A rejected
  constructor returns the default value from `operator()` in the no-exceptions build.
- Tests `test_p4_pitch_shift_fractional_steps` (torchaudio F.pitch_shift(x, 1000, 0.5) golden; 2000 / -240 steps
  raise) and transform `test_p4_time_transforms` (PitchShift(0.5) == functional, golden; bins_per_octave 0 and -240
  steps raise in the constructor).

## Acceptance criteria
- [x] Red on 4eb755b: fractional golden off by up to 0.64; 2000 steps ended in c10::Error "invalid size"; the
  constructor did not raise std::invalid_argument.
- [x] Green; A2 probe T_ps_bpo0 / T_ps_huge_neg / ps_huge_steps raise std::invalid_argument.
