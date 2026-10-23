# Task 11 — review round 2 follow-ups (parity + robustness reviewers)
id: 2026-10-23/task11
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
Close the 12 findings two reviewers raised against the task04-task10 changes, each with a test that fails before
the fix and passes after.

## Findings
- P1/R1 (medium): T::PitchShift on float64 input used its float32 kernel cast up (2.8e-5 to 4.9e-4 off
  torchaudio); progress02 claimed bit-exact / ~1e-8.
- P2 (low): create_dct built in float64 then cast (1.9e-5 off), so MFCC/LFCC were not bit-exact.
- P3 (low): equal rates threw on lowpass_filter_width / rolloff <= 0; torchaudio returns the input.
- P4 (low): an invalid normalize_method passed when normalized was false.
- P5 (low): spectral_centroid bin frequencies were float64 for float64 input (torchaudio: float32).
- R2 (medium): no test pinned the transform kernel's int64-phase detail (only nf = 1 was tested).
- R3 (medium): with LIB_TORCH_MEDIA_NO_EXCEPTIONS, pitch_shift returned zeros, mfcc/melspectrogram ended in a
  c10::Error, T::Resample(0) used an undefined kernel.
- R4 (low): the no-exceptions test covered ~5 of the `return handle_exceptions` sites.
- R5 (low): griffinlim on an integer spectrogram threw (window dtype taken before pow).
- R6 (low): negative win/hop were silently derived in the functional layer.
- R7 (low): inverse_spectrogram accepted "true" but rejected "false".

## Scope
In: `_functional.hpp` (spectrogram, mel string checks, create_dct, mfcc, griffinlim, _check_resample_args,
spectral_centroid, inverse_spectrogram, pitch_shift), `_functional_methods_options.hpp` (derive helpers),
`_transform_time.hpp` (Resample guard, PitchShift kernel dtype), `globel_include.hpp` (comment), tests,
gen_golden.py, progress02.
Out: per-dtype PitchShift kernel cache, up-front validation in the remaining composites (progress02 Open).

## Deliverables
- Fixes as recorded in progress02 D3-D7, D10, D12.
- Tests: functional `test_r2_*` (7 functions), transform `test_r2_pitch_shift_float64`,
  `test_r2_resample_kernel_and_equal_rates`; `audio_test_noexcept` rewritten to check every tensor-returning
  error site (59 checks, composites through `returns_default`); goldens in both gen_golden.py "task11" sections.

## Acceptance criteria
- [x] Red before the fixes: functional 308/324 (create_dct, centroid, equal-rate resample, normalize_method,
  negative win/hop, bins_per_octave 0, integer griffinlim, "false"), transform 197/202 (PitchShift float64,
  equal-rate Resample), noexcept 50/59 (pitch_shift, Resample(0), spectrogram, melspectrogram, mfcc).
- [x] Mutants: M15 (phase arange in float64) fails transform 3 checks; M51 (griffinlim momentum `return` dropped)
  fails noexcept.
- [x] Green; parity probe on this tree: create_dct, MFCC/LFCC (<= 3-D), spectral_centroid float64, F/T Resample,
  PitchShift float32, F::pitch_shift float64 bit-exact; T::PitchShift float64 4.6e-8 (window cast, as torchaudio).
- [x] ctest green (6 tests); union line coverage 100% for `_functional*.hpp`, `_transform*.hpp`, `_io.hpp`.
