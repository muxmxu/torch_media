# Task 53 — torchaudio goldens for the loose and self-referential tests
id: 2026-10-23/task53
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Pin phase_vocoder, inverse_spectrogram options, detect_pitch_frequency, InverseMelScale, LFCC, SpeedPerturbation, the
beamforming transforms (incl. online MVDR) and the three vad float32 details to torchaudio 2.5.1 (A2-12, A3-16,
progress04 vad notes). Each new test must fail on a deliberate mutation.

## Scope
In: functional and transform tests and gen_golden.py. No library change.

## Deliverables
- functional: `test_p8_phase_vocoder_goldens`, `test_p8_inverse_spectrogram_options`, `test_p8_detect_pitch_golden`,
  `test_p8_vad_float32_details` (trigger levels on the float32 peak of torchaudio's running mean, two input variants).
- transform: `test_p8_feature_time_goldens` (InverseMelScale, LFCC dB / log_lf, SpeedPerturbation seeds 7 / 2024),
  `test_p8_souden_rtf_mvdr_goldens`, `test_p8_mvdr_goldens`, `test_p8_mvdr_online_goldens`.
- Deleted: the 1e-4 phase_vocoder, 1-Hz detect_pitch, 1e-1 InverseMelScale and 1e-2 LFCC pins, and the MVDR tests
  that compared against this library's functional ops (`test_souden_rtf_mvdr`, `test_mvdr_offline`, `test_mvdr_online`
  and their helpers).

## Acceptance criteria
- [x] All goldens bit-exact on this tree (point checks at 0, float64 sums at 1e-9, complex64 sums at 1e-4).
- [x] Mutation check (afollow/impl/mut, 21 mutants of the touched code, built from the real test sources): all fail.
  For vad, a float64 mean update and a double trigger comparison are killed at T_a (k = 0); a float64 first FFT at
  T_b (k = 9).
