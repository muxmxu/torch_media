# Task 15 — spectral/time ops: rank checks and no-exceptions returns
id: 2026-10-23/task15
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
No out-of-bounds read or segfault on 0-D/1-D input in the phase vocoder, InverseMelScale, pitch shift and pitch
detection paths. Transform call sites return the default value in the no-exceptions build.

## Findings
- A2-01 (medium): phase_vocoder / TimeStretch read `shape[size-2]` out of bounds on 1-D / 0-D input.
- A2-02 (medium): InverseMelScale did the same.
- A2-03 (medium): pitch_shift / PitchShift / detect_pitch_frequency segfaulted on 0-D input.
- A2-09 (low): TimeStretch / InverseMelScale `(void) handle_exceptions` then continued; the message blamed a missing
  fixed_rate for a bad overriding_rate.

## Scope
In: `_functional.hpp` (phase_vocoder, `_stretch_waveform`, pitch_shift, detect_pitch_frequency), `_transform_time.hpp`
(TimeStretch, PitchShift), `_transform_feature.hpp` (InverseMelScale), tests.

## Deliverables
- dim checks returning `handle_exceptions<tensor_t, std::invalid_argument>`. phase_vocoder checks after the rate == 1
  early return (torchaudio order). TimeStretch names the bad overriding_rate.
- Tests `test_p4_rank_and_degenerate_checks` (functional), `test_p4_rank_checks` and the overriding_rate message
  (transform), and noexcept checks (TimeStretch without a rate, InverseMelScale n_mels mismatch / 1-D, phase_vocoder
  1-D, `_stretch_waveform` / pitch_shift / detect_pitch 0-D).

## Acceptance criteria
- [x] Red on 4eb755b: TimeStretch 1-D raised a garbage-shape c10::Error; the overriding_rate message check failed.
  The functional 0-D cases sit in the crash-class test, and the noexcept checks come after a crashing vad check; the
  red runs segfaulted before reaching them (the verifier's ASan/no-exceptions probes are the red evidence there).
- [x] Green; A2 probe crash cases (pv_1d, ims_1d, ps_0d, dp_0d) now raise std::invalid_argument; ASan/UBSan clean.
