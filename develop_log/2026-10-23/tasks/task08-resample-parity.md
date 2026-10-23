# Task 08 — resample / speed / pitch_shift parity and validation
id: 2026-10-23/task08
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
Resampling (functional and transform) matches torchaudio bit-for-bit and rejects invalid arguments cleanly.

## Findings
G4-2 (zero/negative rates: UB), G4-3 (lowpass_filter_width/rolloff unchecked), G4-4 (same rate not identity),
G4-5 (float32-only kernel), G4-6 (double-precision target length), G4-11 (kernel multiply order), G4-12
(transform kernel precision), G4-14 (0-D input: UB), G4-18 (resample tests); new: Resample/Speed/PitchShift
buffers CPU float32, speed/pitch_shift inherit; pitch_shift `llround` vs Python round-half-even.

## Scope
In: `_functional.hpp` (`_check_resample_args`, `_sinc_resample_kernel`, `_apply_sinc_resample_kernel`, resample,
_stretch_waveform rounding), `_transform_time.hpp` (Resample dtype option + constructor validation, PitchShift
window device, TimeStretch phase-advance device).
Out: sinc_interp_kaiser / beta, non-integer rates (G4-15).

## Deliverables
- Same-rate identity; argument + waveform checks (std::invalid_argument); kernel in the waveform dtype on its
  device, torchaudio op order; float32 target length; transform kernel = torchaudio `dtype=None` path.
- Tests: `test_resample_parity` (functional: f64 golden, 180697-sample length, bit-exact f32 kernel, speed /
  pitch_shift identity + float64, -12-step odd-length golden), transform `test_resample_parity`.

## Acceptance criteria
- [x] Red on e7041b5: same-rate output != input, resample(x, 0, 16000) threw c10 "non-positive stride",
  T::Resample kernel 1e-5 off, 180697 samples -> 65560 (torchaudio 65559), Resample(0, 16000) constructed,
  float64 transforms threw, pitch_shift(-12) last sample off by 0.178.
- [x] Green; probe: F::resample (48k->16k, 44.1k->16k, 16k->22.05k, float64, same rate, long), T::Resample
  (lfw 64, 16k->22.05k, dtype=float64), speed, pitch_shift (float32 and float64) all bit-exact.
- [x] ctest green.
