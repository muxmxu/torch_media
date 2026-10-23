# Task 50 — biquad designers: coefficients as torchaudio computes them
id: 2026-10-23/task50
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Make the float32 output of the eleven `*_biquad` designers (and loudness, which uses two of them) bit-exact with
torchaudio 2.5.1 (A1-10).

## Findings
- A1-10: the designers computed coefficients in double and cast them to the waveform's dtype; torchaudio computes
  them as 0-D tensors in that dtype. float32 was up to 6.8e-3 of peak off (band, noise, 30 Hz at 44.1 kHz).
- New: clang contracted riaa / deemph's double math into FMAs (float64 riaa 1.1e-14..2.7e-14 off).

## Scope
In: `_functional_filtering.hpp` (designers, a Tensor `biquad` overload), functional tests, gen_golden.py.
Out: lfilter itself (already bit-exact, progress04 D1).

## Deliverables
- `detail::as_coeff` (torch.as_tensor) and `biquad(waveform, Tensor b0, ..., Tensor a2)`.
- allpass, lowpass, highpass, bandpass, bandreject, equalizer, band, bass, treble: torchaudio's tensor ops in order.
- deemph / riaa: double math with one product per statement.
- `test_p8_biquad_designers_in_dtype`: one torchaudio golden per designer at the index where the old coefficients
  were furthest off (tolerance 0), loudness at 48 / 22.05 kHz, riaa / deemph float64, a float64 dtype check, MPS
  against CPU.

## Acceptance criteria
- [x] Red on 080ae69: 18 failures (every float32 designer except allpass / bandreject, both loudness rates, the float64
  riaa points; deemph was already exact).
- [x] Green; scratch probe (afollow/impl/probe_biquad) 62 -> 94 / 94 rows bit-exact (designers float32 / float64,
  loudness at five rates).
- [x] lowpass_biquad 2 s stereo 0.335 ms (torchaudio 0.346 ms).
