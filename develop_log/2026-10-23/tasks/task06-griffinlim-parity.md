# Task 06 — griffinlim parity (length, rand_init, validation, dtype)
id: 2026-10-23/task06
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
`functional::griffinlim` / `transform::GriffinLim` reproduce torchaudio op-for-op.

## Findings
G2-2 (length only on the final istft), G2-6 (rand_init distribution), G2-7 (momentum unchecked), G2-9 (rank < 2
out-of-bounds read), G2-10 (default window float32), G2-12 (loose / non-discriminating tests); new:
inverse_spectrogram rank < 2 out-of-bounds read (also validates the `normalized` string now).

## Scope
In: `_functional.hpp` (griffinlim, inverse_spectrogram), `_functional_methods_options.hpp` (griffinlim_option
win/hop 0), transform GriffinLim/InverseSpectrogram window device.

## Deliverables
- length in every istft; `torch::rand(complex)` init; momentum in [0, 1) and rank >= 2 enforced; default window
  in the spectrogram's dtype/device; derived win/hop.
- Tests: `test_griffinlim_parity` (functional, goldens length=18 and seed-0 rand_init), transform
  `test_spectral_parity`; existing griffinlim goldens tightened to 1e-3 (sum) / 1e-6 (sample) and 1e-5.

## Acceptance criteria
- [x] Red on e7041b5: length=18 off by 2.68, rand_init off by 1.98, momentum 1.0/1.5/-0.5 accepted, 1-D input
  threw c10 "shape '[-1, 527, 5]'" (garbage size from the out-of-bounds read).
- [x] Green; probe on [2,129,63] (length 4000, rand_init seed 0, T.GriffinLim default) bit-exact; float64 with a
  float64 window bit-exact.
- [x] ctest green.
