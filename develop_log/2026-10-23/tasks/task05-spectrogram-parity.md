# Task 05 — spectrogram parity (frame_length, N-D, derived win/hop)
id: 2026-10-23/task05
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
`functional::spectrogram` (and every wrapper) matches torchaudio's normalization, batching and defaults.

## Findings
G2-1 (frame_length divided by win_length), G2-3 / G3-5 (N-D input threw), G2-5 / G3-6 (hard-coded win 400 /
hop 200), G2-8 (unknown method silently ignored), G2-11 (tests could not see any of this); new: transform
windows cached on the CPU.

## Scope
In: `_functional.hpp` (spectrogram, melspectrogram option plumbing), `_functional_methods_options.hpp`
(win/hop defaults 0; `functional::detail::derive_*`), `_transform_spectral.hpp` (re-export the helpers; windows
moved to the input's device).
Out: `spectrogram_option::_return_complex` default (kept, D3).

## Deliverables
- frame_length -> `torch::stft(normalized=true)`; window norm after stft; unknown method / 0-D input throw.
- Pack to (-1, L), restore leading dims.
- win/hop 0 derive from n_fft through one helper shared with the transform layer.
- Tests: `test_spectrogram_frame_length_parity`, `test_spectrogram_nd_and_defaults` (functional),
  `test_spectral_parity` (transform); the old "none" method pin removed.

## Acceptance criteria
- [x] Red on e7041b5: frame_length off by 12.44 (n_fft=win=8) / 7.76 (win 6), "none" accepted, [2,3,64] threw
  "stft: expected a 1D or 2D tensor", n_fft-only option threw.
- [x] Green; probe vs torchaudio on [2,3,4000] (power 2, frame_length 400 and 512/256, window norm, complex,
  T.Spectrogram(n_fft=1024)) bit-exact.
- [x] ctest green.
