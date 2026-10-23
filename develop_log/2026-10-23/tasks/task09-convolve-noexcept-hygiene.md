# Task 09 — convolve checks, LIB_TORCH_MEDIA_NO_EXCEPTIONS path, header hygiene
id: 2026-10-23/task09
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
Convolution validates its operands consistently, the no-exceptions build never computes from invalid input,
and the headers are self-contained and free of reserved names and dead code.

## Findings
G1-4 (handle_exceptions return discarded), G1-5 (leading dims not checked), G1-9 (unscoped enum, no default
mode), G4-17 (audio_util.hpp); new: fftconvolve has no prelude, `<torchmedia.hpp>` not self-contained,
reserved guard `_LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP`, internal variable_factories.h include, unused-parameter
warnings in globel_include.hpp.

## Scope
In: `globel_include.hpp`, `_functional.hpp`, `_functional_{filtering,beamforming,alignment}.hpp`,
`_transform.hpp`, `_transform_effects.hpp`, `audio_util.hpp` (deleted), new `unit_test/audio/noexcept/`.

## Deliverables
- `enum class convolve_mode`, default `full`; `_align_convolve_operands` shared by convolve/fftconvolve.
- `handle_exceptions` returns `DefaultReturnType` with `[[maybe_unused]]` params; all tensor-returning functional
  error sites `return` it.
- `<torch/torch.h>` instead of partial torch headers; guard renamed; audio_util.hpp removed (no users).
- Tests: `test_convolve_shape_checks` (functional), `audio_test_noexcept` (new ctest target built with
  -DLIB_TORCH_MEDIA_NO_EXCEPTIONS), io test includes `<torchmedia.hpp>` first.

## Acceptance criteria
- [x] Red on e7041b5: fftconvolve(0-D) threw c10 "Dimension specified as -1", non-broadcastable convolve threw
  c10::Error, no-exceptions build returned the full result for a bad mode and computed 0-D input (5 failures,
  then abort), io test failed to compile (`torch/fft.h: unknown type name 'Tensor'`).
- [x] Green; -Wall -Wextra on a TU using the headers: no globel_include.hpp warnings in either mode (baseline
  had 1 / 2 unused-parameter warnings).
- [x] ctest green (6 tests).
