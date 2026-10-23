# Task 03 — Port torchiva AuxIVA-IP / OverIVA / ILRMA to torchmedia::audio::separation
id: 2026-10-23/task03
parent: 2026-10-23/progress03
status: done
owner: code_agent

## Objective
`separation::auxiva_ip` / `separation::ilrma` separate a complex multichannel STFT and match torchiva 0.1.1 AuxIVA_IP
point-wise for the same seeded input.

## Scope
In:
- NEW `_audio/_separation.hpp` (option struct, enum, two free functions, `detail` helpers), included from `audio.hpp`.
- NEW ctest target `audio_test_separation` (`unit_test/audio/separation/{CMakeLists.txt, main.cpp, gen_golden.py}`).
- THIRD_PARTY_LICENSES torchiva entry; CLAUDE.md namespace list + coverage line.
Out:
- Any STFT/iSTFT code (consume `transform::Spectrogram` / `transform::InverseSpectrogram`).
- Edits to `_functional*.hpp` / `_transform*.hpp` (a parallel branch owns them).
- Other torchiva separators, beamformers, WPE; MPS/CUDA enablement.

## Inputs (read first, priority order)
1. `.venv/lib/python3.12/site-packages/torchiva/{auxiva_ip.py, base.py, linalg.py, models/classic.py, parameters.py}` —
   the algorithm to port, op for op.
2. `develop_log/2026-06-01/progress02-audio-transform-layer.md` — option-struct idiom (`_field` + fluent setters).
3. `.venv/lib/python3.12/site-packages/pyroomacoustics/bss/{auxiva.py, ilrma.py}` — independent cross-check.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_separation.hpp`, `libtorchmedia/include/torchmedia/audio.hpp`
- `unit_test/audio/CMakeLists.txt`, `unit_test/audio/separation/*`

## Deliverables
- `_separation.hpp` — `enum class source_model { laplace, gauss, nmf }`; `auxiva_option` (`_n_iter`, `_n_src`, `_model`,
  `_n_basis`, `_proj_back_mic`, `_eps`, `_load` (review round, progress03 D7) + same-name setters); `auxiva_ip(const_tensor_lref_t, const auxiva_option & = {})`
  and `ilrma(const_tensor_lref_t, auxiva_option = {})`, `[..., C, F, N]` complex -> `[..., S, F, N]`.
- `main.cpp` — torchiva goldens (laplace, gauss, ILRMA, OverIVA, complex128, batch, proj_back_mic -1 / non-zero /
  >= n_src, eps / load 1e-3), closed-form separation with a single global permutation, compact-array load fixture,
  projection-back scale equivariance, model floors, waveform pipeline, all validation branches.
- `gen_golden.py` — regenerates every golden constant from torchiva with the same seeds.

## Steps
1. **Port** — follow `AuxIVA_IP.forward` line by line; keep torchiva's op order and einsum equations; NMF init drawn with
   `uniform_` T then V; everything under `NoGradGuard`, every tensor from `X.options()`.
2. **Goldens** — seeded `randn` inputs (libtorch generator == PyTorch), print |Y|^2, a seeded complex checksum and four
   elements per case; bake into `main.cpp`.
3. **Closed form / pipeline / validation tests**; coverage run; clang-format.
4. **Cross-check** (dev-time only) vs torchiva full tensors and vs pyroomacoustics; record numbers in progress03.

## Acceptance criteria
- [x] Full build green; `ctest --test-dir build` 6/6 green (`audio_test_separation` 265/265 checks after the review
  round; 224/224 before).
- [x] `_separation.hpp` 100% line coverage (169/169; also 100% functions 16/16, regions 80/80, branches 52/52).
- [x] C++ output bit-identical to torchiva on all 16 golden cases (full tensors, dev harness).
- [x] Closed-form test: one permutation across all frequencies, min SIR > 15 dB, mean > 30 dB for laplace/gauss/nmf.
- [x] Every validation branch throws `std::invalid_argument`.
- [x] clang-format applied to the new/changed C++ lines.

## Constraints
- Header-only, `inline`, C++23, `std::format` for messages (no fmt).
- No new dependency; torchiva / pyroomacoustics are dev-time references only.

## Notes / Assumptions
- Assumption: callers pass a contiguous STFT when they need bitwise parity with torchiva (layout changes reduction
  order; see progress03 G1).
- Resolved: default diagonal load is 1e-8; torchiva parity via `load(1e-5)` (progress03 D7).
- Deferred: complex64 projection-back equilibration (progress03 G5) and an MPS path (progress03 Open/TODO).

## Result
Done 2026-10-23; review findings fixed the same day (progress03 Agent log). Numbers in progress03 "Measurements".
Default load decided by the orchestrator (progress03 D7).
