# Progress 08 — Any compatible libtorch: drop torch::linalg::, measure 2.14.1
id: 2026-10-24/progress08
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-24/progress07]
supersedes: [2026-10-24/progress07 D2 "Not done" item (version enforcement): rejected by Mux]
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/{_functional.hpp, _functional_beamforming.hpp}
  - readme.md, doc/{porting-from-python.md, reproducibility.md}, CLAUDE.md

## Goal
TorchMedia builds against any libtorch that keeps the API, not only 2.5.1, and the docs say what a different version
changes.

## Context / Motivation
progress07 left open whether CMake should reject a libtorch other than 2.5.x. Mux ruled (2026-10-24):
"只要 libtorch 兼容，有什么问题呢，非要用版本给用户添麻烦是干什么" — no version pinning. Checking that stance showed the
headers themselves did not compile against Homebrew's libtorch 2.14.1: 7 calls used the C++ frontend namespace
`torch::linalg::` (solve x5, eigh, eigvals), whose header `torch/csrc/api/include/torch/linalg.h` no longer exists there.

## Decisions

### D1 — ATen functions instead of `torch::linalg::`
- `torch::linalg::solve / eigh / eigvals` -> `torch::linalg_solve / linalg_eigh / linalg_eigvals`. In 2.5.1 the
  removed wrappers are one-line forwards to exactly these functions, so the kernels and results are unchanged.
- CLAUDE.md gains the rule (cross-version APIs; goldens stay PyTorch 2.5.1).

### D2 — Docs: a version is a parity condition, not a requirement
readme Requirements / Use-it note, porting "libtorch version", reproducibility "The libtorch version" (with the
2.14.1 measurement below).

## Verification
- `-fsyntax-only -Wall -Wextra` of `torchmedia.hpp` (with the I/O implementation): 0 errors against 2.5.1 and 2.14.1
  (before: 7 errors against 2.14.1).
- libtorch 2.5.1: full Release build, ctest 24/24.
- libtorch 2.14.1 (Homebrew, `/opt/homebrew/lib/libtorch.dylib`): full build of all 24 test targets, 0 errors; every
  binary runs to completion (no exception, no crash). 371 of 20,710 checks fail, all value comparisons against the
  PyTorch 2.5.1 goldens:
  - 322 differ by less than 5e-7 and 41 by less than 1e-3 (newer kernels' rounding);
  - SDR of a short signal moves by 2.55 dB and `sdr_cg` by 3e-3 (ill-conditioned solve), and a PIT tie picks the
    other permutation;
  - float16 `gaussian_noise` draws a different stream (max 0.56); MPS uint8 bilinear resize is 1 level off; a float16
    UQI input that 2.5.1 rejected now computes;
  - InverseMelScale returns a different number of exact zeros after its least-squares solve (59 in 2.5.1).
  None points at a TorchMedia defect. The goldens are not version-aware; ctest is the 2.5.1 gate.

## Open
- Whether the new values equal Python PyTorch 2.14 on the same calls is not measured (no 2.14 Python environment).
