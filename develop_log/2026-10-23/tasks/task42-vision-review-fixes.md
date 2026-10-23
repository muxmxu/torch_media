# Task 42 — Vision review-round fixes
id: 2026-10-23/task42
parent: 2026-10-23/progress07
status: done
owner: code_agent

## Objective
Fix the verified review findings on the vision geometry core (progress07 D8), each with a test that fails before
and passes after.

## Scope
In:
- `_vision/_functional.hpp`: `to_dtype` int -> float with an out-of-place `mul`; `rgb_to_grayscale` casts integer
  inputs to float32 before the chain.
- `unit_test/vision/functional/{gen_golden.py, golden.hpp, main.cpp}`: goldens `to_dtype_u8_f16`,
  `to_dtype_u8_bf16`, `to_dtype_i32_f16`, `resize_bicubic_noaa_up_i8`, `resize_bicubic_noaa_up_i32`;
  `GOLDEN_HAS_MPS` guard; uint8 MPS grayscale against the CPU golden.
Out:
- `readme.md` wording (R4, Mux's call); the non-CPU uint8 bicubic clamp value check (no device here runs it).

## Acceptance criteria
- [x] Before the header fix: to_dtype_u8_f16 / u8_bf16 / i32_f16 fail (max |d| 4.9e-4 / 3.9e-3 / 2.3e-5) and the
  MPS uint8 grayscale call aborts the process; after: 258/258.
- [x] Mutants dropping `kInt8` / `kInt32` from resize's `round_()` list fail the new goldens.
- [x] `gen_golden.py` with MPS patched off writes 97 entries and `GOLDEN_HAS_MPS 0`; `main.cpp` compiles against it.
- [x] Existing golden entries unchanged by the regeneration; full ctest green; 100% lines for `_vision/*.hpp`;
  clang-format clean.

## Constraints
- Do not touch `_audio/` or `unit_test/audio/`; no commit.
