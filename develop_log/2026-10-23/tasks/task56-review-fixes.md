# Task 56 — review fixes R1-R7
id: 2026-10-23/task56
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Fix the seven review findings on task50-task55 (progress08 D9), each real one with a test that fails before.

## Scope
In: `_functional_filtering.hpp` (as_coeff, mps_flush comment), `_transform_time.hpp` (PitchShift window),
`libtorchmedia/CMakeLists.txt`, functional / transform tests and gen_golden.py, readme.md (via afollow/readme-check).

## Deliverables
- R1: `detail::as_coeff` on the CPU when `loop_on_host`; MPS checks of three low-corner designers in
  `test_p8_device_input_runs_host_loops`.
- R2: MPS autograd limit documented (mps_flush comment, readme Behaviour, progress08 Open / TODO).
- R3: `target_compile_options(torch_media INTERFACE ...-ffp-contract=on)`; readme Behaviour note.
- R4 / R5: `test_p8r_contraction_rounding_goldens` (functional), `test_p8r_lfcc_pitch_shift_goldens` (transform);
  goldens in the gen_golden.py "progress08 review fixes" sections.
- R6: PitchShift window cast to the input's dtype; task11 float64 goldens at tolerance 0; the kernel-cache test
  compares float64 with a fresh PitchShift instead of functional::pitch_shift.
- R7: readme vision snippet comment ([C, H, W], C from the file).

## Acceptance criteria
- [x] Red: R1 3 checks, R3 14 checks (under `-ffp-contract=fast`), R6 7 checks; R4 / R5 goldens fail under the
  mutants flanger_fb_fma, vad_mean_fma, istft_window_norm, lfcc_matmul_order, pv_round_mul.
- [x] Green: ctest 10 / 10, union line coverage 4273 / 4273, -Wall -Wextra clean, git clang-format clean.
