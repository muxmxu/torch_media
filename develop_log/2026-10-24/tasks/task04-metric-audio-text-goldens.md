# Task 04 — Audio / text goldens, coverage and parity sweep
id: 2026-10-24/task04
parent: 2026-10-24/progress01
status: done
owner: code_agent

## Objective
Pin task01-03 in ctest against torchmetrics / pystoi / mir_eval / scipy / sklearn, reach 100% line coverage of
`_audio.hpp` and `_text.hpp`, and measure parity on a randomised sweep.

## Scope
In:
- `unit_test/metric/{CMakeLists.txt, gen_golden.py, golden_{audio,text}.hpp, test_{audio,text}.cpp}`; ctest targets
  `metric_test_audio`, `metric_test_text`.
- `unit_test/metric/sweep/{sweep_audio_gen.py, sweep.cpp, cov_union.sh}` (opt-in `metric_sweep`, not in ctest).
  Scratch (not committed): `scratchpad/metric-audio/{build-cov, sweep1, sweep2}`.
Out:
- Parity off arm64.

## Acceptance criteria
- [x] Speech-like inputs from a generator shared with C++ (checksums), batch dims, float32 / float64 / float16,
  silent references and predictions, short STOI signals at the pystoi length boundary, tied scores, CJK / mixed
  strings; every validation branch; batched updates equal torchmetrics' accumulated states; forward() returns the
  batch alone on a stateful instance.
- [x] Full Release build + ctest green; `_audio.hpp` / `_text.hpp` 100% lines in their owning binaries, `_common.hpp`
  100% as the union.
- [x] Sweep numbers recorded in progress01 D9.
