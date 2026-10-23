# Task 13 — Classification goldens, coverage and parity sweep
id: 2026-10-24/task13
parent: 2026-10-24/progress02
status: done
owner: code_agent

## Objective
Pin tasks 10-12 in ctest against torchmetrics, reach 100% line coverage of the three `_classification_*.hpp` headers,
and measure parity on a randomised sweep.

## Scope
In:
- `unit_test/metric/{gen_golden_classification.py, golden_classification.hpp, classification_driver.hpp,
  classification_test_util.hpp, test_classification_{stat,curve,misc}.cpp, CMakeLists.txt}`; ctest targets
  `metric_test_classification_{stat,curve,misc}`.
- `unit_test/metric/sweep/{sweep_classification_gen.py, sweep_classification.cpp, compare.py, cov_union.sh}` (opt-in
  `metric_sweep_classification`, not in ctest). Scratch (not committed): `scratchpad/metric-cls/{build-cov, impl}`.
Out:
- Parity off arm64.

## Acceptance criteria
- [x] Random cases over the three tasks, every option, ignore_index, logits vs probabilities vs labels, ties, NaN
  scores, absent classes, one-class targets; functional and two-update class compared; every validation branch;
  forward() returns the batch alone on a stateful instance; reset; compute before update.
- [x] Full Release build + ctest green; the three headers 100% lines in their owning binaries, `_common.hpp` 100% as the
  union.
- [x] Sweep numbers recorded in progress02 D10.
