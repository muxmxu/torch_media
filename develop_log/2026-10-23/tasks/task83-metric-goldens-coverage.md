# Task 83 — Metric goldens, coverage and parity sweep
id: 2026-10-23/task83
parent: 2026-10-23/progress11
status: done
owner: code_agent

## Objective
Pin task80-82 against pycocotools / torchmetrics / torchvision in ctest, reach 100% line coverage of every
`_metric/*.hpp`, and measure parity on a randomised sweep.

## Scope
In:
- `unit_test/metric/{CMakeLists.txt, gen_golden.py, golden_{detection,image,segmentation}.hpp, metric_test_util.hpp,
  test_{detection,image,segmentation}.cpp}`; ctest targets `metric_test_{detection,image,segmentation}`.
- `unit_test/metric/sweep/{sweep_gen.py, gen_golden_voc.py, sweep.cpp, compare.py, cov_union.sh}` (moved from the
  scratch directory on 2026-10-24 so D10 / D9 can be reproduced from the repo; opt-in CMake target `metric_sweep`,
  not in ctest). Scratch (not committed): `scratchpad/metric-vision/build-cov`.
Out:
- Parity off arm64.

## Acceptance criteria
- [x] COCO-like datasets (120 images, crowds, all area ranges, empty images, classes only in predictions / targets,
  score ties), maxDets truncation, degenerate sets, segm masks; image / segmentation inputs from a generator shared
  with C++ and checksummed; every validation branch; batched updates equal one shot.
- [x] Full build + ctest green; 100% line coverage for every `_metric/*.hpp` (per owning binary; `_common.hpp` union).
- [x] Sweep numbers recorded in progress11 D10.
- [x] Review fixes (2026-10-24, progress11 Agent log): forward() on stateful instances, multi-image batches,
  boundary / crowd-mask / many-class goldens, segm state size, MPS detection; mutants of each killed.
