# Task 72 — Seeded goldens, ImageNet training recipe, coverage and parity sweep
id: 2026-10-23/task72
parent: 2026-10-23/progress10
status: done
owner: code_agent

## Objective
Pin task70 / task71 against torchvision 0.20.1 in ctest and measure the seeded parity on a dev sweep.

## Scope
In:
- `unit_test/vision/transform/gen_golden.py` -> `golden.hpp`: seeded cases (outputs, "is the input" flags, next draw),
  LinearTransformation goldens, the ImageNet training recipe for seeds 0-2; `main.cpp` replays them.
- `unit_test/vision/functional/`: goldens for the new functional ops and the symmetric pad guard.
- Scratch (not committed): `scratchpad/vision3/impl/sweep/{sweep_gen.py, probe.cpp}` (CPU and MPS sweeps),
  `scratchpad/vision3/impl/mutants/run.py`.
Out:
- x86 parity (no machine here).

## Acceptance criteria
- [x] Every new class: several seeds / parameter sets, uint8 and float32, batch dims, edge cases (crop larger than the
  image with pad_if_needed, degenerate ranges, p = 0 / 1); all bit-exact on arm64.
- [x] Full build + ctest green; 100% line coverage for every `_vision/*.hpp`.
- [x] Sweep numbers and surviving mutants recorded in progress10 D8 / D9.
