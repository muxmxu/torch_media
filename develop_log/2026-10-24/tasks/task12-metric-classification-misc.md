# Task 12 — Classification: calibration, hinge, ranking
id: 2026-10-24/task12
parent: 2026-10-24/progress02
status: done
owner: code_agent

## Objective
Port torchmetrics 1.9.0 CalibrationError (l1 / l2 / max) and HingeLoss (squared, crammer-singer / one-vs-all) for
binary / multiclass, and the multilabel CoverageError, RankingAveragePrecision and RankingLoss.

## Scope
In:
- `libtorchmedia/include/torchmedia/_metric/_classification_misc.hpp` (NEW).
Out:
- Nothing else.

## Inputs (read first, priority order)
1. `torchmetrics/functional/classification/{calibration_error,hinge,ranking}.py` and their classes.

## Deliverables
- 7 functionals and 7 classes.

## Acceptance criteria
- [x] Goldens bit-exact (task13); float64 hinge sums for float64 inputs (progress11 D5) documented.
