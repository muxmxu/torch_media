# Task 82 — Segmentation metrics
id: 2026-10-23/task82
parent: 2026-10-23/progress11
status: done
owner: code_agent

## Objective
Port torchmetrics.segmentation MeanIoU, DiceScore and GeneralizedDiceScore (functional + class) to
`torchmedia::metric::segmentation`.

## Scope
In:
- `_metric/_segmentation.hpp`: num_classes, include_background, per_class, input_format one_hot / index, DiceScore's
  average and aggregation_level, GeneralizedDiceScore's weight_type.
Out:
- The "mixed" input format; HausdorffDistance (not cheap: distance transforms, edges).

## Inputs (read first, priority order)
1. `.venv/.../torchmetrics/functional/segmentation/{mean_iou, dice, generalized_dice, utils}.py`.
2. `.venv/.../torchmetrics/segmentation/*.py`; `torchmetrics/utilities/compute.py` `_safe_divide`.

## Deliverables
- The header; progress11 D6.

## Acceptance criteria
- [x] Bit-exact with torchmetrics (functional and class, every option combination in the goldens) and on the sweep.
- [x] Argument / input checks as `std::invalid_argument`.

## Constraints
- float32 results like torchmetrics; torchmetrics' GeneralizedDice weight indexing kept.
