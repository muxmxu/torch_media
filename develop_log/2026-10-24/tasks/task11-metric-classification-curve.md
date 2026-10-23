# Task 11 — Classification: curve family
id: 2026-10-24/task11
parent: 2026-10-24/progress02
status: done
owner: code_agent

## Objective
Port torchmetrics 1.9.0 PrecisionRecallCurve, ROC, AUROC, AveragePrecision, LogAUC, EER, PrecisionAtFixedRecall,
RecallAtFixedPrecision, SensitivityAtSpecificity and SpecificityAtSensitivity (binary / multiclass / multilabel;
functionals and classes) with exact and binned thresholds.

## Scope
In:
- `libtorchmedia/include/torchmedia/_metric/_classification_curve.hpp` (NEW): thresholds_spec, curve result types,
  option structs, the exact / binned states, the curve computations, the reductions, class template and aliases.
Out:
- BinaryPrecisionRecallCurve's class-only `normalization` kwarg; plot().

## Inputs (read first, priority order)
1. `torchmetrics/functional/classification/{precision_recall_curve,roc,auroc,average_precision,logauc,eer,
   precision_fixed_recall,recall_fixed_precision,sensitivity_specificity,specificity_sensitivity}.py`, the classes,
   `torchmetrics/utilities/{compute,data}.py` (two different `interp`s).

## Deliverables
- 30 functionals and 30 classes; binary results as tuples, per-class results as one tensor per class (progress02 D1).

## Acceptance criteria
- [x] Exact and binned states equal one-shot results across updates; torchmetrics' binned comparison-dtype rule
  (vectorized vs loop) reproduced; class quirks of D7 documented.
- [x] Goldens bit-exact (task13); the BinaryEER / audio EER relation measured (progress02 D9).
