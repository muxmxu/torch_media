# Task 10 — Classification: stat-score and confusion-matrix families
id: 2026-10-24/task10
parent: 2026-10-24/progress02
status: done
owner: code_agent

## Objective
Port torchmetrics 1.9.0 StatScores, Accuracy, Precision, Recall, FBetaScore, F1Score, Specificity,
NegativePredictiveValue, HammingDistance, ExactMatch, ConfusionMatrix, JaccardIndex, MatthewsCorrCoef and CohenKappa
(binary / multiclass / multilabel; functionals and classes) to `torchmedia::metric::classification`, op for op.

## Scope
In:
- `libtorchmedia/include/torchmedia/_metric/_classification_stat.hpp` (NEW): shared helpers (logit normalization,
  bincount, the tensor checks), option structs, functionals, class templates and aliases; `_common.hpp` gains
  `safe_divide` (moved from `_segmentation.hpp`); `metric.hpp` includes the new headers.
Out:
- The task= wrapper classes; validate_args (always on).

## Inputs (read first, priority order)
1. `torchmetrics/functional/classification/{stat_scores,accuracy,precision_recall,f_beta,specificity,
   negative_predictive_value,hamming,exact_match,confusion_matrix,jaccard,matthews_corrcoef,cohen_kappa}.py` and the
   classes of the same names — the reference.
2. `develop_log/2026-10-23/progress11-metric-vision.md` D1 / D7 / D8 — conventions.

## Deliverables
- The 40 functionals and 40 classes of the families above, option structs per D1 of progress02.

## Acceptance criteria
- [x] Every torchmetrics argument available (threshold, top_k, average, multidim_average, ignore_index,
  zero_division, beta, normalize, weights); torchmetrics' input formatting and validation (D2).
- [x] Goldens bit-exact (task13); deliberate differences listed in progress02 D7.

## Constraints
- Header-only, `inline` free functions, no new dependency; validation before any state change.
