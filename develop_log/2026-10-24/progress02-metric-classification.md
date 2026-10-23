# Progress 02 — Metric module, phase 3: classification metrics (torchmetrics.classification parity)
id: 2026-10-24/progress02
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress11, 2026-10-24/progress01]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_metric/_classification_stat.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_classification_curve.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_classification_misc.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_common.hpp (safe_divide moved here from _segmentation.hpp)
  - libtorchmedia/include/torchmedia/_metric/_segmentation.hpp
  - libtorchmedia/include/torchmedia/metric.hpp
  - unit_test/metric/{CMakeLists.txt, gen_golden_classification.py, golden_classification.hpp, classification_driver.hpp,
    classification_test_util.hpp, test_classification_{stat,curve,misc}.cpp} (NEW except CMakeLists.txt)
  - unit_test/metric/sweep/{sweep_classification_gen.py, sweep_classification.cpp (NEW), compare.py, cov_union.sh}
  - CLAUDE.md

## Goal
Phase 3 of `torchmedia::metric` (progress11, progress01): `metric::classification` mirroring torchmetrics 1.9.0
torchmetrics.classification for the binary / multiclass / multilabel variants — StatScores, Accuracy, Precision,
Recall, F1Score, FBetaScore, Specificity, NegativePredictiveValue, ConfusionMatrix, HammingDistance, ExactMatch,
JaccardIndex, MatthewsCorrCoef, CohenKappa, AUROC, AveragePrecision, ROC, PrecisionRecallCurve, LogAUC, EER,
PrecisionAtFixedRecall, RecallAtFixedPrecision, SensitivityAtSpecificity, SpecificityAtSensitivity, CalibrationError,
HingeLoss and the multilabel ranking metrics CoverageError, RankingAveragePrecision, RankingLoss. Out of scope: Open.

## Context / Motivation
Mux asked for a measurement module; the phase-3 design (metric list, options, parity bar, tests) was set by the
orchestrator under Mux's delegation. Layout and conventions are progress11's (D1, D7, D8) and progress01's: a functional
plus a torchmetrics-named class with update / compute / reset / forward, `xxx_option` structs, goldens from a Python
generator, `std::invalid_argument` through `handle_exceptions`, deliberate differences listed (D7 below).

## Decisions

### D1 — Layout and API shape
- Namespace `torchmedia::metric::classification`, three headers aggregated by `metric.hpp`: `_classification_stat.hpp`
  (shared helpers; the stat-score family, ExactMatch, the confusion-matrix family), `_classification_curve.hpp`
  (PrecisionRecallCurve, ROC, AUROC, AveragePrecision, LogAUC, EER, the fixed-point metrics) and
  `_classification_misc.hpp` (CalibrationError, HingeLoss, ranking).
- Functionals keep torchmetrics' names (`binary_accuracy`, `multiclass_auroc`, `multilabel_ranking_loss`, ...; 77
  functionals) and take `(preds, target, option)`; classes keep torchmetrics' names (`BinaryAccuracy`,
  `MulticlassAUROC`, ...; 77 classes) and are aliases of a few templates over a family state (progress01's WER-family
  precedent): `detail::stat_class<Opt, K>`, `exact_match_class<Opt>`, `confmat_class<Opt, K>`, `curve_class<Opt, K>`,
  `calibration_error_class<Opt>`, `hinge_loss_class<Opt>`, `ranking_class<K>`; the logic lives in non-template functions.
  `metric_base` (progress11 D1) provides forward / operator() / reset.
- Options are shared by family, one struct per task, with the union of the family's keyword arguments (a field a metric
  does not take is ignored; required arguments are validated):
  `binary_ / multiclass_ / multilabel_stat_scores_option` (threshold, num_classes ({} only with average micro, like
  torchmetrics) / num_labels, top_k, average, multidim_average, ignore_index, zero_division, beta — FBetaScore only;
  also ExactMatch), `binary_ / multiclass_ / multilabel_confusion_matrix_option` (normalize — ConfusionMatrix,
  average / zero_division — JaccardIndex, weights — CohenKappa), `binary_curve_option` (thresholds, ignore_index,
  max_fpr — AUROC, fpr_range — LogAUC, min_precision / min_recall / min_specificity / min_sensitivity — the fixed-point
  metrics, required there), `multiclass_ / multilabel_curve_option` (PrecisionRecallCurve, ROC, EER (multiclass average
  none / micro / macro), fixed-point), `multiclass_ / multilabel_auc_option` (AUROC, AveragePrecision, LogAUC; average
  default macro), `binary_ / multiclass_calibration_error_option`, `binary_ / multiclass_hinge_loss_option`,
  `multilabel_ranking_option`. Enums: `average_mode {micro, macro, weighted, none}` (none = torchmetrics' "none" /
  None), `multidim_average_mode`, `normalize_mode {none, true_, pred, all}` (true_ = "true"), `kappa_weights_mode`,
  `calibration_norm_mode`, `hinge_multiclass_mode`.
- thresholds: `thresholds_spec = std::variant<std::monostate, int64_t, std::vector<double>, tensor_t>` with setters for
  a count (`torch.linspace(0, 1, n)`), a named vector, a braced list (also `{0.5}`) and a tensor; {} is torchmetrics'
  None (exact curves).
- Results: tensors with torchmetrics' dtypes (int64 counts, float32 rates, thresholds in the score dtype, ...). Curves:
  binary `curve_t = std::tuple<tensor_t, tensor_t, tensor_t>`; multiclass / multilabel `curve_list_t` (three vectors):
  one tensor per class — exact curves as torchmetrics' lists, binned curves as the rows of torchmetrics' [C, T (+ 1)]
  tensors with the shared thresholds repeated — or one tensor each for an averaged (micro / macro) curve. Fixed-point
  metrics return `{value, threshold}` pairs.
- Choosing the class instead of torchmetrics' task= wrapper (`Accuracy(task="multiclass", num_classes=3)`): call the
  task's own form — `MulticlassAccuracy(multiclass_stat_scores_option().num_classes(3))`,
  `multiclass_accuracy(preds, target, opt)`. The wrappers (and the Fairness / GroupStatRates metrics) are not provided.

### D2 — Input formatting = torchmetrics
- Logits: `normalize_logits_if_needed` — on the CPU a floating tensor with any value outside [0, 1] (NaN included) goes
  through sigmoid (binary / multilabel) or softmax over dim 1 (multiclass curves, calibration, hinge); other devices
  decide on values < 0 or > 1 only (`torch.where`). The stat family applies it to the whole input before ignore_index,
  the curves after dropping ignored entries (binary / multiclass), as torchmetrics.
- Comparisons with Python-float options (threshold, min_*) are tensor-scalar ops: the scalar takes the tensor's dtype.
- ignore_index: stat family marks ignored targets -1 (binary / multilabel), drops them (multiclass global, top_k 1) or
  writes -1 one-hot rows / an extra class (samplewise, top_k > 1; `ignore_index` inside [0, C) or not); curves drop them
  (binary / multiclass), multilabel binned curves skip them, multilabel exact states keep them until compute (per label)
  and the ranking metrics score them -4 L like torchmetrics' format.
- Validation as torchmetrics' `_xxx_tensor_validation`, including its quirk that the multiclass unique-value checks run
  only for integer preds (`for t in ((target,),) + ((preds,),) if not preds.is_floating_point() else ()`).
- `_bincount`: `torch.bincount`, or the arange / eq form on MPS and in deterministic mode (same counts).

### D3 — Stat-score and confusion-matrix families, op for op
- `_multiclass_stat_scores_update`'s three paths: global top_k 1 (bincount confusion matrix; micro: equality counts),
  samplewise / top_k > 1 (one-hot comparison, `_refine_preds_oh`: the target class where it is among the top_k scores,
  else the top-1); binary / multilabel `.squeeze()` of the sums (a one-sample samplewise binary state then fails in
  `stack(dim=1)` like torchmetrics).
- Reductions `_accuracy_reduce`, `_precision_recall_reduce`, `_fbeta_reduce` (beta^2 as CPython's `pow`),
  `_specificity_reduce`, `_negative_predictive_value_reduce`, `_hamming_distance_reduce` with
  `_adjust_weights_safe_divide` (macro weights drop classes with tp + fp + fn = 0, or tp + fn = 0 for top_k > 1 — only
  where torchmetrics passes top_k: accuracy, precision, recall, NPV); `_safe_divide` with zero_division.
- MatthewsCorrCoef: Python's builtin `sum()` over 1-D float32 tensors (sequential float32 adds, not torch.sum), the
  int64 +-1 early returns of a perfect / inverted 2 x 2 table, the eps branches of a zero denominator (the last four
  reachable only through float32 rounding of huge counts: 2^25 true negatives, tested). CohenKappa weights none /
  linear / quadratic; JaccardIndex average binary / micro (minus the ignore_index class' denominator) / macro / weighted
  / none; ConfusionMatrix normalize with NaN -> 0; ExactMatch (ignored positions count as right).

### D4 — Curve family
- Exact curves: `_binary_clf_curve` (unstable descending argsort; values at tie-group ends do not depend on the order),
  tps from a float32 cumsum, precision / recall with the appended (1, 0) point, recall all ones when every target is 0
  (torchmetrics checks `target == 0` whatever pos_label is), ROC from (0, 0) at threshold 1 with zero fpr / tpr for a
  class without negatives / positives (`zeros_like(thresholds)`: the score dtype).
- Binned curves: the [T, (C,) 2, 2] confusion matrices of `preds >= threshold`, computed per threshold. torchmetrics'
  vectorized update compares in the promoted dtype of preds and the thresholds and its loop (more than 50000 binary
  values / more than 1e6 multiclass `M C C`; multilabel always vectorized) against a 0-d threshold, i.e. in preds'
  dtype: torchmedia follows the same rule (a float64 threshold just above 0.3f counts 0.3f as negative up to 50000
  values and as positive beyond; tested and checked against torchmetrics).
- Macro-averaged curves accumulate `interp` in place like torchmetrics (the float32 sum stays float32 when a class
  curve is float64); AUROC max_fpr: McClish standardization (bucketize, lerp); AveragePrecision
  `-sum(diff(recall) * precision)` with NaN -> 0 on stacked curves only; LogAUC with torchmetrics.utilities.data.interp
  (numpy-style: argsorted xp, plain-division slopes, searchsorted) — not the compute.interp of the macro curves; EER at
  the first ROC point minimising |fpr - fnr|; fixed points: `_lexargmax` over (recall, precision, threshold),
  Python's `max()` over (precision, recall, threshold) tuples (first maximum kept, NaN never wins), argmax for
  sensitivity / specificity, threshold NaN for a 0 value and 1e6 (float16: inf) without a qualifying point.
  AUROC / AveragePrecision weights: positives per class (exact: target counts; binned: the first threshold's
  positives).

### D5 — CalibrationError, HingeLoss, ranking
- CalibrationError: binary confidence = the positive-class probability with accuracy = target; multiclass = the top
  softmax probability (float32) and whether its class is the target; `bucketize(right) - 1` into n_bins + 1 slots,
  l1 / max / l2 (`sqrt(ce) if ce > 0 else torch.tensor(0)`: int64 0).
- HingeLoss on torchmetrics' formatted scores (binary: sigmoid of logits; multiclass: softmax), crammer-singer (target
  score minus the best other) or one-vs-all (per class), squared.
- CoverageError / RankingAveragePrecision (`_rank_data` via unique + cumsum, the per-sample loop in float32) /
  RankingLoss (double argsort ranks; samples with no or only relevant labels count as 0, element count unchanged).

### D6 — Classes
- States as torchmetrics: int64 tp / fp / tn / fn sums starting at `zeros(size)` (or per-sample batches for
  samplewise), int64 confusion matrices, exact curve batches or summed binned confusion matrices, calibration batches,
  float32 hinge / ranking sums. `compute()` applies Metric's `_squeeze_if_scalar` to every returned tensor (a curve's
  one-threshold thresholds become 0-d; rows of a stacked per-class curve are not squeezed, as torchmetrics' [C, n]
  tensor is not). Before any update: torchmetrics' zero state (stat / confusion matrix / binned / hinge / ranking) or
  torch.cat of no batch (c10::Error: samplewise stat, exact curves, calibration).
- torchmetrics quirks kept and pinned: NegativePredictiveValue's classes ignore zero_division; ExactMatch samplewise
  divides by the last batch's position count; MultilabelJaccardIndex's class reduces without ignore_index (its
  functional uses it; golden case95 / case96).
- Exact-curve and calibration states store a copy of each update (detach().clone()), unlike torchmetrics' by-reference
  list states, so a caller that refills one buffer between updates (`from_blob` over a reused array, `copy_` into a
  preallocated tensor) cannot rewrite earlier updates (decided on merge, see Agent log; same for audio
  EqualErrorRate / MinDCF and the detection IoU per-class labels).

### D7 — Deviations (deliberate)
- Stricter checks, `std::invalid_argument` before any state change: zero_division in {0, 1} everywhere (torchmetrics:
  stat classes and FBeta functionals only), top_k >= 1 and max_fpr in (0, 1] and min_* in [0, 1] (torchmetrics' checks
  `not isinstance(x, int) and x < 1` never fire for typed values), fpr_range always for LogAUC (torchmetrics: binary
  functional; multiclass with validate_args; multilabel functional never; BinaryLogAUC's class default
  validate_args=False), empty threshold lists / tensors, 0-d stat inputs and 1-D multilabel inputs (torchmetrics:
  IndexError), a batch-size mismatch between multiclass preds [N, C, ...] and target (torchmetrics compares shape[2:]
  only, then broadcasts or fails in torch), an average on the fixed-point metrics (no such argument there).
- Classes always check their inputs: torchmetrics' BinaryAUROC constructor never sets validate_args and BinaryLogAUC
  defaults it to False, so their updates check nothing.
- MulticlassPrecisionRecallCurve / ROC / EER classes with average micro and binned thresholds keep the [T, 2, 2] state
  their update produces; torchmetrics allocates [T, C, 2, 2] and raises (or mis-broadcasts silently when T == C).
- Float64 hinge sums for float64 inputs (progress11 D5).
- ExactMatch samplewise forward() accumulates like update() (the last batch's count); torchmetrics' forward merges
  `total` with its "mean" reduction (differs only for batches with different position counts).
- The stat functionals with num_classes {} (average micro) use one class like the classes (torchmetrics' functionals
  other than multiclass_accuracy raise a TypeError there).
- Multilabel binned curves (PrecisionRecallCurve, ROC, AUROC, AveragePrecision, LogAUC, EER, the fixed-point
  metrics) skip ignored entries with a mask, so a non-int64 target with ignore_index gives the int64 target's result
  (tested for uint8). torchmetrics codes them with its -4 L sentinel, which wraps in uint8, and its bincount raises
  ("shape '[3, 2, 2, 2]' is invalid"). Still raising like torchmetrics on such targets: multilabel ConfusionMatrix /
  JaccardIndex / MatthewsCorrCoef with ignore_index (the same sentinel wrap) and the multiclass binned curves (one_hot
  needs int64).
- Threshold tensors move to the input's device. LogAUC's shared option defaults to average macro (torchmetrics'
  functionals); its LogAUC classes default to None — set `average(average_mode::none)` for that.
- Not provided: validate_args (always on), BinaryPrecisionRecallCurve's class-only `normalization` kwarg (used by the
  fairness metrics), plot(), the task= wrappers, Fairness / GroupStatRates.

### D8 — Validation and no-exceptions builds
- As progress11 D8: `xxx_error(...) -> std::string`, `return detail::invalid(msg)` / `return detail::fail(msg)`,
  constructors assign fallback options (num_classes / num_labels 2, the fixed-point minimum 0) before failing.
  torchmetrics' torch-level errors stay c10::Error: out-of-range labels in one_hot / bincount, an empty input after
  ignore_index (IndexError on the empty curve), a NaN recall column in `_lexargmax` (IndexError), torch.cat of no batch.
  `audio_test_noexcept` (torchmedia.hpp with `LIB_TORCH_MEDIA_NO_EXCEPTIONS`) builds the new headers.

### D9 — BinaryEER vs metric::audio::equal_error_rate
- Same inputs, different conventions: `classification::binary_eer(preds, target)` is torchmetrics' — ROC from
  `_binary_clf_curve` (preds in [0, 1] as given, else float32 sigmoid; thresholds optional), EER = (fpr + fnr) / 2 at
  the first ROC point minimising |fpr - fnr|, no interpolation, float32 rates. `audio::equal_error_rate(target_scores,
  nontarget_scores)` (progress01 D5) is the speaker-verification form: float64 sklearn ROC, linear interpolation between
  the two points around the fpr = fnr crossing.
- Measured (scratch `metric-cls/impl/eer_cmp.py`, 2000 random trials, 5-2000 target / non-target Gaussian scores,
  separation 0-3, 0-3 decimal rounding for ties, scores mapped monotonically into (0, 1) in float64 so torchmetrics
  applies no sigmoid): equal in 3 trials; |difference| max 0.031, mean 5.9e-4, median 2.0e-4; always within half the
  larger fpr / fnr step at the crossing (2000 / 2000). Equal whenever a ROC point has fpr = fnr (closed-form test).
- Logits differ beyond that: torchmetrics' float32 sigmoid saturates (18, 19, 20, 21 -> 1.0), merging ROC points:
  target logits {18, 19, 20} vs non-target {21, 0} give 0.25 (BinaryEER) and 0.5 (audio EER) — tested. Use the audio
  form for verification trials (raw scores, interpolated EER), BinaryEER for torchmetrics parity.

### D10 — Tests, goldens, coverage, measured parity
- `gen_golden_classification.py` (separate from gen_golden.py, which it does not rerun) draws random cases
  (`make_case`) and evaluates every metric of the task with torchmetrics (`expected`): the functional on the whole
  input and the class over the two halves, curve tuple / list elements as `<key>.<i>`, a raised exception as
  `<key>!`; D7's class differences are expressed through torchmetrics itself (validate_args forced on, the micro binned
  state fixed, float64 hinge sums). Cases cover the three tasks, probabilities on a 0.05 grid (ties), uniform, logits
  and labels, float32 / float64 / float16, extra dims, NaN scores, ignore_index (-1, 255, 0, C - 1), absent classes,
  one-class targets, all-same predictions, every option above (thresholds None / count / list / float32 or float64
  tensor, unsorted too). `targeted()` adds 7 hand-made cases for what the random draws left unpinned (a mutation run
  found survivors there): multiclass micro global top_k 1 (case90), top_k 2 with a predicted class absent from the
  target (case91), top_k 2 with ignore_index -1 global / samplewise (case92 / case93), multiclass jaccard micro with
  ignore_index 1 (case94), MultilabelJaccardIndex functional vs class with ignore_index 0, micro / macro (case95 /
  case96). `golden_classification.hpp` holds the 97 cases (7034 outputs, 932 of them torchmetrics exceptions) as text
  (a std::map initialiser table took minutes to compile), parsed by `classification_test_util.hpp`;
  `classification_driver.hpp` evaluates the same keys with torchmedia for the ctest binaries and the sweep.
- ctest `metric_test_classification_stat` (4284 checks), `..._curve` (7430), `..._misc` (822): every golden output
  bit-exact (dtype, shape, values, raised-or-not), plus closed-form values, every validation branch, class rejection
  with state untouched, forward() on a stateful instance returns the batch alone, reset, compute before update, the
  binned comparison-dtype rule at its switch points (binary 50000 / 50002 values, multiclass 1e6 / 1e6 + 4 `M C C`,
  micro 50000 / 50002, multilabel always promoted), one-threshold stacked rows staying 1-D, a uint8 multilabel target
  (D7), MCC's rounding branches and its Python `sum()` (a 20-class table whose value torch.sum changes), the EER
  relation (D9), deterministic bincount, and MPS against the CPU: stat counts, PrecisionAtFixedRecall, binned
  multilabel / multiclass ROC on [N, L, E] / [N, C, E] inputs exact, logit AUROCs within 1e-6 (MPS sigmoid / softmax
  round differently).
- MPS parity is measured against torchmetrics run on MPS, not against the CPU values (scratch
  `metric-cls/review-conventions-robustness/{py/mps_expected.py, probe/probe_mps2.cpp}`, rerun on the fixed headers:
  `metric-cls/fix/probe_mps2.out`): the 71 float64-free golden cases, 4948 outputs — 4109 equal (dtype, shape, values),
  830 raised on both, 9 raised by torchmetrics only (case19, its broken micro binned class, D7), 0 mismatches.
  Degenerate MatthewsCorrCoef tables differ between MPS and the CPU in torchmetrics too (float32 `s**2` rounding:
  [[0, 0], [13, 6]] is -0.1818 on the CPU, 0.0 on MPS), so the CPU goldens do not carry over to MPS.
- Coverage (Debug + `TORCHMEDIA_COVERAGE`, scratch `metric-cls/build-cov`): `_classification_stat.hpp` 1004/1004,
  `_classification_curve.hpp` 1014/1014, `_classification_misc.hpp` 314/314 lines in their owning binaries;
  `_common.hpp` 88/88 as the union (`cov_union.sh`, now over the eight metric binaries); `_segmentation.hpp` 291/291
  (safe_divide moved to `_common.hpp`), `_audio.hpp` 987/987, `_detection.hpp` 853/853, `_image.hpp` 997/997,
  `_text.hpp` 236/236 unchanged.
- Sweep (`sweep/sweep_classification_gen.py` + opt-in `metric_sweep_classification` + `compare.py`; 2% of the sweep
  cases have 3000-20000 samples, i.e. torchmetrics' loop updates). Seeds 1, 2, 3 with 3000 cases each (2951 binary,
  3053 multiclass, 2996 multilabel; 156 large): 605335 outputs (every metric of the task, functional and two-update
  class, every curve element), 503047 values bit-exact (dtype, shape and values, NaN == NaN), 102288 raised in both
  (label preds for the score metrics, samplewise without extra dims, ...), 0 with equal values but another dtype or
  shape, 0 mismatches. compare.py prints these per metric and as totals (it counts an entry as bit-exact only with the
  same dtype and the same unflattened shape); rerun on the fixed headers (scratch `metric-cls/fix/sw{1,2,3}`) with
  the same totals. Earlier runs during the work found the bugs listed in Issues (LogAUC's interp, the in-place macro
  sum) and torchmetrics' class bugs of D7.
- Mutation check (review harness `metric-cls/review-conventions-robustness/mutate.py`, every mutant force-rebuilt):
  17 / 28 mutants failed ctest before the review fixes; with the targeted goldens and the new direct checks 26 / 28
  do. The two survivors do not change any output: the binned class weights read at the first vs the last threshold
  (tp + fn does not depend on the threshold) and `ignore_index == C - 1` routed as out of range (the -1 target rows
  hide those predictions either way).

## Tasks
- [x] [task10 — stat-score and confusion-matrix families](tasks/task10-metric-classification-stat.md)
- [x] [task11 — curve family](tasks/task11-metric-classification-curve.md)
- [x] [task12 — calibration, hinge, ranking](tasks/task12-metric-classification-misc.md)
- [x] [task13 — goldens, coverage, parity sweep](tasks/task13-metric-classification-goldens.md)

## Issues / Gotchas
- torchmetrics' LogAUC imports `interp` from utilities.data (numpy-style), the curves' macro averaging the one from
  utilities.compute: the same name, different formulas (slopes by plain division, searchsorted vs a sum of ge).
- In-place accumulation matters: `mean_tpr += interp(...)` keeps the float32 sum when one class curve is float64
  (`zeros_like(thresholds)` of float64 scores); an out-of-place sum promoted it (1 float32 ulp).
- A class updated batch by batch normalizes logits per batch (the softmax / sigmoid decision is per update), so it can
  differ from the functional on the concatenation — torchmetrics' too.
- `torch::full({}, 1e6, float16 options)` raises (overflow check); torchmetrics' `torch.tensor(1e6, dtype=float16)` is
  inf: build it from a float64 tensor.
- A braced `{0.5}` binds to an integer parameter before `std::vector<double>`: the thresholds setters take an
  `std::initializer_list<double>` too.
- llvm-cov counts the closing brace of a `case` block after a `return` as an uncovered line: keep case bodies
  brace-free.
- Review fixes (2026-10-24): (1) PrecisionAtFixedRecall always raised on MPS: its float64 tie-break called
  `.to(kFloat64)` on the MPS tensors ("Cannot convert a MPS Tensor to float64"; `.to(kCPU, kFloat64)` in one call
  fails too) — it now moves to the CPU first. (2) Multilabel binned curves on MPS gave wrong counts without an error
  for [N, L, ...] inputs: the formatted [M, L] scores and target are non-contiguous views there, and MPS (torch 2.5.1)
  computes `&` of two non-contiguous bool views wrongly — the binned update now makes them contiguous (counts are
  integers, so the CPU results do not move; exact curves keep torchmetrics' layout, which matters to 1 ulp).
  (3) A mutation run showed 11 of 28 mutants surviving ctest at 100% line coverage (random cases that never drew
  multiclass micro global top_k 1, top_k 2 with an absent class, ...); the goldens gained 7 targeted cases and the
  curve / stat tests direct checks (D10), and the 9 survivors that change a value now fail. (4) compare.py compared
  float64-flattened values only, so the sweep's dtype / shape claim rested on an uncommitted script; it now checks
  dtype and shape itself and prints the totals.
- Python: `import torchmetrics.functional.classification.matthews_corrcoef as M` binds the *function* of that name
  (the package re-exports it over the submodule); patch a module through `sys.modules[f.__module__]`.
- The exact-curve and calibration classes store their inputs by reference (D6). torchmetrics does the same, and so do
  `audio::EqualErrorRate` (`.detach()`) and the detection IoU classes' labels: two updates through one reused buffer
  give BinaryAUROC 0.0 instead of 0.5 (review probe). Whether the module should clone its stored batches is open (Open).

## Open / TODO (carry-over)
- Fairness / GroupStatRates (not media), the task= wrapper classes, BinaryPrecisionRecallCurve's `normalization`
  kwarg, plot().
- Parity off arm64 not measured (the tests allow 1e-5 there).
- readme.md does not mention the metric module yet — Mux's call.

## Agent log
- 2026-10-24 [ai] Read torchmetrics 1.9.0 functional/classification and classification (all files above, plus
  utilities.compute / data / checks and Metric.forward / _wrap_compute). Implemented D1-D8 (3 headers, 77 functionals,
  77 classes). Goldens bit-exact (90 cases, 6330 outputs); ctest metric_test_classification_{stat,curve,misc}
  3899 / 6455 / 758 checks; coverage 100% (D10); sweep 605335 outputs, 0 mismatches (D10); D9 measured. The new
  headers and tests build without warnings under -Wall -Wextra; clang-format on every new / changed C++ line. Full
  Release build + ctest 18 / 18 green. Not committed.
- 2026-10-24 [ai] Review fixes (findings P1, P2, CR1-CR6). Fixed: PrecisionAtFixedRecall on MPS (CPU before float64)
  and multilabel binned curves on MPS for [N, L, ...] inputs (contiguous before the bool ops); both tests failed before
  the fix (MPS raise; 3 label rows off). Pinned: the uint8-target multilabel binned deviation (D7, new test), the
  binned loop / vectorized switch for multiclass / micro / multilabel, one-threshold stacked rows, MCC's Python sum,
  and 7 targeted golden cases (D10); mutation re-run: 26 / 28 killed, the 2 survivors equivalent. compare.py checks
  dtype and shape and prints totals; sweep rerun 605335 outputs, 503047 bit-exact, 102288 raised on both, 0
  mismatches. MPS claims reworded (parity with torchmetrics on MPS: 4948 outputs, 0 mismatches). Documented, left to
  Mux: stored input batches are references (CR5; Open). Goldens 97 cases / 7034 outputs; ctest
  metric_test_classification_{stat,curve,misc} 4284 / 7430 / 822 checks; coverage `_classification_stat.hpp`
  1004/1004, `_classification_curve.hpp` 1014/1014, `_classification_misc.hpp` 314/314, `_common.hpp` 88/88 (union);
  full Release build + ctest 18 / 18 green; clang-format clean. Not committed.
- 2026-10-24 [ai] Orchestrator decision on CR5 (Mux's delegation): metric states own their data. C++ inference loops
  commonly reuse one input buffer (from_blob over a reused array, copy_ into a preallocated tensor), which silently
  rewrote earlier updates. Exact-curve and calibration states, audio EqualErrorRate / MinDCF and the detection IoU
  per-class labels now store detach().clone(). Four regression tests (update, overwrite the buffer, compare with a
  reference) fail with the old aliasing code and pass now. Deliberate difference from torchmetrics (it keeps references).

