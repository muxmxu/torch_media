# Progress 11 — Metric module, phase 1: detection, image quality and segmentation metrics (torchmetrics parity)
id: 2026-10-23/progress11
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress07, 2026-10-23/progress09, 2026-10-23/progress10, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/metric.hpp (NEW, aggregator; included by torchmedia.hpp)
  - libtorchmedia/include/torchmedia/_metric/_common.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_detection.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_image.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_segmentation.hpp (NEW)
  - libtorchmedia/include/torchmedia.hpp
  - unit_test/metric/{CMakeLists.txt, gen_golden.py, golden_{detection,image,segmentation}.hpp,
    metric_test_util.hpp, test_{detection,image,segmentation}.cpp} (NEW)
  - unit_test/metric/sweep/{sweep_gen.py, gen_golden_voc.py, sweep.cpp, compare.py, cov_union.sh} (NEW; opt-in
    target metric_sweep, not in ctest)
  - unit_test/CMakeLists.txt
  - CLAUDE.md

## Goal
A measurement module `torchmedia::metric` mirroring torchmetrics 1.9: phase 1 covers the vision side — detection
(torchvision.ops box ops, the IoU metrics, MeanAveragePrecision equal to pycocotools COCOeval, VOC devkit AP), image
quality (torchmetrics.image minus the model-based metrics) and segmentation (MeanIoU, DiceScore,
GeneralizedDiceScore). Audio and classification metrics are later phases. Out of scope: Open.

## Context / Motivation
Mux asked for a measurement module; the design (layout, metric list, parity bars, test plan) was set by the
orchestrator under Mux's delegation. Conventions follow the transform layers (progress07 D5, progress09 D7,
progress10): option structs with `_field` members and same-name fluent setters, classes with operator() / forward(),
`std::invalid_argument` for invalid input through `handle_exceptions`, dtype / device following the input.

## Decisions

### D1 — Layout and API shape
- Decision: `_metric/{_detection.hpp, _image.hpp, _segmentation.hpp}` plus `_metric/_common.hpp` (shared: the class
  base, torchmetrics' `reduce`, shape checks, numpy's float64 summation order, `TORCHMEDIA_FP_CONTRACT_OFF` guarded
  with the same definition as `_vision/_functional.hpp`), aggregated by `metric.hpp`, which `torchmedia.hpp` includes.
  Namespaces `torchmedia::metric::{detection, image, segmentation}` (room for `audio` / `classification`).
- Every metric has a stateless snake_case functional (torchmetrics.functional / torchvision.ops names and argument
  order; the many keyword arguments of a metric live in its `xxx_option` struct, shared by the functional and the
  class) and a stateful class named like the torchmetrics class with `update(...)`, `compute()`, `reset()`,
  `forward(...)` / `operator()(...)` (torchmetrics' forward: the value of this batch alone, accumulated as well) and
  `options()`. The shared CRTP base `metric::detail::metric_base` implements forward / operator() / reset by building
  a fresh instance from `options()`.
- Enums carry the string options (`reduction_mode`, `box_format_mode`, `iou_type_mode`, `average_mode`,
  `normalize_mode`, `input_format_mode`, `dice_average_mode`, `aggregation_level_mode`, `weight_type_mode`,
  `voc_interpolation_mode`); `{}` is torchmetrics' None for vector options (data_range, dim, thresholds), like
  progress10 D4. torchmetrics' tuple returns are separate `_with_image` / `_with_map` forms
  (`structural_similarity_index_measure_with_image`, `root_mean_squared_error_using_sliding_window_with_map`,
  `StructuralSimilarityIndexMeasure::compute_with_image`).
- Results: tensors in the input's dtype / device for image / segmentation and the IoU classes (torchmetrics' float32
  where it converts). MeanAveragePrecision and VOC evaluate in float64 on the CPU whatever the input device (MPS has
  no float64: inputs are moved to the CPU before the cast), so their result tensors are CPU float64 / int32 / int64;
  class `compute()` squeezes a one-element result to 0-d (torchmetrics `_squeeze_if_scalar`). The IoU classes return
  torchmetrics' dict as `std::map<std::string, tensor_t>` (`"iou"`, `"iou/cl_3"`, ...). MeanAveragePrecision returns
  `mean_average_precision_result` (D3).
- Per-image detection inputs are the structs `detection_prediction {boxes, scores, labels, masks}` and
  `detection_target {boxes, labels, iscrowd, area, masks}` (torchmetrics' dicts; an undefined tensor is a missing key).
- Why: one functional + one class per metric is torchmetrics' shape; the option struct is the codebase's keyword
  argument; building fresh instances keeps forward / reset trivially right for every class.

### D2 — torchvision.ops box ops (bit-exact)
- `box_area`, `box_iou`, `generalized_box_iou`, `distance_box_iou`, `complete_box_iou` (eps 1e-7), `box_convert`
  (xyxy / xywh / cxcywh) call the same libtorch ops in the same order as torchvision 0.20.1 ops/boxes.py, including
  `_upcast` (float16 -> float32, int8 / int16 -> int32; integer boxes keep integer areas) and CIoU's
  `4 / (torch.pi ** 2)` through libm `pow` like CPython. `box_convert` to the same format returns a clone (any shape);
  otherwise a last dim other than 4 is `std::invalid_argument` (torchvision: ValueError from unpacking). Shape errors
  torchvision leaves to torch stay `c10::Error`.

### D3 — MeanAveragePrecision = pycocotools COCOeval through torchmetrics
- Conversion as torchmetrics `_get_safe_item_values` / `_get_coco_format`: boxes to xywh with `box_convert` in the
  input dtype, then float64; area = w * h in float64 unless the target's area > 0; iscrowd nonzero = crowd; image ids
  in update order, annotation order kept. average micro maps every label to class 0; `classes` are the sorted labels
  of predictions and targets (int32).
- COCOeval evaluate / accumulate / summarize reproduced in float64 (`detail::coco_eval`): bbIou (maskApi.c, under
  `TORCHMEDIA_FP_CONTRACT_OFF`), per (image, category) detections in stable descending score order (numpy mergesort;
  NaN last) truncated to maxDets[-1], ground truths stable-sorted ignore-last, the greedy matching with crowd reuse,
  `min(t, 1 - 1e-10)` and the "stop at the first ignored gt once matched to a regular one" rule, unmatched detections
  outside the area range ignored; accumulate's concatenation over images, cumulative TP / FP, precision
  `tp / (fp + tp + np.spacing(1))` (one detection gives 1 / (1 + 2^-52), not 1), the precision envelope,
  `searchsorted(left)` recall lookups that stop at the first threshold past the last recall (Python's swallowed
  IndexError: unsorted rec_thresholds leave the rest 0), -1 where a category has no non-ignored ground truth.
- summarize: means of the entries > -1 in numpy's C order and numpy's float64 `add.reduce` order (pairwise inside each
  8192-element buffer chunk, `detail::np_sum`, measured on numpy 1.26: a plain sum differed in ~1 of 3 large arrays);
  `map` uses COCOeval's literal maxDets = 100 (so it is -1 when 100 is not a threshold, as in torchmetrics), the
  others maxDets[0..2]; iou thresholds 0.5 / 0.75 by exact equality.
- Defaults are torchmetrics': `torch.linspace(0.5, 0.95, 10)` and `torch.linspace(0, 1, 101)` in float32 (0.55 is
  0.550000011920929), not pycocotools' float64 `np.linspace` (they change results; golden `map_da_numpy_thr`).
  max_detection_thresholds: 3 values, sorted.
- class_metrics: torchmetrics re-evaluates each class alone; a category's evaluation is independent of the others,
  so the per-class values are read from the macro evaluation (one extra macro evaluation under micro).
- iou_type segm: dense masks (pycocotools rleIou on counts: intersection / union, crowd: / detection area; areas are
  pixel counts). As in torchmetrics, an image without masks is left out of that side's COCO dataset (detections on an
  image without target masks are not counted) and an empty side gives -1 everywhere.
- segm memory: COCOeval's mask IoUs depend on one image only, so update() computes each image's full [D x G] IoU
  matrix (every detection against every target, input order; float64 pixel counts via `mm`, detections in blocks of
  16 rows to bound the float64 copy) and keeps it with the areas, scores, labels and crowd flags; the masks are
  dropped. compute() reads the (category, maxDets-truncated) blocks from it. State per image: D x G doubles (about
  9 KB for 100 detections x 11 targets) instead of (D + G) x H x W bytes (33.8 MB at 640 x 480, which made a
  5000-image COCO-val segm run need ~165 GB). Measured with the review probe (480 x 640, 100 detections and 3-11
  targets per image): max RSS 0.606 GB at 20 images, 0.611 GB at 80 (was 1.25 / 3.28 GB), stats equal to
  torchmetrics rounded; ctest pins it through libtorch's memory-profiling hook (D9).
- Result: `mean_average_precision_result` with torchmetrics' keys as pycocotools' float64 values (torchmetrics rounds
  them to float32; the goldens assert that torchmetrics equals them rounded): map, map_50, map_75, map_small /
  medium / large, mar_1 / mar_10 / mar_100 (the recalls at max_detection_thresholds[0 / 1 / 2] — torchmetrics renames
  the keys), mar_small / medium / large (doubles); map_per_class / mar_100_per_class (float64 [C], or {-1} without
  class_metrics) and classes.

### D4 — VOC average precision (devkit)
- `voc_average_precision(preds, target, opt)`: per class, detections over all images in descending score order,
  matched to the target of the same image with the largest IoU (first on ties): IoU > threshold (strict) and not
  taken -> TP, taken -> FP, difficult -> ignored, else FP; recall over non-difficult targets, precision
  `tp / max(tp + fp, eps)`; AP all-point (VOC2010+, envelope area) or 11-point (VOC2007, recall 0, 0.1 = 1 * 0.1, ...).
  `pixel_inclusive` (default true) is the devkit's `+ 1` on widths / heights; difficult flags come from `iscrowd`.
  Result: map over classes with a non-difficult target (-1 without), ap_per_class (-1 for such classes), classes.
- Deviation from the devkit: ties take image order then input order (stable); the devkit's quicksort leaves them in an
  unspecified order. Checked against an independent numpy implementation of the devkit formula (gen_golden.py
  `voc_eval_np`, mergesort).

### D5 — Image quality metrics
- PSNR, SSIM (gaussian / uniform, 2-D and 3-D), MS-SSIM (relu / simple / none), UQI, TotalVariation, SAM, ERGAS, RASE,
  RMSE-SW, SCC, VIF: the torchmetrics call sequence op for op. Python's `scalar / tensor` is
  `tensor.reciprocal() * scalar` (Tensor.__rdiv__), which PSNR (`10 / log(base)`), RASE (`100 / target_mean`) and
  GeneralizedDice (`1.0 / support`) rely on; Python scalars stay Python-double arithmetic (`c1 = (k1 * range) ** 2`,
  `int(3.5 * sigma + 0.5)` under `TORCHMEDIA_FP_CONTRACT_OFF`); data_range None makes c1 / c2 tensors in the input
  dtype, as in torchmetrics; `Tensor.norm(dim=1)` is `linalg_vector_norm`.
- torchmetrics quirks kept (each pinned by a golden): SSIM's gaussian kernel size comes from sigma (kernel_size only
  checked, and used for the 3-D depth pad, which lands on the last dim); UQI pads the width by the height's pad;
  RASE divides the target's window mean by window_size^2 twice; SCC's symmetric pad takes the whole side for a zero
  pad (`x[-0:]`); TotalVariation subtracts integer images in their dtype; GeneralizedDice's infinite-weight
  replacement reads the transposed max table (D6).
- Class states start as torchmetrics' float32 zero and accumulate out of place (`state + value`): float32 inputs give
  torchmetrics' bits, float64 inputs keep a float64 state (torchmetrics' in-place float32 state rounds them) on the
  input's device. Classes that keep every batch (ERGAS, RASE, UQI / SAM reduction none) evaluate the concatenation.
- Not provided (D-out): LPIPS, FID, KID, MiFID, DISTS, PPL, ARNIQA (model-based), D_lambda, D_s, QNR (pansharpening),
  PSNRB, image_gradients.

### D6 — Segmentation metrics
- `mean_iou` / MeanIoU, `dice_score` / DiceScore (average micro / macro / weighted / none, aggregation samplewise /
  global), `generalized_dice_score` / GeneralizedDiceScore (weight square / simple / linear), each with
  include_background and input_format one_hot (integer one-hot, or floating scores argmax'ed) / index (int64 class
  maps, num_classes required; mean_iou accepts [H, W] and [L] like torchmetrics). Results float32 (torchmetrics
  `_safe_divide`), functionals per sample, classes as torchmetrics (MeanIoU: IoU sum / count of present (sample,
  class) pairs, -1 per class never present; DiceScore: NaN-mean over samples; GeneralizedDiceScore: mean over samples).
- GeneralizedDice: a class without target pixels takes `max(weights, 0).repeat(N, 1).T.flatten()[i]` at its flat
  index i, torchmetrics' indexing (the right class only for N = 1 or N = C), kept for parity and pinned by the
  `gdice_transposed_{square,simple}` goldens (N = 4, predictions in classes absent from the sample's target; the
  direct lookup gives other values there).
- MeanIoU before any update (and after reset) returns torchmetrics' initial zero state: -1 per class (num_classes,
  minus one without background, or one class when num_classes is inferred later), NaN overall.
- Not provided: the "mixed" input format, HausdorffDistance (distance transforms and edge extraction: not cheap; Open).

### D7 — Deviations from torchmetrics (deliberate)
- IoU classes with class_metrics: an image with predictions but no targets is skipped in the per-class sums
  (torchmetrics raises an IndexError); the global value counts its zero diagonal like torchmetrics.
- Stricter input checks, `std::invalid_argument` before any state change: labels / scores 1-D, iscrowd / area lengths,
  the IoU classes also check prediction boxes against labels, mismatched prediction / target mask sizes (pycocotools:
  IoU -1), negative max_detection_thresholds (Python slicing), empty MS-SSIM betas, kernel_size of 0 or > 3 values.
- No warnings, no `extended_summary`, no ("bbox", "segm") pair, no faster_coco_eval backend. MeanAveragePrecision with
  no image returns -1 stats and map_per_class {-1} (torchmetrics omits the per-class keys there).
- A reset MeanIoU infers num_classes again from the next one-hot batch (torchmetrics keeps the first inference).
- MeanIoU with num_classes = 1 and include_background = false: the class keeps the single channel like both
  functionals (mean_iou gives [1, 1] for identical inputs in torchmetrics and here) and returns that IoU;
  torchmetrics' class sizes its state C - 1 = 0 and returns NaN (per_class: an empty tensor). Before any update both
  give the empty / NaN zero state.
- VisualInformationFidelity (functional and update) requires 4-D [B, C, H, W] input (torchmetrics: IndexError for
  3-D) and the class checks size and shape on update (torchmetrics' class checks nothing: broadcastable batch sizes
  pass there, other mismatches raise from torch).
- MeanAveragePrecision / VOC results on the CPU for MPS inputs (D1).
- Float64 class states (D5).

### D8 — Validation and no-exceptions builds
- Every functional and update validates first through a `xxx_error(...) -> std::string` check and then either
  `return detail::invalid(msg)` (functional: `handle_exceptions` return value) or `return detail::fail(msg)` (update:
  state untouched); constructors assign their fallback (defaults) before failing. No line sits after a throw, so the
  throw lines stay covered and a no-exceptions build never computes on invalid input.
- torch's own errors stay `c10::Error` where torchmetrics leaves them to torch (one_hot of out-of-range or non-int64
  classes, a Half reflection pad, reflect pads wider than the image, `torch.cat` of an empty state).

### D9 — Tests, goldens, coverage
- `unit_test/metric/gen_golden.py` writes `golden_{detection,image,segmentation}.hpp`: pycocotools COCOeval float64
  stats on torchmetrics' conversion (asserting torchmetrics' float32 results equal them rounded), torchmetrics /
  torchvision values, and the numpy VOC devkit. Detection data are baked: a COCO-like set of 120 images (6 classes +
  one only in predictions + one only in targets, crowds, given areas, all area ranges, empty images on either side,
  2-decimal scores with exact ties), 25 images x up to 160 detections (maxDets truncation), no-prediction /
  no-target / all-crowd sets, 14 segm images of rectangle-minus-hole masks (integer parameters, rebuilt in C++).
  Image / segmentation inputs come from a splitmix64 generator mirrored in `metric_test_util.hpp` (checksums pin it).
- Review additions (2026-10-24): `map_de` (40 COCO-like images over 12 classes, so the summary arrays exceed numpy's
  8192-element buffer, plus IoU exactly 0.5, areas exactly 32^2 / 96^2 and an unmatched detection of area 32^2) with
  `voc_de_cont` (VOC: IoU exactly 0.5 is no match); `map_df` (float64 boxes with IoU 1 - 2.5e-11, matched at threshold
  1.0 only through `min(t, 1 - 1e-10)`); `map_dg_segm` (segm detections inside crowd masks); `vif_anti` (locally
  anti-correlated halves, VIF's g < 0 branch); `ssim_dr_target` (data_range None from the target's wider range);
  `gdice_transposed_{square,simple}` (D6). New detection data use a separate generator, so the earlier goldens are
  byte-identical.
- ctest `metric_test_detection` (307 checks), `metric_test_image` (280), `metric_test_segmentation` (153): every golden
  bit-exact on arm64 (`kTol` 0; 1e-5 elsewhere), batches through forward() / operator() / update() equal one shot,
  forward() on an instance that already holds state returns that batch alone (every class), one 2-image update of
  each image class equals two 1-image updates (per-image counters), reset and the MeanIoU empty state, every
  validation branch (VIF's update included), the segm state size (net CPU bytes through libtorch's PROFILER_STATE
  memory-reporting hook: < 1 MB after 4 images of 104 masks of 256 x 256, 27 MB before D3's change), MPS device /
  dtype checks when available (detection: mAP / VOC / IoU classes with class_metrics).
- Mutation check of those tests (fixer scratch `metric-vision/fix/mutate.py`, every mutant force-rebuilt): forward()
  returning the accumulated value, SSIM counting batches, crowd mask IoU over the union, VIF without the g < 0
  branch, SSIM's data_range from preds only, np_sum without 8192-chunks, `<=` on the IoU threshold, `<=` / `>=` on
  the area bounds, VOC `>=`, no `min(t, 1 - 1e-10)`, GeneralizedDice's direct lookup, unique on the MPS device and
  an unchecked VIF update: 15 / 15 killed.
- Coverage (Debug + `TORCHMEDIA_COVERAGE`, scratch `metric-vision/build-cov`, `llvm-cov report` per owning binary;
  `unit_test/metric/sweep/cov_union.sh` prints both): `_detection.hpp` 853/853, `_image.hpp` 997/997,
  `_segmentation.hpp` 299/299 lines; `_common.hpp` 73/73 as the per-line union over the three binaries. llvm-cov's multi-object merge under-reports here (inline header functions
  hash differently per binary: "430 functions have mismatched data"), hence per owning binary.

### D10 — Measured parity (dev sweep, `unit_test/metric/sweep`, not in ctest)
- `sweep_gen.py N seed` + `sweep.cpp` (opt-in target `metric_sweep`; cases through `torch::pickle_load`) +
  `compare.py`, run from a scratch directory (usage in sweep_gen.py); seeds 1 and 2, N = 600 (re-run from the repo
  copy after the 2026-10-24 review fixes, same counts):
  - MeanAveragePrecision vs pycocotools: 1200 random datasets (1-39 images, 1-5 of 10 labels, crowds, given areas,
    1-3 decimal score ties, up to 130 detections per image, all box formats; 1000 bbox + 200 segm), random iou /
    rec thresholds (unsorted rec too), random maxDets, macro / micro, class_metrics: 1200 / 1200 bit-exact
    (12 stats and per-class values; max |delta| 0).
  - VOC vs the numpy devkit: 1000 / 1000 bit-exact (thresholds 0.3 / 0.5 / 0.7, both interpolations, +1 on / off).
  - Image metrics vs torchmetrics (float32 and float64, [1-3, 2-4, 20-69, 20-69], random kernel / sigma /
    data_range / window): PSNR, SSIM, UQI, TV, SAM, ERGAS, RMSE-SW map, RASE, SCC 1200 / 1200 each, VIF 384 / 384,
    MS-SSIM 314 / 314 bit-exact.
  - Segmentation vs torchmetrics: mean_iou, dice (none, weighted), generalized_dice (square per class, simple)
    600 / 600 each bit-exact.
- The goldens (D9) are bit-exact too, including the torchvision box ops in float32 / float64 / int64 / int16 /
  float16.

## Tasks
- [x] [task80 — metric scaffolding and detection metrics](tasks/task80-metric-detection.md)
- [x] [task81 — image quality metrics](tasks/task81-metric-image.md)
- [x] [task82 — segmentation metrics](tasks/task82-metric-segmentation.md)
- [x] [task83 — goldens, coverage, parity sweep](tasks/task83-metric-goldens-coverage.md)

## Issues / Gotchas
- numpy's `np.mean` is not a plain left-to-right sum (pairwise per 8192-element chunk): COCOeval's summary means need
  that order for bit-exactness.
- pycocotools' precision is `tp / (tp + fp + 2^-52)`: one correct detection gives 0.9999999999999998, so a perfect
  result is just below 1.
- torchmetrics' default thresholds are float32 values; feeding pycocotools' float64 defaults changes mAP.
- Python `scalar / tensor` is reciprocal-then-multiply, not a division.
- llvm-cov's multi-binary merge drops inline header functions with per-binary hash differences; measure each header
  with the binary that owns it (as progress10 did) and union per line for shared headers. Template instantiations are
  counted separately, so the IoU classes keep their logic in a non-template base.
- A test macro that expands its argument several times re-ran `forward()` (fixed with a local); keep stateful calls
  out of multi-use macro arguments.
- Keeping per-image dense masks in a class state does not scale (33.8 MB per 640 x 480 image with 100 detections);
  keep what compute() needs (D3). libtorch's PROFILER_STATE `MemoryReportingInfoBase` hook counts net CPU tensor
  bytes, enough to pin a state size in a test.
- MPS has no float64 and no `_unique`: move to the CPU before a float64 cast or a unique.
- SSIM / MS-SSIM classes with data_range None take each batch's own range (torchmetrics too): a 2-image update is not
  two 1-image updates there; pin data_range when comparing batch splits.
- COCO's 'all' area range is [0, 1e10]: a box of area above 1e10 is ignored, so threshold-boundary goldens need
  areas inside it.
- forward() checks on a fresh instance cannot tell the batch value from the accumulated one; check it on an
  instance that already holds state.

## Open / TODO (carry-over)
- HausdorffDistance, the "mixed" segmentation input format; PSNRB, image_gradients; PanopticQuality; mAP
  extended_summary (precision / recall arrays) and the ("bbox", "segm") pair; keypoint OKS.
- Model-based metrics (LPIPS, FID, KID, MiFID, DISTS, PPL, ARNIQA) and the pansharpening indices (D_lambda, D_s, QNR).
- Audio (torchmetrics.audio: SI-SDR, PESQ-free ones) and classification metrics: later phases.
- Parity off arm64 is not measured (the tests allow 1e-5 there; mAP / VOC are pure float64 C++ and platform-free).
- readme.md does not mention the metric module yet — Mux's call.

## Agent log
- 2026-10-23 [ai] Read pycocotools 2.0.11 cocoeval.py, torchmetrics 1.9.0 detection (mean_ap, helpers, iou family),
  image (functional + classes), segmentation (functional + classes, utils) and torchvision 0.20.1 ops/boxes.py.
  Implemented D1-D8 (5 headers, 26 classes, 33 functionals). Tests: detection 283, image 248, segmentation 123 checks,
  all goldens bit-exact; coverage D9; sweep D10. Full Release build + ctest green (13/13). clang-format applied to
  every new / changed C++ file. Not committed.
- 2026-10-24 [ai] Review fixes (parity and conventions reviews). Fixed, each with a test that failed before:
  segm state keeps per-image mask IoUs instead of masks (D3; RSS flat in the image count); MeanAveragePrecision /
  VOC on MPS (CPU before the float64 cast) and the IoU classes' class_metrics on MPS (`unique` on the CPU); MeanIoU
  compute() before update / after reset (torchmetrics' zero state, D6); VisualInformationFidelity::update runs the
  functional's checks (shared `vif_input_error`, 4-D required, D7 / D8); `coco_area_rng` is `inline constexpr` (one
  object across TUs). Test gaps closed: forward() on a stateful instance and 2-image batches for every class, goldens
  for the crowd-mask denominator, VIF g < 0, SSIM data_range None, numpy's 8192-chunk sum (12 classes), the IoU /
  area / VOC threshold boundaries, `min(t, 1 - 1e-10)` and GeneralizedDice's transposed lookup (D9; 15 / 15 mutants
  killed). Documented, not changed: MeanIoU with one class and no background (D7). The sweep and the coverage-union
  script moved into `unit_test/metric/sweep` (opt-in `metric_sweep`); seeds 1 and 2 re-run from there: all
  bit-exact, same counts as D10. The tests build without warnings under -Wall -Wextra (designated initializers,
  inline generator helpers). torchmedia.hpp gains its missing final newline (clang-format). Coverage D9. Full Release
  build + ctest green. Not committed.
