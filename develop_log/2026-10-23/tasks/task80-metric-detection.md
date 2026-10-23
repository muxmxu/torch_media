# Task 80 — Metric module scaffolding and detection metrics
id: 2026-10-23/task80
parent: 2026-10-23/progress11
status: done
owner: code_agent

## Objective
Create `torchmedia::metric` and its detection part: torchvision.ops box ops, the torchmetrics IoU metrics,
MeanAveragePrecision numerically equal to pycocotools COCOeval, and VOC devkit average precision.

## Scope
In:
- `libtorchmedia/include/torchmedia/metric.hpp` (aggregator, included by `torchmedia.hpp`), `_metric/_common.hpp`
  (class base, reduce, checks, numpy summation order), `_metric/_detection.hpp`.
- box_area, box_iou, generalized / distance / complete_box_iou, box_convert; intersection_over_union and its G / D / C
  variants (functional + classes); mean_average_precision / MeanAveragePrecision (bbox and dense-mask segm, macro /
  micro, class_metrics, thresholds / maxDets options); voc_average_precision.
Out:
- extended_summary, ("bbox", "segm") pairs, faster_coco_eval, keypoints, PanopticQuality.

## Inputs (read first, priority order)
1. `.venv/.../pycocotools/cocoeval.py` — evaluate / accumulate / summarize to reproduce.
2. `.venv/.../torchmetrics/detection/{mean_ap.py, helpers.py, iou.py, giou.py, diou.py, ciou.py}` and
   `functional/detection/*.py` — conversion to COCO, class semantics.
3. `.venv/.../torchvision/ops/{boxes.py, _box_convert.py, _utils.py}` — box ops.
4. `develop_log/2026-10-23/progress10` D4 / D5 — option and validation conventions.

## Deliverables
- The headers above; progress11 D1-D4, D7, D8.

## Acceptance criteria
- [x] Box ops bit-exact with torchvision for float32 / float64 / int64 / int16 / float16 boxes.
- [x] MeanAveragePrecision's 12 stats and per-class values equal pycocotools' float64 values on every golden and on
  the randomised sweep (progress11 D10).
- [x] VOC AP equal to an independent numpy devkit implementation.
- [x] Invalid input -> `std::invalid_argument` before any state change; classes support update / compute / reset /
  forward.

## Constraints
- Header-only, inline, C++23, no new dependency; Python-order scalar math under `TORCHMEDIA_FP_CONTRACT_OFF`.
