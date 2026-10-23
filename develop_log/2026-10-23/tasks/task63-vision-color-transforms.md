# Task 63 — GaussianBlur and the p-gated color transform classes
id: 2026-10-23/task63
parent: 2026-10-23/progress09
status: done
owner: code_agent

## Objective
`torchmedia::vision::transform` classes whose seeded runs equal torchvision v2's: GaussianBlur, RandomInvert,
RandomAutocontrast, RandomEqualize, RandomGrayscale, RandomPosterize, RandomSolarize, RandomAdjustSharpness.

## Scope
In:
- `_vision/_transform.hpp`: option structs (`gaussian_blur_option`, `random_apply_option`,
  `random_posterize_option`, `random_solarize_option`, `random_adjust_sharpness_option`), the classes,
  `detail::skip` (also used by the flips) and `detail::RandomApply<Op>` (progress09 D7).
- Goldens in `unit_test/vision/transform/gen_golden.py`, tests in `main.cpp`.
Out:
- ColorJitter and the random geometric families (carry-over).

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/_color.py`, `_misc.py` (GaussianBlur), `_transform.py`
   (`_RandomApplyTransform.forward`), `_utils.py` (`_setup_number_or_seq`).

## Acceptance criteria
- [x] GaussianBlur: with `torch::manual_seed(5)` three calls equal torchvision's outputs; a fixed sigma still draws
  once per call (next draw equal).
- [x] Each p-gated class: 12-call applied / skipped sequence and the next draw equal torchvision's; applied calls
  equal the functional op, skipped calls return the input tensor.
- [x] Constructor validation (`std::invalid_argument`) like torchvision; 100% line coverage of `_transform.hpp`.
