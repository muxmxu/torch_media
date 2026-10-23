# Task 81 — Image quality metrics
id: 2026-10-23/task81
parent: 2026-10-23/progress11
status: done
owner: code_agent

## Objective
Port torchmetrics.image / torchmetrics.functional.image's non-model metrics to `torchmedia::metric::image`, point-wise
equal to torchmetrics.

## Scope
In:
- `_metric/_image.hpp`: PSNR, SSIM (gaussian / uniform, 2-D / 3-D, full image, contrast sensitivity), MS-SSIM, UQI,
  TotalVariation, SpectralAngleMapper, ERGAS, RASE, RMSE-SW, SpatialCorrelationCoefficient,
  VisualInformationFidelity: functional + class each.
Out:
- LPIPS, FID, KID, MiFID, DISTS, PPL, ARNIQA, D_lambda, D_s, QNR, PSNRB, image_gradients.

## Inputs (read first, priority order)
1. `.venv/.../torchmetrics/functional/image/{psnr, ssim, uqi, tv, sam, ergas, rase, rmse_sw, scc, vif, utils}.py`.
2. `.venv/.../torchmetrics/image/*.py` — class states and compute.
3. `.venv/.../torch/_tensor.py` — `__rdiv__` (reciprocal * scalar).

## Deliverables
- The header; progress11 D5.

## Acceptance criteria
- [x] Every functional and class bit-exact with torchmetrics on arm64 for float32 inputs (classes) and float32 /
  float64 inputs (functionals), goldens and sweep.
- [x] torchmetrics quirks kept and documented (progress11 D5); float64 class states documented (D7).
- [x] Every argument check of torchmetrics present as `std::invalid_argument`.

## Constraints
- Same libtorch ops in the same order as torchmetrics; dtype / device follow the input.
