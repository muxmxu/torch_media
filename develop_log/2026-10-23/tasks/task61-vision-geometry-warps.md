# Task 61 — Vision functional geometry: rotate / affine / perspective / elastic
id: 2026-10-23/task61
parent: 2026-10-23/progress09
status: done
owner: code_agent

## Objective
Grid-sampled warps identical to torchvision 0.20.1's `rotate_image`, `affine_image`, `perspective_image` and
`elastic_image` (face alignment, deskew).

## Scope
In:
- `_vision/_functional.hpp`: `rotate`, `affine`, `perspective`, `elastic`; `detail::inverse_affine_matrix`,
  `affine_output_size`, `affine_grid`, `grid_error`, `apply_grid_transform`, `py_mod`; the
  `TORCHMEDIA_FP_CONTRACT_OFF` macro (progress09 D3, D4).
Out:
- RandomRotation / RandomAffine / RandomPerspective / ElasticTransform classes (carry-over); bounding boxes / masks.

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/functional/_geometry.py` (`_get_inverse_affine_matrix`,
   `_compute_affine_output_size_tensor`, `_apply_grid_transform`, `_assert_grid_transform_inputs`, `_affine_grid`,
   `_perspective_grid`, `_create_identity_grid`) and `transforms/functional.py` (`_get_perspective_coeffs`).

## Acceptance criteria
- [x] Goldens bit-exact (uint8 / float32 / float64, batch, nearest / bilinear and bicubic for elastic, fill None /
  scalar / per channel, center, expand, the 0 / 90 / 180 / 270 fast paths, integral and fractional perspective
  points, coefficients, CPU float16 elastic).
- [x] Dev sweep: 3458 rotate, 1081 affine, 702 perspective, 243 elastic cases bit-exact (or raising on both sides).
- [x] Without the FP-contract pragma the sweep shows mismatches (186 cases), with it none (D4).
- [x] Validation (`std::invalid_argument`) for interpolation, fill length, center, translate, scale, shear, points,
  coefficients, displacement shape, rank.

## Constraints
- Same libtorch ops as torchvision (`grid_sampler`, `bmm`, `linspace` in the same dtypes).
