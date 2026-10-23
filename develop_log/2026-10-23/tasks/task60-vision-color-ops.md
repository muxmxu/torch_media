# Task 60 — Vision functional color ops (torchvision v2 parity)
id: 2026-10-23/task60
parent: 2026-10-23/progress09
status: done
owner: code_agent

## Objective
`torchmedia::vision::functional` gains torchvision 0.20.1's color kernels, point-wise identical to torchvision on
uint8 and floating images with any leading dims.

## Scope
In:
- `_vision/_functional.hpp`: `adjust_contrast`, `adjust_saturation`, `adjust_sharpness`, `adjust_hue`,
  `adjust_gamma`, `posterize`, `solarize`, `autocontrast`, `equalize`, `gaussian_blur`, `erase`; `detail::luma`,
  `detail::blend`, `detail::rgb_to_hsv`, `detail::hsv_to_rgb`, `detail::bad_channels` (progress09 D2).
- Goldens in `unit_test/vision/functional/gen_golden.py`, tests in `main.cpp`.
Out:
- ColorJitter / RandomPhotometricDistort (carry-over), `permute_channels`, `gaussian_noise`, PIL kernels.

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/functional/_color.py`, `_misc.py` (`gaussian_blur_image`), `_augment.py`
   (`erase_image`).
2. progress07 D1-D6 (conventions), progress09 D2, D4, D8.

## Deliverables
- The 11 ops with torchvision's op order (in-place sequences included) and validation (`std::invalid_argument`).

## Acceptance criteria
- [x] Goldens bit-exact on arm64 for uint8 / float32 / float64, 1- and 3-channel, 4-D and 5-D, every branch
  (1-channel early returns, H <= 2 sharpness, constant channels in autocontrast / equalize, posterize bits >= value
  bits, inplace erase).
- [x] Dev sweep (progress09 G1): every color op bit-exact against torchvision on CPU; MPS equal to torchvision-on-MPS
  except adjust_hue, where MPS equals torchvision's CPU result (D8).
- [x] Validation paths covered; 100% line coverage of `_functional.hpp`; ctest green.

## Constraints
- Do not touch `_audio/`, `unit_test/audio/`, `readme.md`, `LISCENSE`.

## Notes / Assumptions
- `gaussian_blur`'s default sigma is computed with FMA contraction off (D4).
