# Task 41 — Vision transform classes + ImageNet recipe golden
id: 2026-10-23/task41
parent: 2026-10-23/progress07
status: done
owner: code_agent

## Objective
`torchmedia::vision::transform` offers the deterministic torchvision v2 classes needed for inference preprocessing,
and the classic `Resize(256) + CenterCrop(224) + ToDtype(float32, scale) + Normalize` recipe matches torchvision end
to end.

## Scope
In:
- `_vision/_transform.hpp` (renamed from the empty `_vision/_transform`), included from `vision.hpp`: Resize,
  CenterCrop, Pad, Normalize, ToDtype, Grayscale, RandomHorizontalFlip, RandomVerticalFlip, FiveCrop, TenCrop
  (progress07 D5).
- `unit_test/vision/transform/{main.cpp, gen_golden.py, golden.hpp, CMakeLists.txt}`, ctest `vision_test_transform`.
Out:
- RandomResizedCrop / RandomCrop and other random transforms; Compose.

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/{_geometry.py, _misc.py, _color.py, _transform.py}` — constructors, and
   `_RandomApplyTransform.forward` for the flip draw.
2. `develop_log/2026-06-01/progress02` D1, D2, D5 — class / option-struct idiom.

## Deliverables
- Option structs with `_field` members and same-name fluent setters; `operator()` + `forward()`; constructor
  validation like torchvision's classes.

## Acceptance criteria
- [x] Each class equals its functional op with the same arguments (`torch::equal`), including `forward`.
- [x] Recipe on a synthetic 375 x 500 uint8 image (input checksums match Python's): uint8 order (resize first) and
  float order (ToDtype first) match the torchvision Compose output exactly on arm64 (14 x 14 grid per channel, float64
  sum and sum of squares over all 150528 values); the intermediate Resize(256) is 3 x 256 x 341 uint8.
- [x] Flips: with `torch::manual_seed` 7 / 11 the 12-call flip sequences equal torchvision's, and the next
  `torch::rand({1})` equals torchvision's next draw (one draw per call); p = 0 never and p = 1 always flips.
- [x] Constructor validation throws `std::invalid_argument` (size, padding, p incl. NaN); outputs follow an MPS input.
- [x] `_transform.hpp` 100% line coverage (`llvm-cov report`); full ctest green.

## Constraints
- No cached tensors (outputs follow the input's device / dtype); do not touch `_audio/` or `unit_test/audio/`.

## Notes / Assumptions
- Resize(256) with the default antialias cannot run on MPS in torch 2.5.1 (no `_upsample_bilinear2d_aa` kernel),
  in torchvision as here.
