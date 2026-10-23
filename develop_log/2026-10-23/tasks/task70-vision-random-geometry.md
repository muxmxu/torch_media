# Task 70 — Random geometry transform classes with seeded torchvision parity
id: 2026-10-23/task70
parent: 2026-10-23/progress10
status: done
owner: code_agent

## Objective
`torchmedia::vision::transform` random geometry classes whose seeded runs equal torchvision 0.20.1 v2's: RandomCrop,
RandomResizedCrop, RandomResize, RandomShortestSize, ScaleJitter, RandomZoomOut, RandomRotation, RandomAffine,
RandomPerspective, ElasticTransform.

## Scope
In:
- `_vision/_transform_geometry.hpp`: option structs, the classes, `detail::setup_angle` / `check_center` /
  `setup_number_or_seq` (progress10 D1, D3-D5).
Out:
- Bounding boxes / masks (RandomIoUCrop), PIL.

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/_geometry.py` (`__init__`, `_get_params`, `_transform` of each class).
2. `.venv/.../torchvision/transforms/v2/_utils.py`, `transforms/transforms.py` (`_setup_size`, `_setup_angle`,
   `_check_sequence_input`), `transforms/functional.py` (`_get_perspective_coeffs`).

## Acceptance criteria
- [x] Same torch draws, order and dtypes as each `_get_params`; seeded outputs, "returns the input" calls and the next
  draw equal torchvision's (goldens, sweep).
- [x] Constructor validation like torchvision's `__init__` (`std::invalid_argument`); torch-level errors stay
  `c10::Error`.
- [x] 100% line coverage of `_transform_geometry.hpp`.
