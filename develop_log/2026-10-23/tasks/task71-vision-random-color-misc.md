# Task 71 — Color / misc random classes, containers, functional ops, header split
id: 2026-10-23/task71
parent: 2026-10-23/progress10
status: done
owner: code_agent

## Objective
ColorJitter, RandomPhotometricDistort, RandomChannelPermutation, RGB, RandomErasing, GaussianNoise, Identity,
LinearTransformation, ConvertImageDtype and the containers Compose / RandomApply / RandomChoice / RandomOrder, equal to
torchvision 0.20.1 v2 under the same seed; the transform layer split into themed headers.

## Scope
In:
- `_vision/_transform.hpp` -> aggregator of `_transform_misc.hpp`, `_transform_geometry.hpp`, `_transform_color.hpp`
  (phase 1-2 classes moved unchanged; progress10 D2).
- `_vision/_functional.hpp`: `grayscale_to_rgb`, `permute_channels`, `gaussian_noise`; the symmetric pad guard
  (progress10 D6).
- `transform_fn` containers over any class or lambda (progress10 D4).
Out:
- AutoAugment family, MixUp / CutMix, JPEG (carry-over).

## Inputs (read first, priority order)
1. `.venv/.../torchvision/transforms/v2/_color.py`, `_augment.py`, `_misc.py`, `_container.py`, `_transform.py`.
2. `.venv/.../torchvision/transforms/v2/functional/_color.py`, `_misc.py` (the three kernels).

## Acceptance criteria
- [x] ColorJitter draws `randperm(4)` then one uniform per configured factor; RandomPhotometricDistort's gate / factor /
  coin / permutation sequence; RandomErasing's attempts, value and position draws; containers' gate / multinomial /
  randperm — all equal torchvision's (goldens, sweep).
- [x] Functional goldens for the three new ops; symmetric pad wider than the image raises `c10::IndexError` on CPU and
  MPS.
- [x] 100% line coverage of `_transform_misc.hpp`, `_transform_color.hpp`, `_functional.hpp`.
