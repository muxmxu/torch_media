# Task 41 — Route raw throws; composites and constructors honour the contract
id: 2026-10-24/task41
parent: 2026-10-24/progress05
status: done
owner: code_agent

## Objective
No library error bypasses the user's choice, no composite computes on a part's fallback, and every rejected
constructor leaves its object in the state progress05 D2 documents.

## Scope
In:
- Raw throws -> `handle_exceptions` (progress05 D4): `_audio/_io.hpp` (decoder `error` member, guarded destructors,
  `open` / `load` / `info`), `_audio/_separation.hpp`, `_vision/_io.hpp` (pnm reported at `load_image`).
- Composite checks (D5) in audio functional / kaldi / transforms, vision functional / transforms / io, data loaders.
- Constructor states (D2): audio MelScale, InverseMelScale, MFCC, LFCC, AmplitudeToDB, Vol; vision RandomCrop /
  RandomResizedCrop size, LinearTransformation, RandomErasing scale, RandomZoomOut side_range, RandomPerspective
  distortion_scale, RandomAffine translate, `setup_angle`, RandomChoice p.
Out:
- Pre-existing non-error-site defects (progress05 Open: speed with orig_freq 0, 0-D lfilter).

## Inputs (read first, priority order)
1. `develop_log/2026-10-24/progress05-silent-mode-contract.md` — D2, D4, D5.
2. `develop_log/2026-10-23/progress10-vision-random-transforms.md` — D5 (callable classes).

## Deliverables
- The headers listed in progress05 `files:`; each new check is a one-line `return` with a short comment.

## Acceptance criteria
- [x] Only two raw throws remain (pnm parser, caught at `load_image`; the image loader rewrapping a c10::Error).
- [x] Default build: every existing test unchanged and green (same types and messages).
- [x] Silent build: the composite / constructor sections of `error_policy_test_silent` pass without exceptions.
