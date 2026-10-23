# Task 31 — ImageFolder PNM defaults
id: 2026-10-24/task31
parent: 2026-10-24/progress04
status: done
owner: code_agent

## Objective
With task30 in place, `image_extensions(load_image(true))` lists .ppm / .pgm again (torchvision's IMG_EXTENSIONS order),
and progress03 records the fix.

## Scope
In:
- `libtorchmedia/include/torchmedia/_data/_folder.hpp` (extension list, comments), `unit_test/data/test_folder.cpp`,
  `develop_log/2026-10-24/progress03-data-layer.md` (D3, D9, Issues P2, Open, Agent log), `CLAUDE.md` (Vision I/O).
Out:
- The decode_image default loader (JPEG / PNG only, like torchvision.io).

## Acceptance criteria
- [x] A folder with PNGs, a 16-bit binary .pgm and an ASCII .ppm: the default loader lists the PNGs; load_image(true)
  lists all four with PNM = sample / maxval, and a Stack loader batches them. Red before the change.
- [x] data tests green (data_test_folder, data_test_dataset, data_test_noexcept); `_folder.hpp` 100% lines.
