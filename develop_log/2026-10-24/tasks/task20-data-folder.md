# Task 20 — Data: DatasetFolder, ImageFolder, AudioFolder
id: 2026-10-24/task20
parent: 2026-10-24/progress03
status: done
owner: code_agent

## Objective
Port torchvision 0.20.1 `DatasetFolder` / `ImageFolder` (with `find_classes`, `make_dataset`,
`has_file_allowed_extension`) to `torchmedia::data` as torch::data datasets, and add an AudioFolder over the library's
audio decoder.

## Scope
In:
- `libtorchmedia/include/torchmedia/_data/_dataset.hpp` (shared types, `detail::invalid`, `detail::join`),
  `_data/_folder.hpp` (path-dataset base, walk, the three free functions, DatasetFolder, `image_loader` /
  `image_extensions` / ImageFolder, `audio_loader` / `audio_extensions` / AudioFolder), `torchmedia/data.hpp`,
  `torchmedia.hpp`.
Out:
- Re-implementing loader machinery (make_data_loader, samplers, Stack stay libtorch's); `~` expansion of root.

## Inputs (read first, priority order)
1. `torchvision/datasets/folder.py` (0.20.1) — the reference, incl. `sorted(os.walk(..., followlinks=True))`.
2. `dependence/libtorch/include/torch/csrc/api/include/torch/data/{datasets/base.h, datasets/map.h,
   dataloader/stateless.h}` — the Dataset contract and the per-worker dataset copy.
3. `_vision/_io.hpp`, `_audio/_io.hpp`, `_audio/_functional.hpp` (resample) — the decoders reused.

## Deliverables
- The API of progress03 D1-D3, D7, D8.

## Acceptance criteria
- [x] Sample order, targets, class_to_idx and error messages equal torchvision's on the golden tree (task22).
- [x] Datasets work with `make_data_loader` (Stack, drop_last, workers > 0) and share their sample list across copies.
- [x] Invalid input: std::invalid_argument / std::runtime_error / std::out_of_range through handle_exceptions; the
  no-exceptions build returns defaults.

## Constraints
- Header-only, `inline` free functions, no new dependency; option structs `_field` + fluent setters.
