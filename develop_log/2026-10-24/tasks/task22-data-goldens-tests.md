# Task 22 — Data: goldens, tests, coverage
id: 2026-10-24/task22
parent: 2026-10-24/progress03
status: done
owner: code_agent

## Objective
Pin tasks 20-21 in ctest against torchvision / torch goldens and closed-form checks, with 100% line coverage of every
`_data/*.hpp`.

## Scope
In:
- `unit_test/data/{CMakeLists.txt, gen_golden.py, golden.hpp, test_folder.cpp, test_dataset.cpp, test_noexcept.cpp}`;
  ctest `data_test_folder`, `data_test_dataset`, `data_test_noexcept`; `unit_test/CMakeLists.txt`.
  Scratch (not committed): `scratchpad/data/{build-cov, impl}`.
Out:
- Parity off macOS arm64.

## Acceptance criteria
- [x] gen_golden.py builds the golden tree (nested dirs, case, hidden / non-ASCII classes, symlinks incl. broken ones,
  unlistable dir, empty classes, wrong extensions) and records torchvision's outputs; the C++ test rebuilds the same
  tree in `temp_directory_path()` with files from save_image / save_audio and removes it.
- [x] DataLoader integration: batch sizes, drop_last, padded batches + lengths, workers > 0 see every sample once.
- [x] random_split goldens for several seeds; every validation branch; no-exceptions build.
- [x] Full Release build + ctest green; `_collate.hpp`, `_folder.hpp`, `_manifest.hpp` 100% lines in their owning
  binaries, `_dataset.hpp` 100% as the per-line union (progress03 D9).
