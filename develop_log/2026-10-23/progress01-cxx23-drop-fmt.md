# Progress 01 — Move to C++23 and drop the fmt dependency
id: 2026-10-23/progress01
date: 2026-10-23
author: human+ai
status: active
refs: [2026-05-30/progress01]
supersedes: 2026-05-30/progress01 (D6 only)
commits: []
files:
  - CMakeLists.txt, libtorchmedia/CMakeLists.txt
  - libtorchmedia/include/torchmedia/{globel_include.hpp,basic.hpp}
  - unit_test/audio/io/main.cpp
  - script/download_dep, readme.md, CLAUDE.md

## Goal
Remove `fmt` as a dependency; formatting/printing uses the standard library (`std::format` / `std::print`).
The project baseline becomes C++23 so `std::print` is available everywhere.

## Context / Motivation
fmt was barely used: `globel_include.hpp` included `<fmt/core.h>` without using it, `basic.hpp::print_tensor`
printed an already-formatted string, and `unit_test/audio/io/main.cpp` had 8 `fmt::print` calls. No fmt-only
features (`fmt::join`, styles, ostream adapters). One fewer bundled dependency fits the header-only goal.

## Decisions

### D1 — C++23 baseline (supersedes 2026-05-30/progress01 D6)
- Decision: `set(CMAKE_CXX_STANDARD 23)`; `torch_media` INTERFACE target requires `cxx_std_23`.
  The `TORCHMEDIA_CXX_STANDARD` cache variable is removed (C++20 can no longer compile `<print>`).
- Why: Mux chose C++23 so library and tests can use `std::print`. Measured 2026-10-23 on Apple clang 21:
  `std::print` + libtorch 2.5.1 headers compile, link and run under `-std=c++23` (`__cpp_lib_print=202207`);
  under C++20 `std::print` does not exist. D6's concern (libc++ C++23 gaps) did not reproduce for this use.
- Impact: consumers of the header-only library need a C++23 compiler/stdlib with `<print>`
  (expected: GCC 14+, Clang/libc++ 18+, MSVC 17.7+ — not measured here). Removing the cache variable also
  means a stale `TORCHMEDIA_CXX_STANDARD=20` in an old `build/CMakeCache.txt` is ignored, not silently applied.
- Alternatives considered: stay C++20 with `std::cout << std::format(...)` — rejected by Mux.

### D2 — Drop fmt entirely
- Decision: no `#include <fmt/...>`, no `FMT_HEADER_ONLY`, no `dependence/fmt` include path, no fmt entry in
  `script/download_dep`. `print_tensor` uses `std::print`; tests use `std::print`.
- Impact: `dependence/fmt` is no longer referenced (left on disk; delete locally if wanted).

## Tasks
- [x] [task01 — switch to C++23 and remove fmt](tasks/task01-cxx23-drop-fmt.md)

## Issues / Gotchas
- macOS: `std::format` of floating point needs a deployment target >= 13.3 (`to_chars` availability);
  `-mmacosx-version-min=12.0` fails to compile. `CMAKE_OSX_DEPLOYMENT_TARGET` is currently unset (host default).
- libtorch's `c10/util/strong_type.h` trips `-Winvalid-specialization` (an error) unless included as a system
  header; CMake's imported-target `-isystem` handles this — keep it that way in hand-written compile lines.

## Open / TODO (carry-over)
- readme.md example and dependency list are stale beyond fmt (SOX, `torchmedia::basic::to_string`,
  `spectrogram_options_t{}.windows(...)`) — out of scope here.

## Agent log
- 2026-10-23 [ai] Measured std::format/std::print support with libtorch headers; Mux chose C++23.
  Implemented task01 in an isolated worktree (the audio audit workflow was compiling probes against the
  C++20 headers at the time). Fresh configure+build: `-std=c++23`, zero fmt flags, ctest 4/4 green;
  `-DTORCHMEDIA_WITH_PLOT=ON` build ctest 5/5 green. Awaiting Mux review.
