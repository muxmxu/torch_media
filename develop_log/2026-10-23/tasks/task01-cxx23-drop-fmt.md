# Task 01 — Switch to C++23 and remove fmt
id: 2026-10-23/task01
parent: 2026-10-23/progress01
status: done
owner: code_agent

## Objective
Build as C++23 and remove every use of / reference to fmt.

## Scope
In: CMake standard + target features, fmt include path / define, headers and tests using fmt, download
script, readme/CLAUDE.md mentions, D6 supersede note.
Out: other readme staleness; deleting `dependence/fmt` from disk.

## Inputs (read first, priority order)
1. `develop_log/2026-10-23/progress01-cxx23-drop-fmt.md` — D1, D2.

## Deliverables
- `CMakeLists.txt` — `set(CMAKE_CXX_STANDARD 23)`, no `TORCHMEDIA_CXX_STANDARD`.
- `libtorchmedia/CMakeLists.txt` — include dir only `include/`; no `FMT_HEADER_ONLY`; `cxx_std_23`.
- `globel_include.hpp` — no fmt include. `basic.hpp` — `<print>`, `print_tensor` via `std::print`.
- `unit_test/audio/io/main.cpp` — `std::print`.
- `script/download_dep` — no fmt entry. `readme.md`, `CLAUDE.md` — no fmt; C++23 stated.

## Acceptance criteria
- [x] `git grep -i fmt` (excluding develop_log, `_vendor/`) finds nothing in code/build files.
- [x] Fresh configure + build green; compile flags contain `-std=c++23` and no fmt path/define.
- [x] `ctest` 4/4 green; with `-DTORCHMEDIA_WITH_PLOT=ON` 5/5 green.

## Result
Done 2026-10-23. See progress01 Agent log.
