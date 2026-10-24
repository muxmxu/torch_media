# Task 60 — doc/ pages, readme pointers, CMP0126
id: 2026-10-24/task60
parent: 2026-10-24/progress07
status: done
owner: code_agent

## Objective
Write `doc/porting-from-python.md`, `doc/deviations.md` and `doc/reproducibility.md`; point the readme at them; make a
parent project's plain `set(TORCHMEDIA_DEPENDENCE_DIR ...)` take effect.

## Scope
In:
- The three pages, the readme (intro, Documentation section, vision row, dependence-dir note), CLAUDE.md
  Documentation section, `cmake_policy(SET CMP0126 NEW)`.
Out:
- A markdown API reference; libtorch version enforcement in CMake.

## Acceptance criteria
- [x] Every statement traces to a progress note or the headers; superseded decisions are not described.
- [x] The porting example compiles as an `add_subdirectory` consumer with `-Wall -Wextra` and prints what the doc says.
- [x] A consumer's plain `set(TORCHMEDIA_DEPENDENCE_DIR ...)` selects that libtorch.
- [x] ctest green after the CMake change.
