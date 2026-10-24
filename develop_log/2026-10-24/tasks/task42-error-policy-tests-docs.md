# Task 42 — error_policy tests in both builds; docs
id: 2026-10-24/task42
parent: 2026-10-24/progress05
status: done
owner: code_agent

## Objective
The same failing calls are checked in both builds, the existing no-exceptions binaries enforce the new fallback, and
the policy is written down in the header, CLAUDE.md and the readme.

## Scope
In:
- `unit_test/error_policy/{main.cpp, CMakeLists.txt}` -> `error_policy_test_throw` / `error_policy_test_silent`.
- `unit_test/audio/noexcept/main.cpp` (`is_default`: undefined + recorded message), `unit_test/data/test_noexcept.cpp`
  (messages per call).
- `global_include.hpp` comment, CLAUDE.md "Error policy" convention, readme.md's LIB_TORCH_MEDIA_NO_EXCEPTIONS line.

## Acceptance criteria
- [x] One representative failing call per public area (audio functional / transform / io / separation / kaldi, vision
  functional / transform / io, each metric area, data): throwing build throws the expected type and message; silent
  build returns the fallback and `last_error()` names it.
- [x] Mutants: a `torch::empty({1})` fallback at one site, or no `last_error` update, fail a test (progress05 Agent log).
- [x] Full build + ctest green; touched headers at 100% lines (union over the test binaries, progress02 D11).
