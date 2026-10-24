# Task 40 — Silent-mode contract and last_error()
id: 2026-10-24/task40
parent: 2026-10-24/progress05
status: done
owner: code_agent

## Objective
A `LIB_TORCH_MEDIA_NO_EXCEPTIONS` call that fails returns one documented fallback per return type and records its
message for `torchmedia::last_error()`; the default build is unchanged.

## Scope
In:
- `global_include.hpp`: `last_error()` / `clear_last_error()` (thread_local in the silent build, `""` / no-op in the
  throwing build), `handle_exceptions` records the message, the policy comment (progress05 D1-D3, D6).
- Every `handle_exceptions(torch::empty({1}), msg)` -> `handle_exceptions(msg)` (82 values, incl. forced_align's pair).
Out:
- Catching c10::Error; changing any exception type or message of the default build.

## Inputs (read first, priority order)
1. `develop_log/2026-10-24/progress05-silent-mode-contract.md` — D1-D3, inventory.
2. `develop_log/2026-10-23/progress02-audio-audit-followup.md` — D7 / D10 (the `return handle_exceptions` rule).

## Deliverables
- `libtorchmedia/include/torchmedia/global_include.hpp` — the API above; the explicit-fallback overload takes `msg`
  without a default (progress05 Gotchas).

## Acceptance criteria
- [x] `grep 'torch::empty({1})'` finds no fallback in the library headers.
- [x] `error_policy_test_silent` checks `last_error()` semantics (set, kept on success, thread-local, cleared).
- [x] Default build: ctest green, `error_policy_test_throw` sees the same exceptions with empty `last_error()`.

## Constraints
- Header-only, function-local thread_local; no new dependency.
