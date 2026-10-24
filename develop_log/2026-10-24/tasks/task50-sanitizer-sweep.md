# Task 50 — Full sanitizer sweep; speed / lfilter crash fixes
id: 2026-10-24/task50
parent: 2026-10-24/progress06
status: done
owner: code_agent

## Objective
Run every ctest binary under ASan + UBSan (+ -Wall -Wextra) and fix the two pre-existing crashes from progress05 Open.

## Scope
In:
- `speed` / `Speed` with orig_freq 0, `lfilter` with a 0-D waveform; their tests in both builds.
- One sanitizer build of all test targets, results recorded in progress06.
Out:
- A CMake sanitizer option; warnings inside vendored code.

## Acceptance criteria
- [x] The new tests fail against the unfixed headers and pass after the fix.
- [x] ctest 24/24 under ASan + UBSan with no sanitizer report; no warning from a non-vendored header.
- [x] Release ctest green; the changed lines stay at 100% coverage (the new branches are the tested error paths).
