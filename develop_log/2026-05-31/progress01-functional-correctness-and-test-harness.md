# Progress 01 — Functional correctness fixes + assertion test harness
id: 2026-05-31/progress01
date: 2026-05-31
author: human+ai
status: done
refs: [2026-05-30/progress01]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_functional.hpp
  - libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp
  - unit_test/test_util.hpp, unit_test/audio/functional/{main.cpp,CMakeLists.txt}
  - CMakeLists.txt (coverage option), unit_test/CMakeLists.txt, CLAUDE.md (testing policy)

## Goal
Stand up an assertion-based test harness with coverage, then fix the 5 confirmed correctness bugs in the
audio functional layer via red-test -> fix -> green-test. Chosen scope: option A.

## Context / Motivation
A 4-way, web-verified audit (see Agent log) confirmed 5 correctness-breaking bugs in `_functional.hpp`,
and that the existing functional test is eyeball-only (`print_tensor` + `torch::save`), not assertion-based,
with dead test functions. A `.venv` (torch/torchaudio 2.5.1) is now available locally as a golden-value source
(htk != slaney mel filterbank confirmed empirically).

## Decisions

### D1 — Five confirmed correctness bugs (each fixed behind a failing test first)
1. `mel_filter_bank`: the `slaney` branch duplicates the HTK formula -> implement the piecewise Slaney scale
   (`f < 1000`: `mel = 3f/200`; `f >= 1000`: `mel = 15 + ln(f/1000)/(ln(6.4)/27)`; inverse symmetric).
2. `amplitude_to_DB`: remove internal `pow(2)`; parameterize `multiplier` (10 power / 20 magnitude);
   `db = multiplier*log10(clamp(x,amin)) - multiplier*db_multiplier` (subtract, not multiply); top_db per-sample.
3. `db_to_amplitude`: `ref * 10^(0.1*x*power)` (was `ref * 10^(x/(20*power))`).
4. `convolve`: record original x length BEFORE the swap so `same`/`valid` use the FIRST input's length.
5. `spectrogram`: pass `normalized=false` into `torch::stft`; normalize once externally (no double-normalize).
- Evidence: torchaudio.functional source semantics; htk!=slaney confirmed via the .venv torchaudio.
- Code-Agent Impact: changes confined to `_functional.hpp` (+ `amplitude_to_db_option` gains a `multiplier`
  field). Public function names unchanged.

### D2 — Test harness: header-only assertion macros + ctest (no third-party test lib)
- `unit_test/test_util.hpp` with `TM_CHECK` / `TM_CHECK_CLOSE` / `TM_CHECK_TENSOR_CLOSE` (over `torch::allclose`);
  each test executable returns `tm_test::summary()`; registered via `add_test`. No gtest/Catch/doctest.
- Why: CLAUDE.md says no heavy deps / prefer header-only; matches the existing `audio_test_io` add_test pattern.

### D3 — Golden reference values: closed-form + libtorch self-reference + torchaudio (.venv)
- CI-portable tests use closed-form truths and libtorch self-reference (bare `torch::stft` etc.). The `.venv`
  torchaudio is the development-time point-wise cross-check (confirm fixes where closed-form is impractical).

### D4 — Coverage: LLVM source-based, target 100% for `_functional*`, exclude `_vendor/`
- CMake `TORCHMEDIA_COVERAGE` option adds `-fprofile-instr-generate -fcoverage-mapping`; measure with
  `xcrun llvm-cov`, `--ignore-filename-regex='_vendor/.*'`; target 100% line coverage for `_audio/_functional*.hpp`.

## Tasks
- [x] [task01 — assertion test harness + coverage + ctest wiring](tasks/task01-assertion-test-harness.md)
- [x] [task02 — red-test then fix the 5 functional bugs](tasks/task02-fix-5-functional-bugs.md)

## Issues / Gotchas
- The existing functional test hard-codes `.to("mps")` and loads a wav fixture — replace with CPU-only synthetic
  tensors so tests are portable and assertion-based.
- spectrogram normalization: verify whether torchaudio divides by `window.pow(2).sum().sqrt()` once for power=2.
- (found via task01 smoke test) **bug #6** — spectrogram `return_complex=false` path: `torch::abs` runs
  elementwise over `torch::stft`'s view_as_real `[.., freq, time, 2]` output, so it does NOT compute the
  complex magnitude and shifts `size(-2)` onto the time axis. Fix folded into task02 #5: call `torch::stft`
  with `return_complex=true` internally, then `abs()` the complex result before `pow(power)`.

## Open / TODO (carry-over)
- Tier-1 expansion: `create_dct`+`mfcc`, `griffinlim`, `resample` — done in 2026-05-31/progress02;
  `inverse_spectrogram` — done in 2026-06-01/progress01.
- `mel_filter_bank` vectorization (replace per-element `.item()` loops) — done in 2026-10-23/progress02 (task07).

## Agent log
- 2026-05-31 [ai] Ran a 4-way audit workflow; confirmed 5 bugs (D1) and designed the test/coverage strategy
  (D2-D4). Set up a local .venv (torch/torchaudio 2.5.1) and verified torchaudio works as a golden source
  (htk != slaney). Maintainer chose scope A. Authored task01 (harness) + task02 (fixes). Implementing task01 next.
- 2026-05-31 [ai] task01 done: header-only TM_* assertion harness (unit_test/test_util.hpp), TORCHMEDIA_COVERAGE
  CMake option, ctest registration. task02 done: 6 bugs fixed (D1's 5 + bug#6 spectrogram abs-of-view_as_real),
  each via red-test -> fix -> green with golden values from the .venv torchaudio; removed dead
  _check_shape_compatible. Result: ctest 100% green; _functional.hpp & _functional_methods_options.hpp at
  100% region/function/line/branch coverage (llvm-cov, _vendor/ excluded). Added gen_golden.py.
- 2026-10-23 [ai] Status corrected to `done` (both tasks were done; flagged stale by the 2026-10-23 audit);
  carry-overs annotated with where they landed.
