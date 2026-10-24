# Progress 06 — Full ASan + UBSan sweep; speed orig_freq 0 and lfilter 0-D fixes
id: 2026-10-24/progress06
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-24/progress05, 2026-10-23/progress02, 2026-10-23/progress04, 2026-10-23/progress05]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/{_functional.hpp, _functional_filtering.hpp, _transform_time.hpp}
  - unit_test/audio/{functional, transform, noexcept}/main.cpp

## Goal
Every ctest binary runs clean under AddressSanitizer + UndefinedBehaviorSanitizer, and the two pre-existing defects
found while building the silent-mode contract (progress05 Open) are fixed.

## Context / Motivation
The progress notes record sanitizer runs for the audio batches (progress02, 04, 05) and vision (2026-10-23 worklog),
none for AuxIVA (progress03), kaldi (progress06), metric (progress11, 2026-10-24/progress01, 02), data (progress03) or
the PNM reader (progress04). The silent-mode work (progress05) found two crashes that no test reached:
- `functional::speed` / `transform::Speed` with `orig_freq` 0: `std::gcd(0, 0)` is 0 and the next line divides by it
  (UB; torchaudio raises ZeroDivisionError).
- `lfilter` on a 0-D waveform with 1-D coefficients: `shape.back()` of an empty vector (UB; torchaudio raises
  IndexError on `shape[-1]`).
Scope set under Mux's delegation; no CMake option is added (configure flags are enough).

## Decisions

### D1 — speed: gcd floored at 1
- Decision: `g = std::max(std::gcd(source, target), 1)`, so `orig_freq` 0 reaches `resample(0, 0)` / `Resample(0, 0)`,
  which rejects it ("Original frequency and desired frequency should be positive.", std::invalid_argument; silent
  build: undefined tensor + last_error()).
- Why: one expression, no new message; the rate check stays in one place (`_check_resample_args`).

### D2 — lfilter: a 0-D waveform is rejected before the branches
- Decision: `waveform.dim() <= 0` returns `handle_exceptions` with torchaudio's own message for the 2-D-coefficient
  case, "Expected waveform to have a positive number of dimensions. Found: 0", for every coefficient shape.
- Deviation: torchaudio raises that ValueError only for 2-D coefficients with batching; with 1-D coefficients it raises
  IndexError, and with 2-D coefficients and `batching=false` a 0-D waveform is stacked first (an error unless there is
  one filter). All three are now the same std::invalid_argument. The 2-D batching message for a 0-D waveform was
  "Expected number of batches in waveform and coeffs to be the same." before.

### D3 — Sanitizer build recipe (no CMake option)
`cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined
-fno-omit-frame-pointer -Wall -Wextra" "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"`, then
`ASAN_OPTIONS=detect_container_overflow=0 UBSAN_OPTIONS=print_stacktrace=1 ctest --test-dir build-asan`
(container-overflow checks are off because libtorch is not instrumented).

## Results
- Build: exit 0. All 72 warnings are in the vendored `_vision/_vendor/stb_image_write.h` (65 -Wmissing-field-initializers,
  7 deprecated `sprintf`), emitted once per TU that defines TORCHMEDIA_IO_IMPLEMENTATION; none in our headers. Left as
  is (vendored code is not edited for warnings).
- ctest 24/24 under ASan + UBSan, 20,710 checks; 0 AddressSanitizer reports, 0 UBSan `runtime error`. The binaries link
  `libclang_rt.asan_osx_dynamic.dylib`.

## Open
- The stb_image_write warnings reach a consumer's implementation TU under -Wextra (-Werror builds would fail there).
  A `#pragma clang/GCC diagnostic` wrapper around the vendored includes would silence them; not done (no request).
- The known stb_vorbis `lookup1_values` float-to-int cast (progress05 Open) only shows on fuzzed input; the suite does
  not reach it.

## Agent log
- 2026-10-24: fixes + 5 tests (functional: speed orig_freq 0, lfilter 0-D with 1-D and 2-D coefficients; transform:
  Speed orig_freq 0; noexcept: lfilter 0-D, speed orig_freq 0). Sanitizer sweep as above.
