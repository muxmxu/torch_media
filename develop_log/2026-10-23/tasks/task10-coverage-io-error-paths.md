# Task 10 — coverage: _io.hpp and previously untested error paths
id: 2026-10-23/task10
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
`_io.hpp` and every `_functional*.hpp` / `_transform*.hpp` reach 100% line coverage (union over test binaries).

## Scope
In: `unit_test/audio/io/main.cpp` (rewritten on test_util.hpp), functional test
`test_coverage_gaps_filtering_beamforming_alignment`, `unit_test/test_util.hpp` (`TM_RUN`, CPU-first compare).
Out: I/O behaviour changes (D9).

## Deliverables
- io: const char* / std::filesystem::path overloads, save rejections (1-D, float64, unwritable path), load of a
  missing file and of a non-WAV file.
- functional: _lfilter_core/lfilter validation, riaa 48 k / 88.2 k goldens, flanger triangular golden + bad
  interpolation / > 4 channels, vad empty cepstrum band, rtf_power/rtf_evd validation + n_iter=1 golden,
  forced_align ending on a token + out-of-range target.

## Acceptance criteria
- [x] Before: `_io.hpp` 16 missed lines, `_functional_alignment.hpp` 5, `_functional_beamforming.hpp` 22,
  `_functional_filtering.hpp` 34 (union over functional + transform + io binaries, e7041b5).
- [x] After: 0 missed lines in all twelve `_audio` headers (`_vendor/` excluded).
