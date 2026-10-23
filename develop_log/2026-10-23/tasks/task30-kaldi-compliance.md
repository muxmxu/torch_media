# Task 30 — Port torchaudio.compliance.kaldi
id: 2026-10-23/task30
parent: 2026-10-23/progress06
status: done
owner: code_agent

## Objective
`torchmedia::audio::compliance::kaldi` reproduces torchaudio.compliance.kaldi 2.5.1 (every public function, every
keyword with its default), bit-exact on CPU, with torchaudio's validation as std::invalid_argument.

## Scope
In:
- `_audio/_compliance_kaldi.hpp` (new, included from `audio.hpp`): fbank, mfcc, spectrogram, get_mel_banks, mel_scale,
  mel_scale_scalar, inverse_mel_scale, inverse_mel_scale_scalar, vtln_warp_freq, vtln_warp_mel_freq; options
  `spectrogram_option` / `fbank_option` / `mfcc_option`; `enum class window_type`; private helpers in `kaldi::detail`.
- `unit_test/audio/compliance/` (new ctest `audio_test_compliance`), kaldi sites in `audio_test_noexcept`.
- CLAUDE.md namespace list.
Out:
- `_functional*.hpp` / `_transform*.hpp` (owned by a later audio follow-up; `functional::create_dct` is consumed).
- A batch dimension; front-end scaling / CMVN / LFR of the downstream toolkits.

## Inputs (read first, priority order)
1. `.venv/lib/python3.12/site-packages/torchaudio/compliance/kaldi.py` — the reference, op order included.
2. `develop_log/2026-06-01/progress02` (option idiom), `develop_log/2026-10-23/progress02` D2-D11 (semantics, errors,
   no-exceptions mode, coverage union).

## Deliverables
- The header per progress06 D1-D5.
- `unit_test/audio/compliance/gen_golden.py` writing `golden.hpp` (cases and C++ calls); `main.cpp` with the golden
  loop, validation, min_duration, detail helpers, option chaining, memory layout and MPS tests.

## Acceptance criteria
- [x] Goldens for defaults, WeNet / WeSpeaker / 3D-Speaker (80 bins, dither 0, energy_floor 0), seeded dither 1.0,
  FunASR (hamming), htk_compat, use_energy / raw_energy, all five windows, snip_edges false, round_to_power_of_two
  false (incl. a frame count that rounds up), subtract_mean, VTLN != 1, mfcc with lifter and num_ceps (incl. 0 and a
  negative slice), spectrogram, 8 kHz and 16 kHz, float64: all 52 cases bit-exact with torchaudio (full outputs via `gen_golden.py --dump` + `TM_GOLDEN_DIR`).
- [x] Tolerances derived from torchaudio's measured self-variation (golden.hpp header).
- [x] Every assert and extra check throws std::invalid_argument (test_validation), each one the check that rejects
  at least one test input (spectrogram rows for the framing checks fbank's mel-bank check would otherwise shadow);
  every error site returns the
  default value without exceptions (audio_test_noexcept).
- [x] 100% line coverage of `_compliance_kaldi.hpp` (audio_test_compliance alone).
- [x] Full build and ctest green; clang-format applied; -Wall -Wextra clean.

## Constraints
- Header-only, `inline`, C++23; no edits to `_functional*.hpp` / `_transform*.hpp`. No commit.

## Notes / Assumptions
- Deviations (progress06 D5): empty output in the input's dtype for all three functions below min_duration; strides
  of the reflected tensor without snip_edges; frame shift clamped at 2n + 1 samples.

## Review round 1
- P1 / T1 / T2 / D1 / S1 fixed as recorded in progress06 Issues and Agent log (mfcc num_ceps checks matching
  torchaudio's failing set, golden `fbank_no_snip_shift9`, three spectrogram validation rows, header comment,
  redundant vtln_warp_mel_freq check removed).
