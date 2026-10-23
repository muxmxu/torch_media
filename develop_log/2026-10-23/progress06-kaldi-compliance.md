# Progress 06 — Kaldi-compatible features: port torchaudio.compliance.kaldi
id: 2026-10-23/progress06
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-06-01/progress02, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_compliance_kaldi.hpp (NEW)
  - libtorchmedia/include/torchmedia/audio.hpp
  - unit_test/audio/compliance/{CMakeLists.txt, main.cpp, gen_golden.py, golden.hpp (generated)} (NEW)
  - unit_test/audio/CMakeLists.txt, unit_test/audio/noexcept/main.cpp
  - CLAUDE.md

## Goal
Port every public function of torchaudio.compliance.kaldi 2.5.1 (fbank, mfcc, spectrogram, get_mel_banks,
mel_scale(_scalar), inverse_mel_scale(_scalar), vtln_warp_freq, vtln_warp_mel_freq) so that speaker / ASR front ends
(WeNet, WeSpeaker, 3D-Speaker, FunASR) produce torchaudio's features from C++. Out of scope: a batch dimension
(torchaudio has none), the functional / transform layers (owned by a later audio follow-up; consumed, not edited).

## Context / Motivation
Most pretrained speaker and ASR models were trained on `torchaudio.compliance.kaldi.fbank`; a C++ deployment needs
the same features to the bit, which torchmedia::audio::functional (torchaudio.functional semantics) does not give.
The design (header, namespace, options, semantics, tests) was set by the orchestrator under Mux's delegation; D4's
extra checks and D5's deviations are the implementer's choices, awaiting Mux's review.

## Decisions

### D1 — One header, namespace torchmedia::audio::compliance::kaldi
- Decision: `_audio/_compliance_kaldi.hpp`, included from `audio.hpp`. Public: the ten torchaudio functions. Private
  helpers live in `kaldi::detail` under torchaudio's names without the underscore (`get_window`, `get_strided`,
  `get_log_energy`, `subtract_column_mean`, `get_dct_matrix`, `get_lifter_coeffs`, `next_power_of_2`,
  `get_waveform_and_window_properties`, `feature_window_function`) plus `check_mel_banks` / `check_vtln_warp`, which
  return the error string so a composite can validate before computing. `get_dct_matrix` uses
  `functional::create_dct` (already bit-exact with torchaudio).
- Impact: `fbank(waveform, opt = {})`, `mfcc(...)`, `spectrogram(...)`; `get_mel_banks(...)` returns
  `std::tuple<bins, center_freqs>` (float32 CPU, center_freqs [num_bins, 1] as torchaudio actually returns).

### D2 — Options: one keyword set per function, setters by deducing this
- Decision: `spectrogram_option` carries spectrogram's 15 keywords; `detail::mel_option` adds the 8 mel / VTLN ones;
  `fbank_option` adds use_log_fbank / use_power, `mfcc_option` adds cepstral_lifter / num_ceps. Every field is
  `_name` with torchaudio's default and a same-name setter. The setters use C++23 explicit object parameters
  (`this auto &&self` -> `decltype(auto)`), so they return the most derived option and a chain can mix base and
  derived keywords in any order. `window_type` is `enum class window_type {hamming, hanning, povey, rectangular,
  blackman}`. mfcc builds its fbank_option by assigning its `mel_option` base.
- Why: inheritance keeps one definition per keyword (27 setters instead of 67); a base setter returning the base type
  would break `fbank_option().dither(1).num_mel_bins(80)`.
- Impact: a setter called on a temporary returns an rvalue reference (lvalue reference on a named option); copying the
  result (`const auto o = fbank_option().dither(1);`) or passing it to `fbank` is the intended use.

### D3 — Semantics op for op, dtype / device follow the input
- Decision: the C++ mirrors torchaudio's operations and their order: frames by `as_strided` (snip_edges, or the
  [2, 1, 0, 0, 1, 2] reflection), dither `randn(frames.shape)` from the global generator, DC removal, raw energy,
  replicate-pad pre-emphasis, window, zero pad, rfft, float `epsilon` in the input's dtype, mel banks / DCT / lifter
  built in float32 on the CPU and moved to the input's device and dtype. VTLN's masked writes become a `torch::where`
  chain with the same precedence.
- Why: bit-exact parity, including seeded dither (libtorch's CPU generator gives Python's stream).

### D4 — Validation: torchaudio's asserts plus the failures it reaches later
- Decision: every torchaudio assert is a `std::invalid_argument` with the same condition (channel, window size in
  [2, n], shift > 0, even padded size, pre-emphasis in [0, 1], sample_frequency > 0, num_bins > 3, even
  window_length_padded, low / high vs Nyquist, the VTLN ranges, the vtln_warp_freq inflection points,
  num_ceps <= num_mel_bins). Also `std::invalid_argument`, up front: a waveform that is not 2-D floating point
  (torchaudio: IndexError / AssertionError / RuntimeError), a negative energy_floor (ValueError from math.log), a
  zero vtln_warp (ZeroDivisionError), a non-finite frame shift (OverflowError / ValueError from int()), an out-of-range
  window_type value, and the num_ceps values mfcc only fails on after computing: num_ceps < 0 with a non-zero
  cepstral_lifter (RuntimeError from arange in the lifter) and no DCT column left with use_energy or htk_compat
  (IndexError on feature[:, 0]). With `LIB_TORCH_MEDIA_NO_EXCEPTIONS` every site returns `handle_exceptions`' value; fbank checks
  the mel options before computing frames, and mfcc returns fbank's value when it is not 2-D.
- Why: progress02 D7 (composites check before computing; invalid input never reaches the computation).
- Impact: the extra input checks sit with the asserts, so they also fire for input shorter than min_duration
  (torchaudio would return the empty tensor first for a negative energy_floor). The mel-option checks keep
  torchaudio's place (after min_duration); torchaudio builds the banks after the frames, but nothing is drawn in
  between, so moving them ahead of the frames changes only which of two errors is reported.

### D5 — Deviations from torchaudio (implementer's choices)
- Input shorter than min_duration returns `empty({0})` in the input's dtype / device from all three functions:
  torchaudio's fbank does that, its spectrogram returns float32 CPU, and its mfcc raises (matmul of the empty fbank).
- Without snip_edges, the frame strides come from the reflected tensor; torchaudio takes them from the input, which
  reads the wrong samples (or fails) for a non-contiguous waveform. Contiguous input is unaffected.
- A frame shift above 2n + 1 samples is clamped to 2n + 1: same frames (one with snip_edges, none without), and the
  int64 cast stays defined; torchaudio raises TypeError once the shift no longer fits int64.
- A negative num_ceps with cepstral_lifter 0 keeps torchaudio's Python-slice meaning (all DCT columns but the last
  -num_ceps; golden `mfcc_neg_ceps_no_lifter`), and num_ceps 0 without use_energy / htk_compat gives [m, 0]
  (`mfcc_zero_ceps`): torchaudio returns these, so they are not rejected.
- Half / bfloat16 input passes the floating-point check and then fails inside torch (c10::Error), as in torchaudio.

### D6 — Tests: generated goldens, tolerance from torchaudio's own spread
- Decision: `gen_golden.py` holds one spec per case and writes `golden.hpp` with both the C++ call and the torchaudio
  result (shape, dtype, sum, checksum sum(Y * Z) with Z = randn(shape, float64) under seed 99, max |Y|, four
  elements). Inputs are seeded randn (16 kHz [2, 8000] with 1000 samples of digital silence, 8 kHz [1, 4000]); dither
  cases reseed before the call. Tolerance: 2 x torchaudio's own float32 spread (6.4e-6 x max(1, max |Y|): float32 vs
  float64 input; 5.3e-6 for the same values passed non-contiguous); float64 1e-12 x max(1, max |Y|) (torchaudio's
  float64 output did not move, measured 0). `gen_golden.py --dump DIR` plus `TM_GOLDEN_DIR=DIR` compares every full
  output and prints max |delta| per case.
- Why: the separation tests' checksum idiom, without a hand-copied case list that could drift from the reference.

## Tasks
- [x] [task30 — port torchaudio.compliance.kaldi](tasks/task30-kaldi-compliance.md)

## Issues / Gotchas
- C++23 implicit move (P2266): in `auto f(this auto &&self) -> auto & { return self; }` on a temporary, `self` is an
  xvalue in the return statement and does not bind to `auto &`; the setters return `decltype(auto)`.
- macOS make compares mtimes in whole seconds: a header edited within the second of the last build is not rebuilt.
  When running mutants, delete the test's object file before building (a stale binary first "caught" a mutant).
- torchaudio's own output depends on the input's memory layout (non-contiguous input: up to 1.3e-4 abs on the
  spectrogram), because `as_strided` frames over strided storage take other reduction kernels. That spread is part of
  the tolerance basis; `test_layout` asserts the C++ stays within it with snip_edges and is exact without.
- Review round 1 (2 reviewers, 5 findings, all re-checked and fixed; task30 "Review round 1"):
  - P1: mfcc num_ceps -1 (lifter on) and 0 with htk_compat raised c10::Error, not std::invalid_argument (D4). The
    suggested `num_ceps < 0` rejection would have broken parity: torchaudio returns a Python slice for a negative
    num_ceps when cepstral_lifter is 0. num_ceps 0 with use_energy (IndexError in torchaudio) was missing from the
    finding. The checks now match torchaudio's failing set exactly (D4, D5).
  - T1: no golden had n mod shift >= shift / 2 without snip_edges, so a truncating frame count survived; added
    `fbank_no_snip_shift9` (shift 144, 8000 mod 144 = 80).
  - T2: the window-size >= 2, even-padded-size and sample_frequency > 0 checks were each shadowed by check_mel_banks
    in the fbank tests; three spectrogram rows now make each the rejecting check.
  - D1: the header comment now lists every up-front check (frame shift, window_type, num_ceps) and the
    min_duration ordering.
  - S1: vtln_warp_mel_freq's own check_vtln_warp duplicated vtln_warp_freq's and is gone (as in torchaudio); the
    noexcept row for it still returns the default.
- `fbank_linear_magnitude` (use_log_fbank and use_power off) has max |Y| 2.1e6, so its element tolerance is ~27 in
  absolute terms; the log cases are the tight ones.

## Open / TODO (carry-over)
- A batch dimension ([..., channel, time]) is not supported, as in torchaudio; a caller loops over utterances.
- The FunASR / WeNet front ends also scale the waveform (x 32768) and apply CMVN / LFR outside kaldi.fbank; that stays
  with the caller.

## Agent log
- 2026-10-23 [ai] Implemented task30. Parity vs torchaudio 2.5.1 (every full output, `TM_GOLDEN_DIR`): all 49 golden
  cases bit-exact (max |delta| 0) — 35 feature calls (25 fbank incl. WeNet / WeSpeaker-style 80-bin dither 0 and
  energy_floor 0, seeded dither 1.0, FunASR hamming, htk_compat, use_energy / raw_energy, all five windows, snip_edges
  false (pad > 0, = 0, < 0), round_to_power_of_two false, subtract_mean, VTLN 0.9 / 1.15 / 0.85 at 8 kHz, linear
  magnitude, channel 1 without pre-emphasis / DC removal, huge shift with one and with zero frames, 8 kHz, three
  float64 incl. seeded dither; 6 mfcc incl. lifter 0 / 12, num_ceps 20 / 23, htk_compat with and without use_energy,
  8 kHz float64 dithered; 4 spectrogram) and 14 helper outputs (mel banks and center frequencies with and without
  VTLN, vtln_warp_freq / vtln_warp_mel_freq, inverse_mel_scale, DCT matrix, lifter, five windows). MPS: mfcc matches
  the CPU within 1e-4 x max |Y| (asserted). Mutants caught: povey exponent 0.86 (29 cases), htk C0 unscaled
  (mfcc_htk), torch-style reflect padding (no-snip cases), float32 DC mean (float64 cases), dither drawn column-major
  (the 4 dithered cases); a 1e-5 relative change of the sqrt(2) scale stays within tolerance, as it should.
  audio_test_compliance 447/447, audio_test_noexcept 101/101 (8 new kaldi sites). Coverage of
  `_compliance_kaldi.hpp` from audio_test_compliance alone: 453/453 lines, 236/236 regions, 49/49 functions,
  153/154 branches (the remaining one is the window_type switch's fall-through for out-of-range values, rejected up
  front). -Wall -Wextra clean; clang-format applied. Not committed.
- 2026-10-23 [ai] Review round 1 fixes (P1, T1, T2, D1, S1; see Issues). Each fixed finding has a test that failed
  before or catches the mutant: the 4 new mfcc num_ceps rows (compliance) and 2 (noexcept) failed on the old header
  with c10::Error; in a scratch copy, a truncating no-snip frame count fails only `fbank_no_snip_shift9` (shape),
  and deleting the window >= 2, even padded size or sample_frequency > 0 check each fails only its new spectrogram
  row. Final: 52 golden cases (35 + 3 new feature calls, 14 helpers), all bit-exact against torchaudio 2.5.1 with
  `TM_GOLDEN_DIR` (526/526 checks with the dumps); tolerance basis unchanged (5.3e-6 / 6.36e-6 float32, 0 float64).
  audio_test_compliance 474/474, audio_test_noexcept 103/103, ctest 9/9. Coverage of `_compliance_kaldi.hpp` from
  audio_test_compliance alone: 461/461 lines, 248/248 regions, 49/49 functions, 163/164 branches (the window_type
  switch fall-through, as before). clang-format clean. Not committed.
