# Progress 02 — Audio audit follow-up (verified defects)
id: 2026-10-23/progress02
date: 2026-10-23
author: ai
status: active
refs: [2026-05-30/progress01, 2026-05-31/progress01, 2026-06-01/progress01, 2026-06-01/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/globel_include.hpp
  - libtorchmedia/include/torchmedia/_audio/_functional.hpp
  - libtorchmedia/include/torchmedia/_audio/_functional_methods_options.hpp
  - libtorchmedia/include/torchmedia/_audio/{_functional_filtering,_functional_beamforming,_functional_alignment}.hpp
  - libtorchmedia/include/torchmedia/_audio/{_transform,_transform_spectral,_transform_feature,_transform_time,_transform_effects}.hpp
  - libtorchmedia/include/torchmedia/_audio/_io.hpp
  - libtorchmedia/include/torchmedia/_audio/audio_util.hpp (deleted)
  - unit_test/test_util.hpp, unit_test/audio/{functional,transform,io}/main.cpp, unit_test/audio/*/gen_golden.py
  - unit_test/audio/noexcept/ (new), unit_test/audio/CMakeLists.txt

## Goal
Fix the audio defects confirmed by the 2026-10-23 audit (4 parity probes vs torchaudio 2.5.1, each finding
independently re-reproduced by an adversarial verifier, plus a completeness critic). Each fix lands red-test first.

## Context / Motivation
The audit ran against 4f7a460, i.e. BEFORE the 2026-06-01 functional/transform commits (they had not been pulled
on this machine). Findings in code those commits rewrote (e.g. mel_scale rank collapse, missing inverse_spectrogram,
no transform classes) are stale and must be re-verified against the current tree before fixing. `_io.hpp` was not
touched by those commits, so its findings stand as measured.

Re-verification on e7041b5 (four probes, one per audit group): G1-8 (fftconvolve) and G3-4 (mel_scale rank) were
fixed upstream; G1-9, G3-8, G4-12 and G4-18 changed shape; every other finding still held, mostly also in the new
transform layer. It also produced new findings: fftconvolve has no operand check, AmplitudeToDB accepts a negative
top_db, `<torchmedia.hpp>` does not compile as the first include, transform classes cache their buffers on the
CPU in float32, inverse_spectrogram reads out of bounds on 1-D input, MelSpectrogram/MFCC build and discard
default sub-transforms, LFCC and MelScale use `sample_rate / 2.0` instead of `sample_rate // 2`, and
speed/pitch_shift inherit the resample defects. While fixing, one more parity bug turned up: pitch_shift rounded
the stretch length half away from zero where Python's `round()` goes to even.

Decisions D2-D10 were set by the audit-follow-up workflow orchestrator (stated as acting under Mux's delegation);
D11 and the per-item choices under D10 are the implementer's; the orchestrator accepted all of them on merge
(Mux delegated these calls). Mux reviews the pushed commit.

## Decisions

### D1 — load_audio must always copy out of the decoder buffer (audit G4-1, high)
- Decision: `transpose(0, 1).clone(torch::MemoryFormat::Contiguous)` instead of `.contiguous()`.
- Why: for 1 channel ([frames, 1]) or 1 frame ([1, channels]) the transpose is already contiguous, so
  `.contiguous()` returned a view of the dr_wav buffer that is freed on the next line. Verified by ASan
  (heap-use-after-free at `drwav_free`) and in Release by corrupted first frames of small mono files
  (allocator free-list metadata). Mono is the normal ASR input. The 2026-05-30/progress01 gotcha predicted this.
- Impact: one extra copy is already what the multi-channel path paid; behaviour otherwise unchanged.

### D2 — amplitude_to_DB mirrors torchaudio, including its defaults (G1-1, G1-2, G1-6, G1-7, G3-1)
- Decision: the top_db cutoff packs the input to (-1, C, F, T) with C = shape[-3] for > 2-D input (else 1) and
  takes amax over (-3, -2, -1); 1-D input with top_db throws. `amplitude_to_db_option` defaults mirror
  `AmplitudeToDB()`: multiplier 10, amin 1e-10, db_multiplier 0 (ref 1.0), top_db disabled; `set_top_db()`
  enables it (torchaudio: passing top_db). Option fields and `db_to_amplitude` parameters are `double`. amin is
  used as given (no FLT_MIN floor). Every in-repo caller that relies on top_db enables it explicitly
  (functional mfcc, transform MFCC/LFCC through AmplitudeToDB(top_db=80)). transform::AmplitudeToDB rejects a
  negative top_db (torchaudio ValueError).
- Why: per-[F, T] cutoffs gave every channel its own floor (up to 188 dB error on stereo MFCC); the old default
  subtracted 10 dB everywhere; float parameters cost ~1e-7 relative on float64 input.
- Impact: callers that relied on the old functional default (db_multiplier 1, top_db 80 always on) get the
  torchaudio default; none existed in the tree. Complex input is still converted with abs() (see Open).

### D3 — spectrogram mirrors torchaudio's normalization, packing and default derivation (G2-1, G2-3, G2-5, G2-8, G3-5, G3-6)
- Decision: "frame_length" is `torch::stft(normalized=true)` (amplitude x n_fft^-0.5); "window" divides by
  sqrt(sum(win^2)) after stft; any other method throws std::invalid_argument, also when `normalized` is false.
  Input is packed to (-1, L) and the leading dims restored. `win_length`/`hop_length` 0 derive `n_fft` /
  `win_length / 2` through one shared helper (`functional::detail::derive_*`; the transform layer re-exports it);
  only 0 derives, a negative value is passed on and torch raises, as in torchaudio. The functional option defaults
  (spectrogram, mel_spectrogram, griffinlim) are now 0, so the default n_fft=400 still gives 400/200.
  `spectrogram_option::_return_complex` keeps its functional default (true).
- Why: frame_length was off by sqrt(n_fft)/win (20x at n_fft 400); N-D input threw; setting only n_fft threw
  or silently kept win 400/hop 200; an unknown method was a silent no-op.
- Impact: every spectral wrapper (MelSpectrogram, MFCC, LFCC, SpectralCentroid, both layers) accepts N-D input.

### D4 — griffinlim mirrors torchaudio op-for-op (G2-2, G2-6, G2-7, G2-9, G2-10)
- Decision: `length` is passed to every istft (not only the last); momentum outside [0, 1) and rank < 2 throw;
  the default window has the spectrogram's dtype and device; rand_init draws `torch::rand(..., complex dtype)`
  exactly like torchaudio (same CPU generator stream after `manual_seed`); win/hop derive per D3.
- Why: length != inferred diverged by 100% of peak; the random init had the wrong distribution; momentum and
  rank were unchecked (rank < 2 was an out-of-bounds read). The default window takes the dtype of the magnitude
  after `pow(1 / power)`, so an integer spectrogram still works.
- Impact: functional and transform GriffinLim are bit-exact with torchaudio for float32, rand_init both ways.

### D5 — mel filterbank, MelSpectrogram, MFCC (G3-2, G3-3, G3-7, G3-9, G3-10, G3-11, G3-12)
- Decision: `mel_filter_bank` is vectorized and mirrors `melscale_fbanks` + `_create_triangular_filterbank`
  op-for-op (integer `sample_rate // 2` for bin frequencies and the default f_max), keeping the public
  [n_mels, n_freqs] layout; invalid `norm`/`mel_scale` (and `create_dct` norm) throw. `create_dct` is built in
  float32 in torchaudio's op order and returned as the transposed view. `melspectrogram`/`mfcc`
  move fb and DCT to the spectrogram's device and dtype, reject f_min > effective f_max, and always use a
  one-sided STFT (torchaudio deprecated `onesided`). `mfcc_option.sample_rate` is the rate and overrides
  `mel.sample_rate` (torchaudio MFCC semantics, as transform::MFCC already did); n_mfcc > n_mels throws.
- Why: per-element `.item()` loops took 18-94 ms per bank (300-900x torchaudio); odd rates differed by up to
  5e-4; float64 / MPS input threw; mfcc ignored its own sample_rate.
- Impact: fb, create_dct and MelSpectrogram/MFCC/LFCC (functional and transform) are bit-exact with torchaudio on
  CPU for <= 3-D input; see Issues for the >= 4-D and float64 remainders.

### D6 — resample mirrors torchaudio and validates its arguments (G4-2..G4-6, G4-11, G4-12, G4-14)
- Decision: orig/new > 0 is checked first, then orig == new returns the input, then lowpass_filter_width > 0 and
  rolloff > 0 are checked (torchaudio's order: equal rates build no kernel, so their filter arguments are never
  checked), plus floating-point input and dim >= 1 (std::invalid_argument; rolloff > 0 is stricter than
  torchaudio, which only fails at 0). The functional kernel is computed in the waveform's dtype on its device, in
  torchaudio's multiply order; target length is the float32 ceil torchaudio computes. transform::Resample caches
  the kernel like torchaudio's `dtype=None` (float64 compute, float32 cache) with an optional `dtype`, validates in
  its constructor, and moves the kernel to each input's device/dtype on use. transform::PitchShift caches a
  float32 kernel and, like torchaudio's lazy init, builds it in the input's dtype for any other input.
- Why: same-rate calls attenuated content (1.3e-2), zero/negative rates were UB, non-float32 input threw,
  long inputs gained a sample, and the cached kernel was 1e-5 off torchaudio T.Resample.
- Impact: F::resample, F::speed, F::pitch_shift, T::Resample/Speed are bit-exact with torchaudio in every probed
  case (float32 and float64); T::PitchShift is bit-exact for float32 and, for float64, bit-exact with F::pitch_shift
  and 4.6e-8 from `T.PitchShift(...).double()` (its float32 window buffer cast up, the same gap torchaudio has
  between its own F and T).

### D7 — convolve: scoped mode enum with a torchaudio default, one shared operand check (G1-4, G1-5, G1-9)
- Decision: `enum class convolve_mode`, `convolve`/`fftconvolve` default to `full`; both share
  `_align_convolve_operands` (0-D and non-broadcastable leading dims throw std::invalid_argument; the existing,
  documented extension of left-padding a lower-rank operand is kept). In a LIB_TORCH_MEDIA_NO_EXCEPTIONS build
  every tensor-returning functional error site now `return`s `handle_exceptions`' value, and `handle_exceptions`
  has an explicit return type and `[[maybe_unused]]` parameters (no -Wunused-parameter in either mode).
  Composites check their arguments before calling their parts (pitch_shift: bins_per_octave != 0 and the derived
  rates; melspectrogram/mfcc: norm, mel_scale, DCT norm, f_min), and transform::Resample returns the default
  value when its constructor rejected the arguments. `audio_test_noexcept` checks every tensor-returning site.
- Why: unscoped enumerators collided with `torch::full`; fftconvolve had no checks; the no-exceptions path
  computed garbage from invalid input.
- Impact: callers spell `convolve_mode::full` etc. (tests and transform::convolve_option updated).

### D8 — header hygiene (new findings + audit hygiene items)
- Decision: rename `_LIB_TORCH_MEDIA_AUDIO_TRANSFOMR_HPP` to `LIB_TORCH_MEDIA_AUDIO_TRANSFORM_HPP`; drop the
  internal `torch/csrc/autograd/generated/variable_factories.h` include and the partial torch headers
  (`<torch/fft.h>`, `<torch/linalg.h>`, ...) in favour of `<torch/torch.h>`, so `<torchmedia.hpp>` compiles as
  the first include; delete `audio_util.hpp` (G4-17: unreachable, unused, nonsensical defaults).
- Impact: the io test now includes `<torchmedia.hpp>` before any torch header to keep this true.

### D9 — I/O scope
- Decision: PCM16 rounding bias (G4-7), 8-bit mapping (G4-9), float/other-format save (G4-8, G4-10), save
  validation (G4-13) and more formats (G4-16) are deferred to a later I/O progress. Here only tests are added so
  `_io.hpp` reaches 100% line coverage (overloads, save rejections, missing/garbage-file loads).

### D10 — low-severity and new findings
- Decision: fix when cheap and non-breaking, otherwise list under Open/TODO. Fixed on that basis: transform
  buffers (window, fb, DCT, filter, kernel, phase advance) follow the input's device/dtype on each call (the C++
  stand-in for torchaudio's `module.to()`); inverse_spectrogram rank check and normalized-string validation
  ("true"/"false" stay accepted as torchaudio's bool True/False); LFCC/MelScale `sample_rate // 2`;
  MelSpectrogram/MFCC member-initializer lists; pitch_shift round-half-even; spectral_centroid bin frequencies
  are a float32 linspace like torchaudio (float64 input was 2e-5 Hz off).

### D11 — coverage is measured as the union over the test binaries
- Decision: headers are shared by `audio_test_{functional,transform,io,noexcept}`, so a header line counts as
  covered when any binary runs it; profiles are merged per binary (one profraw each) and combined line by line.
  The no-exceptions-only branch (transform::Resample after a rejected constructor) is covered by noexcept.
- Why: merging all raw profiles into one profdata reports mismatched data for inline functions.

### D12 — review round 2 (two reviewers, 12 findings; implementer's choices, awaiting Mux's review)
- Decision: fix all 12, each with a red test (task11). The choices that go beyond the round-1 decisions:
  - T::PitchShift (P1/R1): a non-float32 input rebuilds the kernel in its dtype per call instead of adding a
    per-dtype cache (a `shortcut:` comment marks it); the float32 kernel stays eager (D8 of 2026-06-01/progress02).
  - Resample equal rates (P3): torchaudio's order rather than the stricter round-1 behaviour (rejecting a bad
    filter width for a call that never filters bought nothing and diverged from torchaudio).
  - Negative win/hop (R6): passed on to torch (c10::Error, as base e7041b5 and torchaudio raise) rather than a new
    std::invalid_argument; the shared helper then needs no error channel, which an `int` return cannot carry in a
    LIB_TORCH_MEDIA_NO_EXCEPTIONS build.
  - inverse_spectrogram "false" (R7): accepted as "none", the counterpart of the already accepted "true".
- Why: each was a parity or validation gap the reviewers reproduced; see task11 for the evidence.

## Tasks
- [x] [task02 — fix the mono / single-frame load_audio use-after-free](tasks/task02-fix-mono-load-uaf.md)
- [x] [task04 — amplitude_to_DB parity](tasks/task04-amplitude-to-db-parity.md)
- [x] [task05 — spectrogram parity](tasks/task05-spectrogram-parity.md)
- [x] [task06 — griffinlim parity](tasks/task06-griffinlim-parity.md)
- [x] [task07 — mel filterbank / MelSpectrogram / MFCC parity](tasks/task07-mel-mfcc-parity.md)
- [x] [task08 — resample / speed / pitch_shift parity](tasks/task08-resample-parity.md)
- [x] [task09 — convolve checks, no-exceptions path, header hygiene](tasks/task09-convolve-noexcept-hygiene.md)
- [x] [task10 — coverage: _io.hpp and untested error paths](tasks/task10-coverage-io-error-paths.md)
- [x] [task11 — review round 2 follow-ups](tasks/task11-review-round2-followups.md)

## Issues / Gotchas
- A 1 s mono round trip did NOT catch the bug: a 64 KB freed block is left intact. The regression test uses a
  4000-frame mono file with a non-zero first sample plus a [2, 1] single-frame file, which fail without the fix
  (max err 0.5) and pass with it; the same test is clean under ASan.
- torchaudio's resample kernel with `dtype=None` is NOT computed purely in float64: its phase `arange` is int64,
  so `/ new_freq` rounds to float32 before meeting the float64 index. `_sinc_resample_kernel` reproduces this
  (`TensorOptions` without a dtype selects that path) — computing the phase in float64 is 1 ulp off.
- `TensorOptions::has_dtype()` distinguishes "no dtype" from float32; `tensor_options_t()` without `.dtype()`
  is the transform's torchaudio-`dtype=None` path, `waveform.options()` the functional one.
- MPS lacks `aten::unfold_backward` (istft) and `aten::angle` (phase_vocoder), so GriffinLim, InverseSpectrogram
  and PitchShift cannot run on MPS in libtorch 2.5.1 regardless of buffer placement; the MPS checks in the tests
  cover Spectrogram, MFCC and Resample and are skipped when MPS is unavailable.
- MPS tensors cannot be cast to float64 directly; `tm_test::tensor_close` now moves to CPU first.
- `TM_RUN` (test_util.hpp) records an unexpected exception as a failure, so one throwing red test does not hide
  the rest of the binary's red evidence.
- Float64 MelSpectrogram/MFCC/GriffinLim use a natively float64 Hann window (torchaudio's `.double()` casts its
  float32 buffer: 3e-6 abs on mel_f64); with a float64 window both are bit-exact. T::PitchShift float64 keeps
  torchaudio F's float32 window, hence its 4.6e-8 gap to `T.PitchShift.double()`.
- MFCC on >= 4-D input differs by ~2.5e-7 relative although the op is torchaudio's (`matmul(x.mT, dct).mT`):
  libtorch C++ and the PyTorch wheel take different batched-matmul kernels there; a per-batch matmul in Python
  reproduces the C++ output exactly. <= 3-D input is bit-exact.
- Measuring T::PitchShift at 16 kHz with n_steps 2: orig_freq 17959 is coprime with 16000, so the kernel is
  [16000, 1, 17973] (1.15 GB in float32, as in torchaudio). Tests use 11025 Hz / 3 steps (gcd 7) instead.

## Open / TODO (carry-over)
- G1-3: amplitude_to_DB still accepts complex input via abs(); torchaudio raises. Throwing would break callers
  that pass a magnitude-domain complex spectrogram with multiplier 20 (correct today) — needs Mux's call.
- G1-9 (rest): `db_to_amplitude_option` is an empty, unused public struct; removing it is a (tiny) API break.
- G3-8: functional/transform mel_scale accept a 1-D spectrogram as one frame (torchaudio raises IndexError).
- G3-11 (rest): no "empty mel filter" warning (torchaudio warns via `warnings.warn`).
- G3-13: functional option structs have no fluent setters/window field and share names with the transform
  structs while using different defaults (`spectrogram_option::_return_complex`).
- G4-15: no `sinc_interp_kaiser`/beta, integer rates only. (The PitchShift float64 kernel note from round 1 was
  wrong by three orders of magnitude, 2.8e-5 to 4.9e-4; fixed in D12/task11.)
- T::PitchShift rebuilds the kernel per call for non-float32 input (D12 shortcut); add a per-dtype cache if that
  path gets hot. PitchShift/TimeStretch still derive win/hop with their own `> 0` checks (negatives derive).
- I/O (D9): G4-7 PCM16 bias, G4-8 save formats, G4-9 8-bit mapping, G4-10 (C++ is the more faithful decoder),
  G4-13 save validation, G4-16 FLAC/MP3/OGG + frame_offset/num_frames, and quantization tests (G4-18 rest).
- LIB_TORCH_MEDIA_NO_EXCEPTIONS: `detail::assert_psd_matrices` is void, so its callers still continue after a
  failed check; transform constructors (except Resample) continue after a failed validation; composites other
  than pitch_shift/melspectrogram/mfcc (e.g. speed, spectral_centroid, the transform wrappers) still pass a
  callee's default value on instead of validating up front.
- readme.md stays stale (vision "coming soon", "CPU, CUDA, MPS" claims) — out of scope.

## Agent log
- 2026-10-23 [ai] Audit workflow (9 agents) done; G4-1 confirmed by probe + verifier. Fixed red->green, ASan clean,
  ctest green.
- 2026-10-23 [ai] Re-verified all 52 findings on e7041b5 (four probes); recorded D2-D11. Implemented task04-task10
  red-test first: the new tests failed on e7041b5 (functional 33 failures/aborts, transform 14, noexcept 5 +
  abort, io compile error) and pass after the fixes. Point-wise probe vs torchaudio 2.5.1 on audit-sized inputs:
  amplitude_to_DB (3/4/5-D), spectrogram (N-D, frame_length, window), melspectrogram, mel filterbanks,
  griffinlim (length, rand_init), resample/speed/pitch_shift (float32 + float64) and T::Resample are bit-exact;
  MFCC/LFCC within 3.4e-7 relative. Timing (4 threads): mel_filter_bank(201x128) 17.8 -> 0.048 ms,
  MelSpectrogram() ctor 36.8 -> 0.053 ms, MFCC() ctor 74.5 -> 0.066 ms, fn::mfcc on 1 s 18.7 -> 0.175 ms.
  Release build green; ctest 6/6 (functional 305/305, transform 191/191, io 16/16, noexcept 8/8). Coverage
  (union, D11): every `_functional*.hpp`, `_transform*.hpp` and `_io.hpp` at 100% lines (alignment, beamforming
  and filtering error paths and `_io.hpp` were below 100% before). git clang-format applied. Not committed —
  awaiting review.
- 2026-10-23 [ai] Review round 2 (12 findings: P1-P5 parity reviewer, R1-R7 robustness reviewer), task11. Every
  finding reproduced, then fixed red-test first: before the fixes functional failed 16 checks/aborts
  (create_dct 1.9e-5, centroid 6e-6 Hz, equal-rate lfw 0 threw, normalize_method/negative hop unchecked,
  bins_per_octave 0 c10 "step must be nonzero", integer griffinlim c10, "false" threw), transform 5 (PitchShift
  float64 6.4e-5 off, equal-rate Resample threw), noexcept 9 (pitch_shift zero-padded result, composites ending in
  c10::Error, Resample(0) undefined kernel). Mutants M15 (phase arange in float64) and M51 ("return" dropped at the
  griffinlim momentum site) now fail transform and noexcept respectively. Parity probe (the reviewers' probe, 109 result
  rows, rebuilt on this tree): 92 bit-exact; non-exact = float64 window (mel_f64, mfcc_f64, T_ps_f64 4.6e-8), >= 4-D MFCC
  batched matmul (mfcc_4d, T_mfcc_sr, 2.5e-7 rel), and 7 MPS rows; gl_3d_short/T_gl raise the same error in
  torchaudio. Release build green; ctest 6/6 (functional 328/328, transform 203/203, io 16/16, noexcept 59/59);
  -Wall -Wextra clean on the audio headers in both modes. Coverage (union of the four audio binaries, D11): every
  `_functional*.hpp`, `_transform*.hpp` and `_io.hpp` at 100% lines. git clang-format applied. Not committed.
- 2026-10-23 [ai] Orchestrator: accepted D2-D11, rebased onto ab03326 (AuxIVA) without conflicts; full build,
  ctest 7/7 (incl. audio_test_separation and the new audio_test_noexcept). CLAUDE.md coverage policy now covers every
  `_audio/*.hpp` (union, D11).
