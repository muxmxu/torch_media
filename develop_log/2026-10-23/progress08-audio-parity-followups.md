# Progress 08 — Audio parity follow-ups: biquad coefficients, phaser/flanger loops, PitchShift cache, goldens, readme
id: 2026-10-23/progress08
date: 2026-10-23
author: ai
status: active
refs: [2026-10-23/progress02, 2026-10-23/progress04]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_functional_filtering.hpp
  - libtorchmedia/include/torchmedia/_audio/_functional_beamforming.hpp
  - libtorchmedia/include/torchmedia/_audio/_transform_time.hpp
  - libtorchmedia/include/torchmedia/_audio/_transform_beamform.hpp
  - unit_test/audio/{functional,transform,noexcept}/main.cpp
  - unit_test/audio/{functional,transform}/gen_golden.py
  - readme.md, LICENSE (renamed from LISCENSE), script/download_dep
  - libtorchmedia/CMakeLists.txt (review D9)

## Goal
Close the progress04 open items A1-10, A2-11, A2-12, A3-12, A3-16, the vad precision notes and the phaser/flanger loop
note; record A1-11 as kept; rewrite the readme for the current library. Red test first for every behaviour change.

## Context / Motivation
Work list set by the follow-up workflow orchestrator, stated as acting under Mux's delegation, on branch
feat/audio-parity-followups at 080ae69. Its rule: parity with torchaudio 2.5.1 wins over extra accuracy, because users
need C++ inference preprocessing identical to training-time torchaudio. D1, D2, D4-D8 follow that list. D3 goes beyond
it (see there) and needs Mux's confirmation.

## Decisions

### D1 — the biquad designers compute their coefficients as torchaudio does (A1-10)
- Decision: the nine frequency / Q / gain designers (allpass, lowpass, highpass, bandpass, bandreject, equalizer,
  band, bass, treble) build b0..a2 as 0-D tensors in the waveform's dtype and device, op for op as torchaudio
  (`detail::as_coeff` is `torch.as_tensor`), through a new `biquad(waveform, Tensor b0, ..., Tensor a2)` overload.
  deemph_biquad and riaa_biquad are designed in Python double math in torchaudio: they stay in double, rewritten with
  one product per statement so that clang's default `-ffp-contract=on` does not fuse `a * b + c`.
- Why: parity rule. float32 outputs were up to 6.8e-3 of peak off (band, noise mode, 30 Hz) and 9.5e-4..1.8e-3 at a
  20 Hz lowpass. float64 riaa was 1.1e-14..2.7e-14 off: clang contracted its double math into FMAs.
- Impact: float32 designer and loudness outputs move onto torchaudio's values. Cost: lowpass_biquad on 2 s stereo
  0.335 ms (a biquad with double coefficients 0.326 ms; torchaudio 0.346 ms).

### D2 — phaser and flanger run torchaudio's per-sample loop as a CPU scalar loop (progress04 loop note)
- Decision: torchaudio 2.5.1 has no compiled phaser or flanger op (its library has only `_lfilter`,
  `_lfilter_core_loop` and `_overdrive_core_loop`); both are Python loops over tensors. Here float32/float64 input
  without autograd runs that loop as a scalar loop per row (phaser: per channel; flanger: per (batch, channel)) under
  `AT_DISPATCH_FLOATING_TYPES` + `at::parallel_for`. One rounding per torchaudio tensor op (separate statements, so
  contraction fuses nothing), Python-style remainders for the delay and LFO indices, and the float32 channel-phase
  arithmetic of torchaudio's `(channel_idxs * lfo_length * channel_phase + 0.5).to(int64)`. Autograd and other dtypes
  keep the tensor-op loop. The wave table is computed on the CPU in float64 and moved to the device (MPS has no
  float64). flanger raises std::invalid_argument for an LFO or delay buffer below 1 sample or non-finite lengths
  (torchaudio: ZeroDivisionError / negative dimension; here it was `% 0` or an undefined int cast).
- Why: 4 s of 16 kHz audio took 202 ms (phaser) and 1119 ms (flanger), and both failed on MPS.
- Impact: same values (bit-exact before and after, see Measured parity); the scalar loops are pinned by goldens that
  kill an fma, a C `%`, a float64 decay product and a float64 channel phase.

### D3 — float32/float64 input on another device runs the scalar loops on a host copy (implementer's call; Mux to confirm)
- Decision: lfilter (`_lfilter_core`), overdrive, phaser and flanger copy float32/float64 input without autograd from
  a non-CPU device to the CPU, run the scalar loop there and return the result on the input's device
  (`detail::loop_on_host`). Autograd and other dtypes keep the on-device tensor-op loop, which now synchronises MPS
  every 8192 steps (`detail::mps_flush`, a `shortcut:`).
- Why: the work list kept the tensor-op loop "for autograd / other devices". On MPS it is not usable:
  - libtorch 2.5.1's MPS backend returns garbage once ~64k index_put_ writes are queued without a sync. Minimal
    repro: 70000 per-column `index_put_` into one MPS tensor, max error 6.9e13 (40000 writes: exact; a sync every 8192:
    exact). The PyTorch 2.5.1 wheel gives nan, and torchaudio's own lowpass_biquad on 64000 MPS samples is 8.3e21 off.
    Measured thresholds: overdrive between 56000 and 64000 samples, flanger between 24000 and 32000 (two writes per
    sample), phaser between 33000 and 64000. A garbage phaser run also corrupted later MPS calls in the same process.
  - lfilter's conv1d on MPS raises past 65536 samples ("Output channels > 65536 not supported at the MPS device").
  - With the sync the loops are correct but slow: phaser 70000 samples 75 s, flanger 40000 samples 21 s, overdrive
    70000 samples 13 s, lowpass 65000 samples 7 s. The host loop takes < 1 ms and gives torchaudio's CPU values.
- Impact (behaviour change): MPS/CUDA float32/float64 lfilter, biquad, the designers, filtfilt, deemphasis,
  overdrive, phaser and flanger return torchaudio's CPU values on the input's device (lfilter on MPS was up
  to 7.4e-4 off on a double pole, MPS float32 matmul; the p4 MPS checks are now exact). The designers compute their
  coefficients on the host too (`detail::as_coeff`, review D9; before that MPS sin / cos / exp put a 15 Hz highpass
  1.55e-3 of peak off). loudness gets torchaudio's CPU K-weighting filters, but its gating ops (square, mean, log10)
  still run on the device.
- Alternatives considered: the sync alone (keeps the compute on the device, but 0.1-1 ms per sample and lfilter still
  raises past 65536 samples).

### D4 — transform::PitchShift caches one resample kernel per (dtype, device) (A2-11)
- Decision: a `shared_ptr` cache with a mutex (the const call stays thread-safe; copies share the cache), seeded
  with the eager float32 CPU kernel. Any other (dtype, device) is built on first use in the input's dtype and device,
  as torchaudio's lazy kernel init does; a float32 input on another device now gets a kernel built there instead of a
  per-call copy of the CPU one. Removes the progress02 D12 `shortcut:`. The STFT window follows the input's dtype
  since review D9 (torchaudio's module after `.double()`).
- Impact: float64, 8 kHz / 0.5 step (a 4000 x 4131 kernel): 62-77 ms on every call before; now 90.8 ms on the first
  call and 3.7-4.0 ms after.

### D5 — torchaudio goldens replace the loose and self-referential pins (A2-12, A3-16, vad notes)
- Decision: phase_vocoder (two rates, was 3x4 at 1e-4), inverse_spectrogram options (window / frame_length
  normalization, center false, length + pad, two-sided), detect_pitch_frequency (exact median and sum, was within
  1 Hz), InverseMelScale (was 1e-1), LFCC (dB and log_lf, was 1e-2), SpeedPerturbation (torchaudio's seeded factor
  draw, was only self-reproducible), SoudenMVDR / RTFMVDR / MVDR offline (three solutions), default mask_n, cfloat,
  online (two solutions, two calls) and online multi-mask (all compared with this library's functional ops before),
  and the three vad float32 details. Values are bit-exact (point checks at 0, float64 sums at 1e-9).
- vad: the details (float32 running mean, float32 trigger comparison, float32 first FFT) move the running mean by an
  ulp, which only decides at a trigger level on its peak. gen_golden.py replays the mean with torchaudio's `_measure`.
  T_a lies just above the float32 peak (same float32, larger double) and T_b is the next float32. At T_a torchaudio
  keeps 17600 samples, and a float64 mean update or double comparison never triggers. At T_b torchaudio keeps none,
  and a float64 first FFT triggers (input variant k = 9).
- Why: a parity regression could pass every replaced check.
- Impact: every new test fails under a deliberate mutation (21 mutants in scratch copies, all killed; see Measured
  parity). The replaced loose checks are deleted.

### D6 — one-hot Tensor reference_channel for rtf_power, SoudenMVDR and RTFMVDR (A3-12)
- Decision: `functional::rtf_power(psd_s, psd_n, Tensor reference_channel, ...)` (shared `detail::rtf_power` with a
  column selector; torchaudio's einsum with the one-hot cast to psd_n's dtype). SoudenMVDR and RTFMVDR take a
  template `Ref` (int64_t or a one-hot Tensor) in operator() and forward and forward it to the functional overloads.
  MVDR's stv_power solution now passes torchaudio's one-hot u to rtf_power (it used the int path; same values here).

### D7 — A1-11 stays: Vol, Vad and Loudness keep their defaults
- Decision (orchestrator, under Mux's delegation): Vol (gain 1.0), Vad and Loudness (sample_rate 16000) keep defaults
  where torchaudio requires the argument. Default-constructible transforms need them, and removing them breaks every
  default-constructed caller. Vol with power gain 0 already raises (progress04).

### D8 — readme, LICENSE, download_dep
- Decision: readme.md rewritten for the current library (scope table, requirements, build, CMake use,
  TORCHMEDIA_IO_IMPLEMENTATION, an audio and a vision example, behaviour, license). Every snippet is a scratch file
  that was compiled and run against this tree, and the readme embeds it verbatim. The build snippet ran in a
  fresh-checkout copy (download_dep, configure, build, ctest 10/10). LISCENSE renamed to LICENSE (git mv).
- download_dep fix: it extracted libtorch to dependence/libtorch/libtorch/ (a single top-level folder moved into the
  freshly created destination), where CMake expects dependence/libtorch/share/cmake/Torch. `_move_extracted` now
  unwraps that folder. Checked with a dummy archive: nested before, flat after.

### D9 — review fixes (7 findings R1-R7; implementer's calls under the work list's parity rule, Mux to confirm)
- R1: `detail::as_coeff` builds the coefficient tensors on the CPU when `loop_on_host(waveform)`, so the designers on
  MPS give torchaudio's CPU values (D3's claim). Red: 15 Hz highpass / 37.5 Hz allpass / 60 Hz noise band at 22050 Hz
  on 2 x 70000 MPS samples were not equal to the CPU output.
- R2 (docs): the 8192-step `mps_flush` fixes the forward pass only. Per the reviewer's probe, a backward pass through
  the on-device loops on MPS exhausts GPU memory beyond about 20k samples (NaN / garbage output or an abort; torchaudio
  raises). Documented at `mps_flush`, in the readme and below; no guard (the limit depends on the GPU's memory).
- R3: `torch_media` adds `-ffp-contract=on` to its users (GCC, Clang, AppleClang). GCC's GNU-mode default, `fast`,
  fuses across statements and breaks the scalar loops' parity; the readme's `add_subdirectory` recipe compiles in
  GNU mode. `on` (within one expression) is the configuration every golden was generated and checked with (Apple
  clang's default); a GCC that treats `on` as `off` still gives the scalar loops' values (one op per statement), only
  overdrive's one-expression update would then be unfused. Red: functional built with
  `-DCMAKE_CXX_FLAGS=-ffp-contract=fast` failed 14 checks (phaser / flanger / riaa / deemph goldens, both new vad
  checks); green with the interface flag (it follows CMAKE_CXX_FLAGS on the command line).
- R4 / R5: new goldens kill the five surviving rounding-order mutants: float32 flanger at regen 33 and -60 (the
  feedback product rounds), vad inputs k = 2 / 3 whose running-mean peak a fused update moves one ulp down / up,
  complex128 inverse_spectrogram with "window" normalization and a 256-sample Hamming window (its pow(2).sum().sqrt()
  and norm() differ), LFCC on 1-D input at 8 kHz / n_fft 256 / hop 100 / log_lf (the batched goldens take another
  matmul kernel), and float32 PitchShift at 8 kHz / 1.5 steps / n_fft 512 (phase_vocoder's phase wrap).
- R6: transform::PitchShift casts its float32 Hann window to the input's dtype. torchaudio's module only takes
  float64 after `.double()`, which casts the window buffer the same way; float64 output moves by 5e-8..8e-8 onto
  that module's (it equalled F.pitch_shift before, which keeps a float32 window). This reverses the progress02
  choice ("T::PitchShift float64 keeps torchaudio F's float32 window") under the parity rule: the transform matches
  the transform. Red: the three task11 `.double()` goldens at tolerance 0 failed.
- R7: the readme's vision snippet said load_image returns [3, H, W]; it returns the file's channel count.

### Behaviour changes for existing callers (one-line justification each)
- float32 biquad designer and loudness values move onto torchaudio's (by up to 7e-3 of peak at a 30 Hz band): parity.
- float64 riaa / deemph values move by ~1e-14 relative: torchaudio's double math has no FMA.
- MPS/CUDA float32/float64 input without grad to lfilter-based ops, overdrive, phaser and flanger is computed on a
  host copy: MPS returned garbage or raised for long input (D3).
- flanger with an LFO or delay buffer below 1 sample raises std::invalid_argument: torchaudio raises; here it was UB.
- SoudenMVDR / RTFMVDR operator() and forward are templates over the reference type: int arguments still compile;
  a braced reference argument no longer deduces.
- Copies of a PitchShift share its kernel cache: the kernels are a function of the configuration.
- float64 transform::PitchShift moves by up to ~8e-8 (window cast to float64): torchaudio's `.double()` module (D9).
- MPS float32/float64 biquad designers move onto torchaudio's CPU values (coefficients on the host): D3 parity (D9).
- Users of the `torch_media` CMake target compile with `-ffp-contract=on` on GCC / clang: scalar-loop parity (D9).

## Tasks
- [x] [task50 — biquad designer coefficients in the waveform's dtype](tasks/task50-biquad-coefficients-in-dtype.md)
- [x] [task51 — phaser / flanger scalar loops, device inputs on host loops](tasks/task51-phaser-flanger-scalar-loops.md)
- [x] [task52 — PitchShift kernel cache per dtype / device](tasks/task52-pitch-shift-kernel-cache.md)
- [x] [task53 — torchaudio goldens for the loose and self-referential tests](tasks/task53-torchaudio-goldens.md)
- [x] [task54 — one-hot Tensor reference_channel](tasks/task54-onehot-reference-channel.md)
- [x] [task55 — readme, LICENSE, download_dep](tasks/task55-readme-license.md)
- [x] [task56 — review fixes R1-R7](tasks/task56-review-fixes.md)

## Measured parity
- Biquad probe (94 rows: the 11 designers at 16 / 44.1 kHz (deemph and riaa at their rates), 20-55 Hz corner cases,
  float32 and float64, and loudness at 11.025 / 16 / 22.05 / 44.1 / 48 kHz): 62 -> 94 bit-exact. Before, float32
  relative to peak (16 / 44.1 kHz): band noise 30 Hz 1.2e-3 / 6.8e-3, bass 20 Hz 5.1e-4 / 3.6e-3, bass 100 Hz
  5.8e-5 / 6.6e-4, lowpass 20 Hz 9.5e-4 / 1.8e-3, treble 20 Hz 5.6e-4 / 5.5e-4, highpass 20 Hz 1.4e-4 / exact,
  bandpass skirt 55 Hz 4.1e-5 / exact, the 1-3 kHz designers 2.5e-7..3.6e-6;
  loudness float32 9.8e-8..1.4e-4 relative; float64 riaa 1.1e-14..2.7e-14, band noise 3.6e-13 / 6.4e-13.
- Phaser / flanger probe (11 rows: float32 and float64, 2-D and 3-D, 4 channels, triangle, quadratic, regen 50 / -95
  / 90): 11 / 11 bit-exact before and after. Best of 3 on the niced shared host, C++ before -> after (torchaudio):
  phaser 4 s @ 16 kHz 201.9 -> 0.37 ms (371 ms), flanger 4 s 1118.7 -> 0.40 ms (1760 ms), flanger 2x3x8000
  151.7 -> 0.32 ms (294 ms). MPS: every float32 row raised (float64 wave table) -> 8 / 8 bit-exact.
- Mutation check (scratch copies of the headers, the real test sources): 21 / 21 mutants fail the suite.
  phase_vocoder round -> floor, time steps in float64; inverse_spectrogram window norm divided, pad trim,
  center ignored; detect_pitch in float64; vad float64 mean, double trigger, float64 first FFT; rtf_power one-hot with
  conj(phi); flanger fma, C `%`, float64 channel phase; phaser float64 decay product; InverseMelScale lstsq in float64;
  LFCC log offset; SpeedPerturbation factor shift; MVDR online blend swapped, no cdouble promotion, perturbed first
  online call; PitchShift without cache. pv_steps_f64, ims_lstsq_f64 and lfcc_log_eps fail only the new tests.
- Review round (D9): the reviewer's survivors flanger_fb_fma, vad_mean_fma, istft_window_norm, lfcc_matmul_order and
  pv_round_mul, plus reverts of the R1 and R6 fixes: 7 / 7 now fail the suite (scratch afollow/fix). R3: functional
  under `-ffp-contract=fast` 601 / 615 before the interface flag, 615 / 615 after.

## Issues / Gotchas
- llvm-cov marks every line of an inline function that a binary does not run as executable with 0 hits, comment and
  blank lines included, while binaries that run it skip those lines. The union (D11 of progress02) excludes comment
  and blank lines, as progress04 did.
- The vad and float32 channel-phase goldens are ulp-level, like the other bit-exact goldens: generated on arm64 with
  this libtorch build.
- Contraction: the scalar loops are written one rounding per torchaudio op and need `-ffp-contract=on` or `off`;
  the `torch_media` target pins `on` (D9). A consumer that does not use the target must not compile with `fast` (GCC's
  GNU-mode default). Not checked with a real GCC (the macOS libtorch ABI blocks it); clang `-ffp-contract=fast` was
  the stand-in. overdrive's one-expression update is fused under `on` like the arm64 wheel; where `on` means `off`
  it would not be.

## Open / TODO (carry-over)
- LICENSE is empty (0 bytes; LISCENSE was empty too): add the MIT text with the copyright holder (Mux).
- D3 goes beyond the work list's wording: Mux to confirm.
- Autograd and float16 / bfloat16 input on MPS still run the tensor-op loops: the forward pass is correct with the
  8192-step sync, but 0.1-1 ms per sample, and lfilter's conv1d still raises past 65536 MPS samples there. A backward
  pass through them exhausts GPU memory beyond about 20k samples (reviewer's probe: overdrive 20000 / 30000 samples
  give NaN gradients with exit code 0, 70000 aborts; lowpass 30000 output 0.369 off). Not guarded (D9 R2).
- D9 (R6) reverses a progress02 choice and R3 adds a compile flag to every user of the target: Mux to confirm.
- progress02 Open items other than the PitchShift kernel and the stale readme are unchanged.

## Agent log
- 2026-10-23 [ai] Implemented task50-task55 red-test first.
  - Red: functional 18 failures for the designers (task50); MPS phaser / flanger raised, both timing checks and the
    degenerate flanger checks failed (task51); lfilter / overdrive MPS not exact and the 70000-sample MPS lowpass
    raised (task51, D3); PitchShift second call not 4x faster (task52); the one-hot overloads did not compile (task54).
  - Green: Release build, ctest 10 / 10 (functional 596 / 596, transform 348 / 348, noexcept 108 / 108, io 1118 /
    1118, io_optout 26 / 26, separation 265 / 265, compliance 474 / 474, the three vision binaries). -Wall -Wextra clean
    on the three changed test TUs (both exception modes); each changed header compiles on its own.
  - Union line coverage (seven audio binaries, scratch afollow/build-cov, comment and blank lines excluded): 4272
    lines in 14 `_audio/*.hpp`, 0 missed.
  - git clang-format applied to changed lines. Not committed; awaiting review.
- 2026-10-23 [ai] Review fixes R1-R7 (task56), red first.
  - Red: functional 612 / 615 (R1 MPS designers), functional under `-ffp-contract=fast` 601 / 615 (R3), transform
    356 / 363 (R6 `.double()` goldens at tolerance 0); the R4 / R5 goldens pass on the fixed code and fail under the
    five mutants they target.
  - Green: Release build, ctest 10 / 10 (functional 615 / 615, transform 363 / 363, noexcept 108 / 108, io 1118 / 1118,
    io_optout 26 / 26, separation 265 / 265, compliance 474 / 474, the three vision binaries). -Wall -Wextra clean on
    the functional, transform and noexcept TUs.
  - Union line coverage (seven audio binaries, scratch afollow/build-cov, comment and blank lines excluded): 4273
    lines in 14 `_audio/*.hpp`, 0 missed.
  - readme snippets rebuilt and rerun in afollow/readme-check against a fresh copy of this tree (the consumer build
    gets `-std=gnu++23 -ffp-contract=on`); readme.md regenerated from them. git clang-format clean. Not committed.
