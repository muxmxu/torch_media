# Progress 04 — Audio surface audit: filtering/effects, spectral/time/augment, beamforming/alignment
id: 2026-10-23/progress04
date: 2026-10-23
author: ai
status: active
refs: [2026-06-01/progress01, 2026-06-01/progress02, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/audio.hpp
  - libtorchmedia/include/torchmedia/_audio/{_functional,_functional_filtering,_functional_alignment,_functional_beamforming}.hpp
  - libtorchmedia/include/torchmedia/_audio/{_transform_time,_transform_feature,_transform_effects,_transform_beamform,_transform_augment}.hpp
  - unit_test/audio/{functional,transform,noexcept}/main.cpp
  - unit_test/audio/{functional,transform}/gen_golden.py

## Goal
Fix the confirmed findings of the 2026-10-23 surface audit of the audio functional/transform layer (everything
progress02 did not cover: IIR filtering and SoX effects, spectral/time/augment ops, beamforming and CTC alignment),
red test first, keeping the progress02 decisions (D2-D11). Low-severity items are fixed when cheap and non-breaking;
the rest are listed under Open.

## Context / Motivation
The audit (branch feat/audio-surface-audit at master 4eb755b) ran three parity probes against torchaudio 2.5.1
(A1 filtering-effects, A2 spectral-time-augment, A3 beamforming-alignment; scratch dirs `surface/probe-A*`), each
finding re-reproduced by an adversarial verifier (`surface/verify-A*`). Method per probe: inputs generated in
Python, raw float32/float64 exchange, one C++ probe executable with one try/catch per case, point-wise comparison
(max abs / rel-to-peak, shapes, dtypes, error type), separate ASan/UBSan builds for the crash cases, best-of-N
timing on 4 threads. 42 findings were confirmed: 1 high, 13 medium, 28 low. The high one: lfilter, and every op
built on it, was 500-1300x slower than torchaudio (a per-sample loop of ~8 tensor ops, ~5 us per sample).

## Confirmed findings and status

| id | severity | kind | function | status | task |
|---|---|---|---|---|---|
| A1-01 | high | performance | lfilter (+ biquad, designers, filtfilt, deemphasis, loudness) | fixed | task12 |
| A1-02 | medium | device-dtype | vad / Vad (float64 threw, MPS corrupted the allocator) | fixed | task13 |
| A1-03 | low | correctness | loudness gate step rounding (11025 Hz) | fixed | task14 |
| A1-04 | medium | validation-crash | vad search_time <= 0 / measure_freq | fixed | task13 |
| A1-05 | medium | performance | overdrive | fixed | task12 |
| A1-06 | low | validation-crash | phaser degenerate buffer lengths | fixed | task14 |
| A1-07 | low | validation-crash | flanger on 1-D input | fixed | task14 |
| A1-08 | low | validation-crash | compute_deltas unknown mode | fixed | task14 |
| A1-09 | low | validation-crash | add_noise shape checks, float arange mask | fixed | task14 |
| A1-10 | low | correctness | biquad designers compute coefficients in double | open (Mux's call) | — |
| A1-11 | low | default-mismatch | Vol power gain 0; Vol/Vad/Loudness silent defaults | partly (Vol power 0 raises) | task14 |
| A1-12 | low | docs | dcshift / dither intentional differences | fixed (documented + pinned) | task14 |
| A1-13 | low | api-design | _functional.hpp did not include the filtering header | fixed | task14 |
| A1-14 | medium | test-gap | filtering/effects tests pinned little | fixed | task12-14 |
| A2-01 | medium | validation-crash | phase_vocoder / TimeStretch rank | fixed | task15 |
| A2-02 | medium | validation-crash | InverseMelScale rank | fixed | task15 |
| A2-03 | medium | validation-crash | pitch_shift / PitchShift / detect_pitch_frequency on 0-D | fixed | task15 |
| A2-04 | low | validation-crash | PitchShift constructor, pitch_shift int cast | fixed | task16 |
| A2-05 | low | device-dtype | mask_along_axis RNG dtype/device | fixed | task17 |
| A2-06 | medium | api-design | pitch_shift n_steps was int | fixed | task16 |
| A2-07 | low | correctness | detect_pitch_frequency 1-ulp Hz, NCCF norm | fixed | task17 |
| A2-08 | low | validation-crash | detect_pitch_frequency arguments | fixed | task17 |
| A2-09 | low | validation-crash | TimeStretch / InverseMelScale no-exceptions returns | fixed | task15 |
| A2-10 | low | api-design | intentional API differences undocumented | fixed (documented) | task17 |
| A2-11 | low | performance | PitchShift rebuilds the kernel for non-float32 | open | — |
| A2-12 | low | test-gap | spectral/time tests pinned weakly | partly | task15-17 |
| A3-1 | medium | validation-crash | forced_align T >= L + R | fixed | task18 |
| A3-2 | low | correctness | forced_align tie-breaking | fixed | task18 |
| A3-3 | low | device-dtype | forced_align accumulation dtype | fixed | task18 |
| A3-4 | medium | validation-crash | forced_align blank range | fixed | task18 |
| A3-5 | medium | validation-crash | forced_align shape/length contract | fixed | task18 |
| A3-6 | medium | validation-crash | psd mask shape | fixed | task19 |
| A3-7 | low | validation-crash | mvdr_weights_rtf validation | fixed | task19 |
| A3-8 | low | validation-crash | apply_beamforming leading dims | fixed | task19 |
| A3-9 | low | validation-crash | merge_tokens shape/length | fixed (breaking, see D5) | task18 |
| A3-10 | low | device-dtype | forced_align paths dtype | fixed | task18 |
| A3-11 | low | performance | forced_align backpointer memory | fixed | task18 |
| A3-12 | low | api-design | one-hot Tensor reference_channel for rtf_power / SoudenMVDR / RTFMVDR | open | — |
| A3-13 | low | docs | frechet_distance stricter check | fixed (documented) | task19 |
| A3-14 | low | validation-crash | beamforming no-exceptions build continued after a failed check | fixed | task19 |
| A3-15 | medium | test-gap | alignment tests pinned two hand cases | fixed | task18 |
| A3-16 | low | test-gap | beamforming goldens: one complex64 case, transforms self-referential | partly | task19 |

Every high and medium finding is fixed. Verifier corrections were taken over: phaser delay_ms=0 is inside torchaudio's
documented range (torchaudio fails there too), torchaudio's PitchShift(16000, -240) fails at the first call, not in its
constructor, and A1-12's "outputs are bit-exact" does not hold for GPDF (see D6).

## Decisions

### D1 — lfilter and overdrive run torchaudio's compiled scalar loops on the CPU (A1-01, A1-05)
- Decision: `_lfilter_core` mirrors torchaudio's compiled `_lfilter` op: the FIR part uses b/a0 in a grouped conv1d,
  then the recurrence `y[i + n - 1] = x[i] - sum_k y[i + k] * a_flipped[k]` runs as an `AT_DISPATCH_FLOATING_TYPES`
  pointer loop with `at::parallel_for` over (batch, channel) rows. This applies to float32/float64 CPU input without
  autograd. Other devices, other dtypes and inputs that require grad keep the differentiable tensor-op loop (torchaudio's
  `_lfilter_core_generic_loop`, which torchaudio also uses on non-CPU, non-CUDA devices). overdrive gets the same split,
  with overdrive.cpp's double-literal arithmetic. The inner step is an explicit `std::fma(-y, c, acc)`.
- Why: the per-sample tensor-op loop cost ~5 us per sample (1 h of 44.1 kHz audio ~13 min). Task01 of 2026-06-01 had
  scoped the compiled path out without measuring it. The fma: torchaudio's arm64 reference wheel compiles `acc -= y * c`
  to a fused multiply-subtract (a numpy model with fma reproduces its output bit-exactly, one without does not), so an
  explicit fma is bit-exact with it and does not depend on this compiler's contraction setting.
- Impact: lfilter on CPU float32/64 without grad is no longer recorded by autograd. That is fine, because grad-requiring
  inputs take the generic loop: the test runs backward through it. float32 results move by ~1e-7 relative, onto
  torchaudio's values.
- Alternatives considered: plain `acc -= y * c`. Not chosen: it was 1.6e-7..8.3e-7 of peak off torchaudio on order-2/3
  filters and 3e-2 on an ill-conditioned order-6 float32 filter, and its result would depend on -ffp-contract.

### D2 — vad keeps torchaudio's float32 measurement state (A1-02, A1-04)
- Decision: windows, spectra and noise mults are float32 on the input's device, as in torchaudio. Each window product
  is computed in the input dtype and rounded to float32 on assignment, as torchaudio's `dftBuf`. The per-measure
  bookkeeping (`measures`, `mean_meas`) is float32 on the host, read through accessors. The mean update uses float32
  arithmetic in two statements (no FMA), and comparisons go against float32(trigger_level), mirroring torchaudio's
  float32 tensor ops. The loop sizes are validated before any int cast: a period of at least 1 sample, at least 1
  measure, a finite measure_duration and positive lifter frequencies. Otherwise std::invalid_argument.
- Why: float64 input threw in `accessor<float>`. MPS input wrote through a CPU accessor into a device buffer: the trim
  was correct, but the process crashed at exit (SIGSEGV in the MPS allocator). search_time 0 wrote into an empty tensor,
  and measure_freq 0 (or > 2 x sample_rate) looped forever.
- Impact: float32, float64 and MPS input give torchaudio's trim (A1 probe: vad 1-D/2-D/3-D, 8/16/44.1 kHz,
  pre-trigger, float64: all bit-exact). The output dtype and device follow the input (it is a slice of the input).

### D3 — crash and UB sites raise std::invalid_argument where torchaudio raises (A1-04, A1-06..A1-09, A2-01..A2-04, A2-08, A2-09, A3-1, A3-4..A3-8, A3-14)
- Decision: add a check, placed before the first read it guards, to phaser (buffer lengths >= 1), flanger (dim >= 2),
  compute_deltas (only replicate/constant/reflect/circular), add_noise (torchaudio's rank and length checks; the mask
  is an int64 arange on lengths' device), phase_vocoder (dim >= 2, after the rate == 1 early return as in torchaudio),
  InverseMelScale (dim >= 2), `_stretch_waveform` / pitch_shift / PitchShift (dim >= 1), detect_pitch_frequency
  (dim >= 1, positive sample_rate, frame_time, freq_low, freq_high, win_length >= 1), Vol (power gain <= 0),
  psd (mask (..., freq, time)), mvdr_weights_rtf (torchaudio's five checks, both overloads) and apply_beamforming
  (leading dims). Transform `operator()` sites now `return handle_exceptions<tensor_t, ...>` instead of `(void)` and
  continuing: TimeStretch, InverseMelScale, MVDR, PitchShift. The TimeStretch message names the bad overriding_rate.
  `detail::assert_psd_matrices` (void) becomes `detail::psd_matrices_error` (returns a message), and its three callers
  return.
- Why: progress02 D3/D7 (invalid input raises std::invalid_argument; the no-exceptions build returns the default value).
  These sites segfaulted, read out of bounds, hung, or silently returned a wrong-rank or wrong-value result.
- Impact: closes the progress02 Open items "`assert_psd_matrices` is void" and "constructors continue" for PitchShift
  (it now remembers a rejected configuration and returns the default value). Other constructors are unchanged.

### D4 — pitch_shift takes fractional steps; PitchShift validates in its constructor (A2-06, A2-04)
- Decision: `n_steps` is `double` in `functional::pitch_shift`, `_stretch_waveform` and `pitch_shift_option`
  (source-compatible for int callers). A shared `_pitch_shift_orig_freq` checks bins_per_octave != 0 and
  orig_freq = sample_rate * 2^(n_steps/bins_per_octave) in [1, INT_MAX] before the int cast, then the resample
  arguments. The PitchShift constructor runs it (stricter than torchaudio, which fails only at the first call).
- Why: torchaudio documents "(fractional) steps". An implicit double-to-int conversion silently turned 0.5 into 0 (no
  shift; 12002 of 12002 samples off). The unchecked cast was UB for extreme steps.

### D5 — forced_align is a port of torchaudio's CPU kernel (A3-1..A3-5, A3-9..A3-11, A3-15)
- Decision: port compute.cpp. The trellis is windowed with start/end, the inner step prefers x2, then x1, and stays
  only when neither is strictly larger, a terminal tie ends on the last token, accumulation is in the log_probs dtype
  (`AT_DISPATCH_FLOATING_TYPES_AND_HALF`), and the backpointers are int8. All checks are taken over: 3-D log_probs and
  2-D targets with batch 1, float16/32/64 log_probs, int32/int64 targets, non-empty targets, blank in [0, C), no blank
  and no out-of-range value in targets, max(lengths) == T / L, and T >= L + repeats. Paths have the targets' dtype and
  device. Relaxed: non-contiguous or non-CPU log_probs are copied (torchaudio's CPU op raises). Stricter: negative
  targets raise (torchaudio reads out of bounds). merge_tokens requires 1-D tokens/scores of equal length.
- Why: the old DP broke ties differently and accumulated float32-cast input in double. It skipped T >= L + R (and
  returned an all-blank path), read out of bounds for a bad blank, flattened 2-row targets into one sequence, ignored
  lengths, and returned int64 paths.
- Impact (breaking): `merge_tokens` on a (1, T) input now raises. Justification: torchaudio's contract; flattening a
  (B, T) input silently merged spans across rows. Pass `paths[0]` / `scores[0]`. Paths of int32 targets are int32.
  Memory on T=10000, L=2000: peak RSS 433.7 MB -> 144 MB (~42 MB for the op, torchaudio ~40 MB).

### D6 — intentional differences are documented where the code is, and pinned by tests (A1-12, A2-10, A3-13)
- Decision: document these in the function or option comments:
  - dcshift and dither never mutate their input.
  - GPDF dither sums the 7 samples of the unmodified input. torchaudio sums into a view of one input sample, so it
    mutates the caller's tensor and re-reads the grown sample when an index repeats. With seed 0 on a 2x6 input, its
    GPDF output differs at 3 elements, without noise shaping too. This corrects A1-12's "outputs are bit-exact".
  - Mono GPDF works.
  - frechet_distance is stricter (torchaudio's chained comparison accepts mu (4,) with sigma (3, 3)).
  - detect_pitch_frequency accepts win_length 1-2 (torchaudio fails on torch.cat([])).
  - TimeStretch hop_length 0 and LFCC f_max 0 mean "derive"; torchaudio's explicit 0 cannot be expressed.
  - The masking parameters have defaults.
  - PitchShift always uses a Hann window.
  - TimeStretch does not warn on non-complex input.
  - float64 works directly in PitchShift/Speed/LFCC/InverseMelScale.
- Why: so that a later "parity fix" does not reintroduce input mutation or the aliasing.

### D7 — mask_along_axis draws its two uniforms like torchaudio (A2-05)
- Decision: `torch::rand({1})` (CPU, float32) whatever the spectrogram's dtype/device; mask_along_axis_iid keeps the
  input's options (as torchaudio does).
- Impact (behaviour): under a given seed, masks for float64, integer and MPS spectrograms change, and now equal
  torchaudio's. Integer spectrograms work (they threw "check_uniform_bounds not implemented for 'Long'").

### D8 — the other low findings: fixed when cheap (D10 of progress02)
- Fixed: loudness rounds half to even (`std::nearbyint`, A1-03). `_functional.hpp` includes `_functional_filtering.hpp`
  (A1-13; the transform test now includes `_transform_effects.hpp` first, which did not compile before).
  detect_pitch_frequency uses `reciprocal(EPS + idx) * sample_rate` (Python `int / tensor`) and
  `linalg_vector_norm` (A2-07).
- Not fixed: see Open.

### Behaviour changes for existing callers (one-line justification each)
- `merge_tokens((1, T) ...)` raises: torchaudio requires 1-D, and flattening a (B, T) input merged spans across rows.
- `compute_deltas` with an unknown or capitalised mode raises: torchaudio raises, and a silent replicate hid typos.
- `add_noise` with mismatched ranks (e.g. 0-D snr) raises: torchaudio raises, and broadcasting changed the output rank.
- `forced_align` returns int32 paths for int32 targets: torchaudio does (D-rule "dtype follows the input").
- Seeded masks of float64 / integer / MPS spectrograms change: they now equal torchaudio's (D7).
- Invalid `PitchShift` options raise in the constructor: the old path was UB, then a c10::Error at the call.
- `Vol` with gain_type power and gain 0 raises instead of returning silence: torchaudio raises (math.log10(0)).
- lfilter/overdrive float32 values move by ~1e-7 relative: they now equal torchaudio's compiled output (D1).
- add_noise float32 values move by up to 1 ulp: the energies are `linalg_vector_norm(...) ** 2`, as in torchaudio.

## Tasks
- [x] [task12 — lfilter / overdrive compiled loops](tasks/task12-lfilter-overdrive-compiled-loops.md)
- [x] [task13 — vad dtype/device and validation](tasks/task13-vad-dtype-device-validation.md)
- [x] [task14 — effects validation, loudness rounding, dither/dcshift docs, header include](tasks/task14-effects-validation-docs.md)
- [x] [task15 — rank checks and no-exceptions returns in spectral/time ops](tasks/task15-spectral-time-rank-checks.md)
- [x] [task16 — fractional pitch_shift steps, PitchShift validation](tasks/task16-pitch-shift-fractional-steps.md)
- [x] [task17 — mask RNG, detect_pitch parity/validation, API-difference docs](tasks/task17-mask-rng-detect-pitch-docs.md)
- [x] [task18 — forced_align kernel port, merge_tokens checks](tasks/task18-forced-align-kernel-port.md)
- [x] [task19 — beamforming shape checks and complex128 goldens](tasks/task19-beamforming-checks-goldens.md)

## Measured parity (after the fixes, re-running the audit probes on this tree)
- A1 probe (176 rows): bit-exact vs torchaudio rose from 89 to 137 rows.
  - Every lfilter (12) and filtfilt (4) case, overdrive (5), vad (8, float64 included), dither TPDF/RPDF and GPDF
    without index repeats, phaser and flanger are bit-exact. loudness at 11025 Hz: 1.8e-7 of peak (was 0.0977 LU).
  - Left: float32 biquad designers 1e-7..2e-5 of peak, and bass at fc 20 Hz 8.6e-4 (A1-10, coefficient precision; float64
    designers bit-exact). loudness at 22.05/44.1/48 kHz 3.6e-4..1.6e-3 LU, through the same designers. add_noise was
    1e-15 (float64) / 7e-7 (float32 batched), from a sum-of-squares energy; bit-exact since the review (Agent log).
- A2 probe (132 rows): 109 exact, 20 raise in both libraries (all crash cases now raise std::invalid_argument).
  - phase_vocoder, TimeStretch, detect_pitch_frequency (float32/64, 3-D, noise), mask_along_axis (float64 included)
    and SpecAugment(iid_masks=False, float64) are exact.
  - Left, both intentional: ts_hop0 (hop 0 derives, D6) and T_ps_f64_double (2.5e-7, the progress02 window shortcut).
    lfcc_fmax0 (torchaudio all-NaN, D6).
- A3 probe: forced_align paths 575/575 equal to torchaudio, scores exact (was 122 path mismatches). Every
  beamforming output (87 rows) is bit-exact. Every torchaudio raise maps to std::invalid_argument, except
  FA_noncontig (relaxed, D5) and FD_mismatch (stricter, D6).
- Timing (best of N, 4 threads, niced host shared with another build; C++ before -> after, torchaudio):

  | case | before | after | torchaudio |
  |---|---|---|---|
  | lfilter, 2 s mono | 161 ms | 0.2 ms | 0.19-0.23 ms |
  | lowpass_biquad, 2 s stereo | 176-208 ms | 0.3 ms | 0.27-0.41 ms |
  | filtfilt, 2 s stereo | 346-413 ms | 0.6 ms | 0.64-0.69 ms |
  | loudness, 2 s stereo 16 kHz | 351-438 ms | 0.7 ms | 0.75-0.81 ms |
  | lfilter, 10 s mono | 806 ms | 1.1 ms | 0.80 ms |
  | overdrive, 2 s stereo | 163-189 ms | 0.3 ms | 0.56-0.70 ms |
  | forced_align, T=3000 L=400 | 2.96 ms | 2.79 ms | 9.67 ms |

  A 16-row batch takes ~20 ms in both libraries: libtorch 2.5.1's conv1d takes 18 ms at batch 16 (the same in the
  PyTorch wheel).

## Issues / Gotchas
- The arm64 torchaudio wheel's compiled lfilter loop is FMA-contracted. A clang build of the plain `acc -= y * c` here
  was not contracted, which left ~1e-7 differences that look like a conv1d difference. They are not: the Python conv1d
  plus a non-fused scalar loop equals the old C++ output exactly.
- torchaudio's GPDF `gaussian = waveform[c][t]` is a view, and `gaussian += ...` writes into the input. The test golden
  replays torchaudio's draws on an unaliased copy (gen_golden.py `_gpdf_unaliased`).
- detect_pitch_frequency with win_length 1 returns 101 frames on 16001 samples, where win 30 returns 86: the median
  window trims (win_length - 1) / 2 + 1 frames, as in torchaudio.
- MPS float32 matmul in the generic lfilter loop differs from the CPU scalar loop by up to 7.4e-4 on a double pole at
  0.99 (tested at 5e-3 abs).
- phaser and flanger still run per-sample tensor-op loops (114 ms / 683 ms on 4 s of 16 kHz audio) and fail on MPS
  (float64 wave table). Neither was a confirmed finding (Open).
- Red runs: segfaulting tests are placed last, so the counts before them are kept. Hanging cases ran under a perl
  `alarm`.

## Open / TODO (carry-over)
- A1-10: biquad designers compute coefficients in double, while torchaudio computes them as tensors in the waveform's
  dtype. It is the only remaining non-bit-exact source in the lfilter-based ops (float32: up to 8.6e-4 of peak at
  fc 20 Hz, where C++ is closer to the float64 truth). Mux to decide; progress02 treated create_dct's 1.9e-5 as a
  parity bug.
- A1-11 (rest): Vol (gain 1.0), Vad and Loudness (sample_rate 16000) have defaults where torchaudio requires the
  argument. Removing them breaks every default-constructed caller.
- A2-11: PitchShift rebuilds its kernel per call for non-float32 input (the progress02 `shortcut:`). At 16 kHz / +1 step
  that kernel is 1.1 GB in float32.
- A2-12 (rest): loose pins remain: phase_vocoder 3x4 at 1e-4, InverseMelScale at 1e-1, LFCC at 1e-2, detect_pitch
  median within 1 Hz, SpeedPerturbation only self-reproducible, inverse_spectrogram options not pinned to torchaudio.
- A3-12: no one-hot Tensor reference_channel overload for rtf_power, SoudenMVDR or RTFMVDR (torchaudio: Union[int, Tensor]).
- A3-16 (rest): the transform MVDR/SoudenMVDR/RTFMVDR tests still compare against this library's functional ops.
- phaser / flanger performance and MPS support (not audited findings).

## Agent log
- 2026-10-23 [ai] Implemented task12-task19 against the 42 confirmed findings: every high and medium fixed; of the 28 low,
  22 fixed or documented, 3 partly fixed, 3 left open with reasons.
  - Red first: the new tests failed on 4eb755b. functional: 39 failure records, then SIGSEGV in the vad search_time=0
    test (the crash-class tests after it never ran). transform: 204/212. noexcept: SIGSEGV at its first new check. A TU including only
    `_transform_effects.hpp` did not compile (2 errors).
  - Green: Release build, ctest 7/7 (functional 446/446, transform 218/218, noexcept 93/93, io 16/16, separation
    265/265). The three changed binaries are clean under ASan+UBSan (MPS checks on). -Wall -Wextra clean on
    `<torchmedia/audio.hpp>` in both exception modes.
  - Union line coverage (D11 rule, five audio binaries, build in scratch `surface/build-cov`): every `_audio/*.hpp`
    at 100% (13 headers, `_separation.hpp` included).
  - Parity and timing: as recorded above. git clang-format applied to changed lines. Not committed; awaiting review.
- 2026-10-23 [ai, reviewer] Independent review of the diff (scratch `surface/review`).
  - Own probe on inputs the tests do not use, against torchaudio 2.5.1: lfilter with a0 = 3 / 0.7 / 1.3 and an order-6
    a0 = 1.7 filter (float32/64), non-contiguous input, batching=false, filtfilt, deemphasis, loudness at 11025/8000 Hz,
    overdrive 3-D and float64, vad (22.05 kHz float32/64/MPS, 8 kHz 3-D, pre-trigger 3 s), compute_deltas
    constant/reflect/circular, masking (float64 p=0.5, int32, 4-D, MPS), SpecAugment(iid=False, p=0.7), phase_vocoder
    complex128 4-D, fractional / negative-bins pitch_shift, detect_pitch float64, and 240 random forced_align cases
    (float16/32/64, int32/int64, any blank, ties and repeats). All bit-exact, except add_noise (fixed below). Same run
    clean under ASan+UBSan. Timing equals torchaudio (2 x 10 s at 44.1 kHz: lowpass 4.3 ms vs 4.4 ms).
  - Mutation check on a copy: 7 of 15 mutants of the new code passed the suite. Four are now killed (task12, task14,
    task17 review notes): no fma, conv1d(b) / a0, sqrt(sum) in the NCCF, and a float16 lengths mask. Three vad mutants
    still pass (float64 mean update, double trigger compare, float64 first FFT). The probe cannot tell them from
    torchaudio either, so they are left unpinned. The other 8 (overdrive float literal, loudness llround, mask RNG
    options, Hz division, int n_steps, forced_align inner / terminal tie rule and path dtype) fail the suite.
  - Fixed: add_noise energies via `linalg_vector_norm(...).pow(2)`. Red: the 4 new add_noise goldens failed (1 ulp, and
    3.4e-5 on the float16 one).
  - Union line coverage re-measured from a fresh coverage build (lcov export of the five audio binaries, comment and
    blank lines excluded): 0 missed lines in every `_audio/*.hpp`. Every audio header compiles on its own.
- 2026-10-23 [ai] Open items A1-10, A1-11 (kept, D7), A2-11, A2-12, A3-12, A3-16, the vad notes and the phaser / flanger
  loops: handled in [progress08](progress08-audio-parity-followups.md).
