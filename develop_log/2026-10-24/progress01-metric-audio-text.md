# Progress 01 — Metric module, phase 2: audio and text metrics (torchmetrics / pystoi / mir_eval parity)
id: 2026-10-24/progress01
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress11, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_metric/_audio.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_text.hpp (NEW)
  - libtorchmedia/include/torchmedia/_metric/_common.hpp (metric_base builds option-less classes)
  - libtorchmedia/include/torchmedia/metric.hpp
  - unit_test/metric/{CMakeLists.txt, gen_golden.py, golden_{audio,text}.hpp (NEW), test_{audio,text}.cpp (NEW)}
  - unit_test/metric/sweep/{sweep_audio_gen.py (NEW), sweep.cpp, cov_union.sh}
  - CLAUDE.md

## Goal
Phase 2 of `torchmedia::metric` (progress11): `metric::audio` mirroring torchmetrics.audio (SNR, SI-SDR, SI-SNR,
C-SI-SNR, SDR, SA-SDR, PIT, STOI = pystoi), mir_eval's `bss_eval_sources`, speaker-verification EER / minDCF, and
`metric::text` mirroring torchmetrics.text (WER, CER, MER, WIL, WIP, EditDistance). Out of scope: Open.

## Context / Motivation
Mux asked for a measurement module; the phase-2 design (metric list, references, parity bars, tests) was set by the
orchestrator under Mux's delegation. Layout and conventions are progress11's (D1, D7, D8): a functional plus a
torchmetrics-named class with update / compute / reset / forward, `xxx_option` structs, goldens from `gen_golden.py`,
`std::invalid_argument` through `handle_exceptions`, deliberate differences listed (D7 below).

## Decisions

### D1 — Layout and API shape
- Decision: `_metric/_audio.hpp` (`torchmedia::metric::audio`) and `_metric/_text.hpp` (`torchmedia::metric::text`),
  aggregated by `metric.hpp`. Functionals keep torchmetrics' names and positional arguments (`signal_noise_ratio(preds,
  target, zero_mean)`, `short_time_objective_intelligibility(preds, target, fs, extended)`); metrics with several
  keyword arguments take an option struct shared with the class (`signal_distortion_ratio_option`,
  `source_aggregated_signal_distortion_ratio_option`, `permutation_invariant_training_option`,
  `short_time_objective_intelligibility_option`, `min_dcf_option`, `edit_distance_option`; `zero_mean_option` is
  shared by the SNR / SI-SDR / C-SI-SNR classes). `metric_base` now default-constructs classes without options
  (ScaleInvariantSignalNoiseRatio, EqualErrorRate, the WER family).
- Classes: SignalNoiseRatio, ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio,
  ComplexScaleInvariantSignalNoiseRatio, SignalDistortionRatio, SourceAggregatedSignalDistortionRatio,
  PermutationInvariantTraining, ShortTimeObjectiveIntelligibility, EqualErrorRate, MinDCF; WordErrorRate, CharErrorRate,
  MatchErrorRate, WordInfoLost, WordInfoPreserved (aliases of one template over a non-template count struct),
  EditDistance.
- Text inputs are `std::vector<std::string>` (UTF-8); a single string is `{s}` in the functionals and `update()`
  (torchmetrics wraps a str the same way). No `std::string` overloads: a braced `{"..."}` argument would be ambiguous
  between the two. `forward()` / `operator()` are `metric_base`'s variadic templates, so they need a named
  `std::vector<std::string>` (a braced list does not deduce).
- PIT: `metric_func` is any `std::function<tensor_t(const tensor_t &, const tensor_t &)>` (bind other arguments in a
  lambda); `pit_mode {speaker_wise, permutation_wise}`, `eval_func_mode {max, min}`; returns `{best_metric,
  best_perm}`; `pit_permutate` provided.
- `bss_eval_sources(reference_sources, estimated_sources, compute_permutation)` keeps mir_eval's argument order and
  returns `bss_eval_sources_result {sdr, sir, sar, perm}` (float64 / int64, CPU).

### D2 — torchmetrics audio, op for op
- SNR / SI-SDR / SI-SNR / C-SI-SNR / SA-SDR: the same libtorch calls in the same order (`eps =
  torch.finfo(preds.dtype).eps` incl. float16 / bfloat16, `x.pow(2)`, `10 * log10`), result dtype of the input.
- SDR: torchmetrics' own algorithm — float64, L2 normalisation clamped at 1e-6, FFT auto / cross correlation with
  `n_fft = 2^ceil(log2(2T - 1))` (filter taps beyond n_fft are dropped like torchmetrics' slicing), `load_diag` on lag 0,
  `torch.linalg.solve` of the `as_strided` symmetric Toeplitz, or `use_cg_iter` = fast_bss_eval 0.1.4's
  `toeplitz_conjugate_gradient` ported op for op (circulant preconditioner with its float32 `arange / n` weights, FFT
  Toeplitz product, the residual refresh `epoch + 1 % 5 == 0` that never fires); `coh / (1 - coh)`; float32 result
  unless preds is float64. A silent target without load_diag stays torch's singular-matrix error (as torchmetrics).
- PIT: speaker-wise metric matrix `[batch, target, pred]` filled like torchmetrics (copy into `empty` of the first
  result's dtype), exhaustive search (`torch.max/min(dim=1)`, first index on ties) for spk < 3, scipy's
  `linear_sum_assignment` for spk >= 3 — ported from `rectangular_lsap.cpp` (Crouse's shortest augmenting path, its
  reverse-order `remaining` list and "prefer an unassigned column" tie rule; NaN / -inf (after negation when
  maximizing) or an infeasible matrix raise like scipy). Permutation-wise: `index_select` over itertools-ordered
  permutations, `repeat_interleave` of the targets, mean over the trailing values.
- Class states: torchmetrics' `sum` (float32 zero) / `total` (int64), accumulated out of place as in progress11 D5:
  float32 inputs give torchmetrics' bits, float64 values keep a float64 state.

### D3 — STOI = pystoi 0.4.1
- `stoi(x = target, y = preds, fs)` per signal over the last dim (torchmetrics loops over the flattened batch):
  resampling to 10 kHz exactly as pystoi (`_resample_window_oct`: Kaiser beta 0.1102 (60 - 8.7), numpy's `i0`
  Chebyshev series, `np.sinc`, the window normalised by numpy's pairwise sum) through `scipy.signal.resample_poly`'s
  polyphase `upfirdn` — same padding (`n_pre_pad`, `n_pre_remove`), same tap order and the fused multiply-adds of the
  arm64 scipy build (`std::fma`; a plain multiply-add differed in 67 of 129 outputs); `np.hanning(258)[1:-1]`; framing
  `range(0, len - 256, 128)`; silent frames 40 dB below the loudest clean frame dropped and the windowed frames
  overlap-added; 512-point rfft STFT; the 15-band 1/3-octave matrix (`thirdoct`, argmin bins); 30-frame segments;
  normalisation, clipping at `10^(15/20)`, mean / norm removal; extended STOI's row / column normalisation (scaled by
  the reciprocal norm like pystoi's diagonal matmul; rows / columns constant up to rounding scaled by 0, D7). Computed
  with libtorch float64 on the CPU.
- Returns float64 [...] on the CPU (torchmetrics' default `keep_same_device=False`; that option is not provided).
  Fewer than 30 STFT frames after silent-frame removal gives 1e-5 (pystoi's warning value). Inputs shorter than one
  256-sample frame at 10 kHz (resampled length <= 256, e.g. 409 samples at 16 kHz) are `std::invalid_argument` —
  pystoi raises an AxisError there.

### D4 — bss_eval_sources = mir_eval 0.8.2
- 512-tap time-invariant distortion filters: FFT (`n_fft = 2^ceil(log2(nsampl + 511))`) inner products of delayed
  references, the Gram matrix assembled as mir_eval's loop leaves it (block (r, c) = ss(r, c), block (c, r) =
  ss(r, c)^T for r >= c; diagonal blocks transposed), solved per projection (`linalg_solve_ex`, and least squares
  (gelsd) when LU reports a singular matrix, mir_eval's `LinAlgError` fallback), filtered sums by FFT convolution;
  SDR / SIR / SAR with `_safe_db` (inf for a zero denominator); the permutation of the largest mean SIR (numpy
  pairwise mean, first maximum in itertools order, NaN wins like `np.argmax`); `compute_permutation=false` evaluates
  estimate j against reference j only. mir_eval's per-call Gram matrices are computed once (the values are the same).
- Validation as mir_eval: same shape, 1-D promoted to one source, empty inputs give empty results, an all-zero
  reference or estimate and more than 100 sources raise; additionally complex and > 2-D inputs raise.

### D5 — Speaker verification (no torchmetrics counterpart)
- ROC = `sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)` of target (label 1) / non-target scores:
  one point per distinct score from the highest down, after (FPR, FNR) = (0, 1) at threshold +inf; FPR = fp / N,
  FNR = 1 - tp / P in float64 (tied scores form one point).
- `equal_error_rate(target_scores, nontarget_scores)`: the first point i with d_i = FNR_i - FPR_i <= 0 and the
  previous one joined linearly: EER = FPR_{i-1} + t (FPR_i - FPR_{i-1}), t = d_{i-1} / (d_{i-1} - d_i); FPR_i when
  d_i = 0.
- `min_dcf(target_scores, nontarget_scores, {p_target = 0.01, c_miss = 1, c_fa = 1})`: min over all ROC points
  (every distinct threshold plus "reject all") of `c_miss * FNR * p_target + c_fa * FPR * (1 - p_target)`, divided by
  `min(c_miss * p_target, c_fa * (1 - p_target))` (normalised NIST form).
- Both float64 0-d; scores 1-D, non-empty, finite; classes keep every batch and evaluate the concatenation.
  gen_golden.py's `eer_np` / `min_dcf_np` are independent numpy / sklearn implementations of these formulas.

### D6 — Text metrics = torchmetrics.text
- UTF-8 decoded to code points (invalid sequences — bad lead / continuation bytes, truncation, overlongs, surrogates,
  > U+10FFFF — are `std::invalid_argument`); words split like Python `str.split()` (runs of the 29 `str.isspace()`
  code points: ASCII \t \n \v \f \r, U+001C-001F, space, U+0085, U+00A0, U+1680, U+2000-200A, U+2028, U+2029,
  U+202F, U+205F, U+3000). CER counts code points like `list(str)` (whitespace included, no normalisation).
- WER / CER / MER / WIL / WIP: per-pair Levenshtein distance (the existing
  `torchmedia::audio::functional::edit_distance`, identical to torchmetrics' `_edit_distance`), float32 sums in
  torchmetrics' order (errors, target / preds token counts, max lengths; WIL / WIP subtract the max lengths per
  update), `errors / total`, MER `/ sum of max lengths`, WIP `(e / N_t) (e / N_p)`, WIL `1 - WIP`. Empty corpora give
  torchmetrics' NaN / inf.
- EditDistance: torchmetrics' `_LevenshteinEditDistance` (reference = target), i.e. sacrebleu's TER DP that fills only
  a band of 25 (or `ceil(ratio / 2 + 25)`) cells around the length-ratio diagonal except in the last row — so strings
  whose optimal alignment leaves the band get more than the Levenshtein distance (goldens include two such pairs:
  86 vs 60 at substitution cost 2, 73 vs 60, and three pairs past its upper edge: 43 / 86 / 60 vs 30 / 60 / 60 at
  cost 1). Reductions: none (int32 [N]), sum (int64), mean (float32); no pair gives int32 0; the class keeps
  torchmetrics' int64 sum / count (mean) or the per-pair list (none), and its compute / forward squeeze a one-element
  result to 0-d (Metric's `_squeeze_if_scalar`; the functional keeps [1]).
- Substitution costs are Python ints in torchmetrics. The DP caps the cost at 2^31: cells never exceed the beam's
  `undefined` (1e16), so there is no int64 overflow, every distance that fits int32 is unchanged and the others stay
  above int32. Capping at 2 would not do: where consecutive beam rows do not overlap (length ratio in (49, 50]) the
  DP must substitute (`edit_distance(["ab"], ["x" * 100], substitution_cost=1000)` = 1100, 103 at cost 3).
- jiwer 4.0 is a cross-check only: its WER / CER equal torchmetrics on corpora with single-space separators and
  non-empty references (gen_golden.py asserts it; the sweep counts it). It differs by design elsewhere (splits on " "
  only, strips CER input, MER / WIL / WIP from the alignment's hits instead of torchmetrics' max-length formulas).

### D7 — Deviations (deliberate)
- Stricter checks, `std::invalid_argument` before any state change: WER-family `preds` / `target` of different
  lengths (torchmetrics zips and silently drops the tail), invalid UTF-8, a negative `substitution_cost` in the
  functional (torchmetrics checks it only in the class), non-floating SNR-family inputs (torchmetrics: TypeError from
  `finfo`), C-SI-SNR shapes compared before the reshape (torchmetrics accepts different shapes of equal size), SDR
  `filter_length >= 1` and `use_cg_iter >= 0`, SDR of 0-d inputs or an empty time axis (torchmetrics: IndexError /
  math domain error in `log2(2T - 1)`), STOI `fs > 0` and real inputs, a missing PIT `metric_func`, an EditDistance
  result beyond int32 (torchmetrics: RuntimeError in the int32 conversion, or ValueError when the trace meets an
  undefined beam cell).
- Float64 class states for float64 values (progress11 D5); ShortTimeObjectiveIntelligibility therefore returns
  float64 (torchmetrics' float32 state rounds pystoi's float64 values). STOI of a 1-D input with too few frames is
  float64 1e-5 (torchmetrics: `torch.tensor(1e-5)`, float32).
- Extended STOI omits pystoi's `EPS * np.random.standard_normal` dither (2.2e-16 absolute, a different value every
  call): results agree within 1e-15 (D9) wherever the dither is negligible. It is not where a row (a band envelope over
  a 30-frame segment) or a column (after row normalisation) is constant up to rounding: >= ~0.4 s of zero-filled
  dropout in the prediction, the segments where one speech frame borders such a dropout (every row then normalises to
  the same vector), a constant or hop-periodic stretch, an all-zero reference or prediction. pystoi's dither turns such
  a row / column into a random unit vector — a random contribution of mean 0, std ~1e-3 on the result. torchmedia
  scales rows / columns whose mean-removed norm is <= 1e-13 of their norm by 0, i.e. returns the expected value of
  pystoi's result (0 for an all-zero signal; pystoi: random, std 2.7e-3); other inputs keep their bits. (Without the
  threshold these inputs gave NaN (0 * inf), which also made the class mean NaN; mapping only exact-zero norms to 0
  still left 1e-4-level rounding noise from the border segments.)
- bss_eval_sources with one source: SIR is inf (no interference by construction; the two projections are one
  computation here). mir_eval runs them as separate FFT / LAPACK calls on separately allocated copies, which are not
  always bit-identical: inf or 277-298 dB of rounding noise (3 of 133 one-source sweep cases; with threaded OpenBLAS
  it also changed run to run). Underdetermined inputs (nsampl <= nsrc * 512: the filters fit the estimate) have
  artifact / interference energies at rounding level on both sides, so SAR (100-155 dB) and SIR differ there (D9).
- STOI's resampling follows the arm64 scipy wheel's fused multiply-adds; other scipy builds round each product (a
  1-ulp-level difference).

### D8 — Validation and no-exceptions builds
- As progress11 D8: `xxx_error(...) -> std::string` checks, `return detail::invalid(msg)` / `return detail::fail(msg)`,
  constructors assign their fallback before failing. Errors torchmetrics leaves to torch stay `c10::Error`: the
  singular SDR solve without load_diag, `torch.cat` of no batch (EqualErrorRate / MinDCF / EditDistance(none) before
  any update). `audio_test_noexcept` (which includes `torchmedia.hpp`) builds the new headers with
  `LIB_TORCH_MEDIA_NO_EXCEPTIONS`.
- PIT's `linear_sum_assignment` check (NaN / -inf / infeasible metric matrix, spk >= 3) needs the metric matrix, so
  it runs inside the computation: without exceptions the functional returns undefined tensors and
  PermutationInvariantTraining::update skips the accumulation (checked at run time in `audio_test_noexcept`).

### D9 — Tests, goldens, coverage, measured parity
- `gen_golden.py` gains the audio / text sections and writes `golden_{audio,text}.hpp` (the phase-1 goldens are
  byte-identical). mir_eval's values are not bit-reproducible run to run (threaded OpenBLAS solves: ~1e-13 dB; with
  `OPENBLAS_NUM_THREADS=1`, which the script pins, ~5e-15 dB from allocation-dependent BLAS / FFT paths), so a rerun
  may rewrite a few `bss*` digits — far inside the 1e-9 dB test tolerance (keep the committed digits when nothing
  else in them changed). `estoi_dropout` is pystoi with the dither removed and the D7 rule (a numpy
  `row_col_normalize`), asserted within 4 standard errors of 200 seeded pystoi runs. Inputs: `speech()` — a jittered
  pulse train through two resonators in syllables with exact-zero pauses
  (exercising STOI's silent-frame removal), only `+ - * /` in Python float order, mirrored in `test_audio.cpp` under
  `TORCHMEDIA_FP_CONTRACT_OFF` and pinned by checksums; [2, 3, 4000] sources at 8 kHz; STOI at 8 / 10 / 16 / 22.05 /
  44.1 kHz; tied score grids for EER; English / Chinese / Japanese / mixed-script / emoji / empty strings and every
  `str.split()` separator kind for text.
- ctest `metric_test_audio` (415 checks) and `metric_test_text` (147): torchmetrics goldens bit-exact on arm64
  (float32 / float64 / float16; SNR family, SDR incl. load_diag / CG / short signals / a silent target, PIT values and
  permutations incl. ties, scipy LSA on 24 tie-heavy integer matrices, EER / minDCF, every text metric); STOI within
  1e-12 and bss_eval within 1e-9 dB (measured below); classes: batched updates equal torchmetrics' accumulated
  states, forward() on a stateful instance returns the batch alone, reset; every validation branch, each with an
  input that only it rejects (a targeted mutation run, Agent log); boundaries: float16 eps (silent float16 SNR),
  exactly 30 STOI frames, a non-contiguous STOI batch, the bss permutation tie rule, both EditDistance beam edges;
  extended STOI of a 0.5 s zero-filled dropout and of all-zero signals; MPS SI-SDR (functional and class) when
  available.
- Coverage (Debug + `TORCHMEDIA_COVERAGE`, scratch `metric-audio/build-cov`, `cov_union.sh` now over the five metric
  binaries): `_audio.hpp` 987/987 and `_text.hpp` 236/236 lines in their owning binaries; `_common.hpp` 80/80 as the
  union (the union's `_audio.hpp` list shows only comment lines of functions other binaries instantiate). Phase-1
  headers keep progress11 D9's numbers (853 / 997 / 299).
- Sweep (`sweep/sweep_audio_gen.py` + the extended `metric_sweep` driver + `compare.py`; seeds 1 and 2, N = 200: per
  seed 200 cases of each kind and 50 STOI cases):
  - Bit-exact, 400 / 400 per metric: SNR, SI-SDR, SI-SNR, SA-SDR (both scale modes), C-SI-SNR (198 / 198 even
    lengths), SDR (filter_length 16-512, CG 0-19 iterations, load_diag, zero_mean, float32 / float64, 100-2500
    samples), PIT values and permutations (2-5 speakers, speaker- / permutation-wise, max / min, tied speakers), EER /
    minDCF (1-400 target / 1-1500 non-target scores rounded to 0-3 decimals, random p_target / costs), WER / CER /
    MER / WIL / WIP / EditDistance (none; random multi-script corpora with every separator kind, substitution cost
    0-2).
  - STOI vs pystoi (8-48 kHz incl. 11.025 / 12 / 22.05 / 32 / 44.1 kHz, 0.33-2 s speech-like, half extended):
    STOI 26 / 50 bit-exact, max |delta| 1.1e-16; extended STOI 8 / 50 bit-exact, max |delta| 5.0e-16 (pystoi's
    random dither included).
  - bss_eval_sources vs mir_eval (1-3 sources, 600-3000 samples): permutations 400 / 400 equal; well-determined
    (nsampl > nsrc * 512, 175 cases) max |delta| SDR 4.3e-14, SIR 1.4e-12, SAR 7.1e-14 dB; one source (133) SDR / SAR
    3.6e-14 dB, SIR inf (mir_eval inf in 130, 277-298 dB in 3, D7); underdetermined (92) SDR 4.3e-14 dB, SIR up to
    1.2e-3 dB and SAR up to 41 dB apart at their 100-155 dB rounding-noise level.
  - jiwer WER / CER equal torchmetrics on all 137 single-space corpora of the two seeds.

## Tasks
- [x] [task01 — torchmetrics audio metrics and STOI](tasks/task01-metric-audio-torchmetrics.md)
- [x] [task02 — bss_eval_sources and speaker-verification metrics](tasks/task02-metric-bss-speaker.md)
- [x] [task03 — text metrics](tasks/task03-metric-text.md)
- [x] [task04 — goldens, coverage and parity sweep](tasks/task04-metric-audio-text-goldens.md)

## Issues / Gotchas
- scipy's `upfirdn` on the arm64 wheel is compiled with FMA contraction; `std::fma` reproduces it bit for bit.
- numpy 1.26's threaded OpenBLAS `solve` is not run-to-run reproducible (pin `OPENBLAS_NUM_THREADS=1` in reference
  scripts that call LAPACK); even single-threaded, mir_eval's results move by ~5e-15 dB between runs.
- torchmetrics' SDR solve is singular for a silent target unless `load_diag` is given.
- `torch::fft::rfft` of an empty batch with padding raises; STOI returns 1e-5 before the STFT when frames are few.
- Python's `scalar - tensor` is `rsub`; C++ `1 - t` is `full(1).sub_(t)`: the same values. `scalar / tensor` is not
  (progress11).
- An unqualified `audio::...` inside `torchmedia::metric` resolves to `metric::audio`; name
  `::torchmedia::audio::functional` explicitly.
- `pystoi` length boundary: 409 samples at 16 kHz resample to 256 (invalid), 410 to 257 (one frame, 1e-5).
- Review (2026-10-24), extended STOI: a zero-filled stretch of >= ~0.4 s in the prediction gave NaN (0 * inf in the
  row scaling), and one such utterance made the class mean NaN. Mapping zero norms to 0 removes the NaN but not the
  segments where one speech frame borders the dropout: there all rows normalise to the same vector, the columns are
  constant up to rounding and their 1e-17 residue is scaled to unit norm — numpy and libtorch already differed by 7e-5
  on identical segments. Fixed by the relative 1e-13 test of D7.
- Review: the suggested `min(substitution_cost, 2)` for the int64 overflow changes results where beam rows do not
  overlap (D6); the cap is 2^31 instead.
- mir_eval's permutation for exactly tied mean SIRs (identical estimates) follows its run-to-run rounding noise (one
  run gave [1, 0]); the tie rule (first maximum, like `np.argmax`) is tested as a closed-form expectation.

## Open / TODO (carry-over)
- PESQ (ITU-T P.862 licence), DNSMOS / NISQA and other model-based metrics, SRMR; LSD / MCD (no single definition).
- BLEU / SacreBLEU / chrF / ROUGE / TER / EED / SQuAD / perplexity / BERTScore / InfoLM (NLP, not media).
- mir_eval `bss_eval_images` and the framewise variants; torchmetrics' `keep_same_device`.
- Classification metrics (later phase). Parity off arm64 not measured (tests allow 1e-5 there).
- readme.md does not mention the metric module yet — Mux's call.

## Agent log
- 2026-10-24 [ai] Read torchmetrics 1.9.0 audio (snr, sdr, pit, stoi; functional + classes) and text (helper, wer,
  cer, mer, wil, wip, edit), fast_bss_eval 0.1.4 cgd, pystoi 0.4.1, mir_eval 0.8.2 separation, scipy 1.17.1
  resample_poly / upfirdn / linear_sum_assignment, sklearn 1.9.1 roc_curve, jiwer 4.0.0. Implemented D1-D8 (2 headers,
  16 classes, 18 functionals). Tests: metric_test_audio 396, metric_test_text 140 checks; goldens bit-exact except
  STOI / bss_eval (tolerances 1e-12 / 1e-9 dB, measured in D9). Coverage D9; sweep seeds 1 and 2 (D9). Full Release
  build + ctest green (15 / 15); the new tests build without warnings under -Wall -Wextra; clang-format on every new /
  changed C++ line. Not committed.
- 2026-10-24 [ai] Review fixes (parity and conventions reviews). Extended STOI: rows / columns constant up to
  rounding scaled by 0 (D7; was NaN for zero-filled dropouts and all-zero signals). EditDistance: compute / forward
  squeeze one pair to 0-d; substitution cost capped at 2^31 (signed overflow near INT64_MAX) and distances beyond
  int32 rejected (were wrapped). PermutationInvariantTraining::update skips an undefined result in no-exceptions
  builds (it called `.sum()` on it). SDR rejects 0-d inputs and an empty time axis (UB float-to-int cast of
  log2(-1)). `stoi_*` / `bss_*` constants `inline constexpr`; the unused `squeeze_if_scalar` using-declaration dropped
  from `_audio.hpp`; D1 and the `_text.hpp` comment state that forward() / operator() need a named vector. Tests and
  goldens: a 0.5 s dropout (STOI and extended), all-zero extended STOI = 0, the dropout in the class mean, a silent
  float16 SNR, exactly 30 STOI frames, a transposed STOI batch, same-shape non-(..., 2) C-SI-SNR inputs, one silent
  bss source on either side, the bss tie rule, the `fs > 0` message, SDR empty / 0-d, the upper beam edge (3 pairs),
  huge substitution costs, one pair under reduction none; a run-time PIT check in `audio_test_noexcept`. Every new
  bug test fails on the pre-fix headers (metric_test_audio 5 failures, metric_test_text 4, audio_test_noexcept 1) and
  16 targeted mutants (the reviewer's surviving mutants re-created, plus each fix reverted or weakened, e.g. zero norms
  only) are all killed. Parity: the reviewer's probe rebuilt on the fixed headers gives bit-identical results in all
  2050 other outputs of its stoi / stoi60 / misc / gap / pit cases; only the 3 former NaN cases changed, now within 0.03 / 0.46 / 1.14 standard errors of 200 seeded pystoi
  runs (450 / 600 ms packet loss: 0.457396 / 0.329485 vs 0.457395 / 0.329517; all-zero preds 0 vs 0.00022, std
  2.7e-3). Final: metric_test_audio 415, metric_test_text 147, audio_test_noexcept 111 checks; coverage
  `_audio.hpp` 987/987, `_text.hpp` 236/236, `_common.hpp` 80/80 (union), phase-1 853 / 997 / 299 unchanged; full
  Release build + ctest 15 / 15 green; -Wall -Wextra clean on the three changed tests; clang-format clean.
