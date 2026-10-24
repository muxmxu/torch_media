# Progress 05 — One contract for the silent (LIB_TORCH_MEDIA_NO_EXCEPTIONS) build, and last_error()
id: 2026-10-24/progress05
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress02, 2026-10-23/progress04, 2026-10-23/progress10, 2026-10-23/progress11,
       2026-10-24/progress01, 2026-10-24/progress02, 2026-10-24/progress03]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/global_include.hpp
  - libtorchmedia/include/torchmedia/_audio/{_functional,_functional_alignment,_functional_beamforming,_functional_filtering,_compliance_kaldi,_io,_separation}.hpp
  - libtorchmedia/include/torchmedia/_audio/{_transform_augment,_transform_beamform,_transform_effects,_transform_feature,_transform_time}.hpp
  - libtorchmedia/include/torchmedia/_vision/{_functional,_io,_transform_color,_transform_geometry,_transform_misc}.hpp
  - libtorchmedia/include/torchmedia/_data/_folder.hpp
  - libtorchmedia/include/torchmedia/_metric/{_common,_detection}.hpp
  - unit_test/error_policy/ (new), unit_test/vision/noexcept/ (new), unit_test/CMakeLists.txt,
    unit_test/vision/CMakeLists.txt, unit_test/audio/noexcept/main.cpp, unit_test/data/test_noexcept.cpp
  - CLAUDE.md, readme.md

## Goal
The error mechanism is a user choice (Mux: "你喜欢异常——那就异常，你喜欢闭嘴——那就闭嘴。交给用户决定。"): the default
build throws, a `LIB_TORCH_MEDIA_NO_EXCEPTIONS` build stays silent. Make the silent side a contract a caller can rely
on: one documented fallback per return type, every library error routed through it, nothing computed from a fallback,
and `torchmedia::last_error()` to learn why. The default build behaves exactly as before.

## Context / Motivation
progress02 D7/D10 made callers `return handle_exceptions(...)` and composites validate first. What was left (measured
on master 60873fe, see the inventory): 81 of the 131 `handle_exceptions` sites returned `torch::empty({1})` (82
values: forced_align's pair had two) — a defined 1-element tensor of uninitialised memory — while the rest returned
an undefined tensor, so a silent caller
could not tell failure from success; 24 raw `throw`s (audio / vision I/O, separation, the image loader) ignored the
switch; composites passed a part's fallback on (into torch ops: c10::Error or garbage); and there was no way to learn
why a call failed. Decisions D1-D7 were set by the workflow orchestrator under Mux's delegation; the per-site choices
are the implementer's. Mux reviews.

## Decisions

### D1 — One fallback per return type
- Decision (silent build):

  | returns | fallback | callers test |
  |---|---|---|
  | `tensor_t` | undefined tensor (`tensor_t{}`) | `!out.defined()` |
  | struct / pair / tuple (`forced_align`, `get_mel_banks`, `curve_t`, `bss_eval_sources_result`, `voc_average_precision_result`, `mean_average_precision_result`, `load_audio_t`, `example_t`, `padded_batch`, `speed`) | value-initialised: tensors undefined, optionals empty, numbers at their default member values (0 unless the struct says otherwise: the detection results' `map` fields stay -1, the value of a valid call without targets) (`load_audio_t` = {undefined, 0}; `speed` = {undefined, nullopt}) | its tensor fields |
  | containers (`five_crop` / `ten_crop`, `merge_tokens`, `random_split`, `make_dataset`, `find_classes`) | empty | `.empty()` |
  | `audio_info` (`info`, `info_from_memory`) | value-initialised (`sample_rate` 0, `encoding` "") | `sample_rate == 0` |
  | `loader_fn` (`audio_loader`) | empty `std::function` (a dataset then rejects it) | `!fn` |
  | `void` (metric `update`) | no-op, state untouched | `last_error()` |
  | metric `forward()` / `operator()` (the batch's `compute()`) | value-initialised result of that type (tensor metrics: undefined) when the batch or the state update was rejected; state untouched | as its type |
  | bool APIs that already return bool (`save_audio`, `save_image`) | `false` in both builds; not an exception site, `last_error()` untouched | the bool |
  | a dataset sample whose loader failed | `data` undefined, target kept; the data transform does not run | `!ex.data.defined()` |
- Why: one rule per type; an undefined tensor is the only tensor value no successful call returns.
- Impact: all `torch::empty({1})` fallbacks removed (`handle_exceptions<T, E>(msg)` returns `T{}`).

### D2 — Constructors: the object stays callable, in a documented state
- audio transforms: every later call returns the fallback and records `"<Class>: invalid arguments."` (Resample,
  PitchShift as before; now also MelScale, InverseMelScale, MFCC, LFCC, AmplitudeToDB, Vol; Speed / SpeedPerturbation
  through their Resample). TimeMasking / FrequencyMasking (p) and GriffinLim (momentum) keep the option and the
  functional op rejects it on every call (its own message).
- metric classes: options reset to the documented fallback defaults (progress11 D8 / progress02 D8, unchanged).
- vision transforms (progress10 D5 "callable"): a size that `setup_size` rejected (RandomCrop, RandomResizedCrop) and a
  rejected LinearTransformation make every call fail with `"<Class>: invalid arguments."`; options the functional op
  validates per call are kept (Resize, Pad, CenterCrop, FiveCrop, TenCrop, GaussianBlur kernel / sigma, the center of
  RandomRotation / RandomAffine); the others are reset — to torchvision's default (RandomErasing scale, RandomZoomOut
  side_range, RandomPerspective distortion_scale) or to a neutral value (RandomCrop padding: none, RandomAffine
  translate / scale: none, a rejected degrees / shear angle: (0, 0), ElasticTransform alpha / sigma: (0, 0), ColorJitter
  ranges: no jitter, RandomChoice p summing to 0: equal weights, empty Compose: identity). A p outside [0, 1] is kept
  as given (draws compare against it: never / always applied).
- data: DatasetFolder / ImageFolder / AudioFolder / ManifestDataset / Subset / ConcatDataset are empty (size 0, `get`
  returns the undefined example); PadCollate's multiple falls back to 1 (progress03 D8, unchanged).
- Why: the task allowed documenting the existing state; the audio transforms and the vision classes listed above
  computed with the rejected option (garbage, or a c10::Error from a reversed `uniform_` / zero-size resize, or UB from
  `int64(NaN)` in RandomShortestSize), which is no documented state.

### D3 — `torchmedia::last_error()` / `clear_last_error()`
- Decision: `global_include.hpp`; silent build: a function-local `thread_local std::string` set by every
  `handle_exceptions` failure; `last_error()` returns a copy, success does not clear it, `clear_last_error()` does
  (errno-like). Throwing build: `last_error()` returns `""` and `clear_last_error()` is a no-op (no storage, nothing on
  the throwing path changes).
- Why: one tiny header-only channel; per-thread so loader workers do not race.
- Impact: define the macro for the whole program (both sides are `inline`; mixing TUs is an ODR violation, as
  `handle_exceptions` already was).

### D4 — Every library error goes through handle_exceptions
- Decision: the raw throws now `return handle_exceptions<T, E>(msg)` with the same exception type and message:
  `audio::io` (`load` / `open` / `info`; decoders record a failed open in `decoder::error` and `open()` reports it,
  their destructors skip the uninit of a stream that never opened), `audio::separation::auxiva_ip`, `vision::io`
  (`load_image`, `decode_image`, `read_file`, `stb_decode`). Two throws stay: `pnm_decode`'s parser fails from nested
  helpers and `load_image` reports its `std::runtime_error` through `handle_exceptions` (caught at that one boundary;
  the default build rethrows the same type and message); the image loader still rewraps a non-library exception
  (c10::Error) with the file name.
- Why: those areas ignored the user's choice.

### D5 — Composites stop at a part's fallback
- Decision: a composite that processes a part's result checks it (`if (!x.defined()) return x;`), so nothing is
  computed from a fallback and `last_error()` keeps the part's message: `lfilter`, `filtfilt`, `spectral_centroid`,
  `speed`, `melspectrogram`, `mfcc`, kaldi `vtln_warp_mel_freq`; MelSpectrogram, MFCC, LFCC, SpecAugment (loop
  guard), MVDR (both PSDs), SoudenMVDR, RTFMVDR, Speed; vision `solarize`, `equalize`, the 11 `query_chw` callers
  (`query_chw` returns {-1, -1, -1}), RandomPhotometricDistort / ColorJitter colour chains; `decode_image` (PNG),
  `read_image`; the data image / audio loaders, `path_dataset::get` (silent build only: no transform on a failed
  loader's fallback), `folder_list` (stops after `find_classes`); the vision containers Compose / RandomApply /
  RandomOrder (after each part, silent build only, for the same reason as `path_dataset::get`); metric
  `forward()` (a rejected batch update skips the state update and returns the value-initialised result) and
  `mean_average_precision` (a rejected option or input), both through `metric::detail::silently_failed`, which
  clears the slot around the call and restores the earlier message when it succeeds (silent build only).
- Exactness: a guard only fires on a part's fallback, never on an undefined *input* (the op still rejects that, as
  before): `speed` / Speed drop the lengths only when resample rejected a defined waveform, SpecAugment and the colour
  chains keep calling the op while the input itself is undefined, `path_dataset::get`'s skip is under
  `#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS` (a user loader may return an undefined tensor in the default build), Vol's
  call-time check is limited to the amplitude gain type its constructor rejects, the vision containers check after a
  part, so an undefined input still reaches the first one. `error_policy` checks two undefined inputs in both builds
  and a Compose one in the silent build.
- Why: with an undefined fallback, passing it on ends in a c10::Error — the silent build would throw after all.

### D6 — libtorch's c10::Error is not ours to silence
- Decision: what torch rejects itself (and what torchaudio / torchvision / torchmetrics leave to torch, see the
  earlier progresses) still throws c10::Error in both builds. `handle_exceptions` never catches.

### D7 — Tests: both builds of the same calls
- Decision: `unit_test/error_policy/main.cpp` builds twice. For one representative failing call per public area
  (audio functional / transform / io / separation / kaldi, vision functional / transform / io, every metric area, data)
  `error_policy_test_throw` checks the expected exception type and message and an empty `last_error()`;
  `error_policy_test_silent` checks the fallback and that `last_error()` names the failure; silent-only sections cover
  D2 / D5 and the `last_error()` semantics (thread-local, kept on success, cleared). `audio_test_noexcept` and
  `data_test_noexcept` now require an undefined fallback plus a recorded message per site; `vision_test_noexcept`
  (new) does the same for every error site of `_vision/_functional.hpp` and `_vision/_io.hpp`, and
  `audio_test_noexcept` for every `auxiva_ip` check.
- Mutation check (Agent log): a `torch::empty({1})` fallback at one site fails both silent binaries; dropping the
  `last_error` update fails every silent binary.

## Inventory (master 60873fe -> this change)
`handle_exceptions` call sites per header (HEAD / now), the `torch::empty({1})` fallbacks removed, raw `throw`s
(HEAD / now), helper calls (`detail::fail`, `detail::invalid`, vision `functional::detail::invalid`), and the
fallback types now in use.

| header | sites | empty({1}) | raw throws | helpers | fallback types now |
|---|---|---|---|---|---|
| `_audio/_compliance_kaldi.hpp` | 8 / 8 | 7 | 0 / 0 | 0 | tensor 7, tuple 1 |
| `_audio/_functional.hpp` | 37 / 37 | 37 | 0 / 0 | 0 | tensor 37 |
| `_audio/_functional_alignment.hpp` | 2 / 2 | 2 (incl. the pair's second) | 0 / 0 | 8 (`fail` lambda) | tensor pair, vector |
| `_audio/_functional_beamforming.hpp` | 11 / 11 | 11 | 0 / 0 | 0 | tensor 11 |
| `_audio/_functional_filtering.hpp` | 17 / 17 | 17 | 0 / 0 | 0 | tensor 17 |
| `_audio/_io.hpp` | 0 / 5 | 0 | 8 / 0 | 0 | `unique_ptr<decoder>` 3 (nullptr), `load_audio_t` 2 |
| `_audio/_separation.hpp` | 0 / 8 | 0 | 8 / 0 | 0 | tensor 8 |
| `_audio/_transform_augment.hpp` | 1 / 1 | 0 | 0 / 0 | 0 | ctor (`int`, discarded) |
| `_audio/_transform_beamform.hpp` | 2 / 2 | 2 | 0 / 0 | 0 | tensor 2 |
| `_audio/_transform_effects.hpp` | 2 / 3 | 1 | 0 / 0 | 0 | ctor 1, tensor 2 |
| `_audio/_transform_feature.hpp` | 8 / 12 | 2 | 0 / 0 | 0 | ctor 4, tensor 7, bool 1 (`check_f_min_max`) |
| `_audio/_transform_spectral.hpp` | 1 / 1 | 0 | 0 / 0 | 0 | ctor 1 |
| `_audio/_transform_time.hpp` | 5 / 5 | 3 | 0 / 0 | 0 | ctor 2, tensor 3 |
| `_data/_collate.hpp` | 0 / 0 | 0 | 0 / 0 | 2 | ctor (multiple -> 1), `padded_batch` |
| `_data/_dataset.hpp` | 1 / 1 | 0 | 0 / 0 | 7 | `invalid<T, E>`: index vectors, examples, splits |
| `_data/_folder.hpp` | 0 / 0 | 0 | 1 / 1 | 10 | sample lists, `find_classes` pair, `loader_fn`, examples |
| `_data/_manifest.hpp` | 0 / 0 | 0 | 0 / 0 | 3 | sample list |
| `_metric/_common.hpp` | 2 / 2 | 0 | 0 / 0 | 1 | `fail` (void), `invalid<T>` (T{}), `forward` value-initialised (`silently_failed`) |
| `_metric/_audio.hpp` | 0 / 0 | 0 | 0 / 0 | 25 | tensor, pair, struct, update no-op, ctor defaults |
| `_metric/_classification_curve.hpp` | 0 / 0 | 0 | 0 / 0 | 4 | `curve_t` / tensor, update, ctor |
| `_metric/_classification_misc.hpp` | 0 / 0 | 0 | 0 / 0 | 12 | tensor, update, ctor |
| `_metric/_classification_stat.hpp` | 0 / 0 | 0 | 0 / 0 | 12 | tensor, update, ctor |
| `_metric/_detection.hpp` | 0 / 0 | 0 | 0 / 0 | 7 | tensor, struct, update, ctor |
| `_metric/_image.hpp` | 0 / 0 | 0 | 0 / 0 | 26 | tensor, update, ctor |
| `_metric/_segmentation.hpp` | 0 / 0 | 0 | 0 / 0 | 9 | tensor, update, ctor |
| `_metric/_text.hpp` | 0 / 0 | 0 | 0 / 0 | 5 | tensor, update, ctor |
| `_vision/_functional.hpp` | 23 / 23 | 0 | 0 / 0 | 32 (+ `rank_error` message builder) | tensor 21, vector 2 |
| `_vision/_io.hpp` | 2 / 9 | 0 | 7 / 1 (pnm, caught in `load_image`) | 6 (pnm `fail` lambda) | tensor 9 |
| `_vision/_transform_color.hpp` | 3 / 3 | 0 | 0 / 0 | 0 | `range_t` 3 (nullopt: no jitter) |
| `_vision/_transform_geometry.hpp` | 4 / 5 | 0 | 0 / 0 | 11 | ctor 2, `vector<double>` 3 ({0, 0}) |
| `_vision/_transform_misc.hpp` | 2 / 2 | 0 | 0 / 0 | 13 | `fail` (void), `query_chw` {-1, -1, -1} |

Total: 131 -> 157 sites (the routed raw throws and the new call-time checks of D2 / D5), `torch::empty({1})` values
82 -> 0,
raw throws 24 -> 2 (D4). Helpers: metric / vision-transform `detail::fail` (void: constructors assign their fallback
first, updates return before touching state), metric / data `detail::invalid<T>` (`T{}`), vision
`functional::detail::invalid` (undefined tensor), `rank_error` / `*_error` message builders (no state),
`setup_size` ({0, 0} + `size_ok`), `setup_angle` / `setup_number_or_seq` ({0, 0}), ColorJitter `check_input`
(nullopt), `check_f_min_max` (false), `query_chw` ({-1, -1, -1}), audio I/O `open` (nullptr).

## Tasks
- [x] [task40 — silent-mode contract and last_error()](tasks/task40-silent-contract-last-error.md)
- [x] [task41 — route raw throws, composites and constructor states](tasks/task41-silent-routing-composites.md)
- [x] [task42 — error_policy tests (both builds), docs](tasks/task42-error-policy-tests-docs.md)

## Issues / Gotchas
- `handle_exceptions<bool, E>("literal")` picked the explicit-fallback overload: the literal converts to `bool`
  (a standard conversion beats `const char *` -> `std::string`), so the message was dropped (silent: returned
  `true`; throwing: an empty `what()`). The explicit-fallback overload's `msg` has no default any more, so a lone
  message never binds there; `error_policy` checks the MelScale message in both builds.
- An undefined tensor's `dim()` is 0 and `sizes()` is empty (no throw), but `device()`, `options()`, `to()`, `size(i)`
  and every op throw: a composite that passes a fallback on either throws c10::Error or fails a later check with a
  misleading message — hence D5.
- `query_chw`'s old fallback {0, 0, 0} drove RandomShortestSize into `int64(0 * inf)` (UB) and other random classes
  into reversed `uniform_` / `randint` ranges.

- An undefined fallback is only as good as the code that receives it: a metric `forward()` computed the metric of an
  empty batch (BinaryAccuracy 0, SNR NaN) and the vision containers handed a part's fallback to the next part
  (c10::Error from a flip, or a later part's message). Found in review; fixed in D5.
- `last_error()` holds the most recent message: where a constructor argument is built by a failing helper before the
  constructor runs (AudioFolder with an invalid `audio_folder_option`: `audio_loader` fails, then DatasetFolder rejects
  the empty loader), the message is the later one ("DatasetFolder: loader must be callable").

## Open / TODO (carry-over)
- `functional::speed` / `transform::Speed` with `orig_freq` 0 divide by `gcd(0, 0)` (SIGFPE in both builds;
  torchaudio: ZeroDivisionError). Pre-existing, not an error site.
- `lfilter` on a 0-D waveform with 1-D coefficients reads `shape.back()` of an empty vector (pre-existing UB).
- A p outside [0, 1] in the vision random classes is kept (D2), not rejected per call.
- The vision transform classes have no per-site table like `vision_test_noexcept`: their sites are `detail::fail`
  (constructors, D2) and `query_chw` (one check per caller in `error_policy`), not one fallback per site.

## Agent log
- 2026-10-24 [ai] Inventory (table above), then task40-task42. Release build green; ctest 23/23 (new:
  `error_policy_test_throw` 58 checks, `error_policy_test_silent` 130 checks; `audio_test_noexcept` 111,
  `data_test_noexcept` 14). Mutants (run on the 54 / 126-check revision of error_policy): M1 `torch::empty({1})` back at
  `_apply_convolve_mode` -> `error_policy_test_silent` 1 and `audio_test_noexcept` 3 failures; M2 the `last_error`
  update dropped -> `error_policy_test_silent` 80, `data_test_noexcept` 12, `audio_test_noexcept` 109 failures.
  `-Wall -Wextra` clean on the headers in both modes. Coverage (TORCHMEDIA_COVERAGE Release build, per-line union over
  all 23 test binaries, D11; comment / blank / preprocessor lines excluded, they read 0 in binaries that instantiate a
  function without running it): every touched header at 100% lines (19 headers incl. `global_include.hpp`). git
  clang-format applied to the changed lines. Not committed.
- 2026-10-24 [ai] Review round (F1-F5). F1: metric `forward()` / `operator()` and the functional
  `mean_average_precision` returned a defined result for rejected input (the metric of an empty batch / computed
  defaults); now the value-initialised result (D1, D5). F2: Compose / RandomApply / RandomOrder passed a part's
  fallback on (c10::Error from a flip, or the later part's message); now they stop after a failing part (D5). F3:
  `vision_test_noexcept` (new, 64 checks: every `_vision/_functional.hpp` / `_vision/_io.hpp` error site) and 8
  `auxiva_ip` checks in `audio_test_noexcept`. F4: RandomAffine clears a rejected scale (no scaling, D2). F5: CLAUDE.md
  names the two remaining raw throws; D1 states the struct default member values. Before the fix,
  `error_policy_test_silent` failed 8 checks (F1 7, F4 1) plus an unexpected c10::Error (F2). Both-build checks added
  for the forward, functional MAP and Compose calls (the throwing build raises the same std::invalid_argument).
  Mutants (single-TU builds against a copy of the headers): `torch::empty({1})` at to_dtype, adjust_hue, load_image's
  PNM catch, decode_image truncated and read_file -> `vision_test_noexcept` 5 failures, one per site; at auxiva_ip
  n_iter and load -> `audio_test_noexcept` 2; `silently_failed` never reporting + functional MAP returning its
  computed result + RandomAffine keeping the scale -> `error_policy_test_silent` 12; the container guard dropped in
  Compose / RandomApply / RandomOrder (one each) -> c10::Error in `test_silent_vision_containers`; the earlier message
  not restored -> 1; the `last_error` update dropped -> `vision_test_noexcept` 64 of 64. Release build green; ctest
  24/24 (`error_policy_test_throw` 64 checks, `error_policy_test_silent` 155, `vision_test_noexcept` 64,
  `audio_test_noexcept` 119, `data_test_noexcept` 14). Coverage (same method as above, 24 binaries): all 21 touched
  headers at 100% lines (now incl. `_metric/_common.hpp`, `_metric/_detection.hpp`). git clang-format clean. Not
  committed.
