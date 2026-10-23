# Progress 10 — Vision random transforms: torchvision v2 augmentation classes with seeded parity
id: 2026-10-23/progress10
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress07, 2026-10-23/progress09, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_transform.hpp (now an aggregator)
  - libtorchmedia/include/torchmedia/_vision/_transform_misc.hpp (NEW)
  - libtorchmedia/include/torchmedia/_vision/_transform_geometry.hpp (NEW)
  - libtorchmedia/include/torchmedia/_vision/_transform_color.hpp (NEW)
  - libtorchmedia/include/torchmedia/_vision/_functional.hpp
  - unit_test/vision/transform/{main.cpp, gen_golden.py, golden.hpp}
  - unit_test/vision/functional/{main.cpp, gen_golden.py, golden.hpp}
  - CLAUDE.md

## Goal
Phase 3 of the vision module: the torchvision 0.20.1 `transforms.v2` random augmentation classes and the containers, so
that a training pipeline (e.g. the ImageNet recipe RandomResizedCrop + RandomHorizontalFlip + ColorJitter + ToDtype +
Normalize) runs in C++ and, seeded with `torch::manual_seed(s)`, gives the same outputs as Python seeded with
`torch.manual_seed(s)`. Images only (`[..., C, H, W]`, uint8 / floating). Out of scope: D7 / Open.

## Context / Motivation
Phases 1-2 (progress07, progress09) made every functional op bit-exact and added the deterministic and p-gated
classes; the random geometric / color families were carry-over. The design (class list, RNG rule, header split, test
plan) was set by the orchestrator under Mux's delegation; conventions are progress07 D5 / progress09 D7.

## Decisions

### D1 — Classes and parity bar
- Decision: in `torchmedia::vision::transform`: RandomCrop, RandomResizedCrop, RandomResize, RandomShortestSize,
  ScaleJitter, ColorJitter, RandomPhotometricDistort, RandomChannelPermutation, RandomRotation, RandomAffine,
  RandomPerspective, ElasticTransform, RandomErasing, RandomZoomOut, GaussianNoise, Identity, LinearTransformation,
  RGB, ConvertImageDtype; containers Compose, RandomApply, RandomChoice, RandomOrder over
  `transform_fn = std::function<tensor_t(const tensor_t &)>` (any class here or any lambda). Option structs
  `xxx_option` with `_field` members and fluent setters, `operator()` + `forward()`, as before.
- Every random class replicates its torchvision class's `_get_params` (and `_RandomApplyTransform.forward`) exactly:
  the same torch calls, in the same order, with the same dtypes, on the global CPU generator (GaussianNoise:
  `randn_like` on the image's device, like torchvision). Seeded runs are bit-identical (D9).
- Threads: instances are const and safe to share, but concurrent calls interleave their draws on the global generator,
  so seeded parity needs the calls serialized (one thread, or per-thread seeding with serialized calls); e.g. not a
  multi-worker data loader.
- Why: a C++ training / augmentation pipeline must reproduce the Python one under the same seed, not merely be
  statistically similar; the functional ops are already exact, so exact params give exact outputs.

### D2 — Header split
- Decision: `_transform.hpp` became an aggregator (like `_audio/_transform.hpp`) of three themed headers mirroring
  torchvision's modules: `_transform_misc.hpp` (shared `detail` helpers, the p-only `random_flip_option` /
  `random_apply_option`, containers, Identity, Normalize, ToDtype, ConvertImageDtype, LinearTransformation,
  GaussianBlur, GaussianNoise, RandomErasing), `_transform_geometry.hpp` (Resize ... TenCrop, the random geometry
  classes), `_transform_color.hpp` (Grayscale, RGB, the p-gated color classes, ColorJitter,
  RandomPhotometricDistort, RandomChannelPermutation). The phase 1-2 classes moved unchanged (GaussianBlur now draws
  through `detail::uniform`, the same call).
- Why: the single header would have been ~1750 lines (orchestrator threshold ~1500).

### D3 — RNG replication details (each pinned by a golden or the sweep)
- Draws: `torch::empty({1}).uniform_(lo, hi)` (float32, read back as double), `torch::randint(lo, hi, {1})`,
  `torch::randperm(n)`, `torch::multinomial(float32 p, 1)`; gates `torch.rand(1) < p` / `>= p` are tensor comparisons,
  i.e. in float32 (`detail::rand_lt` / `skip`).
- Arithmetic torchvision does on tensors stays on float32 tensors: RandomZoomOut's canvas / offsets and ScaleJitter's
  size (`int(orig_w * r)` with r a float32 tensor; golden `jitter_scale_float32_u8`: 11 * f32(13 / 11) gives 12 where
  double gives 13). Python scalar arithmetic stays double: Python `round` is half-to-even (`std::nearbyint`), `int()`
  truncates, ElasticTransform's `int(8 * sigma + 1)` runs under `TORCHMEDIA_FP_CONTRACT_OFF`.
- RandomResizedCrop / RandomErasing: `log(ratio)` precomputed in float32 (`torch.log(torch.tensor(ratio))`), 10
  attempts, RandomResizedCrop then falls back to the whole image center-cropped to the nearest ratio bound;
  RandomErasing computes h with `* ratio` and w with `/ ratio` (the reverse of RandomResizedCrop), needs a strict fit,
  draws the value tensor (`normal_` for 'random') before the position, and returns the input when nothing fits.
- Draw counts: ColorJitter always draws `randperm(4)` (also with no factor configured) and then one uniform per
  configured factor; RandomRotation / RandomAffine draw even for a zero range; RandomCrop draws only along a side
  longer than the crop and returns the input itself when there is nothing to pad or crop; every p-gated class draws its
  gate first (also for p = 0 / 1); RandomPhotometricDistort draws gate + factor per op, the contrast coin, then gate +
  `randperm(C)`.
- torchvision quirks kept: ElasticTransform blurs dx and dy with both sigmas (`gaussian_blur(d, [k, k], sigma)`) and
  scales dx by alpha / H, dy by alpha / W; ScaleJitter's base scale is `min(target_size[1] / H, target_size[0] / W)`;
  RandomPerspective solves the coefficients from integral corner points through `functional::perspective` (the same
  `_get_perspective_coeffs`).

### D4 — Option encodings
- `{}` is torchvision's None throughout: RandomCrop `padding`, RandomAffine `translate` / `scale` / `shear`, ColorJitter
  factors, `center`. `fill` defaults to `{0}` like torchvision's `fill=0` (the masked fill path of the grid ops, which
  differs from `{}` = None for bilinear); one value or one per channel.
- Ranges torchvision indexes as `[0]`, `[1]` are `std::array<double, 2>` (RandomResizedCrop / RandomErasing
  scale and ratio, RandomZoomOut side_range, ScaleJitter scale_range / target_size, RandomPhotometricDistort's
  ranges); ranges with a number form are vectors: ColorJitter `{x}` (center +- x, brightness / contrast / saturation
  clipped at 0) or `{lo, hi}`; degrees / shear `{d}` = (-d, d); ElasticTransform alpha / sigma `{x}` or `{x, y}`.
- RandomErasing `value`: `{v}`, one per channel, or `{}` = torchvision's `'random'` (standard normal per pixel).
- Containers take their arguments directly (`RandomApply(transforms, p = 0.5)`, `RandomChoice(transforms, p = {})`),
  no option struct: torchvision's positional signature, and a transform list is not an option.
- `random_apply_option` serves the p-gated colour classes only (RandomApply(transforms, random_apply_option) does not
  compile). The phase-2 template behind those classes is now `detail::p_gated` (was `detail::RandomApply`, which
  clashed with the container's name).

### D5 — Validation
- Constructors validate as torchvision's `__init__` (`std::invalid_argument`): sizes, padding length, p, scale in
  [0, 1] (RandomErasing), side_range, degrees / shear / center lengths and signs, translate in [0, 1], scale > 0,
  distortion_scale in [0, 1], alpha / sigma lengths, ColorJitter ranges / bounds, LinearTransformation shape / device /
  dtype, RandomChoice p length, Compose non-empty. Classes torchvision does not validate (RandomResize,
  RandomShortestSize, ScaleJitter, RandomPhotometricDistort, GaussianNoise) validate nothing here either.
- Call-time errors torchvision raises itself (ValueError / TypeError, and its own RuntimeError, e.g. the unsafe
  float32 -> int32 ConvertImageDtype of phase 1's to_dtype) are `std::invalid_argument` (crop larger than the padded
  image, RandomErasing value length, LinearTransformation input size / device, GaussianNoise dtype / sigma, rank < 2);
  errors torchvision leaves to torch stay torch's own exception, `c10::Error` (`uniform_` with from > to, an empty
  `randint` range, a zero resize, a reflect pad wider than the image, a fill of the wrong length) or a plain
  `std::runtime_error` (a fill that overflows uint8, e.g. RandomZoomOut fill {300}; a RandomResizedCrop of a 0-row
  image). Catch `std::exception` to catch them all.
- By design `std::invalid_argument` where torchvision fails incidentally on a 2-D image (IndexError): RGB, and phase
  2's ColorJitter / RandomPhotometricDistort colour ops.
- No-exceptions builds: every fallback assignment runs before the throw (or is the `handle_exceptions` return value),
  so the classes stay in a callable state without out-of-bounds indexing, and the throw lines stay covered.

### D6 — New functional ops and functional fixes
- `grayscale_to_rgb(img)`, `permute_channels(img, permutation)` (Python indexing: negatives wrap),
  `gaussian_noise(img, mean = 0, sigma = 0.1, clip = true)`: torchvision 0.20.1's kernels behind RGB,
  RandomChannelPermutation / RandomPhotometricDistort and GaussianNoise, with functional goldens.
- `pad` with `padding_mode::symmetric` wider than the image now raises `c10::IndexError` (`TORCH_CHECK_INDEX`) on every
  device, exactly when torch's CPU gather does: only for a non-empty output (an empty batch or zero channels pad like
  torchvision). Before, MPS gathered out of bounds and returned garbage (found by the MPS sweep through
  RandomCrop(pad_if_needed, symmetric); torchvision on MPS still returns garbage there). Golden
  `pad_symmetric_full_u8` pins the widest valid pad; the guard's MPS check was red before, the empty-output checks
  (review CR2) too.
- `detail::inverse_affine_matrix` (rotate / affine, phase 2) follows torchvision v2's `_get_inverse_affine_matrix`
  (v2's op order; transforms.functional's v1 copy orders b, d and the translation terms differently) and takes sin
  through `detail::py_sin`: clang on Darwin merged each sin / cos pair of one angle into `__sincos_stret`, whose sin
  is 1 ulp off libm's `sin` (Python's `math.sin`) for ~0.2% of angles. Only float64 images showed it (theta stays
  float64; uint8 / float32 cast it to float32). Matrix now bit-exact on 20000 random parameter sets (was 44 off);
  goldens `rot_center_f64`, `aff_shear_f64` were red before (review SP-1).
- `perspective`: grid entries that are inf / NaN (a zero denominator: corners crossing, e.g. RandomPerspective with
  distortion_scale near 1 at any image size, ~1 call in 100 at 1.0 on 224 x 224) are set to -4 (off the image), so
  those pixels take the fill (0 without one). torch's CPU bilinear `grid_sample` dies with SIGBUS on them, as does
  torchvision; its nearest already gives the fill for an inf coordinate (golden `perspective_zero_den_nearest_f32`).
  Finite grids are untouched, so every other output is unchanged (review CR1; the bilinear checks crashed before).

### D7 — Deviations from torchvision (deliberate)
- An empty RandomApply / RandomOrder returns the input (torchvision raises UnboundLocalError when it applies one);
  RandomChoice with p summing to 0 throws `std::invalid_argument` (torchvision: ZeroDivisionError).
- torchvision's `padding=0` (int, None) and `[0]` (no-op padding) differ only in the error text ("padded "); here
  `{0}` is the latter.
- No warnings: torchvision warns for (max, min) scale / ratio ranges, then `uniform_` raises at call time on both sides.
- MPS: hue (progress09 D8) and the symmetric pad (D6).
- A non-finite perspective grid gives the fill (D6) where torchvision crashes (bilinear) or, for a NaN coordinate next
  to an in-range one, reads an implementation-defined pixel (nearest).

### D8 — Tests, goldens, coverage
- `gen_golden.py` gains seeded cases: N calls after `torch.manual_seed(seed)` record every output (dtype, shape,
  values), whether it is the input object, and the next `torch.rand(1)` (pins the draw count); `check_case` replays
  them (odd calls through `forward()`). 74 cases (63 + 11 from the review) over every new class (uint8, float32, float64 for ColorJitter /
  GaussianNoise, batch dims, 2-D input, p = 0 / 1, crop larger than the image with pad_if_needed, every padding form
  and mode, the three RandomResizedCrop fallbacks, degenerate ColorJitter ranges, fill None / scalar / per channel,
  even elastic kernels, strict-fit erasing, in-place erasing, the containers), LinearTransformation goldens, and the
  ImageNet training recipe (Compose of RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter(0.4, 0.4, 0.4),
  ToDtype, Normalize) for seeds 0, 1, 2 on the synthetic 375 x 500 image (14 x 14 grid + sums, next draw).
- Review additions: a batched case for ScaleJitter, RandomPhotometricDistort, ElasticTransform, RandomZoomOut,
  GaussianNoise, RandomApply and RandomOrder (float32); float64 bilinear RandomRotation (center) / RandomAffine
  (shear); RandomRotation(0) (draws); RandomPerspective(0.6) on 1 x 7 x 6 (`h // 2`); `lt_f64`; the inclusive
  constructor bounds (translate 1, distortion_scale 1, erasing scale 0); RandomPerspective(1.0) at the seed where
  torchvision crashes.
- `vision_test_transform` 987 checks (180 before phase 3), `vision_test_functional` 506 (467), `vision_test_io` 136.
- Coverage: 100% lines for every `_vision/*.hpp` (D9 numbers).
- Mutation check (scratch `vision3/impl/mutants/run.py`, Release): 28 mutants of draw order, draw count, dtype and
  rounding; 25 fail the tests. The 3 survivors compute an intermediate in float64 instead of float32 (RandomResizedCrop's
  log ratio, RandomZoomOut's canvas, RandomChoice's probabilities): the 35388-output sweep cannot tell them apart either
  (results differ only within ~1e-7 of a rounding boundary); the code keeps torchvision's dtypes. The review's 7
  surviving mutants (scratch `vision3/review-conventions-robustness/mut`) prompted the review additions above; the 6
  re-applied here (perspective float halving, RandomRotation no draw for a zero range, LinearTransformation without
  the dtype cast, the three tightened constructor bounds) all fail them.

### D9 — Measured parity (dev sweep, scratch `vision3/impl/sweep`, not in ctest)
- CPU (`sweep_gen.py 4 3`): 14378 cases (class x params x image x seed; images uint8 / float32 / float64 from 1 x 7 x 6
  to 3 x 32 x 40, batch and 5-D; 2-4 seeds; 2-6 calls each) = 35388 outputs: 35388 bit-exact (max |delta| 0), every
  "is the input" flag and every next draw equal; 1220 cases raise on both sides at the same call, with the mapping
  ValueError -> `std::invalid_argument`, RuntimeError / IndexError -> `c10::Error`; 0 mismatches.
- MPS float32 (`sweep_gen.py 2 2 ... mps`): 5238 cases, 8156 outputs, 7996 exact, 1508 raise on both sides
  (incl. torch's missing MPS bicubic / antialias kernels). Mismatches: the 64 hue-configured ColorJitter /
  RandomPhotometricDistort cases (torchvision's MPS adjust_hue bug, progress09 D8: off by up to 1.0; ours is within
  2.8e-6 of torchvision's CPU output on the same cases) and 28 RandomCrop symmetric pad_if_needed cases (D6).
- The sweep missed the float64 matrix ulp (D6, SP-1): its float64 rotations / affines did not hit one of the ~0.2%
  angles. The review's probe (17520 + 11300 + 106500 runs) found 23 such float64 mismatches and nothing else.
- `torch::manual_seed` seeds the MPS generator too (checked: MPS `randn` equals Python's after the same seed), so
  GaussianNoise on MPS equals torchvision on MPS.

## Tasks
- [x] [task70 — random geometry classes](tasks/task70-vision-random-geometry.md)
- [x] [task71 — color / misc classes, containers, functional ops, header split](tasks/task71-vision-random-color-misc.md)
- [x] [task72 — seeded goldens, ImageNet training recipe, coverage, parity sweep](tasks/task72-vision-random-goldens.md)

## Issues / Gotchas
- torch 2.5.1 `grid_sample` (CPU, bilinear) dies with SIGBUS on the inf / NaN grid of a degenerate perspective, at
  any image size once distortion_scale is near 1 (review CR1: 224 x 224 at 1.0, seed 25, second call; 1 x 7 x 6 at
  0.8, seed 0). torchvision crashes the same way; `functional::perspective` now moves non-finite grid entries off the
  image (D6). The sweep had skipped distortion >= 0.8 on 1 x 7 x 6 and so understated it.
- Darwin clang merges `std::sin(x)` / `std::cos(x)` of one x into `__sincos_stret` (no errno on Darwin), 1 ulp off
  libm's `sin` for some x. Python-parity scalar trig takes sin through `detail::py_sin` (D6).
- torchvision has two `_get_inverse_affine_matrix`: transforms.functional (v1) and transforms.v2.functional (v2), with
  different op orders (bit-different for a center or a shear). The v2 classes use v2's; compare against that one.
- Python `int(t)` of a float tensor truncates its float32 value; `t < p` compares in float32 (a double comparison
  differs when float32(p) != p). Tensor math in torchvision's `_get_params` must stay tensor math.
- `uniform_(from, to)` on a float32 tensor casts from / to to float32 first, which is why the float64 log-ratio
  mutant is practically equivalent.
- A coverage trap: any statement after a throwing `detail::fail(...)` in the same block never runs in an exceptions
  build; put no-exceptions fallbacks before the throw or return `handle_exceptions<T, E>(fallback, msg)`.
- `Tensor::cpu()` on a CPU tensor returns the tensor itself: clone outputs before an in-place transform mutates them
  (the sweep's RandomErasing(inplace) cases).

## Open / TODO (carry-over)
- AutoAugment family (AutoAugment, RandAugment, TrivialAugmentWide, AugMix) — next phase; MixUp / CutMix, JPEG.
- Bounding boxes / masks / videos semantics (RandomIoUCrop, SanitizeBoundingBoxes, the box / mask kernels), PIL.
- `_audio/_functional_filtering.hpp` has scalar sin / cos pairs of one angle (biquad coefficients); clang may merge
  them the same way on Darwin. Not checked against torchaudio here (audio is out of this phase's scope).
- Parity off arm64 (x86 AVX2 uint8 resize, `normal_` AVX2 path) is not measured; the tests allow 1 level / 1e-5 there.
- `readme.md` still says vision is "coming soon" (progress07 R4) — Mux's call.

## Agent log
- 2026-10-23 [ai] Read torchvision 0.20.1 `_geometry.py`, `_color.py`, `_augment.py`, `_misc.py`, `_container.py`,
  `_transform.py`, `_utils.py` and the v1 setup helpers; implemented D1-D6 (header split, 19 classes + 4 containers,
  3 functional ops, the symmetric pad guard). Tests: transform 889, functional 498, io 136 checks; full build + ctest
  green (10/10). Coverage (`llvm-cov report`, per binary): `_functional.hpp` 1016/1016, `_transform_misc.hpp` 285/285,
  `_transform_geometry.hpp` 570/570, `_transform_color.hpp` 193/193, `_io.hpp` 203/203 lines (the union script over
  the three vision binaries agrees: 100% of every `_vision/*.hpp`).
  Parity: D9. Mutants: D8. git clang-format applied. Not committed.
- 2026-10-23 [ai] Review round (seeded-parity and conventions-robustness reviews). Fixed with a test that failed before:
  SP-1 (float64 rotate / affine matrix 1 ulp off: Darwin sincos merge; `detail::py_sin`; goldens `rot_center_f64`,
  `aff_shear_f64` red before; matrix 20000 / 20000 bit-exact vs v2 after, 44 off before), CR1 (non-finite perspective
  grid -> fill; the functional zero-denominator checks and RandomPerspective(1.0) seed 13 died with SIGBUS before),
  CR2 (symmetric pad guard only for a non-empty output; empty batch / zero channels threw before). Test gaps closed:
  CR3 (`persp_odd_f32`, `rot_zero_u8`, `lt_f64`), CR4 (inclusive constructor bounds), CR5 (batched / float cases);
  the 6 re-applied mutants all fail. Docs: CR6 (threads: D1, `_transform_misc.hpp`, CLAUDE.md), CR7
  (`detail::p_gated`, alias comment, D4), CR8 + SP-2 (D5 error mapping; to_dtype's invalid_argument kept: the phase-1
  functional convention maps torchvision's own RuntimeError there). Not taken: SP-1's suggested v1 op order (the
  reviewer's reference was transforms.functional's `_get_inverse_affine_matrix`; the v2 classes use v2's, which the
  code already followed: with that order 2486 / 5000 centered rotations would have broken). Existing goldens
  regenerate byte-identical. Full Release build + ctest 10/10 green; transform 987, functional 506, io 136 checks.
  Coverage (Release + TORCHMEDIA_COVERAGE, scratch `vision3/build-cov`, per owning binary): `_functional.hpp`
  1025/1025, `_transform_misc.hpp` 285/285, `_transform_geometry.hpp` 570/570, `_transform_color.hpp` 193/193,
  `_io.hpp` 203/203 lines. clang-format clean on every changed file. Not committed.
