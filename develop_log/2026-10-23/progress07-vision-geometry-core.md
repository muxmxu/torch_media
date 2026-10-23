# Progress 07 — Vision geometry core: torchvision.transforms.v2 parity (functional + transform classes)
id: 2026-10-23/progress07
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-05-31/progress03, 2026-05-31/progress04, 2026-06-01/progress02, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_functional.hpp
  - libtorchmedia/include/torchmedia/_vision/_transform.hpp (renamed from the empty `_vision/_transform`)
  - libtorchmedia/include/torchmedia/vision.hpp
  - unit_test/vision/functional/{main.cpp, gen_golden.py, golden.hpp (NEW, generated)}
  - unit_test/vision/transform/{main.cpp, gen_golden.py, golden.hpp (generated), CMakeLists.txt} (NEW)
  - unit_test/vision/CMakeLists.txt, CLAUDE.md

## Goal
Make the vision module usable for C++ inference-time preprocessing: the torchvision 0.20.1 `transforms.v2.functional`
tensor kernels for resize / crop / center_crop / pad / resized_crop / five_crop / ten_crop / to_dtype, a re-audit of
the 7 initial ops (2026-05-31/progress03), and deterministic transform classes, all point-wise identical to
torchvision. Out of scope: random crops / color jitter / affine family, PIL semantics, bounding boxes and masks.

## Context / Motivation
The vision side had 7 float-only ops and image I/O. The standard classification recipe
`Resize(256) + CenterCrop(224) + ToDtype(float32, scale=True) + Normalize(mean, std)` could not be written, and the
existing ops diverged from torchvision on uint8 (and partly on float). The design below was set by the orchestrator
under Mux's delegation; the conventions are those of the audio transform layer (2026-06-01/progress02 D1, D2, D5) and
the audio audit (2026-10-23/progress02 D2-D11).

## Decisions

### D1 — Reference and parity bar
- Decision: mirror the torchvision 0.20.1 `transforms/v2/functional/*_image` kernels op for op on `[..., C, H, W]`
  tensors (uint8 and floating, any leading dims), calling the same libtorch ops (`at::upsample_*` /
  `at::_upsample_*_aa` exactly as `F.interpolate` dispatches them, `torch::pad`, slicing, index gathers). Invalid
  arguments throw `std::invalid_argument` through `handle_exceptions` (torchvision: ValueError / TypeError /
  RuntimeError / ZeroDivisionError); what torchvision leaves to torch (zero output size, a reflect pad wider than the
  image, a fill of the wrong length) stays a `c10::Error`.
- Why: inference preprocessing must reproduce the training pipeline; calling the same kernels makes the results
  bit-identical instead of merely close.
- Impact: results follow torch's kernels, including their platform dependence (D3).

### D2 — Functional API (positional mirror of torchvision)
- `enum class interpolation_mode { nearest, nearest_exact, bilinear, bicubic }`,
  `enum class padding_mode { constant, edge, reflect, symmetric }`.
- `resize(img, size, interpolation = bilinear, max_size = nullopt, antialias = true)`: `size` is `{h, w}`,
  `{shorter edge}` or `{}` (torchvision v2 `None`, longer edge = max_size). Same-size returns `img` itself.
- `crop(img, top, left, height, width)` zero-pads the out-of-bounds part; `center_crop(img, output_size)`
  (`{s}` or `{h, w}`) pads an image smaller than the crop and anchors at Python `round((H - h) / 2)` (half to even).
- `pad(img, padding, fill = {}, mode = constant)`: padding `{all}`, `{lr, tb}` or `{l, t, r, b}`; fill `{}` (0),
  one value, or one per channel (constant mode only).
- `resized_crop(img, top, left, height, width, size, interpolation, antialias)`; `five_crop(img, size)` and
  `ten_crop(img, size, vertical_flip)` return `std::vector<tensor_t>` in torchvision order (tl, tr, bl, br, center,
  then the same for the flipped image).
- `to_dtype(img, dtype = float32, scale = false)`: torchvision's rules (int -> float `* 1 / max`, float -> int
  `* (max + 1 - 1e-3)` then truncate without clamping, int -> int bit shifts, uint16 through division / multiply;
  float32 -> int32/int64 and float64 -> int64 rejected). The int -> float scale is an out-of-place `mul` (D8).
- `normalize` gains torchvision's `inplace`. **Breaking:** `center_crop(img, h, w)` became
  `center_crop(img, {h, w})` (the only caller was the old test).
- Call `resize` / `pad` qualified (`functional::resize`): an unqualified call also finds `at::resize` / `at::pad` by
  ADL and is ambiguous with a braced size.

### D3 — uint8 resize dispatch exactly as torchvision
- Decision: uint8 is interpolated natively for nearest / nearest_exact, for bicubic on CPU, and for bilinear on CPU
  only when `at::get_cpu_capability()` contains "AVX2" (torchvision `_do_native_uint8_resize_on_cpu`); otherwise
  through float32 and `round_()` back (bicubic clamped to [0, 255] first). Other integer dtypes always go through
  float32 and are rounded; float16 / bfloat16 go through float32 without rounding.
- Why: the native uint8 and float paths differ by up to one level, so matching torchvision on a given machine means
  taking the same path it takes there.
- Impact: this arm64 machine reports "NO AVX", so bilinear uint8 goes through float32 here. An x86 AVX2 machine takes
  torch's native uint8 kernel, as torchvision does there; the tests allow one uint8 level off aarch64 (D6).

### D4 — Re-audit of the 7 initial ops (fixed)
- `invert`: was `1.0 - img` for every dtype (uint8 returned a float tensor of `1 - x`); now `1 - x` for floating,
  `~x` for uint8, `x ^ max` for signed integers (bool rejected).
- `adjust_brightness`: clamped every dtype to [0, 1] (uint8 came back as float32 in [0, 1]); now clamps to the dtype
  maximum and casts integer results back (truncation); validates factor >= 0 and 1 or 3 channels.
- `rgb_to_grayscale`: different op order (`0.2989*r + 0.587*g + 0.114*b` is a few ulps off torchvision's
  `r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)` in float32 and float64) and uint8 returned float; now
  torchvision's chain, cast back to the input dtype, 1-channel input cloned / repeated, `num_output_channels = 3`
  returned as an expanded view, validation of channels and num_output_channels. Integer inputs enter the chain as
  float32 (D8).
- `normalize`: mean / std went through float32 even for float64 images; now converted straight to the image dtype;
  validation (floating input, >= 3 dims, no zero std).
- `center_crop`: floor anchor (`(H - h) / 2`) instead of half-to-even rounding (a 2 x 2 crop of a 5 x 7 image started one
  row too high) and no padding for oversize crops; now torchvision's kernel.
- `hflip` / `vflip`: already identical.

### D5 — Transform layer
- Decision: `_vision/_transform.hpp`, namespace `torchmedia::vision::transform`, included from `vision.hpp`. Classes
  `Resize`, `CenterCrop`, `Pad`, `Normalize`, `ToDtype`, `Grayscale`, `RandomHorizontalFlip`, `RandomVerticalFlip`,
  `FiveCrop`, `TenCrop`; option structs `resize_option`, `center_crop_option`, `pad_option`, `normalize_option`,
  `to_dtype_option`, `grayscale_option`, `random_flip_option` (shared by both flips), `five_crop_option`,
  `ten_crop_option` with `_field` members and same-name fluent setters; `operator()` + `forward()`. Constructors
  validate like the torchvision classes (size 1 or 2 values / None with max_size, padding 1, 2 or 4 values,
  p in [0, 1]); everything else is checked by the functional op at call time, as in torchvision. No `Crop` (torchvision
  v2 has none).
- The flips draw `torch::rand({1})` from the global CPU generator once per call and flip unless `draw >= p`
  (torchvision v2 `_RandomApplyTransform`), so `torch::manual_seed` reproduces torchvision's flips; a non-flip
  returns the input tensor itself.
- The classes hold no tensors (Normalize builds mean / std per call like torchvision), so outputs follow the input's
  device and dtype without a cache.

### D6 — Tests, goldens and parity bar in ctest
- Decision: `gen_golden.py` writes `golden.hpp` (seeded inputs plus every expected output; float32 printed with 9
  significant digits and float64 with `repr`, both exact round trips; each entry carries its dtype). The tests
  compare dtype, shape and values with zero tolerance on aarch64 (where the goldens were generated) and allow one
  integer level / 1e-5 elsewhere (D3). The ImageNet recipe runs on a synthetic 375 x 500 image built from integer ops
  in both languages (input checksums asserted); its output is pinned by a 3 x 14 x 14 pixel grid plus the float64 sum
  and sum of squares over all pixels.
- Coverage: 100% line coverage for every `_vision/*.hpp` except `_vendor/` (CLAUDE.md updated). MPS has no bicubic /
  antialias kernels in torch 2.5.1, so the non-CPU uint8 bicubic clamp line runs on the `meta` device (shape / dtype
  checked) — covered but not value-asserted (no device here runs it); MPS bilinear (u8 and f32) is checked against
  torchvision-on-MPS goldens, guarded by `GOLDEN_HAS_MPS`, which `gen_golden.py` writes into `golden.hpp` so a
  golden regenerated on a host without MPS still compiles.

### D7 — Deviations from torchvision (deliberate)
- `resize` rejects a `size` with more than 2 values (torchvision's functional silently uses the first two; its
  `Resize` class rejects them).
- A symmetric pad whose negative padding crops an edge to zero length returns the empty image; torchvision raises
  IndexError there (its empty index list becomes a float tensor). The only mismatch in the 2348-case sweep (G1).
- `fill` and `brightness_factor` are doubles: a vector fill goes through float32 like torchvision's float list (a
  Python int list would go through int64: same values for in-range integers); torchvision with a Python *int*
  brightness factor multiplies in the integer dtype and wraps, this always computes in floating point and clamps.
- torchvision's channels_last restride for single-image batches is skipped (`shortcut:` comment in `resize`): it only
  preserves the output memory format, the values are equal.

### D8 — Review round (two reviewers, 6 findings; implementer's choices, awaiting Mux's review)
- `to_dtype` int -> float16 / bfloat16 with scale (P1 / R1, medium): 1-ulp errors (u8 -> f16 / bf16) and NaN instead
  of inf (int32 / int64 -> f16) because libtorch's in-place `Tensor::mul_(Scalar)` rounds `1 / max` to the reduced
  dtype first; Python's `mul_(float)` keeps it at float opmath. Fixed with the out-of-place `mul` (same path as
  Python); goldens u8 -> f16, u8 -> bf16 and int32 -> f16 added (red before). `ToDtype` calls the functional op.
- `rgb_to_grayscale` on a uint8 MPS tensor (P2, low): torch 2.5.1's MPSGraph aborts the process on
  `mul(uint8, float scalar)` (torchvision aborts the same way). Integer inputs are now cast to float32 before the
  chain — torch's own promotion, so CPU values are unchanged (`gray_u8` golden still bit-exact) and MPS uint8 now
  equals the CPU golden (red before: SIGABRT).
- Golden regeneration without MPS (R2, low): `GOLDEN_HAS_MPS` guard (D6); a no-MPS regeneration (97 entries)
  compiles (`-fsyntax-only`).
- Resize rounding for int8 / int32 (R3, low): torchvision goldens `resize_bicubic_noaa_up_{i8,i32}` added; mutants
  dropping `kInt8` / `kInt32` from the `round_()` list now fail. The non-CPU uint8 bicubic clamp stays covered but not
  value-asserted (D6).
- README "vision processing is coming soon!" (R4, low): left for Mux (README wording is his call; Open / TODO).
- Why: each was reproduced in scratch before the fix; see task42.

## Tasks
- [x] [task40 — vision functional geometry core + re-audit of the initial ops](tasks/task40-vision-functional-geometry.md)
- [x] [task41 — vision transform classes + ImageNet recipe golden](tasks/task41-vision-transform-layer.md)
- [x] [task42 — review-round fixes (to_dtype half / bfloat16, MPS grayscale, golden guards)](tasks/task42-vision-review-fixes.md)

## Issues / Gotchas
- G1 — Parity sweep (dev-time, not in ctest): 2348 cases through a scratch C++ probe vs torchvision on the full
  outputs — resize 1324 (4 modes x antialias on/off x uint8 / float32 / float64 x 10 shapes from 2 x 3 to 375 x 500,
  1- and 2-value sizes, max_size, None, non-integer up / down / mixed factors, batch and 5-D, int16), crop / center_crop /
  resized_crop 120 each, pad 119, to_dtype 280 (7 target dtypes, scale on / off), five / ten crop, flips, grayscale,
  brightness, invert, normalize, and both ImageNet recipes on a synthetic and a random 480 x 640 image. 2347 bit-exact
  (max |delta| 0), 2 cases raise on both sides, 1 mismatch (D7 symmetric-pad IndexError). The 7 to_dtype targets did
  not include float16 / bfloat16; the review sweep (G2) found the mismatch there.
- G2 — Review parity sweep (reviewer's probe, 1447 cases incl. MPS, channels_last / strided views, CLIP and ImageNet
  recipes): 1353 bit-exact, 86 raise on both sides, 8 mismatch = 7 x to_dtype int -> f16 / bf16 (fixed, D8) + the D7
  size-with-3-values deviation. Resize timing equals torchvision within noise (same at:: kernels).
- libtorch `Tensor::mul_(Scalar)` on a float16 / bfloat16 tensor rounds the scalar to that dtype before multiplying;
  `mul(Scalar)` (and Python's `mul_(float)`) compute with the scalar at float opmath. Prefer the out-of-place form
  (or a float64 `scalar_tensor`) for reduced-precision outputs.
- `torch::tensor(std::vector<double>)` without a dtype gives float32 (the default dtype), like Python's
  `torch.tensor([...floats])`; `normalize` passes kFloat64 explicitly and converts to the image dtype.
- 9-digit float32 goldens compare exactly only after rounding them to float32 again (the decimal itself is not the
  float32 value).
- GNU Make 3.81 compares mtimes at one-second resolution: a header restored within the same second as the last
  compile is not rebuilt (hit while mutation-testing the goldens).
- `llvm-cov export -format=lcov` reports comment lines inside executed regions with count 0; `llvm-cov report` (the
  project's measure) counts them correctly.
- Old-vs-new mutation checks: the floor anchor, a missing uint8 `round_()`, the old grayscale op order and the old
  uint8 invert are each caught by the goldens.

## Open / TODO (carry-over)
- Random geometry / color (RandomResizedCrop, RandomCrop, ColorJitter, rotate / affine / perspective, gaussian_blur),
  a Compose helper.
- `io::load_image` returns float32 in [0, 1]; torchvision's `decode_image` returns uint8, so the uint8 recipe path
  needs a uint8 load mode to be reachable from a file.
- `readme.md` line 3 still says "vision processing is coming soon!" (review R4) — Mux's wording call.
- Parity on an x86 AVX2 machine is not measured (no such machine here); the tests allow one uint8 level there.

## Agent log
- 2026-10-23 [ai] Read the torchvision 0.20.1 v2 kernels and classes; implemented D2-D5 in `_vision/_functional.hpp`
  and `_vision/_transform.hpp` (renamed from the empty `_transform`); wired `vision.hpp`. New goldens and tests:
  `vision_test_functional` 245 checks, new `vision_test_transform` 72 checks, all bit-exact on arm64; parity sweep G1.
  Full build + ctest green (9/9). `llvm-cov report`: `_functional.hpp` 439/439, `_transform.hpp` 149/149, `_io.hpp`
  38/38 lines. git clang-format applied. Not committed (orchestrator commits).
- 2026-10-23 [ai] Review round (D8 / task42): re-checked the 6 findings; fixed P1/R1 (to_dtype f16 / bf16), P2 (MPS
  uint8 grayscale), R2 (GOLDEN_HAS_MPS), R3 (int8 / int32 resize goldens), each red before / green after; R4 left for
  Mux. `vision_test_functional` 258 checks (was 245), `vision_test_transform` 72, `vision_test_io` 10; full build +
  ctest green (9/9). `llvm-cov report` per binary (union, 2026-10-23/progress02 D11): `_functional.hpp` 439/439,
  `_transform.hpp` 149/149, `_io.hpp` 38/38 lines. clang-format clean. Not committed.
