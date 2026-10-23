# Progress 09 — Vision color ops, rotate / affine / perspective, torchvision.io-style decoding
id: 2026-10-23/progress09
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress07, 2026-10-23/progress02, 2026-10-23/progress05, 2026-05-31/progress04]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_functional.hpp
  - libtorchmedia/include/torchmedia/_vision/_transform.hpp
  - libtorchmedia/include/torchmedia/_vision/_io.hpp
  - libtorchmedia/include/torchmedia/_vision/_vendor/stb_image.h (one local patch, D6)
  - unit_test/vision/functional/{main.cpp, gen_golden.py, golden.hpp}
  - unit_test/vision/transform/{main.cpp, gen_golden.py, golden.hpp}
  - unit_test/vision/io/{main.cpp, gen_golden.py (NEW), golden.hpp (NEW, generated)}
  - THIRD_PARTY_LICENSES, CLAUDE.md

## Goal
Phase 2 of the vision module: the remaining torchvision 0.20.1 `transforms.v2.functional` color kernels, the
grid-sampled geometry kernels that inference needs (face alignment, deskew), uint8 image decoding like
`torchvision.io.decode_image`, and the deterministic (seeded-parity) transform classes over them. Out of scope: the
random geometric / color families (carry-over), PIL semantics, bounding boxes / masks, video.

## Context / Motivation
Phase 1 (progress07) left rotate / affine / perspective, the color ops and a uint8 load path as carry-over, so a
torchvision `decode_image -> rotate / affine -> ...` pipeline could not be reproduced in C++. The design (op list,
io scope, which classes) was set by the orchestrator under Mux's delegation; conventions are progress07's D1-D6.

## Decisions

### D1 — Parity bar and conventions (unchanged from progress07)
- Decision: every op mirrors the torchvision 0.20.1 v2 `*_image` kernel on `[..., C, H, W]` (uint8 and floating, any
  leading dims) with the same libtorch ops in the same order (in-place sequences included), so float results are
  bit-identical and uint8 rounding / truncation matches. Invalid arguments throw `std::invalid_argument`; what
  torchvision leaves to torch (a reflect pad wider than the image, a fill that does not broadcast in `elastic`) stays
  a `c10::Error`. Goldens: `gen_golden.py -> golden.hpp`, exact on arm64, 1 level / 1e-5 elsewhere.
- Why: inference preprocessing must reproduce training; same kernels make it exact.

### D2 — Functional color API (`torchmedia::vision::functional`)
- `adjust_contrast(img, factor)`, `adjust_saturation(img, factor)`, `adjust_sharpness(img, factor)`,
  `adjust_hue(img, hue_factor)` (with torchvision's `_rgb_to_hsv` / `_hsv_to_rgb` as `detail::`),
  `adjust_gamma(img, gamma, gain = 1)`, `posterize(img, bits)`, `solarize(img, threshold)`, `autocontrast(img)`,
  `equalize(img)`, `gaussian_blur(img, kernel_size, sigma = {})`, `erase(img, i, j, h, w, v, inplace = false)`.
- The luma shared by grayscale / contrast / saturation is `detail::luma` (integers enter as float32 first, phase 1's
  MPS fix); `detail::blend` is torchvision's `_blend`. `rgb_to_grayscale` and `adjust_brightness` now use the shared
  helpers (values unchanged, goldens still bit-exact).
- `gaussian_blur`: `kernel_size` `{k}` is torchvision's int k, `{kx, ky}` its sequence; `sigma` `{}` = None
  (`k * 0.15 + 0.35`), `{s}` or `{sx, sy}`; a sigma `<= 0` throws, a NaN sigma passes like torchvision's `s <= 0.0`
  check (NaN output).

### D3 — Functional geometry API
- `rotate(img, angle, interpolation = nearest, expand = false, center = {}, fill = {})`,
  `affine(img, angle, translate, scale, shear, interpolation = nearest, fill = {}, center = {})`,
  `perspective(img, startpoints, endpoints, interpolation = bilinear, fill = {}, coefficients = {})`,
  `elastic(img, displacement, interpolation = bilinear, fill = {})` (elastic was cheap: the identity grid plus the
  shared grid transform).
- `fill`: `{}` is torchvision's `None` (zero padding, no mask channel), `{v}` a scalar, or one value per channel; a
  non-empty fill goes through torchvision's sampled ones-mask blend (`{0}` therefore differs from `{}` for bilinear,
  as in torchvision). `center`: `{}` or `{x, y}` pixels. `shear`: `{s}` is a 1-element list (`{s, s}`); torchvision's
  scalar `s` is `{s, 0}`. Points are `std::array<double, 2>`: integral coordinates are solved like Python ints
  (exact rows), a row with a fractional coordinate is rounded to float32 like torchvision's Python-float list.
- Shared `detail::` pieces mirror torchvision: `inverse_affine_matrix`, `affine_output_size` (float32 corner math),
  `affine_grid` (float32 base linspaces even for float64 theta), `apply_grid_transform` (grid_sampler, mask fill,
  round back), `grid_error` (`_assert_grid_transform_inputs` + rank + center). `rotate` keeps torchvision's exact
  fast paths (0 / 180, and 90 / 270 when square or expand; `center` disables them).

### D4 — Python-order scalar math: no FMA contraction
- Decision: functions doing torchvision's Python-float scalar arithmetic (`inverse_affine_matrix`, the default
  `gaussian_blur` sigma) start with `TORCHMEDIA_FP_CONTRACT_OFF` (`_Pragma("clang fp contract(off)")` on clang; GCC in
  ISO mode and MSVC do not contract by default). The libpng gamma table splits `max * pow + .5` into two statements.
- Why: clang contracts `a * b + c` into an FMA by default; measured without the pragma: 186 of 4676 rotate / affine /
  blur cases differ (affine matrices by up to 4.7e-14, and through nearest sampling whole pixels: max 0.84 in float).
- Impact: consumers must not build with `-ffast-math` / `-ffp-contract=fast` (they disregard the pragma; same 186
  mismatches measured). GCC in GNU mode on an FMA target (`-std=gnu++23` on aarch64 Linux) contracts too.

### D5 — `decode_image` / `read_image` / `read_file` (`torchmedia::vision::io`)
- Decision: `enum class image_read_mode { unchanged, gray, gray_alpha, rgb, rgb_alpha }`;
  `decode_image(data, mode = unchanged)` takes encoded bytes (1-D uint8 CPU tensor) and returns `[C, H, W]` uint8
  (uint16 for 16-bit PNG), `read_file(path)` the bytes, `read_image(path, mode)` both. JPEG and PNG only, detected
  from the signature; torchvision 0.20.1 also decodes GIF and WebP, which raise `std::runtime_error` here
  (carry-over). JPEG supports unchanged / gray / rgb (torchvision raises for the alpha modes). The float
  `load_image` / `save_image` API is unchanged.
- A JPEG must reach its EOI marker (marker segments skipped by length, then the entropy-coded data scanned):
  truncated data raises `std::runtime_error` ("image is incomplete or truncated") as torchvision does when
  libjpeg-turbo runs out of input; stb_image alone would pad the missing tail. Bytes after the EOI are ignored.
- PNG decodes natively through stb_image, then reproduces libpng's transforms as torchvision configures them:
  gray = `(9794 R + 19234 G + 3740 B) >> 15` (`png_set_rgb_to_gray(png, 1, 0.2989, 0.587)` in 15-bit fixed point,
  truncating for 8-bit, rounding for 16-bit); with a significant file gamma (gAMA chunk, or sRGB = 45455) the
  conversion runs in linear light through libpng's gamma tables (`png_build_8bit_table` / `png_build_16bit_table`,
  gray pixels unchanged); the file gamma follows libpng's colour-space rules (`png_colorspace_set_gamma` /
  `_sRGB`): a gAMA / sRGB chunk with a bad length or after PLTE is skipped, an out-of-range or second gAMA, a second
  sRGB or an sRGB intent >= 4 invalidates the colour space (the gamma stored so far stays, later gAMA / sRGB chunks
  are skipped), and sRGB wins over gAMA in either order; a tRNS chunk on a gray / RGB image is ignored (stb's alpha
  dropped); the added alpha is torchvision's filler `(1 << file bit depth) - 1` (so a 1 / 2 / 4-bit gray file gets
  alpha 1 / 3 / 15, a torchvision quirk kept for parity).
- Argument errors (`dtype`, rank, empty, alpha mode for JPEG) throw `std::invalid_argument`; undecodable or
  truncated data, unsupported formats and unreadable files throw `std::runtime_error` (as `load_image` does);
  `read_file` also throws on a directory or an empty file (torchvision: "unable to mmap" / "Expected a non empty
  file").
- Deviations: a palette PNG in unchanged mode returns its colours (RGB / RGBA with tRNS); torchvision returns the raw
  palette indices, which stb_image cannot produce. Palette PNGs without tRNS in gray_alpha / rgb_alpha: torchvision
  0.20.1 writes the packed 3-byte RGB (or 1-byte gray) pixels into the 4- (2-) channel buffer, so every channel is
  scrambled (colour channels off by up to 188 levels at 480 x 640) and the tail is uninitialised memory (the same on
  repeated calls for some files, not for others); here the colours are kept and alpha is 255. Arithmetic-coded JPEGs
  raise (stb_image has no arithmetic decoder; torchvision decodes them). CMYK / Adobe JPEGs: stb converts to RGB, so
  unchanged mode returns 3 channels where torchvision returns 4; rgb mode is within 1 level, gray mode within 2
  (mean <= 0.52). Not handled: `apply_exif_orientation`, the PNG sBIT shift of libpng's 16-bit gamma tables, ICC
  profiles libpng recognises as sRGB.

### D6 — JPEG decoding goes through stb_image (measured, not exact); three stb_image chroma-upsampling patches
- Decision: keep stb_image (no libjpeg-turbo dependency) and document the measured difference to
  `torchvision.io.decode_jpeg` (libjpeg-turbo, ISLOW IDCT, fancy upsampling). Patch stb_image v2.30 where its chroma
  upsampling departs from libjpeg-turbo's `jdsample.c` (each marked `torchmedia patch`, recorded in
  THIRD_PARTY_LICENSES with sha256 before / after): (1) `stbi__resample_row_h_2`: the second-to-last output sample of
  every 4:2:2 (h2v1) chroma row weighted the far sample by 3/4 instead of the near one; (2) the same function rounds
  the left sample of each output pair with +1, not +2 (`h2v1_fancy_upsample`); (3) a 2x horizontally subsampled
  chroma row of <= 2 samples (images <= 4 px wide) is replicated instead of fancy-upsampled (`jinit_upsampler` only
  takes the fancy path for `downsampled_width > 2`). All three are in plain C code that stb has no SIMD variant of, so
  they hold on every platform.
- Why: (1) put one wrong column into every 4:2:2 JPEG (121 levels off in the test golden); (2) left 4:2:2 at mean
  |delta| 0.24 (max 4) against 0.05 for 4:4:4; (3) made images <= 4 px wide decode up to 92 levels off.
- Measured (dev sweep, 255 JPEG files: the 44 PIL files of the first sweep, both reviewers' PIL / cjpeg / ImageMagick
  files and a cjpeg sampling sweep; 1 x 1 to 480 x 640, quality 50-100, baseline / progressive / optimized / restart
  markers / float DCT, every common sampling), RGB / unchanged mode, per file:
  4:4:4 max 3, mean <= 0.053; 4:2:2 (incl. progressive) max 3, mean <= 0.053; 4:1:1 max 3, mean <= 0.03;
  4:2:0 max 3, mean <= 0.122 (<= 0.30 for images 1-2 px tall); images <= 4 px wide max 2; 4:4:0 (h1v2) max 4,
  mean <= 0.38. Over all decodes except arithmetic / CMYK: 94.89% of values exact, 3.64% off by 1, 1.47% by 2,
  0.004% by 3, mean |delta| 0.066 (before patches 2-3: 92.27% exact, mean 0.102; 4:2:2 max 4 / mean <= 0.24; narrow
  images up to 92). Gray mode (Y plane only on both sides) and grayscale JPEGs: max 1, >= 99.4% exact. The ctest
  bound is max 3 (1 for gray) and mean <= 0.06 per file (0.12 for 4:2:0).
- Residual rounding gaps (not patched): libjpeg-turbo's h2v2 fancy upsampling rounds alternately +8 / +7 where stb
  uses +8 (4:2:0) — stb has an x86 SSE2 variant of that kernel, so a C-only patch would split x86 from arm64; for
  4:4:0 (h1v2) libjpeg-turbo alternates +1 / +2, but matching that bias in `stbi__resample_row_v_2` only halved the
  gap (mean 0.37 -> 0.20), so the remainder has another cause (not traced).
- Alternatives considered: vendoring libjpeg-turbo (heavy, not header-only); patching stb's IDCT / colour conversion
  to libjpeg-turbo's integer arithmetic (large vendor diff) — both left for Mux if exact JPEG parity is required.

### D7 — Transform classes
- `GaussianBlur(gaussian_blur_option)`: `_kernel_size` `{k}` / `{kx, ky}`, `_sigma` `{s}` / `{min, max}` (default
  `{0.1, 2.0}`), validated like torchvision; every call draws `torch::empty({1}).uniform_(min, max)` (also when
  min == max) and blurs with `{s, s}`, so a seeded run equals torchvision's.
- `RandomInvert`, `RandomAutocontrast`, `RandomEqualize` (`random_apply_option`, p = 0.5), `RandomGrayscale`
  (p = 0.1, output keeps the channel count), `RandomPosterize` (`random_posterize_option`: bits, p),
  `RandomSolarize` (`random_solarize_option`: threshold, p), `RandomAdjustSharpness`
  (`random_adjust_sharpness_option`: sharpness_factor, p): one `torch::rand({1})` draw per call, the op runs unless
  draw >= p (`_RandomApplyTransform`), a skipped call returns the input tensor. They share
  `detail::RandomApply<Op>`; the flips use the same `detail::skip(p)`, and `random_apply_option` is an alias of
  phase 1's `random_flip_option` (one p-only option struct). torchvision has no defaults for bits / threshold /
  sharpness_factor; the defaults here (8, NaN, 1) leave uint8 images unchanged, but 8 bits quantises a floating image
  to 256 levels.
- ColorJitter, RandomRotation, RandomAffine, RandomPerspective, RandomCrop, RandomResizedCrop, RandomErasing,
  ElasticTransform, RandomPhotometricDistort, RandomChannelPermutation: carry-over.

### D8 — Deviations from torchvision (deliberate)
- `adjust_hue` on MPS: torchvision 0.20.1 copies its `select` table with `non_blocking=True` from a CPU tensor that
  is freed immediately, so on MPS it indexes garbage (wrong by up to 1.0, every call; reproduced in isolation: 20/20
  copies wrong). Here the copy is blocking and MPS equals torchvision's CPU result bit for bit.
- `adjust_contrast` / `adjust_saturation` on a uint8 MPS tensor abort the process in torchvision (MPSGraph
  `mul(uint8, float)`); here they run (contrast equals the CPU golden, saturation within 1 level of CPU on 7 of 40).
- `posterize` rejects bits < 0 (torchvision: ValueError from the shift) and, for floating images, bits > 62
  (torchvision overflows the Scalar at 63); a bool image is rejected (torchvision TypeError).
- `rotate` / `affine` reject a center that is not `{}` or `{x, y}` (torchvision zips silently); every grid op needs
  `[..., C, H, W]` (torchvision fails on unpacking); `rotate`'s fast paths still accept 2-D input like torchvision.
- `gaussian_blur` with `{k}`: torchvision's functional rejects a 1-element list (only an int); `{k}` is the int form.
- Decode deviations: D5.

### D9 — Tests, goldens, coverage
- `vision_test_functional` 467 checks (258 before phase 2): every new op on uint8 / float32 (float64 where the
  kernel has a float64 path; float16 / bfloat16 for adjust_gamma's gain), 3-D / batched / 5-D, odd sizes (5 x 7,
  11 x 13), every branch (1-channel returns, constant channels, step == 0 equalize, fast-path rotations incl. a
  float64 bilinear exactness check, expand, center, all fill kinds, nearest / bilinear / bicubic elastic with and
  without fill, CPU float16 elastic, non-dyadic fractional perspective points on float32 and float64), validation,
  and MPS goldens from torchvision-on-MPS (blur, sharpness, rotate, perspective, equalize u8) plus MPS hue / u8
  contrast against the CPU goldens (D8).
- `vision_test_transform` 180 checks (72): GaussianBlur seeded outputs and draw count, each p-gated class's 12-call
  sequence and next draw equal torchvision's, applied calls equal the functional op.
- `vision_test_io` 136 checks (10 before phase 2): 22 small PNGs (gray, gray+alpha, RGB, RGBA, 1-bit, palette +/-
  tRNS, RGB+tRNS, gAMA 1/2.2/0.97, sRGB, gAMA 1.0, 16-bit RGB / gray, 16-bit + gAMA, and the libpng colour-space
  cases of D5: duplicate gAMA, sRGB then gAMA, bad-length sRGB, sRGB intent 7, out-of-range gAMA then sRGB, gAMA
  after PLTE) bit-exact in the torchvision-sane modes; 6 JPEGs (4:4:4 / 4:2:2 / 4:2:0 / gray / 3 px wide / restart
  markers) within D6's per-file bounds; truncated, fill-byte and trailing-byte JPEGs; read_file / read_image incl. a
  directory and an empty file; every error path.
- Coverage (`llvm-cov report` per binary, progress02 D11): `_functional.hpp` 981/981, `_transform.hpp` 225/225,
  `_io.hpp` 203/203 lines.

### D10 — Review round (two reviewers, 12 findings; fixer's choices, awaiting Mux's review)
- Truncated JPEG accepted (P1, medium): stb_image pads a cut-off stream, so a partial download decoded to a
  full-size image with filler rows; torchvision raises. Fixed with the EOI walk of D5 (red before: no-EOI and
  missing-last-byte JPEGs decoded; the reviewer's 60% file too).
- `adjust_gamma` gain on float16 / bfloat16 (R1, medium): the in-place `mul_(gain)` rounded the gain to the image
  dtype (progress07 D8's libtorch gotcha); now out of place. Goldens `gamma_gain_{f16, bf16}` (red before: 4.9e-4 /
  3.9e-3).
- `read_file` returned uninitialised memory for a directory (R3, medium): the read result is now checked and an empty
  file rejected, both raising `std::runtime_error` like torchvision (red before: both returned).
- JPEG bound did not hold beyond the first sweep (R2 medium, P5 low): re-measured on 255 files (D6). Two more stb
  patches (h2v1 +1 rounding, replication for chroma <= 2 samples wide) fix the cause for 4:2:2 (max 4 -> 3, mean 0.24
  -> 0.05) and narrow images (92 -> 2 levels); goldens `jpeg_narrow` (61 before) and a 0.06 per-file mean bound that
  fails without the rounding patch (0.167). 4:4:0 (max 4, mean 0.38), 4:2:0 on 1-2 px tall images (mean 0.30) and
  CMYK in gray mode (max 2) are documented in D5 / D6; the test comment's "160 JPEGs" is gone.
- PNG gamma chunks (R5, low): duplicate gAMA, bad-length sRGB and, found while checking it, an sRGB intent >= 4, an
  out-of-range gAMA followed by sRGB and gAMA / sRGB after PLTE all diverged from libpng (max 51-58 levels in gray
  mode). `read_png_header` now follows libpng's colour-space rules (D5); 7 goldens, 5 red before.
- Untested branches (R4, low): goldens `posterize_white_f32` (1.0 hits the clamp), `perspective_frac_{f32, f64}`,
  `blur_default_sigma_f64`, `elastic_bicubic_fill_f32`, PNG sRGB-then-gAMA / gAMA 0.97 / out-of-range gAMA, and
  float64 bilinear checks of the rotate fast paths. 22 mutants (the reviewer's survivors F3, F8, F17, F18, F22b, F24,
  F30, IO27b, IO31, IO33 plus every new branch of this round) all fail the tests (Release, -O3).
- NaN sigma (R6, low): `gaussian_blur` now uses torchvision's `s <= 0.0` check, so NaN passes (NaN output) instead of
  being a new deviation.
- Docs (P2, P3, P4, R7, low): arithmetic-coded JPEG and GIF / WebP stated as decoded by torchvision but not here
  (D5, carry-over); the palette alpha-mode deviation reworded from the reviewer's measurement (D5, `_io.hpp` comment);
  `random_apply_option` aliases `random_flip_option` and RandomPosterize's default documented for float images (D7).
- Why: each code finding was reproduced before the fix; see task64.

## Tasks
- [x] [task60 — functional color ops](tasks/task60-vision-color-ops.md)
- [x] [task61 — functional geometry: rotate / affine / perspective / elastic](tasks/task61-vision-geometry-warps.md)
- [x] [task62 — torchvision.io-style decode_image + stb_image h2v1 patch](tasks/task62-vision-decode-image.md)
- [x] [task63 — GaussianBlur and the p-gated color transform classes](tasks/task63-vision-color-transforms.md)
- [x] [task64 — review-round fixes (JPEG EOI, gamma f16, read_file, stb, PNG gamma)](tasks/task64-vision-review-fixes.md)

## Issues / Gotchas
- G1 — Dev parity sweep (scratch probe, not in ctest; `scratchpad/vision2/impl/sweep`): CPU 6634 cases over the 15
  ops (u8 / f32 / f64, 8 shapes from 3 x 2 x 9 to 3 x 120 x 160 incl. 5-D, every interpolation / expand / center /
  fill kind, integral and fractional perspective points, coefficients): 6616 bit-exact, 18 raise on both sides,
  0 mismatches. MPS float32 2214 cases: 2135 exact, 39 raise on both, 40 mismatches = all adjust_hue (D8); MPS uint8
  2388 cases (contrast / saturation excluded: torchvision aborts): 2307 exact, 41 raise on both, 40 = adjust_hue.
  Decode: 8-bit PNG 66 files x 5 modes and 16-bit 15 x 5 bit-exact (incl. gamma); palette: gray / rgb exact, D5's
  two deviations; JPEG: D6.
- libtorch `torch::conv2d(x, w, {}, 1, 0, 1, groups)` is ambiguous (the `string_view padding` overload takes `0`);
  pass `torch::IntArrayRef{0}`.
- `torch::pickle_load(buf).toListRef()` dangles: keep the `IValue` alive.
- `torch.save` cannot pickle uint16 tensors in torch 2.5.1 (store as int32).
- MPS `tensor.to("mps", non_blocking=True)` from a CPU tensor that is then freed copies garbage (the D8 hue bug).
- torchvision's `gaussian_blur_image` builds the kernel in the image dtype but its affine `_affine_grid` linspaces in
  the default dtype (float32), also for float64 images; both mirrored.
- `llvm-cov show` marks comment lines inside executed regions with count 0; the union script skips comment lines,
  `llvm-cov report` per binary counts them correctly.
- G2 — Review-round sweeps (scratch, `scratchpad/vision2/fix`): JPEG 255 files x unchanged / gray / rgb (D6 numbers);
  PNG: the first sweep's 81 files x 5 modes and both reviewers' 290 files unchanged apart from the 2 R5 files that
  now match (remaining misses: D5's palette deviations and the documented 16-bit sBIT gap); libpng gAMA / sRGB
  behaviour pinned on 20 hand-made files (D5). No JPEG among the 255 is rejected by the EOI check.
- `unit_test/test_util.hpp` prints its failure lines to stderr; capture stderr when scripting test runs.
- libjpeg-turbo's fancy upsampling rounds alternately (+1 / +2 for h2v1 and h1v2, +8 / +7 for h2v2); stb_image uses
  the larger bias everywhere. The h2v1 case is patched (D6), the others are not.

## Open / TODO (carry-over)
- Random families: ColorJitter, RandomRotation, RandomAffine, RandomPerspective, RandomCrop, RandomResizedCrop,
  RandomErasing, ElasticTransform, RandomPhotometricDistort, RandomChannelPermutation; GaussianNoise, RGB; Compose.
- Decoding: EXIF orientation, GIF / WebP and arithmetic-coded JPEG (torchvision 0.20.1 decodes them), CMYK JPEG,
  palette indices in unchanged mode, sBIT, the 4:2:0 / 4:4:0 rounding residuals (D6), exact JPEG (D6 alt.).
- Parity off arm64 (x86 AVX2 / FMA, stb's SSE2 JPEG path) is not measured here; the tests allow 1 level / 1e-5.
- GCC GNU-mode contraction on FMA targets (D4) is documented, not guarded.
- `readme.md` still says vision is "coming soon" (progress07 R4) — Mux's call.

## Agent log
- 2026-10-23 [ai] Read the torchvision 0.20.1 v2 kernels (`_color.py`, `_geometry.py`, `_misc.py`, `_augment.py`,
  `_color.py` / `_misc.py` classes, `_RandomApplyTransform`) and `torchvision.io`; probed torchvision's PNG decode per
  color type / mode / chunk to pin libpng's behaviour (D5). Implemented D2-D7; dev sweep G1 (CPU 6616/6616 exact);
  found and patched the stb h2v1 bug (D6, red before: max 121); found torchvision's MPS hue and palette-alpha bugs
  (D5, D8). Tests: functional 450, transform 180, io 109 checks; full build + ctest green (10/10). Coverage 100% lines
  for the three `_vision` headers. git clang-format applied (vendor file excluded). Not committed.
- 2026-10-23 [ai] Review round (D10 / task64): re-checked the 12 findings against torchvision 0.20.1 in the venv and
  the reviewers' repro files; fixed P1 (EOI check), R1 (gamma gain out of place), R3 (read_file), R5 (libpng
  colour-space rules), R6 (NaN sigma), R2 (two more stb upsampling patches, D6 re-measured on 255 JPEGs), R4 (goldens
  and checks; 22/22 mutants killed), R7 (option alias); P2-P5 and the rest of R2 / R7 are documentation (D5-D7).
  Each code fix was red before (io 124/135, functional gamma f16 / bf16 and NaN sigma, pre-fix stb: jpeg_422 mean
  0.167 and jpeg_narrow max 61). Tests: functional 467, transform 180, io 136 checks; full build + ctest green
  (10/10). `llvm-cov report`: `_functional.hpp` 981/981, `_transform.hpp` 225/225, `_io.hpp` 203/203 lines. git
  clang-format clean (vendor file excluded). Not committed.
