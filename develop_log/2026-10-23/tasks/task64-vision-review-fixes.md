# Task 64 — Vision phase-2 review-round fixes
id: 2026-10-23/task64
parent: 2026-10-23/progress09
status: done
owner: code_agent

## Objective
Fix the verified review findings on the phase-2 vision work (progress09 D10), each with a test that fails before
and passes after; correct the documentation the reviewers measured as wrong.

## Scope
In:
- `_vision/_io.hpp`: JPEG EOI check (truncated data raises); `read_file` checks the read and rejects empty files;
  `read_png_header` keeps the gamma by libpng's colour-space rules; comments on the palette alpha modes and JPEG.
- `_vision/_vendor/stb_image.h`: two more chroma-upsampling patches (h2v1 +1 rounding, replication for chroma
  <= 2 samples wide); THIRD_PARTY_LICENSES sha256.
- `_vision/_functional.hpp`: `adjust_gamma` gain with an out-of-place `mul`; `gaussian_blur` sigma check as
  torchvision's `s <= 0.0`.
- `_vision/_transform.hpp`: `random_apply_option` aliases `random_flip_option`; RandomPosterize default comment.
- Tests / goldens: io `png_rgb_{two_gama, srgb_gama1, srgb_len2, srgb_intent7, gama_tiny_srgb, gama097}`,
  `png_palette_gama_after_plte`, `jpeg_narrow`, `jpeg_rst`, truncated / fill-byte / trailing-byte JPEGs,
  `read_file` on a directory and an empty file, per-file JPEG mean bounds; functional `gamma_gain_{f16, bf16}`,
  `posterize_white_f32`, `blur_default_sigma_f64`, `perspective_frac_{f32, f64}`, `elastic_bicubic_fill_f32`, NaN
  sigma, rotate fast paths on float64 bilinear.
- Docs: progress09 D2 / D5 / D6 / D7 / D9 / D10, CLAUDE.md (vision I/O line).
Out:
- GIF / WebP / arithmetic-coded JPEG decoding, the 4:2:0 / 4:4:0 rounding residuals (D6), EXIF orientation.

## Acceptance criteria
- [x] Before the fixes: `vision_test_io` 124/135 (two-gAMA, bad-length sRGB, sRGB intent 7, out-of-range gAMA + sRGB,
  gAMA after PLTE: max 51; jpeg_narrow max 61; JPEG without EOI accepted twice; read_file on a directory / empty file
  returned), `vision_test_functional` gamma f16 / bf16 off by 4.9e-4 / 3.9e-3 and NaN sigma rejected; with the
  pre-fix stb_image, jpeg_422 mean 0.167 > 0.06 and jpeg_narrow max 61. After: all green.
- [x] 22 mutants (Release -O3) fail the tests: every new branch, and review R4's survivors (float posterize clamp,
  fractional perspective rows, float32 coefficients, blur FMA pragma, bicubic fill, sRGB precedence, gamma
  significance band, gAMA range, rotate 180 / square 90-270 fast paths via float64 bilinear exactness).
- [x] JPEG sweep (255 files) and PNG sweeps (81 + 290 files) re-run against torchvision; existing golden entries
  unchanged by the regeneration; full ctest green; 100% lines for `_vision/*.hpp`; clang-format clean.

## Constraints
- Do not touch `_audio/`, `unit_test/audio/`, `readme.md` or LICENSE; no commit.
