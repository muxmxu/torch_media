# Task 62 — torchvision.io-style decode_image / read_image + stb_image h2v1 patch
id: 2026-10-23/task62
parent: 2026-10-23/progress09
status: done
owner: code_agent

## Objective
Decode JPEG / PNG bytes to uint8 `[C, H, W]` like `torchvision.io.decode_image` with an `image_read_mode`; PNG exact,
JPEG differences measured and bounded.

## Scope
In:
- `_vision/_io.hpp`: `image_read_mode`, `decode_image`, `read_file`, `read_image`; `detail::read_png_header`,
  libpng gray / gamma emulation, `stb_decode` (progress09 D5).
- `_vision/_vendor/stb_image.h`: the `stbi__resample_row_h_2` patch (D6); THIRD_PARTY_LICENSES entry.
- `unit_test/vision/io/{gen_golden.py, golden.hpp}` (new) and decode tests in `main.cpp`.
Out:
- GIF / WebP, EXIF orientation, CMYK JPEG, exact JPEG (D6 alternatives).

## Inputs (read first, priority order)
1. `.venv/.../torchvision/io/image.py`; torchvision's decode behaviour per PNG color type / mode / chunk, probed
   empirically (progress09 D5).
2. libpng `png_do_rgb_to_gray`, `png_build_gamma_table`, `png_build_8bit_table` / `png_build_16bit_table`.

## Acceptance criteria
- [x] PNG goldens bit-exact in every torchvision-sane mode: gray, gray+alpha, RGB, RGBA, 1-bit, palette (+tRNS),
  RGB+tRNS, gAMA 1/2.2, sRGB, gAMA 1.0, 16-bit RGB / gray (+gAMA).
- [x] JPEG (4:4:4 / 4:2:2 / 4:2:0 / gray): max |delta| <= 3 (1 for gray), mean <= 0.25; the 4:2:2 golden fails
  without the stb patch (max 121).
- [x] Errors: `std::invalid_argument` for bad tensors / JPEG alpha modes, `std::runtime_error` for unreadable files,
  unsupported formats, truncated data.
- [x] `load_image` / `save_image` unchanged; 100% line coverage of `_io.hpp`.
