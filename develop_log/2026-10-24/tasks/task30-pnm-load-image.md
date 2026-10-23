# Task 30 — PNM reader in load_image
id: 2026-10-24/task30
parent: 2026-10-24/progress04
status: done
owner: code_agent

## Objective
`vision::io::load_image` decodes PNM P1-P6 (8 / 16-bit, any maxval, comments, ASCII and binary) to float32
sample / maxval, with errors instead of silently wrong values.

## Scope
In:
- `libtorchmedia/include/torchmedia/_vision/_io.hpp`: `detail::pnm_decode` and the dispatch in `load_image`.
- `unit_test/vision/io/{gen_golden.py, golden.hpp, main.cpp}`: PNM goldens from a spec decoder (PIL cross-check), error
  cases, decode_image / read_image still rejecting PNM.
Out:
- stb_image.h (no patch), decode_image / read_image formats, PAM / PFM, 16-bit PNG precision in load_image.

## Inputs (read first, priority order)
1. `develop_log/2026-10-24/progress03-data-layer.md` — finding P2 and the Open note.
2. netpbm spec (pbm.html, pgm.html, ppm.html) — header, comment and raster rules.
3. `_vendor/stb_image.h` `stbi__pnm_*` — what is wrong there (no swap, no maxval, P5 / P6 only).

## Deliverables
- `_io.hpp` — the reader per progress04 D2 / D3; load_image comment states the contract.
- `gen_golden.py` / `golden.hpp` — `pnm_*`, `pnm_*_samples`, `pnm_*_maxval`; old entries unchanged.
- `main.cpp` — `test_load_pnm`.

## Acceptance criteria
- [x] Tests red against the unfixed header (16-bit values, maxval scaling, ASCII), green after.
- [x] Each golden equal exactly; every error reason reached; decode_image / read_image raise for PNM.
- [x] `_vision/_io.hpp` 100% lines in vision_test_io; clang-format clean; full build + ctest green.

## Constraints
- Header-only, inline; no allocation sized by an unchecked header.
