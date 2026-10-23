# Progress 04 — PNM in load_image; readme sections for metric and data
id: 2026-10-24/progress04
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-24/progress03, 2026-05-31/progress04, 2026-10-23/progress09, 2026-10-23/progress11,
       2026-10-24/progress01, 2026-10-24/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_vision/_io.hpp
  - libtorchmedia/include/torchmedia/_data/_folder.hpp
  - unit_test/vision/io/{gen_golden.py, golden.hpp, main.cpp}
  - unit_test/data/test_folder.cpp
  - readme.md, CLAUDE.md, develop_log/2026-10-24/progress03-data-layer.md

## Goal
`vision::io::load_image` reads PNM P1-P6 correctly (8 / 16-bit, maxval, comments, ASCII), so ImageFolder's
`load_image(true)` lists .ppm / .pgm again (progress03 finding P2); the readme documents `torchmedia::metric` and
`torchmedia::data` with compiled examples. decode_image / read_image are unchanged.

## Context / Motivation
progress03's review found that load_image (stb_image v2.30) returned the low byte of 16-bit binary PGM / PPM samples
(1000 -> 232, no error) and threw on ASCII PNM; the folder defaults dropped .ppm / .pgm as a workaround. The readme
rewritten on 2026-10-23 predates the metric (progress11, progress01, progress02) and data (progress03) modules. Scope
set by the orchestrator under Mux's delegation.

## Decisions

### D1 — A netpbm reader in `_vision/_io.hpp`, not an stb_image patch
- Decision: `detail::pnm_decode` (about 90 lines) reads P1-P6; `load_image` sends a file whose first two bytes are
  `P1`..`P6` to it and every other file to stb_image as before. stb_image.h is not touched.
- Why: stb's PNM path is wrong in three independent ways — 16-bit samples are copied without the big-endian swap,
  maxval is ignored (a maxval-15 file returns raw 0..15, so `/ 255` is wrong too), and P1-P4 are not read at all. A
  patch would have to add byte swapping, maxval scaling and two ASCII parsers plus P4 bit unpacking inside the vendored
  C code (no smaller than the reader), re-applied on every re-vendor, and stbi_load's uint8 API would still drop
  16-bit precision. The reader is C++ in the library's own header, tested against the spec and fully covered.
- Alternatives considered: `stbi_load_16` + byte swap + `>> 8` (progress03's Open note) — fixes binary 16-bit only;
  maxval scaling and ASCII would remain missing.

### D2 — Format rules (netpbm spec; libnetpbm where the spec is terse)
- Whitespace is blank / TAB / CR / LF (the spec's list; VT and FF are not whitespace — PIL and stb accept them).
- `#` through the next CR / LF is a comment and separates tokens, as libnetpbm's `pm_getc` turns it into the newline
  (`7#c\n5` is 7 then 5; PIL joins it to 75). A comment right after the last header token ends at the raster's
  whitespace byte (`255#c\n<raster>`; PIL reads the comment into the raster).
- P4-P6: exactly one whitespace byte separates the header from the raster (a raster may start with LF / blank / `#`).
  Samples are 1 byte for maxval < 256, else 2 bytes big-endian. P4 rows are padded to whole bytes (padding bits are
  ignored). P1 bits need no separator. Bytes after the raster (a second image) are not read.
- Limits: width / height in [1, 2^24] (stb_image's STBI_MAX_DIMENSIONS), maxval in [1, 65535], plain samples <= 65535
  (digit runs saturate, no overflow); the raster's minimum byte count is checked before allocating (a plain sample
  takes at least one byte), so a forged header cannot make it allocate.
- Errors: `std::runtime_error("Could not load image: {path} (PNM: {reason})")`, reasons "expected a decimal integer",
  "value outside [lo, hi]", "no whitespace byte before the raster", "truncated raster", "expected a PBM bit",
  "sample above maxval" (raw and plain; PIL raises for plain samples and clamps raw ones).

### D3 — Output contract
- `load_image(pnm)` = float32 [C, H, W] = sample / maxval, divided in float32 (C = 1 for P1 / P2 / P4 / P5, 3 for P3 /
  P6). PBM: 1 = black -> 0.0, 0 = white -> 1.0. 8-bit maxval-255 files give the same values as the stb path did
  (`x / 255.0`).
- 16-bit PNM keeps full precision; a 16-bit PNG through load_image is still reduced by stb to its high byte. PIL
  quantizes non-255 maxvals to 8 bits (16 for gray "I"): it agrees within its rounding.
- Memory: an 8-bit raster stays a uint8 view of the file buffer (16-bit / plain / PBM: one int32 tensor), the maxval
  check is a max() reduction, and one float32 [C, H, W] buffer is filled by `copy_` (convert + transpose) and divided
  in place — the same float32 division, goldens unchanged. Peak RSS for a 4000 x 3000 P6 (Release, 92 MB baseline):
  8-bit 758 -> 311 MB, 16-bit 943 -> 531 MB; the same pixels as PNG through stb 438 MB (review finding F3).
- `decode_image` / `read_image` keep raising "unsupported image file, only JPEG and PNG are supported." for PNM, as
  torchvision.io.decode_image does (tested).

### D4 — ImageFolder / readme
- `image_extensions(image_folder_option().load_image(true))` = {.jpg, .jpeg, .png, .ppm, .bmp, .pgm} (torchvision's
  IMG_EXTENSIONS order); the decode_image default stays {.jpg, .jpeg, .png}. progress03 D3 / D9 / Issues / Open and the
  `_folder.hpp` comments updated.
- readme: table rows for vision::io (decode_image / read_image / load_image incl. PNM), the five metric namespaces and
  data with their parity references; a metric example (COCO mAP over two images + WER) and a data example (AudioFolder +
  PadCollate through `make_data_loader`). Every snippet is copied from a program compiled and run against the worktree
  headers (scratch `pnm/readme-check`, linked through the `torch_media` target); the inline fragments in the prose
  (`random_split(dataset, {0.8, 0.2}, seed)`, ImageFolder with Stack and load_image(true), `word_error_rate(preds,
  target)`) compile in `inline.cpp` there. The hobby / no-liability notice is unchanged.

### D5 — Tests, goldens, coverage
- `gen_golden.py` adds 11 PNM files with a numpy decoder written from the spec (`pnm_reference`) and asserts PIL
  against it where PIL reads the variant (exact for maxval 255, 65535 gray and PBM; else within PIL's rounding). The
  goldens are the file bytes, the decoded integer samples and maxval: P5 (comment line, raster starting with LF / blank
  / `#`), P5 16-bit (1000 among the samples), P5 maxval 1000 (TAB / CRLF / CR separators, a comment), P5 maxval 15, P6,
  P6 maxval 4095, P2 (CRLF, comment and TAB in the raster), P3, P1 (packed and spaced bits, a comment), P4 (width 11,
  padding bits set) and a libnetpbm-comment case (`7#width\n5 25#maxval\n<raster>` + a trailing second image; no PIL).
  The existing 74 entries regenerate byte-identical.
- ctest `vision_test_io` (170 checks): each golden exactly (float32 sample / maxval, max |delta| 0), closed forms
  (16-bit 1000 -> 1000 / 65535; PBM white = 1), every error reason (incl. VT as separator, a 23-digit height, a
  2^24 x 2^24 P6 / P1 header with one byte of raster, a comment running to EOF), decode_image / read_image on PNM still
  raise. `data_test_folder` (224 checks): a folder mixing two PNGs, a 16-bit binary .pgm and an ASCII .ppm — the default
  loader lists the PNGs; load_image(true) all four (PNM = sample / maxval), stacked by a loader.
- Red first: against the unfixed header vision_test_io 138/143 (P5 16-bit max |delta| 0.89, maxval 1000 0.73, maxval
  15 0.94, P6 4095 0.90; P2 threw "Could not load image") and data_test_folder 216/219 (extension list, folder size,
  index 2 out of range).
- Coverage (Debug + `TORCHMEDIA_COVERAGE`, scratch `pnm/build-cov`): `_vision/_io.hpp` 323/323 lines in
  vision_test_io (regions 334/335: the missed region is the pre-existing `png_reciprocal` overflow fallback);
  `_data/_folder.hpp` 259/259 lines in data_test_folder.
- Dev sweep (scratch `pnm/impl/sweep`, not committed): 600 random files (P1-P6, maxval 1..65535 incl. 255 / 256 /
  65535 edges, random separators, comments between and right after tokens, plain P1 with and without separators, P4
  padding bits 0 / 1, rasters starting with whitespace bytes) — load_image equals the numpy reference bit for bit on
  600 / 600; PIL agrees on the 293 files without libnetpbm-only comment placement.

## Tasks
- [x] [task30 — PNM reader in load_image](tasks/task30-pnm-load-image.md)
- [x] [task31 — ImageFolder PNM defaults](tasks/task31-imagefolder-pnm.md)
- [x] [task32 — readme: metric and data](tasks/task32-readme-metric-data.md)

## Issues / Gotchas
- `std::string("...\0...")` stops at the first NUL: the error tests build their byte strings with the `s` literal.
- A comment that runs to EOF can only occur before the raster, so it is always an error case ("no whitespace byte
  before the raster"); a trailing comment after the last plain sample is never parsed.
- `make_data_loader(...)` used directly as a range-for expression dangles (the unique_ptr temporary dies before the
  loop body); the test keeps the loader in a variable.

## Open / TODO (carry-over)
- load_image reduces 16-bit PNG to 8 bits (stb `stbi_load`) while 16-bit PNM keeps full precision; `stbi_load_16` for
  16-bit PNG would make them consistent (changes load_image's values for 16-bit PNG).
- PAM (P7) and PFM (`Pf`, which PIL reads) are not supported (stb: "unknown image type").
- The readme's vision functional / transform row still lists only the 2026-10-23 core ops (not the color / geometry /
  random classes of progress09 / progress10); out of this progress's scope.

## Agent log
- 2026-10-24 [ai] D1-D5: gen_golden.py PNM section (PIL-asserted, old entries byte-identical), red tests (vision_test_io
  138/143, data_test_folder 216/219), `detail::pnm_decode` + load_image dispatch, ImageFolder PNM defaults, progress03 /
  CLAUDE.md notes. vision_test_io 170, data_test_folder 224 checks; coverage `_io.hpp` 320/320, `_folder.hpp` 259/259;
  sweep 600 / 600 bit-exact vs the spec decoder, 293 PIL-checked. readme metric / data sections from compiled programs
  (outputs equal torchmetrics + pycocotools: map 0.3842, map_50 0.8317, map_75 0.1683, mar_100 0.3833, wer 0.2222).
  Full Release build + ctest 21 / 21 green; clang-format clean on the changed C++ files. Not committed.
- 2026-10-24 [ai] Review fixes. F1: progress03 D3's PIL deviation note was wrong for 16-bit gray PNM (PIL opens it as
  I and convert("RGB") clips > 255 to 255; only RGB / maxval < 256 is rescaled to 8 bits) — reworded, checked with
  Pillow 12.2.0. F2: the readme's vision::io row now names the palette-PNG (unchanged mode) and 16-bit sBIT exceptions
  to "PNG bit-exact" (progress09 D5). F3: pnm_decode peak memory (D3 Memory); outputs on the five measured files are
  byte-identical to the previous decoder. vision_test_io 170/170, data_test_folder 224/224, `_io.hpp` 323/323 lines;
  Release build + ctest 21 / 21 green; clang-format clean.
