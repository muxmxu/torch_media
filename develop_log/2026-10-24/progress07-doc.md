# Progress 07 — doc/: porting guide, deviations, reproducibility
id: 2026-10-24/progress07
date: 2026-10-24
author: ai
status: active            # written and checked; awaiting Mux's review
refs: [2026-10-24/progress05, 2026-10-24/progress06, 2026-10-23/progress05, 2026-10-23/progress09,
       2026-10-23/progress10, 2026-10-23/progress11]
supersedes:
commits: []
files:
  - doc/{porting-from-python.md, deviations.md, reproducibility.md}
  - readme.md, CLAUDE.md, CMakeLists.txt

## Goal
A user can port Python preprocessing to TorchMedia and know where its results differ, without reading develop_log.

## Context / Motivation
Mux asked for a markdown document set in `doc/` and whether it is needed, preferring documentation in code / comments.
Recommendation given and accepted by Mux ("可以，停掉目前的 workflow，完成 doc 的部分"): API documentation stays in
the header comments (each port mirrors a Python function whose documentation already explains the parameters; a
markdown API reference would duplicate it and drift). `doc/` holds only what no single function documents, which until
now lived in 17 developer-facing progress notes.

## Decisions

### D1 — Three pages, no API reference
- `porting-from-python.md`: namespaces and headers, the names that differ (found by diffing the headers against
  `torchaudio.functional.__all__`, `torchaudio.transforms.__all__`, torchvision v2 and torchmetrics), option structs /
  `None` / enums / tuples, classes and thread use, dtype / device, errors, building, one compiled example, what is not
  ported.
- `deviations.md`: every intentional or measured difference, grouped by module, each section linked to its progress
  notes. Only the current state (superseded decisions dropped, e.g. progress03 D2's raw throw, now routed by
  progress05).
- `reproducibility.md`: what was measured where, what changes the bits (CPU dispatch, device, FP contraction,
  decoder), seeded-parity ops and their conditions.
- readme: an intro pointer, a Documentation section, the vision row brought up to date (it still listed only the
  first 7 ops of 2026-05-31), and a note on `TORCHMEDIA_DEPENDENCE_DIR`.

### D2 — CMake: a parent's `set(TORCHMEDIA_DEPENDENCE_DIR ...)` is honoured (CMP0126 NEW)
- Found while compiling the porting example as an out-of-tree `add_subdirectory` consumer: under the library's
  `cmake_minimum_required(VERSION 3.20)` (CMP0126 OLD), `set(... CACHE PATH ...)` removed the parent's normal variable,
  the prefix pointed at a directory without libtorch, and `find_package(Torch)` silently took Homebrew's libtorch
  2.14.1, which fails to compile the headers ("no member named 'linalg' in namespace 'torch'").
- Decision: `cmake_policy(SET CMP0126 NEW)` before the cache default. Verified: the same consumer with a plain `set()`
  now finds `dependence/libtorch`.
- Not done (Mux's call): failing configuration when the found libtorch is not 2.5.x. A missing `dependence/` still
  falls through to any system libtorch; the readme and porting page say so.

### D3 — Keeping the pages true
CLAUDE.md gains a Documentation section: a change to a Python difference updates `doc/deviations.md` in the same
commit; doc / readme code is compiled and run, stated output is that program's output.

## Verification
- The porting example was built as an `add_subdirectory` consumer against libtorch 2.5.1 with `-Wall -Wextra` (no
  warning) and run: `spec [1, 257, 32], rotated [3, 80, 74], flipped [3, 64, 48], accuracy 0.8333`; the same calls in
  torchaudio 2.5.1 / torchvision 0.20.1 / torchmetrics 1.9.0 give the same shapes and 0.8333333730697632.
- Each claim was checked against its progress note, and the API claims against the headers. The checks changed four
  first drafts:
  - metric state copies are scoped to the states that keep inputs;
  - MVDR's online mode is not shareable across threads;
  - only RPDF dither draws (TPDF is deterministic);
  - the torchvision aliases.

## Open
- `mel_filter_bank` carries two Chinese inline comments (`_audio/_functional.hpp:243-244`), against the English-only
  rule; untouched (not in this scope).

## Agent log
- 2026-10-24: pages written, example compiled and cross-checked, CMake fix verified; main-checkout build + ctest
  before push (task60).
