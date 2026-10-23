# torch_media — Project Guide for AI Collaboration

torch_media (a.k.a. **TorchMedia / libtorchmedia**) is a **header-only C++ library** that
reimplements **torchaudio / torchvision** functionality on top of **native libtorch
(`torch::Tensor`) operators**. MIT licensed. Audio is partially implemented; vision is stubs.

## Collaboration model (development, not research)

A software-development workflow with three roles. Note the "AI assistant" and the "code agent"
are usually the same Claude Code instance wearing two hats:

- **Human (Mux)** — owns decisions, scope, direction, and conventions. The AI proposes; Mux decides.
- **AI (planner/recorder hat)** — turns Mux's intent into a `progress` note (records the decision,
  the why, and the implementation impact) and breaks it into executable `task` files.
- **AI (implementer hat / code agent)** — executes a `task`: inspects and changes the codebase.

Authority (confirmed 2026-05-30): **Mux makes the key decisions; the AI writes progress/task docs
and writes code; code changes are reviewed by Mux before they count as done.**

## Workflow

1. Mux raises an idea / problem (conversation in Chinese).
2. AI creates or updates a **progress** entry under `develop_log/<date>/` recording the decision,
   rationale, and impact.
3. AI breaks the work into one or more **task** files under `develop_log/<date>/tasks/`.
4. AI implements the task (code), keeping builds and unit tests green.
5. AI appends to the progress `Agent log`; Mux reviews.

See `develop_log/README.md` for file conventions, templates, and naming.

## Languages

- **Conversation with Mux: Chinese.**
- **All written artifacts (CLAUDE.md, progress, task, README): English** (code-agent-facing).
- Code identifiers and comments: English.

## Code conventions

- **Header-only**: implementation lives in `.hpp` under `libtorchmedia/include/torchmedia/`.
  Keep free functions `inline`; vendored single-header C libs are gated behind an implementation
  macro to avoid ODR violations (see I/O direction below).
- **Style**: `.clang-format` (LLVM base, IndentWidth 4, ColumnLimit 120, namespaces indented).
  Run clang-format before committing.
- **Naming**: `snake_case` for functions/variables; option structs as `xxx_option` with fluent
  setters; namespaces `torchmedia::{audio,vision}::{io,functional,transform}` and `torchmedia::audio::separation`.
- **No heavy runtime deps**: prefer header-only / vendored single-header over system libraries.

## Dependencies & build

- Deps live in `dependence/` (libtorch; matplotplusplus for the optional plot module) — fetched by `script/download_dep`.
- Build system: **CMake** (>= 3.20). libtorch is found via `dependence/libtorch`
  (`find_package(Torch)`). No formatting library: use `std::format` / `std::print`.
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` — configure.
  - `cmake --build build -j` — build all targets.
  - `cmake --build build --target audio_test_io && ./build/unit_test/audio/io/audio_test_io` — build & run the I/O test.
  - `ctest --test-dir build` — run registered tests.
- C++ standard: **C++23** (headers use `<print>`; see develop_log/2026-10-23/progress01).
- Audio I/O: WAV via vendored header-only `dr_wav`; the implementation block is emitted in the one TU
  that defines `TORCHMEDIA_IO_IMPLEMENTATION` (each test's `main.cpp` does this).
- Unit tests live in `unit_test/` and double as usage examples.
- Optional plotting: `torchmedia::plot` (matplot++ / gnuplot) is **opt-in** via `-DTORCHMEDIA_WITH_PLOT=ON`
  (needs gnuplot at runtime + `dependence/matplotplusplus`). The core audio/vision stays header-only and
  does not require it; `torchmedia.hpp` does not include `plot.hpp` — use `#include <torchmedia/plot.hpp>`.

## Testing & coverage

- Every code change to library headers (especially `_audio/_functional*.hpp`) MUST ship with
  assertion-based tests using `unit_test/test_util.hpp` (`TM_CHECK`, `TM_CHECK_TENSOR_CLOSE`) and
  registered with ctest. No eyeball / print-only tests.
- A new or changed function is not "done" until covered; target **100% line coverage** for
  `_audio/_functional*.hpp` and `_audio/_separation.hpp` (vendored `_vendor/` is excluded).
- Golden reference values: prefer closed-form truths and libtorch self-reference (portable, no deps).
  Use the local `.venv` (torch/torchaudio 2.5.1) as a development-time point-wise cross-check.
- `ctest --test-dir build` must be green before a task is marked done.
- Coverage run: `cmake -S . -B build -DTORCHMEDIA_COVERAGE=ON`, then `llvm-profdata merge` +
  `xcrun llvm-cov` (with `--ignore-filename-regex='_vendor/.*'`).

## Audio I/O direction (decided 2026-05-30 — see develop_log/2026-05-30/progress01)

- Drop the FFmpeg-from-source dependency. Use **vendored single-header** decoders:
  `dr_wav` (+write), `dr_flac`, `dr_mp3`, `stb_vorbis`. WAV first.
- `load_audio` returns the native sample rate; **resampling is a separate torch-native op**, not
  in the loader.
- ODR: vendored implementations gated behind `TORCHMEDIA_IO_IMPLEMENTATION` (consumer defines it
  once, in one translation unit).
- FFmpeg becomes optional (`TORCHMEDIA_WITH_FFMPEG`, default OFF) for compressed-format encoding /
  Opus / future video decoding only.
