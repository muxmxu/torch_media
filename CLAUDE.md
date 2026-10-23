# torch_media — Project Guide for AI Collaboration

torch_media (a.k.a. **TorchMedia / libtorchmedia**) is a **header-only C++ library** that
reimplements **torchaudio / torchvision** functionality on top of **native libtorch
(`torch::Tensor`) operators**. MIT licensed. Audio covers torchaudio.functional and torchaudio.transforms
(see develop_log/2026-06-01); vision covers the core torchvision.transforms.v2 resize / crop / pad / dtype / color /
rotate / affine / perspective ops as functional ops and transform classes, point-wise identical to torchvision
(develop_log/2026-10-23/progress07, progress09), the random augmentation classes and containers with seeded parity
(progress10), plus image I/O (torchvision.io-style uint8 decode_image). Metrics (`torchmedia::metric`, torchmetrics
1.9 parity, develop_log/2026-10-23/progress11): detection (torchvision.ops box ops, IoU metrics, MeanAveragePrecision
equal to pycocotools COCOeval, VOC AP), image quality and segmentation; audio (SNR family, SDR, PIT, STOI = pystoi,
mir_eval bss_eval_sources, EER / minDCF) and text (WER family, EditDistance) (develop_log/2026-10-24/progress01);
classification (torchmetrics.classification binary / multiclass / multilabel: stat scores, confusion-matrix metrics,
curves / AUROC / AP / fixed-point metrics, calibration, hinge, ranking; develop_log/2026-10-24/progress02).

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
  setters; namespaces `torchmedia::{audio,vision}::{io,functional,transform}`, `torchmedia::audio::separation`,
  `torchmedia::audio::compliance::kaldi` (torchaudio.compliance.kaldi) and `torchmedia::metric::{detection,image,
  segmentation,audio,text,classification}` (torchmetrics: a functional per metric plus a class with update / compute /
  reset / forward, progress11 D1; audio / text: develop_log/2026-10-24/progress01; classification: progress02).
- **No heavy runtime deps**: prefer header-only / vendored single-header over system libraries.
- **Seeded parity**: a random vision transform class replicates its torchvision class's parameter draws (same torch
  calls, order and dtypes, on the global generator), so `torch::manual_seed(s)` reproduces `torch.manual_seed(s)`
  (develop_log/2026-10-23/progress10 D3) — for serialized calls: concurrent calls interleave the draws (progress10 D1).
- **Python-order scalar math**: a function that reproduces Python float arithmetic (e.g. torchvision's affine matrix)
  starts with `TORCHMEDIA_FP_CONTRACT_OFF` (clang otherwise fuses `a * b + c` into FMAs and breaks bit-exactness)
  and takes the sin of an angle whose cos it also needs through `detail::py_sin` (Darwin clang merges the pair into
  `__sincos_stret`, 1 ulp off `math.sin`; progress10 D6);
  `-ffast-math` / `-ffp-contract=fast` builds are not bit-exact (develop_log/2026-10-23/progress09 D4). The
  `torch_media` CMake target adds `-ffp-contract=on` for consumers so GCC never fuses across statements; the audio
  scalar loops (lfilter / phaser / flanger / vad) rely on that (progress08) — keep both mechanisms.

## Dependencies & build

- Deps live in `dependence/` (libtorch; matplotplusplus for the optional plot module) — fetched by `script/download_dep`.
- Build system: **CMake** (>= 3.20). libtorch is found via `dependence/libtorch`
  (`find_package(Torch)`). No formatting library: use `std::format` / `std::print`.
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` — configure.
  - `cmake --build build -j` — build all targets.
  - `cmake --build build --target audio_test_io && ./build/unit_test/audio/io/audio_test_io` — build & run the I/O test.
  - `ctest --test-dir build` — run registered tests.
- C++ standard: **C++23** (headers use `<print>`; see develop_log/2026-10-23/progress01).
- Audio I/O: WAV (dr_wav), FLAC (dr_flac), MP3 (dr_mp3) and Ogg Vorbis (stb_vorbis), vendored single-header and
  pinned in THIRD_PARTY_LICENSES; the implementation blocks are emitted in the one TU that defines
  `TORCHMEDIA_IO_IMPLEMENTATION` (each test's `main.cpp` does this). Saving is WAV only.
- Vision I/O: stb_image / stb_image_write (same implementation macro). `stb_image.h` carries `torchmedia patch`es
  (JPEG chroma upsampling, THIRD_PARTY_LICENSES): re-apply them when re-vendoring. JPEG decodes differ from
  torchvision (libjpeg-turbo) by up to 3 levels (4 for 4:4:0 chroma); PNG is exact apart from documented palette /
  sBIT cases (develop_log/2026-10-23/progress09 D5, D6).
- Unit tests live in `unit_test/` and double as usage examples.
- Optional plotting: `torchmedia::plot` (matplot++ / gnuplot) is **opt-in** via `-DTORCHMEDIA_WITH_PLOT=ON`
  (needs gnuplot at runtime + `dependence/matplotplusplus`). The core audio/vision stays header-only and
  does not require it; `torchmedia.hpp` does not include `plot.hpp` — use `#include <torchmedia/plot.hpp>`.

## Testing & coverage

- Every code change to library headers (especially `_audio/_functional*.hpp`) MUST ship with
  assertion-based tests using `unit_test/test_util.hpp` (`TM_CHECK`, `TM_CHECK_TENSOR_CLOSE`) and
  registered with ctest. No eyeball / print-only tests.
- A new or changed function is not "done" until covered; target **100% line coverage** for
  every `_audio/*.hpp`, `_vision/*.hpp` and `_metric/*.hpp` header (vendored `_vendor/` is excluded), measured as the
  union over the audio / vision / metric test binaries (develop_log/2026-10-23/progress02 D11): each header with the
  binary that owns it, a shared header per line over the binaries (llvm-cov's multi-object merge drops inline header
  functions whose hashes differ per binary; progress11 D9).
- Golden reference values: prefer closed-form truths and libtorch self-reference (portable, no deps).
  Use the local `.venv` (torch/torchaudio 2.5.1, torchvision 0.20.1, torchmetrics 1.9.0, pycocotools 2.0.11, pystoi
  0.4.1, mir_eval 0.8.2, jiwer 4.0.0, scipy, scikit-learn) as a development-time point-wise cross-check; pin
  `OPENBLAS_NUM_THREADS=1` in reference scripts that solve with numpy (threaded solves change run to run).
- `ctest --test-dir build` must be green before a task is marked done.
- Coverage run: `cmake -S . -B build -DTORCHMEDIA_COVERAGE=ON`, then `llvm-profdata merge` +
  `xcrun llvm-cov` (with `--ignore-filename-regex='_vendor/.*'`).

## Audio I/O direction (decided 2026-05-30 — see develop_log/2026-05-30/progress01)

- Drop the FFmpeg-from-source dependency. Use **vendored single-header** decoders:
  `dr_wav` (+write), `dr_flac`, `dr_mp3`, `stb_vorbis` — all landed (develop_log/2026-10-23/progress05).
- `load_audio` / `info` detect the container from the leading bytes (never the extension) and mirror torchaudio.load /
  torchaudio.info (frame_offset / num_frames, memory input); `save_audio` mirrors torchaudio.save for WAV, PCM
  bit-exact. Documented deviations (float WAV kept exact, MP3 gapless lengths): progress05 D4-D8.
- Decode trusted input only (stb_vorbis has the largest CVE record); its short macros are saved / restored by
  `_audio/_io_stb_vorbis_impl.hpp` — keep that list in sync when re-vendoring. `stb_vorbis.c` carries two local
  `torchmedia patch` fixes (5.1 residue length, comment-header bounds; THIRD_PARTY_LICENSES): re-apply them when
  re-vendoring. A consumer that compiles a codec itself opts out with
  `TORCHMEDIA_NO_{DR_WAV,DR_FLAC,DR_MP3,STB_VORBIS}_IMPLEMENTATION`.
- `load_audio` returns the native sample rate; **resampling is a separate torch-native op**, not
  in the loader.
- ODR: vendored implementations gated behind `TORCHMEDIA_IO_IMPLEMENTATION` (consumer defines it
  once, in one translation unit).
- FFmpeg becomes optional (`TORCHMEDIA_WITH_FFMPEG`, default OFF) for compressed-format encoding /
  Opus / future video decoding only.
