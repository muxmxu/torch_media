# Progress 01 — Remove FFmpeg, switch audio I/O to header-only single-header decoders
id: 2026-05-30/progress01
date: 2026-05-30
author: human+ai
status: active
refs: []
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_io.hpp
  - libtorchmedia/include/torchmedia/_audio/_vendor/dr_wav.h  (vendored)
  - CMakeLists.txt + libtorchmedia/ & unit_test/audio/io/ CMakeLists.txt
  - unit_test/audio/io/main.cpp, unit_test/audio/functional/main.cpp
  - script/download_dep, THIRD_PARTY_LICENSES, .gitignore
  - removed: xmake.lua, ffmpeg_xmake.lua, libtorchmedia_rule.lua, **/xmake.lua

## Goal
Remove the FFmpeg (built-from-source) dependency from the audio I/O layer and replace it with
vendored, header-only single-header decoders, starting with WAV. Keep the public
`load_audio` / `save_audio` API unchanged.

## Context / Motivation
TorchMedia is marketed as header-only ("drop the headers and go"), yet the current I/O path links
FFmpeg and `script/download_dep` clones + builds FFmpeg from source — the heaviest possible
dependency form, which contradicts the project's identity. The I/O layer was being migrated
SOX -> FFmpeg (WIP in `_io.hpp`). Mux wants a dependency-light / "native" approach.

## Decisions

### D1 — There is no stdlib "native" C++ audio I/O; target "no install/link/build" instead
- Decision: Frame the goal as "no external dependency that must be installed, linked, or built",
  achieved via vendored single-header libraries — not a non-existent stdlib audio API.
- Why: The C++ standard library has zero audio/format awareness. The realistic dependency spectrum
  is system-lib -> vendored-source-build -> single-header -> hand-rolled. FFmpeg-from-source sits at
  the heaviest end.
- Impact: Drives backend selection (D2) and removal of FFmpeg from the build scripts (D5).

### D2 — Use vendored single-header decoders (dr_libs + stb_vorbis); WAV first
- Decision: Adopt `dr_wav` (read + write), `dr_flac`, `dr_mp3`, `stb_vorbis` (decode) as the default
  I/O backend. Ship WAV (dr_wav) first, then FLAC / MP3 / OGG decode.
- Why: All are header-only and public-domain / MIT-0 / MIT (compatible with our MIT). Decode speed
  is on par with FFmpeg / libsndfile (WAV decode is I/O-bound, not CPU-bound). Perfect fit for the
  header-only model. `dr_wav` exposes `drwav_read_pcm_frames_f32` returning a contiguous interleaved
  f32 buffer — ideal impedance match for `torch::from_blob`.
- Impact: Add `_audio/_vendor/`; rewrite `_io.hpp` load/save; add a `THIRD_PARTY_LICENSES` file.
  Structural gap to accept: only WAV has a header-only *encoder*, so `save_audio` is WAV-only for now.
- Alternatives considered: hand-rolled WAV (robust impl ~600-1800 LOC, effectively reinvents dr_wav —
  keep only as a zero-vendored-code fallback); AudioFile.h (stores `vector<vector<double>>`, forces an
  extra copy, poor libtorch fit); libnyquist (needs a CMake build — defeats the goal); miniaudio
  (a playback/capture device engine wrapping the same dr_libs — overkill).

### D3 — Resampling leaves the loader; loader reports native sample rate
- Decision: `load_audio` returns audio at its native sample rate. Resampling becomes a separate
  torch-native op (band-limited sinc expressed as an ATen convolution), implemented later in
  functional/transform.
- Why: The current `swr_convert` call uses in == out rate (format/layout only), so dropping
  swresample loses no resampling capability. A torch-native resampler runs on CPU/CUDA/MPS and fits
  the project's mission ("reimplement torchaudio with native operators"). Keeps the I/O layer tiny.
- Impact: No resampler in `_io.hpp`. A future progress adds `functional::resample`.

### D4 — ODR strategy for vendored single-header impls in a header-only library
- Decision: Gate vendored implementations behind `TORCHMEDIA_IO_IMPLEMENTATION`; the consumer defines
  it in exactly one `.cpp` before including torchmedia. Default stays header-only.
- Why: `dr_*.h` implementation blocks are non-inline C-linkage definitions; naive inclusion in every
  translation unit causes "multiple definition" link errors.
- Impact: Document the one-line requirement; optionally ship a tiny impl `.cpp` as an alternative.

### D5 — FFmpeg downgraded to optional, off by default
- Decision: Keep FFmpeg only behind `TORCHMEDIA_WITH_FFMPEG` (default OFF), dynamically linked against
  a system install — never built from source.
- Why: Compressed-format *encoding*, Opus, and future video decoding (vision side) genuinely need
  FFmpeg, but it must not be the default audio dependency.
- Impact: Remove FFmpeg from the default build config and from `script/download_dep`; gate any future
  FFmpeg path behind the flag.

### D6 — C++ standard is C++20
- Decision: Build as C++20 (overridable to C++23 via `-DTORCHMEDIA_CXX_STANDARD=23`).
- Why: Apple Clang 21 compiles both C++20 and C++23 cleanly, but libtorch 2.5.1 ships built against
  C++17 and libc++'s C++23 library support still has gaps; C++20 is the safe, sufficient baseline
  (the codebase already uses `<ranges>`).
- Impact: `CMAKE_CXX_STANDARD=20`, `cxx_std_20` on the torch_media target.
- Superseded 2026-10-23 by [2026-10-23/progress01](../2026-10-23/progress01-cxx23-drop-fmt.md) (C++23, fmt removed).

### D7 — Migrate the build system from xmake to CMake
- Decision: Replace xmake with CMake (>= 3.20) as the build system.
- Why: The xmake setup was messy for this personal project; CMake is the de-facto standard for
  libtorch (`find_package(Torch)`) and for vendoring header-only deps.
- Impact: Root `CMakeLists.txt` + per-dir CMakeLists; `torch_media` is an INTERFACE target linking
  `${TORCH_LIBRARIES}` with fmt header-only; all xmake `.lua` removed. The stale CMakeLists used
  `find_package(SOX)` / `find_package(LibTorch)` — replaced.

## Tasks
- [x] [task01 — vendor dr_wav, rewrite the WAV I/O path, drop FFmpeg link](tasks/task01-vendor-dr_wav-wav-path.md)

## Issues / Gotchas
- Current `_io.hpp` does a per-channel `.clone()` (around line 164) — replace with a single
  `from_blob({frames, channels}).transpose(0,1).contiguous()`; cheap win.
- `torch::from_blob` does not take ownership: clone (then free the decoder buffer) or pass a deleter
  that frees the dr_wav allocation.
- stb_vorbis has the largest fuzzing/CVE surface of these libs and a non-thread-safe decoder handle —
  feed trusted audio only (relevant once OGG support lands).

## Open / TODO (carry-over)
- ~~FLAC / MP3 / OGG decode backends (dr_flac / dr_mp3 / stb_vorbis)~~ — done in
  [2026-10-23/progress05](../2026-10-23/progress05-audio-io-formats.md).
- torch-native `functional::resample` (sinc-as-convolution) — separate progress.
- `TORCHMEDIA_WITH_FFMPEG` optional path. (`THIRD_PARTY_LICENSES` landed with task01; stale entry corrected
  2026-10-23 after the audit flagged it.)
- ~~24-bit / float32 WAV round-trip tests~~ — done in 2026-10-23/progress05 (every save encoding, plus U8 / S16 / S24 /
  S32 / F32 / F64 decode fixtures). THIRD_PARTY_LICENSES now records the pinned commit and sha256 of every vendored
  decoder (progress05 D1).

## Agent log
- 2026-05-30 [ai] Ran a 4-way, web-verified research workflow on dependency-light C++ audio I/O
  (single-header decoders, hand-rolled WAV, framework backends, perf/integration). Recorded
  decisions D1-D7. Authored task01.
- 2026-05-30 [ai] Implemented task01 end-to-end: vendored dr_wav.h v0.14.6; rewrote `_io.hpp` on dr_wav
  (load f32 interleaved -> [channels, samples]; save f32 -> PCM16); migrated build xmake -> CMake (C++20);
  removed all FFmpeg wiring + xmake `.lua`; added `THIRD_PARTY_LICENSES`. Fixed two bugs found during
  full-build verification: `load_audio("literal")` string/path overload ambiguity (added `const char*`
  overloads), and a `.gitignore` rule that was swallowing `CMakeLists.txt`. `audio_test_io` round-trip
  passes (max abs err 6.1e-05), zero ffmpeg/sox in the link line; full build (io + functional + vision) green.
- 2026-10-23 [ai] FLAC / MP3 / Ogg Vorbis decode, frame ranges, info() and torchaudio.save parity landed in
  2026-10-23/progress05 (task20-22); the related Open items above are closed.
