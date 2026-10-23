# Task 20 — Vendor dr_flac / dr_mp3 / stb_vorbis, re-pin dr_wav, isolate stb_vorbis macros
id: 2026-10-23/task20
parent: 2026-10-23/progress05
status: done
owner: code_agent

## Objective
The FLAC, MP3 and Ogg Vorbis decoders are available to `_io.hpp` as unmodified, provenance-recorded vendored files
whose implementations are emitted only in the `TORCHMEDIA_IO_IMPLEMENTATION` translation unit, without leaking macros.

## Scope
In:
- `_audio/_vendor/{dr_flac.h, dr_mp3.h, stb_vorbis.c}` (NEW), `dr_wav.h` re-pinned to the same dr_libs commit.
- `_audio/_io_stb_vorbis_impl.hpp` (NEW): push/pop of stb_vorbis's unprefixed macros around its implementation.
- Include / implementation block at the top of `_audio/_io.hpp`; THIRD_PARTY_LICENSES.
Out:
- Any edit to a vendored file (license text stays inline, files are byte-identical to upstream).
- Encoders other than dr_wav; FFmpeg.

## Inputs (read first, priority order)
1. `develop_log/2026-05-30/progress01-remove-ffmpeg-dr_wav-io.md` — D2 (decoder choice), D4 (ODR gate).
2. Upstream file headers / license sections of each downloaded file.

## Deliverables
- Vendored files fetched with curl from `raw.githubusercontent.com/<repo>/<commit>/<file>` into new empty directories
  under the session scratchpad, read (header, license, a scan for process / network / env calls), then copied.
- THIRD_PARTY_LICENSES entry per library: upstream URL, commit, version string, sha256, license.
- `_io_stb_vorbis_impl.hpp`: one `push_macro` / `pop_macro` pair per macro the stb_vorbis implementation defines
  beyond its `STB_` configuration (43, from a clang `-dM` diff of a TU with and without the implementation).

## Acceptance criteria
- [x] sha256 of each vendored file equals the value in THIRD_PARTY_LICENSES (dr_flac 96b3dfca..., dr_mp3 91f264e8...,
  dr_wav 999aecea..., stb_vorbis 4c7cb2ff...).
- [x] Every test binary that defines `TORCHMEDIA_IO_IMPLEMENTATION` (audio_test_io, vision_test_io) links; the other
  binaries compile with declarations only; full build has no new warning (the stb_vorbis always-false compare
  warning is silenced inside the wrapper only).
- [x] `-dM` diff: dr_flac / dr_mp3 / dr_wav define only prefixed macros; stb_vorbis's 43 are restored after the include.

## Constraints
- Pinned commit or tag, never a branch head. The dr_libs release tags (flac-0.13.3, mp3-0.7.3) were tried first and
  rejected (progress05 D1): dr_mp3 0.7.3 returns 0 frames after a seek on a LAME-tagged MP3 opened from a file.

## Notes / Assumptions
- Assumption: stb_vorbis.c at 1ee679c (2021-07-12) is its newest revision; later stb commits do not touch it.
- Question (for Mux): pinning dr_libs to an untagged commit 2-3 days old vs a tag with known MP3 seek / overflow
  bugs — progress05 D1 chose the commit; re-pin to tags when released.
- Later (review round, progress05 D1 / D2): stb_vorbis.c is no longer unmodified — it carries two marked local
  fixes (5.1 residue length, comment-header bounds); M_PI and STB_VORBIS_HEADER_ONLY are also saved / restored, and
  `TORCHMEDIA_NO_*_IMPLEMENTATION` opt-outs exist.
