# Progress 05 — Audio I/O: FLAC / MP3 / Ogg Vorbis decode, frame ranges, info(), torchaudio.save parity
id: 2026-10-23/progress05
date: 2026-10-23
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-05-30/progress01, 2026-10-23/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_io.hpp
  - libtorchmedia/include/torchmedia/_audio/_io_stb_vorbis_impl.hpp (NEW)
  - libtorchmedia/include/torchmedia/_audio/_vendor/{dr_flac.h, dr_mp3.h, stb_vorbis.c (patched)} (NEW, vendored), dr_wav.h (re-pinned)
  - unit_test/audio/io/{main.cpp, gen_fixtures.py (NEW), fixtures.hpp (NEW, generated), optout_main.cpp (NEW),
    user_codecs.cpp (NEW), CMakeLists.txt}
  - THIRD_PARTY_LICENSES, CLAUDE.md, develop_log/2026-05-30/progress01-remove-ffmpeg-dr_wav-io.md

## Goal
Finish the I/O direction of 2026-05-30/progress01: decode FLAC, MP3 and Ogg Vorbis with vendored single-header
decoders, add torchaudio.load's frame_offset / num_frames, memory input and torchaudio.info, and make save_audio
mirror torchaudio.save for WAV (encodings, defaults, bit-exact PCM quantization). This closes the I/O audit findings
deferred by 2026-10-23/progress02 D9 (G4-7, G4-8, G4-9, G4-10, G4-13, G4-16). Out of scope: encoders other than WAV,
Opus / AAC / video (the optional FFmpeg path of progress01 D5), normalize=False, channels_first, resampling.

## Context / Motivation
`load_audio` read WAV only and `save_audio` always wrote truncating PCM16 (0.0 became -1). The audit measured the gaps
against torchaudio 2.5.1 (sox backend, the only backend in the local `.venv`). Decisions D1-D9 below follow the design
the orchestrator set under Mux's delegation; the deviations from that design are marked "Deviation" with the reason.

## Decisions

### D1 — Vendored decoders, pinned by commit
- Decision: vendor `dr_flac.h`, `dr_mp3.h` (dr_libs) and `stb_vorbis.c` (stb) into `_audio/_vendor/` and re-pin
  `dr_wav.h` to the same dr_libs commit. The dr_libs files are unmodified; `stb_vorbis.c` carries two local fixes
  (below). Provenance (URL, commit, version string, sha256) is in
  THIRD_PARTY_LICENSES: dr_libs commit b4264c5c1f61c1d460aced4cf4d8a8dec197ca48 (2026-10-21), stb commit
  1ee679ca2ef753a528db5ba6801e1067b40481b8 (stb_vorbis v1.22, its last change).
- Why: the newest dr_libs tags (flac-0.13.3, mp3-0.7.3, 2026-01-17) were tried first and fail this feature: with
  dr_mp3 0.7.3 a file-based decoder on a LAME-tagged MP3 returns 0 frames after any seek or partial read (fixed
  upstream 2026-10-20: "Fix seeking in streams with an encoder delay", "Don't reset the stream cursor after the
  Xing/Info frame"), and the tags also predate a Xing/Info heap-buffer-overflow fix and dr_flac seek / UB fixes. The
  old dr_wav (commit 243e26f, 2026-04-26) lacked ~17 upstream fixes (malformed fmt/fact chunks, overflows, large seeks).
- Impact: the design asked for "a pinned release tag or commit"; this is a commit 2-3 days old with no tag. Re-pin to
  the next release tags when they ship (Open).
- Local stb_vorbis patches (review round, findings PR-1 / PR-2 / SEC-1; the pin is already stb_vorbis's newest
  revision, so re-pinning cannot fix them). Each change is marked `torchmedia patch`; THIRD_PARTY_LICENSES records the
  upstream sha256 4c7cb2ff... and the patched sha256 1e011afe.... Re-apply both when re-vendoring.
  1. `decode_residue`: `actual_size = rtype == 2 ? n*ch : n` (was `n*2`). Residue type 2 interleaves all `ch`
     channels of its submap into one vector of n*ch values; libvorbis's 5.1 mapping at 44.1 / 48 kHz puts 5 channels
     in one type-2 submap, and v1.22 clamped its range to 2 channels, so 6-channel files decoded to garbage (peaks
     ~20-100) with the right length and no error. 2-5 and 8 channels, and 6 channels at 8 / 22.05 kHz (other
     mapping), were unaffected. Measured after the fix: max diff vs torchaudio <= 1.53e-5 (2^-16) on every 6-channel
     file at 8 / 22.05 / 44.1 / 48 kHz.
  2. `start_decoder`, comment header: the vendor length, each comment length and the comment count are bounded by
     the bytes left in the stream (memory: unread bytes; file: the stream length; capped at 2^28 so `setup_malloc`'s
     int size cannot overflow), `comment_list` is zeroed after allocation and its count reset when the list cannot
     be allocated. v1.22 freed uninitialized `comment_list` entries when a later comment failed (SIGSEGV / SIGABRT
     from one corrupt byte, e.g. a comment count of 0x4F000001), and a corrupt length allocated and read up to
     ~2 GB (a 358-byte file took 1.7 s and 1.17 GB RSS). Comment headers larger than 256 MB are now rejected.
  ```diff
  -   unsigned int actual_size = rtype == 2 ? n*2 : n;
  +   unsigned int actual_size = rtype == 2 ? n*ch : n; // torchmedia patch: ...
  +   uint32 limit; // torchmedia patch
  +   limit = USE_MEMORY(f) ? (uint32) (f->stream_end - f->stream) : f->stream_len;
  +   if (limit > (1u << 28)) limit = 1u << 28;
      len = get32_packet(f);
  +   if (len < 0 || (uint32) len >= limit)            return error(f, VORBIS_invalid_setup);
      ...
  +      if ((uint32) f->comment_list_length > limit / 4) { f->comment_list_length = 0; return error(f, VORBIS_invalid_setup); }
         f->comment_list = (char**) setup_malloc(f, sizeof(char*) * (f->comment_list_length));
  -      if (f->comment_list == NULL)                  return error(f, VORBIS_outofmem);
  +      if (f->comment_list == NULL)                  { f->comment_list_length = 0; return error(f, VORBIS_outofmem); }
  +      memset(f->comment_list, 0, sizeof(char*) * f->comment_list_length);
      ...
         len = get32_packet(f);
  +      if (len < 0 || (uint32) len >= limit)         return error(f, VORBIS_invalid_setup); // torchmedia patch
  ```

### D2 — ODR gate and stb_vorbis macro isolation
- Decision: all four implementations are emitted only in the TU that defines `TORCHMEDIA_IO_IMPLEMENTATION` (as
  dr_wav was). stb_vorbis is included with `STB_VORBIS_HEADER_ONLY` everywhere; the implementation TU includes it a
  second time through `_io_stb_vorbis_impl.hpp`, which `push_macro`/`pop_macro`s the 43 unprefixed macros the
  implementation defines (`C`, `L`, `R`, `TRUE`, `FALSE`, `CHECK`, `DECODE`, `EOP`, ...; list from a clang `-dM`
  diff) plus `M_PI` (stb_vorbis defines it as a float literal where `<math.h>` does not, e.g. MSVC without
  `_USE_MATH_DEFINES`; review SEC-5) and silences one always-false overflow-guard warning (`-Wtautological-compare`)
  inside it. The list covers the default configuration (not `STB_VORBIS_DIVIDE_TABLE` / `STB_VORBIS_NO_CRT`).
  `DR_MP3_FLOAT_OUTPUT` is defined there too (float synthesis, no int16 rounding of the decoded samples).
- Opt-outs (review SEC-4): `TORCHMEDIA_NO_DR_WAV_IMPLEMENTATION`, `TORCHMEDIA_NO_DR_FLAC_IMPLEMENTATION`,
  `TORCHMEDIA_NO_DR_MP3_IMPLEMENTATION`, `TORCHMEDIA_NO_STB_VORBIS_IMPLEMENTATION` leave that implementation to a
  consumer that compiles the library itself (its copy must be the same version; a dr_mp3 built without
  `DR_MP3_FLOAT_OUTPUT` decodes through int16). `STB_VORBIS_HEADER_ONLY` is saved / restored with push_macro, so a
  consumer's own definition survives the include, and it is undefined around the implementation include so it cannot
  suppress the implementation. `audio_test_io_optout` links torchmedia's implementation TU with a consumer TU that
  compiles dr_wav and stb_vorbis itself (before: duplicate symbols, and `#error` on the lost macro).
- Why: without the save/restore, `#define C ...` and friends leak into every header that follows `_io.hpp` in the
  implementation TU (`_separation.hpp`, `_transform.hpp`, vision, user code). dr_flac / dr_mp3 only define prefixed
  macros (checked the same way).
- Impact: stb_vorbis still declares global-scope typedefs (`int8`, `uint32`, ...) and static functions in the
  implementation TU; harmless here, noted for users who define their own.

### D3 — Container detection by leading bytes
- Decision: `RIFF`/`RF64` + `WAVE`, W64 (`riff` GUID + `wave`) -> dr_wav; `fLaC` -> dr_flac; `OggS` whose first
  packet is `\x01vorbis` -> stb_vorbis; `ID3` or an MPEG frame sync with non-reserved layer bits -> dr_mp3; anything
  else throws std::runtime_error naming the path ("<memory>" for buffers). The extension is never used (the tests
  load extension-less files).
- Why: design item 2; the layer-bits check keeps AAC ADTS (also 0xFFF sync) out of dr_mp3.
- Impact: AIFF / RIFX (dr_wav could read them), Ogg FLAC and Opus are rejected as unsupported (Open).
- Review additions: an ID3v2 tag (10 bytes + syncsafe size, + 10 with the footer flag) followed by `fLaC` goes to
  dr_flac with the stream starting at the marker (PR-5: some taggers write this; torchaudio reads it; dr_flac's own
  ID3 skip is broken because its seeks assume `fLaC` at byte 0, so a tagged file path is read into memory first);
  every other `ID3` file still goes to dr_mp3. An MPEG frame sync after zero padding within the first 512 bytes is
  MP3 (PR-6; dr_mp3 resyncs). dr_mp3 then does not see the LAME / Xing tag, so such a file is decoded like an
  untagged one (no gapless trimming; it matches torchaudio at lag 0, 131 dB, on the reviewer's file). Other leading
  junk is still unrecognized.

### D4 — Decoded values (fixes G4-9, keeps G4-10 as a deliberate choice)
- Decision: every non-float WAV encoding and FLAC are read as full-scale int32 and scaled by 2^-31, i.e. exactly
  x / 2^(bits-1) for <= 24-bit PCM and (x - 128) / 128 for 8-bit (torchaudio's mapping; dr_wav's own f32 path is
  x / 127.5 - 1). Float WAV is returned as stored (values beyond [-1, 1] kept), 32-bit PCM rounds to nearest.
  MP3 / Vorbis come from the decoders' float output. The result is always a contiguous [C, N] float32 tensor made in
  one copy from a torch-owned decode buffer (no aliasing of decoder memory, cf. progress02 D1).
- Why: bit-exact torchaudio parity where torchaudio is exact. Deviation from "WAV variants match torchaudio.load bit
  for bit": torchaudio's sox backend routes every sample through int32 and back with (x + 64) & ~127, so for S32 it
  differs from the exact value by up to 5.96e-08 (32-38 % of the fixture samples), and for float WAV it clips to
  [-1, 1] (up to 0.25 / 0.5 off on the fixtures, whose values reach 1.25 / 1.5). Matching that would destroy data the
  file holds; the audit (G4-10) already called the exact decode the more faithful one.
- Impact: U8 / S16 / S24 / FLAC / W64 / RF64 / mu-law / A-law / IMA ADPCM equal torchaudio.load bit for bit (asserted
  in gen_fixtures.py and main.cpp); S32 / F32 / F64 equal the exact stored values.

### D5 — load_option (torchaudio frame_offset / num_frames) and memory input
- Decision: `load_option{_frame_offset = 0, _num_frames = -1}` with fluent setters; `load_audio(path, opt)` seeks to
  the offset and decodes only the requested range; `load_audio_from_memory(data, size, opt)`. A negative offset, and
  num_frames 0 or < -1, throw std::invalid_argument before the file is opened; an offset past the end gives [C, 0];
  a range running past the end is clipped. A failed seek (the data ends before the offset, e.g. a truncated MP3 whose
  Info tag claims more frames) also gives [C, 0].
- Why: torchaudio semantics. Deviation: the design listed only "num_frames < -1"; torchaudio 2.5.1 also rejects 0
  ("num_frames must be -1 or greater than 0"), so 0 throws too.
- Impact: the existing `load_audio(path)` calls are unchanged (default argument).
- Bounded decode (review PR-4 / SEC-3): `load` reads blocks of ~2^20 samples (`max(1, 2^20 / channels)` frames)
  until the requested count or until the decoder returns nothing, then `torch::cat`s them. Memory follows what
  actually decodes; before, the whole claimed range was allocated up front, so a few corrupt header bytes requested
  terabytes (FLAC STREAMINFO 2^36-1: 1.65 TB virtual on macOS) or threw `c10::Error` (RF64 ds64 sampleCount 2^50:
  4.5 PB) instead of returning the real frames.

### D6 — info() / audio_info
- Decision: `audio_info{sample_rate, num_frames, num_channels, bits_per_sample, encoding}`, `info(path)` (one
  `std::filesystem::path` overload accepts literals, strings and paths) and `info_from_memory`. Encodings follow
  torchaudio: "PCM_S", "PCM_U", "PCM_F", "ULAW", "ALAW", "UNKNOWN" (ADPCM, bits 4), "FLAC" (bits from STREAMINFO),
  "MP3" and "VORBIS" (bits 0). num_frames is the decodable length: MP3 uses dr_mp3's count (Xing/LAME tag minus
  delay and padding, else a header scan), and a FLAC stream whose STREAMINFO length is 0 (unknown) is counted by one
  decode pass so that seeking and slicing work.
- Why: design item 2. All fixture fields equal torchaudio.info except the documented MP3 length (D7) and the
  unknown-length FLAC case (torchaudio reports 0 there).
- Review additions: a Vorbis stream whose last page stb_vorbis cannot find (it only looks in the final 64 KB, so
  e.g. > 64 KB of trailing junk) reported 0 frames and loaded as [C, 0] (PR-3). It is now counted by decoding once,
  and seeks for it decode from the start (stb_vorbis cannot seek without the length). 32-bit FLAC throws
  "Unsupported 32-bit FLAC" from `info` and `load` (SEC-2): dr_flac reads the 32-bit frame-header code as reserved
  and decoded no frame, so it loaded as [C, 0] without an error. num_frames stays the header's claim for corrupt
  files (e.g. an MP3 Xing frame count of 1e6 on a 1199-frame file); `load_audio` returns what decodes.

### D7 — MP3 / Vorbis parity with torchaudio (measured, not faked)
- Decision: assert the measured agreement, document the length differences.
  | file | ours (frames) | torchaudio sox (frames) | alignment | SNR | max abs diff |
  |---|---|---|---|---|---|
  | fixture mp3_mono, 8 kHz, LAME tag + ID3v2 | 800 (= source) | 2880 | ours[n] = ta[n + 1681] | 131.8 dB | 3.0e-7 |
  | fixture mp3_stereo, 8 kHz, no tag | 2304 | 1728 | lag 0 | 131.0 dB | 3.4e-7 |
  | 3 s 44.1 kHz stereo, CBR 128, LAME tag | 132300 (= source) | 133632 | lag 2257, overlap 131375 | 129.7 dB | 6.4e-7 |
  | same, VBR -V 2 (Xing) | 132300 (= source) | 133623 | lag 2257, overlap 131366 | 130.0 dB | 5.5e-7 |
  | same, CBR 128, no tag (`lame -t`) | 133632 | 132480 | lag 0 | 129.7 dB | 6.4e-7 |
  | fixture ogg_mono / ogg_stereo, 8 kHz | 800 | 800 | lag 0 | 93.0 / 92.2 dB | 1.53e-5 |
  | 3 s 44.1 kHz stereo Vorbis q3 | 132300 | 132300 | lag 0 | 87.7 dB | 1.53e-5 |
  | fixture ogg_51, 44.1 kHz 6 ch (5.1 mapping), after the D1 patch | 220 | 220 | lag 0 | 78-90 dB / ch | 1.53e-5 |
- Why: dr_mp3 is gapless: with a LAME/Xing tag it drops the encoder delay + 529-frame decoder delay and the padding,
  returning exactly the source length. libmad under sox emits the Info frame as silence plus both delays (1681 frames
  at 8 kHz MPEG-2.5 = 576 + 576 + 529; 2257 at 44.1 kHz = 1152 + 576 + 529), keeps the padding, and never emits
  the last MP3 frame, so with a tag its output also misses the last 925 source frames of the 44.1 kHz file. Where both
  decode the same frames they agree to ~130 dB. Vorbis: torchaudio's sox reads libvorbis through 16-bit `ov_read`
  (round half up), so its output sits on the 2^-15 grid; rounding stb_vorbis's float output the same way reproduces it
  exactly on the fixtures and on 99.95 % of the 3 s file samples (max 1 LSB).
- Impact: main.cpp asserts MP3 max diff <= 1e-6 and SNR > 120 dB on the overlap with the stated lengths and lag, and
  Vorbis max diff <= 2^-15 and SNR > 85 dB with equal length.
- Clipping (review PR-7): the figures above hold for content within full scale. torchaudio's sox path clips decoded
  MP3 and Vorbis to [-1, 1]; we return the decoders' overshoot, like float WAV (D4). Loud sources (0.99 peaks plus
  noise; reviewer's measurement): MP3 62-65 dB, max diff 0.085 (our peak 1.085); Vorbis 28-67 dB, max diff 0.53
  (peak 1.53). On samples with |x| < 0.999 the max diff is 6.6e-7 (MP3) and 1.5e-5 (Vorbis), so clipping is the whole
  gap; `torch::clamp(x, -1, 1)` reproduces torchaudio there. The fixtures stay below full scale.

### D8 — save_audio mirrors torchaudio.save for WAV (fixes G4-7, G4-8, G4-13)
- Decision: `save_option{_encoding = "", _bits_per_sample = 0}` with setters; torchaudio's derivation (measured):
  bits alone pick "PCM_U" for 8 and "PCM_S" otherwise; neither picks from the dtype (float32 -> 32-bit float, int32
  -> PCM_S 32, int16 -> PCM_S 16, uint8 -> PCM_U 8); "PCM_S" alone is 16-bit, "PCM_U" 8, "PCM_F" 32. Accepted:
  PCM_S 16/24/32, PCM_U 8, PCM_F 32/64 for float32 / int32 / int16 / uint8 tensors. PCM is quantized as torchaudio's
  sox path does: float -> int32 by x * 2^31 clamped and truncated (NaN -> 0, as torchaudio gives on arm64), int16
  x * 2^16, uint8 (x - 128) * 2^24, then round half up to the target width with clipping only at the positive end;
  data chunks are byte-identical with torchaudio.save for every dtype x PCM encoding (golden bytes in fixtures.hpp).
  The tensor is converted in blocks of 65536 frames so the int64 / double temporaries stay bounded.
- Behaviour change: the 3-argument `save_audio(tensor, path, sample_rate)` keeps compiling but now writes a 32-bit
  float WAV for float32 input (torchaudio's default), not 16-bit PCM.
- Deviations (each asserted by the test): float32 input is stored exactly in PCM_F 32/64 (torchaudio's sox path clips
  to [-1, 1] and rounds to 2^-24 / 2^-31 steps, e.g. 1.0 -> 0.9999999995 in 64-bit); int32 -> PCM_F 32 rounds to
  nearest (torchaudio rounds half up to 2^-24, so 1 -> 0); "64-bit" signed PCM is rejected (torchaudio silently
  writes 32-bit); ULAW / ALAW are rejected (no G.711 encoder here; torchaudio supports them).
- Validation: false, and no file created, for rank != 2, dtypes other than float32 / int32 / int16 / uint8 (torchaudio
  also rejects float64), 0 or > 65535 channels, sample_rate <= 0 (torchaudio writes 48000 for 0 and a broken file for
  negatives), and every unsupported (encoding, bits) pair. I/O failures still return false.

### D9 — Fixtures embedded as bytes, generated once
- Decision: `unit_test/audio/io/gen_fixtures.py` builds 26 fixtures (25 of <= 3.3 KB at 8 kHz: WAV U8 / S16 / S24 /
  S32 / F32 / F64 and FLAC 16 / 24, each mono and stereo, from a closed-form integer ramp; G.711 mu-law / A-law,
  IMA ADPCM, W64, RF64 from the S16 one; MP3 and Vorbis mono / stereo from two tones; plus `ogg_51`, 220 frames of
  44.1 kHz 6-channel Vorbis, 7.5 KB because libvorbis's 5.1 setup header alone is ~7 KB) and one refused file
  (`flac_32_mono`, 918 bytes, a 32-bit FLAC of the S32 ramp, emitted outside `fixtures[]`) with its own RIFF writer (sox
  cannot write exact float WAV: it converts through int32), sox, flac, lame and ffmpeg, and embeds them as byte arrays
  in `fixtures.hpp` with torchaudio.info and, where main.cpp has no closed form, torchaudio.load as integers k / 2^24
  (exact: the sox backend's output is on that grid). It also embeds torchaudio.save golden data chunks and its
  (encoding, bits) derivation table. `fixtures.hpp` is ~256 KB of generated text.
- Why: the repo ignores `*.wav` / `*.mp3`; the test must not need Python or encoders.

## Tasks
- [x] [task20 — vendor dr_flac / dr_mp3 / stb_vorbis, re-pin dr_wav, isolate stb_vorbis macros](tasks/task20-vendor-flac-mp3-vorbis.md)
- [x] [task21 — load_audio: container detection, frame ranges, memory input, info()](tasks/task21-load-formats-ranges-info.md)
- [x] [task22 — save_audio: torchaudio.save encodings and bit-exact PCM quantization](tasks/task22-save-torchaudio-parity.md)

## Issues / Gotchas
- `Tensor::to(kFloat32, ..., MemoryFormat::Contiguous)` on a transposed 2-D float tensor returns it unchanged
  (`suggest_memory_format()` is Contiguous for 2-D), so stereo MP3 / Vorbis / float WAV first came back non-contiguous.
  `torch::empty({C, N}).copy_(raw.t())` makes the [C, N] result in one copy for every dtype.
- `.contiguous()` keeps the stride of a size-1 dim, so `view(kUInt8)` on a mono [N, 1] int32 block failed; the 24-bit
  packer uses `reshape(-1)` first.
- `torch::tensor(std::vector<int32_t>)` builds an int64 tensor in libtorch 2.5.1; pass the dtype explicitly.
- sox needs a file extension for W64 (and to pick other readers): gen_fixtures.py writes `<name>.<ext>`; main.cpp
  writes `<name>.bin` on purpose.
- sox writes a random Ogg stream serial, so rerunning gen_fixtures.py changes the Ogg bytes (and their goldens) even
  when nothing else changed. MP3 / FLAC / WAV fixtures are reproducible.
- Truncating the Ogg fixtures to 60 % cuts into the Vorbis setup header, so they fail to open (asserted) rather than
  yield a prefix like the other formats.
- Review round (two reviewers: torchaudio parity + robustness; security / vendoring / ODR), all re-checked and fixed
  with a test that fails before and passes after (before = the reviewed headers rebuilt in scratch):
  - PR-1 (high) 5.1 Vorbis decoded to garbage: stb_vorbis patch 1 (D1); fixture `ogg_51` (before: max diff 91).
  - PR-2 / SEC-1 (high) a corrupt Vorbis comment header crashed (SIGSEGV / SIGABRT) in info / load / memory load:
    stb_vorbis patch 2 (D1); `test_corrupt_and_unusual_files` corrupts the comment count (0x4F000001, 1000), the
    vendor length and a comment length of the ogg_mono fixture (before: SIGSEGV, rc 139). All 9 reviewer repro files
    now throw std::runtime_error in 0.13 s (5 runs x 3 modes each; before: 3 crashed every run, 2 intermittently).
  - PR-3 (medium) Ogg + > 64 KB trailing junk loaded as [C, 0]: counted by decoding (D6); test appends 70000 bytes.
  - PR-4 / SEC-3 (medium) header-sized up-front allocation: bounded blocks (D5); test sets the RF64 sampleCount to
    2^50 (before: c10::Error, 4.5 PB) and the FLAC total to 2^36-1.
  - PR-5 (medium) ID3v2 + FLAC rejected: D3; test prepends a 660-byte v2.4 tag with footer (path, memory, slice).
  - PR-6 (low) zero padding before an MP3 frame rejected: D3; test prepends 64 zero bytes to mp3_stereo (bit-equal).
  - PR-7 (low) MP3 / Vorbis clipping undocumented: D7 and the load_audio comment (documentation only).
  - SEC-2 (medium) 32-bit FLAC loaded as [C, 0]: refused (D6); fixture `flac_32_mono`.
  - SEC-4 (low) no implementation opt-out, STB_VORBIS_HEADER_ONLY clobbered: D2; `audio_test_io_optout`.
  - SEC-5 (low) M_PI leaked as a float literal on platforms without it: pushed / popped (D2). Checked with a scratch
    TU that `#undef`s M_PI before the include (before: `static_assert(sizeof(M_PI) == sizeof(double))` failed); no
    ctest, since it needs a platform without M_PI.
  - SEC-6 (low) audio_test_io left ~27 files in the working directory: main() now runs in a fresh
    `temp_directory_path()/torchmedia_audio_test_io_<random>` and removes it.

## Open / TODO (carry-over)
- Re-pin dr_libs to release tags (wav / flac / mp3) once tags newer than b4264c5 exist.
- Not detected / not decoded: AIFF / RIFX (dr_wav supports them), Ogg FLAC, Opus, AAC (FFmpeg path, progress01 D5).
- Not implemented from torchaudio.load / save: normalize=False (raw integers), channels_first=False, format hints,
  file-like objects, ULAW / ALAW encoding, non-WAV save.
- MP3 seeking is a decode from the start (dr_mp3 brute force) and info() of an untagged MP3 scans every frame header;
  `drmp3_calculate_seek_points` could bound repeated slicing of long files if that becomes hot.
- stb_vorbis counts samples in `unsigned int`: streams beyond 2^32 frames are not supported.
- stb_vorbis is patched locally (D1): re-apply when re-vendoring. UBSan still reports a float-to-int overflow in its
  `lookup1_values` (stb_vorbis.c:1244, `(int) floor(pow(r+1, dim))` with a huge `dim`) on corrupt codebooks; the
  result is a wrong but bounded value count that the setup then rejects or decodes as garbage, and ASan fuzzing
  (~13.6k Ogg mutations after the patches, plus the reviewers' runs) found no memory error. Left unpatched.
- 32-bit FLAC is refused (dr_flac lacks the frame-header code); torchaudio reads it.
- A zero-padded MP3 is decoded without gapless trimming (D3); an ID3v2-tagged FLAC opened by path is read into
  memory whole; a Vorbis file whose last page stb_vorbis cannot find seeks by decoding from the start.
- info().num_frames is the header's claim for corrupt files (e.g. an MP3 Xing count far beyond the data); load_audio
  returns what decodes. Clamping it would need a decode pass.
- Not covered by tests (branches, not lines): a short write / failing `drwav_uninit` in save_audio, a big-endian host
  (24-bit packing picks the host-order bytes), an ID3-tagged FLAC file that shrinks between detection and decoding.
- dr_wav's write side still writes a plain PCM fmt chunk for 24 / 32-bit (torchaudio / sox write
  WAVE_FORMAT_EXTENSIBLE); every reader tested here accepts both.

## Agent log
- 2026-10-23 [ai] Implemented D1-D9 (task20-22). Probed torchaudio 2.5.1 first: save quantization (emulated
  bit-exactly), the (encoding, bits) table, validation, frame_offset / num_frames semantics, and the sox decode paths
  (int32 round trip, 16-bit Vorbis, libmad delays). Tried the dr_libs release tags first; dr_mp3 0.7.3 returned no
  frames after a seek on a LAME-tagged file opened from disk, so all dr_libs files were pinned to b4264c5 instead.
  Release build green; ctest 7/7 (audio_test_io 1066/1066 checks). Coverage (TORCHMEDIA_COVERAGE, audio_test_io,
  `_vendor/` excluded): `_io.hpp` 248/248 lines, 35/35 functions, 170/175 branches (the five listed under Open);
  the union over the audio binaries (progress02 D11) is therefore 100 % for `_io.hpp`. git clang-format clean on the
  changed files (vendored files untouched). torchaudio.load / info read the files save_audio wrote in the test
  (float, 24-bit, int16 / int32 / uint8 defaults, empty). audio_test_io is clean under ASan + UBSan. Not committed.
- 2026-10-23 [ai] Review round (fixer): re-checked the 13 findings of the parity / robustness and security / vendoring
  reviews against the reviewers' repro files and fixed all 13 (Issues above; stb_vorbis patched in two places, D1).
  Every fix has a check that fails on the reviewed headers rebuilt in scratch and passes now (SEC-5: a scratch TU,
  PR-7: documentation). gen_fixtures.py rerun: adds `ogg_51` and `flac_32_mono`; only the Ogg byte arrays of
  ogg_mono / ogg_stereo changed (random sox stream serial), their goldens did not. Release build green; ctest 8/8
  (audio_test_io 1118/1118 checks, audio_test_io_optout 26/26). Coverage (TORCHMEDIA_COVERAGE, `_vendor/`
  excluded): `_io.hpp` 313/313 lines, 38/38 functions, 208/213 branches (the Open items). ASan + UBSan:
  audio_test_io and audio_test_io_optout clean; the security reviewer's in-process fuzzer, rebuilt on the patched
  headers, ran ~13.6k mutated Ogg inputs (anywhere, and from the comment page on) plus 3000 magic-prefixed random
  inputs and 3000 mutations spread over all fixtures: no ASan report, every failure a std::runtime_error, worst input
  0.3 s; UBSan reported only the known `lookup1_values` float cast (Open). git clang-format clean on the changed lines (vendored files not
  reformatted). Not committed.
