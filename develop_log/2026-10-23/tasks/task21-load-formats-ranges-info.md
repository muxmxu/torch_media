# Task 21 — load_audio: container detection, frame ranges, memory input, info()
id: 2026-10-23/task21
parent: 2026-10-23/progress05
status: done
owner: code_agent

## Objective
`load_audio` decodes WAV (RIFF / RF64 / W64, every dr_wav encoding), FLAC, MP3 and Ogg Vorbis detected by magic bytes,
honours torchaudio's frame_offset / num_frames, reads from memory too, and `info()` mirrors torchaudio.info.

## Scope
In:
- `_audio/_io.hpp`: `load_option`, `audio_info`, `detail::{source, decoder, wav/flac/mp3/vorbis_decoder, open, load}`,
  `load_audio(path, opt)` (string / const char* / filesystem::path), `load_audio_from_memory`, `info`, `info_from_memory`.
- `unit_test/audio/io/{gen_fixtures.py, fixtures.hpp, main.cpp}` (decode, slicing, info, truncation, error paths).
Out:
- normalize=False, channels_first, format hints, file-like objects, resampling (progress05 Open).

## Inputs (read first, priority order)
1. `scratchpad/audit_findings_confirmed.json` G4-9 (8-bit mapping), G4-10 (float WAV), G4-16 (formats, ranges, info).
2. torchaudio 2.5.1 sox backend behaviour, measured in `.venv` (frame range validation, int32 round trip, 16-bit
   Vorbis, libmad delays) — progress05 D4-D7.

Code to inspect/change:
- `libtorchmedia/include/torchmedia/_audio/_io.hpp`, `unit_test/audio/io/*`

## Deliverables
- Decoders read into torch-owned buffers (int32 full scale for non-float WAV and FLAC, float32 otherwise); one copy to
  a contiguous [C, N] float32 result.
- Validation before opening: frame_offset >= 0, num_frames -1 or > 0 (std::invalid_argument); missing / unsupported /
  undecodable input throws std::runtime_error naming the path ("<memory>" for buffers).
- `gen_fixtures.py` + generated `fixtures.hpp` (25 fixtures, torchaudio.info, torchaudio.load goldens as k / 2^24).

## Acceptance criteria
- [x] Closed-form fixtures (WAV U8 / S16 / S24 / S32 / F32 / F64, FLAC 16 / 24, W64, RF64; mono and stereo) decode
  bit-exactly to pcm24() / pcm32() values; gen_fixtures.py asserts torchaudio.load gives the same for the integer
  <= 24-bit ones and prints the sox deviation for S32 / F32 / F64.
- [x] mu-law, A-law and IMA ADPCM fixtures equal torchaudio.load bit for bit; info() fields equal torchaudio.info.
- [x] MP3: overlap with torchaudio within 1e-6 and SNR > 120 dB at the documented lag; lengths 800 (gapless, tag) and
  torchaudio + 576 (no tag). Vorbis: same length, within 2^-15, SNR > 85 dB.
- [x] For every fixture and nine (offset, count) ranges incl. clipped and past-the-end ones, path and memory slices
  equal the slice of the full decode.
- [x] FLAC with STREAMINFO length 0 decodes, reports 200 frames and slices; truncated files yield a prefix (or, for
  Ogg cut inside the setup header, runtime_error); offset beyond a truncated stream gives [C, 0].
- [x] Error paths: invalid ranges (path and memory), missing file (load and info), text file, empty buffer, RIFF / W64
  that is not WAVE, bare Ogg page, Opus-like Ogg, ADTS, 0xFF without sync, garbage after each decoder's signature
  (path and memory), Vorbis buffer > INT_MAX.

## Constraints
- Detection by content only; the tests load extension-less `<name>.bin` files.

## Notes / Assumptions
- Deviation (progress05 D5): num_frames 0 throws like torchaudio, although the design listed only < -1.
- Deviation (progress05 D4): S32 / float WAV follow the stored values, not torchaudio's sox int32 round trip.
