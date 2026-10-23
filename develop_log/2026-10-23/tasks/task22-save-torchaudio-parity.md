# Task 22 — save_audio: torchaudio.save encodings and bit-exact PCM quantization
id: 2026-10-23/task22
parent: 2026-10-23/progress05
status: done
owner: code_agent

## Objective
`save_audio` writes WAV like torchaudio.save: same (encoding, bits) derivation and defaults, byte-identical PCM data,
and invalid arguments rejected with false before any file is created.

## Scope
In:
- `_audio/_io.hpp`: `save_option`, `detail::encode_wav_block`, `save_audio(tensor, path, sample_rate, opt = {})` for
  std::string / const char* / filesystem::path (the 3-argument calls keep compiling).
- `unit_test/audio/io/main.cpp` save tests; save goldens in `gen_fixtures.py` / `fixtures.hpp`.
Out:
- ULAW / ALAW and non-WAV encoders; WAVE_FORMAT_EXTENSIBLE headers.

## Inputs (read first, priority order)
1. Audit G4-7 (PCM16 bias), G4-8 (formats / defaults), G4-13 (validation) in `scratchpad/audit_findings_confirmed.json`.
2. torchaudio.save measurements (progress05 D8): rounding, clipping, NaN / inf, dtype paths, (encoding, bits) table.

## Deliverables
- Quantization: float -> int32 (x * 2^31, NaN -> 0, clamp, truncate), int16 x * 2^16, uint8 (x - 128) * 2^24, then
  round half up to 8 / 16 / 24 bits with positive-end clipping; float32 stored exactly as PCM_F; conversion in
  65536-frame blocks.
- Golden data chunks from torchaudio.save for float32 (special values, rounding ties, NaN / inf, 200 uniform values),
  int16, int32 and uint8 inputs in PCM_S 16 / 24 / 32, PCM_U 8, and PCM_F 32 / 64 for int16 / uint8 (int32: 64).

## Acceptance criteria
- [x] Every golden data chunk is byte-identical (21 cases); the derivation table (4 dtypes x 6 encodings x 7 bit
  depths) matches torchaudio.info of torchaudio's file, except the asserted deviations (64-bit PCM_S, ULAW / ALAW).
- [x] PCM_F 32 / 64 store float32 input exactly (default for float32); int32 -> PCM_F 32 is x / 2^31 rounded to nearest.
- [x] Round trip for every encoding within half an LSB of the saturated input; int16 / int32 / uint8 defaults exact
  (uint8 decodes to (x - 128) / 128); a 70001-frame stereo, non-contiguous tensor crosses the block boundary in float
  and 24-bit; a [2, 0] tensor gives an empty valid file.
- [x] false and no file for 1-D, 3-D, float64, int8, 0 channels, 65536 channels, sample_rate 0 / negative, and each
  unsupported (encoding, bits) pair; false for an unwritable path.
- [x] torchaudio.load reads the files save_audio writes (checked by hand on the test outputs).

## Constraints
- Keep the bool return for both invalid arguments and I/O failure (design item 3).

## Notes / Assumptions
- Behaviour change: 3-argument save_audio of a float32 tensor now writes a 32-bit float WAV (was PCM16).
- NaN -> 0 matches torchaudio on arm64 (where the measurement ran); torchaudio's own NaN cast is platform-defined.
