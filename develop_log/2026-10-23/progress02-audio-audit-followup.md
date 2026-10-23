# Progress 02 — Audio audit follow-up (verified defects)
id: 2026-10-23/progress02
date: 2026-10-23
author: ai
status: active
refs: [2026-05-30/progress01, 2026-05-31/progress01, 2026-06-01/progress01, 2026-06-01/progress02]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_io.hpp
  - unit_test/audio/io/main.cpp

## Goal
Fix the audio defects confirmed by the 2026-10-23 audit (4 parity probes vs torchaudio 2.5.1, each finding
independently re-reproduced by an adversarial verifier, plus a completeness critic). Each fix lands red-test first.

## Context / Motivation
The audit ran against 4f7a460, i.e. BEFORE the 2026-06-01 functional/transform commits (they had not been pulled
on this machine). Findings in code those commits rewrote (e.g. mel_scale rank collapse, missing inverse_spectrogram,
no transform classes) are stale and must be re-verified against the current tree before fixing. `_io.hpp` was not
touched by those commits, so its findings stand as measured.

## Decisions

### D1 — load_audio must always copy out of the decoder buffer (audit G4-1, high)
- Decision: `transpose(0, 1).clone(torch::MemoryFormat::Contiguous)` instead of `.contiguous()`.
- Why: for 1 channel ([frames, 1]) or 1 frame ([1, channels]) the transpose is already contiguous, so
  `.contiguous()` returned a view of the dr_wav buffer that is freed on the next line. Verified by ASan
  (heap-use-after-free at `drwav_free`) and in Release by corrupted first frames of small mono files
  (allocator free-list metadata). Mono is the normal ASR input. The 2026-05-30/progress01 gotcha predicted this.
- Impact: one extra copy is already what the multi-channel path paid; behaviour otherwise unchanged.

## Tasks
- [x] [task02 — fix the mono / single-frame load_audio use-after-free](tasks/task02-fix-mono-load-uaf.md)
- [ ] re-verify the remaining audit findings against the post-2026-06-01 tree, then fix the confirmed ones

## Issues / Gotchas
- A 1 s mono round trip did NOT catch the bug: a 64 KB freed block is left intact. The regression test uses a
  4000-frame mono file with a non-zero first sample plus a [2, 1] single-frame file, which fail without the fix
  (max err 0.5) and pass with it; the same test is clean under ASan.

## Open / TODO (carry-over)
- Audit findings to re-verify on the current tree: amplitude_to_DB top_db axes (G1-1) and default db_multiplier
  (G1-2); spectrogram frame_length normalization (G2-1) and >=3-D input (G2-3); griffinlim length in the loop
  (G2-2); melspectrogram/mfcc device+dtype (G3-2), filter-bank speed (G3-3), mfcc_option.sample_rate (G3-7),
  odd sample rates (G3-9); resample same-rate/validation/dtype (G4-2..G4-5); io PCM16 bias, 8-bit mapping,
  save validation (G4-7, G4-9, G4-13); header hygiene (unscoped enums, reserved guard names).

## Agent log
- 2026-10-23 [ai] Audit workflow (9 agents) done; G4-1 confirmed by probe + verifier. Fixed red->green, ASan clean,
  ctest green.
