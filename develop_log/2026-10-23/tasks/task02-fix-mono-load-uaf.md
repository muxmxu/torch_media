# Task 02 — Fix the mono / single-frame load_audio use-after-free
id: 2026-10-23/task02
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
`load_audio` returns a tensor that owns its data for every WAV shape.

## Scope
In: `_io.hpp` load path; a regression test in `unit_test/audio/io/main.cpp`.
Out: the other io findings (PCM16 bias, 8-bit mapping, save validation) — later task.

## Deliverables
- `_io.hpp` — `clone(torch::MemoryFormat::Contiguous)` + corrected comment.
- `unit_test/audio/io/main.cpp` — mono (4000 frames, non-zero first sample) and [2, 1] round trips.

## Acceptance criteria
- [x] New test fails before the fix (mono max err 0.5) and passes after.
- [x] `audio_test_io` clean under `-fsanitize=address`.
- [x] ctest green.

## Result
Done 2026-10-23.
