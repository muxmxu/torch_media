# Task 52 — PitchShift kernel cache per dtype / device
id: 2026-10-23/task52
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Stop transform::PitchShift from rebuilding its resample kernel on every non-float32 call (A2-11).

## Scope
In: `_transform_time.hpp` (PitchShift), transform test.

## Deliverables
- `kernel_cache` (mutex + list) behind a `shared_ptr`, seeded with the eager float32 CPU kernel; `kernel_for` builds a
  missing (dtype, device) kernel in the input's options on first use (torchaudio's lazy init).
- `test_p8_pitch_shift_kernel_cache`: a second float64 call at least 4x faster than the first (8 kHz / 0.5 step),
  float32 equal to functional::pitch_shift, float64 equal to a fresh PitchShift (review D9: its window is float64
  now), a copy gives the same output.

## Acceptance criteria
- [x] Red on 080ae69: `second * 4 < first` failed (62-77 ms per call).
- [x] Green: 90.8 ms first call, 3.7-4.0 ms after; mutant without the cache lookup fails.
