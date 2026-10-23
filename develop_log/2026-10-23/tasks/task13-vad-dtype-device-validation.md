# Task 13 — vad: float32 measurement state like torchaudio, validated loop sizes
id: 2026-10-23/task13
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
vad accepts float64 and non-CPU input with torchaudio's trim, and rejects arguments that would index empty buffers or
never terminate.

## Findings
- A1-02 (medium): float64 threw `expected scalar type Float but found Double`; MPS input crashed the process at exit.
- A1-04 (medium): search_time <= 0 segfaulted; measure_freq 0 (or > 2 x sample_rate) never terminated.
- A1-14 (part): vad tested on one float32 CPU mono tone only.

## Scope
In: `_functional_filtering.hpp` (`_measure`, `vad`), functional/transform/noexcept tests, gen_golden.py.
Out: vad performance (already faster than torchaudio).

## Deliverables
- Windows, spectra, mults float32 on the input's device; `measures` / `mean_meas` float32 on the host; window product
  rounded to float32; float32 mean update and trigger comparisons.
- One up-front check (period >= 1 sample, >= 1 measure, finite measure_duration, positive lifter frequencies,
  dim >= 1) returning `handle_exceptions<tensor_t, std::invalid_argument>`.
- Tests `test_p4_vad_dtype_device` (stereo float32/float64 with pre-trigger -> torchaudio's 18400; 1-D float64; MPS),
  `test_p4_vad_validation`, transform Vad float64 and search_time 0, a noexcept check.

## Acceptance criteria
- [x] Red on 4eb755b: float64 threw (c10::Error), search_time 0 SIGSEGV.
- [x] Green; A1 probe vad rows (1-D/2-D/3-D, float64, 8/16/44.1 kHz, pre-trigger) bit-exact; vad on MPS exits 0.
