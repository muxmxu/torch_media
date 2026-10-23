# Task 18 — forced_align: port torchaudio's CPU kernel and checks; merge_tokens checks
id: 2026-10-23/task18
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
forced_align returns torchaudio's paths and scores for every dtype and tie pattern, and raises where torchaudio raises.

## Findings
- A3-1 (medium): no T >= L + R check (all-blank path instead of an error). The 2026-06-01 task15 acceptance claimed it.
- A3-2 (low): different tie rules (inner and terminal).
- A3-3 (low): float32 cast + double accumulation instead of the input dtype (float64 / float16 paths differed).
- A3-4 (medium): blank not range-checked (negative blank read before the buffer).
- A3-5 (medium): shapes/lengths not enforced (2-row targets flattened, lengths ignored, 2-D log_probs accepted).
- A3-9 (low): merge_tokens accepted unequal lengths (NaN spans) and non-1-D input.
- A3-10 (low): paths always int64. A3-11 (low): int64 vector<vector> backpointers (8x memory).
- A3-15 (medium): two hand-built test cases only.

## Scope
In: `_functional_alignment.hpp`, functional/noexcept tests, gen_golden.py.

## Deliverables
- Port of compute.cpp's `forced_align_impl`: start/end window, tie rules, terminal rule,
  `AT_DISPATCH_FLOATING_TYPES_AND_HALF`, int8 backpointers. All Python and kernel checks as std::invalid_argument.
  Paths in the targets' dtype/device. Non-contiguous / non-CPU log_probs copied (relaxed); negative targets raise
  (stricter).
- merge_tokens: 1-D, equal-length tokens/scores (breaking for (1, T) input; justification in progress04 D5).
- Tests `test_p4_forced_align_parity` (uniform ties, T == L + R, float64 1e-10 noise, float16, blank 2 with int32
  targets and score goldens, matching lengths) and `test_p4_forced_align_validation` (12 raises + 2 merge_tokens raises).
  Noexcept checks for every new site.

## Acceptance criteria
- [x] Red on 4eb755b: tie, float64, float16 and int32 goldens failed; every validation check failed until blank 5
  aborted the function with a c10::Error.
- [x] Green; A3 probe: 575/575 paths equal torchaudio (was 122 mismatches), scores exact. T=10000/L=2000 peak RSS
  433.7 MB -> 144 MB; T=3000/L=400 2.79 ms (torchaudio 9.67 ms).
