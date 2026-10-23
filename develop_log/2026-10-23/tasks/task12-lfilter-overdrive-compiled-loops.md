# Task 12 — lfilter / overdrive: torchaudio's compiled scalar loops on the CPU
id: 2026-10-23/task12
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
Remove the ~5 us-per-sample tensor-op recurrence from lfilter (and everything built on it) and overdrive, matching
torchaudio's compiled output bit-exactly, without losing autograd or non-CPU support.

## Findings
- A1-01 (high): lfilter 500-1300x slower than torchaudio (2 s mono 161 ms vs 0.23 ms; 10 s 806 ms).
- A1-05 (medium): overdrive ~230x slower (2 s stereo 163 ms vs 0.70 ms).
- A1-14 (part): no float64 / 3-D / long-signal / speed coverage for lfilter.

## Scope
In: `_functional_filtering.hpp` (`_lfilter_core`, `overdrive`), functional tests, gen_golden.py.
Out: biquad designer coefficient precision (A1-10, open); phaser/flanger loops (not audited findings).

## Inputs
1. torchaudio `functional/filtering.py:925-997, 1102-1170`; libtorchaudio `lfilter.cpp` / `overdrive.cpp` semantics.
2. `surface/probe-A1-filtering-effects` (`bench.cpp` scalar loop, `compare.py`).

## Deliverables
- `_lfilter_core`: FIR with (b / a0) through a grouped conv1d, then the `AT_DISPATCH_FLOATING_TYPES` +
  `at::parallel_for` recurrence with `std::fma(-y, c, acc)` for CPU float32/float64 without grad. Otherwise the
  generic tensor-op loop.
- `overdrive`: accessor loop with overdrive.cpp's double-literal arithmetic, under the same condition. Otherwise the
  generic loop.
- Tests `test_p4_lfilter_compiled_loop`, `test_p4_overdrive_compiled_loop`: torchaudio goldens on a (2, 3, 4000)
  input with per-channel filters (a0 = 2 and a double pole at 0.99), float64 and float32. The grad path must agree
  with the fast path and run backward. MPS equals CPU when MPS is available. A 10 s run must take < 100 ms (it took
  ~800 ms).
- Review (2026-10-23): those order-2 goldens did not pin the fma or the b/a0-before-conv1d order (a0 = 2 divides
  exactly, and this compiler gives the same float32 values with a plain `acc -= y * c` there): both mutants passed.
  Added an order-6 a0 = 1.7 filter (butter(5, 0.1) * 1.7) with bit-exact float32 goldens; both mutants now fail.

## Acceptance criteria
- [x] Red on 4eb755b: float32 goldens off by 1.9e-5 / 4.95e-4; both timing checks failed; overdrive float32 goldens off.
- [x] Green; A1 probe: all 12 lfilter, 4 filtfilt and 5 overdrive cases bit-exact vs torchaudio (float32 and float64).
- [x] Timing after: lfilter 2 s mono 0.2 ms, 10 s 1.1 ms, overdrive 2 s stereo 0.3 ms (torchaudio 0.19-0.23 / 0.80 / 0.56-0.70).

## Notes
- Without the fma, float32 was 1.6e-7..8.3e-7 of peak off on order-2/3 filters and 3e-2 on an order-6 one: the arm64
  wheel's loop is FMA-contracted (numpy model check in progress04 Issues).
