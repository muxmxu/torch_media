# Task 51 — phaser / flanger scalar loops; device input on the host loops
id: 2026-10-23/task51
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Run phaser and flanger at compiled speed, bit-exact with torchaudio 2.5.1, keep autograd, and make them work on MPS
(progress04 loop note).

## Findings
- torchaudio 2.5.1 has no compiled phaser / flanger op: both are Python loops over tensors. The C++ tensor-op loops
  took 202 ms / 1119 ms on 4 s of 16 kHz audio and raised on MPS (float64 wave table).
- flanger with speed > sample_rate or speed 0, or a delay buffer < 1 sample, hit `int64 % 0` / an undefined cast.
- New (D3): libtorch 2.5.1's MPS backend corrupts > ~64k queued index_put_ writes, so the tensor-op loops of lfilter,
  overdrive, phaser and flanger returned garbage on long MPS input (torchaudio too); lfilter's MPS conv1d raises past
  65536 samples.

## Scope
In: `_functional_filtering.hpp` (`_generate_wave_table`, phaser, flanger, `_lfilter_core`, overdrive, `detail::`
helpers), functional and noexcept tests, gen_golden.py.

## Deliverables
- Wave table computed on the CPU, moved to the requested device.
- phaser / flanger: `AT_DISPATCH_FLOATING_TYPES` + `at::parallel_for` scalar loops for float32/float64 without grad
  (separate statements per torchaudio op, Python-style remainders, float32 channel phase); otherwise the tensor-op loop.
- flanger validation (std::invalid_argument).
- `detail::loop_on_host`: lfilter, overdrive, phaser, flanger run float32/float64 non-grad input from another device
  on a host copy. `detail::mps_flush`: the tensor-op loops sync MPS every 8192 steps.
- Tests `test_p8_phaser_flanger_loops` (goldens + float64 sums incl. negative LFO / phase and the float32 channel
  phase, autograd backward, float16, timing < 50 ms, degenerate flanger raises), `test_p8_device_input_runs_host_loops`
  (70000-sample MPS lowpass / overdrive / phaser / flanger equal to CPU, flanger < 100 ms, an 8200-sample autograd run
  on MPS); the p4 lfilter / overdrive MPS checks tightened to equality; noexcept `test_p8_sites`.

## Acceptance criteria
- [x] Red: both MPS calls raised, both timing checks failed, speed 20000 / speed 0 escaped as c10::Error; lfilter /
  overdrive MPS not equal to CPU, 70000-sample MPS lowpass raised.
- [x] Green; probe (afollow/impl/probe_mod) 11 / 11 bit-exact, 4 s phaser 0.37 ms, flanger 0.40 ms; MPS 8 / 8 exact.
- [x] Mutants killed: fma in the linear step, C `%`, float64 decay product, float64 channel phase.
