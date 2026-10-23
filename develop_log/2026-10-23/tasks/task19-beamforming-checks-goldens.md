# Task 19 — beamforming: torchaudio's shape checks, no-exceptions returns, complex128 goldens
id: 2026-10-23/task19
parent: 2026-10-23/progress04
status: done
owner: code_agent

## Objective
The beamforming functionals reject the inputs torchaudio rejects, stop in the no-exceptions build, and are pinned to
torchaudio on batched complex128 input.

## Findings
- A3-6 (medium): psd accepted a per-channel or broadcastable mask (wrong rank or values).
- A3-7 (low): mvdr_weights_rtf (both overloads) validated nothing.
- A3-8 (low): apply_beamforming broadcast mismatched leading dims.
- A3-13 (low): frechet_distance's stricter check undocumented.
- A3-14 (low): `detail::assert_psd_matrices` was void and MVDR `(void)`-discarded its checks.
- A3-16 (part): one complex64 unbatched golden at ~1e-3.

## Scope
In: `_functional_beamforming.hpp`, `_transform_beamform.hpp` (MVDR), `_functional.hpp` (frechet comment), tests,
gen_golden.py.
Out: one-hot Tensor overloads for rtf_power / SoudenMVDR / RTFMVDR (A3-12, open); torchaudio goldens for the
transform MVDR tests (A3-16 rest).

## Deliverables
- `detail::psd_matrices_error` and `detail::rtf_psd_n_error` (return a message; callers return handle_exceptions),
  psd mask check, apply_beamforming leading-dims check (before the complex check, torchaudio order), MVDR returns.
- Test `test_p4_beamforming_goldens_and_validation`: complex128 batch-2 goldens for psd (unnormalized mask),
  mvdr_weights_souden and rtf_power(n_iter 5) + mvdr_weights_rtf at 1e-12, plus 9 raises. Noexcept checks for
  souden/rtf_power/rtf weights/psd/apply_beamforming/MVDR.

## Acceptance criteria
- [x] Red on 4eb755b: psd and mvdr_weights_rtf checks failed, then a non-square psd_n escaped as a c10::Error.
- [x] Green; A3 probe: every beamforming output (87 rows) bit-exact and every torchaudio raise mapped to
  std::invalid_argument.
