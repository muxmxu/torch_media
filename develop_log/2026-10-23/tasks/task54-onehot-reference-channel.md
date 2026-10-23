# Task 54 — one-hot Tensor reference_channel
id: 2026-10-23/task54
parent: 2026-10-23/progress08
status: done
owner: code_agent

## Objective
Accept torchaudio's `Union[int, Tensor]` reference_channel in rtf_power, SoudenMVDR and RTFMVDR (A3-12).

## Scope
In: `_functional_beamforming.hpp`, `_transform_beamform.hpp`, functional / transform / noexcept tests.

## Deliverables
- `detail::rtf_power` (shared body with a column selector) behind the int64_t and Tensor overloads; the Tensor one
  is torchaudio's `einsum("...c,...c->...", [phi, ref.to(psd_n.dtype)[..., None, None, :]])`.
- SoudenMVDR / RTFMVDR `operator()` and `forward` templated on the reference type.
- MVDR stv_power passes its one-hot u, as torchaudio's `_get_mvdr_vector`.
- Tests: `test_p8_rtf_power_onehot` (n_iter 4 and 1 goldens, int equals one-hot, validation), one-hot cases in
  `test_p8_souden_rtf_mvdr_goldens`, noexcept checks.

## Acceptance criteria
- [x] Red: the one-hot calls did not compile.
- [x] Green: torchaudio goldens bit-exact; conj(phi) mutant fails.
