# Progress 03 — Blind source separation: AuxIVA-IP / OverIVA / ILRMA (torchmedia::audio::separation)
id: 2026-10-23/progress03
date: 2026-10-23
author: ai
status: done              # code + tests done, review findings fixed, default load decided (D7)
refs: [2026-06-01/progress02, 2026-10-23/progress01]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_audio/_separation.hpp (NEW)
  - libtorchmedia/include/torchmedia/audio.hpp (include the new header)
  - unit_test/audio/separation/{CMakeLists.txt, main.cpp, gen_golden.py} (NEW ctest target audio_test_separation)
  - unit_test/audio/CMakeLists.txt (wire in the new subdir)
  - THIRD_PARTY_LICENSES (torchiva entry)
  - CLAUDE.md (namespace list + coverage policy line only)

## Goal
Add STFT-domain blind source separation to libtorchmedia: AuxIVA with iterative-projection updates (Laplace / Gauss
source models), its overdetermined variant OverIVA (n_src < n_chan) and ILRMA (NMF source model), as header-only
native-libtorch free functions that match torchiva 0.1.1 point-wise. Out of scope: other torchiva separators
(AuxIVA-IP2, T-ISS, FIVE), beamformers, WPE, and any STFT code (the existing transform layer provides it).

## Context / Motivation
torchaudio has no BSS; libtorchmedia already has the STFT/iSTFT pair (functional + transform layer, 2026-06-01/progress02),
so a separation stage completes a wav -> wav multichannel pipeline:
`wav [C, L] -> transform::Spectrogram(return_complex) -> [C, F, T] -> separation::auxiva_ip -> [S, F, T] ->
transform::InverseSpectrogram -> [S, L]`. An earlier draft (written against an older tree, with its own
`inverse_spectrogram`) was reused for the algorithm core and rewritten for the current tree and option idiom.

## Decisions

### D1 — Namespace, header and shape: free functions in `torchmedia::audio::separation`
- Decision: new header `_audio/_separation.hpp`, included from `audio.hpp`; namespace
  `torchmedia::audio::separation`; helpers in a nested `detail` namespace. Free functions only — no class wrapper.
- Why: separation is neither a torchaudio functional nor a transform; it has no state worth caching across calls
  (the NMF bases/activations are per-signal state, created lazily inside one call), so a class would add nothing.
- Impact: CLAUDE.md namespace list and coverage policy line now name `separation` / `_separation.hpp`.

### D2 — API in the transform-layer option idiom
- Decision:
  `enum class source_model { laplace, gauss, nmf };`
  `auxiva_option { _n_iter = 10; _n_src = -1 (=> n_chan; < n_chan => OverIVA); _model = laplace; _n_basis = 2 (NMF rank);
  _proj_back_mic = 0 (-1 => off); _eps = 1e-5; _load = 1e-8 (D7; torchiva 1e-5) }` with same-name fluent setters returning `*this`;
  `auxiva_ip(const_tensor_lref_t X, const auxiva_option & = {}) -> tensor_t` and
  `ilrma(const_tensor_lref_t X, auxiva_option = {}) -> tensor_t` (= auxiva_ip with model forced to nmf).
  X: complex64/complex128 STFT `[..., n_chan, n_freq, n_frames]` -> `[..., n_src, n_freq, n_frames]`, any batch dims.
- Why: matches the `_field` + fluent-setter convention of the transform layer; enums replace torchiva's model objects.
- Impact: `eps` is the floor of the source models, the variance normalization, the scale resolution and the row-norm
  clamps; `load` is the diagonal load of every solve (D7). torchiva `AuxIVA_IP(eps=e)` with `Model(eps=e)` is
  `eps(e).load(e)` here; gen_golden.py passes the same e to both sides (torchiva's model eps and separator eps both
  default to `eps_models = 1e-5`, so the eps defaults match; the load default differs, D7).
- Validation (`std::invalid_argument`, message names the bad value): `X.dim() < 3`, X not complex64/complex128
  (complex32 has no solver kernels; the message names the dtype), `n_src == 0 || n_src < -1 || n_src > n_chan`,
  `n_iter < 0`, `proj_back_mic` outside `[-1, n_chan)`, `n_basis < 1`, eps or load not finite and > 0 (rejects NaN,
  inf, 0, negatives). Thrown directly (not via `handle_exceptions`) because continuing after an invalid option would
  run the solver on garbage.

### D3 — torchiva 0.1.1 is the porting reference; pyroomacoustics is only a cross-check
- Decision: port torchiva `AuxIVA_IP.forward` step for step (`auxiva_ip.py`, `base.py`, `models/classic.py`,
  `linalg.solve_loaded_general`, `parameters.eps_models`): W = eye(n_src, n_chan); Y = X[..., :n_src]; Cx; per iteration
  the model weights are computed BEFORE the per-source variance normalization g; Y, W, weights rescaled by g; per-source
  IP update via the row-normalized, diagonally loaded `solve_loaded_general(load, eps)` (torchiva: load = eps); scale resolution;
  OverIVA orthogonal constraint J recomputed before every source update; final demix; projection back (with J for
  OverIVA). Same ops in the same order and the same einsum equations (the .venv has no opt_einsum, so Python and C++
  contract einsum operands in the same left-to-right order).
- Why: torchiva is the authors' (Scheibler) tensor-native, batch-capable reference, written in torch ops, so a
  libtorch port can be checked bit for bit. pyroomacoustics (numpy) uses different scaling (V / n_frames, no g
  normalization), no diagonal loading, a least-squares projection back onto the mic signal, and numpy RNG for the ILRMA
  init — useful as an independent algorithmic check, not as a point-wise reference.
- Impact: THIRD_PARTY_LICENSES gains a torchiva (MIT, Copyright (c) 2022 Robin Scheibler, Kohei Saijo) entry and the
  header carries a one-line attribution.

### D4 — Seeded-RNG parity for ILRMA
- Decision: the NMF state is drawn lazily on the first model call exactly as torchiva does —
  `T = 0.9 * zeros([B, F, K], real dtype of X).uniform_() + 0.1`, then V the same way with `[B, N, K]` — from the global
  generator. With the same `torch::manual_seed` the C++ output equals torchiva's (libtorch's CPU generator is PyTorch's).
- Why: makes ILRMA testable point-wise instead of only statistically.
- Impact: callers who want reproducible ILRMA seed the global generator; nothing else in auxiva_ip draws randomness.

### D5 — Deliberate deviations from torchiva
- OverIVA projection back with `proj_back_mic >= n_src`: torchiva broadcasts its already per-frequency `e1` a second
  time over WW's batch shape, so it returns `[..., n_freq, ..., n_src, n_freq, n_frames]` (verified: batch (3, 2),
  F=5 -> output (3, 2, 5, 3, 2, 1, 5, 8)). We return the per-frequency result. torchiva's leading slices equal it only
  up to rounding: bit-exact at small n_freq (F <= 65 measured), but at F=257 (4 mics, n_src 2, N=400, 30 iterations)
  the slices disagree with each other by up to 2.7e-4 in complex64 (1e-13 in complex128) and the C++ result matches
  none of them exactly (max |d| 8.4e-5, rel 1.8e-5) — rounding, not wrong numbers; mic -1/0/1 stay bit-exact on the
  same input. So the goldens for this case stay small (F=17, `[0]` / `[0, 0]` slice); a large-F golden would need a
  relative tolerance. The math was checked: with the completed demixing matrix `[[W1, W2], [J, -I]]` the
  mixing-matrix row m is `e_m^T (W1 + W2 J)^-1` (m < n_src) or `J[m - n_src] (W1 + W2 J)^-1`.
- `n_src = -1` means n_chan; torchiva's `n_src=None` default crashes (`TypeError` building W's shape).
- Projection back (changed in review, findings Q2): torchiva passes `load=eps` to its projection back but ignores it
  (always load 1e-5) and clamps the row norms of W^T at 1e-5. W scales as 1 / |X|, so for int16-scale audio
  (|X| ~ 1e7) that clamp fires and the output loses up to 4 dB SDR (2x2 room, gauss: 15.04 -> 10.89 dB). Ours uses
  `auxiva_option::load` and a 1e-30 row floor (only guards all-zero rows: a dead mic gives exactly-zero W columns).
  At `load(1e-5)` the result is bit-identical to torchiva whenever torchiva's clamp does not fire (every
  unit-scale input measured: 55/55 default-eps reviewer configurations, all 16 goldens); it differs by design when
  load != 1e-5 and projection back is on, so the goldens with non-default eps/load use `proj_back_mic(-1)`.

### D6 — Device / dtype / autograd
- Decision: every internal tensor is built from `X.options()` (eye, NMF init), so the output follows X's device and
  dtype; the whole call runs under `torch::NoGradGuard` (output never requires grad).
- Status: CPU verified. MPS (available on this machine) FAILS inside libtorch: "norm ops are not supported for complex
  yet" (`linalg_vector_norm` on complex, aten/src/ATen/native/mps/operations/ReduceOps.mm) — same failure in torchiva
  on MPS, so not specific to the port. Replacing the norm (sqrt of summed |a|^2) is not enough: the next op,
  `linalg_solve` (`aten::_linalg_solve_ex.result`), is not implemented on MPS either, so MPS also needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1` or a CPU round-trip for the solves (review RC4; the norm rewrite alone kept the CPU
  goldens passing at the test tolerances, bitwise identity with torchiva not re-measured). CUDA: no GPU here,
  untested. A meta-device test proves no stray CPU tensor. The `auxiva_ip` header comment states this.

### D7 — Diagonal load as its own knob (`auxiva_option::load`), default 1e-8 (review Q1)
- Decision: split torchiva's separator eps into `eps` (floors / clamps) and `load` (diagonal load of the IP solve,
  the OverIVA J solve and the projection back). Default `load = 1e-8`; `load(1e-5)` reproduces torchiva bit for bit
  (the parity goldens pass it explicitly via `tiva()`). Decided by the orchestrator under Mux's delegation after the
  fixer left it open: users need separation that works; torchiva parity is a verification tool, not the goal.
- Why: with torchiva's 1e-5 load, AuxIVA / ILRMA do not separate realistic 3-mic compact-array rooms (G2), and the
  diagonal load is the cause (only lowering the load restores pra-level SDR; lowering the model eps or the row-norm
  clamp changes nothing). A single eps knob cannot express the working setting safely: in complex64, lowering eps to
  <= 1e-8 also lowers the Gauss model floor, and together with the (then numerically unloaded) solve the Gauss model
  diverges on the compact-array fixture: max |Y| 1.1e3 at eps 1e-8 and 6.3e5 at 1e-10 (C++, unit-test fixture), all
  NaN at 1e-10 in torchiva on a variant with a 0.05 diffuse term; eps 1e-5 with load 1e-8 or 1e-10 stays at 1.03.
- Evidence for the default: at 1e-8 the measured 3x3 rooms reach pra-level SDR in complex64 as well as complex128
  (table below; no NaN / divergence), 2x2 rooms and OverIVA are unchanged or better; at 1e-5 every 3x3 room fails.
- Impact: new setter `load()`, validated like eps; `test_compact_array_load` pins that torchiva's 1e-5 leaves a
  nearly-coherent compact-array mixture mixed and that the default load separates it (determined and OverIVA).

## Tasks
- [x] [task03 — AuxIVA-IP / OverIVA / ILRMA port + tests](tasks/task03-auxiva-ip.md)

## Measurements (dev-time, 2026-10-23, macOS arm64, libtorch 2.5.1 == .venv torch 2.5.1)
Scripts and dumps: scratchpad `auxiva2/impl/` (dev.cpp dumps C++ outputs as raw complex128; compare_torchiva.py,
compare_pra.py, sens.py, gauss_floor.py, overiva_noisefree.py). Not part of the repo.
- C++ vs torchiva, full output tensors, 15 seeded cases (laplace, gauss, proj_back_mic -1 / 1, n_iter 0, eps 1e-3,
  ILRMA c64 / c128 rank 3 / batch, OverIVA mic 0 / mic 2 / batch mic 2 / n_src 1 NMF, c128 laplace, batch laplace):
  **bit-identical (max |d| = 0)** in all 15. Thread count (1/4/10 on either side) does not change this.
- Rounding sensitivity (same values, transposed memory layout -> different reduction order, torchiva vs itself):
  c64 max |d| 0.76e-6 .. 2.3e-6, c128 ~1.2e-15. Test tolerances: 2e-5 (c64), 1e-10 (c128).
- vs pyroomacoustics 0.10.1 (`pra.bss.auxiva` / `pra.bss.ilrma`, layout (n_frames, n_freq, n_chan)), 30 iterations,
  synthetic STFT F=129, N=400, super-Gaussian (IVA) or rank-2 NMF sources, random complex A_f, 40 dB noise floor, both
  complex128 and complex64 (C++ == torchiva bit-identical on all of these too):

  | case | coherence per (src, freq), no proj back | mean SIR ours / pra | proj-back rel. diff |
  |---|---|---|---|
  | AuxIVA laplace 3x3 | min 0.99998, median 1.0000000 | 40.93 / 40.95 dB | 6.5e-2 |
  | OverIVA laplace 2 src / 3 mic | min 1.000000 | 48.30 / 48.29 dB | 3.5e-2 |
  | ILRMA rank 2, 2x2 (different inits) | min 0.9986, median 0.99999 | 37.36 / 37.24 dB | 4.7e-2 |
  | AuxIVA gauss 3x3 | median 0.9998, 120/387 bins < 0.99 | 46.11 / 27.12 dB (pra min -20.8) | 3.2e-1 |

  The projection-back difference is expected (pra: least-squares scale onto mic 0; ours/torchiva: row of W^-1).
  pra's Gauss lands in a different stationary point in ~1/3 of the bins; torchiva and ours separate them (min 21 dB).
  The r-floor (pra 1e-15 vs 1e-5) was ruled out as the cause (torchiva with eps 1e-15 still gives 46.8 dB / min 19.5);
  not investigated further.
- Unit-test margins (closed form, 3x3, F=33, N=400, 30 iterations): min / mean SIR laplace 37.2 / 48.0, gauss
  19.5 / 47.5, nmf 33.3 / 47.7 dB (thresholds 15 / 30); one consistent permutation over all frequencies for all three;
  projection-back gain vs A_f[0, p] median rel. error 5.0e-3 (threshold 1e-2). Waveform pipeline (2x2 instantaneous,
  16000 samples, n_fft 512, hop 128): SDR 38.7 / 30.0 dB vs mic-0 images (threshold 20).
- Speed: c64 3x129x400, 30 iterations: C++ 132 ms (laplace) / 138 ms (nmf) vs torchiva 138 / 143 ms (same kernels).

Review-round measurements (fixer, same machine; scripts in scratchpad `auxiva2/fix/`, built on the reviewers'
pyroomacoustics harness `auxiva2/review-quality-pyroom/src/common.py`; not part of the repo):
- Rooms: pyroomacoustics ShoeBox 6x5x3 m, line array 5 cm spacing, sources at 1.5 m, 10 s pseudo-speech, 50 dB noise
  floor, 50 iterations, BSS-eval v3 SDR (dB, mean over sources, vs the mic-0 images). eps 1e-5 throughout; pra =
  `pra.bss.auxiva` / `ilrma` on the same STFT (reviewer's run).

  | room | model | load 1e-5 (torchiva) c128 / c64 | load 1e-8 (default) c128 / c64 | load 1e-10 c128 / c64 | pra |
  |---|---|---|---|---|---|
  | 3x3 RT60 0.2 | laplace | -2.04 / -1.75 | 11.49 / 11.48 | 11.48 / 11.48 | 10.91 |
  | 3x3 RT60 0.2 | gauss | 0.17 / -0.22 | 11.65 / 11.66 | 11.65 / 11.65 | 10.69 |
  | 3x3 RT60 0.2 | nmf | 6.63 / 6.64 | 10.80 / 10.51 | 10.83 / 10.53 | 6.10 |
  | 3x3 RT60 0.3 | laplace | -2.17 / -4.16 | 3.55 / 2.06 | 3.71 / 2.01 | 3.66 |
  | 3x3 RT60 0.3 | gauss | -2.26 / -1.54 | 4.51 / 4.48 | 4.51 / 4.47 | 4.29 |
  | 3x3 RT60 0.3 | nmf | -0.57 / -0.45 | 5.15 / 3.34 | 4.74 / 3.33 | -0.03 |
  | 3x3 RT60 0.4 | laplace | -6.18 / -1.70 | 6.03 / 6.05 | 6.00 / 5.99 | 5.74 |
  | 3x3 RT60 0.4 | gauss | -4.19 / -3.30 | 5.99 / 5.96 | 5.99 / 6.00 | 5.41 |
  | 3x3 RT60 0.4 | nmf | -2.24 / -7.84 | -1.18 / 5.34 | -1.36 / 5.38 | 1.75 |
  | 2x2 RT60 0.2 | laplace / gauss / nmf | 12.16 / 12.36 / 11.96 (c128) | 12.17 / 12.36 / 11.97 | same | 12.02 / 12.09 / 9.98 |
  | 2x2 RT60 0.3 | laplace / gauss / nmf | 10.15 / 10.50 / 8.08 (c128) | 10.30 / 10.54 / 8.06 | same | 9.72 / 9.54 / 8.84 |
  | 2x2 RT60 0.4 | laplace / gauss / nmf | 6.04 / 5.85 / 6.52 (c128) | 6.17 / 5.95 / 6.42 | same | 6.06 / 4.83 / 5.37 |
  | OverIVA 2 src / 3 mic RT60 0.2 | laplace | 5.14 / 5.07 | 12.88 / 12.24 | 12.85 / 12.93 | 12.28 |
  | OverIVA 2 src / 3 mic RT60 0.3 | laplace | 7.44 / 7.07 | 10.74 / 10.24 | 10.69 / 10.62 | 10.61 |

  2x2 c64 matches c128 within 0.1 dB except nmf, which is init / rounding sensitive (c64 nmf at load 1e-5 / 1e-8:
  RT60 0.2 11.69 / 11.70, RT60 0.3 10.73 / 10.69, RT60 0.4 5.66 / 4.99). At load 1e-8 (c128) the port is within 0.1 dB
  of pra or above it in 18 of 20 cells; the two below are ILRMA (3x3 RT60 0.4 -1.18 vs 1.75, 2x2 RT60 0.3 8.06 vs
  8.84; pra's ILRMA uses a different init).
  The 1e-5 column reproduces the reviewer's torchiva numbers (C++ == torchiva at load 1e-5).
- Input scale (2x2 RT60 0.3, eps / load 1e-5, mixture x1 vs x32768 = int16 full scale, max |X| 5.8e3 vs 1.9e8):
  laplace 10.15 / 10.15, gauss 10.50 / 10.50 (before the projection-back fix: 7.38 / 2.38 at x32768); nmf 8.08 /
  9.66 (ILRMA's own floors are absolute; not a loss).
- Parity re-run (reviewer's 65 configurations, `auxiva2/fix/parity/`): at eps 1e-5, 55/55 bit-identical to stock
  torchiva; with torchiva's projection back patched to our semantics (load = eps, row floor 1e-30), 65/65
  bit-identical, including eps 1e-8 / 1e-3 / 1e-1, silent channels, scale 1e-4 / 1e4, OverIVA mic >= n_src.
- Compact-array unit fixture (c64, 3 mics 5 cm, 35..600 Hz, weak diffuse term, 30 iterations): torchiva's load (1e-5) min SIR
  -2.2 dB with permutation flips; default load (1e-8) min / mean SIR laplace 35.5 / 47.0, gauss 39.5 / 47.2, nmf 32.3 / 42.3,
  one permutation; projected-back gain error vs A_f[0, p] max 1.6e-2 / 1.4e-2 (0.61 / 0.75 if the projection back
  kept a 1e-5 load). OverIVA (2 of the sources, -60 dB floor): load 1e-5 min 0.05 dB with flips; default (1e-8) mean 43.3.

## Issues / Gotchas
- G1 — Parity is bitwise only for the same memory layout. A strided input (e.g. `torch.from_numpy` of a non-C-contiguous
  numpy einsum result) changes reduction order; in complex64 over 30 iterations this grew to max |d| 1e-2 (|Y| up to 14)
  in a first cross-check run. Feed contiguous X when comparing; the goldens use contiguous seeded inputs.
- G2 — torchiva's 1e-5 diagonal load fails compact / reverberant arrays (review Q1, Q3; replaces the earlier
  "noise-free OverIVA only" note, which was wrong). The IP step solves row-normalized normal equations
  `(A^H A + load I) x = A^H b`; with nearly coherent channels (low frequencies, small spacing) `sigma_min^2` of the
  normalized WV falls below 1e-5, the load dominates, and the bins stay mixed or flip permutation: in a 3x3 room
  (5 cm, RT60 0.2) 0-300 Hz reaches 0.99 dB SIR with 40% permutation flips at 1e-5 vs 8.57 dB / 20% at eps = load
  = 1e-10 (reviewer's run);
  bands above 1 kHz are unchanged. More iterations do not help (n_iter 10 / 20 / 50 / 200: -2.16 / -2.44 / -2.04 /
  -1.95 dB). The mic-spacing sweep (3x3, RT60 0.3) closes the gap only at 20 cm (3 cm -3.13 vs pra 2.63; 5 cm -2.17 vs
  3.66; 10 cm 1.20 vs 3.02; 20 cm 2.53 vs 2.46). The earlier i.i.d. random-A_f cross-checks are well conditioned and
  could not show this; noise-free OverIVA (rank-deficient V) is the same mechanism. Fixed: the default load is 1e-8 (D7,
  table above).
- G3 — torchiva's model eps and separator eps are independent objects; ours are `eps` (all floors) and `load` (all
  diagonal loads) (D2, D7).
- G4 — An absolute clamp still makes the IP stage weakly scale dependent at the default eps: for X x 2^16 and up the
  scale-resolution / row-norm clamps fire (max |d| 2.9e-2 vs the unscaled run, 1e-9 with either clamp at 1e-30). SDR
  is unaffected in rooms (x1 .. x32768 identical, input-scale row above), so it is kept for torchiva parity; feed
  STFTs of float audio in [-1, 1] (what `io::load_audio` returns) when bitwise reproducibility matters.
- G5 — complex64 projection back can lose the per-frequency gains when W is very ill-conditioned (found while writing
  the compact-array test): ILRMA on the compact fixture has cond(W) 2e4 .. 8e4 (spread of the per-source scales);
  `solve_loaded_general` forms the normal equations, which square that beyond float32, so the projected-back gains are
  off by median 0.55 / max 0.96 (SIR is unaffected; complex128 gives 4.7e-3, an exact double-precision solve 8.3e-3).
  torchiva computes the same. Row-equilibrating WW before the solve fixes it in complex64 (median 8.4e-3), but changes
  every default-load projection back by ~1e-6 relative (the load is not invariant to that rescaling), breaking
  bitwise torchiva parity, so it is not applied. Workaround: run ILRMA in complex128. See Open/TODO.

## Open / TODO (carry-over)
- G5 (complex64 projection back for ill-conditioned W): deferred. Row-equilibrating WW would move the load-1e-5
  torchiva goldens with projection back (incl. complex128 at 1e-10 tolerance); revisit together with a
  relative-tolerance golden scheme. Until then: run ILRMA in complex128 when the projected-back gains matter.
- MPS: needs the complex-norm rewrite in `solve_loaded_general` (sqrt of summed |a|^2) AND either
  `PYTORCH_ENABLE_MPS_FALLBACK=1` or a CPU round-trip for `linalg_solve` (D6); decide whether to gate it on device.
- CUDA: verify on a GPU machine.
- The pyroomacoustics room harness (review Q3) lives only in the session scratchpad; if Mux wants the room check
  reproducible, add it as a dev-time script next to gen_golden.py (pseudo-speech + ShoeBox + BSS-eval, ~250 lines incl.
  a C++ driver). The portable regression for the mechanism is `test_compact_array_load`.
- Not ported: torchiva AuxIVA-IP2, T-ISS, FIVE, beamformers (MWF/MVDR/GEV), WPE, cost-function monitoring (`verbose`).
- No wav -> wav convenience wrapper; users compose `transform::Spectrogram` + `auxiva_ip` + `transform::InverseSpectrogram`
  (shown in `test_waveform_pipeline`). Re-run `audio_test_separation` after the parallel spectrogram fixes
  (feat/audio-audit-fixes) merge — the pipeline test consumes the transform STFT classes.

## Agent log
- 2026-10-23 [ai] Implemented task03. NEW `_separation.hpp` (auxiva_ip, ilrma, auxiva_option, source_model; detail:
  mag_sq, solve_loaded_general, orthogonal_constraint, projection_back, nmf_weights, source_weights), included from
  `audio.hpp`. NEW `audio_test_separation` (224 checks): 15 torchiva goldens (|Y|^2 sum, seeded complex checksum,
  4 elements each), ilrma-forces-nmf, batch-item independence, closed-form separation for laplace/gauss/nmf with a
  single global permutation, projection-back gain vs the true mic-0 image, wav pipeline through transform
  Spectrogram/InverseSpectrogram, option defaults/setters, meta-device propagation, every validation branch, no-grad.
  Full build green, `ctest` 6/6 green. Coverage (Debug, `-DTORCHMEDIA_COVERAGE=ON`): `_separation.hpp` 100% lines
  (160/160), functions (15/15), regions (70/70), branches (44/44). clang-format applied. Cross-checks recorded above.
  THIRD_PARTY_LICENSES + CLAUDE.md updated. Not committed — awaiting Mux's review.
- 2026-10-23 [ai] Review round (fixer). Fixed: (Q1) new `auxiva_option::load` for every diagonal load, default 1e-5
  (torchiva), `eps` keeps the floors; (Q2) projection back uses `load` and a 1e-30 row floor instead of torchiva's
  fixed 1e-5 load / clamp — int16-scale input now separates like float input; (P2/RC1) dtype check requires
  complex64/complex128 and names the dtype; (RC5) eps and load must be finite; (RC2) `test_model_floor` (eps reaches
  all three model floors, closed form) and a torchiva OverIVA golden with eps/load 1e-3 (J in the loop); the eps 1e-3
  gauss golden now runs without projection back; (RC3) accepting boundaries n_src == n_chan and n_basis 1; (Q3)
  `test_compact_array_load` (portable coherent-channel fixture; determined + OverIVA; pins that load reaches the IP
  solve, J and the projection back); `test_projection_back_scale` (a(W / c) = c a(W) exactly). Docs: P1 (D5 and
  gen_golden comment: slices equal up to rounding), RC4 (MPS needs norm rewrite + linalg_solve fallback; header
  comment names device support), RC6 (THIRD_PARTY_LICENSES preamble), G2 rewritten with room evidence, new D7, G4,
  G5. Mutation check of the new wiring (12 mutants: row floor, projection-back load, J load in and before the loop, IP
  load, three model floors, dtype check, eps / load finiteness, load > 0): all killed. `audio_test_separation`
  265/265; full build green, `ctest` 6/6. Coverage (Debug, `-DTORCHMEDIA_COVERAGE=ON`): `_separation.hpp` 100% lines
  (169/169), functions (16/16), regions (80/80), branches (52/52). clang-format applied. Not committed.
- 2026-10-23 [ai] Orchestrator decision D7 (Mux's delegation): default `load` 1e-8. Tests: torchiva goldens pass
  `tiva()` (= `load(1e-5)`), `test_compact_array_load` now pins torchiva's load failing and the default separating,
  the defaults test asserts 1e-8. `audio_test_separation` 265/265, ctest 6/6, clang-format clean.
