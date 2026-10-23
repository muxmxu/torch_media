# Task 07 — mel filterbank / MelSpectrogram / MFCC / LFCC parity
id: 2026-10-23/task07
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
The mel/cepstral path matches torchaudio numerically, runs on the input's device/dtype, and is fast.

## Findings
G3-2 (CPU float32 fb/DCT), G3-3 (per-element filterbank loops), G3-7 (mfcc_option.sample_rate ignored), G3-9
(`sample_rate / 2.0`), G3-10 (n_mfcc > n_mels), G3-11 (bad strings, f_min > f_max), G3-12 (onesided=false);
new: MelSpectrogram/MFCC discard default sub-transforms, LFCC and `check_f_min_max` use `/ 2.0`, transform
MelScale/InverseMelScale/MFCC/LFCC buffers stay CPU float32.

## Scope
In: `_functional.hpp` (mel_filter_bank, _create_triangular_filterbank moved up, melspectrogram, create_dct,
mfcc), `_functional_methods_options.hpp` (`detail::effective_f_max`, option comments), `_transform_feature.hpp`.
Out: empty-filter warning, functional option-struct API (G3-13) — progress02 Open.

## Deliverables
- Vectorized `mel_filter_bank` mirroring melscale_fbanks op-for-op ([n_mels, n_freqs] layout kept).
- `mfcc`: sample_rate overrides mel.sample_rate, n_mfcc <= n_mels, DCT on the data's device/dtype, top_db on.
- Transforms: buffers moved per call; member-initializer lists; `effective_f_max` everywhere.
- Tests: `test_mel_filter_bank_parity` (incl. a < 0.5 s bound for 20 large banks), `test_melspectrogram_mfcc_parity`,
  transform `test_feature_parity` / `test_buffers_follow_input`.

## Acceptance criteria
- [x] Red on e7041b5: fb at 11025 Hz off by 4.7e-5, 20 banks > 0.5 s, bad strings accepted, mfcc sr=8000 off by
  42.1, stereo mfcc off by 39.8, n_mfcc/f_min not rejected, onesided=false threw, float64 MelSpectrogram threw,
  LFCC at 11025 off by 2.5e-3, MelScale(11025, f_min=5512.25) accepted.
- [x] Green; probe: filterbanks (16 k/22.05 k/11.025 k, htk/slaney) and MelSpectrogram bit-exact, MFCC/LFCC
  <= 3.5e-7 relative. Timing 17.8 -> 0.048 ms per bank, MFCC() ctor 74.5 -> 0.066 ms.
- [x] ctest green.
