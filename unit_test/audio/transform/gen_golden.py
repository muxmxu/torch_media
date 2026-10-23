#!/usr/bin/env python3
"""Golden reference values for the audio transform unit tests, computed with torchaudio 2.5.1.

Run in the project venv:
    .venv/bin/python unit_test/audio/transform/gen_golden.py

Inputs are explicit (deterministic, no RNG) so the printed constants are portable across the
Python reference and the C++ libtorch implementation. The C++ tests additionally assert that each
transform class delegates to its golden-verified torchmedia::audio::functional counterpart
(libtorch self-reference, no venv needed at build time).
"""
import torch
import torchaudio.transforms as T

torch.set_printoptions(precision=8)


def show(name, value):
    print(f"{name} = {value}")


# Deterministic explicit input: a length-16 ramp (no RNG -> identical in C++).
N = 16
x = torch.arange(N, dtype=torch.float32)
NFFT, WIN, HOP = 8, 8, 4

# ---- Spectrogram (real power spectrogram, default power=2.0) ----
spec = T.Spectrogram(n_fft=NFFT, win_length=WIN, hop_length=HOP)(x)
show("spectrogram_shape", list(spec.shape))            # [5, 5]
show("spectrogram", spec.flatten().tolist())

# ---- SpectralCentroid ----
sc = T.SpectralCentroid(sample_rate=16000, n_fft=NFFT, win_length=WIN, hop_length=HOP)(x)
show("spectral_centroid_shape", list(sc.shape))        # [5]
show("spectral_centroid", sc.flatten().tolist())

# ---- GriffinLim (deterministic: rand_init=False) ----
mag = T.Spectrogram(n_fft=NFFT, win_length=WIN, hop_length=HOP, power=2.0)(x)
gl = T.GriffinLim(n_fft=NFFT, win_length=WIN, hop_length=HOP, power=2.0,
                  n_iter=8, momentum=0.99, rand_init=False, length=N)(mag)
show("griffinlim_shape", list(gl.shape))               # [16]
show("griffinlim", [float(f"{v:.9g}") for v in gl.flatten().tolist()])

# ---- InverseSpectrogram (round-trip of the complex STFT recovers x) ----
spec_c = T.Spectrogram(n_fft=NFFT, win_length=WIN, hop_length=HOP, power=None)(x)
inv = T.InverseSpectrogram(n_fft=NFFT, win_length=WIN, hop_length=HOP)(spec_c, length=N)
show("inverse_spectrogram_shape", list(inv.shape))     # [16]
show("inverse_spectrogram", inv.flatten().tolist())
show("inverse_spectrogram_recon_max_err", float((inv - x).abs().max()))

# ============================ task20: mel & cepstral ============================
N_MELS, N_STFT = 4, NFFT // 2 + 1  # 4 mels, 5 stft bins
SR = 16000

# ---- AmplitudeToDB (power) ----
adb = T.AmplitudeToDB("power", top_db=80.0)(spec)
show("amplitude_to_db_shape", list(adb.shape))
show("amplitude_to_db", adb.flatten().tolist())

# ---- MelScale (htk, norm=None) over the power spectrogram ----
ms = T.MelScale(n_mels=N_MELS, sample_rate=SR, n_stft=N_STFT)(spec)
show("mel_scale_shape", list(ms.shape))
show("mel_scale", ms.flatten().tolist())

# ---- MelSpectrogram ----
mel = T.MelSpectrogram(sample_rate=SR, n_fft=NFFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS)(x)
show("mel_spectrogram_shape", list(mel.shape))
show("mel_spectrogram", mel.flatten().tolist())

# ---- MFCC (n_mfcc<=n_mels) ----
mfcc = T.MFCC(sample_rate=SR, n_mfcc=N_MELS,
              melkwargs=dict(n_fft=NFFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS))(x)
show("mfcc_shape", list(mfcc.shape))
show("mfcc", mfcc.flatten().tolist())

# ---- LFCC (n_lfcc<=n_filter) ----
lfcc = T.LFCC(sample_rate=SR, n_filter=N_MELS, n_lfcc=N_MELS,
              speckwargs=dict(n_fft=NFFT, win_length=WIN, hop_length=HOP))(x)
show("lfcc_shape", list(lfcc.shape))
show("lfcc", lfcc.flatten().tolist())

# ---- InverseMelScale (lstsq, driver='gels') ----
# Use a full-rank fb (9 stft bins, 4 mels) so the default 'gels' QR driver succeeds.
NFFT2, HOP2 = 16, 8
N_STFT2 = NFFT2 // 2 + 1  # 9
x2 = torch.arange(32, dtype=torch.float32)
spec2 = T.Spectrogram(n_fft=NFFT2, win_length=NFFT2, hop_length=HOP2)(x2)        # (9, T2)
ms2 = T.MelScale(n_mels=N_MELS, sample_rate=SR, n_stft=N_STFT2)(spec2)           # (4, T2)
show("inverse_mel_scale_in_shape", list(ms2.shape))
inv_ms = T.InverseMelScale(n_stft=N_STFT2, n_mels=N_MELS, sample_rate=SR, driver="gels")(ms2)
show("inverse_mel_scale_shape", list(inv_ms.shape))                              # (9, T2)
show("inverse_mel_scale", inv_ms.flatten().tolist())

# ============================ task22: augmentation (Fade) ============================
# Masking/SpecAugment are RNG-driven -> tested by seeded delegation-equivalence in C++ (no golden).
# Fade is deterministic. Apply to ones(8) so the result is the fade_in * fade_out product.
ones8 = torch.ones(8)
for shp in ["linear", "exponential", "logarithmic", "quarter_sine", "half_sine"]:
    show(f"fade_{shp}", T.Fade(fade_in_len=3, fade_out_len=3, fade_shape=shp)(ones8).tolist())

# ======================= 2026-10-23 audit follow-up (progress02, task04-task09) =======================
# The functional gen_golden.py prints the shared values (amplitude_to_DB packing, frame_length, griffinlim
# length/rand_init, MFCC stereo); only transform-specific goldens are here.
# LFCC at an odd rate: f_max defaults to float(sample_rate // 2), so 11025 Hz equals the 16 kHz golden above.
show("lfcc_11025", T.LFCC(sample_rate=11025, n_filter=N_MELS, n_lfcc=N_MELS,
                          speckwargs=dict(n_fft=NFFT, win_length=WIN, hop_length=HOP))(x).flatten().tolist())
# Resample caches a kernel computed in float64 and stored as float32 (dtype=None).
show("resample_kernel_48k_16k", [float(f"{float(v):.9g}") for v in T.Resample(48000, 16000).kernel.flatten()])
show("resample_44k1_16k_len", T.Resample(44100, 16000)(torch.zeros(1, 180697)).shape[-1])  # 65559

# ======================= task11 -- review round 2 (progress02 D12) =======================
import math
_t = torch.arange(3001, dtype=torch.float64)
_x2 = torch.stack([torch.sin(_t * 0.05), 0.5 * torch.cos(_t * 0.031)])
# PitchShift on float64 input builds its kernel in float64 (lazy init with dtype=input.dtype).
_ps = T.PitchShift(11025, 3, n_fft=400).double()(_x2)
show("pitch_shift_f64 [0,2964]/[0,2954]/[0,2977]", [repr(float(_ps[0, i])) for i in (2964, 2954, 2977)])
# Resample kernel at an nf > 1 pair (44.1k -> 16k: nf = 160): float64 checksums of the float32 kernel.
_k = T.Resample(44100, 16000).kernel.double()
_wts = torch.sin(torch.arange(_k.numel(), dtype=torch.float64) * 0.7).reshape(_k.shape)
show("resample_kernel_44k1_16k shape/sum/wsum/[38717]",
     [tuple(_k.shape), repr(float(_k.sum())), repr(float((_k * _wts).sum())), repr(float(_k.flatten()[38717]))])
