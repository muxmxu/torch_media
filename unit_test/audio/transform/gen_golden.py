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

# ======================= progress04 -- surface audit (2026-10-23, task12-task19) =======================
# SpecAugment(iid_masks=False) on float64: torch.rand(1) is drawn on the CPU in float32, so the masks equal float32's.
_xs = torch.arange(1, 1 + 2 * 30 * 40, dtype=torch.float64).reshape(2, 30, 40)
torch.manual_seed(0)
_ys = T.SpecAugment(2, 10, 2, 10, iid_masks=False, zero_masking=True)(_xs)
show("specaugment_f64_seed0 sum/zeros", [repr(float(_ys.sum())), int((_ys == 0).sum())])  # 1852636.0, 744
# PitchShift with fractional n_steps (the functional value is pinned in the functional gen_golden.py).
_pp = T.PitchShift(1000, 0.5)(torch.sin(torch.arange(3001.0) * 0.05).unsqueeze(0))
show("pitch_shift_frac [0,1500]", repr(float(_pp[0, 1500])))

# ======================= progress08 -- audio parity follow-ups (2026-10-23, task53 / task54) =======================
import warnings

import torchaudio.functional as F


def _p8(n):
    t = torch.arange(n, dtype=torch.float64)
    return torch.stack([0.4 * torch.sin(t * 0.013) + 0.3 * torch.sin(t * 0.71),
                        0.3 * torch.cos(t * 0.0021) - 0.2 * torch.sin(t * 1.9 + 0.3)])


def _pts(y, idx):  # values at idx (repr), then the float64 sum of the whole output
    return [repr(float(y[i])) for i in idx] + [repr(float(y.double().sum()))]


# task53 -- InverseMelScale, LFCC, SpeedPerturbation (seeded factor draw).
_f = torch.arange(40, dtype=torch.float64).reshape(40, 1)
_t = torch.arange(30, dtype=torch.float64)
_mel = (1.0 + torch.sin(0.3 * _f + 0.17 * _t) ** 2 * (1 + _f / 10)).float()
_y = T.InverseMelScale(n_stft=201, n_mels=40, sample_rate=16000)(_mel)
show("p8 InverseMelScale shape / points+sum / zeros", [list(_y.shape), _pts(_y, ((0, 0), (37, 12), (120, 29))),
                                                       int((_y == 0).sum())])
_x = _p8(8000).float()
_y = T.LFCC(sample_rate=16000, n_filter=40, n_lfcc=20, speckwargs=dict(n_fft=400, hop_length=160))(_x)
show("p8 LFCC", [list(_y.shape), _pts(_y, ((0, 0, 0), (1, 19, 50), (0, 7, 23)))])
_y = T.LFCC(sample_rate=16000, n_filter=40, n_lfcc=13, log_lf=True, speckwargs=dict(n_fft=256))(_x)
show("p8 LFCC log_lf", [list(_y.shape), _pts(_y, ((0, 0, 0), (1, 12, 50), (0, 7, 23)))])
_x1 = _p8(4000)[:1].float()
_sp = T.SpeedPerturbation(16000, [0.9, 1.0, 1.1])
for _seed in (7, 2024):
    torch.manual_seed(_seed)
    _y, _l = _sp(_x1, torch.tensor([4000.0]))
    show(f"p8 SpeedPerturbation seed {_seed}", [list(_y.shape), float(_l[0]),
                                                _pts(_y, ((0, 0), (0, 1000), (0, _y.shape[-1] - 1)))])


# task53 / task54 -- SoudenMVDR, RTFMVDR (int and one-hot Tensor reference_channel), MVDR offline / online / cfloat.
def _bf(k):
    n = torch.arange(3 * 33 * 20, dtype=torch.float64)
    spec = torch.complex(torch.sin(n * (0.37 + 0.01 * k)), torch.cos(n * (0.11 + 0.02 * k))).reshape(3, 33, 20)
    m = torch.arange(33 * 20, dtype=torch.float64)
    return spec, (0.5 + 0.4 * torch.sin(m * (0.23 + 0.05 * k))).reshape(33, 20)


def _bfp(y):
    return [[repr(float(y[i].real)), repr(float(y[i].imag))] for i in ((0, 0), (17, 9), (32, 19))] + \
        [repr(float(y.abs().sum()))]


warnings.filterwarnings("ignore")  # MVDR warns when mask_n is not given
_spec, _ms = _bf(0)
_ps, _pn = T.PSD()(_spec, _ms), T.PSD()(_spec, 1 - _ms)
_hot = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
show("p8 SoudenMVDR ref 1 / one-hot", [_bfp(T.SoudenMVDR()(_spec, _ps, _pn, 1)), _bfp(T.SoudenMVDR()(_spec, _ps, _pn, _hot))])
_rtf = F.rtf_evd(_ps)
show("p8 RTFMVDR ref 1 / one-hot", [_bfp(T.RTFMVDR()(_spec, _rtf, _pn, 1)), _bfp(T.RTFMVDR()(_spec, _rtf, _pn, _hot))])
for _sol in ("ref_channel", "stv_evd", "stv_power"):
    show(f"p8 MVDR ref 2 {_sol}", _bfp(T.MVDR(ref_channel=2, solution=_sol)(_spec, _ms, 1 - _ms)))
show("p8 MVDR default mask_n", _bfp(T.MVDR()(_spec, _ms)))
_y = T.MVDR(ref_channel=1)(_spec.to(torch.complex64), _ms.float(), (1 - _ms).float())
show("p8 MVDR cfloat", [str(_y.dtype), _bfp(_y)])
_spec2, _ms2 = _bf(1)
for _sol in ("ref_channel", "stv_power"):
    _mv = T.MVDR(ref_channel=1, solution=_sol, online=True)
    show(f"p8 MVDR online {_sol} call 1 / call 2", [_bfp(_mv(_spec, _ms, 1 - _ms)), _bfp(_mv(_spec2, _ms2, 1 - _ms2))])
_mm1, _mm2 = torch.stack([_ms, _ms.flip(-1), 0.5 * _ms]), torch.stack([_ms2, 1 - _ms2, _ms2.flip(0)])
_mv = T.MVDR(solution="stv_evd", multi_mask=True, online=True)
_mv(_spec, _mm1, 1 - _mm1)
show("p8 MVDR online multi_mask call 2", _bfp(_mv(_spec2, _mm2, 1 - _mm2)))

# progress08 review fixes (R5 / R6). Inputs where the rounding order shows: a 1-D LFCC input (another matmul kernel than
# the batched one above), PitchShift at 1.5 steps (phase_vocoder's phase wrap rounds differently in ~300 samples).
_x = _p8(6000)[0].float()
_y = T.LFCC(sample_rate=8000, n_filter=32, n_lfcc=13, log_lf=True, speckwargs=dict(n_fft=256, hop_length=100))(_x)
show("p8r LFCC 8k log_lf 1-D", [list(_y.shape), _pts(_y, ((0, 0), (12, 60), (7, 23), (3, 41)))])
_y = T.PitchShift(8000, 1.5, n_fft=512)(_p8(4000).float())
show("p8r PitchShift f32", [list(_y.shape), _pts(_y, ((0, 100), (1, 2000), (0, 3999)))])
# float64: torchaudio's module takes float64 only after .double(), which casts its float32 window buffer up.
_y = T.PitchShift(8000, 1.5, n_fft=512).double()(_p8(4000))
show("p8r PitchShift f64 .double()", [list(_y.shape), _pts(_y, ((0, 100), (1, 2000), (0, 3999)))])
