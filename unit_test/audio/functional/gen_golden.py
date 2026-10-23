#!/usr/bin/env python3
"""Golden reference values for the audio functional unit tests, computed with torchaudio 2.5.1.

Run in the project venv:
    .venv/bin/python unit_test/audio/functional/gen_golden.py

Filterbanks and melspectrogram are deterministic and cross-language comparable, so the printed
constants are hard-coded into main.cpp. Randomized-input tests use libtorch self-reference instead
(e.g. normalized spectrogram == unnormalized / Sum(win^2)).
"""
import torch
import torchaudio.functional as F
import torchaudio.transforms as T


def show(name, value):
    print(f"{name} = {value}")


# bug#3 -- db_to_amplitude(x, ref, power) = ref * (10^(0.1*x))^power
show("db_to_amp(20,1,1)", float(F.DB_to_amplitude(torch.tensor(20.0), 1.0, 1.0)))    # 100
show("db_to_amp(20,1,0.5)", float(F.DB_to_amplitude(torch.tensor(20.0), 1.0, 0.5)))  # 10
show("db_to_amp(10,2,1)", float(F.DB_to_amplitude(torch.tensor(10.0), 2.0, 1.0)))    # 20

# bug#2 -- amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db); no internal square
show("amp_to_DB power [1,.1,.01,.001] mult=10",
     F.amplitude_to_DB(torch.tensor([[1., .1, .01, .001]]), 10.0, 1e-10, 0.0, 80.0).tolist())  # [0,-10,-20,-30]
show("amp_to_DB mag [1,.5,.25] mult=20",
     F.amplitude_to_DB(torch.tensor([[1., .5, .25]]), 20.0, 1e-10, 0.0, 80.0).tolist())          # [0,-6.02,-12.04]

# bug#4 -- convolve mode lengths ('same' = first input length)
a = torch.tensor([1., 2., 3., 4., 5.]).view(1, -1)
b = torch.tensor([1., 1., 1.]).view(1, -1)
for m in ["full", "same", "valid"]:
    show(f"convolve {m} len", F.convolve(a, b, mode=m).shape[-1])  # 7 / 5 / 3

# bug#1 -- mel filterbank: slaney differs from htk; slaney sum (norm=None)
fb_s = F.melscale_fbanks(201, 0.0, 8000.0, 64, 16000, norm=None, mel_scale="slaney")
fb_h = F.melscale_fbanks(201, 0.0, 8000.0, 64, 16000, norm=None, mel_scale="htk")
show("fb_slaney != fb_htk", bool(not torch.allclose(fb_s, fb_h)))
show("fb_slaney.sum", round(float(fb_s.sum()), 3))  # 194.677

# bug#5 -- spectrogram normalized (power=2): unnormalized/normalized == Sum(win^2)
win = torch.hann_window(400)
show("Sum(win^2)", float(win.pow(2).sum()))  # 150

# melspectrogram (htk) end-to-end golden, deterministic sine input
sig = torch.sin(2 * torch.pi * 440.0 * torch.arange(16000.0) / 16000.0).unsqueeze(0)
mel = T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512, hop_length=256, n_mels=64,
                       f_min=0.0, f_max=8000.0, power=2.0, norm=None, center=True,
                       pad_mode="reflect", mel_scale="htk")(sig)
show("melspec htk shape", tuple(mel.shape))          # (1, 64, 63)
show("melspec htk sum", round(float(mel.sum()), 3))  # 1548267.875

# Tier 1 -- create_dct (DCT-II matrix)
for norm in (None, "ortho"):
    d = F.create_dct(4, 8, norm)
    show(f"create_dct(4,8,{norm}) sum/[0,0]/[3,2]",
         [round(float(d.sum()), 6), round(float(d[0, 0]), 6), round(float(d[3, 2]), 6)])

# Tier 1 -- mfcc (DCT over dB / log mel)
mk = dict(n_fft=512, win_length=512, hop_length=256, n_mels=64, f_min=0.0, f_max=8000.0,
          power=2.0, norm=None, mel_scale="htk", center=True, pad_mode="reflect")
for log_mels in (False, True):
    mf = T.MFCC(sample_rate=16000, n_mfcc=13, norm="ortho", log_mels=log_mels, melkwargs=mk)(sig)
    show(f"mfcc(log_mels={log_mels}) shape/sum/[0,0,0]",
         [tuple(mf.shape), round(float(mf.sum()), 4), round(float(mf[0, 0, 0]), 4)])

# Tier 1 -- griffinlim (deterministic, rand_init=False)
wavg = torch.sin(2 * torch.pi * 5.0 * torch.arange(2000.0) / 2000.0).unsqueeze(0)
wing = torch.hann_window(256)
specg = F.spectrogram(wavg, pad=0, window=wing, n_fft=256, hop_length=64, win_length=256, power=2.0,
                      normalized=False, center=True, pad_mode="reflect", onesided=True)
recg = F.griffinlim(specg, window=wing, n_fft=256, hop_length=64, win_length=256, power=2.0,
                    n_iter=32, momentum=0.99, length=None, rand_init=False)
show("griffinlim shape/sum/[0,1000]", [tuple(recg.shape), repr(float(recg.sum())), repr(float(recg[0, 1000]))])

# Tier 1 -- resample
xr = torch.sin(2 * torch.pi * 3.0 * torch.arange(64.0) / 64.0).unsqueeze(0)
for (a, b) in [(64, 32), (64, 48)]:
    rr = F.resample(xr, a, b, lowpass_filter_width=6, rolloff=0.99)
    show(f"resample {a}->{b} shape/sum/[0,1]", [tuple(rr.shape), round(float(rr.sum()), 6), round(float(rr[0, 1]), 6)])

# task01 -- lfilter / biquad / filtfilt (IIR core). Fixed deterministic signal.
fwav = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
show("lfilter a=[1,-.3,.05] b=[.5,.1,0]",
     [round(float(x), 8) for x in F.lfilter(fwav, torch.tensor([1.0, -0.3, 0.05]), torch.tensor([0.5, 0.1, 0.0]))])
show("biquad (.2,.2,.2 / 1,-.5,.1)",
     [round(float(x), 8) for x in F.biquad(fwav, 0.2, 0.2, 0.2, 1.0, -0.5, 0.1)])
show("filtfilt a=[1,-.3,.05] b=[.5,.1,0]",
     [round(float(x), 8) for x in F.filtfilt(fwav, torch.tensor([1.0, -0.3, 0.05]), torch.tensor([0.5, 0.1, 0.0]))])
fimp = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
show("lfilter one-pole a=[1,-.5] b=[1,0] (0.5^n)",
     [round(float(x), 8) for x in F.lfilter(fimp, torch.tensor([1.0, -0.5]), torch.tensor([1.0, 0.0]))])
_a2 = torch.tensor([[1.0, -0.3, 0.05], [1.0, -0.5, 0.1]])
_b2 = torch.tensor([[0.5, 0.1, 0.0], [0.2, 0.2, 0.2]])
_o2 = F.lfilter(torch.stack([fwav, fwav * 0.5]), _a2, _b2, batching=True)
show("lfilter batched row1", [round(float(x), 8) for x in _o2[1]])

# task02 -- biquad designers. filt_wav(), sr=16000, f=2000, Q=0.707, gain=6 (deemph/riaa: own rates).
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
def _bd(n, t): show(n, [round(float(x), 7) for x in t])
_bd("allpass", F.allpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("lowpass", F.lowpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("highpass", F.highpass_biquad(_fw, 16000, 2000.0, 0.707))
_bd("bandpass_F", F.bandpass_biquad(_fw, 16000, 2000.0, 0.707, const_skirt_gain=False))
_bd("bandpass_T", F.bandpass_biquad(_fw, 16000, 2000.0, 0.707, const_skirt_gain=True))
_bd("bandreject", F.bandreject_biquad(_fw, 16000, 2000.0, 0.707))
_bd("equalizer_g6", F.equalizer_biquad(_fw, 16000, 2000.0, 6.0, 0.707))
_bd("band_F", F.band_biquad(_fw, 16000, 2000.0, 0.707, noise=False))
_bd("band_T", F.band_biquad(_fw, 16000, 2000.0, 0.707, noise=True))
_bd("bass_g6", F.bass_biquad(_fw, 16000, 6.0, 100.0, 0.707))
_bd("treble_g6", F.treble_biquad(_fw, 16000, 6.0, 3000.0, 0.707))
for _r in (44100, 48000):
    _bd(f"deemph_{_r}", F.deemph_biquad(_fw, _r))
for _r in (44100, 48000, 88200, 96000):
    _bd(f"riaa_{_r}", F.riaa_biquad(_fw, _r))

# task03 -- simple effects contrast/dcshift/gain on filt_wav()
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
def _ef(n, t): show(n, [round(float(x), 7) for x in t])
_ef("contrast_75", F.contrast(_fw, 75.0)); _ef("contrast_0", F.contrast(_fw, 0.0))
_ef("gain_6", F.gain(_fw, 6.0))
_ef("dcshift_0.3", F.dcshift(_fw, 0.3)); _ef("dcshift_-0.3", F.dcshift(_fw, -0.3))
_ef("dcshift_lim_pos", F.dcshift(_fw.clone(), 0.5, limiter_gain=0.1))
_ef("dcshift_lim_neg", F.dcshift(_fw.clone(), -0.5, limiter_gain=0.1))

# task06 -- companding & emphasis on filt_wav()
_fw = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
show("mulaw_enc", [int(x) for x in F.mu_law_encoding(_fw, 256)])
show("mulaw_dec", [round(float(x), 7) for x in F.mu_law_decoding(F.mu_law_encoding(_fw, 256), 256)])
show("preemph_097", [round(float(x), 7) for x in F.preemphasis(_fw, 0.97)])
show("deemph_097", [round(float(x), 7) for x in F.deemphasis(_fw, 0.97)])

# task09 -- compute_deltas / linear_fbanks / spectral_centroid
_sg = torch.tensor([[1.,2.,3.,4.,5.,6.],[6.,5.,4.,3.,2.,1.]])
show("deltas_w5", F.compute_deltas(_sg, 5).tolist())
show("deltas_w3", F.compute_deltas(_sg, 3).tolist())
show("linfbank_6_3", F.linear_fbanks(6, 0.0, 8000.0, 3, 16000).tolist())

# task07 -- fftconvolve / add_noise / speed
_x = torch.tensor([[1.,2.,3.,4.]]); _y = torch.tensor([[1.,1.,1.]])
for _m in ("full","valid","same"):
    show(f"fftconv_{_m}", F.fftconvolve(_x, _y, mode=_m).tolist())
_w = torch.tensor([[1.,2.,3.,4.],[0.5,-0.5,0.5,-0.5]]); _n = torch.tensor([[0.1,0.1,0.1,0.1],[0.2,-0.2,0.2,-0.2]])
show("add_noise", F.add_noise(_w, _n, torch.tensor([10.0,3.0])).tolist())
_sp = torch.sin(2*torch.pi*2.0*torch.arange(16.0)/16.0).reshape(1,16)
_o,_l = F.speed(_sp, 16, 2.0, torch.tensor([16])); show("speed_out", [round(float(x),6) for x in _o.flatten()]); show("speed_len", _l.tolist())

# task13 -- frechet_distance (edit_distance is non-tensor, closed-form tested in C++)
_I = torch.eye(2)
show("frechet_shift", round(float(F.frechet_distance(torch.tensor([0.,0.]), _I, torch.tensor([1.,1.]), _I)), 6))
show("frechet_diag", round(float(F.frechet_distance(torch.tensor([1.,2.]), torch.tensor([[2.,0.],[0.,3.]]), torch.tensor([0.,0.]), _I)), 6))

# task18 -- dither (TPDF deterministic; RPDF/GPDF RNG-dependent)
_dw = torch.tensor([[0.1,-0.2,0.3,-0.4,0.5,-0.6],[0.05,-0.05,0.15,-0.15,0.25,-0.25]])
show("dither_tpdf", F.dither(_dw, "TPDF", noise_shaping=False).tolist())
show("dither_tpdf_ns", F.dither(_dw, "TPDF", noise_shaping=True).tolist())

# task08 -- phase_vocoder / inverse_spectrogram / pitch_shift (deterministic inputs)
import math as _m
_A = torch.arange(1,13,dtype=torch.float32).reshape(1,3,4)
_spec = torch.complex(_A, _A*0.5); _pa = torch.linspace(0, _m.pi, 3)[...,None]
_pv = F.phase_vocoder(_spec, 1.5, _pa)
show("pv_real", [round(float(x),6) for x in _pv.real.flatten()])
show("pv_imag", [round(float(x),6) for x in _pv.imag.flatten()])
_t = torch.arange(2000.0)/16000.0; _sine = torch.sin(2*_m.pi*440*_t).reshape(1,2000); _win = torch.hann_window(400)
_S = F.spectrogram(_sine, pad=0, window=_win, n_fft=400, hop_length=200, win_length=400, power=None, normalized=False)
show("inv_recon_err", round(float((F.inverse_spectrogram(_S,2000,0,_win,400,200,400,False)-_sine).abs().max()),6))
_ps = F.pitch_shift(_sine, 16000, 4, n_fft=512)
show("ps_sum", round(float(_ps.sum()),5)); show("ps_500", round(float(_ps[0,500]),6))

# task12 -- loudness (ITU-R BS.1770)
import math as _m
_t = torch.arange(16000.0)/16000.0; _tone = torch.sin(2*_m.pi*1000*_t)
show("loudness_2ch", round(float(F.loudness(torch.stack([_tone, 0.5*_tone]), 16000)),5))
show("loudness_mono", round(float(F.loudness(_tone.reshape(1,16000), 16000)),5))

# task14 -- beamforming (deterministic complex PD matrices)
_A = torch.arange(1,25,dtype=torch.float32).reshape(2,3,4); _spec = torch.complex(_A, _A*0.1)
show("psd_re", [round(float(x),5) for x in F.psd(_spec).real.flatten()])
_Ms = torch.complex(torch.arange(1,13,dtype=torch.float32).reshape(3,2,2), torch.arange(1,13,dtype=torch.float32).reshape(3,2,2)*0.2)
_ps = _Ms @ _Ms.conj().transpose(-1,-2) + 2*torch.eye(2)
_Mn = torch.complex(torch.arange(13,25,dtype=torch.float32).reshape(3,2,2)*0.1, torch.arange(1,13,dtype=torch.float32).reshape(3,2,2)*0.3)
_pn = _Mn @ _Mn.conj().transpose(-1,-2) + 2*torch.eye(2)
_ws = F.mvdr_weights_souden(_ps,_pn,0)
show("souden_re", [round(float(x),5) for x in _ws.real.flatten()]); show("souden_im", [round(float(x),5) for x in _ws.imag.flatten()])

# task04 -- overdrive / phaser / flanger
_w = torch.tensor([0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8]).reshape(1,8)
show("overdrive", [round(float(x),6) for x in F.overdrive(_w,20.0,20.0).flatten()])
show("phaser_sine", [round(float(x),6) for x in F.phaser(_w,8000,sinusoidal=True).flatten()])
_wf = torch.tensor([[0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8],[0.2,-0.1,0.4,-0.3,0.6,-0.5,0.8,-0.7]])
show("flanger_lin", [round(float(x),6) for x in F.flanger(_wf,8000,interpolation="linear").flatten()])

# task11 -- sliding_window_cmn / detect_pitch_frequency
import math as _m
_sg = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,10.]])
show("cmn_basic", F.sliding_window_cmn(_sg,3,2,False,False).tolist())
show("cmn_center", F.sliding_window_cmn(_sg,3,2,True,False).tolist())
show("cmn_normvars", F.sliding_window_cmn(_sg,4,2,False,True).tolist())
_sine = torch.sin(2*_m.pi*220*torch.arange(8000.0)/8000.0).reshape(1,8000)
show("pitch_median", round(float(F.detect_pitch_frequency(_sine,8000).median()),3))

# task05 -- vad (silence-then-tone trim length)
import math as _m
_sr=16000; _t=torch.arange(int(0.5*_sr))/_sr
_sig=torch.cat([torch.zeros(int(0.5*_sr)), 0.5*torch.sin(2*_m.pi*300*_t)]).reshape(1,-1)
show("vad_out_len", int(F.vad(_sig,_sr).size(-1)))

# ======================= 2026-10-23 audit follow-up (progress02, task04-task09) =======================
import math

def g9(t):  # 9 significant digits: enough to pin float32 values
    return [float(f"{float(v):.9g}") for v in t.flatten()]

# task04 -- amplitude_to_DB: top_db cutoff over (C, F, T) per packed batch item; AmplitudeToDB() defaults
_c = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
_c[1] *= 1e-6
show("adb3_t40", g9(F.amplitude_to_DB(_c, 10.0, 1e-10, 0.0, 40.0)))
show("adb4_t40", g9(F.amplitude_to_DB(torch.stack([_c, _c * 1e-3]), 10.0, 1e-10, 0.0, 40.0)))
show("adb_default", g9(T.AmplitudeToDB()(_c)))
show("adb_amin0", F.amplitude_to_DB(torch.tensor([[0.0, 1e-45, 1.0]]), 10.0, 0.0, 0.0, None).tolist())
show("adb_f64", [repr(float(v)) for v in F.amplitude_to_DB(torch.tensor([[0.5, 0.25, 2.0]], dtype=torch.float64),
                                                            20.0, 1e-10, math.log10(2.0), None).flatten()])
show("db_to_amp_f64", [repr(float(v)) for v in F.DB_to_amplitude(torch.tensor([3.0, -7.5], dtype=torch.float64), 0.1, 0.3)])

# task05 -- spectrogram normalized="frame_length" == torch.stft(normalized=True) (amplitude * n_fft^-0.5)
_r = torch.arange(16, dtype=torch.float32)
show("spec_fl_w8", g9(F.spectrogram(_r, 0, torch.hann_window(8), 8, 4, 8, 1.0, "frame_length")))
show("spec_fl_w6", g9(F.spectrogram(_r, 0, torch.hann_window(6), 8, 4, 6, 1.0, "frame_length")))

# task06 -- griffinlim: length passed to every istft; rand_init = torch.rand(complex) (seed 0)
_mag = F.spectrogram(_r, 0, torch.hann_window(8), 8, 4, 8, 2.0, False)
show("gl_len18", g9(F.griffinlim(_mag, torch.hann_window(8), 8, 4, 8, 2.0, 8, 0.99, 18, False)))
torch.manual_seed(0)
show("gl_rand_seed0", g9(F.griffinlim(_mag, torch.hann_window(8), 8, 4, 8, 2.0, 4, 0.5, None, True)))

# task07 -- mel filterbank at an odd rate (sample_rate // 2 for bins and default f_max); MFCC sample_rate + stereo top_db
for _ms, _nm in (("htk", None), ("slaney", "slaney")):
    show(f"fb11025_{_ms}", g9(F.melscale_fbanks(9, 0.0, float(11025 // 2), 3, 11025, _nm, _ms).t()))
_s = torch.sin(2 * math.pi * 0.11 * torch.arange(64.0))
_st = torch.stack([_s, 1e-4 * torch.cos(2 * math.pi * 0.23 * torch.arange(64.0))])
_mk = dict(n_fft=16, hop_length=8, n_mels=6)
show("mfcc_sr8000", g9(T.MFCC(sample_rate=8000, n_mfcc=4, melkwargs=_mk)(_s)))
show("mfcc_stereo", g9(T.MFCC(sample_rate=16000, n_mfcc=4, melkwargs=_mk)(_st)))

# task08 -- resample: kernel dtype follows the waveform; float32 target length; torchaudio kernel multiply order
_xr = torch.sin(2 * torch.pi * 3.0 * torch.arange(64.0) / 64.0).unsqueeze(0)
_r64 = F.resample(_xr.double(), 64, 48)
show("resample_f64 sum/[0,1]", [repr(float(_r64.sum())), repr(float(_r64[0, 1]))])
show("resample_len_180697", F.resample(torch.zeros(1, 180697), 44100, 16000).shape[-1])  # 65559
import torchaudio.functional.functional as _FF
_k, _wd = _FF._get_sinc_resample_kernel(48000, 16000, 16000, dtype=torch.float32)
show("kernel_f32_48k_16k width", _wd)
show("kernel_f32_48k_16k", [repr(float(v)) for v in _k.flatten()])
# pitch_shift at -12 steps on an odd length: Python round() is half-to-even (stretch length 1000, not 1001)
_ps12 = F.pitch_shift(torch.sin(torch.arange(2001.0) * 0.05).unsqueeze(0), 16000, -12, n_fft=512)
show("pitch_shift_m12 [2000]/[1999]/[1000]", [float(_ps12[0, 2000]), float(_ps12[0, 1999]), float(_ps12[0, 1000])])

# coverage gaps (filtering / beamforming / alignment paths never exercised before)
for _r in (48000, 88200):
    show(f"riaa_{_r}", g9(F.riaa_biquad(torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]), _r)))
_wf = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8, -0.7]])
show("flanger_tri", g9(F.flanger(_wf, 8000, modulation="triangular", interpolation="linear")))
_lg = torch.tensor([[2., 0, 0], [0, 3, 0], [0, 2, 0], [2, 0, 0], [0, 0, 3], [0, 0, 4]]).reshape(1, 6, 3)
_p, _sc = F.forced_align(torch.log_softmax(_lg, -1), torch.tensor([[1, 2]], dtype=torch.int32), blank=0)
show("forced_align_last_token", [_p.tolist(), g9(_sc)])
_Ms = torch.complex(torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2), torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2) * 0.2)
_Mn = torch.complex(torch.arange(13, 25, dtype=torch.float32).reshape(3, 2, 2) * 0.1, torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2) * 0.3)
_r1 = F.rtf_power(_Ms @ _Ms.conj().transpose(-1, -2) + 2 * torch.eye(2), _Mn @ _Mn.conj().transpose(-1, -2) + 2 * torch.eye(2), 0, n_iter=1)
show("rtf_power_n1", [g9(_r1.real), g9(_r1.imag)])

# ======================= task11 -- review round 2 (progress02 D12) =======================
# create_dct is built in float32 (a float64 build cast down is up to 1.9e-5 off at these entries).
_d40, _d13 = F.create_dct(40, 128, None), F.create_dct(13, 40, "ortho")
show("dct40x128 [122,36]/[84,34]/[98,28]", [repr(float(_d40[r, c])) for r, c in ((122, 36), (84, 34), (98, 28))])
show("dct13x40o [34,12]/[27,11]", [repr(float(_d13[r, c])) for r, c in ((34, 12), (27, 11))])
# spectral_centroid on float64 input: torchaudio's bin frequencies are a float32 linspace.
_t = torch.arange(3001, dtype=torch.float64)
_x2 = torch.stack([torch.sin(_t * 0.05), 0.5 * torch.cos(_t * 0.031)])
_sc = F.spectral_centroid(_x2, 16000, 0, torch.hann_window(384, dtype=torch.float64), 384, 96, 384)
show("sc_f64_16k [0,0]/[0,31]/[0,1]", [repr(float(_sc[0, i])) for i in (0, 31, 1)])

# ======================= progress04 -- surface audit (2026-10-23, task12-task19) =======================
import torchaudio.functional.filtering as _FF
import torchaudio.functional.functional as _FN

# task12 -- lfilter: torchaudio's compiled _lfilter (b/a0 FIR conv1d, then the IIR loop, fused multiply-subtract in
# this arm64 wheel); 3-D input with per-channel filters, an a0 != 1 filter and a double pole at 0.99.
_t = torch.arange(4000, dtype=torch.float64)
_k = torch.arange(6, dtype=torch.float64).reshape(6, 1)
_lx = (0.5 * torch.sin(_t * (0.01 + 0.007 * _k)) + 0.3 * torch.sin(_t * (0.3 + 0.05 * _k))).reshape(2, 3, 4000)
_la = torch.tensor([[1.0, -1.6, 0.8], [2.0, -0.5, 0.1], [1.0, -1.98, 0.9801]], dtype=torch.float64)
_lb = torch.tensor([[0.2, 0.3, 0.1], [0.5, 0.0, -0.5], [0.01, 0.02, 0.01]], dtype=torch.float64)
for _dt in (torch.float64, torch.float32):
    _y = F.lfilter(_lx.to(_dt), _la.to(_dt), _lb.to(_dt), clamp=False)
    show(f"lfilter3d_{_dt} [0,0,3999]/[0,1,123]/[1,2,3999]/[1,2,2000]/[0,2,57]",
         [repr(float(_y[i])) for i in ((0, 0, 3999), (0, 1, 123), (1, 2, 3999), (1, 2, 2000), (0, 2, 57))])
# Ill-conditioned order 6 (scipy butter(5, 0.1) * 1.7, so a0 = 1.7): pins the fused multiply-subtract and the b/a0
# order; the order-2 filters above give the same float32 values without them.
_a6 = torch.tensor([1.7, -6.773723303340971, 10.939274053468974, -8.931145789598856, 3.6807259465310254,
                    -0.6118780166080454], dtype=torch.float64)
_b6 = torch.tensor([0.00010165282662900549, 0.0005082641331450275, 0.001016528266290055, 0.001016528266290055,
                    0.0005082641331450275, 0.00010165282662900549], dtype=torch.float64)
for _dt in (torch.float64, torch.float32):
    _y = F.lfilter(_lx.to(_dt), _a6.to(_dt), _b6.to(_dt), clamp=False)
    show(f"lfilter_order6_{_dt} [0,0,3999]/[1,1,2500]/[1,2,777]",
         [repr(float(_y[i])) for i in ((0, 0, 3999), (1, 1, 2500), (1, 2, 777))])
# overdrive: torchaudio's compiled _overdrive_core_loop.
_ox = torch.stack([0.5 * torch.sin(_t[:3000] * 0.02) + 0.2 * torch.sin(_t[:3000] * 0.9),
                   0.35 * torch.sin(_t[:3000] * 0.05) - 0.1 * torch.cos(_t[:3000] * 1.3)])
for _dt in (torch.float64, torch.float32):
    _y = F.overdrive(_ox.to(_dt), 30.0, 40.0)
    show(f"overdrive_{_dt} [0,2999]/[1,1234]/[0,7]", [repr(float(_y[i])) for i in ((0, 2999), (1, 1234), (0, 7))])

# task13 -- vad on stereo float32/float64 with a pre-trigger (torchaudio keeps float32 measurement buffers).
_tv = torch.arange(16000, dtype=torch.float64) / 16000
_v = torch.zeros(2, 24000, dtype=torch.float64)
_v[0, 8000:] = 0.5 * torch.sin(2 * math.pi * 300 * _tv) * (1 + 0.3 * torch.sin(2 * math.pi * 3 * _tv))
_v[1, 9000:] = 0.3 * torch.sin(2 * math.pi * 520 * _tv[:15000])
for _dt in (torch.float32, torch.float64):
    _o = F.vad(_v.to(_dt), 16000, pre_trigger_time=0.1)
    show(f"vad_stereo_{_dt} len/dtype", [_o.shape[-1], str(_o.dtype)])  # 18400 for both

# task14 -- loudness at 11025 Hz: the gating step is Python round(1102.5) = 1102.
_tl = torch.arange(33075, dtype=torch.float64) / 11025
_lx2 = torch.stack([0.3 * torch.sin(2 * math.pi * 440 * _tl) * (1 + 0.5 * torch.sin(2 * math.pi * 3 * _tl)),
                    0.2 * torch.sin(2 * math.pi * 1000 * _tl)]).float()
show("loudness_11025", repr(float(F.loudness(_lx2, 11025))))
# dither, seeded: RPDF is torchaudio's output. GPDF: torchaudio sums into a VIEW of one input sample (mutating the
# input and re-reading it when an index repeats); the golden is its draws and math on an unaliased copy.
_dw = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [0.05, -0.05, 0.15, -0.15, 0.25, -0.25]])
torch.manual_seed(0)
show("dither_rpdf_seed0", F.dither(_dw.clone(), "RPDF").flatten().tolist())


def _gpdf_unaliased(w):
    cs, ts = w.size(0) - 1, w.size(1) - 1
    rc = int(torch.randint(cs, [1]).item()) if cs > 0 else 0
    rt = int(torch.randint(ts, [1]).item()) if ts > 0 else 0
    g = w[rc][rt].clone()
    for _ in range(6):
        c = int(torch.randint(cs, [1]).item())
        g = g + w[c][int(torch.randint(ts, [1]).item())]
    return torch.round(w * 32766 + g) / 32768


torch.manual_seed(0)
_g = _gpdf_unaliased(_dw)
show("dither_gpdf_seed0 (unaliased)", _g.flatten().tolist())
show("dither_gpdf_ns_seed0 (unaliased)", _FF._add_noise_shaping(_g, _dw).flatten().tolist())
# add_noise: the lengths mask is an int64 arange. A float16 one also rounds lengths 2049 to 2048 in the comparison,
# which drops index 2048 (the only signal sample here, so no noise would be added).
_w16 = torch.zeros(1, 2052, dtype=torch.float16)
_w16[0, 2048] = 1.0
_o16 = F.add_noise(_w16, torch.ones(1, 2052, dtype=torch.float16), torch.tensor([10.0], dtype=torch.float16),
                   torch.tensor([2049]))
show("add_noise_f16_mask [0,0]", repr(float(_o16[0, 0])))  # 0.006977081298828125
# add_noise energies are torch.linalg.vector_norm(...) ** 2 (a plain sum of squares is 1 ulp off in float32).
_tt = torch.arange(1000, dtype=torch.float64)
_kk = torch.arange(3, dtype=torch.float64).reshape(3, 1)
_an = F.add_noise((0.4 * torch.sin(_tt * (0.05 + 0.02 * _kk) + _kk)).float(),
                  (0.3 * torch.cos(_tt * (0.17 + 0.03 * _kk))).float(), torch.tensor([10.0, 3.0, -2.0]),
                  torch.tensor([1000, 700, 3]))
show("add_noise_f32 [2,356]/[2,315]/[2,179]", [repr(float(_an[2, i])) for i in (356, 315, 179)])

# task16 -- fractional n_steps (torchaudio documents "(fractional) steps"); 1 kHz keeps the kernel small.
_pp = F.pitch_shift(torch.sin(torch.arange(3001.0) * 0.05).unsqueeze(0), 1000, 0.5)
show("pitch_shift_frac [0,100]/[0,1500]/[0,2900]", [repr(float(_pp[0, i])) for i in (100, 1500, 2900)])

# task17 -- mask_along_axis draws torch.rand(1) on the CPU in float32, whatever the spectrogram's dtype.
_ms = torch.arange(1, 1 + 2 * 30 * 100, dtype=torch.float64).reshape(2, 30, 100)
torch.manual_seed(0)
show("mask_f64_seed0 time cols", (F.mask_along_axis(_ms, 20, 0.0, 2)[0, 0] == 0).nonzero().flatten().tolist())
torch.manual_seed(0)
show("mask_i64_seed0 time cols", (F.mask_along_axis(_ms.long(), 20, 0.0, 2)[0, 0] == 0).nonzero().flatten().tolist())
torch.manual_seed(0)
show("mask_f64_seed0 freq rows", (F.mask_along_axis(_ms, 10, 0.0, 1)[0, :, 0] == 0).nonzero().flatten().tolist())
# detect_pitch_frequency: freq = reciprocal(EPS + idx) * sample_rate (Python int / tensor); vector_norm in the NCCF.
_tp = torch.arange(16001, dtype=torch.float64) / 16000
_dp = torch.stack([torch.sin(2 * math.pi * 180 * _tp) + 0.4 * torch.sin(2 * math.pi * 360 * _tp + 0.3),
                   torch.sin(2 * math.pi * 233 * _tp) * (1 + 0.2 * torch.sin(2 * math.pi * 5 * _tp)),
                   torch.sin(2 * math.pi * 97 * _tp) + 0.1 * torch.sin(2 * math.pi * 1100 * _tp),
                   torch.sin(2 * math.pi * 410 * _tp + 1.0)]).reshape(2, 2, 16001).float()
_pf = F.detect_pitch_frequency(_dp, 16000)
show("detect_pitch shape/sum", [list(_pf.shape), repr(float(_pf.double().sum()))])
show("detect_pitch [0,0,5]/[0,1,50]/[1,0,85]/[1,1,0]",
     [repr(float(_pf[i])) for i in ((0, 0, 5), (0, 1, 50), (1, 0, 85), (1, 1, 0))])
_nc = _FN._compute_nccf(_dp[0, 0, :4000], 16000, 0.01, 85)
show("nccf [3,40]/[10,88]/[24,150]", [repr(float(_nc[i])) for i in ((3, 40), (10, 88), (24, 150))])
# On a noise-like signal vector_norm and sqrt(sum(x * x)) differ (up to 5e-7 relative); on the tones above they agree.
_tn = torch.arange(4000, dtype=torch.float64)
_ncn = _FN._compute_nccf((torch.sin(_tn * _tn * 1e-3) + 0.5 * torch.sin(_tn * 2.3)).float(), 16000, 0.01, 85)
show("nccf_noise [24,158]/[24,157]/[24,156]", [repr(float(_ncn[24, i])) for i in (158, 157, 156)])

# task18 -- forced_align: the compiled kernel's tie rules, terminal rule and native-dtype accumulation.
_p, _ = F.forced_align(torch.log_softmax(torch.zeros(1, 10, 5), -1), torch.tensor([[1, 2, 3]]))
show("fa_tie_uniform", _p.flatten().tolist())
_lg = torch.log_softmax(torch.sin(torch.arange(30.0) * 1.3).reshape(1, 6, 5), -1)
show("fa_tight_ok (T == L + R)", F.forced_align(_lg, torch.tensor([[1, 1, 2, 2]]))[0].flatten().tolist())
_i = torch.arange(32, dtype=torch.float64)
_lg64 = torch.log_softmax((torch.floor((_i * 2) % 3) + 1e-10 * torch.sin(_i * 1.7 + 2)).reshape(1, 8, 4), -1)
show("fa_f64 / as float32", [F.forced_align(_lg64, torch.tensor([[1, 2, 3]]))[0].flatten().tolist(),
                              F.forced_align(_lg64.float(), torch.tensor([[1, 2, 3]]))[0].flatten().tolist()])
_lg16 = torch.log_softmax(3 * torch.sin(torch.arange(60.0) * (0.37 + 0.01 * 217)).reshape(1, 12, 5), -1).half()
show("fa_f16 logp", _lg16.flatten().tolist())
show("fa_f16 / as float32", [F.forced_align(_lg16, torch.tensor([[1, 2, 3, 4]]))[0].flatten().tolist(),
                             F.forced_align(_lg16.float(), torch.tensor([[1, 2, 3, 4]]))[0].flatten().tolist()])
_lgb = torch.log_softmax(torch.cos(torch.arange(40.0) * 0.9).reshape(1, 8, 5), -1)
_p, _s = F.forced_align(_lgb, torch.tensor([[1, 3, 4]], dtype=torch.int32), blank=2)
show("fa_blank2 paths/dtype", [_p.flatten().tolist(), str(_p.dtype)])
show("fa_blank2 scores", [repr(float(v)) for v in _s.flatten()])

# task19 -- beamforming complex128 batched goldens (functional layer, pinned to torchaudio).
_n = torch.arange(2 * 3 * 5 * 7, dtype=torch.float64)
_spec = torch.complex(torch.sin(_n * 0.37), torch.cos(_n * 0.11)).reshape(2, 3, 5, 7)
_mask = (0.5 + 0.4 * torch.sin(torch.arange(2 * 5 * 7, dtype=torch.float64) * 0.23)).reshape(2, 5, 7)
_psd = F.psd(_spec, _mask, normalize=False)
show("psd128 [1,4,2,0]/[0,0,1,2]", [[repr(float(_psd[i].real)), repr(float(_psd[i].imag))] for i in ((1, 4, 2, 0), (0, 0, 1, 2))])
_psdn = F.psd(_spec, 1 - _mask)
_ws = F.mvdr_weights_souden(F.psd(_spec, _mask), _psdn, 1)
show("souden128 [1,3,2]/[0,0,0]", [[repr(float(_ws[i].real)), repr(float(_ws[i].imag))] for i in ((1, 3, 2), (0, 0, 0))])
_wr = F.mvdr_weights_rtf(F.rtf_power(F.psd(_spec, _mask), _psdn, 2, n_iter=5), _psdn, 0)
show("rtf_power5+mvdr_rtf128 [1,2,1]/[0,4,2]", [[repr(float(_wr[i].real)), repr(float(_wr[i].imag))] for i in ((1, 2, 1), (0, 4, 2))])
