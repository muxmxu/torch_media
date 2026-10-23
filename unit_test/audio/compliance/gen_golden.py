#!/usr/bin/env python3
"""Regenerate golden.hpp for audio_test_compliance from torchaudio.compliance.kaldi 2.5.1 (dev-time only).

    .venv/bin/python -I unit_test/audio/compliance/gen_golden.py [--dump DIR]

Every case is one call on a seeded torch.randn input that main.cpp rebuilds bit for bit (libtorch's CPU generator gives
the same stream); dither cases reseed right before the call, so the dither draw matches too. Per case golden.hpp holds
the output shape and dtype, its sum, the checksum sum(Y * Z) with Z = randn(Y.shape, float64) under seed 99, max |Y| and four
elements. The C++ call is generated from the same spec, so the two sides cannot drift apart.

The tolerance basis is torchaudio's own variation, printed into golden.hpp: the same input values passed
non-contiguous, and float64 instead of float32 input, move its output by the measured fraction of max |Y|.

--dump DIR also writes each full torchaudio output (raw, native dtype); TM_GOLDEN_DIR=DIR ./audio_test_compliance then
prints the element-wise max |delta| of every case (the parity numbers in develop_log 2026-10-23/progress06).
"""
import argparse
import os

import torch
import torchaudio.compliance.kaldi as K


def wave16():  # main.cpp: wave16()
    torch.manual_seed(0)
    w = torch.randn(2, 8000) * 3000
    w[0, 2000:3000] = 0  # digital silence: frames 13-16 hit the epsilon / energy floors
    return w


def wave8():  # main.cpp: wave8()
    torch.manual_seed(1)
    return torch.randn(1, 4000) * 3000


INPUTS = {"wave16": wave16, "wave8": wave8}
WENET = dict(num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0, energy_floor=0.0)

# (name, function, input, float64, kwargs, dither seed)
FEATURES = [
    ("fbank_default", "fbank", "wave16", False, {}, None),
    # WeNet / WeSpeaker / 3D-Speaker inference front end
    ("fbank_80_dither0_floor0", "fbank", "wave16", False, WENET, None),
    # WeNet training: dither 1.0 from the seeded global generator
    ("fbank_80_dither1", "fbank", "wave16", False, {**WENET, "dither": 1.0}, 123),
    # FunASR WavFrontend
    ("fbank_funasr", "fbank", "wave16", False, {**WENET, "window_type": "hamming", "snip_edges": True}, None),
    ("fbank_htk_energy", "fbank", "wave16", False, dict(use_energy=True, htk_compat=True), None),
    ("fbank_energy_not_raw", "fbank", "wave16", False, dict(use_energy=True, raw_energy=False), None),
    ("fbank_hanning", "fbank", "wave16", False, dict(window_type="hanning"), None),
    ("fbank_rectangular", "fbank", "wave16", False, dict(window_type="rectangular"), None),
    ("fbank_blackman", "fbank", "wave16", False, dict(window_type="blackman", blackman_coeff=0.4), None),
    ("fbank_no_snip", "fbank", "wave16", False, dict(snip_edges=False), None),
    ("fbank_no_snip_trim", "fbank", "wave16", False, dict(snip_edges=False, frame_length=10.0, frame_shift=25.0), None),
    # (n + shift / 2) / shift rounds up here: 8000 mod 144 = 80 >= 72 (a truncating count drops the last frame)
    ("fbank_no_snip_shift9", "fbank", "wave16", False, dict(snip_edges=False, frame_shift=9.0), None),
    ("fbank_no_snip_pad0", "fbank", "wave16", False, dict(snip_edges=False, frame_length=20.0, frame_shift=20.0), None),
    ("fbank_no_pow2", "fbank", "wave16", False, dict(round_to_power_of_two=False), None),
    ("fbank_subtract_mean", "fbank", "wave16", False, {**WENET, "subtract_mean": True}, None),
    ("fbank_vtln_0.9", "fbank", "wave16", False, dict(vtln_warp=0.9), None),
    ("fbank_vtln_1.15", "fbank", "wave16", False, dict(vtln_warp=1.15, vtln_low=200.0, vtln_high=7000.0), None),
    ("fbank_linear_magnitude", "fbank", "wave16", False, dict(use_log_fbank=False, use_power=False), None),
    ("fbank_channel1_plain", "fbank", "wave16", False,
     dict(channel=1, preemphasis_coefficient=0.0, remove_dc_offset=False, low_freq=100.0, high_freq=7000.0,
          min_duration=0.4), None),
    ("fbank_huge_shift", "fbank", "wave16", False, dict(frame_shift=1e6), None),
    ("fbank_huge_shift_no_frames", "fbank", "wave16", False,
     dict(frame_shift=1e6, snip_edges=False, preemphasis_coefficient=0.0), None),
    ("fbank_8k", "fbank", "wave8", False, dict(sample_frequency=8000.0, num_mel_bins=40, high_freq=-200.0), None),
    ("fbank_8k_vtln", "fbank", "wave8", False,
     dict(sample_frequency=8000.0, vtln_warp=0.85, vtln_low=60.0, vtln_high=-400.0, low_freq=40.0), None),
    ("fbank_f64", "fbank", "wave16", True, {}, None),
    ("fbank_f64_80_dither1", "fbank", "wave16", True, {**WENET, "dither": 1.0}, 7),
    ("fbank_f64_8k_htk", "fbank", "wave8", True,
     dict(sample_frequency=8000.0, use_energy=True, htk_compat=True, window_type="hamming"), None),
    ("mfcc_default", "mfcc", "wave16", False, {}, None),
    ("mfcc_40_20_no_lifter", "mfcc", "wave16", False, dict(num_mel_bins=40, num_ceps=20, cepstral_lifter=0.0), None),
    ("mfcc_htk_energy", "mfcc", "wave16", False, dict(htk_compat=True, use_energy=True), None),
    ("mfcc_htk", "mfcc", "wave16", False, dict(htk_compat=True, cepstral_lifter=12.0), None),
    ("mfcc_energy_mean", "mfcc", "wave16", False, dict(use_energy=True, subtract_mean=True, num_ceps=23), None),
    # num_ceps is a Python slice of the DCT columns: 0 gives none, -1 all but the last (no lifter: arange(-1) fails)
    ("mfcc_zero_ceps", "mfcc", "wave16", False, dict(num_ceps=0), None),
    ("mfcc_neg_ceps_no_lifter", "mfcc", "wave16", False, dict(num_ceps=-1, cepstral_lifter=0.0), None),
    ("mfcc_8k_f64", "mfcc", "wave8", True, dict(sample_frequency=8000.0, dither=0.5), 5),
    ("spectrogram_default", "spectrogram", "wave16", False, {}, None),
    ("spectrogram_mean_rect", "spectrogram", "wave16", False,
     dict(subtract_mean=True, raw_energy=False, window_type="rectangular", energy_floor=0.0), None),
    ("spectrogram_dither", "spectrogram", "wave16", False, dict(dither=1.0, snip_edges=False), 11),
    ("spectrogram_8k_f64", "spectrogram", "wave8", True, dict(sample_frequency=8000.0, round_to_power_of_two=False),
     None),
]

def grid():  # main.cpp: torch::linspace(0, 8000, 161)
    return torch.linspace(0, 8000, 161)


GRID = "torch::linspace(0, 8000, 161)"
# (name, torchaudio call, C++ expression)
HELPERS = [
    ("mel_banks", lambda: K.get_mel_banks(23, 512, 16000., 20., 0., 100., -500., 1.)[0],
     "std::get<0>(kd::get_mel_banks(23, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0))"),
    ("mel_banks_center", lambda: K.get_mel_banks(23, 512, 16000., 20., 0., 100., -500., 1.)[1],
     "std::get<1>(kd::get_mel_banks(23, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0))"),
    ("mel_banks_vtln", lambda: K.get_mel_banks(40, 256, 8000., 64., -100., 100., -500., 1.1)[0],
     "std::get<0>(kd::get_mel_banks(40, 256, 8000.0, 64.0, -100.0, 100.0, -500.0, 1.1))"),
    ("mel_banks_vtln_center", lambda: K.get_mel_banks(40, 256, 8000., 64., -100., 100., -500., 1.1)[1],
     "std::get<1>(kd::get_mel_banks(40, 256, 8000.0, 64.0, -100.0, 100.0, -500.0, 1.1))"),
    ("vtln_warp_freq", lambda: K.vtln_warp_freq(100., 7500., 20., 8000., 0.9, grid()),
     f"kd::vtln_warp_freq(100.0, 7500.0, 20.0, 8000.0, 0.9, {GRID})"),
    ("vtln_warp_mel_freq", lambda: K.vtln_warp_mel_freq(100., 7500., 20., 8000., 1.2, K.mel_scale(grid())),
     f"kd::vtln_warp_mel_freq(100.0, 7500.0, 20.0, 8000.0, 1.2, kd::mel_scale({GRID}))"),
    ("inverse_mel_scale", lambda: K.inverse_mel_scale(grid() / 2), f"kd::inverse_mel_scale({GRID} / 2)"),
    ("dct_matrix", lambda: K._get_dct_matrix(13, 23), "kdd::get_dct_matrix(13, 23)"),
    ("lifter_coeffs", lambda: K._get_lifter_coeffs(13, 22.), "kdd::get_lifter_coeffs(13, 22.0)"),
] + [(f"window_{t}", lambda t=t: K._feature_window_function(t, 400, 0.42, "cpu", torch.float32),
      f"kdd::feature_window_function(kd::window_type::{t}, 400, 0.42, torch::kFloat)")
     for t in ("hamming", "hanning", "povey", "rectangular", "blackman")]


def cpp_value(key, v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if key == "window_type":
        return f"kd::window_type::{v}"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def run_feature(fn, inp, f64, kwargs, seed):
    w = INPUTS[inp]()
    if f64:
        w = w.double()
    if seed is not None:
        torch.manual_seed(seed)
    return getattr(K, fn)(w, **kwargs)


def cpp_feature(fn, inp, f64, kwargs, seed):
    opt = f"kd::{fn}_option()" + "".join(f".{k}({cpp_value(k, v)})" for k, v in kwargs.items())
    w = f"{inp}()" + (".to(torch::kDouble)" if f64 else "")
    reseed = f"torch::manual_seed({seed}); " if seed is not None else ""
    return f"[] {{ const auto w = {w}; {reseed}return kd::{fn}(w, {opt}); }}"


def golden(y):
    yd = y.double()
    torch.manual_seed(99)
    chk = (yd * torch.randn(y.shape, dtype=torch.float64)).sum().item()
    flat = yd.flatten()
    n = flat.numel()
    idx = sorted({0, n // 3, n // 2 + 1, n - 1}) if n else []
    els = ", ".join(f"{{{i}, {flat[i].item()!r}}}" for i in idx)
    shape = ", ".join(str(s) for s in y.shape)
    maxabs = yd.abs().max().item() if n else 0.0
    f64 = "true" if y.dtype == torch.float64 else "false"
    return f"{{{{{shape}}}, {f64}, {yd.sum().item()!r}, {chk!r}, {maxabs!r}, {{{els}}}}}"


def self_variation(f64):
    """max over the non-dither feature cases of |torchaudio(x) - torchaudio(same values, other layout or dtype)|."""
    worst_layout = worst_dtype = 0.0
    for _, fn, inp, is64, kw, seed in FEATURES:
        if seed is not None or is64 != f64:
            continue
        w = INPUTS[inp]()
        w = w.double() if f64 else w
        y = getattr(K, fn)(w, **kw)
        if y.numel() == 0:
            continue
        scale = max(1.0, y.abs().max().item())
        if kw.get("snip_edges", True):  # without snip_edges torchaudio strides a non-contiguous input wrongly
            yn = getattr(K, fn)(w.t().contiguous().t(), **kw)
            worst_layout = max(worst_layout, (y - yn).abs().max().item() / scale)
        other = getattr(K, fn)(w.float() if f64 else w.double(), **kw)
        worst_dtype = max(worst_dtype, (y.double() - other.double()).abs().max().item() / scale)
    return worst_layout, worst_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="also write every full output as DIR/<case>.bin (raw, native dtype)")
    args = ap.parse_args()
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    rows = []
    for name, fn, inp, f64, kw, seed in FEATURES:
        y = run_feature(fn, inp, f64, kw, seed)
        rows.append((name, cpp_feature(fn, inp, f64, kw, seed), y))
    for name, call, cpp in HELPERS:
        rows.append((name, f"[] {{ return {cpp}; }}", call()))
    lay32, dt32 = self_variation(False)
    lay64, dt64 = self_variation(True)
    out = [
        "// Generated by unit_test/audio/compliance/gen_golden.py (torchaudio.compliance.kaldi 2.5.1). Do not edit by",
        "// hand; rerun the script. Included by main.cpp after wave16() / wave8() and the kd / kdd aliases.",
        "// clang-format off",
        "#pragma once",
        "",
        "// torchaudio's own variation, as a fraction of max(1, max |Y|) over the non-dither cases:",
        f"//   float32: {lay32:.3g} for a non-contiguous input of the same values, {dt32:.3g} against float64 input",
        f"//   float64: {lay64:.3g} for a non-contiguous input of the same values, {dt64:.3g} against float32 input",
        f"inline constexpr double torchaudio_variation_f32 = {max(lay32, dt32)!r};",
        f"inline constexpr double torchaudio_variation_f64 = {lay64!r};",
        "",
        "inline const kaldi_case kaldi_cases[] = {",
    ]
    for name, cpp, y in rows:
        out.append(f'    {{"{name}", {cpp},')
        out.append(f"     {golden(y)}}},")
        if args.dump:
            with open(os.path.join(args.dump, f"{name}.bin"), "wb") as f:
                f.write(y.contiguous().numpy().tobytes())
    out.append("};")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.hpp")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path}: {len(rows)} cases; variation f32 {lay32:.3g}/{dt32:.3g}, f64 {lay64:.3g}/{dt64:.3g}")


if __name__ == "__main__":
    main()
