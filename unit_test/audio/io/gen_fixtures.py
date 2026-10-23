#!/usr/bin/env python3
"""Regenerate fixtures.hpp for audio_test_io (dev-time only: the test itself needs neither Python nor an encoder).

Every fixture is a tiny encoded file (<= ~4 KB) built from a closed-form or fixed signal, written by this script's own
RIFF writer or by the local encoders (sox, flac, lame, ffmpeg), and embedded as bytes because the repo ignores *.wav /
*.mp3. Next to it go torchaudio 2.5.1's info() and, where the test does not use the closed form, its load() output
(sox backend). The checks that hold by construction are asserted here: torchaudio decodes the closed-form integer PCM
and FLAC fixtures to exactly pcm24() / 2^(bits - 1), which main.cpp recomputes. Known decoder differences are printed.

    .venv/bin/python -I unit_test/audio/io/gen_fixtures.py [--keep DIR]   # rewrites fixtures.hpp next to this file
"""
import argparse
import os
import struct
import subprocess
import tempfile

import numpy as np
import torch
import torchaudio

SR = 8000
N = 200  # frames of the closed-form PCM fixtures (25 ms)
N_LOSSY = 800  # frames of the MP3 / Vorbis source (0.1 s: a few codec frames)
PCM_SUBTYPE = bytes.fromhex("0100000000001000800000aa00389b71")  # KSDATAFORMAT_SUBTYPE_PCM


def pcm24(channels):
    """Closed-form 24-bit source, mirrored by pcm24() in main.cpp: wrapping ramps that start with -2^23, 2^23-1, 0."""
    n = np.arange(N, dtype=np.int64)
    c = np.arange(channels, dtype=np.int64)[:, None]
    v = (n * 40503 + c * 1234567) % (1 << 24) - (1 << 23)
    v[:, :3] = [-(1 << 23), (1 << 23) - 1, 0]
    return v


def pcm32(channels):
    """pcm24() with a non-zero low byte, mirrored by pcm32() in main.cpp."""
    n = np.arange(N, dtype=np.int64)
    c = np.arange(channels, dtype=np.int64)[:, None]
    return pcm24(channels) * 256 + (n * 37 + c * 11) % 256


def riff_wav(samples, bits, tag, rate=SR):
    """WAV bytes for interleaved-able [C, N] `samples` (numpy, already in the stored dtype). tag 1: PCM, 3: IEEE
    float, 0xFFFE: WAVE_FORMAT_EXTENSIBLE PCM (what sox and torchaudio write for 24 / 32-bit)."""
    channels = samples.shape[0]
    if bits == 24:
        u = (samples.T.reshape(-1) & 0xFFFFFF).astype("<u4")
        data = np.stack([u & 255, (u >> 8) & 255, u >> 16], axis=1).astype(np.uint8).tobytes()
    else:
        data = np.ascontiguousarray(samples.T).tobytes()
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    if tag == 0xFFFE:
        fmt += struct.pack("<HHI", 22, bits, 0) + PCM_SUBTYPE
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if tag == 3:
        body += b"fact" + struct.pack("<II", 4, samples.shape[1])
    body += b"data" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def data_chunk(raw):
    """The data chunk of a RIFF/WAVE file image."""
    i = 12
    while i + 8 <= len(raw):
        tag, size = raw[i : i + 4], struct.unpack("<I", raw[i + 4 : i + 8])[0]
        if tag == b"data":
            return raw[i + 8 : i + 8 + size]
        i += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def k24(x):
    """torchaudio's sox backend yields multiples of 2^-24: store them exactly as integers."""
    k = x.astype(np.float64) * (1 << 24)
    assert np.array_equal(k, np.round(k)), "torchaudio output is not on the 2^-24 grid"
    return k.astype(np.int64)


def best_lag(decoded, source):
    """Lag L at which decoded[:, L + n] best matches source[:, n] (normalized correlation)."""
    scores = []
    for lag in range(decoded.shape[1] - source.shape[1] + 1):
        seg = decoded[:, lag : lag + source.shape[1]]
        scores.append((seg * source).sum() / (np.linalg.norm(seg) * np.linalg.norm(source) + 1e-30))
    return int(np.argmax(scores))


def build(tmp):
    fixtures = []  # (name, path, bytes, golden or None, golden_lag); sox needs the extension, main.cpp does not
    rejected = []  # (name, bytes): valid files load_audio must refuse with std::runtime_error

    def add(name, raw, ext=".wav", golden=None, lag=0):
        path = os.path.join(tmp, name + ext)
        with open(path, "wb") as f:
            f.write(raw)
        fixtures.append([name, path, raw, golden, lag])
        return path

    def read(path):
        with open(path, "rb") as f:
            return f.read()

    # Closed-form PCM WAV in every torchaudio.save encoding, plus FLAC / W64 / RF64 / G.711 / ADPCM from it.
    for label, ch in (("mono", 1), ("stereo", 2)):
        v, s32 = pcm24(ch), pcm32(ch)
        add(f"wav_u8_{label}", riff_wav(((v >> 16) + 128).astype(np.uint8), 8, 1))
        p16 = add(f"wav_s16_{label}", riff_wav((v >> 8).astype("<i2"), 16, 1))
        p24 = add(f"wav_s24_{label}", riff_wav(v, 24, 0xFFFE))
        add(f"wav_s32_{label}", riff_wav(s32.astype("<i4"), 32, 0xFFFE))
        add(f"wav_f32_{label}", riff_wav((v * 1.25 / 2**23).astype("<f4"), 32, 3))
        add(f"wav_f64_{label}", riff_wav((s32 / 2**31 * 1.5).astype("<f8"), 64, 3))
        for bits, src in ((16, p16), (24, p24)):
            out = os.path.join(tmp, f"flac{bits}_{label}.flac")
            run("flac", "--silent", "--no-padding", "-8", "-f", "-o", out, src)
            add(f"flac_{bits}_{label}", read(out), ".flac")
    p16 = os.path.join(tmp, "wav_s16_mono.wav")
    for enc in ("u-law", "a-law", "ima-adpcm"):
        out = os.path.join(tmp, f"{enc}.wav")
        run("sox", "-D", p16, "-e", enc, out)
        add("wav_" + enc.replace("-", "") + "_mono", read(out), golden=True)
    run("sox", p16, os.path.join(tmp, "s16.w64"))
    add("w64_s16_mono", read(os.path.join(tmp, "s16.w64")), ".w64")
    run("ffmpeg", "-loglevel", "error", "-y", "-i", p16, "-rf64", "always", "-c", "copy",
        os.path.join(tmp, "rf64.wav"))
    add("rf64_s16_mono", read(os.path.join(tmp, "rf64.wav")))

    # Lossy: two tones per channel, 16-bit source. The mono MP3 carries an ID3v2 tag and LAME's gapless Info frame,
    # the stereo one neither (detected by frame sync; dr_mp3 then cannot trim delay / padding).
    t = np.arange(N_LOSSY) / SR
    tone = np.stack([0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 1250 * t + 1),
                     0.4 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)])
    for label, ch in (("mono", 1), ("stereo", 2)):
        src16 = np.round(tone[:ch] * 32767).astype("<i2")
        src = os.path.join(tmp, f"tone_{label}.wav")
        with open(src, "wb") as f:
            f.write(riff_wav(src16, 16, 1))
        mp3 = os.path.join(tmp, f"{label}.mp3")
        extra = ("--add-id3v2", "--tt", "torchmedia") if ch == 1 else ("-t",)
        run("lame", "--silent", "-b", "32", *extra, src, mp3)
        ta = torchaudio.load(mp3)[0].numpy()
        # Gapless file: the golden starts where libmad's output lines up with the source (Info frame + delays).
        lag = best_lag(ta, src16 / 32768.0) if ch == 1 else 0
        add(f"mp3_{label}", read(mp3), ".mp3", golden=True, lag=lag)
        ogg = os.path.join(tmp, f"{label}.ogg")
        run("sox", "-D", src, "-C", "-1", ogg)
        add(f"ogg_{label}", read(ogg), ".ogg", golden=True)

    # 5.1 Vorbis: libvorbis's 6-channel mapping at 44.1 / 48 kHz puts 5 channels in one residue-type-2 submap, which
    # stb_vorbis v1.22 misdecoded (patched, THIRD_PARTY_LICENSES). Its setup header alone is ~7 KB.
    t51 = np.arange(220) / 44100
    src51 = np.round(0.4 * np.sin(2 * np.pi * np.array([440, 550, 660, 770, 880, 990])[:, None] * t51) * 32767)
    src = os.path.join(tmp, "tone_51.wav")
    with open(src, "wb") as f:
        f.write(riff_wav(src51.astype("<i2"), 16, 1, 44100))
    ogg = os.path.join(tmp, "51.ogg")
    run("sox", "-D", src, "-C", "-1", ogg)
    add("ogg_51", read(ogg), ".ogg", golden=True)

    # 32-bit FLAC (torchaudio reads it; dr_flac takes its frame-header bit-depth code as reserved)
    out = os.path.join(tmp, "flac32.flac")
    run("flac", "--silent", "--no-padding", "-f", "-o", out, os.path.join(tmp, "wav_s32_mono.wav"))
    assert torchaudio.info(out).bits_per_sample == 32
    rejected.append(("flac_32_mono", read(out)))

    rows = []
    for name, path, raw, golden, lag in fixtures:
        info = torchaudio.info(path)
        ta = torchaudio.load(path)[0].numpy()
        ch = info.num_channels
        expect = {  # closed form, as main.cpp computes it
            "wav_u8": (pcm24(ch) >> 16) / 128.0,
            "wav_s16": (pcm24(ch) >> 8) / 32768.0,
            "flac_16": (pcm24(ch) >> 8) / 32768.0,
            "w64_s16": (pcm24(ch) >> 8) / 32768.0,
            "rf64_s16": (pcm24(ch) >> 8) / 32768.0,
            "wav_s24": pcm24(ch) / 2.0**23,
            "flac_24": pcm24(ch) / 2.0**23,
            "wav_s32": (pcm32(ch) / 2.0**31).astype(np.float32),
            "wav_f32": (pcm24(ch) * 1.25 / 2**23).astype(np.float32),
            "wav_f64": (pcm32(ch) / 2**31 * 1.5).astype(np.float32),
        }.get(name.rsplit("_", 1)[0])
        if expect is not None and name.split("_")[1] in ("s32", "f32", "f64"):
            diff = np.abs(ta - expect)
            print(f"{name}: torchaudio (sox) differs from the stored values by up to {diff.max():.3g} "
                  f"({np.mean(diff > 0):.0%} of samples)")
        elif expect is not None:
            assert np.array_equal(ta, expect.astype(np.float32)), f"{name}: torchaudio != closed form"
        print(f"{name}: {len(raw)} bytes, torchaudio {info}")
        g = k24(ta[:, lag:]) if golden else None
        rows.append((name, raw, info, g, lag))
    return rows, rejected


def save_goldens(tmp):
    """torchaudio.save data chunks for a float32 probe and integer inputs, plus its (encoding, bits) derivation."""
    special = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -2.0, 1 / 32768, -1 / 32768, 0.25, 0.99999, 1 - 2**-24, -1 + 2**-24,
               1 - 2**-16, 1 - 2**-17, 1e-10, -1e-10, float("nan"), float("inf"), float("-inf")]
    special += [s * (k + 0.5) / 2**b for b in (7, 15, 23) for k in (0, 1, 2, 77) for s in (1, -1)]  # rounding ties
    rng = np.random.default_rng(0)
    probe = np.concatenate([np.array(special, np.float32), rng.uniform(-1.25, 1.25, 200).astype(np.float32)])
    i16 = np.array([0, 1, -1, 2, -2, 127, 128, -128, -129, 255, 256, 12345, -12345, 32767, -32768], np.int16)
    i32 = [0, 1, -1, 63, 64, 65, -64, -65, 127, 128, 129, 32767, 32768, 32769, -32768, -32769, 65535, 65536,
           123456789, -123456789, 2**31 - 1, -(2**31), 2**31 - 64, 2**31 - 32768, 2**31 - 32769, 2**23, 2**23 + 1]
    i32 = np.array(i32 + [k * 65536 + 32768 for k in (-3, 0, 5)] + [k * 2**24 + 2**23 for k in (-2, 0, 3)], np.int32)
    u8 = np.arange(256, dtype=np.uint8)
    inputs = {"float32": probe, "int16": i16, "int32": i32, "uint8": u8}
    path = os.path.join(tmp, "save.wav")
    goldens = []
    for dtype, values in inputs.items():
        for enc, bits in (("PCM_S", 16), ("PCM_S", 24), ("PCM_S", 32), ("PCM_U", 8), ("PCM_F", 32), ("PCM_F", 64)):
            if enc == "PCM_F" and (dtype == "float32" or (dtype == "int32" and bits == 32)):
                continue  # torchaudio's sox path clips / rounds these; main.cpp checks the exact values instead
            torchaudio.save(path, torch.from_numpy(values).reshape(1, -1), SR, encoding=enc, bits_per_sample=bits)
            with open(path, "rb") as f:
                goldens.append((dtype, enc, bits, data_chunk(f.read())))
    combos = []
    for dtype in inputs:
        for enc in ("", "PCM_S", "PCM_U", "PCM_F", "ULAW", "ALAW"):
            for bits in (0, 8, 12, 16, 24, 32, 64):
                tensor = torch.from_numpy(inputs[dtype][:4]).reshape(1, -1)
                try:
                    torchaudio.save(path, tensor, SR, encoding=enc or None, bits_per_sample=bits or None)
                    info = torchaudio.info(path)
                    combos.append((dtype, enc, bits, info.encoding, info.bits_per_sample))
                except RuntimeError:
                    combos.append((dtype, enc, bits, "", 0))
    return inputs, goldens, combos


def array(ctype, name, values, per_line=24):
    values = [str(int(v)) for v in values]
    lines = [", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)]
    return f"    inline constexpr {ctype} {name}[] = {{\n            " + ",\n            ".join(lines) + "};\n"


def emit(rows, rejected, inputs, goldens, combos, out):
    s = ["// Generated by unit_test/audio/io/gen_fixtures.py (torchaudio 2.5.1 sox backend, sox, flac, lame, ffmpeg).",
         "// Do not edit by hand; rerun the script. Audio is embedded as bytes because the repo ignores *.wav / *.mp3.",
         "// clang-format off", "#pragma once", "#include <cstddef>", "#include <cstdint>", "",
         "namespace tm_fixture {",
         f"    inline constexpr int sample_rate = {SR};",
         f"    inline constexpr int64_t pcm_frames = {N}; // closed-form fixtures: pcm24() / pcm32() in main.cpp",
         f"    inline constexpr int64_t lossy_frames = {N_LOSSY}; // MP3 / Vorbis source length", "",
         "    // One encoded file with torchaudio.info() and, for codecs without a closed form, torchaudio.load() from",
         "    // frame golden_lag on, as integers k with sample = k / 2^24 (null: main.cpp computes the closed form).",
         "    struct fixture {",
         "        const char *name;",
         "        const unsigned char *data;",
         "        size_t size;",
         "        int sample_rate;",
         "        int64_t num_frames;",
         "        int num_channels;",
         "        int bits_per_sample;",
         "        const char *encoding;",
         "        const int32_t *golden;",
         "        int64_t golden_frames;",
         "        int64_t golden_lag;",
         "    };", ""]
    for name, raw, _, g, _ in rows:
        s.append(array("unsigned char", name, raw))
        if g is not None:
            s.append(array("int32_t", name + "_golden", g.T.reshape(-1)))  # interleaved [frames, channels]
    s.append("    inline constexpr fixture fixtures[] = {")
    for name, raw, info, g, lag in rows:
        golden = f"{name}_golden, {g.shape[1]}" if g is not None else "nullptr, 0"
        s.append(f'            {{"{name}", {name}, sizeof {name}, {info.sample_rate}, {info.num_frames}, '
                 f'{info.num_channels}, {info.bits_per_sample}, "{info.encoding}", {golden}, {lag}}},')
    s.append("    };\n")
    s.append("    // Valid files load_audio refuses (std::runtime_error).")
    for name, raw in rejected:
        s.append(array("unsigned char", name, raw))
    s.append("    // torchaudio.save inputs ([1, n]): float32 bit patterns (NaN / inf, rounding ties, uniform noise).")
    s.append(array("uint32_t", "save_float32", inputs["float32"].view(np.uint32)))
    s.append(array("int16_t", "save_int16", inputs["int16"]))
    s.append(array("int32_t", "save_int32", inputs["int32"]))
    s.append(array("uint8_t", "save_uint8", inputs["uint8"]))
    s.append("    // torchaudio.save(input, encoding, bits_per_sample) data-chunk bytes.")
    for i, (_, _, _, data) in enumerate(goldens):
        s.append(array("unsigned char", f"save_golden_{i}", data))
    s.append("    struct save_golden {\n        const char *dtype;\n        const char *encoding;\n        int bits;\n"
             "        const unsigned char *data;\n        size_t size;\n    };")
    s.append("    inline constexpr save_golden save_goldens[] = {")
    for i, (dtype, enc, bits, _) in enumerate(goldens):
        s.append(f'            {{"{dtype}", "{enc}", {bits}, save_golden_{i}, sizeof save_golden_{i}}},')
    s.append("    };\n")
    s.append("    // torchaudio.save(encoding, bits_per_sample) on a WAV path -> torchaudio.info of the file (\"\": raised).")
    s.append("    struct save_combo {\n        const char *dtype;\n        const char *encoding;\n        int bits;\n"
             "        const char *ta_encoding;\n        int ta_bits;\n    };")
    s.append("    inline constexpr save_combo save_combos[] = {")
    for dtype, enc, bits, ta_enc, ta_bits in combos:
        s.append(f'            {{"{dtype}", "{enc}", {bits}, "{ta_enc}", {ta_bits}}},')
    s.append("    };")
    s.append("} // namespace tm_fixture")
    s.append("// clang-format on\n")
    with open(out, "w") as f:
        f.write("\n".join(s))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", help="keep the encoded files in this directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or tmp
        os.makedirs(work, exist_ok=True)
        rows, rejected = build(work)
        inputs, goldens, combos = save_goldens(work)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.hpp")
    emit(rows, rejected, inputs, goldens, combos, out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
