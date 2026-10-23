#!/usr/bin/env python3
"""Golden values for the vision I/O decode tests from torchvision 0.20.1 (torchvision.io.decode_image, libpng /
libjpeg-turbo), torch 2.5.1.

Run in the project venv; it rewrites golden.hpp next to this file:
    .venv/bin/python unit_test/vision/io/gen_golden.py

Small PNG / JPEG files are generated with PIL (and by hand for 16-bit PNG), baked into golden.hpp as bytes together
with torchvision's decode in each read mode. PNG decodes compare exactly; JPEG within the measured stb_image vs
libjpeg-turbo bound (develop_log/2026-10-23/progress09). uint16 outputs are stored as int32.
"""
import io
import pathlib
import struct
import zlib

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torchvision.io import ImageReadMode as M, decode_image

rng = np.random.default_rng(20261023)
entries = []


def emit(name, t):
    t = t.to(torch.int32) if t.dtype == torch.uint16 else t
    shape = ", ".join(str(s) for s in t.shape)
    vals = ", ".join(str(int(v)) for v in t.flatten().tolist())
    entries.append(f'    inline const entry {name}{{"{str(t.dtype).removeprefix("torch.")}", {{{shape}}}, {{{vals}}}}};')


def file(name, data, modes):
    """Emit the encoded bytes and torchvision's decode for each mode."""
    t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    emit(name, t)
    for m in modes:
        emit(f"{name}_{m.name.lower()}", decode_image(t, m))


def png(im, **kw):
    b = io.BytesIO()
    im.save(b, "PNG", **kw)
    return b.getvalue()


def gamma_chunk(tag, data):
    info = PngImagePlugin.PngInfo()
    info.add(tag, data)
    return info


def png_raw(arr, color_type, chunks=(), depth=16):
    """A PNG written by hand (PIL cannot write 16-bit RGB or chunks in a chosen order); chunks go before IDAT."""
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].astype(">u2" if depth == 16 else "u1").tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 0))
            + b"".join(chunk(t, d) for t, d in chunks) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


rgb = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
alpha = rng.integers(0, 256, (5, 7), dtype=np.uint8)
gray = rgb[..., 0]
file("png_l", png(Image.fromarray(gray)), [M.UNCHANGED, M.GRAY_ALPHA, M.RGB, M.RGB_ALPHA])
file("png_la", png(Image.fromarray(np.dstack([gray, alpha]), "LA")), [M.UNCHANGED, M.GRAY, M.RGB_ALPHA])
file("png_rgb", png(Image.fromarray(rgb)), [M.UNCHANGED, M.GRAY, M.GRAY_ALPHA])
file("png_rgba", png(Image.fromarray(np.dstack([rgb, alpha]))), [M.UNCHANGED, M.GRAY, M.RGB])
file("png_1bit", png(Image.fromarray(gray > 128)), [M.UNCHANGED, M.GRAY_ALPHA])  # alpha filler 1 (torchvision)
palette = Image.fromarray(rgb).quantize(64)
file("png_palette", png(palette), [M.GRAY, M.RGB])  # UNCHANGED: torchvision returns indices (deviation)
file("png_palette_trns", png(palette, transparency=5), [M.GRAY_ALPHA, M.RGB_ALPHA])
file("png_rgb_trns", png(Image.fromarray(rgb), transparency=tuple(int(v) for v in rgb[0, 0])), [M.UNCHANGED, M.RGB_ALPHA])
# gray conversion in linear light (libpng gamma tables) for a significant file gamma, plain otherwise
file("png_rgb_gama", png(Image.fromarray(rgb), pnginfo=gamma_chunk(b"gAMA", struct.pack(">I", 45455))), [M.GRAY, M.GRAY_ALPHA])
file("png_rgba_srgb", png(Image.fromarray(np.dstack([rgb, alpha])), pnginfo=gamma_chunk(b"sRGB", b"\x00")), [M.GRAY])
file("png_rgb_gama1", png(Image.fromarray(rgb), pnginfo=gamma_chunk(b"gAMA", struct.pack(">I", 100000))), [M.GRAY])
rgb16 = rng.integers(0, 65536, (5, 7, 3), dtype=np.uint16)
rgb16[0, :3] = rgb16[0, :3, :1]  # a few gray pixels
file("png16_rgb", png_raw(rgb16, 2), [M.UNCHANGED, M.GRAY, M.RGB_ALPHA])
file("png16_rgb_gama", png_raw(rgb16, 2, [(b"gAMA", struct.pack(">I", 45455))]), [M.GRAY])
file("png16_l", png_raw(rgb16[..., 0], 0), [M.GRAY_ALPHA])
# gAMA / sRGB kept as libpng keeps them (png_colorspace_set_gamma / _sRGB): the first gAMA wins, sRGB wins over gAMA in
# either order, a bad length or a chunk after PLTE is skipped, an out-of-range gAMA or an sRGB intent >= 4 invalidates
# the colour space (later gAMA / sRGB skipped); gamma within 5% of 1.0 is not significant
GAMA = lambda v: (b"gAMA", struct.pack(">I", v))
SRGB = (b"sRGB", b"\x00")
for name, chunks in (("two_gama", [GAMA(45455), GAMA(100000)]), ("srgb_gama1", [SRGB, GAMA(100000)]),
                     ("srgb_len2", [(b"sRGB", b"\x00\x00")]), ("srgb_intent7", [(b"sRGB", b"\x07")]),
                     ("gama_tiny_srgb", [GAMA(10), SRGB]), ("gama097", [GAMA(97000)])):
    file(f"png_rgb_{name}", png_raw(rgb, 2, chunks, 8), [M.GRAY])
file("png_palette_gama_after_plte", png_raw(gray % 16, 3, [(b"PLTE", rgb.reshape(-1, 3)[:16].tobytes()), GAMA(45455)], 8),
     [M.GRAY])

# JPEG: a photo-like 33 x 17 image (odd size) in each chroma subsampling, plus a grayscale JPEG
yy, xx = np.mgrid[0:33, 0:17]
photo = np.stack([128 + 100 * np.sin(xx / 3.0) * np.cos(yy / 5.0), xx * 15.0,
                  255.0 * (((xx - 8) ** 2 + (yy - 16) ** 2) < 64)], -1) + rng.normal(0, 12, (33, 17, 3))
photo = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8))
for sub, name in ((0, "444"), (1, "422"), (2, "420")):
    b = io.BytesIO()
    photo.save(b, "JPEG", quality=75, subsampling=sub)
    file(f"jpeg_{name}", b.getvalue(), [M.UNCHANGED, M.GRAY])
b = io.BytesIO()
photo.convert("L").save(b, "JPEG", quality=90)
file("jpeg_gray", b.getvalue(), [M.UNCHANGED, M.RGB])
# chroma only 2 samples wide: libjpeg-turbo replicates instead of fancy upsampling
b = io.BytesIO()
photo.crop((0, 0, 3, 16)).save(b, "JPEG", quality=95, subsampling=2)
file("jpeg_narrow", b.getvalue(), [M.UNCHANGED])
b = io.BytesIO()
photo.save(b, "JPEG", quality=75, subsampling=2, restart_marker_blocks=1)  # RSTn markers in the entropy-coded data
file("jpeg_rst", b.getvalue(), [M.UNCHANGED])

# PNM (P1-P6) for io::load_image, decoded here from the netpbm spec (netpbm.sourceforge.net/doc/pbm.html, pgm.html,
# ppm.html): whitespace is blank / TAB / CR / LF; '#' through the next CR / LF is a comment and separates tokens (as
# libnetpbm's pm_getc turns it into the newline); the raster of P4-P6 follows the single whitespace byte after the last
# header token; samples are 2 big-endian bytes when maxval > 255; P4 rows are padded to whole bytes; P1 bits need no
# separator; 1 = black. load_image's contract: float32 sample / maxval (PBM: 1 - bit). torchvision's decode_image reads
# no PNM, so the cross-check is PIL, where it reads the variant (exact for maxval 255 / 65535 gray and PBM, else within
# its 8- / 16-bit rounding).
WS = b" \t\r\n"


def pnm_reference(d):
    kind, i = d[1] - 48, 2

    def skip():
        nonlocal i
        while i < len(d) and (d[i] in WS or d[i] == 35):
            i += 1
            if d[i - 1] == 35:
                while i < len(d) and d[i] not in b"\r\n":
                    i += 1

    def uint():
        nonlocal i
        skip()
        j = i
        while i < len(d) and 48 <= d[i] <= 57:
            i += 1
        assert i > j, "integer expected"
        return int(d[j:i])

    w, h = uint(), uint()
    maxval, c = (1 if kind in (1, 4) else uint()), (3 if kind in (3, 6) else 1)
    if kind == 1:
        bits = []
        while len(bits) < w * h:
            skip()
            assert d[i] in b"01"
            bits.append(d[i] - 48)
            i += 1
        s = 1 - np.array(bits)
    elif kind in (2, 3):
        s = np.array([uint() for _ in range(w * h * c)])
    else:
        if d[i] == 35:  # a comment right after the last token ends at the raster's whitespace byte
            while d[i] not in b"\r\n":
                i += 1
        assert d[i] in WS
        i += 1
        if kind == 4:
            rb = (w + 7) // 8
            s = 1 - np.unpackbits(np.frombuffer(d, np.uint8, h * rb, i).reshape(h, rb), axis=1)[:, :w]
        else:
            s = np.frombuffer(d, ">u2" if maxval > 255 else "u1", w * h * c, i)
    s = s.astype(np.int64).reshape(h, w, c).transpose(2, 0, 1)
    assert 0 < maxval < 65536 and s.max() <= maxval
    return s, maxval


def pnm(name, data, pil=True):
    s, maxval = pnm_reference(data)
    if pil:
        im = Image.open(io.BytesIO(data))
        a = np.asarray(im).astype(np.int64).reshape(s.shape[1], s.shape[2], -1).transpose(2, 0, 1)
        if im.mode == "1":
            assert (a == s).all(), name
        else:
            top = 65535 if im.mode == "I" else 255
            assert np.abs(a / top - s / maxval).max() <= (0 if maxval == top else 0.5 / top + 1e-12), name
    emit(name, torch.frombuffer(bytearray(data), dtype=torch.uint8))
    emit(f"{name}_samples", torch.from_numpy(np.ascontiguousarray(s)))
    emit(f"{name}_maxval", torch.tensor(maxval))


g8 = rng.integers(0, 256, (5, 7), dtype=np.uint8)
g8[0, :3] = [10, 32, 35]  # raster starting with LF / blank / '#': only one whitespace byte ends the header
pnm("pnm_p5", b"P5\n# a comment line\n7 5\n255\n" + g8.tobytes())
g16 = rng.integers(0, 65536, (5, 7), dtype=np.uint16)
g16[0, :3] = [1000, 0, 65535]  # stb_image v2.30 returns the low byte of 1000 (232)
pnm("pnm_p5_16", b"P5 7 5 65535\n" + g16.astype(">u2").tobytes())
pnm("pnm_p5_1000", b"P5\t7\r\n5 #c\r1000\r" + (g16 % 1001).astype(">u2").tobytes())  # 16-bit, scaled by maxval
pnm("pnm_p5_15", b"P5 7 5 15\n" + (g8 % 16).tobytes())  # 8-bit storage, maxval 15
pnm("pnm_p6", b"P6\n7 5\n255\n" + rgb.tobytes())
pnm("pnm_p6_4095", b"P6\n7 5\n4095\n" + (rgb16 % 4096).astype(">u2").tobytes())
plain = lambda a: "\n".join(" ".join(str(int(v)) for v in row) for row in a.reshape(a.shape[0], -1)).encode()
pnm("pnm_p2", b"P2\r\n# plain\r\n7 5\r\n1000\r\n" + plain(g16[:2] % 1001) + b"  # in the raster\n\t" + plain(g16[2:] % 1001))
pnm("pnm_p3", b"P3 7 5 255 " + plain(rgb) + b"\n")
bits = rng.integers(0, 2, (5, 7))
rows = ["".join(str(v) for v in r) for r in bits]
pnm("pnm_p1", ("P1\n7 5\n" + rows[0] + "\n" + " ".join(rows[1]) + "#c\n" + "\n".join(rows[2:])).encode())
packed = np.packbits(np.pad(rng.integers(0, 2, (5, 11)), ((0, 0), (0, 5)), constant_values=1), axis=1)
pnm("pnm_p4", b"P4\n11 5\n" + packed.tobytes())  # padding bits set: ignored
# a comment between digits splits the token (libnetpbm), one after maxval ends at the raster's whitespace byte, and a
# second image after the raster is not read; PIL joins split tokens and reads the comment into the raster
pnm("pnm_comments", b"P5 7#width\n5 25#maxval\n" + (g8 % 26).tobytes() + b"P5 1 1 255\n\x00", pil=False)

out = pathlib.Path(__file__).with_name("golden.hpp")
out.write_text(
    "// Generated by gen_golden.py from torchvision 0.20.1 / torch 2.5.1 on arm64 -- do not edit.\n"
    "// clang-format off\n#pragma once\n#include <cstdint>\n#include <string>\n#include <vector>\n\n"
    "namespace golden {\n    struct entry {\n        std::string dtype; // torch dtype name; uint16 decodes are stored as int32\n"
    "        std::vector<int64_t> shape;\n        std::vector<int64_t> data;\n    };\n\n"
    + "\n".join(entries)
    + "\n} // namespace golden\n// clang-format on\n"
)
print(f"wrote {out} ({len(entries)} entries)")
