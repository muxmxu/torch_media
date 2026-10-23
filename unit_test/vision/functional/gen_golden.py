#!/usr/bin/env python3
"""Golden values for the vision functional tests from torchvision 0.20.1 (transforms.v2.functional, torch 2.5.1).

Run in the project venv; it rewrites golden.hpp next to this file:
    .venv/bin/python unit_test/vision/functional/gen_golden.py

Inputs are seeded random images baked into golden.hpp together with the outputs, so the C++ test feeds torchvision's
exact input. Float32 values are printed with 9 significant digits and float64 values with repr, both of which round
trip exactly, so the test can compare bit for bit. Generated on arm64 (torch reports CPU capability "NO AVX").
"""
import pathlib

import torch
import torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode as IM

torch.manual_seed(20261023)
entries = []


def emit(name, t):
    t = t.detach().cpu()
    if t.dtype == torch.float32:
        vals = [f"{v:.9g}" for v in t.flatten().tolist()]
    elif t.dtype == torch.float64:
        vals = [repr(v) for v in t.flatten().tolist()]
    else:
        vals = [str(int(v)) for v in t.flatten().tolist()]
    shape = ", ".join(str(s) for s in t.shape)
    dtype = str(t.dtype).removeprefix("torch.")
    entries.append(f'    inline const entry {name}{{"{dtype}", {{{shape}}}, {{{", ".join(vals)}}}}};')


def u8(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8)


f32 = lambda x: x.to(torch.float32) / 255
f64 = lambda x: x.to(torch.float64) / 255

S = u8(3, 5, 7)  # upscaling and general ops
L = u8(3, 11, 13)  # downscaling
B = u8(2, 3, 6, 10)  # batch
emit("in_s", S)
emit("in_l", L)
emit("in_b", B)

# ---- resize: every mode, uint8 / float32, up by 1.6 x 1.571 and down by 2.75 x 2.6 ----
MODES = {
    "nearest": (IM.NEAREST, True),
    "nearest_exact": (IM.NEAREST_EXACT, True),
    "bilinear": (IM.BILINEAR, True),
    "bilinear_noaa": (IM.BILINEAR, False),
    "bicubic": (IM.BICUBIC, True),
    "bicubic_noaa": (IM.BICUBIC, False),
}
for mname, (mode, aa) in MODES.items():
    for dname, conv in (("u8", lambda x: x), ("f32", f32)):
        emit(f"resize_{mname}_up_{dname}", F.resize(conv(S), [8, 11], interpolation=mode, antialias=aa))
        emit(f"resize_{mname}_down_{dname}", F.resize(conv(L), [4, 5], interpolation=mode, antialias=aa))
emit("resize_bilinear_down_f64", F.resize(f64(L), [4, 5], interpolation=IM.BILINEAR, antialias=True))
emit("resize_bicubic_down_f64", F.resize(f64(L), [4, 5], interpolation=IM.BICUBIC, antialias=True))
emit("resize_bilinear_noaa_up_f64", F.resize(f64(S), [8, 11], interpolation=IM.BILINEAR, antialias=False))
emit("resize_bicubic_noaa_up_f64", F.resize(f64(S), [8, 11], interpolation=IM.BICUBIC, antialias=False))
# other integer dtypes go through float32 and are rounded back (in range: bicubic overshoot stays inside int8)
emit("resize_bicubic_noaa_up_i8", F.resize((S >> 1).to(torch.int8) - 64, [8, 11], interpolation=IM.BICUBIC, antialias=False))
emit("resize_bicubic_noaa_up_i32", F.resize(S.to(torch.int32) * 1000 - 128000, [8, 11], interpolation=IM.BICUBIC, antialias=False))
# batch dims, mixed up (h) / down (w) scaling
emit("resize_batch_bilinear_u8", F.resize(B, [4, 7]))
emit("resize_batch_bicubic_f32", F.resize(f32(B), [9, 4], interpolation=IM.BICUBIC))
# size = [shorter edge], max_size cap and size None on a tall image [3, 11, 5]
T = L[..., :5]
emit("resize_short_f32", F.resize(f32(L), [6]))  # 11 x 13 -> 6 x 7
emit("resize_max_size_f32", F.resize(f32(T), [4], max_size=6))  # 11 x 5 -> 6 x 3
emit("resize_size_none_f32", F.resize(f32(T), None, max_size=6))  # 11 x 5 -> 6 x 2
# a non-CPU device takes the float32 path for uint8 bilinear (MPS has no bicubic / antialias kernels in 2.5.1)
HAS_MPS = torch.backends.mps.is_available()
if HAS_MPS:
    emit("resize_mps_bilinear_noaa_u8", F.resize(S.to("mps"), [8, 11], antialias=False))
    emit("resize_mps_bilinear_noaa_f32", F.resize(f32(S).to("mps"), [8, 11], antialias=False))

# ---- crop / center_crop / resized_crop / five_crop / ten_crop ----
emit("crop_in_f32", F.crop(f32(S), 1, 2, 3, 4))
emit("crop_oob_tl_u8", F.crop(S, -2, -1, 5, 6))
emit("crop_oob_br_f32", F.crop(f32(S), 3, 4, 4, 6))
emit("crop_2d_u8", F.crop(S[0], 1, 1, 2, 3))
emit("center_crop_2x2_u8", F.center_crop(S, [2, 2]))  # anchors round(1.5)=2, round(2.5)=2
emit("center_crop_3x4_u8", F.center_crop(S, [3, 4]))
emit("center_crop_one_f32", F.center_crop(f32(L), [4]))
emit("center_crop_pad_both_u8", F.center_crop(S, [8, 10]))
emit("center_crop_pad_h_f32", F.center_crop(f32(S), [6, 7]))  # padded to exactly the crop
emit("center_crop_pad_h_crop_w_u8", F.center_crop(S, [8, 3]))
emit("resized_crop_f32", F.resized_crop(f32(L), 1, 2, 7, 9, [5, 4]))
emit("resized_crop_bicubic_u8", F.resized_crop(L, 2, 1, 8, 10, [13, 6], interpolation=IM.BICUBIC, antialias=False))
emit("five_crop_u8", torch.stack(F.five_crop(S, [3, 4])))
emit("ten_crop_u8", torch.stack(F.ten_crop(S, [3, 4])))
emit("ten_crop_vflip_f32", torch.stack(F.ten_crop(f32(S), [4], vertical_flip=True)))

# ---- pad: modes, fills, dtypes; padding [left, top, right, bottom] ----
P4 = [1, 2, 3, 4]
emit("pad_constant_u8", F.pad(S, P4, fill=7))
emit("pad_constant_f32", F.pad(f32(S), P4, fill=0.5))
emit("pad_one_f32", F.pad(f32(S), [2]))
emit("pad_two_u8", F.pad(S, [1, 3], fill=[200]))
emit("pad_vector_u8", F.pad(S, P4, fill=[10.0, 20.0, 30.0]))
emit("pad_vector_f64", F.pad(f64(S), P4, fill=[0.1, 0.2, 0.3]))  # fill goes through float32
emit("pad_vector_neg_f32", F.pad(f32(S), [-1, 2, 1, -2], fill=[0.25, 0.5, 0.75]))
emit("pad_edge_u8", F.pad(S, P4, padding_mode="edge"))
emit("pad_edge_f32", F.pad(f32(S), P4, padding_mode="edge"))
emit("pad_reflect_u8", F.pad(S, P4, padding_mode="reflect"))
emit("pad_reflect_f32", F.pad(f32(S), P4, padding_mode="reflect"))
emit("pad_symmetric_u8", F.pad(S, P4, padding_mode="symmetric"))
emit("pad_symmetric_neg_f32", F.pad(f32(S), [-1, 2, 1, -2], padding_mode="symmetric"))
emit("pad_constant_neg_u8", F.pad(S, [-1, 2, 1, -2]))
emit("pad_batch_reflect_f32", F.pad(f32(B), [2, 1], padding_mode="reflect"))

# ---- dtype conversion ----
emit("to_dtype_u8_f32", F.to_dtype(S, torch.float32, scale=True))
emit("to_dtype_u8_f64", F.to_dtype(S, torch.float64, scale=True))
emit("to_dtype_f32_u8", F.to_dtype(f32(L), torch.uint8, scale=True))
emit("to_dtype_f64_u8", F.to_dtype(f64(L), torch.uint8, scale=True))
emit("to_dtype_f32_i16", F.to_dtype(f32(S), torch.int16, scale=True))
emit("to_dtype_f32_u16", F.to_dtype(f32(S), torch.uint16, scale=True).to(torch.int32))
emit("to_dtype_u8_i16", F.to_dtype(S, torch.int16, scale=True))
emit("to_dtype_i16_u8", F.to_dtype(F.to_dtype(S, torch.int16, scale=True) + 77, torch.uint8, scale=True))
emit("to_dtype_u8_u16", F.to_dtype(S, torch.uint16, scale=True).to(torch.int32))
emit("to_dtype_u16_u8", F.to_dtype((S.to(torch.int32) * 257).to(torch.uint16), torch.uint8, scale=True))
emit("to_dtype_noscale_f32", F.to_dtype(f32(S) * 250, torch.uint8))
# half / bfloat16 outputs, stored as float32 (exact); the 1 / max scale must not be rounded to the output dtype
emit("to_dtype_u8_f16", F.to_dtype(S, torch.float16, scale=True).to(torch.float32))
emit("to_dtype_u8_bf16", F.to_dtype(S, torch.bfloat16, scale=True).to(torch.float32))
emit("to_dtype_i32_f16", F.to_dtype(S.to(torch.int32) * 200, torch.float16, scale=True).to(torch.float32))

# ---- color ----
emit("gray_u8", F.rgb_to_grayscale(S))
emit("gray3_f32", F.rgb_to_grayscale(f32(S), num_output_channels=3))
emit("gray_f64", F.rgb_to_grayscale(f64(B)))
emit("brightness_u8", F.adjust_brightness(S, 1.7))
emit("brightness_f32", F.adjust_brightness(f32(S), 1.7))
emit("brightness_f64", F.adjust_brightness(f64(S), 0.6))
emit("invert_u8", F.invert(S))
emit("invert_f32", F.invert(f32(S)))
emit("invert_i16", F.invert(S.to(torch.int16) * 100 - 9000))
emit("invert_i8", F.invert(S.to(torch.int8)))  # wraps 128..255 to negatives first
emit("invert_i32", F.invert(S.to(torch.int32) * 1000003 - 99999999))
# int64 inputs near the extremes, so x ^ INT64_MAX is small enough to be exact in the double-typed golden
I64 = S.to(torch.int64) * 1000
emit("invert_i64", F.invert(torch.where(S % 2 == 0, torch.iinfo(torch.int64).max - I64, torch.iinfo(torch.int64).min + I64)))
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
emit("normalize_f32", F.normalize(f32(S), *IMAGENET))
emit("normalize_f64", F.normalize(f64(B), *IMAGENET))
emit("normalize_scalar_f32", F.normalize(f32(S), [0.5], [0.25]))

# ---- color ops (progress09): uint8 / float32 / float64, 1- and 3-channel, batch and 5-D ----
G1 = S[:1]  # 1-channel [1, 5, 7]
emit("contrast_u8", F.adjust_contrast(S, 0.6))
emit("contrast_f32", F.adjust_contrast(f32(S), 1.7))
emit("contrast_f64", F.adjust_contrast(f64(B), 0.4))
emit("contrast_gray_u8", F.adjust_contrast(G1, 1.5))
emit("contrast_gray_f32", F.adjust_contrast(f32(G1), 0.5))
emit("saturation_u8", F.adjust_saturation(S, 0.3))
emit("saturation_f32", F.adjust_saturation(f32(L), 1.8))
emit("saturation_f64", F.adjust_saturation(f64(B), 0.0))
emit("sharpness_u8", F.adjust_sharpness(L, 2.0))
emit("sharpness_f32", F.adjust_sharpness(f32(S), 0.4))
emit("sharpness_f64", F.adjust_sharpness(f64(L), 1.5))
emit("sharpness_5d_f32", F.adjust_sharpness(f32(B).reshape(2, 1, 3, 6, 10), 3.0))
emit("hue_u8", F.adjust_hue(S, 0.25))
emit("hue_f32", F.adjust_hue(f32(L), -0.4))
emit("hue_f64", F.adjust_hue(f64(B), 0.1))
emit("hue_batch_u8", F.adjust_hue(B, -0.5))
emit("gamma_u8", F.adjust_gamma(S, 0.5))
emit("gamma_gain_u8", F.adjust_gamma(S, 2.2, 1.3))
emit("gamma_f32", F.adjust_gamma(f32(L), 1.8))
emit("gamma_gain_f64", F.adjust_gamma(f64(B), 0.7, 0.8))
# half / bfloat16 with a gain: Python's mul_(float) keeps the float opmath (libtorch's mul_(Scalar) would not)
emit("gamma_gain_f16", F.adjust_gamma(f32(S).half(), 2.2, 0.7).float())
emit("gamma_gain_bf16", F.adjust_gamma(f32(L).bfloat16(), 1.0, 1.2).float())
emit("posterize_u8", F.posterize(S, 3))
emit("posterize_f32", F.posterize(f32(S), 2))
emit("posterize_white_f32", F.posterize(torch.cat([f32(S), torch.ones(3, 1, 7)], -2), 3))  # 1.0 hits the clamp
emit("posterize_i16", F.posterize(S.to(torch.int16) * 100, 5))
emit("solarize_u8", F.solarize(S, 128.0))
emit("solarize_f32", F.solarize(f32(L), 0.4))
LOW = S // 3 + 60  # [60, 145]: stretched by autocontrast
CONST = torch.cat([S[:2], torch.full_like(S[:1], 99)])  # a constant channel: left as is
emit("autocontrast_u8", F.autocontrast(LOW))
emit("autocontrast_f32", F.autocontrast(f32(L) * 0.5 + 0.2))
emit("autocontrast_const_u8", F.autocontrast(CONST))
emit("equalize_u8", F.equalize(S))
emit("equalize_batch_u8", F.equalize(B))
emit("equalize_f32", F.equalize(f32(L)))
emit("equalize_const_u8", F.equalize(CONST))  # step == 0 for the constant channel
emit("blur_u8", F.gaussian_blur(L, 3))
emit("blur_f32", F.gaussian_blur(f32(L), [5, 3], [1.2, 0.7]))
emit("blur_f64", F.gaussian_blur(f64(B), 3, 0.9))
emit("blur_5d_f32", F.gaussian_blur(f32(B).reshape(2, 1, 3, 6, 10), [3, 5]))
emit("blur_default_sigma_f64", F.gaussian_blur(f64(L), [5, 9]))  # k * 0.15 + 0.35 in Python order (no FMA)
emit("erase_u8", F.erase(S, 1, 2, 3, 4, torch.tensor([7, 8, 9], dtype=torch.uint8).view(3, 1, 1)))
emit("erase_f32", F.erase(f32(B), -3, -4, 2, 9, torch.tensor(0.25)))

# ---- geometry (progress09): rotate / affine / perspective / elastic ----
NE, BI, BC = IM.NEAREST, IM.BILINEAR, IM.BICUBIC
emit("rotate_30_u8", F.rotate(S, 30.0))
emit("rotate_30_fill_f32", F.rotate(f32(L), 30.0, BI, fill=[0.1, 0.5, 0.9]))
emit("rotate_expand_u8", F.rotate(L, 37.5, BI, expand=True, fill=[200.0]))
emit("rotate_expand_nearest_f32", F.rotate(f32(S), -110.0, NE, expand=True, fill=[0.0]))
emit("rotate_center_f64", F.rotate(f64(S), -20.0, BI, center=[1.0, 2.0]))
emit("rotate_batch_f32", F.rotate(f32(B), 45.0, NE, fill=[0.3]))
emit("rotate_90_rect_u8", F.rotate(S, 90.0))  # non-square, no expand: grid path
emit("affine_u8", F.affine(S, 25.0, [1.0, -2.0], 0.9, [10.0, -5.0]))
emit("affine_fill_f32", F.affine(f32(L), -40.0, [0.5, 1.5], 1.2, [15.0], BI, [0.2, 0.4, 0.6], [3.0, 4.0]))
emit("affine_f64", F.affine(f64(B), 10.0, [0.0, 0.0], 1.1, [0.0, 20.0], BI, [0.0]))
emit("affine_u8_bilinear", F.affine(L, 123.0, [-2.0, 0.0], 0.75, [-8.0, 3.0], BI, [255.0, 0.0, 128.0]))
SP = [[0, 0], [12, 0], [12, 10], [0, 10]]
emit("perspective_u8", F.perspective(L, SP, [[1, 0], [11, 2], [12, 9], [0, 10]]))
emit("perspective_float_f32", F.perspective(f32(L), SP, [[0.5, 0.25], [11.5, 0.0], [12.0, 8.25], [1.25, 10.0]], BI, [0.7]))
# non-dyadic fractional points: float32 rows of the system; float32 coefficients, visible on a float64 image
emit("perspective_frac_f32", F.perspective(f32(L), SP, [[0.1, 0.3], [11.7, 0.2], [12, 9.9], [0.4, 10]], BI))
emit("perspective_frac_f64", F.perspective(f64(L), [[0.1, 0.2], [12.3, 0], [12, 10], [0, 10]], [[1, 0], [11, 2], [12, 9], [0, 10]], BI))
COEFFS = [1.1, 0.05, -0.5, -0.02, 0.95, 0.7, 0.001, -0.002]
emit("perspective_coeffs_f64", F.perspective(f64(B), None, None, NE, None, COEFFS))
emit("perspective_nearest_fill_u8", F.perspective(S, [[0, 0], [6, 0], [6, 4], [0, 4]], [[1, 1], [5, 0], [6, 4], [0, 3]], NE, [9.0, 99.0, 199.0]))
# a zero denominator on x + y = 7 (inf grid entries, NaN at (6, 1)): nearest takes the fill there (bilinear: SIGBUS)
ZERO_DEN = [1.0, 0.0, 0.0, 0.0, 1.0, -1.5, -0.125, -0.125]
emit("perspective_zero_den_nearest_f32", F.perspective(f32(S), None, None, NE, [0.25, 0.5, 0.75], ZERO_DEN))
DS = (torch.rand(1, 5, 7, 2) - 0.5) * 0.4
DL = (torch.rand(1, 11, 13, 2) - 0.5) * 0.4
emit("disp_s", DS)
emit("disp_l", DL)
emit("elastic_u8", F.elastic(S, DS))
emit("elastic_bicubic_f32", F.elastic(f32(L), DL, BC))
emit("elastic_nearest_fill_f64", F.elastic(f64(S), DS, NE, [0.5]))
emit("elastic_fill_u8", F.elastic(L, DL, BI, [10.0, 20.0, 30.0]))
emit("elastic_half", F.elastic(f32(S).half(), DS).float())  # CPU float16: warped in float32, cast back
emit("elastic_bicubic_fill_f32", F.elastic(f32(S), DS, BC, [0.3]))  # bicubic fill: the blend, not nearest's where
if HAS_MPS:  # torchvision on MPS (float32); its MPS adjust_hue is wrong (non_blocking copy race), see progress09
    emit("blur_mps_f32", F.gaussian_blur(f32(L).to("mps"), [5, 3], [1.2, 0.7]))
    emit("sharpness_mps_f32", F.adjust_sharpness(f32(S).to("mps"), 0.4))
    emit("rotate_mps_f32", F.rotate(f32(L).to("mps"), 30.0, BI, fill=[0.1, 0.5, 0.9]))
    emit("perspective_mps_f32", F.perspective(f32(L).to("mps"), SP, [[1, 0], [11, 2], [12, 9], [0, 10]], BI))
    emit("equalize_mps_u8", F.equalize(B.to("mps")))

# ---- permute_channels / grayscale_to_rgb / gaussian_noise (progress10) ----
emit("pad_symmetric_full_u8", F.pad(S, [7, 5], padding_mode="symmetric"))  # the widest valid symmetric pad
emit("permute_channels_u8", F.permute_channels(S, [2, 0, 1]))
emit("permute_channels_batch_f32", F.permute_channels(f32(B), [1, 2, 0]))
emit("permute_channels_neg_u8", F.permute_channels(S, [-1, 0, 1]))  # Python indexing: -1 is the last channel
torch.manual_seed(77)
emit("gaussian_noise_f32", F.gaussian_noise(f32(S)))
torch.manual_seed(78)
emit("gaussian_noise_f64", F.gaussian_noise(f64(S), mean=0.1, sigma=0.3, clip=False))
torch.manual_seed(79)
emit("gaussian_noise_half", F.gaussian_noise(f32(S).half(), mean=0.05, sigma=0.2).float())

out = pathlib.Path(__file__).with_name("golden.hpp")
out.write_text(
    "// Generated by gen_golden.py from torchvision 0.20.1 / torch 2.5.1 on arm64 -- do not edit.\n"
    "// clang-format off\n#pragma once\n#include <cstdint>\n#include <string>\n#include <vector>\n\n"
    f"#define GOLDEN_HAS_MPS {int(HAS_MPS)}\n\n"
    "namespace golden {\n    struct entry {\n        std::string dtype; // torch dtype name, e.g. uint8\n"
    "        std::vector<int64_t> shape;\n        std::vector<double> data;\n    };\n\n"
    + "\n".join(entries)
    + "\n} // namespace golden\n// clang-format on\n"
)
print(f"wrote {out} ({len(entries)} entries)")
