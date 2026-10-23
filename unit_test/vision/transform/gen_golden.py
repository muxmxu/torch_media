#!/usr/bin/env python3
"""Golden values for the vision transform tests from torchvision 0.20.1 (transforms.v2, torch 2.5.1).

Run in the project venv; it rewrites golden.hpp next to this file:
    .venv/bin/python unit_test/vision/transform/gen_golden.py

The ImageNet recipe runs on a synthetic 375 x 500 photo-like image built from integer arithmetic only, so main.cpp
rebuilds it bit for bit (checked through the input checksums). Its 3 x 224 x 224 output is pinned by a 14 x 14 grid of
pixels per channel plus the float64 sum and sum of squares over every pixel. Generated on arm64.

Random classes (progress10): each seeded case runs N calls after torch.manual_seed(seed) and records every output
(dtype, shape, values), whether it is the input object itself, and the next torch.rand(1) (pins the draw count).
"""
import pathlib

import torch
import torchvision.transforms.v2 as T

lines = []


def lit(t):
    """An `entry` literal: dtype name, shape, values (float32 with 9 digits, float64 with repr: exact round trips)."""
    t = t.detach().cpu()
    if t.dtype == torch.float32:
        vals = [f"{v:.9g}" for v in t.flatten().tolist()]
    elif t.dtype == torch.float64:
        vals = [repr(v) for v in t.flatten().tolist()]
    else:
        vals = [str(int(v)) for v in t.flatten().tolist()]
    shape = ", ".join(str(v) for v in t.shape)
    return f'{{"{str(t.dtype).removeprefix("torch.")}", {{{shape}}}, {{{", ".join(vals)}}}}}'


def emit(name, t):
    lines.append(f"    inline const entry {name}{lit(t)};")


def case(name, t, x, seed, calls=3):
    """Seeded calls of transform t on x (x is reused, so an in-place transform keeps mutating it, as in C++)."""
    torch.manual_seed(seed)
    outs, same = [], []
    for _ in range(calls):
        o = t(x)
        same.append(int(o is x))
        outs.append(o.clone())
    nxt = float(torch.rand(1))
    lines.append(f"    inline const seeded_case {name}{{{seed}, {{{', '.join(lit(o) for o in outs)}}}, "
                 f"{{{', '.join(map(str, same))}}}, {nxt!r}}};")


def scalar(name, v):
    lines.append(f"    inline constexpr {'double' if isinstance(v, float) else 'int64_t'} {name} = {v!r};")


def values(name, t):
    vals = [f"{v:.9g}" for v in t.flatten().tolist()] if t.dtype == torch.float32 else [str(v) for v in t.tolist()]
    lines.append(f"    inline const std::vector<double> {name}{{{', '.join(vals)}}};")


def synthetic_image(h=375, w=500):
    """Gradients + a disc + stripes + hashed noise, uint8 [3, h, w]; integer ops only (mirrored in main.cpp)."""
    y = torch.arange(h).view(h, 1)
    x = torch.arange(w).view(1, w)
    channels = []
    for c in range(3):
        base = torch.div(x * (200 + 20 * c), w, rounding_mode="floor") + torch.div(
            y * (150 - 30 * c), h, rounding_mode="floor"
        )
        disc = (x - 330) ** 2 + (y - 150) ** 2 < 80 * 80
        base = torch.where(disc, 220 - 70 * c, base)
        stripes = (y >= 260) & (y < 330) & (torch.div(x, 7, rounding_mode="floor").remainder(2) == 1)
        base = torch.where(stripes, 30 + 90 * c, base)
        noise = ((x * 7919 + y * 104729 + c * 15485863) * 2654435761).remainder(1000003).remainder(17) - 8
        channels.append((base + noise).clamp(0, 255))
    return torch.stack(channels).to(torch.uint8)


img = synthetic_image()
flat = img.flatten().to(torch.int64)
scalar("image_sum", int(flat.sum()))
scalar("image_weighted_sum", int((flat * (torch.arange(flat.numel()) % 251 + 1)).sum()))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
recipes = {
    # the usual order: resize the uint8 image, then convert and normalize
    "recipe_u8": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToDtype(torch.float32, scale=True),
                            T.Normalize(MEAN, STD)]),
    # a float [0, 1] image (e.g. from io::load_image)
    "recipe_f32": T.Compose([T.ToDtype(torch.float32, scale=True), T.Resize(256), T.CenterCrop(224),
                             T.Normalize(MEAN, STD)]),
}
for name, recipe in recipes.items():
    out = recipe(img)
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32
    values(f"{name}_grid", out[:, ::16, ::16])
    scalar(f"{name}_sum", float(out.double().sum()))
    scalar(f"{name}_sum_sq", float(out.double().square().sum()))

# random flips: one torch.rand(1) per call on the global CPU generator
x = torch.zeros(1, 1, 2)
for name, cls, p, seed in (("hflip", T.RandomHorizontalFlip, 0.5, 7), ("vflip", T.RandomVerticalFlip, 0.3, 11)):
    torch.manual_seed(seed)
    t = cls(p)
    flips = [int(t(x) is not x) for _ in range(12)]
    values(f"{name}_flips", torch.tensor(flips))
    scalar(f"{name}_next_draw", float(torch.rand(1)))

# GaussianBlur draws sigma = torch.empty(1).uniform_(min, max) on every call (progress09)
blur_in = (torch.arange(3 * 9 * 11) % 37).float().reshape(3, 9, 11) / 36
torch.manual_seed(5)
blur = T.GaussianBlur(5, (0.5, 2.0))
for i in range(3):
    values(f"blur_out{i}", blur(blur_in))
torch.manual_seed(6)
fixed = T.GaussianBlur([3, 5], 0.8)
values("blur_fixed_out", fixed(blur_in))
scalar("blur_fixed_next_draw", float(torch.rand(1)))

# p-gated colour transforms (_RandomApplyTransform): one torch.rand(1) per call
x8 = (torch.arange(3 * 4 * 5) * 37 % 256).to(torch.uint8).reshape(3, 4, 5)
for name, t, seed in (("invert", T.RandomInvert(), 21), ("autocontrast", T.RandomAutocontrast(0.7), 22),
                      ("equalize", T.RandomEqualize(0.4), 23), ("grayscale", T.RandomGrayscale(), 24),
                      ("posterize", T.RandomPosterize(3, 0.6), 25), ("solarize", T.RandomSolarize(100.0, 0.5), 26),
                      ("sharpness", T.RandomAdjustSharpness(2.0, 0.3), 27)):
    torch.manual_seed(seed)
    values(f"{name}_applied", torch.tensor([int(t(x8) is not x8) for _ in range(12)]))
    scalar(f"{name}_next_draw", float(torch.rand(1)))

# ======== random transform classes (progress10): seeded cases ========
from torchvision.transforms import InterpolationMode as IM  # noqa: E402

torch.manual_seed(20261024)
S = torch.randint(0, 256, (3, 9, 11), dtype=torch.uint8)
B = torch.randint(0, 256, (2, 3, 6, 10), dtype=torch.uint8)
TI = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
E = torch.randint(0, 256, (1, 24, 26), dtype=torch.uint8)
for name, t in (("in_s", S), ("in_b", B), ("in_t", TI), ("in_e", E)):
    emit(name, t)
f32 = lambda x: x.to(torch.float32) / 255  # noqa: E731
f64 = lambda x: x.to(torch.float64) / 255  # noqa: E731

# RandomCrop: one randint per side longer than the crop; pad first (padding, then pad_if_needed)
case("crop_u8", T.RandomCrop([5, 7]), S, 1)
case("crop_pad_f32", T.RandomCrop(6, padding=2, fill=0.5), f32(S), 2)
case("crop_pad2_reflect_u8", T.RandomCrop([7, 9], padding=[1, 3], padding_mode="reflect"), B, 3)
case("crop_pad4_symmetric_f32", T.RandomCrop([8, 12], padding=[1, 2, 3, 4], padding_mode="symmetric"), f32(B), 4)
case("crop_pad_if_needed_u8", T.RandomCrop([12, 14], pad_if_needed=True, fill=[10, 20, 30]), S, 5)
case("crop_pad_if_needed_edge_f32", T.RandomCrop([12, 5], pad_if_needed=True, padding_mode="edge"), f32(S), 6)
case("crop_full_u8", T.RandomCrop([9, 11]), S, 7, calls=2)  # nothing to do: the input itself, no draw
case("crop_2d_u8", T.RandomCrop([3, 4]), S[0], 58)  # a 2-D image is one channel
case("crop_rows_u8", T.RandomCrop([9, 5]), S, 59)  # full height: only the column draw
# RandomResizedCrop: 10 attempts, then the fallbacks
case("rrc_u8", T.RandomResizedCrop([5, 6]), S, 8, calls=4)
case("rrc_bicubic_f32", T.RandomResizedCrop(7, interpolation=IM.BICUBIC, antialias=False), f32(S), 9)
case("rrc_batch_u8", T.RandomResizedCrop([4, 5], scale=(0.3, 0.6), ratio=(0.5, 2.0)), B, 10)
case("rrc_fallback_wide_f32", T.RandomResizedCrop([5, 6], scale=(0.9, 1.0), ratio=(0.2, 0.25)), f32(S), 11, calls=1)
case("rrc_fallback_tall_f32", T.RandomResizedCrop([5, 6], scale=(0.9, 1.0), ratio=(5.0, 6.0)), f32(S), 12, calls=1)
case("rrc_fallback_whole_u8", T.RandomResizedCrop([5, 6], scale=(2.0, 3.0)), S, 13, calls=1)
# RandomResize / RandomShortestSize / ScaleJitter
case("rresize_u8", T.RandomResize(4, 9), S, 14)
case("rresize_nearest_f32", T.RandomResize(3, 8, interpolation=IM.NEAREST, antialias=False), f32(B), 15)
case("shortest_u8", T.RandomShortestSize([4, 6, 8], max_size=9), S, 16)
case("shortest_f32", T.RandomShortestSize([4, 7]), f32(B), 17)
case("jitter_scale_u8", T.ScaleJitter((8, 10), scale_range=(0.5, 1.5)), S, 18)
case("jitter_scale_bicubic_f32", T.ScaleJitter((12, 10), (0.5, 1.2), interpolation=IM.BICUBIC), f32(S), 19)
case("jitter_scale_float32_u8", T.ScaleJitter((13, 12), (1.0, 1.0)), S, 62, calls=1)  # 11 * f32(13 / 11): 12, not 13
case("jitter_scale_batch_f32", T.ScaleJitter((8, 10), scale_range=(0.5, 1.5)), f32(B), 64)
# ColorJitter: randperm(4), then one uniform per configured factor
case("cj_u8", T.ColorJitter(0.4, 0.4, 0.4), S, 20, calls=4)
case("cj_hue_f32", T.ColorJitter(0.4, 0.4, 0.4, 0.1), f32(S), 21)
case("cj_batch_u8", T.ColorJitter(brightness=(0.5, 1.5), hue=(-0.2, 0.3)), B, 22)
case("cj_none_u8", T.ColorJitter(), S, 23, calls=2)  # still draws randperm(4)
case("cj_identity_ranges_f32", T.ColorJitter(brightness=0, contrast=(1, 1), saturation=0.5, hue=0), f32(S), 24)
case("cj_f64", T.ColorJitter(0.3, (0.2, 0.9), (0.0, 2.0), (-0.5, 0.5)), f64(S), 25)
case("cj_clip_u8", T.ColorJitter(brightness=1.5, contrast=1.2), S, 60)  # 1 - x < 0: the range starts at 0
# RandomPhotometricDistort: 4 gated factors, the contrast coin, the gated permutation
case("pd_u8", T.RandomPhotometricDistort(), S, 26, calls=6)
case("pd_p1_f32", T.RandomPhotometricDistort(p=1.0), f32(S), 27, calls=4)
case("pd_p0_u8", T.RandomPhotometricDistort(p=0.0), S, 28, calls=2)
case("pd_batch_f32", T.RandomPhotometricDistort(p=1.0), f32(B), 65, calls=2)
case("perm_u8", T.RandomChannelPermutation(), S, 29, calls=4)
case("perm_batch_f32", T.RandomChannelPermutation(), f32(B), 30)
# RandomRotation / RandomAffine / RandomPerspective (class default fill 0 = the masked fill path)
case("rot_u8", T.RandomRotation(30), S, 31)
case("rot_expand_f32", T.RandomRotation((-10, 40), IM.BILINEAR, expand=True, fill=[0.2, 0.4, 0.6]), f32(S), 32)
case("rot_center_none_f32", T.RandomRotation(25, IM.BILINEAR, center=[3, 4], fill=None), f32(B), 33)
case("aff_u8", T.RandomAffine(20, translate=(0.2, 0.3), scale=(0.8, 1.2), shear=10), S, 34)
case("aff_shear4_f32", T.RandomAffine((-45, 10), shear=(-5, 5, -20, 3), interpolation=IM.BILINEAR, fill=0.25),
     f32(S), 35)
case("aff_center_none_f32", T.RandomAffine(15, translate=(0.1, 0.1), interpolation=IM.BILINEAR, fill=None,
                                           center=[2.5, 3]), f32(B), 36)
case("aff_zero_u8", T.RandomAffine(0), S, 37, calls=1)
case("rot_zero_u8", T.RandomRotation(0), S, 70, calls=1)  # a zero range still draws
# float64 keeps theta in float64: libm sin / cos, not a merged sincos (1 ulp apart at these seeds' angles)
case("rot_center_f64", T.RandomRotation((-60, 150), IM.BILINEAR, center=[2, 6], fill=None), f64(S), 418, calls=2)
case("aff_shear_f64", T.RandomAffine(30, translate=(0.2, 0.1), shear=(-10, 15, 5, 20), interpolation=IM.BILINEAR,
                                     fill=0.5), f64(S), 39, calls=2)
case("persp_u8", T.RandomPerspective(), S, 38, calls=4)
case("persp_nearest_fill_f32", T.RandomPerspective(0.8, p=1.0, interpolation=IM.NEAREST, fill=[0.5, 0.1, 0.9]),
     f32(S), 39)
case("persp_none_batch_f32", T.RandomPerspective(0.3, p=1.0, fill=None), f32(B), 40)
case("persp_odd_f32", T.RandomPerspective(0.6, p=1.0), f32(S[:1, :7, :6]), 71)  # 7 // 2 = 3: bound 2, not 3
# ElasticTransform: dx then dy, each rand + gaussian_blur (k = int(8 sigma + 1) made odd) + alpha / size
case("elastic_default_f32", T.ElasticTransform(), f32(E), 41, calls=2)
case("elastic_nearest_u8", T.ElasticTransform((20.0, 30.0), (0.625, 0.5), IM.NEAREST), S, 42)  # k 6 -> 7, k 5
case("elastic_nosigma_bicubic_f32", T.ElasticTransform(10.0, 0.0, IM.BICUBIC, fill=None), f32(S), 43)
case("elastic_even_kernel_f32", T.ElasticTransform((20.0, 30.0), (0.625, 0.5)), f32(S), 63, calls=1)  # k 6 -> 7
case("elastic_batch_f32", T.ElasticTransform(20.0, 0.5), f32(B), 66, calls=2)
# RandomErasing: p gate, 10 attempts (h uses *, w uses / the ratio), then the value tensor and the position
case("erase_u8", T.RandomErasing(), S, 44, calls=6)
case("erase_random_f32", T.RandomErasing(p=1.0, value="random"), f32(S), 45)
case("erase_channel_f32", T.RandomErasing(p=1.0, value=[0.1, 0.2, 0.3]), f32(B), 46)
case("erase_inplace_u8", T.RandomErasing(p=1.0, value=7, inplace=True), S.clone(), 47)
case("erase_fallback_f32", T.RandomErasing(p=1.0, scale=(0.99, 1.0), ratio=(1.0, 1.0)), f32(S), 48, calls=2)
case("erase_exact_fit_f32", T.RandomErasing(p=1.0, scale=(0.99, 1.0), ratio=(1.0, 1.0)), f32(S[..., :9]), 61,
     calls=2)  # a 9 x 9 rectangle does not fit strictly inside 9 x 9
# RandomZoomOut / GaussianNoise
case("zoom_u8", T.RandomZoomOut(), TI, 49, calls=4)
case("zoom_fill_f32", T.RandomZoomOut(fill=[0.1, 0.5, 0.9], side_range=(1.5, 2.0), p=1.0), f32(TI), 50)
case("zoom_p0_u8", T.RandomZoomOut(p=0.0), S, 51, calls=2)
case("zoom_batch_u8", T.RandomZoomOut(side_range=(1.0, 1.5), p=1.0), B, 67, calls=2)
case("noise_f32", T.GaussianNoise(), f32(S), 52, calls=2)
case("noise_noclip_f64", T.GaussianNoise(0.1, 0.3, clip=False), f64(S), 53, calls=2)
case("noise_batch_f32", T.GaussianNoise(sigma=0.2), f32(B), 68, calls=2)
# containers
case("apply_u8", T.RandomApply([T.ColorJitter(0.3, 0.3), T.RandomRotation(20)], p=0.6), S, 54, calls=6)
case("apply_batch_f32", T.RandomApply([T.ColorJitter(0.3, 0.3), T.RandomRotation(20)], p=0.6), f32(B), 69,
     calls=4)
case("choice_u8", T.RandomChoice([T.RandomHorizontalFlip(1.0), T.RandomVerticalFlip(1.0), T.RandomRotation(20)],
                                 p=[1, 2, 3]), S, 55, calls=6)
case("choice_equal_u8", T.RandomChoice([T.RandomHorizontalFlip(1.0), T.RandomVerticalFlip(1.0)]), S, 56, calls=6)
case("order_u8", T.RandomOrder([T.ColorJitter(0.3), T.RandomRotation(20), T.Grayscale(3)]), S, 57, calls=4)
case("order_batch_f32", T.RandomOrder([T.ColorJitter(0.3), T.RandomRotation(20), T.Grayscale(3)]), f32(B), 72)
# deterministic: LinearTransformation on a 12-element image (integer-built matrix and mean, mirrored in main.cpp)
LT_M = ((torch.arange(144) * 7 % 11) - 5).to(torch.float32).reshape(12, 12) / 8
LT_MEAN = (torch.arange(12) % 5).to(torch.float32) / 10
lt = T.LinearTransformation(LT_M, LT_MEAN)
emit("lt_f32", lt(f32(S[:, :2, :2])))
emit("lt_batch_u8", lt(B[..., :2, :2]))
emit("lt_f64", lt(f64(S[:, :2, :2])))  # the float32 matrix is cast to the input dtype
# the ImageNet training recipe (torchvision references/classification presets, ColorJitter 0.4)
train = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4),
                   T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    out = train(img)
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32
    values(f"train{seed}_grid", out[:, ::16, ::16])
    scalar(f"train{seed}_sum", float(out.double().sum()))
    scalar(f"train{seed}_sum_sq", float(out.double().square().sum()))
    scalar(f"train{seed}_next_draw", float(torch.rand(1)))

out = pathlib.Path(__file__).with_name("golden.hpp")
out.write_text(
    "// Generated by gen_golden.py from torchvision 0.20.1 / torch 2.5.1 on arm64 -- do not edit.\n"
    "// clang-format off\n#pragma once\n#include <cstdint>\n#include <string>\n#include <vector>\n\nnamespace golden {\n"
    "    struct entry {\n        std::string dtype; // torch dtype name, e.g. uint8\n        std::vector<int64_t> shape;\n"
    "        std::vector<double> data;\n    };\n    struct seeded_case {\n        uint64_t seed;\n"
    "        std::vector<entry> outs; // one per call\n        std::vector<int> same; // output is the input object\n"
    "        double next_draw; // torch.rand(1) after the calls\n    };\n\n"
    + "\n".join(lines)
    + "\n} // namespace golden\n// clang-format on\n"
)
print(f"wrote {out} ({len(lines)} entries)")
