#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
#define LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
#include <ATen/Version.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../globel_include.hpp"

// Image tensors follow torchvision's [..., C, H, W] layout (uint8 in [0, 255] or floating in [0, 1]) with any number
// of leading batch dims. Every op mirrors the torchvision 0.20.1 transforms.v2.functional tensor kernel (`*_image`)
// and calls the same libtorch ops, so results match point-wise (develop_log/2026-10-23/progress07). Invalid arguments
// throw std::invalid_argument (torchvision: ValueError / TypeError / RuntimeError); what torchvision leaves to torch
// (e.g. a zero output size, a reflect pad wider than the image) stays a c10::Error.
// Call resize / pad qualified (functional::resize): unqualified calls also find at::resize / at::pad by ADL.
namespace torchmedia::vision::functional {

    enum class interpolation_mode { nearest, nearest_exact, bilinear, bicubic };
    enum class padding_mode { constant, edge, reflect, symmetric };

    namespace detail {
        // torchvision _max_value: the largest value of an integer dtype, 1 for floating (and any other) dtype.
        inline auto max_value(torch::Dtype dtype) -> int64_t {
            switch (dtype) {
                case torch::kUInt8:
                    return 255;
                case torch::kInt8:
                    return 127;
                case torch::kInt16:
                    return 32767;
                case torch::kUInt16:
                    return 65535;
                case torch::kInt32:
                    return 2147483647;
                case torch::kInt64:
                    return std::numeric_limits<int64_t>::max();
                default:
                    return 1;
            }
        }

        // torchvision _num_value_bits; 0 where torchvision raises (non-integer dtypes).
        inline auto num_value_bits(torch::Dtype dtype) -> int {
            switch (dtype) {
                case torch::kUInt8:
                    return 8;
                case torch::kInt8:
                    return 7;
                case torch::kInt16:
                    return 15;
                case torch::kUInt16:
                    return 16;
                case torch::kInt32:
                    return 31;
                case torch::kInt64:
                    return 63;
                default:
                    return 0;
            }
        }

        inline auto rank_error(const char *op, const tensor_t &img, int64_t min_dim) -> std::string {
            return std::format("{}: expected a tensor with at least {} dims ({}), got {} dims.", op, min_dim,
                               min_dim == 2 ? "..., H, W" : "..., C, H, W", img.dim());
        }

        // Symmetric padding by index gathering, torchvision _pad_symmetric on a [N, C, H, W] tensor; `padding` is
        // [left, right, top, bottom] and negative entries crop first.
        inline auto pad_symmetric(const tensor_t &img, std::vector<int64_t> padding) -> tensor_t {
            auto x = img;
            const auto neg = [&](size_t i) { return -std::min<int64_t>(padding[i], 0); };
            x = x.slice(-2, neg(2), x.size(-2) - neg(3)).slice(-1, neg(0), x.size(-1) - neg(1));
            for (auto &p: padding) {
                p = std::max<int64_t>(p, 0);
            }
            const auto opts = torch::TensorOptions().dtype(torch::kLong).device(x.device());
            const auto indices = [&](int64_t before, int64_t n, int64_t after) {
                return torch::cat({torch::arange(before - 1, -1, -1, opts), torch::arange(n, opts),
                                   -torch::arange(1, after + 1, opts)});
            };
            const auto y = indices(padding[2], x.size(-2), padding[3]);
            const auto xs = indices(padding[0], x.size(-1), padding[1]);
            using torch::indexing::Slice;
            return x.index({Slice(), Slice(), y.unsqueeze(1), xs.unsqueeze(0)});
        }

        // torchvision _pad_with_scalar_fill: `padding` is torch order [left, right, top, bottom]; needs >= 3 dims.
        inline auto pad_with_scalar_fill(const tensor_t &img, const std::vector<int64_t> &padding, double fill,
                                         padding_mode mode) -> tensor_t {
            auto shape = img.sizes().vec();
            int64_t batch = 1;
            for (size_t i = 0; i + 3 < shape.size(); ++i) {
                batch *= shape[i];
            }
            auto x = img.reshape({batch, img.size(-3), img.size(-2), img.size(-1)});
            if (mode == padding_mode::constant) {
                x = torch::pad(x, padding, "constant", fill);
            } else if (mode == padding_mode::symmetric) {
                x = pad_symmetric(x, padding);
            } else {
                // torch only reflect / replicate pads floating tensors.
                const bool cast = !x.is_floating_point();
                x = torch::pad(cast ? x.to(torch::kFloat32) : x, padding,
                               mode == padding_mode::edge ? "replicate" : "reflect");
                if (cast) {
                    x = x.to(img.scalar_type());
                }
            }
            shape[shape.size() - 2] = x.size(-2);
            shape.back() = x.size(-1);
            return x.reshape(shape);
        }

        // torchvision _compute_resized_output_size (v2: an empty `size` is None and needs max_size). Returns an error
        // message, or "" with {new_h, new_w} in `out`.
        inline auto resized_output_size(int64_t h, int64_t w, const std::vector<int64_t> &size,
                                        std::optional<int64_t> max_size, std::array<int64_t, 2> &out) -> std::string {
            if (size.size() > 2) {
                return std::format("size must have 1 or 2 elements, or be empty with max_size, got {}.", size.size());
            }
            if (size.size() == 2) {
                if (max_size) {
                    return "max_size should only be passed if size is None or specifies the length of the smaller "
                           "edge.";
                }
                out = {size[0], size[1]};
                return "";
            }
            const auto [short_edge, long_edge] = std::minmax(h, w);
            // Python int(a * b / c): exact integer product, correctly rounded true division, truncation.
            const auto scaled = [](int64_t a, int64_t b, int64_t c) {
                return static_cast<int64_t>(static_cast<double>(a * b) / static_cast<double>(c));
            };
            int64_t new_short = 0;
            int64_t new_long = 0;
            if (size.empty()) {
                if (!max_size) {
                    return "max_size must be an integer when size is None.";
                }
                if (long_edge == 0) {
                    return "division by zero (the image has no pixels).";
                }
                new_short = scaled(*max_size, short_edge, long_edge);
                new_long = *max_size;
            } else {
                if (short_edge == 0) {
                    return "division by zero (the image has an empty edge).";
                }
                new_short = size[0];
                new_long = scaled(size[0], long_edge, short_edge);
                if (max_size) {
                    if (*max_size <= size[0]) {
                        return std::format("max_size = {} must be strictly greater than the requested size for the "
                                           "smaller edge size = {}.",
                                           *max_size, size[0]);
                    }
                    if (new_long > *max_size) {
                        new_short = scaled(*max_size, new_short, new_long);
                        new_long = *max_size;
                    }
                }
            }
            out = w <= h ? std::array<int64_t, 2>{new_long, new_short} : std::array<int64_t, 2>{new_short, new_long};
            return "";
        }

        // torchvision _do_native_uint8_resize_on_cpu (eager mode).
        inline auto native_uint8_resize_on_cpu(interpolation_mode mode) -> bool {
            if (mode == interpolation_mode::bilinear) {
                return at::get_cpu_capability().find("AVX2") != std::string::npos;
            }
            return mode == interpolation_mode::bicubic;
        }

        // torch.nn.functional.interpolate on a 4-D tensor, align_corners=False for the (bi)linear/cubic modes.
        inline auto interpolate(const tensor_t &x, const std::array<int64_t, 2> &hw, interpolation_mode mode,
                                bool antialias) -> tensor_t {
            const torch::IntArrayRef size(hw);
            switch (mode) {
                case interpolation_mode::nearest:
                    return at::upsample_nearest2d(x, size);
                case interpolation_mode::nearest_exact:
                    return at::_upsample_nearest_exact2d(x, size);
                case interpolation_mode::bilinear:
                    return antialias ? at::_upsample_bilinear2d_aa(x, size, false)
                                     : at::upsample_bilinear2d(x, size, false);
                default:
                    return antialias ? at::_upsample_bicubic2d_aa(x, size, false)
                                     : at::upsample_bicubic2d(x, size, false);
            }
        }

        inline auto five_crop_error(const tensor_t &img, const std::vector<int64_t> &size) -> std::string {
            if (size.empty() || size.size() > 2) {
                return "Please provide only two dimensions (h, w) for size.";
            }
            if (img.dim() < 2) {
                return rank_error("five_crop", img, 2);
            }
            if (size.back() > img.size(-1) || size[0] > img.size(-2)) {
                return std::format("Requested crop size ({}, {}) is bigger than input size ({}, {}).", size[0],
                                   size.back(), img.size(-2), img.size(-1));
            }
            return "";
        }
    } // namespace detail

    inline auto hflip(const tensor_t &img) -> tensor_t { return img.flip(-1); }

    inline auto vflip(const tensor_t &img) -> tensor_t { return img.flip(-2); }

    // RGB -> luma L = 0.2989 R + 0.587 G + 0.114 B in torchvision's op order, cast back to the input dtype (uint8
    // truncates); a 1-channel input is copied or repeated. num_output_channels = 3 returns an expanded view.
    inline auto rgb_to_grayscale(const tensor_t &img, int num_output_channels = 1) -> tensor_t {
        if (num_output_channels != 1 && num_output_channels != 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    std::format("rgb_to_grayscale: num_output_channels must be 1 or 3, got {}.", num_output_channels));
        }
        if (img.dim() < 3 || (img.size(-3) != 1 && img.size(-3) != 3)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "rgb_to_grayscale: expected a [..., 1 or 3, H, W] image.");
        }
        if (img.size(-3) == 1) {
            if (num_output_channels == 1) {
                return img.clone();
            }
            std::vector<int64_t> reps(img.dim(), 1);
            reps[img.dim() - 3] = 3;
            return img.repeat(reps);
        }
        // integers in float32 up front (torch's own promotion, same values): MPS mul(uint8, float) aborts the process
        const auto rgb = (img.is_floating_point() ? img : img.to(torch::kFloat32)).unbind(-3);
        auto l = rgb[0].mul(0.2989).add_(rgb[1], 0.587).add_(rgb[2], 0.114).unsqueeze(-3).to(img.scalar_type());
        return num_output_channels == 3 ? l.expand(img.sizes()) : l;
    }

    // Per-channel (img - mean) / std on a floating image; mean / std are rounded to the image dtype like torchvision.
    inline auto normalize(const tensor_t &img, const std::vector<double> &mean, const std::vector<double> &stddev,
                          bool inplace = false) -> tensor_t {
        if (!img.is_floating_point()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("normalize: input tensor should be a float "
                                                                      "tensor.");
        }
        if (img.dim() < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("normalize", img, 3));
        }
        if (std::ranges::find(stddev, 0.0) != stddev.end()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "normalize: std evaluated to zero, leading to division by zero.");
        }
        const auto m = torch::tensor(mean, torch::kFloat64).to(img.options()).view({-1, 1, 1});
        const auto s = torch::tensor(stddev, torch::kFloat64).to(img.options()).view({-1, 1, 1});
        return (inplace ? img.sub_(m) : img.sub(m)).div_(s);
    }

    // Crop [top, top + height) x [left, left + width); the part outside the image is zero-padded (torchvision crop).
    inline auto crop(const tensor_t &img, int64_t top, int64_t left, int64_t height, int64_t width) -> tensor_t {
        if (img.dim() < 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("crop", img, 2));
        }
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        const auto right = left + width;
        const auto bottom = top + height;
        if (left < 0 || top < 0 || right > w || bottom > h) {
            if (img.dim() < 3) {
                return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("crop", img, 3));
            }
            const auto inside =
                    img.slice(-2, std::max<int64_t>(top, 0), bottom).slice(-1, std::max<int64_t>(left, 0), right);
            const std::vector<int64_t> padding{
                    std::max<int64_t>(std::min<int64_t>(right, 0) - left, 0),
                    std::max<int64_t>(right - std::max(w, left), 0),
                    std::max<int64_t>(std::min<int64_t>(bottom, 0) - top, 0),
                    std::max<int64_t>(bottom - std::max(h, top), 0),
            };
            return detail::pad_with_scalar_fill(inside, padding, 0.0, padding_mode::constant);
        }
        return img.slice(-2, top, bottom).slice(-1, left, right);
    }

    // Center crop to output_size ({s} or {h, w}); an image smaller than the crop is zero-padded first. The anchor is
    // Python round((H - h) / 2), i.e. half to even.
    inline auto center_crop(const tensor_t &img, const std::vector<int64_t> &output_size) -> tensor_t {
        if (output_size.empty() || output_size.size() > 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "center_crop: output_size must have 1 or 2 elements (h, w).");
        }
        if (img.dim() < 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("center_crop", img, 2));
        }
        const auto crop_h = output_size[0];
        const auto crop_w = output_size.back();
        if (img.numel() == 0) {
            auto shape = img.sizes().vec();
            shape[shape.size() - 2] = crop_h;
            shape.back() = crop_w;
            return img.reshape(shape);
        }
        auto x = img;
        if (crop_h > x.size(-2) || crop_w > x.size(-1)) {
            const auto h = x.size(-2);
            const auto w = x.size(-1);
            x = torch::pad(x,
                           {crop_w > w ? (crop_w - w) / 2 : 0, crop_w > w ? (crop_w - w + 1) / 2 : 0,
                            crop_h > h ? (crop_h - h) / 2 : 0, crop_h > h ? (crop_h - h + 1) / 2 : 0},
                           "constant", 0.0);
            if (crop_h == x.size(-2) && crop_w == x.size(-1)) {
                return x;
            }
        }
        const auto top = static_cast<int64_t>(std::nearbyint(static_cast<double>(x.size(-2) - crop_h) / 2.0));
        const auto left = static_cast<int64_t>(std::nearbyint(static_cast<double>(x.size(-1) - crop_w) / 2.0));
        return x.slice(-2, top, top + crop_h).slice(-1, left, left + crop_w);
    }

    // Pad by padding = {all}, {left/right, top/bottom} or {left, top, right, bottom}. fill: {} (0), one value, or one
    // value per channel (constant mode only). Integer images are edge/reflect padded through float32 like torchvision.
    inline auto pad(const tensor_t &img, const std::vector<int64_t> &padding, const std::vector<double> &fill = {},
                    padding_mode mode = padding_mode::constant) -> tensor_t {
        std::vector<int64_t> tp; // torch order: left, right, top, bottom
        switch (padding.size()) {
            case 1:
                tp = {padding[0], padding[0], padding[0], padding[0]};
                break;
            case 2:
                tp = {padding[0], padding[0], padding[1], padding[1]};
                break;
            case 4:
                tp = {padding[0], padding[2], padding[1], padding[3]};
                break;
            default:
                return handle_exceptions<tensor_t, std::invalid_argument>(std::format(
                        "pad: padding must be an int or a 1, 2, or 4 element tuple, not a {} element tuple.",
                        padding.size()));
        }
        if (fill.size() > 1 && mode != padding_mode::constant) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "pad: a non-scalar fill is only supported with padding_mode::constant.");
        }
        if (img.dim() < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("pad", img, 3));
        }
        if (fill.size() <= 1) {
            return detail::pad_with_scalar_fill(img, tp, fill.empty() ? 0.0 : fill[0], mode);
        }
        auto out = detail::pad_with_scalar_fill(img, tp, 0.0, padding_mode::constant);
        // torchvision builds the fill tensor in the default dtype (float32), then casts it to the image dtype.
        const std::vector<float> fill_f(fill.begin(), fill.end());
        const auto f = torch::tensor(fill_f, torch::TensorOptions().device(img.device()))
                               .to(img.scalar_type())
                               .reshape({-1, 1, 1});
        if (tp[2] > 0) {
            out.slice(-2, 0, tp[2]).copy_(f);
        }
        if (tp[0] > 0) {
            out.slice(-1, 0, tp[0]).copy_(f);
        }
        if (tp[3] > 0) {
            out.slice(-2, out.size(-2) - tp[3]).copy_(f);
        }
        if (tp[1] > 0) {
            out.slice(-1, out.size(-1) - tp[1]).copy_(f);
        }
        return out;
    }

    // Resize to size = {h, w}, {shorter edge} (aspect kept, optionally capped by max_size) or {} (longer edge =
    // max_size). antialias applies to bilinear / bicubic only. uint8 is interpolated natively where torchvision does
    // (nearest modes; bicubic on CPU; bilinear on AVX2 CPUs), otherwise through float32 and rounded back (bicubic
    // clamped first). A same-size request returns `img` itself.
    inline auto resize(const tensor_t &img, const std::vector<int64_t> &size,
                       interpolation_mode interpolation = interpolation_mode::bilinear,
                       std::optional<int64_t> max_size = std::nullopt, bool antialias = true) -> tensor_t {
        if (img.dim() < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("resize", img, 3));
        }
        const auto c = img.size(-3);
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        std::array<int64_t, 2> hw{};
        if (const auto err = detail::resized_output_size(h, w, size, max_size, hw); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("resize: " + err);
        }
        if (hw[0] == h && hw[1] == w) {
            return img;
        }
        auto out_shape = img.sizes().vec();
        out_shape[out_shape.size() - 2] = hw[0];
        out_shape.back() = hw[1];
        if (img.numel() == 0) {
            return img.reshape(out_shape);
        }
        const bool nearest =
                interpolation == interpolation_mode::nearest || interpolation == interpolation_mode::nearest_exact;
        const auto dtype = img.scalar_type();
        const bool uint8_ok = nearest || (img.device().is_cpu() && detail::native_uint8_resize_on_cpu(interpolation));
        const bool need_cast =
                !(dtype == torch::kFloat32 || dtype == torch::kFloat64 || (dtype == torch::kUInt8 && uint8_ok));
        // shortcut: torchvision's channels_last restride for 1-image batches is skipped; it only keeps the output's
        // memory format (values are equal). Add it if a caller needs channels_last outputs.
        auto x = img.reshape({-1, c, h, w});
        x = detail::interpolate(need_cast ? x.to(torch::kFloat32) : x, hw, interpolation, antialias && !nearest);
        if (need_cast) {
            if (interpolation == interpolation_mode::bicubic && dtype == torch::kUInt8) {
                x = x.clamp_(0, 255);
            }
            if (dtype == torch::kUInt8 || dtype == torch::kInt8 || dtype == torch::kInt16 || dtype == torch::kInt32 ||
                dtype == torch::kInt64) {
                x = x.round_();
            }
            x = x.to(dtype);
        }
        return x.reshape(out_shape);
    }

    inline auto resized_crop(const tensor_t &img, int64_t top, int64_t left, int64_t height, int64_t width,
                             const std::vector<int64_t> &size,
                             interpolation_mode interpolation = interpolation_mode::bilinear, bool antialias = true)
            -> tensor_t {
        if (img.dim() < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(detail::rank_error("resized_crop", img, 3));
        }
        return resize(crop(img, top, left, height, width), size, interpolation, std::nullopt, antialias);
    }

    // The four corner crops and the center crop, in torchvision order: tl, tr, bl, br, center.
    inline auto five_crop(const tensor_t &img, const std::vector<int64_t> &size) -> std::vector<tensor_t> {
        if (const auto err = detail::five_crop_error(img, size); !err.empty()) {
            return handle_exceptions<std::vector<tensor_t>, std::invalid_argument>("five_crop: " + err);
        }
        const auto ch = size[0];
        const auto cw = size.back();
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        return {crop(img, 0, 0, ch, cw), crop(img, 0, w - cw, ch, cw), crop(img, h - ch, 0, ch, cw),
                crop(img, h - ch, w - cw, ch, cw), center_crop(img, {ch, cw})};
    }

    // five_crop of the image, then five_crop of its horizontal (or vertical) flip: 10 tensors.
    inline auto ten_crop(const tensor_t &img, const std::vector<int64_t> &size, bool vertical_flip = false)
            -> std::vector<tensor_t> {
        if (const auto err = detail::five_crop_error(img, size); !err.empty()) {
            return handle_exceptions<std::vector<tensor_t>, std::invalid_argument>("ten_crop: " + err);
        }
        auto crops = five_crop(img, size);
        const auto flipped = five_crop(vertical_flip ? vflip(img) : hflip(img), size);
        crops.insert(crops.end(), flipped.begin(), flipped.end());
        return crops;
    }

    // Convert to dtype; with scale, rescale between value ranges like torchvision: int -> float divides by the int
    // max, float -> int multiplies by (max + 1 - 1e-3) and truncates (no clamping), int -> int shifts bits.
    inline auto to_dtype(const tensor_t &img, torch::Dtype dtype = torch::kFloat32, bool scale = false) -> tensor_t {
        const auto in = img.scalar_type();
        if (in == dtype) {
            return img;
        }
        if (!scale) {
            return img.to(dtype);
        }
        const bool float_out = c10::isFloatingType(dtype);
        if (img.is_floating_point()) {
            if (float_out) {
                return img.to(dtype);
            }
            if ((in == torch::kFloat32 && (dtype == torch::kInt32 || dtype == torch::kInt64)) ||
                (in == torch::kFloat64 && dtype == torch::kInt64)) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        std::format("to_dtype: the conversion from {} to {} cannot be performed safely.",
                                    c10::toString(in), c10::toString(dtype)));
            }
            return img.mul(static_cast<double>(detail::max_value(dtype)) + 1.0 - 1e-3).to(dtype);
        }
        if (float_out) {
            // out of place: Tensor::mul_(Scalar) rounds the scalar to half / bfloat16 first, Python's mul_(float) not
            return img.to(dtype).mul(1.0 / static_cast<double>(detail::max_value(in)));
        }
        const int in_bits = detail::num_value_bits(in);
        const int out_bits = detail::num_value_bits(dtype);
        if (in_bits == 0 || out_bits == 0) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "to_dtype: scaling between these dtypes is only defined for integer dtypes.");
        }
        const int shift = std::abs(in_bits - out_bits);
        if (in_bits > out_bits) {
            // torch has no uint16 shifts: true division (to float32), then truncation.
            return (in == torch::kUInt16 ? img.div(int64_t{1} << shift) : img.bitwise_right_shift(shift)).to(dtype);
        }
        return dtype == torch::kUInt16 ? img.to(dtype).mul(int64_t{1} << shift)
                                       : img.to(dtype).bitwise_left_shift_(shift);
    }

    // Scale by brightness_factor and clamp to [0, dtype max]; integer images are computed in float32 and truncated.
    inline auto adjust_brightness(const tensor_t &img, double brightness_factor) -> tensor_t {
        if (brightness_factor < 0) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    std::format("adjust_brightness: brightness_factor ({}) is not non-negative.", brightness_factor));
        }
        if (img.dim() < 3 || (img.size(-3) != 1 && img.size(-3) != 3)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "adjust_brightness: expected a [..., 1 or 3, H, W] image.");
        }
        const auto out = img.mul(brightness_factor).clamp_(0, detail::max_value(img.scalar_type()));
        return img.is_floating_point() ? out : out.to(img.scalar_type());
    }

    // 1 - x for floating images, ~x for uint8, x ^ max for signed integer images (keeps the sign bit).
    inline auto invert(const tensor_t &img) -> tensor_t {
        if (img.is_floating_point()) {
            return torch::rsub(img, 1.0);
        }
        if (img.scalar_type() == torch::kUInt8) {
            return img.bitwise_not();
        }
        if (detail::num_value_bits(img.scalar_type()) == 0) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "invert: only floating and integer images are supported.");
        }
        return img.bitwise_xor(detail::max_value(img.scalar_type()));
    }

} // namespace torchmedia::vision::functional
#endif // LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
