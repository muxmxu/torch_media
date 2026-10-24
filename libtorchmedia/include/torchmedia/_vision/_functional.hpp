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
#include <numbers>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../global_include.hpp"

// Scalar math that mirrors Python float arithmetic must not be fused into FMAs (clang contracts a * b + c within an
// expression by default; GCC in ISO mode and MSVC do not). Used as the first statement of such function bodies.
#if defined(__clang__)
#define TORCHMEDIA_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define TORCHMEDIA_FP_CONTRACT_OFF
#endif

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
            // torch's CPU gather raises IndexError for a pad wider than the image unless the output is empty; MPS would
            // read out of bounds
            const auto h = x.size(-2);
            const auto w = x.size(-1);
            const bool empty =
                    x.size(0) * x.size(1) * (h + padding[2] + padding[3]) * (w + padding[0] + padding[1]) == 0;
            const bool fits = empty || (padding[0] <= w && padding[1] <= w && padding[2] <= h && padding[3] <= h);
            TORCH_CHECK_INDEX(fits, std::format("pad: symmetric padding wider than the image ({} x {}).", h, w));
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

        inline auto invalid(std::string msg) -> tensor_t {
            return handle_exceptions<tensor_t, std::invalid_argument>(std::move(msg));
        }

        inline auto bad_channels(const tensor_t &img) -> bool {
            return img.dim() < 3 || (img.size(-3) != 1 && img.size(-3) != 3);
        }

        // The float luma [..., 1, H, W] of a 3-channel image (torchvision _rgb_to_grayscale_image, preserve_dtype
        // False). Integers enter as float32 up front (torch's own promotion, same values): MPS mul(uint8, float)
        // aborts.
        inline auto luma(const tensor_t &img) -> tensor_t {
            const auto rgb = (img.is_floating_point() ? img : img.to(torch::kFloat32)).unbind(-3);
            return rgb[0].mul(0.2989).add_(rgb[1], 0.587).add_(rgb[2], 0.114).unsqueeze(-3);
        }

        // torchvision _blend: img1 * ratio + img2 * (1 - ratio), clamped to the dtype range, integers truncated back.
        inline auto blend(const tensor_t &img1, const tensor_t &img2, double ratio) -> tensor_t {
            const auto out = img1.mul(ratio).add_(img2, 1.0 - ratio).clamp_(0, max_value(img1.scalar_type()));
            return img1.is_floating_point() ? out : out.to(img1.scalar_type());
        }

        // torchvision _rgb_to_hsv / _hsv_to_rgb on float [..., 3, H, W], the same in-place op sequence.
        inline auto rgb_to_hsv(const tensor_t &image) -> tensor_t {
            const auto ch = image.unbind(-3);
            auto [minc, maxc] = torch::aminmax(image, -3);
            const auto eqc = maxc == minc;
            const auto channels_range = maxc - minc;
            const auto ones = torch::ones_like(maxc);
            const auto s = channels_range / torch::where(eqc, ones, maxc);
            const auto divisor = torch::where(eqc, ones, channels_range).unsqueeze_(-3);
            const auto c = ((maxc.unsqueeze(-3) - image) / divisor).unbind(-3);
            auto rc = c[0], gc = c[1], bc = c[2];
            auto maxc_neq_r = maxc != ch[0];
            auto maxc_eq_g = maxc == ch[1];
            const auto hg = rc.add(2.0).sub_(bc).mul_(maxc_eq_g & maxc_neq_r);
            const auto hr = bc.sub_(gc).mul_(~maxc_neq_r);
            const auto hb = gc.add_(4.0).sub_(rc).mul_(maxc_neq_r.logical_and_(maxc_eq_g.logical_not_()));
            auto h = hr.add_(hg).add_(hb);
            h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0);
            return torch::stack({h, s, maxc}, -3);
        }

        inline auto hsv_to_rgb(const tensor_t &img) -> tensor_t {
            const auto hsv = img.unbind(-3);
            const auto &s = hsv[1];
            const auto &v = hsv[2];
            const auto h6 = hsv[0].mul(6);
            auto i = torch::floor(h6);
            const auto f = h6.sub_(i);
            i = i.to(torch::kInt32);
            auto sxf = s * f;
            auto one_minus_s = torch::rsub(s, 1.0);
            const auto q = torch::rsub(sxf, 1.0).mul_(v).clamp_(0.0, 1.0);
            const auto t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0);
            const auto p = one_minus_s.mul_(v).clamp_(0.0, 1.0);
            i.remainder_(6);
            const auto vpqt = torch::stack({v, p, q, t}, -3);
            auto select = torch::tensor({0, 2, 1, 1, 3, 0, 3, 0, 0, 2, 1, 1, 1, 1, 3, 0, 0, 2}, torch::kLong)
                                  .view({3, 6})
                                  .to(img.device());
            select = select.index({torch::indexing::Slice(), i});
            if (select.dim() > 3) {
                select = select.movedim(0, -3);
            }
            return vpqt.gather(-3, select);
        }

        // Python float `a % b` for b > 0 (fmod, shifted into [0, b)).
        inline auto py_mod(double a, double b) -> double {
            const double m = std::fmod(a, b);
            return m < 0 ? m + b : (m == 0 ? 0.0 : m);
        }

        // Python math.sin: clang (Darwin) merges sin(x) and cos(x) of one x into __sincos_stret, whose sin can be 1 ulp
        // off libm's sin; the volatile copy keeps the calls apart.
        inline auto py_sin(double x) -> double {
            const volatile double v = x;
            return std::sin(v);
        }

        // torchvision v2 _get_inverse_affine_matrix (inverted=True): the output -> input pixel map around `center`
        // (relative to the image center), in Python's double op order (v2's, not transforms.functional's).
        inline auto inverse_affine_matrix(const std::array<double, 2> &center, double angle,
                                          const std::array<double, 2> &translate, double scale,
                                          const std::array<double, 2> &shear) -> std::array<double, 6> {
            TORCHMEDIA_FP_CONTRACT_OFF
            constexpr double deg_to_rad = std::numbers::pi / 180.0; // math.radians
            const double rot = angle * deg_to_rad;
            const double sx = shear[0] * deg_to_rad;
            const double sy = shear[1] * deg_to_rad;
            const double cos_sy = std::cos(sy);
            const double tan_sx = std::tan(sx);
            const double rot_minus_sy = rot - sy;
            const double cx_plus_tx = center[0] + translate[0];
            const double cy_plus_ty = center[1] + translate[1];
            const double a = std::cos(rot_minus_sy) / cos_sy;
            const double b = -(a * tan_sx + py_sin(rot));
            const double c = py_sin(rot_minus_sy) / cos_sy;
            const double d = std::cos(rot) - c * tan_sx;
            std::array<double, 6> m{d / scale, -b / scale, 0.0, -c / scale, a / scale, 0.0};
            m[2] += center[0] - m[0] * cx_plus_tx - m[1] * cy_plus_ty;
            m[5] += center[1] - m[3] * cx_plus_tx - m[4] * cy_plus_ty;
            return m;
        }

        // torchvision _compute_affine_output_size_tensor (float32 corner math, 1e-4 truncation): {w, h}.
        inline auto affine_output_size(const std::array<double, 6> &m, int64_t w, int64_t h) -> std::array<int64_t, 2> {
            const double hw = 0.5 * static_cast<double>(w);
            const double hh = 0.5 * static_cast<double>(h);
            const auto f32 = [](std::vector<double> v) {
                return torch::tensor(v, torch::kFloat64).to(torch::kFloat32);
            };
            const auto pts = f32({-hw, -hh, 1.0, -hw, hh, 1.0, hw, hh, 1.0, hw, -hh, 1.0}).view({4, 3});
            const auto theta = f32({m.begin(), m.end()}).view({2, 3});
            auto [min_vals, max_vals] = torch::matmul(pts, theta.t()).aminmax(0);
            const auto halfs = f32({hw, hh});
            min_vals.add_(halfs);
            max_vals.add_(halfs);
            const double tol = 1e-4;
            const double inv_tol = 1.0 / tol;
            const auto cmax = max_vals.mul_(inv_tol).trunc_().mul_(tol).ceil_();
            const auto cmin = min_vals.mul_(inv_tol).trunc_().mul_(tol).floor_();
            const auto size = cmax.sub_(cmin);
            return {static_cast<int64_t>(size[0].item<float>()), static_cast<int64_t>(size[1].item<float>())};
        }

        // torchvision _affine_grid: the sampling grid of an output of oh x ow pixels, normalised by the input w x h.
        // The base linspaces are default-dtype (float32) even for a float64 theta, as in torchvision.
        inline auto affine_grid(const tensor_t &theta, int64_t w, int64_t h, int64_t ow, int64_t oh) -> tensor_t {
            const auto dev = torch::TensorOptions().device(theta.device());
            auto base_grid = torch::empty({1, oh, ow, 3}, theta.options());
            const double fow = static_cast<double>(ow);
            const double foh = static_cast<double>(oh);
            base_grid.select(-1, 0).copy_(torch::linspace((1.0 - fow) * 0.5, (fow - 1.0) * 0.5, ow, dev));
            base_grid.select(-1, 1).copy_(
                    torch::linspace((1.0 - foh) * 0.5, (foh - 1.0) * 0.5, oh, dev).unsqueeze_(-1));
            base_grid.select(-1, 2).fill_(1);
            const auto scale =
                    torch::tensor({0.5 * static_cast<double>(w), 0.5 * static_cast<double>(h)}, torch::kFloat64)
                            .to(theta.options());
            const auto rescaled_theta = theta.transpose(1, 2).div_(scale);
            return base_grid.view({1, oh * ow, 3}).bmm(rescaled_theta).view({1, oh, ow, 2});
        }

        // torchvision _assert_grid_transform_inputs plus the [..., C, H, W] rank the grid ops need.
        inline auto grid_error(const char *op, const tensor_t &img, interpolation_mode mode,
                               const std::vector<double> &fill, const std::vector<double> &center = {}) -> std::string {
            if (img.dim() < 3) {
                return rank_error(op, img, 3);
            }
            if (!center.empty() && center.size() != 2) {
                return std::format("{}: center must be empty or {{x, y}}.", op);
            }
            if (fill.size() > 1 && static_cast<int64_t>(fill.size()) != img.size(-3)) {
                return std::format("{}: the number of elements in 'fill' cannot broadcast to match the number of "
                                   "channels of the image ({} != {}).",
                                   op, fill.size(), img.size(-3));
            }
            if (mode != interpolation_mode::nearest && mode != interpolation_mode::bilinear) {
                return std::format("{}: only nearest and bilinear interpolation are supported.", op);
            }
            return "";
        }

        // torchvision _apply_grid_transform: grid_sample with zero padding; a non-empty fill samples a ones mask
        // alongside the image and blends the fill in where the mask is not 1. Integer images are rounded back.
        inline auto apply_grid_transform(const tensor_t &img, tensor_t grid, interpolation_mode mode,
                                         const std::vector<double> &fill) -> tensor_t {
            const auto c = img.size(-3);
            const auto ih = img.size(-2);
            const auto iw = img.size(-1);
            auto out_shape = img.sizes().vec();
            out_shape[out_shape.size() - 2] = grid.size(1);
            out_shape.back() = grid.size(2);
            if (img.numel() == 0) {
                return img.reshape(out_shape);
            }
            auto x = img.reshape({-1, c, ih, iw});
            const auto batch = x.size(0);
            const bool fp = img.scalar_type() == grid.scalar_type();
            if (!fp) {
                x = x.to(grid.scalar_type());
            }
            if (batch > 1) {
                grid = grid.expand({batch, -1, -1, -1});
            }
            if (!fill.empty()) {
                x = torch::cat({x, torch::ones({batch, 1, ih, iw}, x.options())}, 1);
            }
            // torch.nn.functional.grid_sample: mode bilinear 0, nearest 1, bicubic 2; padding zeros 0
            const int64_t m = mode == interpolation_mode::bilinear ? 0 : mode == interpolation_mode::nearest ? 1 : 2;
            x = torch::grid_sampler(x, grid, m, 0, false);
            if (!fill.empty()) {
                const auto mask = x.slice(1, c).expand({-1, c, -1, -1});
                x = x.slice(1, 0, c);
                const auto fill_img = torch::tensor(fill, torch::kFloat64).to(x.options()).view({1, -1, 1, 1});
                x = mode == interpolation_mode::nearest ? torch::where(mask < 0.5, fill_img.expand_as(x), x)
                                                        : x.sub_(fill_img).mul_(mask).add_(fill_img);
            }
            return (fp ? x : x.round_().to(img.scalar_type())).reshape(out_shape);
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
        if (detail::bad_channels(img)) {
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
        auto l = detail::luma(img).to(img.scalar_type());
        return num_output_channels == 3 ? l.expand(img.sizes()) : l;
    }

    // A 1-channel image repeated to 3 channels; an image with >= 3 channels is returned as is.
    inline auto grayscale_to_rgb(const tensor_t &img) -> tensor_t {
        return img.dim() >= 3 && img.size(-3) >= 3 ? img : rgb_to_grayscale(img, 3);
    }

    // Reorder the channels: output channel k is input channel permutation[k] (Python indexing, negatives wrap).
    inline auto permute_channels(const tensor_t &img, const std::vector<int64_t> &permutation) -> tensor_t {
        if (img.dim() < 3) {
            return detail::invalid(detail::rank_error("permute_channels", img, 3));
        }
        const auto c = img.size(-3);
        if (static_cast<int64_t>(permutation.size()) != c) {
            return detail::invalid(
                    std::format("permute_channels: length of permutation does not match number of channels: {} != {}",
                                permutation.size(), c));
        }
        if (img.numel() == 0) {
            return img;
        }
        using torch::indexing::Slice;
        const auto index = torch::tensor(permutation, torch::TensorOptions().dtype(torch::kLong).device(img.device()));
        return img.reshape({-1, c, img.size(-2), img.size(-1)}).index({Slice(), index}).reshape(img.sizes());
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
        if (detail::bad_channels(img)) {
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

    // Blend with the mean luma of each image (torchvision adjust_contrast: integer luma floored first).
    inline auto adjust_contrast(const tensor_t &img, double contrast_factor) -> tensor_t {
        if (contrast_factor < 0) {
            return detail::invalid(
                    std::format("adjust_contrast: contrast_factor ({}) is not non-negative.", contrast_factor));
        }
        if (detail::bad_channels(img)) {
            return detail::invalid("adjust_contrast: expected a [..., 1 or 3, H, W] image.");
        }
        const bool fp = img.is_floating_point();
        auto gray = img.size(-3) == 3 ? detail::luma(img) : (fp ? img : img.to(torch::kFloat32));
        if (!fp && img.size(-3) == 3) {
            gray = gray.floor_();
        }
        return detail::blend(img, gray.mean({-3, -2, -1}, true), contrast_factor);
    }

    // Blend with the per-pixel luma (integer luma floored first); a 1-channel image is returned as is.
    inline auto adjust_saturation(const tensor_t &img, double saturation_factor) -> tensor_t {
        if (saturation_factor < 0) {
            return detail::invalid(
                    std::format("adjust_saturation: saturation_factor ({}) is not non-negative.", saturation_factor));
        }
        if (detail::bad_channels(img)) {
            return detail::invalid("adjust_saturation: expected a [..., 1 or 3, H, W] image.");
        }
        if (img.size(-3) == 1) {
            return img;
        }
        auto gray = detail::luma(img);
        return detail::blend(img, img.is_floating_point() ? gray : gray.floor_(), saturation_factor);
    }

    // Blend the interior pixels with a 3 x 3 smoothing (1, 5 center / 13) of the image; the border is kept.
    inline auto adjust_sharpness(const tensor_t &img, double sharpness_factor) -> tensor_t {
        if (detail::bad_channels(img)) {
            return detail::invalid("adjust_sharpness: expected a [..., 1 or 3, H, W] image.");
        }
        if (sharpness_factor < 0) {
            return detail::invalid(
                    std::format("adjust_sharpness: sharpness_factor ({}) is not non-negative.", sharpness_factor));
        }
        const auto c = img.size(-3);
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        if (img.numel() == 0 || h <= 2 || w <= 2) {
            return img;
        }
        const bool fp = img.is_floating_point();
        const auto kdtype = fp ? img.scalar_type() : torch::kFloat32;
        const double a = 1.0 / 13.0;
        const double b = 5.0 / 13.0;
        const auto kernel = torch::tensor({a, a, a, a, b, a, a, a, a}, torch::kFloat64)
                                    .to(torch::TensorOptions().dtype(kdtype).device(img.device()))
                                    .view({3, 3})
                                    .expand({c, 1, 3, 3});
        auto out = (img.dim() > 4 ? img.reshape({-1, c, h, w}) : img).to(kdtype, false, true);
        auto blurred = torch::conv2d(out, kernel, {}, 1, torch::IntArrayRef{0}, 1, c);
        if (!fp) {
            blurred = blurred.round_();
        }
        auto view = out.slice(-2, 1, h - 1).slice(-1, 1, w - 1);
        view.add_(blurred.sub_(view), 1.0 - sharpness_factor);
        out = out.clamp_(0, detail::max_value(img.scalar_type()));
        if (!fp) {
            out = out.to(img.scalar_type());
        }
        return img.dim() > 4 ? out.reshape(img.sizes()) : out;
    }

    // Shift the hue by hue_factor (in [-0.5, 0.5]) through HSV in float32; a 1-channel image is returned as is.
    inline auto adjust_hue(const tensor_t &img, double hue_factor) -> tensor_t {
        if (!(hue_factor >= -0.5 && hue_factor <= 0.5)) {
            return detail::invalid(std::format("adjust_hue: hue_factor ({}) is not in [-0.5, 0.5].", hue_factor));
        }
        if (detail::bad_channels(img)) {
            return detail::invalid("adjust_hue: expected a [..., 1 or 3, H, W] image.");
        }
        if (img.size(-3) == 1 || img.numel() == 0) {
            return img;
        }
        auto hsv = detail::rgb_to_hsv(to_dtype(img, torch::kFloat32, true)).unbind(-3);
        hsv[0].add_(hue_factor).remainder_(1.0);
        return to_dtype(detail::hsv_to_rgb(torch::stack(hsv, -3)), img.scalar_type(), true);
    }

    // gain * x^gamma on the [0, 1] scale (integers are scaled to float32 and back), clamped only when gain != 1.
    inline auto adjust_gamma(const tensor_t &img, double gamma, double gain = 1.0) -> tensor_t {
        if (gamma < 0) {
            return detail::invalid("adjust_gamma: gamma should be a non-negative real number.");
        }
        auto out = img.is_floating_point() ? img.pow(gamma) : to_dtype(img, torch::kFloat32, true).pow_(gamma);
        if (gain != 1.0) {
            out = out.mul(gain).clamp_(0.0, 1.0); // out of place: see to_dtype (mul_(Scalar) on half / bfloat16)
        }
        return to_dtype(out, img.scalar_type(), true);
    }

    // Keep the `bits` most significant bits (floating images: floor to 2^bits levels).
    inline auto posterize(const tensor_t &img, int64_t bits) -> tensor_t {
        if (img.is_floating_point()) {
            if (bits < 0 || bits > 62) {
                return detail::invalid(
                        std::format("posterize: bits must be in [0, 62] for floating images, got {}.", bits));
            }
            const int64_t levels = int64_t{1} << bits;
            return img.mul(levels).floor_().clamp_(0, levels - 1).mul_(1.0 / static_cast<double>(levels));
        }
        const int64_t num_bits = detail::num_value_bits(img.scalar_type());
        if (num_bits == 0 || bits < 0) {
            return detail::invalid("posterize: expected an integer or floating image and bits >= 0.");
        }
        if (bits >= num_bits) {
            return img;
        }
        return img.bitwise_and(((int64_t{1} << bits) - 1) << (num_bits - bits));
    }

    // Invert the pixels >= threshold.
    inline auto solarize(const tensor_t &img, double threshold) -> tensor_t {
        if (threshold > static_cast<double>(detail::max_value(img.scalar_type()))) {
            return detail::invalid(std::format(
                    "solarize: threshold should be less or equal the maximum value of the dtype, but got {}.",
                    threshold));
        }
        const auto inverted = invert(img);
        return inverted.defined() ? torch::where(img >= threshold, inverted, img) : inverted; // undefined: rejected
    }

    // Stretch each channel to the full range (a constant channel is left as is).
    inline auto autocontrast(const tensor_t &img) -> tensor_t {
        if (detail::bad_channels(img)) {
            return detail::invalid("autocontrast: expected a [..., 1 or 3, H, W] image.");
        }
        if (img.numel() == 0) {
            return img;
        }
        const auto bound = detail::max_value(img.scalar_type());
        const bool fp = img.is_floating_point();
        auto f = fp ? img : img.to(torch::kFloat32);
        auto minimum = f.amin({-2, -1}, true);
        auto maximum = f.amax({-2, -1}, true);
        const auto eq_idxs = maximum == minimum;
        auto inv_scale = maximum.sub_(minimum).mul_(1.0 / static_cast<double>(bound));
        minimum.index_put_({eq_idxs}, 0.0);
        inv_scale.index_put_({eq_idxs}, 1.0);
        const auto diff = fp ? f.sub(minimum) : f.sub_(minimum);
        return diff.div_(inv_scale).clamp_(0, bound).to(img.scalar_type());
    }

    // Histogram equalization per channel with PIL's lookup table, computed on the uint8 version of the image.
    inline auto equalize(const tensor_t &img) -> tensor_t {
        if (img.numel() == 0) {
            return img;
        }
        const auto u8 = to_dtype(img, torch::kUInt8, true);
        if (!u8.defined()) {
            return u8; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: to_dtype rejected the dtype
        }
        if (u8.dim() < 2) {
            return detail::invalid(detail::rank_error("equalize", img, 2));
        }
        auto lut_shape = u8.sizes().vec();
        lut_shape.resize(lut_shape.size() - 2);
        const auto flat = u8.flatten(-2).to(torch::kLong);
        lut_shape.push_back(256);
        auto hist = flat.new_zeros(lut_shape, torch::kInt32);
        hist.scatter_add_(-1, flat, hist.new_ones({1}).expand_as(flat));
        auto cum_hist = hist.cumsum(-1);
        const auto index = cum_hist.argmax(-1).unsqueeze_(-1);
        auto step = torch::rsub(hist.gather(-1, index), flat.size(-1)).div_(255, "floor");
        const auto valid = step.ne(0).unsqueeze_(-1);
        cum_hist = cum_hist.slice(-1, 0, 255);
        cum_hist.add_(torch::floor_divide(step, 2));
        cum_hist.div_(step.clamp_min_(1), "floor").clamp_(0, 255);
        auto lut = cum_hist.to(torch::kUInt8);
        lut_shape.back() = 1;
        lut = torch::cat({lut.new_zeros({1}).expand(lut_shape), lut}, -1);
        const auto equalized = lut.gather(-1, flat).view_as(u8);
        return to_dtype(torch::where(valid, equalized, u8), img.scalar_type(), true);
    }

    // Gaussian blur with reflect padding. kernel_size {k} or {kx, ky} (odd, positive); sigma {} (0.3 * ((k - 1) *
    // 0.5 - 1) + 0.8 = k * 0.15 + 0.35 per axis), {s} or {sx, sy}. Integer images are blurred in float32 and rounded.
    inline auto gaussian_blur(const tensor_t &img, std::vector<int64_t> kernel_size, std::vector<double> sigma = {})
            -> tensor_t {
        TORCHMEDIA_FP_CONTRACT_OFF
        if (kernel_size.size() == 1) {
            kernel_size.push_back(kernel_size[0]);
        }
        if (kernel_size.size() != 2) {
            return detail::invalid(
                    std::format("gaussian_blur: kernel_size must have 1 or 2 elements, got {}.", kernel_size.size()));
        }
        for (const auto k: kernel_size) {
            if (k % 2 == 0 || k < 0) {
                return detail::invalid(
                        std::format("gaussian_blur: kernel_size should have odd and positive integers, got {}.", k));
            }
        }
        if (sigma.empty()) {
            for (const auto k: kernel_size) {
                sigma.push_back(static_cast<double>(k) * 0.15 + 0.35);
            }
        } else if (sigma.size() == 1) {
            sigma.push_back(sigma[0]);
        }
        if (sigma.size() != 2 || sigma[0] <= 0.0 || sigma[1] <= 0.0) { // torchvision's `s <= 0.0`: NaN passes
            return detail::invalid("gaussian_blur: sigma should have 1 or 2 positive values.");
        }
        if (img.dim() < 3) {
            return detail::invalid(detail::rank_error("gaussian_blur", img, 3));
        }
        if (img.numel() == 0) {
            return img;
        }
        const bool fp = img.is_floating_point();
        const auto opts = torch::TensorOptions().dtype(fp ? img.scalar_type() : torch::kFloat32).device(img.device());
        const auto kernel1d = [&](int64_t k, double s) {
            const double lim = static_cast<double>(k - 1) / (2.0 * std::sqrt(2.0));
            return torch::softmax(torch::linspace(-lim, lim, k, opts).div(s).pow(2).neg(), 0);
        };
        const auto c = img.size(-3);
        const auto kernel = (kernel1d(kernel_size[1], sigma[1]).unsqueeze(-1) * kernel1d(kernel_size[0], sigma[0]))
                                    .expand({c, 1, kernel_size[1], kernel_size[0]});
        auto x = img.dim() == 3 ? img.unsqueeze(0) : img.reshape({-1, c, img.size(-2), img.size(-1)});
        x = torch::pad(fp ? x : x.to(torch::kFloat32),
                       {kernel_size[0] / 2, kernel_size[0] / 2, kernel_size[1] / 2, kernel_size[1] / 2}, "reflect");
        x = torch::conv2d(x, kernel, {}, 1, torch::IntArrayRef{0}, 1, c).reshape(img.sizes());
        return fp ? x : x.round_().to(img.scalar_type());
    }

    // img + mean + sigma * N(0, 1) noise drawn with randn_like (the image's device generator), clamped to [0, 1]
    // with clip. Floating images only.
    inline auto gaussian_noise(const tensor_t &img, double mean = 0.0, double sigma = 0.1, bool clip = true)
            -> tensor_t {
        if (!img.is_floating_point()) {
            return detail::invalid(std::format("gaussian_noise: input tensor is expected to be in float dtype, got {}.",
                                               c10::toString(img.scalar_type())));
        }
        if (sigma < 0) {
            return detail::invalid(std::format("gaussian_noise: sigma shouldn't be negative, got {}.", sigma));
        }
        const auto out = img.add(torch::randn_like(img).mul(sigma).add(mean));
        return clip ? out.clamp(0, 1) : out;
    }

    // Set img[..., i:i+h, j:j+w] = v (Python slice semantics); a copy unless inplace.
    inline auto erase(const tensor_t &img, int64_t i, int64_t j, int64_t h, int64_t w, const tensor_t &v,
                      bool inplace = false) -> tensor_t {
        auto out = inplace ? img : img.clone();
        using torch::indexing::Ellipsis;
        using torch::indexing::Slice;
        out.index_put_({Ellipsis, Slice(i, i + h), Slice(j, j + w)}, v);
        return out;
    }

    // ---------------- geometry: grid-sampled warps (torchvision rotate / affine / perspective / elastic)
    // ---------------- interpolation: nearest or bilinear (elastic also bicubic). fill: {} = None (zero padding, no
    // mask), {v} or one value per channel (blended in through a sampled mask, as torchvision does). center: {} = image
    // center, or {x, y} in pixel coordinates.

    // Rotate counter-clockwise by angle degrees; expand grows the output to hold the whole rotated image. Multiples of
    // 90 degrees about the image center are exact transposes (180, and 90 / 270 when square or expand).
    inline auto rotate(const tensor_t &img, double angle,
                       interpolation_mode interpolation = interpolation_mode::nearest, bool expand = false,
                       const std::vector<double> &center = {}, const std::vector<double> &fill = {}) -> tensor_t {
        angle = detail::py_mod(angle, 360.0);
        if (center.empty()) {
            if (angle == 0) {
                return img.clone();
            }
            if (angle == 180) {
                return torch::rot90(img, 2, {-2, -1});
            }
            if (expand || img.size(-1) == img.size(-2)) {
                if (angle == 90) {
                    return torch::rot90(img, 1, {-2, -1});
                }
                if (angle == 270) {
                    return torch::rot90(img, 3, {-2, -1});
                }
            }
        }
        if (auto err = detail::grid_error("rotate", img, interpolation, fill, center); !err.empty()) {
            return detail::invalid(err);
        }
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        std::array<double, 2> center_f{0.0, 0.0};
        if (!center.empty()) { // relative to the image center
            center_f = {center[0] - static_cast<double>(w) * 0.5, center[1] - static_cast<double>(h) * 0.5};
        }
        const auto m = detail::inverse_affine_matrix(center_f, -angle, {0.0, 0.0}, 1.0, {0.0, 0.0});
        const auto out_wh = expand ? detail::affine_output_size(m, w, h) : std::array<int64_t, 2>{w, h};
        const auto dtype = img.is_floating_point() ? img.scalar_type() : torch::kFloat32;
        const auto theta = torch::tensor(torch::ArrayRef<double>(m.data(), m.size()), torch::kFloat64)
                                   .to(torch::TensorOptions().dtype(dtype).device(img.device()))
                                   .reshape({1, 2, 3});
        return detail::apply_grid_transform(img, detail::affine_grid(theta, w, h, out_wh[0], out_wh[1]), interpolation,
                                            fill);
    }

    // Affine warp: rotate by angle degrees (clockwise), scale, shear ({s} = {s, s}; torchvision's scalar s is
    // {s, 0}) in degrees, then translate by {tx, ty} pixels, all about center.
    inline auto affine(const tensor_t &img, double angle, const std::vector<double> &translate, double scale,
                       std::vector<double> shear, interpolation_mode interpolation = interpolation_mode::nearest,
                       const std::vector<double> &fill = {}, const std::vector<double> &center = {}) -> tensor_t {
        if (translate.size() != 2) {
            return detail::invalid("affine: translate should be a sequence of length 2.");
        }
        if (scale <= 0.0) {
            return detail::invalid("affine: scale should be positive.");
        }
        if (shear.size() == 1) {
            shear.push_back(shear[0]);
        }
        if (shear.size() != 2) {
            return detail::invalid("affine: shear should contain one or two values.");
        }
        if (auto err = detail::grid_error("affine", img, interpolation, fill, center); !err.empty()) {
            return detail::invalid(err);
        }
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        std::array<double, 2> center_f{0.0, 0.0};
        if (!center.empty()) {
            center_f = {center[0] - static_cast<double>(w) * 0.5, center[1] - static_cast<double>(h) * 0.5};
        }
        const auto m = detail::inverse_affine_matrix(center_f, angle, {translate[0], translate[1]}, scale,
                                                     {shear[0], shear[1]});
        const auto dtype = img.is_floating_point() ? img.scalar_type() : torch::kFloat32;
        const auto theta = torch::tensor(torch::ArrayRef<double>(m.data(), m.size()), torch::kFloat64)
                                   .to(torch::TensorOptions().dtype(dtype).device(img.device()))
                                   .reshape({1, 2, 3});
        return detail::apply_grid_transform(img, detail::affine_grid(theta, w, h, w, h), interpolation, fill);
    }

    // Perspective warp mapping the 4 startpoints (tl, tr, br, bl; {x, y}) to the endpoints, or given directly by the 8
    // coefficients (a..h: x' = (ax + by + c) / (gx + hy + 1), y' = (dx + ey + f) / (gx + hy + 1)). Integral point
    // coordinates are solved exactly like torchvision's Python ints; a row with a fractional coordinate is rounded to
    // float32 like its Python float list.
    inline auto perspective(const tensor_t &img, const std::vector<std::array<double, 2>> &startpoints,
                            const std::vector<std::array<double, 2>> &endpoints,
                            interpolation_mode interpolation = interpolation_mode::bilinear,
                            const std::vector<double> &fill = {}, const std::vector<double> &coefficients = {})
            -> tensor_t {
        const bool points = !startpoints.empty() && !endpoints.empty();
        std::string err;
        if (!coefficients.empty()) {
            err = points ? "the startpoints/endpoints and the coefficients shouldn't be defined concurrently."
                  : coefficients.size() != 8 ? "coefficients should have 8 float values."
                                             : "";
        } else if (!points) {
            err = "either the startpoints/endpoints or the coefficients must be given.";
        } else if (startpoints.size() != 4 || endpoints.size() != 4) {
            err = std::format("please provide exactly four corners, got {} startpoints and {} endpoints.",
                              startpoints.size(), endpoints.size());
        }
        if (!err.empty()) {
            return detail::invalid("perspective: " + err);
        }
        if (auto e = detail::grid_error("perspective", img, interpolation, fill); !e.empty()) {
            return detail::invalid(e);
        }
        std::vector<double> c = coefficients;
        if (c.empty()) { // torchvision _get_perspective_coeffs: float64 least squares (gels), rounded to float32
            auto a = torch::zeros({8, 8}, torch::kFloat64);
            std::vector<double> b;
            const auto integral = [](double v) { return v == std::trunc(v); };
            for (int64_t i = 0; i < 4; ++i) {
                const auto &p1 = endpoints[i];
                const auto &p2 = startpoints[i];
                for (int64_t k = 0; k < 2; ++k) {
                    std::vector<double> row{0, 0, 0, 0, 0, 0, -p2[k] * p1[0], -p2[k] * p1[1]};
                    row[3 * k] = p1[0];
                    row[3 * k + 1] = p1[1];
                    row[3 * k + 2] = 1;
                    auto r = torch::tensor(row, torch::kFloat64);
                    if (!(integral(p1[0]) && integral(p1[1]) && integral(p2[k]))) {
                        r = r.to(torch::kFloat32).to(torch::kFloat64);
                    }
                    a[2 * i + k].copy_(r);
                    b.push_back(p2[k]);
                }
            }
            const auto res =
                    std::get<0>(torch::linalg_lstsq(a, torch::tensor(b, torch::kFloat64), std::nullopt, "gels"))
                            .to(torch::kFloat32)
                            .to(torch::kFloat64);
            c.assign(res.data_ptr<double>(), res.data_ptr<double>() + 8);
        }
        // torchvision _perspective_grid
        const auto oh = img.size(-2);
        const auto ow = img.size(-1);
        const auto opts = torch::TensorOptions()
                                  .dtype(img.is_floating_point() ? img.scalar_type() : torch::kFloat32)
                                  .device(img.device());
        const auto theta1 =
                torch::tensor({c[0], c[1], c[2], c[3], c[4], c[5]}, torch::kFloat64).to(opts).view({1, 2, 3});
        const auto theta2 = torch::tensor({c[6], c[7], 1.0, c[6], c[7], 1.0}, torch::kFloat64).to(opts).view({1, 2, 3});
        const double d = 0.5;
        auto base_grid = torch::empty({1, oh, ow, 3}, opts);
        base_grid.select(-1, 0).copy_(torch::linspace(d, static_cast<double>(ow) + d - 1.0, ow, opts));
        base_grid.select(-1, 1).copy_(torch::linspace(d, static_cast<double>(oh) + d - 1.0, oh, opts).unsqueeze_(-1));
        base_grid.select(-1, 2).fill_(1);
        const auto scale =
                torch::tensor({0.5 * static_cast<double>(ow), 0.5 * static_cast<double>(oh)}, torch::kFloat64).to(opts);
        const auto rescaled_theta1 = theta1.transpose(1, 2).div_(scale);
        const auto flat = base_grid.view({1, oh * ow, 3});
        // A zero denominator (crossing corners, e.g. distortion_scale near 1) gives inf / NaN, on which torch's CPU
        // bilinear grid_sample dies with SIGBUS (torchvision too): those pixels go off the image and take the fill.
        const auto grid = flat.bmm(rescaled_theta1)
                                  .div_(flat.bmm(theta2.transpose(1, 2)))
                                  .sub_(1.0)
                                  .nan_to_num_(-4.0, -4.0, -4.0);
        return detail::apply_grid_transform(img, grid.view({1, oh, ow, 2}), interpolation, fill);
    }

    // Displace every output pixel by displacement [1, H, W, 2] (normalised [-1, 1] grid units) from the identity grid.
    inline auto elastic(const tensor_t &img, const tensor_t &displacement,
                        interpolation_mode interpolation = interpolation_mode::bilinear,
                        const std::vector<double> &fill = {}) -> tensor_t {
        if (img.dim() < 3) {
            return detail::invalid(detail::rank_error("elastic", img, 3));
        }
        if (interpolation == interpolation_mode::nearest_exact) {
            return detail::invalid("elastic: nearest_exact interpolation is not supported by grid_sample.");
        }
        const auto h = img.size(-2);
        const auto w = img.size(-1);
        if (displacement.sizes() != torch::IntArrayRef({1, h, w, 2})) {
            return detail::invalid(std::format("elastic: displacement shape should be (1, {}, {}, 2).", h, w));
        }
        // torchvision patch: CPU float16 is warped in float32
        const bool cpu_half = img.device().is_cpu() && img.scalar_type() == torch::kHalf;
        const auto x = cpu_half ? img.to(torch::kFloat32) : img;
        const auto opts = torch::TensorOptions()
                                  .dtype(x.is_floating_point() ? x.scalar_type() : torch::kFloat32)
                                  .device(img.device());
        auto grid = torch::empty({1, h, w, 2}, opts); // torchvision _create_identity_grid
        const double fw = static_cast<double>(w);
        const double fh = static_cast<double>(h);
        grid.select(-1, 0).copy_(torch::linspace((1.0 - fw) / fw, (fw - 1.0) / fw, w, opts));
        grid.select(-1, 1).copy_(torch::linspace((1.0 - fh) / fh, (fh - 1.0) / fh, h, opts).unsqueeze_(-1));
        const auto out = detail::apply_grid_transform(x, grid.add_(displacement.to(opts)), interpolation, fill);
        return cpu_half ? out.to(torch::kHalf) : out;
    }

} // namespace torchmedia::vision::functional
#endif // LIB_TORCH_MEDIA_VISION_FUNCTIONAL_HPP
