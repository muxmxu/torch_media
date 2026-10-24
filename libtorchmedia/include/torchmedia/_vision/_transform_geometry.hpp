#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_TRANSFORM_GEOMETRY_HPP
#define LIB_TORCH_MEDIA_VISION_TRANSFORM_GEOMETRY_HPP
// Geometry transform classes (torchvision 0.20.1 transforms.v2 _geometry.py): resize / crop / pad / flips and the
// random crop, resize, rotation, affine, perspective, elastic and zoom-out families. Conventions: _transform_misc.hpp.
#include "_transform_misc.hpp"

namespace torchmedia::vision::transform {
    // ---------------- Resize ----------------
    typedef struct resize_option {
        std::vector<int64_t> _size;
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        std::optional<int64_t> _max_size;
        bool _antialias = true;

        auto size(std::vector<int64_t> v) -> resize_option & {
            _size = std::move(v);
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> resize_option & {
            _interpolation = m;
            return *this;
        }
        auto max_size(int64_t v) -> resize_option & {
            _max_size = v;
            return *this;
        }
        auto antialias(bool b) -> resize_option & {
            _antialias = b;
            return *this;
        }
    } resize_option_t;

    class Resize {
    public:
        // size: {h, w}, {shorter edge}, or {} (None) with max_size.
        explicit Resize(resize_option opt) : _opt(std::move(opt)) {
            if (_opt._size.size() > 2 || (_opt._size.empty() && !_opt._max_size)) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "Resize: size can be one or two integers, or empty (None) with max_size.");
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::resize(img, _opt._size, _opt._interpolation, _opt._max_size, _opt._antialias);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        resize_option _opt;
    };

    // ---------------- CenterCrop ----------------
    typedef struct center_crop_option {
        std::vector<int64_t> _size;

        auto size(std::vector<int64_t> v) -> center_crop_option & {
            _size = std::move(v);
            return *this;
        }
    } center_crop_option_t;

    class CenterCrop {
    public:
        explicit CenterCrop(center_crop_option opt) : _size(std::move(opt._size)) {
            detail::check_size("CenterCrop", _size);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t { return functional::center_crop(img, _size); }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        std::vector<int64_t> _size;
    };

    // ---------------- Pad ----------------
    typedef struct pad_option {
        std::vector<int64_t> _padding;
        std::vector<double> _fill; // {} = 0, one value, or one per channel
        functional::padding_mode _padding_mode = functional::padding_mode::constant;

        auto padding(std::vector<int64_t> v) -> pad_option & {
            _padding = std::move(v);
            return *this;
        }
        auto fill(std::vector<double> v) -> pad_option & {
            _fill = std::move(v);
            return *this;
        }
        auto padding_mode(functional::padding_mode m) -> pad_option & {
            _padding_mode = m;
            return *this;
        }
    } pad_option_t;

    class Pad {
    public:
        explicit Pad(pad_option opt) : _opt(std::move(opt)) {
            const auto n = _opt._padding.size();
            if (n != 1 && n != 2 && n != 4) {
                (void) handle_exceptions<int, std::invalid_argument>(std::format(
                        "Pad: padding must be an int or a 1, 2, or 4 element tuple, not a {} element tuple.", n));
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::pad(img, _opt._padding, _opt._fill, _opt._padding_mode);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        pad_option _opt;
    };

    // ---------------- RandomHorizontalFlip / RandomVerticalFlip ----------------
    class RandomHorizontalFlip {
    public:
        explicit RandomHorizontalFlip(random_flip_option opt = {}) : _p(opt._p) {
            detail::check_p("RandomHorizontalFlip", _p);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return detail::skip(_p) ? img : functional::hflip(img);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        double _p;
    };

    class RandomVerticalFlip {
    public:
        explicit RandomVerticalFlip(random_flip_option opt = {}) : _p(opt._p) {
            detail::check_p("RandomVerticalFlip", _p);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return detail::skip(_p) ? img : functional::vflip(img);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        double _p;
    };

    // ---------------- FiveCrop / TenCrop ----------------
    typedef struct five_crop_option {
        std::vector<int64_t> _size;

        auto size(std::vector<int64_t> v) -> five_crop_option & {
            _size = std::move(v);
            return *this;
        }
    } five_crop_option_t;

    class FiveCrop {
    public:
        explicit FiveCrop(five_crop_option opt) : _size(std::move(opt._size)) { detail::check_size("FiveCrop", _size); }

        auto operator()(const_tensor_lref_t img) const -> std::vector<tensor_t> {
            return functional::five_crop(img, _size);
        }
        auto forward(const_tensor_lref_t img) const -> std::vector<tensor_t> { return (*this)(img); }

    private:
        std::vector<int64_t> _size;
    };

    typedef struct ten_crop_option {
        std::vector<int64_t> _size;
        bool _vertical_flip = false;

        auto size(std::vector<int64_t> v) -> ten_crop_option & {
            _size = std::move(v);
            return *this;
        }
        auto vertical_flip(bool b) -> ten_crop_option & {
            _vertical_flip = b;
            return *this;
        }
    } ten_crop_option_t;

    class TenCrop {
    public:
        explicit TenCrop(ten_crop_option opt) : _opt(std::move(opt)) { detail::check_size("TenCrop", _opt._size); }

        auto operator()(const_tensor_lref_t img) const -> std::vector<tensor_t> {
            return functional::ten_crop(img, _opt._size, _opt._vertical_flip);
        }
        auto forward(const_tensor_lref_t img) const -> std::vector<tensor_t> { return (*this)(img); }

    private:
        ten_crop_option _opt;
    };

    namespace detail {
        // torchvision _setup_angle: {d} -> {-d, d} (d >= 0), or a sequence of 2 (or 4 with `four`) values.
        inline auto setup_angle(const char *cls, const char *name, std::vector<double> v, bool four = false)
                -> std::vector<double> {
            if (v.size() == 1) {
                if (v[0] < 0) {
                    return handle_exceptions<std::vector<double>, std::invalid_argument>(
                            {0.0, 0.0}, std::format("{}: if {} is a single number, it must be positive.", cls, name));
                }
                return {-v[0], v[0]};
            }
            if (v.size() != 2 && !(four && v.size() == 4)) {
                return handle_exceptions<std::vector<double>, std::invalid_argument>(
                        {0.0, 0.0},
                        std::format("{}: {} should be a sequence of length {}.", cls, name, four ? "2 or 4" : "2"));
            }
            return v;
        }

        inline auto check_center(const char *cls, const std::vector<double> &center) -> void {
            if (!center.empty() && center.size() != 2) {
                fail(std::format("{}: center should be a sequence of length 2.", cls));
            }
        }

        // torchvision _setup_number_or_seq: {x} -> {x, x}, or {x, y}.
        inline auto setup_number_or_seq(const char *cls, const char *name, std::vector<double> v)
                -> std::vector<double> {
            if (v.size() == 1) {
                return {v[0], v[0]};
            }
            if (v.size() != 2) {
                return handle_exceptions<std::vector<double>, std::invalid_argument>(
                        {0.0, 0.0}, std::format("{}: if {} is a sequence its length should be 1 or 2, got {}.", cls,
                                                name, v.size()));
            }
            return v;
        }
    } // namespace detail

    // ---------------- RandomCrop ----------------
    typedef struct random_crop_option {
        std::vector<int64_t> _size; // {s} or {h, w}
        std::vector<int64_t> _padding; // {} = None, {all}, {left/right, top/bottom} or {left, top, right, bottom}
        bool _pad_if_needed = false;
        std::vector<double> _fill{0.0}; // one value, or one per channel (constant mode)
        functional::padding_mode _padding_mode = functional::padding_mode::constant;

        auto size(std::vector<int64_t> v) -> random_crop_option & {
            _size = std::move(v);
            return *this;
        }
        auto padding(std::vector<int64_t> v) -> random_crop_option & {
            _padding = std::move(v);
            return *this;
        }
        auto pad_if_needed(bool b) -> random_crop_option & {
            _pad_if_needed = b;
            return *this;
        }
        auto fill(std::vector<double> v) -> random_crop_option & {
            _fill = std::move(v);
            return *this;
        }
        auto padding_mode(functional::padding_mode m) -> random_crop_option & {
            _padding_mode = m;
            return *this;
        }
    } random_crop_option_t;

    // Pad by `padding`, then (pad_if_needed) grow a short side by its shortfall on both ends, then crop a random
    // window. Draws torch.randint only along a side longer than the crop; nothing to pad or crop returns the input.
    class RandomCrop {
    public:
        explicit RandomCrop(random_crop_option opt) :
            _opt(std::move(opt)), _size(detail::setup_size("RandomCrop", _opt._size)) {
            const auto n = _opt._padding.size();
            if (n != 0 && n != 1 && n != 2 && n != 4) {
                _opt._padding.clear(); // a no-exceptions build then crops without padding
                detail::fail(std::format(
                        "RandomCrop: padding must be an int or a 1, 2, or 4 element tuple, not a {} element tuple.",
                        n));
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (!detail::size_ok(_opt._size)) { // LIB_TORCH_MEDIA_NO_EXCEPTIONS: the constructor rejected the size
                return functional::detail::invalid("RandomCrop: invalid arguments.");
            }
            auto [c, h, w] = detail::query_chw("RandomCrop", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto &p = _opt._padding;
            std::array<int64_t, 4> pad{}; // left, top, right, bottom
            if (!p.empty()) {
                pad = p.size() == 1   ? std::array{p[0], p[0], p[0], p[0]}
                      : p.size() == 2 ? std::array{p[0], p[1], p[0], p[1]}
                                      : std::array{p[0], p[1], p[2], p[3]};
                h += pad[1] + pad[3];
                w += pad[0] + pad[2];
            }
            const auto [ch, cw] = _size;
            if (_opt._pad_if_needed) {
                if (h < ch) {
                    const auto diff = ch - h;
                    pad[1] += diff;
                    pad[3] += diff;
                    h += 2 * diff;
                }
                if (w < cw) {
                    const auto diff = cw - w;
                    pad[0] += diff;
                    pad[2] += diff;
                    w += 2 * diff;
                }
            }
            if (h < ch || w < cw) {
                return functional::detail::invalid(std::format(
                        "RandomCrop: required crop size ({}, {}) is larger than {}input image size ({}, {}).", ch, cw,
                        p.empty() ? "" : "padded ", h, w));
            }
            const auto top = h > ch ? detail::randint(0, h - ch + 1) : 0;
            const auto left = w > cw ? detail::randint(0, w - cw + 1) : 0;
            auto out = img;
            if (pad != std::array<int64_t, 4>{}) {
                out = functional::pad(out, {pad[0], pad[1], pad[2], pad[3]}, _opt._fill, _opt._padding_mode);
            }
            return h > ch || w > cw ? functional::crop(out, top, left, ch, cw) : out;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_crop_option _opt;
        std::array<int64_t, 2> _size;
    };

    // ---------------- RandomResizedCrop ----------------
    typedef struct random_resized_crop_option {
        std::vector<int64_t> _size; // {s} or {h, w}
        std::array<double, 2> _scale{0.08, 1.0}; // crop area / image area
        std::array<double, 2> _ratio{3.0 / 4.0, 4.0 / 3.0}; // crop aspect ratio (w / h)
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        bool _antialias = true;

        auto size(std::vector<int64_t> v) -> random_resized_crop_option & {
            _size = std::move(v);
            return *this;
        }
        auto scale(std::array<double, 2> v) -> random_resized_crop_option & {
            _scale = v;
            return *this;
        }
        auto ratio(std::array<double, 2> v) -> random_resized_crop_option & {
            _ratio = v;
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_resized_crop_option & {
            _interpolation = m;
            return *this;
        }
        auto antialias(bool b) -> random_resized_crop_option & {
            _antialias = b;
            return *this;
        }
    } random_resized_crop_option_t;

    // Crop a random area (scale) with a random log-uniform aspect ratio (ratio), resized to size: up to 10 attempts,
    // then the whole image center-cropped to the nearest ratio bound.
    class RandomResizedCrop {
    public:
        explicit RandomResizedCrop(random_resized_crop_option opt) :
            _opt(std::move(opt)), _size(detail::setup_size("RandomResizedCrop", _opt._size)),
            _log_ratio(detail::log_ratio(_opt._ratio)) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (!detail::size_ok(_opt._size)) { // LIB_TORCH_MEDIA_NO_EXCEPTIONS: the constructor rejected the size
                return functional::detail::invalid("RandomResizedCrop: invalid arguments.");
            }
            const auto [c, height, width] = detail::query_chw("RandomResizedCrop", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto area = static_cast<double>(height * width);
            int64_t top = 0;
            int64_t left = 0;
            int64_t h = 0;
            int64_t w = 0;
            bool found = false;
            for (int attempt = 0; attempt < 10 && !found; ++attempt) {
                const double target_area = area * detail::uniform(_opt._scale[0], _opt._scale[1]);
                const double aspect_ratio =
                        torch::exp(torch::empty({1}).uniform_(_log_ratio[0], _log_ratio[1])).item<double>();
                w = detail::py_round(std::sqrt(target_area * aspect_ratio));
                h = detail::py_round(std::sqrt(target_area / aspect_ratio));
                if (0 < w && w <= width && 0 < h && h <= height) {
                    top = detail::randint(0, height - h + 1);
                    left = detail::randint(0, width - w + 1);
                    found = true;
                }
            }
            if (!found) {
                const double in_ratio = static_cast<double>(width) / static_cast<double>(height);
                const double min_ratio = std::min(_opt._ratio[0], _opt._ratio[1]);
                const double max_ratio = std::max(_opt._ratio[0], _opt._ratio[1]);
                w = width;
                h = height;
                if (in_ratio < min_ratio) {
                    h = detail::py_round(static_cast<double>(w) / min_ratio);
                } else if (in_ratio > max_ratio) {
                    w = detail::py_round(static_cast<double>(h) * max_ratio);
                }
                top = (height - h) / 2;
                left = (width - w) / 2;
            }
            return functional::resized_crop(img, top, left, h, w, {_size[0], _size[1]}, _opt._interpolation,
                                            _opt._antialias);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_resized_crop_option _opt;
        std::array<int64_t, 2> _size;
        std::array<double, 2> _log_ratio;
    };

    // ---------------- RandomResize / RandomShortestSize / ScaleJitter ----------------
    typedef struct random_resize_option {
        int64_t _min_size = 0;
        int64_t _max_size = 0; // exclusive
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        bool _antialias = true;

        auto min_size(int64_t v) -> random_resize_option & {
            _min_size = v;
            return *this;
        }
        auto max_size(int64_t v) -> random_resize_option & {
            _max_size = v;
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_resize_option & {
            _interpolation = m;
            return *this;
        }
        auto antialias(bool b) -> random_resize_option & {
            _antialias = b;
            return *this;
        }
    } random_resize_option_t;

    // Resize the shorter edge to torch.randint(min_size, max_size).
    class RandomResize {
    public:
        explicit RandomResize(random_resize_option opt) : _opt(opt) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto size = detail::randint(_opt._min_size, _opt._max_size);
            return functional::resize(img, {size}, _opt._interpolation, std::nullopt, _opt._antialias);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_resize_option _opt;
    };

    typedef struct random_shortest_size_option {
        std::vector<int64_t> _min_size; // candidate shorter-edge sizes
        std::optional<int64_t> _max_size; // cap on the longer edge
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        bool _antialias = true;

        auto min_size(std::vector<int64_t> v) -> random_shortest_size_option & {
            _min_size = std::move(v);
            return *this;
        }
        auto max_size(int64_t v) -> random_shortest_size_option & {
            _max_size = v;
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_shortest_size_option & {
            _interpolation = m;
            return *this;
        }
        auto antialias(bool b) -> random_shortest_size_option & {
            _antialias = b;
            return *this;
        }
    } random_shortest_size_option_t;

    // Scale so the shorter edge is a random pick of min_size (longer edge capped by max_size), sizes truncated.
    class RandomShortestSize {
    public:
        explicit RandomShortestSize(random_shortest_size_option opt) : _opt(std::move(opt)) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto [c, h, w] = detail::query_chw("RandomShortestSize", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto pick = detail::randint(0, static_cast<int64_t>(_opt._min_size.size()));
            double r = static_cast<double>(_opt._min_size[pick]) / static_cast<double>(std::min(h, w));
            if (_opt._max_size) {
                r = std::min(r, static_cast<double>(*_opt._max_size) / static_cast<double>(std::max(h, w)));
            }
            const auto new_h = static_cast<int64_t>(static_cast<double>(h) * r);
            const auto new_w = static_cast<int64_t>(static_cast<double>(w) * r);
            return functional::resize(img, {new_h, new_w}, _opt._interpolation, std::nullopt, _opt._antialias);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_shortest_size_option _opt;
    };

    typedef struct scale_jitter_option {
        std::array<int64_t, 2> _target_size{}; // {a, b}: base scale min(b / H, a / W), torchvision's indexing
        std::array<double, 2> _scale_range{0.1, 2.0};
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        bool _antialias = true;

        auto target_size(std::array<int64_t, 2> v) -> scale_jitter_option & {
            _target_size = v;
            return *this;
        }
        auto scale_range(std::array<double, 2> v) -> scale_jitter_option & {
            _scale_range = v;
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> scale_jitter_option & {
            _interpolation = m;
            return *this;
        }
        auto antialias(bool b) -> scale_jitter_option & {
            _antialias = b;
            return *this;
        }
    } scale_jitter_option_t;

    // Resize by min(target_size[1] / H, target_size[0] / W) * a uniform scale in scale_range (float32 math).
    class ScaleJitter {
    public:
        explicit ScaleJitter(scale_jitter_option opt) : _opt(opt) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto [c, h, w] = detail::query_chw("ScaleJitter", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto &s = _opt._scale_range;
            const auto scale = torch::rand({1}).mul(s[1] - s[0]).add(s[0]);
            const auto r = scale.mul(std::min(static_cast<double>(_opt._target_size[1]) / static_cast<double>(h),
                                              static_cast<double>(_opt._target_size[0]) / static_cast<double>(w)));
            const auto new_w = static_cast<int64_t>(r.mul(w).item<double>());
            const auto new_h = static_cast<int64_t>(r.mul(h).item<double>());
            return functional::resize(img, {new_h, new_w}, _opt._interpolation, std::nullopt, _opt._antialias);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        scale_jitter_option _opt;
    };

    // ---------------- RandomZoomOut ----------------
    typedef struct random_zoom_out_option {
        std::vector<double> _fill{0.0}; // one value, or one per channel
        std::array<double, 2> _side_range{1.0, 4.0}; // canvas side / image side
        double _p = 0.5;

        auto fill(std::vector<double> v) -> random_zoom_out_option & {
            _fill = std::move(v);
            return *this;
        }
        auto side_range(std::array<double, 2> v) -> random_zoom_out_option & {
            _side_range = v;
            return *this;
        }
        auto p(double v) -> random_zoom_out_option & {
            _p = v;
            return *this;
        }
    } random_zoom_out_option_t;

    // With probability p, place the image at a random position on a canvas side_range times larger, filled with fill.
    class RandomZoomOut {
    public:
        explicit RandomZoomOut(random_zoom_out_option opt = {}) : _opt(std::move(opt)) {
            detail::check_p("RandomZoomOut", _opt._p);
            const auto &s = _opt._side_range;
            if (s[0] < 1.0 || s[0] > s[1]) {
                const auto msg = std::format("RandomZoomOut: invalid side range provided ({}, {}).", s[0], s[1]);
                _opt._side_range = random_zoom_out_option()._side_range; // a no-exceptions build then uses the default
                detail::fail(msg);
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (detail::skip(_opt._p)) {
                return img;
            }
            const auto [c, h, w] = detail::query_chw("RandomZoomOut", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto &s = _opt._side_range;
            const auto r = torch::rand({1}).mul(s[1] - s[0]).add(s[0]);
            const auto canvas_w = static_cast<int64_t>(r.mul(w).item<double>());
            const auto canvas_h = static_cast<int64_t>(r.mul(h).item<double>());
            const auto r2 = torch::rand({2});
            const auto left = static_cast<int64_t>(r2[0].mul(canvas_w - w).item<double>());
            const auto top = static_cast<int64_t>(r2[1].mul(canvas_h - h).item<double>());
            return functional::pad(img, {left, top, canvas_w - (left + w), canvas_h - (top + h)}, _opt._fill);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_zoom_out_option _opt;
    };

    // ---------------- RandomRotation ----------------
    // fill (here and in RandomAffine / RandomPerspective / ElasticTransform): torchvision's default 0 is {0}; {} is
    // None (zero padding without the fill mask, which differs from {0} for bilinear); one value or one per channel.
    typedef struct random_rotation_option {
        std::vector<double> _degrees; // {d} = (-d, d), or {min, max}
        functional::interpolation_mode _interpolation = functional::interpolation_mode::nearest;
        bool _expand = false;
        std::vector<double> _center; // {} = image center, or {x, y}
        std::vector<double> _fill{0.0};

        auto degrees(std::vector<double> v) -> random_rotation_option & {
            _degrees = std::move(v);
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_rotation_option & {
            _interpolation = m;
            return *this;
        }
        auto expand(bool b) -> random_rotation_option & {
            _expand = b;
            return *this;
        }
        auto center(std::vector<double> v) -> random_rotation_option & {
            _center = std::move(v);
            return *this;
        }
        auto fill(std::vector<double> v) -> random_rotation_option & {
            _fill = std::move(v);
            return *this;
        }
    } random_rotation_option_t;

    class RandomRotation {
    public:
        explicit RandomRotation(random_rotation_option opt) : _opt(std::move(opt)) {
            _opt._degrees = detail::setup_angle("RandomRotation", "degrees", std::move(_opt._degrees));
            detail::check_center("RandomRotation", _opt._center);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto angle = detail::uniform(_opt._degrees[0], _opt._degrees[1]);
            return functional::rotate(img, angle, _opt._interpolation, _opt._expand, _opt._center, _opt._fill);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_rotation_option _opt;
    };

    // ---------------- RandomAffine ----------------
    typedef struct random_affine_option {
        std::vector<double> _degrees; // {d} = (-d, d), or {min, max}
        std::vector<double> _translate; // {} = None, or {max_dx, max_dy} as fractions of W / H in [0, 1]
        std::vector<double> _scale; // {} = None, or {min, max} (> 0)
        std::vector<double> _shear; // {} = None, {s} = x in (-s, s), {xmin, xmax} or {xmin, xmax, ymin, ymax}
        functional::interpolation_mode _interpolation = functional::interpolation_mode::nearest;
        std::vector<double> _fill{0.0};
        std::vector<double> _center; // {} = image center, or {x, y}

        auto degrees(std::vector<double> v) -> random_affine_option & {
            _degrees = std::move(v);
            return *this;
        }
        auto translate(std::vector<double> v) -> random_affine_option & {
            _translate = std::move(v);
            return *this;
        }
        auto scale(std::vector<double> v) -> random_affine_option & {
            _scale = std::move(v);
            return *this;
        }
        auto shear(std::vector<double> v) -> random_affine_option & {
            _shear = std::move(v);
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_affine_option & {
            _interpolation = m;
            return *this;
        }
        auto fill(std::vector<double> v) -> random_affine_option & {
            _fill = std::move(v);
            return *this;
        }
        auto center(std::vector<double> v) -> random_affine_option & {
            _center = std::move(v);
            return *this;
        }
    } random_affine_option_t;

    // Draws, in torchvision's order: angle, then (translate) tx, ty rounded to pixels, (scale) scale, (shear) shear x
    // and, with 4 shear values, shear y.
    class RandomAffine {
    public:
        explicit RandomAffine(random_affine_option opt) : _opt(std::move(opt)) {
            constexpr auto cls = "RandomAffine";
            _opt._degrees = detail::setup_angle(cls, "degrees", std::move(_opt._degrees));
            if (!_opt._translate.empty()) {
                if (_opt._translate.size() != 2) {
                    _opt._translate.clear();
                    detail::fail("RandomAffine: translate should be a sequence of length 2.");
                }
                if (std::ranges::any_of(_opt._translate, [](double t) { return !(0.0 <= t && t <= 1.0); })) {
                    _opt._translate.clear(); // a no-exceptions build then does not translate
                    detail::fail("RandomAffine: translation values should be between 0 and 1.");
                }
            }
            if (!_opt._scale.empty()) {
                if (_opt._scale.size() != 2) {
                    _opt._scale.clear();
                    detail::fail("RandomAffine: scale should be a sequence of length 2.");
                }
                if (std::ranges::any_of(_opt._scale, [](double s) { return s <= 0; })) {
                    _opt._scale.clear(); // a no-exceptions build then does not scale
                    detail::fail("RandomAffine: scale values should be positive.");
                }
            }
            if (!_opt._shear.empty()) {
                _opt._shear = detail::setup_angle(cls, "shear", std::move(_opt._shear), true);
            }
            detail::check_center(cls, _opt._center);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto [c, h, w] = detail::query_chw("RandomAffine", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto angle = detail::uniform(_opt._degrees[0], _opt._degrees[1]);
            std::vector<double> translate{0.0, 0.0};
            if (!_opt._translate.empty()) {
                const double max_dx = _opt._translate[0] * static_cast<double>(w);
                const double max_dy = _opt._translate[1] * static_cast<double>(h);
                translate[0] = static_cast<double>(detail::py_round(detail::uniform(-max_dx, max_dx)));
                translate[1] = static_cast<double>(detail::py_round(detail::uniform(-max_dy, max_dy)));
            }
            const double scale = _opt._scale.empty() ? 1.0 : detail::uniform(_opt._scale[0], _opt._scale[1]);
            std::vector<double> shear{0.0, 0.0};
            if (!_opt._shear.empty()) {
                shear[0] = detail::uniform(_opt._shear[0], _opt._shear[1]);
                if (_opt._shear.size() == 4) {
                    shear[1] = detail::uniform(_opt._shear[2], _opt._shear[3]);
                }
            }
            return functional::affine(img, angle, translate, scale, shear, _opt._interpolation, _opt._fill,
                                      _opt._center);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_affine_option _opt;
    };

    // ---------------- RandomPerspective ----------------
    typedef struct random_perspective_option {
        double _distortion_scale = 0.5; // in [0, 1]
        double _p = 0.5;
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        std::vector<double> _fill{0.0};

        auto distortion_scale(double v) -> random_perspective_option & {
            _distortion_scale = v;
            return *this;
        }
        auto p(double v) -> random_perspective_option & {
            _p = v;
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> random_perspective_option & {
            _interpolation = m;
            return *this;
        }
        auto fill(std::vector<double> v) -> random_perspective_option & {
            _fill = std::move(v);
            return *this;
        }
    } random_perspective_option_t;

    // With probability p, move each corner inwards by a random offset of up to distortion_scale * half the side.
    class RandomPerspective {
    public:
        explicit RandomPerspective(random_perspective_option opt = {}) : _opt(std::move(opt)) {
            detail::check_p("RandomPerspective", _opt._p);
            if (!(0.0 <= _opt._distortion_scale && _opt._distortion_scale <= 1.0)) {
                _opt._distortion_scale = random_perspective_option()._distortion_scale; // no-exceptions: the default
                detail::fail("RandomPerspective: distortion_scale value should be between 0 and 1.");
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (detail::skip(_opt._p)) {
                return img;
            }
            const auto [c, h, w] = detail::query_chw("RandomPerspective", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto bound_h = static_cast<int64_t>(_opt._distortion_scale * static_cast<double>(h / 2)) + 1;
            const auto bound_w = static_cast<int64_t>(_opt._distortion_scale * static_cast<double>(w / 2)) + 1;
            const auto point = [](int64_t x, int64_t y) {
                return std::array{static_cast<double>(x), static_cast<double>(y)};
            };
            std::vector<std::array<double, 2>> endpoints; // x drawn before y: argument evaluation order is unspecified
            const auto x_tl = detail::randint(0, bound_w);
            endpoints.push_back(point(x_tl, detail::randint(0, bound_h)));
            const auto x_tr = detail::randint(w - bound_w, w);
            endpoints.push_back(point(x_tr, detail::randint(0, bound_h)));
            const auto x_br = detail::randint(w - bound_w, w);
            endpoints.push_back(point(x_br, detail::randint(h - bound_h, h)));
            const auto x_bl = detail::randint(0, bound_w);
            endpoints.push_back(point(x_bl, detail::randint(h - bound_h, h)));
            const std::vector startpoints{point(0, 0), point(w - 1, 0), point(w - 1, h - 1), point(0, h - 1)};
            return functional::perspective(img, startpoints, endpoints, _opt._interpolation, _opt._fill);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_perspective_option _opt;
    };

    // ---------------- ElasticTransform ----------------
    typedef struct elastic_transform_option {
        std::vector<double> _alpha{50.0}; // {a} or {ax, ay}: displacement magnitude
        std::vector<double> _sigma{5.0}; // {s} or {sx, sy}: smoothness of the displacement field
        functional::interpolation_mode _interpolation = functional::interpolation_mode::bilinear;
        std::vector<double> _fill{0.0};

        auto alpha(std::vector<double> v) -> elastic_transform_option & {
            _alpha = std::move(v);
            return *this;
        }
        auto sigma(std::vector<double> v) -> elastic_transform_option & {
            _sigma = std::move(v);
            return *this;
        }
        auto interpolation(functional::interpolation_mode m) -> elastic_transform_option & {
            _interpolation = m;
            return *this;
        }
        auto fill(std::vector<double> v) -> elastic_transform_option & {
            _fill = std::move(v);
            return *this;
        }
    } elastic_transform_option_t;

    // Displace every pixel by a uniform [-1, 1] field (dx, then dy) blurred with gaussian_blur(k = int(8 sigma + 1)
    // made odd, sigma = both sigmas, as torchvision does) and scaled by alpha / H (dx) and alpha / W (dy).
    class ElasticTransform {
    public:
        explicit ElasticTransform(elastic_transform_option opt = {}) : _opt(std::move(opt)) {
            _opt._alpha = detail::setup_number_or_seq("ElasticTransform", "alpha", std::move(_opt._alpha));
            _opt._sigma = detail::setup_number_or_seq("ElasticTransform", "sigma", std::move(_opt._sigma));
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto [c, h, w] = detail::query_chw("ElasticTransform", img);
            if (c < 0) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: query_chw rejected img
            }
            const auto field = [&](int axis, int64_t size) {
                auto d = torch::rand({1, 1, h, w}).mul(2).sub(1);
                if (_opt._sigma[axis] > 0.0) {
                    d = functional::gaussian_blur(d, {odd_kernel(_opt._sigma[axis])}, _opt._sigma);
                }
                return d.mul(_opt._alpha[axis]).div(size);
            };
            const auto dx = field(0, h);
            const auto dy = field(1, w);
            return functional::elastic(img, torch::cat({dx, dy}, 1).permute({0, 2, 3, 1}), _opt._interpolation,
                                       _opt._fill);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        static auto odd_kernel(double sigma) -> int64_t {
            TORCHMEDIA_FP_CONTRACT_OFF
            const auto k = static_cast<int64_t>(8 * sigma + 1);
            return k % 2 == 0 ? k + 1 : k;
        }

        elastic_transform_option _opt;
    };

} // namespace torchmedia::vision::transform
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_GEOMETRY_HPP
