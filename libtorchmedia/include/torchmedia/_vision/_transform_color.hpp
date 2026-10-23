#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_TRANSFORM_COLOR_HPP
#define LIB_TORCH_MEDIA_VISION_TRANSFORM_COLOR_HPP
// Color transform classes (torchvision 0.20.1 transforms.v2 _color.py). Conventions: _transform_misc.hpp.
#include "_transform_misc.hpp"

namespace torchmedia::vision::transform {
    // ---------------- Grayscale ----------------
    typedef struct grayscale_option {
        int _num_output_channels = 1;

        auto num_output_channels(int n) -> grayscale_option & {
            _num_output_channels = n;
            return *this;
        }
    } grayscale_option_t;

    class Grayscale {
    public:
        explicit Grayscale(grayscale_option opt = {}) : _num_output_channels(opt._num_output_channels) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::rgb_to_grayscale(img, _num_output_channels);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        int _num_output_channels;
    };

    // ---------------- p-gated color transforms ----------------
    // Random{Invert, Autocontrast, Equalize, Grayscale, Posterize, Solarize, AdjustSharpness}: torchvision v2
    // _RandomApplyTransform subclasses, one torch.rand(1) draw per call, the op runs unless draw >= p.
    namespace detail {
        // A p-gated functional op; Op is a callable (tensor) -> tensor.
        template<class Op>
        class p_gated {
        public:
            p_gated(const char *cls, double p, Op op) : _p(p), _op(std::move(op)) { check_p(cls, p); }
            auto operator()(const_tensor_lref_t img) const -> tensor_t { return skip(_p) ? img : _op(img); }
            auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

        private:
            double _p;
            Op _op;
        };

        struct invert_op {
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::invert(x); }
        };
        struct autocontrast_op {
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::autocontrast(x); }
        };
        struct equalize_op {
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::equalize(x); }
        };
        // RandomGrayscale keeps the channel count (torchvision: num_output_channels = num_input_channels).
        struct grayscale_op {
            auto operator()(const_tensor_lref_t x) const -> tensor_t {
                return functional::rgb_to_grayscale(x, x.dim() < 3 ? 1 : static_cast<int>(x.size(-3)));
            }
        };
        struct posterize_op {
            int64_t bits;
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::posterize(x, bits); }
        };
        struct solarize_op {
            double threshold;
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::solarize(x, threshold); }
        };
        struct sharpness_op {
            double factor;
            auto operator()(const_tensor_lref_t x) const -> tensor_t { return functional::adjust_sharpness(x, factor); }
        };
    } // namespace detail

    class RandomInvert : public detail::p_gated<detail::invert_op> {
    public:
        explicit RandomInvert(random_apply_option opt = {}) : p_gated("RandomInvert", opt._p, {}) {}
    };

    class RandomAutocontrast : public detail::p_gated<detail::autocontrast_op> {
    public:
        explicit RandomAutocontrast(random_apply_option opt = {}) : p_gated("RandomAutocontrast", opt._p, {}) {}
    };

    class RandomEqualize : public detail::p_gated<detail::equalize_op> {
    public:
        explicit RandomEqualize(random_apply_option opt = {}) : p_gated("RandomEqualize", opt._p, {}) {}
    };

    // Default p = 0.1 (torchvision).
    class RandomGrayscale : public detail::p_gated<detail::grayscale_op> {
    public:
        explicit RandomGrayscale(random_apply_option opt = random_apply_option().p(0.1)) :
            p_gated("RandomGrayscale", opt._p, {}) {}
    };

    // bits / threshold / sharpness_factor have no default in torchvision; set them. The defaults here (8 bits, a NaN
    // threshold, factor 1) leave uint8 images unchanged, but 8 bits quantises a floating image to 256 levels.
    typedef struct random_posterize_option {
        int64_t _bits = 8;
        double _p = 0.5;
        auto bits(int64_t v) -> random_posterize_option & {
            _bits = v;
            return *this;
        }
        auto p(double v) -> random_posterize_option & {
            _p = v;
            return *this;
        }
    } random_posterize_option_t;

    class RandomPosterize : public detail::p_gated<detail::posterize_op> {
    public:
        explicit RandomPosterize(random_posterize_option opt) : p_gated("RandomPosterize", opt._p, {opt._bits}) {}
    };

    typedef struct random_solarize_option {
        double _threshold = std::numeric_limits<double>::quiet_NaN();
        double _p = 0.5;
        auto threshold(double v) -> random_solarize_option & {
            _threshold = v;
            return *this;
        }
        auto p(double v) -> random_solarize_option & {
            _p = v;
            return *this;
        }
    } random_solarize_option_t;

    class RandomSolarize : public detail::p_gated<detail::solarize_op> {
    public:
        explicit RandomSolarize(random_solarize_option opt) : p_gated("RandomSolarize", opt._p, {opt._threshold}) {}
    };

    typedef struct random_adjust_sharpness_option {
        double _sharpness_factor = 1.0;
        double _p = 0.5;
        auto sharpness_factor(double v) -> random_adjust_sharpness_option & {
            _sharpness_factor = v;
            return *this;
        }
        auto p(double v) -> random_adjust_sharpness_option & {
            _p = v;
            return *this;
        }
    } random_adjust_sharpness_option_t;

    class RandomAdjustSharpness : public detail::p_gated<detail::sharpness_op> {
    public:
        explicit RandomAdjustSharpness(random_adjust_sharpness_option opt) :
            p_gated("RandomAdjustSharpness", opt._p, {opt._sharpness_factor}) {}
    };


    // ---------------- RGB ----------------
    // A 1-channel image repeated to 3 channels; >= 3 channels returned as is.
    class RGB {
    public:
        auto operator()(const_tensor_lref_t img) const -> tensor_t { return functional::grayscale_to_rgb(img); }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }
    };

    namespace detail {
        // ColorJitter / RandomPhotometricDistort op k: 0 brightness, 1 contrast, 2 saturation, 3 hue.
        inline auto adjust_color(int64_t k, const tensor_t &img, double factor) -> tensor_t {
            switch (k) {
                case 0:
                    return functional::adjust_brightness(img, factor);
                case 1:
                    return functional::adjust_contrast(img, factor);
                case 2:
                    return functional::adjust_saturation(img, factor);
                default:
                    return functional::adjust_hue(img, factor);
            }
        }

        inline auto randperm(int64_t n) -> std::vector<int64_t> {
            const auto p = torch::randperm(n);
            return {p.data_ptr<int64_t>(), p.data_ptr<int64_t>() + p.numel()};
        }
    } // namespace detail

    // ---------------- ColorJitter ----------------
    typedef struct color_jitter_option {
        std::vector<double> _brightness; // {} = None, {x} = [max(0, 1 - x), 1 + x], or {min, max} (>= 0)
        std::vector<double> _contrast; // as brightness
        std::vector<double> _saturation; // as brightness
        std::vector<double> _hue; // {} = None, {x} = [-x, x], or {min, max}, within [-0.5, 0.5]

        auto brightness(std::vector<double> v) -> color_jitter_option & {
            _brightness = std::move(v);
            return *this;
        }
        auto contrast(std::vector<double> v) -> color_jitter_option & {
            _contrast = std::move(v);
            return *this;
        }
        auto saturation(std::vector<double> v) -> color_jitter_option & {
            _saturation = std::move(v);
            return *this;
        }
        auto hue(std::vector<double> v) -> color_jitter_option & {
            _hue = std::move(v);
            return *this;
        }
    } color_jitter_option_t;

    // Every call draws torch.randperm(4) (the op order), then one uniform factor per configured op in the order
    // brightness, contrast, saturation, hue, and applies the configured ops in the permuted order. A range equal to
    // the identity ({1, 1}, hue {0, 0}) is None, as in torchvision.
    class ColorJitter {
    public:
        explicit ColorJitter(color_jitter_option opt = {}) :
            _range{check_input("brightness", opt._brightness, 1.0, 0.0, kInf, true),
                   check_input("contrast", opt._contrast, 1.0, 0.0, kInf, true),
                   check_input("saturation", opt._saturation, 1.0, 0.0, kInf, true),
                   check_input("hue", opt._hue, 0.0, -0.5, 0.5, false)} {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto order = detail::randperm(4);
            std::array<std::optional<double>, 4> factor;
            for (size_t k = 0; k < 4; ++k) {
                if (_range[k]) {
                    factor[k] = detail::uniform((*_range[k])[0], (*_range[k])[1]);
                }
            }
            auto out = img;
            for (const auto k: order) {
                if (factor[k]) {
                    out = detail::adjust_color(k, out, *factor[k]);
                }
            }
            return out;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        static constexpr double kInf = std::numeric_limits<double>::infinity();
        using range_t = std::optional<std::array<double, 2>>;

        // torchvision ColorJitter._check_input
        static auto check_input(const char *name, const std::vector<double> &v, double center, double lo, double hi,
                                bool clip_first_on_zero) -> range_t {
            if (v.empty()) {
                return std::nullopt;
            }
            if (v.size() > 2) {
                return handle_exceptions<range_t, std::invalid_argument>(
                        std::format("ColorJitter: {} should be a single number or a sequence with length 2.", name));
            }
            if (v.size() == 1 && v[0] < 0) {
                return handle_exceptions<range_t, std::invalid_argument>(
                        std::format("ColorJitter: if {} is a single number, it must be non negative.", name));
            }
            std::array<double, 2> r{v[0], v.back()};
            if (v.size() == 1) {
                r = {center - v[0], center + v[0]};
                if (clip_first_on_zero) {
                    r[0] = std::max(r[0], 0.0);
                }
            }
            if (!(lo <= r[0] && r[0] <= r[1] && r[1] <= hi)) {
                return handle_exceptions<range_t, std::invalid_argument>(
                        std::format("ColorJitter: {} values should be between ({}, {}), but got [{}, {}].", name, lo,
                                    hi, r[0], r[1]));
            }
            return r[0] == r[1] && r[1] == center ? std::nullopt : range_t{r};
        }

        std::array<range_t, 4> _range;
    };

    // ---------------- RandomPhotometricDistort ----------------
    typedef struct random_photometric_distort_option {
        std::array<double, 2> _brightness{0.875, 1.125};
        std::array<double, 2> _contrast{0.5, 1.5};
        std::array<double, 2> _saturation{0.5, 1.5};
        std::array<double, 2> _hue{-0.05, 0.05};
        double _p = 0.5;

        auto brightness(std::array<double, 2> v) -> random_photometric_distort_option & {
            _brightness = v;
            return *this;
        }
        auto contrast(std::array<double, 2> v) -> random_photometric_distort_option & {
            _contrast = v;
            return *this;
        }
        auto saturation(std::array<double, 2> v) -> random_photometric_distort_option & {
            _saturation = v;
            return *this;
        }
        auto hue(std::array<double, 2> v) -> random_photometric_distort_option & {
            _hue = v;
            return *this;
        }
        auto p(double v) -> random_photometric_distort_option & {
            _p = v;
            return *this;
        }
    } random_photometric_distort_option_t;

    // SSD's photometric distortion: brightness, contrast (before or after the next two, a coin flip), saturation and
    // hue, each with probability p, then a channel permutation with probability p. torchvision validates nothing here.
    class RandomPhotometricDistort {
    public:
        explicit RandomPhotometricDistort(random_photometric_distort_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto c = detail::query_chw("RandomPhotometricDistort", img)[0];
            const std::array ranges{_opt._brightness, _opt._contrast, _opt._saturation, _opt._hue};
            std::array<std::optional<double>, 4> factor;
            for (size_t k = 0; k < 4; ++k) {
                if (detail::rand_lt(_opt._p)) {
                    factor[k] = detail::uniform(ranges[k][0], ranges[k][1]);
                }
            }
            const bool contrast_before = detail::rand_lt(0.5);
            const auto permutation = detail::rand_lt(_opt._p) ? detail::randperm(c) : std::vector<int64_t>{};
            auto out = img;
            const auto apply = [&](int64_t k) {
                if (factor[k]) {
                    out = detail::adjust_color(k, out, *factor[k]);
                }
            };
            apply(0);
            if (contrast_before) {
                apply(1);
            }
            apply(2);
            apply(3);
            if (!contrast_before) {
                apply(1);
            }
            return permutation.empty() ? out : functional::permute_channels(out, permutation);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_photometric_distort_option _opt;
    };

    // ---------------- RandomChannelPermutation ----------------
    class RandomChannelPermutation {
    public:
        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto c = detail::query_chw("RandomChannelPermutation", img)[0];
            return functional::permute_channels(img, detail::randperm(c));
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }
    };
} // namespace torchmedia::vision::transform
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_COLOR_HPP
