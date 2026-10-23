#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
#define LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
// Transform classes (torchvision 0.20.1 transforms.v2 parity) over the vision functional layer: the constructor
// validates like the torchvision class, operator() / forward() call the functional op. They hold no tensors, so the
// output follows the input's device and dtype. The random flips draw torch::rand({1}) from the global CPU generator
// exactly like torchvision v2, so a torch::manual_seed gives the same flips. See develop_log/2026-10-23/progress07.
#include "_functional.hpp"

namespace torchmedia::vision::transform {
    namespace detail {
        inline auto check_size(const char *cls, const std::vector<int64_t> &size) -> void {
            if (size.empty() || size.size() > 2) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        std::format("{}: please provide only two dimensions (h, w) for size.", cls));
            }
        }

        inline auto check_p(const char *cls, double p) -> void {
            if (!(p >= 0.0 && p <= 1.0)) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        std::format("{}: p should be a floating point value in the interval [0.0, 1.0].", cls));
            }
        }

        // torchvision v2 _RandomApplyTransform: one torch.rand(1) draw per call; the transform is skipped if draw >= p.
        inline auto skip(double p) -> bool { return (torch::rand({1}) >= p).item<bool>(); }
    } // namespace detail

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

    // ---------------- Normalize ----------------
    typedef struct normalize_option {
        std::vector<double> _mean;
        std::vector<double> _std;
        bool _inplace = false;

        auto mean(std::vector<double> v) -> normalize_option & {
            _mean = std::move(v);
            return *this;
        }
        auto std(std::vector<double> v) -> normalize_option & {
            _std = std::move(v);
            return *this;
        }
        auto inplace(bool b) -> normalize_option & {
            _inplace = b;
            return *this;
        }
    } normalize_option_t;

    class Normalize {
    public:
        explicit Normalize(normalize_option opt) : _opt(std::move(opt)) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::normalize(img, _opt._mean, _opt._std, _opt._inplace);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        normalize_option _opt;
    };

    // ---------------- ToDtype ----------------
    typedef struct to_dtype_option {
        torch::Dtype _dtype = torch::kFloat32;
        bool _scale = false;

        auto dtype(torch::Dtype d) -> to_dtype_option & {
            _dtype = d;
            return *this;
        }
        auto scale(bool b) -> to_dtype_option & {
            _scale = b;
            return *this;
        }
    } to_dtype_option_t;

    class ToDtype {
    public:
        explicit ToDtype(to_dtype_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::to_dtype(img, _opt._dtype, _opt._scale);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        to_dtype_option _opt;
    };

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

    // ---------------- RandomHorizontalFlip / RandomVerticalFlip ----------------
    typedef struct random_flip_option {
        double _p = 0.5;

        auto p(double v) -> random_flip_option & {
            _p = v;
            return *this;
        }
    } random_flip_option_t;

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

    // ---------------- GaussianBlur ----------------
    typedef struct gaussian_blur_option {
        std::vector<int64_t> _kernel_size; // {k} or {kx, ky}, odd and positive
        std::vector<double> _sigma{0.1, 2.0}; // {s} or {min, max}
        auto kernel_size(std::vector<int64_t> v) -> gaussian_blur_option & {
            _kernel_size = std::move(v);
            return *this;
        }
        auto sigma(std::vector<double> v) -> gaussian_blur_option & {
            _sigma = std::move(v);
            return *this;
        }
    } gaussian_blur_option_t;

    class GaussianBlur {
    public:
        explicit GaussianBlur(gaussian_blur_option opt) : _opt(std::move(opt)) {
            auto &k = _opt._kernel_size;
            auto &s = _opt._sigma;
            if (k.size() == 1) {
                k.push_back(k[0]);
            }
            if (s.size() == 1) {
                s.push_back(s[0]);
            }
            if (k.size() != 2 || k[0] <= 0 || k[1] <= 0 || k[0] % 2 == 0 || k[1] % 2 == 0) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "GaussianBlur: kernel size should be one or two odd and positive numbers.");
            }
            if (s.size() != 2 || !(0.0 < s[0] && s[0] <= s[1])) {
                s.assign(2, 0.0); // a no-exceptions build then fails in functional::gaussian_blur, not on _sigma[1]
                (void) handle_exceptions<int, std::invalid_argument>(
                        "GaussianBlur: sigma values should be positive and of the form (min, max).");
            }
        }

        // torchvision v2 draws sigma = torch.empty(1).uniform_(min, max) on every call, even when min == max.
        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto sigma = torch::empty({1}).uniform_(_opt._sigma[0], _opt._sigma[1]).item<double>();
            return functional::gaussian_blur(img, _opt._kernel_size, {sigma, sigma});
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        gaussian_blur_option _opt;
    };

    // ---------------- p-gated color transforms ----------------
    // Random{Invert, Autocontrast, Equalize, Grayscale, Posterize, Solarize, AdjustSharpness}: torchvision v2
    // _RandomApplyTransform subclasses, one torch.rand(1) draw per call, the op runs unless draw >= p.
    using random_apply_option = random_flip_option; // the flips are p-gated the same way: one option struct
    using random_apply_option_t = random_flip_option;

    namespace detail {
        // A p-gated functional op; Op is a callable (tensor) -> tensor.
        template<class Op>
        class RandomApply {
        public:
            RandomApply(const char *cls, double p, Op op) : _p(p), _op(std::move(op)) { check_p(cls, p); }
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

    class RandomInvert : public detail::RandomApply<detail::invert_op> {
    public:
        explicit RandomInvert(random_apply_option opt = {}) : RandomApply("RandomInvert", opt._p, {}) {}
    };

    class RandomAutocontrast : public detail::RandomApply<detail::autocontrast_op> {
    public:
        explicit RandomAutocontrast(random_apply_option opt = {}) : RandomApply("RandomAutocontrast", opt._p, {}) {}
    };

    class RandomEqualize : public detail::RandomApply<detail::equalize_op> {
    public:
        explicit RandomEqualize(random_apply_option opt = {}) : RandomApply("RandomEqualize", opt._p, {}) {}
    };

    // Default p = 0.1 (torchvision).
    class RandomGrayscale : public detail::RandomApply<detail::grayscale_op> {
    public:
        explicit RandomGrayscale(random_apply_option opt = random_apply_option().p(0.1)) :
            RandomApply("RandomGrayscale", opt._p, {}) {}
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

    class RandomPosterize : public detail::RandomApply<detail::posterize_op> {
    public:
        explicit RandomPosterize(random_posterize_option opt) : RandomApply("RandomPosterize", opt._p, {opt._bits}) {}
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

    class RandomSolarize : public detail::RandomApply<detail::solarize_op> {
    public:
        explicit RandomSolarize(random_solarize_option opt) : RandomApply("RandomSolarize", opt._p, {opt._threshold}) {}
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

    class RandomAdjustSharpness : public detail::RandomApply<detail::sharpness_op> {
    public:
        explicit RandomAdjustSharpness(random_adjust_sharpness_option opt) :
            RandomApply("RandomAdjustSharpness", opt._p, {opt._sharpness_factor}) {}
    };

} // namespace torchmedia::vision::transform
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
