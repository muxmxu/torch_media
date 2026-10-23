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

        // torchvision v2 _RandomApplyTransform: one torch.rand(1) draw per call; flip unless draw >= p.
        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return (torch::rand({1}) >= _p).item<bool>() ? img : functional::hflip(img);
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
            return (torch::rand({1}) >= _p).item<bool>() ? img : functional::vflip(img);
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

} // namespace torchmedia::vision::transform
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_HPP
