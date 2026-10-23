#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_TRANSFORM_MISC_HPP
#define LIB_TORCH_MEDIA_VISION_TRANSFORM_MISC_HPP
// Transform classes (torchvision 0.20.1 transforms.v2 parity) over the vision functional layer: the constructor
// validates like the torchvision class, operator() / forward() call the functional op. Random classes draw from the
// global CPU generator with the same torch calls, in the same order, as torchvision v2's _get_params, so a
// torch::manual_seed gives the same outputs as torch.manual_seed (develop_log/2026-10-23/progress07, progress09,
// progress10). Instances are const and safe to share across threads, but concurrent calls interleave their draws on
// the global generator: seeded parity needs the calls serialized (one thread). This header: the shared helpers,
// torchvision's _misc / _augment / _container classes.
#include <functional>
#include <numeric>
#include "_functional.hpp"

namespace torchmedia::vision::transform {
    namespace detail {
        inline auto fail(std::string msg) -> void {
            (void) handle_exceptions<int, std::invalid_argument>(std::move(msg));
        }

        inline auto check_size(const char *cls, const std::vector<int64_t> &size) -> void {
            if (size.empty() || size.size() > 2) {
                fail(std::format("{}: please provide only two dimensions (h, w) for size.", cls));
            }
        }

        // torchvision _setup_size: {s} or {h, w} -> {h, w} ({0, 0} after a failed check in a no-exceptions build).
        inline auto setup_size(const char *cls, const std::vector<int64_t> &size) -> std::array<int64_t, 2> {
            check_size(cls, size);
            return size.empty() || size.size() > 2 ? std::array<int64_t, 2>{} : std::array{size[0], size.back()};
        }

        inline auto check_p(const char *cls, double p) -> void {
            if (!(p >= 0.0 && p <= 1.0)) {
                fail(std::format("{}: p should be a floating point value in the interval [0.0, 1.0].", cls));
            }
        }

        // torchvision v2 _RandomApplyTransform: one torch.rand(1) draw per call; the transform is skipped if draw >= p.
        inline auto skip(double p) -> bool { return (torch::rand({1}) >= p).item<bool>(); }

        // torch.rand(1) < p, compared in float32 like the tensor comparison in torchvision.
        inline auto rand_lt(double p) -> bool { return (torch::rand({1}) < p).item<bool>(); }

        // torch.empty(1).uniform_(lo, hi).item() (float32 draw).
        inline auto uniform(double lo, double hi) -> double {
            return torch::empty({1}).uniform_(lo, hi).item<double>();
        }

        // torch.randint(lo, hi, size=(1,)).item()
        inline auto randint(int64_t lo, int64_t hi) -> int64_t { return torch::randint(lo, hi, {1}).item<int64_t>(); }

        // Python int(round(x)): half to even.
        inline auto py_round(double x) -> int64_t { return static_cast<int64_t>(std::nearbyint(x)); }

        // torch.log(torch.tensor(ratio)): float32, as RandomResizedCrop / RandomErasing precompute it.
        inline auto log_ratio(const std::array<double, 2> &ratio) -> std::array<double, 2> {
            const auto l = torch::log(torch::tensor({ratio[0], ratio[1]}, torch::kFloat64).to(torch::kFloat32));
            return {l[0].item<double>(), l[1].item<double>()};
        }

        // torchvision query_chw on one image [..., H, W] (a 2-D image has one channel).
        inline auto query_chw(const char *cls, const tensor_t &img) -> std::array<int64_t, 3> {
            if (img.dim() < 2) {
                return handle_exceptions<std::array<int64_t, 3>, std::invalid_argument>(std::format(
                        "{}: expected a tensor with at least 2 dims (..., H, W), got {} dims.", cls, img.dim()));
            }
            return {img.dim() < 3 ? 1 : img.size(-3), img.size(-2), img.size(-1)};
        }
    } // namespace detail

    // The p-only option of the p-gated classes (flips, RandomInvert, ...).
    typedef struct random_flip_option {
        double _p = 0.5;

        auto p(double v) -> random_flip_option & {
            _p = v;
            return *this;
        }
    } random_flip_option_t;
    // For the p-gated colour classes (RandomInvert, ...); the RandomApply container takes p directly.
    using random_apply_option = random_flip_option;
    using random_apply_option_t = random_flip_option;

    // ---------------- containers (torchvision Compose / RandomApply / RandomChoice / RandomOrder) ----------------
    // Any transform class or callable (const tensor_t &) -> tensor_t.
    using transform_fn = std::function<tensor_t(const_tensor_lref_t)>;

    class Compose {
    public:
        explicit Compose(std::vector<transform_fn> transforms) : _transforms(std::move(transforms)) {
            if (_transforms.empty()) {
                detail::fail("Compose: pass at least one transform.");
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            auto out = img;
            for (const auto &t: _transforms) {
                out = t(out);
            }
            return out;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        std::vector<transform_fn> _transforms;
    };

    // All transforms in order with probability p (one torch.rand(1) draw per call), else the input itself. An empty
    // list returns the input (torchvision raises UnboundLocalError when it applies one).
    class RandomApply {
    public:
        explicit RandomApply(std::vector<transform_fn> transforms, double p = 0.5) :
            _transforms(std::move(transforms)), _p(p) {
            detail::check_p("RandomApply", p);
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (detail::skip(_p)) {
                return img;
            }
            auto out = img;
            for (const auto &t: _transforms) {
                out = t(out);
            }
            return out;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        std::vector<transform_fn> _transforms;
        double _p;
    };

    // One transform picked by torch.multinomial(p normalised to sum 1, as float32); p {} = equal weights.
    class RandomChoice {
    public:
        explicit RandomChoice(std::vector<transform_fn> transforms, std::vector<double> p = {}) :
            _transforms(std::move(transforms)), _p(std::move(p)) {
            if (_p.empty()) {
                _p.assign(_transforms.size(), 1.0);
            } else if (_p.size() != _transforms.size()) {
                const auto msg =
                        std::format("RandomChoice: length of p doesn't match the number of transforms: {} != {}",
                                    _p.size(), _transforms.size());
                _p.assign(_transforms.size(), 1.0);
                detail::fail(msg);
            }
            const double total = std::accumulate(_p.begin(), _p.end(), 0.0);
            if (total == 0.0 && !_p.empty()) {
                detail::fail("RandomChoice: p sums to zero."); // torchvision: ZeroDivisionError
            }
            for (auto &v: _p) {
                v /= total;
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto p = torch::tensor(_p, torch::kFloat64).to(torch::kFloat32);
            return _transforms[torch::multinomial(p, 1).item<int64_t>()](img);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        std::vector<transform_fn> _transforms;
        std::vector<double> _p;
    };

    // All transforms in the order of torch.randperm(n). An empty list returns the input (torchvision raises).
    class RandomOrder {
    public:
        explicit RandomOrder(std::vector<transform_fn> transforms) : _transforms(std::move(transforms)) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto perm = torch::randperm(static_cast<int64_t>(_transforms.size()));
            auto out = img;
            for (int64_t i = 0; i < perm.numel(); ++i) {
                out = _transforms[perm[i].item<int64_t>()](out);
            }
            return out;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        std::vector<transform_fn> _transforms;
    };

    // ---------------- Identity ----------------
    class Identity {
    public:
        auto operator()(const_tensor_lref_t img) const -> tensor_t { return img; }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return img; }
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

    // ---------------- ToDtype / ConvertImageDtype ----------------
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

    typedef struct convert_image_dtype_option {
        torch::Dtype _dtype = torch::kFloat32;

        auto dtype(torch::Dtype d) -> convert_image_dtype_option & {
            _dtype = d;
            return *this;
        }
    } convert_image_dtype_option_t;

    // to_dtype with scale = true (torchvision's v1-compatible ConvertImageDtype).
    class ConvertImageDtype {
    public:
        explicit ConvertImageDtype(convert_image_dtype_option opt = {}) : _dtype(opt._dtype) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t { return functional::to_dtype(img, _dtype, true); }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        torch::Dtype _dtype;
    };

    // ---------------- LinearTransformation ----------------
    typedef struct linear_transformation_option {
        tensor_t _transformation_matrix; // [D, D], D = C * H * W
        tensor_t _mean_vector; // [D], same device and dtype

        auto transformation_matrix(tensor_t m) -> linear_transformation_option & {
            _transformation_matrix = std::move(m);
            return *this;
        }
        auto mean_vector(tensor_t v) -> linear_transformation_option & {
            _mean_vector = std::move(v);
            return *this;
        }
    } linear_transformation_option_t;

    // (img.reshape(-1, D) - mean_vector) @ transformation_matrix, reshaped back (e.g. ZCA whitening).
    class LinearTransformation {
    public:
        explicit LinearTransformation(linear_transformation_option opt) : _opt(std::move(opt)) {
            const auto &m = _opt._transformation_matrix;
            const auto &v = _opt._mean_vector;
            if (m.size(0) != m.size(1)) {
                detail::fail(std::format("LinearTransformation: transformation_matrix should be square, got [{}, {}] "
                                         "rectangular matrix.",
                                         m.size(0), m.size(1)));
            } else if (v.size(0) != m.size(0)) {
                detail::fail(std::format("LinearTransformation: mean_vector should have the same length {} as any one "
                                         "of the dimensions of the transformation_matrix [{}, {}].",
                                         v.size(0), m.size(0), m.size(1)));
            } else if (m.device() != v.device()) {
                detail::fail("LinearTransformation: input tensors should be on the same device.");
            } else if (m.scalar_type() != v.scalar_type()) {
                detail::fail("LinearTransformation: input tensors should have the same dtype.");
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto &m = _opt._transformation_matrix;
            const auto n = img.size(-3) * img.size(-2) * img.size(-1);
            if (n != m.size(0)) {
                return functional::detail::invalid(std::format(
                        "LinearTransformation: input tensor and transformation matrix have incompatible shape. "
                        "[{} x {} x {}] != {}",
                        img.size(-3), img.size(-2), img.size(-1), m.size(0)));
            }
            if (img.device().type() != _opt._mean_vector.device().type()) {
                return functional::detail::invalid("LinearTransformation: input tensor should be on the same device "
                                                   "as transformation matrix and mean vector.");
            }
            const auto flat = img.reshape({-1, n}) - _opt._mean_vector;
            return torch::mm(flat, m.to(flat.scalar_type())).reshape(img.sizes());
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        linear_transformation_option _opt;
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
                detail::fail("GaussianBlur: kernel size should be one or two odd and positive numbers.");
            }
            if (s.size() != 2 || !(0.0 < s[0] && s[0] <= s[1])) {
                s.assign(2, 0.0); // a no-exceptions build then fails in functional::gaussian_blur, not on _sigma[1]
                detail::fail("GaussianBlur: sigma values should be positive and of the form (min, max).");
            }
        }

        // torchvision v2 draws sigma = torch.empty(1).uniform_(min, max) on every call, even when min == max.
        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            const auto sigma = detail::uniform(_opt._sigma[0], _opt._sigma[1]);
            return functional::gaussian_blur(img, _opt._kernel_size, {sigma, sigma});
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        gaussian_blur_option _opt;
    };

    // ---------------- GaussianNoise ----------------
    typedef struct gaussian_noise_option {
        double _mean = 0.0;
        double _sigma = 0.1;
        bool _clip = true;

        auto mean(double v) -> gaussian_noise_option & {
            _mean = v;
            return *this;
        }
        auto sigma(double v) -> gaussian_noise_option & {
            _sigma = v;
            return *this;
        }
        auto clip(bool b) -> gaussian_noise_option & {
            _clip = b;
            return *this;
        }
    } gaussian_noise_option_t;

    // Adds mean + sigma * randn_like(img) (drawn from the image device's generator); floating images only.
    class GaussianNoise {
    public:
        explicit GaussianNoise(gaussian_noise_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            return functional::gaussian_noise(img, _opt._mean, _opt._sigma, _opt._clip);
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        gaussian_noise_option _opt;
    };

    // ---------------- RandomErasing ----------------
    typedef struct random_erasing_option {
        double _p = 0.5;
        std::array<double, 2> _scale{0.02, 0.33}; // erased area / image area
        std::array<double, 2> _ratio{0.3, 3.3}; // aspect ratio range
        std::vector<double> _value{0.0}; // one value, one per channel, or {} = torchvision's 'random' (N(0, 1))
        bool _inplace = false;

        auto p(double v) -> random_erasing_option & {
            _p = v;
            return *this;
        }
        auto scale(std::array<double, 2> v) -> random_erasing_option & {
            _scale = v;
            return *this;
        }
        auto ratio(std::array<double, 2> v) -> random_erasing_option & {
            _ratio = v;
            return *this;
        }
        auto value(std::vector<double> v) -> random_erasing_option & {
            _value = std::move(v);
            return *this;
        }
        auto inplace(bool b) -> random_erasing_option & {
            _inplace = b;
            return *this;
        }
    } random_erasing_option_t;

    // With probability p, erase a random rectangle (10 attempts at a random area and aspect ratio that fits strictly
    // inside the image; none fits: the input is returned).
    class RandomErasing {
    public:
        explicit RandomErasing(random_erasing_option opt = {}) :
            _opt(std::move(opt)), _log_ratio(detail::log_ratio(_opt._ratio)) {
            detail::check_p("RandomErasing", _opt._p);
            if (_opt._scale[0] < 0 || _opt._scale[1] > 1) {
                detail::fail("RandomErasing: scale should be between 0 and 1.");
            }
        }

        auto operator()(const_tensor_lref_t img) const -> tensor_t {
            if (detail::skip(_opt._p)) {
                return img;
            }
            const auto [c, h, w] = detail::query_chw("RandomErasing", img);
            const auto &value = _opt._value;
            if (!value.empty() && value.size() != 1 && static_cast<int64_t>(value.size()) != c) {
                return functional::detail::invalid(
                        std::format("RandomErasing: if value is a sequence, it should have either a single value or {} "
                                    "(number of input channels).",
                                    c));
            }
            const auto area = static_cast<double>(h * w);
            for (int attempt = 0; attempt < 10; ++attempt) {
                const double erase_area = area * detail::uniform(_opt._scale[0], _opt._scale[1]);
                const double aspect_ratio =
                        torch::exp(torch::empty({1}).uniform_(_log_ratio[0], _log_ratio[1])).item<double>();
                const auto eh = detail::py_round(std::sqrt(erase_area * aspect_ratio));
                const auto ew = detail::py_round(std::sqrt(erase_area / aspect_ratio));
                if (!(eh < h && ew < w)) {
                    continue;
                }
                const auto v = value.empty()
                                       ? torch::empty({c, eh, ew}, torch::kFloat32).normal_()
                                       : torch::tensor(value, torch::kFloat64).to(torch::kFloat32).view({-1, 1, 1});
                const auto i = detail::randint(0, h - eh + 1);
                const auto j = detail::randint(0, w - ew + 1);
                return functional::erase(img, i, j, eh, ew, v, _opt._inplace);
            }
            return img;
        }
        auto forward(const_tensor_lref_t img) const -> tensor_t { return (*this)(img); }

    private:
        random_erasing_option _opt;
        std::array<double, 2> _log_ratio;
    };

} // namespace torchmedia::vision::transform
#endif // LIB_TORCH_MEDIA_VISION_TRANSFORM_MISC_HPP
