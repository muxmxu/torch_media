#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_IMAGE_HPP
#define LIB_TORCH_MEDIA_METRIC_IMAGE_HPP
// Image quality metrics (develop_log/2026-10-23/progress11): torchmetrics 1.9 torchmetrics.functional.image and
// torchmetrics.image, the same libtorch calls in the same order (Python's `scalar / tensor` is
// tensor.reciprocal() * scalar), so results match point-wise in the input dtype. Classes accumulate their state in the
// input's dtype and device (torchmetrics' float32 states round float64 inputs; develop_log/2026-10-23/progress11 D5).
// Model-based metrics (LPIPS, FID, KID, MiFID, DISTS, PPL, ARNIQA) and the pansharpening indices (D_lambda, D_s, QNR,
// PSNRB) are out of scope.
#include "_common.hpp"

namespace torchmedia::metric::image {
    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;
        using metric::detail::reduce;
        using metric::detail::shape_error;
        using metric::detail::squeeze_if_scalar;

        inline auto dtype_error(const char *fn, const tensor_t &preds, const tensor_t &target) -> std::string {
            if (preds.scalar_type() == target.scalar_type()) {
                return shape_error(fn, preds, target);
            }
            return std::format("{}: expected `preds` and `target` to have the same data type, got {} and {}.", fn,
                               c10::toString(preds.scalar_type()), c10::toString(target.scalar_type()));
        }

        inline auto bchw_error(const char *fn, const tensor_t &preds) -> std::string {
            return preds.dim() == 4 ? std::string{}
                                    : std::format("{}: expected `preds` and `target` to have BxCxHxW shape, got {}.",
                                                  fn, c10::str(preds.sizes()));
        }

        // Python's round(window_size / 2) (half to even).
        inline auto half_window(int64_t window_size) -> int64_t {
            return static_cast<int64_t>(std::nearbyint(static_cast<double>(window_size) / 2));
        }

        // torchmetrics' `state += value` on a float32 zero state, kept in value's dtype / device.
        inline auto accumulate(const tensor_t &state, const tensor_t &value) -> tensor_t { return state + value; }
        inline auto zero_state() -> tensor_t { return torch::tensor(0.0, torch::kFloat32); }

        // `torch.cat` of the stored batches (torchmetrics dim_zero_cat).
        inline auto cat(const std::vector<tensor_t> &xs) -> tensor_t {
            std::vector<tensor_t> v;
            for (const auto &x: xs) {
                v.push_back(x.dim() == 0 ? x.unsqueeze(0) : x);
            }
            return torch::cat(v, 0);
        }
    } // namespace detail

    // ---------------- PeakSignalNoiseRatio ----------------
    typedef struct peak_signal_noise_ratio_option {
        std::vector<double> _data_range; // required: {range} or {min, max} (inputs clamped to it)
        double _base = 10.0;
        reduction_mode _reduction = reduction_mode::elementwise_mean; // over the per-`dim` values
        std::vector<int64_t> _dim; // {} = None: one value over all elements

        auto data_range(std::vector<double> v) -> peak_signal_noise_ratio_option & {
            _data_range = std::move(v);
            return *this;
        }
        auto base(double v) -> peak_signal_noise_ratio_option & {
            _base = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> peak_signal_noise_ratio_option & {
            _reduction = v;
            return *this;
        }
        auto dim(std::vector<int64_t> v) -> peak_signal_noise_ratio_option & {
            _dim = std::move(v);
            return *this;
        }
    } peak_signal_noise_ratio_option_t;

    namespace detail {
        inline auto psnr_error(const peak_signal_noise_ratio_option &opt) -> std::string {
            return opt._data_range.size() == 1 || opt._data_range.size() == 2
                           ? std::string{}
                           : "peak_signal_noise_ratio: data_range must be {range} or {min, max}.";
        }

        // torchmetrics' data_range tensor: float32 on the CPU.
        inline auto psnr_data_range(const peak_signal_noise_ratio_option &opt) -> tensor_t {
            const auto &r = opt._data_range;
            return torch::tensor(r.size() == 2 ? r[1] - r[0] : r[0], torch::kFloat32);
        }

        inline auto psnr_clamp(const tensor_t &x, const peak_signal_noise_ratio_option &opt) -> tensor_t {
            const auto &r = opt._data_range;
            return r.size() == 2 ? torch::clamp(x, r[0], r[1]) : x;
        }

        // _psnr_update: {sum of squared errors, number of observations}
        inline auto psnr_update(const tensor_t &preds, const tensor_t &target, const std::vector<int64_t> &dim)
                -> std::pair<tensor_t, tensor_t> {
            const auto p = preds.is_floating_point() ? preds : preds.to(torch::kFloat32);
            const auto t = target.is_floating_point() ? target : target.to(torch::kFloat32);
            const auto opt = torch::TensorOptions().dtype(torch::kInt64).device(t.device());
            if (dim.empty()) {
                return {torch::sum(torch::pow(p - t, 2)), torch::tensor(t.numel(), opt)};
            }
            const auto diff = p - t;
            const auto sse = torch::sum(diff * diff, dim);
            int64_t n = 1;
            for (const auto d: dim) {
                n *= t.size(d);
            }
            return {sse, torch::tensor(n, opt).expand_as(sse)};
        }

        inline auto psnr_compute(const tensor_t &sse, const tensor_t &num_obs, const tensor_t &data_range, double base,
                                 reduction_mode r) -> tensor_t {
            const auto psnr_base_e = 2 * torch::log(data_range) - torch::log(sse / num_obs);
            return reduce(psnr_base_e * (torch::log(torch::tensor(base, torch::kFloat32)).reciprocal() * 10), r);
        }
    } // namespace detail

    // torchmetrics peak_signal_noise_ratio: 10 / log(base) * (2 log(range) - log(MSE)); integer inputs as float32.
    inline auto peak_signal_noise_ratio(const tensor_t &preds, const tensor_t &target,
                                        const peak_signal_noise_ratio_option &opt) -> tensor_t {
        if (auto e = detail::psnr_error(opt); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        const auto [sse, n] =
                detail::psnr_update(detail::psnr_clamp(preds, opt), detail::psnr_clamp(target, opt), opt._dim);
        return detail::psnr_compute(sse, n, detail::psnr_data_range(opt), opt._base, opt._reduction);
    }

    class PeakSignalNoiseRatio : public metric::detail::metric_base<PeakSignalNoiseRatio> {
    public:
        explicit PeakSignalNoiseRatio(peak_signal_noise_ratio_option opt) : _opt(std::move(opt)) {
            if (auto e = detail::psnr_error(_opt); !e.empty()) {
                _opt._data_range = {1.0};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            const auto [sse, n] =
                    detail::psnr_update(detail::psnr_clamp(preds, _opt), detail::psnr_clamp(target, _opt), _opt._dim);
            if (_opt._dim.empty()) {
                _sse = detail::accumulate(_sse, sse);
                _total = detail::accumulate(_total, n);
            } else {
                _sse_list.push_back(sse.flatten());
                _total_list.push_back(n.flatten());
            }
        }

        auto compute() const -> tensor_t {
            const bool all = _opt._dim.empty();
            return detail::squeeze_if_scalar(
                    detail::psnr_compute(all ? _sse : torch::cat(_sse_list), all ? _total : torch::cat(_total_list),
                                         detail::psnr_data_range(_opt), _opt._base, _opt._reduction));
        }

        auto options() const -> const peak_signal_noise_ratio_option & { return _opt; }

    private:
        peak_signal_noise_ratio_option _opt;
        tensor_t _sse = detail::zero_state(), _total = torch::tensor(0, torch::kInt64);
        std::vector<tensor_t> _sse_list, _total_list;
    };

    // ---------------- StructuralSimilarityIndexMeasure / MultiScaleStructuralSimilarityIndexMeasure ----------------
    typedef struct structural_similarity_index_measure_option {
        bool _gaussian_kernel = true; // false: uniform kernel of kernel_size
        std::vector<double> _sigma{
                1.5}; // one value or one per spatial dim; the gaussian's size is 2 int(3.5 s + .5) + 1
        std::vector<int64_t> _kernel_size{11}; // one value or one per spatial dim (odd, > 0)
        reduction_mode _reduction = reduction_mode::elementwise_mean; // over the per-image values
        std::vector<double> _data_range; // {} = None (max - min of each input), {range} or {min, max} (clamped)
        double _k1 = 0.01;
        double _k2 = 0.03;
        bool _return_full_image = false; // the image of structural_similarity_index_measure_with_image
        bool _return_contrast_sensitivity = false; // ... or the per-image contrast sensitivity

        auto gaussian_kernel(bool v) -> structural_similarity_index_measure_option & {
            _gaussian_kernel = v;
            return *this;
        }
        auto sigma(std::vector<double> v) -> structural_similarity_index_measure_option & {
            _sigma = std::move(v);
            return *this;
        }
        auto kernel_size(std::vector<int64_t> v) -> structural_similarity_index_measure_option & {
            _kernel_size = std::move(v);
            return *this;
        }
        auto reduction(reduction_mode v) -> structural_similarity_index_measure_option & {
            _reduction = v;
            return *this;
        }
        auto data_range(std::vector<double> v) -> structural_similarity_index_measure_option & {
            _data_range = std::move(v);
            return *this;
        }
        auto k1(double v) -> structural_similarity_index_measure_option & {
            _k1 = v;
            return *this;
        }
        auto k2(double v) -> structural_similarity_index_measure_option & {
            _k2 = v;
            return *this;
        }
        auto return_full_image(bool v) -> structural_similarity_index_measure_option & {
            _return_full_image = v;
            return *this;
        }
        auto return_contrast_sensitivity(bool v) -> structural_similarity_index_measure_option & {
            _return_contrast_sensitivity = v;
            return *this;
        }
    } structural_similarity_index_measure_option_t;

    enum class normalize_mode { none, relu, simple };

    typedef struct multiscale_structural_similarity_index_measure_option {
        bool _gaussian_kernel = true;
        std::vector<int64_t> _kernel_size{11};
        std::vector<double> _sigma{1.5};
        reduction_mode _reduction = reduction_mode::elementwise_mean;
        std::vector<double> _data_range;
        double _k1 = 0.01;
        double _k2 = 0.03;
        std::vector<double> _betas{0.0448, 0.2856, 0.3001, 0.2363, 0.1333}; // one per scale
        normalize_mode _normalize = normalize_mode::relu;

        auto gaussian_kernel(bool v) -> multiscale_structural_similarity_index_measure_option & {
            _gaussian_kernel = v;
            return *this;
        }
        auto kernel_size(std::vector<int64_t> v) -> multiscale_structural_similarity_index_measure_option & {
            _kernel_size = std::move(v);
            return *this;
        }
        auto sigma(std::vector<double> v) -> multiscale_structural_similarity_index_measure_option & {
            _sigma = std::move(v);
            return *this;
        }
        auto reduction(reduction_mode v) -> multiscale_structural_similarity_index_measure_option & {
            _reduction = v;
            return *this;
        }
        auto data_range(std::vector<double> v) -> multiscale_structural_similarity_index_measure_option & {
            _data_range = std::move(v);
            return *this;
        }
        auto k1(double v) -> multiscale_structural_similarity_index_measure_option & {
            _k1 = v;
            return *this;
        }
        auto k2(double v) -> multiscale_structural_similarity_index_measure_option & {
            _k2 = v;
            return *this;
        }
        auto betas(std::vector<double> v) -> multiscale_structural_similarity_index_measure_option & {
            _betas = std::move(v);
            return *this;
        }
        auto normalize(normalize_mode v) -> multiscale_structural_similarity_index_measure_option & {
            _normalize = v;
            return *this;
        }
    } multiscale_structural_similarity_index_measure_option_t;

    namespace detail {
        // The SSIM parameters shared by both forms (kernel_size / sigma already expanded to the spatial dims).
        struct ssim_params {
            bool gaussian_kernel;
            std::vector<double> sigma;
            std::vector<int64_t> kernel_size;
            std::vector<double> data_range;
            double k1, k2;
        };

        enum class ssim_image { none, full, contrast_sensitivity };

        template<class Opt>
        inline auto make_ssim_params(const Opt &opt, int64_t ndim) -> ssim_params {
            const auto n = static_cast<size_t>(ndim == 5 ? 3 : 2);
            return {opt._gaussian_kernel,
                    opt._sigma.size() == 1 ? std::vector<double>(n, opt._sigma[0]) : opt._sigma,
                    opt._kernel_size.size() == 1 ? std::vector<int64_t>(n, opt._kernel_size[0]) : opt._kernel_size,
                    opt._data_range,
                    opt._k1,
                    opt._k2};
        }

        // _ssim_check_inputs (target cast to preds' dtype by the caller).
        inline auto ssim_input_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            if (auto e = shape_error("structural_similarity_index_measure", preds, target); !e.empty()) {
                return e;
            }
            return preds.dim() == 4 || preds.dim() == 5
                           ? std::string{}
                           : std::format("Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape. Got {}.",
                                         c10::str(preds.sizes()));
        }

        // _ssim_update's argument checks.
        inline auto ssim_param_error(const ssim_params &p, int64_t ndim, bool full, bool cs) -> std::string {
            const auto n = static_cast<size_t>(ndim - 2);
            if (p.kernel_size.size() != n || p.sigma.size() != n) {
                return std::format("`kernel_size` and `sigma` need one value or {} values (got {} and {}).", n,
                                   p.kernel_size.size(), p.sigma.size());
            }
            if (full && cs) {
                return "Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive.";
            }
            if (std::any_of(p.kernel_size.begin(), p.kernel_size.end(),
                            [](int64_t k) { return k % 2 == 0 || k <= 0; })) {
                return "Expected `kernel_size` to have odd positive number.";
            }
            if (std::any_of(p.sigma.begin(), p.sigma.end(), [](double s) { return s <= 0; })) {
                return "Expected `sigma` to have positive number.";
            }
            return p.data_range.size() > 2 ? "data_range must be {}, {range} or {min, max}." : std::string{};
        }

        // torchmetrics.functional.image.utils._gaussian: [1, k]
        inline auto gaussian(int64_t k, double sigma, const torch::TensorOptions &opt) -> tensor_t {
            const auto dist =
                    torch::arange((1.0 - static_cast<double>(k)) / 2, (1.0 + static_cast<double>(k)) / 2, 1, opt);
            const auto gauss = torch::exp(-torch::pow(dist / sigma, 2) / 2);
            return (gauss / gauss.sum()).unsqueeze(0);
        }

        // _ssim_update (checked): {per-image SSIM [B], the requested image or undefined}
        inline auto ssim_update(tensor_t preds, tensor_t target, const ssim_params &p, ssim_image want)
                -> std::pair<tensor_t, tensor_t> {
            TORCHMEDIA_FP_CONTRACT_OFF
            const bool is_3d = preds.dim() == 5;
            tensor_t c1t, c2t; // data_range None: tensors in preds' dtype
            double c1 = 0, c2 = 0;
            if (p.data_range.empty()) {
                const auto a = preds.max() - preds.min();
                const auto b = target.max() - target.min();
                const auto dr = (b > a).item<bool>() ? b : a;
                c1t = torch::pow(p.k1 * dr, 2);
                c2t = torch::pow(p.k2 * dr, 2);
            } else {
                double dr = p.data_range[0];
                if (p.data_range.size() == 2) {
                    preds = torch::clamp(preds, p.data_range[0], p.data_range[1]);
                    target = torch::clamp(target, p.data_range[0], p.data_range[1]);
                    dr = p.data_range[1] - p.data_range[0];
                }
                c1 = std::pow(p.k1 * dr, 2);
                c2 = std::pow(p.k2 * dr, 2);
            }
            const auto plus_c = [](const tensor_t &x, const tensor_t &ct, double cd) {
                return ct.defined() ? x + ct : x + cd;
            };
            const auto opt = preds.options();
            const auto channel = preds.size(1);
            std::vector<int64_t> gks;
            for (const auto s: p.sigma) {
                gks.push_back(static_cast<int64_t>(3.5 * s + 0.5) * 2 + 1);
            }
            const auto &ks = p.gaussian_kernel ? gks : p.kernel_size;
            const int64_t pad_h = (ks[0] - 1) / 2, pad_w = (ks[1] - 1) / 2;
            int64_t pad_d = 0;
            tensor_t kernel;
            if (p.gaussian_kernel) {
                kernel = torch::matmul(gaussian(gks[0], p.sigma[0], opt).t(), gaussian(gks[1], p.sigma[1], opt));
            }
            if (is_3d) {
                pad_d = (p.kernel_size[2] - 1) / 2;
                preds = torch::pad(preds, {pad_d, pad_d, pad_w, pad_w, pad_h, pad_h}, "reflect");
                target = torch::pad(target, {pad_d, pad_d, pad_w, pad_w, pad_h, pad_h}, "reflect");
                if (p.gaussian_kernel) {
                    kernel = torch::mul(kernel.unsqueeze(-1).repeat({1, 1, gks[2]}),
                                        gaussian(gks[2], p.sigma[2], opt).expand({gks[0], gks[1], gks[2]}));
                }
            } else {
                preds = torch::pad(preds, {pad_w, pad_w, pad_h, pad_h}, "reflect");
                target = torch::pad(target, {pad_w, pad_w, pad_h, pad_h}, "reflect");
            }
            std::vector<int64_t> shape{channel, 1};
            if (p.gaussian_kernel) {
                shape.insert(shape.end(), kernel.sizes().begin(), kernel.sizes().end());
                kernel = kernel.expand(shape);
            } else {
                shape.insert(shape.end(), p.kernel_size.begin(), p.kernel_size.end());
                kernel = torch::ones(shape, opt) /
                         torch::prod(torch::tensor(p.kernel_size, opt.dtype(torch::kInt64)).to(opt.dtype()));
            }
            const auto input_list =
                    torch::cat({preds, target, preds * preds, target * target, preds * target}); // (5 B, C, ...)
            const auto outputs = is_3d ? torch::conv3d(input_list, kernel, {}, 1, int64_t{0}, 1, channel)
                                       : torch::conv2d(input_list, kernel, {}, 1, int64_t{0}, 1, channel);
            const auto o = outputs.split(preds.size(0));
            const auto dtype = preds.scalar_type();
            const auto mu_pred_sq = o[0].pow(2);
            const auto mu_target_sq = o[1].pow(2);
            const auto mu_pred_target = o[0] * o[1];
            const auto sigma_pred_sq = torch::clamp(o[2] - mu_pred_sq, 0.0);
            const auto sigma_target_sq = torch::clamp(o[3] - mu_target_sq, 0.0);
            const auto sigma_pred_target = o[4] - mu_pred_target;
            const auto upper = plus_c(2 * sigma_pred_target.to(dtype), c2t, c2);
            const auto lower = plus_c((sigma_pred_sq + sigma_target_sq).to(dtype), c2t, c2);
            const auto ssim_full = (plus_c(2 * mu_pred_target, c1t, c1) * upper) /
                                   (plus_c(mu_pred_sq + mu_target_sq, c1t, c1) * lower);
            const auto b = ssim_full.size(0);
            const auto sim = ssim_full.reshape({b, -1}).mean(-1);
            if (want == ssim_image::contrast_sensitivity) {
                auto cs = upper / lower;
                if (is_3d) {
                    cs = cs.slice(-3, pad_h, -pad_h).slice(-2, pad_w, -pad_w).slice(-1, pad_d, -pad_d);
                } else {
                    cs = cs.slice(-2, pad_h, -pad_h).slice(-1, pad_w, -pad_w);
                }
                return {sim, cs.reshape({cs.size(0), -1}).mean(-1)};
            }
            return {sim, want == ssim_image::full ? ssim_full : tensor_t{}};
        }

        inline auto ssim_want(const structural_similarity_index_measure_option &opt) -> ssim_image {
            return opt._return_full_image             ? ssim_image::full
                   : opt._return_contrast_sensitivity ? ssim_image::contrast_sensitivity
                                                      : ssim_image::none;
        }

        // Checked SSIM of one batch: {per-image SSIM, image}; an error leaves both undefined.
        inline auto ssim_batch(const tensor_t &preds, const tensor_t &target,
                               const structural_similarity_index_measure_option &opt, std::string &error)
                -> std::pair<tensor_t, tensor_t> {
            error = ssim_input_error(preds, target);
            if (!error.empty()) {
                return {};
            }
            const auto p = make_ssim_params(opt, preds.dim());
            error = ssim_param_error(p, preds.dim(), opt._return_full_image, opt._return_contrast_sensitivity);
            if (!error.empty()) {
                return {};
            }
            return ssim_update(preds, target.to(preds.scalar_type()), p, ssim_want(opt));
        }
    } // namespace detail

    // torchmetrics structural_similarity_index_measure over [B, C, H, W] or [B, C, D, H, W] (target cast to preds'
    // dtype): the reduced per-image SSIM. The _with_image form also returns the full SSIM map (return_full_image) or
    // the per-image contrast sensitivity (return_contrast_sensitivity), torchmetrics' tuple.
    inline auto structural_similarity_index_measure(const tensor_t &preds, const tensor_t &target,
                                                    const structural_similarity_index_measure_option &opt = {})
            -> tensor_t {
        std::string e;
        const auto sim = detail::ssim_batch(preds, target, opt, e).first;
        return e.empty() ? detail::reduce(sim, opt._reduction) : detail::invalid(std::move(e));
    }

    inline auto structural_similarity_index_measure_with_image(const tensor_t &preds, const tensor_t &target,
                                                               const structural_similarity_index_measure_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        if (detail::ssim_want(opt) == detail::ssim_image::none) {
            return detail::invalid<std::pair<tensor_t, tensor_t>>(
                    "structural_similarity_index_measure_with_image: set return_full_image or "
                    "return_contrast_sensitivity.");
        }
        std::string e;
        const auto [sim, img] = detail::ssim_batch(preds, target, opt, e);
        if (!e.empty()) {
            return detail::invalid<std::pair<tensor_t, tensor_t>>(std::move(e));
        }
        return {detail::reduce(sim, opt._reduction), img};
    }

    class StructuralSimilarityIndexMeasure : public metric::detail::metric_base<StructuralSimilarityIndexMeasure> {
    public:
        explicit StructuralSimilarityIndexMeasure(structural_similarity_index_measure_option opt = {}) :
            _opt(std::move(opt)) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            std::string e;
            const auto [sim, img] = detail::ssim_batch(preds, target, _opt, e);
            if (!e.empty()) {
                return detail::fail(std::move(e));
            }
            if (img.defined()) {
                _images.push_back(img);
            }
            if (_opt._reduction == reduction_mode::none) {
                _similarity_list.push_back(sim);
            } else {
                _similarity = detail::accumulate(_similarity, sim.sum());
                _total += preds.size(0);
            }
        }

        // The accumulated SSIM: mean / sum over all images, or all per-image values (reduction none).
        auto compute() const -> tensor_t {
            switch (_opt._reduction) {
                case reduction_mode::elementwise_mean:
                    return detail::squeeze_if_scalar(_similarity / static_cast<double>(_total));
                case reduction_mode::sum:
                    return detail::squeeze_if_scalar(_similarity);
                default:
                    return detail::squeeze_if_scalar(detail::cat(_similarity_list));
            }
        }

        // torchmetrics' (similarity, images) with return_full_image / return_contrast_sensitivity.
        auto compute_with_image() const -> std::pair<tensor_t, tensor_t> {
            if (detail::ssim_want(_opt) == detail::ssim_image::none) {
                return detail::invalid<std::pair<tensor_t, tensor_t>>(
                        "StructuralSimilarityIndexMeasure: set return_full_image or return_contrast_sensitivity.");
            }
            return {compute(), detail::squeeze_if_scalar(detail::cat(_images))};
        }

        auto options() const -> const structural_similarity_index_measure_option & { return _opt; }

    private:
        structural_similarity_index_measure_option _opt;
        tensor_t _similarity = detail::zero_state();
        int64_t _total = 0;
        std::vector<tensor_t> _similarity_list, _images;
    };

    namespace detail {
        inline auto ms_ssim_option_error(const multiscale_structural_similarity_index_measure_option &opt)
                -> std::string {
            if (opt._kernel_size.empty() || opt._kernel_size.size() > 3) {
                return "Argument `kernel_size` expected to be a sequence of size 2 or 3, or a single int.";
            }
            return opt._betas.empty() ? "Argument `betas` needs at least one value." : std::string{};
        }

        // _multiscale_ssim_update with its checks (an error leaves the result undefined).
        inline auto ms_ssim_batch(const tensor_t &preds, const tensor_t &target,
                                  const multiscale_structural_similarity_index_measure_option &opt, std::string &error)
                -> tensor_t {
            error = ms_ssim_option_error(opt);
            if (error.empty()) {
                error = ssim_input_error(preds, target);
            }
            if (!error.empty()) {
                return {};
            }
            const auto p = make_ssim_params(opt, preds.dim());
            const auto nb = static_cast<int64_t>(opt._betas.size());
            const auto betas_div = std::max<int64_t>(1, nb - 1) * std::max<int64_t>(1, nb - 1);
            if (preds.size(-1) < (int64_t{1} << nb) || preds.size(-2) < (int64_t{1} << nb)) {
                error = std::format("For {} `betas`, the image height and width must be larger than or equal to {}.",
                                    nb, int64_t{1} << nb);
            } else if (preds.size(-2) / betas_div <= p.kernel_size[0] - 1) {
                error = std::format("For {} `betas` and kernel size {}, the image height must be larger than {}.", nb,
                                    p.kernel_size[0], (p.kernel_size[0] - 1) * betas_div);
            } else if (preds.size(-1) / betas_div <= p.kernel_size[1] - 1) {
                error = std::format("For {} `betas` and kernel size {}, the image width must be larger than {}.", nb,
                                    p.kernel_size[1], (p.kernel_size[1] - 1) * betas_div);
            } else {
                error = ssim_param_error(p, preds.dim(), false, true);
            }
            if (!error.empty()) {
                return {};
            }
            auto x = preds;
            auto y = target.to(preds.scalar_type());
            std::vector<tensor_t> mcs;
            tensor_t sim;
            for (int64_t s = 0; s < nb; ++s) {
                auto [si, cs] = ssim_update(x, y, p, ssim_image::contrast_sensitivity);
                if (opt._normalize == normalize_mode::relu) {
                    si = torch::relu(si);
                    cs = torch::relu(cs);
                }
                sim = si;
                mcs.push_back(cs);
                x = preds.dim() == 4 ? torch::avg_pool2d(x, {2, 2}) : torch::avg_pool3d(x, {2, 2, 2});
                y = preds.dim() == 4 ? torch::avg_pool2d(y, {2, 2}) : torch::avg_pool3d(y, {2, 2, 2});
            }
            mcs.back() = sim;
            auto stack = torch::stack(mcs);
            if (opt._normalize == normalize_mode::simple) {
                stack = (stack + 1) / 2;
            }
            const auto betas = torch::tensor(opt._betas, torch::TensorOptions().dtype(torch::kFloat32))
                                       .to(stack.device())
                                       .view({-1, 1});
            return torch::prod(torch::pow(stack, betas), 0);
        }
    } // namespace detail

    // torchmetrics multiscale_structural_similarity_index_measure (normalize relu by default).
    inline auto multiscale_structural_similarity_index_measure(
            const tensor_t &preds, const tensor_t &target,
            const multiscale_structural_similarity_index_measure_option &opt = {}) -> tensor_t {
        std::string e;
        const auto v = detail::ms_ssim_batch(preds, target, opt, e);
        return e.empty() ? detail::reduce(v, opt._reduction) : detail::invalid(std::move(e));
    }

    class MultiScaleStructuralSimilarityIndexMeasure
        : public metric::detail::metric_base<MultiScaleStructuralSimilarityIndexMeasure> {
    public:
        explicit MultiScaleStructuralSimilarityIndexMeasure(
                multiscale_structural_similarity_index_measure_option opt = {}) : _opt(std::move(opt)) {
            if (auto e = detail::ms_ssim_option_error(_opt); !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            std::string e;
            const auto v = detail::ms_ssim_batch(preds, target, _opt, e);
            if (!e.empty()) {
                return detail::fail(std::move(e));
            }
            if (_opt._reduction == reduction_mode::none) {
                _similarity_list.push_back(v);
            } else {
                _similarity = detail::accumulate(_similarity, v.sum());
            }
            _total += preds.size(0);
        }

        auto compute() const -> tensor_t {
            switch (_opt._reduction) {
                case reduction_mode::elementwise_mean:
                    return detail::squeeze_if_scalar(_similarity / static_cast<double>(_total));
                case reduction_mode::sum:
                    return detail::squeeze_if_scalar(_similarity);
                default:
                    return detail::squeeze_if_scalar(detail::cat(_similarity_list));
            }
        }

        auto options() const -> const multiscale_structural_similarity_index_measure_option & { return _opt; }

    private:
        multiscale_structural_similarity_index_measure_option _opt;
        tensor_t _similarity = detail::zero_state();
        int64_t _total = 0;
        std::vector<tensor_t> _similarity_list;
    };

    // ---------------- UniversalImageQualityIndex ----------------
    typedef struct universal_image_quality_index_option {
        std::array<int64_t, 2> _kernel_size{11, 11}; // odd, > 0
        std::array<double, 2> _sigma{1.5, 1.5};
        reduction_mode _reduction = reduction_mode::elementwise_mean;

        auto kernel_size(std::array<int64_t, 2> v) -> universal_image_quality_index_option & {
            _kernel_size = v;
            return *this;
        }
        auto sigma(std::array<double, 2> v) -> universal_image_quality_index_option & {
            _sigma = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> universal_image_quality_index_option & {
            _reduction = v;
            return *this;
        }
    } universal_image_quality_index_option_t;

    namespace detail {
        inline auto uqi_error(const tensor_t &preds, const tensor_t &target,
                              const universal_image_quality_index_option &opt) -> std::string {
            for (auto e: {dtype_error("universal_image_quality_index", preds, target),
                          bchw_error("universal_image_quality_index", preds)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            if (opt._kernel_size[0] % 2 == 0 || opt._kernel_size[0] <= 0 || opt._kernel_size[1] % 2 == 0 ||
                opt._kernel_size[1] <= 0) {
                return "Expected `kernel_size` to have odd positive number.";
            }
            return opt._sigma[0] <= 0 || opt._sigma[1] <= 0 ? "Expected `sigma` to have positive number."
                                                            : std::string{};
        }

        // _uqi_compute (checked), not yet reduced.
        inline auto uqi_map(const tensor_t &preds_in, const tensor_t &target_in,
                            const universal_image_quality_index_option &opt) -> tensor_t {
            const auto o = preds_in.options();
            const auto channel = preds_in.size(1);
            const auto &ks = opt._kernel_size;
            const auto kernel = torch::matmul(gaussian(ks[0], opt._sigma[0], o).t(), gaussian(ks[1], opt._sigma[1], o))
                                        .expand({channel, 1, ks[0], ks[1]});
            const int64_t pad_h = (ks[0] - 1) / 2, pad_w = (ks[1] - 1) / 2;
            // torchmetrics pads the width by pad_h and the height by pad_w
            const auto preds = torch::pad(preds_in, {pad_h, pad_h, pad_w, pad_w}, "reflect");
            const auto target = torch::pad(target_in, {pad_h, pad_h, pad_w, pad_w}, "reflect");
            const auto input_list = torch::cat({preds, target, preds * preds, target * target, preds * target});
            const auto out = torch::conv2d(input_list, kernel, {}, 1, int64_t{0}, 1, channel).split(preds.size(0));
            const auto mu_pred_sq = out[0].pow(2);
            const auto mu_target_sq = out[1].pow(2);
            const auto mu_pred_target = out[0] * out[1];
            const auto sigma_pred_sq = torch::clamp(out[2] - mu_pred_sq, 0.0);
            const auto sigma_target_sq = torch::clamp(out[3] - mu_target_sq, 0.0);
            const auto sigma_pred_target = out[4] - mu_pred_target;
            const auto upper = 2 * sigma_pred_target;
            const auto lower = sigma_pred_sq + sigma_target_sq;
            const auto dt = sigma_pred_sq.scalar_type();
            const double eps = dt == torch::kFloat64    ? 0x1p-52
                               : dt == torch::kFloat16  ? 0x1p-10
                               : dt == torch::kBFloat16 ? 0x1p-7
                                                        : 0x1p-23; // torch.finfo(dtype).eps
            const auto uqi = ((2 * mu_pred_target) * upper) / ((mu_pred_sq + mu_target_sq) * lower + eps);
            return uqi.slice(-2, pad_h, -pad_h).slice(-1, pad_w, -pad_w);
        }
    } // namespace detail

    // torchmetrics universal_image_quality_index over [B, C, H, W] (same dtype and shape).
    inline auto universal_image_quality_index(const tensor_t &preds, const tensor_t &target,
                                              const universal_image_quality_index_option &opt = {}) -> tensor_t {
        if (auto e = detail::uqi_error(preds, target, opt); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::reduce(detail::uqi_map(preds, target, opt), opt._reduction);
    }

    class UniversalImageQualityIndex : public metric::detail::metric_base<UniversalImageQualityIndex> {
    public:
        explicit UniversalImageQualityIndex(universal_image_quality_index_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::uqi_error(preds, target, _opt); !e.empty()) {
                return detail::fail(std::move(e));
            }
            if (_opt._reduction == reduction_mode::none) {
                _preds.push_back(preds);
                _target.push_back(target);
                return;
            }
            _sum = detail::accumulate(_sum, torch::sum(detail::uqi_map(preds, target, _opt)));
            _numel += preds.size(0) * preds.size(1) * (preds.size(2) - _opt._kernel_size[0] + 1) *
                      (preds.size(3) - _opt._kernel_size[1] + 1);
        }

        auto compute() const -> tensor_t {
            if (_opt._reduction == reduction_mode::none) {
                return detail::squeeze_if_scalar(detail::uqi_map(detail::cat(_preds), detail::cat(_target), _opt));
            }
            return detail::squeeze_if_scalar(
                    _opt._reduction == reduction_mode::sum ? _sum : _sum / static_cast<double>(_numel));
        }

        auto options() const -> const universal_image_quality_index_option & { return _opt; }

    private:
        universal_image_quality_index_option _opt;
        tensor_t _sum = detail::zero_state();
        int64_t _numel = 0;
        std::vector<tensor_t> _preds, _target;
    };

    // ---------------- TotalVariation ----------------
    namespace detail {
        // _total_variation_update: per-image sum of absolute neighbour differences.
        inline auto tv_score(const tensor_t &img) -> tensor_t {
            const auto diff1 = img.slice(-2, 1) - img.slice(-2, 0, -1);
            const auto diff2 = img.slice(-1, 1) - img.slice(-1, 0, -1);
            return diff1.abs().sum({1, 2, 3}) + diff2.abs().sum({1, 2, 3});
        }

        inline auto tv_reduce(const tensor_t &score, int64_t n, reduction_mode r) -> tensor_t {
            return r == reduction_mode::elementwise_mean ? score.sum() / n
                   : r == reduction_mode::sum            ? score.sum()
                                                         : score;
        }

        inline auto tv_error(const tensor_t &img) -> std::string {
            return img.dim() == 4
                           ? std::string{}
                           : std::format("Expected input `img` to be an 4D tensor, but got {}", c10::str(img.sizes()));
        }
    } // namespace detail

    // torchmetrics total_variation of [B, C, H, W] (reduction sum by default; elementwise_mean is torchmetrics' "mean":
    // the sum over images / B). Integer images subtract in their dtype, like torchmetrics.
    inline auto total_variation(const tensor_t &img, reduction_mode reduction = reduction_mode::sum) -> tensor_t {
        if (auto e = detail::tv_error(img); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::tv_reduce(detail::tv_score(img), img.size(0), reduction);
    }

    typedef struct total_variation_option {
        reduction_mode _reduction = reduction_mode::sum;

        auto reduction(reduction_mode v) -> total_variation_option & {
            _reduction = v;
            return *this;
        }
    } total_variation_option_t;

    class TotalVariation : public metric::detail::metric_base<TotalVariation> {
    public:
        explicit TotalVariation(total_variation_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &img) -> void {
            if (auto e = detail::tv_error(img); !e.empty()) {
                return detail::fail(std::move(e));
            }
            const auto score = detail::tv_score(img);
            if (_opt._reduction == reduction_mode::none) {
                _scores.push_back(score);
            } else {
                _score = detail::accumulate(_score, score.sum());
            }
            _num += img.size(0);
        }

        auto compute() const -> tensor_t {
            return detail::squeeze_if_scalar(detail::tv_reduce(
                    _opt._reduction == reduction_mode::none ? detail::cat(_scores) : _score, _num, _opt._reduction));
        }

        auto options() const -> const total_variation_option & { return _opt; }

    private:
        total_variation_option _opt;
        tensor_t _score = detail::zero_state();
        int64_t _num = 0;
        std::vector<tensor_t> _scores;
    };

    // ---------------- SpectralAngleMapper ----------------
    namespace detail {
        inline auto sam_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            for (auto e:
                 {dtype_error("spectral_angle_mapper", preds, target), bchw_error("spectral_angle_mapper", preds)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            return preds.size(1) > 1 ? std::string{}
                                     : "spectral_angle_mapper: expected more than one channel in `preds` and `target`.";
        }

        inline auto sam_map(const tensor_t &preds, const tensor_t &target) -> tensor_t {
            const auto dot_product = (preds * target).sum(1);
            // Tensor.norm(dim=1) is linalg.vector_norm(x, 2, [1]) on the CPU
            const auto preds_norm = torch::linalg_vector_norm(preds, 2, at::IntArrayRef{1}, false, std::nullopt);
            const auto target_norm = torch::linalg_vector_norm(target, 2, at::IntArrayRef{1}, false, std::nullopt);
            return torch::clamp(dot_product / (preds_norm * target_norm), -1, 1).acos();
        }
    } // namespace detail

    // torchmetrics spectral_angle_mapper: per-pixel angle (radians) between the channel vectors, [B, C > 1, H, W].
    inline auto spectral_angle_mapper(const tensor_t &preds, const tensor_t &target,
                                      reduction_mode reduction = reduction_mode::elementwise_mean) -> tensor_t {
        if (auto e = detail::sam_error(preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::reduce(detail::sam_map(preds, target), reduction);
    }

    typedef struct spectral_angle_mapper_option {
        reduction_mode _reduction = reduction_mode::elementwise_mean;

        auto reduction(reduction_mode v) -> spectral_angle_mapper_option & {
            _reduction = v;
            return *this;
        }
    } spectral_angle_mapper_option_t;

    class SpectralAngleMapper : public metric::detail::metric_base<SpectralAngleMapper> {
    public:
        explicit SpectralAngleMapper(spectral_angle_mapper_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::sam_error(preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            if (_opt._reduction == reduction_mode::none) {
                _preds.push_back(preds);
                _target.push_back(target);
                return;
            }
            _sum = detail::accumulate(_sum, torch::sum(detail::sam_map(preds, target)));
            _numel += preds.size(0) * preds.size(2) * preds.size(3);
        }

        auto compute() const -> tensor_t {
            if (_opt._reduction == reduction_mode::none) {
                return detail::squeeze_if_scalar(detail::sam_map(detail::cat(_preds), detail::cat(_target)));
            }
            return detail::squeeze_if_scalar(
                    _opt._reduction == reduction_mode::sum ? _sum : _sum / static_cast<double>(_numel));
        }

        auto options() const -> const spectral_angle_mapper_option & { return _opt; }

    private:
        spectral_angle_mapper_option _opt;
        tensor_t _sum = detail::zero_state();
        int64_t _numel = 0;
        std::vector<tensor_t> _preds, _target;
    };

    // ---------------- ErrorRelativeGlobalDimensionlessSynthesis ----------------
    typedef struct error_relative_global_dimensionless_synthesis_option {
        double _ratio = 4; // high / low resolution pixel size ratio
        reduction_mode _reduction = reduction_mode::elementwise_mean;

        auto ratio(double v) -> error_relative_global_dimensionless_synthesis_option & {
            _ratio = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> error_relative_global_dimensionless_synthesis_option & {
            _reduction = v;
            return *this;
        }
    } error_relative_global_dimensionless_synthesis_option_t;

    namespace detail {
        inline auto ergas_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            auto e = dtype_error("error_relative_global_dimensionless_synthesis", preds, target);
            return e.empty() ? bchw_error("error_relative_global_dimensionless_synthesis", preds) : e;
        }

        inline auto ergas_compute(const tensor_t &preds_in, const tensor_t &target_in, double ratio, reduction_mode r)
                -> tensor_t {
            const auto b = preds_in.size(0), c = preds_in.size(1), hw = preds_in.size(2) * preds_in.size(3);
            const auto preds = preds_in.reshape({b, c, hw});
            const auto target = target_in.reshape({b, c, hw});
            const auto diff = preds - target;
            const auto sum_squared_error = torch::sum(diff * diff, 2);
            const auto rmse_per_band = torch::sqrt(sum_squared_error / hw);
            const auto mean_target = torch::mean(target, 2);
            const auto ergas = 100 / ratio * torch::sqrt(torch::sum(torch::pow(rmse_per_band / mean_target, 2), 1) / c);
            return reduce(ergas, r);
        }
    } // namespace detail

    // torchmetrics error_relative_global_dimensionless_synthesis over [B, C, H, W].
    inline auto
    error_relative_global_dimensionless_synthesis(const tensor_t &preds, const tensor_t &target,
                                                  const error_relative_global_dimensionless_synthesis_option &opt = {})
            -> tensor_t {
        if (auto e = detail::ergas_error(preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::ergas_compute(preds, target, opt._ratio, opt._reduction);
    }

    // Keeps every batch (torchmetrics' list states) and evaluates their concatenation.
    class ErrorRelativeGlobalDimensionlessSynthesis
        : public metric::detail::metric_base<ErrorRelativeGlobalDimensionlessSynthesis> {
    public:
        explicit ErrorRelativeGlobalDimensionlessSynthesis(
                error_relative_global_dimensionless_synthesis_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::ergas_error(preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _preds.push_back(preds);
            _target.push_back(target);
        }

        auto compute() const -> tensor_t {
            return detail::squeeze_if_scalar(
                    detail::ergas_compute(detail::cat(_preds), detail::cat(_target), _opt._ratio, _opt._reduction));
        }

        auto options() const -> const error_relative_global_dimensionless_synthesis_option & { return _opt; }

    private:
        error_relative_global_dimensionless_synthesis_option _opt;
        std::vector<tensor_t> _preds, _target;
    };

    // ---------------- RootMeanSquaredErrorUsingSlidingWindow / RelativeAverageSpectralError ----------------
    namespace detail {
        // torchmetrics _uniform_filter: window mean over a symmetric (edge-including) pad of window // 2.
        inline auto uniform_filter(const tensor_t &inputs, int64_t window_size) -> tensor_t {
            const int64_t pad = window_size / 2, outer_pad = window_size % 2;
            auto x = inputs;
            for (const int64_t dim: {2, 3}) {
                const auto n = x.size(dim);
                const auto idx = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
                const auto lo = x.index_select(dim, torch::arange(pad - 1, -1, -1, idx));
                const auto hi = x.index_select(dim, torch::arange(n - 1, n - pad - outer_pad, -1, idx));
                x = torch::cat({lo, x, hi}, dim);
            }
            auto weight = torch::ones({1, 1, window_size, window_size}, x.options());
            weight /= window_size * window_size;
            const auto bias = torch::zeros({1}, x.options());
            std::vector<tensor_t> channels;
            for (int64_t c = 0; c < x.size(1); ++c) {
                channels.push_back(torch::conv2d(x.select(1, c).unsqueeze(1), weight, bias));
            }
            return torch::cat(channels, 1);
        }

        inline auto window_error(int64_t window_size) -> std::string {
            return window_size >= 1 ? std::string{} : "Argument `window_size` is expected to be a positive integer.";
        }

        inline auto rmse_sw_error(const tensor_t &preds, const tensor_t &target, int64_t window_size) -> std::string {
            for (auto e:
                 {window_error(window_size), dtype_error("root_mean_squared_error_using_sliding_window", preds, target),
                  bchw_error("root_mean_squared_error_using_sliding_window", preds)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            const auto half = half_window(window_size);
            return half < target.size(2) && half < target.size(3)
                           ? std::string{}
                           : std::format("Parameter `round(window_size / 2)` is expected to be smaller than {} but "
                                         "got {}.",
                                         std::min(target.size(2), target.size(3)), half);
        }

        // _rmse_sw_update for one batch: {sum over images of the cropped per-image map's mean, summed rmse map}.
        inline auto rmse_sw_batch(const tensor_t &preds, const tensor_t &target, int64_t window_size)
                -> std::pair<tensor_t, tensor_t> {
            const auto rmse_map = torch::sqrt(uniform_filter(torch::pow(target - preds, 2), window_size));
            const auto crop = half_window(window_size);
            return {rmse_map.slice(2, crop, -crop).slice(3, crop, -crop).sum(0).mean(), rmse_map.sum(0)};
        }
    } // namespace detail

    // torchmetrics root_mean_squared_error_using_sliding_window over [B, C, H, W]; the _with_map form also returns the
    // per-pixel RMSE map averaged over the images (torchmetrics' return_rmse_map).
    inline auto root_mean_squared_error_using_sliding_window_with_map(const tensor_t &preds, const tensor_t &target,
                                                                      int64_t window_size = 8)
            -> std::pair<tensor_t, tensor_t> {
        if (auto e = detail::rmse_sw_error(preds, target, window_size); !e.empty()) {
            return detail::invalid<std::pair<tensor_t, tensor_t>>(std::move(e));
        }
        const auto [val, map] = detail::rmse_sw_batch(preds, target, window_size);
        const auto total =
                torch::tensor(target.size(0), torch::TensorOptions().dtype(torch::kInt64).device(target.device()));
        return {val / total, map / total};
    }

    inline auto root_mean_squared_error_using_sliding_window(const tensor_t &preds, const tensor_t &target,
                                                             int64_t window_size = 8) -> tensor_t {
        return root_mean_squared_error_using_sliding_window_with_map(preds, target, window_size).first;
    }

    typedef struct sliding_window_option {
        int64_t _window_size = 8;

        auto window_size(int64_t v) -> sliding_window_option & {
            _window_size = v;
            return *this;
        }
    } sliding_window_option_t;

    class RootMeanSquaredErrorUsingSlidingWindow
        : public metric::detail::metric_base<RootMeanSquaredErrorUsingSlidingWindow> {
    public:
        explicit RootMeanSquaredErrorUsingSlidingWindow(sliding_window_option opt = {}) : _opt(opt) {
            if (auto e = detail::window_error(_opt._window_size); !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::rmse_sw_error(preds, target, _opt._window_size); !e.empty()) {
                return detail::fail(std::move(e));
            }
            const auto [val, map] = detail::rmse_sw_batch(preds, target, _opt._window_size);
            _val_sum = detail::accumulate(_val_sum, val);
            _total += target.size(0);
        }

        auto compute() const -> tensor_t { return detail::squeeze_if_scalar(_val_sum / static_cast<double>(_total)); }

        auto options() const -> const sliding_window_option & { return _opt; }

    private:
        sliding_window_option _opt;
        tensor_t _val_sum = detail::zero_state();
        int64_t _total = 0;
    };

    // torchmetrics relative_average_spectral_error over [B, C, H, W] (window_size >= 1). Kept torchmetrics quirk: the
    // target's window mean is divided by window_size^2 a second time, so RASE scales with window_size^2.
    inline auto relative_average_spectral_error(const tensor_t &preds, const tensor_t &target, int64_t window_size = 8)
            -> tensor_t {
        if (auto e = detail::rmse_sw_error(preds, target, window_size); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        const auto total = torch::tensor(static_cast<double>(target.size(0)),
                                         torch::TensorOptions().dtype(torch::kFloat32).device(target.device()));
        const auto rmse_map = detail::rmse_sw_batch(preds, target, window_size).second / total;
        const auto target_sum =
                torch::sum(detail::uniform_filter(target, window_size) / (window_size * window_size), 0);
        const auto target_mean = (target_sum / total).mean(0);
        const auto rase_map = (target_mean.reciprocal() * 100) * torch::sqrt(torch::mean(torch::pow(rmse_map, 2), 0));
        const auto crop = detail::half_window(window_size);
        return torch::mean(rase_map.slice(0, crop, -crop).slice(1, crop, -crop));
    }

    // Keeps every batch and evaluates their concatenation (torchmetrics; inputs are checked at compute()).
    class RelativeAverageSpectralError : public metric::detail::metric_base<RelativeAverageSpectralError> {
    public:
        explicit RelativeAverageSpectralError(sliding_window_option opt = {}) : _opt(opt) {
            if (auto e = detail::window_error(_opt._window_size); !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            _preds.push_back(preds);
            _target.push_back(target);
        }

        auto compute() const -> tensor_t {
            return relative_average_spectral_error(detail::cat(_preds), detail::cat(_target), _opt._window_size);
        }

        auto options() const -> const sliding_window_option & { return _opt; }

    private:
        sliding_window_option _opt;
        std::vector<tensor_t> _preds, _target;
    };

    // ---------------- SpatialCorrelationCoefficient ----------------
    typedef struct spatial_correlation_coefficient_option {
        tensor_t _hp_filter; // undefined = the 3 x 3 Laplacian {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}
        int64_t _window_size = 8;
        reduction_mode _reduction = reduction_mode::elementwise_mean; // or none (per image); torchmetrics' "mean"

        auto hp_filter(tensor_t v) -> spatial_correlation_coefficient_option & {
            _hp_filter = std::move(v);
            return *this;
        }
        auto window_size(int64_t v) -> spatial_correlation_coefficient_option & {
            _window_size = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> spatial_correlation_coefficient_option & {
            _reduction = v;
            return *this;
        }
    } spatial_correlation_coefficient_option_t;

    namespace detail {
        inline auto scc_error(const tensor_t &preds, const tensor_t &target, int64_t window_size) -> std::string {
            if (auto e = shape_error("spatial_correlation_coefficient", preds, target); !e.empty()) {
                return e;
            }
            if (preds.dim() != 3 && preds.dim() != 4) {
                return std::format("Expected `preds` and `target` to have BxCxHxW or BxHxW shape. Got {}.",
                                   c10::str(preds.sizes()));
            }
            if (window_size <= 0) {
                return std::format("Expected `window_size` to be a positive integer. Got {}.", window_size);
            }
            return window_size > preds.size(-2) || window_size > preds.size(-1)
                           ? std::format("Expected `window_size` to be less than or equal to the size of the image. "
                                         "Got window_size: {} and image size: {}x{}.",
                                         window_size, preds.size(-2), preds.size(-1))
                           : std::string{};
        }

        // torchmetrics _symmetric_reflect_pad_2d with (left, right, top, bottom); a zero right / bottom pad takes the
        // whole side, like Python's x[-0:].
        inline auto symmetric_reflect_pad(const tensor_t &img, int64_t l, int64_t r, int64_t t, int64_t b) -> tensor_t {
            const auto padded = torch::cat({img.slice(3, 0, l).flip({3}), img, img.slice(3, -r).flip({3})}, 3);
            return torch::cat({padded.slice(2, 0, t).flip({2}), padded, padded.slice(2, -b).flip({2})}, 2);
        }

        // _scc_per_channel_compute on float32 [B, 1, H, W]
        inline auto scc_channel(const tensor_t &preds, const tensor_t &target, const tensor_t &hp, int64_t ws)
                -> tensor_t {
            const auto window = torch::ones({1, 1, ws, ws}, preds.options()) / (ws * ws);
            const auto kh = hp.size(2), kw = hp.size(3);
            const auto hp_flip = hp.flip({2, 3});
            const auto laplacian = [&](const tensor_t &x) {
                return torch::conv2d(symmetric_reflect_pad(x, (kw - 1) / 2, kw / 2, (kh - 1) / 2, kh / 2), hp_flip) *
                       2.0;
            };
            const int64_t l = ws / 2, r = (ws - 1) / 2;
            const auto p = torch::pad(laplacian(preds), {l, r, l, r});
            const auto t = torch::pad(laplacian(target), {l, r, l, r});
            const auto preds_mean = torch::conv2d(p, window);
            const auto target_mean = torch::conv2d(t, window);
            auto preds_var = torch::conv2d(torch::pow(p, 2), window) - torch::pow(preds_mean, 2);
            auto target_var = torch::conv2d(torch::pow(t, 2), window) - torch::pow(target_mean, 2);
            const auto cov = torch::conv2d(t * p, window) - target_mean * preds_mean;
            preds_var.masked_fill_(preds_var < 0, 0);
            target_var.masked_fill_(target_var < 0, 0);
            auto den = torch::sqrt(target_var) * torch::sqrt(preds_var);
            const auto idx = den == 0;
            den.masked_fill_(idx, 1);
            return (cov / den).masked_fill_(idx, 0);
        }

        // [B, C, H', W'] per-pixel SCC (checked inputs), computed in float32.
        inline auto scc_map(const tensor_t &preds_in, const tensor_t &target_in, const tensor_t &hp_filter, int64_t ws)
                -> tensor_t {
            auto preds = preds_in.to(torch::kFloat32);
            auto target = target_in.to(preds_in.scalar_type()).to(torch::kFloat32);
            if (preds.dim() == 3) {
                preds = preds.unsqueeze(1);
                target = target.unsqueeze(1);
            }
            const auto hp = (hp_filter.defined() ? hp_filter : torch::tensor({{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}))
                                    .unsqueeze(0)
                                    .unsqueeze(0)
                                    .to(preds.device(), torch::kFloat32);
            std::vector<tensor_t> per_channel;
            for (int64_t c = 0; c < preds.size(1); ++c) {
                per_channel.push_back(
                        scc_channel(preds.select(1, c).unsqueeze(1), target.select(1, c).unsqueeze(1), hp, ws));
            }
            return torch::cat(per_channel, 1);
        }
    } // namespace detail

    // torchmetrics spatial_correlation_coefficient over [B, C, H, W] or [B, H, W] (float32 result).
    inline auto spatial_correlation_coefficient(const tensor_t &preds, const tensor_t &target,
                                                const spatial_correlation_coefficient_option &opt = {}) -> tensor_t {
        if (opt._reduction == reduction_mode::sum) {
            return detail::invalid("spatial_correlation_coefficient: reduction must be elementwise_mean or none.");
        }
        if (auto e = detail::scc_error(preds, target, opt._window_size); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        const auto m = detail::scc_map(preds, target, opt._hp_filter, opt._window_size);
        return opt._reduction == reduction_mode::none ? torch::mean(m, {1, 2, 3}) : torch::mean(m);
    }

    // Mean over images of the per-image SCC (the option's reduction is not used, as in torchmetrics).
    class SpatialCorrelationCoefficient : public metric::detail::metric_base<SpatialCorrelationCoefficient> {
    public:
        explicit SpatialCorrelationCoefficient(spatial_correlation_coefficient_option opt = {}) :
            _opt(std::move(opt)) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::scc_error(preds, target, _opt._window_size); !e.empty()) {
                return detail::fail(std::move(e));
            }
            const auto m = detail::scc_map(preds, target, _opt._hp_filter, _opt._window_size);
            _score = detail::accumulate(_score, torch::sum(torch::mean(m, {1, 2, 3})));
            _total += preds.size(0);
        }

        auto compute() const -> tensor_t { return detail::squeeze_if_scalar(_score / static_cast<double>(_total)); }

        auto options() const -> const spatial_correlation_coefficient_option & { return _opt; }

    private:
        spatial_correlation_coefficient_option _opt;
        tensor_t _score = detail::zero_state();
        int64_t _total = 0;
    };

    // ---------------- VisualInformationFidelity ----------------
    typedef struct visual_information_fidelity_option {
        double _sigma_n_sq = 2.0; // variance of the visual noise
        reduction_mode _reduction = reduction_mode::elementwise_mean; // or none (per image); torchmetrics' "mean"

        auto sigma_n_sq(double v) -> visual_information_fidelity_option & {
            _sigma_n_sq = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> visual_information_fidelity_option & {
            _reduction = v;
            return *this;
        }
    } visual_information_fidelity_option_t;

    namespace detail {
        // torchmetrics vif _filter: normalised [n, n] gaussian with sigma n / 5.
        inline auto vif_filter(double win_size, double sigma, const torch::TensorOptions &opt) -> tensor_t {
            const auto coords = torch::arange(win_size, opt) - (win_size - 1) / 2;
            const auto g = torch::pow(coords, 2);
            auto k = torch::exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * std::pow(sigma, 2.0)));
            k /= torch::sum(k);
            return k;
        }

        // _vif_per_channel on [B, H, W]: [B]
        inline auto vif_channel(const tensor_t &preds_in, const tensor_t &target_in, double sigma_n_sq_v) -> tensor_t {
            const auto opt = preds_in.options();
            auto preds = preds_in.unsqueeze(1);
            auto target = target_in.unsqueeze(1);
            const auto eps = torch::tensor(1e-10, opt);
            const auto sigma_n_sq = torch::tensor(sigma_n_sq_v, opt);
            auto preds_vif = torch::zeros({preds.size(0)}, opt);
            auto target_vif = torch::zeros({preds.size(0)}, opt);
            for (int scale = 0; scale < 4; ++scale) {
                const double n = std::pow(2.0, 4 - scale) + 1;
                const auto kernel = vif_filter(n, n / 5, opt).unsqueeze(0).unsqueeze(0);
                if (scale > 0) {
                    target = torch::conv2d(target, kernel).slice(2, 0, {}, 2).slice(3, 0, {}, 2);
                    preds = torch::conv2d(preds, kernel).slice(2, 0, {}, 2).slice(3, 0, {}, 2);
                }
                const auto mu_target = torch::conv2d(target, kernel);
                const auto mu_preds = torch::conv2d(preds, kernel);
                const auto mu_target_sq = torch::pow(mu_target, 2);
                const auto mu_preds_sq = torch::pow(mu_preds, 2);
                const auto mu_target_preds = mu_target * mu_preds;
                auto sigma_target_sq = torch::clamp(torch::conv2d(torch::pow(target, 2), kernel) - mu_target_sq, 0.0);
                const auto sigma_preds_sq =
                        torch::clamp(torch::conv2d(torch::pow(preds, 2), kernel) - mu_preds_sq, 0.0);
                const auto sigma_target_preds = torch::conv2d(target * preds, kernel) - mu_target_preds;
                auto g = sigma_target_preds / (sigma_target_sq + eps);
                auto sigma_v_sq = sigma_preds_sq - g * sigma_target_preds;
                auto mask = sigma_target_sq < eps;
                g = g.masked_fill(mask, 0);
                sigma_v_sq = torch::where(mask, sigma_preds_sq, sigma_v_sq);
                sigma_target_sq = sigma_target_sq.masked_fill(mask, 0);
                mask = sigma_preds_sq < eps;
                g = g.masked_fill(mask, 0);
                sigma_v_sq = sigma_v_sq.masked_fill(mask, 0);
                mask = g < 0;
                sigma_v_sq = torch::where(mask, sigma_preds_sq, sigma_v_sq);
                g = g.masked_fill(mask, 0);
                sigma_v_sq = torch::clamp_min(sigma_v_sq, eps);
                preds_vif +=
                        torch::sum(torch::log10(1.0 + torch::pow(g, 2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)),
                                   {1, 2, 3});
                target_vif += torch::sum(torch::log10(1.0 + sigma_target_sq / sigma_n_sq), {1, 2, 3});
            }
            return preds_vif / target_vif;
        }

        // Per image [B]: the mean over channels of the per-channel VIF.
        inline auto vif_batch(const tensor_t &preds, const tensor_t &target, double sigma_n_sq) -> tensor_t {
            std::vector<tensor_t> per_channel;
            for (int64_t c = 0; c < preds.size(1); ++c) {
                per_channel.push_back(vif_channel(preds.select(1, c), target.select(1, c), sigma_n_sq));
            }
            return per_channel.size() > 1 ? torch::stack(per_channel, 0).mean(0) : per_channel[0];
        }

        // torchmetrics' size and shape checks (ValueError there; a 3-D input is an IndexError there).
        inline auto vif_input_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            for (const auto *t: {&preds, &target}) {
                if (t->dim() != 4 || t->size(-1) < 41 || t->size(-2) < 41) {
                    return std::format("Invalid size of {}. Expected [B, C, H, W] of at least 41x41, but got {}.",
                                       t == &preds ? "preds" : "target", c10::str(t->sizes()));
                }
            }
            return shape_error("visual_information_fidelity", preds, target);
        }

        inline auto vif_reduction_error(reduction_mode r) -> std::string {
            return r == reduction_mode::sum ? "visual_information_fidelity: reduction must be elementwise_mean or none."
                                            : std::string{};
        }
    } // namespace detail

    // torchmetrics visual_information_fidelity (pixel domain) over [B, C, H >= 41, W >= 41].
    inline auto visual_information_fidelity(const tensor_t &preds, const tensor_t &target,
                                            const visual_information_fidelity_option &opt = {}) -> tensor_t {
        if (auto e = detail::vif_input_error(preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        if (auto e = detail::vif_reduction_error(opt._reduction); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        const auto v = detail::vif_batch(preds, target, opt._sigma_n_sq);
        return opt._reduction == reduction_mode::none ? v : v.mean();
    }

    class VisualInformationFidelity : public metric::detail::metric_base<VisualInformationFidelity> {
    public:
        explicit VisualInformationFidelity(visual_information_fidelity_option opt = {}) : _opt(opt) {
            auto e = detail::vif_reduction_error(_opt._reduction);
            if (e.empty() && !(_opt._sigma_n_sq >= 0)) {
                e = std::format("Argument `sigma_n_sq` is expected to be a positive float or int, but got {}",
                                _opt._sigma_n_sq);
            }
            if (!e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::vif_input_error(preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _scores.push_back(detail::vif_batch(preds, target, _opt._sigma_n_sq));
        }

        auto compute() const -> tensor_t {
            const auto v = detail::cat(_scores);
            return detail::squeeze_if_scalar(_opt._reduction == reduction_mode::none ? v : v.mean());
        }

        auto options() const -> const visual_information_fidelity_option & { return _opt; }

    private:
        visual_information_fidelity_option _opt;
        std::vector<tensor_t> _scores;
    };
} // namespace torchmedia::metric::image
#endif // LIB_TORCH_MEDIA_METRIC_IMAGE_HPP
