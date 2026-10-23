#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_AUDIO_HPP
#define LIB_TORCH_MEDIA_METRIC_AUDIO_HPP
// Audio metrics (develop_log/2026-10-24/progress01): torchmetrics 1.9 torchmetrics.audio (SNR, SI-SDR, SI-SNR,
// C-SI-SNR, SDR with torchmetrics' own Toeplitz solve / fast_bss_eval's conjugate gradient, SA-SDR, PIT, STOI = pystoi
// 0.4.1) with the same libtorch calls, mir_eval 0.8.2 bss_eval_sources, and speaker-verification EER / minDCF. PESQ,
// DNSMOS, NISQA and SRMR are out of scope.
#include <functional>
#include <numeric>
#include "_common.hpp"

namespace torchmedia::metric::audio {
    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;
        using metric::detail::shape_error;

        // torch.finfo(dtype).eps of a floating dtype.
        inline auto eps_of(const tensor_t &x) -> double {
            switch (x.scalar_type()) {
                case torch::kFloat64:
                    return std::numeric_limits<double>::epsilon();
                case torch::kFloat16:
                    return 0.0009765625;
                case torch::kBFloat16:
                    return 0.0078125;
                default:
                    return std::numeric_limits<float>::epsilon();
            }
        }

        // Same shape and real floating inputs (torchmetrics: _check_same_shape, torch.finfo).
        inline auto input_error(const char *fn, const tensor_t &preds, const tensor_t &target) -> std::string {
            if (!preds.is_floating_point() || !target.is_floating_point()) {
                return std::format("{}: expected real floating point `preds` and `target`, got {} and {}.", fn,
                                   c10::toString(preds.scalar_type()), c10::toString(target.scalar_type()));
            }
            return shape_error(fn, preds, target);
        }

        inline auto zero_state() -> tensor_t { return torch::tensor(0.0, torch::kFloat32); }

        // torchmetrics' `sum += batch.sum(); total += batch.numel()` (out of place: float64 values keep a float64
        // state, as in the image classes).
        struct mean_state {
            tensor_t sum = zero_state(), total = torch::tensor(0, torch::kInt64);
            auto add(const tensor_t &batch) -> void {
                sum = sum + batch.sum();
                total = total + batch.numel();
            }
            auto mean() const -> tensor_t { return sum / total; }
        };

        inline auto snr(const tensor_t &preds, const tensor_t &target, bool zero_mean) -> tensor_t {
            const double eps = eps_of(preds);
            auto p = preds, t = target;
            if (zero_mean) {
                t = t - torch::mean(t, {-1}, true);
                p = p - torch::mean(p, {-1}, true);
            }
            const auto noise = t - p;
            return 10 * torch::log10((torch::sum(t.pow(2), {-1}) + eps) / (torch::sum(noise.pow(2), {-1}) + eps));
        }

        inline auto si_sdr(const tensor_t &preds, const tensor_t &target, bool zero_mean) -> tensor_t {
            const double eps = eps_of(preds);
            auto p = preds, t = target;
            if (zero_mean) {
                t = t - torch::mean(t, {-1}, true);
                p = p - torch::mean(p, {-1}, true);
            }
            const auto alpha = (torch::sum(p * t, {-1}, true) + eps) / (torch::sum(t.pow(2), {-1}, true) + eps);
            const auto t_scaled = alpha * t;
            const auto noise = t_scaled - p;
            return 10 *
                   torch::log10((torch::sum(t_scaled.pow(2), {-1}) + eps) / (torch::sum(noise.pow(2), {-1}) + eps));
        }
    } // namespace detail

    // zero_mean of SignalNoiseRatio, ScaleInvariantSignalDistortionRatio and ComplexScaleInvariantSignalNoiseRatio.
    typedef struct zero_mean_option {
        bool _zero_mean = false;

        auto zero_mean(bool v) -> zero_mean_option & {
            _zero_mean = v;
            return *this;
        }
    } zero_mean_option_t;

    // ---------------- SignalNoiseRatio ----------------
    // torchmetrics signal_noise_ratio over the last dim: 10 log10((|t|^2 + eps) / (|t - p|^2 + eps)), eps of preds'
    // dtype; result [...].
    inline auto signal_noise_ratio(const tensor_t &preds, const tensor_t &target, bool zero_mean = false) -> tensor_t {
        if (auto e = detail::input_error("signal_noise_ratio", preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::snr(preds, target, zero_mean);
    }

    class SignalNoiseRatio : public metric::detail::metric_base<SignalNoiseRatio> {
    public:
        explicit SignalNoiseRatio(zero_mean_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::input_error("signal_noise_ratio", preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::snr(preds, target, _opt._zero_mean));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const zero_mean_option & { return _opt; }

    private:
        zero_mean_option _opt;
        detail::mean_state _state;
    };

    // ---------------- ScaleInvariantSignalDistortionRatio / ScaleInvariantSignalNoiseRatio ----------------
    // torchmetrics scale_invariant_signal_distortion_ratio over the last dim; result [...].
    inline auto scale_invariant_signal_distortion_ratio(const tensor_t &preds, const tensor_t &target,
                                                        bool zero_mean = false) -> tensor_t {
        if (auto e = detail::input_error("scale_invariant_signal_distortion_ratio", preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::si_sdr(preds, target, zero_mean);
    }

    // SI-SDR with zero_mean (torchmetrics scale_invariant_signal_noise_ratio).
    inline auto scale_invariant_signal_noise_ratio(const tensor_t &preds, const tensor_t &target) -> tensor_t {
        return scale_invariant_signal_distortion_ratio(preds, target, true);
    }

    class ScaleInvariantSignalDistortionRatio
        : public metric::detail::metric_base<ScaleInvariantSignalDistortionRatio> {
    public:
        explicit ScaleInvariantSignalDistortionRatio(zero_mean_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::input_error("scale_invariant_signal_distortion_ratio", preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::si_sdr(preds, target, _opt._zero_mean));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const zero_mean_option & { return _opt; }

    private:
        zero_mean_option _opt;
        detail::mean_state _state;
    };

    class ScaleInvariantSignalNoiseRatio : public metric::detail::metric_base<ScaleInvariantSignalNoiseRatio> {
    public:
        auto update(const tensor_t &preds, const tensor_t &target) -> void { _si_sdr.update(preds, target); }
        auto compute() const -> tensor_t { return _si_sdr.compute(); }

    private:
        ScaleInvariantSignalDistortionRatio _si_sdr{zero_mean_option().zero_mean(true)};
    };

    // ---------------- ComplexScaleInvariantSignalNoiseRatio ----------------
    namespace detail {
        inline auto c_si_snr_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            if ((preds.dim() < 3 || preds.size(-1) != 2) || (target.dim() < 3 || target.size(-1) != 2)) {
                return std::format(
                        "complex_scale_invariant_signal_noise_ratio: predictions and targets are expected to "
                        "have the shape (..., frequency, time, 2), but got {} and {}.",
                        c10::str(preds.sizes()), c10::str(target.sizes()));
            }
            return input_error("complex_scale_invariant_signal_noise_ratio", preds, target);
        }

        inline auto as_real(const tensor_t &x) -> tensor_t { return x.is_complex() ? torch::view_as_real(x) : x; }

        inline auto c_si_snr(const tensor_t &preds, const tensor_t &target, bool zero_mean) -> tensor_t {
            auto shape = preds.sizes().slice(0, preds.dim() - 3).vec();
            shape.push_back(-1);
            return si_sdr(preds.reshape(shape), target.reshape(shape), zero_mean);
        }
    } // namespace detail

    // torchmetrics complex_scale_invariant_signal_noise_ratio: complex [..., F, T] or real [..., F, T, 2] inputs,
    // SI-SDR over the flattened (F, T, 2); result [...].
    inline auto complex_scale_invariant_signal_noise_ratio(const tensor_t &preds, const tensor_t &target,
                                                           bool zero_mean = false) -> tensor_t {
        const auto p = detail::as_real(preds), t = detail::as_real(target);
        if (auto e = detail::c_si_snr_error(p, t); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::c_si_snr(p, t, zero_mean);
    }

    class ComplexScaleInvariantSignalNoiseRatio
        : public metric::detail::metric_base<ComplexScaleInvariantSignalNoiseRatio> {
    public:
        explicit ComplexScaleInvariantSignalNoiseRatio(zero_mean_option opt = {}) : _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            const auto p = detail::as_real(preds), t = detail::as_real(target);
            if (auto e = detail::c_si_snr_error(p, t); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::c_si_snr(p, t, _opt._zero_mean));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const zero_mean_option & { return _opt; }

    private:
        zero_mean_option _opt;
        detail::mean_state _state;
    };

    // ---------------- SignalDistortionRatio ----------------
    typedef struct signal_distortion_ratio_option {
        std::optional<int64_t> _use_cg_iter; // {} = torch.linalg.solve; n = n conjugate-gradient iterations
        int64_t _filter_length = 512;
        bool _zero_mean = false;
        std::optional<double> _load_diag; // added to the autocorrelation's lag 0

        auto use_cg_iter(std::optional<int64_t> v) -> signal_distortion_ratio_option & {
            _use_cg_iter = v;
            return *this;
        }
        auto filter_length(int64_t v) -> signal_distortion_ratio_option & {
            _filter_length = v;
            return *this;
        }
        auto zero_mean(bool v) -> signal_distortion_ratio_option & {
            _zero_mean = v;
            return *this;
        }
        auto load_diag(std::optional<double> v) -> signal_distortion_ratio_option & {
            _load_diag = v;
            return *this;
        }
    } signal_distortion_ratio_option_t;

    namespace detail {
        // torchmetrics _symmetric_toeplitz: [..., L] -> [..., L, L].
        inline auto symmetric_toeplitz(const tensor_t &v) -> tensor_t {
            const auto vec_exp = torch::cat({torch::flip(v, {-1}), v.slice(-1, 1)}, -1);
            const auto len = v.size(-1);
            auto size = vec_exp.sizes().vec(), stride = vec_exp.strides().vec();
            size.back() = len;
            size.push_back(len);
            stride.back() = 1;
            stride.push_back(1);
            return vec_exp.as_strided(size, stride).flip({-1});
        }

        // fast_bss_eval 0.1.4 toeplitz_conjugate_gradient (circulant preconditioner, FFT Toeplitz product). Its
        // residual refresh `epoch + 1 % 5 == 0` never fires; kept.
        inline auto toeplitz_conjugate_gradient(const tensor_t &acf, const tensor_t &xcorr, int64_t n_iter)
                -> tensor_t {
            const auto n = acf.size(-1);
            const auto bw = torch::arange(1, n, torch::TensorOptions().device(acf.device())) / static_cast<double>(n);
            const auto w = acf.slice(-1, 1) * bw.flip({-1});
            const auto col_precond = torch::cat({acf.slice(-1, 0, 1), w + w.flip({-1})}, -1);
            auto c = torch::fft::rfft(col_precond, c10::nullopt, -1);
            c = (c / torch::clamp(torch::real(c).pow(2) + torch::imag(c).pow(2), 1e-6)).conj();
            const auto precond = [&](const tensor_t &lhs) {
                return torch::fft::irfft(c.unsqueeze(-1) * torch::fft::rfft(lhs, n, -2), n, -2);
            };
            auto pad_shape = acf.sizes().vec();
            pad_shape.back() = 1;
            const auto circ_col = torch::cat({acf, acf.new_zeros(pad_shape), acf.slice(-1, 1).flip({-1})}, -1);
            const auto c_forward = torch::fft::rfft(circ_col, c10::nullopt, -1);
            const auto forward = [&](const tensor_t &lhs) {
                return torch::fft::irfft(torch::fft::rfft(lhs, 2 * n, -2) * c_forward.unsqueeze(-1), 2 * n, -2)
                        .slice(-2, 0, n);
            };
            const auto inner = [](const tensor_t &a, const tensor_t &b) {
                return torch::einsum("...cd,...cd->...d", {a.conj(), b});
            };
            const auto b = xcorr.unsqueeze(-1);
            auto x = torch::zeros_like(b);
            auto r = b - forward(x);
            auto z = precond(r);
            auto p = z;
            auto rsold = inner(r, z);
            for (int64_t epoch = 0; epoch < n_iter; ++epoch) {
                const auto ap = forward(p);
                const auto alpha = rsold / torch::clamp(inner(p, ap), 1e-6);
                x = x + alpha.unsqueeze(-2) * p;
                r = r - alpha.unsqueeze(-2) * ap;
                z = precond(r);
                const auto rsnew = inner(r, z);
                const auto beta = rsnew / torch::clamp(rsold, 1e-6);
                p = z + beta.unsqueeze(-2) * p;
                rsold = rsnew;
            }
            return x.select(-1, 0);
        }

        inline auto sdr_option_error(const signal_distortion_ratio_option &opt) -> std::string {
            return opt._filter_length < 1 || (opt._use_cg_iter && *opt._use_cg_iter < 0)
                           ? std::format("signal_distortion_ratio: expected filter_length >= 1 and use_cg_iter >= 0, "
                                         "got {} and {}.",
                                         opt._filter_length, opt._use_cg_iter.value_or(0))
                           : std::string{};
        }

        inline auto sdr_error(const tensor_t &preds, const tensor_t &target, const signal_distortion_ratio_option &opt)
                -> std::string {
            if (auto e = sdr_option_error(opt); !e.empty()) {
                return e;
            }
            if (auto e = input_error("signal_distortion_ratio", preds, target); !e.empty()) {
                return e;
            }
            // torchmetrics: log2(2T - 1) raises for T = 0 (math domain error)
            return preds.dim() == 0 || preds.size(-1) == 0
                           ? std::format("signal_distortion_ratio: expected a non-empty time axis, got {}.",
                                         c10::str(preds.sizes()))
                           : std::string{};
        }

        inline auto sdr(const tensor_t &preds, const tensor_t &target, const signal_distortion_ratio_option &opt)
                -> tensor_t {
            auto p = preds.to(torch::kFloat64), t = target.to(torch::kFloat64);
            if (opt._zero_mean) {
                p = p - p.mean({-1}, true);
                t = t - t.mean({-1}, true);
            }
            t = t / torch::clamp(torch::linalg_vector_norm(t, 2, {-1}, true), 1e-6);
            p = p / torch::clamp(torch::linalg_vector_norm(p, 2, {-1}, true), 1e-6);
            // _compute_autocorr_crosscorr
            const auto n_fft = int64_t{1} << static_cast<int>(
                                       std::ceil(std::log2(static_cast<double>(p.size(-1) + t.size(-1) - 1))));
            const auto t_fft = torch::fft::rfft(t, n_fft, -1);
            auto r_0 = torch::fft::irfft(torch::real(t_fft).pow(2) + torch::imag(t_fft).pow(2), n_fft)
                               .slice(-1, 0, opt._filter_length);
            const auto p_fft = torch::fft::rfft(p, n_fft, -1);
            const auto b = torch::fft::irfft(t_fft.conj() * p_fft, n_fft, -1).slice(-1, 0, opt._filter_length);
            if (opt._load_diag) {
                r_0.select(-1, 0).add_(*opt._load_diag);
            }
            const auto sol = opt._use_cg_iter ? toeplitz_conjugate_gradient(r_0, b, *opt._use_cg_iter)
                                              : torch::linalg_solve(symmetric_toeplitz(r_0), b);
            const auto coh = torch::einsum("...l,...l->...", {b, sol});
            const auto val = 10.0 * torch::log10(coh / (1 - coh));
            return preds.scalar_type() == torch::kFloat64 ? val : val.to(torch::kFloat32);
        }
    } // namespace detail

    // torchmetrics signal_distortion_ratio: preds and target L2-normalised in float64, the filter_length-tap distortion
    // filter from the target's autocorrelation (Toeplitz solve, or fast_bss_eval's conjugate gradient), SDR =
    // 10 log10(coh / (1 - coh)); float32 result unless preds is float64.
    inline auto signal_distortion_ratio(const tensor_t &preds, const tensor_t &target,
                                        const signal_distortion_ratio_option &opt = {}) -> tensor_t {
        if (auto e = detail::sdr_error(preds, target, opt); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::sdr(preds, target, opt);
    }

    class SignalDistortionRatio : public metric::detail::metric_base<SignalDistortionRatio> {
    public:
        explicit SignalDistortionRatio(signal_distortion_ratio_option opt = {}) : _opt(opt) {
            if (auto e = detail::sdr_option_error(_opt); !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::sdr_error(preds, target, _opt); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::sdr(preds, target, _opt));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const signal_distortion_ratio_option & { return _opt; }

    private:
        signal_distortion_ratio_option _opt;
        detail::mean_state _state;
    };

    // ---------------- SourceAggregatedSignalDistortionRatio ----------------
    typedef struct source_aggregated_signal_distortion_ratio_option {
        bool _scale_invariant = true; // one scale for all speakers
        bool _zero_mean = false;

        auto scale_invariant(bool v) -> source_aggregated_signal_distortion_ratio_option & {
            _scale_invariant = v;
            return *this;
        }
        auto zero_mean(bool v) -> source_aggregated_signal_distortion_ratio_option & {
            _zero_mean = v;
            return *this;
        }
    } source_aggregated_signal_distortion_ratio_option_t;

    namespace detail {
        inline auto sa_sdr_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            if (auto e = input_error("source_aggregated_signal_distortion_ratio", preds, target); !e.empty()) {
                return e;
            }
            return preds.dim() < 2 ? std::format("source_aggregated_signal_distortion_ratio: the preds and target "
                                                 "should have the shape (..., spk, time), but {} found",
                                                 c10::str(preds.sizes()))
                                   : std::string{};
        }

        inline auto sa_sdr(const tensor_t &preds, const tensor_t &target,
                           const source_aggregated_signal_distortion_ratio_option &opt) -> tensor_t {
            const double eps = eps_of(preds);
            auto p = preds, t = target;
            if (opt._zero_mean) {
                t = t - torch::mean(t, {-1}, true);
                p = p - torch::mean(p, {-1}, true);
            }
            if (opt._scale_invariant) {
                const auto alpha = ((p * t).sum({-1}, true).sum({-2}, true) + eps) /
                                   (t.pow(2).sum({-1}, true).sum({-2}, true) + eps);
                t = alpha * t;
            }
            const auto distortion = t - p;
            return 10 *
                   torch::log10((t.pow(2).sum({-1}).sum({-1}) + eps) / (distortion.pow(2).sum({-1}).sum({-1}) + eps));
        }
    } // namespace detail

    // torchmetrics source_aggregated_signal_distortion_ratio of [..., spk, time]: result [...].
    inline auto
    source_aggregated_signal_distortion_ratio(const tensor_t &preds, const tensor_t &target,
                                              const source_aggregated_signal_distortion_ratio_option &opt = {})
            -> tensor_t {
        if (auto e = detail::sa_sdr_error(preds, target); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::sa_sdr(preds, target, opt);
    }

    class SourceAggregatedSignalDistortionRatio
        : public metric::detail::metric_base<SourceAggregatedSignalDistortionRatio> {
    public:
        explicit SourceAggregatedSignalDistortionRatio(source_aggregated_signal_distortion_ratio_option opt = {}) :
            _opt(opt) {}

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::sa_sdr_error(preds, target); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::sa_sdr(preds, target, _opt));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const source_aggregated_signal_distortion_ratio_option & { return _opt; }

    private:
        source_aggregated_signal_distortion_ratio_option _opt;
        detail::mean_state _state;
    };

    // ---------------- PermutationInvariantTraining ----------------
    enum class pit_mode { speaker_wise, permutation_wise };
    enum class eval_func_mode { max, min };
    // metric_func(preds, target) -> per-sample values (any metric here, other arguments bound by a lambda).
    using metric_func_t = std::function<tensor_t(const tensor_t &, const tensor_t &)>;

    typedef struct permutation_invariant_training_option {
        metric_func_t _metric_func; // required
        pit_mode _mode = pit_mode::speaker_wise;
        eval_func_mode _eval_func = eval_func_mode::max; // larger (max) or smaller (min) is better

        auto metric_func(metric_func_t v) -> permutation_invariant_training_option & {
            _metric_func = std::move(v);
            return *this;
        }
        auto mode(pit_mode v) -> permutation_invariant_training_option & {
            _mode = v;
            return *this;
        }
        auto eval_func(eval_func_mode v) -> permutation_invariant_training_option & {
            _eval_func = v;
            return *this;
        }
    } permutation_invariant_training_option_t;

    namespace detail {
        // itertools.permutations(range(spk)) as int64 [spk!, spk] (lexicographic).
        inline auto permutations(int64_t spk, const torch::Device &device) -> tensor_t {
            std::vector<int64_t> p(spk), all;
            std::iota(p.begin(), p.end(), 0);
            do {
                all.insert(all.end(), p.begin(), p.end());
            } while (std::next_permutation(p.begin(), p.end()));
            return torch::tensor(all).view({-1, spk}).to(device);
        }

        // scipy.optimize.linear_sum_assignment of a square cost matrix (rectangular_lsap.cpp: Crouse's shortest
        // augmenting path, same tie-breaking): col4row, or empty for NaN / -inf entries (after negation when
        // maximizing) or an infeasible matrix.
        inline auto linear_sum_assignment(std::vector<double> cost, int64_t n, bool maximize) -> std::vector<int64_t> {
            const double inf = std::numeric_limits<double>::infinity();
            for (auto &c: cost) {
                c = maximize ? -c : c;
                if (std::isnan(c) || c == -inf) {
                    return {};
                }
            }
            std::vector<double> u(n, 0.0), v(n, 0.0), spc(n);
            std::vector<int64_t> path(n, -1), col4row(n, -1), row4col(n, -1), remaining(n);
            std::vector<bool> sr(n), sc(n);
            for (int64_t cur = 0; cur < n; ++cur) {
                double min_val = 0.0;
                int64_t num_remaining = n, sink = -1, i = cur;
                for (int64_t it = 0; it < n; ++it) {
                    remaining[it] = n - it - 1;
                }
                std::fill(sr.begin(), sr.end(), false);
                std::fill(sc.begin(), sc.end(), false);
                std::fill(spc.begin(), spc.end(), inf);
                while (sink == -1) {
                    int64_t index = -1;
                    double lowest = inf;
                    sr[i] = true;
                    for (int64_t it = 0; it < num_remaining; ++it) {
                        const auto j = remaining[it];
                        const double r = min_val + cost[i * n + j] - u[i] - v[j];
                        if (r < spc[j]) {
                            path[j] = i;
                            spc[j] = r;
                        }
                        if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) {
                            lowest = spc[j];
                            index = it;
                        }
                    }
                    min_val = lowest;
                    if (min_val == inf) {
                        return {};
                    }
                    const auto j = remaining[index];
                    if (row4col[j] == -1) {
                        sink = j;
                    } else {
                        i = row4col[j];
                    }
                    sc[j] = true;
                    remaining[index] = remaining[--num_remaining];
                }
                u[cur] += min_val;
                for (int64_t k = 0; k < n; ++k) {
                    if (sr[k] && k != cur) {
                        u[k] += min_val - spc[col4row[k]];
                    }
                }
                for (int64_t k = 0; k < n; ++k) {
                    if (sc[k]) {
                        v[k] -= min_val - spc[k];
                    }
                }
                int64_t j = sink;
                while (true) {
                    const auto r = path[j];
                    row4col[j] = r;
                    std::swap(col4row[r], j);
                    if (r == cur) {
                        break;
                    }
                }
            }
            return col4row;
        }

        inline auto pit_error(const tensor_t &preds, const tensor_t &target,
                              const permutation_invariant_training_option &opt) -> std::string {
            if (!opt._metric_func) {
                return "permutation_invariant_training: `metric_func` is required.";
            }
            const auto head = [](const tensor_t &x) { return x.sizes().slice(0, std::min<int64_t>(2, x.dim())); };
            if (head(preds) != head(target)) {
                return "permutation_invariant_training: predictions and targets are expected to have the same shape at "
                       "the batch and speaker dimensions";
            }
            return target.dim() < 2
                           ? std::format("permutation_invariant_training: inputs must be of shape [batch, spk, "
                                         "...], got {} and {} instead",
                                         c10::str(target.sizes()), c10::str(preds.sizes()))
                           : std::string{};
        }

        inline auto best(const tensor_t &x, eval_func_mode eval) -> std::tuple<tensor_t, tensor_t> {
            return eval == eval_func_mode::max ? torch::max(x, 1) : torch::min(x, 1);
        }

        inline auto pit(const tensor_t &preds, const tensor_t &target, const permutation_invariant_training_option &opt)
                -> std::pair<tensor_t, tensor_t> {
            const auto batch = target.size(0), spk = target.size(1);
            const auto &f = opt._metric_func;
            if (opt._mode == pit_mode::permutation_wise) {
                const auto perms = permutations(spk, preds.device());
                const auto n_perm = perms.size(0);
                auto shape = preds.sizes().vec();
                shape[0] = batch * n_perm;
                const auto ppreds = torch::index_select(preds, 1, perms.reshape(-1)).reshape(shape);
                const auto ptarget = target.repeat_interleave(n_perm, 0);
                const auto metric_of_ps = torch::mean(f(ppreds, ptarget).reshape({batch, n_perm, -1}), {-1});
                const auto [best_metric, best_idx] = best(metric_of_ps, opt._eval_func);
                return {best_metric, perms.index({best_idx})};
            }
            const auto first = f(preds.select(1, 0), target.select(1, 0));
            auto mtx = torch::empty({batch, spk, spk}, first.options());
            mtx.select(2, 0).select(1, 0).copy_(first);
            for (int64_t t = 0; t < spk; ++t) {
                for (int64_t p = 0; p < spk; ++p) {
                    if (t != 0 || p != 0) {
                        mtx.select(2, p).select(1, t).copy_(f(preds.select(1, p), target.select(1, t)));
                    }
                }
            }
            if (spk < 3) {
                const auto ps = permutations(spk, mtx.device());
                const auto bps = ps.t().unsqueeze(0).expand({batch, spk, ps.size(0)});
                const auto [best_metric, best_idx] = best(torch::gather(mtx, 2, bps).mean(1), opt._eval_func);
                return {best_metric, ps.index({best_idx})};
            }
            const auto m = mtx.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
            std::vector<int64_t> perm;
            for (int64_t b = 0; b < batch; ++b) {
                const auto *d = m[b].data_ptr<double>();
                const auto col4row = linear_sum_assignment(std::vector<double>(d, d + spk * spk), spk,
                                                           opt._eval_func == eval_func_mode::max);
                if (col4row.empty()) {
                    return {invalid("permutation_invariant_training: the metric matrix contains invalid numeric "
                                    "entries or is infeasible for linear_sum_assignment."),
                            tensor_t{}};
                }
                perm.insert(perm.end(), col4row.begin(), col4row.end());
            }
            const auto best_perm = torch::tensor(perm).view({batch, spk}).to(mtx.device());
            return {torch::gather(mtx, 2, best_perm.unsqueeze(-1)).mean({-1, -2}), best_perm};
        }
    } // namespace detail

    // torchmetrics permutation_invariant_training of [batch, spk, ...]: {best metric [batch], best permutation [batch,
    // spk]} (perm[b][t] = the prediction matched to target t). speaker-wise calls metric_func(preds[:, p], target[:,
    // t]) for every pair (best permutation exhaustively for spk < 3, by scipy's linear_sum_assignment otherwise);
    // permutation-wise calls it once on every permutation of the speakers.
    inline auto permutation_invariant_training(const tensor_t &preds, const tensor_t &target,
                                               const permutation_invariant_training_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        if (auto e = detail::pit_error(preds, target, opt); !e.empty()) {
            return {detail::invalid(std::move(e)), tensor_t{}};
        }
        return detail::pit(preds, target, opt);
    }

    // torchmetrics pit_permutate: preds [batch, spk, ...] reordered by perm [batch, spk].
    inline auto pit_permutate(const tensor_t &preds, const tensor_t &perm) -> tensor_t {
        std::vector<tensor_t> out;
        for (int64_t b = 0; b < preds.size(0); ++b) {
            out.push_back(torch::index_select(preds[b], 0, perm[b]));
        }
        return torch::stack(out);
    }

    class PermutationInvariantTraining : public metric::detail::metric_base<PermutationInvariantTraining> {
    public:
        // A missing metric_func fails here (and in every update of a no-exceptions build).
        explicit PermutationInvariantTraining(permutation_invariant_training_option opt) : _opt(std::move(opt)) {
            if (!_opt._metric_func) {
                detail::fail("permutation_invariant_training: `metric_func` is required.");
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::pit_error(preds, target, _opt); !e.empty()) {
                return detail::fail(std::move(e));
            }
            // undefined: the linear_sum_assignment check failed in a no-exceptions build (state untouched)
            if (const auto m = detail::pit(preds, target, _opt).first; m.defined()) {
                _state.add(m);
            }
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const permutation_invariant_training_option & { return _opt; }

    private:
        permutation_invariant_training_option _opt;
        detail::mean_state _state;
    };

    // ---------------- ShortTimeObjectiveIntelligibility (pystoi 0.4.1) ----------------
    typedef struct short_time_objective_intelligibility_option {
        int64_t _fs = 0; // required: sampling rate (Hz)
        bool _extended = false; // extended STOI (Jensen & Taal 2016)

        auto fs(int64_t v) -> short_time_objective_intelligibility_option & {
            _fs = v;
            return *this;
        }
        auto extended(bool v) -> short_time_objective_intelligibility_option & {
            _extended = v;
            return *this;
        }
    } short_time_objective_intelligibility_option_t;

    namespace detail {
        inline constexpr int64_t stoi_fs = 10000, stoi_frame = 256, stoi_hop = 128, stoi_nfft = 512, stoi_n = 30;

        // numpy.i0 for |x| <= 8: exp(x) * chbevl(x / 2 - 2, _i0A) in numpy's evaluation order.
        inline auto np_i0(double x) -> double {
            TORCHMEDIA_FP_CONTRACT_OFF
            static constexpr std::array<double, 30> a{
                    -4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16, 1.715391285555133e-15,
                    -1.1685332877993451e-14, 7.676185498604936e-14,  -4.856446783111929e-13, 2.95505266312964e-12,
                    -1.726826291441556e-11,  9.675809035373237e-11,  -5.189795601635263e-10, 2.6598237246823866e-09,
                    -1.300025009986248e-08,  6.046995022541919e-08,  -2.670793853940612e-07, 1.1173875391201037e-06,
                    -4.4167383584587505e-06, 1.6448448070728896e-05, -5.754195010082104e-05, 0.00018850288509584165,
                    -0.0005763755745385824,  0.0016394756169413357,  -0.004324309995050576,  0.010546460394594998,
                    -0.02373741480589947,    0.04930528423967071,    -0.09490109704804764,   0.17162090152220877,
                    -0.3046826723431984,     0.6767952744094761};
            const double y = x / 2.0 - 2;
            double b0 = a[0], b1 = 0.0, b2 = 0.0;
            for (size_t i = 1; i < a.size(); ++i) {
                b2 = b1;
                b1 = b0;
                b0 = y * b1 - b2 + a[i];
            }
            return std::exp(x) * (0.5 * (b0 - b2));
        }

        // pystoi resample_oct(x, 10000, fs): its Kaiser-windowed sinc (utils._resample_window_oct, normalised to
        // sum 1) through scipy.signal.resample_poly, upfirdn's summation order with the fused multiply-adds of the
        // arm64 scipy build.
        inline auto stoi_resample(const std::vector<double> &x, int64_t fs) -> std::vector<double> {
            TORCHMEDIA_FP_CONTRACT_OFF
            const int64_t g = std::gcd(stoi_fs, fs), up = stoi_fs / g, down = fs / g;
            const double stop = 1.0 / (2 * static_cast<double>(std::max(up, down)));
            const double rejection = -20 * -3.0;
            const double half = std::ceil((rejection - 8) / (28.714 * (stop / 10)));
            const double beta = 0.1102 * (rejection - 8.7), i0_beta = np_i0(beta);
            const double gain = 2 * static_cast<double>(up) * stop, pi = 3.141592653589793;
            const auto len_h = static_cast<int64_t>(2 * half + 1);
            std::vector<double> h(len_h);
            for (int64_t k = 0; k < len_h; ++k) {
                const double xs = 2 * stop * (static_cast<double>(k) - half);
                const double ys = pi * (xs == 0 ? 1.0e-20 : xs);
                const double r = (static_cast<double>(k) - half) / half;
                h[k] = np_i0(beta * std::sqrt(1 - r * r)) / i0_beta * (gain * (std::sin(ys) / ys));
            }
            const double total = metric::detail::np_sum(h);
            // resample_poly: zero pads around h * up so that output n sits at upfirdn index n + n_pre_remove
            const auto n_in = static_cast<int64_t>(x.size());
            const int64_t n_out = (n_in * up + down - 1) / down, pre_pad = down - (len_h - 1) / 2 % down,
                          pre_remove = ((len_h - 1) / 2 + pre_pad) / down;
            const int64_t phase_len = (pre_pad + len_h + up - 1) / up; // taps per phase (post pads are zeros)
            std::vector<double> hp(phase_len * up, 0.0);
            for (int64_t k = 0; k < len_h; ++k) {
                hp[pre_pad + k] = h[k] / total * static_cast<double>(up);
            }
            std::vector<double> y(n_out);
            for (int64_t n = 0; n < n_out; ++n) {
                const int64_t m = (n + pre_remove) * down, t = m % up, xi = m / up;
                double acc = 0.0;
                for (int64_t k = phase_len - 1; k >= 0; --k) {
                    if (xi - k >= 0 && xi - k < n_in) {
                        acc = std::fma(x[xi - k], hp[t + up * k], acc);
                    }
                }
                y[n] = acc;
            }
            return y;
        }

        // Length after pystoi's resampling to 10 kHz.
        inline auto stoi_length(int64_t n, int64_t fs) -> int64_t {
            const int64_t g = std::gcd(stoi_fs, fs);
            return fs == stoi_fs ? n : (n * (stoi_fs / g) + fs / g - 1) / (fs / g);
        }

        // np.hanning(258)[1:-1] and pystoi's 1/3-octave band matrix thirdoct(10000, 512, 15, 150) [15, 257].
        inline auto stoi_tables() -> std::pair<tensor_t, tensor_t> {
            TORCHMEDIA_FP_CONTRACT_OFF
            const double pi = 3.141592653589793;
            std::vector<double> w(stoi_frame);
            for (int64_t k = 0; k < stoi_frame; ++k) {
                w[k] = 0.5 + 0.5 * std::cos(pi * static_cast<double>(2 * k - (stoi_frame - 1)) /
                                            static_cast<double>(stoi_frame + 1));
            }
            const int64_t n_bins = stoi_nfft / 2 + 1;
            auto obm = torch::zeros({15, n_bins}, torch::kFloat64);
            const auto bin = [&](double freq) {
                int64_t best = 0;
                double best_d = std::numeric_limits<double>::infinity();
                for (int64_t i = 0; i < n_bins; ++i) {
                    const double d = static_cast<double>(i) * (10000.0 / 512) - freq;
                    if (d * d < best_d) {
                        best_d = d * d;
                        best = i;
                    }
                }
                return best;
            };
            for (int64_t i = 0; i < 15; ++i) {
                const double k = static_cast<double>(i);
                obm[i].slice(0, bin(150 * std::pow(2.0, (2 * k - 1) / 6)), bin(150 * std::pow(2.0, (2 * k + 1) / 6)))
                        .fill_(1.0);
            }
            return {torch::tensor(w, torch::kFloat64), obm};
        }

        // pystoi.stoi(x = clean, y = processed, fs, extended) of one signal pair (float64 CPU 1-D, long enough).
        inline auto stoi_row(const tensor_t &x_in, const tensor_t &y_in, int64_t fs, bool extended, const tensor_t &w,
                             const tensor_t &obm) -> double {
            auto x = x_in, y = y_in;
            if (fs != stoi_fs) {
                const auto rs = [&](const tensor_t &s) {
                    return torch::tensor(
                            stoi_resample(std::vector<double>(s.data_ptr<double>(), s.data_ptr<double>() + s.numel()),
                                          fs),
                            torch::kFloat64);
                };
                x = rs(x);
                y = rs(y);
            }
            const double eps = std::numeric_limits<double>::epsilon();
            const auto count = [](const tensor_t &s) { // len(range(0, len(s) - 256, 128)), len(s) >= 256
                return (s.size(0) - stoi_frame + stoi_hop - 1) / stoi_hop;
            };
            const auto frames = [&](const tensor_t &s) {
                return s.unfold(0, stoi_frame, stoi_hop).slice(0, 0, count(s)) * w;
            };
            // remove_silent_frames: drop frames 40 dB below the loudest clean frame, overlap-add the rest
            auto xf = frames(x), yf = frames(y);
            const auto energies = 20 * torch::log10(torch::linalg_vector_norm(xf, 2, {1}) + eps);
            const auto keep = (energies.max() - 40 - energies) < 0;
            const auto ola = [&](const tensor_t &f) {
                const auto k = f.size(0);
                auto out = torch::zeros({(k + 1) * stoi_hop}, torch::kFloat64);
                out.slice(0, 0, k * stoi_hop).add_(f.slice(1, 0, stoi_hop).reshape(-1));
                out.slice(0, stoi_hop).add_(f.slice(1, stoi_hop).reshape(-1));
                return out;
            };
            const auto xs = ola(xf.index({keep})), ys = ola(yf.index({keep}));
            if (count(xs) < stoi_n) {
                return 1e-5; // pystoi: not enough STFT frames for the intermediate intelligibility measure
            }
            const auto spec = [&](const tensor_t &s) { return torch::fft::rfft(frames(s), stoi_nfft); };
            const auto x_spec = spec(xs), y_spec = spec(ys);
            // 1/3-octave band envelopes, segments of 30 frames [J, 15, 30]
            const auto seg = [&](const tensor_t &s) {
                return torch::sqrt(obm.matmul(s.abs().square().t())).unfold(1, stoi_n, 1).permute({1, 0, 2});
            };
            const auto x_seg = seg(x_spec), y_seg = seg(y_spec);
            if (extended) {
                // row_col_normalize without pystoi's EPS * N(0, 1) dither (develop_log/2026-10-24/progress01 D7):
                // mean removal, then 1 / norm like pystoi's diagonal matmul. A row / column that is constant up to
                // rounding (spread <= 1e-13 of its norm: zero-filled dropouts, one speech frame next to one, an
                // all-zero signal) scales by 0, the expected value of what the dither makes of it.
                const auto center_scale = [](const tensor_t &s, int64_t dim) {
                    const auto c = s - s.mean({dim}, true);
                    const auto n = c.square().sum({dim}, true).sqrt();
                    return c * torch::where(n > 1e-13 * s.square().sum({dim}, true).sqrt(), n.reciprocal(), 0.0);
                };
                const auto normalize = [&](const tensor_t &s) { return center_scale(center_scale(s, -1), 1); };
                const auto xn = normalize(x_seg);
                return (torch::sum(xn * normalize(y_seg) / stoi_n) / xn.size(0)).item<double>();
            }
            const auto norm2 = [](const tensor_t &s) { return torch::linalg_vector_norm(s, 2, {2}, true); };
            const auto y_norm = y_seg * (norm2(x_seg) / (norm2(y_seg) + eps));
            auto y_prime = torch::minimum(y_norm, x_seg * (1 + std::pow(10.0, 15.0 / 20)));
            y_prime = y_prime - y_prime.mean({2}, true);
            auto x_c = x_seg - x_seg.mean({2}, true);
            y_prime = y_prime / (norm2(y_prime) + eps);
            x_c = x_c / (norm2(x_c) + eps);
            return (torch::sum(y_prime * x_c) / (x_c.size(0) * x_c.size(1))).item<double>();
        }

        inline auto stoi_error(const tensor_t &preds, const tensor_t &target, int64_t fs) -> std::string {
            if (preds.is_complex() || target.is_complex() || fs <= 0) {
                return std::format(
                        "short_time_objective_intelligibility: expected real inputs and fs > 0, got fs = {}.", fs);
            }
            if (auto e = shape_error("short_time_objective_intelligibility", preds, target); !e.empty()) {
                return e;
            }
            const auto n = preds.dim() == 0 ? 1 : preds.size(-1);
            return (preds.numel() == 0 && n > 0) || stoi_length(n, fs) > stoi_frame
                           ? std::string{}
                           : std::format(
                                     "short_time_objective_intelligibility: {} samples at {} Hz leave no 256-sample "
                                     "frame at 10 kHz (pystoi raises).",
                                     n, fs);
        }

        inline auto stoi(const tensor_t &preds, const tensor_t &target, int64_t fs, bool extended) -> tensor_t {
            const auto n = preds.size(-1);
            const auto p = preds.detach().to(torch::kCPU).to(torch::kFloat64).reshape({-1, n}).contiguous();
            const auto t = target.detach().to(torch::kCPU).to(torch::kFloat64).reshape({-1, n}).contiguous();
            const auto [w, obm] = stoi_tables();
            auto out = torch::empty({p.size(0)}, torch::kFloat64);
            for (int64_t b = 0; b < p.size(0); ++b) {
                out[b] = stoi_row(t[b], p[b], fs, extended, w, obm);
            }
            return out.reshape(preds.sizes().slice(0, preds.dim() - 1));
        }
    } // namespace detail

    // pystoi.stoi(target, preds, fs, extended) per signal over the last dim (torchmetrics delegates to it): float64 CPU
    // result [...]; 1e-5 when fewer than 30 STFT frames remain after silent-frame removal. Inputs shorter than one
    // 256-sample frame at 10 kHz are invalid (pystoi raises). Extended STOI omits pystoi's random 2.2e-16 dither: band
    // envelopes constant up to rounding over a 30-frame segment (e.g. >= 0.4 s of zero-filled dropout) contribute 0,
    // the expected value of pystoi's random contribution.
    inline auto short_time_objective_intelligibility(const tensor_t &preds, const tensor_t &target, int64_t fs,
                                                     bool extended = false) -> tensor_t {
        if (auto e = detail::stoi_error(preds, target, fs); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::stoi(preds, target, fs, extended);
    }

    class ShortTimeObjectiveIntelligibility : public metric::detail::metric_base<ShortTimeObjectiveIntelligibility> {
    public:
        explicit ShortTimeObjectiveIntelligibility(short_time_objective_intelligibility_option opt) : _opt(opt) {
            if (_opt._fs <= 0) {
                const auto fs = _opt._fs;
                _opt._fs = detail::stoi_fs;
                detail::fail(std::format("short_time_objective_intelligibility: expected fs > 0, got {}.", fs));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            if (auto e = detail::stoi_error(preds, target, _opt._fs); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _state.add(detail::stoi(preds, target, _opt._fs, _opt._extended));
        }
        auto compute() const -> tensor_t { return _state.mean(); }
        auto options() const -> const short_time_objective_intelligibility_option & { return _opt; }

    private:
        short_time_objective_intelligibility_option _opt;
        detail::mean_state _state;
    };

    // ---------------- bss_eval_sources (mir_eval 0.8.2) ----------------
    struct bss_eval_sources_result {
        tensor_t sdr, sir, sar; // float64 [nsrc], per reference source
        tensor_t perm; // int64 [nsrc]: the estimate matched to each reference
    };

    namespace detail {
        inline constexpr int64_t bss_flen = 512, bss_max_sources = 100;

        inline auto bss_error(const tensor_t &ref, const tensor_t &est) -> std::string {
            if (ref.sizes() != est.sizes() || ref.dim() != 2) {
                return std::format("bss_eval_sources: expected reference and estimated sources of the same shape "
                                   "[nsrc, nsampl] or [nsampl], got {} and {}.",
                                   c10::str(ref.sizes()), c10::str(est.sizes()));
            }
            if (ref.numel() == 0) {
                return {};
            }
            if (ref.is_complex() || est.is_complex() || ref.size(0) > bss_max_sources) {
                return std::format("bss_eval_sources: expected real sources and at most {} of them.", bss_max_sources);
            }
            const auto silent = [](const tensor_t &x) { return (x == 0).all(1).any().item<bool>(); };
            return silent(ref) || silent(est) ? "bss_eval_sources: all the reference and estimated sources should be "
                                                "non-silent (not all-zeros)."
                                              : std::string{};
        }

        inline auto safe_db(double num, double den) -> double {
            return den == 0 ? std::numeric_limits<double>::infinity() : 10 * std::log10(num / den);
        }
    } // namespace detail

    // mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, compute_permutation): SDR / SIR / SAR
    // of each reference with 512-tap time-invariant distortion filters (FFT inner products, the Gram system solved per
    // projection), and the estimate permutation of the largest mean SIR (first in itertools order on ties; identity
    // without compute_permutation). Inputs [nsrc, nsampl] or [nsampl], float64 on the CPU; empty inputs give empty
    // results.
    inline auto bss_eval_sources(const tensor_t &reference_sources, const tensor_t &estimated_sources,
                                 bool compute_permutation = true) -> bss_eval_sources_result {
        const auto as2d = [](const tensor_t &x) { return x.dim() == 1 ? x.unsqueeze(0) : x; };
        if (auto e = detail::bss_error(as2d(reference_sources), as2d(estimated_sources)); !e.empty()) {
            return detail::invalid<bss_eval_sources_result>(std::move(e));
        }
        const auto ref = as2d(reference_sources).detach().to(torch::kCPU).to(torch::kFloat64);
        const auto est = as2d(estimated_sources).detach().to(torch::kCPU).to(torch::kFloat64);
        if (ref.numel() == 0) {
            const auto empty = torch::empty({0}, torch::kFloat64);
            return {empty, empty, empty, torch::empty({0}, torch::kInt64)};
        }
        const auto flen = detail::bss_flen, nsrc = ref.size(0), nsampl = ref.size(1);
        const auto n_fft =
                int64_t{1} << static_cast<int>(std::ceil(std::log2(static_cast<double>(nsampl + flen) - 1.0)));
        const auto sf = torch::fft::fft(ref, n_fft, 1), sef = torch::fft::fft(est, n_fft, 1);
        const auto lag = torch::arange(flen);
        // toeplitz(r = c[:flen], c = [c[0], c[-1], ...]): element (a, b) = c[(b - a) mod n_fft]
        const auto toeplitz_idx = torch::remainder(lag.unsqueeze(0) - lag.unsqueeze(1), n_fft);
        const auto d_idx = torch::remainder(-lag, n_fft);
        // Gram matrix of the delayed references; mir_eval's loop leaves block (r, c) = ss(r, c), (c, r) = ss(r, c)^T
        // for r >= c
        auto gram = torch::zeros({nsrc * flen, nsrc * flen}, torch::kFloat64);
        for (int64_t r = 0; r < nsrc; ++r) {
            for (int64_t c = 0; c <= r; ++c) {
                const auto ss = torch::real(torch::fft::ifft(sf[r] * sf[c].conj())).index({toeplitz_idx});
                gram.slice(0, r * flen, (r + 1) * flen).slice(1, c * flen, (c + 1) * flen).copy_(ss);
                gram.slice(0, c * flen, (c + 1) * flen).slice(1, r * flen, (r + 1) * flen).copy_(ss.t());
            }
        }
        const auto rf = torch::fft::rfft(ref, n_fft, 1);
        // _project: least-squares filters of the given references (lstsq when the Gram matrix is singular, like
        // mir_eval), filtered sum of length nsampl + flen - 1
        const auto project = [&](const tensor_t &g, const tensor_t &d, int64_t first) {
            auto [c, info] = torch::linalg_solve_ex(g, d);
            if (info.item<int64_t>() != 0) {
                c = std::get<0>(torch::linalg_lstsq(g, d.unsqueeze(-1), c10::nullopt, "gelsd")).squeeze(-1);
            }
            auto out = torch::zeros({nsampl + flen - 1}, torch::kFloat64);
            for (int64_t i = 0; i < c.size(0) / flen; ++i) {
                out += torch::fft::irfft(torch::fft::rfft(c.slice(0, i * flen, (i + 1) * flen), n_fft) * rf[first + i],
                                         n_fft)
                               .slice(0, 0, nsampl + flen - 1);
            }
            return out;
        };
        const auto sq = [](const tensor_t &x) { return torch::sum(x.pow(2)).item<double>(); };
        std::vector<double> sdr(nsrc * nsrc), sir(nsrc * nsrc), sar(nsrc * nsrc);
        for (int64_t je = 0; je < nsrc; ++je) {
            std::vector<tensor_t> blocks;
            for (int64_t i = 0; i < nsrc; ++i) {
                blocks.push_back(torch::real(torch::fft::ifft(sf[i] * sef[je].conj())).index({d_idx}));
            }
            const auto d = torch::cat(blocks);
            const auto p_all = project(gram, d, 0);
            for (int64_t jt = 0; jt < nsrc; ++jt) {
                if (!compute_permutation && jt != je) {
                    continue;
                }
                const auto blk = [&](const tensor_t &x) { return x.slice(-1, jt * flen, (jt + 1) * flen); };
                const auto s_true = torch::cat({ref[jt], torch::zeros({flen - 1}, torch::kFloat64)});
                const auto e_spat = project(blk(gram.slice(0, jt * flen, (jt + 1) * flen)), blk(d), jt) - s_true;
                const auto e_interf = p_all - s_true - e_spat;
                auto e_artif = -s_true - e_spat - e_interf;
                e_artif.slice(0, 0, nsampl).add_(est[je]);
                const auto s_filt = s_true + e_spat;
                const auto k = je * nsrc + jt;
                sdr[k] = detail::safe_db(sq(s_filt), sq(e_interf + e_artif));
                sir[k] = detail::safe_db(sq(s_filt), sq(e_interf));
                sar[k] = detail::safe_db(sq(s_filt + e_interf), sq(e_artif));
            }
        }
        std::vector<int64_t> popt(nsrc);
        std::iota(popt.begin(), popt.end(), 0);
        if (compute_permutation) {
            auto p = popt;
            double best = -std::numeric_limits<double>::infinity();
            bool first = true;
            do { // np.argmax of the mean SIR: first maximum, NaN wins
                std::vector<double> v(nsrc);
                for (int64_t k = 0; k < nsrc; ++k) {
                    v[k] = sir[p[k] * nsrc + k];
                }
                const double m = metric::detail::np_sum(v) / static_cast<double>(nsrc);
                if (first || (!std::isnan(best) && (m > best || std::isnan(m)))) {
                    best = m;
                    popt = p;
                    first = false;
                }
            } while (std::next_permutation(p.begin(), p.end()));
        }
        std::vector<double> o_sdr(nsrc), o_sir(nsrc), o_sar(nsrc);
        for (int64_t k = 0; k < nsrc; ++k) {
            const auto i = popt[k] * nsrc + k;
            o_sdr[k] = sdr[i];
            o_sir[k] = sir[i];
            o_sar[k] = sar[i];
        }
        return {torch::tensor(o_sdr, torch::kFloat64), torch::tensor(o_sir, torch::kFloat64),
                torch::tensor(o_sar, torch::kFloat64), torch::tensor(popt)};
    }

    // ---------------- Speaker verification: EqualErrorRate / MinDCF ----------------
    typedef struct min_dcf_option {
        double _p_target = 0.01; // prior of a target trial, in (0, 1)
        double _c_miss = 1.0; // cost of a miss (> 0)
        double _c_fa = 1.0; // cost of a false alarm (> 0)

        auto p_target(double v) -> min_dcf_option & {
            _p_target = v;
            return *this;
        }
        auto c_miss(double v) -> min_dcf_option & {
            _c_miss = v;
            return *this;
        }
        auto c_fa(double v) -> min_dcf_option & {
            _c_fa = v;
            return *this;
        }
    } min_dcf_option_t;

    namespace detail {
        inline auto scores_error(const char *fn, const tensor_t &target_scores, const tensor_t &nontarget_scores)
                -> std::string {
            for (const auto *s: {&target_scores, &nontarget_scores}) {
                if (s->dim() != 1 || s->numel() == 0 || s->is_complex() || !torch::isfinite(*s).all().item<bool>()) {
                    return std::format("{}: expected non-empty 1-D finite target and non-target scores, got {} and {}.",
                                       fn, c10::str(target_scores.sizes()), c10::str(nontarget_scores.sizes()));
                }
            }
            return {};
        }

        inline auto dcf_option_error(const min_dcf_option &opt) -> std::string {
            return opt._p_target > 0 && opt._p_target < 1 && opt._c_miss > 0 && opt._c_fa > 0
                           ? std::string{}
                           : std::format("min_dcf: expected 0 < p_target < 1, c_miss > 0 and c_fa > 0, got {}, {}, {}.",
                                         opt._p_target, opt._c_miss, opt._c_fa);
        }

        // sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False) as (fpr, fnr = 1 - tpr): one point per
        // distinct score from the highest down, after the (0, 1) point of threshold +inf.
        inline auto roc(const tensor_t &target_scores, const tensor_t &nontarget_scores)
                -> std::pair<std::vector<double>, std::vector<double>> {
            const auto to_vec = [](const tensor_t &x) {
                const auto c = x.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
                return std::vector<double>(c.data_ptr<double>(), c.data_ptr<double>() + c.numel());
            };
            std::vector<std::pair<double, int>> s;
            for (const auto v: to_vec(target_scores)) {
                s.emplace_back(v, 1);
            }
            for (const auto v: to_vec(nontarget_scores)) {
                s.emplace_back(v, 0);
            }
            std::sort(s.begin(), s.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            const auto pos = static_cast<double>(target_scores.numel()),
                       neg = static_cast<double>(nontarget_scores.numel());
            std::vector<double> fpr{0.0}, fnr{1.0 - 0.0 / pos};
            double tp = 0, fp = 0;
            for (size_t i = 0; i < s.size(); ++i) {
                (s[i].second ? tp : fp) += 1;
                if (i + 1 == s.size() || s[i + 1].first != s[i].first) {
                    fpr.push_back(fp / neg);
                    fnr.push_back(1.0 - tp / pos);
                }
            }
            return {fpr, fnr};
        }

        inline auto eer(const tensor_t &target_scores, const tensor_t &nontarget_scores) -> tensor_t {
            TORCHMEDIA_FP_CONTRACT_OFF
            const auto [fpr, fnr] = roc(target_scores, nontarget_scores);
            size_t i = 1;
            while (fnr[i] - fpr[i] > 0) { // the last point (1, 0) ends it
                ++i;
            }
            const double d0 = fnr[i - 1] - fpr[i - 1], d1 = fnr[i] - fpr[i];
            const double v = d1 == 0 ? fpr[i] : fpr[i - 1] + d0 / (d0 - d1) * (fpr[i] - fpr[i - 1]);
            return torch::tensor(v, torch::kFloat64);
        }

        inline auto min_dcf(const tensor_t &target_scores, const tensor_t &nontarget_scores, const min_dcf_option &opt)
                -> tensor_t {
            TORCHMEDIA_FP_CONTRACT_OFF
            const auto [fpr, fnr] = roc(target_scores, nontarget_scores);
            double best = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < fpr.size(); ++i) {
                best = std::min(best, opt._c_miss * fnr[i] * opt._p_target + opt._c_fa * fpr[i] * (1 - opt._p_target));
            }
            return torch::tensor(best / std::min(opt._c_miss * opt._p_target, opt._c_fa * (1 - opt._p_target)),
                                 torch::kFloat64);
        }
    } // namespace detail

    // Equal error rate of target / non-target trial scores (higher = more likely target), float64 0-d: on the ROC of
    // sklearn.metrics.roc_curve(drop_intermediate=False), the first point i with FNR_i <= FPR_i and the previous one
    // are joined linearly: EER = FPR_{i-1} + t (FPR_i - FPR_{i-1}), t = d_{i-1} / (d_{i-1} - d_i), d = FNR - FPR
    // (FPR_i when d_i = 0). Tied scores form one ROC point.
    inline auto equal_error_rate(const tensor_t &target_scores, const tensor_t &nontarget_scores) -> tensor_t {
        if (auto e = detail::scores_error("equal_error_rate", target_scores, nontarget_scores); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::eer(target_scores, nontarget_scores);
    }

    // Normalised minimum detection cost (NIST SRE form), float64 0-d: min over the ROC points above (every distinct
    // score threshold plus "reject all") of c_miss FNR p_target + c_fa FPR (1 - p_target), divided by
    // min(c_miss p_target, c_fa (1 - p_target)).
    inline auto min_dcf(const tensor_t &target_scores, const tensor_t &nontarget_scores, const min_dcf_option &opt = {})
            -> tensor_t {
        if (auto e = detail::dcf_option_error(opt); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        if (auto e = detail::scores_error("min_dcf", target_scores, nontarget_scores); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::min_dcf(target_scores, nontarget_scores, opt);
    }

    // Keep every batch's scores; compute() is the functional over all of them.
    class EqualErrorRate : public metric::detail::metric_base<EqualErrorRate> {
    public:
        auto update(const tensor_t &target_scores, const tensor_t &nontarget_scores) -> void {
            if (auto e = detail::scores_error("equal_error_rate", target_scores, nontarget_scores); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _target.push_back(target_scores.detach().clone());
            _nontarget.push_back(nontarget_scores.detach().clone());
        }
        auto compute() const -> tensor_t { return equal_error_rate(torch::cat(_target), torch::cat(_nontarget)); }

    private:
        std::vector<tensor_t> _target, _nontarget;
    };

    class MinDCF : public metric::detail::metric_base<MinDCF> {
    public:
        explicit MinDCF(min_dcf_option opt = {}) : _opt(opt) {
            if (auto e = detail::dcf_option_error(_opt); !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &target_scores, const tensor_t &nontarget_scores) -> void {
            if (auto e = detail::scores_error("min_dcf", target_scores, nontarget_scores); !e.empty()) {
                return detail::fail(std::move(e));
            }
            _target.push_back(target_scores.detach().clone());
            _nontarget.push_back(nontarget_scores.detach().clone());
        }
        auto compute() const -> tensor_t { return min_dcf(torch::cat(_target), torch::cat(_nontarget), _opt); }
        auto options() const -> const min_dcf_option & { return _opt; }

    private:
        min_dcf_option _opt;
        std::vector<tensor_t> _target, _nontarget;
    };
} // namespace torchmedia::metric::audio
#endif // LIB_TORCH_MEDIA_METRIC_AUDIO_HPP
