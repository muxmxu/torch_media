#ifndef LIB_TORCH_MEDIA_AUDIO_SEPARATION_HPP
#define LIB_TORCH_MEDIA_AUDIO_SEPARATION_HPP
// STFT-domain blind source separation: AuxIVA-IP / OverIVA / ILRMA-IP (see develop_log 2026-10-23/progress03).
// Ported from torchiva 0.1.1 (MIT, Copyright (c) 2022 Robin Scheibler, Kohei Saijo; see THIRD_PARTY_LICENSES).
#include <cmath>
#include <format>
#include <stdexcept>
#include "../globel_include.hpp"

namespace torchmedia::audio::separation {
    enum class source_model { laplace, gauss, nmf };

    typedef struct auxiva_option {
        int _n_iter = 10;
        int _n_src = -1; // -1 => n_chan; n_src < n_chan => OverIVA (orthogonal constraint)
        source_model _model = source_model::laplace;
        int _n_basis = 2; // NMF rank (ILRMA)
        int _proj_back_mic = 0; // -1 => no projection back
        double _eps = 1e-5; // torchiva eps: floor of the source models, the normalizations and the row-norm clamps
        double _load = 1e-8; // diagonal load of every solve (IP update, OverIVA J, projection back); torchiva: 1e-5

        auto n_iter(const int n) -> auxiva_option & {
            _n_iter = n;
            return *this;
        }
        auto n_src(const int n) -> auxiva_option & {
            _n_src = n;
            return *this;
        }
        auto model(const source_model m) -> auxiva_option & {
            _model = m;
            return *this;
        }
        auto n_basis(const int n) -> auxiva_option & {
            _n_basis = n;
            return *this;
        }
        auto proj_back_mic(const int m) -> auxiva_option & {
            _proj_back_mic = m;
            return *this;
        }
        auto eps(const double e) -> auxiva_option & {
            _eps = e;
            return *this;
        }
        auto load(const double l) -> auxiva_option & {
            _load = l;
            return *this;
        }
    } auxiva_option_t;

    namespace detail {
        inline auto mag_sq(const_tensor_lref_t x) -> tensor_t {
            return torch::real(x).square() + torch::imag(x).square();
        }

        // torchiva linalg.solve_loaded_general: row-normalize A, then solve (A^H A + load I) x = A^H b.
        inline auto solve_loaded_general(const_tensor_lref_t A, const_tensor_lref_t b, const double load,
                                         const double eps) -> tensor_t {
            const auto weights = torch::reciprocal(torch::clamp(torch::linalg_vector_norm(A, 2, {-1}, true), eps));
            const auto load_eye = load * torch::eye(A.size(-1), A.options());
            const auto A2 = A * weights;
            const auto lhs = torch::einsum("...km,...kn->...mn", {A2.conj(), A2});
            const auto rhs = torch::einsum("...km,...kn->...mn", {A2.conj(), b * weights});
            return torch::linalg_solve(lhs + load_eye, rhs);
        }

        // OverIVA orthogonal constraint J [..., n_freq, n_chan - n_src, n_src]; undefined when n_src == n_chan.
        inline auto orthogonal_constraint(const_tensor_lref_t W, const_tensor_lref_t Cx, const double load,
                                          const double eps) -> tensor_t {
            const auto n_src = W.size(-2);
            if (n_src == W.size(-1)) {
                return {};
            }
            const auto A = torch::matmul(W, Cx);
            return solve_loaded_general(A.slice(-1, 0, n_src), A.slice(-1, n_src), load, eps).transpose(-2, -1).conj();
        }

        // Scale each source to its image on mic ref_mic, i.e. the row ref_mic of the (OverIVA-completed) mixing matrix.
        // torchiva ignores the load it passes here (always 1e-5) and clamps row norms at 1e-5; W scales as 1/|X|, so
        // that clamp made the output depend on the input scale. A 1e-30 clamp only guards all-zero rows (dead mic).
        inline auto projection_back(const_tensor_lref_t Y, const_tensor_lref_t W, const_tensor_lref_t J,
                                    const int64_t ref_mic, const double load) -> tensor_t {
            constexpr double row_floor = 1e-30;
            const auto n_src = Y.size(-3);
            tensor_t WW, e1; // e1 broadcasts against WW inside solve_loaded_general
            if (!J.defined()) {
                WW = W;
                e1 = torch::eye(W.size(-1), W.options()).slice(1, ref_mic, ref_mic + 1);
            } else {
                const auto W_s = W.slice(-1, 0, n_src);
                WW = W_s + torch::matmul(W.slice(-1, n_src), J);
                // torchiva broadcasts this already per-frequency e1 once more over WW's batch shape, which yields a
                // [..., n_freq, ..., n_src, n_freq, n_frames] output for ref_mic >= n_src; we keep it per-frequency.
                e1 = ref_mic < n_src ? torch::eye(n_src, W.options()).slice(1, ref_mic, ref_mic + 1)
                                     : J.slice(-2, ref_mic - n_src, ref_mic - n_src + 1).transpose(-2, -1);
            }
            const auto a = solve_loaded_general(WW.transpose(-2, -1), e1, load, row_floor);
            return Y * a.transpose(-3, -2);
        }

        // NMF (ILRMA) source model, one multiplicative update per call; T/V are the lazily drawn per-call state.
        inline auto nmf_weights(const_tensor_lref_t Y, const int n_basis, const double eps, tensor_t &T, tensor_t &V)
                -> tensor_t {
            const auto n_freq = Y.size(-2);
            const auto n_frames = Y.size(-1);
            const auto P = mag_sq(Y.reshape({-1, n_freq, n_frames}));
            const auto inv_estimate = [eps](const tensor_t &t, const tensor_t &v) {
                return torch::reciprocal(torch::clamp(torch::bmm(t, v.transpose(-2, -1)), eps));
            };
            if (!T.defined()) { // uniform in [0.1, 1.0): same generator, order and shapes as torchiva
                T = 0.9 * torch::zeros({P.size(0), n_freq, n_basis}, P.options()).uniform_() + 0.1;
                V = 0.9 * torch::zeros({P.size(0), n_frames, n_basis}, P.options()).uniform_() + 0.1;
            }
            auto iR = inv_estimate(T, V);
            T = torch::clamp(T * torch::sqrt(torch::bmm(P * torch::square(iR), V) / torch::bmm(iR, V)), eps);
            iR = inv_estimate(T, V);
            const auto iR_T = iR.transpose(-2, -1);
            const auto P_T = P.transpose(-2, -1);
            V = torch::clamp(V * torch::sqrt(torch::bmm(P_T * torch::square(iR_T), T) / torch::bmm(iR_T, T)), eps);
            return inv_estimate(T, V).reshape(Y.sizes());
        }

        // Source model weights 1 / r(Y), shaped to broadcast against Y [..., n_src, n_freq, n_frames].
        inline auto source_weights(const_tensor_lref_t Y, const auxiva_option &opt, tensor_t &T, tensor_t &V)
                -> tensor_t {
            switch (opt._model) {
                case source_model::laplace:
                    return torch::reciprocal(torch::clamp(2.0 * torch::sqrt(mag_sq(Y).sum(-2, true)), opt._eps));
                case source_model::gauss:
                    return torch::reciprocal(torch::clamp(mag_sq(Y).mean(-2, true), opt._eps));
                default:
                    return nmf_weights(Y, opt._n_basis, opt._eps, T, V);
            }
        }
    } // namespace detail

    // AuxIVA with iterative projection updates (Ono 2011); OverIVA (Scheibler & Ono 2019) when n_src < n_chan.
    // X: complex STFT [..., n_chan, n_freq, n_frames] (complex64/complex128); returns [..., n_src, n_freq, n_frames].
    // Defaults follow torchiva 0.1.1 except the diagonal load: its 1e-5 leaves the low-frequency bins of compact
    // arrays (e.g. 3 mics 5 cm apart) mixed, so ours is 1e-8; load(1e-5) reproduces torchiva bit for bit (progress03
    // G2, D7). Devices: CPU tested, CUDA untested, MPS unsupported (libtorch lacks complex norm and linalg_solve).
    inline auto auxiva_ip(const_tensor_lref_t X, const auxiva_option &opt = {}) -> tensor_t {
        if (X.dim() < 3) {
            throw std::invalid_argument(
                    std::format("auxiva_ip: X must be [..., n_chan, n_freq, n_frames], got {} dims", X.dim()));
        }
        if (X.scalar_type() != torch::kComplexFloat && X.scalar_type() != torch::kComplexDouble) {
            throw std::invalid_argument(std::format("auxiva_ip: X must be a complex64 or complex128 STFT, got {}",
                                                    c10::toString(X.scalar_type())));
        }
        const auto n_chan = X.size(-3);
        const auto n_freq = X.size(-2);
        const auto n_frames = X.size(-1);
        if (opt._n_src == 0 || opt._n_src < -1 || opt._n_src > n_chan) {
            throw std::invalid_argument(
                    std::format("auxiva_ip: n_src must be -1 or in [1, n_chan={}], got {}", n_chan, opt._n_src));
        }
        if (opt._n_iter < 0) {
            throw std::invalid_argument(std::format("auxiva_ip: n_iter must be >= 0, got {}", opt._n_iter));
        }
        if (opt._proj_back_mic < -1 || opt._proj_back_mic >= n_chan) {
            throw std::invalid_argument(std::format("auxiva_ip: proj_back_mic must be -1 or in [0, n_chan={}), got {}",
                                                    n_chan, opt._proj_back_mic));
        }
        if (opt._n_basis < 1) {
            throw std::invalid_argument(std::format("auxiva_ip: n_basis must be >= 1, got {}", opt._n_basis));
        }
        if (!(std::isfinite(opt._eps) && opt._eps > 0.0)) {
            throw std::invalid_argument(std::format("auxiva_ip: eps must be finite and > 0, got {}", opt._eps));
        }
        if (!(std::isfinite(opt._load) && opt._load > 0.0)) {
            throw std::invalid_argument(std::format("auxiva_ip: load must be finite and > 0, got {}", opt._load));
        }
        torch::NoGradGuard no_grad;
        const int64_t n_src = opt._n_src == -1 ? n_chan : opt._n_src;
        const double eps = opt._eps;
        const double load = opt._load;

        auto w_shape = X.sizes().vec();
        w_shape.resize(w_shape.size() - 3);
        w_shape.insert(w_shape.end(), {n_freq, n_src, n_chan});
        auto W = torch::eye(n_src, n_chan, X.options()).expand(w_shape).clone(); // [..., n_freq, n_src, n_chan]
        const auto evec = torch::eye(n_chan, X.options());

        auto Y = X.slice(-3, 0, n_src).clone();
        const auto Cx = torch::einsum("...cfn,...dfn->...fcd", {X, X.conj()}) / n_frames;
        auto J = detail::orthogonal_constraint(W, Cx, load, eps);

        tensor_t nmf_T, nmf_V;
        for (int epoch = 0; epoch < opt._n_iter; ++epoch) {
            // the model sees Y before the per-source variance normalization below
            auto weights = detail::source_weights(Y, opt, nmf_T, nmf_V);
            const auto g = torch::clamp(detail::mag_sq(Y).mean({-2, -1}, true), eps);
            const auto g_sqrt = torch::sqrt(g);
            Y = Y / torch::clamp(g_sqrt, eps);
            W = W / torch::clamp(g_sqrt.transpose(-3, -2), eps);
            weights = weights * g;

            for (int64_t k = 0; k < n_src; ++k) {
                J = detail::orthogonal_constraint(W, Cx, load, eps);
                // weighted spatial covariance of source k [..., n_freq, n_chan, n_chan]
                const auto V = torch::einsum("...cfn,...dfn->...fcd", {X * weights.slice(-3, k, k + 1), X.conj()});
                const auto WV = J.defined() ? torch::cat({torch::matmul(W, V),
                                                          torch::matmul(J, V.slice(-2, 0, n_src)) - V.slice(-2, n_src)},
                                                         -2)
                                            : torch::matmul(W, V);
                auto new_w = detail::solve_loaded_general(WV, evec.slice(1, k, k + 1), load, eps).conj().select(-1, 0);
                const auto scale = torch::real(torch::einsum("...c,...d,...cd->...", {new_w, new_w.conj(), V}));
                new_w = (new_w / torch::sqrt(torch::clamp(scale.unsqueeze(-1), eps))).unsqueeze(-2);
                W = torch::cat({W.slice(-2, 0, k), new_w, W.slice(-2, k + 1, n_src)}, -2);
            }
            Y = torch::einsum("...fcd,...dfn->...cfn", {W, X});
        }

        if (opt._proj_back_mic >= 0) {
            Y = detail::projection_back(Y, W, J, opt._proj_back_mic, load);
        }
        return Y;
    }

    // ILRMA (NMF source model, rank opt._n_basis) with IP updates: auxiva_ip with model = nmf.
    inline auto ilrma(const_tensor_lref_t X, auxiva_option opt = {}) -> tensor_t {
        return auxiva_ip(X, opt.model(source_model::nmf));
    }
} // namespace torchmedia::audio::separation
#endif // LIB_TORCH_MEDIA_AUDIO_SEPARATION_HPP
