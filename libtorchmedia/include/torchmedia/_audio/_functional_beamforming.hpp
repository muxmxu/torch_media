#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
#include <torch/torch.h>
#include "../globel_include.hpp"

// Multichannel beamforming (mirrors the MVDR / RTF ops in torchaudio.functional). All operate on
// COMPLEX tensors and use torch::linalg::solve / eigh. reference_channel is modelled with int and
// Tensor (one-hot) overloads.
namespace torchmedia::audio::functional {
    namespace detail {
        inline auto assert_psd_matrices(const_tensor_lref_t psd_s, const_tensor_lref_t psd_n) -> void {
            if (psd_s.dim() < 3 || psd_n.dim() < 3) {
                handle_exceptions<tensor_t, std::invalid_argument>(
                        torch::empty({1}), "Expected at least 3D (..., freq, channel, channel) for psd_s/psd_n.");
            }
            if (!(psd_s.is_complex() && psd_n.is_complex())) {
                handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                   "psd_s and psd_n must be complex.");
            }
            if (psd_s.sizes() != psd_n.sizes()) {
                handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                   "psd_s and psd_n must have the same shape.");
            }
            if (psd_s.size(-1) != psd_s.size(-2)) {
                handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                   "The last two dimensions of psd_s must match.");
            }
        }
    } // namespace detail

    // Trace along the last two (channel, channel) dims.
    inline auto _compute_mat_trace(const_tensor_lref_t input, int64_t dim1 = -1, int64_t dim2 = -2) -> tensor_t {
        return torch::diagonal(input, 0, dim1, dim2).sum(-1);
    }

    // Tikhonov regularisation: add (reg * trace.real + eps) on the diagonal.
    inline auto _tik_reg(const_tensor_lref_t mat, double reg = 1e-7, double eps = 1e-8) -> tensor_t {
        using namespace torch::indexing;
        const int64_t C = mat.size(-1);
        const auto eye = torch::eye(C, mat.options());
        auto epsilon = torch::real(_compute_mat_trace(mat)).index({Ellipsis, None, None}) * reg;
        epsilon = epsilon + eps;
        return mat + epsilon * eye;
    }

    // Cross-channel PSD matrix (mirrors torchaudio.functional.psd).
    inline auto psd(const_tensor_lref_t specgram_in, const c10::optional<tensor_t> &mask = c10::nullopt,
                    bool normalize = true, double eps = 1e-10) -> tensor_t {
        using namespace torch::indexing;
        const auto specgram = specgram_in.transpose(-3, -2); // (..., freq, channel, time)
        auto psd_mat = torch::einsum("...ct,...et->...tce", {specgram, specgram.conj()});
        if (mask.has_value()) {
            auto m = mask.value();
            if (normalize) {
                m = m / (m.sum(-1, /*keepdim=*/true) + eps);
            }
            psd_mat = psd_mat * m.index({Ellipsis, None, None});
        }
        return psd_mat.sum(-3);
    }

    // MVDR weights, Souden formulation (mirrors mvdr_weights_souden). reference_channel as int.
    inline auto mvdr_weights_souden(const_tensor_lref_t psd_s, tensor_t psd_n, int64_t reference_channel,
                                    bool diagonal_loading = true, double diag_eps = 1e-7, double eps = 1e-8)
            -> tensor_t {
        using namespace torch::indexing;
        detail::assert_psd_matrices(psd_s, psd_n);
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg::solve(psd_n, psd_s, /*left=*/true);
        const auto ws = numerator / (_compute_mat_trace(numerator).index({Ellipsis, None, None}) + eps);
        return ws.index({Ellipsis, Slice(), reference_channel});
    }

    // MVDR weights, Souden formulation. reference_channel as a one-hot Tensor.
    inline auto mvdr_weights_souden(const_tensor_lref_t psd_s, tensor_t psd_n, const_tensor_lref_t reference_channel,
                                    bool diagonal_loading = true, double diag_eps = 1e-7, double eps = 1e-8)
            -> tensor_t {
        using namespace torch::indexing;
        detail::assert_psd_matrices(psd_s, psd_n);
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg::solve(psd_n, psd_s, /*left=*/true);
        const auto ws = numerator / (_compute_mat_trace(numerator).index({Ellipsis, None, None}) + eps);
        const auto ref = reference_channel.to(psd_n.scalar_type());
        return torch::einsum("...c,...c->...", {ws, ref.index({Ellipsis, None, None, Slice()})});
    }

    // MVDR weights from an RTF/steering vector (mirrors mvdr_weights_rtf). reference_channel optional int.
    inline auto mvdr_weights_rtf(const_tensor_lref_t rtf, tensor_t psd_n,
                                 c10::optional<int64_t> reference_channel = c10::nullopt, bool diagonal_loading = true,
                                 double diag_eps = 1e-7, double eps = 1e-8) -> tensor_t {
        using namespace torch::indexing;
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg::solve(psd_n, rtf.unsqueeze(-1), /*left=*/true).squeeze(-1);
        const auto denominator = torch::einsum("...d,...d->...", {rtf.conj(), numerator});
        auto beamform_weights = numerator / (torch::real(denominator).unsqueeze(-1) + eps);
        if (reference_channel.has_value()) {
            const auto scale = rtf.index({Ellipsis, reference_channel.value()}).conj();
            beamform_weights = beamform_weights * scale.unsqueeze(-1);
        }
        return beamform_weights;
    }

    // MVDR weights from an RTF, reference_channel as a one-hot Tensor.
    inline auto mvdr_weights_rtf(const_tensor_lref_t rtf, tensor_t psd_n, const_tensor_lref_t reference_channel,
                                 bool diagonal_loading = true, double diag_eps = 1e-7, double eps = 1e-8) -> tensor_t {
        using namespace torch::indexing;
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg::solve(psd_n, rtf.unsqueeze(-1), /*left=*/true).squeeze(-1);
        const auto denominator = torch::einsum("...d,...d->...", {rtf.conj(), numerator});
        auto beamform_weights = numerator / (torch::real(denominator).unsqueeze(-1) + eps);
        const auto ref = reference_channel.to(psd_n.scalar_type());
        const auto scale = torch::einsum("...c,...c->...", {rtf.conj(), ref.index({Ellipsis, None, Slice()})});
        return beamform_weights * scale.unsqueeze(-1);
    }

    // RTF/steering vector by eigenvalue decomposition (mirrors rtf_evd).
    inline auto rtf_evd(const_tensor_lref_t psd_s) -> tensor_t {
        using namespace torch::indexing;
        if (!psd_s.is_complex()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), "psd_s must be complex.");
        }
        if (psd_s.size(-1) != psd_s.size(-2)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                      "The last two dimensions of psd_s must match.");
        }
        const auto eig = torch::linalg::eigh(psd_s, "L");
        const auto v = std::get<1>(eig); // eigenvectors (columns), ascending eigenvalues
        return v.index({Ellipsis, -1}); // principal eigenvector (max eigenvalue)
    }

    // RTF/steering vector by the power method (mirrors rtf_power). reference_channel as int.
    inline auto rtf_power(const_tensor_lref_t psd_s, tensor_t psd_n, int64_t reference_channel, int n_iter = 3,
                          bool diagonal_loading = true, double diag_eps = 1e-7) -> tensor_t {
        using namespace torch::indexing;
        detail::assert_psd_matrices(psd_s, psd_n);
        if (n_iter <= 0) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    torch::empty({1}), "The number of iterations must be greater than 0.");
        }
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto phi = torch::linalg::solve(psd_n, psd_s, /*left=*/true);
        auto rtf = phi.index({Ellipsis, reference_channel}).unsqueeze(-1); // (..., freq, channel, 1)
        if (n_iter >= 2) {
            for (int k = 0; k < n_iter - 2; ++k) {
                rtf = torch::matmul(phi, rtf);
            }
            rtf = torch::matmul(psd_s, rtf);
        } else {
            rtf = torch::matmul(psd_n, rtf);
        }
        return rtf.squeeze(-1);
    }

    // Apply precomputed beamforming weights (mirrors apply_beamforming).
    inline auto apply_beamforming(const_tensor_lref_t beamform_weights, const_tensor_lref_t specgram) -> tensor_t {
        if (!(beamform_weights.is_complex() && specgram.is_complex())) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                      "beamform_weights and specgram must be complex.");
        }
        return torch::einsum("...fc,...cft->...ft", {beamform_weights.conj(), specgram});
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
