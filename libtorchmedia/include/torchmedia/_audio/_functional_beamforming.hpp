#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
#include <torch/torch.h>
#include "../global_include.hpp"

// Multichannel beamforming (mirrors the MVDR / RTF ops in torchaudio.functional). All operate on
// COMPLEX tensors and use torch::linalg_solve / linalg_eigh. reference_channel is modelled with int and
// Tensor (one-hot) overloads.
namespace torchmedia::audio::functional {
    namespace detail {
        // torchaudio's psd_s / psd_n checks. Returns an error message or "" (callers return handle_exceptions, so the
        // LIB_TORCH_MEDIA_NO_EXCEPTIONS build stops too).
        inline auto psd_matrices_error(const_tensor_lref_t psd_s, const_tensor_lref_t psd_n) -> str_t {
            if (psd_s.dim() < 3 || psd_n.dim() < 3) {
                return "Expected at least 3D (..., freq, channel, channel) for psd_s/psd_n.";
            }
            if (!(psd_s.is_complex() && psd_n.is_complex())) {
                return "psd_s and psd_n must be complex.";
            }
            if (psd_s.sizes() != psd_n.sizes()) {
                return "psd_s and psd_n must have the same shape.";
            }
            if (psd_s.size(-1) != psd_s.size(-2)) {
                return "The last two dimensions of psd_s must match.";
            }
            return {};
        }

        // torchaudio mvdr_weights_rtf's checks: rtf (..., freq, channel), psd_n (..., freq, channel, channel).
        inline auto rtf_psd_n_error(const_tensor_lref_t rtf, const_tensor_lref_t psd_n) -> str_t {
            if (rtf.dim() < 2 || psd_n.dim() < 3) {
                return "Expected at least 2D rtf (..., freq, channel) and 3D psd_n (..., freq, channel, channel).";
            }
            if (!(rtf.is_complex() && psd_n.is_complex())) {
                return "rtf and psd_n must be complex.";
            }
            if (rtf.sizes() != psd_n.sizes().slice(0, psd_n.dim() - 1)) {
                return "The dimensions of rtf and of psd_n without its last dimension must be the same.";
            }
            if (psd_n.size(-1) != psd_n.size(-2)) {
                return "The last two dimensions of psd_n must match.";
            }
            return {};
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
        if (mask.has_value() &&
            (mask->dim() < 1 ||
             mask->sizes().slice(0, mask->dim() - 1) != specgram.sizes().slice(0, specgram.dim() - 2) ||
             mask->size(-1) != specgram.size(-1))) { // e.g. a per-channel mask: PSD(multi_mask)
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "The dimensions of mask except the channel dimension should be the same as "
                    "specgram: expected (..., freq, time).");
        }
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
        if (const auto err = detail::psd_matrices_error(psd_s, psd_n); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg_solve(psd_n, psd_s, /*left=*/true);
        const auto ws = numerator / (_compute_mat_trace(numerator).index({Ellipsis, None, None}) + eps);
        return ws.index({Ellipsis, Slice(), reference_channel});
    }

    // MVDR weights, Souden formulation. reference_channel as a one-hot Tensor.
    inline auto mvdr_weights_souden(const_tensor_lref_t psd_s, tensor_t psd_n, const_tensor_lref_t reference_channel,
                                    bool diagonal_loading = true, double diag_eps = 1e-7, double eps = 1e-8)
            -> tensor_t {
        using namespace torch::indexing;
        if (const auto err = detail::psd_matrices_error(psd_s, psd_n); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg_solve(psd_n, psd_s, /*left=*/true);
        const auto ws = numerator / (_compute_mat_trace(numerator).index({Ellipsis, None, None}) + eps);
        const auto ref = reference_channel.to(psd_n.scalar_type());
        return torch::einsum("...c,...c->...", {ws, ref.index({Ellipsis, None, None, Slice()})});
    }

    // MVDR weights from an RTF/steering vector (mirrors mvdr_weights_rtf). reference_channel optional int.
    inline auto mvdr_weights_rtf(const_tensor_lref_t rtf, tensor_t psd_n,
                                 c10::optional<int64_t> reference_channel = c10::nullopt, bool diagonal_loading = true,
                                 double diag_eps = 1e-7, double eps = 1e-8) -> tensor_t {
        using namespace torch::indexing;
        if (const auto err = detail::rtf_psd_n_error(rtf, psd_n); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg_solve(psd_n, rtf.unsqueeze(-1), /*left=*/true).squeeze(-1);
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
        if (const auto err = detail::rtf_psd_n_error(rtf, psd_n); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (diagonal_loading) {
            psd_n = _tik_reg(psd_n, diag_eps);
        }
        const auto numerator = torch::linalg_solve(psd_n, rtf.unsqueeze(-1), /*left=*/true).squeeze(-1);
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
            return handle_exceptions<tensor_t, std::invalid_argument>("psd_s must be complex.");
        }
        if (psd_s.size(-1) != psd_s.size(-2)) {
            return handle_exceptions<tensor_t, std::invalid_argument>("The last two dimensions of psd_s must match.");
        }
        const auto eig = torch::linalg_eigh(psd_s, "L");
        const auto v = std::get<1>(eig); // eigenvectors (columns), ascending eigenvalues
        return v.index({Ellipsis, -1}); // principal eigenvector (max eigenvalue)
    }

    namespace detail {
        // rtf_power for either reference_channel form: `column` takes phi's reference column (..., freq, channel).
        template<class Column>
        inline auto rtf_power(const_tensor_lref_t psd_s, tensor_t psd_n, const Column &column, int n_iter,
                              bool diagonal_loading, double diag_eps) -> tensor_t {
            if (const auto err = psd_matrices_error(psd_s, psd_n); !err.empty()) {
                return handle_exceptions<tensor_t, std::invalid_argument>(err);
            }
            if (n_iter <= 0) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        "The number of iterations must be greater than 0.");
            }
            if (diagonal_loading) {
                psd_n = _tik_reg(psd_n, diag_eps);
            }
            const auto phi = torch::linalg_solve(psd_n, psd_s, /*left=*/true);
            auto rtf = column(phi).unsqueeze(-1); // (..., freq, channel, 1)
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
    } // namespace detail

    // RTF/steering vector by the power method (mirrors rtf_power). reference_channel as int.
    inline auto rtf_power(const_tensor_lref_t psd_s, const_tensor_lref_t psd_n, int64_t reference_channel,
                          int n_iter = 3, bool diagonal_loading = true, double diag_eps = 1e-7) -> tensor_t {
        return detail::rtf_power(
                psd_s, psd_n,
                [&](const tensor_t &phi) { return phi.index({torch::indexing::Ellipsis, reference_channel}); }, n_iter,
                diagonal_loading, diag_eps);
    }

    // RTF/steering vector by the power method, reference_channel as a one-hot Tensor (..., channel).
    inline auto rtf_power(const_tensor_lref_t psd_s, const_tensor_lref_t psd_n, const_tensor_lref_t reference_channel,
                          int n_iter = 3, bool diagonal_loading = true, double diag_eps = 1e-7) -> tensor_t {
        using namespace torch::indexing;
        return detail::rtf_power(
                psd_s, psd_n,
                [&](const tensor_t &phi) {
                    const auto ref = reference_channel.to(phi.scalar_type());
                    return torch::einsum("...c,...c->...", {phi, ref.index({Ellipsis, None, None, Slice()})});
                },
                n_iter, diagonal_loading, diag_eps);
    }

    // Apply precomputed beamforming weights (mirrors apply_beamforming).
    inline auto apply_beamforming(const_tensor_lref_t beamform_weights, const_tensor_lref_t specgram) -> tensor_t {
        if (beamform_weights.dim() < 2 || specgram.dim() < 3 ||
            beamform_weights.sizes().slice(0, beamform_weights.dim() - 2) !=
                    specgram.sizes().slice(0, specgram.dim() - 3)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "The dimensions except the last two of beamform_weights should be the same as "
                    "the dimensions except the last three of specgram.");
        }
        if (!(beamform_weights.is_complex() && specgram.is_complex())) {
            return handle_exceptions<tensor_t, std::invalid_argument>("beamform_weights and specgram must be complex.");
        }
        return torch::einsum("...fc,...cft->...ft", {beamform_weights.conj(), specgram});
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_BEAMFORMING_HPP
