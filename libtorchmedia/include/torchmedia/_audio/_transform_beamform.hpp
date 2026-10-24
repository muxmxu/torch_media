#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_BEAMFORM_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_BEAMFORM_HPP
// Multi-channel beamforming transform classes (torchaudio.transforms parity): PSD, SoudenMVDR,
// RTFMVDR, and the full MVDR (incl. the online recursive PSD accumulation). MVDR is the only class
// in the layer with mutable cross-call state, so its call operator is non-const (see develop_log
// 2026-06-01/progress02 D4/G2/G7 and task25).
#include "_functional.hpp"
#include "_functional_beamforming.hpp"

namespace torchmedia::audio::transform {
    // ---------------- PSD ----------------
    typedef struct psd_option {
        bool _multi_mask = false;
        bool _normalize = true;
        double _eps = 1e-15; // torchaudio PSD default (functional::psd defaults to 1e-10)

        auto multi_mask(bool b) -> psd_option & {
            _multi_mask = b;
            return *this;
        }
        auto normalize(bool b) -> psd_option & {
            _normalize = b;
            return *this;
        }
        auto eps(double e) -> psd_option & {
            _eps = e;
            return *this;
        }
    } psd_option_t;

    class PSD {
    public:
        explicit PSD(psd_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t specgram, const c10::optional<tensor_t> &mask = c10::nullopt) const
                -> tensor_t {
            c10::optional<tensor_t> m = mask;
            if (m.has_value() && _opt._multi_mask) {
                m = m.value().mean(-3); // average a (..., channel, freq, time) mask over channels
            }
            return functional::psd(specgram, m, _opt._normalize, _opt._eps);
        }

        auto forward(const_tensor_lref_t specgram, const c10::optional<tensor_t> &mask = c10::nullopt) const
                -> tensor_t {
            return (*this)(specgram, mask);
        }

    private:
        psd_option _opt;
    };

    // ---------------- SoudenMVDR ----------------
    class SoudenMVDR {
    public:
        SoudenMVDR() = default;
        // reference_channel: an int64_t index or a one-hot Tensor (..., channel), as torchaudio's Union[int, Tensor].
        template<class Ref>
        auto operator()(const_tensor_lref_t specgram, const_tensor_lref_t psd_s, const_tensor_lref_t psd_n,
                        const Ref &reference_channel, bool diagonal_loading = true, double diag_eps = 1e-7,
                        double eps = 1e-8) const -> tensor_t {
            const auto w =
                    functional::mvdr_weights_souden(psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps);
            return w.defined() ? functional::apply_beamforming(w, specgram) : w; // undefined: rejected (no-exceptions)
        }
        template<class Ref>
        auto forward(const_tensor_lref_t specgram, const_tensor_lref_t psd_s, const_tensor_lref_t psd_n,
                     const Ref &reference_channel, bool diagonal_loading = true, double diag_eps = 1e-7,
                     double eps = 1e-8) const -> tensor_t {
            return (*this)(specgram, psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps);
        }
    };

    // ---------------- RTFMVDR ----------------
    class RTFMVDR {
    public:
        RTFMVDR() = default;
        // reference_channel: an int64_t index or a one-hot Tensor (..., channel), as torchaudio's Union[int, Tensor].
        template<class Ref>
        auto operator()(const_tensor_lref_t specgram, const_tensor_lref_t rtf, const_tensor_lref_t psd_n,
                        const Ref &reference_channel, bool diagonal_loading = true, double diag_eps = 1e-7,
                        double eps = 1e-8) const -> tensor_t {
            const auto w = functional::mvdr_weights_rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps);
            return w.defined() ? functional::apply_beamforming(w, specgram) : w; // undefined: rejected (no-exceptions)
        }
        template<class Ref>
        auto forward(const_tensor_lref_t specgram, const_tensor_lref_t rtf, const_tensor_lref_t psd_n,
                     const Ref &reference_channel, bool diagonal_loading = true, double diag_eps = 1e-7,
                     double eps = 1e-8) const -> tensor_t {
            return (*this)(specgram, rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps);
        }
    };

    // ---------------- MVDR (incl. online recursive PSD; mutable state) ----------------
    enum class mvdr_solution { ref_channel, stv_evd, stv_power };

    typedef struct mvdr_option {
        int _ref_channel = 0;
        mvdr_solution _solution = mvdr_solution::ref_channel;
        bool _multi_mask = false;
        bool _diag_loading = true;
        double _diag_eps = 1e-7;
        bool _online = false;

        auto ref_channel(int c) -> mvdr_option & {
            _ref_channel = c;
            return *this;
        }
        auto solution(mvdr_solution s) -> mvdr_option & {
            _solution = s;
            return *this;
        }
        auto multi_mask(bool b) -> mvdr_option & {
            _multi_mask = b;
            return *this;
        }
        auto diag_loading(bool b) -> mvdr_option & {
            _diag_loading = b;
            return *this;
        }
        auto diag_eps(double e) -> mvdr_option & {
            _diag_eps = e;
            return *this;
        }
        auto online(bool b) -> mvdr_option & {
            _online = b;
            return *this;
        }
    } mvdr_option_t;

    class MVDR {
    public:
        explicit MVDR(mvdr_option opt = {}) : _opt(opt), _psd(psd_option().multi_mask(opt._multi_mask)) {}

        // Non-const: online mode accumulates the running PSDs across calls (G2).
        auto operator()(const_tensor_lref_t specgram, const_tensor_lref_t mask_s,
                        const c10::optional<tensor_t> &mask_n_opt = c10::nullopt) -> tensor_t {
            const auto dtype = specgram.scalar_type();
            if (specgram.dim() < 3) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        "MVDR: expected at least a 3D tensor (..., channel, freq, time).");
            }
            if (!specgram.is_complex()) {
                return handle_exceptions<tensor_t, std::invalid_argument>("MVDR: specgram must be a complex tensor.");
            }
            // Promote cfloat -> cdouble for internal numerical stability (G7).
            const tensor_t spec = dtype == torch::kComplexFloat ? specgram.to(torch::kComplexDouble) : specgram;
            const tensor_t mask_n = mask_n_opt.has_value() ? mask_n_opt.value() : (1.0 - mask_s);

            const auto psd_s = _psd(spec, mask_s);
            const auto psd_n = _psd(spec, mask_n);
            if (!psd_s.defined() || !psd_n.defined()) {
                return {}; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: PSD rejected a mask
            }

            // One-hot reference vector u over the leading (..., channel) dims.
            auto lead = spec.sizes().vec();
            lead.pop_back(); // drop time
            lead.pop_back(); // drop freq
            auto u = torch::zeros(lead, tensor_options_t().dtype(torch::kComplexDouble).device(spec.device()));
            u.select(-1, _opt._ref_channel).fill_(1);

            const tensor_t w = _opt._online ? get_updated_mvdr_vector(psd_s, psd_n, mask_s, mask_n, u)
                                            : get_mvdr_vector(psd_s, psd_n, u);
            return functional::apply_beamforming(w, spec).to(dtype);
        }

        auto forward(const_tensor_lref_t specgram, const_tensor_lref_t mask_s,
                     const c10::optional<tensor_t> &mask_n_opt = c10::nullopt) -> tensor_t {
            return (*this)(specgram, mask_s, mask_n_opt);
        }

    private:
        // Dispatch on `solution` (mirrors torchaudio's module-level _get_mvdr_vector).
        auto get_mvdr_vector(const tensor_t &psd_s, const tensor_t &psd_n, const tensor_t &u) const -> tensor_t {
            constexpr double eps = 1e-8;
            if (_opt._solution == mvdr_solution::ref_channel) {
                return functional::mvdr_weights_souden(psd_s, psd_n, u, _opt._diag_loading, _opt._diag_eps, eps);
            }
            tensor_t stv;
            if (_opt._solution == mvdr_solution::stv_evd) {
                stv = functional::rtf_evd(psd_s);
            } else { // stv_power
                stv = functional::rtf_power(psd_s, psd_n, u, 3, _opt._diag_loading, _opt._diag_eps); // torchaudio: u
            }
            return functional::mvdr_weights_rtf(stv, psd_n, u, _opt._diag_loading, _opt._diag_eps, eps);
        }

        // Recursive online PSD update (higuchi2017online): blends the new frame into the running PSD.
        static auto update_psd(const tensor_t &old_psd, const tensor_t &mask_sum, const tensor_t &new_psd,
                               const tensor_t &mask) -> tensor_t {
            const auto denom = mask_sum + mask.sum(-1);
            const auto numerator = (mask_sum / denom).unsqueeze(-1).unsqueeze(-1);
            const auto denominator = (1.0 / denom).unsqueeze(-1).unsqueeze(-1);
            return old_psd * numerator + new_psd * denominator;
        }

        auto get_updated_mvdr_vector(tensor_t psd_s, tensor_t psd_n, tensor_t mask_s, tensor_t mask_n,
                                     const tensor_t &u) -> tensor_t {
            if (_opt._multi_mask) {
                mask_s = mask_s.mean(-3);
                mask_n = mask_n.mean(-3);
            }
            if (!_initialized) {
                _psd_s = psd_s;
                _psd_n = psd_n;
                _mask_sum_s = mask_s.sum(-1);
                _mask_sum_n = mask_n.sum(-1);
                _initialized = true;
                return get_mvdr_vector(psd_s, psd_n, u);
            }
            psd_s = update_psd(_psd_s, _mask_sum_s, psd_s, mask_s);
            psd_n = update_psd(_psd_n, _mask_sum_n, psd_n, mask_n);
            _psd_s = psd_s;
            _psd_n = psd_n;
            _mask_sum_s = _mask_sum_s + mask_s.sum(-1);
            _mask_sum_n = _mask_sum_n + mask_n.sum(-1);
            return get_mvdr_vector(psd_s, psd_n, u);
        }

        mvdr_option _opt;
        PSD _psd;
        bool _initialized = false;
        tensor_t _psd_s;
        tensor_t _psd_n;
        tensor_t _mask_sum_s;
        tensor_t _mask_sum_n;
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_BEAMFORM_HPP
