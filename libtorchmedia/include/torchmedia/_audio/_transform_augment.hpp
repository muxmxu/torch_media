#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_AUGMENT_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_AUGMENT_HPP
// Augmentation transform classes (torchaudio.transforms parity): FrequencyMasking, TimeMasking
// (over the _AxisMasking base), SpecAugment, and Fade. Masking delegates to the functional
// mask_along_axis(_iid) ops; Fade synthesizes its fade curves itself (no functional counterpart).
// See develop_log 2026-06-01/progress02 (D5 fade_shape enum, G5 RNG) and task22.
#include "_functional.hpp"

namespace torchmedia::audio::transform {
    namespace detail {
        // Mirrors torchaudio's _AxisMasking: the relative axis (1=freq, 2=time, defined for the 3D
        // case) is remapped to the absolute dim via `axis + dim - 3`, then dispatched on iid_masks.
        class AxisMasking {
        public:
            AxisMasking(int mask_param, int axis, bool iid_masks, double p = 1.0) :
                _mask_param(mask_param), _axis(axis), _iid_masks(iid_masks), _p(p) {}

            auto apply(const_tensor_lref_t specgram, double mask_value) const -> tensor_t {
                const int eff_axis = _axis + static_cast<int>(specgram.dim()) - 3;
                if (_iid_masks) {
                    return functional::mask_along_axis_iid(specgram, _mask_param, mask_value, eff_axis, _p);
                }
                return functional::mask_along_axis(specgram, _mask_param, mask_value, eff_axis, _p);
            }

        protected:
            int _mask_param;
            int _axis;
            bool _iid_masks;
            double _p;
        };
    } // namespace detail

    // ---------------- FrequencyMasking ----------------
    // API difference: the mask params (here and in time_masking_option / spec_augment_option) have defaults; torchaudio
    // requires them. Masks are drawn like torchaudio's, so a seed gives the same masks for any dtype/device.
    typedef struct frequency_masking_option {
        int _freq_mask_param = 80;
        bool _iid_masks = false;

        auto freq_mask_param(int n) -> frequency_masking_option & {
            _freq_mask_param = n;
            return *this;
        }
        auto iid_masks(bool b) -> frequency_masking_option & {
            _iid_masks = b;
            return *this;
        }
    } frequency_masking_option_t;

    class FrequencyMasking {
    public:
        explicit FrequencyMasking(frequency_masking_option opt = {}) :
            _mask(opt._freq_mask_param, /*axis=*/1, opt._iid_masks, /*p=*/1.0) {}

        auto operator()(const_tensor_lref_t specgram, double mask_value = 0.0) const -> tensor_t {
            return _mask.apply(specgram, mask_value);
        }

        auto forward(const_tensor_lref_t specgram, double mask_value = 0.0) const -> tensor_t {
            return (*this)(specgram, mask_value);
        }

    private:
        detail::AxisMasking _mask;
    };

    // ---------------- TimeMasking ----------------
    typedef struct time_masking_option {
        int _time_mask_param = 80;
        bool _iid_masks = false;
        double _p = 1.0;

        auto time_mask_param(int n) -> time_masking_option & {
            _time_mask_param = n;
            return *this;
        }
        auto iid_masks(bool b) -> time_masking_option & {
            _iid_masks = b;
            return *this;
        }
        auto p(double v) -> time_masking_option & {
            _p = v;
            return *this;
        }
    } time_masking_option_t;

    class TimeMasking {
    public:
        explicit TimeMasking(time_masking_option opt = {}) :
            _mask(opt._time_mask_param, /*axis=*/2, opt._iid_masks, opt._p) {
            if (!(opt._p >= 0.0 && opt._p <= 1.0)) {
                (void) handle_exceptions<int, std::invalid_argument>("TimeMasking: p must be between 0.0 and 1.0.");
            }
        }

        auto operator()(const_tensor_lref_t specgram, double mask_value = 0.0) const -> tensor_t {
            return _mask.apply(specgram, mask_value);
        }

        auto forward(const_tensor_lref_t specgram, double mask_value = 0.0) const -> tensor_t {
            return (*this)(specgram, mask_value);
        }

    private:
        detail::AxisMasking _mask;
    };

    // ---------------- SpecAugment ----------------
    typedef struct spec_augment_option {
        int _n_time_masks = 2;
        int _time_mask_param = 10;
        int _n_freq_masks = 2;
        int _freq_mask_param = 10;
        bool _iid_masks = true;
        double _p = 1.0;
        bool _zero_masking = false;

        auto n_time_masks(int n) -> spec_augment_option & {
            _n_time_masks = n;
            return *this;
        }
        auto time_mask_param(int n) -> spec_augment_option & {
            _time_mask_param = n;
            return *this;
        }
        auto n_freq_masks(int n) -> spec_augment_option & {
            _n_freq_masks = n;
            return *this;
        }
        auto freq_mask_param(int n) -> spec_augment_option & {
            _freq_mask_param = n;
            return *this;
        }
        auto iid_masks(bool b) -> spec_augment_option & {
            _iid_masks = b;
            return *this;
        }
        auto p(double v) -> spec_augment_option & {
            _p = v;
            return *this;
        }
        auto zero_masking(bool b) -> spec_augment_option & {
            _zero_masking = b;
            return *this;
        }
    } spec_augment_option_t;

    class SpecAugment {
    public:
        explicit SpecAugment(spec_augment_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t specgram) const -> tensor_t {
            const double mask_value = _opt._zero_masking ? 0.0 : specgram.mean().item<double>();
            const int time_dim = static_cast<int>(specgram.dim()) - 1;
            const int freq_dim = time_dim - 1;
            tensor_t out = specgram;
            if (specgram.dim() > 2 && _opt._iid_masks) {
                for (int i = 0; i < _opt._n_time_masks && (out.defined() || !specgram.defined()); ++i) {
                    out = functional::mask_along_axis_iid(out, _opt._time_mask_param, mask_value, time_dim, _opt._p);
                }
                for (int i = 0; i < _opt._n_freq_masks && (out.defined() || !specgram.defined()); ++i) {
                    out = functional::mask_along_axis_iid(out, _opt._freq_mask_param, mask_value, freq_dim, _opt._p);
                }
            } else {
                for (int i = 0; i < _opt._n_time_masks && (out.defined() || !specgram.defined()); ++i) {
                    out = functional::mask_along_axis(out, _opt._time_mask_param, mask_value, time_dim, _opt._p);
                }
                for (int i = 0; i < _opt._n_freq_masks && (out.defined() || !specgram.defined()); ++i) {
                    out = functional::mask_along_axis(out, _opt._freq_mask_param, mask_value, freq_dim, _opt._p);
                }
            }
            return out;
        }

        auto forward(const_tensor_lref_t specgram) const -> tensor_t { return (*this)(specgram); }

    private:
        spec_augment_option _opt;
    };

    // ---------------- Fade ----------------
    enum class fade_shape { linear, exponential, logarithmic, quarter_sine, half_sine };

    typedef struct fade_option {
        int _fade_in_len = 0;
        int _fade_out_len = 0;
        fade_shape _fade_shape = fade_shape::linear;

        auto fade_in_len(int n) -> fade_option & {
            _fade_in_len = n;
            return *this;
        }
        auto fade_out_len(int n) -> fade_option & {
            _fade_out_len = n;
            return *this;
        }
        auto shape(fade_shape s) -> fade_option & {
            _fade_shape = s;
            return *this;
        }
    } fade_option_t;

    class Fade {
    public:
        explicit Fade(fade_option opt = {}) : _opt(opt) {}

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            const int64_t length = waveform.size(-1);
            const auto options = tensor_options_t().dtype(torch::kFloat).device(waveform.device());
            return _fade_in(length, options) * _fade_out(length, options) * waveform;
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        static constexpr double pi() { return 3.14159265358979323846; }

        auto _fade_in(int64_t length, const tensor_options_t &options) const -> tensor_t {
            auto fade = torch::linspace(0, 1, _opt._fade_in_len, options);
            const auto ones = torch::ones(length - _opt._fade_in_len, options);
            switch (_opt._fade_shape) {
                case fade_shape::exponential:
                    fade = torch::pow(2, fade - 1) * fade;
                    break;
                case fade_shape::logarithmic:
                    fade = torch::log10(0.1 + fade) + 1;
                    break;
                case fade_shape::quarter_sine:
                    fade = torch::sin(fade * pi() / 2);
                    break;
                case fade_shape::half_sine:
                    fade = torch::sin(fade * pi() - pi() / 2) / 2 + 0.5;
                    break;
                case fade_shape::linear:
                default:
                    break;
            }
            return torch::cat({fade, ones}).clamp_(0, 1);
        }

        auto _fade_out(int64_t length, const tensor_options_t &options) const -> tensor_t {
            auto fade = torch::linspace(0, 1, _opt._fade_out_len, options);
            const auto ones = torch::ones(length - _opt._fade_out_len, options);
            switch (_opt._fade_shape) {
                case fade_shape::exponential:
                    fade = torch::pow(2, -fade) * (1 - fade);
                    break;
                case fade_shape::logarithmic:
                    fade = torch::log10(1.1 - fade) + 1;
                    break;
                case fade_shape::quarter_sine:
                    fade = torch::sin(fade * pi() / 2 + pi() / 2);
                    break;
                case fade_shape::half_sine:
                    fade = torch::sin(fade * pi() + pi() / 2) / 2 + 0.5;
                    break;
                case fade_shape::linear:
                default:
                    fade = -fade + 1;
                    break;
            }
            return torch::cat({ones, fade}).clamp_(0, 1);
        }

        fade_option _opt;
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_AUGMENT_HPP
