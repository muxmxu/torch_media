#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_EFFECTS_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_EFFECTS_HPP
// Companding / emphasis / feature / amplitude transform classes (torchaudio.transforms parity):
// MuLawEncoding, MuLawDecoding, Preemphasis, Deemphasis, ComputeDeltas, SlidingWindowCmn, Loudness,
// Vad (thin config holders over functional ops) and Vol (gain-mode dispatch + clamp). See
// develop_log 2026-06-01/progress02 (D5 gain_type enum) and task23.
#include <cmath>
#include <optional>
#include "_functional.hpp"
#include "_functional_filtering.hpp"

namespace torchmedia::audio::transform {
    // ---------------- MuLawEncoding ----------------
    typedef struct mu_law_encoding_option {
        int _quantization_channels = 256;
        auto quantization_channels(int q) -> mu_law_encoding_option & {
            _quantization_channels = q;
            return *this;
        }
    } mu_law_encoding_option_t;

    class MuLawEncoding {
    public:
        explicit MuLawEncoding(mu_law_encoding_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t x) const -> tensor_t {
            return functional::mu_law_encoding(x, _opt._quantization_channels);
        }
        auto forward(const_tensor_lref_t x) const -> tensor_t { return (*this)(x); }

    private:
        mu_law_encoding_option _opt;
    };

    // ---------------- MuLawDecoding ----------------
    typedef struct mu_law_decoding_option {
        int _quantization_channels = 256;
        auto quantization_channels(int q) -> mu_law_decoding_option & {
            _quantization_channels = q;
            return *this;
        }
    } mu_law_decoding_option_t;

    class MuLawDecoding {
    public:
        explicit MuLawDecoding(mu_law_decoding_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t x_mu) const -> tensor_t {
            return functional::mu_law_decoding(x_mu, _opt._quantization_channels);
        }
        auto forward(const_tensor_lref_t x_mu) const -> tensor_t { return (*this)(x_mu); }

    private:
        mu_law_decoding_option _opt;
    };

    // ---------------- Preemphasis ----------------
    typedef struct preemphasis_option {
        double _coeff = 0.97;
        auto coeff(double c) -> preemphasis_option & {
            _coeff = c;
            return *this;
        }
    } preemphasis_option_t;

    class Preemphasis {
    public:
        explicit Preemphasis(preemphasis_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            return functional::preemphasis(waveform, _opt._coeff);
        }
        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        preemphasis_option _opt;
    };

    // ---------------- Deemphasis ----------------
    typedef struct deemphasis_option {
        double _coeff = 0.97;
        auto coeff(double c) -> deemphasis_option & {
            _coeff = c;
            return *this;
        }
    } deemphasis_option_t;

    class Deemphasis {
    public:
        explicit Deemphasis(deemphasis_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            return functional::deemphasis(waveform, _opt._coeff);
        }
        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        deemphasis_option _opt;
    };

    // ---------------- ComputeDeltas ----------------
    typedef struct compute_deltas_option {
        int _win_length = 5;
        std::string _mode = "replicate";
        auto win_length(int w) -> compute_deltas_option & {
            _win_length = w;
            return *this;
        }
        auto mode(const std::string &m) -> compute_deltas_option & {
            _mode = m;
            return *this;
        }
    } compute_deltas_option_t;

    class ComputeDeltas {
    public:
        explicit ComputeDeltas(compute_deltas_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t specgram) const -> tensor_t {
            return functional::compute_deltas(specgram, _opt._win_length, _opt._mode);
        }
        auto forward(const_tensor_lref_t specgram) const -> tensor_t { return (*this)(specgram); }

    private:
        compute_deltas_option _opt;
    };

    // ---------------- SlidingWindowCmn ----------------
    typedef struct sliding_window_cmn_option {
        int _cmn_window = 600;
        int _min_cmn_window = 100;
        bool _center = false;
        bool _norm_vars = false;
        auto cmn_window(int w) -> sliding_window_cmn_option & {
            _cmn_window = w;
            return *this;
        }
        auto min_cmn_window(int w) -> sliding_window_cmn_option & {
            _min_cmn_window = w;
            return *this;
        }
        auto center(bool c) -> sliding_window_cmn_option & {
            _center = c;
            return *this;
        }
        auto norm_vars(bool n) -> sliding_window_cmn_option & {
            _norm_vars = n;
            return *this;
        }
    } sliding_window_cmn_option_t;

    class SlidingWindowCmn {
    public:
        explicit SlidingWindowCmn(sliding_window_cmn_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t specgram) const -> tensor_t {
            return functional::sliding_window_cmn(specgram, _opt._cmn_window, _opt._min_cmn_window, _opt._center,
                                                  _opt._norm_vars);
        }
        auto forward(const_tensor_lref_t specgram) const -> tensor_t { return (*this)(specgram); }

    private:
        sliding_window_cmn_option _opt;
    };

    // ---------------- Loudness ----------------
    typedef struct loudness_option {
        int _sample_rate = 16000;
        auto sample_rate(int s) -> loudness_option & {
            _sample_rate = s;
            return *this;
        }
    } loudness_option_t;

    class Loudness {
    public:
        explicit Loudness(loudness_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            return functional::loudness(waveform, _opt._sample_rate);
        }
        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        loudness_option _opt;
    };

    // ---------------- Vad ----------------
    typedef struct vad_option {
        int _sample_rate = 16000;
        double _trigger_level = 7.0;
        double _trigger_time = 0.25;
        double _search_time = 1.0;
        double _allowed_gap = 0.25;
        double _pre_trigger_time = 0.0;
        double _boot_time = 0.35;
        double _noise_up_time = 0.1;
        double _noise_down_time = 0.01;
        double _noise_reduction_amount = 1.35;
        double _measure_freq = 20.0;
        std::optional<double> _measure_duration = std::nullopt;
        double _measure_smooth_time = 0.4;
        double _hp_filter_freq = 50.0;
        double _lp_filter_freq = 6000.0;
        double _hp_lifter_freq = 150.0;
        double _lp_lifter_freq = 2000.0;

        auto sample_rate(int s) -> vad_option & {
            _sample_rate = s;
            return *this;
        }
        auto trigger_level(double v) -> vad_option & {
            _trigger_level = v;
            return *this;
        }
        auto trigger_time(double v) -> vad_option & {
            _trigger_time = v;
            return *this;
        }
        auto search_time(double v) -> vad_option & {
            _search_time = v;
            return *this;
        }
        auto allowed_gap(double v) -> vad_option & {
            _allowed_gap = v;
            return *this;
        }
        auto pre_trigger_time(double v) -> vad_option & {
            _pre_trigger_time = v;
            return *this;
        }
        auto boot_time(double v) -> vad_option & {
            _boot_time = v;
            return *this;
        }
        auto noise_up_time(double v) -> vad_option & {
            _noise_up_time = v;
            return *this;
        }
        auto noise_down_time(double v) -> vad_option & {
            _noise_down_time = v;
            return *this;
        }
        auto noise_reduction_amount(double v) -> vad_option & {
            _noise_reduction_amount = v;
            return *this;
        }
        auto measure_freq(double v) -> vad_option & {
            _measure_freq = v;
            return *this;
        }
        auto measure_duration(double v) -> vad_option & {
            _measure_duration = v;
            return *this;
        }
        auto measure_smooth_time(double v) -> vad_option & {
            _measure_smooth_time = v;
            return *this;
        }
        auto hp_filter_freq(double v) -> vad_option & {
            _hp_filter_freq = v;
            return *this;
        }
        auto lp_filter_freq(double v) -> vad_option & {
            _lp_filter_freq = v;
            return *this;
        }
        auto hp_lifter_freq(double v) -> vad_option & {
            _hp_lifter_freq = v;
            return *this;
        }
        auto lp_lifter_freq(double v) -> vad_option & {
            _lp_lifter_freq = v;
            return *this;
        }
    } vad_option_t;

    class Vad {
    public:
        explicit Vad(vad_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            return functional::vad(waveform, _opt._sample_rate, _opt._trigger_level, _opt._trigger_time,
                                   _opt._search_time, _opt._allowed_gap, _opt._pre_trigger_time, _opt._boot_time,
                                   _opt._noise_up_time, _opt._noise_down_time, _opt._noise_reduction_amount,
                                   _opt._measure_freq, _opt._measure_duration, _opt._measure_smooth_time,
                                   _opt._hp_filter_freq, _opt._lp_filter_freq, _opt._hp_lifter_freq,
                                   _opt._lp_lifter_freq);
        }
        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        vad_option _opt;
    };

    // ---------------- Vol ----------------
    enum class vol_gain_type { amplitude, power, db };

    typedef struct vol_option {
        double _gain = 1.0;
        vol_gain_type _gain_type = vol_gain_type::amplitude;
        auto gain(double g) -> vol_option & {
            _gain = g;
            return *this;
        }
        auto gain_type(vol_gain_type t) -> vol_option & {
            _gain_type = t;
            return *this;
        }
    } vol_option_t;

    class Vol {
    public:
        explicit Vol(vol_option opt = {}) : _opt(opt) {
            if ((_opt._gain_type == vol_gain_type::amplitude || _opt._gain_type == vol_gain_type::power) &&
                _opt._gain < 0.0) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "Vol: gain must be positive for amplitude/power gain_type.");
            }
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            tensor_t out;
            switch (_opt._gain_type) {
                case vol_gain_type::db:
                    out = functional::gain(waveform, _opt._gain);
                    break;
                case vol_gain_type::power:
                    if (!(_opt._gain > 0.0)) { // torchaudio: math.log10 domain error
                        return handle_exceptions<tensor_t, std::invalid_argument>(
                                "Vol: gain must be positive for gain_type power.");
                    }
                    out = functional::gain(waveform, 10.0 * std::log10(_opt._gain));
                    break;
                case vol_gain_type::amplitude:
                default:
                    if (_opt._gain_type == vol_gain_type::amplitude && _opt._gain < 0.0) { // the ctor rejected it
                        return handle_exceptions<tensor_t, std::invalid_argument>("Vol: invalid arguments.");
                    }
                    out = waveform * _opt._gain;
                    break;
            }
            return torch::clamp(out, -1.0, 1.0);
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        vol_option _opt;
    };

    // ---------------- Convolve / FFTConvolve (reuse functional::convolve_mode) ----------------
    typedef struct convolve_option {
        functional::convolve_mode _mode = functional::convolve_mode::full;
        auto mode(functional::convolve_mode m) -> convolve_option & {
            _mode = m;
            return *this;
        }
    } convolve_option_t;

    class Convolve {
    public:
        explicit Convolve(convolve_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t x, const_tensor_lref_t y) const -> tensor_t {
            return functional::convolve(x, y, _opt._mode);
        }
        auto forward(const_tensor_lref_t x, const_tensor_lref_t y) const -> tensor_t { return (*this)(x, y); }

    private:
        convolve_option _opt;
    };

    class FFTConvolve {
    public:
        explicit FFTConvolve(convolve_option opt = {}) : _opt(opt) {}
        auto operator()(const_tensor_lref_t x, const_tensor_lref_t y) const -> tensor_t {
            return functional::fftconvolve(x, y, _opt._mode);
        }
        auto forward(const_tensor_lref_t x, const_tensor_lref_t y) const -> tensor_t { return (*this)(x, y); }

    private:
        convolve_option _opt;
    };

    // ---------------- AddNoise (no stored config) ----------------
    class AddNoise {
    public:
        AddNoise() = default;
        auto operator()(const_tensor_lref_t waveform, const_tensor_lref_t noise, const_tensor_lref_t snr,
                        const c10::optional<tensor_t> &lengths = c10::nullopt) const -> tensor_t {
            return functional::add_noise(waveform, noise, snr, lengths);
        }
        auto forward(const_tensor_lref_t waveform, const_tensor_lref_t noise, const_tensor_lref_t snr,
                     const c10::optional<tensor_t> &lengths = c10::nullopt) const -> tensor_t {
            return (*this)(waveform, noise, snr, lengths);
        }
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_EFFECTS_HPP
