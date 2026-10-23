#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_SPECTRAL_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_SPECTRAL_HPP
// Spectral transform classes (torchaudio.transforms parity): Spectrogram, InverseSpectrogram,
// GriffinLim, SpectralCentroid. Each is a stateful class that precomputes/caches the analysis
// window once in its constructor and delegates the math to torchmedia::audio::functional. This is
// the first task of the transform layer; it establishes the class idiom (see develop_log
// 2026-06-01/progress02 D1/D2/D5/D6 and task19).
#include "_functional.hpp"
#include "_functional_methods_options.hpp"

namespace torchmedia::audio::transform {
    namespace detail {
        // torchaudio default-derivation (win_length -> n_fft, hop_length -> win_length / 2), shared with functional.
        using functional::detail::derive_hop_length;
        using functional::detail::derive_win_length;
        // Cache the analysis window once (float32, like torchaudio's registered buffer). An explicit
        // window passed via the option overrides the default Hann window. Calls move it to the input's device
        // (the C++ stand-in for torchaudio's module.to(device)).
        inline auto make_window(const tensor_t &explicit_window, int win_length) -> tensor_t {
            return explicit_window.defined() ? explicit_window
                                             : torch::hann_window(win_length, tensor_options_t().dtype(torch::kFloat));
        }
    } // namespace detail

    // ---------------- Spectrogram ----------------
    typedef struct spectrogram_option {
        int _pad = 0;
        int _n_fft = 400;
        int _win_length = 0; // 0 => derive from n_fft
        int _hop_length = 0; // 0 => derive from win_length
        double _power = 2.0;
        bool _normalized = false;
        std::string _normalize_method = "window"; // "window" | "frame_length"
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        bool _return_complex = false; // torchaudio: power!=None => real power spectrogram
        tensor_t _window = {}; // explicit window override (defaults to Hann)

        auto pad(int p) -> spectrogram_option & {
            _pad = p;
            return *this;
        }
        auto n_fft(int n) -> spectrogram_option & {
            _n_fft = n;
            return *this;
        }
        auto win_length(int w) -> spectrogram_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> spectrogram_option & {
            _hop_length = h;
            return *this;
        }
        auto power(double p) -> spectrogram_option & {
            _power = p;
            return *this;
        }
        auto normalized(bool n) -> spectrogram_option & {
            _normalized = n;
            return *this;
        }
        auto normalize_method(const std::string &n) -> spectrogram_option & {
            _normalize_method = n;
            return *this;
        }
        auto center(bool c) -> spectrogram_option & {
            _center = c;
            return *this;
        }
        auto pad_mode(const std::string &p) -> spectrogram_option & {
            _pad_mode = p;
            return *this;
        }
        auto onesided(bool o) -> spectrogram_option & {
            _onesided = o;
            return *this;
        }
        auto return_complex(bool r) -> spectrogram_option & {
            _return_complex = r;
            return *this;
        }
        auto window(const tensor_t &w) -> spectrogram_option & {
            _window = w;
            return *this;
        }
    } spectrogram_option_t;

    class Spectrogram {
    public:
        explicit Spectrogram(spectrogram_option opt = {}) : _opt(std::move(opt)) {
            _opt._win_length = detail::derive_win_length(_opt._win_length, _opt._n_fft);
            _opt._hop_length = detail::derive_hop_length(_opt._hop_length, _opt._win_length);
            _window = detail::make_window(_opt._window, _opt._win_length);
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            auto fopt = functional::spectrogram_option()
                                .pad(_opt._pad)
                                .window(_window.to(waveform.device()))
                                .n_fft(_opt._n_fft)
                                .hop_length(_opt._hop_length)
                                .win_length(_opt._win_length)
                                .power(_opt._power)
                                .normalized(_opt._normalized)
                                .normalize_method(_opt._normalize_method)
                                .center(_opt._center)
                                .pad_mode(_opt._pad_mode)
                                .onesided(_opt._onesided)
                                .return_complex(_opt._return_complex);
            return functional::spectrogram(waveform, fopt);
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

        auto window() const -> const_tensor_lref_t { return _window; }

    private:
        spectrogram_option _opt;
        tensor_t _window;
    };

    // ---------------- InverseSpectrogram ----------------
    typedef struct inverse_spectrogram_option {
        int _pad = 0;
        int _n_fft = 400;
        int _win_length = 0;
        int _hop_length = 0;
        std::string _normalized = "none"; // "none" | "window" | "frame_length"
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        tensor_t _window = {};

        auto pad(int p) -> inverse_spectrogram_option & {
            _pad = p;
            return *this;
        }
        auto n_fft(int n) -> inverse_spectrogram_option & {
            _n_fft = n;
            return *this;
        }
        auto win_length(int w) -> inverse_spectrogram_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> inverse_spectrogram_option & {
            _hop_length = h;
            return *this;
        }
        auto normalized(const std::string &n) -> inverse_spectrogram_option & {
            _normalized = n;
            return *this;
        }
        auto center(bool c) -> inverse_spectrogram_option & {
            _center = c;
            return *this;
        }
        auto pad_mode(const std::string &p) -> inverse_spectrogram_option & {
            _pad_mode = p;
            return *this;
        }
        auto onesided(bool o) -> inverse_spectrogram_option & {
            _onesided = o;
            return *this;
        }
        auto window(const tensor_t &w) -> inverse_spectrogram_option & {
            _window = w;
            return *this;
        }
    } inverse_spectrogram_option_t;

    class InverseSpectrogram {
    public:
        explicit InverseSpectrogram(inverse_spectrogram_option opt = {}) : _opt(std::move(opt)) {
            _opt._win_length = detail::derive_win_length(_opt._win_length, _opt._n_fft);
            _opt._hop_length = detail::derive_hop_length(_opt._hop_length, _opt._win_length);
            _window = detail::make_window(_opt._window, _opt._win_length);
        }

        auto operator()(const_tensor_lref_t spectrogram, c10::optional<int64_t> length = c10::nullopt) const
                -> tensor_t {
            return functional::inverse_spectrogram(spectrogram, length, _opt._pad, _window.to(spectrogram.device()),
                                                   _opt._n_fft, _opt._hop_length, _opt._win_length, _opt._normalized,
                                                   _opt._center, _opt._pad_mode, _opt._onesided);
        }

        auto forward(const_tensor_lref_t spectrogram, c10::optional<int64_t> length = c10::nullopt) const -> tensor_t {
            return (*this)(spectrogram, length);
        }

        auto window() const -> const_tensor_lref_t { return _window; }

    private:
        inverse_spectrogram_option _opt;
        tensor_t _window;
    };

    // ---------------- GriffinLim ----------------
    typedef struct griffinlim_option {
        int _n_fft = 400;
        int _n_iter = 32;
        int _win_length = 0;
        int _hop_length = 0;
        double _power = 2.0;
        double _momentum = 0.99;
        int _length = -1; // -1 => inferred from the spectrogram
        bool _rand_init = true; // torchaudio default
        tensor_t _window = {};

        auto n_fft(int n) -> griffinlim_option & {
            _n_fft = n;
            return *this;
        }
        auto n_iter(int n) -> griffinlim_option & {
            _n_iter = n;
            return *this;
        }
        auto win_length(int w) -> griffinlim_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> griffinlim_option & {
            _hop_length = h;
            return *this;
        }
        auto power(double p) -> griffinlim_option & {
            _power = p;
            return *this;
        }
        auto momentum(double m) -> griffinlim_option & {
            _momentum = m;
            return *this;
        }
        auto length(int l) -> griffinlim_option & {
            _length = l;
            return *this;
        }
        auto rand_init(bool r) -> griffinlim_option & {
            _rand_init = r;
            return *this;
        }
        auto window(const tensor_t &w) -> griffinlim_option & {
            _window = w;
            return *this;
        }
    } griffinlim_option_t;

    class GriffinLim {
    public:
        explicit GriffinLim(griffinlim_option opt = {}) : _opt(std::move(opt)) {
            if (!(_opt._momentum >= 0.0 && _opt._momentum < 1.0)) {
                (void) handle_exceptions<int, std::invalid_argument>("GriffinLim: momentum must be in [0, 1).");
            }
            _opt._win_length = detail::derive_win_length(_opt._win_length, _opt._n_fft);
            _opt._hop_length = detail::derive_hop_length(_opt._hop_length, _opt._win_length);
            _window = detail::make_window(_opt._window, _opt._win_length);
        }

        auto operator()(const_tensor_lref_t specgram) const -> tensor_t {
            functional::griffinlim_option g;
            g.n_fft = _opt._n_fft;
            g.hop_length = _opt._hop_length;
            g.win_length = _opt._win_length;
            g.window = _window.to(specgram.device());
            g.power = _opt._power;
            g.n_iter = _opt._n_iter;
            g.momentum = _opt._momentum;
            g.length = _opt._length;
            g.rand_init = _opt._rand_init;
            return functional::griffinlim(specgram, g);
        }

        auto forward(const_tensor_lref_t specgram) const -> tensor_t { return (*this)(specgram); }

        auto window() const -> const_tensor_lref_t { return _window; }

    private:
        griffinlim_option _opt;
        tensor_t _window;
    };

    // ---------------- SpectralCentroid ----------------
    typedef struct spectral_centroid_option {
        int _sample_rate = 16000;
        int _pad = 0;
        int _n_fft = 400;
        int _win_length = 0;
        int _hop_length = 0;
        tensor_t _window = {};

        auto sample_rate(int s) -> spectral_centroid_option & {
            _sample_rate = s;
            return *this;
        }
        auto pad(int p) -> spectral_centroid_option & {
            _pad = p;
            return *this;
        }
        auto n_fft(int n) -> spectral_centroid_option & {
            _n_fft = n;
            return *this;
        }
        auto win_length(int w) -> spectral_centroid_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> spectral_centroid_option & {
            _hop_length = h;
            return *this;
        }
        auto window(const tensor_t &w) -> spectral_centroid_option & {
            _window = w;
            return *this;
        }
    } spectral_centroid_option_t;

    class SpectralCentroid {
    public:
        explicit SpectralCentroid(spectral_centroid_option opt = {}) : _opt(std::move(opt)) {
            _opt._win_length = detail::derive_win_length(_opt._win_length, _opt._n_fft);
            _opt._hop_length = detail::derive_hop_length(_opt._hop_length, _opt._win_length);
            _window = detail::make_window(_opt._window, _opt._win_length);
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            return functional::spectral_centroid(waveform, _opt._sample_rate, _opt._pad, _window.to(waveform.device()),
                                                 _opt._n_fft, _opt._hop_length, _opt._win_length);
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

        auto window() const -> const_tensor_lref_t { return _window; }

    private:
        spectral_centroid_option _opt;
        tensor_t _window;
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_SPECTRAL_HPP
