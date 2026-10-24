//
// Created by Mux on 25-1-17.
//

#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
#include "../global_include.hpp"

namespace torchmedia::audio::functional {
    namespace detail {
        // torchaudio default derivation, shared with the transform layer: win_length 0 => n_fft, hop_length 0 =>
        // win_length / 2 (torchaudio's None); a negative value is kept, so torch raises on it like torchaudio does.
        // f_max <= 0 => float(sample_rate // 2) (integer halving, as in torchaudio MelScale/LFCC).
        inline auto derive_win_length(int win_length, int n_fft) -> int { return win_length == 0 ? n_fft : win_length; }
        inline auto derive_hop_length(int hop_length, int win_length) -> int {
            return hop_length == 0 ? win_length / 2 : hop_length;
        }
        inline auto effective_f_max(double f_max, int sample_rate) -> double {
            return f_max > 0.0 ? f_max : static_cast<double>(sample_rate / 2);
        }
    } // namespace detail

    typedef struct spectrogram_option {
        int _pad = 0;
        tensor_t _window = {};
        int _n_fft = 400;
        int _hop_length = 0; // 0 => win_length / 2
        int _win_length = 0; // 0 => n_fft
        double _power = 2.0;
        bool _normalized = false;
        std::string _normalize_method = "window"; // "window" | "frame_length"; anything else throws (even unused)
        bool _center = true;
        std::string _pad_mode = "reflect";
        bool _onesided = true;
        bool _return_complex = true; // when true, power becomes optional;

        auto pad(const int p) -> spectrogram_option & {
            _pad = p;
            return *this;
        }

        auto window(const tensor_t &w) -> spectrogram_option & {
            _window = w;
            return *this;
        }

        auto n_fft(const int n) -> spectrogram_option & {
            _n_fft = n;
            return *this;
        }

        auto hop_length(const int h) -> spectrogram_option & {
            _hop_length = h;
            return *this;
        }

        auto win_length(const int w) -> spectrogram_option & {
            _win_length = w;
            return *this;
        }

        auto power(const double p) -> spectrogram_option & {
            _power = p;
            return *this;
        }

        auto normalized(const bool n) -> spectrogram_option & {
            _normalized = n;
            return *this;
        }

        auto normalize_method(const std::string &n) -> spectrogram_option & {
            _normalize_method = n;
            return *this;
        }

        auto center(const bool c) -> spectrogram_option & {
            _center = c;
            return *this;
        }

        auto pad_mode(const std::string &p) -> spectrogram_option & {
            _pad_mode = p;
            return *this;
        }

        auto onesided(const bool o) -> spectrogram_option & {
            _onesided = o;
            return *this;
        }

        auto return_complex(const bool r) -> spectrogram_option & {
            _return_complex = r;
            return *this;
        }
    } spectrogram_option_t;

    // Defaults mirror torchaudio AmplitudeToDB(): power (multiplier 10), amin 1e-10, ref 1.0 (db_multiplier 0) and
    // no top_db. set_top_db() enables the cutoff (torchaudio: passing top_db); set_apply_top_db(false) disables it.
    typedef struct amplitude_to_db_option {
        double multiplier = 10.0; // 10 for a power spectrogram, 20 for a magnitude spectrogram
        double amin = 1e-10;
        double top_db = 80.0; // used only when apply_top_db
        double db_multiplier = 0.0; // log10(max(amin, ref))
        bool apply_top_db = false;

        auto set_multiplier(const double m) -> amplitude_to_db_option & {
            multiplier = m;
            return *this;
        }

        auto set_amin(const double a) -> amplitude_to_db_option & {
            amin = a;
            return *this;
        }

        auto set_top_db(const double t) -> amplitude_to_db_option & {
            top_db = t;
            apply_top_db = true;
            return *this;
        }

        auto set_db_multiplier(const double m) -> amplitude_to_db_option & {
            db_multiplier = m;
            return *this;
        }

        auto set_apply_top_db(const bool b) -> amplitude_to_db_option & {
            apply_top_db = b;
            return *this;
        }
    } amplitude_to_db_option_t;

    typedef struct mel_spectrogram_option {
        int sample_rate = 16000;
        int n_fft = 400;
        int win_length = 0; // 0 => n_fft
        int hop_length = 0; // 0 => win_length / 2
        int pad = 0;

        double f_min = 0.0;
        double f_max = 0.0; // <= 0 => sample_rate // 2
        int n_mels = 128;

        double power = 2.0;
        bool normalized = false;
        bool center = true;
        std::string pad_mode = "reflect";
        bool onesided = true; // deprecated and ignored, like torchaudio: the spectrogram is always one-sided
        std::string norm; // "" (None) | "slaney"
        std::string mel_scale = "htk"; // "htk" | "slaney"
    } mel_spectrogram_option_t;

    typedef struct db_to_amplitude_option {
    } db_to_amplitude_option_t;

    typedef struct mfcc_option {
        int sample_rate = 16000; // the rate, as in torchaudio MFCC(sample_rate, melkwargs): overrides mel.sample_rate
        int n_mfcc = 40;
        std::string norm = "ortho"; // DCT normalization ("ortho" or "")
        bool log_mels = false;      // log(mel + 1e-6) instead of amplitude_to_DB
        double top_db = 80.0; // torchaudio MFCC: AmplitudeToDB("power", top_db=80)
        mel_spectrogram_option mel{}; // mel spectrogram parameters
    } mfcc_option_t;

    typedef struct griffinlim_option {
        int n_fft = 400;
        int hop_length = 0; // 0 => win_length / 2
        int win_length = 0; // 0 => n_fft
        tensor_t window = {}; // defaults to hann_window(win_length) in the spectrogram's dtype
        double power = 2.0;
        int n_iter = 32;
        double momentum = 0.99; // must be in [0, 1)
        int length = -1;        // -1 => inferred from the spectrogram
        bool rand_init = false; // false => deterministic
    } griffinlim_option_t;

    typedef struct resample_option {
        int lowpass_filter_width = 6;
        double rolloff = 0.99; // cutoff = rolloff * Nyquist of the lower rate
    } resample_option_t;
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_METHODS_OPTIONS_HPP
