#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_FEATURE_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_FEATURE_HPP
// Mel / cepstral transform classes (torchaudio.transforms parity): AmplitudeToDB, MelScale,
// InverseMelScale, MelSpectrogram, MFCC, LFCC. The "beyond-wrapping" content lives here: cached
// filterbank/DCT matrices applied by matmul, an lstsq mel inverse with no functional counterpart,
// and composition of the cached spectral sub-transforms (see develop_log 2026-06-01/progress02
// G3/G4 and task20). Depends on _transform_spectral.hpp for Spectrogram.
#include "_functional.hpp"
#include "_transform_spectral.hpp"

namespace torchmedia::audio::transform {
    // ---------------- AmplitudeToDB ----------------
    enum class db_stype { power, magnitude };

    typedef struct amplitude_to_db_option {
        db_stype _stype = db_stype::power;
        double _top_db = 80.0;
        bool _has_top_db = false; // torchaudio default top_db=None => no clamping

        auto stype(db_stype s) -> amplitude_to_db_option & {
            _stype = s;
            return *this;
        }
        auto top_db(double t) -> amplitude_to_db_option & {
            _top_db = t;
            _has_top_db = true;
            return *this;
        }
    } amplitude_to_db_option_t;

    class AmplitudeToDB {
    public:
        explicit AmplitudeToDB(amplitude_to_db_option opt = {}) : _opt(opt) {
            if (_opt._has_top_db && _opt._top_db < 0.0) {
                (void) handle_exceptions<int, std::invalid_argument>("AmplitudeToDB: top_db must be positive value.");
            }
        }

        auto operator()(const_tensor_lref_t signal) const -> tensor_t {
            const auto o = functional::amplitude_to_db_option()
                                   .set_multiplier(_opt._stype == db_stype::power ? 10.0 : 20.0)
                                   .set_amin(1e-10)
                                   .set_db_multiplier(0.0)
                                   .set_top_db(_opt._top_db)
                                   .set_apply_top_db(_opt._has_top_db);
            return functional::amplitude_to_DB(signal, o);
        }

        auto forward(const_tensor_lref_t signal) const -> tensor_t { return (*this)(signal); }

    private:
        amplitude_to_db_option _opt;
    };

    // ---------------- MelScale ----------------
    typedef struct mel_scale_option {
        int _n_mels = 128;
        int _sample_rate = 16000;
        int _n_stft = 201; // n_fft/2 + 1
        double _f_min = 0.0;
        double _f_max = 0.0; // 0 => Nyquist
        std::string _norm = ""; // "" (None) | "slaney"
        std::string _mel_scale = "htk"; // "htk" | "slaney"

        auto n_mels(int n) -> mel_scale_option & {
            _n_mels = n;
            return *this;
        }
        auto sample_rate(int s) -> mel_scale_option & {
            _sample_rate = s;
            return *this;
        }
        auto n_stft(int n) -> mel_scale_option & {
            _n_stft = n;
            return *this;
        }
        auto f_min(double f) -> mel_scale_option & {
            _f_min = f;
            return *this;
        }
        auto f_max(double f) -> mel_scale_option & {
            _f_max = f;
            return *this;
        }
        auto norm(const std::string &n) -> mel_scale_option & {
            _norm = n;
            return *this;
        }
        auto mel_scale(const std::string &m) -> mel_scale_option & {
            _mel_scale = m;
            return *this;
        }
    } mel_scale_option_t;

    namespace detail {
        // torchaudio raises when f_min > f_max (f_max defaulting to float(sample_rate // 2)).
        inline void check_f_min_max(double f_min, double f_max, int sample_rate) {
            if (f_min > functional::detail::effective_f_max(f_max, sample_rate)) {
                (void) handle_exceptions<int, std::invalid_argument>("MelScale: require f_min <= f_max.");
            }
        }
    } // namespace detail

    class MelScale {
    public:
        explicit MelScale(mel_scale_option opt = {}) : _opt(opt) {
            detail::check_f_min_max(_opt._f_min, _opt._f_max, _opt._sample_rate);
            _fb = functional::mel_filter_bank(_opt._n_mels, _opt._f_min, _opt._f_max, _opt._sample_rate, _opt._n_stft,
                                              _opt._norm, _opt._mel_scale);
        }

        // _fb is (n_mels, n_stft); functional::mel_scale projects it (rank-preserving matmul:
        // (..., freq, time) -> (..., n_mels, time)) after moving it to the input's device and dtype.
        auto operator()(const_tensor_lref_t specgram) const -> tensor_t {
            return functional::mel_scale(specgram, _fb.to(specgram.device(), specgram.scalar_type()));
        }

        auto forward(const_tensor_lref_t specgram) const -> tensor_t { return (*this)(specgram); }

        auto fb() const -> const_tensor_lref_t { return _fb; }

    private:
        mel_scale_option _opt;
        tensor_t _fb;
    };

    // ---------------- InverseMelScale ----------------
    enum class lstsq_driver { gels, gelsy, gelsd, gelss };

    namespace detail {
        inline auto driver_name(lstsq_driver d) -> c10::string_view {
            switch (d) {
                case lstsq_driver::gelsy:
                    return "gelsy";
                case lstsq_driver::gelsd:
                    return "gelsd";
                case lstsq_driver::gelss:
                    return "gelss";
                default: // lstsq_driver::gels (the option default)
                    return "gels";
            }
        }
    } // namespace detail

    typedef struct inverse_mel_scale_option {
        int _n_stft = 201;
        int _n_mels = 128;
        int _sample_rate = 16000;
        double _f_min = 0.0;
        double _f_max = 0.0;
        std::string _norm = "";
        std::string _mel_scale = "htk";
        lstsq_driver _driver = lstsq_driver::gels; // torchaudio default

        auto n_stft(int n) -> inverse_mel_scale_option & {
            _n_stft = n;
            return *this;
        }
        auto n_mels(int n) -> inverse_mel_scale_option & {
            _n_mels = n;
            return *this;
        }
        auto sample_rate(int s) -> inverse_mel_scale_option & {
            _sample_rate = s;
            return *this;
        }
        auto f_min(double f) -> inverse_mel_scale_option & {
            _f_min = f;
            return *this;
        }
        auto f_max(double f) -> inverse_mel_scale_option & {
            _f_max = f;
            return *this;
        }
        auto norm(const std::string &n) -> inverse_mel_scale_option & {
            _norm = n;
            return *this;
        }
        auto mel_scale(const std::string &m) -> inverse_mel_scale_option & {
            _mel_scale = m;
            return *this;
        }
        auto driver(lstsq_driver d) -> inverse_mel_scale_option & {
            _driver = d;
            return *this;
        }
    } inverse_mel_scale_option_t;

    class InverseMelScale {
    public:
        explicit InverseMelScale(inverse_mel_scale_option opt = {}) : _opt(opt) {
            detail::check_f_min_max(_opt._f_min, _opt._f_max, _opt._sample_rate);
            // _fb is (n_mels, n_stft) == torchaudio's fb.transpose(-1, -2): the lstsq coefficient matrix A.
            _fb = functional::mel_filter_bank(_opt._n_mels, _opt._f_min, _opt._f_max, _opt._sample_rate, _opt._n_stft,
                                              _opt._norm, _opt._mel_scale);
        }

        auto operator()(const_tensor_lref_t melspec) const -> tensor_t {
            const auto shape = melspec.sizes().vec();
            if (shape[shape.size() - 2] != _opt._n_mels) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "InverseMelScale: input n_mels does not match the configured n_mels.");
            }
            const int64_t time = shape.back();
            const auto packed = melspec.reshape({-1, static_cast<int64_t>(_opt._n_mels), time}); // (B, n_mels, time)
            const auto A = _fb.to(melspec.device(), melspec.scalar_type()).unsqueeze(0); // (1, n_mels, n_stft)
            const auto solution = std::get<0>(torch::linalg_lstsq(
                    A, packed, c10::nullopt, c10::optional<c10::string_view>(detail::driver_name(_opt._driver))));
            const auto specgram = torch::relu(solution); // (B, n_stft, time)
            std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
            out_shape.push_back(_opt._n_stft);
            out_shape.push_back(time);
            return specgram.reshape(out_shape);
        }

        auto forward(const_tensor_lref_t melspec) const -> tensor_t { return (*this)(melspec); }

        auto fb() const -> const_tensor_lref_t { return _fb; }

    private:
        inverse_mel_scale_option _opt;
        tensor_t _fb;
    };

    // ---------------- MelSpectrogram (compose Spectrogram + MelScale) ----------------
    typedef struct mel_spectrogram_option {
        int _sample_rate = 16000;
        int _n_fft = 400;
        int _win_length = 0;
        int _hop_length = 0;
        int _pad = 0;
        int _n_mels = 128;
        double _f_min = 0.0;
        double _f_max = 0.0;
        double _power = 2.0;
        bool _normalized = false;
        bool _center = true;
        std::string _pad_mode = "reflect";
        std::string _norm = "";
        std::string _mel_scale = "htk";
        tensor_t _window = {};

        auto sample_rate(int s) -> mel_spectrogram_option & {
            _sample_rate = s;
            return *this;
        }
        auto n_fft(int n) -> mel_spectrogram_option & {
            _n_fft = n;
            return *this;
        }
        auto win_length(int w) -> mel_spectrogram_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> mel_spectrogram_option & {
            _hop_length = h;
            return *this;
        }
        auto pad(int p) -> mel_spectrogram_option & {
            _pad = p;
            return *this;
        }
        auto n_mels(int n) -> mel_spectrogram_option & {
            _n_mels = n;
            return *this;
        }
        auto f_min(double f) -> mel_spectrogram_option & {
            _f_min = f;
            return *this;
        }
        auto f_max(double f) -> mel_spectrogram_option & {
            _f_max = f;
            return *this;
        }
        auto power(double p) -> mel_spectrogram_option & {
            _power = p;
            return *this;
        }
        auto normalized(bool n) -> mel_spectrogram_option & {
            _normalized = n;
            return *this;
        }
        auto center(bool c) -> mel_spectrogram_option & {
            _center = c;
            return *this;
        }
        auto pad_mode(const std::string &p) -> mel_spectrogram_option & {
            _pad_mode = p;
            return *this;
        }
        auto norm(const std::string &n) -> mel_spectrogram_option & {
            _norm = n;
            return *this;
        }
        auto mel_scale(const std::string &m) -> mel_spectrogram_option & {
            _mel_scale = m;
            return *this;
        }
        auto window(const tensor_t &w) -> mel_spectrogram_option & {
            _window = w;
            return *this;
        }
    } mel_spectrogram_option_t;

    class MelSpectrogram {
    public:
        // Members are built in the initializer list (Spectrogram derives win/hop from 0), so no default-configured
        // sub-transform is constructed and thrown away.
        explicit MelSpectrogram(mel_spectrogram_option opt = {}) :
            _opt(std::move(opt)), _spectrogram(spectrogram_option()
                                                       .pad(_opt._pad)
                                                       .n_fft(_opt._n_fft)
                                                       .win_length(_opt._win_length)
                                                       .hop_length(_opt._hop_length)
                                                       .power(_opt._power)
                                                       .normalized(_opt._normalized)
                                                       .center(_opt._center)
                                                       .pad_mode(_opt._pad_mode)
                                                       .onesided(true)
                                                       .return_complex(false)
                                                       .window(_opt._window)),
            _mel_scale(mel_scale_option()
                               .n_mels(_opt._n_mels)
                               .sample_rate(_opt._sample_rate)
                               .n_stft(_opt._n_fft / 2 + 1)
                               .f_min(_opt._f_min)
                               .f_max(_opt._f_max)
                               .norm(_opt._norm)
                               .mel_scale(_opt._mel_scale)) {}

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t { return _mel_scale(_spectrogram(waveform)); }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        mel_spectrogram_option _opt;
        Spectrogram _spectrogram;
        MelScale _mel_scale;
    };

    // ---------------- MFCC (compose MelSpectrogram + AmplitudeToDB + cached DCT) ----------------
    typedef struct mfcc_option {
        int _sample_rate = 16000;
        int _n_mfcc = 40;
        int _dct_type = 2;
        std::string _norm = "ortho";
        bool _log_mels = false;
        mel_spectrogram_option _mel{};

        auto sample_rate(int s) -> mfcc_option & {
            _sample_rate = s;
            return *this;
        }
        auto n_mfcc(int n) -> mfcc_option & {
            _n_mfcc = n;
            return *this;
        }
        auto dct_type(int d) -> mfcc_option & {
            _dct_type = d;
            return *this;
        }
        auto norm(const std::string &n) -> mfcc_option & {
            _norm = n;
            return *this;
        }
        auto log_mels(bool l) -> mfcc_option & {
            _log_mels = l;
            return *this;
        }
        auto mel(const mel_spectrogram_option &m) -> mfcc_option & {
            _mel = m;
            return *this;
        }
    } mfcc_option_t;

    class MFCC {
    public:
        explicit MFCC(mfcc_option opt = {}) :
            _opt(std::move(opt)), _melspec(with_sample_rate(_opt._mel, _opt._sample_rate)),
            _amplitude_to_db(amplitude_to_db_option().stype(db_stype::power).top_db(80.0)) {
            if (_opt._dct_type != 2) {
                (void) handle_exceptions<int, std::invalid_argument>("MFCC: only dct_type == 2 is supported.");
            }
            if (_opt._n_mfcc > _opt._mel._n_mels) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "MFCC: cannot select more MFCC coefficients than mel bins.");
            }
            _dct = functional::create_dct(_opt._n_mfcc, _opt._mel._n_mels, _opt._norm); // (n_mels, n_mfcc)
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            const auto mel = _melspec(waveform);
            const auto feat = _opt._log_mels ? torch::log(mel + 1e-6) : _amplitude_to_db(mel);
            return torch::matmul(feat.transpose(-2, -1), _dct.to(feat.device(), feat.scalar_type())).transpose(-2, -1);
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        // torchaudio: MelSpectrogram(sample_rate=sample_rate, **melkwargs) -- MFCC's rate overrides the mel option's.
        static auto with_sample_rate(mel_spectrogram_option mel, int sample_rate) -> mel_spectrogram_option {
            mel._sample_rate = sample_rate;
            return mel;
        }

        mfcc_option _opt;
        MelSpectrogram _melspec;
        AmplitudeToDB _amplitude_to_db;
        tensor_t _dct;
    };

    // ---------------- LFCC (compose Spectrogram + AmplitudeToDB + cached linear fbank + DCT) ----------------
    typedef struct lfcc_option {
        int _sample_rate = 16000;
        int _n_filter = 128;
        int _n_lfcc = 40;
        int _dct_type = 2;
        double _f_min = 0.0;
        double _f_max = 0.0;
        std::string _norm = "ortho";
        bool _log_lf = false;
        spectrogram_option _spec{};

        auto sample_rate(int s) -> lfcc_option & {
            _sample_rate = s;
            return *this;
        }
        auto n_filter(int n) -> lfcc_option & {
            _n_filter = n;
            return *this;
        }
        auto n_lfcc(int n) -> lfcc_option & {
            _n_lfcc = n;
            return *this;
        }
        auto dct_type(int d) -> lfcc_option & {
            _dct_type = d;
            return *this;
        }
        auto f_min(double f) -> lfcc_option & {
            _f_min = f;
            return *this;
        }
        auto f_max(double f) -> lfcc_option & {
            _f_max = f;
            return *this;
        }
        auto norm(const std::string &n) -> lfcc_option & {
            _norm = n;
            return *this;
        }
        auto log_lf(bool l) -> lfcc_option & {
            _log_lf = l;
            return *this;
        }
        auto spec(const spectrogram_option &s) -> lfcc_option & {
            _spec = s;
            return *this;
        }
    } lfcc_option_t;

    class LFCC {
    public:
        explicit LFCC(lfcc_option opt = {}) : _opt(opt) {
            if (_opt._dct_type != 2) {
                (void) handle_exceptions<int, std::invalid_argument>("LFCC: only dct_type == 2 is supported.");
            }
            if (_opt._n_lfcc > _opt._spec._n_fft) {
                (void) handle_exceptions<int, std::invalid_argument>(
                        "LFCC: cannot select more LFCC coefficients than fft bins.");
            }
            _spectrogram = Spectrogram(_opt._spec);
            _amplitude_to_db = AmplitudeToDB(amplitude_to_db_option().stype(db_stype::power).top_db(80.0));
            const int n_freqs = _opt._spec._n_fft / 2 + 1;
            const double eff_f_max = functional::detail::effective_f_max(_opt._f_max, _opt._sample_rate);
            _filter = functional::linear_fbanks(n_freqs, _opt._f_min, eff_f_max, _opt._n_filter,
                                                _opt._sample_rate); // (n_freqs, n_filter)
            _dct = functional::create_dct(_opt._n_lfcc, _opt._n_filter, _opt._norm); // (n_filter, n_lfcc)
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            auto spec = _spectrogram(waveform); // (..., n_freqs, time)
            const auto like_spec = [&spec](const tensor_t &t) { return t.to(spec.device(), spec.scalar_type()); };
            spec = torch::matmul(spec.transpose(-2, -1), like_spec(_filter)).transpose(-2, -1); // (..., n_filter, time)
            spec = _opt._log_lf ? torch::log(spec + 1e-6) : _amplitude_to_db(spec);
            return torch::matmul(spec.transpose(-2, -1), like_spec(_dct)).transpose(-2, -1); // (..., n_lfcc, time)
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        lfcc_option _opt;
        Spectrogram _spectrogram;
        AmplitudeToDB _amplitude_to_db;
        tensor_t _filter;
        tensor_t _dct;
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_FEATURE_HPP
