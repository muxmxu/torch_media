#ifndef LIB_TORCH_MEDIA_AUDIO_TRANSFORM_TIME_HPP
#define LIB_TORCH_MEDIA_AUDIO_TRANSFORM_TIME_HPP
// Time-domain transform classes (torchaudio.transforms parity): Resample, Speed, SpeedPerturbation,
// PitchShift, TimeStretch. The headline state is the cached sinc resample kernel (Resample) and the
// cached phase-advance vector (TimeStretch); Speed/SpeedPerturbation/PitchShift reuse the resampling
// machinery. See develop_log 2026-06-01/progress02 (G1 kernel split, D8 eager kernel) and task21.
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>
#include "_functional.hpp"

namespace torchmedia::audio::transform {
    // ---------------- Resample ----------------
    typedef struct resample_option {
        int _orig_freq = 16000;
        int _new_freq = 16000;
        int _lowpass_filter_width = 6;
        double _rolloff = 0.99;
        // Kernel precision (torchaudio `dtype`): unset => computed in float64 and cached as float32.
        c10::optional<torch::ScalarType> _dtype = c10::nullopt;

        auto orig_freq(int f) -> resample_option & {
            _orig_freq = f;
            return *this;
        }
        auto new_freq(int f) -> resample_option & {
            _new_freq = f;
            return *this;
        }
        auto lowpass_filter_width(int w) -> resample_option & {
            _lowpass_filter_width = w;
            return *this;
        }
        auto rolloff(double r) -> resample_option & {
            _rolloff = r;
            return *this;
        }
        auto dtype(torch::ScalarType d) -> resample_option & {
            _dtype = d;
            return *this;
        }
    } resample_option_t;

    class Resample {
    public:
        // The kernel is cached once (torchaudio dtype semantics) and moved to each input's device and dtype on use.
        explicit Resample(resample_option opt = {}) : _opt(opt) {
            functional::resample_option fopt;
            fopt.lowpass_filter_width = _opt._lowpass_filter_width;
            fopt.rolloff = _opt._rolloff;
            if (const auto err = functional::_check_resample_args(_opt._orig_freq, _opt._new_freq, fopt);
                !err.empty()) {
                (void) handle_exceptions<int, std::invalid_argument>("Resample: " + err);
                return; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: never build a kernel from invalid arguments
            }
            if (_opt._orig_freq != _opt._new_freq) {
                _gcd = std::gcd(_opt._orig_freq, _opt._new_freq);
                auto kw = functional::_sinc_resample_kernel(_opt._orig_freq, _opt._new_freq, _gcd, fopt,
                                                            _opt._dtype ? tensor_options_t().dtype(*_opt._dtype)
                                                                        : tensor_options_t());
                _kernel = kw.first;
                _width = kw.second;
            }
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            if (_opt._orig_freq == _opt._new_freq) {
                return waveform; // identity short-circuit
            }
            if (!_kernel.defined()) { // LIB_TORCH_MEDIA_NO_EXCEPTIONS: the constructor rejected the arguments
                return handle_exceptions<tensor_t, std::invalid_argument>("Resample: invalid arguments.");
            }
            return functional::_apply_sinc_resample_kernel(waveform, _opt._orig_freq, _opt._new_freq, _gcd, _kernel,
                                                           _width);
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

        auto kernel() const -> const_tensor_lref_t { return _kernel; }

    private:
        resample_option _opt;
        int _gcd = 1;
        int _width = 0;
        tensor_t _kernel;
    };

    // ---------------- Speed (speed change via resampling) ----------------
    typedef struct speed_option {
        int _orig_freq = 16000;
        double _factor = 1.0;

        auto orig_freq(int f) -> speed_option & {
            _orig_freq = f;
            return *this;
        }
        auto factor(double f) -> speed_option & {
            _factor = f;
            return *this;
        }
    } speed_option_t;

    class Speed {
    public:
        explicit Speed(speed_option opt = {}) : _opt(opt) {
            int source = static_cast<int>(_opt._factor * _opt._orig_freq);
            int target = _opt._orig_freq;
            const int g = std::gcd(source, target);
            _source = source / g;
            _target = target / g;
            _resampler = Resample(resample_option().orig_freq(_source).new_freq(_target));
        }

        auto operator()(const_tensor_lref_t waveform, const c10::optional<tensor_t> &lengths = c10::nullopt) const
                -> std::pair<tensor_t, c10::optional<tensor_t>> {
            const auto out = _resampler(waveform);
            if (!lengths.has_value() || (!out.defined() && waveform.defined())) { // the ctor rejected the factor
                return {out, c10::nullopt};
            }
            return {out,
                    torch::ceil(lengths.value() * _target / static_cast<double>(_source)).to(lengths.value().dtype())};
        }

        auto forward(const_tensor_lref_t waveform, const c10::optional<tensor_t> &lengths = c10::nullopt) const
                -> std::pair<tensor_t, c10::optional<tensor_t>> {
            return (*this)(waveform, lengths);
        }

    private:
        speed_option _opt;
        int _source = 1;
        int _target = 1;
        Resample _resampler;
    };

    // ---------------- SpeedPerturbation (random factor per call) ----------------
    typedef struct speed_perturbation_option {
        int _orig_freq = 16000;
        std::vector<double> _factors = {0.9, 1.0, 1.1};

        auto orig_freq(int f) -> speed_perturbation_option & {
            _orig_freq = f;
            return *this;
        }
        auto factors(const std::vector<double> &f) -> speed_perturbation_option & {
            _factors = f;
            return *this;
        }
    } speed_perturbation_option_t;

    class SpeedPerturbation {
    public:
        explicit SpeedPerturbation(speed_perturbation_option opt = {}) : _opt(opt) {
            for (const double f: _opt._factors) {
                _speeders.emplace_back(speed_option().orig_freq(_opt._orig_freq).factor(f));
            }
        }

        // Draws a factor uniformly at random per call (global RNG; seed with torch::manual_seed for
        // reproducibility) -> matches torchaudio.transforms.SpeedPerturbation.
        auto operator()(const_tensor_lref_t waveform, const c10::optional<tensor_t> &lengths = c10::nullopt) const
                -> std::pair<tensor_t, c10::optional<tensor_t>> {
            const int64_t idx = torch::randint(static_cast<int64_t>(_speeders.size()), {1}).item<int64_t>();
            return _speeders[idx](waveform, lengths);
        }

        auto forward(const_tensor_lref_t waveform, const c10::optional<tensor_t> &lengths = c10::nullopt) const
                -> std::pair<tensor_t, c10::optional<tensor_t>> {
            return (*this)(waveform, lengths);
        }

        auto size() const -> int64_t { return static_cast<int64_t>(_speeders.size()); }

    private:
        speed_perturbation_option _opt;
        std::vector<Speed> _speeders;
    };

    // ---------------- PitchShift (stft -> phase_vocoder -> istft -> resample, eager float32 kernel) ----------------
    // API differences: always a Hann window (torchaudio: window_fn/wkwargs); float64 input works directly and gives
    // torchaudio's module after .double() (torchaudio needs that call; its float32 window buffer is cast up, as here).
    typedef struct pitch_shift_option {
        int _sample_rate = 16000;
        double _n_steps = 0; // fractional steps allowed (torchaudio: "(fractional) steps to shift")
        int _bins_per_octave = 12;
        int _n_fft = 512;
        int _win_length = 0; // 0 => n_fft
        int _hop_length = 0; // 0 => win_length / 4

        auto sample_rate(int s) -> pitch_shift_option & {
            _sample_rate = s;
            return *this;
        }
        auto n_steps(double n) -> pitch_shift_option & {
            _n_steps = n;
            return *this;
        }
        auto bins_per_octave(int b) -> pitch_shift_option & {
            _bins_per_octave = b;
            return *this;
        }
        auto n_fft(int n) -> pitch_shift_option & {
            _n_fft = n;
            return *this;
        }
        auto win_length(int w) -> pitch_shift_option & {
            _win_length = w;
            return *this;
        }
        auto hop_length(int h) -> pitch_shift_option & {
            _hop_length = h;
            return *this;
        }
    } pitch_shift_option_t;

    class PitchShift {
    public:
        explicit PitchShift(pitch_shift_option opt = {}) : _opt(opt) {
            _win_length = _opt._win_length > 0 ? _opt._win_length : _opt._n_fft;
            _hop_length = _opt._hop_length > 0 ? _opt._hop_length : _win_length / 4;
            _window = torch::hann_window(_win_length, tensor_options_t().dtype(torch::kFloat));
            // Stricter than torchaudio, which accepts these in its constructor and fails in the first call.
            if (const auto err = functional::_pitch_shift_orig_freq(_opt._sample_rate, _opt._n_steps,
                                                                    _opt._bins_per_octave, _orig_freq);
                !err.empty()) {
                (void) handle_exceptions<int, std::invalid_argument>("PitchShift: " + err);
                _orig_freq = 0; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: operator() returns the default value
                return;
            }
            if (_orig_freq != _opt._sample_rate) {
                _gcd = std::gcd(_orig_freq, _opt._sample_rate);
                auto kw = functional::_sinc_resample_kernel(_orig_freq, _opt._sample_rate, _gcd,
                                                            functional::resample_option(),
                                                            tensor_options_t().dtype(torch::kFloat));
                _kernels->list.push_back(kw.first);
                _width = kw.second;
            }
        }

        auto operator()(const_tensor_lref_t waveform) const -> tensor_t {
            if (_orig_freq <= 0 || waveform.dim() < 1) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        "PitchShift: invalid arguments or a 0-D waveform.");
            }
            const auto stretched = functional::_stretch_waveform(
                    waveform, _opt._n_steps, _opt._bins_per_octave, _opt._n_fft, c10::optional<int>(_win_length),
                    c10::optional<int>(_hop_length), c10::optional<tensor_t>(_window.to(waveform.options())));
            tensor_t shifted = stretched;
            if (_orig_freq != _opt._sample_rate) {
                shifted = functional::_apply_sinc_resample_kernel(stretched, _orig_freq, _opt._sample_rate, _gcd,
                                                                  kernel_for(waveform), _width);
            }
            return functional::_fix_waveform_shape(shifted, waveform.sizes().vec());
        }

        auto forward(const_tensor_lref_t waveform) const -> tensor_t { return (*this)(waveform); }

    private:
        // One kernel per (dtype, device): the eager float32 CPU one (the common case), any other built on first use
        // in the input's dtype and device, as torchaudio's lazy kernel init does. Shared by copies; the mutex keeps a
        // const call thread-safe.
        struct kernel_cache {
            std::mutex mutex;
            std::vector<tensor_t> list;
        };

        auto kernel_for(const_tensor_lref_t waveform) const -> tensor_t {
            const std::lock_guard<std::mutex> lock(_kernels->mutex);
            for (const auto &k: _kernels->list) {
                if (k.scalar_type() == waveform.scalar_type() && k.device() == waveform.device()) {
                    return k;
                }
            }
            _kernels->list.push_back(functional::_sinc_resample_kernel(_orig_freq, _opt._sample_rate, _gcd,
                                                                       functional::resample_option(),
                                                                       waveform.options())
                                             .first);
            return _kernels->list.back();
        }

        pitch_shift_option _opt;
        int _win_length = 0;
        int _hop_length = 0;
        int _orig_freq = 0;
        int _gcd = 1;
        int _width = 0;
        tensor_t _window;
        std::shared_ptr<kernel_cache> _kernels = std::make_shared<kernel_cache>();
    };

    // ---------------- TimeStretch (phase vocoder, cached phase-advance) ----------------
    typedef struct time_stretch_option {
        int _n_freq = 201;
        // 0 => (n_freq-1)  [= n_fft/2 with n_fft=(n_freq-1)*2]; torchaudio's explicit hop_length=0 is not expressible.
        int _hop_length = 0;
        double _fixed_rate = -1.0; // < 0 => unset (must pass a rate to operator()); input must be complex (no warning)

        auto n_freq(int n) -> time_stretch_option & {
            _n_freq = n;
            return *this;
        }
        auto hop_length(int h) -> time_stretch_option & {
            _hop_length = h;
            return *this;
        }
        auto fixed_rate(double r) -> time_stretch_option & {
            _fixed_rate = r;
            return *this;
        }
    } time_stretch_option_t;

    class TimeStretch {
    public:
        explicit TimeStretch(time_stretch_option opt = {}) : _opt(opt) {
            const int n_fft = (_opt._n_freq - 1) * 2;
            const int hop = _opt._hop_length > 0 ? _opt._hop_length : n_fft / 2;
            const double pi = std::acos(-1.0);
            _phase_advance =
                    torch::linspace(0, pi * hop, _opt._n_freq, tensor_options_t().dtype(torch::kFloat)).unsqueeze(-1);
        }

        auto operator()(const_tensor_lref_t complex_specgrams,
                        c10::optional<double> overriding_rate = c10::nullopt) const -> tensor_t {
            const double rate = overriding_rate.has_value() ? overriding_rate.value() : _opt._fixed_rate;
            if (!(rate > 0.0)) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        overriding_rate.has_value() ? "TimeStretch: overriding_rate must be positive."
                                                    : "TimeStretch: no fixed_rate set; pass a valid rate to the call.");
            }
            return functional::phase_vocoder(complex_specgrams, rate, _phase_advance.to(complex_specgrams.device()));
        }

        auto forward(const_tensor_lref_t complex_specgrams, c10::optional<double> overriding_rate = c10::nullopt) const
                -> tensor_t {
            return (*this)(complex_specgrams, overriding_rate);
        }

        auto phase_advance() const -> const_tensor_lref_t { return _phase_advance; }

    private:
        time_stretch_option _opt;
        tensor_t _phase_advance;
    };
} // namespace torchmedia::audio::transform
#endif // LIB_TORCH_MEDIA_AUDIO_TRANSFORM_TIME_HPP
