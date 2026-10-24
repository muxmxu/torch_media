#ifndef LIB_TORCH_MEDIA_AUDIO_COMPLIANCE_KALDI_HPP
#define LIB_TORCH_MEDIA_AUDIO_COMPLIANCE_KALDI_HPP
// Kaldi-compatible features: a port of torchaudio.compliance.kaldi 2.5.1 (fbank, mfcc, spectrogram and the mel / VTLN
// helpers), op for op so the outputs match torchaudio's (see develop_log 2026-10-23/progress06). The waveform is
// [channel, time]; the output is [frames, features] in the input's dtype and on its device. Deviations from
// torchaudio: input shorter than min_duration gives an empty tensor in the input's dtype from all three (torchaudio's
// spectrogram returns float32, its mfcc raises); and these are std::invalid_argument up front, also for input shorter
// than min_duration: non-floating input, a negative energy_floor, a zero vtln_warp, a non-finite frame shift, an
// out-of-range window_type, and the num_ceps values mfcc fails on after computing (torchaudio raises later, a
// RuntimeError / ValueError / ZeroDivisionError / OverflowError / IndexError, or returns the empty tensor first).
#include <algorithm>
#include <bit>
#include <cmath>
#include <expected>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include "../global_include.hpp"
#include "_functional.hpp" // create_dct

namespace torchmedia::audio::compliance::kaldi {
    enum class window_type { hamming, hanning, povey, rectangular, blackman };

    // Every keyword of torchaudio.compliance.kaldi.spectrogram, same defaults; fbank / mfcc options extend it.
    typedef struct spectrogram_option {
        double _blackman_coeff = 0.42;
        int _channel = -1; // < 0 => 0
        double _dither = 0.0;
        double _energy_floor = 1.0;
        double _frame_length = 25.0; // ms
        double _frame_shift = 10.0; // ms
        double _min_duration = 0.0; // s
        double _preemphasis_coefficient = 0.97;
        bool _raw_energy = true;
        bool _remove_dc_offset = true;
        bool _round_to_power_of_two = true;
        double _sample_frequency = 16000.0;
        bool _snip_edges = true;
        bool _subtract_mean = false;
        kaldi::window_type _window_type = kaldi::window_type::povey;

        // Deducing this: the setters return the most derived option (an rvalue on a temporary), so
        // fbank_option().dither(1).num_mel_bins(80) chains across the base.
        auto blackman_coeff(this auto &&self, const double v) -> decltype(auto) {
            self._blackman_coeff = v;
            return self;
        }
        auto channel(this auto &&self, const int v) -> decltype(auto) {
            self._channel = v;
            return self;
        }
        auto dither(this auto &&self, const double v) -> decltype(auto) {
            self._dither = v;
            return self;
        }
        auto energy_floor(this auto &&self, const double v) -> decltype(auto) {
            self._energy_floor = v;
            return self;
        }
        auto frame_length(this auto &&self, const double v) -> decltype(auto) {
            self._frame_length = v;
            return self;
        }
        auto frame_shift(this auto &&self, const double v) -> decltype(auto) {
            self._frame_shift = v;
            return self;
        }
        auto min_duration(this auto &&self, const double v) -> decltype(auto) {
            self._min_duration = v;
            return self;
        }
        auto preemphasis_coefficient(this auto &&self, const double v) -> decltype(auto) {
            self._preemphasis_coefficient = v;
            return self;
        }
        auto raw_energy(this auto &&self, const bool v) -> decltype(auto) {
            self._raw_energy = v;
            return self;
        }
        auto remove_dc_offset(this auto &&self, const bool v) -> decltype(auto) {
            self._remove_dc_offset = v;
            return self;
        }
        auto round_to_power_of_two(this auto &&self, const bool v) -> decltype(auto) {
            self._round_to_power_of_two = v;
            return self;
        }
        auto sample_frequency(this auto &&self, const double v) -> decltype(auto) {
            self._sample_frequency = v;
            return self;
        }
        auto snip_edges(this auto &&self, const bool v) -> decltype(auto) {
            self._snip_edges = v;
            return self;
        }
        auto subtract_mean(this auto &&self, const bool v) -> decltype(auto) {
            self._subtract_mean = v;
            return self;
        }
        auto window_type(this auto &&self, const kaldi::window_type v) -> decltype(auto) {
            self._window_type = v;
            return self;
        }
    } spectrogram_option_t;

    namespace detail {
        // The mel-bank keywords fbank and mfcc share.
        struct mel_option : spectrogram_option {
            double _high_freq = 0.0; // <= 0 => offset from Nyquist
            bool _htk_compat = false;
            double _low_freq = 20.0;
            int _num_mel_bins = 23;
            bool _use_energy = false;
            double _vtln_high = -500.0; // < 0 => offset from Nyquist
            double _vtln_low = 100.0;
            double _vtln_warp = 1.0;

            auto high_freq(this auto &&self, const double v) -> decltype(auto) {
                self._high_freq = v;
                return self;
            }
            auto htk_compat(this auto &&self, const bool v) -> decltype(auto) {
                self._htk_compat = v;
                return self;
            }
            auto low_freq(this auto &&self, const double v) -> decltype(auto) {
                self._low_freq = v;
                return self;
            }
            auto num_mel_bins(this auto &&self, const int v) -> decltype(auto) {
                self._num_mel_bins = v;
                return self;
            }
            auto use_energy(this auto &&self, const bool v) -> decltype(auto) {
                self._use_energy = v;
                return self;
            }
            auto vtln_high(this auto &&self, const double v) -> decltype(auto) {
                self._vtln_high = v;
                return self;
            }
            auto vtln_low(this auto &&self, const double v) -> decltype(auto) {
                self._vtln_low = v;
                return self;
            }
            auto vtln_warp(this auto &&self, const double v) -> decltype(auto) {
                self._vtln_warp = v;
                return self;
            }
        };
    } // namespace detail

    // Every keyword of torchaudio.compliance.kaldi.fbank, same defaults.
    typedef struct fbank_option : detail::mel_option {
        bool _use_log_fbank = true;
        bool _use_power = true;

        auto use_log_fbank(this auto &&self, const bool v) -> decltype(auto) {
            self._use_log_fbank = v;
            return self;
        }
        auto use_power(this auto &&self, const bool v) -> decltype(auto) {
            self._use_power = v;
            return self;
        }
    } fbank_option_t;

    // Every keyword of torchaudio.compliance.kaldi.mfcc, same defaults.
    typedef struct mfcc_option : detail::mel_option {
        double _cepstral_lifter = 22.0; // 0 => no liftering
        int _num_ceps = 13;

        auto cepstral_lifter(this auto &&self, const double v) -> decltype(auto) {
            self._cepstral_lifter = v;
            return self;
        }
        auto num_ceps(this auto &&self, const int v) -> decltype(auto) {
            self._num_ceps = v;
            return self;
        }
    } mfcc_option_t;

    inline auto inverse_mel_scale_scalar(const double mel_freq) -> double {
        return 700.0 * (std::exp(mel_freq / 1127.0) - 1.0);
    }

    inline auto inverse_mel_scale(const_tensor_lref_t mel_freq) -> tensor_t {
        return 700.0 * ((mel_freq / 1127.0).exp() - 1.0);
    }

    inline auto mel_scale_scalar(const double freq) -> double { return 1127.0 * std::log(1.0 + freq / 700.0); }

    inline auto mel_scale(const_tensor_lref_t freq) -> tensor_t { return 1127.0 * (1.0 + freq / 700.0).log(); }

    namespace detail {
        struct window_properties {
            tensor_t waveform; // the selected channel, [time]
            int64_t window_shift, window_size, padded_window_size;
        };

        // numeric_limits<float>::epsilon() in the input's dtype / device, also for float64 (as torchaudio).
        inline auto get_epsilon(const tensor_options_t &options) -> tensor_t {
            return torch::scalar_tensor(std::numeric_limits<float>::epsilon(), options);
        }

        // Smallest power of 2 >= x (x >= 0; 1 for 0).
        inline auto next_power_of_2(const int64_t x) -> int64_t {
            return static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(x)));
        }

        // [m, window_size] frames of a 1-D waveform. snip_edges: only the frames that fit; otherwise
        // (n + shift / 2) / shift frames over the waveform reflected at both ends ([2, 1, 0, 0, 1, 2]). The strides
        // come from the strided tensor itself (torchaudio takes them from the input, wrong for a non-contiguous one).
        inline auto get_strided(tensor_t waveform, const int64_t window_size, const int64_t window_shift,
                                const bool snip_edges) -> tensor_t {
            const auto num_samples = waveform.size(0);
            int64_t m = 0;
            if (snip_edges) {
                if (num_samples < window_size) {
                    return torch::empty({0, 0}, waveform.options());
                }
                m = 1 + (num_samples - window_size) / window_shift;
            } else {
                const auto reversed = torch::flip(waveform, {0});
                m = (num_samples + window_shift / 2) / window_shift;
                const auto pad = window_size / 2 - window_shift / 2;
                waveform = pad > 0 ? torch::cat({reversed.slice(0, num_samples - pad), waveform, reversed})
                                   : torch::cat({waveform.slice(0, -pad), reversed}); // trim the front
            }
            const auto stride = waveform.stride(0);
            return waveform.as_strided({m, window_size}, {window_shift * stride, stride});
        }

        inline auto feature_window_function(const window_type type, const int64_t window_size,
                                            const double blackman_coeff, const tensor_options_t &options) -> tensor_t {
            switch (type) {
                case window_type::hanning:
                    return torch::hann_window(window_size, false, options);
                case window_type::hamming:
                    return torch::hamming_window(window_size, false, 0.54, 0.46, options);
                case window_type::povey: // like hanning but goes to zero at the edges
                    return torch::hann_window(window_size, false, options).pow(0.85);
                case window_type::rectangular:
                    return torch::ones({window_size}, options);
                case window_type::blackman:
                    break; // other values are rejected by get_waveform_and_window_properties
            }
            // not torch::blackman_window: Kaldi's coefficient is configurable
            const double a = 2 * std::numbers::pi / static_cast<double>(window_size - 1);
            const auto n = torch::arange(window_size, options);
            return blackman_coeff - 0.5 * torch::cos(a * n) + (0.5 - blackman_coeff) * torch::cos(2 * a * n);
        }

        inline auto get_log_energy(const_tensor_lref_t strided_input, const_tensor_lref_t epsilon,
                                   const double energy_floor) -> tensor_t {
            const auto log_energy = torch::maximum(strided_input.pow(2).sum(1), epsilon).log(); // [m]
            if (energy_floor == 0.0) {
                return log_energy;
            }
            return torch::maximum(log_energy, torch::scalar_tensor(std::log(energy_floor), strided_input.options()));
        }

        // torchaudio's asserts as an error string, plus the input and option checks torchaudio fails on later.
        inline auto get_waveform_and_window_properties(const_tensor_lref_t waveform, const spectrogram_option &o)
                -> std::expected<window_properties, std::string> {
            using error = std::unexpected<std::string>;
            if (waveform.dim() != 2 || !waveform.is_floating_point()) {
                return error(std::format("kaldi: expected a floating-point [channel, time] waveform, got {}-D {}",
                                         waveform.dim(), c10::toString(waveform.scalar_type())));
            }
            const int64_t channel = std::max(o._channel, 0);
            if (channel >= waveform.size(0)) {
                return error(std::format("Invalid channel {} for size {}", channel, waveform.size(0)));
            }
            const auto wave = waveform.select(0, channel);
            const auto num_samples = wave.size(0);
            // Python int(): truncation; kept in double until range-checked
            const double shift = std::trunc(o._sample_frequency * o._frame_shift * 0.001);
            const double size = std::trunc(o._sample_frequency * o._frame_length * 0.001);
            if (!(2 <= size && size <= static_cast<double>(num_samples))) {
                return error(std::format("choose a window size {} that is [2, {}]", size, num_samples));
            }
            if (!(0 < shift && std::isfinite(shift))) {
                return error("`window_shift` must be greater than 0");
            }
            const auto window_size = static_cast<int64_t>(size);
            const auto padded_window_size = o._round_to_power_of_two ? next_power_of_2(window_size) : window_size;
            if (padded_window_size % 2 != 0) {
                return error("the padded `window_size` must be divisible by two. use `round_to_power_of_two` or "
                             "change `frame_length`");
            }
            if (!(0.0 <= o._preemphasis_coefficient && o._preemphasis_coefficient <= 1.0)) {
                return error("`preemphasis_coefficient` must be between [0,1]");
            }
            if (!(o._sample_frequency > 0)) {
                return error("`sample_frequency` must be greater than zero");
            }
            if (o._energy_floor < 0.0) {
                return error(std::format("`energy_floor` must be >= 0 (log of {})", o._energy_floor));
            }
            if (const auto t = std::to_underlying(o._window_type);
                t < 0 || t > std::to_underlying(window_type::blackman)) {
                return error(std::format("Invalid window type {}", t));
            }
            // A shift beyond 2n + 1 gives the frames of 2n + 1 (one with snip_edges, none without); keeps the cast
            // defined for huge finite shifts.
            const auto window_shift = static_cast<int64_t>(std::min(shift, 2.0 * static_cast<double>(num_samples) + 1));
            return window_properties{wave, window_shift, window_size, padded_window_size};
        }

        // Frames after dither, DC removal, pre-emphasis, windowing and zero padding ([m, padded_window_size]), and
        // their log energy ([m], before pre-emphasis when raw_energy).
        inline auto get_window(const_tensor_lref_t waveform, const int64_t padded_window_size,
                               const int64_t window_size, const int64_t window_shift, const spectrogram_option &o)
                -> std::tuple<tensor_t, tensor_t> {
            namespace F = torch::nn::functional;
            const auto options = waveform.options();
            const auto epsilon = get_epsilon(options);
            auto strided_input = get_strided(waveform, window_size, window_shift, o._snip_edges);
            if (o._dither != 0.0) { // global generator, same draw as torchaudio
                strided_input = strided_input + torch::randn(strided_input.sizes(), options) * o._dither;
            }
            if (o._remove_dc_offset) {
                strided_input = strided_input - torch::mean(strided_input, 1).unsqueeze(1);
            }
            tensor_t signal_log_energy;
            if (o._raw_energy) {
                signal_log_energy = get_log_energy(strided_input, epsilon, o._energy_floor);
            }
            if (o._preemphasis_coefficient != 0.0) { // x[j] -= c * x[max(0, j - 1)]
                const auto offset =
                        F::pad(strided_input.unsqueeze(0), F::PadFuncOptions({1, 0}).mode(torch::kReplicate))
                                .squeeze(0);
                strided_input = strided_input - o._preemphasis_coefficient * offset.slice(1, 0, -1);
            }
            strided_input =
                    strided_input *
                    feature_window_function(o._window_type, window_size, o._blackman_coeff, options).unsqueeze(0);
            if (padded_window_size != window_size) {
                strided_input = torch::constant_pad_nd(strided_input, {0, padded_window_size - window_size}, 0);
            }
            if (!o._raw_energy) {
                signal_log_energy = get_log_energy(strided_input, epsilon, o._energy_floor);
            }
            return {strided_input, signal_log_energy};
        }

        inline auto subtract_column_mean(const_tensor_lref_t tensor, const bool subtract_mean) -> tensor_t {
            return subtract_mean ? tensor - torch::mean(tensor, 0).unsqueeze(0) : tensor;
        }

        // [num_mel_bins, num_ceps]: torchaudio's ortho DCT with Kaldi's first column sqrt(1 / num_mel_bins).
        inline auto get_dct_matrix(const int64_t num_ceps, const int64_t num_mel_bins) -> tensor_t {
            auto dct_matrix = functional::create_dct(static_cast<int>(num_mel_bins), static_cast<int>(num_mel_bins));
            dct_matrix.select(1, 0).fill_(std::sqrt(1 / static_cast<double>(num_mel_bins)));
            return dct_matrix.slice(1, 0, num_ceps);
        }

        // [num_ceps] cepstral liftering coefficients (C0 is not affected).
        inline auto get_lifter_coeffs(const int64_t num_ceps, const double cepstral_lifter) -> tensor_t {
            const auto i = torch::arange(num_ceps);
            return 1.0 + 0.5 * cepstral_lifter * torch::sin(std::numbers::pi * i / cepstral_lifter);
        }

        inline auto check_vtln_warp(const double vtln_low_cutoff, const double vtln_high_cutoff, const double low_freq,
                                    const double high_freq, const double vtln_warp_factor) -> std::string {
            if (!(vtln_low_cutoff > low_freq)) {
                return "be sure to set the vtln_low option higher than low_freq";
            }
            if (!(vtln_high_cutoff < high_freq)) {
                return "be sure to set the vtln_high option lower than high_freq [or negative]";
            }
            if (vtln_warp_factor == 0.0) {
                return "vtln_warp_factor must be non-zero";
            }
            const double l = vtln_low_cutoff * std::max(1.0, vtln_warp_factor);
            const double h = vtln_high_cutoff * std::min(1.0, vtln_warp_factor);
            if (!(l > low_freq && h < high_freq)) {
                return std::format("vtln: inflection points l {} and h {} must lie in (low_freq {}, high_freq {})", l,
                                   h, low_freq, high_freq);
            }
            return {};
        }

        inline auto check_mel_banks(const int64_t num_bins, const int64_t window_length_padded,
                                    const double sample_freq, const double low_freq, double high_freq,
                                    const double vtln_low, double vtln_high, const double vtln_warp_factor)
                -> std::string {
            if (!(num_bins > 3)) {
                return "Must have at least 3 mel bins";
            }
            if (window_length_padded % 2 != 0) {
                return std::format("window_length_padded {} must be even", window_length_padded);
            }
            const double nyquist = 0.5 * sample_freq;
            if (high_freq <= 0.0) {
                high_freq += nyquist;
            }
            if (!((0.0 <= low_freq && low_freq < nyquist) && (0.0 < high_freq && high_freq <= nyquist) &&
                  (low_freq < high_freq))) {
                return std::format("Bad values in options: low-freq {} and high-freq {} vs. nyquist {}", low_freq,
                                   high_freq, nyquist);
            }
            if (vtln_high < 0.0) {
                vtln_high += nyquist;
            }
            if (vtln_warp_factor == 1.0) {
                return {};
            }
            if (!((low_freq < vtln_low && vtln_low < high_freq) && (0.0 < vtln_high && vtln_high < high_freq) &&
                  (vtln_low < vtln_high))) {
                return std::format("Bad values in options: vtln-low {} and vtln-high {}, versus low-freq {} and "
                                   "high-freq {}",
                                   vtln_low, vtln_high, low_freq, high_freq);
            }
            return check_vtln_warp(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor);
        }
    } // namespace detail

    // Kaldi's VTLN warp of frequencies in Hz: piecewise linear with inflection points l and h, F(f) = f / warp in
    // between, identity outside [low_freq, high_freq].
    inline auto vtln_warp_freq(const double vtln_low_cutoff, const double vtln_high_cutoff, const double low_freq,
                               const double high_freq, const double vtln_warp_factor, const_tensor_lref_t freq)
            -> tensor_t {
        if (const auto err =
                    detail::check_vtln_warp(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor);
            !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), err);
        }
        const double l = vtln_low_cutoff * std::max(1.0, vtln_warp_factor);
        const double h = vtln_high_cutoff * std::min(1.0, vtln_warp_factor);
        const double scale = 1.0 / vtln_warp_factor;
        const double Fl = scale * l;
        const double Fh = scale * h;
        const double scale_left = (Fl - low_freq) / (l - low_freq);
        const double scale_right = (high_freq - Fh) / (high_freq - h);
        // torchaudio's masked writes, later ones winning: outside, before l, before h, after h
        const auto outside_low_high_freq = torch::lt(freq, low_freq) | torch::gt(freq, high_freq);
        return torch::where(outside_low_high_freq, freq,
                            torch::where(torch::lt(freq, l), low_freq + scale_left * (freq - low_freq),
                                         torch::where(torch::lt(freq, h), scale * freq,
                                                      high_freq + scale_right * (freq - high_freq))));
    }

    inline auto vtln_warp_mel_freq(const double vtln_low_cutoff, const double vtln_high_cutoff, const double low_freq,
                                   const double high_freq, const double vtln_warp_factor, const_tensor_lref_t mel_freq)
            -> tensor_t {
        return mel_scale(vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor,
                                        inverse_mel_scale(mel_freq)));
    }

    // (bins [num_bins, window_length_padded / 2], center_freqs [num_bins, 1]), float32 on the CPU as in torchaudio.
    inline auto get_mel_banks(const int64_t num_bins, const int64_t window_length_padded, const double sample_freq,
                              const double low_freq, double high_freq, const double vtln_low, double vtln_high,
                              const double vtln_warp_factor) -> std::tuple<tensor_t, tensor_t> {
        if (const auto err = detail::check_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq,
                                                     vtln_low, vtln_high, vtln_warp_factor);
            !err.empty()) {
            return handle_exceptions<std::tuple<tensor_t, tensor_t>, std::invalid_argument>(err);
        }
        const double nyquist = 0.5 * sample_freq;
        if (high_freq <= 0.0) {
            high_freq += nyquist;
        }
        if (vtln_high < 0.0) {
            vtln_high += nyquist;
        }
        const double fft_bin_width = sample_freq / static_cast<double>(window_length_padded);
        const double mel_low_freq = mel_scale_scalar(low_freq);
        const double mel_high_freq = mel_scale_scalar(high_freq);
        // num_bins + 1: the bins spread out to the sides
        const double mel_freq_delta = (mel_high_freq - mel_low_freq) / static_cast<double>(num_bins + 1);
        const auto bin = torch::arange(num_bins).unsqueeze(1);
        auto left_mel = mel_low_freq + bin * mel_freq_delta; // [num_bins, 1]
        auto center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta;
        auto right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta;
        if (vtln_warp_factor != 1.0) {
            left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel);
            center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel);
            right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel);
        }
        const auto center_freqs = inverse_mel_scale(center_mel);
        const auto mel = mel_scale(fft_bin_width * torch::arange(window_length_padded / 2, torch::kFloat)).unsqueeze(0);
        const auto up_slope = (mel - left_mel) / (center_mel - left_mel);
        const auto down_slope = (right_mel - mel) / (right_mel - center_mel);
        if (vtln_warp_factor == 1.0) { // left < center < right: min of the two slopes, negatives clamped
            return {torch::maximum(torch::zeros(1), torch::minimum(up_slope, down_slope)), center_freqs};
        }
        // warping can reorder left, center and right
        const auto up_idx = torch::gt(mel, left_mel) & torch::le(mel, center_mel);
        const auto down_idx = torch::gt(mel, center_mel) & torch::lt(mel, right_mel);
        return {torch::where(down_idx, down_slope, torch::where(up_idx, up_slope, torch::zeros_like(up_slope))),
                center_freqs};
    }

    // Log power spectrum [frames, padded_window_size / 2 + 1]; column 0 is the frame's log energy.
    inline auto spectrogram(const_tensor_lref_t waveform, const spectrogram_option &opt = {}) -> tensor_t {
        const auto props = detail::get_waveform_and_window_properties(waveform, opt);
        if (!props) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), props.error());
        }
        const auto &[wave, window_shift, window_size, padded_window_size] = *props;
        if (static_cast<double>(wave.size(0)) < opt._min_duration * opt._sample_frequency) {
            return torch::empty({0}, waveform.options()); // signal is too short
        }
        const auto [strided_input, signal_log_energy] =
                detail::get_window(wave, padded_window_size, window_size, window_shift, opt);
        auto power_spectrum =
                torch::maximum(torch::fft::rfft(strided_input).abs().pow(2.0), detail::get_epsilon(waveform.options()))
                        .log();
        power_spectrum.select(1, 0).copy_(signal_log_energy);
        return detail::subtract_column_mean(power_spectrum, opt._subtract_mean);
    }

    // Mel filterbank energies [frames, num_mel_bins (+ 1 with use_energy: last column if htk_compat, else first)].
    inline auto fbank(const_tensor_lref_t waveform, const fbank_option &opt = {}) -> tensor_t {
        const auto props = detail::get_waveform_and_window_properties(waveform, opt);
        if (!props) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), props.error());
        }
        const auto &[wave, window_shift, window_size, padded_window_size] = *props;
        if (static_cast<double>(wave.size(0)) < opt._min_duration * opt._sample_frequency) {
            return torch::empty({0}, waveform.options()); // signal is too short
        }
        // validated before the frames are drawn (torchaudio builds the banks after, but draws nothing in between)
        if (const auto err =
                    detail::check_mel_banks(opt._num_mel_bins, padded_window_size, opt._sample_frequency, opt._low_freq,
                                            opt._high_freq, opt._vtln_low, opt._vtln_high, opt._vtln_warp);
            !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), err);
        }
        const auto [strided_input, signal_log_energy] =
                detail::get_window(wave, padded_window_size, window_size, window_shift, opt);
        auto spectrum = torch::fft::rfft(strided_input).abs(); // [m, padded_window_size / 2 + 1]
        if (opt._use_power) {
            spectrum = spectrum.pow(2.0);
        }
        const auto banks =
                std::get<0>(get_mel_banks(opt._num_mel_bins, padded_window_size, opt._sample_frequency, opt._low_freq,
                                          opt._high_freq, opt._vtln_low, opt._vtln_high, opt._vtln_warp))
                        .to(waveform.device(), waveform.scalar_type());
        // the banks have no Nyquist column: pad it with zeros
        auto mel_energies = torch::mm(spectrum, torch::constant_pad_nd(banks, {0, 1}, 0).t()); // [m, num_mel_bins]
        if (opt._use_log_fbank) {
            mel_energies = torch::maximum(mel_energies, detail::get_epsilon(waveform.options())).log();
        }
        if (opt._use_energy) {
            const auto energy = signal_log_energy.unsqueeze(1);
            mel_energies =
                    opt._htk_compat ? torch::cat({mel_energies, energy}, 1) : torch::cat({energy, mel_energies}, 1);
        }
        return detail::subtract_column_mean(mel_energies, opt._subtract_mean);
    }

    // MFCC [frames, num_ceps]: DCT of the log mel energies; with use_energy C0 is the log energy; htk_compat moves C0
    // (or the energy) to the last column.
    inline auto mfcc(const_tensor_lref_t waveform, const mfcc_option &opt = {}) -> tensor_t {
        if (opt._num_ceps > opt._num_mel_bins) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    torch::empty({1}), std::format("num_ceps cannot be larger than num_mel_bins: {} vs {}",
                                                   opt._num_ceps, opt._num_mel_bins));
        }
        // torchaudio fails on these later (arange(num_ceps) for the lifter, feature[:, 0] with no cepstra); otherwise a
        // negative num_ceps is its Python slice of the DCT columns: all but the last -num_ceps
        if (opt._num_ceps < 0 && opt._cepstral_lifter != 0.0) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    torch::empty({1}), std::format("a negative num_ceps ({}) needs cepstral_lifter 0", opt._num_ceps));
        }
        const auto num_cepstra = opt._num_ceps < 0 ? std::max<int64_t>(int64_t{opt._num_mel_bins} + opt._num_ceps, 0)
                                                   : int64_t{opt._num_ceps};
        if (num_cepstra == 0 && (opt._use_energy || opt._htk_compat)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    torch::empty({1}),
                    std::format("num_ceps {} leaves no C0 column for use_energy / htk_compat", opt._num_ceps));
        }
        fbank_option fbank_opt; // log power mel energies (use_log_fbank, use_power), mean subtracted at the end
        static_cast<detail::mel_option &>(fbank_opt) = opt;
        fbank_opt._subtract_mean = false;
        auto feature = fbank(waveform, fbank_opt); // [m, num_mel_bins (+ 1)]
        if (feature.dim() != 2) {
            return feature; // shorter than min_duration, or fbank's default value without exceptions
        }
        tensor_t signal_log_energy;
        if (opt._use_energy) {
            signal_log_energy = feature.select(1, opt._htk_compat ? opt._num_mel_bins : 0);
            const int mel_offset = opt._htk_compat ? 0 : 1;
            feature = feature.slice(1, mel_offset, opt._num_mel_bins + mel_offset);
        }
        const auto device = waveform.device();
        const auto dtype = waveform.scalar_type();
        feature = feature.matmul(detail::get_dct_matrix(opt._num_ceps, opt._num_mel_bins).to(device, dtype));
        if (opt._cepstral_lifter != 0.0) {
            feature *= detail::get_lifter_coeffs(opt._num_ceps, opt._cepstral_lifter).unsqueeze(0).to(device, dtype);
        }
        if (opt._use_energy) {
            feature.select(1, 0).copy_(signal_log_energy);
        }
        if (opt._htk_compat) {
            auto energy = feature.select(1, 0).unsqueeze(1);
            feature = feature.slice(1, 1);
            if (!opt._use_energy) {
                energy *= std::sqrt(2.0); // undo the ortho DCT's C0 scale
            }
            feature = torch::cat({feature, energy}, 1);
        }
        return detail::subtract_column_mean(feature, opt._subtract_mean);
    }
} // namespace torchmedia::audio::compliance::kaldi
#endif // LIB_TORCH_MEDIA_AUDIO_COMPLIANCE_KALDI_HPP
