#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>
#include <utility>
#include "../global_include.hpp"
#include "_functional_filtering.hpp" // lfilter / biquads for deemphasis and loudness
#include "_functional_methods_options.hpp"

namespace torchmedia::audio::functional {
    // torchaudio convolve modes. Scoped, so full/valid/same do not collide with torch::full and friends.
    enum class convolve_mode { full, valid, same };

    inline auto _apply_convolve_mode(tensor_t conv_result, const int64_t x_length, const int64_t y_length,
                                     const convolve_mode mode) -> tensor_t {
        if (mode == convolve_mode::valid) {
            const auto target_length = std::max(x_length, y_length) - std::min(x_length, y_length) + 1;
            const auto start_idx = (conv_result.size(-1) - target_length) / 2;
            return conv_result.slice(-1, start_idx, start_idx + target_length);
        }
        if (mode == convolve_mode::same) {
            const auto start_idx = (conv_result.size(-1) - x_length) / 2;
            return conv_result.slice(-1, start_idx, start_idx + x_length);
        }
        if (mode != convolve_mode::full) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Unrecognized convolve mode. Use full, valid, or same.");
        }
        return conv_result;
    }

    // Operand check shared by convolve/fftconvolve (torchaudio _check_shape_compatible), returning an error message
    // or "". Deliberate extension: a lower-rank operand is left-padded with size-1 dims instead of being rejected.
    inline auto _align_convolve_operands(tensor_t &x, tensor_t &y) -> str_t {
        if (x.dim() == 0 || y.dim() == 0) {
            return "Inputs must be at least 1D.";
        }
        while (x.dim() < y.dim()) {
            x = x.unsqueeze(0);
        }
        while (y.dim() < x.dim()) {
            y = y.unsqueeze(0);
        }
        for (int64_t i = 0; i < x.dim() - 1; ++i) {
            if (x.size(i) != y.size(i) && x.size(i) != 1 && y.size(i) != 1) {
                return "Leading dimensions of x and y are not broadcastable.";
            }
        }
        return {};
    }

    inline auto db_to_amplitude(tensor_t x, double ref, double power) -> tensor_t {
        // torchaudio: DB_to_amplitude(x, ref, power) = ref * (10^(0.1*x))^power
        return ref * torch::pow(torch::pow(10.0, 0.1 * x), power);
    }

    inline auto convolve(tensor_t x, tensor_t y, const convolve_mode mode = convolve_mode::full) -> tensor_t {
        if (const auto err = _align_convolve_operands(x, y); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        const auto n_dims_x = x.dim();

        // torchaudio's 'same'/'valid' crop is defined w.r.t. the ORIGINAL first input x, so record its
        // length BEFORE the swap. Convolution is commutative, so swapping to keep the longer signal first
        // is fine for the math; only the final crop length must use the original x length.
        const auto original_x_size = x.size(-1);
        const auto original_y_size = y.size(-1);
        if (x.size(-1) < y.size(-1)) {
            std::swap(x, y);
        }

        const auto x_size = x.size(-1);
        const auto y_size = y.size(-1);
        const auto leading_dims_count = n_dims_x - 1;
        const auto shape_x = x.sizes();
        const auto shape_y = y.sizes();

        // 2. 广播 (Broadcasting) 计算共同形状
        std::vector<int64_t> new_shape(leading_dims_count);
        for (int i = 0; i < leading_dims_count; i++) {
            new_shape[i] = std::max(shape_x[i], shape_y[i]);
        }

        auto broadcast_shape_x = new_shape;
        broadcast_shape_x.push_back(x_size);
        x = x.broadcast_to(broadcast_shape_x);

        auto broadcast_shape_y = new_shape;
        broadcast_shape_y.push_back(y_size);
        y = y.broadcast_to(broadcast_shape_y);

        // 3. 计算总信号数量 (Flatten Batch)
        auto num_signals = 1LL;
        for (int i = 0; i < leading_dims_count; i++) {
            num_signals *= new_shape[i];
        }

        // 4. Reshape 为分组卷积 (Grouped Convolution) 兼容的形状
        // 关键修复:
        // Input  => [1, num_signals, x_size]  (Batch=1, Channels=num_signals)
        // Weight => [num_signals, 1, y_size]  (Out=num_signals, In/Groups=1)
        const auto reshaped_x = x.reshape({1, num_signals, x_size});

        // 注意：Convolution 实际上是互相关，严格数学卷积需要 flip kernel。
        // 原版代码有 flip，这里保留。
        const auto reshaped_y = y.flip(-1).reshape({num_signals, 1, y_size});

        // 5. 执行卷积
        // padding(y_size - 1) 是为了实现 'full' 模式，之后再裁切
        const auto conv_out =
                torch::nn::functional::conv1d(reshaped_x, reshaped_y,
                                              torch::nn::functional::Conv1dFuncOptions()
                                                      .stride(1)
                                                      .groups(num_signals) // 分组数 = 信号数，实现 Depthwise
                                                      .padding(y_size - 1));

        // 6. 恢复形状
        // conv_out 目前是 [1, num_signals, out_len]
        const auto output_length = conv_out.size(-1);
        auto output_shape = new_shape;
        output_shape.push_back(output_length);

        // 先 reshape 去掉 batch=1 维度，再 reshape 回广播后的维度
        const auto result = conv_out.reshape(output_shape);

        // 7. 应用裁剪模式 (Full/Valid/Same) — crop length uses the ORIGINAL x/y lengths
        return _apply_convolve_mode(result, original_x_size, original_y_size, mode);
    }


    // Mirrors torchaudio.functional.amplitude_to_DB: multiplier * log10(clamp(x, amin)) - multiplier * db_multiplier,
    // then, with top_db, a floor at max - top_db where the max is taken over (C, F, T) of each item of the input
    // packed as (-1, C, F, T), C = shape[-3] for > 2-D input else 1 (so all channels of one clip share the cutoff).
    // Complex input is converted to its magnitude first (torchaudio raises instead).
    inline auto amplitude_to_DB(tensor_t signal, const amplitude_to_db_option &option = {}) -> tensor_t {
        if (signal.is_complex()) {
            signal = signal.abs(); // sqrt(real^2 + imag^2)
        }
        // `signal` is already power (multiplier=10) or magnitude (multiplier=20) — do NOT square here.
        auto db = option.multiplier * torch::log10(torch::clamp_min(signal, option.amin));
        db = db - option.multiplier * option.db_multiplier;
        if (option.apply_top_db) {
            if (db.dim() < 2) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        "amplitude_to_DB: top_db needs at least (freq, time) dimensions.");
            }
            const auto shape = db.sizes().vec();
            const int64_t channels = db.dim() > 2 ? shape[shape.size() - 3] : 1;
            const auto packed = db.reshape({-1, channels, shape[shape.size() - 2], shape.back()});
            const auto floor = (packed.amax({-3, -2, -1}) - option.top_db).view({-1, 1, 1, 1});
            db = torch::maximum(packed, floor).reshape(shape);
        }
        return db;
    }


    // Mirrors torchaudio.functional.spectrogram: pad, pack (..., time) to (-1, time), stft ("frame_length" is stft's
    // normalized=true, i.e. * n_fft^-0.5), unpack, "window" norm (/ sqrt(sum(win^2))), then power unless complex.
    // win_length / hop_length 0 derive from n_fft like the transform layer. normalize_method is checked even when
    // normalized is false (torchaudio has one `normalized` argument, so it never holds an unused invalid method).
    inline auto spectrogram(tensor_t signal, const spectrogram_option &option) -> tensor_t {
        if (option._normalize_method != "frame_length" && option._normalize_method != "window") {
            return handle_exceptions<tensor_t, std::invalid_argument>("Invalid normalize_method: \"" +
                                                                      option._normalize_method +
                                                                      "\" (use \"window\" or \"frame_length\").");
        }
        const bool frame_length_norm = option._normalized && option._normalize_method == "frame_length";
        const bool window_norm = option._normalized && option._normalize_method == "window";
        if (signal.dim() < 1) {
            return handle_exceptions<tensor_t, std::invalid_argument>("spectrogram: expected a (..., time) waveform.");
        }
        if (option._pad > 0) {
            signal = torch::constant_pad_nd(signal, {option._pad, option._pad}, 0);
        }

        const int n_fft = option._n_fft;
        const int win_length = detail::derive_win_length(option._win_length, n_fft);
        const int hop_length = detail::derive_hop_length(option._hop_length, win_length);
        const tensor_t window =
                option._window.defined()
                        ? option._window
                        : torch::hann_window(win_length,
                                             tensor_options_t().dtype(signal.dtype()).device(signal.device()));
        const auto shape = signal.sizes().vec();
        auto spec_f = torch::stft(signal.reshape({-1, shape.back()}), n_fft, hop_length, win_length, window,
                                  option._center, option._pad_mode, frame_length_norm, option._onesided,
                                  /*return_complex=*/true);
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(spec_f.size(-2));
        out_shape.push_back(spec_f.size(-1));
        spec_f = spec_f.reshape(out_shape);
        if (window_norm) {
            spec_f = spec_f / window.pow(2).sum().sqrt();
        }
        if (option._return_complex) {
            return spec_f;
        }
        return torch::pow(torch::abs(spec_f), option._power);
    }


    // Triangular filterbank shared by mel/linear fbanks (mirrors torchaudio's _create_triangular_filterbank).
    inline auto _create_triangular_filterbank(const_tensor_lref_t all_freqs, const_tensor_lref_t f_pts) -> tensor_t {
        using namespace torch::indexing;
        const auto f_diff = f_pts.index({Slice(1, None)}) - f_pts.index({Slice(None, -1)}); // (n_filter+1,)
        const auto slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1); // (n_freqs, n+2)
        const auto down_slopes = (-1.0 * slopes.index({Slice(), Slice(None, -2)})) / f_diff.index({Slice(None, -1)});
        const auto up_slopes = slopes.index({Slice(), Slice(2, None)}) / f_diff.index({Slice(1, None)});
        return torch::clamp_min(torch::minimum(down_slopes, up_slopes), 0.0);
    }

    // String checks shared by mel_filter_bank and its composites (melspectrogram, mfcc), which validate up front so
    // a LIB_TORCH_MEDIA_NO_EXCEPTIONS build never computes from a default-valued filterbank. "" when valid.
    inline auto _check_mel_strings(const std::string &norm, const std::string &mel_scale) -> str_t {
        if (!norm.empty() && norm != "slaney") {
            return "norm must be \"\" (None) or \"slaney\".";
        }
        if (mel_scale != "htk" && mel_scale != "slaney") {
            return "mel_scale must be \"htk\" or \"slaney\".";
        }
        return {};
    }
    inline auto _check_melspectrogram_args(const mel_spectrogram_option &opt) -> str_t {
        if (opt.f_min > detail::effective_f_max(opt.f_max, opt.sample_rate)) {
            return "melspectrogram: require f_min <= f_max.";
        }
        return _check_mel_strings(opt.norm, opt.mel_scale);
    }
    inline auto _check_dct_norm(const std::string &norm) -> str_t {
        return norm.empty() || norm == "ortho" ? str_t{} : str_t{"norm must be \"ortho\" or \"\" (None)."};
    }

    // Mel filterbank of shape [n_mels, n_freqs] (torchaudio melscale_fbanks transposed). Mirrors melscale_fbanks
    // op-for-op, including the integer sample_rate // 2 for the bin frequencies and the default f_max (f_max <= 0).
    // Built in float32 on the CPU like torchaudio; callers move it to the data's device/dtype.
    inline auto mel_filter_bank(int n_mels, const double f_min, double f_max, const int sample_rate,
                                int n_stft_bins, // 通常 = n_fft/2 + 1
                                const std::string &norm, // "slaney" 或 ""
                                const std::string &mel_scale // "htk" / "slaney"
                                ) -> tensor_t {
        using namespace torch::indexing;
        if (const auto err = _check_mel_strings(norm, mel_scale); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        f_max = detail::effective_f_max(f_max, sample_rate);
        const bool htk = mel_scale == "htk";
        // Slaney: linear below 1 kHz, logarithmic above (torchaudio _hz_to_mel / _mel_to_hz with f_min = 0).
        const double f_sp = 200.0 / 3.0;
        const double min_log_hz = 1000.0;
        const double min_log_mel = min_log_hz / f_sp;
        const double logstep = std::log(6.4) / 27.0;
        const auto hz_to_mel = [&](const double freq) -> double {
            if (htk) {
                return 2595.0 * std::log10(1.0 + freq / 700.0);
            }
            return freq >= min_log_hz ? min_log_mel + std::log(freq / min_log_hz) / logstep : freq / f_sp;
        };

        const auto all_freqs = torch::linspace(0, sample_rate / 2, n_stft_bins);
        const auto m_pts = torch::linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2);
        tensor_t f_pts;
        if (htk) {
            f_pts = 700.0 * (torch::pow(10.0, m_pts / 2595.0) - 1.0);
        } else {
            f_pts = f_sp * m_pts;
            const auto log_t = m_pts >= min_log_mel;
            f_pts.index_put_({log_t}, min_log_hz * torch::exp(logstep * (m_pts.index({log_t}) - min_log_mel)));
        }
        auto fb = _create_triangular_filterbank(all_freqs, f_pts); // (n_freqs, n_mels)
        if (norm == "slaney") {
            // Slaney-style mel is scaled to be approx constant energy per channel.
            fb = fb * (2.0 / (f_pts.index({Slice(2, n_mels + 2)}) - f_pts.index({Slice(None, n_mels)}))).unsqueeze(0);
        }
        return fb.t().contiguous();
    }

    inline auto mel_scale(const_tensor_lref_t spec, const_tensor_lref_t fb) -> tensor_t {
        // fb is [n_mels, n_freqs]; spec is [..., n_freqs, time]. matmul broadcasts over the leading
        // dims, projecting to [..., n_mels, time] and PRESERVING the input rank (torchaudio contract).
        // (An earlier version collapsed all leading dims into a single batch axis, which added a
        // spurious dimension for 1D/2D input and flattened batch/channel structure for >3D input.)
        return matmul(fb, spec);
    }

    // Mirrors torchaudio.transforms.MelSpectrogram (always one-sided; f_min must not exceed the effective f_max).
    // The filterbank is moved to the spectrogram's device and dtype.
    inline auto melspectrogram(const_tensor_lref_t waveform, const mel_spectrogram_option &opt) -> tensor_t {
        if (const auto err = _check_melspectrogram_args(opt); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        spectrogram_option sp_opt;
        sp_opt._pad = opt.pad;
        sp_opt._n_fft = opt.n_fft;
        sp_opt._win_length = opt.win_length;
        sp_opt._hop_length = opt.hop_length;
        sp_opt._center = opt.center;
        sp_opt._pad_mode = opt.pad_mode;
        sp_opt._onesided = true; // torchaudio deprecated `onesided`; the fb needs n_fft / 2 + 1 bins
        sp_opt._power = opt.power; // 1 => 幅度谱, 2 => 功率谱
        sp_opt._normalized = opt.normalized;
        sp_opt._return_complex = false; // 我们要实数谱，后面才能和 fb 做乘法

        const auto spec = spectrogram(waveform, sp_opt); // [..., freq, time]
        if (!spec.defined()) {
            return spec; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: spectrogram rejected the waveform
        }
        const auto fb = mel_filter_bank(opt.n_mels, opt.f_min, opt.f_max, opt.sample_rate, opt.n_fft / 2 + 1, opt.norm,
                                        opt.mel_scale)
                                .to(spec.device(), spec.scalar_type()); // [n_mels, n_stft_bins]
        return mel_scale(spec, fb); // [..., n_mels, time]
    }

    // DCT-II matrix of shape [n_mels, n_mfcc] (mirrors torchaudio.functional.create_dct op-for-op, in float32).
    // dct[k,n] = cos(pi/n_mels * (n+0.5) * k); norm=="" -> *2; norm=="ortho" -> row0 *= 1/sqrt(2), then
    // *sqrt(2/n_mels); returned transposed (a view, like torchaudio).
    inline auto create_dct(int n_mfcc, int n_mels, const std::string &norm = "ortho") -> tensor_t {
        if (const auto err = _check_dct_norm(norm); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        const double pi = std::acos(-1.0);
        const auto n = torch::arange(n_mels, torch::kFloat); // [n_mels]
        const auto k = torch::arange(n_mfcc, torch::kFloat).unsqueeze(1); // [n_mfcc, 1]
        auto dct = torch::cos(pi / n_mels * (n + 0.5) * k); // [n_mfcc, n_mels]
        if (norm.empty()) {
            dct.mul_(2.0);
        } else { // "ortho"
            dct[0].mul_(1.0 / std::sqrt(2.0));
            dct.mul_(std::sqrt(2.0 / n_mels));
        }
        return dct.t();
    }

    // MFCC: DCT over the dB (or log) mel spectrogram (mirrors torchaudio.transforms.MFCC). opt.sample_rate is the
    // rate (torchaudio MFCC(sample_rate, melkwargs)) and overrides opt.mel.sample_rate.
    inline auto mfcc(const_tensor_lref_t waveform, const mfcc_option &opt) -> tensor_t {
        if (opt.n_mfcc > opt.mel.n_mels) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Cannot select more MFCC coefficients than mel bins.");
        }
        auto mel_opt = opt.mel;
        mel_opt.sample_rate = opt.sample_rate;
        auto err = _check_dct_norm(opt.norm);
        if (err.empty()) {
            err = _check_melspectrogram_args(mel_opt);
        }
        if (!err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        const auto dct_mat = create_dct(opt.n_mfcc, opt.mel.n_mels, opt.norm); // [n_mels, n_mfcc]
        const auto mel_spec = melspectrogram(waveform, mel_opt); // [..., n_mels, time]
        if (!mel_spec.defined()) {
            return mel_spec; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: the waveform was rejected
        }
        tensor_t feat;
        if (opt.log_mels) {
            feat = torch::log(mel_spec + 1e-6);
        } else {
            feat = amplitude_to_DB(mel_spec, amplitude_to_db_option().set_top_db(opt.top_db)); // power, ref 1.0
        }
        // apply the DCT along the mel axis: [..., n_mels, time] -> [..., n_mfcc, time]
        return torch::matmul(feat.transpose(-2, -1), dct_mat.to(feat.device(), feat.scalar_type())).transpose(-2, -1);
    }

    // Griffin-Lim phase reconstruction (mirrors torchaudio.functional.griffinlim op-for-op: `length` goes to every
    // istft and rand_init draws torch.rand(complex)). The default window follows the spectrogram's dtype and device.
    inline auto griffinlim(const_tensor_lref_t specgram, const griffinlim_option &opt) -> tensor_t {
        if (!(opt.momentum >= 0.0 && opt.momentum < 1.0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>("momentum must be in range [0, 1).");
        }
        if (specgram.dim() < 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "griffinlim: expected a (..., freq, time) spectrogram.");
        }
        const int win_length = detail::derive_win_length(opt.win_length, opt.n_fft);
        const int hop_length = detail::derive_hop_length(opt.hop_length, win_length);
        const double momentum = opt.momentum / (1.0 + opt.momentum);
        const c10::optional<int64_t> length = opt.length > 0 ? c10::optional<int64_t>(opt.length) : c10::nullopt;

        const auto shape = specgram.sizes().vec();
        const auto spec =
                specgram.reshape({-1, shape[shape.size() - 2], shape.back()}).pow(1.0 / opt.power); // magnitude
        // dtype after pow: floating even for an integer spectrogram.
        const auto window = opt.window.defined() ? opt.window : torch::hann_window(win_length, spec.options());
        const auto complex_options = spec.options().dtype(c10::toComplexType(spec.scalar_type()));
        auto angles =
                opt.rand_init ? torch::rand(spec.sizes(), complex_options) : torch::ones(spec.sizes(), complex_options);

        tensor_t tprev = torch::zeros({}, spec.options());
        for (int i = 0; i < opt.n_iter; i++) {
            const auto inverse = torch::istft(spec * angles, opt.n_fft, hop_length, win_length, window,
                                              /*center=*/true, /*normalized=*/false, /*onesided=*/true, length,
                                              /*return_complex=*/false);
            const auto rebuilt = torch::stft(inverse, opt.n_fft, hop_length, win_length, window,
                                             /*center=*/true, /*pad_mode=*/"reflect", /*normalized=*/false,
                                             /*onesided=*/true, /*return_complex=*/true);
            angles = rebuilt;
            if (momentum > 0.0) {
                angles = angles - tprev * momentum;
            }
            angles = angles / (angles.abs() + 1e-16);
            tprev = rebuilt;
        }

        const auto waveform = torch::istft(spec * angles, opt.n_fft, hop_length, win_length, window,
                                           /*center=*/true, /*normalized=*/false, /*onesided=*/true, length,
                                           /*return_complex=*/false);
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
        out_shape.push_back(waveform.size(-1));
        return waveform.reshape(out_shape);
    }

    // Resample argument checks shared by resample(), pitch_shift() and transform::Resample (torchaudio's, plus
    // rolloff > 0, which torchaudio leaves to a ZeroDivisionError). Returns an error message or "". Equal rates
    // build no kernel, so, as in torchaudio, the filter arguments are not checked for them.
    inline auto _check_resample_args(int orig_freq, int new_freq, const resample_option &opt) -> str_t {
        if (orig_freq <= 0 || new_freq <= 0) {
            return "Original frequency and desired frequency should be positive.";
        }
        if (orig_freq == new_freq) {
            return {};
        }
        if (opt.lowpass_filter_width <= 0) {
            return "Low pass filter width should be positive.";
        }
        if (!(opt.rolloff > 0.0)) {
            return "rolloff should be positive.";
        }
        return {};
    }

    // Build the [nf, 1, kernel_width] windowed-sinc resampling filterbank (mirrors torchaudio's
    // _get_sinc_resample_kernel op-for-op, sinc_interp_hann). Split out from resample() so transforms can cache it.
    // A dtype in `options` computes the kernel in that dtype (resample(): the waveform's); without one it is computed
    // in float64 and returned as float32, like torchaudio's dtype=None (transform::Resample's default).
    inline auto _sinc_resample_kernel(int orig_freq, int new_freq, int gcd, const resample_option &opt,
                                      const tensor_options_t &options) -> std::pair<tensor_t, int> {
        const int of = orig_freq / gcd;
        const int nf = new_freq / gcd;
        const int lfw = opt.lowpass_filter_width;
        const double base_freq = std::min(of, nf) * opt.rolloff;
        const int width = static_cast<int>(std::ceil(static_cast<double>(lfw) * of / base_freq));
        const double pi = std::acos(-1.0);
        const bool has_dtype = options.has_dtype();
        const auto idx = torch::arange(-width, width + of, has_dtype ? options : options.dtype(torch::kFloat64))
                                 .reshape({1, 1, -1}) /
                         of; // [1,1,K]
        // dtype=None: torchaudio's phase arange is int64, so `/ new_freq` is float32 before it meets the float64 idx.
        const auto phase =
                torch::arange(0, -nf, -1, has_dtype ? options : options.dtype(torch::kLong)).reshape({nf, 1, 1}) /
                nf; // [nf,1,1]
        auto t = ((phase + idx) * base_freq).clamp(-lfw, lfw);
        const auto window = torch::cos(t * pi / lfw / 2).pow(2);
        t = t * pi;
        const double scale = base_freq / of;
        const auto kernels = torch::where(t == 0, torch::ones_like(t), torch::sin(t) / t) * (window * scale);
        return {has_dtype ? kernels : kernels.to(torch::kFloat), width};
    }

    // Apply a precomputed sinc kernel: pad, strided conv1d, crop to the target length (mirrors torchaudio's
    // _apply_sinc_resample_kernel). The kernel is moved to the waveform's device and dtype (a no-op for resample()).
    inline auto _apply_sinc_resample_kernel(const_tensor_lref_t waveform, int orig_freq, int new_freq, int gcd,
                                            const_tensor_lref_t kernels, int width) -> tensor_t {
        if (!waveform.is_floating_point()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Expected floating point type for waveform tensor.");
        }
        if (waveform.dim() < 1) {
            return handle_exceptions<tensor_t, std::invalid_argument>("resample: expected a (..., time) waveform.");
        }
        const int of = orig_freq / gcd;
        const int nf = new_freq / gcd;
        const auto shape = waveform.sizes().vec();
        const int64_t length = shape.back();
        auto wav = waveform.reshape({-1, length});
        const int64_t num_wavs = wav.size(0);
        wav = torch::constant_pad_nd(wav, {width, width + of}, 0);
        auto resampled =
                torch::nn::functional::conv1d(wav.unsqueeze(1), kernels.to(waveform.device(), waveform.scalar_type()),
                                              torch::nn::functional::Conv1dFuncOptions().stride(of));
        resampled = resampled.transpose(1, 2).reshape({num_wavs, -1});
        // torchaudio: ceil(torch.as_tensor(new * length / orig)) -- a float32 tensor, so ceil the float32 value.
        const auto target_length =
                static_cast<int64_t>(std::ceil(static_cast<float>(static_cast<double>(nf) * length / of)));
        resampled = resampled.slice(-1, 0, target_length);
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(resampled.size(-1));
        return resampled.reshape(out_shape);
    }

    // Band-limited sinc resampling (mirrors torchaudio.functional.resample, sinc_interp_hann): equal rates return
    // the input; the kernel is built in the waveform's dtype on its device.
    inline auto resample(const_tensor_lref_t waveform, int orig_freq, int new_freq, const resample_option &opt = {})
            -> tensor_t {
        if (const auto err = _check_resample_args(orig_freq, new_freq, opt); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (orig_freq == new_freq) {
            return waveform;
        }
        const int gcd = std::gcd(orig_freq, new_freq);
        const auto [kernels, width] = _sinc_resample_kernel(orig_freq, new_freq, gcd, opt, waveform.options());
        return _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernels, width);
    }

    // mu-law companding encode (mirrors torchaudio.functional.mu_law_encoding). Returns integer codes 0..mu.
    inline auto mu_law_encoding(tensor_t x, int quantization_channels) -> tensor_t {
        const double mu = quantization_channels - 1.0;
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat);
        }
        const auto x_mu = torch::sign(x) * torch::log1p(mu * torch::abs(x)) / std::log1p(mu);
        return ((x_mu + 1.0) / 2.0 * mu + 0.5).to(torch::kInt64);
    }

    // mu-law companding decode (mirrors torchaudio.functional.mu_law_decoding). Returns a float waveform in [-1, 1].
    inline auto mu_law_decoding(tensor_t x_mu, int quantization_channels) -> tensor_t {
        const double mu = quantization_channels - 1.0;
        if (!x_mu.is_floating_point()) {
            x_mu = x_mu.to(torch::kFloat);
        }
        const auto x = (x_mu / mu) * 2.0 - 1.0;
        return torch::sign(x) * (torch::exp(torch::abs(x) * std::log1p(mu)) - 1.0) / mu;
    }

    // Pre-emphasis FIR y[i] = x[i] - coeff*x[i-1] (mirrors torchaudio.functional.preemphasis).
    inline auto preemphasis(const_tensor_lref_t waveform, double coeff = 0.97) -> tensor_t {
        using namespace torch::indexing;
        auto out = waveform.clone();
        const auto rhs = coeff * waveform.index({Ellipsis, Slice(None, -1)}); // materialise from the original
        out.index({Ellipsis, Slice(1, None)}).sub_(rhs);
        return out;
    }

    // De-emphasis IIR y[i] = x[i] + coeff*y[i-1] via lfilter (mirrors torchaudio.functional.deemphasis).
    inline auto deemphasis(const_tensor_lref_t waveform, double coeff = 0.97) -> tensor_t {
        const auto opts = waveform.options();
        const auto a_coeffs = torch::tensor({1.0, -coeff}, opts);
        const auto b_coeffs = torch::tensor({1.0, 0.0}, opts);
        return lfilter(waveform, a_coeffs, b_coeffs);
    }

    // Delta (velocity) coefficients via a centred linear-regression conv (mirrors compute_deltas).
    inline auto compute_deltas(const_tensor_lref_t specgram, int win_length = 5, const std::string &mode = "replicate")
            -> tensor_t {
        namespace F = torch::nn::functional;
        if (win_length < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>("win_length must be >= 3.");
        }
        if (mode != "replicate" && mode != "constant" && mode != "reflect" && mode != "circular") {
            return handle_exceptions<tensor_t, std::invalid_argument>("Unrecognised padding mode " + mode);
        }
        const auto shape = specgram.sizes().vec();
        const int64_t time = shape.back();
        const auto packed = specgram.reshape({1, -1, time});
        const int64_t channels = packed.size(1);
        const int64_t n = (win_length - 1) / 2;
        const double denom = n * (n + 1) * (2 * n + 1) / 3.0;
        const auto kernel = torch::arange(-n, n + 1, packed.options()).reshape({1, 1, -1}).repeat({channels, 1, 1});
        tensor_t padded;
        if (mode == "constant") {
            padded = F::pad(packed, F::PadFuncOptions({n, n}).mode(torch::kConstant));
        } else if (mode == "reflect") {
            padded = F::pad(packed, F::PadFuncOptions({n, n}).mode(torch::kReflect));
        } else if (mode == "circular") {
            padded = F::pad(packed, F::PadFuncOptions({n, n}).mode(torch::kCircular));
        } else { // "replicate" (torchaudio default)
            padded = F::pad(packed, F::PadFuncOptions({n, n}).mode(torch::kReplicate));
        }
        const auto out = F::conv1d(padded, kernel, F::Conv1dFuncOptions().groups(channels)) / denom;
        return out.reshape(shape);
    }

    // Linear-spaced triangular filterbank (mirrors torchaudio.functional.linear_fbanks).
    inline auto linear_fbanks(int n_freqs, double f_min, double f_max, int n_filter, int sample_rate) -> tensor_t {
        const auto all_freqs = torch::linspace(0, sample_rate / 2, n_freqs);
        const auto f_pts = torch::linspace(f_min, f_max, n_filter + 2);
        return _create_triangular_filterbank(all_freqs, f_pts);
    }

    // Magnitude-weighted average frequency per frame (mirrors torchaudio.functional.spectral_centroid).
    inline auto spectral_centroid(const_tensor_lref_t waveform, int sample_rate, int pad, const_tensor_lref_t window,
                                  int n_fft, int hop_length, int win_length) -> tensor_t {
        auto opt = spectrogram_option()
                           .pad(pad)
                           .window(window)
                           .n_fft(n_fft)
                           .hop_length(hop_length)
                           .win_length(win_length)
                           .power(1.0)
                           .normalized(false)
                           .return_complex(false);
        const auto specgram = spectrogram(waveform, opt);
        if (!specgram.defined()) {
            return specgram; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: spectrogram rejected the waveform
        }
        // float32 like torchaudio's linspace(device=...), whatever the spectrogram's dtype.
        const auto freqs =
                torch::linspace(0, sample_rate / 2, 1 + n_fft / 2, tensor_options_t().device(specgram.device()))
                        .reshape({-1, 1});
        return (freqs * specgram).sum(-2) / specgram.sum(-2);
    }

    // FFT-based convolution along the last dim (mirrors torchaudio.functional.fftconvolve). Shares convolve()'s
    // operand check and crop helper, so it agrees with the time-domain convolve() to FP tolerance.
    inline auto fftconvolve(tensor_t x, tensor_t y, const convolve_mode mode = convolve_mode::full) -> tensor_t {
        if (const auto err = _align_convolve_operands(x, y); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        const int64_t n = x.size(-1) + y.size(-1) - 1;
        const auto fresult = torch::fft::rfft(x, n, -1) * torch::fft::rfft(y, n, -1);
        const auto result = torch::fft::irfft(fresult, n, -1);
        return _apply_convolve_mode(result, x.size(-1), y.size(-1), mode);
    }

    // Scale `noise` to a target per-signal SNR (dB) and add it (mirrors torchaudio.functional.add_noise).
    inline auto add_noise(const_tensor_lref_t waveform, const_tensor_lref_t noise, const_tensor_lref_t snr,
                          const c10::optional<tensor_t> &lengths = c10::nullopt) -> tensor_t {
        if (!(waveform.dim() >= 1 && waveform.dim() == noise.dim() && waveform.dim() - 1 == snr.dim() &&
              (!lengths.has_value() || lengths->dim() == snr.dim()))) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Input leading dimensions don't match.");
        }
        if (waveform.size(-1) != noise.size(-1)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Length dimensions of waveform and noise don't match.");
        }
        tensor_t ws = waveform, ns = noise;
        if (lengths.has_value()) {
            const int64_t length = waveform.size(-1);
            const auto mask = torch::arange(length, lengths->options().dtype(torch::kLong)) < lengths->unsqueeze(-1);
            ws = waveform * mask;
            ns = noise * mask;
        }
        // torch.linalg.vector_norm(...) ** 2 as in torchaudio (a plain sum of squares rounds differently).
        const auto energy_signal = torch::linalg_vector_norm(ws, 2, {-1}, false).pow(2); // (...,)
        const auto energy_noise = torch::linalg_vector_norm(ns, 2, {-1}, false).pow(2); // (...,)
        const auto original_snr_db = 10.0 * (torch::log10(energy_signal) - torch::log10(energy_noise));
        const auto scale = torch::pow(10.0, (original_snr_db - snr) / 20.0); // (...,)
        return waveform + scale.unsqueeze(-1) * noise;
    }

    // Speed change via resampling (mirrors torchaudio.functional.speed). Returns (waveform, out_lengths).
    inline auto speed(const_tensor_lref_t waveform, int orig_freq, double factor,
                      const c10::optional<tensor_t> &lengths = c10::nullopt)
            -> std::pair<tensor_t, c10::optional<tensor_t>> {
        int source_sample_rate = static_cast<int>(factor * orig_freq);
        int target_sample_rate = orig_freq;
        const int g = std::gcd(source_sample_rate, target_sample_rate);
        source_sample_rate /= g;
        target_sample_rate /= g;
        const auto out = resample(waveform, source_sample_rate, target_sample_rate);
        if (!lengths.has_value() || (!out.defined() && waveform.defined())) { // resample rejected it (no-exceptions)
            return {out, c10::nullopt};
        }
        return {out, torch::ceil(lengths.value() * target_sample_rate / static_cast<double>(source_sample_rate))
                             .to(lengths.value().dtype())};
    }

    // Levenshtein edit distance over two sequences (mirrors torchaudio.functional.edit_distance).
    // Non-tensor: works on strings, vectors, etc. Returns a plain integer distance.
    template<class Seq1, class Seq2>
    inline auto edit_distance(const Seq1 &seq1, const Seq2 &seq2) -> int64_t {
        const int64_t len2 = static_cast<int64_t>(std::size(seq2));
        std::vector<int64_t> dold(len2 + 1), dnew(len2 + 1);
        for (int64_t j = 0; j <= len2; ++j) {
            dold[j] = j;
        }
        int64_t i = 0;
        for (const auto &a: seq1) {
            ++i;
            dnew[0] = i;
            int64_t j = 0;
            for (const auto &b: seq2) {
                ++j;
                if (a == b) {
                    dnew[j] = dold[j - 1];
                } else {
                    dnew[j] = std::min({dold[j - 1], dnew[j - 1], dold[j]}) + 1;
                }
            }
            std::swap(dold, dnew);
        }
        return dold[len2];
    }

    // Fréchet distance between two multivariate Gaussians (mirrors torchaudio.functional.frechet_distance).
    // Intentionally stricter: torchaudio's `size(0) != size(1) != mu.size(0)` is a chained comparison, so it accepts
    // e.g. mu (4,) with sigma (3, 3); here any mismatch raises.
    inline auto frechet_distance(const_tensor_lref_t mu_x, const_tensor_lref_t sigma_x, const_tensor_lref_t mu_y,
                                 const_tensor_lref_t sigma_y) -> tensor_t {
        if (mu_x.dim() != 1 || sigma_x.dim() != 2 || sigma_x.size(0) != sigma_x.size(1) ||
            sigma_x.size(0) != mu_x.size(0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "mu_x must be 1-D and sigma_x a matching square matrix.");
        }
        if (mu_x.sizes() != mu_y.sizes() || sigma_x.sizes() != sigma_y.sizes()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("x and y means/covariances must share shapes.");
        }
        const auto a = (mu_x - mu_y).square().sum();
        const auto b = sigma_x.trace() + sigma_y.trace();
        const auto c = torch::real(torch::sqrt(torch::linalg::eigvals(torch::matmul(sigma_x, sigma_y)))).sum();
        return a + b - 2.0 * c;
    }

    // Effective mask width given the proportion cap p (mirrors torchaudio's _get_mask_param).
    inline auto _get_mask_param(int mask_param, double p, int64_t axis_length) -> int64_t {
        if (p == 1.0) {
            return mask_param;
        }
        return std::min<int64_t>(mask_param, static_cast<int64_t>(axis_length * p));
    }

    // SpecAugment: one random band mask shared across the batch (mirrors mask_along_axis).
    inline auto mask_along_axis(const_tensor_lref_t specgram, int mask_param, double mask_value, int axis,
                                double p = 1.0) -> tensor_t {
        const int64_t dim = specgram.dim();
        if (dim < 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Spectrogram must have at least two dimensions.");
        }
        if (axis != dim - 2 && axis != dim - 1) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Only Frequency and Time masking are supported.");
        }
        if (!(p >= 0.0 && p <= 1.0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>("p must be between 0.0 and 1.0.");
        }
        const int64_t mp = _get_mask_param(mask_param, p, specgram.size(axis));
        if (mp < 1) {
            return specgram;
        }
        const auto shape = specgram.sizes().vec();
        auto packed = specgram.reshape({-1, shape[shape.size() - 2], shape.back()});
        const int64_t packed_axis = axis - dim + 3; // 1 (freq) or 2 (time) after packing to 3D
        // torchaudio draws torch.rand(1): CPU float32 whatever the spectrogram's dtype/device (same masks per seed).
        const auto value = torch::rand({1}) * mp;
        const auto min_value = torch::rand({1}) * (packed.size(packed_axis) - value);
        const auto mask_start = min_value.to(torch::kLong).squeeze();
        const auto mask_end = (min_value.to(torch::kLong) + value.to(torch::kLong)).squeeze();
        auto mask = torch::arange(0, packed.size(packed_axis), specgram.options());
        auto maskb = (mask >= mask_start).logical_and(mask < mask_end);
        if (axis == dim - 2) {
            maskb = maskb.unsqueeze(-1);
        }
        packed = packed.masked_fill(maskb, mask_value);
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        return packed.reshape(out_shape);
    }

    // SpecAugment: an independent random band mask per batch example (mirrors mask_along_axis_iid).
    inline auto mask_along_axis_iid(const_tensor_lref_t specgrams, int mask_param, double mask_value, int axis,
                                    double p = 1.0) -> tensor_t {
        const int64_t dim = specgrams.dim();
        if (dim < 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Spectrogram must have at least three dimensions.");
        }
        if (axis != dim - 2 && axis != dim - 1) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Only Frequency and Time masking are supported.");
        }
        if (!(p >= 0.0 && p <= 1.0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>("p must be between 0.0 and 1.0.");
        }
        const int64_t mp = _get_mask_param(mask_param, p, specgrams.size(axis));
        if (mp < 1) {
            return specgrams;
        }
        const auto shape = specgrams.sizes().vec();
        const std::vector<int64_t> leading(shape.begin(), shape.end() - 2);
        const auto value = torch::rand(leading, specgrams.options()) * mp;
        const auto min_value = torch::rand(leading, specgrams.options()) * (specgrams.size(axis) - value);
        const auto mask_start = min_value.to(torch::kLong).unsqueeze(-1).unsqueeze(-1);
        const auto mask_end = (min_value.to(torch::kLong) + value.to(torch::kLong)).unsqueeze(-1).unsqueeze(-1);
        const auto mask = torch::arange(0, specgrams.size(axis), specgrams.options());
        auto out = specgrams.transpose(axis, -1);
        out = out.masked_fill((mask >= mask_start).logical_and(mask < mask_end), mask_value);
        return out.transpose(axis, -1);
    }

    // Phase vocoder: time-stretch a complex STFT by `rate` without changing pitch (mirrors phase_vocoder).
    inline auto phase_vocoder(const_tensor_lref_t complex_specgrams_in, double rate, const_tensor_lref_t phase_advance)
            -> tensor_t {
        using namespace torch::indexing;
        namespace F = torch::nn::functional;
        if (rate == 1.0) {
            return complex_specgrams_in;
        }
        if (complex_specgrams_in.dim() < 2) { // torchaudio raises (RuntimeError) on 0-D / 1-D input
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "phase_vocoder: expected a (..., freq, time) complex spectrogram.");
        }
        const double pi = std::acos(-1.0);
        const auto shape = complex_specgrams_in.sizes().vec();
        auto cs = complex_specgrams_in.reshape({-1, shape[shape.size() - 2], shape.back()});
        const auto real_dtype = torch::real(cs).scalar_type();
        const auto time_steps =
                torch::arange(0, cs.size(-1), rate, tensor_options_t().dtype(real_dtype).device(cs.device()));
        const auto alphas = time_steps - torch::floor(time_steps); // time_steps % 1.0 (time_steps >= 0)
        const auto phase_0 = cs.index({Ellipsis, Slice(None, 1)}).angle();
        cs = F::pad(cs, F::PadFuncOptions({0, 2}));
        const auto cs0 = cs.index_select(-1, time_steps.to(torch::kLong));
        const auto cs1 = cs.index_select(-1, (time_steps + 1).to(torch::kLong));
        const auto norm_0 = cs0.abs();
        const auto norm_1 = cs1.abs();
        auto phase = cs1.angle() - cs0.angle() - phase_advance;
        phase = phase - 2.0 * pi * torch::round(phase / (2.0 * pi));
        phase = phase + phase_advance;
        phase = torch::cat({phase_0, phase.index({Ellipsis, Slice(None, -1)})}, -1);
        const auto phase_acc = torch::cumsum(phase, -1);
        const auto mag = alphas * norm_1 + (1.0 - alphas) * norm_0;
        const auto stretch = torch::polar(mag, phase_acc);
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
        out_shape.push_back(stretch.size(1));
        out_shape.push_back(stretch.size(2));
        return stretch.reshape(out_shape);
    }

    // Inverse spectrogram via ISTFT (mirrors inverse_spectrogram). `normalized`: "none"/"window"/"frame_length", plus
    // "true"/"false" for torchaudio's bool True ("window") / False ("none").
    inline auto inverse_spectrogram(const_tensor_lref_t spectrogram_in, c10::optional<int64_t> length, int pad,
                                    const_tensor_lref_t window, int n_fft, int hop_length, int win_length,
                                    const std::string &normalized = "none", bool center = true,
                                    const std::string &pad_mode = "reflect", bool onesided = true) -> tensor_t {
        using namespace torch::indexing;
        (void) pad_mode; // accepted for spectrogram-parity; ISTFT does not use it
        const bool frame_length_norm = normalized == "frame_length";
        const bool window_norm = normalized == "window" || normalized == "true";
        if (!frame_length_norm && !window_norm && normalized != "none" && normalized != "false") {
            return handle_exceptions<tensor_t, std::invalid_argument>("Invalid normalized parameter: " + normalized);
        }
        if (spectrogram_in.dim() < 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "inverse_spectrogram: expected a (..., freq, time) spectrogram.");
        }
        if (!spectrogram_in.is_complex()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected spectrogram to be complex dtype.");
        }
        auto spec = spectrogram_in;
        if (window_norm) {
            spec = spec * window.pow(2.0).sum().sqrt();
        }
        const auto shape = spec.sizes().vec();
        const auto packed = spec.reshape({-1, shape[shape.size() - 2], shape.back()});
        const c10::optional<int64_t> istft_len =
                length.has_value() ? c10::optional<int64_t>(length.value() + 2 * pad) : c10::nullopt;
        auto waveform = torch::istft(packed, n_fft, hop_length, win_length, window, center, frame_length_norm, onesided,
                                     istft_len, false);
        if (length.has_value() && pad > 0) {
            waveform = waveform.index({Slice(), Slice(pad, -pad)});
        }
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
        out_shape.push_back(waveform.size(-1));
        return waveform.reshape(out_shape);
    }

    // pitch_shift helper: STFT -> phase_vocoder time-stretch -> ISTFT (mirrors _stretch_waveform).
    inline auto _stretch_waveform(const_tensor_lref_t waveform_in, double n_steps, int bins_per_octave = 12,
                                  int n_fft = 512, c10::optional<int> win_length_opt = c10::nullopt,
                                  c10::optional<int> hop_length_opt = c10::nullopt,
                                  const c10::optional<tensor_t> &window_opt = c10::nullopt) -> tensor_t {
        using namespace torch::indexing;
        if (waveform_in.dim() < 1) {
            return handle_exceptions<tensor_t, std::invalid_argument>("pitch_shift: expected a (..., time) waveform.");
        }
        const int hop_length = hop_length_opt.has_value() ? hop_length_opt.value() : n_fft / 4;
        const int win_length = win_length_opt.has_value() ? win_length_opt.value() : n_fft;
        const auto window = window_opt.has_value()
                                    ? window_opt.value()
                                    : torch::hann_window(win_length, tensor_options_t().device(waveform_in.device()));
        const auto shape = waveform_in.sizes().vec();
        const auto waveform = waveform_in.reshape({-1, shape.back()});
        const int64_t ori_len = shape.back();
        const double rate = std::pow(2.0, -n_steps / bins_per_octave);
        const auto spec_f =
                torch::stft(waveform, n_fft, hop_length, win_length, window, true, "reflect", false, true, true);
        const double pi = std::acos(-1.0);
        const auto phase_advance = torch::linspace(0, pi * hop_length, spec_f.size(-2),
                                                   tensor_options_t().dtype(torch::kFloat).device(waveform_in.device()))
                                           .unsqueeze(-1);
        const auto spec_stretch = phase_vocoder(spec_f, rate, phase_advance);
        // Python round(): halves go to even (nearbyint in the default rounding mode), e.g. 2001 / 2 -> 1000.
        const int64_t len_stretch = static_cast<int64_t>(std::nearbyint(static_cast<double>(ori_len) / rate));
        return torch::istft(spec_stretch, n_fft, hop_length, win_length, window, true, false, true,
                            c10::optional<int64_t>(len_stretch), false);
    }

    // pitch_shift helper: crop/pad the resampled waveform back to the original length (mirrors _fix_waveform_shape).
    inline auto _fix_waveform_shape(const_tensor_lref_t waveform_shift, const std::vector<int64_t> &shape) -> tensor_t {
        using namespace torch::indexing;
        namespace F = torch::nn::functional;
        const int64_t ori_len = shape.back();
        const int64_t shift_len = waveform_shift.size(-1);
        tensor_t ws = shift_len > ori_len ? waveform_shift.index({Ellipsis, Slice(None, ori_len)})
                                          : F::pad(waveform_shift, F::PadFuncOptions({0, ori_len - shift_len}));
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(ws.size(-1));
        return ws.reshape(out_shape);
    }

    // pitch_shift's resample source rate int(sample_rate / 2^(-n_steps / bins_per_octave)), checked before the int
    // cast (an out-of-range double -> int is UB). Shared with transform::PitchShift. Returns an error message or "".
    inline auto _pitch_shift_orig_freq(int sample_rate, double n_steps, int bins_per_octave, int &orig_freq) -> str_t {
        if (bins_per_octave == 0) { // torchaudio: ZeroDivisionError
            return "bins_per_octave must be non-zero.";
        }
        const double orig = sample_rate / std::pow(2.0, -n_steps / bins_per_octave);
        if (!(orig >= 1.0 && orig <= std::numeric_limits<int>::max())) {
            return "sample_rate * 2^(n_steps / bins_per_octave) must be in [1, INT_MAX].";
        }
        orig_freq = static_cast<int>(orig);
        return _check_resample_args(orig_freq, sample_rate, resample_option());
    }

    // Pitch shift by n_steps (fractional allowed, as in torchaudio) via time-stretch + resample (mirrors
    // torchaudio.functional.pitch_shift).
    inline auto pitch_shift(const_tensor_lref_t waveform, int sample_rate, double n_steps, int bins_per_octave = 12,
                            int n_fft = 512, c10::optional<int> win_length = c10::nullopt,
                            c10::optional<int> hop_length = c10::nullopt,
                            const c10::optional<tensor_t> &window = c10::nullopt) -> tensor_t {
        int orig_freq = 0;
        if (const auto err = _pitch_shift_orig_freq(sample_rate, n_steps, bins_per_octave, orig_freq); !err.empty()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(err);
        }
        if (waveform.dim() < 1) { // torchaudio: IndexError
            return handle_exceptions<tensor_t, std::invalid_argument>("pitch_shift: expected a (..., time) waveform.");
        }
        const auto waveform_stretch =
                _stretch_waveform(waveform, n_steps, bins_per_octave, n_fft, win_length, hop_length, window);
        const auto waveform_shift = resample(waveform_stretch, orig_freq, sample_rate);
        return _fix_waveform_shape(waveform_shift, waveform.sizes().vec());
    }

    // ITU-R BS.1770-4 K-weighted gated loudness in LKFS (mirrors torchaudio.functional.loudness).
    inline auto loudness(const_tensor_lref_t waveform_in, int sample_rate) -> tensor_t {
        using namespace torch::indexing;
        if (waveform_in.size(-2) > 5) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Only up to 5 channels are supported.");
        }
        const double gate_duration = 0.4, overlap = 0.75, gamma_abs = -70.0, kweight_bias = -0.691;
        // Python round(): halves go to even (11025 Hz: round(1102.5) = 1102).
        const auto gate_samples = static_cast<int64_t>(std::nearbyint(gate_duration * sample_rate));
        const auto step = static_cast<int64_t>(std::nearbyint(gate_samples * (1.0 - overlap)));

        // K-weighting: high-shelf (+4 dB @ 1500 Hz) then a 38 Hz high-pass.
        auto waveform = treble_biquad(waveform_in, sample_rate, 4.0, 1500.0, 1.0 / std::sqrt(2.0));
        waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5);

        auto energy = torch::square(waveform).unfold(-1, gate_samples, step); // (..., channels, n_blocks, gate)
        energy = torch::mean(energy, -1); // (..., channels, n_blocks)

        auto g = torch::tensor({1.0, 1.0, 1.0, 1.41, 1.41}, waveform.options());
        g = g.index({Slice(None, energy.size(-2))});

        auto energy_weighted = torch::sum(g.unsqueeze(-1) * energy, -2); // (..., n_blocks)
        const auto block_loudness = -0.691 + 10.0 * torch::log10(energy_weighted);

        // Absolute gating (> -70 LKFS).
        auto gated_blocks = (block_loudness > gamma_abs).unsqueeze(-2); // (..., 1, n_blocks)
        auto energy_filtered = torch::sum(gated_blocks * energy, -1) / gated_blocks.to(torch::kLong).sum(-1);
        energy_weighted = torch::sum(g * energy_filtered, -1);
        const auto gamma_rel = kweight_bias + 10.0 * torch::log10(energy_weighted) - 10.0;

        // Relative gating (> gamma_rel).
        gated_blocks = torch::logical_and(gated_blocks.squeeze(-2), block_loudness > gamma_rel.unsqueeze(-1));
        gated_blocks = gated_blocks.unsqueeze(-2);
        energy_filtered = torch::sum(gated_blocks * energy, -1) / gated_blocks.to(torch::kLong).sum(-1);
        energy_weighted = torch::sum(g * energy_filtered, -1);
        return kweight_bias + 10.0 * torch::log10(energy_weighted);
    }

    // Kaldi-style sliding-window cepstral mean (and optional variance) normalisation (mirrors
    // sliding_window_cmn). Inherently sequential per-frame with running-sum bookkeeping.
    inline auto sliding_window_cmn(const_tensor_lref_t specgram_in, int cmn_window = 600, int min_cmn_window = 100,
                                   bool center = false, bool norm_vars = false) -> tensor_t {
        using namespace torch::indexing;
        const auto input_shape = specgram_in.sizes().vec();
        const int64_t num_frames = input_shape[input_shape.size() - 2];
        const int64_t num_feats = input_shape.back();
        const auto specgram = specgram_in.reshape({-1, num_frames, num_feats});
        const int64_t num_channels = specgram.size(0);
        auto cur_sum = torch::zeros({num_channels, num_feats}, specgram.options());
        auto cur_sumsq = torch::zeros({num_channels, num_feats}, specgram.options());
        auto cmn = torch::zeros({num_channels, num_frames, num_feats}, specgram.options());
        int64_t last_window_start = -1, last_window_end = -1;
        for (int64_t t = 0; t < num_frames; ++t) {
            int64_t window_start = 0, window_end = 0;
            if (center) {
                window_start = t - cmn_window / 2;
                window_end = window_start + cmn_window;
            } else {
                window_start = t - cmn_window;
                window_end = t + 1;
            }
            if (window_start < 0) {
                window_end -= window_start;
                window_start = 0;
            }
            if (!center && window_end > t) {
                window_end = std::max<int64_t>(t + 1, min_cmn_window);
            }
            if (window_end > num_frames) {
                window_start -= window_end - num_frames;
                window_end = num_frames;
                if (window_start < 0) {
                    window_start = 0;
                }
            }
            if (last_window_start == -1) {
                // NOTE: replicate torchaudio's exact (quirky) slice end `window_end - window_start`.
                const auto input_part =
                        specgram.index({Slice(), Slice(window_start, window_end - window_start), Slice()});
                cur_sum += torch::sum(input_part, 1);
                if (norm_vars) {
                    cur_sumsq += torch::cumsum(input_part.pow(2), 1).index({Slice(), -1, Slice()});
                }
            } else {
                if (window_start > last_window_start) {
                    const auto ftr = specgram.index({Slice(), last_window_start, Slice()});
                    cur_sum -= ftr;
                    if (norm_vars) {
                        cur_sumsq -= ftr.pow(2);
                    }
                }
                if (window_end > last_window_end) {
                    const auto fta = specgram.index({Slice(), last_window_end, Slice()});
                    cur_sum += fta;
                    if (norm_vars) {
                        cur_sumsq += fta.pow(2);
                    }
                }
            }
            const int64_t window_frames = window_end - window_start;
            last_window_start = window_start;
            last_window_end = window_end;
            cmn.index_put_({Slice(), t, Slice()}, specgram.index({Slice(), t, Slice()}) - cur_sum / window_frames);
            if (norm_vars) {
                if (window_frames == 1) {
                    cmn.index_put_({Slice(), t, Slice()}, torch::zeros({num_channels, num_feats}, specgram.options()));
                } else {
                    auto variance = cur_sumsq / window_frames;
                    variance = variance - cur_sum.pow(2) / static_cast<double>(window_frames * window_frames);
                    variance = torch::pow(variance, -0.5);
                    cmn.index_put_({Slice(), t, Slice()}, cmn.index({Slice(), t, Slice()}) * variance);
                }
            }
        }
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 2);
        out_shape.push_back(num_frames);
        out_shape.push_back(num_feats);
        auto out = cmn.reshape(out_shape);
        if (input_shape.size() == 2) {
            out = out.squeeze(0);
        }
        return out;
    }

    // Normalised cross-correlation function for pitch detection (mirrors _compute_nccf).
    inline auto _compute_nccf(const_tensor_lref_t waveform_in, int sample_rate, double frame_time, int freq_low)
            -> tensor_t {
        using namespace torch::indexing;
        namespace F = torch::nn::functional;
        const double EPS = 1e-9;
        const int64_t lags = static_cast<int64_t>(std::ceil(static_cast<double>(sample_rate) / freq_low));
        const int64_t frame_size = static_cast<int64_t>(std::ceil(sample_rate * frame_time));
        const int64_t wlen = waveform_in.size(-1);
        const int64_t num_of_frames = static_cast<int64_t>(std::ceil(static_cast<double>(wlen) / frame_size));
        const int64_t p = lags + num_of_frames * frame_size - wlen;
        const auto waveform = F::pad(waveform_in, F::PadFuncOptions({0, p}));
        std::vector<tensor_t> output_lag;
        for (int64_t lag = 1; lag <= lags; ++lag) {
            const auto s1 = waveform.index({Ellipsis, Slice(None, -lag)})
                                    .unfold(-1, frame_size, frame_size)
                                    .index({Ellipsis, Slice(None, num_of_frames), Slice()});
            const auto s2 = waveform.index({Ellipsis, Slice(lag, None)})
                                    .unfold(-1, frame_size, frame_size)
                                    .index({Ellipsis, Slice(None, num_of_frames), Slice()});
            const auto of = (s1 * s2).sum(-1) / (EPS + torch::linalg_vector_norm(s1, 2, {-1}, false)).pow(2) /
                            (EPS + torch::linalg_vector_norm(s2, 2, {-1}, false)).pow(2);
            output_lag.push_back(of.unsqueeze(-1));
        }
        return torch::cat(output_lag, -1);
    }

    // Elementwise "prefer a if a > thresh*b" combiner (mirrors _combine_max).
    inline auto _combine_max(const std::pair<tensor_t, tensor_t> &a, const std::pair<tensor_t, tensor_t> &b,
                             double thresh = 0.99) -> std::pair<tensor_t, tensor_t> {
        const auto mask = a.first > thresh * b.first;
        const auto nmask = mask.logical_not();
        return {mask * a.first + nmask * b.first, mask * a.second + nmask * b.second};
    }

    // Best lag per frame with the half-lag preference + lag_min and +1 calibration (mirrors _find_max_per_frame).
    inline auto _find_max_per_frame(const_tensor_lref_t nccf, int sample_rate, int freq_high) -> tensor_t {
        using namespace torch::indexing;
        const int64_t lag_min = static_cast<int64_t>(std::ceil(static_cast<double>(sample_rate) / freq_high));
        const auto best = torch::max(nccf.index({Ellipsis, Slice(lag_min, None)}), -1);
        const int64_t half_size = nccf.size(-1) / 2;
        const auto half = torch::max(nccf.index({Ellipsis, Slice(lag_min, half_size)}), -1);
        auto combined = _combine_max({std::get<0>(half), std::get<1>(half)}, {std::get<0>(best), std::get<1>(best)});
        return combined.second + lag_min + 1;
    }

    // Centred median smoothing over a window (mirrors _median_smoothing).
    inline auto _median_smoothing(const_tensor_lref_t indices_in, int win_length) -> tensor_t {
        using namespace torch::indexing;
        namespace F = torch::nn::functional;
        const int64_t pad_length = (win_length - 1) / 2;
        auto indices = F::pad(indices_in, F::PadFuncOptions({pad_length, 0}).mode(torch::kConstant).value(0));
        if (pad_length > 0) {
            const auto fill = indices.index({Ellipsis, pad_length}).unsqueeze(-1);
            const auto rep = torch::cat(std::vector<tensor_t>(pad_length, fill), -1);
            indices.index_put_({Ellipsis, Slice(None, pad_length)}, rep);
        }
        const auto roll = indices.unfold(-1, win_length, 1);
        return std::get<0>(torch::median(roll, -1));
    }

    // Pitch detection via NCCF + median smoothing (mirrors detect_pitch_frequency). Extension: win_length 1 or 2
    // (no / minimal smoothing) works, where torchaudio's _median_smoothing fails on torch.cat([]).
    inline auto detect_pitch_frequency(const_tensor_lref_t waveform_in, int sample_rate, double frame_time = 1e-2,
                                       int win_length = 30, int freq_low = 85, int freq_high = 3400) -> tensor_t {
        if (waveform_in.dim() < 1 || sample_rate <= 0 || freq_low <= 0 || freq_high <= 0 || win_length < 1 ||
            !(frame_time > 0.0 && sample_rate * frame_time < 1e15)) { // torchaudio: ZeroDivisionError / IndexError
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "detect_pitch_frequency: need a (..., time) waveform, positive sample_rate, "
                    "frame_time, freq_low, freq_high and win_length >= 1.");
        }
        const auto shape = waveform_in.sizes().vec();
        const auto waveform = waveform_in.reshape({-1, shape.back()});
        const auto nccf = _compute_nccf(waveform, sample_rate, frame_time, freq_low);
        auto indices = _find_max_per_frame(nccf, sample_rate, freq_high);
        indices = _median_smoothing(indices, win_length);
        const double EPS = 1e-9;
        // Python `int / tensor` is tensor.reciprocal() * int (1 ulp off a correctly rounded division).
        const auto freq = torch::reciprocal(EPS + indices.to(torch::kFloat)) * sample_rate;
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(freq.size(-1));
        return freq.reshape(out_shape);
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_HPP
