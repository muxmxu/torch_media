#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_FILTERING_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_FILTERING_HPP
#include <cmath>
#include <limits>
#include <optional>
#include <torch/torch.h>
#include "../global_include.hpp"

namespace torchmedia::audio::functional {
    namespace detail {
        // float32/float64 input without autograd on another device (MPS, CUDA): the per-sample loops below run their
        // CPU scalar loop on a host copy (torchaudio's CPU values). Their tensor-op form costs 0.1-1 ms per sample on
        // MPS, and libtorch 2.5.1's MPS conv1d raises past 65536 samples (task51).
        inline auto loop_on_host(const_tensor_lref_t t) -> bool {
            return !t.is_cpu() && (t.scalar_type() == torch::kFloat || t.scalar_type() == torch::kDouble) &&
                   !t.requires_grad();
        }

        // shortcut: libtorch 2.5.1's MPS backend (the PyTorch wheel and torchaudio too) returns garbage once ~64k
        // index_put_ writes are queued without a sync; the per-sample tensor loops below flush every 8192 steps.
        // That fixes the forward pass only: a backward pass through these loops on MPS exhausts GPU memory beyond
        // about 20k samples (NaN / garbage, or an abort; torchaudio raises). Drop it once upstream MPS is fixed.
        inline void mps_flush(const_tensor_lref_t t, int64_t step) {
            if (t.is_mps() && step % 8192 == 8191) {
                torch::mps::synchronize();
            }
        }
    } // namespace detail

    // Core IIR difference-equation evaluator (mirrors torchaudio's compiled _lfilter op: FIR with b/a0 as a grouped
    // conv1d, then the recurrence with a/a0). `waveform` is (n_batch, n_channel, n_sample); coeffs are
    // (n_channel, n_order) ordered [a0, a1, ...] (lower delay first). The recurrence runs as torchaudio's scalar CPU
    // loop for float32/float64 CPU input without autograd; otherwise (other devices, autograd, other dtypes) as its
    // differentiable tensor-op loop (_lfilter_core_generic_loop), which costs ~5 us per sample.
    inline auto _lfilter_core(const_tensor_lref_t waveform, const_tensor_lref_t a_coeffs, const_tensor_lref_t b_coeffs)
            -> tensor_t {
        using namespace torch::indexing;
        namespace F = torch::nn::functional;

        if (a_coeffs.sizes() != b_coeffs.sizes()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected coeffs to be the same size.");
        }
        if (waveform.dim() != 3) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected waveform to be 3 dimensional.");
        }
        const int64_t n_channel = waveform.size(1);
        const int64_t n_sample = waveform.size(2);
        const int64_t n_order = a_coeffs.size(1);
        if (n_order <= 0) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected n_order to be positive.");
        }
        if (detail::loop_on_host(waveform) && !a_coeffs.requires_grad() && !b_coeffs.requires_grad()) {
            return _lfilter_core(waveform.cpu(), a_coeffs.cpu(), b_coeffs.cpu()).to(waveform.device());
        }

        // FIR part (b/a0) * x for all samples at once: a depthwise conv1d over the left-padded input.
        const auto a0 = a_coeffs.index({Slice(), Slice(None, 1)}); // (n_channel, 1)
        const auto input_signal_windows =
                F::conv1d(F::pad(waveform, F::PadFuncOptions({n_order - 1, 0})), (b_coeffs / a0).flip(1).unsqueeze(1),
                          F::Conv1dFuncOptions().groups(n_channel))
                        .contiguous(); // (n_batch, n_channel, n_sample)
        const auto a_flipped = (a_coeffs / a0).flip(1).contiguous(); // [.., a1, a0] / a0
        auto padded_output =
                torch::zeros({waveform.size(0), n_channel, n_sample + n_order - 1}, input_signal_windows.options());

        const auto dtype = padded_output.scalar_type();
        if (padded_output.is_cpu() && (dtype == torch::kFloat || dtype == torch::kDouble) &&
            !(input_signal_windows.requires_grad() || a_flipped.requires_grad())) {
            // torchaudio's host_lfilter_core_loop: y[i + n_order - 1] = x[i] - sum_k y[i + k] * a_flipped[k]. Its
            // `acc -= y * c` is compiled to a fused multiply-subtract in the arm64 reference wheel; an explicit fma
            // reproduces it bit-exactly and does not depend on this compiler's contraction setting.
            AT_DISPATCH_FLOATING_TYPES(dtype, "lfilter_core_loop", [&] {
                const scalar_t *in = input_signal_windows.const_data_ptr<scalar_t>();
                const scalar_t *coeff = a_flipped.const_data_ptr<scalar_t>();
                scalar_t *out = padded_output.mutable_data_ptr<scalar_t>();
                const int64_t n_out = padded_output.size(2);
                at::parallel_for(0, padded_output.size(0) * n_channel, 1, [&](int64_t begin, int64_t end) {
                    for (int64_t row = begin; row < end; ++row) {
                        const scalar_t *x = in + row * n_sample;
                        const scalar_t *c = coeff + (row % n_channel) * n_order;
                        scalar_t *y = out + row * n_out;
                        for (int64_t i = 0; i < n_sample; ++i) {
                            scalar_t acc = x[i];
                            for (int64_t k = 0; k < n_order; ++k) {
                                acc = std::fma(-y[i + k], c[k], acc);
                            }
                            y[i + n_order - 1] = acc;
                        }
                    }
                });
            });
        } else {
            const auto a_flipped_3 = a_flipped.unsqueeze(2); // (n_channel, n_order, 1)
            for (int64_t i = 0; i < n_sample; ++i) {
                const auto windowed = padded_output.index({Slice(), Slice(), Slice(i, i + n_order)}); // (nb, nc, no)
                const auto corr = torch::matmul(windowed.transpose(0, 1), a_flipped_3).squeeze(-1).t(); // (nb, nc)
                padded_output.index_put_({Slice(), Slice(), i + n_order - 1},
                                         input_signal_windows.index({Slice(), Slice(), i}) - corr);
                detail::mps_flush(padded_output, i);
            }
        }
        return padded_output.index({Slice(), Slice(), Slice(n_order - 1, None)});
    }

    // Perform an IIR filter by evaluating the difference equation (mirrors torchaudio.functional.lfilter).
    inline auto lfilter(tensor_t waveform, tensor_t a_coeffs, tensor_t b_coeffs, bool clamp = true,
                        bool batching = true) -> tensor_t {
        if (a_coeffs.sizes() != b_coeffs.sizes()) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected coeffs to be the same size.");
        }
        if (a_coeffs.dim() > 2) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Expected coeffs to have at most 2 dimensions.");
        }

        if (a_coeffs.dim() > 1) {
            if (batching) {
                if (waveform.dim() <= 0 || waveform.size(-2) != a_coeffs.size(0)) {
                    return handle_exceptions<tensor_t, std::invalid_argument>(
                            "Expected number of batches in waveform and coeffs to be the same.");
                }
            } else {
                waveform = torch::stack(std::vector<tensor_t>(a_coeffs.size(0), waveform), -2);
            }
        } else {
            a_coeffs = a_coeffs.unsqueeze(0);
            b_coeffs = b_coeffs.unsqueeze(0);
        }

        // pack batch -> (-1, num_filters, time)
        const auto shape = waveform.sizes().vec();
        const auto packed = waveform.reshape({-1, a_coeffs.size(0), shape.back()});
        auto output = _lfilter_core(packed, a_coeffs, b_coeffs);
        if (!output.defined()) {
            return output; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: _lfilter_core rejected empty coefficients
        }

        if (clamp) {
            output = torch::clamp(output, -1.0, 1.0);
        }

        // unpack batch -> original leading dims + (possibly new) time length
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(output.size(-1));
        return output.reshape(out_shape);
    }

    // Second-order (2-pole/2-zero) IIR filter with zero initial conditions
    // (mirrors torchaudio.functional.biquad). Coefficients are UNNORMALISED; lfilter divides by a0.
    inline auto biquad(const_tensor_lref_t waveform, double b0, double b1, double b2, double a0, double a1, double a2)
            -> tensor_t {
        const auto opts = waveform.options();
        const auto a_coeffs = torch::tensor({a0, a1, a2}, opts);
        const auto b_coeffs = torch::tensor({b0, b1, b2}, opts);
        return lfilter(waveform, a_coeffs, b_coeffs);
    }

    // biquad with 0-D (or 1-element) coefficient tensors, as torchaudio's designers pass them.
    inline auto biquad(const_tensor_lref_t waveform, const_tensor_lref_t b0, const_tensor_lref_t b1,
                       const_tensor_lref_t b2, const_tensor_lref_t a0, const_tensor_lref_t a1, const_tensor_lref_t a2)
            -> tensor_t {
        const auto o = waveform.options();
        return lfilter(waveform, torch::cat({a0.to(o).view(1), a1.to(o).view(1), a2.to(o).view(1)}),
                       torch::cat({b0.to(o).view(1), b1.to(o).view(1), b2.to(o).view(1)}));
    }

    // Zero-phase forward-backward IIR filtering (mirrors torchaudio.functional.filtfilt).
    inline auto filtfilt(const_tensor_lref_t waveform, const_tensor_lref_t a_coeffs, const_tensor_lref_t b_coeffs,
                         bool clamp = true) -> tensor_t {
        const auto forward_filtered = lfilter(waveform, a_coeffs, b_coeffs, /*clamp=*/false, /*batching=*/true);
        if (!forward_filtered.defined()) {
            return forward_filtered; // LIB_TORCH_MEDIA_NO_EXCEPTIONS: lfilter rejected the arguments
        }
        return lfilter(forward_filtered.flip(-1), a_coeffs, b_coeffs, /*clamp=*/clamp, /*batching=*/true).flip(-1);
    }

    // ---- biquad filter designers (Audio EQ Cookbook / SoX) — all delegate to biquad (task02) ----
    // Like torchaudio, the frequency / Q / gain designers compute their coefficients as 0-D tensors in the waveform's
    // dtype and device, op for op (float32 input: float32 coefficients, bit-exact with torchaudio; task50).
    namespace detail {
        inline constexpr double pi = 3.14159265358979323846;

        // torch.as_tensor(value, dtype=waveform.dtype, device=waveform.device); on the host when the filter loop runs
        // there (loop_on_host), so the result is torchaudio's CPU one (MPS sin / cos / exp round differently).
        inline auto as_coeff(double value, const_tensor_lref_t waveform) -> tensor_t {
            return torch::scalar_tensor(value, loop_on_host(waveform) ? waveform.options().device(torch::kCPU)
                                                                      : waveform.options());
        }
    } // namespace detail

    // Two-pole all-pass filter (EQ Cookbook APF).
    inline auto allpass_biquad(const_tensor_lref_t waveform, int sample_rate, double central_freq, double Q = 0.707)
            -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(central_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        return biquad(waveform, 1 - alpha, -2 * torch::cos(w0), 1 + alpha, 1 + alpha, -2 * torch::cos(w0), 1 - alpha);
    }

    // Biquad low-pass filter (EQ Cookbook LPF).
    inline auto lowpass_biquad(const_tensor_lref_t waveform, int sample_rate, double cutoff_freq, double Q = 0.707)
            -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(cutoff_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        const auto b0 = (1 - torch::cos(w0)) / 2;
        return biquad(waveform, b0, 1 - torch::cos(w0), b0, 1 + alpha, -2 * torch::cos(w0), 1 - alpha);
    }

    // Biquad high-pass filter (EQ Cookbook HPF).
    inline auto highpass_biquad(const_tensor_lref_t waveform, int sample_rate, double cutoff_freq, double Q = 0.707)
            -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(cutoff_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2.0 / detail::as_coeff(Q, waveform);
        const auto b0 = (1 + torch::cos(w0)) / 2;
        return biquad(waveform, b0, -1 - torch::cos(w0), b0, 1 + alpha, -2 * torch::cos(w0), 1 - alpha);
    }

    // Two-pole band-pass filter (EQ Cookbook BPF). const_skirt_gain=true -> constant skirt gain (peak gain Q).
    inline auto bandpass_biquad(const_tensor_lref_t waveform, int sample_rate, double central_freq, double Q = 0.707,
                                bool const_skirt_gain = false) -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(central_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        const auto temp = const_skirt_gain ? torch::sin(w0) / 2 : alpha;
        return biquad(waveform, temp, detail::as_coeff(0.0, waveform), -temp, 1 + alpha, -2 * torch::cos(w0),
                      1 - alpha);
    }

    // Two-pole band-reject (notch) filter (EQ Cookbook notch).
    inline auto bandreject_biquad(const_tensor_lref_t waveform, int sample_rate, double central_freq, double Q = 0.707)
            -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(central_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        const auto one = detail::as_coeff(1.0, waveform);
        return biquad(waveform, one, -2 * torch::cos(w0), one, 1 + alpha, -2 * torch::cos(w0), 1 - alpha);
    }

    // Peaking-EQ biquad (EQ Cookbook peakingEQ).
    inline auto equalizer_biquad(const_tensor_lref_t waveform, int sample_rate, double center_freq, double gain,
                                 double Q = 0.707) -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(center_freq, waveform) / sample_rate;
        const auto A = torch::exp(detail::as_coeff(gain, waveform) / 40.0 * std::log(10.0));
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        return biquad(waveform, 1 + alpha * A, -2 * torch::cos(w0), 1 - alpha * A, 1 + alpha / A, -2 * torch::cos(w0),
                      1 - alpha / A);
    }

    // Two-pole band filter mimicking SoX 'band' (exponential bandwidth, not the cookbook BPF).
    inline auto band_biquad(const_tensor_lref_t waveform, int sample_rate, double central_freq, double Q = 0.707,
                            bool noise = false) -> tensor_t {
        const auto f = detail::as_coeff(central_freq, waveform);
        const auto w0 = 2 * detail::pi * f / sample_rate;
        const auto bw_Hz = f / detail::as_coeff(Q, waveform);
        const auto a2 = torch::exp(-2 * detail::pi * bw_Hz / sample_rate);
        const auto a1 = -4 * a2 / (1 + a2) * torch::cos(w0);
        auto b0 = torch::sqrt(1 - a1 * a1 / (4 * a2)) * (1 - a2);
        if (noise) {
            const auto mult = torch::sqrt(((1 + a2) * (1 + a2) - a1 * a1) * (1 - a2) / (1 + a2)) / b0;
            b0 = mult * b0;
        }
        const auto zero = detail::as_coeff(0.0, waveform);
        return biquad(waveform, b0, zero, zero, detail::as_coeff(1.0, waveform), a1, a2);
    }

    // Low-shelf (bass tone-control) biquad (EQ Cookbook lowShelf).
    inline auto bass_biquad(const_tensor_lref_t waveform, int sample_rate, double gain, double central_freq = 100.0,
                            double Q = 0.707) -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(central_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        const auto A = torch::exp(detail::as_coeff(gain, waveform) / 40 * std::log(10.0));
        const auto temp1 = 2 * torch::sqrt(A) * alpha;
        const auto temp2 = (A - 1) * torch::cos(w0);
        const auto temp3 = (A + 1) * torch::cos(w0);
        const auto b0 = A * ((A + 1) - temp2 + temp1);
        const auto b1 = 2 * A * ((A - 1) - temp3);
        const auto b2 = A * ((A + 1) - temp2 - temp1);
        const auto a0 = (A + 1) + temp2 + temp1;
        const auto a1 = -2 * ((A - 1) + temp3);
        const auto a2 = (A + 1) + temp2 - temp1;
        return biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0);
    }

    // High-shelf (treble tone-control) biquad (EQ Cookbook highShelf).
    inline auto treble_biquad(const_tensor_lref_t waveform, int sample_rate, double gain, double central_freq = 3000.0,
                              double Q = 0.707) -> tensor_t {
        const auto w0 = 2 * detail::pi * detail::as_coeff(central_freq, waveform) / sample_rate;
        const auto alpha = torch::sin(w0) / 2 / detail::as_coeff(Q, waveform);
        const auto A = torch::exp(detail::as_coeff(gain, waveform) / 40 * std::log(10.0));
        const auto temp1 = 2 * torch::sqrt(A) * alpha;
        const auto temp2 = (A - 1) * torch::cos(w0);
        const auto temp3 = (A + 1) * torch::cos(w0);
        const auto b0 = A * ((A + 1) + temp2 + temp1);
        const auto b1 = -2 * A * ((A - 1) + temp3);
        const auto b2 = A * ((A + 1) + temp2 - temp1);
        const auto a0 = (A + 1) - temp2 + temp1;
        const auto a1 = 2 * ((A - 1) - temp3);
        const auto a2 = (A + 1) - temp2 - temp1;
        return biquad(waveform, b0, b1, b2, a0, a1, a2);
    }

    // ISO 908 CD de-emphasis high-shelf IIR (SoX). Only sample_rate 44100 or 48000.
    inline auto deemph_biquad(const_tensor_lref_t waveform, int sample_rate) -> tensor_t {
        double central_freq = 0.0, width_slope = 0.0, gain = 0.0;
        if (sample_rate == 44100) {
            central_freq = 5283.0;
            width_slope = 0.4845;
            gain = -9.477;
        } else if (sample_rate == 48000) {
            central_freq = 5356.0;
            width_slope = 0.479;
            gain = -9.62;
        } else {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Sample rate must be 44100 (audio-CD) or 48000 (DAT)");
        }
        const double w0 = 2.0 * detail::pi * central_freq / sample_rate;
        const double A = std::exp(gain / 40.0 * std::log(10.0));
        // torchaudio designs these two in Python double math: one product per statement, so that no a * b + c is
        // contracted into a fused multiply-add (clang's default -ffp-contract=on works within an expression).
        const double slope = (A + 1.0 / A) * (1.0 / width_slope - 1.0);
        const double alpha = std::sin(w0) / 2.0 * std::sqrt(slope + 2.0);
        const double temp1 = 2.0 * std::sqrt(A) * alpha;
        const double temp2 = (A - 1.0) * std::cos(w0);
        const double temp3 = (A + 1.0) * std::cos(w0);
        const double b0 = A * ((A + 1.0) + temp2 + temp1);
        const double b1 = -2.0 * A * ((A - 1.0) + temp3);
        const double b2 = A * ((A + 1.0) + temp2 - temp1);
        const double a0 = (A + 1.0) - temp2 + temp1;
        const double a1 = 2.0 * ((A - 1.0) - temp3);
        const double a2 = (A + 1.0) - temp2 - temp1;
        return biquad(waveform, b0, b1, b2, a0, a1, a2);
    }

    // RIAA vinyl playback EQ (SoX). Only sample_rate 44100 / 48000 / 88200 / 96000.
    inline auto riaa_biquad(const_tensor_lref_t waveform, int sample_rate) -> tensor_t {
        double zeros[2] = {0.0, 0.0}, poles[2] = {0.0, 0.0};
        if (sample_rate == 44100) {
            zeros[0] = -0.2014898;
            zeros[1] = 0.9233820;
            poles[0] = 0.7083149;
            poles[1] = 0.9924091;
        } else if (sample_rate == 48000) {
            zeros[0] = -0.1766069;
            zeros[1] = 0.9321590;
            poles[0] = 0.7396325;
            poles[1] = 0.9931330;
        } else if (sample_rate == 88200) {
            zeros[0] = -0.1168735;
            zeros[1] = 0.9648312;
            poles[0] = 0.8590646;
            poles[1] = 0.9964002;
        } else if (sample_rate == 96000) {
            zeros[0] = -0.1141486;
            zeros[1] = 0.9676817;
            poles[0] = 0.8699137;
            poles[1] = 0.9966946;
        } else {
            return handle_exceptions<tensor_t, std::invalid_argument>("Sample rate must be 44.1k, 48k, 88.2k, or 96k");
        }
        double b0 = 1.0, b1 = -(zeros[0] + zeros[1]), b2 = zeros[0] * zeros[1];
        const double a0 = 1.0, a1 = -(poles[0] + poles[1]), a2 = poles[0] * poles[1];
        // Normalise to 0 dB at 1 kHz.
        const double y = 2.0 * detail::pi * 1000.0 / sample_rate;
        // One product per statement (see deemph_biquad).
        const double c1 = std::cos(-y), c2 = std::cos(-2.0 * y), s1 = std::sin(-y), s2 = std::sin(-2.0 * y);
        const double b1c = b1 * c1, b2c = b2 * c2, a1c = a1 * c1, a2c = a2 * c2;
        const double b1s = b1 * s1, b2s = b2 * s2, a1s = a1 * s1, a2s = a2 * s2;
        const double b_re = b0 + b1c + b2c, a_re = a0 + a1c + a2c, b_im = b1s + b2s, a_im = a1s + a2s;
        const double b_re2 = b_re * b_re, b_im2 = b_im * b_im, a_re2 = a_re * a_re, a_im2 = a_im * a_im;
        const double g = 1.0 / std::sqrt((b_re2 + b_im2) / (a_re2 + a_im2));
        b0 *= g;
        b1 *= g;
        b2 *= g;
        return biquad(waveform, b0, b1, b2, a0, a1, a2);
    }

    // ---- simple SoX effects (no IIR state) ----

    // Contrast enhancement (SoX). enhancement_amount in [0, 100].
    inline auto contrast(const_tensor_lref_t waveform, double enhancement_amount = 75.0) -> tensor_t {
        if (!(enhancement_amount >= 0.0 && enhancement_amount <= 100.0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Allowed range of values for enhancement_amount : 0-100");
        }
        const double contrast = enhancement_amount / 750.0;
        const auto temp1 = waveform * (detail::pi / 2.0);
        const auto temp2 = contrast * torch::sin(temp1 * 4.0);
        return torch::sin(temp1 + temp2);
    }

    // DC shift, with an optional peak limiter (SoX). shift in [-2, 2]. Intentional difference: torchaudio writes the
    // result into its input tensor; here the input is left unchanged (same output).
    inline auto dcshift(const_tensor_lref_t waveform, double shift, std::optional<double> limiter_gain = std::nullopt)
            -> tensor_t {
        using namespace torch::indexing;
        if (limiter_gain.has_value()) {
            const double lg = limiter_gain.value();
            const double thr = 1.0 - (std::abs(shift) - lg);
            auto out = waveform.clone();
            if (shift > 0) {
                const auto mask = waveform > thr;
                const auto nmask = mask.logical_not();
                const auto temp = (waveform.index({mask}) - thr) * lg / (1.0 - thr);
                out.index_put_({mask}, torch::clamp_max(temp + thr + shift, thr));
                out.index_put_({nmask}, torch::clamp(waveform.index({nmask}) + shift, -1.0, 1.0));
                return out;
            }
            if (shift < 0) {
                const auto mask = waveform < -thr;
                const auto nmask = mask.logical_not();
                const auto temp = (waveform.index({mask}) + thr) * lg / (1.0 - thr);
                out.index_put_({mask}, torch::clamp_min(temp - thr + shift, -thr));
                out.index_put_({nmask}, torch::clamp(waveform.index({nmask}) + shift, -1.0, 1.0));
                return out;
            }
        }
        return torch::clamp(waveform + shift, -1.0, 1.0);
    }

    // Amplify / attenuate the whole waveform by gain_db decibels (SoX).
    inline auto gain(const_tensor_lref_t waveform, double gain_db = 1.0) -> tensor_t {
        if (gain_db == 0.0) {
            return waveform;
        }
        const double ratio = std::pow(10.0, gain_db / 20.0);
        return waveform * ratio;
    }

    // ---- dither (task18): probability-distribution dithering + optional noise shaping ----

    // Apply a probability distribution (TPDF/RPDF/GPDF) and requantise at 16-bit scale.
    // TPDF is deterministic (Bartlett window); RPDF/GPDF draw random samples (same randint stream as torchaudio).
    // Intentional GPDF difference: torchaudio sums its 7 samples into a VIEW of the first one, so it mutates the
    // caller's tensor and re-reads that grown sample when an index repeats (its noise shaping then sees the mutated
    // input too); here the 7 samples of the unmodified input are summed. Mono GPDF also works (torchaudio raises).
    inline auto _apply_probability_distribution(const_tensor_lref_t waveform_in,
                                                const std::string &density_function = "TPDF") -> tensor_t {
        using namespace torch::indexing;
        const auto shape = waveform_in.sizes().vec();
        const auto waveform = waveform_in.reshape({-1, shape.back()});
        const int64_t channel_size = waveform.size(0) - 1;
        const int64_t time_size = waveform.size(1) - 1;
        const int64_t random_channel = channel_size > 0 ? torch::randint(channel_size, {1}).item<int64_t>() : 0;
        const int64_t random_time = time_size > 0 ? torch::randint(time_size, {1}).item<int64_t>() : 0;

        const int number_of_bits = 16;
        const double up_scaling = std::pow(2.0, number_of_bits - 1) - 2.0; // 32766
        const auto signal_scaled = waveform * up_scaling;
        const double down_scaling = std::pow(2.0, number_of_bits - 1); // 32768

        tensor_t signal_scaled_dis;
        if (density_function == "RPDF") {
            const auto rpdf = waveform.index({random_channel, random_time}) - 0.5;
            signal_scaled_dis = signal_scaled + rpdf;
        } else if (density_function == "GPDF") {
            const int num_rand_variables = 6;
            auto gaussian = waveform.index({random_channel, random_time}).clone();
            for (int k = 0; k < num_rand_variables; ++k) {
                const int64_t rand_chan = channel_size > 0 ? torch::randint(channel_size, {1}).item<int64_t>() : 0;
                const int64_t rand_t = time_size > 0 ? torch::randint(time_size, {1}).item<int64_t>() : 0;
                gaussian = gaussian + waveform.index({rand_chan, rand_t});
            }
            signal_scaled_dis = signal_scaled + gaussian;
        } else { // TPDF (default)
            auto tpdf = torch::bartlett_window(time_size + 1, signal_scaled.options());
            tpdf = tpdf.repeat({channel_size + 1, 1});
            signal_scaled_dis = signal_scaled + tpdf;
        }
        const auto quantised = torch::round(signal_scaled_dis) / down_scaling;
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(quantised.size(-1));
        return quantised.reshape(out_shape);
    }

    // Error-feedback noise shaping: noise_shaped[n] = dithered[n] + error[n-1] (error = dithered - original).
    inline auto _add_noise_shaping(const_tensor_lref_t dithered_in, const_tensor_lref_t waveform_in) -> tensor_t {
        using namespace torch::indexing;
        const auto wf_shape = waveform_in.sizes().vec();
        const auto wf = waveform_in.reshape({-1, wf_shape.back()});
        const auto dshape = dithered_in.sizes().vec();
        const auto dithered = dithered_in.reshape({-1, dshape.back()});
        const auto error = dithered - wf;
        // offset the error by one sample along time (prepend 0, drop last) then add back.
        const auto shifted = torch::constant_pad_nd(error.index({Slice(), Slice(None, -1)}), {1, 0}, 0);
        const auto noise_shaped = dithered + shifted;
        std::vector<int64_t> out_shape(dshape.begin(), dshape.end() - 1);
        out_shape.push_back(noise_shaped.size(-1));
        return noise_shaped.reshape(out_shape);
    }

    // Dither (mirrors torchaudio.functional.dither).
    inline auto dither(const_tensor_lref_t waveform, const std::string &density_function = "TPDF",
                       bool noise_shaping = false) -> tensor_t {
        const auto dithered = _apply_probability_distribution(waveform, density_function);
        if (noise_shaping) {
            return _add_noise_shaping(dithered, waveform);
        }
        return dithered;
    }

    // ---- modulated-delay effects (task04): per-sample sequential loops ----

    inline auto _db2linear(double x) -> double { return std::exp(x * std::log(10.0) / 20.0); }

    // Wave-table generator for phaser/flanger LFOs (mirrors _generate_wave_table). The float64 math runs on the CPU
    // (MPS has no float64) and the int32 / float32 table is moved to `dev`'s device.
    inline auto _generate_wave_table(const std::string &wave_type, const std::string &data_type, int64_t table_size,
                                     double mn, double mx, double phase, const tensor_options_t &dev) -> tensor_t {
        using namespace torch::indexing;
        const int64_t phase_offset = static_cast<int64_t>(phase / detail::pi / 2.0 * table_size + 0.5);
        const auto t = torch::arange(table_size, torch::kInt32);
        const auto point = (t + phase_offset).remainder(table_size);
        auto d = torch::zeros_like(point, torch::kFloat64);
        if (wave_type == "SINE") {
            d = (torch::sin(point.to(torch::kFloat64) / table_size * 2.0 * detail::pi) + 1.0) / 2.0;
        } else { // TRIANGLE
            d = point.to(torch::kFloat64) * 2.0 / table_size;
            const auto value = torch::div(4 * point, table_size, "floor");
            d = torch::where(value == 0, d + 0.5, d);
            d = torch::where(value == 1, 1.5 - d, d);
            d = torch::where(value == 2, 1.5 - d, d);
            d = torch::where(value == 3, d - 1.5, d);
        }
        d = d * (mx - mn) + mn;
        if (data_type == "INT") {
            d = torch::where(d < 0, d - 0.5, d + 0.5).to(torch::kInt32);
        } else {
            d = d.to(torch::kFloat32);
        }
        return d.to(dev.device());
    }

    // Overdrive (SoX): waveshaper + a leaky-integrator recursion.
    inline auto overdrive(const_tensor_lref_t waveform_in, double gain = 20.0, double colour = 20.0) -> tensor_t {
        using namespace torch::indexing;
        if (detail::loop_on_host(waveform_in)) {
            return overdrive(waveform_in.cpu(), gain, colour).to(waveform_in.device());
        }
        const auto actual_shape = waveform_in.sizes().vec();
        const auto waveform = waveform_in.reshape({-1, actual_shape.back()});
        const double g = _db2linear(gain);
        const double col = colour / 200.0;
        auto temp = waveform * g + col;
        const auto mask1 = temp < -1;
        const auto mask2 = temp > 1;
        temp = torch::where(mask1, torch::full_like(temp, -2.0 / 3.0), temp);
        temp = torch::where(mask2, torch::full_like(temp, 2.0 / 3.0), temp);
        const auto mask3 = mask1.logical_not().logical_and(mask2.logical_not());
        temp = torch::where(mask3, temp - temp.pow(3) * (1.0 / 3.0), temp);
        auto output = torch::zeros_like(waveform);
        const int64_t T = waveform.size(-1);
        const auto dtype = waveform.scalar_type();
        if (waveform.is_cpu() && (dtype == torch::kFloat || dtype == torch::kDouble) && !temp.requires_grad()) {
            // torchaudio's compiled _overdrive_core_loop (overdrive.cpp), same double-literal arithmetic.
            const auto wav = waveform.contiguous();
            const auto tmp = temp.contiguous();
            AT_DISPATCH_FLOATING_TYPES(dtype, "overdrive_core_loop", [&] {
                const auto w_acc = wav.accessor<scalar_t, 2>();
                const auto t_acc = tmp.accessor<scalar_t, 2>();
                auto o_acc = output.accessor<scalar_t, 2>();
                at::parallel_for(0, wav.size(0), 1, [&](int64_t begin, int64_t end) {
                    for (int64_t c = begin; c < end; ++c) {
                        scalar_t l_in = 0, l_out = 0;
                        for (int64_t i = 0; i < T; ++i) {
                            l_out = t_acc[c][i] - l_in + 0.995 * l_out;
                            l_in = t_acc[c][i];
                            o_acc[c][i] = w_acc[c][i] * 0.5 + l_out * 0.75;
                        }
                    }
                });
            });
        } else { // other devices, autograd, other dtypes: torchaudio's _overdrive_core_loop_generic
            auto last_in = torch::zeros({waveform.size(0)}, waveform.options());
            auto last_out = torch::zeros({waveform.size(0)}, waveform.options());
            for (int64_t i = 0; i < T; ++i) {
                const auto ti = temp.index({Slice(), i});
                last_out = ti - last_in + 0.995 * last_out;
                last_in = ti;
                output.index_put_({Slice(), i}, waveform.index({Slice(), i}) * 0.5 + last_out * 0.75);
                detail::mps_flush(output, i);
            }
        }
        return torch::clamp(output, -1.0, 1.0).reshape(actual_shape);
    }

    // Phaser (SoX): modulated feedback delay line. float32/float64 CPU input without autograd runs torchaudio's
    // per-sample loop as a scalar loop per row (same float ops, bit-exact); anything else runs it as tensor ops.
    inline auto phaser(const_tensor_lref_t waveform_in, int sample_rate, double gain_in = 0.4, double gain_out = 0.74,
                       double delay_ms = 3.0, double decay = 0.4, double mod_speed = 0.5, bool sinusoidal = true)
            -> tensor_t {
        using namespace torch::indexing;
        const auto actual_shape = waveform_in.sizes().vec();
        auto waveform = waveform_in.reshape({-1, actual_shape.back()});
        const double delay_len = delay_ms * 0.001 * sample_rate + 0.5;
        const double mod_len = sample_rate / mod_speed + 0.5;
        // torchaudio fails here too (ZeroDivisionError / IndexError); unchecked, both lengths index empty buffers.
        if (!(delay_len >= 1.0 && delay_len < 1e15 && mod_len >= 1.0 && mod_len < 1e15)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "phaser: delay_ms and mod_speed must give delay and LFO buffers of >= 1 sample.");
        }
        if (detail::loop_on_host(waveform_in)) {
            return phaser(waveform_in.cpu(), sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)
                    .to(waveform_in.device());
        }
        const auto delay_buf_len = static_cast<int64_t>(delay_len);
        const auto mod_buf_len = static_cast<int64_t>(mod_len);
        const auto mod_buf = _generate_wave_table(sinusoidal ? "SINE" : "TRIANGLE", "INT", mod_buf_len, 1.0,
                                                  static_cast<double>(delay_buf_len), detail::pi / 2.0,
                                                  tensor_options_t()); // on the CPU: read element-wise below
        const int32_t *mod = mod_buf.const_data_ptr<int32_t>(); // values in [1, delay_buf_len]
        waveform = waveform * gain_in;
        const int64_t T = waveform.size(1);
        const auto dtype = waveform.scalar_type();
        tensor_t output;
        if (waveform.is_cpu() && (dtype == torch::kFloat || dtype == torch::kDouble) && !waveform.requires_grad()) {
            const auto in = waveform.contiguous();
            output = torch::empty_like(in);
            AT_DISPATCH_FLOATING_TYPES(dtype, "phaser_loop", [&] {
                const auto dec = static_cast<scalar_t>(decay);
                const scalar_t *src = in.const_data_ptr<scalar_t>();
                scalar_t *dst = output.mutable_data_ptr<scalar_t>();
                at::parallel_for(0, in.size(0), 1, [&](int64_t begin, int64_t end) {
                    std::vector<scalar_t> buf(delay_buf_len);
                    for (int64_t row = begin; row < end; ++row) {
                        std::fill(buf.begin(), buf.end(), scalar_t(0) * dec); // torchaudio: delay_buf * decay
                        const scalar_t *x = src + row * T;
                        scalar_t *y = dst + row * T;
                        int64_t delay_pos = 0, mod_pos = 0;
                        for (int64_t i = 0; i < T; ++i) {
                            const int64_t idx = (delay_pos + mod[mod_pos]) % delay_buf_len;
                            mod_pos = (mod_pos + 1) % mod_buf_len;
                            delay_pos = (delay_pos + 1) % delay_buf_len;
                            const scalar_t temp = x[i] + buf[idx];
                            buf[delay_pos] = temp * dec;
                            y[i] = temp;
                        }
                    }
                });
            });
        } else { // other devices, autograd, other dtypes: torchaudio's loop as tensor ops
            auto delay_buf = torch::zeros({waveform.size(0), delay_buf_len}, waveform.options()) * decay;
            int64_t delay_pos = 0, mod_pos = 0;
            std::vector<tensor_t> out_cols;
            out_cols.reserve(T);
            for (int64_t i = 0; i < T; ++i) {
                const int64_t idx = (delay_pos + mod[mod_pos]) % delay_buf_len;
                mod_pos = (mod_pos + 1) % mod_buf_len;
                delay_pos = (delay_pos + 1) % delay_buf_len;
                const auto temp = waveform.index({Slice(), i}) + delay_buf.index({Slice(), idx});
                delay_buf.index_put_({Slice(), delay_pos}, temp * decay);
                out_cols.push_back(temp);
                detail::mps_flush(delay_buf, i);
            }
            output = torch::stack(out_cols, 1);
        }
        return torch::clamp(output * gain_out, -1.0, 1.0).reshape(actual_shape);
    }

    // Flanger (SoX): modulated circular delay line with feedback and linear/quadratic interpolation.
    inline auto flanger(const_tensor_lref_t waveform_in, int sample_rate, double delay = 0.0, double depth = 2.0,
                        double regen = 0.0, double width = 71.0, double speed = 0.5, double phase = 25.0,
                        const std::string &modulation = "sinusoidal", const std::string &interpolation = "linear")
            -> tensor_t {
        using namespace torch::indexing;
        if (modulation != "sinusoidal" && modulation != "triangular") {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Only \"sinusoidal\" or \"triangular\" modulation.");
        }
        if (interpolation != "linear" && interpolation != "quadratic") {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "Only \"linear\" or \"quadratic\" interpolation.");
        }
        const auto actual_shape = waveform_in.sizes().vec();
        if (actual_shape.size() < 2) { // torchaudio: IndexError
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "flanger: expected a (..., channel, time) waveform.");
        }
        if (actual_shape[actual_shape.size() - 2] > 4) {
            return handle_exceptions<tensor_t, std::invalid_argument>("Max 4 channels allowed");
        }
        auto waveform = waveform_in.reshape({-1, actual_shape[actual_shape.size() - 2], actual_shape.back()});
        const double feedback_gain = regen / 100.0;
        double delay_gain = width / 100.0;
        const double channel_phase = phase / 100.0;
        const double delay_min = delay / 1000.0;
        const double delay_depth = depth / 1000.0;
        const int64_t n_channels = waveform.size(-2);
        const double in_gain = 1.0 / (1.0 + delay_gain);
        delay_gain = delay_gain / (1.0 + delay_gain);
        delay_gain = delay_gain * (1.0 - std::abs(feedback_gain));
        // torchaudio raises on an empty LFO (ZeroDivisionError) or a delay buffer < 1 sample (negative dimension);
        // the bounds also keep every int cast below defined.
        const double buf_len = (delay_min + delay_depth) * sample_rate + 0.5, lfo_len = sample_rate / speed;
        if (!(buf_len > -2.0 && buf_len < 1e15 && std::abs(delay_min * sample_rate) < 1e15 && lfo_len >= 1.0 &&
              lfo_len < 1e15 && std::abs(channel_phase * lfo_len * n_channels) < 1e15)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "flanger: delay, depth, speed and phase must give finite delay and LFO buffers "
                    "of >= 1 sample.");
        }
        if (detail::loop_on_host(waveform_in)) {
            return flanger(waveform_in.cpu(), sample_rate, delay, depth, regen, width, speed, phase, modulation,
                           interpolation)
                    .to(waveform_in.device());
        }
        const int64_t delay_buf_length = static_cast<int64_t>(buf_len) + 2;
        const auto lfo_length = static_cast<int64_t>(lfo_len);
        const double table_min = std::floor(delay_min * sample_rate + 0.5);
        const double table_max = delay_buf_length - 2.0;
        const auto lfo = _generate_wave_table(modulation == "sinusoidal" ? "SINE" : "TRIANGLE", "FLOAT", lfo_length,
                                              table_min, table_max, 3.0 * detail::pi / 2.0, waveform.options());
        const int64_t T = waveform.size(-1);
        const auto dtype = waveform.scalar_type();
        if (waveform.is_cpu() && (dtype == torch::kFloat || dtype == torch::kDouble) && !waveform.requires_grad()) {
            // torchaudio's per-sample loop as a scalar loop per (batch, channel) row: one rounding per torchaudio
            // tensor op (separate statements, so -ffp-contract=on fuses nothing); indices are Python-style remainders.
            const auto in = waveform.contiguous();
            auto output = torch::empty_like(in);
            const float *lf = lfo.const_data_ptr<float>();
            const bool linear = interpolation == "linear";
            const auto pymod = [](int64_t a, int64_t n) { return ((a % n) + n) % n; };
            AT_DISPATCH_FLOATING_TYPES(dtype, "flanger_loop", [&] {
                const auto fb = static_cast<scalar_t>(feedback_gain), ig = static_cast<scalar_t>(in_gain),
                           dg = static_cast<scalar_t>(delay_gain);
                const scalar_t *src = in.const_data_ptr<scalar_t>();
                scalar_t *dst = output.mutable_data_ptr<scalar_t>();
                at::parallel_for(0, in.size(0) * n_channels, 1, [&](int64_t begin, int64_t end) {
                    std::vector<scalar_t> buf(delay_buf_length);
                    for (int64_t row = begin; row < end; ++row) {
                        // (channel_idxs * lfo_length * channel_phase + 0.5).to(int64): float32 tensor arithmetic
                        const float ph =
                                static_cast<float>((row % n_channels) * lfo_length) * static_cast<float>(channel_phase);
                        const int64_t ch_phase = pymod(static_cast<int64_t>(ph + 0.5f), lfo_length);
                        std::fill(buf.begin(), buf.end(), scalar_t(0));
                        const scalar_t *x = src + row * T;
                        scalar_t *y = dst + row * T;
                        scalar_t last = 0;
                        int64_t pos = 0;
                        for (int64_t i = 0; i < T; ++i) {
                            pos = (pos + delay_buf_length - 1) % delay_buf_length;
                            const float d = lf[(i % lfo_length + ch_phase) % lfo_length];
                            const auto frac = static_cast<scalar_t>(d - std::trunc(d)); // torch.frac on float32
                            const auto k = static_cast<int64_t>(std::floor(d));
                            const scalar_t fed = last * fb;
                            buf[pos] = x[i] + fed;
                            const scalar_t d0 = buf[pymod(pos + k, delay_buf_length)];
                            const scalar_t d1 = buf[pymod(pos + k + 1, delay_buf_length)];
                            scalar_t delayed;
                            if (linear) {
                                const scalar_t step = (d1 - d0) * frac;
                                delayed = d0 + step;
                            } else {
                                const scalar_t e2 = buf[pymod(pos + k + 2, delay_buf_length)] - d0;
                                const scalar_t e1 = d1 - d0;
                                const scalar_t half_e2 = e2 * scalar_t(0.5);
                                const scalar_t two_e1 = e1 * scalar_t(2);
                                const scalar_t a = half_e2 - e1;
                                const scalar_t b = two_e1 - half_e2;
                                const scalar_t afrac = a * frac;
                                const scalar_t inner = afrac + b;
                                const scalar_t step = inner * frac;
                                delayed = d0 + step;
                            }
                            last = delayed;
                            const scalar_t dry = x[i] * ig;
                            const scalar_t wet = delayed * dg;
                            y[i] = dry + wet;
                        }
                    }
                });
            });
            return torch::clamp(output, -1.0, 1.0).reshape(actual_shape);
        }
        // Other devices, autograd, other dtypes: torchaudio's loop as tensor ops.
        auto delay_bufs = torch::zeros({waveform.size(0), n_channels, delay_buf_length}, waveform.options());
        auto delay_last = torch::zeros({waveform.size(0), n_channels}, waveform.options());
        auto output = torch::zeros_like(waveform);
        int64_t delay_buf_pos = 0, lfo_pos = 0;
        const auto channel_idxs =
                torch::arange(0, n_channels, tensor_options_t().dtype(torch::kLong).device(waveform.device()));
        const auto cur_channel_phase = (channel_idxs * lfo_length * channel_phase + 0.5).to(torch::kLong); // (C,)
        for (int64_t i = 0; i < T; ++i) {
            delay_buf_pos = (delay_buf_pos + delay_buf_length - 1) % delay_buf_length;
            const auto delay_tensor0 = lfo.index({(lfo_pos + cur_channel_phase).remainder(lfo_length)});
            const auto frac_delay = torch::frac(delay_tensor0);
            auto int_delay = torch::floor(delay_tensor0).to(torch::kLong); // (channels,)
            const auto temp = waveform.index({Slice(), Slice(), i}); // (batch, channels)
            delay_bufs.index_put_({Slice(), Slice(), delay_buf_pos}, temp + delay_last * feedback_gain);
            const auto delayed_0 =
                    delay_bufs.index({Slice(), channel_idxs, (delay_buf_pos + int_delay).remainder(delay_buf_length)});
            int_delay = int_delay + 1;
            const auto delayed_1 =
                    delay_bufs.index({Slice(), channel_idxs, (delay_buf_pos + int_delay).remainder(delay_buf_length)});
            int_delay = int_delay + 1;
            tensor_t delayed;
            if (interpolation == "linear") {
                delayed = delayed_0 + (delayed_1 - delayed_0) * frac_delay;
            } else {
                auto delayed_2 = delay_bufs.index(
                        {Slice(), channel_idxs, (delay_buf_pos + int_delay).remainder(delay_buf_length)});
                delayed_2 = delayed_2 - delayed_0;
                const auto d1 = delayed_1 - delayed_0;
                const auto a = delayed_2 * 0.5 - d1;
                const auto b = d1 * 2.0 - delayed_2 * 0.5;
                delayed = delayed_0 + (a * frac_delay + b) * frac_delay;
            }
            delay_last = delayed;
            output.index_put_({Slice(), Slice(), i}, temp * in_gain + delayed * delay_gain);
            lfo_pos = (lfo_pos + 1) % lfo_length;
            detail::mps_flush(output, i);
        }
        return torch::clamp(output, -1.0, 1.0).reshape(actual_shape);
    }

    // ---- VAD (task05): SoX cepstral-power voice-activity detector ----

    // Single-window cepstral-power measure (mirrors _measure). `spectrum`/`noise_spectrum` are views and
    // are updated in place across windows. Like torchaudio, the work buffers are float32 (the spectrum's dtype)
    // whatever the samples' dtype: a float64 window product is rounded to float32 on assignment.
    inline auto _measure(int64_t measure_len_ws, const tensor_t &samples, tensor_t spectrum, tensor_t noise_spectrum,
                         const tensor_t &spectrum_window, int64_t spectrum_start, int64_t spectrum_end,
                         const tensor_t &cepstrum_window, int64_t cepstrum_start, int64_t cepstrum_end,
                         double noise_reduction_amount, double measure_smooth_time_mult,
                         const tensor_t &noise_up_time_mult, const tensor_t &noise_down_time_mult, int64_t boot_count)
            -> double {
        using namespace torch::indexing;
        const int64_t dft_len_ws = spectrum.size(-1);
        auto dftBuf = torch::zeros({dft_len_ws}, spectrum.options());
        dftBuf.index_put_({Slice(None, measure_len_ws)},
                          (samples * spectrum_window.index({Slice(None, measure_len_ws)})).to(dftBuf.scalar_type()));
        const auto _dftBuf = torch::fft::rfft(dftBuf);
        const double mult =
                boot_count >= 0 ? static_cast<double>(boot_count) / (1.0 + boot_count) : measure_smooth_time_mult;
        auto _d = _dftBuf.index({Slice(spectrum_start, spectrum_end)}).abs();
        spectrum.index({Slice(spectrum_start, spectrum_end)}).mul_(mult).add_(_d * (1 - mult));
        _d = spectrum.index({Slice(spectrum_start, spectrum_end)}).pow(2);
        const auto _zeros = torch::zeros({spectrum_end - spectrum_start}, spectrum.options());
        tensor_t _mult = boot_count >= 0
                                 ? _zeros
                                 : torch::where(_d > noise_spectrum.index({Slice(spectrum_start, spectrum_end)}),
                                                noise_up_time_mult, noise_down_time_mult);
        noise_spectrum.index({Slice(spectrum_start, spectrum_end)}).mul_(_mult).add_(_d * (1 - _mult));
        _d = torch::sqrt(torch::max(
                _zeros, _d - noise_reduction_amount * noise_spectrum.index({Slice(spectrum_start, spectrum_end)})));
        auto _cepstrum_Buf = torch::zeros({dft_len_ws >> 1}, spectrum.options());
        _cepstrum_Buf.index_put_({Slice(spectrum_start, spectrum_end)}, _d * cepstrum_window);
        const auto _cep = torch::fft::rfft(_cepstrum_Buf);
        double result = torch::sum(_cep.index({Slice(cepstrum_start, cepstrum_end)}).abs().pow(2)).item<double>();
        result = result > 0 ? std::log(result / (cepstrum_end - cepstrum_start))
                            : -std::numeric_limits<double>::infinity();
        return std::max(0.0, 21.0 + result);
    }

    // Voice activity detector trimming silence from the front (mirrors torchaudio.functional.vad).
    inline auto vad(const_tensor_lref_t waveform_in, int sample_rate, double trigger_level = 7.0,
                    double trigger_time = 0.25, double search_time = 1.0, double allowed_gap = 0.25,
                    double pre_trigger_time = 0.0, double boot_time = 0.35, double noise_up_time = 0.1,
                    double noise_down_time = 0.01, double noise_reduction_amount = 1.35, double measure_freq = 20.0,
                    std::optional<double> measure_duration = std::nullopt, double measure_smooth_time = 0.4,
                    double hp_filter_freq = 50.0, double lp_filter_freq = 6000.0, double hp_lifter_freq = 150.0,
                    double lp_lifter_freq = 2000.0) -> tensor_t {
        using namespace torch::indexing;
        const double mdur = measure_duration.has_value() ? measure_duration.value() : 2.0 / measure_freq;
        // These sizes index the loop below: 0 measures or a 0-sample period would write out of bounds or never end
        // (torchaudio raises IndexError / ZeroDivisionError / "range() arg 3 must not be zero" there).
        const double period = sample_rate / measure_freq + 0.5, n_measures = std::ceil(search_time * measure_freq);
        if (waveform_in.dim() < 1 || !(period >= 1.0 && period < 1e15 && n_measures >= 1.0 && n_measures < 1e15) ||
            !(sample_rate * mdur + 0.5 < 1e15 && hp_lifter_freq > 0.0 && lp_lifter_freq > 0.0)) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "vad: need a (..., time) waveform, search_time > 0, 0 < measure_freq <= "
                    "2 * sample_rate, a finite measure_duration and positive lifter frequencies.");
        }
        const int64_t measure_len_ws = static_cast<int64_t>(sample_rate * mdur + 0.5);
        const int64_t measure_len_ns = measure_len_ws;
        int64_t dft_len_ws = 16;
        while (dft_len_ws < measure_len_ws) {
            dft_len_ws *= 2;
        }
        const auto measure_period_ns = static_cast<int64_t>(period);
        const auto measures_len = static_cast<int64_t>(n_measures);
        const int64_t search_pre_trigger_len_ns = measures_len * measure_period_ns;
        const int64_t gap_len = static_cast<int64_t>(allowed_gap * measure_freq + 0.5);
        const int64_t fixed_pre_trigger_len_ns = static_cast<int64_t>(pre_trigger_time * sample_rate + 0.5);
        const int64_t samplesLen_ns = fixed_pre_trigger_len_ns + search_pre_trigger_len_ns + measure_len_ns;

        // torchaudio keeps windows, spectra and the mults in float32 on the input's device, whatever its dtype.
        const auto fopts = waveform_in.options().dtype(torch::kFloat);
        auto spectrum_window =
                torch::full({measure_len_ws}, 2.0 / std::sqrt(static_cast<double>(measure_len_ws)), fopts);
        spectrum_window = spectrum_window * torch::hann_window(measure_len_ws, fopts);

        int64_t spectrum_start =
                std::max<int64_t>(static_cast<int64_t>(hp_filter_freq / sample_rate * dft_len_ws + 0.5), 1);
        int64_t spectrum_end = std::min<int64_t>(static_cast<int64_t>(lp_filter_freq / sample_rate * dft_len_ws + 0.5),
                                                 dft_len_ws / 2);

        auto cepstrum_window = torch::full({spectrum_end - spectrum_start},
                                           2.0 / std::sqrt(static_cast<double>(spectrum_end) - spectrum_start), fopts);
        cepstrum_window = cepstrum_window * torch::hann_window(spectrum_end - spectrum_start, fopts);

        int64_t cepstrum_start = static_cast<int64_t>(std::ceil(sample_rate * 0.5 / lp_lifter_freq));
        int64_t cepstrum_end =
                std::min<int64_t>(static_cast<int64_t>(std::floor(sample_rate * 0.5 / hp_lifter_freq)), dft_len_ws / 4);
        if (cepstrum_end <= cepstrum_start) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "cepstrum_start must be smaller than cepstrum_end.");
        }

        const auto noise_up_time_mult = torch::tensor(std::exp(-1.0 / (noise_up_time * measure_freq)), fopts);
        const auto noise_down_time_mult = torch::tensor(std::exp(-1.0 / (noise_down_time * measure_freq)), fopts);
        const double measure_smooth_time_mult = std::exp(-1.0 / (measure_smooth_time * measure_freq));
        const double trigger_meas_time_mult = std::exp(-1.0 / (trigger_time * measure_freq));
        const int64_t boot_count_max = static_cast<int64_t>(boot_time * measure_freq - 0.5);
        int64_t boot_count = 0, measures_index = 0, flushedLen_ns = 0;

        const auto shape = waveform_in.sizes().vec();
        const auto waveform = waveform_in.reshape({-1, shape.back()});
        const int64_t n_channels = waveform.size(0);
        const int64_t ilen = waveform.size(1);

        auto spectrum = torch::zeros({n_channels, dft_len_ws}, fopts);
        auto noise_spectrum = torch::zeros({n_channels, dft_len_ws}, fopts);
        // The per-measure bookkeeping is scalar: float32 like torchaudio's, but on the host (read via accessors).
        auto mean_meas = torch::zeros({n_channels}, torch::kFloat);
        auto measures = torch::zeros({n_channels, measures_len}, torch::kFloat);
        auto m_acc = measures.accessor<float, 2>();
        auto mm_acc = mean_meas.accessor<float, 1>();
        const auto trigger = static_cast<float>(trigger_level); // torchaudio compares float32 tensors with it

        bool has_triggered = false;
        int64_t num_measures_to_flush = 0;
        int64_t pos = 0;
        for (pos = measure_len_ns; pos < ilen; pos += measure_period_ns) {
            for (int64_t i = 0; i < n_channels; ++i) {
                const double meas =
                        _measure(measure_len_ws, waveform.index({i, Slice(pos - measure_len_ws, pos)}),
                                 spectrum.select(0, i), noise_spectrum.select(0, i), spectrum_window, spectrum_start,
                                 spectrum_end, cepstrum_window, cepstrum_start, cepstrum_end, noise_reduction_amount,
                                 measure_smooth_time_mult, noise_up_time_mult, noise_down_time_mult, boot_count);
                m_acc[i][measures_index] = static_cast<float>(meas);
                // float32 tensor arithmetic as in torchaudio (two statements: no fused multiply-add).
                const float decayed = mm_acc[i] * static_cast<float>(trigger_meas_time_mult);
                mm_acc[i] = decayed + static_cast<float>(meas * (1.0 - trigger_meas_time_mult));
                has_triggered = has_triggered || (mm_acc[i] >= trigger);
                if (has_triggered) {
                    const int64_t n = measures_len;
                    int64_t k = measures_index, jTrigger = n, jZero = n, j = 0;
                    for (j = 0; j < n; ++j) {
                        const float mk = m_acc[i][k];
                        if (mk >= trigger && j <= jTrigger + gap_len) {
                            jZero = jTrigger = j;
                        } else if (mk == 0 && jTrigger >= jZero) {
                            jZero = j;
                        }
                        k = (k + n - 1) % n;
                    }
                    const int64_t jj = std::min<int64_t>(n - 1, jZero);
                    num_measures_to_flush = std::min<int64_t>(std::max<int64_t>(num_measures_to_flush, jj), n);
                }
            }
            measures_index = (measures_index + 1) % measures_len;
            if (boot_count >= 0) {
                boot_count = (boot_count == boot_count_max) ? -1 : boot_count + 1;
            }
            if (has_triggered) {
                flushedLen_ns = (measures_len - num_measures_to_flush) * measure_period_ns;
                break;
            }
        }
        if (!has_triggered) {
            std::vector<int64_t> empty_shape(shape.begin(), shape.end() - 1);
            empty_shape.push_back(0);
            return waveform.index({Ellipsis, Slice(0, 0)}).reshape(empty_shape);
        }
        const auto res = waveform.index({Slice(), Slice(pos - samplesLen_ns + flushedLen_ns, None)});
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        out_shape.push_back(res.size(-1));
        return res.reshape(out_shape);
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_FILTERING_HPP
