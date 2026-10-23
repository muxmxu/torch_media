// LIB_TORCH_MEDIA_NO_EXCEPTIONS build: invalid input returns handle_exceptions' default value (torch::empty({1}))
// instead of throwing -- and instead of silently computing a result from the invalid input. One check per
// tensor-returning error site, so dropping any `return` in front of handle_exceptions fails here.
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

namespace fn = torchmedia::audio::functional;
namespace tf = torchmedia::audio::transform;

static auto is_default(const torch::Tensor &t) -> bool { return t.sizes() == torch::IntArrayRef({1}); }
// For composites: an exception escaping from an inner op (torch's c10::Error) counts as a failure, not an abort.
template<class Fn>
static auto returns_default(Fn &&f) -> bool {
    try {
        return is_default(f());
    } catch (const std::exception &) {
        return false;
    }
}

static void test_convolve_resample_deltas() {
    const auto x = torch::tensor({1.f, 2.f, 3.f});
    const auto y = torch::tensor({1.f, 1.f});
    const auto bad_mode = static_cast<fn::convolve_mode>(7);
    TM_CHECK(is_default(fn::convolve(x, y, bad_mode)));
    TM_CHECK(is_default(fn::fftconvolve(x, y, bad_mode)));
    TM_CHECK(is_default(fn::_apply_convolve_mode(torch::arange(6), 4, 3, bad_mode)));
    TM_CHECK(is_default(fn::convolve(torch::tensor(1.f), y, fn::convolve_mode::full))); // 0-D
    TM_CHECK(is_default(fn::convolve(x, torch::tensor(2.f), fn::convolve_mode::full)));
    TM_CHECK(is_default(fn::fftconvolve(torch::randn({2, 10}), torch::randn({3, 4}), fn::convolve_mode::full)));
    TM_CHECK(is_default(fn::resample(x, 0, 16000)));
    TM_CHECK(is_default(fn::resample(x.to(torch::kShort), 2, 1)));
    TM_CHECK(is_default(fn::resample(torch::tensor(1.f), 2, 1)));
    TM_CHECK(is_default(fn::compute_deltas(torch::randn({2, 6}), 2)));
    // composites validate up front: no zero-padded "result" from a failed inner resample (review R3).
    const auto w = torch::randn({1, 4000});
    TM_CHECK(returns_default([&] { return fn::pitch_shift(w, 0, 2); }));
    TM_CHECK(returns_default([&] { return fn::pitch_shift(w, 16000, 2, 0); }));
    TM_CHECK(returns_default([&] { return tf::Resample(tf::resample_option().orig_freq(0).new_freq(16000))(w); }));
}

static void test_spectral() {
    const auto w = torch::randn({1, 4000});
    TM_CHECK(is_default(fn::amplitude_to_DB(torch::rand({5}), fn::amplitude_to_db_option().set_top_db(80.0))));
    TM_CHECK(returns_default([&] { return fn::spectrogram(w, fn::spectrogram_option().normalize_method("foo")); }));
    TM_CHECK(is_default(fn::spectrogram(torch::tensor(1.f), fn::spectrogram_option())));
    TM_CHECK(is_default(fn::mel_filter_bank(4, 0.0, 0.0, 16000, 9, "foo", "htk")));
    TM_CHECK(is_default(fn::mel_filter_bank(4, 0.0, 0.0, 16000, 9, "", "foo")));
    TM_CHECK(is_default(fn::create_dct(4, 8, "foo")));
    fn::mel_spectrogram_option fmin;
    fmin.f_min = 9000.0;
    TM_CHECK(returns_default([&] { return fn::melspectrogram(w, fmin); }));
    fn::mel_spectrogram_option mnorm;
    mnorm.norm = "foo";
    TM_CHECK(returns_default([&] { return fn::melspectrogram(w, mnorm); }));
    fn::mel_spectrogram_option mscale;
    mscale.mel_scale = "foo";
    TM_CHECK(returns_default([&] { return fn::melspectrogram(w, mscale); }));
    fn::mfcc_option many;
    many.n_mfcc = 200; // > n_mels (128)
    TM_CHECK(returns_default([&] { return fn::mfcc(w, many); }));
    fn::mfcc_option dnorm;
    dnorm.norm = "foo";
    TM_CHECK(returns_default([&] { return fn::mfcc(w, dnorm); }));
    fn::mfcc_option mfcc_mel;
    mfcc_mel.mel.mel_scale = "foo";
    TM_CHECK(returns_default([&] { return fn::mfcc(w, mfcc_mel); }));
    fn::mfcc_option mfcc_fmin;
    mfcc_fmin.mel.f_min = 9000.0;
    TM_CHECK(returns_default([&] { return fn::mfcc(w, mfcc_fmin); }));

    const auto mag = torch::rand({9, 5});
    fn::griffinlim_option gm;
    gm.n_fft = 16;
    gm.momentum = 1.0;
    TM_CHECK(is_default(fn::griffinlim(mag, gm)));
    fn::griffinlim_option g16;
    g16.n_fft = 16;
    TM_CHECK(is_default(fn::griffinlim(mag[0], g16)));

    const auto win = torch::hann_window(16);
    const auto spec = torch::ones({9, 5}, torch::kComplexFloat);
    TM_CHECK(is_default(fn::inverse_spectrogram(spec, c10::nullopt, 0, win, 16, 4, 16, "foo")));
    TM_CHECK(is_default(fn::inverse_spectrogram(spec[0], c10::nullopt, 0, win, 16, 4, 16)));
    TM_CHECK(is_default(fn::inverse_spectrogram(mag, c10::nullopt, 0, win, 16, 4, 16)));
}

static void test_masking_metrics_loudness() {
    TM_CHECK(is_default(fn::frechet_distance(torch::zeros({3}), torch::eye(2), torch::zeros({3}), torch::eye(3))));
    TM_CHECK(is_default(fn::frechet_distance(torch::zeros({3}), torch::eye(3), torch::zeros({2}), torch::eye(2))));
    TM_CHECK(is_default(fn::mask_along_axis(torch::randn({6}), 2, 0.0, 0)));
    TM_CHECK(is_default(fn::mask_along_axis(torch::randn({4, 6}), 2, 0.0, 5)));
    TM_CHECK(is_default(fn::mask_along_axis(torch::randn({4, 6}), 2, 0.0, 1, 2.0)));
    TM_CHECK(is_default(fn::mask_along_axis_iid(torch::randn({4, 6}), 2, 0.0, 1)));
    TM_CHECK(is_default(fn::mask_along_axis_iid(torch::randn({2, 4, 6}), 2, 0.0, 0)));
    TM_CHECK(is_default(fn::mask_along_axis_iid(torch::randn({2, 4, 6}), 2, 0.0, 2, 2.0)));
    TM_CHECK(is_default(fn::loudness(torch::randn({6, 1000}), 16000)));
}

static void test_filtering_beamforming_alignment() {
    const auto w3 = torch::randn({1, 1, 8});
    TM_CHECK(is_default(fn::_lfilter_core(w3, torch::ones({1, 3}), torch::ones({1, 2}))));
    TM_CHECK(is_default(fn::_lfilter_core(torch::randn({8}), torch::ones({1, 3}), torch::ones({1, 3}))));
    TM_CHECK(is_default(fn::_lfilter_core(w3, torch::ones({1, 0}), torch::ones({1, 0}))));
    const auto wav = torch::randn({2, 8});
    TM_CHECK(is_default(fn::lfilter(wav, torch::tensor({1.f, 0.f}), torch::tensor({1.f, 0.f, 0.f}))));
    TM_CHECK(is_default(fn::lfilter(wav, torch::zeros({2, 2, 3}), torch::zeros({2, 2, 3}))));
    TM_CHECK(is_default(fn::lfilter(torch::randn({3, 8}), torch::ones({2, 3}), torch::ones({2, 3}))));
    TM_CHECK(is_default(fn::deemph_biquad(wav, 16000)));
    TM_CHECK(is_default(fn::riaa_biquad(wav, 16000)));
    TM_CHECK(is_default(fn::contrast(wav, 150.0)));
    TM_CHECK(is_default(fn::flanger(wav, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "bad", "linear")));
    TM_CHECK(is_default(fn::flanger(wav, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "sinusoidal", "cubic")));
    TM_CHECK(is_default(fn::flanger(torch::randn({5, 8}), 8000)));
    TM_CHECK(is_default(fn::vad(torch::randn({1, 8000}), 16000, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0,
                                std::nullopt, 0.4, 50.0, 6000.0, 2000.0, 150.0)));

    const auto c3 = torch::complex(torch::randn({3, 2, 2}), torch::randn({3, 2, 2}));
    const auto c32 = torch::complex(torch::randn({3, 2, 3}), torch::randn({3, 2, 3}));
    TM_CHECK(is_default(fn::rtf_evd(torch::real(c3))));
    TM_CHECK(is_default(fn::rtf_evd(c32)));
    TM_CHECK(is_default(fn::rtf_power(c3, c3, 0, 0)));
    TM_CHECK(is_default(fn::apply_beamforming(torch::real(c3), torch::real(c3))));

    const auto logp = torch::log_softmax(torch::randn({1, 6, 3}), -1);
    TM_CHECK(is_default(fn::forced_align(logp, torch::tensor({{0, 1}}, torch::kLong)).first));
    TM_CHECK(is_default(fn::forced_align(logp, torch::tensor({{1, 3}}, torch::kLong)).first));
}

int main() {
    TM_RUN(test_convolve_resample_deltas);
    TM_RUN(test_spectral);
    TM_RUN(test_masking_metrics_loudness);
    TM_RUN(test_filtering_beamforming_alignment);
    return tm_test::summary("audio_test_noexcept");
}
