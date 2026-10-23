// Assertion-based tests for the audio functional layer (target: 100% coverage of _functional*.hpp).
// Golden values verified against torchaudio 2.5.1 (see gen_golden.py). Bug regression tests are RED
// against the pre-fix _functional.hpp and GREEN after task02's fixes.
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <type_traits>
#include "test_util.hpp"

using namespace torchmedia::audio::functional;
static const double PI = std::acos(-1.0);

// ---------------- convolve ----------------
static void test_convolve_full_length() {
    auto x = torch::tensor({1.f, 2.f, 3.f, 4.f}).reshape({1, 4});
    auto y = torch::tensor({1.f, 1.f}).reshape({1, 2});
    TM_CHECK(convolve(x, y, convolve_mode::full).size(-1) == 5); // N + M - 1
}

// bug#4: 'same' returns the FIRST input's length; 'valid' = max-min+1 (x shorter than y)
static void test_convolve_modes() {
    auto x = torch::tensor({1.f, 2.f, 3.f}).reshape({1, 3});
    auto y = torch::tensor({1.f, 1.f, 1.f, 1.f, 1.f}).reshape({1, 5});
    TM_CHECK(convolve(x, y, convolve_mode::full).size(-1) == 7);
    TM_CHECK(convolve(x, y, convolve_mode::same).size(-1) == 3); // first input length
    TM_CHECK(convolve(x, y, convolve_mode::valid).size(-1) == 3); // 5 - 3 + 1
}

static void test_convolve_broadcast_and_errors() {
    // differing ndim -> dimension-alignment (unsqueeze) branches
    TM_CHECK(convolve(torch::randn({1, 10}), torch::randn({4}), convolve_mode::full).size(-1) == 13); // unsqueeze y
    TM_CHECK(convolve(torch::randn({5}), torch::randn({1, 3}), convolve_mode::full).size(-1) == 7); // unsqueeze x

    // multi-dim leading dims with broadcasting
    auto x = torch::randn({2, 1, 10});
    auto y = torch::randn({1, 3, 4});
    auto out = convolve(x, y, convolve_mode::full);
    TM_CHECK(out.size(0) == 2 && out.size(1) == 3);

    bool threw_0d = false;
    try {
        convolve(torch::tensor(1.0f), torch::tensor(1.0f), convolve_mode::full);
    } catch (const std::invalid_argument &) {
        threw_0d = true;
    }
    TM_CHECK(threw_0d);

    bool threw_0d_y = false; // x is >=1D, y is 0D -> exercises the y.dim()==0 side of the ||
    try {
        (void) convolve(torch::tensor({1.f, 2.f}).reshape({1, 2}), torch::tensor(1.0f), convolve_mode::full);
    } catch (const std::invalid_argument &) {
        threw_0d_y = true;
    }
    TM_CHECK(threw_0d_y);

    bool threw_mode = false;
    try {
        auto a = torch::tensor({1.f, 2.f}).reshape({1, 2});
        auto b = torch::tensor({1.f}).reshape({1, 1});
        (void) convolve(a, b, static_cast<convolve_mode>(99));
    } catch (const std::invalid_argument &) {
        threw_mode = true;
    }
    TM_CHECK(threw_mode);
}

// ---------------- db_to_amplitude (bug#3) ----------------
static void test_db_to_amplitude() {
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(20.0f), 1.0f, 1.0f).item<float>(), 100.0, 1e-2);
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(20.0f), 1.0f, 0.5f).item<float>(), 10.0, 1e-2);
    TM_CHECK_CLOSE(db_to_amplitude(torch::tensor(10.0f), 2.0f, 1.0f).item<float>(), 20.0, 1e-2);
}

// ---------------- amplitude_to_DB (bug#2) ----------------
static void test_amplitude_to_DB_power() {
    auto spec = torch::tensor({1.0f, 0.1f, 0.01f, 0.001f}).reshape({1, 4});
    auto opt = amplitude_to_db_option().set_db_multiplier(0.0f).set_top_db(80.0f);
    auto db = amplitude_to_DB(spec, opt);
    auto expected = torch::tensor({0.0f, -10.0f, -20.0f, -30.0f}).reshape({1, 4});
    TM_CHECK_TENSOR_CLOSE(db, expected, 1e-3, 1e-4);
}

static void test_amplitude_to_DB_magnitude_complex_flags() {
    // magnitude path (multiplier = 20): golden [0, -6.0206, -12.0412]
    auto mag = torch::tensor({1.0f, 0.5f, 0.25f}).reshape({1, 3});
    auto db = amplitude_to_DB(mag, amplitude_to_db_option().set_multiplier(20.0f).set_db_multiplier(0.0f));
    auto expected = torch::tensor({0.0f, -6.0206f, -12.0412f}).reshape({1, 3});
    TM_CHECK_TENSOR_CLOSE(db, expected, 1e-3, 1e-4);

    // complex input -> internal abs; apply_top_db=false path
    auto c = torch::complex(torch::tensor({3.0f, 0.0f}), torch::tensor({4.0f, 0.0f})).reshape({1, 2});
    auto dbc = amplitude_to_DB(
            c, amplitude_to_db_option().set_multiplier(20.0f).set_db_multiplier(0.0f).set_apply_top_db(false));
    TM_CHECK_CLOSE(dbc[0][0].item<float>(), 20.0f * std::log10(5.0f), 1e-2);

    // top_db clamp path: power [1, 1e-5] -> [0, -50] dB, clamp to max-30 = -30
    auto big = torch::tensor({1.0f, 1e-5f}).reshape({1, 2});
    auto db2 = amplitude_to_DB(
            big, amplitude_to_db_option().set_multiplier(10.0f).set_db_multiplier(0.0f).set_top_db(30.0f));
    TM_CHECK_CLOSE(db2[0][1].item<float>(), -30.0f, 1e-3);
}

// ---------------- spectrogram (bug#5/#6) ----------------
static void test_spectrogram_shape() {
    auto spec = spectrogram(torch::randn({1, 16000}), spectrogram_option().return_complex(true));
    TM_CHECK(spec.is_complex());
    TM_CHECK(spec.size(-2) == 201);
}

static void test_spectrogram_power_abs() {
    auto sig = torch::randn({1, 4000});
    auto win = torch::hann_window(400);
    auto c = spectrogram(sig, spectrogram_option().window(win).return_complex(true));
    auto p = spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false));
    TM_CHECK_TENSOR_CLOSE(p, c.abs().pow(2.0), 1e-3, 1e-4);
}

static void test_spectrogram_normalized() {
    auto sig = torch::randn({1, 4000});
    auto win = torch::hann_window(400);
    auto sp_n = spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false).normalized(true));
    auto sp_un = spectrogram(sig, spectrogram_option().window(win).power(2.0).return_complex(false).normalized(false));
    TM_CHECK_TENSOR_CLOSE(sp_n, sp_un / win.pow(2).sum(), 1e-3, 1e-4);
}

static void test_spectrogram_branches() {
    auto sig = torch::randn({1, 4000});
    // pad > 0 branch
    auto sp_pad = spectrogram(sig, spectrogram_option().pad(100).return_complex(true));
    TM_CHECK(sp_pad.size(-2) == 201);
    // default window (none set -> hann) branch
    auto sp_defwin = spectrogram(sig, spectrogram_option().power(2.0).return_complex(false));
    TM_CHECK(sp_defwin.size(-2) == 201);
    // frame_length normalization branch
    auto sp_fl = spectrogram(sig, spectrogram_option()
                                          .window(torch::hann_window(400))
                                          .power(2.0)
                                          .return_complex(false)
                                          .normalized(true)
                                          .normalize_method("frame_length"));
    TM_CHECK(sp_fl.size(-2) == 201);
    // an unrecognized method now throws: see test_spectrogram_frame_length_parity
}

// ---------------- mel (bug#1) ----------------
static void test_mel_filter_bank_slaney() {
    auto fb_sla = mel_filter_bank(64, 0.0, 8000.0, 16000, 201, "", "slaney");
    auto fb_htk = mel_filter_bank(64, 0.0, 8000.0, 16000, 201, "", "htk");
    TM_CHECK(!torch::allclose(fb_sla, fb_htk));
    TM_CHECK_CLOSE(fb_sla.sum().item<double>(), 194.677, 0.5); // torchaudio golden (norm=None)
}

static void test_mel_filter_bank_branches() {
    // f_max <= 0 -> default to Nyquist; norm="slaney" branch
    auto fb = mel_filter_bank(40, 0.0, 0.0, 16000, 201, "slaney", "htk");
    TM_CHECK(fb.size(0) == 40 && fb.size(1) == 201);
}

static void test_melspectrogram() {
    auto sig = torch::sin(2.0 * PI * 440.0 * torch::arange(16000, torch::kFloat32) / 16000.0).reshape({1, 16000});
    mel_spectrogram_option opt;
    opt.sample_rate = 16000;
    opt.n_fft = 512;
    opt.win_length = 512;
    opt.hop_length = 256;
    opt.n_mels = 64;
    opt.f_min = 0.0;
    opt.f_max = 8000.0;
    opt.power = 2.0;
    opt.mel_scale = "htk";
    opt.norm = "";
    auto mel = melspectrogram(sig, opt);
    TM_CHECK(mel.size(-2) == 64 && mel.size(-1) == 63); // golden shape (1,64,63)
    TM_CHECK_CLOSE(mel.sum().item<double>(), 1548267.875, 1548267.875 * 0.03); // golden sum, 3% tol
}

// ---------------- create_dct (Tier 1) ----------------
static void test_create_dct() {
    auto d_none = create_dct(4, 8, "");
    TM_CHECK(d_none.size(0) == 8 && d_none.size(1) == 4);
    TM_CHECK_CLOSE(d_none.sum().item<double>(), 16.0, 1e-3);
    TM_CHECK_CLOSE(d_none[0][0].item<double>(), 2.0, 1e-4);
    TM_CHECK_CLOSE(d_none[3][2].item<double>(), -1.847759, 1e-4);
    auto d_ortho = create_dct(4, 8, "ortho");
    TM_CHECK_CLOSE(d_ortho.sum().item<double>(), 2.828427, 1e-4);
    TM_CHECK_CLOSE(d_ortho[0][0].item<double>(), 0.353553, 1e-4);
}

// ---------------- mfcc (Tier 1) ----------------
static void test_mfcc() {
    auto sig = torch::sin(2.0 * PI * 440.0 * torch::arange(16000, torch::kFloat32) / 16000.0).reshape({1, 16000});
    mfcc_option opt;
    opt.sample_rate = 16000;
    opt.n_mfcc = 13;
    opt.norm = "ortho";
    opt.mel.sample_rate = 16000;
    opt.mel.n_fft = 512;
    opt.mel.win_length = 512;
    opt.mel.hop_length = 256;
    opt.mel.n_mels = 64;
    opt.mel.f_min = 0.0;
    opt.mel.f_max = 8000.0;
    opt.mel.power = 2.0;
    opt.mel.mel_scale = "htk";
    opt.mel.norm = "";
    auto m = mfcc(sig, opt);
    TM_CHECK(m.size(-2) == 13 && m.size(-1) == 63); // golden shape (1,13,63)
    TM_CHECK_CLOSE(m.sum().item<double>(), -15546.43, 50.0); // golden sum (tol for mel/dB impl diffs)
    TM_CHECK_CLOSE(m[0][0][0].item<double>(), 52.4487, 1.0); // golden [0,0,0]

    // log_mels=true branch
    mfcc_option lopt = opt;
    lopt.log_mels = true;
    auto ml = mfcc(sig, lopt);
    TM_CHECK(ml.size(-2) == 13 && ml.size(-1) == 63);
    TM_CHECK_CLOSE(ml.sum().item<double>(), -4499.624, 20.0); // log_mels golden
    TM_CHECK_CLOSE(ml[0][0][0].item<double>(), 12.0768, 0.5);
}

// ---------------- griffinlim (Tier 1) ----------------
static void test_griffinlim() {
    auto wav = torch::sin(2.0 * PI * 5.0 * torch::arange(2000, torch::kFloat32) / 2000.0).reshape({1, 2000});
    auto win = torch::hann_window(256);
    auto spec = spectrogram(
            wav, spectrogram_option().window(win).n_fft(256).win_length(256).hop_length(64).power(2.0).return_complex(
                         false));
    griffinlim_option g;
    g.n_fft = 256;
    g.hop_length = 64;
    g.win_length = 256;
    g.window = win;
    g.power = 2.0;
    g.n_iter = 32;
    g.momentum = 0.99;
    g.rand_init = false;
    auto rec = griffinlim(spec, g);
    TM_CHECK(rec.size(-1) == 1984); // golden shape
    TM_CHECK_CLOSE(rec.sum().item<double>(), 901.7086791992188, 1e-3); // golden sum (same ATen ops -> bit-exact)
    TM_CHECK_CLOSE(rec[0][1000].item<double>(), -0.035030048340559006, 1e-6); // golden sample
}

static void test_griffinlim_branches() {
    auto wav = torch::sin(2.0 * PI * 5.0 * torch::arange(1024, torch::kFloat32) / 1024.0).reshape({1, 1024});
    auto win = torch::hann_window(256);
    auto spec = spectrogram(
            wav, spectrogram_option().window(win).n_fft(256).win_length(256).hop_length(64).power(2.0).return_complex(
                         false));

    // default window (none set -> hann), momentum=0 branch, explicit length>0 branch
    griffinlim_option g1;
    g1.n_fft = 256;
    g1.hop_length = 64;
    g1.win_length = 256;
    g1.power = 2.0;
    g1.n_iter = 4;
    g1.momentum = 0.0;
    g1.length = 1024;
    auto r1 = griffinlim(spec, g1);
    TM_CHECK(r1.size(-1) == 1024); // explicit length

    // rand_init=true branch (random phase -> just check shape)
    griffinlim_option g2;
    g2.n_fft = 256;
    g2.hop_length = 64;
    g2.win_length = 256;
    g2.window = win;
    g2.power = 2.0;
    g2.n_iter = 4;
    g2.rand_init = true;
    auto r2 = griffinlim(spec, g2);
    TM_CHECK(r2.dim() == 2 && r2.size(0) == 1);
}

// ---------------- resample (Tier 1) ----------------
static void test_resample() {
    auto x = torch::sin(2.0 * PI * 3.0 * torch::arange(64, torch::kFloat32) / 64.0).reshape({1, 64});
    auto r1 = resample(x, 64, 32); // downsample 2x
    TM_CHECK(r1.size(-1) == 32);
    TM_CHECK_CLOSE(r1.sum().item<double>(), 0.049087, 1e-3);
    TM_CHECK_CLOSE(r1[0][1].item<double>(), 0.546815, 1e-3);
    TM_CHECK_CLOSE(r1[0][2].item<double>(), 0.927328, 1e-3);
    auto r2 = resample(x, 64, 48); // non-integer ratio (gcd reduce 4:3)
    TM_CHECK(r2.size(-1) == 48);
    TM_CHECK_CLOSE(r2.sum().item<double>(), 0.020851, 1e-3);
    TM_CHECK_CLOSE(r2[0][1].item<double>(), 0.378224, 1e-3);
}

// ---------------- option setters ----------------
static void test_option_setters() {
    auto so = spectrogram_option()
                      .pad(0)
                      .window(torch::hann_window(400))
                      .n_fft(400)
                      .hop_length(200)
                      .win_length(400)
                      .power(2.0)
                      .normalized(false)
                      .normalize_method("window")
                      .center(true)
                      .pad_mode("reflect")
                      .onesided(true)
                      .return_complex(false);
    TM_CHECK(so._n_fft == 400);
    auto ao = amplitude_to_db_option()
                      .set_multiplier(10.0f)
                      .set_amin(1e-10f)
                      .set_top_db(80.0f)
                      .set_db_multiplier(0.0f)
                      .set_apply_top_db(true);
    TM_CHECK(ao.multiplier == 10.0f);
}

// ---------------- lfilter / biquad / filtfilt (task01) ----------------
// Golden values from torchaudio 2.5.1 (see gen_golden.py). Fixed signal:
static torch::Tensor filt_wav() { return torch::tensor({0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f}); }

static void test_lfilter_identity_fir() {
    auto wav = filt_wav();
    auto a = torch::tensor({1.f, 0.f, 0.f}), b = torch::tensor({1.f, 0.f, 0.f});
    TM_CHECK_TENSOR_CLOSE(lfilter(wav, a, b), wav, 1e-6, 1e-6);
}

static void test_lfilter_one_pole() {
    // a=[1,-0.5], b=[1,0], unit impulse -> closed-form geometric 0.5^n.
    auto imp = torch::zeros({6});
    imp[0] = 1.0;
    auto out = lfilter(imp, torch::tensor({1.f, -0.5f}), torch::tensor({1.f, 0.f}));
    auto expect = torch::tensor({1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f});
    TM_CHECK_TENSOR_CLOSE(out, expect, 1e-6, 1e-6);
}

static void test_lfilter_golden() {
    auto wav = filt_wav();
    auto out = lfilter(wav, torch::tensor({1.f, -0.3f, 0.05f}), torch::tensor({0.5f, 0.1f, 0.f}));
    auto expect = torch::tensor(
            {0.05f, -0.075f, 0.10500001f, -0.13475001f, 0.16432498f, -0.193965f, 0.22359422f, -0.25322351f});
    TM_CHECK_TENSOR_CLOSE(out, expect, 1e-5, 1e-4);
}

static void test_lfilter_clamp() {
    auto wav = filt_wav();
    auto a = torch::tensor({1.f, 0.f, 0.f}), b = torch::tensor({2.f, 0.f, 0.f}); // gain 2 -> exceeds [-1,1]
    auto clamped = lfilter(wav, a, b, /*clamp=*/true);
    TM_CHECK(clamped.max().item<float>() <= 1.0f + 1e-6f && clamped.min().item<float>() >= -1.0f - 1e-6f);
    auto raw = lfilter(wav, a, b, /*clamp=*/false);
    TM_CHECK(raw.max().item<float>() > 1.0f); // 2*0.7 = 1.4
}

static void test_lfilter_batching() {
    auto wav = filt_wav();
    auto a2 = torch::tensor({{1.f, -0.3f, 0.05f}, {1.f, -0.5f, 0.1f}});
    auto b2 = torch::tensor({{0.5f, 0.1f, 0.f}, {0.2f, 0.2f, 0.2f}});
    auto wav2 = torch::stack({wav, wav * 0.5f});

    auto out = lfilter(wav2, a2, b2, /*clamp=*/true, /*batching=*/true);
    TM_CHECK(out.size(0) == 2 && out.size(1) == 8);
    auto row0 = torch::tensor(
            {0.05f, -0.075f, 0.10500001f, -0.13475001f, 0.16432498f, -0.193965f, 0.22359422f, -0.25322351f});
    auto row1 = torch::tensor({0.01f, -0.005f, 0.0165f, -0.02125f, 0.027725f, -0.03401251f, 0.04022124f, -0.04648814f});
    TM_CHECK_TENSOR_CLOSE(out[0], row0, 1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(out[1], row1, 1e-5, 1e-4);

    // batching=false: stacks the 1D waveform into (num_filters, time); row0 == single-filter lfilter golden.
    auto stacked = lfilter(wav, a2, b2, /*clamp=*/true, /*batching=*/false);
    TM_CHECK(stacked.size(0) == 2 && stacked.size(1) == 8);
    TM_CHECK_TENSOR_CLOSE(stacked[0], row0, 1e-5, 1e-4);

    // raises: coeff ndim > 2, and a/b size mismatch.
    bool threw_dim = false, threw_mismatch = false;
    try {
        lfilter(wav, torch::zeros({2, 2, 3}), torch::zeros({2, 2, 3}));
    } catch (const std::invalid_argument &) {
        threw_dim = true;
    }
    try {
        lfilter(wav, torch::tensor({1.f, 0.f}), torch::tensor({1.f, 0.f, 0.f}));
    } catch (const std::invalid_argument &) {
        threw_mismatch = true;
    }
    TM_CHECK(threw_dim);
    TM_CHECK(threw_mismatch);
}

static void test_biquad_golden() {
    auto wav = filt_wav();
    auto out = biquad(wav, 0.2, 0.2, 0.2, 1.0, -0.5, 0.1);
    auto expect =
            torch::tensor({0.02f, -0.01f, 0.033f, -0.04250001f, 0.05545f, -0.06802501f, 0.0804425f, -0.09297627f});
    TM_CHECK_TENSOR_CLOSE(out, expect, 1e-5, 1e-4);
}

static void test_filtfilt_golden() {
    auto wav = filt_wav();
    auto a = torch::tensor({1.f, -0.3f, 0.05f}), b = torch::tensor({0.5f, 0.1f, 0.f});
    auto out = filtfilt(wav, a, b);
    auto expect = torch::tensor({0.01097753f, -0.01735536f, 0.02631715f, -0.03498988f, 0.0442179f, -0.05374512f,
                                 0.04849123f, -0.12661175f});
    TM_CHECK_TENSOR_CLOSE(out, expect, 1e-5, 1e-4);
}

// ---------------- biquad designers (task02) ----------------
// Golden from torchaudio 2.5.1 on filt_wav(), sr=16000, f=2000, Q=0.707, gain=6 (see gen_golden.py).
static void test_biquad_designers_cookbook() {
    auto w = filt_wav();
    const int sr = 16000;
    const double f = 2000.0, Q = 0.707;
    TM_CHECK_TENSOR_CLOSE(allpass_biquad(w, sr, f, Q),
                          torch::tensor({0.0333266f, -0.1295103f, 0.2553281f, -0.3322601f, 0.4454035f, -0.5407003f,
                                         0.6407539f, -0.7422708f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(lowpass_biquad(w, sr, f, Q),
                          torch::tensor({0.0097626f, 0.0092038f, 0.0054235f, 0.0020457f, 0.0001212f, -0.0005675f,
                                         -0.0005754f, -0.0003534f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(highpass_biquad(w, sr, f, Q),
                          torch::tensor({0.0569007f, -0.173959f, 0.2722406f, -0.3681757f, 0.4725806f, -0.5697826f,
                                         0.6709524f, -0.7707819f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(bandpass_biquad(w, sr, f, Q, false),
                          torch::tensor({0.0333367f, -0.0352448f, 0.0223359f, -0.03387f, 0.0272982f, -0.0296499f,
                                         0.029623f, -0.0288646f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(bandpass_biquad(w, sr, f, Q, true),
                          torch::tensor({0.023569f, -0.0249181f, 0.0157915f, -0.0239461f, 0.0192998f, -0.0209625f,
                                         0.0209434f, -0.0204073f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(bandreject_biquad(w, sr, f, Q),
                          torch::tensor({0.0666633f, -0.1647552f, 0.2776641f, -0.36613f, 0.4727018f, -0.5703501f,
                                         0.6703769f, -0.7711354f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(equalizer_biquad(w, sr, f, 6.0, Q),
                          torch::tensor({0.1260223f, -0.2248656f, 0.3136591f, -0.4259155f, 0.5184607f, -0.6203997f,
                                         0.721931f, -0.8194066f}),
                          1e-5, 1e-4);
}

static void test_biquad_designers_sox() {
    auto w = filt_wav();
    const int sr = 16000;
    const double f = 2000.0, Q = 0.707;
    TM_CHECK_TENSOR_CLOSE(band_biquad(w, sr, f, Q, false),
                          torch::tensor({0.0531239f, -0.0690283f, 0.0935173f, -0.124247f, 0.1477777f, -0.1742974f,
                                         0.2010933f, -0.2267115f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(band_biquad(w, sr, f, Q, true),
                          torch::tensor({0.0802433f, -0.1042669f, 0.1412576f, -0.1876745f, 0.2232176f, -0.2632754f,
                                         0.3037504f, -0.3424466f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(bass_biquad(w, sr, 6.0, 100.0, Q),
                          torch::tensor({0.1009684f, -0.1999922f, 0.3009748f, -0.399987f, 0.5009787f, -0.5999843f,
                                         0.7009802f, -0.799984f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(
            treble_biquad(w, sr, 6.0, 3000.0, Q),
            torch::tensor({0.1529666f, -0.364006f, 0.5690225f, -0.7649547f, 0.9644334f, -1.0f, 1.0f, -1.0f}), 1e-5,
            1e-4);
}

static void test_biquad_designers_tables() {
    auto w = filt_wav();
    TM_CHECK_TENSOR_CLOSE(deemph_biquad(w, 44100),
                          torch::tensor({0.0460351f, -0.0719766f, 0.1103325f, -0.1409072f, 0.1764067f, -0.2087693f,
                                         0.2431382f, -0.2762204f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(deemph_biquad(w, 48000),
                          torch::tensor({0.0447693f, -0.0703595f, 0.1082149f, -0.1380917f, 0.1732204f, -0.2048642f,
                                         0.2388327f, -0.2712452f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(riaa_biquad(w, 44100),
                          torch::tensor({0.0238061f, -0.0243101f, 0.0432812f, -0.0472247f, 0.0637447f, -0.0694391f,
                                         0.0847044f, -0.0913019f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(riaa_biquad(w, 96000),
                          torch::tensor({0.0117148f, -0.0115616f, 0.0221101f, -0.0229727f, 0.0326363f, -0.03427f,
                                         0.0432616f, -0.0454811f}),
                          1e-5, 1e-4);
    // unsupported sample rates raise.
    bool d_threw = false, r_threw = false;
    try {
        deemph_biquad(w, 16000);
    } catch (const std::invalid_argument &) {
        d_threw = true;
    }
    try {
        riaa_biquad(w, 16000);
    } catch (const std::invalid_argument &) {
        r_threw = true;
    }
    TM_CHECK(d_threw);
    TM_CHECK(r_threw);
}

// ---------------- simple effects: contrast, dcshift, gain (task03) ----------------
static void test_effects_contrast_dcshift_gain() {
    auto w = filt_wav();
    TM_CHECK_TENSOR_CLOSE(contrast(w, 75.0),
                          torch::tensor({0.2141858f, -0.3979351f, 0.536551f, -0.6342956f, 0.7071068f, -0.7730906f,
                                         0.8438679f, -0.9174136f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(contrast(w, 0.0),
                          torch::tensor({0.1564345f, -0.309017f, 0.4539905f, -0.5877852f, 0.7071068f, -0.8090171f,
                                         0.8910065f, -0.9510565f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(gain(w, 6.0),
                          torch::tensor({0.1995262f, -0.3990525f, 0.5985787f, -0.7981049f, 0.9976311f, -1.1971574f,
                                         1.3966836f, -1.5962099f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(gain(w, 0.0), w, 1e-6, 1e-6); // gain_db==0 short-circuit
    TM_CHECK_TENSOR_CLOSE(dcshift(w, 0.3), torch::tensor({0.4f, 0.1f, 0.6f, -0.1f, 0.8f, -0.3f, 1.0f, -0.5f}), 1e-5,
                          1e-4);
    TM_CHECK_TENSOR_CLOSE(dcshift(w, -0.3), torch::tensor({-0.2f, -0.5f, 0.0f, -0.7f, 0.2f, -0.9f, 0.4f, -1.0f}), 1e-5,
                          1e-4);
    TM_CHECK_TENSOR_CLOSE(dcshift(w, 0.5, 0.1), torch::tensor({0.6f, 0.3f, 0.8f, 0.1f, 1.0f, -0.1f, 0.6f, -0.3f}), 1e-5,
                          1e-4);
    TM_CHECK_TENSOR_CLOSE(dcshift(w, -0.5, 0.1), torch::tensor({-0.4f, -0.7f, -0.2f, -0.9f, 0.0f, -1.0f, 0.2f, -0.6f}),
                          1e-5, 1e-4);
    bool threw = false;
    try {
        contrast(w, 150.0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- companding & emphasis (task06) ----------------
static void test_companding_emphasis() {
    auto w = filt_wav();
    auto enc = mu_law_encoding(w, 256);
    TM_CHECK(enc.dtype() == torch::kInt64);
    auto enc_exp = torch::tensor({203, 37, 228, 21, 239, 12, 247, 5}, torch::kLong);
    TM_CHECK(enc.equal(enc_exp));
    auto dec = mu_law_decoding(enc, 256);
    TM_CHECK_TENSOR_CLOSE(dec,
                          torch::tensor({0.1006746f, -0.1969128f, 0.306334f, -0.3988404f, 0.4966767f, -0.5917979f,
                                         0.7049939f, -0.8037952f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(preemphasis(w, 0.97),
                          torch::tensor({0.1f, -0.297f, 0.494f, -0.691f, 0.888f, -1.085f, 1.2820001f, -1.4790001f}),
                          1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(
            deemphasis(w, 0.97),
            torch::tensor({0.1f, -0.103f, 0.20009f, -0.2059127f, 0.3002647f, -0.3087433f, 0.400519f, -0.4114965f}),
            1e-5, 1e-4);
    // round-trip: deemphasis(preemphasis(x)) == x
    TM_CHECK_TENSOR_CLOSE(deemphasis(preemphasis(w, 0.97), 0.97), w, 1e-5, 1e-4);
}

// ---------------- feature ops: compute_deltas, linear_fbanks, spectral_centroid (task09) ----------------
static void test_feature_ops() {
    auto sg = torch::tensor({{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {6.f, 5.f, 4.f, 3.f, 2.f, 1.f}});
    TM_CHECK_TENSOR_CLOSE(
            compute_deltas(sg, 5),
            torch::tensor({{0.5f, 0.8f, 1.0f, 1.0f, 0.8f, 0.5f}, {-0.5f, -0.8f, -1.0f, -1.0f, -0.8f, -0.5f}}), 1e-5,
            1e-4);
    TM_CHECK_TENSOR_CLOSE(
            compute_deltas(sg, 3),
            torch::tensor({{0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f}, {-0.5f, -1.0f, -1.0f, -1.0f, -1.0f, -0.5f}}), 1e-5,
            1e-4);
    bool threw = false;
    try {
        compute_deltas(sg, 2);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);

    auto lfb = linear_fbanks(6, 0.0, 8000.0, 3, 16000);
    TM_CHECK(lfb.size(0) == 6 && lfb.size(1) == 3);
    auto lfb_exp = torch::tensor({{0.f, 0.f, 0.f},
                                  {0.8f, 0.f, 0.f},
                                  {0.4f, 0.6f, 0.f},
                                  {0.f, 0.6f, 0.4f},
                                  {0.f, 0.f, 0.8f},
                                  {0.f, 0.f, 0.f}});
    TM_CHECK_TENSOR_CLOSE(lfb, lfb_exp, 1e-5, 1e-4);

    torch::manual_seed(0);
    auto wav = torch::randn({1, 4000});
    auto sc = spectral_centroid(wav, 16000, 0, torch::hann_window(400), 400, 200, 400);
    TM_CHECK(sc.size(0) == 1 && sc.size(1) == 21);
    // single 1kHz tone -> centroid near 1000 Hz
    auto t = torch::arange(16000.0) / 16000.0;
    auto tone = torch::sin(2.0 * PI * 1000.0 * t).reshape({1, 16000});
    auto sct = spectral_centroid(tone, 16000, 0, torch::hann_window(400), 400, 200, 400);
    TM_CHECK_CLOSE(sct.mean().item<float>(), 1000.0f, 60.0f);
}

// ---------------- fftconvolve, add_noise, speed (task07) ----------------
static void test_fftconvolve_addnoise_speed() {
    auto x = torch::tensor({{1.f, 2.f, 3.f, 4.f}});
    auto y = torch::tensor({{1.f, 1.f, 1.f}});
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, convolve_mode::full), torch::tensor({{1.f, 3.f, 6.f, 9.f, 7.f, 4.f}}), 1e-4,
                          1e-4);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, convolve_mode::valid), torch::tensor({{6.f, 9.f}}), 1e-4, 1e-4);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, convolve_mode::same), torch::tensor({{3.f, 6.f, 9.f, 7.f}}), 1e-4, 1e-4);
    // fftconvolve agrees with the time-domain convolve()
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, convolve_mode::full), convolve(x, y, convolve_mode::full), 1e-4, 1e-4);

    auto wav = torch::tensor({{1.f, 2.f, 3.f, 4.f}, {0.5f, -0.5f, 0.5f, -0.5f}});
    auto noise = torch::tensor({{0.1f, 0.1f, 0.1f, 0.1f}, {0.2f, -0.2f, 0.2f, -0.2f}});
    auto snr = torch::tensor({10.f, 3.f});
    TM_CHECK_TENSOR_CLOSE(add_noise(wav, noise, snr),
                          torch::tensor({{1.866025f, 2.866025f, 3.866025f, 4.866025f},
                                         {0.853973f, -0.853973f, 0.853973f, -0.853973f}}),
                          1e-4, 1e-4);
    // empirical SNR of the result matches the requested snr (closed-form sanity)
    auto added = add_noise(wav, noise, snr);
    auto applied = added - wav; // == scale * noise
    auto e_sig = (wav * wav).sum(-1);
    auto e_app = (applied * applied).sum(-1);
    auto got_snr = 10.0 * (torch::log10(e_sig) - torch::log10(e_app));
    TM_CHECK_TENSOR_CLOSE(got_snr, snr, 1e-3, 1e-3);

    auto sp_wav = torch::sin(2.0 * PI * 2.0 * torch::arange(16.0) / 16.0).reshape({1, 16});
    auto [out, outlen] = speed(sp_wav, 16, 2.0, torch::tensor({16}));
    TM_CHECK(out.size(0) == 1 && out.size(1) == 8);
    TM_CHECK(outlen.has_value() && outlen.value().item<int64_t>() == 8);
    TM_CHECK_TENSOR_CLOSE(out, resample(sp_wav, 2, 1), 1e-5, 1e-4); // speed == resample(src,tgt)
}

// ---------------- metrics: edit_distance, frechet_distance (task13) ----------------
static void test_metrics() {
    TM_CHECK(edit_distance(std::string("kitten"), std::string("sitting")) == 3);
    TM_CHECK(edit_distance(std::string("abc"), std::string("abc")) == 0);
    TM_CHECK(edit_distance(std::string("abc"), std::string("abd")) == 1); // substitution
    TM_CHECK(edit_distance(std::string("abc"), std::string("ab")) == 1); // deletion
    TM_CHECK(edit_distance(std::vector<int>{1, 2, 3}, std::vector<int>{1, 3}) == 1);

    auto I = torch::eye(2);
    auto mux = torch::tensor({0.f, 0.f}), muy = torch::tensor({1.f, 1.f});
    TM_CHECK_CLOSE(frechet_distance(mux, I, muy, I).item<float>(), 2.0f, 1e-4); // shifted means
    TM_CHECK_CLOSE(frechet_distance(mux, I, mux, I).item<float>(), 0.0f, 1e-4); // identical -> 0
    auto sx = torch::tensor({{2.f, 0.f}, {0.f, 3.f}});
    TM_CHECK_CLOSE(frechet_distance(torch::tensor({1.f, 2.f}), sx, torch::tensor({0.f, 0.f}), I).item<float>(),
                   5.707471f, 1e-3);
    bool threw = false;
    try {
        frechet_distance(torch::randn({2, 2}), I, mux, I); // mu_x not 1-D
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- dither (task18) ----------------
static void test_dither() {
    auto w = torch::tensor({{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f}, {0.05f, -0.05f, 0.15f, -0.15f, 0.25f, -0.25f}});
    // TPDF is deterministic (Bartlett window) -> exact golden.
    auto tpdf = dither(w, "TPDF", false);
    auto tpdf_exp = torch::tensor({{0.10000610f, -0.19998169f, 0.29998779f, -0.39993286f, 0.5f, -0.59994507f},
                                   {0.04998779f, -0.04998779f, 0.15002441f, -0.14996338f, 0.25f, -0.24996948f}});
    TM_CHECK_TENSOR_CLOSE(tpdf, tpdf_exp, 1e-5, 1e-4);
    // TPDF + noise shaping -> exact golden.
    auto tpdf_ns = dither(w, "TPDF", true);
    auto ns_exp = torch::tensor({{0.10000610f, -0.19997558f, 0.30000609f, -0.39994508f, 0.50006711f, -0.59994507f},
                                 {0.04998779f, -0.05000000f, 0.15003662f, -0.14993897f, 0.25003663f, -0.24996948f}});
    TM_CHECK_TENSOR_CLOSE(tpdf_ns, ns_exp, 1e-5, 1e-4);
    // RPDF / GPDF are RNG-dependent: check shape/dtype/finite invariants.
    torch::manual_seed(0);
    for (auto df: {std::string("RPDF"), std::string("GPDF")}) {
        auto d = dither(w, df, false);
        TM_CHECK(d.sizes() == w.sizes() && d.dtype() == w.dtype());
        TM_CHECK(torch::isfinite(d).all().item<bool>());
    }
}

// ---------------- SpecAugment masks (task10) ----------------
static void test_specaugment() {
    // mask_along_axis: ONE shared band along time across all freq rows.
    torch::manual_seed(0);
    auto x = torch::ones({4, 20});
    auto m = mask_along_axis(x, 6, 0.0, /*axis=*/1);
    TM_CHECK(m.sizes() == x.sizes());
    auto colzero = (m == 0).all(0); // (20,) — column fully masked?
    auto nz = colzero.sum().item<int64_t>();
    TM_CHECK(nz < 6); // band width < mask_param
    TM_CHECK((m.ne(x)).sum().item<int64_t>() == nz * 4); // shared across the 4 freq rows

    // mask_along_axis_iid: independent band per batch example.
    torch::manual_seed(0);
    auto xb = torch::ones({3, 4, 20});
    auto mb = mask_along_axis_iid(xb, 6, 0.0, /*axis=*/2);
    TM_CHECK(mb.sizes() == xb.sizes());
    for (int b = 0; b < 3; ++b) {
        auto cz = (mb[b] == 0).all(0).sum().item<int64_t>();
        TM_CHECK(cz < 6);
    }

    // p < 1 clamps the effective mask_param to floor(axis_len * p).
    torch::manual_seed(1);
    auto mp = mask_along_axis(x, 100, 0.0, 1, /*p=*/0.1); // mp = min(100, floor(20*0.1)=2)
    TM_CHECK((mp == 0).all(0).sum().item<int64_t>() < 2);

    // mask_param resolving to < 1 returns the input unchanged.
    TM_CHECK(mask_along_axis(x, 0, 0.0, 1).equal(x));

    // raises
    bool e_dim = false, e_axis = false, e_p = false;
    try {
        mask_along_axis(torch::ones({5}), 3, 0.0, 0);
    } catch (const std::invalid_argument &) {
        e_dim = true;
    }
    try {
        mask_along_axis(x, 3, 0.0, 5);
    } catch (const std::invalid_argument &) {
        e_axis = true;
    }
    try {
        mask_along_axis(x, 3, 0.0, 1, 1.5);
    } catch (const std::invalid_argument &) {
        e_p = true;
    }
    TM_CHECK(e_dim && e_axis && e_p);
}

// ---------------- STFT domain: inverse_spectrogram, phase_vocoder, pitch_shift (task08) ----------------
static void test_stft_domain() {
    // phase_vocoder on a deterministic complex spec.
    auto A = torch::arange(1, 13, torch::kFloat).reshape({1, 3, 4});
    auto spec = torch::complex(A, A * 0.5);
    auto pa = torch::linspace(0, PI, 3).unsqueeze(-1);
    auto pv = phase_vocoder(spec, 1.5, pa);
    TM_CHECK(pv.size(0) == 1 && pv.size(1) == 3 && pv.size(2) == 3);
    auto pv_real_exp = torch::tensor({1.0f, 2.5f, 4.0f, 5.0f, 6.5f, 8.0f, 8.999999f, 10.5f, 12.0f}).reshape({1, 3, 3});
    auto pv_imag_exp = torch::tensor({0.5f, 1.25f, 2.0f, 2.5f, 3.25f, 4.0f, 4.5f, 5.25f, 6.0f}).reshape({1, 3, 3});
    TM_CHECK_TENSOR_CLOSE(torch::real(pv), pv_real_exp, 1e-4, 1e-4);
    TM_CHECK_TENSOR_CLOSE(torch::imag(pv), pv_imag_exp, 1e-4, 1e-4);
    TM_CHECK(phase_vocoder(spec, 1.0, pa).equal(spec)); // rate==1.0 short-circuit

    // inverse_spectrogram round-trip on a sine.
    auto t = torch::arange(2000.0) / 16000.0;
    auto sine = torch::sin(2.0 * PI * 440.0 * t).reshape({1, 2000});
    auto win = torch::hann_window(400);
    auto S = spectrogram(
            sine, spectrogram_option().window(win).n_fft(400).hop_length(200).win_length(400).return_complex(true));
    auto rec = inverse_spectrogram(S, 2000, 0, win, 400, 200, 400, "none");
    TM_CHECK(rec.size(0) == 1 && rec.size(1) == 2000);
    TM_CHECK_TENSOR_CLOSE(rec, sine, 1e-3, 1e-3);
    bool threw = false;
    try {
        inverse_spectrogram(sine, 2000, 0, win, 400, 200, 400, "none"); // real input -> raise
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);

    // pitch_shift on the sine.
    auto ps = pitch_shift(sine, 16000, 4, 12, 512);
    TM_CHECK(ps.size(0) == 1 && ps.size(1) == 2000);
    TM_CHECK_CLOSE(ps.sum().item<float>(), 8.6217f, 1e-2);
    TM_CHECK_CLOSE(ps[0][500].item<float>(), 0.369485f, 1e-3);
}

// ---------------- loudness (task12) ----------------
static void test_loudness() {
    auto t = torch::arange(16000.0) / 16000.0;
    auto tone = torch::sin(2.0 * PI * 1000.0 * t);
    auto stereo = torch::stack({tone, 0.5 * tone}); // (2, 16000)
    TM_CHECK_CLOSE(loudness(stereo, 16000).item<float>(), -2.28488f, 1e-2);
    auto mono = tone.reshape({1, 16000});
    TM_CHECK_CLOSE(loudness(mono, 16000).item<float>(), -3.29946f, 1e-2);
    bool threw = false;
    try {
        loudness(torch::randn({6, 1000}), 16000); // > 5 channels -> raise
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- beamforming (task14) ----------------
static void test_beamforming() {
    auto A = torch::arange(1, 25, torch::kFloat).reshape({2, 3, 4});
    auto spec = torch::complex(A, A * 0.1f); // (channel=2, freq=3, time=4)

    auto P = psd(spec); // (freq=3, ch=2, ch=2)
    TM_CHECK(P.size(0) == 3 && P.size(1) == 2 && P.size(2) == 2);
    TM_CHECK_TENSOR_CLOSE(torch::real(P).flatten(),
                          torch::tensor({30.3f, 151.5f, 151.5f, 854.46002f, 175.74001f, 490.86002f, 490.86002f,
                                         1387.73999f, 450.45999f, 959.5f, 959.5f, 2050.30005f}),
                          1e-2, 1e-4);

    auto Ms = torch::complex(torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}),
                             torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}) * 0.2f);
    auto ps = torch::matmul(Ms, Ms.conj().transpose(-1, -2)) + 2.0f * torch::eye(2);
    auto Mn = torch::complex(torch::arange(13, 25, torch::kFloat).reshape({3, 2, 2}) * 0.1f,
                             torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}) * 0.3f);
    auto pn = torch::matmul(Mn, Mn.conj().transpose(-1, -2)) + 2.0f * torch::eye(2);

    auto ws = mvdr_weights_souden(ps, pn, 0);
    TM_CHECK_TENSOR_CLOSE(torch::real(ws).flatten(),
                          torch::tensor({0.05083f, 0.27646f, 0.13236f, 0.52895f, 0.2109f, 0.55445f}), 1e-3, 1e-3);
    TM_CHECK_TENSOR_CLOSE(torch::imag(ws).flatten(),
                          torch::tensor({0.14018f, -0.08823f, 0.27408f, -0.20779f, 0.3033f, -0.25334f}), 1e-3, 1e-3);
    auto onehot = torch::complex(torch::tensor({1.0f, 0.0f}), torch::zeros({2}));
    TM_CHECK_TENSOR_CLOSE(torch::real(mvdr_weights_souden(ps, pn, onehot)), torch::real(ws), 1e-4, 1e-4);

    auto rtf = rtf_evd(ps);
    TM_CHECK(rtf.size(0) == 3 && rtf.size(1) == 2);
    auto outer = torch::einsum("...c,...d->...cd", {rtf, rtf.conj()});
    TM_CHECK_TENSOR_CLOSE(torch::real(outer).flatten(),
                          torch::tensor({0.16366f, 0.36997f, 0.36997f, 0.83634f, 0.35054f, 0.47714f, 0.47714f, 0.64946f,
                                         0.40583f, 0.49105f, 0.49105f, 0.59417f}),
                          1e-3, 1e-3);

    auto wr = mvdr_weights_rtf(rtf, pn, 0);
    TM_CHECK_TENSOR_CLOSE(torch::real(wr).flatten(),
                          torch::tensor({-0.1576f, 0.51209f, 0.05181f, 0.6966f, 0.16169f, 0.69281f}), 1e-3, 1e-3);
    TM_CHECK_TENSOR_CLOSE(torch::imag(wr).flatten(),
                          torch::tensor({0.19362f, -0.08565f, 0.32209f, -0.23663f, 0.34456f, -0.28476f}), 1e-3, 1e-3);

    auto rp = rtf_power(ps, pn, 0, 3);
    TM_CHECK_TENSOR_CLOSE(torch::real(rp).flatten(),
                          torch::tensor({50.81053f, 134.70436f, 2443.69824f, 3379.28418f, 11043.48828f, 13425.94629f}),
                          1e-1, 1e-3);

    auto enh = apply_beamforming(ws, spec); // (freq=3, time=4)
    TM_CHECK(enh.size(0) == 3 && enh.size(1) == 4);
    TM_CHECK_TENSOR_CLOSE(torch::real(enh).flatten(),
                          torch::tensor({3.54415f, 3.87663f, 4.20912f, 4.5416f, 9.43777f, 10.10571f, 10.77365f,
                                         11.44159f, 13.28244f, 14.05278f, 14.82312f, 15.59346f}),
                          1e-2, 1e-3);

    bool threw = false;
    try {
        apply_beamforming(A, A); // real input -> raise
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- forced alignment (task15) ----------------
static void test_forced_align() {
    auto logits =
            torch::tensor({2.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 2.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 2.f, 0.f, 0.f})
                    .reshape({1, 6, 3});
    auto logp = torch::log_softmax(logits, -1);
    auto targets = torch::tensor({{1, 2}}, torch::kLong);
    auto [paths, scores] = forced_align(logp, targets, c10::nullopt, c10::nullopt, 0);
    TM_CHECK(paths.equal(torch::tensor({{0, 1, 1, 0, 2, 0}}, torch::kLong)));
    TM_CHECK_TENSOR_CLOSE(scores.flatten(),
                          torch::tensor({-0.23954f, -0.09492f, -0.23954f, -0.23954f, -0.09492f, -0.23954f}), 1e-4,
                          1e-4);

    auto spans = merge_tokens(paths[0], scores[0], 0);
    TM_CHECK(spans.size() == 2);
    TM_CHECK(spans[0].token == 1 && spans[0].start == 1 && spans[0].end == 3 && spans[0].length() == 2);
    TM_CHECK_CLOSE(spans[0].score, -0.16723, 1e-4);
    TM_CHECK(spans[1].token == 2 && spans[1].start == 4 && spans[1].end == 5);
    TM_CHECK_CLOSE(spans[1].score, -0.09492, 1e-4);

    // targets containing the blank index raises.
    bool threw = false;
    try {
        forced_align(logp, torch::tensor({{0, 1}}, torch::kLong), c10::nullopt, c10::nullopt, 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- modulated-delay effects: overdrive, phaser, flanger (task04) ----------------
static void test_modulated_effects() {
    auto w = filt_wav().reshape({1, 8});
    TM_CHECK_TENSOR_CLOSE(
            overdrive(w, 20.0, 20.0),
            torch::tensor({{0.55f, -0.6025f, 0.650013f, -0.702488f, 0.750025f, -0.802475f, 0.850037f, -0.902463f}}),
            1e-4, 1e-4);
    TM_CHECK_TENSOR_CLOSE(
            phaser(w, 8000, 0.4, 0.74, 3.0, 0.4, 0.5, true),
            torch::tensor({{0.0296f, -0.04736f, 0.069856f, -0.090458f, 0.111817f, -0.132873f, 0.154051f, -0.17518f}}),
            1e-4, 1e-4);
    auto wf = torch::tensor({{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f},
                             {0.2f, -0.1f, 0.4f, -0.3f, 0.6f, -0.5f, 0.8f, -0.7f}});
    TM_CHECK_TENSOR_CLOSE(flanger(wf, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "sinusoidal", "linear"),
                          torch::tensor({{0.1f, -0.2f, 0.3f, -0.399998f, 0.499996f, -0.599993f, 0.699988f, -0.799981f},
                                         {0.116959f, -0.05848f, 0.233918f, -0.175439f, 0.350877f, -0.292398f, 0.467836f,
                                          -0.409357f}}),
                          1e-4, 1e-4);
    // quadratic interpolation path runs and is shape-correct.
    auto fq = flanger(wf, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "sinusoidal", "quadratic");
    TM_CHECK(fq.sizes() == wf.sizes());
    bool threw = false;
    try {
        flanger(wf, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "bad", "linear");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

// ---------------- sequential feature ops: sliding_window_cmn, detect_pitch_frequency (task11) ----------------
static void test_sequential_feature_ops() {
    auto sg = torch::tensor({{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}, {7.f, 8.f}, {9.f, 10.f}});
    TM_CHECK_TENSOR_CLOSE(sliding_window_cmn(sg, 3, 2, false, false),
                          torch::tensor({{-1.f, -1.f}, {1.f, 1.f}, {2.f, 2.f}, {3.f, 3.f}, {3.f, 3.f}}), 1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(sliding_window_cmn(sg, 3, 2, true, false),
                          torch::tensor({{-2.f, -2.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {2.f, 2.f}}), 1e-5, 1e-4);
    TM_CHECK_TENSOR_CLOSE(
            sliding_window_cmn(sg, 4, 2, false, true),
            torch::tensor(
                    {{-1.f, -1.f}, {1.f, 1.f}, {1.224745f, 1.224745f}, {1.341641f, 1.341641f}, {1.414214f, 1.414214f}}),
            1e-4, 1e-4);

    auto t = torch::arange(8000.0) / 8000.0;
    auto sine = torch::sin(2.0 * PI * 220.0 * t).reshape({1, 8000});
    auto pf = detect_pitch_frequency(sine, 8000, 0.01, 30, 85, 3400);
    TM_CHECK(pf.size(0) == 1 && pf.size(1) == 85);
    TM_CHECK_CLOSE(pf.median().item<float>(), 222.222f, 1.0f); // ~220 Hz (+1 calibration lag)
}

// ---------------- VAD (task05) ----------------
static void test_vad() {
    const int sr = 16000;
    auto t = torch::arange(static_cast<double>(sr / 2)) / sr;
    auto tone = 0.5 * torch::sin(2.0 * PI * 300.0 * t);
    auto sig = torch::cat({torch::zeros({sr / 2}), tone}).reshape({1, -1});
    auto out = vad(sig, sr);
    TM_CHECK(out.size(-1) == 9600); // trims ~6400 silent samples from the front
    // pure silence never triggers -> empty result.
    TM_CHECK(vad(torch::zeros({1, 8000}), sr).size(-1) == 0);
}

// ---------------- coverage-gap tests (error/branch paths in _functional.hpp) ----------------
static bool raises_inv(const std::function<void()> &fnc) {
    try {
        fnc();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static void test_coverage_gaps() {
    // mu_law_encoding: integer input -> the is_floating_point() cast branch.
    TM_CHECK(mu_law_encoding(torch::zeros({4}, torch::kInt32), 256).scalar_type() == torch::kInt64);

    // compute_deltas: the non-default pad modes.
    auto cd_in = torch::randn({2, 12});
    TM_CHECK(compute_deltas(cd_in, 5, "constant").sizes() == cd_in.sizes());
    TM_CHECK(compute_deltas(cd_in, 5, "reflect").sizes() == cd_in.sizes());
    TM_CHECK(compute_deltas(cd_in, 5, "circular").sizes() == cd_in.sizes());

    // add_noise: the lengths-mask branch.
    auto wav = torch::randn({2, 10});
    auto noise = torch::randn({2, 10});
    auto snr = torch::tensor({10.0f, 10.0f});
    auto lengths = torch::tensor({8, 10}, torch::kLong);
    TM_CHECK(add_noise(wav, noise, snr, lengths).sizes() == wav.sizes());

    // frechet_distance: both validation raise paths.
    auto mu3 = torch::zeros({3});
    auto s33 = torch::eye(3);
    TM_CHECK(raises_inv([&] { (void) frechet_distance(mu3, torch::eye(2), mu3, s33); })); // square/size mismatch
    TM_CHECK(raises_inv(
            [&] { (void) frechet_distance(mu3, s33, torch::zeros({2}), torch::eye(2)); })); // x/y shape mismatch

    // mask_along_axis: frequency axis (axis == dim-2) -> the maskb.unsqueeze branch.
    auto spec2d = torch::randn({4, 6});
    TM_CHECK(mask_along_axis(spec2d, 2, 0.0, /*axis=*/0).sizes() == spec2d.sizes());

    // mask_along_axis_iid: validation + early-return paths.
    TM_CHECK(raises_inv([] { (void) mask_along_axis_iid(torch::randn({4, 6}), 2, 0.0, 1); })); // dim < 3
    TM_CHECK(raises_inv([] { (void) mask_along_axis_iid(torch::randn({2, 4, 6}), 2, 0.0, 0); })); // bad axis
    TM_CHECK(raises_inv([] { (void) mask_along_axis_iid(torch::randn({2, 4, 6}), 2, 0.0, 2, 2.0); })); // p out of range
    auto iid_in = torch::randn({2, 4, 6});
    TM_CHECK_TENSOR_CLOSE(mask_along_axis_iid(iid_in, 0, 0.0, 2, 1.0), iid_in, 0.0, 0.0); // mp<1 early return

    // inverse_spectrogram: window-norm branch and the pad-trim branch.
    auto sig = torch::randn({1, 64});
    auto win = torch::hann_window(16);
    auto cspec = spectrogram(
            sig, spectrogram_option().window(win).n_fft(16).hop_length(8).win_length(16).return_complex(true));
    TM_CHECK(inverse_spectrogram(cspec, 64, 0, win, 16, 8, 16, "window").size(-1) == 64);
    auto cspec_p = spectrogram(
            sig, spectrogram_option().pad(2).window(win).n_fft(16).hop_length(8).win_length(16).return_complex(true));
    TM_CHECK(inverse_spectrogram(cspec_p, 64, 2, win, 16, 8, 16, "none").size(-1) == 64);

    // sliding_window_cmn: window-clamp branch (few frames, default min_cmn_window=100).
    TM_CHECK(sliding_window_cmn(torch::randn({5, 4})).sizes() == torch::IntArrayRef({5, 4}));
    // norm_vars=true, cmn_window>1: the cur_sumsq subtraction as the window slides forward.
    auto cmn_nv = sliding_window_cmn(torch::randn({10, 4}), /*cmn_window=*/3, /*min_cmn_window=*/2,
                                     /*center=*/false, /*norm_vars=*/true);
    TM_CHECK(cmn_nv.sizes() == torch::IntArrayRef({10, 4}));
    // norm_vars=true, min_cmn_window=1: t=0 has a single-frame window -> the window_frames==1 zero-fill.
    auto cmn_nv1 = sliding_window_cmn(torch::randn({4, 3}), /*cmn_window=*/2, /*min_cmn_window=*/1,
                                      /*center=*/false, /*norm_vars=*/true);
    TM_CHECK(cmn_nv1.sizes() == torch::IntArrayRef({4, 3}));
}

// ======================== 2026-10-23 audit follow-up (progress02, task04-task09) ========================
// Golden values: gen_golden.py "2026-10-23 audit follow-up" section (torchaudio 2.5.1).
static auto ramp16() -> torch::Tensor { return torch::arange(16, torch::kFloat); }
static auto vec(const std::vector<float> &v) -> torch::Tensor { return torch::tensor(v); }
static auto mps_or_skip() -> c10::optional<torch::Device> {
    return torch::mps::is_available() ? c10::optional<torch::Device>(torch::kMPS) : c10::nullopt;
}

// task04 -- amplitude_to_DB mirrors torchaudio: top_db cutoff over (C, F, T) per packed batch item (G1-1/G3-1),
// AmplitudeToDB() defaults (G1-2), amin without a FLT_MIN floor (G1-6), double-precision parameters (G1-7).
static auto adb_input() -> torch::Tensor {
    auto c = torch::arange(1, 25, torch::kFloat).reshape({2, 3, 4});
    c[1].mul_(1e-6);
    return c;
}
static const std::vector<float> g_adb_ch0 = {0.0f,        3.01030016f, 4.77121258f, 6.02060032f,
                                             6.98970032f, 7.78151274f, 8.45098019f, 9.0309f,
                                             9.54242516f, 10.0f,       10.4139271f, 10.7918119f};

static void test_amplitude_to_DB_top_db_packing() {
    const auto c = adb_input();
    const auto o40 = amplitude_to_db_option().set_db_multiplier(0.0).set_top_db(40.0);
    // 3-D (C, F, T): one cutoff (global max - 40) shared by both channels -> channel 1 is clipped to -29.208.
    auto g3 = g_adb_ch0;
    g3.insert(g3.end(), 12, -29.2081871f);
    TM_CHECK_TENSOR_CLOSE(amplitude_to_DB(c, o40), vec(g3).reshape({2, 3, 4}), 1e-4, 1e-5);
    // 4-D (B, C, F, T): one cutoff per batch item.
    auto g4 = g3;
    for (const float v: {-30.0f, -26.9897003f, -25.2287865f, -23.9793987f, -23.0102997f, -22.2184868f, -21.5490189f,
                         -20.969101f, -20.4575748f, -20.0f, -19.5860729f, -19.2081871f}) {
        g4.push_back(v);
    }
    g4.insert(g4.end(), 12, -59.2081871f);
    TM_CHECK_TENSOR_CLOSE(amplitude_to_DB(torch::stack({c, c * 1e-3}), o40), vec(g4).reshape({2, 2, 3, 4}), 1e-4, 1e-5);
    // 2-D (F, T) packs to one channel: the cutoff is the max over the whole matrix.
    TM_CHECK_TENSOR_CLOSE(amplitude_to_DB(torch::tensor({{1.0f, 1e-6f}, {0.1f, 1e-2f}}), o40),
                          torch::tensor({{0.0f, -40.0f}, {-10.0f, -20.0f}}), 1e-4, 1e-5);
    // top_db needs (F, T): torchaudio raises for 1-D input.
    TM_CHECK(raises_inv([&] { (void) amplitude_to_DB(c.flatten(), o40); }));
}

static void test_amplitude_to_DB_defaults_amin_precision() {
    // default option == torchaudio AmplitudeToDB(): ref 1.0 (db_multiplier 0), no top_db.
    auto gd = g_adb_ch0;
    for (const float v: {-48.8605652f, -48.5387192f, -48.23909f, -47.9587975f, -47.6955109f, -47.4472771f, -47.2124634f,
                         -46.9897003f, -46.7778053f, -46.5757751f, -46.3827209f, -46.1978874f}) {
        gd.push_back(v);
    }
    TM_CHECK_TENSOR_CLOSE(amplitude_to_DB(adb_input()), vec(gd).reshape({2, 3, 4}), 1e-4, 1e-5);
    // amin is used as given: amin=0 maps an exact zero to -inf like torch.clamp(x, min=0) + log10.
    const auto z = amplitude_to_DB(torch::tensor({0.0f, 1e-45f, 1.0f}).reshape({1, 3}),
                                   amplitude_to_db_option().set_amin(0.0));
    TM_CHECK(std::isinf(z[0][0].item<float>()) && z[0][0].item<float>() < 0.0f);
    TM_CHECK_CLOSE(z[0][1].item<float>(), -448.5346984863281, 1e-3);
    // double parameters keep float64 input exact (db_multiplier = log10(2), multiplier 20).
    const auto x64 = torch::tensor({0.5, 0.25, 2.0}, torch::kDouble).reshape({1, 3});
    const auto d64 =
            amplitude_to_DB(x64, amplitude_to_db_option().set_multiplier(20.0).set_db_multiplier(std::log10(2.0)));
    TM_CHECK_TENSOR_CLOSE(d64,
                          torch::tensor({-12.041199826559248, -18.06179973983887, 0.0}, torch::kDouble).reshape({1, 3}),
                          1e-12, 0.0);
    TM_CHECK_TENSOR_CLOSE(db_to_amplitude(torch::tensor({3.0, -7.5}, torch::kDouble), 0.1, 0.3),
                          torch::tensor({0.12302687708123816, 0.05956621435290105}, torch::kDouble), 1e-15, 0.0);
}

// task05 -- spectrogram: frame_length == torch.stft(normalized=true) (G2-1), unknown method throws (G2-8),
// N-D input packed to (-1, L) (G2-3/G3-5), win/hop 0 => derived from n_fft (G2-5/G3-6).
static void test_spectrogram_frame_length_parity() {
    const auto opt = spectrogram_option()
                             .n_fft(8)
                             .win_length(8)
                             .hop_length(4)
                             .power(1.0)
                             .return_complex(false)
                             .normalized(true)
                             .normalize_method("frame_length");
    const auto g8 = vec({1.62132013f,  5.65685368f,     11.3137074f,     16.970562f,      19.2383289f,
                         0.207106829f, 3.13561893f,     5.8165369f,      8.59256077f,     10.0821371f,
                         0.707106769f, 0.292893291f,    0.29289344f,     0.292893618f,    0.612372339f,
                         0.207106829f, 0.0606605262f,   0.0606608614f,   0.0606610291f,   0.207107008f,
                         0.207106665f, 3.37174782e-07f, 6.74349565e-07f, 6.74349565e-07f, 0.146445796f});
    TM_CHECK_TENSOR_CLOSE(spectrogram(ramp16(), opt), g8.reshape({5, 5}), 1e-5, 1e-5);
    // win_length (6) != n_fft (8): the scale is n_fft^-0.5, not 1/win_length.
    auto opt6 = opt;
    opt6.win_length(6);
    const auto g6 =
            vec({0.883883476f,    4.2426405f,  8.48528099f,   12.7279215f,  14.6724653f,  0.37499997f,  3.00390244f,
                 5.87378502f,     8.7729435f,  10.2146606f,   0.353553414f, 0.883883297f, 1.51038051f,  2.18660665f,
                 2.70416307f,     0.37499997f, 0.0884265229f, 0.172907904f, 0.258251131f, 0.359783262f, 0.176776648f,
                 1.68587391e-07f, 0.0f,        0.0f,          0.176776692f});
    TM_CHECK_TENSOR_CLOSE(spectrogram(ramp16(), opt6), g6.reshape({5, 5}), 1e-5, 1e-5);
    // an unknown method throws instead of silently skipping normalization (torchaudio ValueError).
    auto bad = opt;
    bad.normalize_method("none");
    TM_CHECK(raises_inv([&] { (void) spectrogram(ramp16(), bad); }));
}

static void test_spectrogram_nd_and_defaults() {
    torch::manual_seed(0);
    const auto x = torch::randn({2, 3, 64});
    const auto opt = spectrogram_option().n_fft(16).win_length(16).hop_length(8).return_complex(false);
    const auto s = spectrogram(x, opt);
    TM_CHECK(s.sizes() == torch::IntArrayRef({2, 3, 9, 9}));
    TM_CHECK_TENSOR_CLOSE(s[1][2], spectrogram(x[1][2], opt), 1e-6, 1e-6);
    const auto c = spectrogram(torch::randn({2, 2, 2, 64}), spectrogram_option().n_fft(16).return_complex(true));
    TM_CHECK(c.is_complex() && c.sizes() == torch::IntArrayRef({2, 2, 2, 9, 9}));

    // only n_fft set: win_length = n_fft, hop_length = win_length / 2 (torchaudio transform defaults).
    const auto w = torch::randn({1, 1024});
    const auto d = spectrogram(w, spectrogram_option().n_fft(256).return_complex(false));
    TM_CHECK(d.sizes() == torch::IntArrayRef({1, 129, 9}));
    TM_CHECK_TENSOR_CLOSE(
            d, spectrogram(w, spectrogram_option().n_fft(256).win_length(256).hop_length(128).return_complex(false)),
            0.0, 0.0);
    // the default n_fft=400 still derives win 400 / hop 200.
    TM_CHECK_TENSOR_CLOSE(spectrogram(w, spectrogram_option()),
                          spectrogram(w, spectrogram_option().win_length(400).hop_length(200)), 0.0, 0.0);
    mel_spectrogram_option mo;
    mo.n_fft = 1024;
    TM_CHECK(melspectrogram(torch::randn({1, 16000}), mo).sizes() == torch::IntArrayRef({1, 128, 32}));
    TM_CHECK(raises_inv([] { (void) spectrogram(torch::tensor(1.0f), spectrogram_option()); })); // no time dim
}

// task06 -- griffinlim: length in every istft (G2-2), torchaudio rand_init (G2-6), momentum/rank validation
// (G2-7/G2-9), window dtype follows the spectrogram (G2-10), derived win/hop (G2-5); inverse_spectrogram rank.
static auto gl8() -> griffinlim_option {
    griffinlim_option g;
    g.n_fft = 8;
    g.win_length = 8;
    g.hop_length = 4;
    g.n_iter = 8;
    g.momentum = 0.99;
    return g;
}

static void test_griffinlim_parity() {
    const auto mag =
            spectrogram(ramp16(), spectrogram_option().n_fft(8).win_length(8).hop_length(4).return_complex(false));
    auto g = gl8();
    g.length = 18; // != the inferred 16
    const auto g18 = vec({-2.98023224e-08f, 0.990252018f, 2.11276579f, 3.04415154f, 4.02314472f, 4.97991943f,
                          5.23739147f, 6.26294136f, 7.63227892f, 9.41060448f, 12.111124f, 12.6491499f, 12.0032158f,
                          11.616848f, 11.5963821f, 13.2079496f, 14.0422859f, 14.814682f});
    TM_CHECK_TENSOR_CLOSE(griffinlim(mag, g), g18, 1e-5, 1e-5);

    auto gr = gl8();
    gr.n_iter = 4;
    gr.momentum = 0.5;
    gr.rand_init = true;
    torch::manual_seed(0); // torch.rand(complex) draws the same stream in Python and C++
    const auto grand = vec({0.146446496f, 0.303864747f, 2.44942617f, 3.81121278f, 4.06707764f, 4.39426899f, 6.08368301f,
                            7.81170273f, 8.2752037f, 8.28270435f, 9.15467739f, 11.0375671f, 12.1425934f, 12.9706297f,
                            13.7202454f, 13.6415434f});
    TM_CHECK_TENSOR_CLOSE(griffinlim(mag, gr), grand, 1e-5, 1e-5);

    // momentum outside [0, 1) and rank < 2 throw.
    for (const double m: {1.0, 1.5, -0.5}) {
        auto gm = gl8();
        gm.momentum = m;
        TM_CHECK(raises_inv([&] { (void) griffinlim(mag, gm); }));
    }
    TM_CHECK(raises_inv([&] { (void) griffinlim(mag[0], gl8()); }));
    TM_CHECK(raises_inv([] {
        (void) inverse_spectrogram(torch::ones({5}, torch::kComplexFloat), c10::nullopt, 0, torch::hann_window(8), 8, 4,
                                   8);
    }));
    TM_CHECK(raises_inv([] { // unknown normalization (torchaudio: "Invalid normalized parameter")
        (void) inverse_spectrogram(torch::ones({5, 3}, torch::kComplexFloat), c10::nullopt, 0, torch::hann_window(8), 8,
                                   4, 8, "foo");
    }));

    // float64 spectrogram: the default window is float64 too.
    auto gw = gl8();
    gw.window = torch::hann_window(8, torch::kDouble);
    const auto out64 = griffinlim(mag.to(torch::kDouble), gl8());
    TM_CHECK(out64.scalar_type() == torch::kDouble);
    TM_CHECK_TENSOR_CLOSE(out64, griffinlim(mag.to(torch::kDouble), gw), 1e-12, 0.0);

    // only n_fft set: win_length = n_fft, hop_length = win_length / 2.
    griffinlim_option gd;
    gd.n_fft = 8;
    gd.n_iter = 2;
    auto ge = gd;
    ge.win_length = 8;
    ge.hop_length = 4;
    TM_CHECK_TENSOR_CLOSE(griffinlim(mag, gd), griffinlim(mag, ge), 0.0, 0.0);
}

// task07 -- mel filterbank mirrors melscale_fbanks (sample_rate // 2, vectorized: G3-9/G3-3), invalid strings throw
// (G3-11), fb/DCT follow the waveform dtype/device (G3-2), mfcc sample_rate/n_mfcc (G3-7/G3-10), stereo top_db.
static void test_mel_filter_bank_parity() {
    const auto g_htk = vec(
            {-0.0f,        0.793843091f, 0.00830203854f, 0.0f,        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.206156909f,
             0.991697967f, 0.549680114f, 0.0945501179f,  0.0f,        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.450319916f,
             0.905449867f, 0.79108566f,  0.52739042f,    0.26369521f, 0.0f});
    TM_CHECK_TENSOR_CLOSE(mel_filter_bank(3, 0.0, 0.0, 11025, 9, "", "htk"), g_htk.reshape({3, 9}), 1e-7, 1e-6);
    const auto g_sla = vec({-0.0f,
                            0.00137791259f,
                            4.61866402e-05f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            3.22826527e-05f,
                            0.000914578442f,
                            0.000488945283f,
                            1.63211153e-05f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.0f,
                            0.000234882114f,
                            0.00047820303f,
                            0.000368132372f,
                            0.000245421601f,
                            0.0001227108f,
                            0.0f});
    TM_CHECK_TENSOR_CLOSE(mel_filter_bank(3, 0.0, 0.0, 11025, 9, "slaney", "slaney"), g_sla.reshape({3, 9}), 1e-10,
                          1e-6);

    // vectorized: the old per-element .item() loop took ~0.1 s for one 128 x 1025 bank.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) {
        (void) mel_filter_bank(128, 0.0, 0.0, 44100, 1025, "slaney", "slaney");
    }
    TM_CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5);

    TM_CHECK(raises_inv([] { (void) mel_filter_bank(4, 0.0, 0.0, 16000, 9, "foo", "htk"); }));
    TM_CHECK(raises_inv([] { (void) mel_filter_bank(4, 0.0, 0.0, 16000, 9, "", "foo"); }));
    TM_CHECK(raises_inv([] { (void) create_dct(4, 8, "foo"); }));
}

static auto mel16() -> mel_spectrogram_option {
    mel_spectrogram_option mo;
    mo.n_fft = 16;
    mo.win_length = 16;
    mo.hop_length = 8;
    mo.n_mels = 6;
    return mo;
}

static void test_melspectrogram_mfcc_parity() {
    const auto s = torch::sin(2.0 * PI * 0.11 * torch::arange(64, torch::kFloat));
    mfcc_option fo;
    fo.n_mfcc = 4;
    fo.mel = mel16();
    // mfcc_option.sample_rate is the rate (torchaudio MFCC(sample_rate, **melkwargs)); it overrides mel.sample_rate.
    auto f8 = fo;
    f8.sample_rate = 8000;
    const auto g8 = vec({-1.68409872f, -7.84579754f, -8.1784296f,  -7.92232513f, -7.5800581f,  -7.77191257f,
                         -8.15801334f, -7.99654627f, 4.18332243f,  -1.85283589f, 19.6016407f,  19.7420082f,
                         19.6328907f,  19.4981213f,  19.5721436f,  19.7330456f,  19.663784f,   4.27806568f,
                         -2.04555082f, -28.9420853f, -29.2900982f, -29.022089f,  -28.6645241f, -28.8648872f,
                         -29.2687187f, -29.0997066f, -8.29948521f, 3.44941854f,  2.98183155f,  3.1396656f,
                         3.01743221f,  2.8616848f,   2.94792175f,  3.12974548f,  3.05234623f,  -4.13999557f});
    TM_CHECK_TENSOR_CLOSE(mfcc(s, f8), g8.reshape({4, 9}), 1e-3, 1e-5);
    // stereo, channel 1 at -80 dB: torchaudio's top_db=80 cutoff is shared across channels (G3-1).
    const auto st = torch::stack({s, 1e-4 * torch::cos(2.0 * PI * 0.23 * torch::arange(64, torch::kFloat))});
    const auto gst = vec(
            {-19.8038521f,     -28.1426048f,     -28.4079247f,     -28.2037907f,     -27.9293652f,     -28.0834026f,
             -28.3916798f,     -28.2630272f,     -18.9366798f,     -36.8075142f,     -22.4600658f,     -22.2805557f,
             -22.4192848f,     -22.5992641f,     -22.4991531f,     -22.2917538f,     -22.3794785f,     -34.0135078f,
             -39.8719177f,     -56.8293381f,     -57.0345268f,     -56.8762512f,     -56.6677704f,     -56.7841911f,
             -57.0218277f,     -56.921875f,      -42.4429398f,     -29.8638916f,     -19.9228096f,     -19.676405f,
             -19.8660259f,     -20.1204853f,     -19.9777145f,     -19.6915054f,     -19.8110504f,     -31.8583241f,
             -169.484497f,     -169.483749f,     -169.483643f,     -169.484482f,     -169.483871f,     -169.483551f,
             -169.484436f,     -169.484009f,     -169.490204f,     -0.56785965f,     -0.568611145f,    -0.568725586f,
             -0.567874908f,    -0.56848526f,     -0.568817139f,    -0.5679245f,      -0.568359375f,    -0.562156677f,
             -7.62939453e-06f, -7.62939453e-06f, -7.62939453e-06f, -7.62939453e-06f, -7.62939453e-06f, -7.62939453e-06f,
             -7.62939453e-06f, -7.62939453e-06f, -7.62939453e-06f, 0.567882538f,     0.568632126f,     0.568748474f,
             0.567897797f,     0.568508148f,     0.56883812f,      0.567947388f,     0.568380356f,     0.562179565f});
    TM_CHECK_TENSOR_CLOSE(mfcc(st, fo), gst.reshape({2, 4, 9}), 1e-3, 1e-5);

    auto fbad = fo;
    fbad.n_mfcc = 7; // > n_mels (6)
    TM_CHECK(raises_inv([&] { (void) mfcc(s, fbad); }));
    auto fnorm = fo;
    fnorm.norm = "foo";
    TM_CHECK(raises_inv([&] { (void) mfcc(s, fnorm); }));

    // f_min > f_max throws (f_max defaults to sample_rate // 2, so 5512.25 Hz is above it at 11025 Hz).
    auto mb = mel16();
    mb.sample_rate = 11025;
    mb.f_min = 5512.25;
    TM_CHECK(raises_inv([&] { (void) melspectrogram(s, mb); }));
    // onesided is deprecated in torchaudio and always true: onesided=false no longer breaks the fb matmul.
    auto mo1 = mel16();
    mo1.onesided = false;
    TM_CHECK_TENSOR_CLOSE(melspectrogram(s, mo1), melspectrogram(s, mel16()), 0.0, 0.0);

    // fb / DCT follow the waveform dtype (and device).
    const auto m64 = melspectrogram(s.to(torch::kDouble), mel16());
    TM_CHECK(m64.scalar_type() == torch::kDouble);
    TM_CHECK_TENSOR_CLOSE(m64, melspectrogram(s, mel16()), 1e-4, 1e-5);
    TM_CHECK(mfcc(s.to(torch::kDouble), fo).scalar_type() == torch::kDouble);
    if (const auto dev = mps_or_skip()) {
        TM_CHECK_TENSOR_CLOSE(mfcc(s.to(*dev), fo), mfcc(s, fo), 1e-3, 1e-4);
    }
}

// task08 -- resample: identity (G4-4), validation (G4-2/G4-3/G4-14), kernel dtype follows the waveform (G4-5),
// float32 target length (G4-6), torchaudio kernel multiply order (G4-11); speed/pitch_shift inherit them.
static void test_resample_parity() {
    torch::manual_seed(0);
    const auto x = torch::randn({2, 100});
    TM_CHECK(resample(x, 16000, 16000).equal(x));
    TM_CHECK(raises_inv([&] { (void) resample(x, 0, 16000); }));
    TM_CHECK(raises_inv([&] { (void) resample(x, 16000, 0); }));
    TM_CHECK(raises_inv([&] { (void) resample(x, 0, 0); }));
    TM_CHECK(raises_inv([&] { (void) resample(x, -16000, 8000); }));
    resample_option lfw0;
    lfw0.lowpass_filter_width = 0;
    TM_CHECK(raises_inv([&] { (void) resample(x, 16000, 8000, lfw0); }));
    resample_option roll0;
    roll0.rolloff = 0.0;
    TM_CHECK(raises_inv([&] { (void) resample(x, 16000, 8000, roll0); }));
    TM_CHECK(raises_inv([] { (void) resample(torch::tensor(1.0f), 2, 1); }));
    TM_CHECK(raises_inv([&] { (void) resample(x.to(torch::kShort), 2, 1); }));

    const auto xr = torch::sin(2.0 * PI * 3.0 * torch::arange(64, torch::kFloat) / 64.0).reshape({1, 64});
    const auto r64 = resample(xr.to(torch::kDouble), 64, 48);
    TM_CHECK(r64.scalar_type() == torch::kDouble);
    TM_CHECK_CLOSE(r64.sum().item<double>(), 0.020850907711279287, 1e-12);
    TM_CHECK_CLOSE(r64[0][1].item<double>(), 0.3782239573026156, 1e-12);
    TM_CHECK(resample(torch::zeros({1, 180697}), 44100, 16000).size(-1) == 65559);

    const auto [kernel, width] = _sinc_resample_kernel(48000, 16000, 16000, resample_option(),
                                                       torchmedia::tensor_options_t().dtype(torch::kFloat));
    TM_CHECK(width == 19);
    const auto gk = vec(
            {1.59557077e-24f,  -8.17534328e-07f, -0.000183016979f, -0.000538223714f, 0.000244591618f, 0.00264129951f,
             0.00362526113f,   -0.00086147018f,  -0.00897745974f,  -0.0108616734f,   0.00169034069f,  0.0213740002f,
             0.0254518967f,    -0.00251340843f,  -0.0467810147f,   -0.0594799407f,   0.00311386609f,  0.135345891f,
             0.271942765f,     0.330000013f,     0.271942765f,     0.135345891f,     0.00311386609f,  -0.0594799407f,
             -0.0467810147f,   -0.00251340843f,  0.0254518967f,    0.0213740002f,    0.00169034069f,  -0.0108616734f,
             -0.00897745974f,  -0.00086147018f,  0.00362526113f,   0.00264129951f,   0.000244591618f, -0.000538223714f,
             -0.000183016979f, -8.17534328e-07f, 1.59557077e-24f,  1.59557077e-24f,  1.59557077e-24f});
    TM_CHECK_TENSOR_CLOSE(kernel.flatten(), gk, 0.0, 0.0); // bit-exact with torchaudio's float32 kernel

    // speed / pitch_shift reuse resample.
    TM_CHECK(speed(x, 16000, 1.0).first.equal(x));
    TM_CHECK(raises_inv([&] { (void) speed(x, 16000, 0.0); }));
    TM_CHECK(speed(x.to(torch::kDouble), 16000, 1.5).first.scalar_type() == torch::kDouble);
    const auto xl = torch::randn({1, 2000});
    TM_CHECK_TENSOR_CLOSE(pitch_shift(xl, 16000, 0), xl, 1e-5, 1e-5); // only the stft/istft round trip remains
    TM_CHECK(pitch_shift(xl.to(torch::kDouble), 16000, 2).scalar_type() == torch::kDouble);
    // odd length at -12 steps: the stretch length round(2001 / 2) rounds half to even like Python (1000, not 1001),
    // so the last sample is _fix_waveform_shape's zero padding.
    const auto odd = torch::sin(torch::arange(2001, torch::kFloat) * 0.05).reshape({1, 2001});
    const auto ps12 = pitch_shift(odd, 16000, -12);
    TM_CHECK(ps12[0][2000].item<float>() == 0.0f);
    TM_CHECK_CLOSE(ps12[0][1999].item<float>(), -0.16586455702781677, 1e-5);
    TM_CHECK_CLOSE(ps12[0][1000].item<float>(), -0.10505791008472443, 1e-5);
}

// task09 -- convolve/fftconvolve share one input check: 0-D and non-broadcastable leading dims throw
// std::invalid_argument (G1-5 + fftconvolve prelude); lower-rank operands are left-padded (documented extension).
static void test_convolve_shape_checks() {
    const auto full_mode = convolve_mode::full;
    TM_CHECK(raises_inv([&] { (void) fftconvolve(torch::tensor(1.0f), torch::randn({3}), full_mode); }));
    TM_CHECK(raises_inv([&] { (void) fftconvolve(torch::randn({3}), torch::tensor(1.0f), full_mode); }));
    TM_CHECK(raises_inv([&] { (void) fftconvolve(torch::randn({2, 10}), torch::randn({3, 4}), full_mode); }));
    TM_CHECK(raises_inv([&] { (void) convolve(torch::randn({2, 10}), torch::randn({3, 4}), full_mode); }));
    const auto x = torch::randn({1, 10});
    const auto y = torch::randn({4});
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, full_mode), fftconvolve(x, y.unsqueeze(0), full_mode), 1e-6, 1e-6);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(y, x, full_mode), convolve(y, x, full_mode), 1e-5, 1e-5);
    // mode defaults to full (torchaudio), and the scoped enum no longer collides with torch::full (G1-9).
    TM_CHECK_TENSOR_CLOSE(convolve(x, y), convolve(x, y, full_mode), 0.0, 0.0);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y), fftconvolve(x, y, full_mode), 0.0, 0.0);
    static_assert(!std::is_convertible_v<convolve_mode, int>, "convolve_mode must be a scoped enum");
}

// Error/branch paths of the filtering, beamforming and alignment headers (their error sites now return the
// handle_exceptions value; these were never exercised). Goldens: gen_golden.py "coverage gaps" section.
static void test_coverage_gaps_filtering_beamforming_alignment() {
    // _lfilter_core / lfilter validation.
    const auto w3 = torch::randn({1, 1, 8});
    TM_CHECK(raises_inv([&] { (void) _lfilter_core(w3, torch::ones({1, 3}), torch::ones({1, 2})); }));
    TM_CHECK(raises_inv([] { (void) _lfilter_core(torch::randn({8}), torch::ones({1, 3}), torch::ones({1, 3})); }));
    TM_CHECK(raises_inv([&] { (void) _lfilter_core(w3, torch::ones({1, 0}), torch::ones({1, 0})); }));
    TM_CHECK(raises_inv([] { (void) lfilter(torch::randn({3, 8}), torch::ones({2, 3}), torch::ones({2, 3})); }));
    // riaa_biquad 48 kHz / 88.2 kHz tables.
    const auto w = filt_wav();
    TM_CHECK_TENSOR_CLOSE(riaa_biquad(w, 48000),
                          vec({0.0219737832f, -0.0224744882f, 0.0404246412f, -0.0439121127f, 0.059642449f,
                               -0.0647824109f, 0.0792799741f, -0.0853421539f}),
                          1e-6, 1e-5);
    TM_CHECK_TENSOR_CLOSE(riaa_biquad(w, 88200),
                          vec({0.0126710637f, -0.0125759393f, 0.0238931254f, -0.024962673f, 0.0352777243f,
                               -0.0372097716f, 0.046782285f, -0.0493537933f}),
                          1e-6, 1e-5);
    // flanger: triangular LFO golden; bad interpolation and > 4 channels raise.
    const auto wf = torch::stack({w, torch::tensor({0.2f, -0.1f, 0.4f, -0.3f, 0.6f, -0.5f, 0.8f, -0.7f})});
    TM_CHECK_TENSOR_CLOSE(flanger(wf, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "triangular", "linear").flatten(),
                          vec({0.099999994f, -0.19975087f, 0.2991696f, -0.398256123f, 0.497010529f, -0.595432758f,
                               0.693522811f, -0.791280627f, 0.116959058f, -0.0584795289f, 0.233918115f, -0.175438598f,
                               0.350877196f, -0.292397648f, 0.467836231f, -0.409356713f}),
                          1e-5, 1e-5);
    TM_CHECK(raises_inv([&] { (void) flanger(wf, 8000, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "sinusoidal", "cubic"); }));
    TM_CHECK(raises_inv([] { (void) flanger(torch::randn({5, 8}), 8000); }));
    // vad: lifter band that leaves no cepstrum bins.
    TM_CHECK(raises_inv([] {
        (void) vad(torch::randn({1, 8000}), 16000, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, std::nullopt,
                   0.4, 50.0, 6000.0, 2000.0, 150.0);
    }));

    // beamforming validation + rtf_power n_iter == 1.
    const auto c3 = torch::complex(torch::randn({3, 2, 2}), torch::randn({3, 2, 2}));
    TM_CHECK(raises_inv([&] { (void) rtf_power(c3[0], c3[0], 0); })); // < 3-D
    TM_CHECK(raises_inv([&] { (void) rtf_power(torch::real(c3), torch::real(c3), 0); })); // real
    TM_CHECK(raises_inv([&] { (void) rtf_power(c3, c3.index({torch::indexing::Slice(0, 2)}), 0); })); // shape
    const auto c32 = torch::complex(torch::randn({3, 2, 3}), torch::randn({3, 2, 3}));
    TM_CHECK(raises_inv([&] { (void) rtf_power(c32, c32, 0); })); // not square
    TM_CHECK(raises_inv([&] { (void) rtf_power(c3, c3, 0, 0); })); // n_iter <= 0
    TM_CHECK(raises_inv([&] { (void) rtf_evd(torch::real(c3)); }));
    TM_CHECK(raises_inv([&] { (void) rtf_evd(c32); }));
    const auto ms = torch::complex(torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}),
                                   torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}) * 0.2f);
    const auto ps = torch::matmul(ms, ms.conj().transpose(-1, -2)) + 2.0f * torch::eye(2);
    const auto mn = torch::complex(torch::arange(13, 25, torch::kFloat).reshape({3, 2, 2}) * 0.1f,
                                   torch::arange(1, 13, torch::kFloat).reshape({3, 2, 2}) * 0.3f);
    const auto pn = torch::matmul(mn, mn.conj().transpose(-1, -2)) + 2.0f * torch::eye(2);
    const auto r1 = rtf_power(ps, pn, 0, 1);
    TM_CHECK_TENSOR_CLOSE(torch::real(r1).flatten(), vec({7.2f, 11.44f, 65.44001f, 86.32f, 190.24f, 227.76f}), 1e-3,
                          1e-5);
    TM_CHECK_TENSOR_CLOSE(torch::imag(r1).flatten(), torch::zeros({6}), 1e-3, 0.0);

    // forced_align: a path that ends on the last token (not the final blank); max(targets) >= C raises.
    const auto logits =
            torch::tensor({2.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 2.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 4.f})
                    .reshape({1, 6, 3});
    const auto logp = torch::log_softmax(logits, -1);
    const auto [paths, scores] =
            forced_align(logp, torch::tensor({{1, 2}}, torch::kLong), c10::nullopt, c10::nullopt, 0);
    TM_CHECK(paths.equal(torch::tensor({{0, 1, 1, 0, 2, 2}}, torch::kLong)));
    TM_CHECK_CLOSE(scores[0][5].item<float>(), -0.0359762, 1e-5);
    TM_CHECK(raises_inv(
            [&] { (void) forced_align(logp, torch::tensor({{1, 3}}, torch::kLong), c10::nullopt, c10::nullopt, 0); }));
}

// task11 -- review round 2 (progress02 D12). Goldens: gen_golden.py "task11" section (torchaudio 2.5.1).
static auto raises_any(const std::function<void()> &fnc) -> bool { // torch's own c10::Error included
    try {
        fnc();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

static void test_r2_create_dct() {
    // P2: create_dct builds in float32 in torchaudio's op order (float64-then-cast was 1.9e-5 off).
    const auto d40 = create_dct(40, 128, "");
    TM_CHECK_CLOSE(d40[122][36].item<float>(), 0.29344236850738525, 1e-9);
    TM_CHECK_CLOSE(d40[84][34].item<float>(), 0.3419055938720703, 1e-9);
    TM_CHECK_CLOSE(d40[98][28].item<float>(), 0.29347845911979675, 1e-9);
    const auto d13 = create_dct(13, 40, "ortho");
    TM_CHECK(d13.sizes() == torch::IntArrayRef({40, 13}) && d13.scalar_type() == torch::kFloat);
    TM_CHECK_CLOSE(d13[34][12].item<float>(), 0.10151462256908417, 1e-10);
    TM_CHECK_CLOSE(d13[27][11].item<float>(), 0.04362407699227333, 1e-10);
}

static void test_r2_resample_equal_rates() {
    // P3: equal rates return the input before the filter arguments are checked (torchaudio order).
    torch::manual_seed(0);
    const auto x = torch::randn({2, 100});
    resample_option lfw0;
    lfw0.lowpass_filter_width = 0;
    resample_option roll0;
    roll0.rolloff = 0.0;
    TM_CHECK(resample(x, 16000, 16000, lfw0).equal(x));
    TM_CHECK(resample(x, 16000, 16000, roll0).equal(x));
    TM_CHECK(raises_inv([&] { (void) resample(x, 0, 0, lfw0); })); // the rates are still checked first
}

static void test_r2_spectrogram_args() {
    const auto x = torch::randn({2, 1000});
    // P4: normalize_method is validated even when normalized is false.
    TM_CHECK(raises_inv([&] { (void) spectrogram(x, spectrogram_option().normalize_method("foo")); }));
    // R6: only 0 derives win/hop; a negative value reaches torch, which raises (as in torchaudio).
    TM_CHECK(raises_any([&] { (void) spectrogram(x, spectrogram_option().n_fft(64).win_length(64).hop_length(-1)); }));
    TM_CHECK(raises_any([&] { (void) spectrogram(x, spectrogram_option().n_fft(64).win_length(-5)); }));
    griffinlim_option gneg;
    gneg.n_fft = 16;
    gneg.hop_length = -1;
    TM_CHECK(raises_any([&] { (void) griffinlim(torch::rand({9, 5}), gneg); }));
}

static void test_r2_spectral_centroid() {
    // P5: spectral_centroid bin frequencies are float32 like torchaudio (a float64 linspace was 6e-6 Hz off).
    const auto t = torch::arange(3001, torch::kDouble);
    const auto xd = torch::stack({torch::sin(t * 0.05), 0.5 * torch::cos(t * 0.031)});
    const auto sc = spectral_centroid(xd, 16000, 0, torch::hann_window(384, torch::kDouble), 384, 96, 384);
    TM_CHECK(sc.scalar_type() == torch::kDouble && sc.sizes() == torch::IntArrayRef({2, 32}));
    TM_CHECK_CLOSE(sc[0][0].item<double>(), 268.47082870419354, 1e-9);
    TM_CHECK_CLOSE(sc[0][31].item<double>(), 224.7979686856453, 1e-9);
    TM_CHECK_CLOSE(sc[0][1].item<double>(), 214.82245513204614, 1e-9);
}

static void test_r2_composite_checks() {
    const auto x = torch::randn({2, 1000});
    // R3: pitch_shift checks its rates before stretching; bins_per_octave 0 was a division by zero.
    TM_CHECK(raises_inv([&] { (void) pitch_shift(x, 0, 2); }));
    TM_CHECK(raises_inv([&] { (void) pitch_shift(x, 16000, 2, 0); }));
    auto mscale = mel16();
    mscale.mel_scale = "foo";
    TM_CHECK(raises_inv([&] { (void) melspectrogram(x, mscale); }));
    mfcc_option fnorm;
    fnorm.mel = mel16();
    fnorm.n_mfcc = 4;
    fnorm.mel.norm = "foo";
    TM_CHECK(raises_inv([&] { (void) mfcc(x, fnorm); }));
}

static void test_r2_griffinlim_integer_input() {
    // R5: an integer spectrogram works again (the default window takes the dtype after pow).
    const auto magl = torch::randint(0, 5, {9, 5}, torch::kLong);
    griffinlim_option gi;
    gi.n_fft = 16;
    gi.n_iter = 2;
    TM_CHECK_TENSOR_CLOSE(griffinlim(magl, gi), griffinlim(magl.to(torch::kFloat), gi), 0.0, 0.0);
}

static void test_r2_inverse_spectrogram_false() {
    // R7: "false" means no normalization, mirroring the accepted "true" (torchaudio's bool normalized).
    const auto spc = spectrogram(ramp16(), spectrogram_option().n_fft(8).win_length(8).hop_length(4));
    const auto w8 = torch::hann_window(8);
    TM_CHECK_TENSOR_CLOSE(inverse_spectrogram(spc, 16, 0, w8, 8, 4, 8, "false"),
                          inverse_spectrogram(spc, 16, 0, w8, 8, 4, 8, "none"), 0.0, 0.0);
}

// ======================== progress04 -- surface audit (task12-task19) ========================
// Goldens: gen_golden.py "progress04" section (torchaudio 2.5.1).
template<class Fn>
static auto wall_ms(Fn &&f) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    (void) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// task12 -- lfilter / overdrive run torchaudio's compiled scalar loops on the CPU (A1-01, A1-05).
static auto lf3d_input(torch::ScalarType dt) -> std::array<torch::Tensor, 3> {
    const auto t = torch::arange(4000, torch::kDouble);
    const auto k = torch::arange(6, torch::kDouble).reshape({6, 1});
    const auto x =
            (0.5 * torch::sin(t * (0.01 + 0.007 * k)) + 0.3 * torch::sin(t * (0.3 + 0.05 * k))).reshape({2, 3, 4000});
    const auto a = torch::tensor({1.0, -1.6, 0.8, 2.0, -0.5, 0.1, 1.0, -1.98, 0.9801}, torch::kDouble).reshape({3, 3});
    const auto b = torch::tensor({0.2, 0.3, 0.1, 0.5, 0.0, -0.5, 0.01, 0.02, 0.01}, torch::kDouble).reshape({3, 3});
    return {x.to(dt), a.to(dt), b.to(dt)};
}

static void test_p4_lfilter_compiled_loop() {
    const int64_t idx[5][3] = {{0, 0, 3999}, {0, 1, 123}, {1, 2, 3999}, {1, 2, 2000}, {0, 2, 57}};
    const double g64[5] = {0.19155460062639007, 0.011517529585665685, 9.196526035082943, -5.892367795549261,
                           11.238014013084832};
    const double g32[5] = {0.19155456125736237, 0.01151752844452858, 9.196492195129395, -5.892395496368408,
                           11.238103866577148};
    const auto [x64, a64, b64] = lf3d_input(torch::kDouble);
    const auto [x32, a32, b32] = lf3d_input(torch::kFloat);
    const auto y64 = lfilter(x64, a64, b64, /*clamp=*/false);
    const auto y32 = lfilter(x32, a32, b32, /*clamp=*/false);
    TM_CHECK(y64.scalar_type() == torch::kDouble && y32.sizes() == x32.sizes());
    for (int i = 0; i < 5; ++i) {
        TM_CHECK_CLOSE(y64[idx[i][0]][idx[i][1]][idx[i][2]].item<double>(), g64[i], 1e-13); // bit-exact here
        TM_CHECK_CLOSE(y32[idx[i][0]][idx[i][1]][idx[i][2]].item<double>(), g32[i], 1e-6); // ~1 ulp at 10
    }
    // Autograd keeps the differentiable tensor-op loop (torchaudio's generic loop), which agrees with the fast one.
    const auto xg = x64.clone().requires_grad_(true);
    const auto yg = lfilter(xg, a64, b64, false);
    TM_CHECK_TENSOR_CLOSE(yg.detach(), y64, 1e-9, 1e-9);
    yg.sum().backward();
    TM_CHECK(xg.grad().defined() && xg.grad().sizes() == x64.sizes());
    if (const auto mps = mps_or_skip()) { // non-CPU devices take the generic loop too
        const auto xs = x32.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 200)});
        TM_CHECK_TENSOR_CLOSE(lfilter(xs.to(*mps), a32.to(*mps), b32.to(*mps), false).cpu(),
                              lfilter(xs, a32, b32, false), 5e-3, 1e-4); // MPS float32 matmul vs the scalar loop
    }
    // 10 s at 16 kHz: the per-sample tensor-op loop took ~800 ms, the compiled-style loop ~1 ms.
    torch::manual_seed(0);
    const auto x10 = torch::randn({1, 160000}) * 0.1;
    TM_CHECK(wall_ms([&] { return lowpass_biquad(x10, 16000, 1000.0); }) < 100.0);
    // Ill-conditioned order 6 with a0 = 1.7 (butter(5, 0.1) * 1.7): bit-exact only with the fused multiply-subtract
    // and with b/a0 before the conv1d; the order-2 filters above give the same float32 values without either.
    const auto a6 = torch::tensor(
            {1.7, -6.773723303340971, 10.939274053468974, -8.931145789598856, 3.6807259465310254, -0.6118780166080454},
            torch::kDouble);
    const auto b6 = torch::tensor({0.00010165282662900549, 0.0005082641331450275, 0.001016528266290055,
                                   0.001016528266290055, 0.0005082641331450275, 0.00010165282662900549},
                                  torch::kDouble);
    const auto y6_64 = lfilter(x64, a6, b6, false);
    const auto y6_32 = lfilter(x32, a6.to(torch::kFloat), b6.to(torch::kFloat), false);
    const int64_t i6[3][3] = {{0, 0, 3999}, {1, 1, 2500}, {1, 2, 777}};
    const double g6_64[3] = {0.597813830452131, 0.18334021665192268, 0.03366486722150463};
    const double g6_32[3] = {0.5977604985237122, 0.18334224820137024, 0.033612895756959915};
    for (int i = 0; i < 3; ++i) {
        TM_CHECK_CLOSE(y6_64[i6[i][0]][i6[i][1]][i6[i][2]].item<double>(), g6_64[i], 1e-13);
        TM_CHECK_CLOSE(y6_32[i6[i][0]][i6[i][1]][i6[i][2]].item<double>(), g6_32[i], 0.0);
    }
}

static void test_p4_overdrive_compiled_loop() {
    const auto t = torch::arange(3000, torch::kDouble);
    const auto ox = torch::stack({0.5 * torch::sin(t * 0.02) + 0.2 * torch::sin(t * 0.9),
                                  0.35 * torch::sin(t * 0.05) - 0.1 * torch::cos(t * 1.3)});
    const int64_t idx[3][2] = {{0, 2999}, {1, 1234}, {0, 7}};
    const double g64[3] = {-0.7610693966554736, -0.6155380194883864, 0.5359604727307031};
    const double g32[3] = {-0.7610692381858826, -0.6155380606651306, 0.5359604954719543};
    const auto y64 = overdrive(ox, 30.0, 40.0);
    const auto y32 = overdrive(ox.to(torch::kFloat), 30.0, 40.0);
    for (int i = 0; i < 3; ++i) {
        TM_CHECK_CLOSE(y64[idx[i][0]][idx[i][1]].item<double>(), g64[i], 1e-15);
        TM_CHECK_CLOSE(y32[idx[i][0]][idx[i][1]].item<double>(), g32[i], 1e-9); // bit-exact
    }
    const auto og = ox.clone().requires_grad_(true);
    TM_CHECK_TENSOR_CLOSE(overdrive(og, 30.0, 40.0).detach(), y64, 1e-12, 1e-12);
    if (const auto mps = mps_or_skip()) {
        const auto xs = ox.to(torch::kFloat).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 200)});
        TM_CHECK_TENSOR_CLOSE(overdrive(xs.to(*mps), 30.0, 40.0).cpu(), overdrive(xs, 30.0, 40.0), 1e-5, 1e-5);
    }
    torch::manual_seed(0);
    const auto x10 = torch::randn({1, 160000}) * 0.1;
    TM_CHECK(wall_ms([&] { return overdrive(x10); }) < 100.0); // was ~800 ms
}

// task13 -- vad keeps torchaudio's float32 measurement state on the host whatever the input (A1-02), validates (A1-04).
static auto vad_stereo(torch::ScalarType dt) -> torch::Tensor {
    using torch::indexing::Slice;
    const auto tv = torch::arange(16000, torch::kDouble) / 16000;
    auto v = torch::zeros({2, 24000}, torch::kDouble);
    v.index_put_({0, Slice(8000, torch::indexing::None)},
                 0.5 * torch::sin(2.0 * PI * 300.0 * tv) * (1.0 + 0.3 * torch::sin(2.0 * PI * 3.0 * tv)));
    v.index_put_({1, Slice(9000, torch::indexing::None)}, 0.3 * torch::sin(2.0 * PI * 520.0 * tv.slice(0, 0, 15000)));
    return v.to(dt);
}

static void test_p4_vad_dtype_device() {
    for (const auto dt: {torch::kFloat, torch::kDouble}) {
        const auto v = vad_stereo(dt);
        const auto o = vad(v, 16000, 7.0, 0.25, 1.0, 0.25, /*pre_trigger_time=*/0.1);
        TM_CHECK(o.scalar_type() == dt && o.size(-1) == 18400); // torchaudio: 18400 for both dtypes
        TM_CHECK(o.equal(v.slice(-1, 24000 - 18400)));
    }
    TM_CHECK(vad(vad_stereo(torch::kDouble)[0], 16000).dim() == 1); // 1-D float64
    if (const auto mps = mps_or_skip()) { // used to write MPS memory through a CPU accessor (crash at exit)
        const auto o = vad(vad_stereo(torch::kFloat).to(*mps), 16000, 7.0, 0.25, 1.0, 0.25, 0.1);
        TM_CHECK(o.device().is_mps() && o.size(-1) == 18400);
    }
}

static void test_p4_vad_validation() {
    const auto v = vad_stereo(torch::kFloat);
    TM_CHECK(raises_inv([&] { (void) vad(v, 16000, 7.0, 0.25, /*search_time=*/0.0); }));
    TM_CHECK(raises_inv([&] { (void) vad(v, 16000, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, /*mf=*/0.0); }));
    TM_CHECK(raises_inv([&] { (void) vad(v, 16000, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 40000.0); }));
    TM_CHECK(raises_inv([&] {
        (void) vad(v, 16000, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, std::nullopt, 0.4, 50.0, 6000.0,
                   /*hp_lifter_freq=*/0.0);
    }));
}

// task14 -- loudness, dither, compute_deltas, add_noise, phaser, flanger (A1-03, A1-06..A1-09, A1-12).
static void test_p4_loudness_11025() {
    // gate step = Python round(1102.5) = 1102 (half to even); std::llround gave 1103.
    const auto tl = torch::arange(33075, torch::kDouble) / 11025;
    const auto lx =
            torch::stack({0.3 * torch::sin(2.0 * PI * 440.0 * tl) * (1.0 + 0.5 * torch::sin(2.0 * PI * 3.0 * tl)),
                          0.2 * torch::sin(2.0 * PI * 1000.0 * tl)})
                    .to(torch::kFloat);
    TM_CHECK_CLOSE(loudness(lx, 11025).item<double>(), -12.05608081817627, 1e-5);
}

static void test_p4_dither_seeded_and_intentional_differences() {
    const auto dw =
            torch::tensor({{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f}, {0.05f, -0.05f, 0.15f, -0.15f, 0.25f, -0.25f}});
    const auto dw0 = dw.clone();
    torch::manual_seed(0);
    TM_CHECK_TENSOR_CLOSE(dither(dw, "RPDF").flatten(),
                          vec({0.100006103515625f, -0.199981689453125f, 0.29998779296875f, -0.39996337890625f,
                               0.499969482421875f, -0.5999755859375f, 0.04998779296875f, -0.04998779296875f,
                               0.149993896484375f, -0.149993896484375f, 0.25f, -0.25f}),
                          0.0, 0.0);
    // Intentional GPDF difference: torchaudio sums its 7 samples into a VIEW of the first one (mutating the input,
    // and re-reading the grown sample when an index repeats, as here: it differs at [0][4], [1][2], [1][3]). Golden:
    // torchaudio's draws and math on an unaliased copy (gen_golden.py), with and without noise shaping.
    torch::manual_seed(0);
    TM_CHECK_TENSOR_CLOSE(dither(dw, "GPDF").flatten(),
                          vec({0.100006103515625f, -0.199981689453125f, 0.29998779296875f, -0.39996337890625f, 0.5f,
                               -0.599945068359375f, 0.050018310546875f, -0.04998779296875f, 0.1500244140625f,
                               -0.14996337890625f, 0.25f, -0.249969482421875f}),
                          0.0, 0.0);
    torch::manual_seed(0);
    TM_CHECK_TENSOR_CLOSE(dither(dw, "GPDF", true).flatten(),
                          vec({0.100006103515625f, -0.19997557997703552f, 0.30000609159469604f, -0.39997559785842896f,
                               0.5000365972518921f, -0.599945068359375f, 0.050018310546875f, -0.04996948316693306f,
                               0.15003661811351776f, -0.14993897080421448f, 0.2500366270542145f, -0.249969482421875f}),
                          0.0, 0.0);
    TM_CHECK(dw.equal(dw0));
    TM_CHECK(dither(dw[0], "GPDF").sizes() == torch::IntArrayRef({6})); // torchaudio raises on mono GPDF
    (void) dcshift(dw, 0.3, 0.05);
    TM_CHECK(dw.equal(dw0)); // torchaudio's dcshift with a limiter writes into its input
}

static void test_p4_effect_validation() {
    const auto cd = torch::randn({2, 40, 20});
    TM_CHECK(raises_inv([&] { (void) compute_deltas(cd, 5, "foo"); })); // torchaudio: Unrecognised padding mode
    TM_CHECK(raises_inv([&] { (void) compute_deltas(cd, 5, "Replicate"); }));

    const auto w = torch::randn({3, 1000});
    const auto n = torch::randn({3, 1000});
    const auto snr = torch::full({3}, 10.0);
    TM_CHECK(raises_inv([&] { (void) add_noise(w, n, torch::tensor(10.0)); })); // 0-D snr
    TM_CHECK(raises_inv([&] { (void) add_noise(w, n, snr, torch::full({1, 3}, 900, torch::kLong)); }));
    TM_CHECK(raises_inv([&] { (void) add_noise(w, n.slice(-1, 0, 999), snr); })); // length mismatch
    TM_CHECK(add_noise(w, n, snr, torch::full({3}, 900, torch::kLong)).sizes() == w.sizes());
    // The lengths mask is an int64 arange. A float16 one also rounds lengths 2049 to 2048 in the comparison, which
    // drops index 2048: the only signal sample here, so no noise would be added (o16[0][0] would stay 0).
    auto w16 = torch::zeros({1, 2052}, torch::kHalf);
    w16[0][2048] = 1.0;
    const auto o16 = add_noise(w16, torch::ones({1, 2052}, torch::kHalf), torch::tensor({10.0}, torch::kHalf),
                               torch::tensor({2049}, torch::kLong));
    TM_CHECK_CLOSE(o16[0][0].item<double>(), 0.006977081298828125, 0.0); // torchaudio
    // Energies are torch.linalg.vector_norm(...) ** 2 as in torchaudio (a plain sum of squares is 1 ulp off).
    const auto tt = torch::arange(1000, torch::kDouble);
    const auto kk = torch::arange(3, torch::kDouble).reshape({3, 1});
    const auto an = add_noise((0.4 * torch::sin(tt * (0.05 + 0.02 * kk) + kk)).to(torch::kFloat),
                              (0.3 * torch::cos(tt * (0.17 + 0.03 * kk))).to(torch::kFloat),
                              torch::tensor({10.0f, 3.0f, -2.0f}), torch::tensor({1000, 700, 3}, torch::kLong));
    TM_CHECK_CLOSE(an[2][356].item<double>(), 0.6446744203567505, 0.0);
    TM_CHECK_CLOSE(an[2][315].item<double>(), -0.7973425388336182, 0.0);
    TM_CHECK_CLOSE(an[2][179].item<double>(), -0.7008386850357056, 0.0);
}

// task15 / task16 / task17 -- spectral & time ops (A2-04..A2-08).
static void test_p4_pitch_shift_fractional_steps() {
    const double steps = 0.5; // torchaudio: "(fractional) steps"; an int parameter silently truncated it to 0
    const auto xp = torch::sin(torch::arange(3001.0) * 0.05).unsqueeze(0);
    const auto pp = pitch_shift(xp, 1000, steps);
    TM_CHECK_CLOSE(pp[0][100].item<double>(), -0.9880214929580688, 1e-6);
    TM_CHECK_CLOSE(pp[0][1500].item<double>(), 0.25565478205680847, 1e-6);
    TM_CHECK_CLOSE(pp[0][2900].item<double>(), -0.12411279231309891, 1e-6);
    TM_CHECK(raises_inv([&] { (void) pitch_shift(xp, 16000, 2000); })); // orig_freq overflows int (was UB)
    TM_CHECK(raises_inv([&] { (void) pitch_shift(xp, 16000, -240); })); // orig_freq 0
}

static void test_p4_mask_rng_cpu_float32() {
    // torchaudio draws torch.rand(1) (CPU, float32) whatever the spectrogram's dtype/device.
    const auto ms = torch::arange(1, 1 + 2 * 30 * 100, torch::kDouble).reshape({2, 30, 100});
    const auto zero_cols = [](const torch::Tensor &row) { return (row == 0).nonzero().flatten(); };
    torch::manual_seed(0);
    TM_CHECK(zero_cols(mask_along_axis(ms, 20, 0.0, 2)[0][0]).equal(torch::arange(69, 78)));
    torch::manual_seed(0);
    TM_CHECK(zero_cols(mask_along_axis(ms.to(torch::kLong), 20, 0.0, 2)[0][0]).equal(torch::arange(69, 78)));
    torch::manual_seed(0);
    TM_CHECK(zero_cols(mask_along_axis(ms, 10, 0.0, 1)[0].select(1, 0)).equal(torch::arange(19, 23)));
    if (const auto mps = mps_or_skip()) {
        torch::manual_seed(0);
        const auto mm = mask_along_axis(ms.to(torch::kFloat).to(*mps), 20, 0.0, 2).cpu();
        TM_CHECK(zero_cols(mm[0][0]).equal(torch::arange(69, 78)));
    }
}

static auto dp_input() -> torch::Tensor {
    const auto tp = torch::arange(16001, torch::kDouble) / 16000;
    return torch::stack({torch::sin(2.0 * PI * 180.0 * tp) + 0.4 * torch::sin(2.0 * PI * 360.0 * tp + 0.3),
                         torch::sin(2.0 * PI * 233.0 * tp) * (1.0 + 0.2 * torch::sin(2.0 * PI * 5.0 * tp)),
                         torch::sin(2.0 * PI * 97.0 * tp) + 0.1 * torch::sin(2.0 * PI * 1100.0 * tp),
                         torch::sin(2.0 * PI * 410.0 * tp + 1.0)})
            .reshape({2, 2, 16001})
            .to(torch::kFloat);
}

static void test_p4_detect_pitch_parity_and_validation() {
    const auto x = dp_input();
    const auto pf = detect_pitch_frequency(x, 16000);
    TM_CHECK(pf.sizes() == torch::IntArrayRef({2, 2, 86}));
    // Python `int / tensor` is reciprocal() * int: 181.8181915... where a correctly rounded division gives ...7627.
    TM_CHECK(pf[0][0][5].item<float>() == 181.8181915283203f);
    TM_CHECK(pf[0][1][50].item<float>() == 231.88406372070312f);
    TM_CHECK(pf[1][0][85].item<float>() == 98.76543426513672f);
    TM_CHECK(pf[1][1][0].item<float>() == 205.12820434570312f);
    TM_CHECK_CLOSE(pf.to(torch::kDouble).sum().item<double>(), 77573.66893005371, 1e-7);
    const auto nc = _compute_nccf(x[0][0].slice(0, 0, 4000), 16000, 0.01, 85); // torch.linalg.vector_norm
    TM_CHECK_CLOSE(nc[3][40].item<double>(), -0.00801188126206398, 1e-12);
    TM_CHECK_CLOSE(nc[10][88].item<double>(), 0.011401122435927391, 1e-12);
    TM_CHECK_CLOSE(nc[24][150].item<double>(), -0.016313474625349045, 1e-12);
    // On these tones vector_norm and sqrt(sum(x * x)) agree; on a noise-like signal they differ (5e-7 relative).
    const auto tn = torch::arange(4000, torch::kDouble);
    const auto ncn =
            _compute_nccf((torch::sin(tn * tn * 1e-3) + 0.5 * torch::sin(tn * 2.3)).to(torch::kFloat), 16000, 0.01, 85);
    TM_CHECK_CLOSE(ncn[24][158].item<double>(), -0.022163026034832, 0.0);
    TM_CHECK_CLOSE(ncn[24][157].item<double>(), 0.015337648801505566, 0.0);
    TM_CHECK_CLOSE(ncn[24][156].item<double>(), 0.016815871000289917, 0.0);

    TM_CHECK(raises_inv([&] { (void) detect_pitch_frequency(x, 0); }));
    TM_CHECK(raises_inv([&] { (void) detect_pitch_frequency(x, 16000, 0.0); }));
    TM_CHECK(raises_inv([&] { (void) detect_pitch_frequency(x, 16000, 1e-2, 0); }));
    TM_CHECK(raises_inv([&] { (void) detect_pitch_frequency(x, 16000, 1e-2, 30, 0); }));
    TM_CHECK(raises_inv([&] { (void) detect_pitch_frequency(x, 16000, 1e-2, 30, 85, 0); }));
    // Extension: win_length 1 (no smoothing) works; torchaudio's _median_smoothing fails on torch.cat([]).
    TM_CHECK(detect_pitch_frequency(x, 16000, 1e-2, 1).sizes() == torch::IntArrayRef({2, 2, 101}));
}

// task18 -- forced_align mirrors torchaudio's compiled kernel and its checks (A3-1..A3-5, A3-9, A3-10).
static void test_p4_forced_align_parity() {
    const auto lk = [](const std::vector<int64_t> &v) { return torch::tensor(v, torch::kLong).unsqueeze(0); };
    // ties: torchaudio stays on x0 unless x1/x2 is strictly larger, and ends on the token when the terminals tie.
    auto [pt, st] = forced_align(torch::log_softmax(torch::zeros({1, 10, 5}), -1), lk({1, 2, 3}));
    TM_CHECK(pt.equal(lk({1, 2, 3, 3, 3, 3, 3, 3, 3, 3})));
    const auto lg = torch::log_softmax(torch::sin(torch::arange(30.0) * 1.3).reshape({1, 6, 5}), -1);
    TM_CHECK(forced_align(lg, lk({1, 1, 2, 2})).first.equal(lk({1, 0, 1, 2, 0, 2}))); // T == L + R
    // float64 accumulates in float64 (a float32 cast picks [1,0,2,2,3,3,3,3]).
    const auto i = torch::arange(32, torch::kDouble);
    const auto lg64 = torch::log_softmax(
            (torch::floor(torch::remainder(i * 2, 3)) + 1e-10 * torch::sin(i * 1.7 + 2)).reshape({1, 8, 4}), -1);
    TM_CHECK(forced_align(lg64, lk({1, 2, 3})).first.equal(lk({1, 0, 2, 2, 2, 2, 2, 3})));
    // float16 accumulates in Half (a float32 cast picks [1,2,3,3,3,3,3,4,0,0,0,0]).
    const auto lg16 = torch::tensor({-3.224609375,     -1.52734375,   -6.0234375,   -0.30712890625,   -5.23828125,
                                     -2.83203125,      -1.8779296875, -5.86328125,  -0.2467041015625, -5.5234375,
                                     -2.431640625,     -2.23828125,   -5.63671875,  -0.2247314453125, -5.75,
                                     -2.02734375,      -2.599609375,  -5.35546875,  -0.239990234375,  -5.91796875,
                                     -1.6328125,       -2.9609375,    -5.01953125,  -0.2958984375,    -6.02734375,
                                     -1.263671875,     -3.322265625,  -4.65234375,  -0.401123046875,  -6.08203125,
                                     -0.939453125,     -3.69140625,   -4.265625,    -0.56591796875,   -6.09375,
                                     -0.67431640625,   -4.0703125,    -3.8828125,   -0.79736328125,   -6.07421875,
                                     -0.4765625,       -4.45703125,   -3.5078125,   -1.09375,         -6.0234375,
                                     -0.342529296875,  -4.83984375,   -3.142578125, -1.4423828125,    -5.94140625,
                                     -0.26318359375,   -5.19140625,   -2.783203125, -1.826171875,     -5.8125,
                                     -0.2283935546875, -5.50390625,   -2.421875,    -2.2265625,       -5.62890625},
                                    torch::kHalf)
                              .reshape({1, 12, 5});
    TM_CHECK(forced_align(lg16, lk({1, 2, 3, 4})).first.equal(lk({1, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 4})));
    // blank != 0 and int32 targets: paths take the targets' dtype.
    const auto lgb = torch::log_softmax(torch::cos(torch::arange(40.0) * 0.9).reshape({1, 8, 5}), -1);
    const auto [pb, sb] = forced_align(lgb, torch::tensor({{1, 3, 4}}, torch::kInt), c10::nullopt, c10::nullopt, 2);
    TM_CHECK(pb.scalar_type() == torch::kInt && pb.equal(torch::tensor({{1, 2, 3, 3, 2, 2, 4, 2}}, torch::kInt)));
    TM_CHECK_TENSOR_CLOSE(sb.flatten(),
                          vec({-1.2012882232666016f, -1.0840137004852295f, -1.1872434616088867f, -2.324695348739624f,
                               -1.446059226989746f, -1.3689641952514648f, -0.7990177869796753f, -2.0998730659484863f}),
                          0.0, 0.0);
    // lengths that match are accepted.
    TM_CHECK(forced_align(lgb, lk({1, 3, 4}), torch::tensor({8}), torch::tensor({3}), 2).first.size(1) == 8);
}

static void test_p4_forced_align_validation() {
    const auto lk = [](const std::vector<int64_t> &v) { return torch::tensor(v, torch::kLong).unsqueeze(0); };
    const auto lp = torch::log_softmax(torch::sin(torch::arange(40.0)).reshape({1, 8, 5}), -1);
    const auto lp5 = lp.slice(1, 0, 5);
    TM_CHECK(raises_inv([&] { (void) forced_align(lp5, lk({1, 1, 2, 2})); })); // T 5 < L 4 + R 2
    TM_CHECK(raises_inv([&] { (void) forced_align(lp.slice(1, 0, 3), lk({1, 2, 3, 4})); })); // T < L
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, lk({1, 2}), c10::nullopt, c10::nullopt, -1); }));
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, lk({1, 2}), c10::nullopt, c10::nullopt, 5); }));
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, torch::tensor({{1, 2}, {3, 4}}, torch::kLong)); }));
    TM_CHECK(raises_inv([&] { (void) forced_align(torch::cat({lp, lp}), lk({1, 2})); })); // batch 2
    TM_CHECK(raises_inv([&] { (void) forced_align(lp[0], lk({1, 2})); })); // 2-D log_probs
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, torch::tensor({1, 2}, torch::kLong)); })); // 1-D targets
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, torch::empty({1, 0}, torch::kLong)); })); // empty targets
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, lk({1, 2}), torch::tensor({6}), c10::nullopt); }));
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, lk({1, 2}), c10::nullopt, torch::tensor({1})); }));
    TM_CHECK(raises_inv([&] { (void) forced_align(lp, lk({1, -2})); })); // negative target (stricter)

    const auto tok = torch::tensor({0, 1, 1, 0, 2, 2, 2, 0, 0, 4, 4, 0}, torch::kLong);
    const auto sc = torch::randn({12});
    TM_CHECK(raises_inv([&] { (void) merge_tokens(tok, sc.slice(0, 0, 8)); })); // torchaudio: must be the same length
    TM_CHECK(raises_inv([&] { (void) merge_tokens(tok.unsqueeze(0), sc.unsqueeze(0)); })); // must be 1-D
}

// task19 -- beamforming shape checks (A3-6..A3-8) and complex128 batched goldens (A3-16).
static auto bf_input() -> std::pair<torch::Tensor, torch::Tensor> {
    const auto n = torch::arange(2 * 3 * 5 * 7, torch::kDouble);
    const auto spec = torch::complex(torch::sin(n * 0.37), torch::cos(n * 0.11)).reshape({2, 3, 5, 7});
    const auto mask = (0.5 + 0.4 * torch::sin(torch::arange(2 * 5 * 7, torch::kDouble) * 0.23)).reshape({2, 5, 7});
    return {spec, mask};
}

static void test_p4_beamforming_goldens_and_validation() {
    const auto [spec, mask] = bf_input();
    const auto cclose = [](const torch::Tensor &z, double re, double im) {
        return std::abs(torch::real(z).item<double>() - re) < 1e-12 &&
               std::abs(torch::imag(z).item<double>() - im) < 1e-12;
    };
    const auto p = psd(spec, mask, /*normalize=*/false);
    TM_CHECK(p.sizes() == torch::IntArrayRef({2, 5, 3, 3}));
    TM_CHECK(cclose(p[1][4][2][0], 3.588256279720432, 2.540564735105338));
    TM_CHECK(cclose(p[0][0][1][2], 3.1487050893006217, -0.9914267928317912));
    const auto psd_n = psd(spec, 1.0 - mask);
    const auto ws = mvdr_weights_souden(psd(spec, mask), psd_n, 1);
    TM_CHECK(cclose(ws[1][3][2], -0.16985355138179073, -0.0402733696953217));
    TM_CHECK(cclose(ws[0][0][0], 0.015005770558839891, 0.25326959912293007));
    const auto wr = mvdr_weights_rtf(rtf_power(psd(spec, mask), psd_n, 2, 5), psd_n, 0);
    TM_CHECK(cclose(wr[1][2][1], 0.9412228437191267, 0.5980290613611818));
    TM_CHECK(cclose(wr[0][4][2], 0.365298469384894, 0.5759623617597058));

    // psd: the mask must be (..., freq, time) of the specgram (torchaudio ValueError).
    const auto s3 = spec[0].slice(0, 0, 3); // (3, 5, 7)
    TM_CHECK(raises_inv([&] { (void) psd(spec[0], mask[0].expand({3, 5, 7})); })); // per-channel mask
    TM_CHECK(raises_inv([&] { (void) psd(s3, torch::ones({5, 1}, torch::kDouble)); })); // broadcastable mask
    // mvdr_weights_rtf: both overloads validate (5 checks).
    const auto rtf = psd_n.select(-1, 0); // (2, 5, 3)
    const auto onehot = torch::tensor({1.0, 0.0, 0.0}, torch::kDouble);
    TM_CHECK(raises_inv([&] { (void) mvdr_weights_rtf(torch::real(rtf), torch::real(psd_n)); })); // real
    TM_CHECK(raises_inv([&] { (void) mvdr_weights_rtf(rtf[0], psd_n); })); // rtf vs psd_n[:-1] shape
    TM_CHECK(raises_inv([&] { (void) mvdr_weights_rtf(rtf[0][0], psd_n[0]); })); // rtf 1-D
    TM_CHECK(raises_inv([&] { (void) mvdr_weights_rtf(rtf, psd_n.slice(-1, 0, 2)); })); // non-square
    TM_CHECK(raises_inv([&] { (void) mvdr_weights_rtf(rtf[0], psd_n, onehot); }));
    // apply_beamforming: weights' leading dims must equal the specgram's (torchaudio ValueError).
    TM_CHECK(raises_inv([&] { (void) apply_beamforming(ws[0], spec); }));
    TM_CHECK(raises_inv([&] { (void) apply_beamforming(torch::real(ws), torch::real(spec)); })); // not complex
    TM_CHECK(apply_beamforming(ws, spec).sizes() == torch::IntArrayRef({2, 5, 7}));
}

// Crash-class red tests last: before the fixes these read out of bounds or crash the process.
static void test_p4_rank_and_degenerate_checks() {
    const auto c1 = torch::ones({65}, torch::kComplexFloat);
    const auto pa = torch::linspace(0, PI, 65).unsqueeze(-1);
    TM_CHECK(raises_inv([&] { (void) phase_vocoder(c1, 1.5, pa); })); // torchaudio: RuntimeError
    TM_CHECK(raises_inv([&] { (void) phase_vocoder(torch::ones({}, torch::kComplexFloat), 1.5, pa); }));
    TM_CHECK(phase_vocoder(c1, 1.0, pa).equal(c1)); // rate 1 returns early, as in torchaudio
    TM_CHECK(raises_inv([] { (void) flanger(torch::randn({8000}), 8000); })); // torchaudio: IndexError
    const auto w = torch::randn({1, 4000}) * 0.1;
    TM_CHECK(raises_inv([&] { (void) phaser(w, 16000, 0.4, 0.74, 3.0, 0.4, /*mod_speed=*/40000.0); }));
    TM_CHECK(raises_inv([&] { (void) phaser(w, 16000, 0.4, 0.74, /*delay_ms=*/0.0); }));
    TM_CHECK(raises_inv([] { (void) pitch_shift(torch::tensor(0.5f), 11025, 2); }));
    TM_CHECK(raises_inv([] { (void) detect_pitch_frequency(torch::tensor(0.5f), 16000); }));
}

int main() {
    test_coverage_gaps();
    test_convolve_full_length();
    test_convolve_modes();
    test_convolve_broadcast_and_errors();
    test_db_to_amplitude();
    test_amplitude_to_DB_power();
    test_amplitude_to_DB_magnitude_complex_flags();
    test_spectrogram_shape();
    test_spectrogram_power_abs();
    test_spectrogram_normalized();
    test_spectrogram_branches();
    test_mel_filter_bank_slaney();
    test_mel_filter_bank_branches();
    test_melspectrogram();
    test_create_dct();
    test_mfcc();
    test_griffinlim();
    test_griffinlim_branches();
    test_resample();
    test_lfilter_identity_fir();
    test_lfilter_one_pole();
    test_lfilter_golden();
    test_lfilter_clamp();
    test_lfilter_batching();
    test_biquad_golden();
    test_filtfilt_golden();
    test_biquad_designers_cookbook();
    test_biquad_designers_sox();
    test_biquad_designers_tables();
    test_effects_contrast_dcshift_gain();
    test_companding_emphasis();
    test_feature_ops();
    test_fftconvolve_addnoise_speed();
    test_metrics();
    test_dither();
    test_specaugment();
    test_stft_domain();
    test_loudness();
    test_beamforming();
    test_forced_align();
    test_modulated_effects();
    test_sequential_feature_ops();
    test_vad();
    test_option_setters();
    TM_RUN(test_amplitude_to_DB_top_db_packing);
    TM_RUN(test_amplitude_to_DB_defaults_amin_precision);
    TM_RUN(test_spectrogram_frame_length_parity);
    TM_RUN(test_spectrogram_nd_and_defaults);
    TM_RUN(test_griffinlim_parity);
    TM_RUN(test_mel_filter_bank_parity);
    TM_RUN(test_melspectrogram_mfcc_parity);
    TM_RUN(test_resample_parity);
    TM_RUN(test_convolve_shape_checks);
    TM_RUN(test_coverage_gaps_filtering_beamforming_alignment);
    TM_RUN(test_r2_create_dct);
    TM_RUN(test_r2_resample_equal_rates);
    TM_RUN(test_r2_spectrogram_args);
    TM_RUN(test_r2_spectral_centroid);
    TM_RUN(test_r2_composite_checks);
    TM_RUN(test_r2_griffinlim_integer_input);
    TM_RUN(test_r2_inverse_spectrogram_false);
    TM_RUN(test_p4_lfilter_compiled_loop);
    TM_RUN(test_p4_overdrive_compiled_loop);
    TM_RUN(test_p4_vad_dtype_device);
    TM_RUN(test_p4_loudness_11025);
    TM_RUN(test_p4_dither_seeded_and_intentional_differences);
    TM_RUN(test_p4_effect_validation);
    TM_RUN(test_p4_pitch_shift_fractional_steps);
    TM_RUN(test_p4_mask_rng_cpu_float32);
    TM_RUN(test_p4_detect_pitch_parity_and_validation);
    TM_RUN(test_p4_forced_align_parity);
    TM_RUN(test_p4_beamforming_goldens_and_validation);
    TM_RUN(test_p4_forced_align_validation);
    TM_RUN(test_p4_vad_validation);
    TM_RUN(test_p4_rank_and_degenerate_checks);
    return tm_test::summary("audio_test_functional");
}
