// Assertion-based tests for the audio functional layer (target: 100% coverage of _functional*.hpp).
// Golden values verified against torchaudio 2.5.1 (see gen_golden.py). Bug regression tests are RED
// against the pre-fix _functional.hpp and GREEN after task02's fixes.
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::audio::functional;
static const double PI = std::acos(-1.0);

// ---------------- convolve ----------------
static void test_convolve_full_length() {
    auto x = torch::tensor({1.f, 2.f, 3.f, 4.f}).reshape({1, 4});
    auto y = torch::tensor({1.f, 1.f}).reshape({1, 2});
    TM_CHECK(convolve(x, y, full).size(-1) == 5); // N + M - 1
}

// bug#4: 'same' returns the FIRST input's length; 'valid' = max-min+1 (x shorter than y)
static void test_convolve_modes() {
    auto x = torch::tensor({1.f, 2.f, 3.f}).reshape({1, 3});
    auto y = torch::tensor({1.f, 1.f, 1.f, 1.f, 1.f}).reshape({1, 5});
    TM_CHECK(convolve(x, y, full).size(-1) == 7);
    TM_CHECK(convolve(x, y, same).size(-1) == 3); // first input length
    TM_CHECK(convolve(x, y, valid).size(-1) == 3); // 5 - 3 + 1
}

static void test_convolve_broadcast_and_errors() {
    // differing ndim -> dimension-alignment (unsqueeze) branches
    TM_CHECK(convolve(torch::randn({1, 10}), torch::randn({4}), full).size(-1) == 13); // unsqueeze y
    TM_CHECK(convolve(torch::randn({5}), torch::randn({1, 3}), full).size(-1) == 7); // unsqueeze x

    // multi-dim leading dims with broadcasting
    auto x = torch::randn({2, 1, 10});
    auto y = torch::randn({1, 3, 4});
    auto out = convolve(x, y, full);
    TM_CHECK(out.size(0) == 2 && out.size(1) == 3);

    bool threw_0d = false;
    try {
        convolve(torch::tensor(1.0f), torch::tensor(1.0f), full);
    } catch (const std::invalid_argument &) {
        threw_0d = true;
    }
    TM_CHECK(threw_0d);

    bool threw_0d_y = false; // x is >=1D, y is 0D -> exercises the y.dim()==0 side of the ||
    try {
        (void) convolve(torch::tensor({1.f, 2.f}).reshape({1, 2}), torch::tensor(1.0f), full);
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
    // normalized=true but an unrecognized method -> neither normalize branch taken
    auto sp_none = spectrogram(sig, spectrogram_option()
                                            .window(torch::hann_window(400))
                                            .power(2.0)
                                            .return_complex(false)
                                            .normalized(true)
                                            .normalize_method("none"));
    TM_CHECK(sp_none.size(-2) == 201);
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
    TM_CHECK_CLOSE(rec.sum().item<double>(), 901.7087, 5.0); // golden sum
    TM_CHECK_CLOSE(rec[0][1000].item<double>(), -0.03503, 0.02); // golden sample (same ATen ops -> close)
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
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, full), torch::tensor({{1.f, 3.f, 6.f, 9.f, 7.f, 4.f}}), 1e-4, 1e-4);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, valid), torch::tensor({{6.f, 9.f}}), 1e-4, 1e-4);
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, same), torch::tensor({{3.f, 6.f, 9.f, 7.f}}), 1e-4, 1e-4);
    // fftconvolve agrees with the time-domain convolve()
    TM_CHECK_TENSOR_CLOSE(fftconvolve(x, y, full), convolve(x, y, full), 1e-4, 1e-4);

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
    return tm_test::summary("audio_test_functional");
}
