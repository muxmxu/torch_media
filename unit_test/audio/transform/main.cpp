// Assertion-based tests for the audio transform layer (target: 100% coverage of _transform*.hpp).
// Two kinds of checks per class:
//   (1) delegation-equivalence: transform::X(opt)(input) == functional::x(input, equiv_opt) exactly
//       (functional is golden-verified against torchaudio 2.5.1 in progress01 -> no venv needed).
//   (2) golden: a deterministic explicit input matched against baked torchaudio.transforms 2.5.1
//       values (see gen_golden.py) -> pins the torchaudio default derivation (win/hop, power, ...).
// First include on purpose: every audio header must compile on its own (A1-13: _functional.hpp used lfilter and
// treble_biquad without including _functional_filtering.hpp, so this line did not compile).
#include <torchmedia/_audio/_transform_effects.hpp>
// clang-format off
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"
// clang-format on

namespace tf = torchmedia::audio::transform;
namespace fn = torchmedia::audio::functional;
using torchmedia::tensor_options_t;
using torchmedia::tensor_t;

// Deterministic explicit input shared with gen_golden.py: a length-16 ramp.
static auto ramp16() -> tensor_t { return torch::arange(16, tensor_options_t().dtype(torch::kFloat)); }
static auto hann8() -> tensor_t { return torch::hann_window(8, tensor_options_t().dtype(torch::kFloat)); }

// ---------------- Spectrogram ----------------
static void test_spectrogram() {
    const auto x = ramp16();
    const auto S = tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4))(x);

    // (1) delegation-equivalence to functional::spectrogram with an explicit Hann window.
    const auto ref = fn::spectrogram(x, fn::spectrogram_option()
                                                .window(hann8())
                                                .n_fft(8)
                                                .win_length(8)
                                                .hop_length(4)
                                                .power(2.0)
                                                .normalized(false)
                                                .center(true)
                                                .onesided(true)
                                                .return_complex(false));
    TM_CHECK_TENSOR_CLOSE(S, ref, 1e-6, 1e-6);

    // (2) golden vs torchaudio.transforms.Spectrogram.
    TM_CHECK(S.dim() == 2 && S.size(0) == 5 && S.size(1) == 5);
    const std::vector<float> g = {21.0294323f, 255.9999695f, 1023.9998779f, 2304.0f,      2960.9064941f,
                                  0.3431459f,  78.6568527f,  270.6567993f,  590.6567993f, 813.1959229f,
                                  4.0f,        0.6862918f,   0.6862926f,    0.6862934f,   2.9999990f,
                                  0.3431459f,  0.0294376f,   0.0294379f,    0.0294380f,   0.3431463f,
                                  0.3431454f,  0.0f,         0.0f,          0.0f,         0.1715710f};
    const auto golden = torch::tensor(g).reshape({5, 5});
    TM_CHECK_TENSOR_CLOSE(S, golden, 1e-2, 1e-3);
}

// ---------------- SpectralCentroid ----------------
static void test_spectral_centroid() {
    const auto x = ramp16();
    const auto sc = tf::SpectralCentroid(
            tf::spectral_centroid_option().sample_rate(16000).n_fft(8).win_length(8).hop_length(4))(x);

    const auto ref = fn::spectral_centroid(x, 16000, 0, hann8(), 8, 4, 8);
    TM_CHECK_TENSOR_CLOSE(sc, ref, 1e-6, 1e-6);

    TM_CHECK(sc.dim() == 1 && sc.size(0) == 5);
    const auto golden = torch::tensor({2082.258057f, 853.570374f, 753.189697f, 722.340454f, 826.376770f});
    TM_CHECK_TENSOR_CLOSE(sc, golden, 1e-1, 1e-4);
}

// ---------------- GriffinLim ----------------
static void test_griffinlim() {
    const auto x = ramp16();
    const auto mag = tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).power(2.0))(x);

    const auto gl_opt = tf::griffinlim_option()
                                .n_fft(8)
                                .win_length(8)
                                .hop_length(4)
                                .power(2.0)
                                .n_iter(8)
                                .momentum(0.99)
                                .rand_init(false)
                                .length(16);
    const auto out = tf::GriffinLim(gl_opt)(mag);

    // (1) delegation-equivalence to functional::griffinlim.
    fn::griffinlim_option g;
    g.n_fft = 8;
    g.hop_length = 4;
    g.win_length = 8;
    g.window = hann8();
    g.power = 2.0;
    g.n_iter = 8;
    g.momentum = 0.99;
    g.length = 16;
    g.rand_init = false;
    const auto ref = fn::griffinlim(mag, g);
    TM_CHECK_TENSOR_CLOSE(out, ref, 1e-6, 1e-6);

    // (2) golden vs torchaudio.transforms.GriffinLim (deterministic: rand_init=false).
    TM_CHECK(out.dim() == 1 && out.size(0) == 16);
    const auto golden = torch::tensor({-0.29289332f, 0.736765623f, 2.27640057f, 3.30954361f, 3.81394148f, 4.78316498f,
                                       6.18080425f, 6.9954381f, 7.78199816f, 9.00509834f, 10.6329165f, 11.3739462f,
                                       11.7928171f, 12.7641764f, 14.2714882f, 14.6547432f});
    TM_CHECK_TENSOR_CLOSE(out, golden, 1e-5, 1e-6);
}

static void test_griffinlim_momentum_validation() {
    bool threw_high = false;
    try {
        tf::GriffinLim(tf::griffinlim_option().momentum(1.5));
    } catch (const std::invalid_argument &) {
        threw_high = true;
    }
    TM_CHECK(threw_high);

    bool threw_neg = false;
    try {
        tf::GriffinLim(tf::griffinlim_option().momentum(-0.1));
    } catch (const std::invalid_argument &) {
        threw_neg = true;
    }
    TM_CHECK(threw_neg);
}

// ---------------- InverseSpectrogram ----------------
static void test_inverse_spectrogram() {
    const auto x = ramp16();
    const auto spec_c =
            tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).return_complex(true))(x);
    TM_CHECK(spec_c.is_complex());

    const auto inv =
            tf::InverseSpectrogram(tf::inverse_spectrogram_option().n_fft(8).win_length(8).hop_length(4))(spec_c, 16);

    // (1) delegation-equivalence to functional::inverse_spectrogram.
    const auto ref = fn::inverse_spectrogram(spec_c, 16, 0, hann8(), 8, 4, 8, "none", true, "reflect", true);
    TM_CHECK_TENSOR_CLOSE(inv, ref, 1e-6, 1e-6);

    // (2) round-trip recovers the ramp (and matches baked torchaudio values).
    TM_CHECK(inv.dim() == 1 && inv.size(0) == 16);
    TM_CHECK_TENSOR_CLOSE(inv, x, 1e-4, 1e-4);
}

// ---------------- default derivation + caching ----------------
static void test_defaults_and_caching() {
    // win_length defaults to n_fft; hop_length to win_length/2; window cached at win_length.
    const tf::Spectrogram s(tf::spectrogram_option().n_fft(8));
    TM_CHECK(s.window().defined() && s.window().size(0) == 8);

    // Calling twice yields identical results (window cached, not rebuilt).
    const auto x = ramp16();
    TM_CHECK_TENSOR_CLOSE(s(x), s(x), 0.0, 0.0);

    // Explicit-window override path.
    const auto w = torch::hann_window(8, tensor_options_t().dtype(torch::kFloat));
    const tf::Spectrogram s2(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).window(w));
    TM_CHECK_TENSOR_CLOSE(s2.window(), w, 0.0, 0.0);

    // forward() is an alias of operator().
    TM_CHECK_TENSOR_CLOSE(s.forward(x), s(x), 0.0, 0.0);

    // Other classes expose a cached window too (coverage of their window()/forward()).
    const tf::InverseSpectrogram is(tf::inverse_spectrogram_option().n_fft(8));
    TM_CHECK(is.window().defined() && is.window().size(0) == 8);
    const tf::GriffinLim gl(tf::griffinlim_option().n_fft(8));
    TM_CHECK(gl.window().defined() && gl.window().size(0) == 8);
    const tf::SpectralCentroid scn(tf::spectral_centroid_option().n_fft(8));
    TM_CHECK(scn.window().defined() && scn.window().size(0) == 8);
    const auto sc_fwd = scn.forward(x);
    TM_CHECK(sc_fwd.size(0) == 5);
}

// Exercise every fluent setter + the remaining operator()/forward()/default-arg paths (coverage).
static void test_option_setters() {
    const auto x = ramp16();

    auto so = tf::spectrogram_option()
                      .pad(0)
                      .n_fft(8)
                      .win_length(8)
                      .hop_length(4)
                      .power(2.0)
                      .normalized(false)
                      .normalize_method("window")
                      .center(true)
                      .pad_mode("reflect")
                      .onesided(true)
                      .return_complex(false)
                      .window(hann8());
    TM_CHECK(tf::Spectrogram(so)(x).size(0) == 5);

    auto io = tf::inverse_spectrogram_option()
                      .pad(0)
                      .n_fft(8)
                      .win_length(8)
                      .hop_length(4)
                      .normalized("none")
                      .center(true)
                      .pad_mode("reflect")
                      .onesided(true)
                      .window(hann8());
    const auto spec_c =
            tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).return_complex(true))(x);
    const tf::InverseSpectrogram is(io);
    TM_CHECK(is(spec_c).size(0) > 0); // default length = nullopt path
    TM_CHECK(is.forward(spec_c, 16).size(0) == 16);

    auto go = tf::griffinlim_option()
                      .n_fft(8)
                      .n_iter(4)
                      .win_length(8)
                      .hop_length(4)
                      .power(2.0)
                      .momentum(0.5)
                      .length(16)
                      .rand_init(false)
                      .window(hann8());
    const auto mag = tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).power(2.0))(x);
    TM_CHECK(tf::GriffinLim(go).forward(mag).size(0) == 16);

    auto co = tf::spectral_centroid_option().sample_rate(16000).pad(0).n_fft(8).win_length(8).hop_length(4).window(
            hann8());
    TM_CHECK(tf::SpectralCentroid(co).forward(x).size(0) == 5);

    // InverseSpectrogram / GriffinLim forward() aliases exercised above; SpectralCentroid window().
    TM_CHECK(is.window().defined());
}

// ======================== task20: mel & cepstral ========================
static auto power_spec8() -> tensor_t {
    return tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4))(ramp16());
}

static void test_amplitude_to_db() {
    const auto spec = power_spec8();
    const auto adb = tf::AmplitudeToDB(tf::amplitude_to_db_option().stype(tf::db_stype::power).top_db(80.0))(spec);

    const auto ref = fn::amplitude_to_DB(spec, fn::amplitude_to_db_option()
                                                       .set_multiplier(10.0f)
                                                       .set_amin(1e-10f)
                                                       .set_db_multiplier(0.0f)
                                                       .set_top_db(80.0f)
                                                       .set_apply_top_db(true));
    TM_CHECK_TENSOR_CLOSE(adb, ref, 1e-6, 1e-6);

    const std::vector<float> g = {13.2282753f, 24.0823975f,  30.1029987f,  33.6248245f,  34.7142448f,
                                  -4.6452117f, 18.9573650f,  24.3241882f,  27.7133522f,  29.1019516f,
                                  6.0206003f,  -1.6349118f,  -1.6349070f,  -1.6349016f,  4.7712111f,
                                  -4.6452117f, -15.3109760f, -15.3109283f, -15.3109169f, -4.6452065f,
                                  -4.6452184f, -45.2857552f, -45.2857552f, -45.2857552f, -7.6555624f};
    TM_CHECK_TENSOR_CLOSE(adb, torch::tensor(g).reshape({5, 5}), 1e-2, 1e-3);

    // magnitude stype + default (no top_db) path.
    const auto mag = tf::AmplitudeToDB(tf::amplitude_to_db_option().stype(tf::db_stype::magnitude))(spec);
    const auto mag_ref = fn::amplitude_to_DB(spec, fn::amplitude_to_db_option()
                                                           .set_multiplier(20.0f)
                                                           .set_amin(1e-10f)
                                                           .set_db_multiplier(0.0f)
                                                           .set_apply_top_db(false));
    TM_CHECK_TENSOR_CLOSE(mag, mag_ref, 1e-6, 1e-6);
}

static void test_mel_scale() {
    const auto spec = power_spec8();
    const auto ms = tf::MelScale(tf::mel_scale_option().n_mels(4).sample_rate(16000).n_stft(5))(spec);

    const auto fb = fn::mel_filter_bank(4, 0.0, 0.0, 16000, 5, "", "htk");
    TM_CHECK_TENSOR_CLOSE(ms, fn::mel_scale(spec, fb), 1e-6, 1e-6);

    const std::vector<float> g = {0.0f,       0.0f,        0.0f,         0.0f,         0.0f,
                                  0.1296862f, 29.7270203f, 102.2901382f, 223.2286987f, 307.3335571f,
                                  1.2818550f, 49.1131363f, 168.5499573f, 367.6114197f, 506.6636353f,
                                  3.1308622f, 0.5200779f,  0.5200786f,   0.5200793f,   2.3979607f};
    TM_CHECK_TENSOR_CLOSE(ms, torch::tensor(g).reshape({4, 5}), 1e-2, 1e-3);
}

static void test_mel_spectrogram() {
    const auto x = ramp16();
    const auto mel = tf::MelSpectrogram(
            tf::mel_spectrogram_option().sample_rate(16000).n_fft(8).win_length(8).hop_length(4).n_mels(4))(x);

    fn::mel_spectrogram_option mo;
    mo.sample_rate = 16000;
    mo.n_fft = 8;
    mo.win_length = 8;
    mo.hop_length = 4;
    mo.n_mels = 4;
    TM_CHECK_TENSOR_CLOSE(mel, fn::melspectrogram(x, mo), 1e-6, 1e-6);

    // MelSpectrogram == MelScale(Spectrogram) -> same golden as test_mel_scale.
    const std::vector<float> g = {0.0f,       0.0f,        0.0f,         0.0f,         0.0f,
                                  0.1296862f, 29.7270203f, 102.2901382f, 223.2286987f, 307.3335571f,
                                  1.2818550f, 49.1131363f, 168.5499573f, 367.6114197f, 506.6636353f,
                                  3.1308622f, 0.5200779f,  0.5200786f,   0.5200793f,   2.3979607f};
    TM_CHECK_TENSOR_CLOSE(mel, torch::tensor(g).reshape({4, 5}), 1e-2, 1e-3);
}

static void test_mfcc() {
    const auto x = ramp16();
    const auto mfcc = tf::MFCC(tf::mfcc_option().sample_rate(16000).n_mfcc(4).mel(
            tf::mel_spectrogram_option().n_fft(8).win_length(8).hop_length(4).n_mels(4)))(x);

    fn::mfcc_option fmo;
    fmo.sample_rate = 16000;
    fmo.n_mfcc = 4;
    fmo.norm = "ortho";
    fmo.log_mels = false;
    fmo.top_db = 80.0f;
    fmo.mel.sample_rate = 16000;
    fmo.mel.n_fft = 8;
    fmo.mel.win_length = 8;
    fmo.mel.hop_length = 4;
    fmo.mel.n_mels = 4;
    TM_CHECK_TENSOR_CLOSE(mfcc, fn::mfcc(x, fmo), 1e-5, 1e-5);

    const std::vector<float> g = {-27.8944168f, -12.0743141f, -6.7132444f,  -3.3253593f,  1.3844566f,
                                  -40.5234680f, -33.3282433f, -33.3251305f, -33.3244438f, -37.6620178f,
                                  -20.1017418f, -43.7178040f, -49.0788651f, -52.4667473f, -50.5388336f,
                                  -9.1703920f,  -12.1361523f, -12.1436768f, -12.1453485f, -13.9384308f};
    TM_CHECK_TENSOR_CLOSE(mfcc, torch::tensor(g).reshape({4, 5}), 1e-2, 1e-2);
}

static void test_lfcc() {
    const auto x = ramp16();
    const auto lfcc = tf::LFCC(tf::lfcc_option().sample_rate(16000).n_filter(4).n_lfcc(4).spec(
            tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4)))(x);

    // Reference built from the golden-verified functional pieces (no single functional::lfcc exists).
    const auto sp = fn::spectrogram(
            x, fn::spectrogram_option().window(hann8()).n_fft(8).win_length(8).hop_length(4).power(2.0).return_complex(
                       false));
    const auto fb = fn::linear_fbanks(5, 0.0, 8000.0, 4, 16000); // (n_freqs=5, n_filter=4)
    auto s = torch::matmul(sp.transpose(-2, -1), fb).transpose(-2, -1);
    s = fn::amplitude_to_DB(s, fn::amplitude_to_db_option()
                                       .set_multiplier(10.0f)
                                       .set_amin(1e-10f)
                                       .set_db_multiplier(0.0f)
                                       .set_top_db(80.0f)
                                       .set_apply_top_db(true));
    const auto dct = fn::create_dct(4, 4, "ortho");
    const auto ref = torch::matmul(s.transpose(-2, -1), dct).transpose(-2, -1);
    TM_CHECK_TENSOR_CLOSE(lfcc, ref, 1e-5, 1e-5);

    const std::vector<float> g = {-2.7019000f, 4.8032293f,  10.1434975f, 13.5267248f, 23.5368443f,
                                  0.0f,        27.1399155f, 32.0837936f, 35.2117462f, 27.7589417f,
                                  -9.0872974f, -3.6556177f, -3.6290169f, -3.6230688f, -1.5788774f,
                                  0.0000026f,  -2.2019062f, -4.2209806f, -5.5101891f, -4.6593971f};
    TM_CHECK_TENSOR_CLOSE(lfcc, torch::tensor(g).reshape({4, 5}), 1e-2, 1e-2);
}

static void test_inverse_mel_scale() {
    const auto x2 = torch::arange(32, tensor_options_t().dtype(torch::kFloat));
    const auto spec2 = tf::Spectrogram(tf::spectrogram_option().n_fft(16).win_length(16).hop_length(8))(x2);
    const auto ms2 = tf::MelScale(tf::mel_scale_option().n_mels(4).sample_rate(16000).n_stft(9))(spec2);
    const auto inv = tf::InverseMelScale(
            tf::inverse_mel_scale_option().n_stft(9).n_mels(4).sample_rate(16000).driver(tf::lstsq_driver::gels))(ms2);

    TM_CHECK(inv.dim() == 2 && inv.size(0) == 9 && inv.size(1) == 5);
    TM_CHECK(inv.min().item<double>() >= 0.0); // relu non-negativity
    const std::vector<float> g = {0.0f,          0.0f,          0.0f,          0.0f,           0.0f,        8.1963634f,
                                  1257.5067139f, 4329.5068359f, 9449.5068359f, 14458.1894531f, 54.4421539f, 11.4990921f,
                                  11.4992542f,   11.4986076f,   51.3251343f,   2.6294749f,     0.6707515f,  0.6706300f,
                                  0.6711465f,    2.5926456f,    1.3575928f,    0.2156847f,     0.2156659f,  0.2157466f,
                                  1.2904152f,    0.5668065f,    0.0f,          0.0f,           0.0f,        0.4936028f,
                                  0.3778709f,    0.0f,          0.0f,          0.0f,           0.3290685f,  0.1889354f,
                                  0.0f,          0.0f,          0.0f,          0.1645342f,     0.0f,        0.0f,
                                  0.0f,          0.0f,          0.0f};
    TM_CHECK_TENSOR_CLOSE(inv, torch::tensor(g).reshape({9, 5}), 1e-1, 1e-3);
}

static void test_feature_validation() {
    auto threw = [](auto fnc) {
        bool t = false;
        try {
            fnc();
        } catch (const std::invalid_argument &) {
            t = true;
        }
        return t;
    };
    TM_CHECK(threw([] { tf::MelScale(tf::mel_scale_option().f_min(9000.0).f_max(1000.0)); }));
    TM_CHECK(threw([] { tf::MFCC(tf::mfcc_option().dct_type(3)); }));
    TM_CHECK(threw([] { tf::MFCC(tf::mfcc_option().n_mfcc(10).mel(tf::mel_spectrogram_option().n_mels(4))); }));
    TM_CHECK(threw([] { tf::LFCC(tf::lfcc_option().dct_type(3)); }));
    TM_CHECK(threw([] { tf::LFCC(tf::lfcc_option().n_lfcc(100).spec(tf::spectrogram_option().n_fft(8))); }));
    // InverseMelScale: input n_mels mismatch.
    TM_CHECK(threw(
            [] { tf::InverseMelScale(tf::inverse_mel_scale_option().n_stft(9).n_mels(4))(torch::randn({3, 5})); }));
}

static void test_feature_option_setters() {
    const auto x = ramp16();
    const auto spec = power_spec8();

    auto ao = tf::amplitude_to_db_option().stype(tf::db_stype::power).top_db(80.0);
    TM_CHECK(tf::AmplitudeToDB(ao).forward(spec).size(0) == 5);

    auto mso =
            tf::mel_scale_option().n_mels(4).sample_rate(16000).n_stft(5).f_min(0.0).f_max(8000.0).norm("").mel_scale(
                    "htk");
    const tf::MelScale msc(mso);
    TM_CHECK(msc.forward(spec).size(0) == 4 && msc.fb().size(0) == 4);

    auto imo = tf::inverse_mel_scale_option()
                       .n_stft(9)
                       .n_mels(4)
                       .sample_rate(16000)
                       .f_min(0.0)
                       .f_max(8000.0)
                       .norm("")
                       .mel_scale("htk")
                       .driver(tf::lstsq_driver::gelsd);
    const tf::InverseMelScale imsc(imo); // driver = gelsd
    const auto mel_in = torch::abs(torch::randn({4, 7})); // (n_mels=4, time=7), n_stft=9
    TM_CHECK(imsc.fb().size(0) == 4 && imsc.forward(mel_in).size(0) == 9);
    // exercise the remaining driver-name branches by actually solving with each driver
    TM_CHECK(tf::InverseMelScale(tf::inverse_mel_scale_option().n_stft(9).n_mels(4).driver(tf::lstsq_driver::gelsy))(
                     mel_in)
                     .size(0) == 9);
    TM_CHECK(tf::InverseMelScale(tf::inverse_mel_scale_option().n_stft(9).n_mels(4).driver(tf::lstsq_driver::gelss))(
                     mel_in)
                     .size(0) == 9);

    auto mlo = tf::mel_spectrogram_option()
                       .sample_rate(16000)
                       .n_fft(8)
                       .win_length(8)
                       .hop_length(4)
                       .pad(0)
                       .n_mels(4)
                       .f_min(0.0)
                       .f_max(8000.0)
                       .power(2.0)
                       .normalized(false)
                       .center(true)
                       .pad_mode("reflect")
                       .norm("")
                       .mel_scale("htk")
                       .window(hann8());
    TM_CHECK(tf::MelSpectrogram(mlo).forward(x).size(0) == 4);

    auto mfo = tf::mfcc_option()
                       .sample_rate(16000)
                       .n_mfcc(4)
                       .dct_type(2)
                       .norm("ortho")
                       .log_mels(true) // exercise the log path
                       .mel(tf::mel_spectrogram_option().n_fft(8).win_length(8).hop_length(4).n_mels(4));
    TM_CHECK(tf::MFCC(mfo).forward(x).size(0) == 4);

    auto lfo = tf::lfcc_option()
                       .sample_rate(16000)
                       .n_filter(4)
                       .n_lfcc(4)
                       .dct_type(2)
                       .f_min(0.0)
                       .f_max(8000.0)
                       .norm("ortho")
                       .log_lf(true) // exercise the log path
                       .spec(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4));
    TM_CHECK(tf::LFCC(lfo).forward(x).size(0) == 4);
}

// ======================== task21: resample & time-domain ========================
static void test_resample() {
    const auto x = torch::randn({1, 64});
    const tf::Resample r(tf::resample_option().orig_freq(16000).new_freq(8000));

    // delegation-equivalence to functional::resample (same kernel build + apply).
    TM_CHECK_TENSOR_CLOSE(r(x), fn::resample(x, 16000, 8000), 1e-6, 1e-6);
    // identity short-circuit.
    const tf::Resample rid(tf::resample_option().orig_freq(16000).new_freq(16000));
    TM_CHECK_TENSOR_CLOSE(rid(x), x, 0.0, 0.0);
    // kernel cached + reused (call twice identical).
    TM_CHECK(r.kernel().defined());
    TM_CHECK_TENSOR_CLOSE(r(x), r(x), 0.0, 0.0);
    TM_CHECK_TENSOR_CLOSE(r.forward(x), r(x), 0.0, 0.0);
}

static void test_speed() {
    const auto x = torch::randn({1, 64});
    const tf::Speed s(tf::speed_option().orig_freq(16000).factor(1.5));
    const auto out = s(x);
    const auto ref = fn::speed(x, 16000, 1.5);
    TM_CHECK_TENSOR_CLOSE(out.first, ref.first, 1e-6, 1e-6);
    TM_CHECK(!out.second.has_value()); // no lengths passed

    // lengths path: out_lengths == ceil(lengths * target / source).
    const auto lengths = torch::tensor({64}, torch::kLong);
    const auto out_l = s(x, lengths);
    TM_CHECK(out_l.second.has_value());
    const auto ref_l = fn::speed(x, 16000, 1.5, lengths);
    TM_CHECK_TENSOR_CLOSE(out_l.second.value(), ref_l.second.value(), 0.0, 0.0);

    // factor 1.0 -> identity resampler.
    const tf::Speed s1(tf::speed_option().orig_freq(16000).factor(1.0));
    TM_CHECK_TENSOR_CLOSE(s1(x).first, x, 0.0, 0.0);
}

static void test_speed_perturbation() {
    const auto x = torch::randn({1, 64});
    const tf::SpeedPerturbation sp(tf::speed_perturbation_option().orig_freq(16000).factors({0.9, 1.0, 1.1}));
    TM_CHECK(sp.size() == 3);

    // reproducible under a fixed seed.
    torch::manual_seed(123);
    const auto a = sp(x);
    torch::manual_seed(123);
    const auto b = sp.forward(x);
    TM_CHECK_TENSOR_CLOSE(a.first, b.first, 0.0, 0.0);
    TM_CHECK(a.first.dim() == 2);

    // single factor 1.0 -> deterministic identity.
    const tf::SpeedPerturbation sp1(tf::speed_perturbation_option().factors({1.0}));
    TM_CHECK_TENSOR_CLOSE(sp1(x).first, x, 0.0, 0.0);
}

static void test_pitch_shift() {
    const auto x = torch::randn({1, 1024});
    const tf::PitchShift ps(tf::pitch_shift_option().sample_rate(16000).n_steps(4).n_fft(512));
    TM_CHECK_TENSOR_CLOSE(ps(x), fn::pitch_shift(x, 16000, 4, 12, 512), 1e-5, 1e-5);

    // n_steps == 0 -> resample is identity (orig_freq == sample_rate); still shape-preserving.
    const tf::PitchShift ps0(tf::pitch_shift_option().sample_rate(16000).n_steps(0).n_fft(512));
    TM_CHECK(ps0(x).size(-1) == 1024);
}

static void test_time_stretch() {
    const auto wav = torch::randn({1, 64});
    const auto spec = fn::spectrogram(wav, fn::spectrogram_option()
                                                   .window(torch::hann_window(16))
                                                   .n_fft(16)
                                                   .win_length(16)
                                                   .hop_length(8)
                                                   .return_complex(true));
    const int n_freq = 9; // 16/2 + 1
    const tf::TimeStretch ts(tf::time_stretch_option().n_freq(n_freq).hop_length(8).fixed_rate(1.2));

    const double pi = std::acos(-1.0);
    const auto phase_advance =
            torch::linspace(0, pi * 8, n_freq, tensor_options_t().dtype(torch::kFloat)).unsqueeze(-1);
    TM_CHECK_TENSOR_CLOSE(ts(spec), fn::phase_vocoder(spec, 1.2, phase_advance), 1e-6, 1e-6);

    // overriding_rate path + accessor.
    TM_CHECK(ts.phase_advance().size(0) == n_freq);
    const tf::TimeStretch ts_no(tf::time_stretch_option().n_freq(n_freq).hop_length(8)); // no fixed_rate
    TM_CHECK_TENSOR_CLOSE(ts_no(spec, 1.5), fn::phase_vocoder(spec, 1.5, phase_advance), 1e-6, 1e-6);

    // raise when neither fixed_rate nor overriding_rate is set.
    bool threw = false;
    try {
        ts_no(spec);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

static void test_time_option_setters() {
    const auto x = torch::randn({1, 64});
    auto ro = tf::resample_option().orig_freq(16000).new_freq(8000).lowpass_filter_width(6).rolloff(0.99);
    TM_CHECK(tf::Resample(ro).forward(x).size(-1) == 32);

    auto so = tf::speed_option().orig_freq(16000).factor(2.0);
    TM_CHECK(tf::Speed(so).forward(x).first.dim() == 2);

    auto spo = tf::speed_perturbation_option().orig_freq(16000).factors({1.0});
    TM_CHECK(tf::SpeedPerturbation(spo).forward(x).first.dim() == 2);

    auto pso = tf::pitch_shift_option()
                       .sample_rate(16000)
                       .n_steps(2)
                       .bins_per_octave(12)
                       .n_fft(16)
                       .win_length(16)
                       .hop_length(4);
    TM_CHECK(tf::PitchShift(pso).forward(x).size(-1) == 64);

    auto tso = tf::time_stretch_option().n_freq(9).hop_length(8).fixed_rate(1.1);
    const auto spec = fn::spectrogram(x, fn::spectrogram_option()
                                                 .window(torch::hann_window(16))
                                                 .n_fft(16)
                                                 .win_length(16)
                                                 .hop_length(8)
                                                 .return_complex(true));
    TM_CHECK(tf::TimeStretch(tso).forward(spec).size(-2) == 9);
}

// ======================== task22: augmentation ========================
static void test_fade() {
    const auto x = torch::ones({8});
    struct {
        tf::fade_shape shape;
        std::vector<float> g;
    } cases[] = {
            {tf::fade_shape::linear, {0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.0f}},
            {tf::fade_shape::exponential, {0.0f, 0.35355338f, 1.0f, 1.0f, 1.0f, 1.0f, 0.35355338f, 0.0f}},
            {tf::fade_shape::logarithmic, {0.0f, 0.77815127f, 1.0f, 1.0f, 1.0f, 1.0f, 0.77815127f, 0.0f}},
            {tf::fade_shape::quarter_sine, {0.0f, 0.70710677f, 1.0f, 1.0f, 1.0f, 1.0f, 0.70710677f, 0.0f}},
            {tf::fade_shape::half_sine, {0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.0f}},
    };
    for (const auto &c: cases) {
        const auto out = tf::Fade(tf::fade_option().fade_in_len(3).fade_out_len(3).shape(c.shape))(x);
        TM_CHECK_TENSOR_CLOSE(out, torch::tensor(c.g), 1e-5, 1e-5);
    }
    // default (no fade) -> identity.
    TM_CHECK_TENSOR_CLOSE(tf::Fade()(x), x, 0.0, 0.0);
}

static void test_masking() {
    const auto spec2d = torch::randn({4, 6});
    const auto spec3d = torch::randn({2, 4, 6});

    // no-op: mask_param 0 -> returns input unchanged.
    TM_CHECK_TENSOR_CLOSE(tf::FrequencyMasking(tf::frequency_masking_option().freq_mask_param(0))(spec2d), spec2d, 0.0,
                          0.0);

    // seeded delegation-equivalence: FrequencyMasking on 2D -> mask_along_axis at eff_axis = 1+2-3 = 0.
    torch::manual_seed(0);
    const auto fa = tf::FrequencyMasking(tf::frequency_masking_option().freq_mask_param(2))(spec2d, 0.0);
    torch::manual_seed(0);
    const auto fb = fn::mask_along_axis(spec2d, 2, 0.0, /*axis=*/0, 1.0);
    TM_CHECK_TENSOR_CLOSE(fa, fb, 0.0, 0.0);

    // TimeMasking on 2D -> mask_along_axis at eff_axis = 2+2-3 = 1.
    torch::manual_seed(1);
    const auto ta = tf::TimeMasking(tf::time_masking_option().time_mask_param(2))(spec2d, 0.0);
    torch::manual_seed(1);
    const auto tb = fn::mask_along_axis(spec2d, 2, 0.0, /*axis=*/1, 1.0);
    TM_CHECK_TENSOR_CLOSE(ta, tb, 0.0, 0.0);

    // iid path on 3D -> mask_along_axis_iid at eff_axis = 1+3-3 = 1.
    torch::manual_seed(2);
    const auto ia =
            tf::FrequencyMasking(tf::frequency_masking_option().freq_mask_param(2).iid_masks(true))(spec3d, 0.0);
    torch::manual_seed(2);
    const auto ib = fn::mask_along_axis_iid(spec3d, 2, 0.0, /*axis=*/1, 1.0);
    TM_CHECK_TENSOR_CLOSE(ia, ib, 0.0, 0.0);

    // TimeMasking p validation.
    bool threw = false;
    try {
        tf::TimeMasking(tf::time_masking_option().time_mask_param(2).p(2.0));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

static void test_spec_augment() {
    const auto spec3d = torch::randn({2, 4, 6});
    const auto spec2d = torch::randn({4, 6});

    // no masks -> identity.
    TM_CHECK_TENSOR_CLOSE(tf::SpecAugment(tf::spec_augment_option().n_time_masks(0).n_freq_masks(0))(spec3d), spec3d,
                          0.0, 0.0);

    // seeded replication, iid path (dim>2 && iid_masks) with zero_masking.
    auto opt = tf::spec_augment_option()
                       .n_time_masks(2)
                       .time_mask_param(2)
                       .n_freq_masks(1)
                       .freq_mask_param(2)
                       .iid_masks(true)
                       .zero_masking(true);
    torch::manual_seed(7);
    const auto a = tf::SpecAugment(opt)(spec3d);
    torch::manual_seed(7);
    auto b = spec3d;
    for (int i = 0; i < 2; ++i)
        b = fn::mask_along_axis_iid(b, 2, 0.0, /*time_dim=*/2, 1.0);
    b = fn::mask_along_axis_iid(b, 2, 0.0, /*freq_dim=*/1, 1.0);
    TM_CHECK_TENSOR_CLOSE(a, b, 0.0, 0.0);

    // non-iid path (mean mask value) on a 2D input.
    auto opt2 =
            tf::spec_augment_option().n_time_masks(1).time_mask_param(2).n_freq_masks(1).freq_mask_param(2).iid_masks(
                    false);
    torch::manual_seed(9);
    const auto c = tf::SpecAugment(opt2)(spec2d);
    torch::manual_seed(9);
    const double mv = spec2d.mean().item<double>();
    auto d = fn::mask_along_axis(spec2d, 2, mv, /*time_dim=*/1, 1.0);
    d = fn::mask_along_axis(d, 2, mv, /*freq_dim=*/0, 1.0);
    TM_CHECK_TENSOR_CLOSE(c, d, 0.0, 0.0);
}

static void test_augment_option_setters() {
    const auto spec2d = torch::randn({4, 6});
    const auto spec3d = torch::randn({2, 4, 6});
    const auto x = torch::ones({8});

    auto fo = tf::frequency_masking_option().freq_mask_param(2).iid_masks(false);
    TM_CHECK(tf::FrequencyMasking(fo).forward(spec2d).sizes() == spec2d.sizes());

    auto to = tf::time_masking_option().time_mask_param(2).iid_masks(false).p(0.5);
    TM_CHECK(tf::TimeMasking(to).forward(spec2d).sizes() == spec2d.sizes());

    auto sao = tf::spec_augment_option()
                       .n_time_masks(1)
                       .time_mask_param(2)
                       .n_freq_masks(1)
                       .freq_mask_param(2)
                       .iid_masks(true)
                       .p(1.0)
                       .zero_masking(false);
    TM_CHECK(tf::SpecAugment(sao).forward(spec3d).sizes() == spec3d.sizes());

    auto fado = tf::fade_option().fade_in_len(2).fade_out_len(2).shape(tf::fade_shape::quarter_sine);
    TM_CHECK(tf::Fade(fado).forward(x).size(-1) == 8);
}

// ======================== task23: thin wrappers + Vol ========================
static void test_thin_wrappers() {
    // MuLawEncoding / MuLawDecoding.
    const auto x = torch::tensor({-0.5f, 0.0f, 0.5f, 1.0f});
    const auto enc = tf::MuLawEncoding(tf::mu_law_encoding_option().quantization_channels(256))(x);
    TM_CHECK_TENSOR_CLOSE(enc, fn::mu_law_encoding(x, 256), 0.0, 0.0);
    const auto dec = tf::MuLawDecoding()(enc);
    TM_CHECK_TENSOR_CLOSE(dec, fn::mu_law_decoding(enc, 256), 0.0, 0.0);

    // Preemphasis / Deemphasis.
    const auto wav = torch::randn({1, 16});
    TM_CHECK_TENSOR_CLOSE(tf::Preemphasis(tf::preemphasis_option().coeff(0.97))(wav), fn::preemphasis(wav, 0.97), 1e-6,
                          1e-6);
    TM_CHECK_TENSOR_CLOSE(tf::Deemphasis()(wav), fn::deemphasis(wav, 0.97), 1e-6, 1e-6);

    // ComputeDeltas (default + non-default mode).
    const auto spec = torch::randn({2, 12});
    TM_CHECK_TENSOR_CLOSE(tf::ComputeDeltas()(spec), fn::compute_deltas(spec, 5, "replicate"), 1e-6, 1e-6);
    TM_CHECK_TENSOR_CLOSE(tf::ComputeDeltas(tf::compute_deltas_option().win_length(3).mode("reflect"))(spec),
                          fn::compute_deltas(spec, 3, "reflect"), 1e-6, 1e-6);

    // SlidingWindowCmn.
    const auto sc_in = torch::randn({10, 4});
    TM_CHECK_TENSOR_CLOSE(
            tf::SlidingWindowCmn(tf::sliding_window_cmn_option().cmn_window(3).min_cmn_window(2).norm_vars(true))(
                    sc_in),
            fn::sliding_window_cmn(sc_in, 3, 2, false, true), 1e-6, 1e-6);

    // Loudness.
    const auto lwav = torch::randn({1, 16000});
    TM_CHECK_TENSOR_CLOSE(tf::Loudness(tf::loudness_option().sample_rate(16000))(lwav), fn::loudness(lwav, 16000), 1e-5,
                          1e-5);

    // Vad (default params == functional defaults).
    const auto vwav = torch::randn({1, 8000});
    TM_CHECK_TENSOR_CLOSE(tf::Vad(tf::vad_option().sample_rate(16000))(vwav), fn::vad(vwav, 16000), 1e-5, 1e-5);
}

static void test_vol() {
    const auto wav = torch::randn({1, 16}) * 0.1; // small so clamp rarely triggers

    // amplitude: wav * gain, then clamp.
    const auto amp = tf::Vol(tf::vol_option().gain(0.5).gain_type(tf::vol_gain_type::amplitude))(wav);
    TM_CHECK_TENSOR_CLOSE(amp, torch::clamp(wav * 0.5, -1.0, 1.0), 1e-6, 1e-6);

    // db: functional::gain(wav, gain_db), then clamp.
    const auto db = tf::Vol(tf::vol_option().gain(6.0).gain_type(tf::vol_gain_type::db))(wav);
    TM_CHECK_TENSOR_CLOSE(db, torch::clamp(fn::gain(wav, 6.0), -1.0, 1.0), 1e-6, 1e-6);

    // power: functional::gain(wav, 10*log10(gain)), then clamp.
    const auto pw = tf::Vol(tf::vol_option().gain(2.0).gain_type(tf::vol_gain_type::power))(wav);
    TM_CHECK_TENSOR_CLOSE(pw, torch::clamp(fn::gain(wav, 10.0 * std::log10(2.0)), -1.0, 1.0), 1e-6, 1e-6);

    // clamp actually bounds the output.
    const auto big = tf::Vol(tf::vol_option().gain(100.0).gain_type(tf::vol_gain_type::amplitude))(wav);
    TM_CHECK(big.max().item<double>() <= 1.0 && big.min().item<double>() >= -1.0);

    // negative gain raises for amplitude/power.
    bool threw = false;
    try {
        tf::Vol(tf::vol_option().gain(-1.0).gain_type(tf::vol_gain_type::power));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TM_CHECK(threw);
}

static void test_effects_option_setters() {
    const auto wav = torch::randn({1, 16});
    TM_CHECK(tf::MuLawEncoding(tf::mu_law_encoding_option().quantization_channels(128)).forward(wav).dim() == 2);
    TM_CHECK(tf::MuLawDecoding(tf::mu_law_decoding_option().quantization_channels(128))
                     .forward(tf::MuLawEncoding()(wav))
                     .dim() == 2);
    TM_CHECK(tf::Preemphasis(tf::preemphasis_option().coeff(0.9)).forward(wav).dim() == 2);
    TM_CHECK(tf::Deemphasis(tf::deemphasis_option().coeff(0.9)).forward(wav).dim() == 2);
    TM_CHECK(tf::ComputeDeltas(tf::compute_deltas_option().win_length(5).mode("replicate"))
                     .forward(torch::randn({2, 12}))
                     .dim() == 2);
    TM_CHECK(tf::SlidingWindowCmn(
                     tf::sliding_window_cmn_option().cmn_window(5).min_cmn_window(2).center(true).norm_vars(false))
                     .forward(torch::randn({6, 3}))
                     .dim() == 2);
    TM_CHECK(tf::Loudness(tf::loudness_option().sample_rate(16000)).forward(torch::randn({1, 16000})).numel() >= 1);

    // Vad: exercise every setter.
    auto vo = tf::vad_option()
                      .sample_rate(16000)
                      .trigger_level(7.0)
                      .trigger_time(0.25)
                      .search_time(1.0)
                      .allowed_gap(0.25)
                      .pre_trigger_time(0.0)
                      .boot_time(0.35)
                      .noise_up_time(0.1)
                      .noise_down_time(0.01)
                      .noise_reduction_amount(1.35)
                      .measure_freq(20.0)
                      .measure_duration(0.02)
                      .measure_smooth_time(0.4)
                      .hp_filter_freq(50.0)
                      .lp_filter_freq(6000.0)
                      .hp_lifter_freq(150.0)
                      .lp_lifter_freq(2000.0);
    TM_CHECK(tf::Vad(vo).forward(torch::randn({1, 8000})).dim() == 2);

    // Vol setters + amplitude forward.
    auto volo = tf::vol_option().gain(0.5).gain_type(tf::vol_gain_type::amplitude);
    TM_CHECK(tf::Vol(volo).forward(wav).dim() == 2);
}

// ======================== task24: convolution & noise ========================
static void test_convolution_noise() {
    const auto x = torch::randn({1, 8});
    const auto y = torch::randn({1, 3});
    for (const auto mode: {fn::convolve_mode::full, fn::convolve_mode::valid, fn::convolve_mode::same}) {
        TM_CHECK_TENSOR_CLOSE(tf::Convolve(tf::convolve_option().mode(mode))(x, y), fn::convolve(x, y, mode), 1e-5,
                              1e-5);
        TM_CHECK_TENSOR_CLOSE(tf::FFTConvolve(tf::convolve_option().mode(mode))(x, y), fn::fftconvolve(x, y, mode),
                              1e-5, 1e-5);
    }
    // default mode is 'full'; forward aliases.
    TM_CHECK_TENSOR_CLOSE(tf::Convolve().forward(x, y), fn::convolve(x, y), 1e-5, 1e-5);
    TM_CHECK_TENSOR_CLOSE(tf::FFTConvolve().forward(x, y), fn::fftconvolve(x, y), 1e-5, 1e-5);

    // AddNoise: with and without lengths.
    const auto wav = torch::randn({2, 10});
    const auto noise = torch::randn({2, 10});
    const auto snr = torch::tensor({10.0f, 5.0f});
    const tf::AddNoise add_noise;
    TM_CHECK_TENSOR_CLOSE(add_noise(wav, noise, snr), fn::add_noise(wav, noise, snr), 1e-5, 1e-5);
    const auto lengths = torch::tensor({8, 10}, torch::kLong);
    TM_CHECK_TENSOR_CLOSE(add_noise.forward(wav, noise, snr, lengths), fn::add_noise(wav, noise, snr, lengths), 1e-5,
                          1e-5);
}

// ======================== task25: beamforming ========================
static auto cdouble_opts() -> tensor_options_t { return tensor_options_t().dtype(torch::kComplexDouble); }
static auto make_u(int channels, int ref) -> tensor_t {
    auto u = torch::zeros({channels}, cdouble_opts());
    u.select(-1, ref).fill_(1);
    return u;
}
// Reference MVDR weight via the same functional ops the transform dispatches to (exact, not just
// phase-invariant, since both run the identical deterministic computation).
static auto ref_mvdr_weight(tf::mvdr_solution sol, const tensor_t &psd_s, const tensor_t &psd_n, const tensor_t &u,
                            int ref) -> tensor_t {
    if (sol == tf::mvdr_solution::ref_channel)
        return fn::mvdr_weights_souden(psd_s, psd_n, u, true, 1e-7, 1e-8);
    tensor_t stv =
            sol == tf::mvdr_solution::stv_evd ? fn::rtf_evd(psd_s) : fn::rtf_power(psd_s, psd_n, ref, 3, true, 1e-7);
    return fn::mvdr_weights_rtf(stv, psd_n, u, true, 1e-7, 1e-8);
}

static void test_psd() {
    const auto spec = torch::randn({3, 4, 5}, cdouble_opts());
    const auto mask = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));

    TM_CHECK_TENSOR_CLOSE(tf::PSD()(spec, mask), fn::psd(spec, mask, true, 1e-15), 1e-9, 1e-9);
    TM_CHECK_TENSOR_CLOSE(tf::PSD().forward(spec), fn::psd(spec, c10::nullopt, true, 1e-15), 1e-9, 1e-9);

    // multi_mask averages a (channel, freq, time) mask over channels.
    const auto mmask = torch::rand({3, 4, 5}, tensor_options_t().dtype(torch::kDouble));
    TM_CHECK_TENSOR_CLOSE(tf::PSD(tf::psd_option().multi_mask(true).normalize(true).eps(1e-15))(spec, mmask),
                          fn::psd(spec, mmask.mean(-3), true, 1e-15), 1e-9, 1e-9);
}

static void test_souden_rtf_mvdr() {
    const auto spec = torch::randn({3, 4, 5}, cdouble_opts());
    const auto mask_s = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));
    const auto mask_n = 1.0 - mask_s;
    const auto psd_s = fn::psd(spec, mask_s, true, 1e-15);
    const auto psd_n = fn::psd(spec, mask_n, true, 1e-15);

    const auto souden = tf::SoudenMVDR()(spec, psd_s, psd_n, 0);
    TM_CHECK_TENSOR_CLOSE(
            souden, fn::apply_beamforming(fn::mvdr_weights_souden(psd_s, psd_n, (int64_t) 0, true, 1e-7, 1e-8), spec),
            1e-9, 1e-9);
    TM_CHECK_TENSOR_CLOSE(tf::SoudenMVDR().forward(spec, psd_s, psd_n, 0), souden, 0.0, 0.0);

    const auto rtf = fn::rtf_evd(psd_s);
    const auto rtfmvdr = tf::RTFMVDR().forward(spec, rtf, psd_n, 0);
    TM_CHECK_TENSOR_CLOSE(rtfmvdr,
                          fn::apply_beamforming(fn::mvdr_weights_rtf(rtf, psd_n, (int64_t) 0, true, 1e-7, 1e-8), spec),
                          1e-9, 1e-9);
}

static void test_mvdr_offline() {
    const auto spec = torch::randn({3, 4, 5}, cdouble_opts());
    const auto mask_s = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));
    const auto mask_n = 1.0 - mask_s;
    const auto psd_s = fn::psd(spec, mask_s, true, 1e-15);
    const auto psd_n = fn::psd(spec, mask_n, true, 1e-15);
    const auto u = make_u(3, 0);

    for (const auto sol: {tf::mvdr_solution::ref_channel, tf::mvdr_solution::stv_evd, tf::mvdr_solution::stv_power}) {
        tf::MVDR mvdr(tf::mvdr_option().solution(sol).ref_channel(0).online(false));
        const auto out = mvdr(spec, mask_s, mask_n);
        const auto ref = fn::apply_beamforming(ref_mvdr_weight(sol, psd_s, psd_n, u, 0), spec);
        TM_CHECK_TENSOR_CLOSE(out, ref, 1e-9, 1e-9);
    }

    // mask_n defaults to 1 - mask_s.
    tf::MVDR mvdr_def(tf::mvdr_option().solution(tf::mvdr_solution::ref_channel));
    TM_CHECK_TENSOR_CLOSE(
            mvdr_def(spec, mask_s),
            fn::apply_beamforming(ref_mvdr_weight(tf::mvdr_solution::ref_channel, psd_s, psd_n, u, 0), spec), 1e-9,
            1e-9);

    // cfloat input is promoted internally and the output is cast back to cfloat.
    const auto spec_f = spec.to(torch::kComplexFloat);
    const auto out_f = tf::MVDR(tf::mvdr_option())(spec_f, mask_s);
    TM_CHECK(out_f.scalar_type() == torch::kComplexFloat);
}

static void test_mvdr_online() {
    const auto spec1 = torch::randn({3, 4, 5}, cdouble_opts());
    const auto spec2 = torch::randn({3, 4, 5}, cdouble_opts());
    const auto ms1 = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));
    const auto ms2 = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));
    const auto mn1 = 1.0 - ms1;
    const auto mn2 = 1.0 - ms2;
    const auto u = make_u(3, 0);
    const auto sol = tf::mvdr_solution::ref_channel;

    tf::MVDR mvdr(tf::mvdr_option().solution(sol).ref_channel(0).online(true));
    const auto o1 = mvdr(spec1, ms1, mn1);
    const auto o2 = mvdr(spec2, ms2, mn2);

    // frame 1: the first online call equals the offline computation.
    const auto psd_s1 = fn::psd(spec1, ms1, true, 1e-15);
    const auto psd_n1 = fn::psd(spec1, mn1, true, 1e-15);
    TM_CHECK_TENSOR_CLOSE(o1, fn::apply_beamforming(ref_mvdr_weight(sol, psd_s1, psd_n1, u, 0), spec1), 1e-9, 1e-9);

    // frame 2: replicate the recursive PSD blend.
    const auto psd_s2 = fn::psd(spec2, ms2, true, 1e-15);
    const auto psd_n2 = fn::psd(spec2, mn2, true, 1e-15);
    const auto msum_s = ms1.sum(-1);
    const auto msum_n = mn1.sum(-1);
    auto blend = [](const tensor_t &old_psd, const tensor_t &msum, const tensor_t &new_psd, const tensor_t &mask) {
        const auto denom = msum + mask.sum(-1);
        return old_psd * (msum / denom).unsqueeze(-1).unsqueeze(-1) +
               new_psd * (1.0 / denom).unsqueeze(-1).unsqueeze(-1);
    };
    const auto up_s = blend(psd_s1, msum_s, psd_s2, ms2);
    const auto up_n = blend(psd_n1, msum_n, psd_n2, mn2);
    TM_CHECK_TENSOR_CLOSE(o2, fn::apply_beamforming(ref_mvdr_weight(sol, up_s, up_n, u, 0), spec2), 1e-9, 1e-9);

    // online + multi_mask: exercises the per-channel mask averaging in the recursion.
    tf::MVDR mvdr_mm(tf::mvdr_option().solution(sol).multi_mask(true).online(true));
    const auto mms1 = torch::rand({3, 4, 5}, tensor_options_t().dtype(torch::kDouble));
    const auto mms2 = torch::rand({3, 4, 5}, tensor_options_t().dtype(torch::kDouble));
    (void) mvdr_mm(spec1, mms1, 1.0 - mms1);
    const auto mm2 = mvdr_mm(spec2, mms2, 1.0 - mms2);
    TM_CHECK(mm2.size(-2) == 4 && mm2.size(-1) == 5);
}

static void test_mvdr_validation_and_setters() {
    auto raises = [](const std::function<void()> &f) {
        try {
            f();
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    const auto mask = torch::rand({4, 5}, tensor_options_t().dtype(torch::kDouble));
    // ndim < 3.
    TM_CHECK(raises([&] { tf::MVDR()(torch::randn({4, 5}, cdouble_opts()), mask); }));
    // non-complex specgram.
    TM_CHECK(raises([&] { tf::MVDR()(torch::randn({3, 4, 5}), mask); }));

    // exercise every option setter.
    auto mo = tf::mvdr_option()
                      .ref_channel(1)
                      .solution(tf::mvdr_solution::stv_power)
                      .multi_mask(true)
                      .diag_loading(true)
                      .diag_eps(1e-7)
                      .online(false);
    const auto spec = torch::randn({3, 4, 5}, cdouble_opts());
    const auto mmask = torch::rand({3, 4, 5}, tensor_options_t().dtype(torch::kDouble));
    TM_CHECK(tf::MVDR(mo).forward(spec, mmask).size(-2) == 4);

    auto po = tf::psd_option().multi_mask(false).normalize(true).eps(1e-15);
    TM_CHECK(tf::PSD(po)(spec, mask).size(-1) == 3);
}

// ======================== 2026-10-23 audit follow-up (progress02, task04-task09) ========================
// Transform-side regressions of the functional parity fixes. Shared golden values are printed by
// unit_test/audio/functional/gen_golden.py, transform-only ones by gen_golden.py here (torchaudio 2.5.1).
static auto threw_inv(const std::function<void()> &fnc) -> bool {
    try {
        fnc();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}
static auto vec(const std::vector<float> &v) -> tensor_t { return torch::tensor(v); }
static auto mps_or_skip() -> c10::optional<torch::Device> {
    return torch::mps::is_available() ? c10::optional<torch::Device>(torch::kMPS) : c10::nullopt;
}

// task04 -- AmplitudeToDB: torchaudio top_db packing (G1-1), top_db >= 0 (new), exact float64 (G1-7).
static void test_amplitude_to_db_parity() {
    auto c = torch::arange(1, 25, torch::kFloat).reshape({2, 3, 4});
    c[1].mul_(1e-6);
    const tf::AmplitudeToDB t40(tf::amplitude_to_db_option().top_db(40.0));
    const auto out = t40(c); // (C, F, T): channel 1 is clipped to the GLOBAL max - 40 dB
    TM_CHECK_TENSOR_CLOSE(out[1], torch::full({3, 4}, -29.2081871f), 1e-4, 1e-5);
    TM_CHECK_CLOSE(out.max().item<float>(), 10.7918119, 1e-4);
    TM_CHECK(threw_inv([] { tf::AmplitudeToDB(tf::amplitude_to_db_option().top_db(-5.0)); }));
    const auto x64 = torch::tensor({1e-12, 0.3, 2.0}, torch::kDouble).reshape({1, 3});
    TM_CHECK_TENSOR_CLOSE(tf::AmplitudeToDB()(x64), 10.0 * torch::log10(torch::clamp_min(x64, 1e-10)), 1e-12, 0.0);
}

// task05/task06 -- Spectrogram frame_length + unknown method + N-D; GriffinLim length/rank/rand_init.
static void test_spectral_parity() {
    const auto x = ramp16();
    const auto fl = tf::Spectrogram(
            tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4).power(1.0).normalized(true).normalize_method(
                    "frame_length"))(x);
    TM_CHECK_TENSOR_CLOSE(fl[0], vec({1.62132013f, 5.65685368f, 11.3137074f, 16.970562f, 19.2383289f}), 1e-5, 1e-5);
    TM_CHECK(threw_inv(
            [&] { (void) tf::Spectrogram(tf::spectrogram_option().normalized(true).normalize_method("foo"))(x); }));
    const auto nd = tf::Spectrogram(tf::spectrogram_option().n_fft(8))(torch::randn({2, 3, 16}));
    TM_CHECK(nd.sizes() == torch::IntArrayRef({2, 3, 5, 5}));

    const auto mag = tf::Spectrogram(tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4))(x);
    const auto base = tf::griffinlim_option().n_fft(8).win_length(8).hop_length(4).n_iter(8).momentum(0.99);
    auto g18 = base;
    const auto out18 = tf::GriffinLim(g18.rand_init(false).length(18))(mag);
    TM_CHECK_TENSOR_CLOSE(out18.slice(0, 14), vec({11.5963821f, 13.2079496f, 14.0422859f, 14.814682f}), 1e-5, 1e-5);
    auto gr = base;
    torch::manual_seed(0); // default rand_init=true: torch.rand(complex) like torchaudio
    const auto outr = tf::GriffinLim(gr.n_iter(4).momentum(0.5))(mag);
    TM_CHECK_TENSOR_CLOSE(outr.slice(0, 0, 4), vec({0.146446496f, 0.303864747f, 2.44942617f, 3.81121278f}), 1e-5, 1e-5);
    TM_CHECK(threw_inv([&] { (void) tf::GriffinLim(tf::griffinlim_option().n_fft(8))(mag[0]); }));
}

// task07 -- MFCC/LFCC multichannel top_db (G3-1/G1-1), LFCC + MelScale use sample_rate // 2 (new).
static void test_feature_parity() {
    const double pi = std::acos(-1.0);
    const auto s = torch::sin(2.0 * pi * 0.11 * torch::arange(64, torch::kFloat));
    const auto st = torch::stack({s, 1e-4 * torch::cos(2.0 * pi * 0.23 * torch::arange(64, torch::kFloat))});
    const auto mf = tf::MFCC(
            tf::mfcc_option().n_mfcc(4).mel(tf::mel_spectrogram_option().n_fft(16).hop_length(8).n_mels(6)))(st);
    TM_CHECK(mf.sizes() == torch::IntArrayRef({2, 4, 9}));
    const auto mf4 = tf::MFCC(tf::mfcc_option().n_mfcc(4).mel(
            tf::mel_spectrogram_option().n_fft(16).hop_length(8).n_mels(6)))(torch::stack({st, st})); // (B, C, L)
    TM_CHECK(mf4.sizes() == torch::IntArrayRef({2, 2, 4, 9}));
    TM_CHECK_TENSOR_CLOSE(mf4[1], mf, 1e-4, 1e-5);
    TM_CHECK_TENSOR_CLOSE(mf[1][0].slice(0, 0, 3), vec({-169.484497f, -169.483749f, -169.483643f}), 1e-3, 1e-5);

    // LFCC at 11025 Hz: f_max = float(11025 // 2) -> identical to the 16 kHz golden (frequency-scale invariance).
    const auto lf = tf::LFCC(tf::lfcc_option().sample_rate(11025).n_filter(4).n_lfcc(4).spec(
            tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4)))(ramp16());
    const auto g16 = tf::LFCC(tf::lfcc_option().sample_rate(16000).n_filter(4).n_lfcc(4).spec(
            tf::spectrogram_option().n_fft(8).win_length(8).hop_length(4)))(ramp16());
    TM_CHECK_TENSOR_CLOSE(lf, g16, 1e-4, 1e-5);
    TM_CHECK_TENSOR_CLOSE(lf[0], vec({-2.70190024f, 4.80322838f, 10.1434956f, 13.5267315f, 23.5368462f}), 1e-4, 1e-5);
    // f_min in (sr // 2, sr / 2] is above the effective f_max at odd rates.
    TM_CHECK(threw_inv([] { tf::MelScale(tf::mel_scale_option().sample_rate(11025).f_min(5512.25)); }));
}

// task08 -- Resample: kernel computed in float64 then cached as float32 (G4-12), validation in the constructor
// (G4-2/G4-3), float32 target length (G4-6), rank/dtype checks (G4-14/G4-5); Speed/PitchShift reuse it.
static void test_resample_parity() {
    const tf::Resample r(tf::resample_option().orig_freq(48000).new_freq(16000));
    const auto gk = vec({-0.0f,
                         -8.17537966e-07f,
                         -0.000183017488f,
                         -0.000538223714f,
                         0.000244590978f,
                         0.00264129974f,
                         0.00362526067f,
                         -0.000861472392f,
                         -0.00897746626f,
                         -0.0108616697f,
                         0.00169034046f,
                         0.0213740058f,
                         0.025451893f,
                         -0.00251341448f,
                         -0.0467810147f,
                         -0.0594799332f,
                         0.00311387354f,
                         0.135345906f,
                         0.271942794f,
                         0.330000013f,
                         0.271942794f,
                         0.135345906f,
                         0.00311387354f,
                         -0.0594799332f,
                         -0.0467810147f,
                         -0.00251341448f,
                         0.025451893f,
                         0.0213740058f,
                         0.00169034046f,
                         -0.0108616697f,
                         -0.00897746626f,
                         -0.000861472392f,
                         0.00362526067f,
                         0.00264129974f,
                         0.000244590978f,
                         -0.000538223714f,
                         -0.000183017488f,
                         -8.17537966e-07f,
                         -0.0f,
                         -0.0f,
                         -0.0f});
    TM_CHECK(r.kernel().scalar_type() == torch::kFloat);
    // dtype option: the kernel is computed and cached in that dtype (torchaudio Resample(dtype=...)).
    const tf::Resample r64(tf::resample_option().orig_freq(48000).new_freq(16000).dtype(torch::kDouble));
    TM_CHECK(r64.kernel().scalar_type() == torch::kDouble);
    TM_CHECK_TENSOR_CLOSE(r64.kernel(),
                          fn::_sinc_resample_kernel(48000, 16000, 16000, fn::resample_option(),
                                                    tensor_options_t().dtype(torch::kDouble))
                                  .first,
                          0.0, 0.0);
    TM_CHECK_TENSOR_CLOSE(r.kernel().flatten(), gk, 1e-9, 0.0);
    TM_CHECK(tf::Resample(tf::resample_option().orig_freq(44100).new_freq(16000))(torch::zeros({1, 180697})).size(-1) ==
             65559);
    TM_CHECK(threw_inv([] { tf::Resample(tf::resample_option().orig_freq(0).new_freq(16000)); }));
    TM_CHECK(threw_inv([] { tf::Resample(tf::resample_option().orig_freq(16000).new_freq(0)); }));
    TM_CHECK(threw_inv(
            [] { tf::Resample(tf::resample_option().orig_freq(16000).new_freq(8000).lowpass_filter_width(0)); }));
    TM_CHECK(threw_inv([] { tf::Resample(tf::resample_option().orig_freq(16000).new_freq(8000).rolloff(0.0)); }));
    TM_CHECK(threw_inv([&] { (void) r(torch::tensor(1.0f)); }));
    TM_CHECK(threw_inv([&] { (void) r(torch::zeros({1, 30}, torch::kShort)); }));
    TM_CHECK(threw_inv([] { tf::Speed(tf::speed_option().factor(0.0)); }));
}

// Cached buffers (window, fb, DCT, filter, kernel, phase advance) follow the input dtype/device (new finding).
static void test_buffers_follow_input() {
    torch::manual_seed(0);
    const auto x = torch::randn({1, 2048});
    const auto x64 = x.to(torch::kDouble);
    const tf::MelSpectrogram mel(tf::mel_spectrogram_option().n_fft(64).n_mels(8));
    TM_CHECK(mel(x64).scalar_type() == torch::kDouble);
    TM_CHECK_TENSOR_CLOSE(mel(x64), mel(x), 1e-3, 1e-4);
    const tf::MFCC mfcc(tf::mfcc_option().n_mfcc(4).mel(tf::mel_spectrogram_option().n_fft(64).n_mels(8)));
    TM_CHECK(mfcc(x64).scalar_type() == torch::kDouble);
    const tf::LFCC lfcc(tf::lfcc_option().n_filter(8).n_lfcc(4).spec(tf::spectrogram_option().n_fft(64)));
    TM_CHECK(lfcc(x64).scalar_type() == torch::kDouble);
    const auto spec64 = tf::Spectrogram(tf::spectrogram_option().n_fft(64))(x64);
    TM_CHECK(tf::InverseMelScale(tf::inverse_mel_scale_option().n_stft(33).n_mels(8))(
                     tf::MelScale(tf::mel_scale_option().n_stft(33).n_mels(8))(spec64))
                     .scalar_type() == torch::kDouble);
    const tf::Resample r(tf::resample_option().orig_freq(48000).new_freq(16000));
    TM_CHECK_TENSOR_CLOSE(r(x64), fn::resample(x64, 48000, 16000), 1e-6, 0.0);
    TM_CHECK(tf::Speed(tf::speed_option().factor(1.5))(x64).first.scalar_type() == torch::kDouble);
    TM_CHECK(tf::PitchShift(tf::pitch_shift_option().n_steps(12))(x64).scalar_type() == torch::kDouble);
    if (const auto dev = mps_or_skip()) {
        const auto xd = x.to(*dev);
        TM_CHECK_TENSOR_CLOSE(tf::Spectrogram()(xd), tf::Spectrogram()(x), 1e-2, 1e-4);
        TM_CHECK_TENSOR_CLOSE(mfcc(xd), mfcc(x), 1e-2, 1e-4);
        TM_CHECK_TENSOR_CLOSE(r(xd), r(x), 1e-4, 1e-4);
        // (istft-based GriffinLim / PitchShift need aten ops MPS lacks: unfold_backward, angle)
    }
}

// task11 -- review round 2 (progress02 D12). Goldens: gen_golden.py "task11" section (torchaudio 2.5.1).
static void test_r2_pitch_shift_float64() {
    // P1/R1: PitchShift builds the kernel in a non-float32 input's dtype, as torchaudio's lazy init does
    // (the float32 kernel cast up was 6.4e-5 off here). torchaudio T.PitchShift(11025, 3, n_fft=400).double();
    // the ~8e-8 left is its float64-cast window buffer, the same gap torchaudio has between its own F and T.
    const auto t = torch::arange(3001, torch::kDouble);
    const auto x = torch::stack({torch::sin(t * 0.05), 0.5 * torch::cos(t * 0.031)});
    const tf::PitchShift ps(tf::pitch_shift_option().sample_rate(11025).n_steps(3).n_fft(400));
    const auto y = ps(x);
    TM_CHECK(y.scalar_type() == torch::kDouble);
    TM_CHECK_TENSOR_CLOSE(y, fn::pitch_shift(x, 11025, 3, 12, 400), 0.0, 0.0);
    TM_CHECK_CLOSE(y[0][2964].item<double>(), -0.804910047985357, 1e-6);
    TM_CHECK_CLOSE(y[0][2954].item<double>(), -0.5985449840155338, 1e-6);
    TM_CHECK_CLOSE(y[0][2977].item<double>(), -0.6640172823143292, 1e-6);
    const auto xf = x.to(torch::kFloat); // float32 input keeps the cached float32 kernel
    TM_CHECK_TENSOR_CLOSE(ps(xf), fn::pitch_shift(xf, 11025, 3, 12, 400), 0.0, 0.0);
}

static void test_r2_resample_kernel_and_equal_rates() {
    // R2: a kernel with nf > 1 pins torchaudio's dtype=None phase (int64 arange, float32 division); computing the
    // phase in float64 moves single taps by 1.9e-6. Checksums are float64 sums of torchaudio's float32 kernel.
    const auto k = tf::Resample(tf::resample_option().orig_freq(44100).new_freq(16000)).kernel().to(torch::kDouble);
    TM_CHECK(k.sizes() == torch::IntArrayRef({160, 1, 475}));
    const auto wts = torch::sin(torch::arange(k.numel(), torch::kDouble) * 0.7).reshape(k.sizes());
    TM_CHECK_CLOSE((k * wts).sum().item<double>(), -0.5908894421753382, 1e-9);
    TM_CHECK_CLOSE(k.sum().item<double>(), 160.074687641316, 1e-9);
    TM_CHECK_CLOSE(k.flatten()[38717].item<double>(), 0.1638711839914322, 0.0);

    // P3: equal rates skip the filter-argument checks (torchaudio builds no kernel then).
    const auto x = torch::randn({2, 100});
    TM_CHECK(tf::Resample(tf::resample_option().orig_freq(8000).new_freq(8000).lowpass_filter_width(0))(x).equal(x));
    TM_CHECK(tf::Resample(tf::resample_option().orig_freq(8000).new_freq(8000).rolloff(0.0))(x).equal(x));
}

// ======================== progress04 -- surface audit (task12-task19) ========================
// Goldens: gen_golden.py "progress04" sections (torchaudio 2.5.1).
static auto raises_inv(const std::function<void()> &f) -> bool {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static void test_p4_effects_transforms() {
    // Vad follows the input dtype (float64 used to throw in the float32 accessor) -- A1-02.
    const double pi = std::acos(-1.0);
    const auto t = torch::arange(8000, torch::kDouble) / 16000;
    const auto sig = torch::cat({torch::zeros({8000}, torch::kDouble), 0.5 * torch::sin(2.0 * pi * 300.0 * t)});
    const auto v = tf::Vad(tf::vad_option().sample_rate(16000))(sig);
    TM_CHECK(v.scalar_type() == torch::kDouble && v.equal(fn::vad(sig, 16000)));
    TM_CHECK(raises_inv([&] { (void) tf::Vad(tf::vad_option().sample_rate(16000).search_time(0.0))(sig); }));
    // Vol: power gain 0 raises (torchaudio: math.log10(0)) instead of returning silence -- A1-11.
    TM_CHECK(raises_inv([] {
        (void) tf::Vol(tf::vol_option().gain(0.0).gain_type(tf::vol_gain_type::power))(torch::randn({1, 16}));
    }));
    TM_CHECK(tf::Vol(tf::vol_option().gain(0.0))(torch::randn({1, 16})).abs().sum().item<double>() == 0.0);
}

static void test_p4_time_transforms() {
    // PitchShift: fractional n_steps (A2-06) and constructor validation (A2-04).
    const double steps = 0.5;
    const auto xp = torch::sin(torch::arange(3001.0) * 0.05).unsqueeze(0);
    const tf::PitchShift ps(tf::pitch_shift_option().sample_rate(1000).n_steps(steps));
    const auto yp = ps(xp);
    TM_CHECK_TENSOR_CLOSE(yp, fn::pitch_shift(xp, 1000, steps), 0.0, 0.0);
    TM_CHECK_CLOSE(yp[0][1500].item<double>(), 0.25565478205680847, 1e-6); // torchaudio F.pitch_shift(x, 1000, 0.5)
    TM_CHECK(raises_inv([] { tf::PitchShift(tf::pitch_shift_option().bins_per_octave(0)); }));
    TM_CHECK(raises_inv([] { tf::PitchShift(tf::pitch_shift_option().sample_rate(16000).n_steps(-240)); }));
    // TimeStretch: an invalid overriding_rate is reported as such (A2-09).
    const auto spec = torch::ones({9, 4}, torch::kComplexFloat);
    const tf::TimeStretch ts(tf::time_stretch_option().n_freq(9).fixed_rate(1.2));
    std::string msg;
    try {
        (void) ts(spec, 0.0);
    } catch (const std::invalid_argument &e) {
        msg = e.what();
    }
    TM_CHECK(msg.find("overriding_rate") != std::string::npos);
}

static void test_p4_specaugment_f64_seeded() {
    // mask_along_axis draws torch.rand(1) on the CPU in float32 (A2-05): float64 masks equal torchaudio's.
    const auto x = torch::arange(1, 1 + 2 * 30 * 40, torch::kDouble).reshape({2, 30, 40});
    torch::manual_seed(0);
    const auto y = tf::SpecAugment(tf::spec_augment_option()
                                           .n_time_masks(2)
                                           .time_mask_param(10)
                                           .n_freq_masks(2)
                                           .freq_mask_param(10)
                                           .iid_masks(false)
                                           .zero_masking(true))(x);
    TM_CHECK(y.sum().item<double>() == 1852636.0);
    TM_CHECK((y == 0).sum().item<int64_t>() == 744);
}

// Crash-class red tests last: before the fixes these read out of bounds or crash the process (A2-01..A2-03).
static void test_p4_rank_checks() {
    const tf::TimeStretch ts(tf::time_stretch_option().n_freq(65).fixed_rate(1.5));
    TM_CHECK(raises_inv([&] { (void) ts(torch::ones({65}, torch::kComplexFloat)); }));
    const tf::InverseMelScale ims(tf::inverse_mel_scale_option().n_stft(201).n_mels(40));
    TM_CHECK(raises_inv([&] { (void) ims(torch::rand({40})); }));
    TM_CHECK(raises_inv([&] { (void) ims(torch::rand({})); }));
    const tf::PitchShift ps(tf::pitch_shift_option().sample_rate(11025).n_steps(2));
    TM_CHECK(raises_inv([&] { (void) ps(torch::tensor(0.5f)); }));
}

int main() {
    test_spectrogram();
    test_spectral_centroid();
    test_griffinlim();
    test_griffinlim_momentum_validation();
    test_inverse_spectrogram();
    test_defaults_and_caching();
    test_option_setters();
    test_amplitude_to_db();
    test_mel_scale();
    test_mel_spectrogram();
    test_mfcc();
    test_lfcc();
    test_inverse_mel_scale();
    test_feature_validation();
    test_feature_option_setters();
    test_resample();
    test_speed();
    test_speed_perturbation();
    test_pitch_shift();
    test_time_stretch();
    test_time_option_setters();
    test_fade();
    test_masking();
    test_spec_augment();
    test_augment_option_setters();
    test_thin_wrappers();
    test_vol();
    test_effects_option_setters();
    test_convolution_noise();
    test_psd();
    test_souden_rtf_mvdr();
    test_mvdr_offline();
    test_mvdr_online();
    test_mvdr_validation_and_setters();
    TM_RUN(test_amplitude_to_db_parity);
    TM_RUN(test_spectral_parity);
    TM_RUN(test_feature_parity);
    TM_RUN(test_resample_parity);
    TM_RUN(test_buffers_follow_input);
    TM_RUN(test_r2_pitch_shift_float64);
    TM_RUN(test_r2_resample_kernel_and_equal_rates);
    TM_RUN(test_p4_effects_transforms);
    TM_RUN(test_p4_time_transforms);
    TM_RUN(test_p4_specaugment_f64_seeded);
    TM_RUN(test_p4_rank_checks);
    return tm_test::summary("audio_test_transform");
}
