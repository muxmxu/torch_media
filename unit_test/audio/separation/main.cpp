// Assertion-based tests for torchmedia::audio::separation (target: 100% line coverage of _separation.hpp).
// Point-wise goldens come from torchiva 0.1.1 on the same seeded inputs (see gen_golden.py; measured bit-identical on
// macOS arm64 / libtorch 2.5.1). Separation quality is checked against closed-form truth (known sources and
// per-frequency mixing matrices), and once end to end through the STFT transforms.
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::audio::separation;
namespace sd = torchmedia::audio::separation::detail;

struct element {
    int s, f, n;
    double re, im;
};
struct golden {
    double pow_sum, chk_re, chk_im; // sum |Y|^2 and sum Y * conj(Z), Z = randn(Y.shape, complex128) under seed 99
    element els[4]; // elements of the last batch item
};

static auto seeded_randn(const uint64_t seed, const std::vector<int64_t> &shape,
                         const torch::Dtype dtype = torch::kComplexFloat) -> torch::Tensor {
    torch::manual_seed(seed);
    return torch::randn(shape, torch::TensorOptions().dtype(dtype));
}

// tol bounds every element's error; the sums get the matching bounds (relative for |Y|^2, numel * tol for the
// checksum since sum |Z| < numel).
static void check_golden(const torch::Tensor &Y, const std::vector<int64_t> &shape, const golden &g, const double tol) {
    TM_CHECK(Y.sizes() == torch::IntArrayRef(shape));
    const auto Yd = Y.to(torch::kComplexDouble);
    TM_CHECK_CLOSE(Yd.abs().square().sum().item<double>(), g.pow_sum, tol * g.pow_sum);
    torch::manual_seed(99);
    const auto chk = (Yd * torch::randn(Y.sizes(), torch::kComplexDouble).conj()).sum();
    TM_CHECK_CLOSE(torch::real(chk).item<double>(), g.chk_re, tol * static_cast<double>(Y.numel()));
    TM_CHECK_CLOSE(torch::imag(chk).item<double>(), g.chk_im, tol * static_cast<double>(Y.numel()));
    const auto last = Yd.reshape({-1, Y.size(-3), Y.size(-2), Y.size(-1)})[-1];
    for (const auto &e: g.els) {
        const auto v = last[e.s][e.f][e.n];
        TM_CHECK_CLOSE(torch::real(v).item<double>(), e.re, tol);
        TM_CHECK_CLOSE(torch::imag(v).item<double>(), e.im, tol);
    }
}

// complex64 goldens: rounding-order changes move torchiva's own output by <= 2.3e-6, so 2e-5 leaves 10x margin.
constexpr double tol64 = 2e-5;
constexpr double tol128 = 1e-10;

// torchiva's own parameters: its diagonal load is 1e-5, ours defaults to 1e-8 (progress03 G2, D7)
static auto tiva() -> auxiva_option { return auxiva_option().load(1e-5); }

// ---------------- point-wise parity vs torchiva ----------------
static void test_torchiva_parity_auxiva() {
    const auto X0 = seeded_randn(0, {3, 17, 50});
    check_golden(auxiva_ip(X0, tiva()), {3, 17, 50},
                 {886.642933477,
                  28.0634928173,
                  -9.7586595082,
                  {{0, 0, 0, -0.63200378418, -0.673054456711},
                   {0, 8, 25, 0.0308929998428, 0.779585897923},
                   {2, 8, 16, -0.273684442043, -0.0616698339581},
                   {2, 16, 49, 0.0623509846628, -0.00689794309437}}},
                 tol64);
    check_golden(auxiva_ip(X0, tiva().model(source_model::gauss)), {3, 17, 50},
                 {885.851452042,
                  23.8244444064,
                  -4.65276062956,
                  {{0, 0, 0, -0.570748746395, -0.62900865078},
                   {0, 8, 25, -0.256158679724, 0.696917951107},
                   {2, 8, 16, -0.543007135391, -0.0878395214677},
                   {2, 16, 49, 0.189374089241, 0.0400684140623}}},
                 tol64);
    check_golden(auxiva_ip(X0, tiva().proj_back_mic(-1)), {3, 17, 50},
                 {775.349634525,
                  -9.08218537411,
                  -21.5017202153,
                  {{0, 0, 0, -0.359778374434, -0.391248911619},
                   {0, 8, 25, -0.00702634267509, 0.450802743435},
                   {2, 8, 16, -0.275919377804, -1.08933055401},
                   {2, 16, 49, -0.303919762373, 0.132613405585}}},
                 tol64);
    check_golden(auxiva_ip(X0, tiva().proj_back_mic(1)), {3, 17, 50},
                 {801.466970402,
                  -38.9403555884,
                  -1.01056257112,
                  {{0, 0, 0, -0.0161044709384, 0.0862860754132},
                   {0, 8, 25, 0.102250732481, 0.216245174408},
                   {2, 8, 16, 0.534729659557, 0.346814900637},
                   {2, 16, 49, -0.103924907744, -0.0317506939173}}},
                 tol64);
    // n_iter = 0: identity demixing, so projection back onto mic 0 keeps only mic 0 in source 0
    check_golden(auxiva_ip(X0, tiva().n_iter(0)), {3, 17, 50},
                 {894.969671853,
                  27.6052500333,
                  -6.51888815354,
                  {{0, 0, 0, -0.796081006527, -0.814833521843},
                   {0, 8, 25, 0.411730378866, 1.33621203899},
                   {2, 8, 16, 0, 0},
                   {2, 16, 49, 0, 0}}},
                 tol64);
    // torchiva AuxIVA_IP(eps=e) == eps(e).load(e): pins eps / load in the IP solve and the normalizations (the model
    // floor never fires on this input: test_model_floor). No projection back: torchiva runs it at a fixed 1e-5 load.
    check_golden(auxiva_ip(X0, tiva().model(source_model::gauss).eps(1e-3).load(1e-3).proj_back_mic(-1)), {3, 17, 50},
                 {51.1622848619,
                  -3.23702604464,
                  -5.16930174917,
                  {{0, 0, 0, -0.090844206512, -0.0998493283987},
                   {0, 8, 25, -0.0503955557942, 0.113351352513},
                   {2, 8, 16, -0.0618951059878, -0.300394833088},
                   {2, 16, 49, -0.0860370695591, 0.0515948608518}}},
                 tol64);
}

static void test_torchiva_parity_ilrma() {
    // the NMF init is drawn from the global generator right after X, exactly as torchiva does
    check_golden(ilrma(seeded_randn(1, {2, 17, 50}), tiva()), {2, 17, 50},
                 {902.615902609,
                  -22.8623735464,
                  -2.34082508107,
                  {{0, 0, 0, -0.945355534554, -0.531219184399},
                   {0, 8, 25, -0.420837312937, 0.436595886946},
                   {1, 8, 16, -0.0387893356383, 0.11232624948},
                   {1, 16, 49, 0.357435643673, 0.0851719081402}}},
                 tol64);
    check_golden(ilrma(seeded_randn(4, {2, 17, 50}, torch::kComplexDouble), tiva().n_basis(3)), {2, 17, 50},
                 {841.005229321,
                  -2.49512125741,
                  6.48347338441,
                  {{0, 0, 0, 0.506850614287, -1.16105307755},
                   {0, 8, 25, 0.172836402378, 0.153987646753},
                   {1, 8, 16, 0.010528446564, 0.547751267278},
                   {1, 16, 49, -0.0671452729021, 0.212614135423}}},
                 tol128);
    check_golden(ilrma(seeded_randn(7, {2, 2, 9, 30}), tiva()), {2, 2, 9, 30},
                 {530.626096055,
                  9.91734773807,
                  37.0973292504,
                  {{0, 0, 0, -0.5152348876, 0.0869259461761},
                   {0, 4, 15, 0.671725928783, -0.203507691622},
                   {1, 4, 10, 0.361638784409, 0.0741076171398},
                   {1, 8, 29, -0.34399831295, 0.00472643226385}}},
                 tol64);
    // ilrma forces the NMF model whatever the option says
    const auto X = seeded_randn(1, {2, 17, 50});
    torch::manual_seed(5);
    const auto a = ilrma(X, tiva().model(source_model::gauss).n_iter(3));
    torch::manual_seed(5);
    TM_CHECK(torch::equal(a, auxiva_ip(X, tiva().model(source_model::nmf).n_iter(3))));
}

static void test_torchiva_parity_overiva() {
    const auto X2 = seeded_randn(2, {3, 17, 50});
    check_golden(auxiva_ip(X2, tiva().n_src(2)), {2, 17, 50},
                 {827.453218929,
                  9.74255601272,
                  22.5467050135,
                  {{0, 0, 0, -0.731708824635, 0.370404034853},
                   {0, 8, 25, -0.151949852705, 0.102731578052},
                   {1, 8, 16, 0.293463140726, -0.125764042139},
                   {1, 16, 49, -0.15252430737, 0.0587954744697}}},
                 tol64);
    // eps / load also reach the orthogonal constraint J, recomputed before every source update
    check_golden(auxiva_ip(X2, tiva().n_src(2).eps(1e-3).load(1e-3).proj_back_mic(-1)), {2, 17, 50},
                 {516.954158592,
                  9.08917370993,
                  30.9047266391,
                  {{0, 0, 0, -0.429256737232, 0.218690499663},
                   {0, 8, 25, -0.0884497016668, 0.0638977736235},
                   {1, 8, 16, -0.583864033222, -0.54156434536},
                   {1, 16, 49, -0.334573447704, -0.178894400597}}},
                 tol64);
    // reference mic >= n_src: projection through J (torchiva mis-broadcasts this output; golden = its [0] slice, which
    // equals the per-frequency result only up to rounding at large n_freq, so keep these cases small)
    check_golden(auxiva_ip(X2, tiva().n_src(2).proj_back_mic(2)), {2, 17, 50},
                 {88.1914610389,
                  -7.13827241648,
                  5.9983139019,
                  {{0, 0, 0, 0.0895946845412, -0.0684018954635},
                   {0, 8, 25, -0.0220070332289, -0.0411257594824},
                   {1, 8, 16, 0.162222594023, -0.0136899361387},
                   {1, 16, 49, -0.00941232312471, 0.165575057268}}},
                 tol64);
    check_golden(auxiva_ip(seeded_randn(6, {2, 3, 17, 50}), tiva().n_src(2).proj_back_mic(2)), {2, 2, 17, 50},
                 {131.950964231,
                  8.98386745623,
                  -14.6708264284,
                  {{0, 0, 0, 0.0704602524638, 0.0484845750034},
                   {0, 8, 25, -0.0983125269413, -0.0130513058975},
                   {1, 8, 16, 0.00751522369683, 0.0514587201178},
                   {1, 16, 49, 0.0474156178534, 0.0777232050896}}},
                 tol64);
    // single-source extraction with the NMF model, no projection back
    check_golden(ilrma(seeded_randn(8, {3, 17, 50}), tiva().n_src(1).proj_back_mic(-1)), {1, 17, 50},
                 {17.2472623318,
                  5.08123361058,
                  -0.190588474731,
                  {{0, 0, 0, -0.138373807073, 0.10337305069},
                   {0, 8, 25, -0.138287261128, 0.0489996410906},
                   {0, 8, 16, -0.0249700006098, 0.0195383913815},
                   {0, 16, 49, -0.0515649914742, -0.0207029394805}}},
                 tol64);
}

static void test_torchiva_parity_dtype_batch() {
    const auto Y3 = auxiva_ip(seeded_randn(3, {2, 17, 50}, torch::kComplexDouble), tiva());
    TM_CHECK(Y3.scalar_type() == torch::kComplexDouble);
    check_golden(Y3, {2, 17, 50},
                 {841.398174639,
                  -45.4203965998,
                  21.6005454492,
                  {{0, 0, 0, -0.343197940191, 0.231845307382},
                   {0, 8, 25, -0.066795970058, 0.0117406880478},
                   {1, 8, 16, -0.189279380337, 0.730395929568},
                   {1, 16, 49, 0.107649305226, 0.0608065255495}}},
                 tol128);
    const auto X5 = seeded_randn(5, {2, 2, 17, 50});
    const auto Y5 = auxiva_ip(X5, tiva());
    TM_CHECK(Y5.scalar_type() == torch::kComplexFloat);
    check_golden(Y5, {2, 2, 17, 50},
                 {1596.83587961,
                  49.0024458947,
                  -4.32779526811,
                  {{0, 0, 0, 1.17320156097, 1.22570049763},
                   {0, 8, 25, -0.808724701405, 0.318538725376},
                   {1, 8, 16, 0.0949019119143, 0.036747906357},
                   {1, 16, 49, -0.61051851511, 0.224752172828}}},
                 tol64);
    // a batch item separates exactly like the same input on its own
    TM_CHECK_TENSOR_CLOSE(torch::view_as_real(Y5[1]), torch::view_as_real(auxiva_ip(X5[1], tiva())), 1e-5, 1e-4);
}

// ---------------- closed-form separation quality ----------------
// Independent super-Gaussian sources with a frame activation shared across frequencies (the IVA prior), mixed per
// frequency by known A_f. Since Y_f = W_f A_f S_f exactly, the global system W_f A_f = Y_f S_f^+ must be a scaled
// permutation with ONE permutation for all frequencies.
static auto bursty_sources(const int64_t n_src, const int64_t n_freq, const int64_t n_frames) -> torch::Tensor {
    const auto env = torch::exp(1.5 * torch::randn({n_src, 1, n_frames}, torch::kDouble));
    return (env * torch::randn({n_src, n_freq, n_frames}, torch::kComplexDouble)).to(torch::kComplexFloat);
}

// G_f = (Y_f S_f^H) (S_f S_f^H)^-1 [F, S, S]
static auto global_system(const torch::Tensor &Y, const torch::Tensor &S) -> torch::Tensor {
    const auto Sf = S.permute({1, 0, 2}); // [F, S, N]
    const auto SfH = Sf.conj().transpose(-2, -1);
    return torch::linalg_solve(torch::matmul(Sf, SfH), torch::matmul(Y.permute({1, 0, 2}), SfH), false);
}

// SIR [F, S] of each output against its dominant source; one_perm: the same permutation in every frequency
static auto sir_db(const torch::Tensor &Y, const torch::Tensor &S, bool &one_perm) -> torch::Tensor {
    const auto P = global_system(Y, S).abs().square().to(torch::kDouble); // [F, S, S]
    const auto perm = P.argmax(-1); // [F, S]
    one_perm = torch::equal(perm, perm[0].expand_as(perm)) &&
               std::get<0>(torch::sort(perm[0])).equal(torch::arange(S.size(0), torch::kLong));
    const auto target = P.gather(-1, perm.unsqueeze(-1)).squeeze(-1);
    return 10.0 * torch::log10(target / (P.sum(-1) - target));
}

// relative error [F, S] of the projected-back gains G_f[k, p(k)] against the mic-0 image A_f[0, p(k)]
static auto image_error(const torch::Tensor &Y, const torch::Tensor &S, const torch::Tensor &A) -> torch::Tensor {
    const auto G = global_system(Y, S);
    const auto perm = G.abs().argmax(-1)[0];
    const auto gain = G.gather(-1, perm.expand({G.size(0), -1}).unsqueeze(-1)).squeeze(-1);
    const auto image = A.select(1, 0).index_select(-1, perm);
    return (gain - image).abs() / image.abs();
}

static void test_separation_quality() {
    const int64_t n_src = 3, n_freq = 33, n_frames = 400;
    torch::manual_seed(42);
    const auto S = bursty_sources(n_src, n_freq, n_frames);
    const auto A = torch::randn({n_freq, n_src, n_src}, torch::kComplexFloat);
    const auto X = torch::einsum("fcs,sfn->cfn", {A, S});

    for (const auto model: {source_model::laplace, source_model::gauss, source_model::nmf}) {
        torch::manual_seed(0);
        bool one_perm = false;
        const auto sir = sir_db(auxiva_ip(X, auxiva_option().model(model).n_iter(30).proj_back_mic(-1)), S, one_perm);
        TM_CHECK(one_perm);
        TM_CHECK(sir.min().item<double>() > 15.0);
        TM_CHECK(sir.mean().item<double>() > 30.0);
    }

    // with projection back onto mic 0, output k reproduces the mic-0 image A_f[0, p(k)] s_p(k)
    TM_CHECK(image_error(auxiva_ip(X, auxiva_option().n_iter(30)), S, A).median().item<double>() < 1e-2);
}

// Compact array: far-field steering vectors of a 3-mic line array (5 cm spacing, sources at -55 / 10 / 70 deg) at
// 35..600 Hz plus a weak diffuse term, so the channels are nearly coherent. torchiva's 1e-5 diagonal load leaves
// bins mixed here, as in real rooms (progress03 G2); our default load (1e-8) separates them.
static void test_compact_array_load() {
    const int64_t n_freq = 17, n_frames = 400;
    torch::manual_seed(42);
    const auto S = bursty_sources(3, n_freq, n_frames);
    const auto f = torch::linspace(600.0 / n_freq, 600.0, n_freq, torch::kDouble).view({-1, 1, 1});
    const auto theta = torch::tensor({-55.0, 10.0, 70.0}, torch::kDouble) * (std::numbers::pi / 180.0);
    const auto tau = 0.05 / 343.0 * torch::arange(3, torch::kDouble).view({-1, 1}) * torch::sin(theta); // [mic, src]
    const auto phase = -2.0 * std::numbers::pi * f * tau; // [F, mic, src]
    const auto A =
            torch::polar(torch::ones_like(phase), phase) + 0.02 * torch::randn({n_freq, 3, 3}, torch::kComplexDouble);
    const auto X = torch::einsum("fcs,sfn->cfn", {A.to(torch::kComplexFloat), S});

    bool one_perm = true;
    const auto sir_tiva = sir_db(auxiva_ip(X, tiva().n_iter(30).proj_back_mic(-1)), S, one_perm);
    TM_CHECK(!one_perm && sir_tiva.min().item<double>() < 10.0); // the fixture is hard for torchiva's load
    for (const auto model: {source_model::laplace, source_model::gauss, source_model::nmf}) {
        torch::manual_seed(0);
        const auto Y = auxiva_ip(X, auxiva_option().model(model).n_iter(30));
        const auto sir = sir_db(Y, S, one_perm); // projection back rescales each output: SIR and perm unchanged
        TM_CHECK(one_perm);
        TM_CHECK(sir.min().item<double>() > 20.0);
        TM_CHECK(sir.mean().item<double>() > 35.0);
        // the load also reaches the projection back (a 1e-5 load there leaves gain errors up to 0.75). Not for NMF:
        // its W has cond ~1e4..1e5 here, which the normal equations square beyond complex64 (progress03 G5)
        if (model != source_model::nmf) {
            TM_CHECK(image_error(Y, S, A).max().item<double>() < 0.05);
        }
    }
    // OverIVA (sources 0 and 2 on the 3 mics, -60 dB noise floor): the load also reaches the orthogonal constraint J
    const auto pick = torch::tensor({0, 2}, torch::kLong);
    const auto S2 = S.index_select(0, pick);
    torch::manual_seed(7);
    const auto X2 = torch::einsum("fcs,sfn->cfn", {A.index_select(-1, pick).to(torch::kComplexFloat), S2}) +
                    1e-3 * torch::randn({3, n_freq, n_frames}, torch::kComplexFloat);
    const auto sir2 = sir_db(auxiva_ip(X2, auxiva_option().n_src(2).n_iter(30).proj_back_mic(-1)), S2, one_perm);
    TM_CHECK(one_perm);
    TM_CHECK(sir2.mean().item<double>() > 35.0);
}

// The projection back has no absolute floor of its own: W ~ 1 / |X| is ~1e-7 for int16-scale audio, where torchiva's
// fixed 1e-5 row-norm clamp fired and cost up to 4 dB SDR (progress03 D5). Exactly: a(W / c) = c a(W).
static void test_projection_back_scale() {
    const double c = std::ldexp(1.0, 24);
    const auto Y = seeded_randn(0, {3, 17, 50}, torch::kComplexDouble);
    const auto W = seeded_randn(1, {17, 3, 3}, torch::kComplexDouble);
    for (const int64_t mic: {0, 2}) {
        TM_CHECK_TENSOR_CLOSE(torch::view_as_real(sd::projection_back(Y, W / c, {}, mic, 1e-5) / c),
                              torch::view_as_real(sd::projection_back(Y, W, {}, mic, 1e-5)), 1e-12, 1e-12);
    }
    // OverIVA (2 sources, 3 mics): J does not depend on the scale of W; mic 2 >= n_src projects through J
    const auto Y2 = Y.slice(0, 0, 2);
    const auto W2 = W.slice(-2, 0, 2);
    const auto J = seeded_randn(2, {17, 1, 2}, torch::kComplexDouble);
    for (const int64_t mic: {0, 2}) {
        TM_CHECK_TENSOR_CLOSE(torch::view_as_real(sd::projection_back(Y2, W2 / c, J, mic, 1e-5) / c),
                              torch::view_as_real(sd::projection_back(Y2, W2, J, mic, 1e-5)), 1e-12, 1e-12);
    }
}

// eps floors every source model: an all-zero Y gets weights 1 / eps (NMF: T = V = eps, so 1 / max(K eps^2, eps))
static void test_model_floor() {
    const auto Y = torch::zeros({2, 5, 8}, torch::kComplexFloat);
    for (const auto model: {source_model::laplace, source_model::gauss, source_model::nmf}) {
        torch::Tensor T, V;
        const auto w = sd::source_weights(Y, auxiva_option().model(model).eps(0.25), T, V);
        TM_CHECK(w.size(-1) == 8 && w.size(-3) == 2);
        TM_CHECK(torch::allclose(w, torch::full_like(w, 4.0)));
    }
}

// ---------------- end-to-end usage: wav [C, L] -> STFT -> auxiva_ip -> iSTFT -> wav [S, L] ----------------
static void test_waveform_pipeline() {
    namespace tf = torchmedia::audio::transform;
    const int64_t L = 16000;
    torch::manual_seed(3);
    const auto env = torch::exp(1.5 * torch::randn({2, L / 400})).repeat_interleave(400, -1); // bursty sources
    const auto S = torch::randn({2, L}) * env;
    const auto A = torch::tensor({{1.0f, 0.6f}, {0.5f, 1.0f}});
    const auto X = torch::matmul(A, S); // instantaneous 2x2 mixture

    const auto stft = tf::Spectrogram(tf::spectrogram_option().n_fft(512).hop_length(128).return_complex(true));
    const auto istft = tf::InverseSpectrogram(tf::inverse_spectrogram_option().n_fft(512).hop_length(128));
    const auto spec = stft(X); // [2, 257, T] complex64
    const auto y = istft(auxiva_ip(spec, auxiva_option().n_iter(20)), L);
    TM_CHECK(y.sizes() == torch::IntArrayRef({2, L}));

    // each output is the mic-0 image A[0, p] s_p of a distinct source
    const auto perm = torch::matmul(y, S.t()).abs().argmax(-1);
    TM_CHECK(perm[0].item<int64_t>() != perm[1].item<int64_t>());
    for (int64_t k = 0; k < 2; ++k) {
        const auto p = perm[k].item<int64_t>();
        const auto target = A[0][p] * S[p];
        const double sdr_db =
                10.0 * std::log10((target.square().sum() / (y[k] - target).square().sum()).item<double>());
        TM_CHECK(sdr_db > 20.0);
    }
}

// ---------------- options / devices / validation ----------------
static void test_option_setters() {
    const auxiva_option d;
    TM_CHECK(d._n_iter == 10 && d._n_src == -1 && d._model == source_model::laplace && d._n_basis == 2 &&
             d._proj_back_mic == 0 && d._eps == 1e-5 && d._load == 1e-8);
    const auto o = auxiva_option()
                           .n_iter(3)
                           .n_src(2)
                           .model(source_model::gauss)
                           .n_basis(4)
                           .proj_back_mic(-1)
                           .eps(1e-6)
                           .load(1e-9);
    TM_CHECK(o._n_iter == 3 && o._n_src == 2 && o._model == source_model::gauss && o._n_basis == 4 &&
             o._proj_back_mic == -1 && o._eps == 1e-6 && o._load == 1e-9);
}

// every internal tensor follows X's device: on the meta device a stray CPU tensor would throw
static void test_device_follows_input() {
    const auto X = torch::empty({2, 3, 5, 8}, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kMeta));
    for (const auto &opt: {auxiva_option().n_iter(2), auxiva_option().n_iter(2).n_src(2).proj_back_mic(2),
                           auxiva_option().n_iter(2).model(source_model::nmf)}) {
        const auto Y = auxiva_ip(X, opt);
        TM_CHECK(Y.is_meta() && Y.scalar_type() == torch::kComplexFloat);
        TM_CHECK(Y.sizes() == torch::IntArrayRef({2, opt._n_src == -1 ? 3 : opt._n_src, 5, 8}));
    }
}

static auto throws_invalid(const torch::Tensor &X, const auxiva_option &opt) -> bool {
    try {
        (void) auxiva_ip(X, opt);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static void test_validation() {
    const auto X = torch::randn({2, 5, 8}, torch::kComplexFloat);
    TM_CHECK(throws_invalid(torch::randn({5, 8}, torch::kComplexFloat), {})); // dim < 3
    TM_CHECK(throws_invalid(torch::randn({2, 5, 8}), {})); // real input
    TM_CHECK(throws_invalid(X.to(torch::kComplexHalf), {})); // complex32: no solver kernels
    TM_CHECK(throws_invalid(X, auxiva_option().n_src(0)));
    TM_CHECK(throws_invalid(X, auxiva_option().n_src(-2)));
    TM_CHECK(throws_invalid(X, auxiva_option().n_src(3))); // n_src > n_chan
    TM_CHECK(throws_invalid(X, auxiva_option().n_iter(-1)));
    TM_CHECK(throws_invalid(X, auxiva_option().proj_back_mic(-2)));
    TM_CHECK(throws_invalid(X, auxiva_option().proj_back_mic(2))); // == n_chan
    TM_CHECK(throws_invalid(X, auxiva_option().n_basis(0)));
    TM_CHECK(throws_invalid(X, auxiva_option().eps(0.0)));
    TM_CHECK(throws_invalid(X, auxiva_option().eps(-1e-5)));
    TM_CHECK(throws_invalid(X, auxiva_option().eps(std::numeric_limits<double>::quiet_NaN())));
    TM_CHECK(throws_invalid(X, auxiva_option().eps(std::numeric_limits<double>::infinity())));
    TM_CHECK(throws_invalid(X, auxiva_option().load(0.0)));
    TM_CHECK(throws_invalid(X, auxiva_option().load(-1e-5)));
    TM_CHECK(throws_invalid(X, auxiva_option().load(std::numeric_limits<double>::quiet_NaN())));
    TM_CHECK(throws_invalid(X, auxiva_option().load(std::numeric_limits<double>::infinity())));
    // accepting boundaries
    TM_CHECK(!throws_invalid(X, auxiva_option().n_src(-1).n_iter(1).proj_back_mic(1)));
    TM_CHECK(!throws_invalid(X, auxiva_option().n_src(2).n_iter(1))); // n_src == n_chan
    TM_CHECK(!throws_invalid(X, auxiva_option().model(source_model::nmf).n_basis(1).n_iter(1)));
    bool ilrma_throws = false;
    try {
        (void) ilrma(X, auxiva_option().n_basis(0));
    } catch (const std::invalid_argument &) {
        ilrma_throws = true;
    }
    TM_CHECK(ilrma_throws);
}

static void test_no_grad() {
    const auto X = torch::randn({2, 5, 8}, torch::kComplexFloat).requires_grad_(true);
    TM_CHECK(!auxiva_ip(X, auxiva_option().n_iter(1)).requires_grad());
}

int main() {
    test_torchiva_parity_auxiva();
    test_torchiva_parity_ilrma();
    test_torchiva_parity_overiva();
    test_torchiva_parity_dtype_batch();
    test_separation_quality();
    test_compact_array_load();
    test_projection_back_scale();
    test_model_floor();
    test_waveform_pipeline();
    test_option_setters();
    test_device_follows_input();
    test_validation();
    test_no_grad();
    return tm_test::summary("audio_test_separation");
}
