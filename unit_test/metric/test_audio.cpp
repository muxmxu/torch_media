// Assertion-based tests for torchmedia::metric::audio against torchmetrics 1.9 / pystoi 0.4.1 / mir_eval 0.8.2 /
// scipy / sklearn goldens (gen_golden.py -> golden_audio.hpp). Inputs come from gen_golden.py's speech-like generator,
// mirrored here and pinned by checksums.
#include "golden_audio.hpp"
#include "metric_test_util.hpp"

using namespace torchmedia::metric::audio;
namespace D = torchmedia::metric::audio::detail;

// STOI and bss_eval_sources go through other FFT / BLAS / LAPACK kernels than numpy / scipy (develop_log/2026-10-24/
// progress01 D9): measured within 4e-16 (STOI) and 6e-14 dB (BSS); these bounds leave room for other CPUs.
constexpr double kStoiTol = 1e-12, kBssTol = 1e-9;

// gen_golden.py speech(): jittered pulse train + breath noise through two resonators, syllables with silent gaps.
static auto speech(int64_t n, int64_t fs, uint64_t seed) -> torch::Tensor {
    TORCHMEDIA_FP_CONTRACT_OFF
    const auto u = splitmix_u(n, seed);
    const int64_t period = fs / 120, syl = fs / 4;
    std::vector<double> out(n);
    double y1 = 0.0, y2 = 0.0, z1 = 0.0, z2 = 0.0;
    int64_t nxt = 0;
    for (int64_t t = 0; t < n; ++t) {
        double e = (u[t] - 0.5) * 0.02;
        if (t == nxt) {
            e = e + 1.0;
            nxt = t + period + static_cast<int64_t>(u[t] * 8.0);
        }
        const double y = 1.3 * y1 - 0.85 * y2 + e;
        y2 = y1;
        y1 = y;
        const double z = 0.4 * z1 - 0.6 * z2 + y;
        z2 = z1;
        z1 = z;
        const double r = static_cast<double>(t % syl) / static_cast<double>(syl);
        out[t] = z * ((t / syl) % 3 == 2 ? 0.0 : r * (1.0 - r) * 4.0);
    }
    return torch::tensor(out, torch::kFloat64);
}

static auto noisy(const torch::Tensor &x, uint64_t seed, double level) -> torch::Tensor {
    return x + (torch::tensor(splitmix_u(x.numel(), seed), torch::kFloat64).reshape(x.sizes()) - 0.5) * level;
}

static auto make_sources() -> std::pair<torch::Tensor, torch::Tensor> {
    auto src = torch::empty({2, 3, 4000}, torch::kFloat64), est = torch::empty_like(src);
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t s = 0; s < 3; ++s) {
            src[b][s] = speech(4000, 8000, 10 * b + s + 1);
        }
        for (int64_t s = 0; s < 3; ++s) {
            est[b][s] = noisy(0.7 * src[b][(s + b + 1) % 3] + 0.2 * src[b][(s + 2) % 3], 100 + 3 * b + s, 0.1);
        }
    }
    return {src, est};
}

static const auto SOURCES = make_sources();
static const auto T64 = SOURCES.first, P64 = SOURCES.second;
static const auto T32 = T64.to(torch::kFloat32), P32 = P64.to(torch::kFloat32);
static const auto X16 = speech(24000, 16000, 316), D16 = noisy(X16, 400, 0.4);

static auto same(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::allclose(a, b, 0.0, 0.0, true);
}

// the std::invalid_argument message of fn(), empty when it throws none
template<class Fn>
static auto invalid_message(Fn &&fn) -> std::string {
    try {
        (void) fn();
    } catch (const std::invalid_argument &e) {
        return e.what();
    }
    return {};
}

// forward() on an instance that already holds state returns this batch alone
template<class M, class... A>
static auto forward_is_batch(M &m, const M &fresh, const A &...args) -> bool {
    M alone = fresh;
    alone.update(args...);
    return same(m(args...), alone.compute());
}

static void test_inputs() {
    CHECK_CHECKSUM(T64, in_src);
    CHECK_CHECKSUM(P64, in_est);
}

static void test_snr_family() {
    using namespace golden;
    const golden::entry *g[2][2][4] = {{{&snr_f32, &sisdr_f32, &sasdr_f32, &sasdr_nosi_f32},
                                        {&snr_f32_zm, &sisdr_f32_zm, &sasdr_f32_zm, &sasdr_nosi_f32_zm}},
                                       {{&snr_f64, &sisdr_f64, &sasdr_f64, &sasdr_nosi_f64},
                                        {&snr_f64_zm, &sisdr_f64_zm, &sasdr_f64_zm, &sasdr_nosi_f64_zm}}};
    const golden::entry *cls[2][5] = {{&cls_snr_f32, &cls_sisdr_f32, &cls_sisnr_f32, &cls_sasdr_f32, &cls_sdr_f32},
                                      {&cls_snr_f64, &cls_sisdr_f64, &cls_sisnr_f64, &cls_sasdr_f64, &cls_sdr_f64}};
    for (const int f64: {0, 1}) {
        const auto &P = f64 ? P64 : P32;
        const auto &T = f64 ? T64 : T32;
        for (const int zm: {0, 1}) {
            using SA = source_aggregated_signal_distortion_ratio_option;
            check_golden(signal_noise_ratio(P, T, zm), *g[f64][zm][0], "snr", __LINE__);
            check_golden(scale_invariant_signal_distortion_ratio(P, T, zm), *g[f64][zm][1], "sisdr", __LINE__);
            check_golden(source_aggregated_signal_distortion_ratio(P, T, SA().zero_mean(zm)), *g[f64][zm][2], "sasdr",
                         __LINE__);
            check_golden(source_aggregated_signal_distortion_ratio(P, T, SA().zero_mean(zm).scale_invariant(false)),
                         *g[f64][zm][3], "sasdr_nosi", __LINE__);
        }
        check_golden(scale_invariant_signal_noise_ratio(P, T), f64 ? sisnr_f64 : sisnr_f32, "sisnr", __LINE__);
        check_golden(signal_distortion_ratio(P.slice(1, 0, 2).slice(2, 0, 1500), T.slice(1, 0, 2).slice(2, 0, 1500)),
                     f64 ? sdr_f64 : sdr_f32, "sdr", __LINE__);
        // classes: two [3, 1500] updates; float64 inputs keep a float64 state (torchmetrics rounds to float32)
        const auto p = [&](int64_t b) { return P[b].slice(1, 0, 1500); };
        const auto t = [&](int64_t b) { return T[b].slice(1, 0, 1500); };
        SignalNoiseRatio snr;
        ScaleInvariantSignalDistortionRatio sisdr;
        ScaleInvariantSignalNoiseRatio sisnr;
        SourceAggregatedSignalDistortionRatio sasdr;
        SignalDistortionRatio sdr(signal_distortion_ratio_option().filter_length(128));
        snr.update(p(0), t(0));
        sisdr.update(p(0), t(0));
        sisnr.update(p(0), t(0));
        sasdr.update(p(0), t(0));
        sdr.update(p(0), t(0));
        TM_CHECK(forward_is_batch(snr, SignalNoiseRatio(), p(1), t(1)));
        TM_CHECK(forward_is_batch(sisdr, ScaleInvariantSignalDistortionRatio(), p(1), t(1)));
        TM_CHECK(forward_is_batch(sisnr, ScaleInvariantSignalNoiseRatio(), p(1), t(1)));
        TM_CHECK(forward_is_batch(sasdr, SourceAggregatedSignalDistortionRatio(), p(1), t(1)));
        TM_CHECK(forward_is_batch(sdr, SignalDistortionRatio(sdr.options()), p(1), t(1)));
        const torch::Tensor outs[] = {snr.compute(), sisdr.compute(), sisnr.compute(), sasdr.compute(), sdr.compute()};
        for (int k = 0; k < 5; ++k) {
            if (f64) {
                TM_CHECK(outs[k].scalar_type() == torch::kFloat64);
                check_golden(outs[k], *cls[1][k], "cls_f64", __LINE__, 1e-5, false);
            } else {
                check_golden(outs[k], *cls[0][k], "cls_f32", __LINE__);
            }
        }
        // one-shot over both batches equals the accumulated state (up to the order of the float sums)
        TM_CHECK_TENSOR_CLOSE(snr.compute(),
                              signal_noise_ratio(torch::cat({p(0), p(1)}), torch::cat({t(0), t(1)})).mean(), 1e-5,
                              1e-6);
        snr.reset();
        TM_CHECK(std::isnan(snr.compute().item<double>()));
    }
    const auto z = torch::zeros({4000}, torch::kFloat64);
    CHECK_GOLDEN(signal_noise_ratio(P64[0][0], z), snr_silent);
    CHECK_GOLDEN(signal_noise_ratio(P32[0].slice(1, 0, 400).to(torch::kFloat16),
                                    T32[0].slice(1, 0, 400).to(torch::kFloat16)),
                 snr_f16);
    CHECK_GOLDEN(
            signal_noise_ratio(P32[0][0].slice(0, 0, 400).to(torch::kFloat16), torch::zeros({400}, torch::kFloat16)),
            snr_silent_f16); // float16's eps
    const auto bf = scale_invariant_signal_distortion_ratio(P32[0].to(torch::kBFloat16), T32[0].to(torch::kBFloat16));
    TM_CHECK(bf.scalar_type() == torch::kBFloat16);
    TM_CHECK_TENSOR_CLOSE(bf.to(torch::kFloat32), scale_invariant_signal_distortion_ratio(P32[0], T32[0]), 0.5, 0.0);
    CHECK_GOLDEN(scale_invariant_signal_distortion_ratio(P64[0][0], z), sisdr_silent);
    // complex
    const auto cr = T64.reshape({2, 3, 40, 50, 2}), cp = P64.reshape({2, 3, 40, 50, 2});
    CHECK_GOLDEN(complex_scale_invariant_signal_noise_ratio(torch::view_as_complex(cp), torch::view_as_complex(cr)),
                 csisnr);
    CHECK_GOLDEN(complex_scale_invariant_signal_noise_ratio(cp.to(torch::kFloat32), cr.to(torch::kFloat32), true),
                 csisnr_real_zm_f32);
    CHECK_GOLDEN(complex_scale_invariant_signal_noise_ratio(cp[0][0], cr[0][0]), csisnr_3d);
    ComplexScaleInvariantSignalNoiseRatio cm;
    cm.update(cp[0].to(torch::kFloat32), cr[0].to(torch::kFloat32));
    TM_CHECK(forward_is_batch(cm, ComplexScaleInvariantSignalNoiseRatio(), cp[1].to(torch::kFloat32),
                              cr[1].to(torch::kFloat32)));
    CHECK_GOLDEN(cm.compute(), cls_csisnr);
    // checks (std::invalid_argument before any computation / state change)
    const auto i64 = T64.to(torch::kInt64);
    TM_CHECK(throws_invalid([] { return signal_noise_ratio(P64, T64.slice(2, 1)); }));
    TM_CHECK(throws_invalid([&] { return signal_noise_ratio(i64, i64); }));
    TM_CHECK(throws_invalid([&] { return scale_invariant_signal_distortion_ratio(P64, i64); }));
    TM_CHECK(throws_invalid([] { return scale_invariant_signal_noise_ratio(P64, T64[0]); }));
    TM_CHECK(throws_invalid([] { return source_aggregated_signal_distortion_ratio(P64[0][0], T64[0][0]); }));
    TM_CHECK(throws_invalid([] { return source_aggregated_signal_distortion_ratio(P64, T64[0]); }));
    TM_CHECK(throws_invalid([&] { return complex_scale_invariant_signal_noise_ratio(cp.slice(-1, 0, 1), cr); }));
    TM_CHECK(throws_invalid([&] { return complex_scale_invariant_signal_noise_ratio(cp, cr[0][0][0]); }));
    TM_CHECK(throws_invalid([&] { return complex_scale_invariant_signal_noise_ratio(cp, cr.slice(2, 1)); }));
    for (const auto &x: {torch::zeros({3, 2}), torch::zeros({2, 4, 5, 3})}) { // same shapes, not (..., F, T, 2)
        TM_CHECK(throws_invalid([&] { return complex_scale_invariant_signal_noise_ratio(x, x); }));
    }
    SignalNoiseRatio s1;
    ScaleInvariantSignalDistortionRatio s2;
    SourceAggregatedSignalDistortionRatio s3;
    ComplexScaleInvariantSignalNoiseRatio s4;
    TM_CHECK(throws_invalid([&] { return (s1.update(P64, T64[0]), 0); }));
    TM_CHECK(throws_invalid([&] { return (s2.update(P64, T64[0]), 0); }));
    TM_CHECK(throws_invalid([&] { return (s3.update(P64[0][0], T64[0][0]), 0); }));
    TM_CHECK(throws_invalid([&] { return (s4.update(cp, cr[0][0][0]), 0); }));
    TM_CHECK(std::isnan(s1.compute().item<double>()) && std::isnan(s4.compute().item<double>())); // untouched
    if (torch::mps::is_available()) { // dtype / device follow the input
        const auto v = scale_invariant_signal_distortion_ratio(P32.to(torch::kMPS), T32.to(torch::kMPS));
        TM_CHECK(v.device().is_mps());
        TM_CHECK_TENSOR_CLOSE(v.cpu(), G(golden::sisdr_f32).to(torch::kFloat32), 1e-4, 0.0);
        ScaleInvariantSignalDistortionRatio m;
        m.update(P32.to(torch::kMPS), T32.to(torch::kMPS));
        TM_CHECK(m.compute().device().is_mps());
    }
}

static void test_sdr() {
    using O = signal_distortion_ratio_option;
    const auto p = P64.slice(1, 0, 2).slice(2, 0, 1500), t = T64.slice(1, 0, 2).slice(2, 0, 1500);
    CHECK_GOLDEN(signal_distortion_ratio(p, t, O().zero_mean(true).load_diag(1e-3)), sdr_zm_ld);
    CHECK_GOLDEN(signal_distortion_ratio(p.to(torch::kFloat32), t.to(torch::kFloat32), O().use_cg_iter(10)), sdr_cg);
    CHECK_GOLDEN(signal_distortion_ratio(P64[0][0].slice(0, 0, 600), T64[0][0].slice(0, 0, 600),
                                         O().use_cg_iter(0).filter_length(32)),
                 sdr_cg0);
    CHECK_GOLDEN(signal_distortion_ratio(P64[0].slice(1, 0, 1000), T64[0].slice(1, 0, 1000), O().filter_length(48)),
                 sdr_fl);
    CHECK_GOLDEN(signal_distortion_ratio(P64[0].slice(1, 0, 100), T64[0].slice(1, 0, 100)), sdr_short);
    const auto z = torch::zeros({4000}, torch::kFloat64);
    CHECK_GOLDEN(signal_distortion_ratio(P64[0][0], z, O().load_diag(1e-5)), sdr_silent);
    TM_CHECK(throws_any([&] { return signal_distortion_ratio(P64[0][0], z); })); // singular: torch's error
    TM_CHECK(signal_distortion_ratio(p.to(torch::kFloat16), t.to(torch::kFloat16)).scalar_type() == torch::kFloat32);
    TM_CHECK(throws_invalid([&] { return signal_distortion_ratio(p, t, O().filter_length(0)); }));
    TM_CHECK(throws_invalid([&] { return signal_distortion_ratio(p, t, O().use_cg_iter(-1)); }));
    TM_CHECK(throws_invalid([&] { return signal_distortion_ratio(p, t.slice(2, 1)); }));
    // no time axis (torchmetrics: log2(-1), a math domain error)
    TM_CHECK(throws_invalid([] { return signal_distortion_ratio(torch::zeros({2, 0}), torch::zeros({2, 0})); }));
    TM_CHECK(throws_invalid([] { return signal_distortion_ratio(torch::zeros({}), torch::zeros({})); }));
    TM_CHECK(throws_invalid([] { return SignalDistortionRatio(O().filter_length(-3)); }));
    SignalDistortionRatio m;
    TM_CHECK(throws_invalid([&] { return (m.update(p, t.slice(2, 1)), 0); }));
    TM_CHECK(throws_invalid([&] { return (m.update(torch::zeros({2, 0}), torch::zeros({2, 0})), 0); }));
}

static auto mse(const torch::Tensor &p, const torch::Tensor &t) -> torch::Tensor { return (p - t).pow(2).mean({-1}); }

static void test_pit() {
    using namespace golden;
    using O = permutation_invariant_training_option;
    const auto sisdr = [](const torch::Tensor &p, const torch::Tensor &t) {
        return scale_invariant_signal_distortion_ratio(p, t);
    };
    const auto sasdr = [](const torch::Tensor &p, const torch::Tensor &t) {
        return source_aggregated_signal_distortion_ratio(p, t);
    };
    const golden::entry *g[2][2][3][2] = {{{{&pit_sw2_f32, &pit_sw2_f32_perm},
                                            {&pit_min2_f32, &pit_min2_f32_perm},
                                            {&pit_pw2_f32, &pit_pw2_f32_perm}},
                                           {{&pit_sw3_f32, &pit_sw3_f32_perm},
                                            {&pit_min3_f32, &pit_min3_f32_perm},
                                            {&pit_pw3_f32, &pit_pw3_f32_perm}}},
                                          {{{&pit_sw2_f64, &pit_sw2_f64_perm},
                                            {&pit_min2_f64, &pit_min2_f64_perm},
                                            {&pit_pw2_f64, &pit_pw2_f64_perm}},
                                           {{&pit_sw3_f64, &pit_sw3_f64_perm},
                                            {&pit_min3_f64, &pit_min3_f64_perm},
                                            {&pit_pw3_f64, &pit_pw3_f64_perm}}}};
    const O opts[3] = {O().metric_func(sisdr), O().metric_func(mse).eval_func(eval_func_mode::min),
                       O().metric_func(sasdr).mode(pit_mode::permutation_wise)};
    for (const int f64: {0, 1}) {
        for (const int spk: {2, 3}) {
            const auto p = (f64 ? P64 : P32).slice(1, 0, spk), t = (f64 ? T64 : T32).slice(1, 0, spk);
            for (int k = 0; k < 3; ++k) {
                const auto [m, perm] = permutation_invariant_training(p, t, opts[k]);
                check_golden(m, *g[f64][spk - 2][k][0], "pit", __LINE__);
                check_golden(perm, *g[f64][spk - 2][k][1], "pit_perm", __LINE__);
            }
        }
    }
    const auto perm = permutation_invariant_training(P32, T32, opts[1]).second;
    CHECK_GOLDEN(torchmedia::metric::audio::pit_permutate(P32.slice(2, 0, 5), perm), pit_permutate);
    TM_CHECK(same(torchmedia::metric::audio::pit_permutate(P32, perm)[0][0], P32[0][perm[0][0].item<int64_t>()]));
    PermutationInvariantTraining m(O().metric_func(
            [](const torch::Tensor &p, const torch::Tensor &t) { return scale_invariant_signal_noise_ratio(p, t); }));
    m.update(P32.slice(0, 0, 1), T32.slice(0, 0, 1));
    TM_CHECK(forward_is_batch(m, PermutationInvariantTraining(m.options()), P32.slice(0, 1), T32.slice(0, 1)));
    CHECK_GOLDEN(m.compute(), cls_pit);
    // ties (target speakers 0 and 1 identical): torch.max's first index / scipy's tie-breaking
    auto tt = T64.clone();
    tt.select(1, 1).copy_(tt.select(1, 0));
    const auto snr = [](const torch::Tensor &p, const torch::Tensor &t) { return signal_noise_ratio(p, t); };
    for (const int spk: {2, 3}) {
        const auto [v, pm] =
                permutation_invariant_training(P64.slice(1, 0, spk), tt.slice(1, 0, spk), O().metric_func(snr));
        check_golden(v, spk == 2 ? pit_tie2 : pit_tie3, "pit_tie", __LINE__);
        check_golden(pm, spk == 2 ? pit_tie2_perm : pit_tie3_perm, "pit_tie_perm", __LINE__);
    }
    // scipy linear_sum_assignment on integer matrices with many ties
    const golden::entry *lsa[12][3] = {
            {&lsa_in0, &lsa_max0, &lsa_min0}, {&lsa_in1, &lsa_max1, &lsa_min1},    {&lsa_in2, &lsa_max2, &lsa_min2},
            {&lsa_in3, &lsa_max3, &lsa_min3}, {&lsa_in4, &lsa_max4, &lsa_min4},    {&lsa_in5, &lsa_max5, &lsa_min5},
            {&lsa_in6, &lsa_max6, &lsa_min6}, {&lsa_in7, &lsa_max7, &lsa_min7},    {&lsa_in8, &lsa_max8, &lsa_min8},
            {&lsa_in9, &lsa_max9, &lsa_min9}, {&lsa_in10, &lsa_max10, &lsa_min10}, {&lsa_in11, &lsa_max11, &lsa_min11}};
    for (const auto &e: lsa) {
        const auto n = e[0]->shape[0];
        for (const int mx: {1, 0}) {
            const auto got = D::linear_sum_assignment(e[0]->data, n, mx);
            check_golden(torch::tensor(got), *e[mx ? 1 : 2], "lsa", __LINE__);
        }
    }
    // checks
    TM_CHECK(throws_invalid([] { return permutation_invariant_training(P64, T64, O()).first; }));
    TM_CHECK(throws_invalid([&] { return permutation_invariant_training(P64, T64.slice(1, 0, 2), opts[0]).first; }));
    TM_CHECK(throws_invalid([&] { return permutation_invariant_training(P64[0][0], T64[0][0], opts[0]).first; }));
    TM_CHECK(throws_invalid([] { return PermutationInvariantTraining(O()); }));
    TM_CHECK(throws_invalid([&] { return (m.update(P64, T64.slice(1, 0, 2)), 0); }));
    const auto nan_metric = [](const torch::Tensor &p, const torch::Tensor &) {
        return torch::full({p.size(0)}, std::nan(""), p.options());
    };
    const auto inf_metric = [](const torch::Tensor &p, const torch::Tensor &) {
        return torch::full({p.size(0)}, INFINITY, p.options());
    };
    TM_CHECK(throws_invalid(
            [&] { return permutation_invariant_training(P64, T64, O().metric_func(nan_metric)).first; }));
    TM_CHECK(throws_invalid([&] {
        return permutation_invariant_training(P64, T64, O().metric_func(inf_metric).eval_func(eval_func_mode::min))
                .first;
    }));
    TM_CHECK(D::linear_sum_assignment({1.0, -INFINITY, 0.0, 2.0}, 2, false).empty());
}

static void test_stoi() {
    using namespace golden;
    const int64_t rates[] = {8000, 10000, 16000, 22050, 44100};
    const golden::entry *g[5][3] = {{&in_stoi_8000, &stoi_8000, &estoi_8000},
                                    {&in_stoi_10000, &stoi_10000, &estoi_10000},
                                    {&in_stoi_16000, &stoi_16000, &estoi_16000},
                                    {&in_stoi_22050, &stoi_22050, &estoi_22050},
                                    {&in_stoi_44100, &stoi_44100, &estoi_44100}};
    for (int i = 0; i < 5; ++i) {
        const auto fs = rates[i];
        const auto x = speech(fs * 3 / 2, fs, 300 + fs / 1000);
        const auto clean = torch::stack({x, x}), deg = torch::stack({noisy(x, 400, 0.4), noisy(x, 401, 1.5)});
        check_checksum(deg, *g[i][0], "in_stoi", __LINE__);
        check_golden(short_time_objective_intelligibility(deg, clean, fs), *g[i][1], "stoi", __LINE__, kStoiTol);
        check_golden(short_time_objective_intelligibility(deg, clean, fs, true), *g[i][2], "estoi", __LINE__, kStoiTol);
    }
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(D16.to(torch::kFloat32), X16.to(torch::kFloat32), 16000),
                     stoi_f32, kStoiTol);
    const auto preds = torch::stack({torch::stack({D16, X16}), torch::stack({X16, D16})});
    const auto target = torch::stack({torch::stack({X16, X16}), torch::stack({D16, D16})});
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(preds, target, 16000), stoi_batch, kStoiTol);
    const auto z = torch::zeros_like(D16);
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(D16, z, 16000), stoi_silent_ref, kStoiTol);
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(z, X16, 16000), stoi_silent_pred, kStoiTol);
    CHECK_GOLDEN(short_time_objective_intelligibility(D16.slice(0, 0, 6000).unsqueeze(0),
                                                      X16.slice(0, 0, 6000).unsqueeze(0), 16000),
                 stoi_few_frames);
    // extended STOI of a constant band envelope: 0, the expected value of pystoi's random dither (progress01 D7)
    TM_CHECK(short_time_objective_intelligibility(z, X16, 16000, true).item<double>() == 0.0);
    TM_CHECK(short_time_objective_intelligibility(D16, z, 16000, true).item<double>() == 0.0);
    // a 0.5 s zero-filled dropout over speech in the prediction (golden: pystoi without the dither, D7)
    const auto xd = speech(48000, 16000, 330);
    auto yd = noisy(xd, 410, 0.1);
    yd.slice(0, 12000, 20000).zero_();
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(yd, xd, 16000), stoi_dropout, kStoiTol);
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(yd, xd, 16000, true), estoi_dropout, kStoiTol);
    // exactly 30 STFT frames after silent-frame removal: one segment (29 would give 1e-5)
    const auto x30 = speech(4200, 10000, 331);
    CHECK_GOLDEN_TOL(short_time_objective_intelligibility(noisy(x30, 411, 0.1), x30, 10000), stoi_30_frames, kStoiTol);
    // a non-contiguous (transposed) batch equals its contiguous copy
    const auto pt = torch::stack({D16, X16}, 1).t(), tt = torch::stack({X16, X16}, 1).t();
    TM_CHECK(!pt.is_contiguous() &&
             same(short_time_objective_intelligibility(pt, tt, 16000),
                  short_time_objective_intelligibility(pt.contiguous(), tt.contiguous(), 16000)));
    // the shortest valid input: 257 samples at 10 kHz (one frame) gives 1e-5; 256 is invalid like pystoi
    TM_CHECK(short_time_objective_intelligibility(D16.slice(0, 0, 257), X16.slice(0, 0, 257), 10000).item<double>() ==
             1e-5);
    TM_CHECK(throws_invalid(
            [] { return short_time_objective_intelligibility(D16.slice(0, 0, 256), X16.slice(0, 0, 256), 10000); }));
    // 409 samples at 16 kHz resample to 256 (invalid, pystoi raises), 410 to 257 (one frame: 1e-5)
    TM_CHECK(throws_invalid(
            [] { return short_time_objective_intelligibility(D16.slice(0, 0, 409), X16.slice(0, 0, 409), 16000); }));
    TM_CHECK(short_time_objective_intelligibility(D16.slice(0, 0, 410), X16.slice(0, 0, 410), 16000).item<double>() ==
             1e-5);
    TM_CHECK(throws_invalid([] { return short_time_objective_intelligibility(D16, X16.slice(0, 1), 16000); }));
    TM_CHECK(invalid_message([] { return short_time_objective_intelligibility(D16, X16, 0); }).find("fs > 0") !=
             std::string::npos);
    TM_CHECK(throws_invalid(
            [] { return short_time_objective_intelligibility(torch::zeros({3, 0}), torch::zeros({3, 0}), 16000); }));
    TM_CHECK(throws_invalid([] {
        return short_time_objective_intelligibility(torch::view_as_complex(torch::zeros({5000, 2})),
                                                    torch::view_as_complex(torch::zeros({5000, 2})), 16000);
    }));
    TM_CHECK(short_time_objective_intelligibility(torch::zeros({0, 5000}), torch::zeros({0, 5000}), 16000).sizes() ==
             torch::IntArrayRef({0}));
    using O = short_time_objective_intelligibility_option;
    ShortTimeObjectiveIntelligibility m(O().fs(16000));
    m.update(D16, X16);
    TM_CHECK(forward_is_batch(m, ShortTimeObjectiveIntelligibility(m.options()), X16, D16));
    check_golden(m.compute(), cls_stoi, "cls_stoi", __LINE__, 1e-6, false); // float64 here, float32 in torchmetrics
    ShortTimeObjectiveIntelligibility ext(O().fs(16000).extended(true));
    ext.update(D16, X16);
    TM_CHECK(ext.compute().item<double>() ==
             short_time_objective_intelligibility(D16, X16, 16000, true).item<double>());
    ext.update(yd, xd); // the dropout utterance keeps the dataset mean finite
    TM_CHECK_TENSOR_CLOSE(ext.compute(),
                          (G(golden::estoi_dropout) + short_time_objective_intelligibility(D16, X16, 16000, true)) / 2,
                          kStoiTol, 0.0);
    TM_CHECK(throws_invalid([] { return ShortTimeObjectiveIntelligibility(O()); }));
    TM_CHECK(throws_invalid([&] { return (m.update(D16, X16.slice(0, 1)), 0); }));
}

static void test_bss_eval() {
    using namespace golden;
    const auto check = [](const bss_eval_sources_result &r, const golden::entry &sdr, const golden::entry &sir,
                          const golden::entry &sar, const golden::entry &perm, int line) {
        check_golden(r.sdr, sdr, "bss_sdr", line, kBssTol);
        check_golden(r.sir, sir, "bss_sir", line, kBssTol);
        check_golden(r.sar, sar, "bss_sar", line, kBssTol);
        check_golden(r.perm, perm, "bss_perm", line);
    };
    check(bss_eval_sources(T64[0], P64[0]), bss3_sdr, bss3_sir, bss3_sar, bss3_perm, __LINE__);
    check(bss_eval_sources(T32[0].slice(0, 0, 2), P32[0].slice(0, 0, 2)), bss2_sdr, bss2_sir, bss2_sar, bss2_perm,
          __LINE__);
    const auto one = bss_eval_sources(T64[0].slice(0, 0, 1), P64[0].slice(0, 0, 1));
    CHECK_GOLDEN_TOL(one.sdr, bss1_sdr, kBssTol);
    CHECK_GOLDEN_TOL(one.sar, bss1_sar, kBssTol);
    CHECK_GOLDEN(one.perm, bss1_perm);
    TM_CHECK(std::isinf(one.sir.item<double>())); // no interference (mir_eval: inf or 280-300 dB of rounding noise)
    check(bss_eval_sources(T64[1], P64[1], false), bss_noperm_sdr, bss_noperm_sir, bss_noperm_sar, bss_noperm_perm,
          __LINE__);
    const auto r1 = bss_eval_sources(T64[1][0].slice(0, 0, 700), P64[1][0].slice(0, 0, 700));
    CHECK_GOLDEN_TOL(r1.sdr, bss_1d_sdr, kBssTol);
    CHECK_GOLDEN_TOL(r1.sar, bss_1d_sar, kBssTol);
    CHECK_GOLDEN(r1.perm, bss_1d_perm);
    TM_CHECK(std::isinf(r1.sir.item<double>()));
    // identical impulse references: exactly singular Gram matrix, least squares like mir_eval; SIR is rounding noise
    auto dup = torch::zeros({2, 600}, torch::kFloat64);
    dup.select(1, 0).fill_(1.0);
    const auto r = bss_eval_sources(dup, P64[0].slice(0, 0, 2).slice(1, 0, 600), false);
    CHECK_GOLDEN_TOL(r.sdr, bss_dup_sdr, kBssTol);
    CHECK_GOLDEN_TOL(r.sar, bss_dup_sar, kBssTol);
    CHECK_GOLDEN(r.perm, bss_dup_perm);
    TM_CHECK((r.sir > 250).all().item<bool>());
    // identical estimates: both permutations have the same mean SIR, the first (identity) is kept like np.argmax
    // (mir_eval's own choice there follows its run-to-run rounding noise)
    const auto tie = bss_eval_sources(T64[0].slice(0, 0, 2), torch::stack({P64[0][0], P64[0][0]}));
    TM_CHECK(same(tie.perm, torch::tensor({0, 1}, torch::kInt64)));
    const auto e = bss_eval_sources(torch::zeros({0, 0}), torch::zeros({0, 0}));
    TM_CHECK(e.sdr.numel() == 0 && e.perm.numel() == 0 && e.perm.scalar_type() == torch::kInt64);
    TM_CHECK(throws_invalid([] { return bss_eval_sources(T64[0], P64[0].slice(1, 1)).sdr; }));
    TM_CHECK(throws_invalid([] { return bss_eval_sources(T64, P64).sdr; }));
    TM_CHECK(throws_invalid(
            [] { return bss_eval_sources(torch::zeros({2, 100}), P64[0].slice(0, 0, 2).slice(1, 0, 100)).sdr; }));
    TM_CHECK(throws_invalid([] { return bss_eval_sources(T64[0], torch::zeros_like(T64[0])).sdr; }));
    auto one_silent = T64[0].clone(); // a single silent source among non-silent ones, either side
    one_silent[1].zero_();
    TM_CHECK(throws_invalid([&] { return bss_eval_sources(one_silent, P64[0]).sdr; }));
    TM_CHECK(throws_invalid([&] { return bss_eval_sources(T64[0], one_silent).sdr; }));
    TM_CHECK(throws_invalid([] { return bss_eval_sources(torch::ones({101, 8}), torch::ones({101, 8})).sdr; }));
    TM_CHECK(throws_invalid([] {
        const auto c = torch::view_as_complex(torch::ones({2, 8, 2}));
        return bss_eval_sources(c, c).sdr;
    }));
}

static void test_speaker_verification() {
    using namespace golden;
    const auto tar = torch::floor(torch::tensor(splitmix_u(300, 501), torch::kFloat64) * 60) / 40 + 0.5;
    const auto non = torch::floor(torch::tensor(splitmix_u(1000, 502), torch::kFloat64) * 60) / 40;
    CHECK_CHECKSUM(tar, in_tar);
    CHECK_CHECKSUM(non, in_non);
    CHECK_GOLDEN(equal_error_rate(tar, non), eer);
    CHECK_GOLDEN(equal_error_rate(tar.slice(0, 0, 100), non.slice(0, 0, 400)), eer_head);
    using O = min_dcf_option;
    CHECK_GOLDEN(min_dcf(tar, non), mindcf_1_1_1);
    CHECK_GOLDEN(min_dcf(tar, non, O().p_target(0.05)), mindcf_5_1_1);
    CHECK_GOLDEN(min_dcf(tar, non, O().c_miss(10)), mindcf_1_10_1);
    CHECK_GOLDEN(min_dcf(tar, non, O().p_target(0.5).c_fa(3)), mindcf_50_1_3);
    const auto t = [](std::vector<double> v) { return torch::tensor(v, torch::kFloat64); };
    CHECK_GOLDEN(equal_error_rate(t({0.9, 0.8, 0.3}), t({0.7, 0.2, 0.1})), eer_exact);
    CHECK_GOLDEN(min_dcf(t({0.9, 0.8, 0.3}), t({0.7, 0.2, 0.1})), mindcf_exact);
    CHECK_GOLDEN(equal_error_rate(t({3, 2}).to(torch::kFloat32), t({1, 0}).to(torch::kInt64)), eer_perfect);
    CHECK_GOLDEN(min_dcf(t({3, 2}), t({1, 0})), mindcf_perfect);
    CHECK_GOLDEN(equal_error_rate(t({0, 1}), t({2, 3})), eer_inverted);
    CHECK_GOLDEN(min_dcf(t({0, 1}), t({2, 3})), mindcf_inverted);
    CHECK_GOLDEN(equal_error_rate(t({0.5, 0.5}), t({0.5, 0.5})), eer_tied);
    CHECK_GOLDEN(min_dcf(t({0.5, 0.5}), t({0.5, 0.5})), mindcf_tied);
    // classes: two updates equal one shot
    EqualErrorRate e;
    MinDCF d(O().p_target(0.05));
    for (int64_t k = 0; k < 2; ++k) {
        e.update(tar.slice(0, 150 * k, 150 * (k + 1)), non.slice(0, 500 * k, 500 * (k + 1)));
        d.update(tar.slice(0, 150 * k, 150 * (k + 1)), non.slice(0, 500 * k, 500 * (k + 1)));
    }
    CHECK_GOLDEN(e.compute(), eer);
    CHECK_GOLDEN(d.compute(), mindcf_5_1_1);
    TM_CHECK(forward_is_batch(e, EqualErrorRate(), tar.slice(0, 0, 100), non.slice(0, 0, 400)));
    TM_CHECK(forward_is_batch(d, MinDCF(d.options()), tar.slice(0, 0, 100), non.slice(0, 0, 400)));
    e.reset();
    TM_CHECK(throws_any([&] { return e.compute(); })); // torch.cat of no batch
    // checks
    TM_CHECK(throws_invalid([&] { return equal_error_rate(tar, non.slice(0, 0, 0)); }));
    TM_CHECK(throws_invalid([&] { return equal_error_rate(tar.view({-1, 2}), non); }));
    TM_CHECK(throws_invalid([&] { return equal_error_rate(t({0.1, NAN}), non); }));
    TM_CHECK(throws_invalid([&] { return min_dcf(tar, t({INFINITY})); }));
    TM_CHECK(throws_invalid([&] { return min_dcf(tar, non, O().p_target(1.0)); }));
    TM_CHECK(throws_invalid([&] { return min_dcf(tar, non, O().c_miss(0)); }));
    TM_CHECK(throws_invalid([&] { return min_dcf(tar, non, O().c_fa(-1)); }));
    TM_CHECK(throws_invalid([] { return MinDCF(O().p_target(0)); }));
    TM_CHECK(throws_invalid([&] { return (e.update(tar, t({})), 0); }));
    TM_CHECK(throws_invalid([&] { return (d.update(t({NAN}), non), 0); }));
}

int main() {
    TM_RUN(test_inputs);
    TM_RUN(test_snr_family);
    TM_RUN(test_sdr);
    TM_RUN(test_pit);
    TM_RUN(test_stoi);
    TM_RUN(test_bss_eval);
    TM_RUN(test_speaker_verification);
    return tm_test::summary("metric_test_audio");
}
