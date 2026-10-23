// Assertion-based tests for torchmedia::metric::image against torchmetrics 1.9 goldens (gen_golden.py ->
// golden_image.hpp; inputs from the shared splitmix64 generator, checked by checksum).
#include "golden_image.hpp"
#include "metric_test_util.hpp"

using namespace torchmedia::metric;
using namespace torchmedia::metric::image;
using RM = reduction_mode;

static const auto A64 = image_pair({2, 3, 32, 32}, 101);
static const auto p64 = A64.first, t64 = A64.second;
static const auto p32 = p64.to(torch::kFloat32), t32 = t64.to(torch::kFloat32);
static const auto B32 = image_pair({1, 1, 180, 180}, 202);
static const auto C64 = image_pair({2, 2, 48, 48}, 303);
static const auto cp = C64.first.to(torch::kFloat32), ct = C64.second.to(torch::kFloat32);
static const auto D32 = image_pair({1, 2, 12, 14, 16}, 404);
static const auto d3p = D32.first.to(torch::kFloat32), d3t = D32.second.to(torch::kFloat32);

static auto first(const torch::Tensor &x) -> torch::Tensor { return x.slice(0, 0, 1); }
static auto second(const torch::Tensor &x) -> torch::Tensor { return x.slice(0, 1); }

// forward() on an instance that already holds state returns this batch alone, and one 2-image update accumulates like
// two 1-image updates (the per-image counters).
template<class M, class O, class... T>
static void check_batches(const O &opt, const char *name, int line, const T &...x) {
    M acc(opt), one(opt), two(opt);
    acc.update(first(x)...);
    one.update(second(x)...);
    const auto batch = acc(second(x)...);
    tm_test::record(torch::equal(batch, one.compute()), name, __FILE__, line, "forward() on a stateful instance");
    two.update(x...);
    tm_test::record(torch::allclose(two.compute(), acc.compute(), 1e-5, 0), name, __FILE__, line, "2-image update");
}
#define CHECK_BATCHES(M, opt, ...) check_batches<M>((opt), #M, __LINE__, __VA_ARGS__)

static void test_inputs() {
    CHECK_CHECKSUM(torch::stack({p64, t64}), in_a);
    CHECK_CHECKSUM(torch::stack({B32.first, B32.second}).to(torch::kFloat32), in_b);
    CHECK_CHECKSUM(torch::stack({C64.first, C64.second}), in_c);
    CHECK_CHECKSUM(torch::stack({D32.first, D32.second}).to(torch::kFloat32), in_d);
}

static void test_psnr() {
    const auto dr1 = peak_signal_noise_ratio_option().data_range({1.0});
    CHECK_GOLDEN(peak_signal_noise_ratio(p32, t32, dr1), psnr_f32);
    CHECK_GOLDEN(peak_signal_noise_ratio(p64, t64, peak_signal_noise_ratio_option().data_range({0.2, 0.9}).base(2.0)),
                 psnr_f64_clamp_base2);
    const auto dims = peak_signal_noise_ratio_option(dr1).dim({1, 2, 3});
    CHECK_GOLDEN(peak_signal_noise_ratio(p32, t32, peak_signal_noise_ratio_option(dims).reduction(RM::none)),
                 psnr_dim_none);
    CHECK_GOLDEN(peak_signal_noise_ratio(p32, t32, peak_signal_noise_ratio_option(dims).reduction(RM::sum)),
                 psnr_dim_sum);
    CHECK_GOLDEN(peak_signal_noise_ratio(p32, t32, peak_signal_noise_ratio_option(dr1).dim({2})), psnr_dim2_mean);
    const auto u8p = torch::floor(p64 * 255).to(torch::kUInt8), u8t = torch::floor(t64 * 255).to(torch::kUInt8);
    CHECK_GOLDEN(peak_signal_noise_ratio(u8p, u8t, peak_signal_noise_ratio_option().data_range({255})), psnr_u8);
    {
        PeakSignalNoiseRatio m(dr1);
        const auto batch = m(first(p32), first(t32));
        TM_CHECK(torch::equal(batch, peak_signal_noise_ratio(first(p32), first(t32), dr1)));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), psnr_cls);
        // float64 inputs: a float64 state (torchmetrics' float32 state rounds it)
        PeakSignalNoiseRatio d(dr1);
        d.update(first(p64), first(t64));
        d.update(second(p64), second(t64));
        const auto v = d.compute();
        TM_CHECK(v.scalar_type() == torch::kFloat64);
        TM_CHECK_CLOSE(v.item<double>(), peak_signal_noise_ratio(p64, t64, dr1).item<double>(), 1e-12);
        d.reset();
        TM_CHECK(std::isnan(d.compute().item<double>()));
    }
    {
        PeakSignalNoiseRatio m(
                peak_signal_noise_ratio_option().data_range({0.0, 1.0}).dim({1, 2, 3}).reduction(RM::none));
        m.update(first(p32), first(t32));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), psnr_cls_dim_none);
    }
    TM_CHECK(throws_invalid([] { return peak_signal_noise_ratio(p32, t32, peak_signal_noise_ratio_option()); }));
    TM_CHECK(throws_invalid(
            [] { return peak_signal_noise_ratio(p32, t32, peak_signal_noise_ratio_option().data_range({0, 1, 2})); }));
    TM_CHECK(throws_invalid([] { return PeakSignalNoiseRatio(peak_signal_noise_ratio_option()); }));
}

static void test_ssim() {
    using O = structural_similarity_index_measure_option;
    CHECK_GOLDEN(structural_similarity_index_measure(p32, t32), ssim_f32);
    CHECK_GOLDEN(structural_similarity_index_measure(p64, t64), ssim_f64);
    CHECK_GOLDEN(structural_similarity_index_measure(
                         p32, t32, O().gaussian_kernel(false).kernel_size({7}).data_range({1.0}).reduction(RM::none)),
                 ssim_uniform_none);
    CHECK_GOLDEN(structural_similarity_index_measure(
                         p32, t32, O().sigma({1.0, 2.0}).data_range({0.1, 0.8}).k1(0.02).k2(0.05).reduction(RM::sum)),
                 ssim_sigma_sum);
    CHECK_GOLDEN(structural_similarity_index_measure(p32, t64), ssim_target_f64);
    // data_range None: the wider of the two ranges (here the target's)
    CHECK_GOLDEN(structural_similarity_index_measure(p32 * 0.5, t32), ssim_dr_target);
    {
        const auto [s, img] = structural_similarity_index_measure_with_image(p32, t32, O().return_full_image(true));
        CHECK_GOLDEN(s, ssim_full_val);
        CHECK_GOLDEN(img, ssim_full_img);
        // the plain form returns the same similarity
        TM_CHECK(torch::equal(structural_similarity_index_measure(p32, t32, O().return_full_image(true)), s));
    }
    {
        const auto [s, cs] = structural_similarity_index_measure_with_image(
                p64, t64, O().data_range({1.0}).return_contrast_sensitivity(true).reduction(RM::none));
        CHECK_GOLDEN(s, ssim_cs_val);
        CHECK_GOLDEN(cs, ssim_cs);
    }
    CHECK_GOLDEN(structural_similarity_index_measure(d3p, d3t, O().sigma({0.8}).kernel_size({7})), ssim3d_gauss);
    CHECK_GOLDEN(structural_similarity_index_measure(
                         d3p, d3t, O().gaussian_kernel(false).kernel_size({3}).data_range({1.0}).reduction(RM::none)),
                 ssim3d_uniform);
    {
        const auto [s, cs] = structural_similarity_index_measure_with_image(
                d3p, d3t, O().sigma({0.8}).kernel_size({7, 5, 3}).return_contrast_sensitivity(true));
        CHECK_GOLDEN(s, ssim3d_cs_val);
        CHECK_GOLDEN(cs, ssim3d_cs);
    }
    {
        StructuralSimilarityIndexMeasure m(O().return_full_image(true));
        const auto batch = m.forward(first(p32), first(t32));
        TM_CHECK(torch::equal(batch, structural_similarity_index_measure(first(p32), first(t32))));
        m.update(second(p32), second(t32));
        const auto [s, img] = m.compute_with_image();
        CHECK_GOLDEN(s, ssim_cls);
        CHECK_GOLDEN(img, ssim_cls_img);
        TM_CHECK(torch::equal(m.compute(), s));
    }
    for (const auto red: {RM::sum, RM::none}) {
        StructuralSimilarityIndexMeasure m(O().reduction(red).data_range({1.0}));
        m.update(first(p32), first(t32));
        m.update(second(p32), second(t32));
        if (red == RM::sum) {
            CHECK_GOLDEN(m.compute(), ssim_cls_sum);
        } else {
            CHECK_GOLDEN(m.compute(), ssim_cls_none);
        }
    }
    // checks (torchmetrics ValueError / RuntimeError)
    const auto bad = [](const O &o, const torch::Tensor &p, const torch::Tensor &t) {
        return throws_invalid([&] { return structural_similarity_index_measure(p, t, o); });
    };
    TM_CHECK(bad(O(), p32, t32.slice(3, 1)));
    TM_CHECK(bad(O(), p32[0], t32[0]));
    TM_CHECK(bad(O().kernel_size({11, 11, 11}), p32, t32));
    TM_CHECK(bad(O().sigma({1.5, 1.5, 1.5}), p32, t32));
    TM_CHECK(bad(O().return_full_image(true).return_contrast_sensitivity(true), p32, t32));
    TM_CHECK(bad(O().kernel_size({4}), p32, t32));
    TM_CHECK(bad(O().kernel_size({-1}), p32, t32));
    TM_CHECK(bad(O().sigma({0.0}), p32, t32));
    TM_CHECK(bad(O().data_range({0, 1, 2}), p32, t32));
    TM_CHECK(throws_invalid([] { return structural_similarity_index_measure_with_image(p32, t32, O()); }));
    TM_CHECK(throws_invalid(
            [] { return structural_similarity_index_measure_with_image(p32, t32[0], O().return_full_image(true)); }));
    StructuralSimilarityIndexMeasure m;
    TM_CHECK(throws_invalid([&] {
        m.update(p32, t32[0]);
        return 0;
    }));
    TM_CHECK(throws_invalid([&] { return m.compute_with_image(); }));
}

static void test_ms_ssim() {
    using O = multiscale_structural_similarity_index_measure_option;
    const auto [bp, bt] = B32;
    const auto b32p = bp.to(torch::kFloat32), b32t = bt.to(torch::kFloat32);
    CHECK_GOLDEN(multiscale_structural_similarity_index_measure(b32p, b32t), msssim_f32);
    CHECK_GOLDEN(multiscale_structural_similarity_index_measure(b32p, b32t, O().data_range({1.0}).reduction(RM::none)),
                 msssim_f32_dr);
    const auto three = O().betas({0.5, 0.3, 0.2}).kernel_size({5});
    CHECK_GOLDEN(multiscale_structural_similarity_index_measure(p32, t32, O(three).normalize(normalize_mode::simple)),
                 msssim_simple3);
    CHECK_GOLDEN(multiscale_structural_similarity_index_measure(p32, t32, O(three).k1(0.02).k2(0.05)), msssim_k);
    CHECK_GOLDEN(multiscale_structural_similarity_index_measure(p64, t64,
                                                                O(three).normalize(normalize_mode::none)
                                                                        .data_range({1.0})
                                                                        .gaussian_kernel(false)
                                                                        .reduction(RM::sum)),
                 msssim_none3_f64);
    {
        MultiScaleStructuralSimilarityIndexMeasure m(three);
        m(first(p32), first(t32));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), msssim_cls);
    }
    {
        MultiScaleStructuralSimilarityIndexMeasure m(O(three).reduction(RM::none));
        m.update(first(p32), first(t32));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), msssim_cls_none);
        MultiScaleStructuralSimilarityIndexMeasure s(O(three).reduction(RM::sum));
        s.update(first(p32), first(t32));
        s.update(second(p32), second(t32));
        TM_CHECK(torch::allclose(s.compute(), m.compute().sum(), 1e-6, 0));
    }
    // 5-D input (avg_pool3d between the scales)
    const auto v3 =
            multiscale_structural_similarity_index_measure(torch::cat({d3p, d3p}, 2), torch::cat({d3t, d3t}, 2),
                                                           O().betas({0.6, 0.4}).kernel_size({5, 5, 3}).sigma({0.5}));
    TM_CHECK(v3.dim() == 0 && std::isfinite(v3.item<double>()));
    const auto bad = [](const O &o, const torch::Tensor &p, const torch::Tensor &t) {
        return throws_invalid([&] { return multiscale_structural_similarity_index_measure(p, t, o); });
    };
    TM_CHECK(bad(O().kernel_size({}), b32p, b32t));
    TM_CHECK(bad(O().kernel_size({11, 11, 11, 11}), b32p, b32t));
    TM_CHECK(bad(O().betas({}), b32p, b32t));
    TM_CHECK(bad(O(), b32p, b32t.slice(2, 1)));
    TM_CHECK(bad(O(), p32.slice(2, 0, 16), t32.slice(2, 0, 16))); // smaller than 2^5
    TM_CHECK(bad(O(), p32, t32)); // height 32 / 16 <= 10
    TM_CHECK(bad(O(three).kernel_size({5, 31}), p32, t32)); // width
    TM_CHECK(bad(O(three).kernel_size({4}), p32, t32)); // then _ssim_update's checks
    TM_CHECK(throws_invalid([] { return MultiScaleStructuralSimilarityIndexMeasure(O().betas({})); }));
    MultiScaleStructuralSimilarityIndexMeasure m;
    TM_CHECK(throws_invalid([&] {
        m.update(p32, t32);
        return 0;
    }));
}

static void test_uqi() {
    using O = universal_image_quality_index_option;
    CHECK_GOLDEN(universal_image_quality_index(p32, t32), uqi_f32);
    CHECK_GOLDEN(universal_image_quality_index(p64, t64), uqi_f64);
    const auto k57 = O().kernel_size({5, 7}).sigma({1.0, 2.0});
    CHECK_GOLDEN(universal_image_quality_index(p32, t32, O(k57).reduction(RM::none)), uqi_k57_none);
    CHECK_GOLDEN(universal_image_quality_index(p32, t32, O().reduction(RM::sum)), uqi_sum);
    for (const auto red: {RM::elementwise_mean, RM::sum, RM::none}) {
        UniversalImageQualityIndex m(O(k57).reduction(red));
        m(first(p32), first(t32));
        m.update(second(p32), second(t32));
        const auto v = m.compute();
        if (red == RM::elementwise_mean) {
            CHECK_GOLDEN(v, uqi_cls_elementwise_mean);
        } else if (red == RM::sum) {
            CHECK_GOLDEN(v, uqi_cls_sum);
        } else {
            CHECK_GOLDEN(v, uqi_cls_none);
        }
    }
    // float16: torch 2.5.1 has no CPU Half reflection_pad2d (torchmetrics fails the same way)
    TM_CHECK(
            throws_any([] { return universal_image_quality_index(p32.to(torch::kFloat16), t32.to(torch::kFloat16)); }));
    const auto bad = [](const O &o, const torch::Tensor &p, const torch::Tensor &t) {
        return throws_invalid([&] { return universal_image_quality_index(p, t, o); });
    };
    TM_CHECK(bad(O(), p32, t64));
    TM_CHECK(bad(O(), p32, t32.slice(3, 1)));
    TM_CHECK(bad(O(), p32[0], t32[0]));
    TM_CHECK(bad(O().kernel_size({4, 5}), p32, t32));
    TM_CHECK(bad(O().kernel_size({5, -1}), p32, t32));
    TM_CHECK(bad(O().sigma({1.0, 0.0}), p32, t32));
    UniversalImageQualityIndex m;
    TM_CHECK(throws_invalid([&] {
        m.update(p32, t64);
        return 0;
    }));
}

static void test_tv_sam_ergas() {
    CHECK_GOLDEN(total_variation(p32), tv_sum);
    CHECK_GOLDEN(total_variation(p32, RM::elementwise_mean), tv_mean);
    CHECK_GOLDEN(total_variation(p32, RM::none), tv_none);
    CHECK_GOLDEN(total_variation(torch::floor(p64 * 255).to(torch::kUInt8)), tv_u8);
    CHECK_GOLDEN(total_variation(p64, RM::elementwise_mean), tv_f64_mean);
    for (const auto red: {RM::sum, RM::elementwise_mean, RM::none}) {
        TotalVariation m(total_variation_option().reduction(red));
        m(first(p32));
        m.update(second(p32));
        const auto v = m.compute();
        if (red == RM::sum) {
            CHECK_GOLDEN(v, tv_cls_sum);
        } else if (red == RM::elementwise_mean) {
            CHECK_GOLDEN(v, tv_cls_mean);
        } else {
            CHECK_GOLDEN(v, tv_cls_none);
        }
    }
    TM_CHECK(throws_invalid([] { return total_variation(p32[0]); }));
    TotalVariation tv;
    TM_CHECK(throws_invalid([&] {
        tv.update(p32[0]);
        return 0;
    }));

    CHECK_GOLDEN(spectral_angle_mapper(p32, t32), sam_elementwise_mean);
    CHECK_GOLDEN(spectral_angle_mapper(p32, t32, RM::sum), sam_sum);
    CHECK_GOLDEN(spectral_angle_mapper(p32, t32, RM::none), sam_none);
    CHECK_GOLDEN(spectral_angle_mapper(p64, t64), sam_f64);
    for (const auto red: {RM::elementwise_mean, RM::sum, RM::none}) {
        SpectralAngleMapper m(spectral_angle_mapper_option().reduction(red));
        m(first(p32), first(t32));
        m.update(second(p32), second(t32));
        const auto v = m.compute();
        if (red == RM::elementwise_mean) {
            CHECK_GOLDEN(v, sam_cls_elementwise_mean);
        } else if (red == RM::sum) {
            CHECK_GOLDEN(v, sam_cls_sum);
        } else {
            CHECK_GOLDEN(v, sam_cls_none);
        }
    }
    TM_CHECK(throws_invalid([] { return spectral_angle_mapper(p32, t64); }));
    TM_CHECK(throws_invalid([] { return spectral_angle_mapper(p32, t32.slice(2, 1)); }));
    TM_CHECK(throws_invalid([] { return spectral_angle_mapper(p32[0], t32[0]); }));
    TM_CHECK(throws_invalid([] { return spectral_angle_mapper(p32.slice(1, 0, 1), t32.slice(1, 0, 1)); }));
    SpectralAngleMapper sam;
    TM_CHECK(throws_invalid([&] {
        sam.update(p32, t64);
        return 0;
    }));

    using EO = error_relative_global_dimensionless_synthesis_option;
    CHECK_GOLDEN(error_relative_global_dimensionless_synthesis(p32, t32), ergas_f32);
    CHECK_GOLDEN(error_relative_global_dimensionless_synthesis(p32, t32, EO().ratio(2.5).reduction(RM::none)),
                 ergas_r25_none);
    CHECK_GOLDEN(error_relative_global_dimensionless_synthesis(p64, t64, EO().reduction(RM::sum)), ergas_f64_sum);
    ErrorRelativeGlobalDimensionlessSynthesis m(EO().ratio(2.5));
    m(first(p32), first(t32));
    m.update(second(p32), second(t32));
    CHECK_GOLDEN(m.compute(), ergas_cls);
    TM_CHECK(throws_invalid([] { return error_relative_global_dimensionless_synthesis(p32, t64); }));
    TM_CHECK(throws_invalid([] { return error_relative_global_dimensionless_synthesis(p32[0], t32[0]); }));
    TM_CHECK(throws_invalid([&] {
        m.update(p32[0], t32[0]);
        return 0;
    }));
}

static void test_sliding_window() {
    CHECK_GOLDEN(relative_average_spectral_error(p32, t32), rase_f32);
    CHECK_GOLDEN(relative_average_spectral_error(p64, t64, 5), rase_w5_f64);
    CHECK_GOLDEN(relative_average_spectral_error(p32, t32, 3), rase_w3);
    {
        RelativeAverageSpectralError m(sliding_window_option().window_size(5));
        m(first(p32), first(t32));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), rase_cls);
    }
    const auto [r, map] = root_mean_squared_error_using_sliding_window_with_map(p32, t32);
    CHECK_GOLDEN(r, rmse_sw_f32);
    CHECK_GOLDEN(map, rmse_sw_map);
    CHECK_GOLDEN(root_mean_squared_error_using_sliding_window(p64, t64, 3), rmse_sw_w3_f64);
    CHECK_GOLDEN(root_mean_squared_error_using_sliding_window(p32, t32, 1), rmse_sw_w1);
    {
        RootMeanSquaredErrorUsingSlidingWindow m(sliding_window_option().window_size(5));
        m(first(p32), first(t32));
        m.update(second(p32), second(t32));
        CHECK_GOLDEN(m.compute(), rmse_sw_cls);
        TM_CHECK(throws_invalid([&] {
            m.update(p32, t64);
            return 0;
        }));
    }
    TM_CHECK(throws_invalid([] { return root_mean_squared_error_using_sliding_window(p32, t32, 0); }));
    TM_CHECK(throws_invalid([] { return root_mean_squared_error_using_sliding_window(p32, t64); }));
    TM_CHECK(throws_invalid([] { return root_mean_squared_error_using_sliding_window(p32, t32.slice(2, 1)); }));
    TM_CHECK(throws_invalid([] { return root_mean_squared_error_using_sliding_window(p32[0], t32[0]); }));
    TM_CHECK(throws_invalid([] { return root_mean_squared_error_using_sliding_window(p32, t32, 63); }));
    TM_CHECK(throws_invalid([] { return relative_average_spectral_error(p32, t32, 64); }));
    TM_CHECK(throws_invalid([] { return RelativeAverageSpectralError(sliding_window_option().window_size(0)); }));
    TM_CHECK(throws_invalid(
            [] { return RootMeanSquaredErrorUsingSlidingWindow(sliding_window_option().window_size(-1)); }));
}

static void test_scc() {
    using O = spatial_correlation_coefficient_option;
    CHECK_GOLDEN(spatial_correlation_coefficient(p32, t32), scc_f32);
    CHECK_GOLDEN(spatial_correlation_coefficient(p32, t32, O().window_size(5).reduction(RM::none)), scc_none_w5);
    CHECK_GOLDEN(spatial_correlation_coefficient(p32.select(1, 0), t32.select(1, 0)), scc_3d);
    CHECK_GOLDEN(spatial_correlation_coefficient(
                         p64, t64, O().hp_filter(torch::tensor({{1.0, -1.0}, {-1.0, 1.0}})).window_size(4)),
                 scc_hp2);
    CHECK_GOLDEN(spatial_correlation_coefficient(p32, t32, O().hp_filter(torch::tensor({{1.0}})).window_size(3)),
                 scc_hp1);
    CHECK_GOLDEN(spatial_correlation_coefficient(p64, t32), scc_mixed);
    CHECK_GOLDEN(spatial_correlation_coefficient(torch::full({1, 1, 16, 16}, 0.5),
                                                 t32.slice(0, 0, 1).slice(1, 0, 1).slice(2, 0, 16).slice(3, 0, 16)),
                 scc_const);
    SpatialCorrelationCoefficient m(O().window_size(5));
    m(first(p32), first(t32));
    m.update(second(p32), second(t32));
    CHECK_GOLDEN(m.compute(), scc_cls);
    TM_CHECK(throws_invalid([] { return spatial_correlation_coefficient(p32, t32, O().reduction(RM::sum)); }));
    TM_CHECK(throws_invalid([] { return spatial_correlation_coefficient(p32, t32.slice(2, 1)); }));
    TM_CHECK(throws_invalid([] { return spatial_correlation_coefficient(p32[0][0], t32[0][0]); }));
    TM_CHECK(throws_invalid([] { return spatial_correlation_coefficient(p32, t32, O().window_size(0)); }));
    TM_CHECK(throws_invalid([] { return spatial_correlation_coefficient(p32, t32, O().window_size(33)); }));
    TM_CHECK(throws_invalid([&] {
        m.update(p32, t32.slice(2, 1));
        return 0;
    }));
}

static void test_vif() {
    using O = visual_information_fidelity_option;
    CHECK_GOLDEN(visual_information_fidelity(cp, ct), vif_f32);
    CHECK_GOLDEN(visual_information_fidelity(C64.first, C64.second, O().reduction(RM::none)), vif_f64_none);
    CHECK_GOLDEN(visual_information_fidelity(cp, ct, O().sigma_n_sq(0.4)), vif_sigma);
    CHECK_GOLDEN(visual_information_fidelity(cp.slice(1, 0, 1), ct.slice(1, 0, 1), O().reduction(RM::none)), vif_1ch);
    CHECK_GOLDEN(visual_information_fidelity(torch::full({1, 1, 41, 41}, 0.5),
                                             ct.slice(0, 0, 1).slice(1, 0, 1).slice(2, 0, 41).slice(3, 0, 41)),
                 vif_const);
    {
        VisualInformationFidelity m;
        m(first(cp), first(ct));
        m.update(second(cp), second(ct));
        CHECK_GOLDEN(m.compute(), vif_cls);
    }
    {
        VisualInformationFidelity m(O().reduction(RM::none));
        m.update(first(cp).slice(1, 0, 1), first(ct).slice(1, 0, 1));
        m.update(second(cp).slice(1, 0, 1), second(ct).slice(1, 0, 1));
        CHECK_GOLDEN(m.compute(), vif_cls_none_1ch);
    }
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(p32, t32); }));
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(cp, t32); }));
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(cp, ct.slice(0, 0, 1)); }));
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(cp, ct, O().reduction(RM::sum)); }));
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(torch::zeros({41}), torch::zeros({41})); }));
    TM_CHECK(throws_invalid([] { return visual_information_fidelity(cp[0], ct[0]); })); // [B, C, H, W] only
    TM_CHECK(throws_invalid([] { return VisualInformationFidelity(O().reduction(RM::sum)); }));
    TM_CHECK(throws_invalid([] { return VisualInformationFidelity(O().sigma_n_sq(-1.0)); }));
    // locally anti-correlated (g < 0) on the right half
    auto vp = cp.clone();
    vp.slice(3, 24).copy_(1 - ct.slice(3, 24));
    CHECK_GOLDEN(visual_information_fidelity(vp, ct, O().reduction(RM::none)), vif_anti);
    // update() runs the functional's checks and leaves the state untouched
    VisualInformationFidelity m;
    for (const auto &[a, b]: std::vector<std::pair<torch::Tensor, torch::Tensor>>{
                 {p32, t32}, {cp, ct.slice(0, 0, 1)}, {cp, ct.slice(3, 1)}, {cp[0], ct[0]}}) {
        TM_CHECK(throws_invalid([&] {
            m.update(a, b);
            return 0;
        }));
    }
    m.update(cp, ct);
    TM_CHECK(torch::equal(m.compute(), visual_information_fidelity(cp, ct)));
}

static void test_class_batches() {
    const auto dr1 = peak_signal_noise_ratio_option().data_range({1.0});
    const auto sw5 = sliding_window_option().window_size(5);
    CHECK_BATCHES(PeakSignalNoiseRatio, dr1, p32, t32);
    // data_range None takes each batch's own range (as in torchmetrics), so pin it here
    CHECK_BATCHES(StructuralSimilarityIndexMeasure, structural_similarity_index_measure_option().data_range({1.0}), p32,
                  t32);
    CHECK_BATCHES(MultiScaleStructuralSimilarityIndexMeasure,
                  multiscale_structural_similarity_index_measure_option()
                          .betas({0.5, 0.3, 0.2})
                          .kernel_size({5})
                          .data_range({1.0}),
                  p32, t32);
    CHECK_BATCHES(UniversalImageQualityIndex, universal_image_quality_index_option(), p32, t32);
    CHECK_BATCHES(TotalVariation, total_variation_option(), p32);
    CHECK_BATCHES(SpectralAngleMapper, spectral_angle_mapper_option(), p32, t32);
    CHECK_BATCHES(ErrorRelativeGlobalDimensionlessSynthesis, error_relative_global_dimensionless_synthesis_option(),
                  p32, t32);
    CHECK_BATCHES(RelativeAverageSpectralError, sw5, p32, t32);
    CHECK_BATCHES(RootMeanSquaredErrorUsingSlidingWindow, sw5, p32, t32);
    CHECK_BATCHES(SpatialCorrelationCoefficient, spatial_correlation_coefficient_option().window_size(5), p32, t32);
    CHECK_BATCHES(VisualInformationFidelity, visual_information_fidelity_option(), cp, ct);
}

// dtype / device follow the input
static void test_mps() {
    if (!torch::mps::is_available()) {
        return;
    }
    const auto mp = p32.to(torch::kMPS), mt = t32.to(torch::kMPS);
    const auto dr1 = peak_signal_noise_ratio_option().data_range({1.0});
    const auto v = peak_signal_noise_ratio(mp, mt, dr1);
    TM_CHECK(v.device().is_mps());
    TM_CHECK_CLOSE(v.cpu().item<double>(), peak_signal_noise_ratio(p32, t32, dr1).item<double>(), 1e-4);
    TM_CHECK_CLOSE(structural_similarity_index_measure(mp, mt).cpu().item<double>(),
                   structural_similarity_index_measure(p32, t32).item<double>(), 1e-5);
    StructuralSimilarityIndexMeasure m;
    m.update(mp, mt);
    TM_CHECK(m.compute().device().is_mps());
    TM_CHECK_CLOSE(spectral_angle_mapper(mp, mt).cpu().item<double>(), spectral_angle_mapper(p32, t32).item<double>(),
                   1e-5);
}

int main() {
    TM_RUN(test_inputs);
    TM_RUN(test_psnr);
    TM_RUN(test_ssim);
    TM_RUN(test_ms_ssim);
    TM_RUN(test_uqi);
    TM_RUN(test_tv_sam_ergas);
    TM_RUN(test_sliding_window);
    TM_RUN(test_scc);
    TM_RUN(test_vif);
    TM_RUN(test_class_batches);
    TM_RUN(test_mps);
    return tm_test::summary("metric_test_image");
}
