// Sweep driver: evaluates cases.pt (sweep_gen.py / sweep_audio_gen.py) with torchmedia::metric, writes results.pt.
#include <fstream>
#include <torchmedia.hpp>
using namespace torchmedia::metric;
using T = torch::Tensor;

static auto vec_d(const T &t) -> std::vector<double> {
    const auto c = t.to(torch::kFloat64).contiguous();
    return {c.data_ptr<double>(), c.data_ptr<double>() + c.numel()};
}
static auto vec_i(const T &t) -> std::vector<int64_t> {
    const auto c = t.to(torch::kInt64).contiguous();
    return {c.data_ptr<int64_t>(), c.data_ptr<int64_t>() + c.numel()};
}
static auto make_mask(const T &prm) -> T {
    const auto v = vec_i(prm);
    auto m = torch::zeros({24, 28}, torch::kUInt8);
    m.slice(0, v[1], v[3]).slice(1, v[0], v[2]).fill_(1);
    if (v[4])
        m.slice(0, v[1] + (v[3] - v[1]) / 3, v[3] - (v[3] - v[1]) / 3)
                .slice(1, v[0] + (v[2] - v[0]) / 3, v[2] - (v[2] - v[0]) / 3)
                .fill_(0);
    return m;
}
static auto masks(const T &prm) -> T {
    std::vector<T> ms;
    for (int64_t i = 0; i < prm.size(0); ++i)
        ms.push_back(make_mask(prm[i]));
    return ms.empty() ? torch::zeros({0, 24, 28}, torch::kUInt8) : torch::stack(ms);
}

int main() {
    std::ifstream f("cases.pt", std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
    const auto cases = torch::pickle_load(buf).toList();
    c10::List<c10::Dict<std::string, T>> results;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto d = cases.get(ci).toGenericDict();
        const auto get = [&](const char *k) { return d.at(k).toTensor(); };
        const std::string kind = d.at("kind").toStringRef();
        c10::Dict<std::string, T> out;
        try {
            if (kind == "map" || kind == "voc") {
                const bool segm = kind == "map" && get("segm").item<int64_t>();
                const auto fmt = static_cast<detection::box_format_mode>(get("fmt").item<int64_t>());
                const auto pc = vec_i(get("pc")), gc = vec_i(get("gc"));
                std::vector<detection::detection_prediction> P;
                std::vector<detection::detection_target> G;
                int64_t po = 0, go = 0;
                for (size_t i = 0; i < pc.size(); ++i) {
                    const auto pb = get("pb").slice(0, po, po + pc[i]), gb = get("gb").slice(0, go, go + gc[i]);
                    P.push_back({segm ? T() : pb, get("ps").slice(0, po, po + pc[i]),
                                 get("pl").slice(0, po, po + pc[i]), segm ? masks(pb) : T()});
                    G.push_back({segm ? T() : gb, get("gl").slice(0, go, go + gc[i]),
                                 get("gcr").slice(0, go, go + gc[i]), get("ga").slice(0, go, go + gc[i]),
                                 segm ? masks(gb) : T()});
                    po += pc[i];
                    go += gc[i];
                }
                if (kind == "map") {
                    auto opt =
                            detection::mean_average_precision_option()
                                    .box_format(fmt)
                                    .iou_thresholds(vec_d(get("iou")))
                                    .rec_thresholds(vec_d(get("rec")))
                                    .max_detection_thresholds(vec_i(get("mdt")))
                                    .class_metrics(get("cm").item<int64_t>())
                                    .average(get("micro").item<int64_t>() ? detection::average_mode::micro
                                                                          : detection::average_mode::macro)
                                    .iou_type(segm ? detection::iou_type_mode::segm : detection::iou_type_mode::bbox);
                    const auto r = detection::mean_average_precision(P, G, opt);
                    out.insert("stats",
                               torch::tensor({r.map, r.map_50, r.map_75, r.map_small, r.map_medium, r.map_large,
                                              r.mar_1, r.mar_10, r.mar_100, r.mar_small, r.mar_medium, r.mar_large},
                                             torch::kFloat64));
                    out.insert("mpc", r.map_per_class);
                    out.insert("rpc", r.mar_100_per_class);
                } else {
                    const auto r = detection::voc_average_precision(
                            P, G,
                            detection::voc_average_precision_option()
                                    .box_format(fmt)
                                    .iou_threshold(get("thr").item<double>())
                                    .interpolation(get("use07").item<int64_t>()
                                                           ? detection::voc_interpolation_mode::eleven_point
                                                           : detection::voc_interpolation_mode::all_point)
                                    .pixel_inclusive(get("pix").item<int64_t>()));
                    out.insert("stats", torch::cat({torch::tensor({r.map}, torch::kFloat64), r.ap_per_class}));
                }
            } else if (kind == "image") {
                using namespace image;
                const auto p = get("p"), t = get("t");
                const auto ks = get("ks").item<int64_t>(), ws = get("ws").item<int64_t>(), i = get("i").item<int64_t>();
                const auto sg = get("sg").item<double>();
                const auto dr = vec_d(get("dr"));
                const auto h = p.size(2), w = p.size(3);
                out.insert("psnr", peak_signal_noise_ratio(
                                           p, t,
                                           peak_signal_noise_ratio_option()
                                                   .data_range(dr.empty() ? std::vector<double>{1.0} : dr)
                                                   .dim(i % 2 ? std::vector<int64_t>{1, 2, 3} : std::vector<int64_t>{})
                                                   .reduction(i % 2 ? reduction_mode::none
                                                                    : reduction_mode::elementwise_mean)));
                out.insert("ssim", structural_similarity_index_measure(p, t,
                                                                       structural_similarity_index_measure_option()
                                                                               .gaussian_kernel(i % 4)
                                                                               .sigma({sg})
                                                                               .kernel_size({ks})
                                                                               .data_range(dr)
                                                                               .reduction(reduction_mode::none)));
                out.insert(
                        "uqi",
                        universal_image_quality_index(
                                p, t,
                                universal_image_quality_index_option().kernel_size({ks, ks}).sigma({sg, sg}).reduction(
                                        reduction_mode::none)));
                out.insert("tv", total_variation(p, reduction_mode::none));
                out.insert("sam", spectral_angle_mapper(p, t, reduction_mode::none));
                out.insert("ergas", error_relative_global_dimensionless_synthesis(
                                            p, t,
                                            error_relative_global_dimensionless_synthesis_option()
                                                    .ratio(get("ratio").item<double>())
                                                    .reduction(reduction_mode::none)));
                out.insert("rmse", root_mean_squared_error_using_sliding_window_with_map(p, t, ws).second);
                out.insert("rase", relative_average_spectral_error(p, t, ws));
                out.insert("scc", spatial_correlation_coefficient(p, t,
                                                                  spatial_correlation_coefficient_option()
                                                                          .window_size(std::min({ws + 1, h, w}))
                                                                          .reduction(reduction_mode::none)));
                if (h >= 41 && w >= 41)
                    out.insert("vif",
                               visual_information_fidelity(
                                       p, t, visual_information_fidelity_option().reduction(reduction_mode::none)));
                if (std::min(h, w) / 4 > std::max<int64_t>(ks, static_cast<int64_t>(3.5 * sg + 0.5) * 2 + 1))
                    out.insert("msssim", multiscale_structural_similarity_index_measure(
                                                 p, t,
                                                 multiscale_structural_similarity_index_measure_option()
                                                         .kernel_size({ks})
                                                         .sigma({sg})
                                                         .betas({0.4, 0.35, 0.25})
                                                         .data_range(dr)
                                                         .reduction(reduction_mode::none)));
            } else if (kind == "snr") {
                using namespace audio;
                const auto p = get("p"), t = get("t");
                const bool zm = get("zm").item<int64_t>();
                out.insert("snr", signal_noise_ratio(p, t, zm));
                out.insert("sisdr", scale_invariant_signal_distortion_ratio(p, t, zm));
                out.insert("sisnr", scale_invariant_signal_noise_ratio(p, t));
                out.insert("sasdr",
                           source_aggregated_signal_distortion_ratio(p, t,
                                                                     source_aggregated_signal_distortion_ratio_option()
                                                                             .scale_invariant(get("si").item<int64_t>())
                                                                             .zero_mean(zm)));
                if (p.size(-1) % 2 == 0) {
                    const std::vector<int64_t> shape{p.size(0), p.size(1), p.size(2) / 2, 2};
                    out.insert("csisnr",
                               complex_scale_invariant_signal_noise_ratio(p.reshape(shape), t.reshape(shape), zm));
                }
            } else if (kind == "sdr") {
                using namespace audio;
                const auto cg = get("cg").item<int64_t>();
                const auto ld = get("ld").item<double>();
                out.insert("sdr", signal_distortion_ratio(
                                          get("p"), get("t"),
                                          signal_distortion_ratio_option()
                                                  .filter_length(get("fl").item<int64_t>())
                                                  .zero_mean(get("zm").item<int64_t>())
                                                  .use_cg_iter(cg < 0 ? std::nullopt : std::optional<int64_t>(cg))
                                                  .load_diag(ld < 0 ? std::nullopt : std::optional<double>(ld))));
            } else if (kind == "pit") {
                using namespace audio;
                const auto fn = get("fn").item<int64_t>();
                auto opt = permutation_invariant_training_option();
                if (fn == 0) {
                    opt.metric_func(
                            [](const T &a, const T &b) { return scale_invariant_signal_distortion_ratio(a, b); });
                } else if (fn == 1) {
                    opt.metric_func([](const T &a, const T &b) { return signal_noise_ratio(a, b); });
                } else if (fn == 2) {
                    opt.metric_func([](const T &a, const T &b) {
                           return (a - b).pow(2).mean({-1});
                       }).eval_func(eval_func_mode::min);
                } else {
                    opt.metric_func([](const T &a, const T &b) {
                           return source_aggregated_signal_distortion_ratio(a, b);
                       }).mode(pit_mode::permutation_wise);
                }
                const auto [m, perm] = permutation_invariant_training(get("p"), get("t"), opt);
                out.insert("metric", m);
                out.insert("perm", perm);
            } else if (kind == "bss") {
                const auto r = audio::bss_eval_sources(get("r"), get("e"), get("cp").item<int64_t>());
                out.insert("sdr", r.sdr);
                out.insert("sir", r.sir);
                out.insert("sar", r.sar);
                out.insert("perm", r.perm);
            } else if (kind == "sv") {
                const auto prm = vec_d(get("prm"));
                out.insert("eer", audio::equal_error_rate(get("tar"), get("non")));
                out.insert("dcf", audio::min_dcf(get("tar"), get("non"),
                                                 audio::min_dcf_option().p_target(prm[0]).c_miss(prm[1]).c_fa(prm[2])));
            } else if (kind == "stoi") {
                out.insert("stoi", audio::short_time_objective_intelligibility(
                                           get("p").unsqueeze(0), get("t").unsqueeze(0), get("fs").item<int64_t>(),
                                           get("ext").item<int64_t>()));
            } else if (kind == "text") {
                using namespace text;
                const auto strs = [&](const char *k) {
                    std::vector<std::string> v;
                    for (const auto &x: d.at(k).toList()) {
                        v.push_back(x.get().toStringRef());
                    }
                    return v;
                };
                const auto p = strs("p"), t = strs("t");
                out.insert("wer", word_error_rate(p, t));
                out.insert("cer", char_error_rate(p, t));
                out.insert("mer", match_error_rate(p, t));
                out.insert("wil", word_information_lost(p, t));
                out.insert("wip", word_information_preserved(p, t));
                out.insert("edit", edit_distance(p, t,
                                                 edit_distance_option()
                                                         .substitution_cost(get("sc").item<int64_t>())
                                                         .reduction(reduction_mode::none)));
            } else {
                using namespace segmentation;
                const auto p = get("p"), t = get("t");
                const auto c = get("c").item<int64_t>();
                const bool bg = get("bg").item<int64_t>();
                out.insert(
                        "miou",
                        mean_iou(p, t,
                                 mean_iou_option().num_classes(c).include_background(bg).per_class(true).input_format(
                                         input_format_mode::index)));
                const auto dopt = dice_score_option().num_classes(c).include_background(bg).input_format(
                        input_format_mode::index);
                out.insert("dice", dice_score(p, t, dice_score_option(dopt).average(dice_average_mode::none)));
                out.insert("dicew", dice_score(p, t, dice_score_option(dopt).average(dice_average_mode::weighted)));
                const auto gopt = generalized_dice_score_option().num_classes(c).include_background(bg).input_format(
                        input_format_mode::index);
                out.insert("gdice", generalized_dice_score(p, t, generalized_dice_score_option(gopt).per_class(true)));
                out.insert("gdices",
                           generalized_dice_score(
                                   p, t, generalized_dice_score_option(gopt).weight_type(weight_type_mode::simple)));
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "case %zu (%s): %s\n", ci, kind.c_str(), std::string(e.what()).substr(0, 200).c_str());
        }
        results.push_back(out);
    }
    const auto bytes = torch::pickle_save(c10::IValue(results));
    std::ofstream("results.pt", std::ios::binary).write(bytes.data(), bytes.size());
}
