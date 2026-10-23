#pragma once
// Evaluates one classification parity case (gen_golden_classification.py make_case / expected) with
// torchmedia::metric::classification: the same keys "<family>/<f|c>/<metric>[.<i>...]" (f: the functional on the whole
// input, c: the class updated with the two halves), "<key>!" when the call threw. Shared by the ctest binaries
// (golden_classification.hpp cases) and the opt-in sweep (sweep/sweep_classification.cpp, pickled cases).
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <string>
#include <torchmedia.hpp>
#include <utility>
#include <vector>

namespace cls_driver {
    using namespace torchmedia::metric::classification;
    using T = torch::Tensor;
    using outputs = std::map<std::string, T>;

    struct case_view {
        std::function<T(const std::string &)> in; // preds, target, thrv
        std::function<bool(const std::string &)> has; // a parameter (ii, mfpr) or input (thrv) present
        std::function<double(const std::string &)> prm;
    };

    inline void put(outputs &out, const std::string &key, const T &v) { out[key] = v; }
    template<class A, class B>
    inline void put(outputs &out, const std::string &key, const std::pair<A, B> &v) {
        put(out, key + ".0", v.first);
        put(out, key + ".1", v.second);
    }
    inline void put(outputs &out, const std::string &key, const curve_t &v) {
        put(out, key + ".0", std::get<0>(v));
        put(out, key + ".1", std::get<1>(v));
        put(out, key + ".2", std::get<2>(v));
    }
    // per-class curves in torchmetrics' structure: one averaged curve, binned [C, n] tensors with shared thresholds
    // (stacked), or lists
    struct per_class_curves {
        curve_list_t c;
        bool stacked;
    };
    inline void put(outputs &out, const std::string &key, const per_class_curves &v) {
        const auto &[x, y, t] = v.c;
        if (x.size() == 1) {
            put(out, key, curve_t{x[0], y[0], t[0]});
        } else if (v.stacked) {
            const auto n = static_cast<int64_t>(x.size());
            put(out, key, curve_t{torch::stack(x).reshape({n, -1}), torch::stack(y).reshape({n, -1}), t[0]});
        } else {
            for (size_t i = 0; i < x.size(); ++i) {
                put(out, key + ".0." + std::to_string(i), x[i]);
                put(out, key + ".1." + std::to_string(i), y[i]);
                put(out, key + ".2." + std::to_string(i), t[i]);
            }
        }
    }

    template<class Fn>
    inline void run(outputs &out, const std::string &key, Fn &&fn) {
        try {
            put(out, key, fn());
        } catch (const std::exception &e) {
            out[key + "!"] = T();
            if (std::getenv("CLS_DRIVER_VERBOSE")) { // debugging aid
                std::fprintf(stderr, "%s: %s\n", key.c_str(), std::string(e.what()).substr(0, 300).c_str());
            }
        }
    }

    // the class updated with x[:k] and x[k:], k = max(1, N / 2)
    template<class M, class Opt>
    inline auto halves(const Opt &o, const T &p, const T &t) {
        M m(o);
        const auto k = std::max<int64_t>(1, p.size(0) / 2);
        m.update(p.slice(0, 0, k), t.slice(0, 0, k));
        if (p.size(0) > k) {
            m.update(p.slice(0, k), t.slice(0, k));
        }
        return m.compute();
    }

    template<class M, class Opt, class Fn>
    inline void both(outputs &out, const std::string &family, const std::string &name, const case_view &c, const Opt &o,
                     Fn &&fn) {
        const auto p = c.in("preds"), t = c.in("target");
        run(out, family + "/f/" + name, [&] { return fn(p, t, o); });
        run(out, family + "/c/" + name, [&] { return halves<M>(o, p, t); });
    }
    template<class M, class Opt, class Fn>
    inline void both_curves(outputs &out, const std::string &name, const case_view &c, const Opt &o, bool stacked,
                            Fn &&fn) {
        const auto p = c.in("preds"), t = c.in("target");
        run(out, "curve/f/" + name, [&] { return per_class_curves{fn(p, t, o), stacked}; });
        run(out, "curve/c/" + name, [&] { return per_class_curves{halves<M>(o, p, t), stacked}; });
    }

    inline auto avg(double code) -> average_mode { return static_cast<average_mode>(static_cast<int>(code)); }
    inline auto ignore(const case_view &c) -> std::optional<int64_t> {
        return c.has("ii") ? std::optional<int64_t>(static_cast<int64_t>(c.prm("ii"))) : std::nullopt;
    }
    inline auto multidim(const case_view &c) -> multidim_average_mode {
        return c.prm("md") != 0 ? multidim_average_mode::samplewise : multidim_average_mode::global;
    }
    inline auto thresholds(const case_view &c) -> thresholds_spec {
        switch (static_cast<int>(c.prm("tk"))) {
            case 1:
                return static_cast<int64_t>(c.prm("tn"));
            case 2: {
                const auto v = c.in("thrv").to(torch::kFloat64).contiguous();
                return std::vector<double>(v.data_ptr<double>(), v.data_ptr<double>() + v.numel());
            }
            case 3:
                return c.in("thrv");
            default:
                return std::monostate{};
        }
    }

    // ---------------- stat-score and confusion-matrix families ----------------
    inline auto run_stat(const case_view &c) -> outputs {
        outputs out;
        const auto task = static_cast<int>(c.prm("task"));
        const auto C = static_cast<int64_t>(c.prm("C"));
        const auto ii = ignore(c);
        const auto normalize = static_cast<normalize_mode>(static_cast<int>(c.prm("ncm")));
        const auto weights = static_cast<kappa_weights_mode>(static_cast<int>(c.prm("kw")));
        const auto s = [&](auto fn) { return [fn](const T &p, const T &t, const auto &o) { return fn(p, t, o); }; };
        if (task == 0) {
            const auto o = binary_stat_scores_option()
                                   .threshold(c.prm("thr"))
                                   .multidim_average(multidim(c))
                                   .ignore_index(ii)
                                   .zero_division(c.prm("zd"))
                                   .beta(c.prm("beta"));
            both<BinaryStatScores>(out, "stat", "stat_scores", c, o, s(binary_stat_scores));
            both<BinaryAccuracy>(out, "stat", "accuracy", c, o, s(binary_accuracy));
            both<BinaryPrecision>(out, "stat", "precision", c, o, s(binary_precision));
            both<BinaryRecall>(out, "stat", "recall", c, o, s(binary_recall));
            both<BinaryF1Score>(out, "stat", "f1", c, o, s(binary_f1_score));
            both<BinaryFBetaScore>(out, "stat", "fbeta", c, o, s(binary_fbeta_score));
            both<BinarySpecificity>(out, "stat", "specificity", c, o, s(binary_specificity));
            both<BinaryNegativePredictiveValue>(out, "stat", "npv", c, o, s(binary_negative_predictive_value));
            both<BinaryHammingDistance>(out, "stat", "hamming", c, o, s(binary_hamming_distance));
            const auto co = binary_confusion_matrix_option()
                                    .threshold(c.prm("thr"))
                                    .ignore_index(ii)
                                    .normalize(normalize)
                                    .zero_division(c.prm("zd"))
                                    .weights(weights);
            both<BinaryConfusionMatrix>(out, "stat", "confmat", c, co, s(binary_confusion_matrix));
            both<BinaryJaccardIndex>(out, "stat", "jaccard", c, co, s(binary_jaccard_index));
            both<BinaryMatthewsCorrCoef>(out, "stat", "mcc", c, co, s(binary_matthews_corrcoef));
            both<BinaryCohenKappa>(out, "stat", "kappa", c, co, s(binary_cohen_kappa));
        } else if (task == 1) {
            const auto o = multiclass_stat_scores_option()
                                   .num_classes(C)
                                   .average(avg(c.prm("avg")))
                                   .top_k(static_cast<int64_t>(c.prm("topk")))
                                   .multidim_average(multidim(c))
                                   .ignore_index(ii)
                                   .zero_division(c.prm("zd"))
                                   .beta(c.prm("beta"));
            both<MulticlassStatScores>(out, "stat", "stat_scores", c, o, s(multiclass_stat_scores));
            both<MulticlassAccuracy>(out, "stat", "accuracy", c, o, s(multiclass_accuracy));
            both<MulticlassPrecision>(out, "stat", "precision", c, o, s(multiclass_precision));
            both<MulticlassRecall>(out, "stat", "recall", c, o, s(multiclass_recall));
            both<MulticlassF1Score>(out, "stat", "f1", c, o, s(multiclass_f1_score));
            both<MulticlassFBetaScore>(out, "stat", "fbeta", c, o, s(multiclass_fbeta_score));
            both<MulticlassSpecificity>(out, "stat", "specificity", c, o, s(multiclass_specificity));
            both<MulticlassNegativePredictiveValue>(out, "stat", "npv", c, o, s(multiclass_negative_predictive_value));
            both<MulticlassHammingDistance>(out, "stat", "hamming", c, o, s(multiclass_hamming_distance));
            both<MulticlassExactMatch>(out, "stat", "exact", c, o, s(multiclass_exact_match));
            const auto co = multiclass_confusion_matrix_option()
                                    .num_classes(C)
                                    .ignore_index(ii)
                                    .normalize(normalize)
                                    .average(avg(c.prm("avg")))
                                    .zero_division(c.prm("zd"))
                                    .weights(weights);
            both<MulticlassConfusionMatrix>(out, "stat", "confmat", c, co, s(multiclass_confusion_matrix));
            both<MulticlassJaccardIndex>(out, "stat", "jaccard", c, co, s(multiclass_jaccard_index));
            both<MulticlassMatthewsCorrCoef>(out, "stat", "mcc", c, co, s(multiclass_matthews_corrcoef));
            both<MulticlassCohenKappa>(out, "stat", "kappa", c, co, s(multiclass_cohen_kappa));
        } else {
            const auto o = multilabel_stat_scores_option()
                                   .num_labels(C)
                                   .threshold(c.prm("thr"))
                                   .average(avg(c.prm("avg")))
                                   .multidim_average(multidim(c))
                                   .ignore_index(ii)
                                   .zero_division(c.prm("zd"))
                                   .beta(c.prm("beta"));
            both<MultilabelStatScores>(out, "stat", "stat_scores", c, o, s(multilabel_stat_scores));
            both<MultilabelAccuracy>(out, "stat", "accuracy", c, o, s(multilabel_accuracy));
            both<MultilabelPrecision>(out, "stat", "precision", c, o, s(multilabel_precision));
            both<MultilabelRecall>(out, "stat", "recall", c, o, s(multilabel_recall));
            both<MultilabelF1Score>(out, "stat", "f1", c, o, s(multilabel_f1_score));
            both<MultilabelFBetaScore>(out, "stat", "fbeta", c, o, s(multilabel_fbeta_score));
            both<MultilabelSpecificity>(out, "stat", "specificity", c, o, s(multilabel_specificity));
            both<MultilabelNegativePredictiveValue>(out, "stat", "npv", c, o, s(multilabel_negative_predictive_value));
            both<MultilabelHammingDistance>(out, "stat", "hamming", c, o, s(multilabel_hamming_distance));
            both<MultilabelExactMatch>(out, "stat", "exact", c, o, s(multilabel_exact_match));
            const auto co = multilabel_confusion_matrix_option()
                                    .num_labels(C)
                                    .threshold(c.prm("thr"))
                                    .ignore_index(ii)
                                    .normalize(normalize)
                                    .average(avg(c.prm("avg")))
                                    .zero_division(c.prm("zd"));
            both<MultilabelConfusionMatrix>(out, "stat", "confmat", c, co, s(multilabel_confusion_matrix));
            both<MultilabelJaccardIndex>(out, "stat", "jaccard", c, co, s(multilabel_jaccard_index));
            both<MultilabelMatthewsCorrCoef>(out, "stat", "mcc", c, co, s(multilabel_matthews_corrcoef));
        }
        return out;
    }

    // ---------------- curve family ----------------
    template<class Opt>
    inline auto with_min(Opt o, double mn) -> Opt {
        return o.min_precision(mn).min_recall(mn).min_specificity(mn).min_sensitivity(mn);
    }

    inline auto run_curve(const case_view &c) -> outputs {
        outputs out;
        const auto task = static_cast<int>(c.prm("task"));
        const auto C = static_cast<int64_t>(c.prm("C"));
        const auto ii = ignore(c);
        const auto th = thresholds(c);
        const std::pair<double, double> fr{c.prm("flo"), c.prm("fhi")};
        const auto mn = c.prm("mn");
        const bool binned = !std::holds_alternative<std::monostate>(th);
        const auto s = [&](auto fn) { return [fn](const T &p, const T &t, const auto &o) { return fn(p, t, o); }; };
        if (task == 0) {
            auto o = binary_curve_option().ignore_index(ii).fpr_range(fr);
            o._thresholds = th;
            o = with_min(o, mn);
            both<BinaryPrecisionRecallCurve>(out, "curve", "pr", c, o, s(binary_precision_recall_curve));
            both<BinaryROC>(out, "curve", "roc", c, o, s(binary_roc));
            const auto ao = c.has("mfpr") ? binary_curve_option(o).max_fpr(c.prm("mfpr")) : o;
            both<BinaryAUROC>(out, "curve", "auroc", c, ao, s(binary_auroc));
            both<BinaryAveragePrecision>(out, "curve", "ap", c, o, s(binary_average_precision));
            both<BinaryLogAUC>(out, "curve", "logauc", c, o, s(binary_logauc));
            both<BinaryEER>(out, "curve", "eer", c, o, s(binary_eer));
            both<BinaryPrecisionAtFixedRecall>(out, "curve", "par", c, o, s(binary_precision_at_fixed_recall));
            both<BinaryRecallAtFixedPrecision>(out, "curve", "rap", c, o, s(binary_recall_at_fixed_precision));
            both<BinarySensitivityAtSpecificity>(out, "curve", "sas", c, o, s(binary_sensitivity_at_specificity));
            both<BinarySpecificityAtSensitivity>(out, "curve", "spas", c, o, s(binary_specificity_at_sensitivity));
        } else if (task == 1) {
            auto o = multiclass_curve_option().num_classes(C).ignore_index(ii);
            o._thresholds = th;
            const auto co = multiclass_curve_option(o).average(avg(c.prm("cavg")));
            const bool stacked = binned && co._average == average_mode::none;
            both_curves<MulticlassPrecisionRecallCurve>(out, "pr", c, co, stacked,
                                                        s(multiclass_precision_recall_curve));
            both_curves<MulticlassROC>(out, "roc", c, co, stacked, s(multiclass_roc));
            auto ao = multiclass_auc_option().num_classes(C).ignore_index(ii).average(avg(c.prm("aavg")));
            ao._thresholds = th;
            both<MulticlassAUROC>(out, "curve", "auroc", c, ao, s(multiclass_auroc));
            both<MulticlassAveragePrecision>(out, "curve", "ap", c, ao, s(multiclass_average_precision));
            const auto lo = multiclass_auc_option(ao).average(avg(c.prm("lavg"))).fpr_range(fr);
            both<MulticlassLogAUC>(out, "curve", "logauc", c, lo, s(multiclass_logauc));
            both<MulticlassEER>(out, "curve", "eer", c, co, s(multiclass_eer));
            o = with_min(o, mn);
            both<MulticlassPrecisionAtFixedRecall>(out, "curve", "par", c, o, s(multiclass_precision_at_fixed_recall));
            both<MulticlassRecallAtFixedPrecision>(out, "curve", "rap", c, o, s(multiclass_recall_at_fixed_precision));
            both<MulticlassSensitivityAtSpecificity>(out, "curve", "sas", c, o,
                                                     s(multiclass_sensitivity_at_specificity));
            both<MulticlassSpecificityAtSensitivity>(out, "curve", "spas", c, o,
                                                     s(multiclass_specificity_at_sensitivity));
        } else {
            auto o = multilabel_curve_option().num_labels(C).ignore_index(ii);
            o._thresholds = th;
            both_curves<MultilabelPrecisionRecallCurve>(out, "pr", c, o, binned, s(multilabel_precision_recall_curve));
            both_curves<MultilabelROC>(out, "roc", c, o, binned, s(multilabel_roc));
            auto ao = multilabel_auc_option().num_labels(C).ignore_index(ii).average(avg(c.prm("aavg")));
            ao._thresholds = th;
            both<MultilabelAUROC>(out, "curve", "auroc", c, ao, s(multilabel_auroc));
            both<MultilabelAveragePrecision>(out, "curve", "ap", c, ao, s(multilabel_average_precision));
            const auto lo = multilabel_auc_option(ao).average(avg(c.prm("lavg"))).fpr_range(fr);
            both<MultilabelLogAUC>(out, "curve", "logauc", c, lo, s(multilabel_logauc));
            both<MultilabelEER>(out, "curve", "eer", c, o, s(multilabel_eer));
            o = with_min(o, mn);
            both<MultilabelPrecisionAtFixedRecall>(out, "curve", "par", c, o, s(multilabel_precision_at_fixed_recall));
            both<MultilabelRecallAtFixedPrecision>(out, "curve", "rap", c, o, s(multilabel_recall_at_fixed_precision));
            both<MultilabelSensitivityAtSpecificity>(out, "curve", "sas", c, o,
                                                     s(multilabel_sensitivity_at_specificity));
            both<MultilabelSpecificityAtSensitivity>(out, "curve", "spas", c, o,
                                                     s(multilabel_specificity_at_sensitivity));
        }
        return out;
    }

    // ---------------- calibration, hinge, ranking ----------------
    inline auto run_misc(const case_view &c) -> outputs {
        outputs out;
        const auto task = static_cast<int>(c.prm("task"));
        const auto C = static_cast<int64_t>(c.prm("C"));
        const auto ii = ignore(c);
        const auto nb = static_cast<int64_t>(c.prm("nb"));
        const auto norm = static_cast<calibration_norm_mode>(static_cast<int>(c.prm("cen")));
        const auto s = [&](auto fn) { return [fn](const T &p, const T &t, const auto &o) { return fn(p, t, o); }; };
        if (task == 0) {
            const auto ce = binary_calibration_error_option().n_bins(nb).norm(norm).ignore_index(ii);
            both<BinaryCalibrationError>(out, "misc", "ce", c, ce, s(binary_calibration_error));
            const auto h = binary_hinge_loss_option().squared(c.prm("sq") != 0).ignore_index(ii);
            both<BinaryHingeLoss>(out, "misc", "hinge", c, h, s(binary_hinge_loss));
        } else if (task == 1) {
            const auto ce = multiclass_calibration_error_option().num_classes(C).n_bins(nb).norm(norm).ignore_index(ii);
            both<MulticlassCalibrationError>(out, "misc", "ce", c, ce, s(multiclass_calibration_error));
            const auto h =
                    multiclass_hinge_loss_option()
                            .num_classes(C)
                            .squared(c.prm("sq") != 0)
                            .multiclass_mode(static_cast<hinge_multiclass_mode>(static_cast<int>(c.prm("hmode"))))
                            .ignore_index(ii);
            both<MulticlassHingeLoss>(out, "misc", "hinge", c, h, s(multiclass_hinge_loss));
        } else {
            const auto r = multilabel_ranking_option().num_labels(C).ignore_index(ii);
            both<MultilabelCoverageError>(out, "misc", "coverage", c, r, s(multilabel_coverage_error));
            both<MultilabelRankingAveragePrecision>(out, "misc", "rank_ap", c, r,
                                                    s(multilabel_ranking_average_precision));
            both<MultilabelRankingLoss>(out, "misc", "rank_loss", c, r, s(multilabel_ranking_loss));
        }
        return out;
    }
} // namespace cls_driver
