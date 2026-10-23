#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_MISC_HPP
#define LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_MISC_HPP
// Classification metrics on scores (develop_log/2026-10-24/progress02): torchmetrics 1.9 CalibrationError and
// HingeLoss (binary / multiclass) and the multilabel ranking metrics CoverageError, RankingAveragePrecision and
// RankingLoss, functionals and classes with the same libtorch calls in the same order.
#include "_classification_stat.hpp"

namespace torchmedia::metric::classification {
    enum class calibration_norm_mode { l1, l2, max };
    enum class hinge_multiclass_mode { crammer_singer, one_vs_all };

    typedef struct binary_calibration_error_option {
        int64_t _n_bins = 15; // >= 1
        calibration_norm_mode _norm = calibration_norm_mode::l1;
        std::optional<int64_t> _ignore_index;

        auto n_bins(int64_t v) -> binary_calibration_error_option & {
            _n_bins = v;
            return *this;
        }
        auto norm(calibration_norm_mode v) -> binary_calibration_error_option & {
            _norm = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> binary_calibration_error_option & {
            _ignore_index = v;
            return *this;
        }
    } binary_calibration_error_option_t;

    typedef struct multiclass_calibration_error_option {
        int64_t _num_classes = 0; // required (>= 2)
        int64_t _n_bins = 15;
        calibration_norm_mode _norm = calibration_norm_mode::l1;
        std::optional<int64_t> _ignore_index;

        auto num_classes(int64_t v) -> multiclass_calibration_error_option & {
            _num_classes = v;
            return *this;
        }
        auto n_bins(int64_t v) -> multiclass_calibration_error_option & {
            _n_bins = v;
            return *this;
        }
        auto norm(calibration_norm_mode v) -> multiclass_calibration_error_option & {
            _norm = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_calibration_error_option & {
            _ignore_index = v;
            return *this;
        }
    } multiclass_calibration_error_option_t;

    typedef struct binary_hinge_loss_option {
        bool _squared = false;
        std::optional<int64_t> _ignore_index;

        auto squared(bool v) -> binary_hinge_loss_option & {
            _squared = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> binary_hinge_loss_option & {
            _ignore_index = v;
            return *this;
        }
    } binary_hinge_loss_option_t;

    typedef struct multiclass_hinge_loss_option {
        int64_t _num_classes = 0; // required (>= 2)
        bool _squared = false;
        hinge_multiclass_mode _multiclass_mode = hinge_multiclass_mode::crammer_singer;
        std::optional<int64_t> _ignore_index;

        auto num_classes(int64_t v) -> multiclass_hinge_loss_option & {
            _num_classes = v;
            return *this;
        }
        auto squared(bool v) -> multiclass_hinge_loss_option & {
            _squared = v;
            return *this;
        }
        auto multiclass_mode(hinge_multiclass_mode v) -> multiclass_hinge_loss_option & {
            _multiclass_mode = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_hinge_loss_option & {
            _ignore_index = v;
            return *this;
        }
    } multiclass_hinge_loss_option_t;

    // CoverageError, RankingAveragePrecision, RankingLoss.
    typedef struct multilabel_ranking_option {
        int64_t _num_labels = 0; // required (>= 2)
        std::optional<int64_t> _ignore_index;

        auto num_labels(int64_t v) -> multilabel_ranking_option & {
            _num_labels = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multilabel_ranking_option & {
            _ignore_index = v;
            return *this;
        }
    } multilabel_ranking_option_t;

    namespace detail {
        // ---------------- CalibrationError ----------------
        inline auto ce_arg_error(const binary_calibration_error_option &o) -> std::string {
            return o._n_bins >= 1 ? std::string{}
                                  : std::format("Expected argument `n_bins` to be an integer larger than 0, but got {}",
                                                o._n_bins);
        }
        inline auto ce_arg_error(const multiclass_calibration_error_option &o) -> std::string {
            const auto e = count_error("num_classes", o._num_classes);
            return e.empty() ? ce_arg_error(binary_calibration_error_option().n_bins(o._n_bins)) : e;
        }
        inline auto floating_preds_error(const tensor_t &preds) -> std::string {
            return preds.is_floating_point()
                           ? std::string{}
                           : "Expected argument `preds` to be floating tensor with probabilities/logits";
        }
        inline auto ce_tensor_error(const tensor_t &preds, const tensor_t &target,
                                    const binary_calibration_error_option &o) -> std::string {
            const auto e = binary_tensor_error(preds, target, false, o._ignore_index);
            return e.empty() ? floating_preds_error(preds) : e;
        }
        inline auto ce_tensor_error(const tensor_t &preds, const tensor_t &target,
                                    const multiclass_calibration_error_option &o) -> std::string {
            const auto e = multiclass_tensor_error(preds, target, o._num_classes, false, o._ignore_index);
            return e.empty() ? floating_preds_error(preds) : e;
        }

        // {confidences, accuracies}: _binary_calibration_error_update (the positive-class probability and the target)
        inline auto ce_update(const tensor_t &preds, const tensor_t &target, const binary_calibration_error_option &o)
                -> std::pair<tensor_t, tensor_t> {
            return binary_format(preds, target, 0.0, o._ignore_index, false);
        }
        // _multiclass_calibration_error_update: the top probability and whether its class is the target (float32)
        inline auto ce_update(const tensor_t &preds_in, const tensor_t &target_in,
                              const multiclass_calibration_error_option &o) -> std::pair<tensor_t, tensor_t> {
            const auto [preds, target] = multiclass_format(preds_in, target_in, o._ignore_index, false);
            const auto [confidences, predictions] = normalize_logits(preds, true).max(1);
            return {confidences.to(torch::kFloat32), predictions.eq(target).to(torch::kFloat32)};
        }

        // _ce_compute over n_bins equal-width bins (bucketize right, index - 1)
        inline auto ce_compute(const tensor_t &confidences, const tensor_t &accuracies_in, int64_t n_bins,
                               calibration_norm_mode norm) -> tensor_t {
            const auto boundaries = torch::linspace(0, 1, n_bins + 1, confidences.options());
            const auto accuracies = accuracies_in.to(confidences.scalar_type());
            auto acc_bin = torch::zeros({n_bins + 1}, confidences.options());
            auto conf_bin = torch::zeros({n_bins + 1}, confidences.options());
            auto count_bin = torch::zeros({n_bins + 1}, confidences.options());
            const auto indices = torch::bucketize(confidences, boundaries, false, true) - 1;
            count_bin.scatter_add_(0, indices, torch::ones_like(confidences));
            conf_bin.scatter_add_(0, indices, confidences);
            conf_bin = torch::nan_to_num(conf_bin / count_bin);
            acc_bin.scatter_add_(0, indices, accuracies);
            acc_bin = torch::nan_to_num(acc_bin / count_bin);
            const auto prop_bin = count_bin / count_bin.sum();
            if (norm == calibration_norm_mode::l1) {
                return torch::sum(torch::abs(acc_bin - conf_bin) * prop_bin);
            }
            if (norm == calibration_norm_mode::max) {
                return torch::max(torch::abs(acc_bin - conf_bin));
            }
            const auto ce = torch::sum(torch::pow(acc_bin - conf_bin, 2) * prop_bin);
            return (ce > 0).item<bool>() ? torch::sqrt(ce) : torch::tensor(int64_t{0}); // torchmetrics: int64 0
        }

        // the options a class falls back to when its constructor rejects the given ones
        template<class Opt>
        inline auto option_fallback(const Opt &) -> Opt {
            Opt o;
            if constexpr (requires { o._num_classes; }) {
                o._num_classes = 2;
            }
            return o;
        }

        template<class Opt>
        inline auto calibration_error(const tensor_t &preds, const tensor_t &target, const Opt &o) -> tensor_t {
            if (auto e = ce_arg_error(o); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = ce_tensor_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            const auto [confidences, accuracies] = ce_update(preds, target, o);
            return ce_compute(confidences, accuracies, o._n_bins, o._norm);
        }

        // Keeps every batch's confidences / accuracies.
        template<class Opt>
        class calibration_error_class : public metric::detail::metric_base<calibration_error_class<Opt>> {
        public:
            explicit calibration_error_class(Opt opt = {}) : _opt(opt) {
                if (auto e = ce_arg_error(_opt); !e.empty()) {
                    _opt = option_fallback(_opt);
                    fail(std::move(e));
                }
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = ce_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto [confidences, accuracies] = ce_update(preds, target, _opt);
                _confidences.push_back(confidences.detach().clone()); // own the data: callers may reuse their buffer
                _accuracies.push_back(accuracies.detach().clone());
            }

            auto compute() const -> tensor_t {
                return squeeze_if_scalar(ce_compute(cat0(_confidences), cat0(_accuracies), _opt._n_bins, _opt._norm));
            }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            std::vector<tensor_t> _confidences, _accuracies;
        };

        // ---------------- HingeLoss ----------------
        inline auto hinge_arg_error(const binary_hinge_loss_option &) -> std::string { return {}; }
        inline auto hinge_arg_error(const multiclass_hinge_loss_option &o) -> std::string {
            return count_error("num_classes", o._num_classes);
        }
        inline auto hinge_tensor_error(const tensor_t &preds, const tensor_t &target, const binary_hinge_loss_option &o)
                -> std::string {
            const auto e = binary_tensor_error(preds, target, false, o._ignore_index);
            return e.empty() ? floating_preds_error(preds) : e;
        }
        inline auto hinge_tensor_error(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_hinge_loss_option &o) -> std::string {
            const auto e = multiclass_tensor_error(preds, target, o._num_classes, false, o._ignore_index);
            return e.empty() ? floating_preds_error(preds) : e;
        }

        inline auto hinge_measures(const tensor_t &margin, bool squared) -> tensor_t {
            const auto measures = torch::clamp(1 - margin, 0);
            return (squared ? measures.pow(2) : measures).sum(0);
        }
        // margin +x for the target class / -x otherwise
        inline auto signed_margin(const tensor_t &preds, const tensor_t &target) -> tensor_t {
            auto margin = torch::zeros_like(preds);
            margin.index_put_({target}, preds.index({target}));
            margin.index_put_({~target}, -preds.index({~target}));
            return margin;
        }

        // {measures summed over samples, sample count}: _binary_hinge_loss_update (on probabilities: logits go through
        // sigmoid first, as torchmetrics' formatting does)
        inline auto hinge_update(const tensor_t &preds_in, const tensor_t &target_in, const binary_hinge_loss_option &o)
                -> std::pair<tensor_t, tensor_t> {
            const auto [preds, target] = binary_format(preds_in, target_in, 0.0, o._ignore_index, false);
            return {hinge_measures(signed_margin(preds, target.to(torch::kBool)), o._squared),
                    torch::tensor(target.size(0), torch::device(target.device()))};
        }
        // _multiclass_hinge_loss_update: crammer-singer (target score - best other score) or one-vs-all per class
        inline auto hinge_update(const tensor_t &preds_in, const tensor_t &target_in,
                                 const multiclass_hinge_loss_option &o) -> std::pair<tensor_t, tensor_t> {
            auto [preds, target] = multiclass_format(preds_in, target_in, o._ignore_index, false);
            preds = normalize_logits(preds, true);
            const auto oh = torch::zeros(preds.sizes(), target.options())
                                    .scatter_(1, target.to(torch::kLong).unsqueeze(1), 1)
                                    .to(torch::kBool); // to_onehot(target, max(2, C))
            tensor_t margin;
            if (o._multiclass_mode == hinge_multiclass_mode::crammer_singer) {
                margin = preds.index({oh}) - std::get<0>(torch::max(preds.index({~oh}).view({preds.size(0), -1}), 1));
            } else {
                margin = signed_margin(preds, oh);
            }
            return {hinge_measures(margin, o._squared), torch::tensor(oh.size(0), torch::device(oh.device()))};
        }

        inline auto hinge_state(const binary_hinge_loss_option &) -> tensor_t { return torch::zeros({}); }
        inline auto hinge_state(const multiclass_hinge_loss_option &o) -> tensor_t {
            return o._multiclass_mode == hinge_multiclass_mode::crammer_singer ? torch::zeros({})
                                                                               : torch::zeros({o._num_classes});
        }

        template<class Opt>
        inline auto hinge_loss(const tensor_t &preds, const tensor_t &target, const Opt &o) -> tensor_t {
            if (auto e = hinge_arg_error(o); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = hinge_tensor_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            const auto [measures, total] = hinge_update(preds, target, o);
            return measures / total;
        }

        // torchmetrics' float32 measures / int64 total sums (float64 inputs keep a float64 sum, progress11 D5).
        template<class Opt>
        class hinge_loss_class : public metric::detail::metric_base<hinge_loss_class<Opt>> {
        public:
            explicit hinge_loss_class(Opt opt = {}) : _opt(opt) {
                if (auto e = hinge_arg_error(_opt); !e.empty()) {
                    _opt = option_fallback(_opt);
                    fail(std::move(e));
                }
                _measures = hinge_state(_opt);
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = hinge_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto [measures, total] = hinge_update(preds, target, _opt);
                _measures = _measures.to(measures.device()) + measures;
                _total = _total.to(total.device()) + total;
            }

            auto compute() const -> tensor_t { return squeeze_if_scalar(_measures / _total); }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            tensor_t _measures, _total = torch::zeros({}, torch::kLong);
        };

        // ---------------- ranking ----------------
        inline auto ranking_error(const tensor_t &preds, const tensor_t &target, const multilabel_ranking_option &o)
                -> std::string {
            const auto e = multilabel_tensor_error(preds, target, o._num_labels, false, o._ignore_index);
            return e.empty() ? floating_preds_error(preds) : e;
        }

        // _rank_data: 1-based max rank of each value (ties share the highest rank)
        inline auto rank_data(const tensor_t &x) -> tensor_t {
            const auto [values, inverse, counts] = torch::_unique2(x, true, true, true);
            return torch::cumsum(counts, 0).index({inverse});
        }

        enum class ranking_metric { coverage_error, average_precision, loss };

        // {score sum, element count}: _multilabel_coverage_error_update / _ranking_average_precision_update /
        // _ranking_loss_update on the [M, L] formatted scores (ignore_index entries set to -4 L like torchmetrics)
        inline auto ranking_update(ranking_metric k, const tensor_t &preds_in, const tensor_t &target_in,
                                   const multilabel_ranking_option &o) -> std::pair<tensor_t, int64_t> {
            const auto L = o._num_labels;
            const auto [preds, target] = multilabel_format(preds_in, target_in, L, 0.0, o._ignore_index, false);
            if (k == ranking_metric::coverage_error) {
                auto offset = torch::zeros_like(preds);
                offset.index_put_({target == 0}, preds.min().abs() + 10);
                const auto preds_min = std::get<0>((preds + offset).min(1));
                const auto coverage = (preds >= preds_min.unsqueeze(1)).sum(1).to(torch::kFloat32);
                return {coverage.sum(), coverage.numel()};
            }
            if (k == ranking_metric::average_precision) {
                const auto neg = -preds;
                auto score = torch::zeros({}, torch::device(neg.device()));
                for (int64_t i = 0; i < neg.size(0); ++i) {
                    const auto relevant = target[i] == 1;
                    const auto ranking = rank_data(neg[i].index({relevant})).to(torch::kFloat32);
                    if (ranking.numel() > 0 && ranking.numel() < L) {
                        score = score + (ranking / rank_data(neg[i]).index({relevant}).to(torch::kFloat32)).mean();
                    } else {
                        score = score + torch::ones_like(score);
                    }
                }
                return {score, neg.size(0)};
            }
            const auto n = preds.size(0);
            const auto relevant_all = target == 1;
            const auto num_relevant_all = relevant_all.sum(1);
            const auto mask = (num_relevant_all > 0) & (num_relevant_all < L);
            const auto p = preds.index({mask}), relevant = relevant_all.index({mask});
            const auto num_relevant = num_relevant_all.index({mask});
            if (p.size(0) == 0) {
                return {torch::zeros({}, torch::device(p.device())), 1};
            }
            const auto inverse = p.argsort(int64_t{1}).argsort(int64_t{1});
            const auto per_label_loss = ((L - inverse) * relevant).to(torch::kFloat32);
            const auto correction = 0.5 * num_relevant * (num_relevant + 1);
            const auto denom = num_relevant * (L - num_relevant);
            return {((per_label_loss.sum(1) - correction) / denom).sum(), n};
        }

        inline auto ranking(ranking_metric k, const tensor_t &preds, const tensor_t &target,
                            const multilabel_ranking_option &o) -> tensor_t {
            if (auto e = count_error("num_labels", o._num_labels); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = ranking_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            const auto [score, n] = ranking_update(k, preds, target, o);
            return score / n;
        }

        // torchmetrics' float32 measure sum and element count.
        template<ranking_metric K>
        class ranking_class : public metric::detail::metric_base<ranking_class<K>> {
        public:
            explicit ranking_class(multilabel_ranking_option opt = {}) : _opt(opt) {
                if (auto e = count_error("num_labels", _opt._num_labels); !e.empty()) {
                    _opt = multilabel_ranking_option().num_labels(2);
                    fail(std::move(e));
                }
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = ranking_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto [score, n] = ranking_update(K, preds, target, _opt);
                _measure = _measure.to(score.device()) + score;
                _total += n;
            }

            auto compute() const -> tensor_t { return squeeze_if_scalar(_measure / _total); }

            auto options() const -> const multilabel_ranking_option & { return _opt; }

        private:
            multilabel_ranking_option _opt;
            tensor_t _measure = torch::zeros({});
            int64_t _total = 0;
        };
    } // namespace detail

    // ---------------- CalibrationError ----------------
    // Expected (l1), RMS (l2) or maximum calibration error over n_bins equal-width confidence bins; binary confidence
    // is the positive-class probability, multiclass the top class probability (float32).
    inline auto binary_calibration_error(const tensor_t &preds, const tensor_t &target,
                                         const binary_calibration_error_option &opt = {}) -> tensor_t {
        return detail::calibration_error(preds, target, opt);
    }
    inline auto multiclass_calibration_error(const tensor_t &preds, const tensor_t &target,
                                             const multiclass_calibration_error_option &opt) -> tensor_t {
        return detail::calibration_error(preds, target, opt);
    }
    using BinaryCalibrationError = detail::calibration_error_class<binary_calibration_error_option>;
    using MulticlassCalibrationError = detail::calibration_error_class<multiclass_calibration_error_option>;

    // ---------------- HingeLoss ----------------
    // mean of max(0, 1 - margin) (squared: its square); multiclass one-vs-all: per class [C].
    inline auto binary_hinge_loss(const tensor_t &preds, const tensor_t &target,
                                  const binary_hinge_loss_option &opt = {}) -> tensor_t {
        return detail::hinge_loss(preds, target, opt);
    }
    inline auto multiclass_hinge_loss(const tensor_t &preds, const tensor_t &target,
                                      const multiclass_hinge_loss_option &opt) -> tensor_t {
        return detail::hinge_loss(preds, target, opt);
    }
    using BinaryHingeLoss = detail::hinge_loss_class<binary_hinge_loss_option>;
    using MulticlassHingeLoss = detail::hinge_loss_class<multiclass_hinge_loss_option>;

    // ---------------- multilabel ranking ----------------
    // CoverageError: mean count of labels ranked at or above the lowest-scored relevant label; RankingAveragePrecision:
    // label-ranking average precision (1 for a sample with no or only relevant labels); RankingLoss: mean fraction of
    // (relevant, irrelevant) pairs ranked the wrong way (samples with no or only relevant labels count as 0).
    inline auto multilabel_coverage_error(const tensor_t &preds, const tensor_t &target,
                                          const multilabel_ranking_option &opt) -> tensor_t {
        return detail::ranking(detail::ranking_metric::coverage_error, preds, target, opt);
    }
    inline auto multilabel_ranking_average_precision(const tensor_t &preds, const tensor_t &target,
                                                     const multilabel_ranking_option &opt) -> tensor_t {
        return detail::ranking(detail::ranking_metric::average_precision, preds, target, opt);
    }
    inline auto multilabel_ranking_loss(const tensor_t &preds, const tensor_t &target,
                                        const multilabel_ranking_option &opt) -> tensor_t {
        return detail::ranking(detail::ranking_metric::loss, preds, target, opt);
    }
    using MultilabelCoverageError = detail::ranking_class<detail::ranking_metric::coverage_error>;
    using MultilabelRankingAveragePrecision = detail::ranking_class<detail::ranking_metric::average_precision>;
    using MultilabelRankingLoss = detail::ranking_class<detail::ranking_metric::loss>;
} // namespace torchmedia::metric::classification
#endif // LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_MISC_HPP
