#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_CURVE_HPP
#define LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_CURVE_HPP
// Classification metrics on score curves (develop_log/2026-10-24/progress02): torchmetrics 1.9 PrecisionRecallCurve,
// ROC, AUROC, AveragePrecision, LogAUC, EER, PrecisionAtFixedRecall, RecallAtFixedPrecision, SensitivityAtSpecificity
// and SpecificityAtSensitivity (binary / multiclass / multilabel; functionals and classes), the same libtorch calls in
// the same order. thresholds {} (torchmetrics None) gives exact curves over every distinct score (the classes keep
// every batch); a count n (linspace(0, 1, n)), a list or a 1-D tensor gives binned curves (the classes keep a
// [T, (C,) 2, 2] confusion-matrix state). Floating preds outside [0, 1] are logits: sigmoid (binary / multilabel) or
// softmax over the classes (multiclass).
#include <initializer_list>
#include <variant>
#include "_classification_stat.hpp"

namespace torchmedia::metric::classification {
    // {} = exact (every distinct score); a count n >= 2 = torch.linspace(0, 1, n); a list of floats in [0, 1] (float32,
    // like torch.tensor(list)); a non-empty 1-D tensor.
    using thresholds_spec = std::variant<std::monostate, int64_t, std::vector<double>, tensor_t>;

    // torchmetrics' (precision, recall, thresholds) / (fpr, tpr, thresholds) tuples.
    using curve_t = std::tuple<tensor_t, tensor_t, tensor_t>;
    // Per class (multiclass / multilabel): one tensor per class — exact curves as torchmetrics' lists, binned curves as
    // the rows of torchmetrics' [C, T (+ 1)] tensors with the shared thresholds [T] repeated; an averaged curve
    // (average micro / macro) is one tensor each.
    using curve_list_t = std::tuple<std::vector<tensor_t>, std::vector<tensor_t>, std::vector<tensor_t>>;

    // PrecisionRecallCurve, ROC, AUROC, AveragePrecision, LogAUC, EER and the four fixed-point metrics.
    typedef struct binary_curve_option {
        thresholds_spec _thresholds;
        std::optional<int64_t> _ignore_index;
        std::optional<double> _max_fpr; // AUROC: McClish-standardized partial AUC over [0, max_fpr], (0, 1]
        std::pair<double, double> _fpr_range{0.001, 0.1}; // LogAUC: 0 <= lo < hi <= 1
        std::optional<double> _min_precision; // RecallAtFixedPrecision (required, [0, 1])
        std::optional<double> _min_recall; // PrecisionAtFixedRecall
        std::optional<double> _min_specificity; // SensitivityAtSpecificity
        std::optional<double> _min_sensitivity; // SpecificityAtSensitivity

        auto thresholds(int64_t v) -> binary_curve_option & {
            _thresholds = v;
            return *this;
        }
        auto thresholds(std::vector<double> v) -> binary_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto thresholds(std::initializer_list<double> v) -> binary_curve_option & { // a braced list, also {0.5}
            _thresholds = std::vector<double>(v);
            return *this;
        }
        auto thresholds(tensor_t v) -> binary_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> binary_curve_option & {
            _ignore_index = v;
            return *this;
        }
        auto max_fpr(std::optional<double> v) -> binary_curve_option & {
            _max_fpr = v;
            return *this;
        }
        auto fpr_range(std::pair<double, double> v) -> binary_curve_option & {
            _fpr_range = v;
            return *this;
        }
        auto min_precision(std::optional<double> v) -> binary_curve_option & {
            _min_precision = v;
            return *this;
        }
        auto min_recall(std::optional<double> v) -> binary_curve_option & {
            _min_recall = v;
            return *this;
        }
        auto min_specificity(std::optional<double> v) -> binary_curve_option & {
            _min_specificity = v;
            return *this;
        }
        auto min_sensitivity(std::optional<double> v) -> binary_curve_option & {
            _min_sensitivity = v;
            return *this;
        }
    } binary_curve_option_t;

    // PrecisionRecallCurve, ROC, EER (average: none = per class, micro / macro = one averaged curve) and the four
    // fixed-point metrics (average none).
    typedef struct multiclass_curve_option {
        int64_t _num_classes = 0; // required (>= 2)
        thresholds_spec _thresholds;
        average_mode _average = average_mode::none;
        std::optional<int64_t> _ignore_index;
        std::optional<double> _min_precision;
        std::optional<double> _min_recall;
        std::optional<double> _min_specificity;
        std::optional<double> _min_sensitivity;

        auto num_classes(int64_t v) -> multiclass_curve_option & {
            _num_classes = v;
            return *this;
        }
        auto thresholds(int64_t v) -> multiclass_curve_option & {
            _thresholds = v;
            return *this;
        }
        auto thresholds(std::vector<double> v) -> multiclass_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto thresholds(std::initializer_list<double> v) -> multiclass_curve_option & { // a braced list, also {0.5}
            _thresholds = std::vector<double>(v);
            return *this;
        }
        auto thresholds(tensor_t v) -> multiclass_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto average(average_mode v) -> multiclass_curve_option & {
            _average = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_curve_option & {
            _ignore_index = v;
            return *this;
        }
        auto min_precision(std::optional<double> v) -> multiclass_curve_option & {
            _min_precision = v;
            return *this;
        }
        auto min_recall(std::optional<double> v) -> multiclass_curve_option & {
            _min_recall = v;
            return *this;
        }
        auto min_specificity(std::optional<double> v) -> multiclass_curve_option & {
            _min_specificity = v;
            return *this;
        }
        auto min_sensitivity(std::optional<double> v) -> multiclass_curve_option & {
            _min_sensitivity = v;
            return *this;
        }
    } multiclass_curve_option_t;

    // PrecisionRecallCurve, ROC, EER and the four fixed-point metrics.
    typedef struct multilabel_curve_option {
        int64_t _num_labels = 0; // required (>= 2)
        thresholds_spec _thresholds;
        std::optional<int64_t> _ignore_index;
        std::optional<double> _min_precision;
        std::optional<double> _min_recall;
        std::optional<double> _min_specificity;
        std::optional<double> _min_sensitivity;

        auto num_labels(int64_t v) -> multilabel_curve_option & {
            _num_labels = v;
            return *this;
        }
        auto thresholds(int64_t v) -> multilabel_curve_option & {
            _thresholds = v;
            return *this;
        }
        auto thresholds(std::vector<double> v) -> multilabel_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto thresholds(std::initializer_list<double> v) -> multilabel_curve_option & { // a braced list, also {0.5}
            _thresholds = std::vector<double>(v);
            return *this;
        }
        auto thresholds(tensor_t v) -> multilabel_curve_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multilabel_curve_option & {
            _ignore_index = v;
            return *this;
        }
        auto min_precision(std::optional<double> v) -> multilabel_curve_option & {
            _min_precision = v;
            return *this;
        }
        auto min_recall(std::optional<double> v) -> multilabel_curve_option & {
            _min_recall = v;
            return *this;
        }
        auto min_specificity(std::optional<double> v) -> multilabel_curve_option & {
            _min_specificity = v;
            return *this;
        }
        auto min_sensitivity(std::optional<double> v) -> multilabel_curve_option & {
            _min_sensitivity = v;
            return *this;
        }
    } multilabel_curve_option_t;

    // AUROC, AveragePrecision (average macro / weighted / none), LogAUC (macro / none; torchmetrics' LogAUC classes
    // default to none, its functionals to macro).
    typedef struct multiclass_auc_option {
        int64_t _num_classes = 0; // required (>= 2)
        average_mode _average = average_mode::macro;
        thresholds_spec _thresholds;
        std::optional<int64_t> _ignore_index;
        std::pair<double, double> _fpr_range{0.001, 0.1}; // LogAUC

        auto num_classes(int64_t v) -> multiclass_auc_option & {
            _num_classes = v;
            return *this;
        }
        auto average(average_mode v) -> multiclass_auc_option & {
            _average = v;
            return *this;
        }
        auto thresholds(int64_t v) -> multiclass_auc_option & {
            _thresholds = v;
            return *this;
        }
        auto thresholds(std::vector<double> v) -> multiclass_auc_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto thresholds(std::initializer_list<double> v) -> multiclass_auc_option & { // a braced list, also {0.5}
            _thresholds = std::vector<double>(v);
            return *this;
        }
        auto thresholds(tensor_t v) -> multiclass_auc_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_auc_option & {
            _ignore_index = v;
            return *this;
        }
        auto fpr_range(std::pair<double, double> v) -> multiclass_auc_option & {
            _fpr_range = v;
            return *this;
        }
    } multiclass_auc_option_t;

    // AUROC, AveragePrecision (average micro / macro / weighted / none), LogAUC (macro / none).
    typedef struct multilabel_auc_option {
        int64_t _num_labels = 0; // required (>= 2)
        average_mode _average = average_mode::macro;
        thresholds_spec _thresholds;
        std::optional<int64_t> _ignore_index;
        std::pair<double, double> _fpr_range{0.001, 0.1};

        auto num_labels(int64_t v) -> multilabel_auc_option & {
            _num_labels = v;
            return *this;
        }
        auto average(average_mode v) -> multilabel_auc_option & {
            _average = v;
            return *this;
        }
        auto thresholds(int64_t v) -> multilabel_auc_option & {
            _thresholds = v;
            return *this;
        }
        auto thresholds(std::vector<double> v) -> multilabel_auc_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto thresholds(std::initializer_list<double> v) -> multilabel_auc_option & { // a braced list, also {0.5}
            _thresholds = std::vector<double>(v);
            return *this;
        }
        auto thresholds(tensor_t v) -> multilabel_auc_option & {
            _thresholds = std::move(v);
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multilabel_auc_option & {
            _ignore_index = v;
            return *this;
        }
        auto fpr_range(std::pair<double, double> v) -> multilabel_auc_option & {
            _fpr_range = v;
            return *this;
        }
    } multilabel_auc_option_t;

    namespace detail {
        enum class curve_metric {
            pr_curve,
            roc,
            auroc,
            average_precision,
            logauc,
            eer,
            precision_at_recall,
            recall_at_precision,
            sensitivity_at_specificity,
            specificity_at_sensitivity
        };

        inline auto fixed_point(curve_metric k) -> bool { return k >= curve_metric::precision_at_recall; }

        // ---------------- argument checks ----------------
        inline auto thresholds_error(const thresholds_spec &t) -> std::string {
            if (const auto *n = std::get_if<int64_t>(&t); n && *n < 2) {
                return std::format(
                        "If argument `thresholds` is an integer, expected it to be larger than 1, but got {}", *n);
            }
            if (const auto *v = std::get_if<std::vector<double>>(&t)) {
                const bool ok =
                        !v->empty() && std::all_of(v->begin(), v->end(), [](double x) { return x >= 0 && x <= 1; });
                if (!ok) {
                    return "If argument `thresholds` is a list, expected a non-empty list of floats in the [0,1] range";
                }
            }
            if (const auto *x = std::get_if<tensor_t>(&t); x && (!x->defined() || x->dim() != 1 || x->numel() == 0)) {
                return "If argument `thresholds` is a tensor, expected a non-empty 1d tensor";
            }
            return {};
        }

        inline auto fixed_error(curve_metric k, std::optional<double> min_precision, std::optional<double> min_recall,
                                std::optional<double> min_specificity, std::optional<double> min_sensitivity)
                -> std::string {
            const std::pair<const char *, std::optional<double>> arg =
                    k == curve_metric::precision_at_recall          ? std::pair{"min_recall", min_recall}
                    : k == curve_metric::recall_at_precision        ? std::pair{"min_precision", min_precision}
                    : k == curve_metric::sensitivity_at_specificity ? std::pair{"min_specificity", min_specificity}
                                                                    : std::pair{"min_sensitivity", min_sensitivity};
            if (!fixed_point(k) || (arg.second && *arg.second >= 0 && *arg.second <= 1)) {
                return {};
            }
            return std::format("Expected argument `{}` to be a float in the [0,1] range, but got {}", arg.first,
                               arg.second ? std::to_string(*arg.second) : "none");
        }

        inline auto fpr_range_error(curve_metric k, std::pair<double, double> r) -> std::string {
            return k != curve_metric::logauc || (0 <= r.first && r.first < r.second && r.second <= 1)
                           ? std::string{}
                           : std::format("The `fpr_range` should be a tuple of two floats in the range [0, 1], but got "
                                         "({}, {}).",
                                         r.first, r.second);
        }

        inline auto first_error(std::initializer_list<std::string> errors) -> std::string {
            for (const auto &e: errors) {
                if (!e.empty()) {
                    return e;
                }
            }
            return {};
        }

        inline auto curve_arg_error(const binary_curve_option &o, curve_metric k) -> std::string {
            const bool max_fpr_ok = k != curve_metric::auroc || !o._max_fpr || (*o._max_fpr > 0 && *o._max_fpr <= 1);
            return first_error(
                    {thresholds_error(o._thresholds),
                     max_fpr_ok ? std::string{}
                                : std::format("Arguments `max_fpr` should be a float in range (0, 1], but "
                                              "got: {}",
                                              *o._max_fpr),
                     fpr_range_error(k, o._fpr_range),
                     fixed_error(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity)});
        }
        inline auto curve_arg_error(const multiclass_curve_option &o, curve_metric k) -> std::string {
            const bool avg_ok =
                    fixed_point(k) ? o._average == average_mode::none : o._average != average_mode::weighted;
            return first_error(
                    {count_error("num_classes", o._num_classes), thresholds_error(o._thresholds),
                     avg_ok ? std::string{}
                            : "Expected argument `average` to be one of None, 'micro' or 'macro' (None for "
                              "the fixed-point metrics)",
                     fixed_error(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity)});
        }
        inline auto curve_arg_error(const multilabel_curve_option &o, curve_metric k) -> std::string {
            return first_error(
                    {count_error("num_labels", o._num_labels), thresholds_error(o._thresholds),
                     fixed_error(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity)});
        }
        inline auto auc_average_error(average_mode a, curve_metric k, bool multilabel) -> std::string {
            const bool ok = k == curve_metric::logauc ? a == average_mode::macro || a == average_mode::none
                                                      : multilabel || a != average_mode::micro;
            return ok ? std::string{}
                      : std::format("Expected argument `average` to be one of {}",
                                    k == curve_metric::logauc ? "'macro', 'none'"
                                    : multilabel              ? "'micro', 'macro', 'weighted', 'none'"
                                                              : "'macro', 'weighted', 'none'");
        }
        inline auto curve_arg_error(const multiclass_auc_option &o, curve_metric k) -> std::string {
            return first_error({count_error("num_classes", o._num_classes), thresholds_error(o._thresholds),
                                auc_average_error(o._average, k, false), fpr_range_error(k, o._fpr_range)});
        }
        inline auto curve_arg_error(const multilabel_auc_option &o, curve_metric k) -> std::string {
            return first_error({count_error("num_labels", o._num_labels), thresholds_error(o._thresholds),
                                auc_average_error(o._average, k, true), fpr_range_error(k, o._fpr_range)});
        }

        // ---------------- tensor checks ----------------
        inline auto binary_curve_tensor_error(const tensor_t &preds, const tensor_t &target, std::optional<int64_t> ii)
                -> std::string {
            if (auto e = same_shape_error(preds, target); !e.empty()) {
                return e;
            }
            if (target.is_floating_point()) {
                return "Expected argument `target` to be an int or long tensor with ground truth labels";
            }
            if (!preds.is_floating_point()) {
                return "Expected argument `preds` to be an floating tensor with probability/logit scores";
            }
            return binary_values_error(preds, target, ii);
        }
        inline auto curve_tensor_error(const tensor_t &preds, const tensor_t &target, const binary_curve_option &o)
                -> std::string {
            return binary_curve_tensor_error(preds, target, o._ignore_index);
        }
        inline auto multiclass_curve_tensor_error(const tensor_t &preds, const tensor_t &target, int64_t C,
                                                  std::optional<int64_t> ii) -> std::string {
            if (preds.dim() != target.dim() + 1) {
                return std::format("Expected `preds` to have one more dimension than `target` but got {} and {}",
                                   preds.dim(), target.dim());
            }
            if (target.is_floating_point()) {
                return "Expected argument `target` to be an int or long tensor";
            }
            if (!preds.is_floating_point()) {
                return "Expected `preds` to be a float tensor";
            }
            if (preds.size(1) != C) {
                return std::format("Expected `preds.shape[1]` to be equal to the number of classes but got {} and {}.",
                                   preds.size(1), C);
            }
            if (preds.size(0) != target.size(0) || preds.sizes().slice(2) != target.sizes().slice(1)) {
                return std::format(
                        "Expected the shape of `preds` should be (N, C, ...) and the shape of `target` should "
                        "be (N, ...) but got {} and {}",
                        c10::str(preds.sizes()), c10::str(target.sizes()));
            }
            return unique_count(target) > C + (ii ? 1 : 0)
                           ? "Detected more unique values in `target` than `num_classes`."
                           : std::string{};
        }
        inline auto curve_tensor_error(const tensor_t &preds, const tensor_t &target, const multiclass_curve_option &o)
                -> std::string {
            return multiclass_curve_tensor_error(preds, target, o._num_classes, o._ignore_index);
        }
        inline auto curve_tensor_error(const tensor_t &preds, const tensor_t &target, const multiclass_auc_option &o)
                -> std::string {
            return multiclass_curve_tensor_error(preds, target, o._num_classes, o._ignore_index);
        }
        inline auto multilabel_curve_tensor_error(const tensor_t &preds, const tensor_t &target, int64_t L,
                                                  std::optional<int64_t> ii) -> std::string {
            if (auto e = binary_curve_tensor_error(preds, target, ii); !e.empty()) {
                return e;
            }
            return preds.dim() < 2 || preds.size(1) != L
                           ? std::format(
                                     "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number "
                                     "of labels {}, but got shape {}",
                                     L, c10::str(preds.sizes()))
                           : std::string{};
        }
        inline auto curve_tensor_error(const tensor_t &preds, const tensor_t &target, const multilabel_curve_option &o)
                -> std::string {
            return multilabel_curve_tensor_error(preds, target, o._num_labels, o._ignore_index);
        }
        inline auto curve_tensor_error(const tensor_t &preds, const tensor_t &target, const multilabel_auc_option &o)
                -> std::string {
            return multilabel_curve_tensor_error(preds, target, o._num_labels, o._ignore_index);
        }

        // _adjust_threshold_arg; undefined = exact
        inline auto adjust_thresholds(const thresholds_spec &t, const torch::Device &device) -> tensor_t {
            if (const auto *n = std::get_if<int64_t>(&t)) {
                return torch::linspace(0, 1, *n, torch::device(device));
            }
            if (const auto *v = std::get_if<std::vector<double>>(&t)) {
                return torch::tensor(*v, torch::kFloat64).to(device, torch::kFloat32);
            }
            const auto *x = std::get_if<tensor_t>(&t);
            return x ? x->to(device) : tensor_t{};
        }

        // ---------------- formatting and update ----------------
        // torchmetrics' state: (preds, target) for exact curves, or the binned confusion matrices [T, (C,) 2, 2].
        struct curve_state {
            tensor_t preds, target, confmat;
        };

        // confmat[t, c] = [[tn, fp], [fn, tp]] of preds >= thresholds[t] (rows where valid is false skipped). promote:
        // torchmetrics' vectorized update compares in the promoted dtype of preds and the thresholds; its loop (large
        // inputs) compares against a 0-d threshold, i.e. in preds' dtype.
        inline auto binned_confmat(const tensor_t &preds, const tensor_t &pos_in, const tensor_t &valid,
                                   const tensor_t &thresholds, bool promote) -> tensor_t {
            const auto pos = valid.defined() ? pos_in & valid : pos_in;
            const auto neg = valid.defined() ? ~pos_in & valid : ~pos_in;
            std::vector<tensor_t> rows;
            for (int64_t i = 0; i < thresholds.size(0); ++i) {
                const auto hit = preds >= (promote ? thresholds.slice(0, i, i + 1) : thresholds[i]);
                rows.push_back(torch::stack(
                        {(neg & ~hit).sum(0), (neg & hit).sum(0), (pos & ~hit).sum(0), (pos & hit).sum(0)}, -1));
            }
            return torch::stack(rows).view({thresholds.size(0), preds.size(1), 2, 2});
        }

        inline auto drop_ignored(tensor_t &preds, tensor_t &target, std::optional<int64_t> ii) -> void {
            if (ii) {
                const auto keep = target != *ii;
                preds = preds.index({keep});
                target = target.index({keep});
            }
        }

        // _binary_precision_recall_curve_format + _update
        inline auto curve_update(const tensor_t &preds_in, const tensor_t &target_in, const tensor_t &thresholds,
                                 const binary_curve_option &o) -> curve_state {
            auto preds = preds_in.flatten(), target = target_in.flatten();
            drop_ignored(preds, target, o._ignore_index);
            preds = normalize_logits(preds, false);
            if (!thresholds.defined()) {
                return {preds, target, {}};
            }
            return {{},
                    {},
                    binned_confmat(preds.unsqueeze(1), (target == 1).unsqueeze(1), {}, thresholds,
                                   preds.numel() <= 50000)
                            .squeeze(1)};
        }
        // _multiclass_precision_recall_curve_format + _update ([M, C] scores; micro: flattened one-vs-rest)
        inline auto multiclass_curve_update(const tensor_t &preds_in, const tensor_t &target_in,
                                            const tensor_t &thresholds, int64_t C, std::optional<int64_t> ii,
                                            bool micro) -> curve_state {
            auto preds = preds_in.transpose(0, 1).reshape({C, -1}).t(), target = target_in.flatten();
            drop_ignored(preds, target, ii);
            preds = normalize_logits(preds, true);
            if (micro) {
                preds = preds.flatten();
                target = torch::one_hot(target, C).flatten();
            }
            if (!thresholds.defined()) {
                return {preds, target, {}};
            }
            if (micro) {
                return {{},
                        {},
                        binned_confmat(preds.unsqueeze(1), (target == 1).unsqueeze(1), {}, thresholds,
                                       preds.numel() <= 50000)
                                .squeeze(1)};
            }
            return {{},
                    {},
                    binned_confmat(preds, torch::one_hot(target, C).to(torch::kBool), {}, thresholds,
                                   preds.numel() * C <= 1000000)};
        }
        inline auto curve_update(const tensor_t &preds, const tensor_t &target, const tensor_t &thresholds,
                                 const multiclass_curve_option &o) -> curve_state {
            return multiclass_curve_update(preds, target, thresholds, o._num_classes, o._ignore_index,
                                           o._average == average_mode::micro);
        }
        inline auto curve_update(const tensor_t &preds, const tensor_t &target, const tensor_t &thresholds,
                                 const multiclass_auc_option &o) -> curve_state {
            return multiclass_curve_update(preds, target, thresholds, o._num_classes, o._ignore_index, false);
        }
        // _multilabel_precision_recall_curve_format + _update ([M, L]; exact states keep the ignored entries)
        inline auto multilabel_curve_update(const tensor_t &preds_in, const tensor_t &target_in,
                                            const tensor_t &thresholds, int64_t L, std::optional<int64_t> ii)
                -> curve_state {
            auto preds = normalize_logits(preds_in.transpose(0, 1).reshape({L, -1}).t(), false);
            auto target = target_in.transpose(0, 1).reshape({L, -1}).t();
            if (!thresholds.defined()) {
                return {preds, target, {}};
            }
            // contiguous: MPS (torch 2.5.1) gets & of two non-contiguous bool views ([N, L, ...] inputs) wrong
            const auto p = preds.contiguous(), t = target.contiguous();
            const auto valid = ii ? t != *ii : tensor_t{};
            return {{}, {}, binned_confmat(p, t == 1, valid, thresholds, true)};
        }
        inline auto curve_update(const tensor_t &preds, const tensor_t &target, const tensor_t &thresholds,
                                 const multilabel_curve_option &o) -> curve_state {
            return multilabel_curve_update(preds, target, thresholds, o._num_labels, o._ignore_index);
        }
        inline auto curve_update(const tensor_t &preds, const tensor_t &target, const tensor_t &thresholds,
                                 const multilabel_auc_option &o) -> curve_state {
            return multilabel_curve_update(preds, target, thresholds, o._num_labels, o._ignore_index);
        }

        // ---------------- curves ----------------
        // torchmetrics' curve triple: 1-D tensors, per-class tensors [C, n] with shared thresholds, or per-class lists.
        struct curves {
            tensor_t x, y, t;
            std::vector<tensor_t> xs, ys, ts;

            curves() = default;
            curves(tensor_t x_, tensor_t y_, tensor_t t_) : x(std::move(x_)), y(std::move(y_)), t(std::move(t_)) {}

            auto count() const -> int64_t { return x.defined() ? (x.dim() == 2 ? x.size(0) : 1) : int64_t(xs.size()); }
            auto at(int64_t i) const -> curve_t {
                if (!x.defined()) {
                    return {xs[i], ys[i], ts[i]};
                }
                return x.dim() == 2 ? curve_t{x[i], y[i], t} : curve_t{x, y, t};
            }
            // squeeze: Metric.compute's _squeeze_if_scalar on torchmetrics' objects (rows of a stacked [C, n] tensor
            // are never squeezed)
            auto list(bool squeeze) const -> curve_list_t {
                const auto sq = [&](const tensor_t &v, bool row) { return squeeze && !row ? squeeze_if_scalar(v) : v; };
                curve_list_t out;
                for (int64_t i = 0; i < count(); ++i) {
                    const auto [a, b, c] = at(i);
                    const bool row = x.defined() && x.dim() == 2;
                    std::get<0>(out).push_back(sq(a, row));
                    std::get<1>(out).push_back(sq(b, row));
                    std::get<2>(out).push_back(sq(c, false));
                }
                return out;
            }
        };

        // _binary_clf_curve: (fps, tps, thresholds) at each distinct score, highest first
        inline auto clf_curve(const tensor_t &preds_in, const tensor_t &target_in, int64_t pos_label) -> curve_t {
            const auto order = torch::argsort(preds_in, -1, true);
            const auto preds = preds_in.index({order}), target = target_in.index({order});
            const auto distinct = torch::where(preds.slice(0, 1) - preds.slice(0, 0, -1))[0];
            const auto idx = torch::constant_pad_nd(distinct, {0, 1}, target.size(0) - 1);
            const auto tps = torch::cumsum((target == pos_label).to(torch::kLong) * 1.0, 0).index({idx});
            return {1 + idx - tps, tps, preds.index({idx})};
        }

        // _binary_precision_recall_curve_compute
        inline auto binary_pr(const curve_state &s, const tensor_t &thresholds, int64_t pos_label = 1) -> curves {
            const auto nan = std::nan("");
            if (s.confmat.defined()) {
                const auto tps = s.confmat.select(-2, 1).select(-1, 1), fps = s.confmat.select(-2, 0).select(-1, 1),
                           fns = s.confmat.select(-2, 1).select(-1, 0);
                const auto precision = safe_divide(tps, tps + fps, nan), recall = safe_divide(tps, tps + fns, nan);
                return {torch::cat({precision, torch::ones({1}, precision.options())}),
                        torch::cat({recall, torch::zeros({1}, recall.options())}), thresholds};
            }
            const auto [fps, tps, thr] = clf_curve(s.preds, s.target, pos_label);
            const auto precision = tps / (tps + fps);
            auto recall = tps / tps[-1];
            if ((s.target == 0).all().item<bool>()) { // torchmetrics checks target == 0 whatever pos_label is
                recall = torch::ones_like(recall);
            }
            return {torch::cat({precision.flip(0), torch::ones({1}, precision.options())}),
                    torch::cat({recall.flip(0), torch::zeros({1}, recall.options())}), thr.flip(0)};
        }

        // _binary_roc_compute
        inline auto binary_roc(const curve_state &s, const tensor_t &thresholds, int64_t pos_label = 1) -> curves {
            if (s.confmat.defined()) {
                const auto tps = s.confmat.select(-2, 1).select(-1, 1), fps = s.confmat.select(-2, 0).select(-1, 1),
                           fns = s.confmat.select(-2, 1).select(-1, 0), tns = s.confmat.select(-2, 0).select(-1, 0);
                return {safe_divide(fps, fps + tns).flip(0), safe_divide(tps, tps + fns).flip(0), thresholds.flip(0)};
            }
            auto [fps, tps, thr] = clf_curve(s.preds, s.target, pos_label);
            tps = torch::cat({torch::zeros({1}, tps.options()), tps});
            fps = torch::cat({torch::zeros({1}, fps.options()), fps});
            thr = torch::cat({torch::ones({1}, thr.options()), thr});
            const auto fpr = (fps[-1] <= 0).item<bool>() ? torch::zeros_like(thr) : fps / fps[-1];
            const auto tpr = (tps[-1] <= 0).item<bool>() ? torch::zeros_like(thr) : tps / tps[-1];
            return {fpr, tpr, thr};
        }

        // torchmetrics.utilities.compute.interp
        inline auto interp(const tensor_t &x, const tensor_t &xp, const tensor_t &fp) -> tensor_t {
            const auto m = safe_divide(fp.slice(0, 1) - fp.slice(0, 0, -1), xp.slice(0, 1) - xp.slice(0, 0, -1));
            const auto b = fp.slice(0, 0, -1) - m * xp.slice(0, 0, -1);
            auto indices = torch::sum(torch::ge(x.unsqueeze(1), xp.unsqueeze(0)), 1) - 1;
            indices = torch::clamp(indices, 0, m.size(0) - 1);
            return m.index({indices}) * x + b.index({indices});
        }

        // per-class curves of a multiclass / multilabel state ([T, C, 2, 2] binned or [M, C] exact): roc picks
        // _binary_roc_compute, else _binary_precision_recall_curve_compute; multilabel drops ignore_index per label.
        inline auto per_class(const curve_state &s, const tensor_t &thresholds, int64_t C, bool roc, bool multilabel,
                              std::optional<int64_t> ii) -> curves {
            if (s.confmat.defined()) {
                const auto tps = s.confmat.select(-2, 1).select(-1, 1), fps = s.confmat.select(-2, 0).select(-1, 1),
                           fns = s.confmat.select(-2, 1).select(-1, 0), tns = s.confmat.select(-2, 0).select(-1, 0);
                if (roc) {
                    return {safe_divide(fps, fps + tns).flip(0).t(), safe_divide(tps, tps + fns).flip(0).t(),
                            thresholds.flip(0)};
                }
                const auto nan = std::nan("");
                auto precision = safe_divide(tps, tps + fps, nan), recall = safe_divide(tps, tps + fns, nan);
                precision = torch::cat({precision, torch::ones({1, C}, precision.options())});
                recall = torch::cat({recall, torch::zeros({1, C}, recall.options())});
                return {precision.t(), recall.t(), thresholds};
            }
            curves out;
            for (int64_t i = 0; i < C; ++i) {
                curve_state si{s.preds.select(1, i), multilabel ? s.target.select(1, i) : s.target, {}};
                if (multilabel && ii) {
                    const auto keep = si.target != *ii;
                    si = {si.preds.index({keep}), si.target.index({keep}), {}};
                }
                const auto pos_label = multilabel ? 1 : i;
                const auto c = roc ? binary_roc(si, {}, pos_label) : binary_pr(si, {}, pos_label);
                out.xs.push_back(c.x);
                out.ys.push_back(c.y);
                out.ts.push_back(c.t);
            }
            return out;
        }

        // the macro-averaged curve of _multiclass_precision_recall_curve_compute / _multiclass_roc_compute
        inline auto macro_curve(const curves &c, int64_t C, bool roc) -> curves {
            const bool stacked = c.x.defined();
            auto thr = stacked ? c.t.repeat({C}) : torch::cat(c.ts, 0);
            thr = std::get<0>(thr.sort(-1, roc));
            const auto mean_x = std::get<0>((stacked ? c.x.flatten() : torch::cat(c.xs, 0)).sort());
            auto mean_y = torch::zeros_like(mean_x);
            for (int64_t i = 0; i < C; ++i) {
                const auto [xi, yi, ti] = c.at(i);
                mean_y.add_(interp(mean_x, xi, yi)); // in place, like torchmetrics: keeps mean_x's dtype
            }
            return {mean_x, mean_y.div_(C), thr};
        }

        // multiclass precision-recall (roc: ROC) curves with torchmetrics' average (micro / macro / none)
        inline auto multiclass_curves(const curve_state &s, const tensor_t &thresholds, int64_t C, average_mode average,
                                      bool roc) -> curves {
            if (average == average_mode::micro) {
                return roc ? binary_roc(s, thresholds) : binary_pr(s, thresholds);
            }
            const auto c = per_class(s, thresholds, C, roc, false, std::nullopt);
            return average == average_mode::macro ? macro_curve(c, C, roc) : c;
        }

        // ---------------- reductions ----------------
        // _reduce_auroc / _reduce_average_precision / _reduce_logauc tail: none, macro (NaN classes left out), weighted
        inline auto reduce_classes(const tensor_t &res, average_mode average, const tensor_t &weights) -> tensor_t {
            if (average == average_mode::none) {
                return res;
            }
            const auto idx = ~torch::isnan(res);
            if (average == average_mode::macro) {
                return res.index({idx}).mean();
            }
            const auto w = safe_divide(weights.index({idx}), weights.index({idx}).sum());
            return (res.index({idx}) * w).sum();
        }

        // class weights: positives per class (exact: target counts; binned: the first threshold's positives)
        inline auto class_weights(const curve_state &s, int64_t C, bool multilabel) -> tensor_t {
            if (s.confmat.defined()) {
                return s.confmat[0].select(1, 1).sum(-1);
            }
            return multilabel ? (s.target == 1).sum(0).to(torch::kFloat32) : bincount(s.target, C).to(torch::kFloat32);
        }

        inline auto auroc_reduce(const curves &roc, average_mode average, const tensor_t &weights) -> tensor_t {
            tensor_t res;
            if (roc.x.defined()) {
                res = torch::trapz(roc.y, roc.x, 1);
            } else {
                std::vector<tensor_t> v;
                for (size_t i = 0; i < roc.xs.size(); ++i) {
                    v.push_back(torch::trapz(roc.ys[i], roc.xs[i], -1));
                }
                res = torch::stack(v);
            }
            return reduce_classes(res, average, weights);
        }

        // _binary_auroc_compute
        inline auto binary_auroc(const curve_state &s, const tensor_t &thresholds, std::optional<double> max_fpr)
                -> tensor_t {
            const auto roc = binary_roc(s, thresholds);
            const auto &fpr = roc.x, &tpr = roc.y;
            if (!max_fpr || *max_fpr == 1 || (fpr.sum() == 0).item<bool>() || (tpr.sum() == 0).item<bool>()) {
                return torch::trapz(tpr, fpr, -1);
            }
            const auto max_area = torch::tensor(*max_fpr, torch::device(fpr.device()).dtype(torch::kFloat32));
            const auto stop = torch::bucketize(max_area, fpr, true, true).item<int64_t>();
            const auto weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1]);
            const auto interp_tpr = torch::lerp(tpr[stop - 1], tpr[stop], weight);
            const auto partial = torch::trapz(torch::cat({tpr.slice(0, 0, stop), interp_tpr.view(1)}),
                                              torch::cat({fpr.slice(0, 0, stop), max_area.view(1)}), -1);
            const auto min_area = 0.5 * max_area.pow(2);
            return 0.5 * (1 + (partial - min_area) / (max_area - min_area));
        }

        // _binary_average_precision_compute
        inline auto binary_ap(const curve_state &s, const tensor_t &thresholds) -> tensor_t {
            const auto pr = binary_pr(s, thresholds);
            const auto precision = torch::where(torch::isnan(pr.x), torch::zeros_like(pr.x), pr.x);
            const auto recall = torch::where(torch::isnan(pr.y), torch::zeros_like(pr.y), pr.y);
            return -torch::sum((recall.slice(0, 1) - recall.slice(0, 0, -1)) * precision.slice(0, 0, -1));
        }

        // _reduce_average_precision
        inline auto ap_reduce(const curves &pr, average_mode average, const tensor_t &weights) -> tensor_t {
            tensor_t res;
            if (pr.x.defined()) {
                const auto p = torch::where(torch::isnan(pr.x), torch::zeros_like(pr.x), pr.x);
                const auto r = torch::where(torch::isnan(pr.y), torch::zeros_like(pr.y), pr.y);
                res = -torch::sum((r.slice(1, 1) - r.slice(1, 0, -1)) * p.slice(1, 0, -1), 1);
            } else {
                std::vector<tensor_t> v;
                for (size_t i = 0; i < pr.xs.size(); ++i) {
                    const auto &p = pr.xs[i], &r = pr.ys[i];
                    v.push_back(-torch::sum((r.slice(0, 1) - r.slice(0, 0, -1)) * p.slice(0, 0, -1)));
                }
                res = torch::stack(v);
            }
            return reduce_classes(res, average, weights);
        }

        // torchmetrics.utilities.data.interp (LogAUC's numpy-style interp, not the one above): xp argsorted, slopes by
        // plain division, searchsorted (left) - 1
        inline auto interp_np(const tensor_t &x, const tensor_t &xp_in, const tensor_t &fp_in) -> tensor_t {
            const auto order = torch::argsort(xp_in);
            const auto xp = xp_in.index({order}), fp = fp_in.index({order});
            const auto slopes = (fp.slice(0, 1) - fp.slice(0, 0, -1)) / (xp.slice(0, 1) - xp.slice(0, 0, -1));
            const auto indices = torch::clamp(torch::searchsorted(xp, x) - 1, 0, slopes.size(0) - 1);
            return fp.index({indices}) + slopes.index({indices}) * (x - xp.index({indices}));
        }

        // _binary_logauc_compute
        inline auto logauc(const tensor_t &fpr_in, const tensor_t &tpr_in, std::pair<double, double> fpr_range)
                -> tensor_t {
            const auto range = torch::tensor({fpr_range.first, fpr_range.second}, torch::kFloat64)
                                       .to(fpr_in.device(), torch::kFloat32);
            if (fpr_in.numel() < 2 || tpr_in.numel() < 2) {
                return torch::zeros({}, torch::device(fpr_in.device()));
            }
            const auto tpr = std::get<0>(torch::cat({tpr_in, interp_np(range, fpr_in, tpr_in)}).sort());
            const auto log_fpr = torch::log10(std::get<0>(torch::cat({fpr_in, range}).sort()));
            const auto bounds = torch::log10(range);
            const auto lo = torch::where(log_fpr == bounds[0])[0][-1].item<int64_t>();
            const auto hi = torch::where(log_fpr == bounds[1])[0][-1].item<int64_t>();
            return torch::trapz(tpr.slice(0, lo, hi + 1), log_fpr.slice(0, lo, hi + 1), -1) / (bounds[1] - bounds[0]);
        }

        // _reduce_logauc over per-class curves; a 1-D (binary) curve gives its 0-d value
        inline auto logauc_reduce(const curves &roc, std::pair<double, double> fpr_range, average_mode average)
                -> tensor_t {
            if (roc.x.defined() && roc.x.dim() == 1) {
                return logauc(roc.x, roc.y, fpr_range);
            }
            std::vector<tensor_t> v;
            for (int64_t i = 0; i < roc.count(); ++i) {
                const auto [fpr, tpr, thr] = roc.at(i);
                v.push_back(logauc(fpr, tpr, fpr_range));
            }
            return reduce_classes(torch::stack(v), average, {});
        }

        // _binary_eer_compute
        inline auto eer(const tensor_t &fpr, const tensor_t &tpr) -> tensor_t {
            const auto idx = torch::argmin(torch::abs(fpr - (1 - tpr)));
            return (fpr.index({idx}) + (1 - tpr.index({idx}))) / 2;
        }
        // _eer_compute: one value for a 1-D curve, else one per class
        inline auto eer_reduce(const curves &roc) -> tensor_t {
            if (roc.x.defined() && roc.x.dim() == 1) {
                return eer(roc.x, roc.y);
            }
            std::vector<tensor_t> v;
            for (int64_t i = 0; i < roc.count(); ++i) {
                const auto [fpr, tpr, thr] = roc.at(i);
                v.push_back(eer(fpr, tpr));
            }
            return torch::stack(v);
        }

        // _lexargmax: the rows of x that are lexicographically largest (column 0 first)
        inline auto lexargmax(const tensor_t &x) -> tensor_t {
            tensor_t idx;
            for (int64_t k = 0; k < x.size(1); ++k) {
                const auto col = idx.defined() ? x.index({idx, k}) : x.select(1, k);
                const auto z = torch::where(col == col.max())[0];
                idx = idx.defined() ? idx.index({z}) : z;
                if (idx.numel() < 2) {
                    break;
                }
            }
            return idx;
        }

        // _recall_at_precision (best recall with precision >= min, ties to the higher precision / threshold)
        inline auto recall_at_precision(const tensor_t &precision, const tensor_t &recall, const tensor_t &thresholds,
                                        double min_precision) -> std::pair<tensor_t, tensor_t> {
            auto max_recall = torch::zeros({}, recall.options());
            tensor_t best_threshold;
            const auto n = std::min({recall.size(0), precision.size(0), thresholds.size(0)});
            const auto zipped =
                    torch::vstack({recall.slice(0, 0, n), precision.slice(0, 0, n), thresholds.slice(0, 0, n)}).t();
            const auto masked = zipped.index({zipped.select(1, 1) >= min_precision});
            if (masked.size(0) > 0) {
                const auto row = masked[lexargmax(masked)[0]];
                max_recall = row[0];
                best_threshold = row[2];
            }
            if ((max_recall == 0.0).item<bool>()) {
                best_threshold = torch::full({}, std::nan(""), thresholds.options());
            }
            return {max_recall, best_threshold};
        }

        // _precision_at_recall: Python's max() over the (precision, recall, threshold) tuples with recall >= min:
        // lexicographic, the first maximum kept (a NaN never wins a comparison, so a leading NaN precision stays)
        inline auto precision_at_recall(const tensor_t &precision, const tensor_t &recall, const tensor_t &thresholds,
                                        double min_recall) -> std::pair<tensor_t, tensor_t> {
            const auto n = std::min({recall.size(0), precision.size(0), thresholds.size(0)});
            const auto ok = (recall.slice(0, 0, n) >= min_recall).to(torch::kCPU);
            // to the CPU first: MPS has no float64
            const auto cpu64 = [n](const tensor_t &x) { return x.slice(0, 0, n).cpu().to(torch::kFloat64); };
            const auto v = torch::stack({cpu64(precision), cpu64(recall), cpu64(thresholds)}).contiguous();
            const auto *p = v.data_ptr<double>();
            const auto gt = [&](int64_t a, int64_t b) {
                for (int64_t j = 0; j < 3; ++j) {
                    if (!(p[j * n + a] == p[j * n + b])) {
                        return p[j * n + a] > p[j * n + b];
                    }
                }
                return false;
            };
            int64_t best = -1;
            for (int64_t i = 0; i < n; ++i) {
                if (ok[i].item<bool>() && (best < 0 || gt(i, best))) {
                    best = i;
                }
            }
            const auto max_precision = best < 0 ? torch::zeros({}, precision.options()) : precision[best];
            const auto nan_thr = (max_precision == 0.0).item<bool>() || best < 0;
            return {max_precision, nan_thr ? torch::full({}, std::nan(""), thresholds.options()) : thresholds[best]};
        }

        // _sensitivity_at_specificity / _specificity_at_sensitivity: the best value with other >= min (argmax: first)
        inline auto value_at_min(const tensor_t &value, const tensor_t &other, const tensor_t &thresholds, double min)
                -> std::pair<tensor_t, tensor_t> {
            const auto indices = other >= min;
            if (!indices.any().item<bool>()) {
                // torch.tensor(1e6, dtype=float16) is inf (a float16 fill would raise)
                return {torch::zeros({}, value.options()),
                        torch::tensor(1000000.0, torch::device(thresholds.device())).to(thresholds.scalar_type())};
            }
            const auto v = value.index({indices}), t = thresholds.index({indices});
            const auto idx = torch::argmax(v);
            return {v.index({idx}), t.index({idx})};
        }

        // one fixed-point result per curve (pr: precision-recall curves; else ROC curves)
        inline auto fixed_one(curve_metric k, const curve_t &c, double min) -> std::pair<tensor_t, tensor_t> {
            const auto &[a, b, t] = c;
            switch (k) {
                case curve_metric::precision_at_recall:
                    return precision_at_recall(a, b, t, min);
                case curve_metric::recall_at_precision:
                    return recall_at_precision(a, b, t, min);
                case curve_metric::sensitivity_at_specificity:
                    return value_at_min(b, 1 - a, t, min);
                default:
                    return value_at_min(1 - a, b, t, min);
            }
        }
        inline auto fixed_stack(curve_metric k, const curves &c, double min) -> std::pair<tensor_t, tensor_t> {
            std::vector<tensor_t> values, thresholds;
            for (int64_t i = 0; i < c.count(); ++i) {
                const auto [v, t] = fixed_one(k, c.at(i), min);
                values.push_back(v);
                thresholds.push_back(t);
            }
            return {torch::stack(values), torch::stack(thresholds)};
        }
        inline auto fixed_min(curve_metric k, std::optional<double> min_precision, std::optional<double> min_recall,
                              std::optional<double> min_specificity, std::optional<double> min_sensitivity) -> double {
            return *(k == curve_metric::precision_at_recall          ? min_recall
                     : k == curve_metric::recall_at_precision        ? min_precision
                     : k == curve_metric::sensitivity_at_specificity ? min_specificity
                                                                     : min_sensitivity);
        }
        inline auto fixed_uses_roc(curve_metric k) -> bool { return k >= curve_metric::sensitivity_at_specificity; }

        // ---------------- per-task results ----------------
        // binary
        inline auto squeeze_out(const curve_t &c, bool squeeze) -> curve_t {
            const auto sq = [&](const tensor_t &v) { return squeeze ? squeeze_if_scalar(v) : v; };
            return {sq(std::get<0>(c)), sq(std::get<1>(c)), sq(std::get<2>(c))};
        }
        inline auto pr_curve_out(const curve_state &s, const tensor_t &thr, const binary_curve_option &, bool squeeze)
                -> curve_t {
            return squeeze_out(binary_pr(s, thr).at(0), squeeze);
        }
        inline auto roc_out(const curve_state &s, const tensor_t &thr, const binary_curve_option &, bool squeeze)
                -> curve_t {
            return squeeze_out(binary_roc(s, thr).at(0), squeeze);
        }
        inline auto scalar_out(curve_metric k, const curve_state &s, const tensor_t &thr, const binary_curve_option &o)
                -> tensor_t {
            switch (k) {
                case curve_metric::auroc:
                    return binary_auroc(s, thr, o._max_fpr);
                case curve_metric::average_precision:
                    return binary_ap(s, thr);
                case curve_metric::logauc:
                    return logauc_reduce(binary_roc(s, thr), o._fpr_range, average_mode::none);
                default:
                    return eer_reduce(binary_roc(s, thr));
            }
        }
        inline auto fixed_out(curve_metric k, const curve_state &s, const tensor_t &thr, const binary_curve_option &o)
                -> std::pair<tensor_t, tensor_t> {
            const auto c = fixed_uses_roc(k) ? binary_roc(s, thr) : binary_pr(s, thr);
            return fixed_one(k, c.at(0),
                             fixed_min(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity));
        }
        // multiclass
        inline auto pr_curve_out(const curve_state &s, const tensor_t &thr, const multiclass_curve_option &o,
                                 bool squeeze) -> curve_list_t {
            return multiclass_curves(s, thr, o._num_classes, o._average, false).list(squeeze);
        }
        inline auto roc_out(const curve_state &s, const tensor_t &thr, const multiclass_curve_option &o, bool squeeze)
                -> curve_list_t {
            return multiclass_curves(s, thr, o._num_classes, o._average, true).list(squeeze);
        }
        inline auto scalar_out(curve_metric, const curve_state &s, const tensor_t &thr,
                               const multiclass_curve_option &o) -> tensor_t { // EER
            return eer_reduce(multiclass_curves(s, thr, o._num_classes, o._average, true));
        }
        inline auto fixed_out(curve_metric k, const curve_state &s, const tensor_t &thr,
                              const multiclass_curve_option &o) -> std::pair<tensor_t, tensor_t> {
            const auto c = per_class(s, thr, o._num_classes, fixed_uses_roc(k), false, std::nullopt);
            return fixed_stack(k, c,
                               fixed_min(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity));
        }
        inline auto scalar_out(curve_metric k, const curve_state &s, const tensor_t &thr,
                               const multiclass_auc_option &o) -> tensor_t {
            const auto C = o._num_classes;
            switch (k) {
                case curve_metric::auroc:
                    return auroc_reduce(per_class(s, thr, C, true, false, std::nullopt), o._average,
                                        class_weights(s, C, false));
                case curve_metric::average_precision:
                    return ap_reduce(per_class(s, thr, C, false, false, std::nullopt), o._average,
                                     class_weights(s, C, false));
                default:
                    return logauc_reduce(per_class(s, thr, C, true, false, std::nullopt), o._fpr_range, o._average);
            }
        }
        // multilabel
        inline auto pr_curve_out(const curve_state &s, const tensor_t &thr, const multilabel_curve_option &o,
                                 bool squeeze) -> curve_list_t {
            return per_class(s, thr, o._num_labels, false, true, o._ignore_index).list(squeeze);
        }
        inline auto roc_out(const curve_state &s, const tensor_t &thr, const multilabel_curve_option &o, bool squeeze)
                -> curve_list_t {
            return per_class(s, thr, o._num_labels, true, true, o._ignore_index).list(squeeze);
        }
        inline auto scalar_out(curve_metric, const curve_state &s, const tensor_t &thr,
                               const multilabel_curve_option &o) -> tensor_t { // EER
            return eer_reduce(per_class(s, thr, o._num_labels, true, true, o._ignore_index));
        }
        inline auto fixed_out(curve_metric k, const curve_state &s, const tensor_t &thr,
                              const multilabel_curve_option &o) -> std::pair<tensor_t, tensor_t> {
            const auto c = per_class(s, thr, o._num_labels, fixed_uses_roc(k), true, o._ignore_index);
            return fixed_stack(k, c,
                               fixed_min(k, o._min_precision, o._min_recall, o._min_specificity, o._min_sensitivity));
        }
        inline auto scalar_out(curve_metric k, const curve_state &s, const tensor_t &thr,
                               const multilabel_auc_option &o) -> tensor_t {
            const auto L = o._num_labels;
            if (k != curve_metric::logauc && o._average == average_mode::micro) {
                curve_state flat{{}, {}, s.confmat.defined() ? s.confmat.sum(1) : tensor_t{}};
                if (!s.confmat.defined()) {
                    flat.preds = s.preds.flatten();
                    flat.target = s.target.flatten();
                    drop_ignored(flat.preds, flat.target, o._ignore_index);
                }
                return k == curve_metric::auroc ? binary_auroc(flat, thr, std::nullopt) : binary_ap(flat, thr);
            }
            switch (k) {
                case curve_metric::auroc:
                    return auroc_reduce(per_class(s, thr, L, true, true, o._ignore_index), o._average,
                                        class_weights(s, L, true));
                case curve_metric::average_precision:
                    return ap_reduce(per_class(s, thr, L, false, true, o._ignore_index), o._average,
                                     class_weights(s, L, true));
                default:
                    return logauc_reduce(per_class(s, thr, L, true, true, o._ignore_index), o._fpr_range, o._average);
            }
        }

        // squeeze: the class's compute (Metric._squeeze_if_scalar)
        template<curve_metric K, class Opt>
        inline auto curve_result(const curve_state &s, const tensor_t &thr, const Opt &o, bool squeeze) {
            if constexpr (K == curve_metric::pr_curve) {
                return pr_curve_out(s, thr, o, squeeze);
            } else if constexpr (K == curve_metric::roc) {
                return roc_out(s, thr, o, squeeze);
            } else if constexpr (K >= curve_metric::precision_at_recall) {
                const auto [v, t] = fixed_out(K, s, thr, o);
                return squeeze ? std::pair{squeeze_if_scalar(v), squeeze_if_scalar(t)} : std::pair{v, t};
            } else {
                const auto v = scalar_out(K, s, thr, o);
                return squeeze ? squeeze_if_scalar(v) : v;
            }
        }

        template<curve_metric K, class Opt>
        inline auto curve_functional(const tensor_t &preds, const tensor_t &target, const Opt &o)
                -> decltype(curve_result<K>(curve_state{}, tensor_t{}, o, false)) {
            using R = decltype(curve_result<K>(curve_state{}, tensor_t{}, o, false));
            if (auto e = curve_arg_error(o, K); !e.empty()) {
                return invalid<R>(std::move(e));
            }
            if (auto e = curve_tensor_error(preds, target, o); !e.empty()) {
                return invalid<R>(std::move(e));
            }
            const auto thr = adjust_thresholds(o._thresholds, preds.device());
            return curve_result<K>(curve_update(preds, target, thr, o), thr, o, false);
        }

        inline auto binned_shape(const binary_curve_option &, int64_t T) -> std::vector<int64_t> { return {T, 2, 2}; }
        inline auto binned_shape(const multiclass_curve_option &o, int64_t T) -> std::vector<int64_t> {
            return o._average == average_mode::micro ? std::vector<int64_t>{T, 2, 2}
                                                     : std::vector<int64_t>{T, o._num_classes, 2, 2};
        }
        inline auto binned_shape(const multiclass_auc_option &o, int64_t T) -> std::vector<int64_t> {
            return {T, o._num_classes, 2, 2};
        }
        inline auto binned_shape(const multilabel_curve_option &o, int64_t T) -> std::vector<int64_t> {
            return {T, o._num_labels, 2, 2};
        }
        inline auto binned_shape(const multilabel_auc_option &o, int64_t T) -> std::vector<int64_t> {
            return {T, o._num_labels, 2, 2};
        }

        template<class Opt>
        inline auto curve_fallback(const Opt &) -> Opt {
            Opt o;
            if constexpr (requires { o._min_recall; }) {
                o._min_precision = o._min_recall = o._min_specificity = o._min_sensitivity = 0.0;
            }
            if constexpr (requires { o._num_classes; }) {
                o._num_classes = 2;
            }
            if constexpr (requires { o._num_labels; }) {
                o._num_labels = 2;
            }
            return o;
        }

        // torchmetrics' XxxPrecisionRecallCurve state: every formatted batch (exact) or the summed binned confusion
        // matrices; compute() evaluates K on it.
        template<class Opt, curve_metric K>
        class curve_class : public metric::detail::metric_base<curve_class<Opt, K>> {
        public:
            explicit curve_class(Opt opt = {}) : _opt(std::move(opt)) {
                if (auto e = curve_arg_error(_opt, K); !e.empty()) {
                    _opt = curve_fallback(_opt);
                    fail(std::move(e));
                }
                _thresholds = adjust_thresholds(_opt._thresholds, torch::kCPU);
                if (_thresholds.defined()) {
                    _confmat = torch::zeros(binned_shape(_opt, _thresholds.size(0)), torch::kLong);
                }
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = curve_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto thr = _thresholds.defined() ? _thresholds.to(preds.device()) : _thresholds;
                const auto s = curve_update(preds, target, thr, _opt);
                if (s.confmat.defined()) {
                    _confmat = _confmat.to(s.confmat.device()) + s.confmat;
                    return;
                }
                _preds.push_back(s.preds.detach().clone()); // own the data: callers may reuse their input buffer
                _target.push_back(s.target.detach().clone());
            }

            auto compute() const {
                if (_thresholds.defined()) {
                    return curve_result<K>({{}, {}, _confmat}, _thresholds.to(_confmat.device()), _opt, true);
                }
                return curve_result<K>({cat0(_preds), cat0(_target), {}}, _thresholds, _opt, true);
            }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            tensor_t _thresholds, _confmat;
            std::vector<tensor_t> _preds, _target;
        };
    } // namespace detail

    // ---------------- PrecisionRecallCurve ----------------
    // (precision, recall, thresholds): precision ends with 1 and recall with 0 (after the last threshold); exact
    // curves run from the lowest distinct score up, binned curves follow the thresholds (0 / 0 = NaN there).
    inline auto binary_precision_recall_curve(const tensor_t &preds, const tensor_t &target,
                                              const binary_curve_option &opt = {}) -> curve_t {
        return detail::curve_functional<detail::curve_metric::pr_curve>(preds, target, opt);
    }
    inline auto multiclass_precision_recall_curve(const tensor_t &preds, const tensor_t &target,
                                                  const multiclass_curve_option &opt) -> curve_list_t {
        return detail::curve_functional<detail::curve_metric::pr_curve>(preds, target, opt);
    }
    inline auto multilabel_precision_recall_curve(const tensor_t &preds, const tensor_t &target,
                                                  const multilabel_curve_option &opt) -> curve_list_t {
        return detail::curve_functional<detail::curve_metric::pr_curve>(preds, target, opt);
    }
    using BinaryPrecisionRecallCurve = detail::curve_class<binary_curve_option, detail::curve_metric::pr_curve>;
    using MulticlassPrecisionRecallCurve = detail::curve_class<multiclass_curve_option, detail::curve_metric::pr_curve>;
    using MultilabelPrecisionRecallCurve = detail::curve_class<multilabel_curve_option, detail::curve_metric::pr_curve>;

    // ---------------- ROC ----------------
    // (fpr, tpr, thresholds) from (0, 0) at threshold 1 (exact) / the highest threshold (binned); a class without
    // negatives (positives) gets fpr (tpr) 0.
    inline auto binary_roc(const tensor_t &preds, const tensor_t &target, const binary_curve_option &opt = {})
            -> curve_t {
        return detail::curve_functional<detail::curve_metric::roc>(preds, target, opt);
    }
    inline auto multiclass_roc(const tensor_t &preds, const tensor_t &target, const multiclass_curve_option &opt)
            -> curve_list_t {
        return detail::curve_functional<detail::curve_metric::roc>(preds, target, opt);
    }
    inline auto multilabel_roc(const tensor_t &preds, const tensor_t &target, const multilabel_curve_option &opt)
            -> curve_list_t {
        return detail::curve_functional<detail::curve_metric::roc>(preds, target, opt);
    }
    using BinaryROC = detail::curve_class<binary_curve_option, detail::curve_metric::roc>;
    using MulticlassROC = detail::curve_class<multiclass_curve_option, detail::curve_metric::roc>;
    using MultilabelROC = detail::curve_class<multilabel_curve_option, detail::curve_metric::roc>;

    // ---------------- AUROC ----------------
    // trapezoidal area under the ROC; binary max_fpr: McClish-standardized partial area; averages leave NaN classes
    // out.
    inline auto binary_auroc(const tensor_t &preds, const tensor_t &target, const binary_curve_option &opt = {})
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::auroc>(preds, target, opt);
    }
    inline auto multiclass_auroc(const tensor_t &preds, const tensor_t &target, const multiclass_auc_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::auroc>(preds, target, opt);
    }
    inline auto multilabel_auroc(const tensor_t &preds, const tensor_t &target, const multilabel_auc_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::auroc>(preds, target, opt);
    }
    using BinaryAUROC = detail::curve_class<binary_curve_option, detail::curve_metric::auroc>;
    using MulticlassAUROC = detail::curve_class<multiclass_auc_option, detail::curve_metric::auroc>;
    using MultilabelAUROC = detail::curve_class<multilabel_auc_option, detail::curve_metric::auroc>;

    // ---------------- AveragePrecision ----------------
    // -sum((r[i + 1] - r[i]) p[i]) over the precision-recall curve (NaN points as 0).
    inline auto binary_average_precision(const tensor_t &preds, const tensor_t &target,
                                         const binary_curve_option &opt = {}) -> tensor_t {
        return detail::curve_functional<detail::curve_metric::average_precision>(preds, target, opt);
    }
    inline auto multiclass_average_precision(const tensor_t &preds, const tensor_t &target,
                                             const multiclass_auc_option &opt) -> tensor_t {
        return detail::curve_functional<detail::curve_metric::average_precision>(preds, target, opt);
    }
    inline auto multilabel_average_precision(const tensor_t &preds, const tensor_t &target,
                                             const multilabel_auc_option &opt) -> tensor_t {
        return detail::curve_functional<detail::curve_metric::average_precision>(preds, target, opt);
    }
    using BinaryAveragePrecision = detail::curve_class<binary_curve_option, detail::curve_metric::average_precision>;
    using MulticlassAveragePrecision =
            detail::curve_class<multiclass_auc_option, detail::curve_metric::average_precision>;
    using MultilabelAveragePrecision =
            detail::curve_class<multilabel_auc_option, detail::curve_metric::average_precision>;

    // ---------------- LogAUC ----------------
    // area under tpr over log10(fpr) within fpr_range, divided by the range's log10 width (0 for a one-point curve).
    inline auto binary_logauc(const tensor_t &preds, const tensor_t &target, const binary_curve_option &opt = {})
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::logauc>(preds, target, opt);
    }
    inline auto multiclass_logauc(const tensor_t &preds, const tensor_t &target, const multiclass_auc_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::logauc>(preds, target, opt);
    }
    inline auto multilabel_logauc(const tensor_t &preds, const tensor_t &target, const multilabel_auc_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::logauc>(preds, target, opt);
    }
    using BinaryLogAUC = detail::curve_class<binary_curve_option, detail::curve_metric::logauc>;
    using MulticlassLogAUC = detail::curve_class<multiclass_auc_option, detail::curve_metric::logauc>;
    using MultilabelLogAUC = detail::curve_class<multilabel_auc_option, detail::curve_metric::logauc>;

    // ---------------- EER ----------------
    // At the ROC point minimising |fpr - (1 - tpr)| (the first one), (fpr + 1 - tpr) / 2 — no interpolation between ROC
    // points; metric::audio::equal_error_rate interpolates (develop_log/2026-10-24/progress02 D9).
    inline auto binary_eer(const tensor_t &preds, const tensor_t &target, const binary_curve_option &opt = {})
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::eer>(preds, target, opt);
    }
    inline auto multiclass_eer(const tensor_t &preds, const tensor_t &target, const multiclass_curve_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::eer>(preds, target, opt);
    }
    inline auto multilabel_eer(const tensor_t &preds, const tensor_t &target, const multilabel_curve_option &opt)
            -> tensor_t {
        return detail::curve_functional<detail::curve_metric::eer>(preds, target, opt);
    }
    using BinaryEER = detail::curve_class<binary_curve_option, detail::curve_metric::eer>;
    using MulticlassEER = detail::curve_class<multiclass_curve_option, detail::curve_metric::eer>;
    using MultilabelEER = detail::curve_class<multilabel_curve_option, detail::curve_metric::eer>;

    // ---------------- fixed-point metrics ----------------
    // {value, threshold} (per class: [C] each): the highest precision with recall >= min_recall, the highest recall
    // with precision >= min_precision (threshold NaN when the value is 0), the highest sensitivity with specificity >=
    // min_specificity / specificity with sensitivity >= min_sensitivity (none: {0, 1e6}).
    inline auto binary_precision_at_fixed_recall(const tensor_t &preds, const tensor_t &target,
                                                 const binary_curve_option &opt) -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::precision_at_recall>(preds, target, opt);
    }
    inline auto multiclass_precision_at_fixed_recall(const tensor_t &preds, const tensor_t &target,
                                                     const multiclass_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::precision_at_recall>(preds, target, opt);
    }
    inline auto multilabel_precision_at_fixed_recall(const tensor_t &preds, const tensor_t &target,
                                                     const multilabel_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::precision_at_recall>(preds, target, opt);
    }
    inline auto binary_recall_at_fixed_precision(const tensor_t &preds, const tensor_t &target,
                                                 const binary_curve_option &opt) -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::recall_at_precision>(preds, target, opt);
    }
    inline auto multiclass_recall_at_fixed_precision(const tensor_t &preds, const tensor_t &target,
                                                     const multiclass_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::recall_at_precision>(preds, target, opt);
    }
    inline auto multilabel_recall_at_fixed_precision(const tensor_t &preds, const tensor_t &target,
                                                     const multilabel_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::recall_at_precision>(preds, target, opt);
    }
    inline auto binary_sensitivity_at_specificity(const tensor_t &preds, const tensor_t &target,
                                                  const binary_curve_option &opt) -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::sensitivity_at_specificity>(preds, target, opt);
    }
    inline auto multiclass_sensitivity_at_specificity(const tensor_t &preds, const tensor_t &target,
                                                      const multiclass_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::sensitivity_at_specificity>(preds, target, opt);
    }
    inline auto multilabel_sensitivity_at_specificity(const tensor_t &preds, const tensor_t &target,
                                                      const multilabel_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::sensitivity_at_specificity>(preds, target, opt);
    }
    inline auto binary_specificity_at_sensitivity(const tensor_t &preds, const tensor_t &target,
                                                  const binary_curve_option &opt) -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::specificity_at_sensitivity>(preds, target, opt);
    }
    inline auto multiclass_specificity_at_sensitivity(const tensor_t &preds, const tensor_t &target,
                                                      const multiclass_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::specificity_at_sensitivity>(preds, target, opt);
    }
    inline auto multilabel_specificity_at_sensitivity(const tensor_t &preds, const tensor_t &target,
                                                      const multilabel_curve_option &opt)
            -> std::pair<tensor_t, tensor_t> {
        return detail::curve_functional<detail::curve_metric::specificity_at_sensitivity>(preds, target, opt);
    }
    using BinaryPrecisionAtFixedRecall =
            detail::curve_class<binary_curve_option, detail::curve_metric::precision_at_recall>;
    using MulticlassPrecisionAtFixedRecall =
            detail::curve_class<multiclass_curve_option, detail::curve_metric::precision_at_recall>;
    using MultilabelPrecisionAtFixedRecall =
            detail::curve_class<multilabel_curve_option, detail::curve_metric::precision_at_recall>;
    using BinaryRecallAtFixedPrecision =
            detail::curve_class<binary_curve_option, detail::curve_metric::recall_at_precision>;
    using MulticlassRecallAtFixedPrecision =
            detail::curve_class<multiclass_curve_option, detail::curve_metric::recall_at_precision>;
    using MultilabelRecallAtFixedPrecision =
            detail::curve_class<multilabel_curve_option, detail::curve_metric::recall_at_precision>;
    using BinarySensitivityAtSpecificity =
            detail::curve_class<binary_curve_option, detail::curve_metric::sensitivity_at_specificity>;
    using MulticlassSensitivityAtSpecificity =
            detail::curve_class<multiclass_curve_option, detail::curve_metric::sensitivity_at_specificity>;
    using MultilabelSensitivityAtSpecificity =
            detail::curve_class<multilabel_curve_option, detail::curve_metric::sensitivity_at_specificity>;
    using BinarySpecificityAtSensitivity =
            detail::curve_class<binary_curve_option, detail::curve_metric::specificity_at_sensitivity>;
    using MulticlassSpecificityAtSensitivity =
            detail::curve_class<multiclass_curve_option, detail::curve_metric::specificity_at_sensitivity>;
    using MultilabelSpecificityAtSensitivity =
            detail::curve_class<multilabel_curve_option, detail::curve_metric::specificity_at_sensitivity>;
} // namespace torchmedia::metric::classification
#endif // LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_CURVE_HPP
