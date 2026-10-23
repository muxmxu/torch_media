#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_STAT_HPP
#define LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_STAT_HPP
// Classification metrics on thresholded / arg-maxed predictions (develop_log/2026-10-24/progress02): torchmetrics 1.9
// StatScores, Accuracy, Precision, Recall, FBetaScore, F1Score, Specificity, NegativePredictiveValue,
// HammingDistance, ExactMatch, ConfusionMatrix, JaccardIndex, MatthewsCorrCoef and CohenKappa in their binary /
// multiclass / multilabel forms: functionals (binary_accuracy, ...) and classes (BinaryAccuracy, ...) with the same
// libtorch calls in the same order. Inputs as torchmetrics: binary preds / target [N, ...] (floating preds are
// probabilities, or logits when any value is outside [0, 1]: sigmoid); multiclass preds [N, C, ...] scores (arg-maxed)
// or [N, ...] labels with target [N, ...]; multilabel [N, L, ...]. torchmetrics' task= wrappers (Accuracy(task=...))
// are not provided: call the binary_ / multiclass_ / multilabel_ form (BinaryAccuracy, ...) directly.
#include "_common.hpp"

namespace torchmedia::metric::classification {
    // torchmetrics' average argument; none is torchmetrics' "none" / None (one value per class).
    enum class average_mode { micro, macro, weighted, none };
    enum class multidim_average_mode { global, samplewise };
    // ConfusionMatrix normalize: none, true_ (torchmetrics "true": over the targets, i.e. rows), pred (columns), all.
    enum class normalize_mode { none, true_, pred, all };
    enum class kappa_weights_mode { none, linear, quadratic };

    // ---------------- options ----------------
    // StatScores, Accuracy, Precision, Recall, FBetaScore, F1Score, Specificity, NegativePredictiveValue,
    // HammingDistance.
    typedef struct binary_stat_scores_option {
        double _threshold = 0.5; // [0, 1]
        multidim_average_mode _multidim_average = multidim_average_mode::global;
        std::optional<int64_t> _ignore_index;
        double _zero_division = 0.0; // 0 or 1: the value of x / 0 (Precision, Recall, FBetaScore, F1Score, NPV)
        double _beta = 1.0; // FBetaScore only (> 0; F1Score uses 1)

        auto threshold(double v) -> binary_stat_scores_option & {
            _threshold = v;
            return *this;
        }
        auto multidim_average(multidim_average_mode v) -> binary_stat_scores_option & {
            _multidim_average = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> binary_stat_scores_option & {
            _ignore_index = v;
            return *this;
        }
        auto zero_division(double v) -> binary_stat_scores_option & {
            _zero_division = v;
            return *this;
        }
        auto beta(double v) -> binary_stat_scores_option & {
            _beta = v;
            return *this;
        }
    } binary_stat_scores_option_t;

    // Also ExactMatch (num_classes required there; top_k / average / zero_division / beta unused).
    typedef struct multiclass_stat_scores_option {
        std::optional<int64_t> _num_classes; // >= 2; {} only with average micro (top_k 1)
        int64_t _top_k = 1; // a prediction is correct when the target is among the top_k scores
        average_mode _average = average_mode::macro;
        multidim_average_mode _multidim_average = multidim_average_mode::global;
        std::optional<int64_t> _ignore_index;
        double _zero_division = 0.0;
        double _beta = 1.0;

        auto num_classes(std::optional<int64_t> v) -> multiclass_stat_scores_option & {
            _num_classes = v;
            return *this;
        }
        auto top_k(int64_t v) -> multiclass_stat_scores_option & {
            _top_k = v;
            return *this;
        }
        auto average(average_mode v) -> multiclass_stat_scores_option & {
            _average = v;
            return *this;
        }
        auto multidim_average(multidim_average_mode v) -> multiclass_stat_scores_option & {
            _multidim_average = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_stat_scores_option & {
            _ignore_index = v;
            return *this;
        }
        auto zero_division(double v) -> multiclass_stat_scores_option & {
            _zero_division = v;
            return *this;
        }
        auto beta(double v) -> multiclass_stat_scores_option & {
            _beta = v;
            return *this;
        }
    } multiclass_stat_scores_option_t;

    // Also ExactMatch (average / zero_division / beta unused).
    typedef struct multilabel_stat_scores_option {
        int64_t _num_labels = 0; // required (>= 2)
        double _threshold = 0.5;
        average_mode _average = average_mode::macro;
        multidim_average_mode _multidim_average = multidim_average_mode::global;
        std::optional<int64_t> _ignore_index;
        double _zero_division = 0.0;
        double _beta = 1.0;

        auto num_labels(int64_t v) -> multilabel_stat_scores_option & {
            _num_labels = v;
            return *this;
        }
        auto threshold(double v) -> multilabel_stat_scores_option & {
            _threshold = v;
            return *this;
        }
        auto average(average_mode v) -> multilabel_stat_scores_option & {
            _average = v;
            return *this;
        }
        auto multidim_average(multidim_average_mode v) -> multilabel_stat_scores_option & {
            _multidim_average = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multilabel_stat_scores_option & {
            _ignore_index = v;
            return *this;
        }
        auto zero_division(double v) -> multilabel_stat_scores_option & {
            _zero_division = v;
            return *this;
        }
        auto beta(double v) -> multilabel_stat_scores_option & {
            _beta = v;
            return *this;
        }
    } multilabel_stat_scores_option_t;

    // ConfusionMatrix, JaccardIndex, MatthewsCorrCoef, CohenKappa.
    typedef struct binary_confusion_matrix_option {
        double _threshold = 0.5;
        std::optional<int64_t> _ignore_index;
        normalize_mode _normalize = normalize_mode::none; // ConfusionMatrix only
        double _zero_division = 0.0; // JaccardIndex only (0 or 1)
        kappa_weights_mode _weights = kappa_weights_mode::none; // CohenKappa only

        auto threshold(double v) -> binary_confusion_matrix_option & {
            _threshold = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> binary_confusion_matrix_option & {
            _ignore_index = v;
            return *this;
        }
        auto normalize(normalize_mode v) -> binary_confusion_matrix_option & {
            _normalize = v;
            return *this;
        }
        auto zero_division(double v) -> binary_confusion_matrix_option & {
            _zero_division = v;
            return *this;
        }
        auto weights(kappa_weights_mode v) -> binary_confusion_matrix_option & {
            _weights = v;
            return *this;
        }
    } binary_confusion_matrix_option_t;

    typedef struct multiclass_confusion_matrix_option {
        int64_t _num_classes = 0; // required (>= 2)
        std::optional<int64_t> _ignore_index;
        normalize_mode _normalize = normalize_mode::none;
        average_mode _average = average_mode::macro; // JaccardIndex only
        double _zero_division = 0.0;
        kappa_weights_mode _weights = kappa_weights_mode::none;

        auto num_classes(int64_t v) -> multiclass_confusion_matrix_option & {
            _num_classes = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multiclass_confusion_matrix_option & {
            _ignore_index = v;
            return *this;
        }
        auto normalize(normalize_mode v) -> multiclass_confusion_matrix_option & {
            _normalize = v;
            return *this;
        }
        auto average(average_mode v) -> multiclass_confusion_matrix_option & {
            _average = v;
            return *this;
        }
        auto zero_division(double v) -> multiclass_confusion_matrix_option & {
            _zero_division = v;
            return *this;
        }
        auto weights(kappa_weights_mode v) -> multiclass_confusion_matrix_option & {
            _weights = v;
            return *this;
        }
    } multiclass_confusion_matrix_option_t;

    typedef struct multilabel_confusion_matrix_option {
        int64_t _num_labels = 0; // required (>= 2)
        double _threshold = 0.5;
        std::optional<int64_t> _ignore_index;
        normalize_mode _normalize = normalize_mode::none;
        average_mode _average = average_mode::macro; // JaccardIndex only
        double _zero_division = 0.0;

        auto num_labels(int64_t v) -> multilabel_confusion_matrix_option & {
            _num_labels = v;
            return *this;
        }
        auto threshold(double v) -> multilabel_confusion_matrix_option & {
            _threshold = v;
            return *this;
        }
        auto ignore_index(std::optional<int64_t> v) -> multilabel_confusion_matrix_option & {
            _ignore_index = v;
            return *this;
        }
        auto normalize(normalize_mode v) -> multilabel_confusion_matrix_option & {
            _normalize = v;
            return *this;
        }
        auto average(average_mode v) -> multilabel_confusion_matrix_option & {
            _average = v;
            return *this;
        }
        auto zero_division(double v) -> multilabel_confusion_matrix_option & {
            _zero_division = v;
            return *this;
        }
    } multilabel_confusion_matrix_option_t;

    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;
        using metric::detail::safe_divide;
        using metric::detail::squeeze_if_scalar;

        // torchmetrics normalize_logits_if_needed: a CPU tensor with any value outside [0, 1] (NaN included) goes
        // through sigmoid (softmax over dim 1); on other devices only values < 0 or > 1 count.
        inline auto normalize_logits(const tensor_t &x, bool softmax) -> tensor_t {
            const auto f = [&] { return softmax ? torch::softmax(x, 1) : x.sigmoid(); };
            if (x.device().is_cpu()) {
                return torch::all((x >= 0) * (x <= 1)).item<bool>() ? x : f();
            }
            return torch::where(((x < 0) | (x > 1)).any(), f(), x);
        }

        // torchmetrics _bincount: torch.bincount, or its arange / eq form on MPS and in deterministic mode.
        inline auto bincount(const tensor_t &x, int64_t minlength) -> tensor_t {
            if (at::globalContext().deterministicAlgorithms() || x.is_mps()) {
                return torch::eq(x.reshape({-1, 1}), torch::arange(minlength, x.options()).repeat({x.numel(), 1}))
                        .sum(0);
            }
            return torch::bincount(x, {}, minlength);
        }

        // dim_zero_cat: the batches concatenated along dim 0 (a 0-d batch as one element).
        inline auto cat0(const std::vector<tensor_t> &xs) -> tensor_t {
            std::vector<tensor_t> v;
            for (const auto &x: xs) {
                v.push_back(x.dim() == 0 ? x.unsqueeze(0) : x);
            }
            return torch::cat(v);
        }

        inline auto unique_count(const tensor_t &x) -> int64_t {
            return std::get<0>(torch::_unique(x.to(torch::kCPU))).numel(); // MPS has no _unique
        }

        inline auto ignore_text(std::optional<int64_t> ii) -> std::string {
            return ii ? std::format(" and ignore_index {}", *ii) : std::string{};
        }

        // ---------------- argument checks (torchmetrics _xxx_arg_validation; typed arguments need range checks) ---
        inline auto threshold_error(double t) -> std::string {
            return t >= 0 && t <= 1 ? std::string{}
                                    : std::format("Expected argument `threshold` to be a float in the [0,1] range, "
                                                  "but got {}.",
                                                  t);
        }
        inline auto zero_division_error(double z) -> std::string {
            return z == 0 || z == 1 ? std::string{}
                                    : std::format("Expected argument `zero_division` to be 0 or 1, but got {}.", z);
        }
        inline auto beta_error(double b, bool used) -> std::string {
            return !used || b > 0 ? std::string{}
                                  : std::format("Expected argument `beta` to be a float larger than 0, but got {}.", b);
        }
        inline auto count_error(const char *name, int64_t n) -> std::string {
            return n >= 2 ? std::string{}
                          : std::format("Expected argument `{}` to be an integer larger than 1, but got {}", name, n);
        }

        inline auto stat_arg_error(const binary_stat_scores_option &o, bool fbeta) -> std::string {
            for (auto e:
                 {threshold_error(o._threshold), zero_division_error(o._zero_division), beta_error(o._beta, fbeta)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            return {};
        }
        inline auto stat_arg_error(const multiclass_stat_scores_option &o, bool fbeta) -> std::string {
            if (!o._num_classes && o._average != average_mode::micro) {
                return "Argument `num_classes` can only be `None` for `average='micro'`.";
            }
            if (o._num_classes && *o._num_classes < 2) {
                return count_error("num_classes", *o._num_classes);
            }
            if (o._top_k < 1 || o._top_k > o._num_classes.value_or(1)) {
                return std::format("Expected argument `top_k` to be in [1, num_classes], but got {}.", o._top_k);
            }
            for (auto e: {zero_division_error(o._zero_division), beta_error(o._beta, fbeta)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            return {};
        }
        inline auto stat_arg_error(const multilabel_stat_scores_option &o, bool fbeta) -> std::string {
            for (auto e: {count_error("num_labels", o._num_labels), threshold_error(o._threshold),
                          zero_division_error(o._zero_division), beta_error(o._beta, fbeta)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            return {};
        }
        inline auto confmat_arg_error(const binary_confusion_matrix_option &o) -> std::string {
            const auto e = threshold_error(o._threshold);
            return e.empty() ? zero_division_error(o._zero_division) : e;
        }
        inline auto confmat_arg_error(const multiclass_confusion_matrix_option &o) -> std::string {
            const auto e = count_error("num_classes", o._num_classes);
            return e.empty() ? zero_division_error(o._zero_division) : e;
        }
        inline auto confmat_arg_error(const multilabel_confusion_matrix_option &o) -> std::string {
            for (auto e: {count_error("num_labels", o._num_labels), threshold_error(o._threshold),
                          zero_division_error(o._zero_division)}) {
                if (!e.empty()) {
                    return e;
                }
            }
            return {};
        }

        // ---------------- tensor checks (torchmetrics _xxx_tensor_validation) ----------------
        // target only 0 / 1 (or ignore_index); integer preds only 0 / 1.
        inline auto binary_values_error(const tensor_t &preds, const tensor_t &target, std::optional<int64_t> ii)
                -> std::string {
            auto bad = (target != 0) & (target != 1);
            if (ii) {
                bad = bad & (target != *ii);
            }
            if (bad.any().item<bool>()) {
                return std::format("Detected values in `target` other than 0, 1{}.", ignore_text(ii));
            }
            if (!preds.is_floating_point() && ((preds != 0) & (preds != 1)).any().item<bool>()) {
                return "Detected values in `preds` other than 0 and 1 (`preds` is a label tensor).";
            }
            return {};
        }

        inline auto same_shape_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            return preds.sizes() == target.sizes()
                           ? std::string{}
                           : std::format("Predictions and targets are expected to have the same shape, but got {} "
                                         "and {}.",
                                         c10::str(preds.sizes()), c10::str(target.sizes()));
        }

        inline auto binary_tensor_error(const tensor_t &preds, const tensor_t &target, bool samplewise,
                                        std::optional<int64_t> ii) -> std::string {
            if (auto e = same_shape_error(preds, target); !e.empty()) {
                return e;
            }
            if (auto e = binary_values_error(preds, target, ii); !e.empty()) {
                return e;
            }
            return samplewise && preds.dim() < 2
                           ? "Expected input to be at least 2D when multidim_average is set to `samplewise`"
                           : std::string{};
        }

        inline auto multiclass_tensor_error(const tensor_t &preds, const tensor_t &target,
                                            std::optional<int64_t> num_classes, bool samplewise,
                                            std::optional<int64_t> ii) -> std::string {
            if (preds.dim() == target.dim() + 1) {
                if (!preds.is_floating_point()) {
                    return "If `preds` have one dimension more than `target`, `preds` should be a float tensor.";
                }
                if (num_classes && preds.size(1) != *num_classes) {
                    return "If `preds` have one dimension more than `target`, `preds.shape[1]` should be equal to "
                           "number of classes.";
                }
                // torchmetrics compares shape[2:] only (a batch-size mismatch then broadcasts or fails in torch)
                if (preds.size(0) != target.size(0) || preds.sizes().slice(2) != target.sizes().slice(1)) {
                    return "If `preds` have one dimension more than `target`, the shape of `preds` should be (N, C, "
                           "...), and the shape of `target` should be (N, ...).";
                }
                if (samplewise && preds.dim() < 3) {
                    return "If `preds` have one dimension more than `target`, the shape of `preds` should be at least "
                           "3D when multidim_average is set to `samplewise`";
                }
            } else if (preds.dim() == target.dim()) {
                if (auto e = same_shape_error(preds, target); !e.empty()) {
                    return e;
                }
                if (samplewise && preds.dim() < 2) {
                    return "When `preds` and `target` have the same shape, the shape of `preds` should be at least 2D "
                           "when multidim_average is set to `samplewise`";
                }
            } else {
                return "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be "
                       "(N, ...) and `preds` should be (N, C, ...).";
            }
            // torchmetrics' loop `for t in ((target,),) + ((preds,),) if not preds.is_floating_point() else ()` checks
            // nothing for floating preds (the conditional expression covers the whole tuple)
            if (num_classes && !preds.is_floating_point()) {
                const auto limit = *num_classes + (ii ? 1 : 0);
                for (const auto &[t, name]: {std::pair{target, "target"}, std::pair{preds, "preds"}}) {
                    if (unique_count(t) > limit) {
                        return std::format("Detected more unique values in `{}` than expected. Expected only {}.", name,
                                           limit);
                    }
                }
            }
            return {};
        }

        inline auto multilabel_tensor_error(const tensor_t &preds, const tensor_t &target, int64_t num_labels,
                                            bool samplewise, std::optional<int64_t> ii) -> std::string {
            if (auto e = same_shape_error(preds, target); !e.empty()) {
                return e;
            }
            if (preds.dim() < 2 || preds.size(1) != num_labels) {
                return std::format("Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of "
                                   "labels {}, but got shape {}",
                                   num_labels, c10::str(preds.sizes()));
            }
            if (auto e = binary_values_error(preds, target, ii); !e.empty()) {
                return e;
            }
            return samplewise && preds.dim() < 3
                           ? "Expected input to be at least 3D when multidim_average is set to `samplewise`"
                           : std::string{};
        }

        inline auto stat_tensor_error(const tensor_t &preds, const tensor_t &target, const binary_stat_scores_option &o)
                -> std::string {
            // torchmetrics reshapes by preds.shape[0]: 0-d inputs raise an IndexError there
            return preds.dim() == 0 ? "Expected `preds` and `target` with at least one dimension, but got 0-d tensors."
                                    : binary_tensor_error(preds, target,
                                                          o._multidim_average == multidim_average_mode::samplewise,
                                                          o._ignore_index);
        }
        inline auto stat_tensor_error(const tensor_t &preds, const tensor_t &target,
                                      const multiclass_stat_scores_option &o) -> std::string {
            return target.dim() == 0 ? "Expected `target` with at least one dimension, but got a 0-d tensor."
                                     : multiclass_tensor_error(preds, target, o._num_classes,
                                                               o._multidim_average == multidim_average_mode::samplewise,
                                                               o._ignore_index);
        }
        inline auto stat_tensor_error(const tensor_t &preds, const tensor_t &target,
                                      const multilabel_stat_scores_option &o) -> std::string {
            return multilabel_tensor_error(preds, target, o._num_labels,
                                           o._multidim_average == multidim_average_mode::samplewise, o._ignore_index);
        }

        // ---------------- stat scores ----------------
        struct stat_state {
            tensor_t tp, fp, tn, fn;
        };

        inline auto stat_counts(const tensor_t &preds, const tensor_t &target, at::IntArrayRef dims, bool squeeze)
                -> stat_state {
            const auto f = [&](const tensor_t &m) {
                auto s = m.sum(dims);
                return squeeze ? s.squeeze() : s;
            };
            const auto eq = target == preds, ne = target != preds, pos = target == 1, neg = target == 0;
            return {f(eq & pos), f(ne & neg), f(eq & neg), f(ne & pos)};
        }

        // x.reshape(*x.shape[:k], -1)
        inline auto keep_dims(const tensor_t &x, int64_t k) -> tensor_t {
            std::vector<int64_t> s(x.sizes().begin(), x.sizes().begin() + std::min<int64_t>(k, x.dim()));
            s.push_back(-1);
            return x.reshape(s);
        }

        // _binary_stat_scores_format + _binary_stat_scores_update
        inline auto stat_update(const tensor_t &preds_in, const tensor_t &target_in, const binary_stat_scores_option &o)
                -> stat_state {
            auto preds = preds_in.is_floating_point() ? normalize_logits(preds_in, false) > o._threshold : preds_in;
            preds = keep_dims(preds, 1);
            auto target = keep_dims(target_in, 1);
            if (o._ignore_index) {
                target = target.masked_fill(target == *o._ignore_index, -1);
            }
            const auto global = o._multidim_average == multidim_average_mode::global;
            return stat_counts(preds, target, global ? std::vector<int64_t>{0, 1} : std::vector<int64_t>{1}, true);
        }

        // _refine_preds_oh: the target's one-hot where it is among the top_k scores, else the top-1 class.
        inline auto refine_preds_oh(const tensor_t &preds, const tensor_t &target, int64_t top_k) -> tensor_t {
            const auto top = std::get<1>(torch::topk(preds, top_k, 1));
            const auto in_top = torch::any(top == target.unsqueeze(1), 1);
            const auto result = torch::where(in_top, target, top.select(1, 0));
            return torch::zeros({preds.size(0), preds.size(2), preds.size(1)}, preds.options().dtype(torch::kInt32))
                    .scatter_(-1, result.unsqueeze(-1), 1);
        }

        // _multiclass_stat_scores_format + _multiclass_stat_scores_update
        inline auto stat_update(const tensor_t &preds_in, const tensor_t &target_in,
                                const multiclass_stat_scores_option &o) -> stat_state {
            const int64_t nc = o._num_classes.value_or(1), top_k = o._top_k;
            const auto ii = o._ignore_index;
            auto preds = preds_in.dim() == target_in.dim() + 1 && top_k == 1 ? preds_in.argmax(1) : preds_in;
            preds = keep_dims(preds, top_k != 1 ? 2 : 1);
            auto target = keep_dims(target_in, 1);
            if (o._multidim_average == multidim_average_mode::samplewise || top_k != 1) {
                const bool ignore_in = ii && 0 <= *ii && *ii <= nc - 1;
                if (ii && !ignore_in) {
                    auto idx = target == *ii;
                    target = target.masked_fill(idx, nc);
                    idx = preds.dim() > target.dim() ? idx.unsqueeze(1).repeat({1, nc, 1}) : idx;
                    preds = preds.masked_fill(idx, nc);
                }
                const auto width = ii && !ignore_in ? nc + 1 : nc;
                auto preds_oh = top_k > 1 ? refine_preds_oh(preds, target, top_k)
                                          : torch::one_hot(preds.to(torch::kLong), width);
                auto target_oh = torch::one_hot(target.to(torch::kLong), width);
                if (ii && ignore_in) {
                    target_oh.index_put_({target == *ii}, -1);
                } else if (ii) {
                    preds_oh = top_k == 1 ? preds_oh.slice(-1, 0, -1) : preds_oh;
                    target_oh = target_oh.slice(-1, 0, -1);
                    target_oh.index_put_({target == nc}, -1);
                }
                const auto global = o._multidim_average == multidim_average_mode::global;
                return stat_counts(preds_oh, target_oh, global ? std::vector<int64_t>{0, 1} : std::vector<int64_t>{1},
                                   false);
            }
            preds = preds.flatten();
            target = target.flatten();
            if (ii) {
                const auto keep = target != *ii;
                preds = preds.index({keep});
                target = target.index({keep});
            }
            if (o._average == average_mode::micro) {
                const auto tp = (preds == target).sum(), fp = (preds != target).sum(), fn = (preds != target).sum();
                return {tp, fp, nc * preds.numel() - (fp + fn + tp), fn};
            }
            const auto confmat =
                    bincount(target.to(torch::kLong) * nc + preds.to(torch::kLong), nc * nc).reshape({nc, nc});
            const auto tp = confmat.diag(), fp = confmat.sum(0) - tp, fn = confmat.sum(1) - tp;
            return {tp, fp, confmat.sum() - (fp + fn + tp), fn};
        }

        // _multilabel_stat_scores_format + _multilabel_stat_scores_update
        inline auto stat_update(const tensor_t &preds_in, const tensor_t &target_in,
                                const multilabel_stat_scores_option &o) -> stat_state {
            auto preds = preds_in.is_floating_point() ? normalize_logits(preds_in, false) > o._threshold : preds_in;
            preds = keep_dims(preds, 2);
            auto target = keep_dims(target_in, 2);
            if (o._ignore_index) {
                target = target.masked_fill(target == *o._ignore_index, -1);
            }
            const auto global = o._multidim_average == multidim_average_mode::global;
            return stat_counts(preds, target, global ? std::vector<int64_t>{0, -1} : std::vector<int64_t>{-1}, true);
        }

        // _binary_stat_scores_compute / _multiclass_ / _multilabel_stat_scores_compute: [tp, fp, tn, fn, support]
        inline auto stat_scores_compute(const stat_state &s, std::optional<average_mode> average, bool samplewise,
                                        bool multilabel) -> tensor_t {
            if (!average) {
                return torch::stack({s.tp, s.fp, s.tn, s.fn, s.tp + s.fn}, samplewise ? 1 : 0).squeeze();
            }
            const auto res = torch::stack({s.tp, s.fp, s.tn, s.fn, s.tp + s.fn}, -1);
            const auto w = s.tp + s.fn;
            const int64_t d = samplewise ? 1 : 0;
            switch (*average) {
                case average_mode::micro:
                    return res.dim() > 1 ? res.sum(d) : res;
                case average_mode::macro:
                    return res.to(torch::kFloat32).mean(d);
                case average_mode::weighted:
                    return (res * (samplewise && !multilabel ? w / w.sum(-1, true) : w / w.sum()).unsqueeze(-1)).sum(d);
                default:
                    return res;
            }
        }

        // _adjust_weights_safe_divide
        inline auto adjust_weights(const tensor_t &score, average_mode average, bool multilabel, const stat_state &s,
                                   int64_t top_k) -> tensor_t {
            if (average == average_mode::none) {
                return score;
            }
            auto w = average == average_mode::weighted ? s.tp + s.fn : torch::ones_like(score);
            if (average != average_mode::weighted && !multilabel) {
                w.masked_fill_(top_k == 1 ? s.tp + s.fp + s.fn == 0 : s.tp + s.fn == 0, 0.0);
            }
            return safe_divide(w * score, w.sum(-1, true)).sum(-1);
        }

        enum class stat_metric { stat_scores, accuracy, precision, recall, fbeta, f1, specificity, npv, hamming };

        // torchmetrics' _accuracy_reduce, _precision_recall_reduce, _fbeta_reduce, _specificity_reduce,
        // _negative_predictive_value_reduce and _hamming_distance_reduce; average {} is their "binary".
        inline auto stat_reduce(stat_metric k, const stat_state &s, std::optional<average_mode> average,
                                bool samplewise, bool multilabel, int64_t top_k, double zero_division, double beta)
                -> tensor_t {
            auto [tp, fp, tn, fn] = s;
            if (average == average_mode::micro) {
                const int64_t d = samplewise ? 1 : 0;
                tp = tp.sum(d);
                fp = fp.sum(d);
                tn = tn.sum(d);
                fn = fn.sum(d);
            }
            const bool all_classes = !average || multilabel; // accuracy / hamming over tp + tn of every class
            tensor_t score;
            switch (k) {
                case stat_metric::accuracy:
                case stat_metric::hamming:
                    score = all_classes ? safe_divide(tp + tn, tp + tn + fp + fn) : safe_divide(tp, tp + fn);
                    score = k == stat_metric::hamming ? 1 - score : score;
                    top_k = k == stat_metric::hamming ? 1 : top_k;
                    break;
                case stat_metric::precision:
                case stat_metric::recall:
                    score = safe_divide(tp, tp + (k == stat_metric::precision ? fp : fn), zero_division);
                    break;
                case stat_metric::specificity:
                    score = safe_divide(tn, tn + fp);
                    top_k = 1;
                    break;
                case stat_metric::npv:
                    score = safe_divide(tn, tn + fn, zero_division);
                    break;
                default: { // fbeta, f1
                    const double beta2 = std::pow(k == stat_metric::f1 ? 1.0 : beta, 2.0);
                    score = safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp, zero_division);
                    top_k = 1;
                }
            }
            return !average || average == average_mode::micro ? score
                                                              : adjust_weights(score, *average, multilabel, s, top_k);
        }

        // The metric value of a stat state (the class's compute when cls: NegativePredictiveValue's classes ignore
        // zero_division like torchmetrics').
        inline auto stat_value(stat_metric k, const stat_state &s, const binary_stat_scores_option &o, bool cls)
                -> tensor_t {
            const bool sw = o._multidim_average == multidim_average_mode::samplewise;
            if (k == stat_metric::stat_scores) {
                return stat_scores_compute(s, std::nullopt, sw, false);
            }
            const auto zd = cls && k == stat_metric::npv ? 0.0 : o._zero_division;
            return stat_reduce(k, s, std::nullopt, sw, false, 1, zd, o._beta);
        }
        inline auto stat_value(stat_metric k, const stat_state &s, const multiclass_stat_scores_option &o, bool cls)
                -> tensor_t {
            const bool sw = o._multidim_average == multidim_average_mode::samplewise;
            if (k == stat_metric::stat_scores) {
                return stat_scores_compute(s, o._average, sw, false);
            }
            const auto zd = cls && k == stat_metric::npv ? 0.0 : o._zero_division;
            return stat_reduce(k, s, o._average, sw, false, o._top_k, zd, o._beta);
        }
        inline auto stat_value(stat_metric k, const stat_state &s, const multilabel_stat_scores_option &o, bool cls)
                -> tensor_t {
            const bool sw = o._multidim_average == multidim_average_mode::samplewise;
            if (k == stat_metric::stat_scores) {
                return stat_scores_compute(s, o._average, sw, true);
            }
            const auto zd = cls && k == stat_metric::npv ? 0.0 : o._zero_division;
            return stat_reduce(k, s, o._average, sw, true, 1, zd, o._beta);
        }

        template<class Opt>
        inline auto stat_functional(stat_metric k, const tensor_t &preds, const tensor_t &target, const Opt &o)
                -> tensor_t {
            if (auto e = stat_arg_error(o, k == stat_metric::fbeta); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = stat_tensor_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            return stat_value(k, stat_update(preds, target, o), o, false);
        }

        inline auto stat_size(const binary_stat_scores_option &) -> int64_t { return 1; }
        inline auto stat_size(const multiclass_stat_scores_option &o) -> int64_t {
            return o._average == average_mode::micro && o._top_k == 1 ? 1 : o._num_classes.value_or(1);
        }
        inline auto stat_size(const multilabel_stat_scores_option &o) -> int64_t { return o._num_labels; }

        // The options a class falls back to when its constructor rejects the given ones.
        inline auto fallback(const binary_stat_scores_option &) -> binary_stat_scores_option { return {}; }
        inline auto fallback(const multiclass_stat_scores_option &) -> multiclass_stat_scores_option {
            return multiclass_stat_scores_option().num_classes(2);
        }
        inline auto fallback(const multilabel_stat_scores_option &) -> multilabel_stat_scores_option {
            return multilabel_stat_scores_option().num_labels(2);
        }

        // torchmetrics _AbstractStatScores: global sums of tp / fp / tn / fn (starting at zeros(size)), or the
        // per-sample batches.
        template<class Opt, stat_metric K>
        class stat_class : public metric::detail::metric_base<stat_class<Opt, K>> {
        public:
            explicit stat_class(Opt opt = {}) : _opt(opt) {
                if (auto e = stat_arg_error(_opt, K == stat_metric::fbeta); !e.empty()) {
                    _opt = fallback(_opt);
                    fail(std::move(e));
                }
                _sum = {torch::zeros({stat_size(_opt)}, torch::kLong), torch::zeros({stat_size(_opt)}, torch::kLong),
                        torch::zeros({stat_size(_opt)}, torch::kLong), torch::zeros({stat_size(_opt)}, torch::kLong)};
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = stat_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto s = stat_update(preds, target, _opt);
                if (_opt._multidim_average == multidim_average_mode::samplewise) {
                    _batches.push_back(s);
                    return;
                }
                _sum = {_sum.tp.to(s.tp.device()) + s.tp, _sum.fp.to(s.tp.device()) + s.fp,
                        _sum.tn.to(s.tp.device()) + s.tn, _sum.fn.to(s.tp.device()) + s.fn};
            }

            auto compute() const -> tensor_t {
                auto s = _sum;
                if (_opt._multidim_average == multidim_average_mode::samplewise) {
                    std::vector<tensor_t> tp, fp, tn, fn;
                    for (const auto &b: _batches) {
                        tp.push_back(b.tp);
                        fp.push_back(b.fp);
                        tn.push_back(b.tn);
                        fn.push_back(b.fn);
                    }
                    s = {cat0(tp), cat0(fp), cat0(tn), cat0(fn)};
                }
                return squeeze_if_scalar(stat_value(K, s, _opt, true));
            }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            stat_state _sum;
            std::vector<stat_state> _batches;
        };
    } // namespace detail

    // ---------------- StatScores ----------------
    // [tp, fp, tn, fn, support] (int64; average macro / weighted: float32), per sample with multidim_average
    // samplewise, per class with average none.
    inline auto binary_stat_scores(const tensor_t &preds, const tensor_t &target,
                                   const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::stat_scores, preds, target, opt);
    }
    inline auto multiclass_stat_scores(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::stat_scores, preds, target, opt);
    }
    inline auto multilabel_stat_scores(const tensor_t &preds, const tensor_t &target,
                                       const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::stat_scores, preds, target, opt);
    }
    using BinaryStatScores = detail::stat_class<binary_stat_scores_option, detail::stat_metric::stat_scores>;
    using MulticlassStatScores = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::stat_scores>;
    using MultilabelStatScores = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::stat_scores>;

    // ---------------- Accuracy ----------------
    // binary / multilabel: (tp + tn) / all; multiclass: tp / (tp + fn) per class, averaged.
    inline auto binary_accuracy(const tensor_t &preds, const tensor_t &target,
                                const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::accuracy, preds, target, opt);
    }
    inline auto multiclass_accuracy(const tensor_t &preds, const tensor_t &target,
                                    const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::accuracy, preds, target, opt);
    }
    inline auto multilabel_accuracy(const tensor_t &preds, const tensor_t &target,
                                    const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::accuracy, preds, target, opt);
    }
    using BinaryAccuracy = detail::stat_class<binary_stat_scores_option, detail::stat_metric::accuracy>;
    using MulticlassAccuracy = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::accuracy>;
    using MultilabelAccuracy = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::accuracy>;

    // ---------------- Precision / Recall ----------------
    // tp / (tp + fp) and tp / (tp + fn); x / 0 = zero_division.
    inline auto binary_precision(const tensor_t &preds, const tensor_t &target,
                                 const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::precision, preds, target, opt);
    }
    inline auto multiclass_precision(const tensor_t &preds, const tensor_t &target,
                                     const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::precision, preds, target, opt);
    }
    inline auto multilabel_precision(const tensor_t &preds, const tensor_t &target,
                                     const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::precision, preds, target, opt);
    }
    inline auto binary_recall(const tensor_t &preds, const tensor_t &target, const binary_stat_scores_option &opt = {})
            -> tensor_t {
        return detail::stat_functional(detail::stat_metric::recall, preds, target, opt);
    }
    inline auto multiclass_recall(const tensor_t &preds, const tensor_t &target,
                                  const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::recall, preds, target, opt);
    }
    inline auto multilabel_recall(const tensor_t &preds, const tensor_t &target,
                                  const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::recall, preds, target, opt);
    }
    using BinaryPrecision = detail::stat_class<binary_stat_scores_option, detail::stat_metric::precision>;
    using MulticlassPrecision = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::precision>;
    using MultilabelPrecision = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::precision>;
    using BinaryRecall = detail::stat_class<binary_stat_scores_option, detail::stat_metric::recall>;
    using MulticlassRecall = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::recall>;
    using MultilabelRecall = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::recall>;

    // ---------------- FBetaScore / F1Score ----------------
    // (1 + beta^2) tp / ((1 + beta^2) tp + beta^2 fn + fp) with the option's beta (F1: 1).
    inline auto binary_fbeta_score(const tensor_t &preds, const tensor_t &target,
                                   const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::fbeta, preds, target, opt);
    }
    inline auto multiclass_fbeta_score(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::fbeta, preds, target, opt);
    }
    inline auto multilabel_fbeta_score(const tensor_t &preds, const tensor_t &target,
                                       const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::fbeta, preds, target, opt);
    }
    inline auto binary_f1_score(const tensor_t &preds, const tensor_t &target,
                                const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::f1, preds, target, opt);
    }
    inline auto multiclass_f1_score(const tensor_t &preds, const tensor_t &target,
                                    const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::f1, preds, target, opt);
    }
    inline auto multilabel_f1_score(const tensor_t &preds, const tensor_t &target,
                                    const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::f1, preds, target, opt);
    }
    using BinaryFBetaScore = detail::stat_class<binary_stat_scores_option, detail::stat_metric::fbeta>;
    using MulticlassFBetaScore = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::fbeta>;
    using MultilabelFBetaScore = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::fbeta>;
    using BinaryF1Score = detail::stat_class<binary_stat_scores_option, detail::stat_metric::f1>;
    using MulticlassF1Score = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::f1>;
    using MultilabelF1Score = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::f1>;

    // ---------------- Specificity / NegativePredictiveValue / HammingDistance ----------------
    // tn / (tn + fp); tn / (tn + fn) (x / 0 = zero_division in the functionals; torchmetrics' classes use 0);
    // 1 - accuracy.
    inline auto binary_specificity(const tensor_t &preds, const tensor_t &target,
                                   const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::specificity, preds, target, opt);
    }
    inline auto multiclass_specificity(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::specificity, preds, target, opt);
    }
    inline auto multilabel_specificity(const tensor_t &preds, const tensor_t &target,
                                       const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::specificity, preds, target, opt);
    }
    inline auto binary_negative_predictive_value(const tensor_t &preds, const tensor_t &target,
                                                 const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::npv, preds, target, opt);
    }
    inline auto multiclass_negative_predictive_value(const tensor_t &preds, const tensor_t &target,
                                                     const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::npv, preds, target, opt);
    }
    inline auto multilabel_negative_predictive_value(const tensor_t &preds, const tensor_t &target,
                                                     const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::npv, preds, target, opt);
    }
    inline auto binary_hamming_distance(const tensor_t &preds, const tensor_t &target,
                                        const binary_stat_scores_option &opt = {}) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::hamming, preds, target, opt);
    }
    inline auto multiclass_hamming_distance(const tensor_t &preds, const tensor_t &target,
                                            const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::hamming, preds, target, opt);
    }
    inline auto multilabel_hamming_distance(const tensor_t &preds, const tensor_t &target,
                                            const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::stat_functional(detail::stat_metric::hamming, preds, target, opt);
    }
    using BinarySpecificity = detail::stat_class<binary_stat_scores_option, detail::stat_metric::specificity>;
    using MulticlassSpecificity = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::specificity>;
    using MultilabelSpecificity = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::specificity>;
    using BinaryNegativePredictiveValue = detail::stat_class<binary_stat_scores_option, detail::stat_metric::npv>;
    using MulticlassNegativePredictiveValue =
            detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::npv>;
    using MultilabelNegativePredictiveValue =
            detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::npv>;
    using BinaryHammingDistance = detail::stat_class<binary_stat_scores_option, detail::stat_metric::hamming>;
    using MulticlassHammingDistance = detail::stat_class<multiclass_stat_scores_option, detail::stat_metric::hamming>;
    using MultilabelHammingDistance = detail::stat_class<multilabel_stat_scores_option, detail::stat_metric::hamming>;

    // ---------------- ExactMatch ----------------
    namespace detail {
        inline auto exact_match_arg_error(const multiclass_stat_scores_option &o) -> std::string {
            return o._num_classes ? count_error("num_classes", *o._num_classes)
                                  : "Argument `num_classes` is required for ExactMatch.";
        }
        inline auto exact_match_arg_error(const multilabel_stat_scores_option &o) -> std::string {
            const auto e = count_error("num_labels", o._num_labels);
            return e.empty() ? threshold_error(o._threshold) : e;
        }

        // {correct, total}: _multiclass_exact_match_update
        inline auto exact_match_update(const tensor_t &preds_in, const tensor_t &target_in,
                                       const multiclass_stat_scores_option &o) -> std::pair<tensor_t, tensor_t> {
            auto preds = keep_dims(preds_in.dim() == target_in.dim() + 1 ? preds_in.argmax(1) : preds_in, 1);
            const auto target = keep_dims(target_in, 1);
            if (o._ignore_index) {
                preds = preds.masked_fill(target == *o._ignore_index, *o._ignore_index);
            }
            const bool global = o._multidim_average == multidim_average_mode::global;
            const auto correct = (preds == target).sum(1) == preds.size(1);
            return {global ? correct.sum() : correct,
                    torch::tensor(global ? preds.size(0) : int64_t{1}, torch::device(correct.device()))};
        }
        // _multilabel_stat_scores_format + _multilabel_exact_match_update
        inline auto exact_match_update(const tensor_t &preds_in, const tensor_t &target_in,
                                       const multilabel_stat_scores_option &o) -> std::pair<tensor_t, tensor_t> {
            const auto L = o._num_labels;
            auto preds = preds_in.is_floating_point() ? normalize_logits(preds_in, false) > o._threshold : preds_in;
            preds = keep_dims(preds, 2);
            auto target = keep_dims(target_in, 2);
            if (o._ignore_index) {
                target = target.masked_fill(target == *o._ignore_index, -1);
                target = torch::where(target == -1, preds.to(torch::kLong), target);
            }
            const bool global = o._multidim_average == multidim_average_mode::global;
            if (global) {
                preds = preds.movedim(1, -1).reshape({-1, L});
                target = target.movedim(1, -1).reshape({-1, L});
            }
            const auto correct = ((preds == target).sum(1) == L).sum(-1);
            return {correct, torch::tensor(preds.size(global ? 0 : 2), torch::device(correct.device()))};
        }

        template<class Opt>
        class exact_match_class : public metric::detail::metric_base<exact_match_class<Opt>> {
        public:
            explicit exact_match_class(Opt opt = {}) : _opt(opt) {
                if (auto e = exact_match_arg_error(_opt); !e.empty()) {
                    _opt = fallback(_opt);
                    fail(std::move(e));
                }
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = stat_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto [correct, total] = exact_match_update(preds, target, _opt);
                if (_opt._multidim_average == multidim_average_mode::samplewise) {
                    _correct_batches.push_back(correct);
                    _total = total; // torchmetrics keeps the last batch's count
                    return;
                }
                _correct = _correct.to(correct.device()) + correct;
                _total = _total.to(correct.device()) + total;
            }

            auto compute() const -> tensor_t {
                return squeeze_if_scalar(safe_divide(
                        _opt._multidim_average == multidim_average_mode::samplewise ? cat0(_correct_batches) : _correct,
                        _total));
            }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            tensor_t _correct = torch::zeros({1}, torch::kLong), _total = torch::zeros({1}, torch::kLong);
            std::vector<tensor_t> _correct_batches;
        };

        template<class Opt>
        inline auto exact_match(const tensor_t &preds, const tensor_t &target, const Opt &o) -> tensor_t {
            if (auto e = exact_match_arg_error(o); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = stat_tensor_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            const auto [correct, total] = exact_match_update(preds, target, o);
            return safe_divide(correct, total);
        }
    } // namespace detail

    // Fraction of samples whose every position (multiclass) / label (multilabel) is right; ignore_index positions count
    // as right. samplewise: per sample.
    inline auto multiclass_exact_match(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_stat_scores_option &opt) -> tensor_t {
        return detail::exact_match(preds, target, opt);
    }
    inline auto multilabel_exact_match(const tensor_t &preds, const tensor_t &target,
                                       const multilabel_stat_scores_option &opt) -> tensor_t {
        return detail::exact_match(preds, target, opt);
    }
    using MulticlassExactMatch = detail::exact_match_class<multiclass_stat_scores_option>;
    using MultilabelExactMatch = detail::exact_match_class<multilabel_stat_scores_option>;

    // ---------------- confusion-matrix family ----------------
    namespace detail {
        inline auto confmat_tensor_error(const tensor_t &preds, const tensor_t &target,
                                         const binary_confusion_matrix_option &o) -> std::string {
            return binary_tensor_error(preds, target, false, o._ignore_index);
        }
        inline auto confmat_tensor_error(const tensor_t &preds, const tensor_t &target,
                                         const multiclass_confusion_matrix_option &o) -> std::string {
            return multiclass_tensor_error(preds, target, o._num_classes, false, o._ignore_index);
        }
        inline auto confmat_tensor_error(const tensor_t &preds, const tensor_t &target,
                                         const multilabel_confusion_matrix_option &o) -> std::string {
            return multilabel_tensor_error(preds, target, o._num_labels, false, o._ignore_index);
        }

        // _binary_confusion_matrix_format: flattened, ignore_index dropped, probabilities (labels when convert)
        inline auto binary_format(const tensor_t &preds_in, const tensor_t &target_in, double threshold,
                                  std::optional<int64_t> ii, bool convert) -> std::pair<tensor_t, tensor_t> {
            auto preds = preds_in.flatten();
            auto target = target_in.flatten();
            if (ii) {
                const auto keep = target != *ii;
                preds = preds.index({keep});
                target = target.index({keep});
            }
            if (preds.is_floating_point()) {
                preds = normalize_logits(preds, false);
                preds = convert ? preds > threshold : preds;
            }
            return {preds, target};
        }
        // _multiclass_confusion_matrix_format: labels [M] (convert) or scores [M, C]
        inline auto multiclass_format(const tensor_t &preds_in, const tensor_t &target_in, std::optional<int64_t> ii,
                                      bool convert) -> std::pair<tensor_t, tensor_t> {
            auto preds = preds_in.dim() == target_in.dim() + 1 && convert ? preds_in.argmax(1) : preds_in;
            preds = convert ? preds.flatten() : preds.movedim(1, -1).reshape({-1, preds.size(1)});
            auto target = target_in.flatten();
            if (ii) {
                const auto keep = target != *ii;
                preds = preds.index({keep});
                target = target.index({keep});
            }
            return {preds, target};
        }
        // _multilabel_confusion_matrix_format: [M, L], ignore_index entries set to -4 L
        inline auto multilabel_format(const tensor_t &preds_in, const tensor_t &target_in, int64_t L, double threshold,
                                      std::optional<int64_t> ii, bool convert) -> std::pair<tensor_t, tensor_t> {
            auto preds = preds_in;
            if (preds.is_floating_point()) {
                preds = normalize_logits(preds, false);
                preds = convert ? preds > threshold : preds;
            }
            preds = preds.movedim(1, -1).reshape({-1, L});
            auto target = target_in.movedim(1, -1).reshape({-1, L});
            if (ii) {
                const auto idx = target == *ii;
                preds = preds.masked_fill(idx, -4 * L);
                target = target.masked_fill(idx, -4 * L);
            }
            return {preds, target};
        }

        inline auto confmat_update(const tensor_t &preds_in, const tensor_t &target_in,
                                   const binary_confusion_matrix_option &o) -> tensor_t {
            const auto [preds, target] = binary_format(preds_in, target_in, o._threshold, o._ignore_index, true);
            return bincount((target * 2 + preds).to(torch::kLong), 4).reshape({2, 2});
        }
        inline auto confmat_update(const tensor_t &preds_in, const tensor_t &target_in,
                                   const multiclass_confusion_matrix_option &o) -> tensor_t {
            const auto [preds, target] = multiclass_format(preds_in, target_in, o._ignore_index, true);
            const auto C = o._num_classes;
            return bincount(target.to(torch::kLong) * C + preds.to(torch::kLong), C * C).reshape({C, C});
        }
        inline auto confmat_update(const tensor_t &preds_in, const tensor_t &target_in,
                                   const multilabel_confusion_matrix_option &o) -> tensor_t {
            const auto L = o._num_labels;
            const auto [preds, target] = multilabel_format(preds_in, target_in, L, o._threshold, o._ignore_index, true);
            auto mapping = (2 * target + preds + 4 * torch::arange(L, target.options().dtype(torch::kLong))).flatten();
            mapping = mapping.index({mapping >= 0});
            return bincount(mapping, 4 * L).reshape({L, 2, 2});
        }

        // _confusion_matrix_reduce
        inline auto confmat_normalize(const tensor_t &confmat, normalize_mode n) -> tensor_t {
            if (n == normalize_mode::none) {
                return confmat;
            }
            auto c = confmat.is_floating_point() ? confmat : confmat.to(torch::kFloat32);
            c = n == normalize_mode::true_  ? c / c.sum(-1, true)
                : n == normalize_mode::pred ? c / c.sum(-2, true)
                                            : c / c.sum({-2, -1}, true);
            return c.masked_fill(c.isnan(), 0);
        }

        // _jaccard_index_reduce; average {} is "binary"
        inline auto jaccard_reduce(const tensor_t &confmat, std::optional<average_mode> average,
                                   std::optional<int64_t> ii, double zero_division) -> tensor_t {
            const auto c = confmat.to(torch::kFloat32);
            if (!average) {
                return safe_divide(c[1][1], c[0][1] + c[1][0] + c[1][1], zero_division);
            }
            const bool ignore_cond = ii && 0 <= *ii && *ii < c.size(0);
            const bool multilabel = c.dim() == 3;
            const auto tp = multilabel ? c.select(-1, 1).select(-1, 1) : torch::diag(c);
            auto num = tp;
            auto denom = multilabel ? tp + c.select(1, 0).select(-1, 1) + c.select(1, 1).select(-1, 0)
                                    : c.sum(0) + c.sum(1) - tp;
            if (*average == average_mode::micro) {
                num = num.sum();
                denom = ignore_cond ? denom.sum() - denom[*ii] : denom.sum();
            }
            const auto jaccard = safe_divide(num, denom, zero_division);
            if (*average == average_mode::none || *average == average_mode::micro) {
                return jaccard;
            }
            tensor_t w;
            if (*average == average_mode::weighted) {
                w = multilabel ? tp + c.select(1, 1).select(-1, 0) : c.sum(1);
            } else {
                w = torch::ones_like(jaccard);
                if (ignore_cond) {
                    w[*ii] = 0.0;
                }
                if (!multilabel) {
                    w.masked_fill_(c.sum(1) + c.sum(0) == 0, 0.0);
                }
            }
            return (w * jaccard / w.sum()).sum();
        }

        // Python's builtin sum() over a 1-D float32 tensor: 0 + x[0] + x[1] + ... in float32.
        inline auto py_sum(const tensor_t &x) -> tensor_t {
            const auto c = x.to(torch::kCPU).contiguous();
            float acc = 0.0f;
            for (int64_t i = 0; i < c.numel(); ++i) {
                acc += c.data_ptr<float>()[i];
            }
            return torch::full({}, acc, x.options());
        }

        // _matthews_corrcoef_reduce (a perfect / inverted 2 x 2 table returns int64 +-1 like torchmetrics)
        inline auto mcc_reduce(const tensor_t &confmat_in) -> tensor_t {
            const auto confmat = confmat_in.dim() == 3 ? confmat_in.sum(0) : confmat_in;
            const bool two = confmat.numel() == 4;
            const auto v = confmat.reshape(-1);
            if (two) {
                const bool agree = (v[3] + v[0] != 0).item<bool>(), disagree = (v[1] + v[2] != 0).item<bool>();
                if (agree != disagree) {
                    return torch::full({}, agree ? 1 : -1, confmat.options());
                }
            }
            const auto c = confmat.to(torch::kFloat32);
            const auto tk = c.sum(-1), pk = c.sum(-2), tr = torch::trace(c), s = c.sum();
            auto numerator = tr * s - py_sum(tk * pk);
            auto denom = (s.pow(2) - py_sum(pk * pk)) * (s.pow(2) - py_sum(tk * tk));
            if ((denom != 0).item<bool>()) {
                return numerator / torch::sqrt(denom);
            }
            if (!two) {
                return torch::zeros({}, c.options());
            }
            const auto tn = v[0], fp = v[1], fn = v[2], tp = v[3];
            const auto z = [](const tensor_t &x) { return (x == 0).item<bool>(); };
            const auto eps = torch::full({}, std::numeric_limits<float>::epsilon(), c.options());
            if (z(fn) && z(tn)) {
                numerator = torch::sqrt(eps) * (tp - fp);
            } else if (z(fp) && z(tn)) {
                numerator = torch::sqrt(eps) * (tp - fn);
            } else if (z(tp) && z(fn)) {
                numerator = torch::sqrt(eps) * (tn - fp);
            } else if (z(tp) && z(fp)) {
                numerator = torch::sqrt(eps) * (tn - fn);
            } else if (z(tp) || z(tn) || z(fp) || z(fn)) { // reachable through float32 rounding of huge counts only
                numerator = z(tp) ? tn - fp * fn : z(tn) ? tp - fp * fn : tp * tn;
            } else {
                return torch::zeros({}, c.options());
            }
            denom = (tp + fp + eps) * (tp + fn + eps) * (tn + fp + eps) * (tn + fn + eps);
            return numerator / torch::sqrt(denom);
        }

        // _cohen_kappa_reduce
        inline auto kappa_reduce(const tensor_t &confmat, kappa_weights_mode weights) -> tensor_t {
            const auto c = confmat.is_floating_point() ? confmat : confmat.to(torch::kFloat32);
            const auto n = c.size(0);
            const auto sum0 = c.sum(0, true), sum1 = c.sum(1, true);
            const auto expected = torch::matmul(sum1, sum0) / sum0.sum();
            tensor_t w;
            if (weights == kappa_weights_mode::none) {
                w = torch::ones_like(c).flatten();
                w.slice(0, 0, w.numel(), n + 1).fill_(0);
                w = w.reshape({n, n});
            } else {
                w = torch::zeros_like(c) + torch::arange(n, c.options());
                w = weights == kappa_weights_mode::linear ? torch::abs(w - w.t()) : torch::pow(w - w.t(), 2.0);
            }
            return 1 - torch::sum(w * c) / torch::sum(w * expected);
        }

        enum class confmat_metric { confusion_matrix, jaccard, mcc, cohen_kappa };

        inline auto confmat_value(confmat_metric k, const tensor_t &cm, std::optional<average_mode> average,
                                  std::optional<int64_t> ii, normalize_mode normalize, double zero_division,
                                  kappa_weights_mode weights) -> tensor_t {
            switch (k) {
                case confmat_metric::confusion_matrix:
                    return confmat_normalize(cm, normalize);
                case confmat_metric::jaccard:
                    return jaccard_reduce(cm, average, ii, zero_division);
                case confmat_metric::mcc:
                    return mcc_reduce(cm);
                default:
                    return kappa_reduce(cm, weights);
            }
        }
        // cls: torchmetrics' MultilabelJaccardIndex.compute leaves ignore_index out of the reduction.
        inline auto confmat_value(confmat_metric k, const tensor_t &cm, const binary_confusion_matrix_option &o, bool)
                -> tensor_t {
            return confmat_value(k, cm, std::nullopt, std::nullopt, o._normalize, o._zero_division, o._weights);
        }
        inline auto confmat_value(confmat_metric k, const tensor_t &cm, const multiclass_confusion_matrix_option &o,
                                  bool) -> tensor_t {
            return confmat_value(k, cm, o._average, o._ignore_index, o._normalize, o._zero_division, o._weights);
        }
        inline auto confmat_value(confmat_metric k, const tensor_t &cm, const multilabel_confusion_matrix_option &o,
                                  bool cls) -> tensor_t {
            return confmat_value(k, cm, o._average, cls ? std::nullopt : o._ignore_index, o._normalize,
                                 o._zero_division, kappa_weights_mode::none);
        }

        inline auto confmat_shape(const binary_confusion_matrix_option &) -> std::vector<int64_t> { return {2, 2}; }
        inline auto confmat_shape(const multiclass_confusion_matrix_option &o) -> std::vector<int64_t> {
            return {o._num_classes, o._num_classes};
        }
        inline auto confmat_shape(const multilabel_confusion_matrix_option &o) -> std::vector<int64_t> {
            return {o._num_labels, 2, 2};
        }
        inline auto fallback(const binary_confusion_matrix_option &) -> binary_confusion_matrix_option { return {}; }
        inline auto fallback(const multiclass_confusion_matrix_option &) -> multiclass_confusion_matrix_option {
            return multiclass_confusion_matrix_option().num_classes(2);
        }
        inline auto fallback(const multilabel_confusion_matrix_option &) -> multilabel_confusion_matrix_option {
            return multilabel_confusion_matrix_option().num_labels(2);
        }

        template<class Opt>
        inline auto confmat_functional(confmat_metric k, const tensor_t &preds, const tensor_t &target, const Opt &o)
                -> tensor_t {
            if (auto e = confmat_arg_error(o); !e.empty()) {
                return invalid(std::move(e));
            }
            if (auto e = confmat_tensor_error(preds, target, o); !e.empty()) {
                return invalid(std::move(e));
            }
            return confmat_value(k, confmat_update(preds, target, o), o, false);
        }

        // torchmetrics' XxxConfusionMatrix state: the int64 confusion matrix summed over batches.
        template<class Opt, confmat_metric K>
        class confmat_class : public metric::detail::metric_base<confmat_class<Opt, K>> {
        public:
            explicit confmat_class(Opt opt = {}) : _opt(opt) {
                if (auto e = confmat_arg_error(_opt); !e.empty()) {
                    _opt = fallback(_opt);
                    fail(std::move(e));
                }
                _confmat = torch::zeros(confmat_shape(_opt), torch::kLong);
            }

            auto update(const tensor_t &preds, const tensor_t &target) -> void {
                if (auto e = confmat_tensor_error(preds, target, _opt); !e.empty()) {
                    return fail(std::move(e));
                }
                const auto cm = confmat_update(preds, target, _opt);
                _confmat = _confmat.to(cm.device()) + cm;
            }

            auto compute() const -> tensor_t { return squeeze_if_scalar(confmat_value(K, _confmat, _opt, true)); }

            auto options() const -> const Opt & { return _opt; }

        private:
            Opt _opt;
            tensor_t _confmat;
        };
    } // namespace detail

    // ---------------- ConfusionMatrix ----------------
    // int64 [2, 2] / [C, C] / [L, 2, 2] (rows: target, columns: prediction), float32 when normalized (NaN -> 0).
    inline auto binary_confusion_matrix(const tensor_t &preds, const tensor_t &target,
                                        const binary_confusion_matrix_option &opt = {}) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::confusion_matrix, preds, target, opt);
    }
    inline auto multiclass_confusion_matrix(const tensor_t &preds, const tensor_t &target,
                                            const multiclass_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::confusion_matrix, preds, target, opt);
    }
    inline auto multilabel_confusion_matrix(const tensor_t &preds, const tensor_t &target,
                                            const multilabel_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::confusion_matrix, preds, target, opt);
    }
    using BinaryConfusionMatrix =
            detail::confmat_class<binary_confusion_matrix_option, detail::confmat_metric::confusion_matrix>;
    using MulticlassConfusionMatrix =
            detail::confmat_class<multiclass_confusion_matrix_option, detail::confmat_metric::confusion_matrix>;
    using MultilabelConfusionMatrix =
            detail::confmat_class<multilabel_confusion_matrix_option, detail::confmat_metric::confusion_matrix>;

    // ---------------- JaccardIndex ----------------
    // tp / (tp + fp + fn) from the confusion matrix (x / 0 = zero_division), averaged over classes.
    inline auto binary_jaccard_index(const tensor_t &preds, const tensor_t &target,
                                     const binary_confusion_matrix_option &opt = {}) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::jaccard, preds, target, opt);
    }
    inline auto multiclass_jaccard_index(const tensor_t &preds, const tensor_t &target,
                                         const multiclass_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::jaccard, preds, target, opt);
    }
    inline auto multilabel_jaccard_index(const tensor_t &preds, const tensor_t &target,
                                         const multilabel_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::jaccard, preds, target, opt);
    }
    using BinaryJaccardIndex = detail::confmat_class<binary_confusion_matrix_option, detail::confmat_metric::jaccard>;
    using MulticlassJaccardIndex =
            detail::confmat_class<multiclass_confusion_matrix_option, detail::confmat_metric::jaccard>;
    using MultilabelJaccardIndex =
            detail::confmat_class<multilabel_confusion_matrix_option, detail::confmat_metric::jaccard>;

    // ---------------- MatthewsCorrCoef ----------------
    // float32 (int64 +-1 for a perfect / inverted binary table, as torchmetrics); multilabel: over the summed tables.
    inline auto binary_matthews_corrcoef(const tensor_t &preds, const tensor_t &target,
                                         const binary_confusion_matrix_option &opt = {}) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::mcc, preds, target, opt);
    }
    inline auto multiclass_matthews_corrcoef(const tensor_t &preds, const tensor_t &target,
                                             const multiclass_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::mcc, preds, target, opt);
    }
    inline auto multilabel_matthews_corrcoef(const tensor_t &preds, const tensor_t &target,
                                             const multilabel_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::mcc, preds, target, opt);
    }
    using BinaryMatthewsCorrCoef = detail::confmat_class<binary_confusion_matrix_option, detail::confmat_metric::mcc>;
    using MulticlassMatthewsCorrCoef =
            detail::confmat_class<multiclass_confusion_matrix_option, detail::confmat_metric::mcc>;
    using MultilabelMatthewsCorrCoef =
            detail::confmat_class<multilabel_confusion_matrix_option, detail::confmat_metric::mcc>;

    // ---------------- CohenKappa ----------------
    // 1 - sum(w * confmat) / sum(w * expected) with the option's weights (none: 0 on the diagonal, else 1).
    inline auto binary_cohen_kappa(const tensor_t &preds, const tensor_t &target,
                                   const binary_confusion_matrix_option &opt = {}) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::cohen_kappa, preds, target, opt);
    }
    inline auto multiclass_cohen_kappa(const tensor_t &preds, const tensor_t &target,
                                       const multiclass_confusion_matrix_option &opt) -> tensor_t {
        return detail::confmat_functional(detail::confmat_metric::cohen_kappa, preds, target, opt);
    }
    using BinaryCohenKappa = detail::confmat_class<binary_confusion_matrix_option, detail::confmat_metric::cohen_kappa>;
    using MulticlassCohenKappa =
            detail::confmat_class<multiclass_confusion_matrix_option, detail::confmat_metric::cohen_kappa>;
} // namespace torchmedia::metric::classification
#endif // LIB_TORCH_MEDIA_METRIC_CLASSIFICATION_STAT_HPP
