#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_SEGMENTATION_HPP
#define LIB_TORCH_MEDIA_METRIC_SEGMENTATION_HPP
// Segmentation metrics (develop_log/2026-10-23/progress11): torchmetrics 1.9 torchmetrics.segmentation MeanIoU,
// DiceScore and GeneralizedDiceScore with their functional forms, the same libtorch calls (float32 results).
// Inputs: one_hot [N, C, ...] (integer one-hot masks, or floating scores reduced by argmax over C) or index [N, ...]
// class indices (int64, num_classes required). torchmetrics' "mixed" input format and HausdorffDistance are not
// provided.
#include "_common.hpp"

namespace torchmedia::metric::segmentation {
    enum class input_format_mode { one_hot, index };

    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;
        using metric::detail::safe_divide;
        using metric::detail::shape_error;
        using metric::detail::squeeze_if_scalar;

        inline auto format_logits(const tensor_t &x, int64_t num_classes) -> tensor_t {
            return x.is_floating_point() ? torch::one_hot(x.argmax(1), num_classes).movedim(-1, 1) : x;
        }

        // torchmetrics _segmentation_inputs_format: one-hot [N, C, ...] pair (background channel dropped unless
        // include_background); an error leaves both undefined.
        inline auto inputs_format(const char *fn, const tensor_t &preds_in, const tensor_t &target_in,
                                  bool include_background, std::optional<int64_t> num_classes, input_format_mode format,
                                  std::string &error) -> std::pair<tensor_t, tensor_t> {
            error = shape_error(fn, preds_in, target_in);
            if (!error.empty()) {
                return {};
            }
            tensor_t preds, target;
            if (format == input_format_mode::index) {
                preds = torch::one_hot(preds_in, *num_classes).movedim(-1, 1);
                target = torch::one_hot(target_in, *num_classes).movedim(-1, 1);
            } else {
                const auto n = num_classes ? *num_classes : preds_in.dim() < 2 ? 0 : preds_in.size(1);
                if (n == 0) {
                    error = std::format("{}: cannot determine a positive `num_classes` from preds of shape {}.", fn,
                                        c10::str(preds_in.sizes()));
                    return {};
                }
                preds = format_logits(preds_in, n);
                target = format_logits(target_in, n);
            }
            if (preds.dim() < 3) {
                error = std::format("{}: expected both `preds` and `target` to have at least 3 dimensions, but got {}.",
                                    fn, preds.dim());
                return {};
            }
            if (!include_background) {
                preds = preds.size(1) > 1 ? preds.slice(1, 1) : preds;
                target = target.size(1) > 1 ? target.slice(1, 1) : target;
            }
            return {preds, target};
        }

        inline auto spatial_dims(const tensor_t &x) -> std::vector<int64_t> {
            std::vector<int64_t> dims;
            for (int64_t d = 2; d < x.dim(); ++d) {
                dims.push_back(d);
            }
            return dims;
        }

        inline auto num_classes_error(std::optional<int64_t> num_classes, bool required) -> std::string {
            if (required && !num_classes) {
                return "Argument `num_classes` must be provided when `input_format` is 'index'.";
            }
            return num_classes && *num_classes <= 0
                           ? std::format("Expected argument `num_classes` to be a positive integer, but got {}.",
                                         *num_classes)
                           : std::string{};
        }
    } // namespace detail

    // ---------------- MeanIoU ----------------
    typedef struct mean_iou_option {
        std::optional<int64_t> _num_classes; // required for index inputs; one_hot: {} = preds.size(1)
        bool _include_background = true;
        bool _per_class = false;
        input_format_mode _input_format = input_format_mode::one_hot;

        auto num_classes(std::optional<int64_t> v) -> mean_iou_option & {
            _num_classes = v;
            return *this;
        }
        auto include_background(bool v) -> mean_iou_option & {
            _include_background = v;
            return *this;
        }
        auto per_class(bool v) -> mean_iou_option & {
            _per_class = v;
            return *this;
        }
        auto input_format(input_format_mode v) -> mean_iou_option & {
            _input_format = v;
            return *this;
        }
    } mean_iou_option_t;

    namespace detail {
        // _mean_iou_update: per-sample, per-class {intersection, union}.
        inline auto mean_iou_update(const tensor_t &preds_in, const tensor_t &target_in, const mean_iou_option &opt,
                                    std::string &error) -> std::pair<tensor_t, tensor_t> {
            error = num_classes_error(opt._num_classes, opt._input_format == input_format_mode::index);
            if (!error.empty()) {
                return {};
            }
            auto preds = preds_in, target = target_in;
            if (opt._input_format == input_format_mode::index) {
                // _mean_iou_reshape_args: [L] -> [1, 1, L], [H, W] -> [1, H, W]
                preds = preds.dim() == 1 ? preds.view({1, 1, -1}) : preds.dim() == 2 ? preds.unsqueeze(0) : preds;
                target = target.dim() == 1 ? target.view({1, 1, -1}) : target.dim() == 2 ? target.unsqueeze(0) : target;
            }
            const auto [p, t] = inputs_format("mean_iou", preds, target, opt._include_background, opt._num_classes,
                                              opt._input_format, error);
            if (!error.empty()) {
                return {};
            }
            const auto dims = spatial_dims(p);
            const auto intersection = torch::sum(p.bitwise_and(t), dims);
            return {intersection, torch::sum(t, dims) + torch::sum(p, dims) - intersection};
        }
    } // namespace detail

    // torchmetrics mean_iou: per sample [N], the mean IoU over classes present in preds or target; per_class: [N, C]
    // with -1 for a class absent from both.
    inline auto mean_iou(const tensor_t &preds, const tensor_t &target, const mean_iou_option &opt = {}) -> tensor_t {
        std::string e;
        const auto [intersection, uni] = detail::mean_iou_update(preds, target, opt, e);
        if (!e.empty()) {
            return detail::invalid(std::move(e));
        }
        const auto scores = detail::safe_divide(intersection, uni, std::nan(""));
        return opt._per_class ? scores.nan_to_num(-1.0) : scores.nansum(-1) / (uni > 0).sum(-1);
    }

    // Accumulates IoU sums and counts of (sample, class) pairs present in preds or target: compute() is their ratio,
    // per class ([C], -1 for a class never present) or overall.
    class MeanIoU : public metric::detail::metric_base<MeanIoU> {
    public:
        explicit MeanIoU(mean_iou_option opt = {}) : _opt(opt) {
            if (auto e = detail::num_classes_error(_opt._num_classes, _opt._input_format == input_format_mode::index);
                !e.empty()) {
                _opt = {};
                detail::fail(std::move(e));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            std::string e;
            const auto [intersection, uni] = detail::mean_iou_update(preds, target, _opt, e);
            if (!e.empty()) {
                return detail::fail(std::move(e));
            }
            const auto score = detail::safe_divide(intersection, uni, 0.0);
            const auto valid = uni > 0;
            if (!_count.defined()) {
                const auto c = uni.size(1);
                _score = torch::zeros({_opt._per_class ? c : 1}, score.options());
                _count = torch::zeros({c}, score.options());
            }
            if (_opt._per_class) {
                _score = _score + (score * valid).sum(0);
                _count = _count + valid.sum(0);
            } else {
                _score = _score + (score * valid).sum();
                _count = _count + valid.sum();
            }
        }

        auto compute() const -> tensor_t {
            // before any update: torchmetrics' initial zero state (-1 per class, NaN overall)
            const auto c = _opt._num_classes ? *_opt._num_classes - !_opt._include_background : 1;
            const auto score = _score.defined() ? _score : torch::zeros({_opt._per_class ? c : 1});
            const auto out = score / (_count.defined() ? _count : torch::zeros({c}));
            return detail::squeeze_if_scalar(_opt._per_class ? out.nan_to_num(-1.0) : out.nanmean());
        }

        auto options() const -> const mean_iou_option & { return _opt; }

    private:
        mean_iou_option _opt;
        tensor_t _score, _count;
    };

    // ---------------- DiceScore ----------------
    enum class dice_average_mode { micro, macro, weighted, none };
    enum class aggregation_level_mode { samplewise, global };

    typedef struct dice_score_option {
        int64_t _num_classes = 0; // required (> 0)
        bool _include_background = true;
        dice_average_mode _average = dice_average_mode::macro;
        input_format_mode _input_format = input_format_mode::one_hot;
        aggregation_level_mode _aggregation_level = aggregation_level_mode::samplewise;

        auto num_classes(int64_t v) -> dice_score_option & {
            _num_classes = v;
            return *this;
        }
        auto include_background(bool v) -> dice_score_option & {
            _include_background = v;
            return *this;
        }
        auto average(dice_average_mode v) -> dice_score_option & {
            _average = v;
            return *this;
        }
        auto input_format(input_format_mode v) -> dice_score_option & {
            _input_format = v;
            return *this;
        }
        auto aggregation_level(aggregation_level_mode v) -> dice_score_option & {
            _aggregation_level = v;
            return *this;
        }
    } dice_score_option_t;

    namespace detail {
        struct dice_state {
            tensor_t numerator, denominator, support; // [N, C]
        };

        inline auto dice_update(const tensor_t &preds_in, const tensor_t &target_in, const dice_score_option &opt,
                                std::string &error) -> dice_state {
            error = num_classes_error(opt._num_classes, false);
            if (!error.empty()) {
                return {};
            }
            const auto [p, t] = inputs_format("dice_score", preds_in, target_in, opt._include_background,
                                              opt._num_classes, opt._input_format, error);
            if (!error.empty()) {
                return {};
            }
            const auto dims = spatial_dims(t);
            const auto intersection = torch::sum(p * t, dims);
            const auto target_sum = torch::sum(t, dims);
            return {2 * intersection, torch::sum(p, dims) + target_sum, target_sum};
        }

        // _dice_score_compute
        inline auto dice_compute(tensor_t numerator, tensor_t denominator, tensor_t support,
                                 const dice_score_option &opt) -> tensor_t {
            if (opt._aggregation_level == aggregation_level_mode::global) {
                numerator = torch::sum(numerator, 0).unsqueeze(0);
                denominator = torch::sum(denominator, 0).unsqueeze(0);
                support = torch::sum(support, 0);
            }
            const double nan = std::nan("");
            if (opt._average == dice_average_mode::micro) {
                return safe_divide(torch::sum(numerator, -1), torch::sum(denominator, -1), nan);
            }
            const auto dice = safe_divide(numerator, denominator, nan);
            if (opt._average != dice_average_mode::weighted) {
                return opt._average == dice_average_mode::macro ? torch::nanmean(dice, -1) : dice;
            }
            const auto weights = safe_divide(support, torch::sum(support, -1, true), nan);
            return torch::nansum(dice * weights, -1).masked_fill_(dice.isnan().all(-1), nan);
        }
    } // namespace detail

    // torchmetrics dice_score: per sample [N] (aggregation samplewise) or [1] (global); average none -> [., C].
    inline auto dice_score(const tensor_t &preds, const tensor_t &target, const dice_score_option &opt) -> tensor_t {
        std::string e;
        const auto s = detail::dice_update(preds, target, opt, e);
        if (!e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::dice_compute(s.numerator, s.denominator, s.support, opt);
    }

    // Keeps every batch's per-sample counts; compute() is dice_score over all of them, NaN-averaged over samples.
    class DiceScore : public metric::detail::metric_base<DiceScore> {
    public:
        explicit DiceScore(dice_score_option opt) : _opt(opt) {
            if (_opt._num_classes <= 0) {
                const auto n = _opt._num_classes;
                _opt._num_classes = 1;
                detail::fail(std::format("Expected argument `num_classes` to be a positive integer, but got {}.", n));
            }
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            std::string e;
            auto s = detail::dice_update(preds, target, _opt, e);
            if (!e.empty()) {
                return detail::fail(std::move(e));
            }
            _numerator.push_back(s.numerator);
            _denominator.push_back(s.denominator);
            _support.push_back(s.support);
        }

        auto compute() const -> tensor_t {
            return detail::squeeze_if_scalar(
                    detail::dice_compute(torch::cat(_numerator), torch::cat(_denominator), torch::cat(_support), _opt)
                            .nanmean(0));
        }

        auto options() const -> const dice_score_option & { return _opt; }

    private:
        dice_score_option _opt;
        std::vector<tensor_t> _numerator, _denominator, _support;
    };

    // ---------------- GeneralizedDiceScore ----------------
    enum class weight_type_mode { square, simple, linear };

    typedef struct generalized_dice_score_option {
        int64_t _num_classes = 0; // required (> 0)
        bool _include_background = true;
        bool _per_class = false;
        weight_type_mode _weight_type = weight_type_mode::square; // class weight 1 / support^2, 1 / support or 1
        input_format_mode _input_format = input_format_mode::one_hot;

        auto num_classes(int64_t v) -> generalized_dice_score_option & {
            _num_classes = v;
            return *this;
        }
        auto include_background(bool v) -> generalized_dice_score_option & {
            _include_background = v;
            return *this;
        }
        auto per_class(bool v) -> generalized_dice_score_option & {
            _per_class = v;
            return *this;
        }
        auto weight_type(weight_type_mode v) -> generalized_dice_score_option & {
            _weight_type = v;
            return *this;
        }
        auto input_format(input_format_mode v) -> generalized_dice_score_option & {
            _input_format = v;
            return *this;
        }
    } generalized_dice_score_option_t;

    namespace detail {
        // _generalized_dice_update + _generalized_dice_compute: per sample [N] (or [N, C] per class). A class without
        // target pixels (infinite weight) takes the largest finite weight of the batch at its flat position read in
        // the transposed order, torchmetrics' indexing (the same class for a single sample).
        inline auto generalized_dice(const tensor_t &preds_in, const tensor_t &target_in,
                                     const generalized_dice_score_option &opt, std::string &error) -> tensor_t {
            error = opt._num_classes > 0 ? std::string{}
                                         : std::format("Expected argument `num_classes` to be a positive integer, but "
                                                       "got {}.",
                                                       opt._num_classes);
            if (!error.empty()) {
                return {};
            }
            const auto [p, t] = inputs_format("generalized_dice_score", preds_in, target_in, opt._include_background,
                                              opt._num_classes, opt._input_format, error);
            if (!error.empty()) {
                return {};
            }
            const auto dims = spatial_dims(t);
            const auto intersection = torch::sum(p * t, dims);
            const auto target_sum = torch::sum(t, dims);
            const auto cardinality = target_sum + torch::sum(p, dims);
            auto weights = opt._weight_type == weight_type_mode::simple ? target_sum.reciprocal() * 1.0
                           : opt._weight_type == weight_type_mode::linear
                                   ? torch::ones_like(target_sum)
                                   : torch::pow(target_sum, 2).reciprocal() * 1.0;
            auto flat = weights.flatten(); // a view: zeroing the infinities also changes `weights`
            const auto infs = torch::isinf(flat);
            flat.masked_fill_(infs, 0);
            const auto w_max = std::get<0>(torch::max(weights, 0)).repeat({weights.size(0), 1}).t().flatten();
            flat.index_put_({infs}, w_max.index({infs}));
            weights = flat.reshape(weights.sizes());
            auto numerator = 2.0 * intersection * weights;
            auto denominator = cardinality * weights;
            if (!opt._per_class) {
                numerator = torch::sum(numerator, 1);
                denominator = torch::sum(denominator, 1);
            }
            return safe_divide(numerator, denominator, 0.0);
        }
    } // namespace detail

    // torchmetrics generalized_dice_score: per sample [N] (per_class: [N, C]).
    inline auto generalized_dice_score(const tensor_t &preds, const tensor_t &target,
                                       const generalized_dice_score_option &opt) -> tensor_t {
        std::string e;
        const auto v = detail::generalized_dice(preds, target, opt, e);
        return e.empty() ? v : detail::invalid(std::move(e));
    }

    // The mean over samples of generalized_dice_score.
    class GeneralizedDiceScore : public metric::detail::metric_base<GeneralizedDiceScore> {
    public:
        explicit GeneralizedDiceScore(generalized_dice_score_option opt) : _opt(opt) {
            if (_opt._num_classes <= 0) {
                const auto n = _opt._num_classes;
                _opt._num_classes = 1;
                detail::fail(std::format("Expected argument `num_classes` to be a positive integer, but got {}.", n));
            }
            const auto c = _opt._include_background ? _opt._num_classes : _opt._num_classes - 1;
            _score = torch::zeros({_opt._per_class ? c : 1});
        }

        auto update(const tensor_t &preds, const tensor_t &target) -> void {
            std::string e;
            const auto v = detail::generalized_dice(preds, target, _opt, e);
            if (!e.empty()) {
                return detail::fail(std::move(e));
            }
            _score = _score.to(v.device()) + v.sum(0);
            _samples += preds.size(0);
        }

        auto compute() const -> tensor_t { return detail::squeeze_if_scalar(_score / static_cast<double>(_samples)); }

        auto options() const -> const generalized_dice_score_option & { return _opt; }

    private:
        generalized_dice_score_option _opt;
        tensor_t _score;
        int64_t _samples = 0;
    };
} // namespace torchmedia::metric::segmentation
#endif // LIB_TORCH_MEDIA_METRIC_SEGMENTATION_HPP
