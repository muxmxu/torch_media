#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_DETECTION_HPP
#define LIB_TORCH_MEDIA_METRIC_DETECTION_HPP
// Detection metrics (develop_log/2026-10-23/progress11): torchvision 0.20.1 ops/boxes.py box ops (same libtorch calls,
// bit-exact), torchmetrics 1.9 detection IoU metrics, MeanAveragePrecision reproducing pycocotools COCOeval (evaluate /
// accumulate / summarize, float64, numpy's summation order) and VOC devkit average precision.
#include <cstring>
#include <numbers>
#include <numeric>
#include "_common.hpp"

namespace torchmedia::metric::detection {
    enum class box_format_mode { xyxy, xywh, cxcywh };

    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;
        using torch::indexing::None;
        using torch::indexing::Slice;

        // torchvision.ops._utils._upcast: float16 / bfloat16 -> float32, 8 / 16-bit integers -> int32.
        inline auto upcast(const tensor_t &t) -> tensor_t {
            const auto d = t.scalar_type();
            if (t.is_floating_point()) {
                return d == torch::kFloat32 || d == torch::kFloat64 ? t : t.to(torch::kFloat32);
            }
            return d == torch::kInt32 || d == torch::kInt64 ? t : t.to(torch::kInt32);
        }

        inline auto to_xyxy(const tensor_t &boxes, box_format_mode fmt) -> tensor_t {
            const auto c = boxes.unbind(-1);
            if (fmt == box_format_mode::xywh) {
                return torch::stack({c[0], c[1], c[0] + c[2], c[1] + c[3]}, -1);
            }
            return torch::stack({c[0] - 0.5 * c[2], c[1] - 0.5 * c[3], c[0] + 0.5 * c[2], c[1] + 0.5 * c[3]}, -1);
        }

        inline auto from_xyxy(const tensor_t &boxes, box_format_mode fmt) -> tensor_t {
            const auto c = boxes.unbind(-1);
            if (fmt == box_format_mode::xywh) {
                return torch::stack({c[0], c[1], c[2] - c[0], c[3] - c[1]}, -1);
            }
            return torch::stack({(c[0] + c[2]) / 2, (c[1] + c[3]) / 2, c[2] - c[0], c[3] - c[1]}, -1);
        }
    } // namespace detail

    // torchvision.ops.box_convert: [..., 4] boxes between (x1, y1, x2, y2), (x, y, w, h) and (cx, cy, w, h).
    inline auto box_convert(const tensor_t &boxes, box_format_mode in_fmt, box_format_mode out_fmt) -> tensor_t {
        if (in_fmt == out_fmt) {
            return boxes.clone();
        }
        if (boxes.dim() == 0 || boxes.size(-1) != 4) {
            return detail::invalid(
                    std::format("box_convert: expected boxes of shape [..., 4], got {}.", c10::str(boxes.sizes())));
        }
        const auto xyxy = in_fmt == box_format_mode::xyxy ? boxes : detail::to_xyxy(boxes, in_fmt);
        return out_fmt == box_format_mode::xyxy ? xyxy : detail::from_xyxy(xyxy, out_fmt);
    }

    // torchvision.ops.box_area of [N, 4] xyxy boxes (integer boxes stay integer, upcast to int32 / int64).
    inline auto box_area(const tensor_t &boxes) -> tensor_t {
        const auto b = detail::upcast(boxes);
        return (b.select(1, 2) - b.select(1, 0)) * (b.select(1, 3) - b.select(1, 1));
    }

    namespace detail {
        inline auto inter_union(const tensor_t &b1, const tensor_t &b2) -> std::pair<tensor_t, tensor_t> {
            const auto area1 = box_area(b1);
            const auto area2 = box_area(b2);
            const auto lt = torch::max(b1.index({Slice(), None, Slice(None, 2)}), b2.index({Slice(), Slice(None, 2)}));
            const auto rb = torch::min(b1.index({Slice(), None, Slice(2, None)}), b2.index({Slice(), Slice(2, None)}));
            const auto wh = upcast(rb - lt).clamp_min(0);
            const auto inter = wh.select(2, 0) * wh.select(2, 1);
            return {inter, area1.unsqueeze(1) + area2 - inter};
        }

        // The enclosing box's [N, M, 2] width / height.
        inline auto enclosing_wh(const tensor_t &b1, const tensor_t &b2) -> tensor_t {
            const auto lti = torch::min(b1.index({Slice(), None, Slice(None, 2)}), b2.index({Slice(), Slice(None, 2)}));
            const auto rbi = torch::max(b1.index({Slice(), None, Slice(2, None)}), b2.index({Slice(), Slice(2, None)}));
            return upcast(rbi - lti).clamp_min(0);
        }
    } // namespace detail

    // torchvision.ops.box_iou: [N, M] IoU of xyxy boxes.
    inline auto box_iou(const tensor_t &boxes1, const tensor_t &boxes2) -> tensor_t {
        const auto [inter, uni] = detail::inter_union(boxes1, boxes2);
        return inter / uni;
    }

    // torchvision.ops.generalized_box_iou
    inline auto generalized_box_iou(const tensor_t &boxes1, const tensor_t &boxes2) -> tensor_t {
        const auto [inter, uni] = detail::inter_union(boxes1, boxes2);
        const auto iou = inter / uni;
        const auto whi = detail::enclosing_wh(boxes1, boxes2);
        const auto areai = whi.select(2, 0) * whi.select(2, 1);
        return iou - (areai - uni) / areai;
    }

    namespace detail {
        // torchvision _box_diou_iou on upcast boxes: {diou, iou}.
        inline auto diou_iou(const tensor_t &b1, const tensor_t &b2, double eps) -> std::pair<tensor_t, tensor_t> {
            const auto iou = box_iou(b1, b2);
            const auto whi = enclosing_wh(b1, b2);
            const auto diagonal_distance_squared =
                    torch::pow(whi.select(2, 0), 2) + torch::pow(whi.select(2, 1), 2) + eps;
            const auto x_p = (b1.select(1, 0) + b1.select(1, 2)) / 2;
            const auto y_p = (b1.select(1, 1) + b1.select(1, 3)) / 2;
            const auto x_g = (b2.select(1, 0) + b2.select(1, 2)) / 2;
            const auto y_g = (b2.select(1, 1) + b2.select(1, 3)) / 2;
            const auto centers_distance_squared = torch::pow(upcast(x_p.unsqueeze(1) - x_g.unsqueeze(0)), 2) +
                                                  torch::pow(upcast(y_p.unsqueeze(1) - y_g.unsqueeze(0)), 2);
            return {iou - (centers_distance_squared / diagonal_distance_squared), iou};
        }
    } // namespace detail

    // torchvision.ops.distance_box_iou
    inline auto distance_box_iou(const tensor_t &boxes1, const tensor_t &boxes2, double eps = 1e-7) -> tensor_t {
        return detail::diou_iou(detail::upcast(boxes1), detail::upcast(boxes2), eps).first;
    }

    // torchvision.ops.complete_box_iou
    inline auto complete_box_iou(const tensor_t &boxes1, const tensor_t &boxes2, double eps = 1e-7) -> tensor_t {
        const auto b1 = detail::upcast(boxes1);
        const auto b2 = detail::upcast(boxes2);
        const auto [diou, iou] = detail::diou_iou(b1, b2, eps);
        const auto w_pred = (b1.select(1, 2) - b1.select(1, 0)).unsqueeze(1);
        const auto h_pred = (b1.select(1, 3) - b1.select(1, 1)).unsqueeze(1);
        const auto w_gt = b2.select(1, 2) - b2.select(1, 0);
        const auto h_gt = b2.select(1, 3) - b2.select(1, 1);
        // Python's 4 / (torch.pi ** 2): pow through libm, like CPython.
        const auto v = (4 / std::pow(std::numbers::pi, 2.0)) *
                       torch::pow(torch::atan(w_pred / h_pred) - torch::atan(w_gt / h_gt), 2);
        const auto alpha = v / (1 - iou + v + eps);
        return diou - alpha * v;
    }

    // ---------------- torchmetrics.functional.detection IoU variants ----------------
    namespace detail {
        enum class iou_kind { iou, giou, diou, ciou };

        inline auto pairwise_iou(iou_kind k, const tensor_t &a, const tensor_t &b) -> tensor_t {
            switch (k) {
                case iou_kind::iou:
                    return box_iou(a, b);
                case iou_kind::giou:
                    return generalized_box_iou(a, b);
                case iou_kind::diou:
                    return distance_box_iou(a, b);
                default:
                    return complete_box_iou(a, b);
            }
        }

        inline auto boxes_error(const tensor_t &preds, const tensor_t &target) -> std::string {
            for (const auto *t: {&preds, &target}) {
                if (t->dim() != 2 || t->size(-1) != 4) {
                    return std::format("Expected {} to be of shape (N, 4) but got {}", t == &preds ? "preds" : "target",
                                       c10::str(t->sizes()));
                }
            }
            return {};
        }

        // torchmetrics _iou_update (shapes already checked): [N, M] pairwise values, those below the threshold
        // replaced; an empty side gives torchmetrics' float32 zero matrix (square, of the other side's size).
        inline auto iou_update(iou_kind k, const tensor_t &preds, const tensor_t &target,
                               std::optional<double> iou_threshold, double replacement_val) -> tensor_t {
            if (preds.numel() == 0) {
                return torch::zeros({target.size(0), target.size(0)}, target.options().dtype(torch::kFloat32));
            }
            if (target.numel() == 0) {
                return torch::zeros({preds.size(0), preds.size(0)}, preds.options().dtype(torch::kFloat32));
            }
            auto iou = pairwise_iou(k, preds, target);
            if (iou_threshold) {
                iou.masked_fill_(iou < *iou_threshold, replacement_val);
            }
            return iou;
        }

        inline auto iou_functional(iou_kind k, const tensor_t &preds, const tensor_t &target,
                                   std::optional<double> iou_threshold, double replacement_val, bool aggregate)
                -> tensor_t {
            if (auto e = boxes_error(preds, target); !e.empty()) {
                return invalid(std::move(e));
            }
            const auto iou = iou_update(k, preds, target, iou_threshold, replacement_val);
            if (!aggregate) {
                return iou;
            }
            return iou.numel() > 0 ? iou.diag().mean() : torch::tensor(0.0, iou.options().dtype(torch::kFloat32));
        }
    } // namespace detail

    // torchmetrics intersection_over_union: preds [N, 4] / target [M, 4] xyxy boxes; aggregate -> the mean of the
    // diagonal (box i against box i), else the [N, M] matrix. Values below iou_threshold become replacement_val.
    inline auto intersection_over_union(const tensor_t &preds, const tensor_t &target,
                                        std::optional<double> iou_threshold = {}, double replacement_val = 0,
                                        bool aggregate = true) -> tensor_t {
        return detail::iou_functional(detail::iou_kind::iou, preds, target, iou_threshold, replacement_val, aggregate);
    }

    inline auto generalized_intersection_over_union(const tensor_t &preds, const tensor_t &target,
                                                    std::optional<double> iou_threshold = {},
                                                    double replacement_val = 0, bool aggregate = true) -> tensor_t {
        return detail::iou_functional(detail::iou_kind::giou, preds, target, iou_threshold, replacement_val, aggregate);
    }

    inline auto distance_intersection_over_union(const tensor_t &preds, const tensor_t &target,
                                                 std::optional<double> iou_threshold = {}, double replacement_val = 0,
                                                 bool aggregate = true) -> tensor_t {
        return detail::iou_functional(detail::iou_kind::diou, preds, target, iou_threshold, replacement_val, aggregate);
    }

    inline auto complete_intersection_over_union(const tensor_t &preds, const tensor_t &target,
                                                 std::optional<double> iou_threshold = {}, double replacement_val = 0,
                                                 bool aggregate = true) -> tensor_t {
        return detail::iou_functional(detail::iou_kind::ciou, preds, target, iou_threshold, replacement_val, aggregate);
    }

    // ---------------- per-image inputs of the stateful detection metrics (torchmetrics' dicts) ----------------
    // An undefined tensor is a missing key. boxes [N, 4] in the metric's box_format (or an empty tensor), scores [N]
    // floating, labels [N] integer, masks [N, H, W] (iou_type segm; nonzero = inside).
    struct detection_prediction {
        tensor_t boxes;
        tensor_t scores;
        tensor_t labels;
        tensor_t masks;
    };

    // iscrowd [M] (nonzero = crowd; VOC: difficult) and area [M] (> 0 replaces the box / mask area) are optional.
    struct detection_target {
        tensor_t boxes;
        tensor_t labels;
        tensor_t iscrowd;
        tensor_t area;
        tensor_t masks;
    };

    namespace detail {
        // torchmetrics _input_validator (+ _get_coco_format's label / score / box checks) for boxes, or masks.
        inline auto input_error(const std::vector<detection_prediction> &preds,
                                const std::vector<detection_target> &target, bool masks, bool check_scores)
                -> std::string {
            const char *key = masks ? "masks" : "boxes";
            if (preds.size() != target.size()) {
                return std::format("Expected argument `preds` and `target` to have the same length, but got {} and {}",
                                   preds.size(), target.size());
            }
            for (size_t i = 0; i < preds.size(); ++i) {
                const auto &p = preds[i];
                const auto &t = target[i];
                const auto &pv = masks ? p.masks : p.boxes;
                const auto &tv = masks ? t.masks : t.boxes;
                if (!pv.defined() || !p.labels.defined() || (check_scores && !p.scores.defined())) {
                    return std::format("Expected all predictions to define `{}`, `labels`{} (sample {})", key,
                                       check_scores ? " and `scores`" : "", i);
                }
                if (!tv.defined() || !t.labels.defined()) {
                    return std::format("Expected all targets to define `{}` and `labels` (sample {})", key, i);
                }
                if (p.labels.dim() != 1 || t.labels.dim() != 1 || (check_scores && p.scores.dim() != 1)) {
                    return std::format("Expected 1-D labels{} (sample {})", check_scores ? " and scores" : "", i);
                }
                if (tv.dim() == 0 || tv.size(0) != t.labels.size(0) ||
                    (t.iscrowd.defined() && t.iscrowd.numel() != t.labels.size(0)) ||
                    (t.area.defined() && t.area.numel() != t.labels.size(0))) {
                    return std::format("Input '{}', labels, iscrowd and area of sample {} in targets have a different "
                                       "length",
                                       key, i);
                }
                if (pv.dim() == 0 || pv.size(0) != p.labels.size(0) ||
                    (check_scores && p.scores.size(0) != p.labels.size(0))) {
                    return std::format("Input '{}', labels and scores of sample {} in predictions have a different "
                                       "length",
                                       key, i);
                }
                for (const auto *v: {&pv, &tv}) {
                    if (v->numel() > 0 && (masks ? v->dim() != 3 : v->dim() != 2 || v->size(1) != 4)) {
                        return std::format("Invalid input {} of sample {} (expected [N, {}], got {})", key, i,
                                           masks ? "H, W" : "4", c10::str(v->sizes()));
                    }
                }
                if (masks && p.masks.numel() > 0 && t.masks.numel() > 0 &&
                    p.masks.sizes().slice(1) != t.masks.sizes().slice(1)) {
                    return std::format("Prediction and target masks of sample {} differ in size", i);
                }
                if (!at::isIntegralType(p.labels.scalar_type(), true) ||
                    !at::isIntegralType(t.labels.scalar_type(), true)) {
                    return std::format("Invalid input class of sample {} (expected integer labels)", i);
                }
                if (check_scores && !p.scores.is_floating_point()) {
                    return std::format("Invalid input score of sample {} (expected floating scores)", i);
                }
            }
            return {};
        }

        // Boxes [N, 4] (or empty) -> box_format conversion -> float64 rows on the CPU.
        inline auto box_rows(const tensor_t &boxes, box_format_mode in_fmt, box_format_mode out_fmt)
                -> std::vector<std::array<double, 4>> {
            if (boxes.numel() == 0) {
                return {};
            }
            const auto b = box_convert(boxes, in_fmt, out_fmt).to(torch::kCPU).to(torch::kFloat64).contiguous();
            std::vector<std::array<double, 4>> rows(b.size(0));
            std::memcpy(rows.data(), b.data_ptr<double>(), rows.size() * sizeof(rows[0]));
            return rows;
        }

        template<class T>
        inline auto values(const tensor_t &t, torch::Dtype dtype) -> std::vector<T> {
            const auto c = t.reshape({-1}).to(torch::kCPU).to(dtype).contiguous(); // CPU first: MPS has no float64
            return {c.data_ptr<T>(), c.data_ptr<T>() + c.numel()};
        }
    } // namespace detail

    // ---------------- IntersectionOverUnion and its variants (torchmetrics.detection) ----------------
    typedef struct intersection_over_union_option {
        box_format_mode _box_format = box_format_mode::xyxy;
        std::optional<double> _iou_threshold; // values below it are dropped (torchmetrics' -1 marker)
        bool _class_metrics = false;
        bool _respect_labels = true; // only same-label pairs count

        auto box_format(box_format_mode v) -> intersection_over_union_option & {
            _box_format = v;
            return *this;
        }
        auto iou_threshold(std::optional<double> v) -> intersection_over_union_option & {
            _iou_threshold = v;
            return *this;
        }
        auto class_metrics(bool v) -> intersection_over_union_option & {
            _class_metrics = v;
            return *this;
        }
        auto respect_labels(bool v) -> intersection_over_union_option & {
            _respect_labels = v;
            return *this;
        }
    } intersection_over_union_option_t;

    namespace detail {
        // torchmetrics IntersectionOverUnion (and G / D / C subclasses): per image the pairwise matrix of predicted
        // against target boxes, pairs below the threshold or (respect_labels) with different labels marked -1.
        // compute() returns torchmetrics' dict: {"iou": mean of all unmarked values (0 without any)} plus, with
        // class_metrics, "iou/cl_<label>" over the target columns of each label. An image without boxes on one side
        // gives torchmetrics' square zero matrix (its diagonal counts as zeros); such an image is skipped in the
        // per-class sums, where torchmetrics raises an IndexError (develop_log/2026-10-23/progress11 D7).
        // The kind-independent state and logic (one instance, not one per kind).
        class iou_metric_base {
        public:
            iou_metric_base(iou_kind kind, intersection_over_union_option opt) : _kind(kind), _opt(opt) {}

            auto update(const std::vector<detection_prediction> &preds, const std::vector<detection_target> &target)
                    -> void {
                if (auto e = input_error(preds, target, false, false); !e.empty()) {
                    return fail(std::move(e));
                }
                std::vector<std::pair<tensor_t, tensor_t>> boxes;
                for (size_t i = 0; i < preds.size(); ++i) {
                    boxes.emplace_back(safe_boxes(preds[i].boxes), safe_boxes(target[i].boxes));
                    if (auto e = boxes_error(boxes.back().first, boxes.back().second); !e.empty()) {
                        return fail(std::move(e));
                    }
                }
                for (size_t i = 0; i < preds.size(); ++i) {
                    const auto &[det, gt] = boxes[i];
                    auto iou = iou_update(_kind, det, gt, _opt._iou_threshold, -1.0);
                    if (_opt._respect_labels) {
                        const auto label_eq = det.numel() > 0 && gt.numel() > 0
                                                      ? preds[i].labels.unsqueeze(1) == target[i].labels.unsqueeze(0)
                                                      : torch::eye(iou.size(0), iou.options().dtype(torch::kBool));
                        iou.masked_fill_(label_eq.logical_not(), -1.0);
                    }
                    _gt_labels.push_back(target[i].labels.detach().clone()); // own the data (reused buffers)
                    _pred_labels.push_back(preds[i].labels.detach().clone());
                    _iou_matrix.push_back(iou);
                }
            }

            auto compute() const -> std::map<std::string, tensor_t> {
                const std::string name = std::array{"iou", "giou", "diou", "ciou"}[static_cast<size_t>(_kind)];
                std::vector<tensor_t> valid;
                for (const auto &m: _iou_matrix) {
                    if ((m != -1.0).any().item<bool>()) {
                        valid.push_back(m.index({m != -1.0}));
                    }
                }
                auto score = valid.empty() ? torch::tensor(0.0, torch::kFloat32) : torch::cat(valid, 0).mean();
                std::map<std::string, tensor_t> results{{name, score}};
                if (torch::isnan(score).item<bool>()) {
                    results[name] = torch::tensor(0.0, score.options());
                }
                if (_opt._class_metrics && !_gt_labels.empty()) {
                    const auto classes = std::get<0>(
                            at::_unique(torch::cat({torch::cat(_gt_labels), torch::cat(_pred_labels)}).cpu()));
                    for (const auto cl: values<int64_t>(classes, torch::kInt64)) {
                        auto masked_iou = torch::zeros_like(score);
                        auto observed = torch::zeros_like(score);
                        for (size_t i = 0; i < _iou_matrix.size(); ++i) {
                            if (_iou_matrix[i].size(1) != _gt_labels[i].size(0)) {
                                continue;
                            }
                            const auto scores = _iou_matrix[i].index({Slice(), _gt_labels[i] == cl});
                            const auto valid_scores = scores.index({scores != -1.0});
                            masked_iou += valid_scores.sum();
                            observed += valid_scores.numel();
                        }
                        results[std::format("{}/cl_{}", name, cl)] = observed.item<double>() == 0
                                                                             ? torch::tensor(0.0, score.options())
                                                                             : masked_iou / observed;
                    }
                }
                return results;
            }

            auto options() const -> const intersection_over_union_option & { return _opt; }

        private:
            auto safe_boxes(const tensor_t &boxes) const -> tensor_t {
                // torchmetrics _fix_empty_tensors: a 1-D empty tensor becomes [1, 0] (and is then rejected).
                const auto b = boxes.numel() == 0 && boxes.dim() == 1 ? boxes.unsqueeze(0) : boxes;
                return b.numel() > 0 ? box_convert(b, _opt._box_format, box_format_mode::xyxy) : b;
            }

            iou_kind _kind;
            intersection_over_union_option _opt;
            std::vector<tensor_t> _gt_labels, _pred_labels, _iou_matrix;
        };

        template<iou_kind K>
        class iou_metric : public iou_metric_base, public metric::detail::metric_base<iou_metric<K>> {
        public:
            explicit iou_metric(intersection_over_union_option opt = {}) : iou_metric_base(K, opt) {}
        };
    } // namespace detail

    using IntersectionOverUnion = detail::iou_metric<detail::iou_kind::iou>;
    using GeneralizedIntersectionOverUnion = detail::iou_metric<detail::iou_kind::giou>;
    using DistanceIntersectionOverUnion = detail::iou_metric<detail::iou_kind::diou>;
    using CompleteIntersectionOverUnion = detail::iou_metric<detail::iou_kind::ciou>;

    // ---------------- MeanAveragePrecision (torchmetrics.detection, pycocotools backend) ----------------
    enum class iou_type_mode { bbox, segm };
    enum class average_mode { macro, micro };

    typedef struct mean_average_precision_option {
        box_format_mode _box_format = box_format_mode::xyxy;
        iou_type_mode _iou_type = iou_type_mode::bbox;
        std::vector<double> _iou_thresholds; // {} = torchmetrics' torch.linspace(0.5, 0.95, 10) (float32 values)
        std::vector<double> _rec_thresholds; // {} = torch.linspace(0, 1, 101) (float32 values)
        std::vector<int64_t> _max_detection_thresholds; // {} = {1, 10, 100}; else 3 values >= 0 (sorted)
        bool _class_metrics = false;
        average_mode _average = average_mode::macro;

        auto box_format(box_format_mode v) -> mean_average_precision_option & {
            _box_format = v;
            return *this;
        }
        auto iou_type(iou_type_mode v) -> mean_average_precision_option & {
            _iou_type = v;
            return *this;
        }
        auto iou_thresholds(std::vector<double> v) -> mean_average_precision_option & {
            _iou_thresholds = std::move(v);
            return *this;
        }
        auto rec_thresholds(std::vector<double> v) -> mean_average_precision_option & {
            _rec_thresholds = std::move(v);
            return *this;
        }
        auto max_detection_thresholds(std::vector<int64_t> v) -> mean_average_precision_option & {
            _max_detection_thresholds = std::move(v);
            return *this;
        }
        auto class_metrics(bool v) -> mean_average_precision_option & {
            _class_metrics = v;
            return *this;
        }
        auto average(average_mode v) -> mean_average_precision_option & {
            _average = v;
            return *this;
        }
    } mean_average_precision_option_t;

    // torchmetrics' result keys as pycocotools float64 values (torchmetrics rounds them to float32): -1 where COCOeval
    // has nothing to average. mar_1 / mar_10 / mar_100 are the recalls at max_detection_thresholds[0 / 1 / 2] and
    // mar_100_per_class at [2]; map (and map_per_class) uses COCOeval's literal maxDets = 100, so it is -1 when 100 is
    // not a threshold. map_per_class / mar_100_per_class: [num classes] with class_metrics, else {-1}; classes: the
    // sorted labels of predictions and targets (int32).
    struct mean_average_precision_result {
        double map = -1, map_50 = -1, map_75 = -1, map_small = -1, map_medium = -1, map_large = -1;
        double mar_1 = -1, mar_10 = -1, mar_100 = -1, mar_small = -1, mar_medium = -1, mar_large = -1;
        tensor_t map_per_class, mar_100_per_class, classes;
    };

    namespace detail {
        // One image as torchmetrics hands it to COCO: xywh boxes (bbox) or, for segm, every detection's mask IoU with
        // every target (computed in update(), so the masks are not kept), the COCO areas.
        struct coco_image {
            std::vector<std::array<double, 4>> dt_box, gt_box;
            std::vector<double> mask_iou; // segm: [D x G] row-major, input order
            std::vector<double> dt_score, dt_area, gt_area;
            std::vector<int64_t> dt_label, gt_label;
            std::vector<uint8_t> gt_crowd;
            bool in_gt = true, in_dt = true; // segm: an image without masks is left out of that COCO dataset
        };

        struct coco_params {
            std::vector<double> iou_thrs, rec_thrs;
            std::vector<int64_t> max_dets; // sorted
        };

        // COCOeval.evaluateImg's output for one (category, area range, image).
        struct coco_eval_img {
            bool valid = false;
            std::vector<double> scores; // D, score-sorted
            std::vector<uint8_t> matched, ignored; // T x D
            int64_t num_not_ignored_gt = 0;
        };

        inline constexpr std::array<std::array<double, 2>, 4> coco_area_rng{
                {{0.0, 1e10}, {0.0, 32.0 * 32.0}, {32.0 * 32.0, 96.0 * 96.0}, {96.0 * 96.0, 1e10}}};

        // Stable order of scores, highest first (np.argsort(-scores, kind='mergesort'); NaN last).
        inline auto score_order(const std::vector<double> &scores, const std::vector<int64_t> &idx)
                -> std::vector<int64_t> {
            auto order = idx;
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                const double x = -scores[a], y = -scores[b];
                return x < y || (!std::isnan(x) && std::isnan(y));
            });
            return order;
        }

        // pycocotools maskApi bbIou on xywh boxes: [D x G] row-major.
        inline auto bb_iou(const std::vector<std::array<double, 4>> &dt, const std::vector<int64_t> &dts,
                           const std::vector<std::array<double, 4>> &gt, const std::vector<int64_t> &gts,
                           const std::vector<uint8_t> &crowd) -> std::vector<double> {
            TORCHMEDIA_FP_CONTRACT_OFF
            std::vector<double> o(dts.size() * gts.size(), 0.0);
            for (size_t g = 0; g < gts.size(); ++g) {
                const auto &G = gt[gts[g]];
                const double ga = G[2] * G[3];
                for (size_t d = 0; d < dts.size(); ++d) {
                    const auto &D = dt[dts[d]];
                    const double da = D[2] * D[3];
                    const double w = std::fmin(D[2] + D[0], G[2] + G[0]) - std::fmax(D[0], G[0]);
                    if (w <= 0) {
                        continue;
                    }
                    const double h = std::fmin(D[3] + D[1], G[3] + G[1]) - std::fmax(D[1], G[1]);
                    if (h <= 0) {
                        continue;
                    }
                    const double i = w * h;
                    const double u = crowd[gts[g]] ? da : da + ga - i;
                    o[d * gts.size() + g] = i / u;
                }
            }
            return o;
        }

        // pycocotools rleIou on dense [n, H * W] bool masks, every detection against every target ([D x G] row-major):
        // intersection / union pixel counts (crowd: / the detection's area). The counts are exact in float64; the
        // detections go in blocks of 16 rows to bound the float64 copy.
        inline auto mask_iou(const tensor_t &dm, const tensor_t &gm, const std::vector<uint8_t> &crowd)
                -> std::vector<double> {
            const auto D = dm.size(0), G = gm.size(0);
            if (D == 0 || G == 0) {
                return {};
            }
            const auto gf = gm.to(torch::kFloat64);
            const auto ga = values<double>(gf.sum(1), torch::kFloat64);
            std::vector<double> o(D * G, 0.0);
            for (int64_t lo = 0; lo < D; lo += 16) {
                const auto df = dm.slice(0, lo, lo + 16).to(torch::kFloat64);
                const auto inter = values<double>(torch::mm(df, gf.t()), torch::kFloat64);
                const auto da = values<double>(df.sum(1), torch::kFloat64);
                for (size_t d = 0; d < da.size(); ++d) {
                    for (int64_t g = 0; g < G; ++g) {
                        const double i = inter[d * G + g];
                        o[(lo + d) * G + g] = i == 0 ? 0.0 : i / (crowd[g] ? da[d] : da[d] + ga[g] - i);
                    }
                }
            }
            return o;
        }

        // segm: the [dts x gts] block of the image's mask IoUs.
        inline auto mask_iou_block(const coco_image &im, const std::vector<int64_t> &dts,
                                   const std::vector<int64_t> &gts) -> std::vector<double> {
            std::vector<double> o;
            for (const auto d: dts) {
                for (const auto g: gts) {
                    o.push_back(im.mask_iou[d * static_cast<int64_t>(im.gt_crowd.size()) + g]);
                }
            }
            return o;
        }

        // COCOeval.evaluateImg for one image / category / area range; ious [D x G] (dt score order, gt input order).
        inline auto evaluate_img(const coco_image &im, const std::vector<int64_t> &dts, const std::vector<int64_t> &gts,
                                 const std::vector<double> &ious, const std::array<double, 2> &rng,
                                 const coco_params &p) -> coco_eval_img {
            coco_eval_img e;
            if (dts.empty() && gts.empty()) {
                return e;
            }
            e.valid = true;
            const auto G = gts.size(), D = dts.size(), T = p.iou_thrs.size();
            std::vector<uint8_t> g_ig(G);
            for (size_t g = 0; g < G; ++g) {
                const double a = im.gt_area[gts[g]];
                g_ig[g] = im.gt_crowd[gts[g]] || a < rng[0] || a > rng[1];
            }
            std::vector<size_t> gtind(G);
            std::iota(gtind.begin(), gtind.end(), 0);
            std::stable_sort(gtind.begin(), gtind.end(), [&](size_t a, size_t b) { return g_ig[a] < g_ig[b]; });
            std::vector<uint8_t> gt_ig(G), crowd(G);
            for (size_t j = 0; j < G; ++j) {
                gt_ig[j] = g_ig[gtind[j]];
                crowd[j] = im.gt_crowd[gts[gtind[j]]];
                e.num_not_ignored_gt += gt_ig[j] == 0;
            }
            e.matched.assign(T * D, 0);
            e.ignored.assign(T * D, 0);
            std::vector<uint8_t> gtm(G);
            for (size_t t = 0; t < T; ++t) {
                std::fill(gtm.begin(), gtm.end(), 0);
                for (size_t d = 0; d < D; ++d) {
                    double iou = std::min(p.iou_thrs[t], 1 - 1e-10);
                    int64_t m = -1;
                    for (size_t j = 0; j < G; ++j) {
                        if (gtm[j] && !crowd[j]) {
                            continue;
                        }
                        if (m > -1 && gt_ig[m] == 0 && gt_ig[j] == 1) {
                            break;
                        }
                        const double v = ious[d * G + gtind[j]];
                        if (v < iou) {
                            continue;
                        }
                        iou = v;
                        m = static_cast<int64_t>(j);
                    }
                    if (m == -1) {
                        continue;
                    }
                    e.ignored[t * D + d] = gt_ig[m];
                    e.matched[t * D + d] = 1;
                    gtm[m] = 1;
                }
            }
            for (size_t d = 0; d < D; ++d) {
                const double a = im.dt_area[dts[d]];
                e.scores.push_back(im.dt_score[dts[d]]);
                for (size_t t = 0; t < T; ++t) {
                    e.ignored[t * D + d] |= !e.matched[t * D + d] && (a < rng[0] || a > rng[1]);
                }
            }
            return e;
        }

        // COCOeval evaluate + accumulate + summarize over the images in the ground-truth dataset.
        class coco_eval {
        public:
            coco_eval(const std::vector<coco_image> &images, const std::vector<int64_t> &cats, bool micro, bool segm,
                      const coco_params &p) : _p(p), _K(cats.size()) {
                const auto T = p.iou_thrs.size(), R = p.rec_thrs.size(), A = coco_area_rng.size(),
                           M = p.max_dets.size();
                _precision.assign(T * R * _K * A * M, -1.0);
                _recall.assign(T * _K * A * M, -1.0);
                std::vector<const coco_image *> imgs;
                for (const auto &im: images) {
                    if (im.in_gt) {
                        imgs.push_back(&im);
                    }
                }
                const auto I = imgs.size();
                std::vector<coco_eval_img> evals(_K * A * I);
                for (size_t k = 0; k < _K; ++k) {
                    for (size_t i = 0; i < I; ++i) {
                        const auto &im = *imgs[i];
                        std::vector<int64_t> gts, dts;
                        for (size_t g = 0; g < im.gt_label.size(); ++g) {
                            if (micro || im.gt_label[g] == cats[k]) {
                                gts.push_back(static_cast<int64_t>(g));
                            }
                        }
                        for (size_t d = 0; d < im.dt_label.size(); ++d) {
                            if (micro || im.dt_label[d] == cats[k]) {
                                dts.push_back(static_cast<int64_t>(d));
                            }
                        }
                        dts = score_order(im.dt_score, dts);
                        dts.resize(std::min<size_t>(dts.size(), p.max_dets.back()));
                        const auto ious = dts.empty() || gts.empty() ? std::vector<double>{}
                                          : segm ? mask_iou_block(im, dts, gts)
                                                 : bb_iou(im.dt_box, dts, im.gt_box, gts, im.gt_crowd);
                        for (size_t a = 0; a < A; ++a) {
                            evals[(k * A + a) * I + i] = evaluate_img(im, dts, gts, ious, coco_area_rng[a], p);
                        }
                    }
                }
                accumulate(evals, I);
            }

            // COCOeval.summarize's _summarize over category k (or all): the mean of the entries > -1, else -1.
            auto summarize(bool ap, std::optional<double> iou_thr, size_t a, int64_t max_det,
                           std::optional<size_t> only_k = {}) const -> double {
                const auto T = _p.iou_thrs.size(), R = _p.rec_thrs.size(), A = coco_area_rng.size(),
                           M = _p.max_dets.size();
                std::vector<double> s;
                for (size_t t = 0; t < T; ++t) {
                    if (iou_thr && _p.iou_thrs[t] != *iou_thr) {
                        continue;
                    }
                    for (size_t r = 0; r < (ap ? R : 1); ++r) {
                        for (size_t k = 0; k < _K; ++k) {
                            if (only_k && k != *only_k) {
                                continue;
                            }
                            for (size_t m = 0; m < M; ++m) {
                                if (_p.max_dets[m] != max_det) {
                                    continue;
                                }
                                const double v = ap ? _precision[(((t * R + r) * _K + k) * A + a) * M + m]
                                                    : _recall[((t * _K + k) * A + a) * M + m];
                                if (v > -1) {
                                    s.push_back(v);
                                }
                            }
                        }
                    }
                }
                return s.empty() ? -1.0 : metric::detail::np_sum(s) / static_cast<double>(s.size());
            }

        private:
            auto accumulate(const std::vector<coco_eval_img> &evals, size_t I) -> void {
                const auto T = _p.iou_thrs.size(), R = _p.rec_thrs.size(), A = coco_area_rng.size(),
                           M = _p.max_dets.size();
                const double eps = std::numeric_limits<double>::epsilon(); // np.spacing(1)
                for (size_t k = 0; k < _K; ++k) {
                    for (size_t a = 0; a < A; ++a) {
                        for (size_t m = 0; m < M; ++m) {
                            const auto max_det = static_cast<size_t>(_p.max_dets[m]);
                            std::vector<const coco_eval_img *> E;
                            for (size_t i = 0; i < I; ++i) {
                                if (evals[(k * A + a) * I + i].valid) {
                                    E.push_back(&evals[(k * A + a) * I + i]);
                                }
                            }
                            // concatenated detections: (eval, index), then the stable score order
                            std::vector<double> scores;
                            std::vector<std::pair<const coco_eval_img *, size_t>> src;
                            int64_t npig = 0;
                            for (const auto *e: E) {
                                for (size_t d = 0; d < std::min(max_det, e->scores.size()); ++d) {
                                    scores.push_back(e->scores[d]);
                                    src.emplace_back(e, d);
                                }
                                npig += e->num_not_ignored_gt;
                            }
                            if (npig == 0) {
                                continue;
                            }
                            std::vector<int64_t> all(scores.size());
                            std::iota(all.begin(), all.end(), 0);
                            const auto inds = score_order(scores, all);
                            const auto nd = inds.size();
                            for (size_t t = 0; t < T; ++t) {
                                std::vector<double> rc(nd), pr(nd);
                                double tp = 0, fp = 0;
                                for (size_t j = 0; j < nd; ++j) {
                                    const auto [e, d] = src[inds[j]];
                                    const auto D = e->scores.size();
                                    const bool ig = e->ignored[t * D + d], mt = e->matched[t * D + d];
                                    tp += mt && !ig;
                                    fp += !mt && !ig;
                                    rc[j] = tp / static_cast<double>(npig);
                                    pr[j] = tp / (fp + tp + eps);
                                }
                                _recall[((t * _K + k) * A + a) * M + m] = nd ? rc.back() : 0.0;
                                for (size_t j = nd - (nd > 0); j > 0; --j) {
                                    if (pr[j] > pr[j - 1]) {
                                        pr[j - 1] = pr[j];
                                    }
                                }
                                for (size_t r = 0; r < R; ++r) {
                                    const auto pi = static_cast<size_t>(
                                            std::lower_bound(rc.begin(), rc.end(), _p.rec_thrs[r]) - rc.begin());
                                    // q stays 0 from the first threshold past the last recall (Python's IndexError)
                                    const double q = pi < nd ? pr[pi] : 0.0;
                                    _precision[(((t * R + r) * _K + k) * A + a) * M + m] = q;
                                    if (pi >= nd) {
                                        for (size_t r2 = r + 1; r2 < R; ++r2) {
                                            _precision[(((t * R + r2) * _K + k) * A + a) * M + m] = 0.0;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            coco_params _p;
            size_t _K;
            std::vector<double> _precision, _recall; // [T, R, K, A, M], [T, K, A, M]
        };

        // COCOeval's 12 summary stats into the result.
        inline auto coco_stats(const coco_eval &ev, const coco_params &p, mean_average_precision_result &res) -> void {
            const auto md = p.max_dets;
            res.map = ev.summarize(true, {}, 0, 100);
            res.map_50 = ev.summarize(true, 0.5, 0, md[2]);
            res.map_75 = ev.summarize(true, 0.75, 0, md[2]);
            res.map_small = ev.summarize(true, {}, 1, md[2]);
            res.map_medium = ev.summarize(true, {}, 2, md[2]);
            res.map_large = ev.summarize(true, {}, 3, md[2]);
            res.mar_1 = ev.summarize(false, {}, 0, md[0]);
            res.mar_10 = ev.summarize(false, {}, 0, md[1]);
            res.mar_100 = ev.summarize(false, {}, 0, md[2]);
            res.mar_small = ev.summarize(false, {}, 1, md[2]);
            res.mar_medium = ev.summarize(false, {}, 2, md[2]);
            res.mar_large = ev.summarize(false, {}, 3, md[2]);
        }
    } // namespace detail

    // torchmetrics MeanAveragePrecision with the pycocotools backend: update() takes one detection_prediction /
    // detection_target per image (converted like torchmetrics: boxes to xywh, area = w * h or the target's area > 0,
    // iscrowd), compute() runs COCOeval over all images seen so far with torchmetrics' parameters and reproduces its
    // float64 numbers (score ties in COCOeval's stable order, crowd / ignore flags, area ranges, maxDets truncation per
    // image and category, 101-point interpolation, numpy's summation order). average micro evaluates every label as
    // one class. iou_type segm compares masks; update() keeps each image's [D x G] mask IoUs, not the masks (an image
    // without masks is left out of that side's COCO dataset, as in torchmetrics: detections on an image without
    // target masks are ignored). Differences: no extended_summary, no
    // ("bbox", "segm") pair, no warnings; mismatched mask sizes are rejected (pycocotools: IoU -1).
    class MeanAveragePrecision : public metric::detail::metric_base<MeanAveragePrecision> {
    public:
        explicit MeanAveragePrecision(mean_average_precision_option opt = {}) : _opt(std::move(opt)) {
            auto &md = _opt._max_detection_thresholds;
            const auto given = md.size();
            const bool negative = std::any_of(md.begin(), md.end(), [](int64_t v) { return v < 0; });
            if (given != 3 || negative) {
                md = {1, 10, 100};
                if (given != 0) {
                    detail::fail(std::format("MeanAveragePrecision: max_detection_thresholds needs 3 values >= 0, got "
                                             "{} values",
                                             given));
                }
            }
            std::sort(md.begin(), md.end());
            if (_opt._iou_thresholds.empty()) {
                _opt._iou_thresholds = detail::values<double>(torch::linspace(0.5, 0.95, 10), torch::kFloat64);
            }
            if (_opt._rec_thresholds.empty()) {
                _opt._rec_thresholds = detail::values<double>(torch::linspace(0.0, 1.0, 101), torch::kFloat64);
            }
        }

        auto update(const std::vector<detection_prediction> &preds, const std::vector<detection_target> &target)
                -> void {
            const bool segm = _opt._iou_type == iou_type_mode::segm;
            if (auto e = detail::input_error(preds, target, segm, true); !e.empty()) {
                return detail::fail(std::move(e));
            }
            for (size_t i = 0; i < preds.size(); ++i) {
                const auto &p = preds[i];
                const auto &t = target[i];
                detail::coco_image im;
                im.dt_score = detail::values<double>(p.scores, torch::kFloat64);
                im.dt_label = detail::values<int64_t>(p.labels, torch::kInt64);
                im.gt_label = detail::values<int64_t>(t.labels, torch::kInt64);
                const auto crowd = t.iscrowd.defined() ? detail::values<int64_t>(t.iscrowd, torch::kInt64)
                                                       : std::vector<int64_t>(im.gt_label.size(), 0);
                const auto area = t.area.defined() ? detail::values<double>(t.area, torch::kFloat64)
                                                   : std::vector<double>(im.gt_label.size(), 0.0);
                for (const auto c: crowd) {
                    im.gt_crowd.push_back(c != 0);
                }
                if (segm) {
                    const auto dm = flat_masks(p.masks), gm = flat_masks(t.masks);
                    im.in_dt = p.masks.size(0) > 0;
                    im.in_gt = t.masks.size(0) > 0;
                    im.dt_area = detail::values<double>(dm.sum(1), torch::kFloat64);
                    im.gt_area = detail::values<double>(gm.sum(1), torch::kFloat64);
                    im.mask_iou = detail::mask_iou(dm, gm, im.gt_crowd);
                } else {
                    im.dt_box = detail::box_rows(p.boxes, _opt._box_format, box_format_mode::xywh);
                    im.gt_box = detail::box_rows(t.boxes, _opt._box_format, box_format_mode::xywh);
                    for (const auto &b: im.dt_box) {
                        im.dt_area.push_back(b[2] * b[3]);
                    }
                    for (const auto &b: im.gt_box) {
                        im.gt_area.push_back(b[2] * b[3]);
                    }
                }
                for (size_t g = 0; g < area.size(); ++g) {
                    im.gt_area[g] = area[g] > 0 ? area[g] : im.gt_area[g];
                }
                _images.push_back(std::move(im));
            }
        }

        auto compute() const -> mean_average_precision_result {
            mean_average_precision_result res;
            std::vector<int64_t> labels;
            bool any_gt = false, any_dt = false;
            for (const auto &im: _images) {
                labels.insert(labels.end(), im.dt_label.begin(), im.dt_label.end());
                labels.insert(labels.end(), im.gt_label.begin(), im.gt_label.end());
                any_gt |= im.in_gt;
                any_dt |= im.in_dt;
            }
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            res.classes = torch::tensor(labels, torch::kInt64).to(torch::kInt32);
            res.map_per_class = torch::full({1}, -1.0, torch::kFloat64);
            res.mar_100_per_class = res.map_per_class.clone();
            if (!any_gt || !any_dt) {
                return res;
            }
            const bool segm = _opt._iou_type == iou_type_mode::segm;
            const detail::coco_params p{_opt._iou_thresholds, _opt._rec_thresholds, _opt._max_detection_thresholds};
            const bool micro = _opt._average == average_mode::micro;
            const detail::coco_eval ev(_images, micro ? std::vector<int64_t>{0} : labels, micro, segm, p);
            detail::coco_stats(ev, p, res);
            if (_opt._class_metrics) {
                std::optional<detail::coco_eval> macro;
                if (micro) {
                    macro.emplace(_images, labels, false, segm, p);
                }
                const auto &per_class = micro ? *macro : ev;
                std::vector<double> map, mar;
                for (size_t k = 0; k < labels.size(); ++k) {
                    map.push_back(per_class.summarize(true, {}, 0, 100, k));
                    mar.push_back(per_class.summarize(false, {}, 0, p.max_dets[2], k));
                }
                res.map_per_class = torch::tensor(map, torch::kFloat64);
                res.mar_100_per_class = torch::tensor(mar, torch::kFloat64);
            }
            return res;
        }

        auto options() const -> const mean_average_precision_option & { return _opt; }

    private:
        static auto flat_masks(const tensor_t &masks) -> tensor_t {
            const auto n = masks.size(0);
            return masks.to(torch::kCPU).reshape({n, n > 0 ? masks.numel() / n : 0}) != 0;
        }

        mean_average_precision_option _opt;
        std::vector<detail::coco_image> _images;
    };

    // torchmetrics.functional.detection.mean_average_precision
    inline auto mean_average_precision(const std::vector<detection_prediction> &preds,
                                       const std::vector<detection_target> &target,
                                       const mean_average_precision_option &opt = {}) -> mean_average_precision_result {
        MeanAveragePrecision m(opt);
        m.update(preds, target);
        return m.compute();
    }

    // ---------------- VOC average precision (VOC devkit / py-faster-rcnn voc_eval) ----------------
    enum class voc_interpolation_mode { all_point, eleven_point };

    typedef struct voc_average_precision_option {
        box_format_mode _box_format = box_format_mode::xyxy;
        double _iou_threshold = 0.5; // a match needs IoU > threshold
        voc_interpolation_mode _interpolation = voc_interpolation_mode::all_point; // eleven_point: VOC2007
        bool _pixel_inclusive = true; // the devkit's +1 on widths and heights (1-based pixel boxes)

        auto box_format(box_format_mode v) -> voc_average_precision_option & {
            _box_format = v;
            return *this;
        }
        auto iou_threshold(double v) -> voc_average_precision_option & {
            _iou_threshold = v;
            return *this;
        }
        auto interpolation(voc_interpolation_mode v) -> voc_average_precision_option & {
            _interpolation = v;
            return *this;
        }
        auto pixel_inclusive(bool v) -> voc_average_precision_option & {
            _pixel_inclusive = v;
            return *this;
        }
    } voc_average_precision_option_t;

    // map: the mean AP over classes with a non-difficult target (-1 without one); ap_per_class: float64 per class of
    // `classes` (the sorted labels of predictions and targets, int64), -1 for a class without non-difficult targets.
    struct voc_average_precision_result {
        double map = -1;
        tensor_t ap_per_class, classes;
    };

    // Per class: detections over all images in descending score order (stable: image order, then input order, for
    // ties; the devkit's quicksort leaves ties unordered), each matched to the target of its image with the largest
    // IoU (first on ties): IoU > threshold and that target not yet taken -> TP, taken -> FP, difficult (iscrowd) ->
    // ignored; IoU <= threshold -> FP. AP from the precision envelope over all recall points or at recall 0, 0.1,
    // ..., 1.
    inline auto voc_average_precision(const std::vector<detection_prediction> &preds,
                                      const std::vector<detection_target> &target,
                                      const voc_average_precision_option &opt = {}) -> voc_average_precision_result {
        TORCHMEDIA_FP_CONTRACT_OFF
        if (auto e = detail::input_error(preds, target, false, true); !e.empty()) {
            return detail::invalid<voc_average_precision_result>(std::move(e));
        }
        const auto n = preds.size();
        std::vector<std::vector<std::array<double, 4>>> dt_box(n), gt_box(n);
        std::vector<std::vector<double>> dt_score(n);
        std::vector<std::vector<int64_t>> dt_label(n), gt_label(n), difficult(n);
        std::vector<int64_t> labels;
        for (size_t i = 0; i < n; ++i) {
            dt_box[i] = detail::box_rows(preds[i].boxes, opt._box_format, box_format_mode::xyxy);
            gt_box[i] = detail::box_rows(target[i].boxes, opt._box_format, box_format_mode::xyxy);
            dt_score[i] = detail::values<double>(preds[i].scores, torch::kFloat64);
            dt_label[i] = detail::values<int64_t>(preds[i].labels, torch::kInt64);
            gt_label[i] = detail::values<int64_t>(target[i].labels, torch::kInt64);
            difficult[i] = target[i].iscrowd.defined() ? detail::values<int64_t>(target[i].iscrowd, torch::kInt64)
                                                       : std::vector<int64_t>(gt_label[i].size(), 0);
            labels.insert(labels.end(), dt_label[i].begin(), dt_label[i].end());
            labels.insert(labels.end(), gt_label[i].begin(), gt_label[i].end());
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        const double off = opt._pixel_inclusive ? 1.0 : 0.0;
        const double eps = std::numeric_limits<double>::epsilon();
        std::vector<double> aps, valid_aps;
        for (const auto c: labels) {
            int64_t npos = 0;
            std::vector<std::vector<uint8_t>> taken(n);
            std::vector<double> scores;
            std::vector<std::pair<size_t, size_t>> dets; // (image, detection)
            for (size_t i = 0; i < n; ++i) {
                taken[i].assign(gt_label[i].size(), 0);
                for (size_t g = 0; g < gt_label[i].size(); ++g) {
                    npos += gt_label[i][g] == c && difficult[i][g] == 0;
                }
                for (size_t d = 0; d < dt_label[i].size(); ++d) {
                    if (dt_label[i][d] == c) {
                        scores.push_back(dt_score[i][d]);
                        dets.emplace_back(i, d);
                    }
                }
            }
            if (npos == 0) {
                aps.push_back(-1.0);
                continue;
            }
            std::vector<int64_t> all(scores.size());
            std::iota(all.begin(), all.end(), 0);
            std::vector<double> rec, prec;
            double tp = 0, fp = 0;
            for (const auto j: detail::score_order(scores, all)) {
                const auto [i, d] = dets[j];
                const auto &bb = dt_box[i][d];
                double ovmax = -std::numeric_limits<double>::infinity();
                int64_t jmax = -1;
                for (size_t g = 0; g < gt_label[i].size(); ++g) {
                    if (gt_label[i][g] != c) {
                        continue;
                    }
                    const auto &gt = gt_box[i][g];
                    const double iw = std::max(std::min(gt[2], bb[2]) - std::max(gt[0], bb[0]) + off, 0.0);
                    const double ih = std::max(std::min(gt[3], bb[3]) - std::max(gt[1], bb[1]) + off, 0.0);
                    const double inters = iw * ih;
                    const double uni = (bb[2] - bb[0] + off) * (bb[3] - bb[1] + off) +
                                       (gt[2] - gt[0] + off) * (gt[3] - gt[1] + off) - inters;
                    const double ov = inters / uni;
                    if (ov > ovmax) {
                        ovmax = ov;
                        jmax = static_cast<int64_t>(g);
                    }
                }
                if (ovmax > opt._iou_threshold) {
                    if (!difficult[i][jmax]) {
                        tp += !taken[i][jmax];
                        fp += taken[i][jmax];
                        taken[i][jmax] = 1;
                    }
                } else {
                    fp += 1;
                }
                rec.push_back(tp / static_cast<double>(npos));
                prec.push_back(tp / std::max(tp + fp, eps));
            }
            double ap = 0.0;
            if (opt._interpolation == voc_interpolation_mode::eleven_point) {
                for (int s = 0; s <= 10; ++s) {
                    const double thr = s * 0.1; // np.arange(0., 1.1, 0.1)
                    double pmax = 0.0;
                    bool any = false;
                    for (size_t j = 0; j < rec.size(); ++j) {
                        if (rec[j] >= thr) {
                            pmax = any ? std::max(pmax, prec[j]) : prec[j];
                            any = true;
                        }
                    }
                    ap = ap + pmax / 11.;
                }
            } else {
                std::vector<double> mrec{0.0}, mpre{0.0};
                mrec.insert(mrec.end(), rec.begin(), rec.end());
                mpre.insert(mpre.end(), prec.begin(), prec.end());
                mrec.push_back(1.0);
                mpre.push_back(0.0);
                for (size_t j = mpre.size() - 1; j > 0; --j) {
                    mpre[j - 1] = std::max(mpre[j - 1], mpre[j]);
                }
                std::vector<double> terms;
                for (size_t j = 0; j + 1 < mrec.size(); ++j) {
                    if (mrec[j + 1] != mrec[j]) {
                        terms.push_back((mrec[j + 1] - mrec[j]) * mpre[j + 1]);
                    }
                }
                ap = metric::detail::np_sum(terms);
            }
            aps.push_back(ap);
            valid_aps.push_back(ap);
        }
        voc_average_precision_result res;
        res.ap_per_class = torch::tensor(aps, torch::kFloat64);
        res.classes = torch::tensor(labels, torch::kInt64);
        if (!valid_aps.empty()) {
            res.map = metric::detail::np_sum(valid_aps) / static_cast<double>(valid_aps.size());
        }
        return res;
    }
} // namespace torchmedia::metric::detection
#endif // LIB_TORCH_MEDIA_METRIC_DETECTION_HPP
