// Assertion-based tests for torchmedia::metric::classification (curve family: PrecisionRecallCurve, ROC, AUROC,
// AveragePrecision, LogAUC, EER, fixed-point metrics) against torchmetrics 1.9 goldens (gen_golden_classification.py
// -> golden_classification.hpp) plus closed-form, validation and class-semantics checks.
#include "classification_test_util.hpp"

using namespace torchmedia::metric::classification;
using AV = average_mode;

static void test_goldens() { check_golden_cases("curve", cls_driver::run_curve); }

static const auto BP = torch::tensor({0.9f, 0.2f, 0.6f, 0.4f, 0.8f, 0.1f});
static const auto BT = torch::tensor({1, 0, 0, 1, 1, 0});
static const auto MP = torch::tensor({{0.7f, 0.2f, 0.1f},
                                      {0.1f, 0.8f, 0.1f},
                                      {0.3f, 0.3f, 0.4f},
                                      {0.5f, 0.4f, 0.1f},
                                      {0.2f, 0.2f, 0.6f},
                                      {0.1f, 0.6f, 0.3f}});
static const auto MT = torch::tensor({0, 1, 1, 0, 2, 2});
static const auto LP = torch::tensor({{0.9f, 0.1f}, {0.2f, 0.7f}, {0.6f, 0.4f}, {0.3f, 0.8f}});
static const auto LT = torch::tensor({{1, 0}, {0, 1}, {1, 1}, {0, 0}});

static auto same(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::allclose(a, b, 0.0, 0.0, true);
}

template<class A, class B>
static auto same(const std::pair<A, B> &a, const std::pair<A, B> &b) -> bool {
    return same(a.first, b.first) && same(a.second, b.second);
}
static auto same(const curve_t &a, const curve_t &b) -> bool {
    return same(std::get<0>(a), std::get<0>(b)) && same(std::get<1>(a), std::get<1>(b)) &&
           same(std::get<2>(a), std::get<2>(b));
}
static auto same(const curve_list_t &a, const curve_list_t &b) -> bool {
    const auto eq = [](const std::vector<torch::Tensor> &x, const std::vector<torch::Tensor> &y) {
        return x.size() == y.size() &&
               std::equal(x.begin(), x.end(), y.begin(), [](const auto &u, const auto &v) { return same(u, v); });
    };
    return eq(std::get<0>(a), std::get<0>(b)) && eq(std::get<1>(a), std::get<1>(b)) &&
           eq(std::get<2>(a), std::get<2>(b));
}

static void test_closed_form() {
    // descending scores 0.9 0.8 0.6 0.4 0.2 0.1 with labels 1 1 0 1 0 0
    const auto [fpr, tpr, thr] = binary_roc(BP, BT);
    TM_CHECK(same(thr, torch::tensor({1.0f, 0.9f, 0.8f, 0.6f, 0.4f, 0.2f, 0.1f})));
    TM_CHECK(same(fpr, torch::tensor({0.0f, 0.0f, 0.0f, 1.0f / 3, 1.0f / 3, 2.0f / 3, 1.0f})));
    TM_CHECK(same(tpr, torch::tensor({0.0f, 1.0f / 3, 2.0f / 3, 2.0f / 3, 1.0f, 1.0f, 1.0f})));
    TM_CHECK_CLOSE(binary_auroc(BP, BT).item<float>(), 8.0 / 9.0, 1e-7);
    const auto [p, r, t] = binary_precision_recall_curve(BP, BT);
    TM_CHECK(same(p, torch::tensor({0.5f, 0.6f, 0.75f, 2.0f / 3, 1.0f, 1.0f, 1.0f})));
    TM_CHECK(same(r, torch::tensor({1.0f, 1.0f, 1.0f, 2.0f / 3, 2.0f / 3, 1.0f / 3, 0.0f})));
    // binned: duplicate thresholds give equal (precision, recall, threshold) tuples (Python's max keeps the first)
    const auto o = binary_curve_option().thresholds(std::vector<double>{0.5, 0.5}).min_recall(0.5);
    const auto [prec, best] = binary_precision_at_fixed_recall(BP, BT, o);
    TM_CHECK(prec.item<float>() == 2.0f / 3 && best.item<float>() == 0.5f);
    // a single class without positives: AUROC macro leaves its NaN out, weighted gives it weight 0
    const auto mt = torch::tensor({0, 0, 1, 0, 1, 1});
    const auto mo = multiclass_auc_option().num_classes(3).average(AV::none);
    TM_CHECK(multiclass_auroc(MP, mt, mo)[2].item<float>() == 0.0f); // tpr 0 everywhere
    TM_CHECK(multiclass_average_precision(MP, mt, multiclass_auc_option(mo).thresholds(5))[2].item<float>() == 0.0f);
    // threshold setters (a named vector, a braced list, a count, a tensor)
    const std::vector<double> v{0.25, 0.75};
    TM_CHECK(std::get<std::vector<double>>(multiclass_curve_option().thresholds(v)._thresholds) == v);
    TM_CHECK(std::get<std::vector<double>>(multilabel_curve_option().thresholds(v)._thresholds) == v);
    TM_CHECK(std::get<std::vector<double>>(multiclass_auc_option().thresholds(v)._thresholds) == v);
    TM_CHECK(std::get<std::vector<double>>(multilabel_auc_option().thresholds(v)._thresholds) == v);
    TM_CHECK(std::get<std::vector<double>>(binary_curve_option().thresholds(v)._thresholds) == v);
    TM_CHECK(std::holds_alternative<int64_t>(multiclass_curve_option().thresholds(3)._thresholds));
    TM_CHECK(std::holds_alternative<std::vector<double>>(multilabel_curve_option().thresholds({0.1, 0.2})._thresholds));
    TM_CHECK(std::holds_alternative<torch::Tensor>(multilabel_auc_option().thresholds(torch::ones({2}))._thresholds));
    TM_CHECK(std::holds_alternative<int64_t>(multiclass_auc_option().thresholds(4)._thresholds));
    TM_CHECK(std::holds_alternative<std::vector<double>>(binary_curve_option().thresholds({0.1, 0.2})._thresholds));
    TM_CHECK(std::holds_alternative<torch::Tensor>(multiclass_curve_option().thresholds(torch::ones({2}))._thresholds));
    TM_CHECK(std::holds_alternative<int64_t>(multilabel_curve_option().thresholds(2)._thresholds));
    TM_CHECK(std::holds_alternative<torch::Tensor>(binary_curve_option().thresholds(torch::ones({2}))._thresholds));
    TM_CHECK(std::holds_alternative<std::vector<double>>(multiclass_auc_option().thresholds({0.5, 0.6})._thresholds));
    TM_CHECK(std::holds_alternative<int64_t>(multilabel_auc_option().thresholds(2)._thresholds));
    TM_CHECK(std::holds_alternative<std::vector<double>>(multilabel_auc_option().thresholds({0.5})._thresholds));
    TM_CHECK(std::holds_alternative<torch::Tensor>(multiclass_auc_option().thresholds(torch::ones({2}))._thresholds));
    TM_CHECK(std::holds_alternative<std::vector<double>>(multiclass_curve_option().thresholds({0.5})._thresholds));
    TM_CHECK(std::holds_alternative<torch::Tensor>(multilabel_curve_option().thresholds(torch::ones({2}))._thresholds));
}

// torchmetrics' binned update compares in the promoted dtype up to 50000 binary values (vectorized) and against a
// 0-d threshold, i.e. in preds' dtype, above (loop): a float64 threshold just above 0.3f is above 0.3f in the first
// form and equal to it in the second.
static void test_loop_threshold_dtype() {
    // the next double above 0.3f (it rounds to 0.3f in float32)
    const auto t = torch::tensor({std::nextafter(static_cast<double>(0.3f), 1.0)}, torch::kFloat64);
    const auto o = binary_curve_option().thresholds(t);
    for (const int64_t n: {int64_t{50000}, int64_t{50002}}) {
        const auto preds = torch::full({n}, 0.3f);
        const auto target = torch::arange(n) % 2;
        const auto [precision, recall, thr] = binary_precision_recall_curve(preds, target, o);
        TM_CHECK(recall[0].item<float>() == (n > 50000 ? 1.0f : 0.0f));
    }
    // multiclass: the loop beyond 1e6 = M C C values per class; micro: beyond 50000 flattened values
    for (const int64_t m: {int64_t{250000}, int64_t{250001}, int64_t{25000}, int64_t{25001}}) {
        const bool micro = m < 100000;
        const auto preds = torch::full({m, 2}, 0.3f);
        const auto target = torch::arange(m) % 2;
        const auto co = multiclass_curve_option().num_classes(2).thresholds(t).average(micro ? AV::micro : AV::none);
        const auto [precision, recall, thr] = multiclass_precision_recall_curve(preds, target, co);
        TM_CHECK(recall[0][0].item<float>() == (m % 2 ? 1.0f : 0.0f));
    }
    // multilabel: always vectorized (promoted)
    const auto lo = multilabel_curve_option().num_labels(2).thresholds(t);
    const auto [lp, lr, lt] = multilabel_precision_recall_curve(torch::full({4, 2}, 0.3f), LT, lo);
    TM_CHECK(lr[0][0].item<float>() == 0.0f && lr[1][0].item<float>() == 0.0f);
}

// Multilabel binned curves skip ignored entries with a mask, so a uint8 target gives the int64 target's result;
// torchmetrics' -4 * num_labels sentinel wraps in uint8 and its bincount raises (progress02 D7).
static void test_uint8_target() {
    const auto p = torch::tensor({{0.2f, 0.7f}, {0.6f, 0.1f}}), t = torch::tensor({{0, 1}, {1, 2}});
    const auto o = multilabel_curve_option().num_labels(2).thresholds(3).ignore_index(2);
    const auto u8 = multilabel_roc(p, t.to(torch::kUInt8), o);
    TM_CHECK(same(u8, multilabel_roc(p, t, o)) && same(std::get<0>(u8)[0], torch::tensor({0.0f, 0.0f, 1.0f})));
    const auto ao = multilabel_auc_option().num_labels(2).thresholds(3).ignore_index(2);
    MultilabelAUROC m(ao);
    m.update(p, t.to(torch::kUInt8));
    TM_CHECK(same(m.compute(), multilabel_auroc(p, t, ao)));
}

static void test_validation() {
    const auto inv = [](auto fn) { return throws_invalid(fn); };
    using BO = binary_curve_option;
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(1)); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(std::vector<double>{})); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(std::vector<double>{0.2, 1.5})); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(torch::Tensor())); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(torch::ones({2, 2}))); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT, BO().thresholds(torch::ones({0}))); }));
    TM_CHECK(inv([] { return binary_recall_at_fixed_precision(BP, BT, BO()); }));
    TM_CHECK(inv([] { return binary_precision_at_fixed_recall(BP, BT, BO().min_recall(1.5)); }));
    TM_CHECK(inv([] { return binary_sensitivity_at_specificity(BP, BT, BO().min_specificity(-0.5)); }));
    TM_CHECK(inv([] { return binary_specificity_at_sensitivity(BP, BT, BO().min_precision(0.5)); }));
    TM_CHECK(inv([] { return binary_logauc(BP, BT, BO().fpr_range({0.5, 0.1})); }));
    TM_CHECK(inv([] { return binary_logauc(BP, BT, BO().fpr_range({-0.1, 0.5})); }));
    TM_CHECK(!throws_any([] { return binary_auroc(BP, BT, BO().fpr_range({0.5, 0.1})); })); // LogAUC only
    TM_CHECK(inv([] { return binary_auroc(BP, BT, BO().max_fpr(0.0)); }));
    TM_CHECK(inv([] { return binary_auroc(BP, BT, BO().max_fpr(1.5)); }));
    using CO = multiclass_curve_option;
    using CA = multiclass_auc_option;
    using LO = multilabel_curve_option;
    using LA = multilabel_auc_option;
    TM_CHECK(inv([] { return multiclass_roc(MP, MT, CO().num_classes(1)); }));
    TM_CHECK(inv([] { return multiclass_roc(MP, MT, CO().num_classes(3).average(AV::weighted)); }));
    TM_CHECK(inv([] {
        return multiclass_recall_at_fixed_precision(MP, MT, CO().num_classes(3).min_precision(0.5).average(AV::micro));
    }));
    TM_CHECK(inv([] { return multiclass_auroc(MP, MT, CA().num_classes(3).average(AV::micro)); }));
    TM_CHECK(inv([] { return multiclass_logauc(MP, MT, CA().num_classes(3).average(AV::weighted)); }));
    TM_CHECK(inv([] { return multiclass_auroc(MP, MT, CA()); }));
    TM_CHECK(inv([] { return multilabel_roc(LP, LT, LO()); }));
    TM_CHECK(inv([] { return multilabel_auroc(LP, LT, LA()); }));
    TM_CHECK(inv([] { return multilabel_logauc(LP, LT, LA().num_labels(2).average(AV::micro)); }));
    TM_CHECK(!throws_any([] { return multilabel_auroc(LP, LT, LA().num_labels(2).average(AV::micro)); }));
    // tensors
    TM_CHECK(inv([] { return binary_roc(BP, BT.slice(0, 1)); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT.to(torch::kFloat32)); }));
    TM_CHECK(inv([] { return binary_roc(BT, BT); }));
    TM_CHECK(inv([] { return binary_roc(BP, BT * 3); }));
    const auto C3 = CO().num_classes(3);
    TM_CHECK(inv([&] { return multiclass_roc(MT.to(torch::kFloat32), MT, C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP, MT.to(torch::kFloat32), C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP.to(torch::kLong), MT, C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP.slice(1, 0, 2), MT, C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP, MT.slice(0, 1), C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP.unsqueeze(-1), MT.unsqueeze(-1).repeat({1, 2}), C3); }));
    TM_CHECK(inv([&] { return multiclass_roc(MP, torch::tensor({0, 1, 2, 3, 0, 1}), C3); }));
    TM_CHECK(inv([] { return multilabel_roc(LP, LT.to(torch::kFloat32), LO().num_labels(2)); }));
    TM_CHECK(inv([] { return multilabel_roc(LP, LT, LO().num_labels(3)); }));
    TM_CHECK(inv([] { return multilabel_roc(LP[0], LT[0], LO().num_labels(2)); }));
    // torch's own errors stay c10::Error: everything ignored (torchmetrics: IndexError on the empty curve), a NaN
    // recall column in _lexargmax (torchmetrics: IndexError)
    TM_CHECK(throws_any([] { return binary_roc(BP, torch::full({6}, -1, torch::kLong), BO().ignore_index(-1)); }));
    TM_CHECK(throws_any([] {
        return multiclass_recall_at_fixed_precision(MP, torch::zeros({6}, torch::kLong),
                                                    CO().num_classes(3).thresholds(5).min_precision(0.0));
    }));
}

template<class M, class Opt>
static void check_class_validation(const Opt &bad, const Opt &good, const torch::Tensor &p, const torch::Tensor &t,
                                   const torch::Tensor &bad_t) {
    TM_CHECK(throws_invalid([&] { return M(bad); }));
    M m(good);
    m.update(p, t);
    const auto before = m.compute();
    TM_CHECK(throws_invalid([&] {
        m.update(p, bad_t);
        return 0;
    }));
    TM_CHECK(same(m.compute(), before));
}

static void test_class_validation() {
    check_class_validation<BinaryAUROC>(binary_curve_option().max_fpr(2), {}, BP, BT, BT + 2);
    check_class_validation<BinaryRecallAtFixedPrecision>(
            binary_curve_option(), binary_curve_option().min_precision(0.5).thresholds(5), BP, BT, BT + 2);
    check_class_validation<MulticlassEER>(multiclass_curve_option(), multiclass_curve_option().num_classes(3), MP, MT,
                                          MT.slice(0, 1));
    check_class_validation<MulticlassAveragePrecision>(
            multiclass_auc_option(), multiclass_auc_option().num_classes(3).thresholds(4), MP, MT, MT.slice(0, 1));
    check_class_validation<MultilabelSensitivityAtSpecificity>(
            multilabel_curve_option().num_labels(2), multilabel_curve_option().num_labels(2).min_specificity(0.5), LP,
            LT, LT + 2);
    check_class_validation<MultilabelLogAUC>(multilabel_auc_option(), multilabel_auc_option().num_labels(2), LP, LT,
                                             LT + 2);
}

// forward() on an instance that already holds state returns this batch alone and accumulates it; reset() empties.
template<class M, class Opt>
static void check_forward(const Opt &o, const torch::Tensor &p, const torch::Tensor &t) {
    const auto k = p.size(0) / 2;
    M m(o), alone(o), full(o);
    m.update(p.slice(0, 0, k), t.slice(0, 0, k));
    alone.update(p.slice(0, k), t.slice(0, k));
    full.update(p, t);
    TM_CHECK(same(m(p.slice(0, k), t.slice(0, k)), alone.compute()));
    TM_CHECK(same(m.compute(), full.compute()));
    m.reset();
    m.update(p, t);
    TM_CHECK(same(m.compute(), full.compute()));
}

static void test_class_semantics() {
    check_forward<BinaryPrecisionRecallCurve>(binary_curve_option(), BP, BT);
    check_forward<BinaryROC>(binary_curve_option().thresholds(7), BP, BT);
    check_forward<MulticlassPrecisionRecallCurve>(multiclass_curve_option().num_classes(3).average(AV::macro), MP, MT);
    check_forward<MulticlassROC>(multiclass_curve_option().num_classes(3).thresholds(5), MP, MT);
    check_forward<MultilabelROC>(multilabel_curve_option().num_labels(2), LP, LT);
    check_forward<MulticlassAUROC>(multiclass_auc_option().num_classes(3), MP, MT);
    check_forward<MultilabelAveragePrecision>(multilabel_auc_option().num_labels(2).thresholds(4), LP, LT);
    check_forward<BinarySpecificityAtSensitivity>(binary_curve_option().min_sensitivity(0.5), BP, BT);
    // exact state before any update: torch.cat of no batch; binned: the zero confusion matrices (NaN precision)
    TM_CHECK(throws_any([] { return BinaryAUROC().compute(); }));
    const auto [p, r, t] = BinaryPrecisionRecallCurve(binary_curve_option().thresholds(3)).compute();
    TM_CHECK(torch::isnan(p.slice(0, 0, 3)).all().item<bool>() && p[3].item<float>() == 1.0f);
    // the class squeezes one-element outputs (Metric._squeeze_if_scalar): one exact threshold
    BinaryPrecisionRecallCurve one;
    one.update(torch::full({2}, 0.5f), torch::tensor({0, 1}));
    TM_CHECK(std::get<2>(one.compute()).dim() == 0 &&
             std::get<2>(binary_precision_recall_curve(torch::full({2}, 0.5f), torch::tensor({0, 1}))).dim() == 1);
    MultilabelPrecisionRecallCurve ml(multilabel_curve_option().num_labels(2));
    ml.update(torch::full({2, 2}, 0.5f), torch::tensor({{0, 1}, {1, 0}}));
    TM_CHECK(std::get<2>(ml.compute())[0].dim() == 0);
    // ... but not the rows of a stacked per-class curve (torchmetrics' [C, 1] tensor): one binned threshold
    MulticlassROC rows(multiclass_curve_option().num_classes(3).thresholds({0.5}));
    rows.update(MP, MT);
    const auto [fprs, tprs, thrs] = rows.compute();
    TM_CHECK(same(fprs[1], torch::tensor({0.25f})) && same(tprs[1], torch::tensor({0.5f})) && thrs[1].dim() == 0);
}

// classification::binary_eer (torchmetrics: the ROC point closest to fpr = fnr, (fpr + fnr) / 2) against
// metric::audio::equal_error_rate (sklearn ROC, linear interpolation at the crossing; progress02 D9).
static void test_eer_relation() {
    // a ROC point with fpr = fnr: both conventions give it
    const auto tar = torch::tensor({0.9, 0.8, 0.3}), non = torch::tensor({0.7, 0.2, 0.1});
    const auto preds = torch::cat({tar, non}), target = torch::tensor({1, 1, 1, 0, 0, 0});
    TM_CHECK_CLOSE(binary_eer(preds, target).item<double>(), 1.0 / 3, 1e-7); // float32 rates
    TM_CHECK(torchmedia::metric::audio::equal_error_rate(tar, non).item<double>() == 1.0 / 3);
    // a crossing between ROC points (fpr, fnr) (1/3, 1/2) and (2/3, 1/2): torchmetrics averages the first of the two
    // equally near points, the audio EER interpolates to fpr = fnr = 1/2
    const auto tar2 = torch::tensor({0.9, 0.6, 0.4, 0.3}), non2 = torch::tensor({0.8, 0.5, 0.2});
    const auto e_cls = binary_eer(torch::cat({tar2, non2}), torch::tensor({1, 1, 1, 1, 0, 0, 0})).item<double>();
    TM_CHECK_CLOSE(e_cls, 5.0 / 12, 1e-7);
    TM_CHECK(torchmedia::metric::audio::equal_error_rate(tar2, non2).item<double>() == 0.5);
    // logits: torchmetrics' float32 sigmoid maps 18, 19, 20 and 21 to 1.0 (a four-way tie): 0.25; the audio EER of the
    // raw scores is 0.5
    const auto ltar = torch::tensor({18.0f, 19.0f, 20.0f}), lnon = torch::tensor({21.0f, 0.0f});
    TM_CHECK(binary_eer(torch::cat({ltar, lnon}), torch::tensor({1, 1, 1, 0, 0})).item<float>() == 0.25f);
    TM_CHECK(torchmedia::metric::audio::equal_error_rate(ltar, lnon).item<double>() == 0.5);
}

static void test_mps() {
    if (!torch::mps::is_available()) {
        return;
    }
    const auto dev = torch::Device(torch::kMPS);
    const auto logits = torch::logit(BP);
    // MPS sigmoid / softmax round differently from the CPU kernels: compare within float32 rounding
    TM_CHECK_CLOSE(binary_auroc(logits.to(dev), BT.to(dev)).cpu().item<float>(), binary_auroc(logits, BT).item<float>(),
                   1e-6);
    const auto o = multiclass_auc_option().num_classes(3).average(AV::weighted);
    TM_CHECK_CLOSE(multiclass_auroc(MP.to(dev), MT.to(dev), o).cpu().item<float>(),
                   multiclass_auroc(MP, MT, o).item<float>(), 1e-6);
    MulticlassROC m(multiclass_curve_option().num_classes(3).thresholds(5));
    m.update(MP.to(dev), MT.to(dev));
    const auto [fpr, tpr, thr] = m.compute();
    TM_CHECK(fpr[0].device().is_mps());
    // the fixed-point metrics' float64 tie-break runs on the CPU (MPS has no float64)
    const auto po = binary_curve_option().min_recall(0.5);
    const auto [mp, mt] = binary_precision_at_fixed_recall(BP.to(dev), BT.to(dev), po);
    const auto [cp, ct] = binary_precision_at_fixed_recall(BP, BT, po);
    TM_CHECK(same(mp.cpu(), cp) && same(mt.cpu(), ct));
    // [N, L, E] / [N, C, E] scores give non-contiguous [M, L] views; binned counts equal the CPU's
    const auto x = torch::arange(120);
    const auto sp = (x * 37 % 101).to(torch::kFloat32).div(101).reshape({10, 3, 4});
    const auto lt = (x * 53 % 7 < 3).to(torch::kLong).reshape({10, 3, 4});
    const auto lo = multilabel_curve_option().num_labels(3).thresholds(11);
    const auto [lf, lr, lth] = multilabel_roc(sp.to(dev), lt.to(dev), lo);
    const auto [cf, cr, cth] = multilabel_roc(sp, lt, lo);
    for (size_t i = 0; i < cf.size(); ++i) {
        TM_CHECK(same(lf[i].cpu(), cf[i]) && same(lr[i].cpu(), cr[i]));
    }
    const auto ct3 = torch::tensor({0, 1, 2, 2, 1, 0, 0, 1, 2, 1}).unsqueeze(1).repeat({1, 4});
    const auto co = multiclass_curve_option().num_classes(3).thresholds(11);
    const auto [mf, mr, mth] = multiclass_roc(sp.to(dev), ct3.to(dev), co);
    const auto [kf, kr, kth] = multiclass_roc(sp, ct3, co);
    for (size_t i = 0; i < kf.size(); ++i) {
        TM_CHECK(same(mf[i].cpu(), kf[i]) && same(mr[i].cpu(), kr[i]));
    }
}

// States own their data: a caller that reuses one input buffer must not rewrite earlier updates.
static void test_state_owns_inputs() {
    auto p = BP.clone(), t = BT.clone();
    BinaryAUROC m, ref;
    m.update(p, t);
    ref.update(BP, BT);
    p.fill_(0.5f);
    t.fill_(0);
    TM_CHECK(torch::equal(m.compute(), ref.compute()));
}

int main() {
    TM_RUN(test_goldens);
    TM_RUN(test_closed_form);
    TM_RUN(test_loop_threshold_dtype);
    TM_RUN(test_uint8_target);
    TM_RUN(test_validation);
    TM_RUN(test_class_validation);
    TM_RUN(test_class_semantics);
    TM_RUN(test_eer_relation);
    TM_RUN(test_mps);
    TM_RUN(test_state_owns_inputs);
    return tm_test::summary("metric_test_classification_curve");
}
