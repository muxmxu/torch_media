// Assertion-based tests for torchmedia::metric::classification (CalibrationError, HingeLoss and the multilabel ranking
// metrics) against torchmetrics 1.9 goldens (gen_golden_classification.py -> golden_classification.hpp) plus
// closed-form, validation and class-semantics checks.
#include "classification_test_util.hpp"

using namespace torchmedia::metric::classification;

static void test_goldens() { check_golden_cases("misc", cls_driver::run_misc); }

static const auto BP = torch::tensor({0.9f, 0.2f, 0.6f, 0.4f, 0.8f, 0.1f});
static const auto BT = torch::tensor({1, 0, 0, 1, 1, 0});
static const auto MP = torch::tensor({{0.7f, 0.2f, 0.1f},
                                      {0.1f, 0.8f, 0.1f},
                                      {0.3f, 0.3f, 0.4f},
                                      {0.5f, 0.4f, 0.1f},
                                      {0.2f, 0.2f, 0.6f},
                                      {0.1f, 0.6f, 0.3f}});
static const auto MT = torch::tensor({0, 1, 1, 0, 2, 2});
static const auto LP = torch::tensor({{0.9f, 0.1f, 0.5f}, {0.2f, 0.7f, 0.6f}, {0.6f, 0.4f, 0.3f}, {0.3f, 0.8f, 0.1f}});
static const auto LT = torch::tensor({{1, 0, 0}, {0, 1, 1}, {1, 1, 0}, {0, 0, 0}});

static auto same(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::allclose(a, b, 0.0, 0.0, true);
}

static void test_closed_form() {
    // binary calibration: confidence = probability of the positive class, accuracy = target; one bin: |mean p - mean t|
    const auto ce1 = binary_calibration_error(BP, BT, binary_calibration_error_option().n_bins(1));
    TM_CHECK_CLOSE(ce1.item<float>(), std::abs((0.9 + 0.2 + 0.6 + 0.4 + 0.8 + 0.1) / 6 - 0.5), 1e-7);
    // perfectly calibrated -> l2 is torchmetrics' int64 0
    const auto zero = binary_calibration_error(torch::tensor({0.0f, 1.0f}), torch::tensor({0, 1}),
                                               binary_calibration_error_option().norm(calibration_norm_mode::l2));
    TM_CHECK(same(zero, torch::tensor(int64_t{0})));
    // hinge on probabilities: margins +0.9 -0.2 -0.6 +0.4 +0.8 -0.1 -> mean of 1 - margin
    TM_CHECK_CLOSE(binary_hinge_loss(BP, BT).item<float>(), (0.1 + 1.2 + 1.6 + 0.6 + 0.2 + 1.1) / 6, 1e-6);
    // coverage: labels scored at least the lowest relevant one: 1, 2, 2 and 0 (no relevant label: the offset lifts
    // every score above all raw scores)
    const auto o = multilabel_ranking_option().num_labels(3);
    TM_CHECK(multilabel_coverage_error(LP, LT, o).item<float>() == 1.25f);
    // ranking loss / average precision when no sample has both relevant and irrelevant labels: 0 and 1
    const auto all = torch::ones({2, 3}, torch::kLong);
    TM_CHECK(same(multilabel_ranking_loss(LP.slice(0, 0, 2), all, o), torch::tensor(0.0f)));
    TM_CHECK(same(multilabel_ranking_average_precision(LP.slice(0, 0, 2), all, o), torch::tensor(1.0f)));
}

static void test_validation() {
    const auto inv = [](auto fn) { return throws_invalid(fn); };
    using BC = binary_calibration_error_option;
    using MC = multiclass_calibration_error_option;
    TM_CHECK(inv([] { return binary_calibration_error(BP, BT, BC().n_bins(0)); }));
    TM_CHECK(inv([] { return binary_calibration_error(BT, BT); })); // label preds
    TM_CHECK(inv([] { return binary_calibration_error(BP, BT.slice(0, 1)); }));
    TM_CHECK(inv([] { return multiclass_calibration_error(MP, MT, MC()); }));
    TM_CHECK(inv([] { return multiclass_calibration_error(MP, MT, MC().num_classes(3).n_bins(-1)); }));
    TM_CHECK(inv([] { return multiclass_calibration_error(MT, MT, MC().num_classes(3)); }));
    TM_CHECK(inv([] { return multiclass_calibration_error(MP, MT.slice(0, 2), MC().num_classes(3)); }));
    TM_CHECK(inv([] { return binary_hinge_loss(BT, BT); }));
    TM_CHECK(inv([] { return binary_hinge_loss(BP, BT + 2); }));
    TM_CHECK(inv([] { return multiclass_hinge_loss(MP, MT, multiclass_hinge_loss_option()); }));
    TM_CHECK(inv([] { return multiclass_hinge_loss(MT, MT, multiclass_hinge_loss_option().num_classes(3)); }));
    TM_CHECK(inv(
            [] { return multiclass_hinge_loss(MP, MT.slice(0, 1), multiclass_hinge_loss_option().num_classes(3)); }));
    const auto o = multilabel_ranking_option().num_labels(3);
    TM_CHECK(inv([] { return multilabel_coverage_error(LP, LT, multilabel_ranking_option()); }));
    TM_CHECK(inv([&] { return multilabel_ranking_loss(LT, LT, o); })); // label preds
    TM_CHECK(inv([&] { return multilabel_ranking_average_precision(LP, LT.slice(1, 0, 2), o); }));
    TM_CHECK(inv([&] { return multilabel_coverage_error(LP, LT + 2, o); }));
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
    check_class_validation<BinaryCalibrationError>(binary_calibration_error_option().n_bins(0), {}, BP, BT, BT + 2);
    check_class_validation<MulticlassCalibrationError>(multiclass_calibration_error_option(),
                                                       multiclass_calibration_error_option().num_classes(3), MP, MT,
                                                       MT.slice(0, 1));
    TM_CHECK(!throws_any([] { return BinaryHingeLoss(); }));
    check_class_validation<MulticlassHingeLoss>(multiclass_hinge_loss_option(),
                                                multiclass_hinge_loss_option().num_classes(3), MP, MT, MT.slice(0, 1));
    check_class_validation<MultilabelRankingLoss>(multilabel_ranking_option(),
                                                  multilabel_ranking_option().num_labels(3), LP, LT, LT + 2);
    TM_CHECK(throws_invalid([] {
        BinaryHingeLoss m;
        m.update(BP, BT + 2);
        return 0;
    }));
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
    TM_CHECK_TENSOR_CLOSE(m.compute(), full.compute(), 1e-6, 0.0); // float sums in another order
    m.reset();
    m.update(p, t);
    TM_CHECK(same(m.compute(), full.compute()));
}

static void test_class_semantics() {
    check_forward<BinaryCalibrationError>(binary_calibration_error_option().norm(calibration_norm_mode::max), BP, BT);
    check_forward<MulticlassCalibrationError>(multiclass_calibration_error_option().num_classes(3), MP, MT);
    check_forward<BinaryHingeLoss>(binary_hinge_loss_option().squared(true), BP, BT);
    check_forward<MulticlassHingeLoss>(
            multiclass_hinge_loss_option().num_classes(3).multiclass_mode(hinge_multiclass_mode::one_vs_all), MP, MT);
    check_forward<MultilabelCoverageError>(multilabel_ranking_option().num_labels(3), LP, LT);
    check_forward<MultilabelRankingAveragePrecision>(multilabel_ranking_option().num_labels(3), LP, LT);
    check_forward<MultilabelRankingLoss>(multilabel_ranking_option().num_labels(3), LP, LT);
    // float64 inputs keep a float64 hinge sum (progress11 D5; torchmetrics' float32 state would round it)
    BinaryHingeLoss h;
    h.update(BP.to(torch::kFloat64), BT);
    TM_CHECK(h.compute().scalar_type() == torch::kFloat64);
    // before any update: torch.cat of no batch (calibration), 0 / 0 (hinge, ranking)
    TM_CHECK(throws_any([] { return BinaryCalibrationError().compute(); }));
    TM_CHECK(std::isnan(BinaryHingeLoss().compute().item<float>()));
    TM_CHECK(std::isnan(MultilabelRankingLoss(multilabel_ranking_option().num_labels(2)).compute().item<float>()));
}

// States own their data: a caller that reuses one input buffer must not rewrite earlier updates.
static void test_state_owns_inputs() {
    auto p = BP.clone(), t = BT.clone();
    BinaryCalibrationError m, ref;
    m.update(p, t);
    ref.update(BP, BT);
    p.fill_(0.5f);
    t.fill_(0);
    TM_CHECK(torch::equal(m.compute(), ref.compute()));
}

int main() {
    TM_RUN(test_goldens);
    TM_RUN(test_closed_form);
    TM_RUN(test_validation);
    TM_RUN(test_class_validation);
    TM_RUN(test_class_semantics);
    TM_RUN(test_state_owns_inputs);
    return tm_test::summary("metric_test_classification_misc");
}
