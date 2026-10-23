// Assertion-based tests for torchmedia::metric::classification (stat-score and confusion-matrix families) against
// torchmetrics 1.9 goldens (gen_golden_classification.py -> golden_classification.hpp) plus closed-form, validation
// and class-semantics checks.
#include "classification_test_util.hpp"

using namespace torchmedia::metric::classification;
using MD = multidim_average_mode;
using AV = average_mode;

static void test_goldens() { check_golden_cases("stat", cls_driver::run_stat); }

static const auto BP = torch::tensor({0.9f, 0.2f, 0.6f, 0.4f, 0.8f, 0.1f});
static const auto BT = torch::tensor({1, 0, 0, 1, 1, 0});
static const auto MP = torch::tensor({{0.7f, 0.2f, 0.1f},
                                      {0.1f, 0.8f, 0.1f},
                                      {0.3f, 0.3f, 0.4f},
                                      {0.5f, 0.4f, 0.1f},
                                      {0.2f, 0.2f, 0.6f},
                                      {0.1f, 0.6f, 0.3f}});
static const auto MT = torch::tensor({0, 1, 1, 0, 2, 2});

static auto same(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::allclose(a, b, 0.0, 0.0, true);
}

static void test_closed_form() {
    // threshold 0.5: preds 1 0 1 0 1 0 -> tp 2, fp 1, tn 2, fn 1
    TM_CHECK(same(binary_stat_scores(BP, BT), torch::tensor({2, 1, 2, 1, 3})));
    TM_CHECK(binary_accuracy(BP, BT).item<float>() == 4.0f / 6.0f);
    TM_CHECK(same(binary_stat_scores(torch::logit(BP), BT), binary_stat_scores(BP, BT))); // logits: sigmoid first
    // threshold 0.15: tp 3, fp 2, fn 0 -> F2 = 5 * 3 / (5 * 3 + 2), F1 = 6 / 8 (F1Score ignores beta)
    const auto t15 = binary_stat_scores_option().threshold(0.15);
    TM_CHECK(binary_fbeta_score(BP, BT, binary_stat_scores_option(t15).beta(2)).item<float>() == 15.0f / 17.0f);
    TM_CHECK(binary_f1_score(BP, BT, binary_stat_scores_option(t15).beta(3)).item<float>() == 0.75f);
    // multiclass argmax 0 1 2 0 2 1 -> micro accuracy 4 / 6, also without num_classes
    const auto micro = multiclass_stat_scores_option().average(AV::micro);
    TM_CHECK(multiclass_accuracy(MP, MT, micro).item<float>() == 4.0f / 6.0f);
    MulticlassAccuracy m(micro);
    m.update(MP, MT);
    TM_CHECK(m.compute().item<float>() == 4.0f / 6.0f);
    TM_CHECK(same(multiclass_confusion_matrix(MP, MT, multiclass_confusion_matrix_option().num_classes(3)),
                  torch::tensor({{2, 0, 0}, {0, 1, 1}, {0, 1, 1}})));
    // top_k 2: the target is among the two best scores of every sample but the tied one (0.3 0.3 0.4, target 1)
    const auto k2 = multiclass_stat_scores_option().num_classes(3).top_k(2).average(AV::micro);
    const auto rows = torch::tensor({0, 1, 3, 4, 5});
    TM_CHECK(multiclass_accuracy(MP.index_select(0, rows), MT.index_select(0, rows), k2).item<float>() == 1.0f);
    // samplewise with an ignore_index inside [0, C): its one-hot target rows are -1 (torchmetrics' values)
    const auto sp = torch::tensor({{0, 1, 2}, {2, 2, 0}}), st = torch::tensor({{0, 1, 1}, {1, 2, 0}});
    const auto sw = multiclass_stat_scores_option().num_classes(3).ignore_index(1).multidim_average(MD::samplewise);
    TM_CHECK(same(multiclass_accuracy(sp, st, multiclass_stat_scores_option(sw).average(AV::micro)),
                  torch::tensor({1.0f, 1.0f})));
    TM_CHECK(same(multiclass_stat_scores(sp, st, multiclass_stat_scores_option(sw).average(AV::none)),
                  torch::tensor({{{1, 0, 0, 0, 1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}},
                                 {{1, 0, 1, 0, 1}, {0, 0, 2, 0, 0}, {1, 0, 1, 0, 1}}})));
    // NPV: every sample predicted positive -> tn + fn = 0; the functional uses zero_division, torchmetrics' class 0
    const auto ones = torch::ones({4}), t4 = torch::tensor({1, 0, 1, 0});
    const auto zd1 = binary_stat_scores_option().zero_division(1);
    TM_CHECK(binary_negative_predictive_value(ones, t4, zd1).item<float>() == 1.0f);
    BinaryNegativePredictiveValue npv(zd1);
    npv.update(ones, t4);
    TM_CHECK(npv.compute().item<float>() == 0.0f);
    TM_CHECK(binary_precision(torch::zeros({4}), t4, zd1).item<float>() == 1.0f);
    // MCC: a perfect / inverted binary table is int64 +-1 like torchmetrics
    const auto mcc = binary_matthews_corrcoef(BT, BT);
    TM_CHECK(mcc.scalar_type() == torch::kLong && mcc.item<int64_t>() == 1);
    TM_CHECK(binary_matthews_corrcoef(1 - BT, BT).item<int64_t>() == -1);
    // ExactMatch: ignore_index positions count as right
    const auto ep = torch::tensor({{0, 1}, {2, 2}}), et = torch::tensor({{0, -1}, {2, 1}});
    TM_CHECK(multiclass_exact_match(ep, et, multiclass_stat_scores_option().num_classes(3).ignore_index(-1))
                     .item<float>() == 0.5f);
}

static void test_validation() {
    const auto inv = [](auto fn) { return throws_invalid(fn); };
    using BO = binary_stat_scores_option;
    using CO = multiclass_stat_scores_option;
    using LO = multilabel_stat_scores_option;
    // options
    TM_CHECK(inv([] { return binary_accuracy(BP, BT, BO().threshold(1.5)); }));
    TM_CHECK(inv([] { return binary_accuracy(BP, BT, BO().threshold(std::nan(""))); }));
    TM_CHECK(inv([] { return binary_precision(BP, BT, BO().zero_division(0.5)); }));
    TM_CHECK(inv([] { return binary_fbeta_score(BP, BT, BO().beta(0)); }));
    TM_CHECK(!throws_any([] { return binary_accuracy(BP, BT, BO().beta(0)); })); // beta: FBetaScore only
    TM_CHECK(inv([] { return multiclass_accuracy(MP, MT, CO()); })); // num_classes {} needs average micro
    TM_CHECK(inv([] { return multiclass_accuracy(MP, MT, CO().num_classes(1)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MP, MT, CO().num_classes(3).top_k(0)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MP, MT, CO().num_classes(3).top_k(4)); }));
    TM_CHECK(inv([] { return multiclass_recall(MP, MT, CO().num_classes(3).zero_division(2)); }));
    TM_CHECK(inv([] { return multiclass_fbeta_score(MP, MT, CO().num_classes(3).beta(-1)); }));
    const auto LP = torch::rand({4, 3}), LT = torch::randint(0, 2, {4, 3});
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT, LO().num_labels(1)); }));
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT, LO().num_labels(3).threshold(-0.1)); }));
    TM_CHECK(inv([&] { return multilabel_precision(LP, LT, LO().num_labels(3).zero_division(-1)); }));
    TM_CHECK(inv([&] { return multilabel_fbeta_score(LP, LT, LO().num_labels(3).beta(0)); }));
    // binary tensors
    TM_CHECK(inv([] { return binary_accuracy(BP, BT.slice(0, 1)); }));
    TM_CHECK(inv([] { return binary_accuracy(BP, BT + 1); }));
    TM_CHECK(!throws_any([] { return binary_accuracy(BP, BT.masked_fill(BT == 0, 7), BO().ignore_index(7)); }));
    TM_CHECK(inv([] { return binary_accuracy(BT * 2, BT); })); // label preds other than 0 / 1
    TM_CHECK(inv([] { return binary_accuracy(BP, BT, BO().multidim_average(MD::samplewise)); }));
    TM_CHECK(inv([] { return binary_accuracy(BP[0], BT[0]); })); // 0-d (torchmetrics: IndexError)
    // multiclass tensors
    const auto C3 = CO().num_classes(3);
    TM_CHECK(inv([] { return multiclass_accuracy(MP.to(torch::kLong), MT, CO().num_classes(3)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MP, MT, CO().num_classes(4)); }));
    TM_CHECK(inv([] {
        return multiclass_accuracy(MP.unsqueeze(-1), MT.unsqueeze(-1).repeat({1, 2}), CO().num_classes(3));
    }));
    TM_CHECK(inv([&] { return multiclass_accuracy(MP, MT, CO(C3).multidim_average(MD::samplewise)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MT, MT.slice(0, 1), CO().num_classes(3)); }));
    TM_CHECK(inv([&] { return multiclass_accuracy(MT, MT, CO(C3).multidim_average(MD::samplewise)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MP.unsqueeze(0), MT, CO().num_classes(3)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MT, torch::tensor({0, 1, 2, 3, 0, 1}), CO().num_classes(3)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(torch::tensor({0, 1, 2, 3, 0, 1}), MT, CO().num_classes(3)); }));
    TM_CHECK(inv([] { return multiclass_accuracy(MP[0], MT[0], CO().num_classes(3)); })); // 0-d target
    // multilabel tensors
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT.slice(0, 1), LO().num_labels(3)); }));
    TM_CHECK(inv([&] { return multilabel_accuracy(LP[0], LT[0], LO().num_labels(3)); }));
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT, LO().num_labels(2)); }));
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT + 1, LO().num_labels(3)); }));
    TM_CHECK(inv([&] { return multilabel_accuracy(LP, LT, LO().num_labels(3).multidim_average(MD::samplewise)); }));
    // ExactMatch / confusion-matrix family options
    TM_CHECK(inv([] { return multiclass_exact_match(MP, MT, CO()); }));
    TM_CHECK(inv([] { return multiclass_exact_match(MP, MT, CO().num_classes(1)); }));
    TM_CHECK(inv([&] { return multilabel_exact_match(LP, LT, LO().num_labels(1)); }));
    TM_CHECK(inv([&] { return multilabel_exact_match(LP, LT, LO().num_labels(3).threshold(2)); }));
    TM_CHECK(inv([] { return multiclass_exact_match(MP, MT.slice(0, 2), CO().num_classes(3)); }));
    TM_CHECK(inv([] { return binary_confusion_matrix(BP, BT, binary_confusion_matrix_option().threshold(-1)); }));
    TM_CHECK(inv([] { return binary_jaccard_index(BP, BT, binary_confusion_matrix_option().zero_division(3)); }));
    TM_CHECK(inv([] { return binary_confusion_matrix(BP, BT.slice(0, 2)); }));
    TM_CHECK(inv([] { return multiclass_confusion_matrix(MP, MT, multiclass_confusion_matrix_option()); }));
    TM_CHECK(inv([] {
        return multiclass_jaccard_index(MP, MT, multiclass_confusion_matrix_option().num_classes(3).zero_division(9));
    }));
    TM_CHECK(inv([&] { return multilabel_confusion_matrix(LP, LT, multilabel_confusion_matrix_option()); }));
    TM_CHECK(inv([&] {
        return multilabel_confusion_matrix(LP, LT, multilabel_confusion_matrix_option().num_labels(3).threshold(2));
    }));
    // torch's own errors stay c10::Error: out-of-range class labels reach one_hot / bincount
    TM_CHECK(throws_any([] {
        return multiclass_confusion_matrix(torch::tensor({0, 9}), torch::tensor({0, 1}),
                                           multiclass_confusion_matrix_option().num_classes(3));
    }));
}

// The class rejects invalid options (constructor) and inputs (update, state untouched).
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
    const auto LP = torch::rand({4, 3}), LT = torch::randint(0, 2, {4, 3});
    check_class_validation<BinaryAccuracy>(binary_stat_scores_option().threshold(2), {}, BP, BT, BT + 2);
    check_class_validation<MulticlassF1Score>(multiclass_stat_scores_option(),
                                              multiclass_stat_scores_option().num_classes(3), MP, MT, MT.slice(0, 1));
    check_class_validation<MultilabelPrecision>(multilabel_stat_scores_option(),
                                                multilabel_stat_scores_option().num_labels(3), LP, LT, LT + 2);
    check_class_validation<MulticlassExactMatch>(
            multiclass_stat_scores_option(), multiclass_stat_scores_option().num_classes(3), MP, MT, MT.slice(0, 1));
    check_class_validation<MultilabelExactMatch>(multilabel_stat_scores_option(),
                                                 multilabel_stat_scores_option().num_labels(3), LP, LT, LT + 2);
    check_class_validation<BinaryConfusionMatrix>(binary_confusion_matrix_option().threshold(2), {}, BP, BT, BT + 2);
    check_class_validation<MulticlassCohenKappa>(multiclass_confusion_matrix_option(),
                                                 multiclass_confusion_matrix_option().num_classes(3), MP, MT,
                                                 MT.slice(0, 1));
    check_class_validation<MultilabelJaccardIndex>(multilabel_confusion_matrix_option(),
                                                   multilabel_confusion_matrix_option().num_labels(3), LP, LT, LT + 2);
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
    const auto LP = torch::rand({6, 3, 2}), LT = torch::randint(0, 2, {6, 3, 2});
    check_forward<BinaryStatScores>(binary_stat_scores_option(), BP, BT);
    check_forward<BinaryAccuracy>(binary_stat_scores_option().multidim_average(MD::samplewise), BP.view({3, 2}),
                                  BT.view({3, 2}));
    check_forward<MulticlassRecall>(multiclass_stat_scores_option().num_classes(3).average(AV::weighted), MP, MT);
    check_forward<MultilabelHammingDistance>(
            multilabel_stat_scores_option().num_labels(3).multidim_average(MD::samplewise), LP, LT);
    check_forward<MulticlassExactMatch>(multiclass_stat_scores_option().num_classes(3), MP, MT);
    check_forward<MultilabelExactMatch>(multilabel_stat_scores_option().num_labels(3).multidim_average(MD::samplewise),
                                        LP, LT);
    check_forward<MulticlassMatthewsCorrCoef>(multiclass_confusion_matrix_option().num_classes(3), MP, MT);
    check_forward<MultilabelConfusionMatrix>(
            multilabel_confusion_matrix_option().num_labels(3).normalize(normalize_mode::true_), LP, LT);
    // before any update: torchmetrics' zero state (0 / 0 -> 0), or torch.cat of no batch (samplewise lists)
    TM_CHECK(BinaryAccuracy().compute().item<float>() == 0.0f);
    TM_CHECK(same(BinaryConfusionMatrix().compute(), torch::zeros({2, 2}, torch::kLong)));
    TM_CHECK(throws_any(
            [] { return BinaryAccuracy(binary_stat_scores_option().multidim_average(MD::samplewise)).compute(); }));
    // ExactMatch samplewise divides by the last batch's position count (torchmetrics' state)
    MultilabelExactMatch em(multilabel_stat_scores_option().num_labels(2).multidim_average(MD::samplewise));
    em.update(torch::ones({1, 2, 4}), torch::ones({1, 2, 4}, torch::kLong));
    em.update(torch::ones({1, 2, 2}), torch::ones({1, 2, 2}, torch::kLong));
    TM_CHECK(same(em.compute(), torch::tensor({2.0f, 1.0f})));
}

// torchmetrics' arange / eq bincount (deterministic mode, MPS) gives the same counts.
static void test_deterministic_bincount() {
    const auto o = multiclass_confusion_matrix_option().num_classes(3);
    const auto ref = multiclass_confusion_matrix(MP, MT, o);
    const bool was = at::globalContext().deterministicAlgorithms();
    at::globalContext().setDeterministicAlgorithms(true, false);
    const auto det = multiclass_confusion_matrix(MP, MT, o);
    at::globalContext().setDeterministicAlgorithms(was, false);
    TM_CHECK(same(det, ref));
}

// MCC's zero-denominator branches that only float32 rounding of huge counts reaches (2^25 true negatives); values
// from torchmetrics' _matthews_corrcoef_reduce on the same tables.
static void test_mcc_rounding() {
    const int64_t n = int64_t{1} << 25, batch = int64_t{1} << 22;
    const auto feed = [&](BinaryMatthewsCorrCoef &m, int64_t count, int pred, int target) {
        for (int64_t done = 0; done < count; done += batch) {
            const auto k = std::min(batch, count - done);
            m.update(torch::full({k}, pred, torch::kUInt8), torch::full({k}, target, torch::kUInt8));
        }
    };
    BinaryMatthewsCorrCoef tn_big, tp_big;
    feed(tn_big, n, 0, 0);
    feed(tp_big, n, 1, 1);
    auto tp0 = tn_big, fp0 = tn_big, all = tn_big, tn0 = tp_big;
    feed(tp0, 1, 1, 0); // fp 1, fn 1, tp 0: tn - fp fn
    feed(tp0, 1, 0, 1);
    feed(tn0, 1, 1, 0); // tn 0: tp - fp fn
    feed(tn0, 1, 0, 1);
    feed(fp0, 1, 0, 1); // fp 0: tp tn
    feed(fp0, 1, 1, 1);
    feed(all, 1, 1, 0); // all non-zero: 0
    feed(all, 1, 0, 1);
    feed(all, 1, 1, 1);
    TM_CHECK(tp0.compute().item<float>() == 0.999999881f);
    TM_CHECK(tn0.compute().item<float>() == 0.999999881f);
    TM_CHECK(fp0.compute().item<float>() == 0.707106709f);
    TM_CHECK(all.compute().item<float>() == 0.0f);
}

// MatthewsCorrCoef adds tk * pk with Python's sum() (sequential float32 adds): torch.sum's order changes this 20-class
// table's value in the last place (torchmetrics 0.211416379, with torch.sum 0.211416349).
static void test_mcc_py_sum() {
    const int64_t C = 20;
    const auto i = torch::arange(C).unsqueeze(1), j = torch::arange(C).unsqueeze(0);
    const auto counts = ((i + j + 7 * i * j) % 997 + (i == j) * 2000).flatten();
    const auto cell = torch::arange(C * C);
    const auto target = torch::repeat_interleave(torch::div(cell, C, "floor"), counts);
    const auto preds = torch::repeat_interleave(cell % C, counts);
    const auto o = multiclass_confusion_matrix_option().num_classes(C);
    TM_CHECK(multiclass_matthews_corrcoef(preds, target, o).item<float>() == 0.211416379f);
}

static void test_mps() {
    if (!torch::mps::is_available()) {
        return;
    }
    const auto dev = torch::Device(torch::kMPS);
    const auto logits = torch::logit(BP);
    TM_CHECK(same(binary_stat_scores(logits.to(dev), BT.to(dev)).cpu(), binary_stat_scores(logits, BT)));
    const auto o = multiclass_confusion_matrix_option().num_classes(3);
    TM_CHECK(
            same(multiclass_confusion_matrix(MP.to(dev), MT.to(dev), o).cpu(), multiclass_confusion_matrix(MP, MT, o)));
    MulticlassAccuracy m(multiclass_stat_scores_option().num_classes(3));
    m.update(MP.to(dev), MT.to(dev));
    const auto r = m.compute();
    TM_CHECK(r.device().is_mps() && r.cpu().item<float>() == multiclass_accuracy(MP, MT, m.options()).item<float>());
}

int main() {
    TM_RUN(test_goldens);
    TM_RUN(test_closed_form);
    TM_RUN(test_validation);
    TM_RUN(test_class_validation);
    TM_RUN(test_class_semantics);
    TM_RUN(test_deterministic_bincount);
    TM_RUN(test_mcc_rounding);
    TM_RUN(test_mcc_py_sum);
    TM_RUN(test_mps);
    return tm_test::summary("metric_test_classification_stat");
}
