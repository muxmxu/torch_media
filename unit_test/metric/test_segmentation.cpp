// Assertion-based tests for torchmedia::metric::segmentation against torchmetrics 1.9 goldens (gen_golden.py ->
// golden_segmentation.hpp; inputs from the shared splitmix64 generator, checked by checksum).
#include "golden_segmentation.hpp"
#include "metric_test_util.hpp"

using namespace torchmedia::metric::segmentation;
using IF = input_format_mode;

static const auto SEG = seg_pair(4, 5, 16, 16, 505);
static const auto SP = SEG.first, ST = SEG.second;
static const auto LOG = torch::tensor(splitmix_u(4 * 5 * 16 * 16, 606)).reshape({4, 5, 16, 16}).to(torch::kFloat32);
static const auto OH_P = torch::one_hot(SP, 5).movedim(-1, 1), OH_T = torch::one_hot(ST, 5).movedim(-1, 1);

static auto first(const torch::Tensor &x) -> torch::Tensor { return x.slice(0, 0, 2); }
static auto second(const torch::Tensor &x) -> torch::Tensor { return x.slice(0, 2); }

// equal values, NaN == NaN
static auto same(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && torch::allclose(a, b, 0.0, 0.0, true);
}

// forward() on an instance that already holds state returns this batch alone
template<class M, class O>
static auto forward_is_batch(M &m, const O &opt, const torch::Tensor &p, const torch::Tensor &t) -> bool {
    M alone(opt);
    alone.update(p, t);
    return same(m(p, t), alone.compute());
}

static void test_inputs() {
    CHECK_CHECKSUM(torch::stack({SP, ST}), in_seg);
    CHECK_CHECKSUM(LOG, in_logits);
}

static void test_mean_iou() {
    using namespace golden;
    const golden::entry *idx[2][2] = {{&miou_idx_all_bg, &miou_idx_all_nobg}, {&miou_idx_pc_bg, &miou_idx_pc_nobg}};
    const golden::entry *oh[2][2] = {{&miou_oh_all_bg, &miou_oh_all_nobg}, {&miou_oh_pc_bg, &miou_oh_pc_nobg}};
    const golden::entry *cls[2][2] = {{&miou_cls_all_bg, &miou_cls_all_nobg}, {&miou_cls_pc_bg, &miou_cls_pc_nobg}};
    const golden::entry *lazy[2][2] = {{&miou_cls_lazy_all_bg, &miou_cls_lazy_all_nobg},
                                       {&miou_cls_lazy_pc_bg, &miou_cls_lazy_pc_nobg}};
    for (const int pc: {0, 1}) {
        for (const int nobg: {0, 1}) {
            const auto o = mean_iou_option().per_class(pc).include_background(!nobg);
            const auto io = mean_iou_option(o).num_classes(5).input_format(IF::index);
            check_golden(mean_iou(SP, ST, io), *idx[pc][nobg], "miou_idx", __LINE__);
            check_golden(mean_iou(OH_P, OH_T, o), *oh[pc][nobg], "miou_oh", __LINE__);
            MeanIoU m(io);
            m(first(SP), first(ST));
            TM_CHECK(forward_is_batch(m, io, second(SP), second(ST)));
            check_golden(m.compute(), *cls[pc][nobg], "miou_cls", __LINE__);
            MeanIoU l(o); // num_classes from the first one-hot batch
            l.update(first(OH_P), first(OH_T));
            l.update(second(OH_P), second(OH_T));
            check_golden(l.compute(), *lazy[pc][nobg], "miou_cls_lazy", __LINE__);
        }
    }
    CHECK_GOLDEN(mean_iou(LOG, OH_T, mean_iou_option().num_classes(5)), miou_logits);
    const auto io = mean_iou_option().num_classes(5).input_format(IF::index);
    CHECK_GOLDEN(mean_iou(SP[0], ST[0], io), miou_2d);
    CHECK_GOLDEN(mean_iou(SP[0][0], ST[0][0], mean_iou_option(io).per_class(true)), miou_1d);
    // checks
    TM_CHECK(throws_invalid([] { return mean_iou(SP, ST, mean_iou_option().input_format(IF::index)); }));
    TM_CHECK(throws_invalid([] { return mean_iou(OH_P, OH_T, mean_iou_option().num_classes(0)); }));
    TM_CHECK(throws_invalid([] { return mean_iou(OH_P, OH_T.slice(2, 1)); }));
    TM_CHECK(throws_invalid([] { return mean_iou(OH_P.slice(1, 0, 0), OH_T.slice(1, 0, 0)); }));
    TM_CHECK(throws_invalid([] { return mean_iou(SP[0][0], ST[0][0]); })); // one-hot needs [N, C, ...]
    TM_CHECK(throws_invalid([] { return mean_iou(OH_P[0][0], OH_T[0][0]); })); // < 3 dims
    TM_CHECK(throws_invalid([] { return MeanIoU(mean_iou_option().input_format(IF::index)); }));
    TM_CHECK(throws_invalid([] { return MeanIoU(mean_iou_option().num_classes(-1)); }));
    MeanIoU m;
    TM_CHECK(throws_invalid([&] {
        m.update(OH_P, OH_T.slice(0, 1));
        return 0;
    }));
    TM_CHECK(throws_any([] {
        return mean_iou(SP.to(torch::kInt32), ST.to(torch::kInt32),
                        mean_iou_option().num_classes(5).input_format(IF::index));
    }));
    // before any update and after reset: torchmetrics' zero state (-1 per class, NaN overall)
    TM_CHECK(same(MeanIoU(mean_iou_option().num_classes(3).per_class(true)).compute(), torch::full({3}, -1.0)));
    TM_CHECK(same(MeanIoU(mean_iou_option(io).include_background(false).per_class(true)).compute(),
                  torch::full({4}, -1.0)));
    TM_CHECK(std::isnan(MeanIoU().compute().item<double>()));
    TM_CHECK(MeanIoU(mean_iou_option().per_class(true)).compute().item<double>() == -1.0);
    MeanIoU r(io);
    r.update(SP, ST);
    r.reset();
    TM_CHECK(std::isnan(r.compute().item<double>()));
    // one class without background: the class keeps that channel like both functionals (torchmetrics' class sizes
    // its state C - 1 = 0 and returns NaN / an empty tensor; progress11 D7)
    const auto z = torch::zeros({2, 4, 4}, torch::kInt64);
    const auto one = mean_iou_option().num_classes(1).include_background(false).input_format(IF::index);
    MeanIoU c1(one);
    c1.update(z, z);
    TM_CHECK(c1.compute().item<double>() == 1.0);
    TM_CHECK(same(mean_iou(z, z, one), torch::ones({2})));
    TM_CHECK(MeanIoU(mean_iou_option(one).per_class(true)).compute().numel() == 0); // torchmetrics' empty state
}

static void test_dice() {
    using namespace golden;
    using DA = dice_average_mode;
    using AL = aggregation_level_mode;
    const golden::entry *fn[4][2] = {{&dice_micro_samplewise, &dice_micro_global},
                                     {&dice_macro_samplewise, &dice_macro_global},
                                     {&dice_weighted_samplewise, &dice_weighted_global},
                                     {&dice_none_samplewise, &dice_none_global}};
    const golden::entry *cls[4][2] = {{&dice_cls_micro_samplewise, &dice_cls_micro_global},
                                      {&dice_cls_macro_samplewise, &dice_cls_macro_global},
                                      {&dice_cls_weighted_samplewise, &dice_cls_weighted_global},
                                      {&dice_cls_none_samplewise, &dice_cls_none_global}};
    const DA avgs[] = {DA::micro, DA::macro, DA::weighted, DA::none};
    const AL aggs[] = {AL::samplewise, AL::global};
    for (int a = 0; a < 4; ++a) {
        for (int g = 0; g < 2; ++g) {
            const auto o = dice_score_option().num_classes(5).average(avgs[a]).aggregation_level(aggs[g]).input_format(
                    IF::index);
            check_golden(dice_score(SP, ST, o), *fn[a][g], "dice", __LINE__);
            DiceScore m(o);
            m(first(SP), first(ST));
            TM_CHECK(forward_is_batch(m, o, second(SP), second(ST)));
            check_golden(m.compute(), *cls[a][g], "dice_cls", __LINE__);
        }
    }
    CHECK_GOLDEN(dice_score(LOG, OH_T, dice_score_option().num_classes(5).include_background(false)), dice_nobg_logits);
    const auto z = torch::zeros({2, 3, 4, 4}, torch::kInt64);
    CHECK_GOLDEN(dice_score(z, z, dice_score_option().num_classes(3).average(DA::weighted)), dice_weighted_empty);
    TM_CHECK(throws_invalid([] { return dice_score(SP, ST, dice_score_option().input_format(IF::index)); }));
    TM_CHECK(throws_invalid([] { return dice_score(OH_P, OH_T.slice(0, 1), dice_score_option().num_classes(5)); }));
    TM_CHECK(throws_invalid([] { return DiceScore(dice_score_option()); }));
    DiceScore m(dice_score_option().num_classes(5));
    TM_CHECK(throws_invalid([&] {
        m.update(OH_P, OH_T.slice(0, 1));
        return 0;
    }));
}

static void test_generalized_dice() {
    using namespace golden;
    using WT = weight_type_mode;
    const golden::entry *fn[3][2] = {{&gdice_square_all, &gdice_square_pc},
                                     {&gdice_simple_all, &gdice_simple_pc},
                                     {&gdice_linear_all, &gdice_linear_pc}};
    const golden::entry *cls[3][2] = {{&gdice_cls_square_all, &gdice_cls_square_pc},
                                      {&gdice_cls_simple_all, &gdice_cls_simple_pc},
                                      {&gdice_cls_linear_all, &gdice_cls_linear_pc}};
    const WT wts[] = {WT::square, WT::simple, WT::linear};
    for (int w = 0; w < 3; ++w) {
        for (const int pc: {0, 1}) {
            const auto o =
                    generalized_dice_score_option().num_classes(5).weight_type(wts[w]).per_class(pc).input_format(
                            IF::index);
            check_golden(generalized_dice_score(SP, ST, o), *fn[w][pc], "gdice", __LINE__);
            GeneralizedDiceScore m(o);
            m(first(SP), first(ST));
            TM_CHECK(forward_is_batch(m, o, second(SP), second(ST)));
            check_golden(m.compute(), *cls[w][pc], "gdice_cls", __LINE__);
        }
    }
    CHECK_GOLDEN(generalized_dice_score(OH_P, OH_T,
                                        generalized_dice_score_option().num_classes(5).include_background(false)),
                 gdice_nobg_oh);
    CHECK_GOLDEN(generalized_dice_score(LOG, OH_T, generalized_dice_score_option().num_classes(5).per_class(true)),
                 gdice_logits);
    // predictions in classes absent from the sample's target, N = 4: the transposed max-weight lookup matters
    const auto g = torch::meshgrid({torch::arange(4), torch::arange(6), torch::arange(6)}, "ij");
    const auto gp = torch::remainder(7 * g[0] + 3 * g[1] + g[2], 5);
    const auto gt = torch::remainder(torch::remainder(g[1] * g[1] + g[2] + g[0], 9),
                                     torch::tensor({5, 3, 4, 2}).view({4, 1, 1}));
    const auto go = generalized_dice_score_option().num_classes(5).input_format(IF::index);
    CHECK_GOLDEN(generalized_dice_score(gp, gt, go), gdice_transposed_square);
    CHECK_GOLDEN(generalized_dice_score(gp, gt, generalized_dice_score_option(go).weight_type(WT::simple)),
                 gdice_transposed_simple);
    TM_CHECK(throws_invalid([] { return generalized_dice_score(OH_P, OH_T, generalized_dice_score_option()); }));
    TM_CHECK(throws_invalid([] {
        return generalized_dice_score(OH_P, OH_T.slice(0, 1), generalized_dice_score_option().num_classes(5));
    }));
    TM_CHECK(throws_invalid([] { return GeneralizedDiceScore(generalized_dice_score_option().num_classes(-2)); }));
    GeneralizedDiceScore m(generalized_dice_score_option().num_classes(5));
    TM_CHECK(throws_invalid([&] {
        m.update(OH_P, OH_T.slice(0, 1));
        return 0;
    }));
    TM_CHECK(std::isnan(m.compute().item<double>())); // nothing accumulated: 0 / 0
}

int main() {
    TM_RUN(test_inputs);
    TM_RUN(test_mean_iou);
    TM_RUN(test_dice);
    TM_RUN(test_generalized_dice);
    return tm_test::summary("metric_test_segmentation");
}
