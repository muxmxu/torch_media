// Assertion-based tests for torchmedia::metric::detection against torchvision.ops, torchmetrics and pycocotools
// COCOeval goldens (gen_golden.py -> golden_detection.hpp), and an independent numpy VOC devkit implementation.
#include "golden_detection.hpp"
#include "metric_test_util.hpp"

using namespace torchmedia::metric::detection;
using BF = box_format_mode;

struct dataset {
    std::vector<detection_prediction> preds;
    std::vector<detection_target> target;
};

// Per-image predictions / targets from the flattened golden tensors and per-image counts.
template<class E>
static auto load_dataset(const E &pb, const E &ps, const E &pl, const E &pc, const E &gb, const E &gl, const E &gcr,
                         const E &ga, const E &gc, bool crowd, bool area) -> dataset {
    const auto PB = G(pb).to(torch::kFloat32).reshape({-1, 4}), PS = G(ps).to(torch::kFloat32),
               PL = G(pl).to(torch::kInt64), GB = G(gb).to(torch::kFloat32).reshape({-1, 4}),
               GL = G(gl).to(torch::kInt64), GCR = G(gcr).to(torch::kInt64), GA = G(ga).to(torch::kFloat32);
    dataset d;
    int64_t po = 0, go = 0;
    for (size_t i = 0; i < pc.data.size(); ++i) {
        const auto np = static_cast<int64_t>(pc.data[i]), ng = static_cast<int64_t>(gc.data[i]);
        d.preds.push_back({PB.slice(0, po, po + np), PS.slice(0, po, po + np), PL.slice(0, po, po + np), {}});
        d.target.push_back({GB.slice(0, go, go + ng),
                            GL.slice(0, go, go + ng),
                            crowd ? GCR.slice(0, go, go + ng) : torch::Tensor{},
                            area ? GA.slice(0, go, go + ng) : torch::Tensor{},
                            {}});
        po += np;
        go += ng;
    }
    return d;
}
#define LOAD(n, crowd, area)                                                                                           \
    load_dataset(golden::n##_pred_boxes, golden::n##_pred_scores, golden::n##_pred_labels, golden::n##_pred_counts,    \
                 golden::n##_gt_boxes, golden::n##_gt_labels, golden::n##_gt_crowd, golden::n##_gt_area,               \
                 golden::n##_gt_counts, crowd, area)

static const dataset DA = LOAD(da, true, true);

static auto slice(const dataset &d, size_t lo, size_t hi) -> dataset {
    return {{d.preds.begin() + lo, d.preds.begin() + hi}, {d.target.begin() + lo, d.target.begin() + hi}};
}

static auto converted(const dataset &d, BF fmt) -> dataset {
    auto out = d;
    for (auto &p: out.preds) {
        p.boxes = box_convert(p.boxes, BF::xyxy, fmt);
    }
    for (auto &t: out.target) {
        t.boxes = box_convert(t.boxes, BF::xyxy, fmt);
    }
    return out;
}

static auto stats(const mean_average_precision_result &r) -> torch::Tensor {
    return torch::tensor({r.map, r.map_50, r.map_75, r.map_small, r.map_medium, r.map_large, r.mar_1, r.mar_10,
                          r.mar_100, r.mar_small, r.mar_medium, r.mar_large},
                         torch::kFloat64);
}

static double g_map_max_delta = 0.0;

// The 12 stats exactly (pycocotools float64), and the classes.
template<class E>
static void check_map(const mean_average_precision_result &r, const E &want, const E &classes, const char *name,
                      int line) {
    const torch::Tensor got = stats(r), exp = G(want);
    g_map_max_delta = std::max(g_map_max_delta, (got - exp).abs().max().item<double>());
    tm_test::record(torch::equal(got, exp), name, __FILE__, line,
                    "max|delta|=" + std::to_string((got - exp).abs().max().item<double>()));
    tm_test::record(r.classes.scalar_type() == torch::kInt32 &&
                            torch::equal(r.classes.to(torch::kInt64), G(classes).to(torch::kInt64)),
                    name, __FILE__, line, "classes");
}
#define CHECK_MAP(r, n) check_map((r), golden::n, golden::n##_classes, #n, __LINE__)
#define CHECK_MAP_PC(r, n)                                                                                             \
    do {                                                                                                               \
        const auto map_res_ = (r);                                                                                     \
        CHECK_MAP(map_res_, n);                                                                                        \
        check_golden(map_res_.map_per_class, golden::n##_map_pc, #n "_map_pc", __LINE__, 0.0);                         \
        check_golden(map_res_.mar_100_per_class, golden::n##_mar_pc, #n "_mar_pc", __LINE__, 0.0);                     \
    } while (0)

// ---------------- torchvision.ops ----------------
static void test_box_ops() {
    using namespace golden;
    const auto b1 = G(box1_f32).to(torch::kFloat32), b2 = G(box2_f32).to(torch::kFloat32);
    const auto d1 = G(box1_f64), d2 = G(box2_f64);
    const auto i1 = G(box1_i64).to(torch::kInt64), i2 = G(box2_i64).to(torch::kInt64);
    CHECK_GOLDEN(box_area(b1), box_area_f32);
    CHECK_GOLDEN(box_iou(b1, b2), box_iou_f32);
    CHECK_GOLDEN(generalized_box_iou(b1, b2), giou_f32);
    CHECK_GOLDEN(distance_box_iou(b1, b2), diou_f32);
    CHECK_GOLDEN(complete_box_iou(b1, b2), ciou_f32);
    CHECK_GOLDEN(distance_box_iou(b1, b2, 1e-3), diou_eps_f32);
    CHECK_GOLDEN(complete_box_iou(b1, b2, 1e-3), ciou_eps_f32);
    CHECK_GOLDEN(box_area(d1), box_area_f64);
    CHECK_GOLDEN(box_iou(d1, d2), box_iou_f64);
    CHECK_GOLDEN(generalized_box_iou(d1, d2), giou_f64);
    CHECK_GOLDEN(distance_box_iou(d1, d2), diou_f64);
    CHECK_GOLDEN(complete_box_iou(d1, d2), ciou_f64);
    // integer boxes: areas stay integer (int16 upcast to int32), IoUs are float32
    CHECK_GOLDEN(box_area(i1), box_area_i64);
    CHECK_GOLDEN(box_iou(i1, i2), box_iou_i64);
    CHECK_GOLDEN(generalized_box_iou(i1, i2), giou_i64);
    CHECK_GOLDEN(distance_box_iou(i1, i2), diou_i64);
    CHECK_GOLDEN(complete_box_iou(i1, i2), ciou_i64);
    const auto s1 = i1.to(torch::kInt16), s2 = i2.to(torch::kInt16);
    CHECK_GOLDEN(box_area(s1), box_area_i16);
    CHECK_GOLDEN(box_iou(s1, s2), box_iou_i16);
    CHECK_GOLDEN(generalized_box_iou(s1, s2), giou_i16);
    CHECK_GOLDEN(distance_box_iou(s1, s2), diou_i16);
    CHECK_GOLDEN(complete_box_iou(s1, s2), ciou_i16);
    // float16 boxes compute in float32 (torchvision _upcast)
    const auto h1 = b1.to(torch::kFloat16), h2 = b2.to(torch::kFloat16);
    CHECK_GOLDEN(box_area(h1), box_area_f16);
    CHECK_GOLDEN(box_iou(h1, h2), box_iou_f16);
    CHECK_GOLDEN(generalized_box_iou(h1, h2), giou_f16);
    CHECK_GOLDEN(distance_box_iou(h1, h2), diou_f16);
    CHECK_GOLDEN(complete_box_iou(h1, h2), ciou_f16);
}

static void test_box_convert() {
    using namespace golden;
    const auto b1 = G(box1_f32).to(torch::kFloat32);
    const BF fmts[] = {BF::xyxy, BF::xywh, BF::cxcywh};
    const golden::entry *want[3][3] = {{&convert_xyxy_xyxy, &convert_xyxy_xywh, &convert_xyxy_cxcywh},
                                       {&convert_xywh_xyxy, &convert_xywh_xywh, &convert_xywh_cxcywh},
                                       {&convert_cxcywh_xyxy, &convert_cxcywh_xywh, &convert_cxcywh_cxcywh}};
    for (int i = 0; i < 3; ++i) {
        for (int o = 0; o < 3; ++o) {
            check_golden(box_convert(b1, fmts[i], fmts[o]), *want[i][o], "convert", __LINE__);
        }
    }
    const auto i1 = G(box1_i64).to(torch::kInt64);
    CHECK_GOLDEN(box_convert(i1, BF::cxcywh, BF::xyxy), convert_cxcywh_xyxy_i64);
    CHECK_GOLDEN(box_convert(i1, BF::xyxy, BF::cxcywh), convert_xyxy_cxcywh_i64);
    // same format: a copy (torchvision clone), any shape
    const auto c = box_convert(b1, BF::xywh, BF::xywh);
    TM_CHECK(torch::equal(c, b1) && c.data_ptr() != b1.data_ptr());
    TM_CHECK(box_convert(torch::zeros({3}), BF::xyxy, BF::xyxy).numel() == 3);
    TM_CHECK(throws_invalid([&] { return box_convert(torch::zeros({3, 5}), BF::xyxy, BF::xywh); }));
    TM_CHECK(throws_invalid([&] { return box_convert(torch::tensor(1.0), BF::xywh, BF::xyxy); }));
    // batch dims are fine
    TM_CHECK(box_convert(b1.reshape({7, 1, 4}), BF::xyxy, BF::cxcywh).sizes() == torch::IntArrayRef({7, 1, 4}));
}

// ---------------- torchmetrics functional IoU ----------------
static void test_iou_functional() {
    using namespace golden;
    const auto p = G(box1_f32).to(torch::kFloat32).slice(0, 0, 5), t = G(box2_f32).to(torch::kFloat32);
    using fn_t = torch::Tensor (*)(const torch::Tensor &, const torch::Tensor &, std::optional<double>, double, bool);
    const fn_t fns[] = {intersection_over_union, generalized_intersection_over_union, distance_intersection_over_union,
                        complete_intersection_over_union};
    const golden::entry *want[4][4] = {{&tm_iou_agg, &tm_iou_mat, &tm_iou_thr, &tm_iou_thr_agg},
                                       {&tm_giou_agg, &tm_giou_mat, &tm_giou_thr, &tm_giou_thr_agg},
                                       {&tm_diou_agg, &tm_diou_mat, &tm_diou_thr, &tm_diou_thr_agg},
                                       {&tm_ciou_agg, &tm_ciou_mat, &tm_ciou_thr, &tm_ciou_thr_agg}};
    for (int k = 0; k < 4; ++k) {
        check_golden(fns[k](p, t, {}, 0, true), *want[k][0], "iou_agg", __LINE__);
        check_golden(fns[k](p, t, {}, 0, false), *want[k][1], "iou_mat", __LINE__);
        check_golden(fns[k](p, t, 0.05, -2.0, false), *want[k][2], "iou_thr", __LINE__);
        check_golden(fns[k](p, t, 0.05, 0, true), *want[k][3], "iou_thr_agg", __LINE__);
    }
    // defaults
    CHECK_GOLDEN(intersection_over_union(p, t), tm_iou_agg);
    CHECK_GOLDEN(generalized_intersection_over_union(p, t), tm_giou_agg);
    CHECK_GOLDEN(distance_intersection_over_union(p, t), tm_diou_agg);
    CHECK_GOLDEN(complete_intersection_over_union(p, t), tm_ciou_agg);
    // an empty side: torchmetrics' square float32 zero matrix of the other side
    CHECK_GOLDEN(intersection_over_union(torch::zeros({0, 4}), t, {}, 0, false), tm_iou_empty_preds);
    CHECK_GOLDEN(intersection_over_union(p, torch::zeros({0, 4}), {}, 0, false), tm_iou_empty_target);
    CHECK_GOLDEN(intersection_over_union(torch::zeros({0, 4}), torch::zeros({0, 4})), tm_iou_empty_both_agg);
    CHECK_GOLDEN(intersection_over_union(p.to(torch::kFloat64), t.to(torch::kFloat64)), tm_iou_f64_agg);
    // shape checks (torchmetrics ValueError)
    TM_CHECK(throws_invalid([&] { return intersection_over_union(p.slice(1, 0, 3), t); }));
    TM_CHECK(throws_invalid([&] { return intersection_over_union(p, t.flatten()); }));
    TM_CHECK(throws_invalid([&] { return complete_intersection_over_union(torch::zeros({4}), t); }));
}

// ---------------- torchmetrics IoU classes ----------------
static auto iou_items(size_t n) -> dataset {
    dataset d;
    for (const auto i: golden::iou_sel.data) {
        if (d.preds.size() == n) {
            break;
        }
        const auto &p = DA.preds[static_cast<size_t>(i)];
        d.preds.push_back({p.boxes, {}, p.labels, {}});
        d.target.push_back(
                {.boxes = DA.target[static_cast<size_t>(i)].boxes, .labels = DA.target[static_cast<size_t>(i)].labels});
    }
    return d;
}

template<class E>
static void check_dict(const std::map<std::string, torch::Tensor> &got, const E &want, const E &keys,
                       const std::string &prefix, const char *name, int line) {
    std::vector<std::string> names;
    for (const auto k: keys.data) {
        names.push_back(k < 0 ? prefix : prefix + "/cl_" + std::to_string(static_cast<int64_t>(k)));
    }
    std::sort(names.begin(), names.end());
    bool same_keys = got.size() == names.size();
    std::vector<torch::Tensor> vals;
    for (const auto &n: names) {
        const auto it = got.find(n);
        same_keys = same_keys && it != got.end();
        vals.push_back(it == got.end() ? torch::tensor(-99.0) : it->second.reshape({}).to(torch::kFloat64));
    }
    tm_test::record(same_keys, name, __FILE__, line, "keys differ");
    check_golden(torch::stack(vals), want, name, line, kTol, false);
}
#define CHECK_DICT(got, n, prefix) check_dict((got), golden::n, golden::n##_keys, (prefix), #n, __LINE__)

static void test_iou_classes() {
    const auto all = iou_items(1000);
    const auto first = slice(all, 0, 10), rest = slice(all, 10, all.preds.size());
    const auto opt = intersection_over_union_option().class_metrics(true);
    {
        IntersectionOverUnion m(opt);
        m.update(first.preds, first.target);
        m.update(rest.preds, rest.target);
        CHECK_DICT(m.compute(), cls_iou, "iou");
    }
    {
        GeneralizedIntersectionOverUnion m(opt);
        // forward(): this batch's value, accumulated as well
        const auto same_dict = [](const auto &got, const auto &want) {
            bool same = got.size() == want.size();
            for (const auto &[k, v]: got) {
                same = same && want.count(k) && torch::equal(v, want.at(k));
            }
            return same;
        };
        const auto batch = m.forward(first.preds, first.target);
        GeneralizedIntersectionOverUnion only_first(opt), only_rest(opt);
        only_first.update(first.preds, first.target);
        TM_CHECK(same_dict(batch, only_first.compute()));
        // on an instance that already holds state: still this batch alone
        const auto rest_batch = m(rest.preds, rest.target);
        only_rest.update(rest.preds, rest.target);
        TM_CHECK(same_dict(rest_batch, only_rest.compute()));
        CHECK_DICT(m.compute(), cls_giou, "giou");
    }
    {
        DistanceIntersectionOverUnion m(opt);
        m.update(all.preds, all.target);
        CHECK_DICT(m.compute(), cls_diou, "diou");
        m.reset();
        TM_CHECK(m.compute().at("diou").item<double>() == 0.0);
    }
    {
        CompleteIntersectionOverUnion m(opt);
        m.update(all.preds, all.target);
        CHECK_DICT(m.compute(), cls_ciou, "ciou");
    }
    {
        IntersectionOverUnion m(
                intersection_over_union_option().respect_labels(false).iou_threshold(0.3).class_metrics(true));
        m.update(all.preds, all.target);
        CHECK_DICT(m.compute(), cls_iou_norespect_thr, "iou");
    }
    {
        const auto x = converted(all, BF::xywh);
        IntersectionOverUnion m(intersection_over_union_option().box_format(BF::xywh));
        m.update(x.preds, x.target);
        CHECK_DICT(m.compute(), cls_iou_xywh, "iou");
    }
    // predictions without targets: the zero diagonal counts globally; per class the image is skipped (torchmetrics
    // raises an IndexError there)
    auto d = slice(all, 0, 6);
    d.target[5] = {.boxes = torch::zeros({0, 4}), .labels = torch::zeros({0}, torch::kInt64)};
    {
        IntersectionOverUnion m;
        m.update(d.preds, d.target);
        CHECK_DICT(m.compute(), cls_iou_empty_target, "iou");
        IntersectionOverUnion pc(opt);
        pc.update(d.preds, d.target);
        auto res = pc.compute();
        res.erase("iou");
        auto want = slice(all, 0, 5);
        IntersectionOverUnion ref(opt);
        ref.update(want.preds, want.target);
        auto ref_res = ref.compute();
        CHECK_DICT(ref_res, cls_iou_empty_target_pc, "iou");
        ref_res.erase("iou");
        bool same = res.size() == ref_res.size();
        for (const auto &[k, v]: res) {
            same = same && ref_res.count(k) && torch::equal(v, ref_res.at(k));
        }
        TM_CHECK(same);
    }
    {
        auto e = slice(all, 0, 3);
        e.preds.insert(e.preds.begin(), {torch::zeros({0, 4}), {}, torch::zeros({0}, torch::kInt64), {}});
        e.target.insert(e.target.begin(), all.target[0]);
        IntersectionOverUnion m;
        m.update(e.preds, e.target);
        CHECK_DICT(m.compute(), cls_iou_empty_pred, "iou");
    }
    CHECK_DICT(IntersectionOverUnion().compute(), cls_iou_none, "iou");
    {
        // zero-area boxes: IoU 0 / 0 = NaN, reported as 0 (torchmetrics' isnan branch; checked with torchmetrics)
        IntersectionOverUnion z(opt);
        const std::vector<detection_prediction> zp{{torch::zeros({1, 4}), {}, torch::tensor({1}), {}}};
        const std::vector<detection_target> zt{{.boxes = torch::zeros({1, 4}), .labels = torch::tensor({1})}};
        z.update(zp, zt);
        const auto r = z.compute();
        TM_CHECK(r.at("iou").item<double>() == 0.0 && std::isnan(r.at("iou/cl_1").item<double>()));
    }
    TM_CHECK(IntersectionOverUnion(opt).compute().size() == 1); // class_metrics before any update: no classes
    // validation (torchmetrics _input_validator / ValueError)
    IntersectionOverUnion m;
    const auto p1 = slice(all, 0, 1);
    TM_CHECK(throws_invalid([&] {
        m.update(p1.preds, all.target);
        return 0;
    }));
    auto bad = p1;
    bad.preds[0].labels = torch::Tensor();
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.target[0].boxes = torch::Tensor();
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.target[0].labels = bad.target[0].labels.slice(0, 1);
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.preds[0].boxes = bad.preds[0].boxes.slice(1, 0, 3);
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.preds[0] = {torch::zeros({0}), {}, torch::zeros({0}, torch::kInt64), {}}; // 1-D empty boxes: rejected
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.preds[0].labels = bad.preds[0].labels.to(torch::kFloat32);
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    bad = p1;
    bad.target[0].labels = bad.target[0].labels.unsqueeze(1);
    TM_CHECK(throws_invalid([&] {
        m.update(bad.preds, bad.target);
        return 0;
    }));
    TM_CHECK(m.compute().at("iou").item<double>() == 0.0); // nothing was accumulated
}

// ---------------- MeanAveragePrecision ----------------
static void test_map_bbox() {
    const auto cm = mean_average_precision_option().class_metrics(true);
    CHECK_MAP_PC(mean_average_precision(DA.preds, DA.target, cm), map_da);
    CHECK_MAP_PC(
            mean_average_precision(DA.preds, DA.target, mean_average_precision_option(cm).average(average_mode::micro)),
            map_da_micro);
    // updates in batches == one shot; forward() returns the batch's own value
    {
        MeanAveragePrecision m(cm);
        const auto a = slice(DA, 0, 40), b = slice(DA, 40, 77), c = slice(DA, 77, 120);
        CHECK_MAP_PC(m.forward(a.preds, a.target), map_da40);
        const auto rb = m(b.preds, b.target), want_b = mean_average_precision(b.preds, b.target, cm);
        TM_CHECK(torch::equal(stats(rb), stats(want_b)) && torch::equal(rb.map_per_class, want_b.map_per_class));
        m.update(c.preds, c.target);
        CHECK_MAP_PC(m.compute(), map_da);
        m.reset();
        m.update(a.preds, a.target);
        CHECK_MAP_PC(m.compute(), map_da40);
    }
    CHECK_MAP(mean_average_precision(DA.preds, DA.target,
                                     mean_average_precision_option()
                                             .iou_thresholds({0.5, 0.6, 0.75, 0.9})
                                             .rec_thresholds({0.0, 0.25, 0.5, 0.8, 1.0})
                                             .max_detection_thresholds({100, 2, 10})),
              map_da_params);
    {
        // no 100 among maxDets: COCOeval's map (literal maxDets = 100) is -1
        const auto r = mean_average_precision(DA.preds, DA.target,
                                              mean_average_precision_option(cm).max_detection_thresholds({1, 5, 50}));
        CHECK_MAP_PC(r, map_da_maxdet);
    }
    CHECK_MAP(mean_average_precision(
                      DA.preds, DA.target,
                      mean_average_precision_option().rec_thresholds({0.9, 0.1, 0.5}).iou_thresholds({0.55})),
              map_da_unsorted_rec);
    {
        std::vector<double> iou, rec;
        for (int i = 0; i < 10; ++i) { // np.linspace(.5, 0.95, 10)
            iou.push_back(torch::linspace(0.5, 0.95, 10, torch::kFloat64)[i].item<double>());
        }
        for (int i = 0; i <= 100; ++i) {
            rec.push_back(torch::linspace(0.0, 1.0, 101, torch::kFloat64)[i].item<double>());
        }
        // pycocotools' float64 defaults (np.linspace) as explicit thresholds
        const auto r = mean_average_precision(DA.preds, DA.target,
                                              mean_average_precision_option().iou_thresholds(iou).rec_thresholds(rec));
        TM_CHECK(
                torch::equal(torch::tensor(iou), torch::tensor(std::vector<double>{0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                                                                   0.85, 0.8999999999999999, 0.95})));
        CHECK_MAP(r, map_da_numpy_thr);
    }
    for (const auto fmt: {BF::xywh, BF::cxcywh}) {
        const auto x = converted(DA, fmt);
        const auto r = mean_average_precision(x.preds, x.target, mean_average_precision_option().box_format(fmt));
        if (fmt == BF::xywh) {
            CHECK_MAP(r, map_da_xywh);
        } else {
            CHECK_MAP(r, map_da_cxcywh);
        }
    }
    // the defaults are torchmetrics' float32 linspace values
    const MeanAveragePrecision m;
    TM_CHECK(m.options()._iou_thresholds.size() == 10 && m.options()._iou_thresholds[1] == 0.550000011920929);
    TM_CHECK(m.options()._rec_thresholds.size() == 101 && m.options()._rec_thresholds[1] == 0.009999999776482582);
    TM_CHECK((m.options()._max_detection_thresholds == std::vector<int64_t>{1, 10, 100}));
}

static void test_map_edges() {
    // more than 100 detections per image: truncated per image and class
    const auto db = LOAD(db, true, false);
    CHECK_MAP(mean_average_precision(db.preds, db.target), map_db);
    CHECK_MAP(mean_average_precision(db.preds, db.target,
                                     mean_average_precision_option().max_detection_thresholds({120, 5, 50})),
              map_db_maxdet);
    const auto cm = mean_average_precision_option().class_metrics(true);
    const auto dc1 = LOAD(dc1, false, false); // no predictions
    CHECK_MAP_PC(mean_average_precision(dc1.preds, dc1.target, cm), map_dc1);
    const auto dc2 = LOAD(dc2, false, false); // no targets
    CHECK_MAP_PC(mean_average_precision(dc2.preds, dc2.target, cm), map_dc2);
    const auto dc3 = LOAD(dc3, true, false); // only crowds
    CHECK_MAP(mean_average_precision(dc3.preds, dc3.target), map_dc3);
    // no image at all: -1 everywhere, no classes
    const auto r = MeanAveragePrecision(cm).compute();
    TM_CHECK(r.map == -1 && r.mar_large == -1 && r.classes.numel() == 0 && r.classes.scalar_type() == torch::kInt32);
    TM_CHECK(r.map_per_class.numel() == 1 && r.map_per_class.item<double>() == -1);
    // 1-D empty boxes (torchmetrics _fix_empty_tensors) and integer / bool label and crowd dtypes are fine
    std::vector<detection_prediction> p{{torch::zeros({0}), torch::zeros({0}), torch::zeros({0}, torch::kInt32), {}}};
    std::vector<detection_target> t{{torch::tensor({{0.0, 0.0, 10.0, 10.0}}),
                                     torch::tensor({1}, torch::kInt32),
                                     torch::tensor({false}),
                                     {},
                                     {}}};
    const auto e = mean_average_precision(p, t);
    TM_CHECK(e.map == 0 && e.mar_100 == 0 && e.map_medium == -1);
    // a perfect detection: COCOeval's precision tp / (fp + tp + 2^-52) is 1 / (1 + 2^-52) for one detection
    p[0] = {torch::tensor({{0.0, 0.0, 10.0, 10.0}}), torch::tensor({0.9}), torch::tensor({1}, torch::kInt32), {}};
    const auto perfect = mean_average_precision(p, t);
    TM_CHECK(perfect.map < 1.0 && perfect.map > 1.0 - 1e-15 && perfect.mar_100 == 1.0);
}

// 12 classes (numpy's 8192-element summation chunks), IoU exactly at the threshold, areas exactly 32^2 / 96^2
static void test_map_boundaries() {
    const auto de = LOAD(de, true, true);
    CHECK_MAP_PC(mean_average_precision(de.preds, de.target, mean_average_precision_option().class_metrics(true)),
                 map_de);
    // float64 boxes with IoU 1 - 2.5e-11 (area 5e9): a match at threshold 1.0 through min(t, 1 - 1e-10)
    const auto f64 = [](std::vector<double> b) { return torch::tensor(b, torch::kFloat64).reshape({-1, 4}); };
    const auto label0 = torch::tensor({0});
    const std::vector<detection_prediction> fp{
            {f64({0.0, 0.0, 2e10 - 0.5, 0.25}), torch::tensor({0.9}, torch::kFloat32), label0, {}},
            {f64({12.0, 10.0, 50.0, 62.0}), torch::tensor({0.6}, torch::kFloat32), label0, {}}};
    const std::vector<detection_target> ft{{.boxes = f64({0.0, 0.0, 2e10, 0.25}), .labels = label0},
                                           {.boxes = f64({10.0, 10.0, 50.0, 60.0}), .labels = label0}};
    CHECK_MAP(mean_average_precision(fp, ft, mean_average_precision_option().iou_thresholds({0.5, 1.0})), map_df);
}

// segm: masks from the golden's rectangle-minus-hole parameters
static auto make_mask(const torch::Tensor &prm) -> torch::Tensor {
    const auto v = prm.to(torch::kInt64);
    const int64_t x0 = v[0].item<int64_t>(), y0 = v[1].item<int64_t>(), x1 = v[2].item<int64_t>(),
                  y1 = v[3].item<int64_t>();
    auto m = torch::zeros({40, 48}, torch::kUInt8);
    m.slice(0, y0, y1).slice(1, x0, x1).fill_(1);
    if (v[4].item<int64_t>()) {
        m.slice(0, y0 + (y1 - y0) / 3, y1 - (y1 - y0) / 3).slice(1, x0 + (x1 - x0) / 3, x1 - (x1 - x0) / 3).fill_(0);
    }
    return m;
}

static auto masks(const torch::Tensor &params) -> torch::Tensor {
    std::vector<torch::Tensor> ms;
    for (int64_t i = 0; i < params.size(0); ++i) {
        ms.push_back(make_mask(params[i]));
    }
    return ms.empty() ? torch::zeros({0, 40, 48}, torch::kUInt8) : torch::stack(ms);
}

template<class E>
static auto load_segm(const E &pm, const E &ps, const E &pl, const E &pc, const E &gm, const E &gl, const E &gcr,
                      const E &gc) -> dataset {
    const auto PM = G(pm).reshape({-1, 5}), GM = G(gm).reshape({-1, 5});
    const auto PS = G(ps).to(torch::kFloat32), PL = G(pl).to(torch::kInt64), GL = G(gl).to(torch::kInt64),
               GC = G(gcr).to(torch::kInt64);
    dataset d;
    int64_t po = 0, go = 0;
    for (size_t i = 0; i < pc.data.size(); ++i) {
        const auto np = static_cast<int64_t>(pc.data[i]), ng = static_cast<int64_t>(gc.data[i]);
        // bool prediction masks, uint8 target masks
        d.preds.push_back(
                {{}, PS.slice(0, po, po + np), PL.slice(0, po, po + np), masks(PM.slice(0, po, po + np)) != 0});
        d.target.push_back(
                {{}, GL.slice(0, go, go + ng), GC.slice(0, go, go + ng), {}, masks(GM.slice(0, go, go + ng))});
        po += np;
        go += ng;
    }
    return d;
}
#define LOAD_SEGM(n)                                                                                                   \
    load_segm(golden::n##_pred_masks, golden::n##_pred_scores, golden::n##_pred_labels, golden::n##_pred_counts,       \
              golden::n##_gt_masks, golden::n##_gt_labels, golden::n##_gt_crowd, golden::n##_gt_counts)

static void test_map_segm() {
    const auto d = LOAD_SEGM(dd);
    const auto segm = mean_average_precision_option().iou_type(iou_type_mode::segm);
    CHECK_MAP_PC(mean_average_precision(d.preds, d.target, mean_average_precision_option(segm).class_metrics(true)),
                 map_dd_segm);
    CHECK_MAP(
            mean_average_precision(d.preds, d.target, mean_average_precision_option(segm).average(average_mode::micro)),
            map_dd_segm_micro);
    // detections inside crowd masks: crowd IoU is intersection / detection area
    const auto dg = LOAD_SEGM(dg);
    CHECK_MAP_PC(mean_average_precision(dg.preds, dg.target, mean_average_precision_option(segm).class_metrics(true)),
                 map_dg_segm);
    // no prediction masks anywhere: torchmetrics' empty COCO dataset -> -1
    auto none = d;
    for (auto &p: none.preds) {
        p = {{}, torch::zeros({0}), torch::zeros({0}, torch::kInt64), torch::zeros({0})};
    }
    TM_CHECK(mean_average_precision(none.preds, none.target, segm).map == -1);
    // mismatched mask sizes within an image are rejected (pycocotools: IoU -1)
    size_t both = 0;
    while (d.preds[both].masks.size(0) == 0 || d.target[both].masks.size(0) == 0) {
        ++both;
    }
    auto bad = slice(d, both, both + 1);
    bad.preds[0].masks = torch::zeros({bad.preds[0].masks.size(0), 40, 47});
    TM_CHECK(throws_invalid([&] { return mean_average_precision(bad.preds, bad.target, segm); }));
    bad = slice(d, both, both + 1);
    bad.target[0].masks = bad.target[0].masks.flatten(1);
    TM_CHECK(throws_invalid([&] { return mean_average_precision(bad.preds, bad.target, segm); }));
    bad = slice(d, both, both + 1);
    bad.preds[0].masks = torch::Tensor();
    TM_CHECK(throws_invalid([&] { return mean_average_precision(bad.preds, bad.target, segm); }));
}

// Net CPU bytes libtorch allocates while installed (its allocator reports to the thread's PROFILER_STATE debug info).
struct cpu_bytes final : c10::MemoryReportingInfoBase {
    int64_t live = 0;
    void reportMemoryUsage(void *, int64_t size, size_t, size_t, c10::Device device) override {
        live += device.is_cpu() ? size : 0;
    }
    bool memoryProfilingEnabled() const override { return true; }
};

// iou_type segm keeps each image's IoUs, not its masks: the state does not grow with the mask size
static void test_map_segm_memory() {
    const auto bytes = std::make_shared<cpu_bytes>();
    c10::DebugInfoGuard guard(c10::DebugInfoKind::PROFILER_STATE, bytes);
    MeanAveragePrecision m(mean_average_precision_option().iou_type(iou_type_mode::segm));
    for (int i = 0; i < 4; ++i) {
        const std::vector<detection_prediction> p{
                {{}, torch::rand({100}), torch::randint(0, 3, {100}), torch::rand({100, 256, 256}) > 0.5}};
        const std::vector<detection_target> t{
                {.labels = torch::randint(0, 3, {4}), .masks = torch::rand({4, 256, 256}) > 0.5}};
        m.update(p, t);
    }
    // the masks alone would be 4 x 104 x 256^2 bytes (27 MB)
    tm_test::record(bytes->live < (1 << 20), "segm state bytes", __FILE__, __LINE__, std::to_string(bytes->live));
    TM_CHECK(m.compute().map >= 0);
}

static void test_map_validation() {
    const auto one = slice(DA, 0, 1);
    TM_CHECK(throws_invalid([&] { return mean_average_precision(one.preds, slice(DA, 0, 2).target); }));
    const auto expect_invalid = [&](auto &&mutate, int line) {
        auto d = one;
        mutate(d);
        tm_test::record(throws_invalid([&] { return mean_average_precision(d.preds, d.target); }), "map validation",
                        __FILE__, line);
    };
    expect_invalid([](dataset &d) { d.preds[0].scores = torch::Tensor(); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].labels = torch::Tensor(); }, __LINE__);
    expect_invalid([](dataset &d) { d.target[0].labels = torch::Tensor(); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].scores = d.preds[0].scores.slice(0, 1); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].scores = d.preds[0].scores.unsqueeze(1); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].boxes = d.preds[0].boxes.slice(0, 1); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].boxes = torch::tensor(1.0); }, __LINE__);
    expect_invalid([](dataset &d) { d.target[0].boxes = torch::zeros({d.target[0].labels.size(0), 5}); }, __LINE__);
    expect_invalid([](dataset &d) { d.target[0].iscrowd = torch::zeros({1}); }, __LINE__);
    expect_invalid([](dataset &d) { d.target[0].area = torch::zeros({1}); }, __LINE__);
    expect_invalid([](dataset &d) { d.target[0].labels = d.target[0].labels.to(torch::kFloat32); }, __LINE__);
    expect_invalid([](dataset &d) { d.preds[0].scores = d.preds[0].scores.to(torch::kInt64); }, __LINE__);
    TM_CHECK(one.preds[0].labels.size(0) > 1 && one.target[0].labels.size(0) > 1);
    // max_detection_thresholds: 3 values >= 0
    TM_CHECK(throws_invalid(
            [] { return MeanAveragePrecision(mean_average_precision_option().max_detection_thresholds({1, 10})); }));
    TM_CHECK(throws_invalid([] {
        return MeanAveragePrecision(mean_average_precision_option().max_detection_thresholds({1, -10, 100}));
    }));
    // a failed update leaves the state unchanged
    MeanAveragePrecision m;
    auto d = one;
    d.preds[0].scores = torch::Tensor();
    TM_CHECK(throws_invalid([&] {
        m.update(d.preds, d.target);
        return 0;
    }));
    TM_CHECK(m.compute().map == -1);
}

// ---------------- VOC devkit AP ----------------
template<class E>
static void check_voc(const voc_average_precision_result &r, const E &want, const E &classes, const char *name,
                      int line) {
    const torch::Tensor w = G(want);
    tm_test::record(r.map == w[0].item<double>(), name, __FILE__, line, "map");
    tm_test::record(torch::equal(r.ap_per_class, w.slice(0, 1)), name, __FILE__, line, "ap per class");
    tm_test::record(torch::equal(r.classes, G(classes).to(torch::kInt64)), name, __FILE__, line, "classes");
}
#define CHECK_VOC(r, n) check_voc((r), golden::n, golden::voc_classes, #n, __LINE__)

static void test_voc() {
    CHECK_VOC(voc_average_precision(DA.preds, DA.target), voc_all);
    CHECK_VOC(voc_average_precision(DA.preds, DA.target,
                                    voc_average_precision_option().interpolation(voc_interpolation_mode::eleven_point)),
              voc_07);
    CHECK_VOC(voc_average_precision(DA.preds, DA.target, voc_average_precision_option().pixel_inclusive(false)),
              voc_cont);
    CHECK_VOC(voc_average_precision(DA.preds, DA.target,
                                    voc_average_precision_option()
                                            .interpolation(voc_interpolation_mode::eleven_point)
                                            .iou_threshold(0.7)),
              voc_07_thr);
    const auto x = converted(slice(DA, 0, 30), BF::cxcywh);
    const auto r = voc_average_precision(x.preds, x.target, voc_average_precision_option().box_format(BF::cxcywh));
    const auto w = G(golden::voc_cxcywh_30);
    TM_CHECK(r.map == w[0].item<double>() && torch::equal(r.ap_per_class, w.slice(0, 1)));
    // no class with a non-difficult target: map -1
    const auto dc2 = LOAD(dc2, false, false);
    const auto none = voc_average_precision(dc2.preds, dc2.target);
    TM_CHECK(none.map == -1 && (none.ap_per_class == -1).all().item<bool>());
    TM_CHECK(throws_invalid([&] { return voc_average_precision(slice(DA, 0, 1).preds, slice(DA, 0, 2).target); }));
    // IoU exactly 0.5 is not a match (IoU > threshold), 12 classes
    const auto de = LOAD(de, true, true);
    check_voc(voc_average_precision(de.preds, de.target, voc_average_precision_option().pixel_inclusive(false)),
              golden::voc_de_cont, golden::voc_de_classes, "voc_de_cont", __LINE__);
}

static auto to_device(const dataset &d, torch::Device device) -> dataset {
    auto out = d;
    for (auto &p: out.preds) {
        for (auto *t: {&p.boxes, &p.scores, &p.labels, &p.masks}) {
            *t = t->defined() ? t->to(device) : *t;
        }
    }
    for (auto &g: out.target) {
        for (auto *t: {&g.boxes, &g.labels, &g.iscrowd, &g.area, &g.masks}) {
            *t = t->defined() ? t->to(device) : *t;
        }
    }
    return out;
}

// MPS inputs (no float64 there): the float64 COCO / VOC evaluation runs on the CPU, the IoU classes stay on MPS
static void test_mps() {
    if (!torch::mps::is_available()) {
        return;
    }
    const auto d = slice(DA, 0, 30), m = to_device(d, torch::kMPS);
    const auto cm = mean_average_precision_option().class_metrics(true);
    const auto rm = mean_average_precision(m.preds, m.target, cm), rc = mean_average_precision(d.preds, d.target, cm);
    TM_CHECK(torch::equal(stats(rm), stats(rc)) && torch::equal(rm.map_per_class, rc.map_per_class));
    const auto vm = voc_average_precision(m.preds, m.target), vc = voc_average_precision(d.preds, d.target);
    TM_CHECK(vm.map == vc.map && torch::equal(vm.ap_per_class, vc.ap_per_class));
    IntersectionOverUnion im(intersection_over_union_option().class_metrics(true)),
            ic(intersection_over_union_option().class_metrics(true));
    im.update(m.preds, m.target);
    ic.update(d.preds, d.target);
    const auto gm = im.compute(), gc = ic.compute();
    bool same = gm.size() == gc.size() && gm.at("iou").device().is_mps();
    for (const auto &[k, v]: gc) {
        same = same && gm.count(k) && torch::allclose(gm.at(k).cpu(), v, 1e-5, 1e-6);
    }
    TM_CHECK(same);
}

int main() {
    TM_RUN(test_box_ops);
    TM_RUN(test_box_convert);
    TM_RUN(test_iou_functional);
    TM_RUN(test_iou_classes);
    TM_RUN(test_map_bbox);
    TM_RUN(test_map_edges);
    TM_RUN(test_map_boundaries);
    TM_RUN(test_map_segm);
    TM_RUN(test_map_segm_memory);
    TM_RUN(test_map_validation);
    TM_RUN(test_voc);
    TM_RUN(test_mps);
    std::fprintf(stderr, "MeanAveragePrecision max |delta| vs pycocotools: %g\n", g_map_max_delta);
    return tm_test::summary("metric_test_detection");
}
