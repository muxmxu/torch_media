// Assertion-based tests for torchmedia::metric::text against torchmetrics 1.9 goldens (gen_golden.py ->
// golden_text.hpp; jiwer 4.0 cross-checked there): English, Chinese, Japanese, mixed-script and emoji strings, empty
// strings and every whitespace kind of Python's str.split().
#include "golden_text.hpp"
#include "metric_test_util.hpp"

using namespace torchmedia::metric::text;
using torchmedia::metric::reduction_mode;
using strings = std::vector<std::string>;

static const strings P = {"the cat sat on the mat",
                          "hello world",
                          "今天 天气 很 好",
                          "我爱北京天安门",
                          "日本語の テキスト です",
                          "mixed 中文 and English 😀 text",
                          "",
                          "a\tb\nc　d e",
                          "  spaced   out  words ",
                          "one two three four",
                          "東京タワー に 行きました",
                          "naïve café"};
static const strings T = {"the cat sat on a mat",    "hello there world",
                          "今天 天气 真 好",         "我爱北京天安门广场",
                          "日本語 の テキスト です", "mixed 中文 or English text 😀",
                          "nothing predicted",       "a b c d e",
                          "spaced out words",        "",
                          "東京タワーに 行きました", "naive cafe"};

static auto same_value(const torch::Tensor &a, const torch::Tensor &b) -> bool {
    return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && torch::allclose(a, b, 0.0, 0.0, true);
}
static auto head(const strings &v, size_t n) -> strings { return {v.begin(), v.begin() + n}; }
static auto tail(const strings &v, size_t n) -> strings { return {v.begin() + n, v.end()}; }

static void test_rates() {
    using namespace golden;
    const golden::entry *pairs[12][2] = {{&pair0, &pair0_edit}, {&pair1, &pair1_edit},   {&pair2, &pair2_edit},
                                         {&pair3, &pair3_edit}, {&pair4, &pair4_edit},   {&pair5, &pair5_edit},
                                         {&pair6, &pair6_edit}, {&pair7, &pair7_edit},   {&pair8, &pair8_edit},
                                         {&pair9, &pair9_edit}, {&pair10, &pair10_edit}, {&pair11, &pair11_edit}};
    for (size_t k = 0; k < P.size(); ++k) {
        const strings p{P[k]}, t{T[k]};
        const auto v = torch::stack({word_error_rate(p, t), char_error_rate(p, t), match_error_rate(p, t),
                                     word_information_lost(p, t), word_information_preserved(p, t)});
        check_golden(v, *pairs[k][0], "pair", __LINE__);
        check_golden(edit_distance(p, t, edit_distance_option().reduction(reduction_mode::none)), *pairs[k][1],
                     "pair_edit", __LINE__);
    }
    CHECK_GOLDEN(word_error_rate(P, T), wer_all);
    CHECK_GOLDEN(char_error_rate(P, T), cer_all);
    CHECK_GOLDEN(match_error_rate(P, T), mer_all);
    CHECK_GOLDEN(word_information_lost(P, T), wil_all);
    CHECK_GOLDEN(word_information_preserved(P, T), wip_all);
    CHECK_GOLDEN(word_error_rate(head(P, 6), head(T, 6)), wer_head);
    CHECK_GOLDEN(char_error_rate(head(P, 6), head(T, 6)), cer_head);
    CHECK_GOLDEN(match_error_rate(head(P, 6), head(T, 6)), mer_head);
    CHECK_GOLDEN(word_information_lost(head(P, 6), head(T, 6)), wil_head);
    CHECK_GOLDEN(word_information_preserved(head(P, 6), head(T, 6)), wip_head);
    CHECK_GOLDEN(word_error_rate({""}, {""}), wer_empty_target);
    CHECK_GOLDEN(word_error_rate({"a b"}, {" "}), wer_no_target_words);
    // classes: two updates equal torchmetrics' accumulated states; forward() returns the batch alone
    WordErrorRate wer;
    CharErrorRate cer;
    MatchErrorRate mer;
    WordInfoLost wil;
    WordInfoPreserved wip;
    const auto p5 = head(P, 5), t5 = head(T, 5), pr = tail(P, 5), tr = tail(T, 5);
    wer.update(p5, t5);
    cer.update(p5, t5);
    mer.update(p5, t5);
    wil.update(p5, t5);
    wip.update(p5, t5);
    TM_CHECK(wer(pr, tr).item<float>() == word_error_rate(pr, tr).item<float>());
    TM_CHECK(cer(pr, tr).item<float>() == char_error_rate(pr, tr).item<float>());
    TM_CHECK(mer(pr, tr).item<float>() == match_error_rate(pr, tr).item<float>());
    TM_CHECK(wil.forward(pr, tr).item<float>() == word_information_lost(pr, tr).item<float>());
    TM_CHECK(wip.forward(pr, tr).item<float>() == word_information_preserved(pr, tr).item<float>());
    CHECK_GOLDEN(wer.compute(), cls_wer);
    CHECK_GOLDEN(cer.compute(), cls_cer);
    CHECK_GOLDEN(mer.compute(), cls_mer);
    CHECK_GOLDEN(wil.compute(), cls_wil);
    CHECK_GOLDEN(wip.compute(), cls_wip);
    wer.reset();
    TM_CHECK(std::isnan(wer.compute().item<float>()));
    // checks: different lengths (torchmetrics zips silently), invalid UTF-8
    TM_CHECK(throws_invalid([] { return word_error_rate({"a", "b"}, {"a"}); }));
    TM_CHECK(throws_invalid([] { return char_error_rate({"\xff"}, {"a"}); }));
    TM_CHECK(throws_invalid([&] { return (cer.update({"a"}, {"\xe4\xbd"}), 0); }));
    TM_CHECK(cer.compute().item<float>() == static_cast<float>(golden::cls_cer.data[0])); // untouched
}

static void test_tokenisation() {
    namespace D = torchmedia::metric::text::detail;
    std::u32string s;
    // every Python str.isspace() code point separates words; U+200B (zero width space) does not
    const std::string all_ws = "\t\n\v\f\r\x1c\x1d\x1e\x1f \u0085        "
                               "         　";
    TM_CHECK(D::decode_utf8("a" + all_ws + "b​c" + all_ws, s) && D::split(s).size() == 2);
    TM_CHECK(D::decode_utf8("😀é", s) && s == U"😀é");
    for (const std::string bad: {"\x80", "\xc3", "\xc0\xaf", "\xe0\x80\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80",
                                 "\xf8\x88\x80\x80\x80", "\xe4\xbd\x41"}) {
        TM_CHECK(!D::decode_utf8(bad, s));
    }
}

static void test_edit_distance() {
    using namespace golden;
    using O = edit_distance_option;
    std::string s;
    for (int i = 0; i < 100; ++i) {
        s.push_back(static_cast<char>('a' + (i * 7 + i * i) % 26));
    }
    std::string jp;
    for (int i = 0; i < 20; ++i) {
        jp += "日本語";
    }
    // beyond sacrebleu's beam (above the true distance): 2 pairs at its lower edge, 3 at its upper edge
    const std::string x30(30, 'x'), y30(30, 'y');
    auto ep = P, et = T;
    ep.insert(ep.end(), {x30 + s, jp + s, s, s, s + x30});
    et.insert(et.end(), {s + y30, s, y30 + s, jp + s, y30 + s});
    const golden::entry *g[3][3] = {{&edit_none_0, &edit_mean_0, &edit_sum_0},
                                    {&edit_none_1, &edit_mean_1, &edit_sum_1},
                                    {&edit_none_2, &edit_mean_2, &edit_sum_2}};
    for (int sc = 0; sc < 3; ++sc) {
        const auto o = O().substitution_cost(sc);
        check_golden(edit_distance(ep, et, O(o).reduction(reduction_mode::none)), *g[sc][0], "edit_none", __LINE__);
        check_golden(edit_distance(ep, et, o), *g[sc][1], "edit_mean", __LINE__);
        check_golden(edit_distance(ep, et, O(o).reduction(reduction_mode::sum)), *g[sc][2], "edit_sum", __LINE__);
    }
    const golden::entry *cls[3] = {&cls_edit_mean, &cls_edit_sum, &cls_edit_none};
    const reduction_mode reds[3] = {reduction_mode::elementwise_mean, reduction_mode::sum, reduction_mode::none};
    for (int r = 0; r < 3; ++r) {
        EditDistance m(O().substitution_cost(2).reduction(reds[r]));
        m.update(head(ep, 7), head(et, 7));
        TM_CHECK(same_value(m(tail(ep, 7), tail(et, 7)), edit_distance(tail(ep, 7), tail(et, 7), m.options())));
        check_golden(m.compute(), *cls[r], "cls_edit", __LINE__);
    }
    for (const auto r: reds) { // no pair: int32 0
        const auto e = edit_distance({}, {}, O().reduction(r));
        TM_CHECK(e.scalar_type() == torch::kInt32 && e.item<int>() == 0);
    }
    EditDistance none(O().reduction(reduction_mode::none));
    TM_CHECK(throws_any([&] { return none.compute(); })); // torch.cat of no batch
    none.update({}, {});
    TM_CHECK(none.compute().item<int>() == 0);
    TM_CHECK(std::isnan(EditDistance().compute().item<float>())); // 0 / 0
    // one pair, reduction none: the class squeezes to 0-d like torchmetrics' Metric (the functional keeps [1])
    EditDistance one(O().reduction(reduction_mode::none));
    const strings p1{"abc"}, t1{"abd"};
    const auto f1 = one(p1, t1), c1 = one.compute();
    TM_CHECK(f1.dim() == 0 && c1.dim() == 0 && f1.scalar_type() == torch::kInt32 && c1.item<int>() == 1);
    // substitution costs near INT64_MAX (Python ints in torchmetrics), a substitution forced where beam rows do not
    // overlap, and distances beyond int32 (torchmetrics raises)
    const auto x100 = std::string(100, 'x');
    CHECK_GOLDEN(edit_distance({"ab"}, {"cd"}, O().substitution_cost(INT64_MAX).reduction(reduction_mode::none)),
                 edit_huge_cost);
    CHECK_GOLDEN(edit_distance({"ab"}, {x100}, O().substitution_cost(1000).reduction(reduction_mode::none)),
                 edit_gap_cost);
    for (const int64_t c: {int64_t{1} << 40, INT64_MAX}) {
        TM_CHECK(throws_invalid([&] { return edit_distance({"ab"}, {x100}, O().substitution_cost(c)); }));
    }
    TM_CHECK(throws_invalid([] { return edit_distance({"a"}, {"a", "b"}); }));
    TM_CHECK(throws_invalid([] { return edit_distance({"a"}, {"b"}, O().substitution_cost(-1)); }));
    TM_CHECK(throws_invalid([] { return edit_distance({"\xc3\x28"}, {"b"}); }));
    TM_CHECK(throws_invalid([] { return EditDistance(O().substitution_cost(-2)); }));
    EditDistance m;
    TM_CHECK(throws_invalid([&] { return (m.update({"a"}, {}), 0); }));
}

int main() {
    TM_RUN(test_rates);
    TM_RUN(test_tokenisation);
    TM_RUN(test_edit_distance);
    return tm_test::summary("metric_test_text");
}
