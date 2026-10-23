#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_TEXT_HPP
#define LIB_TORCH_MEDIA_METRIC_TEXT_HPP
// Text metrics (develop_log/2026-10-24/progress01): torchmetrics 1.9 torchmetrics.text WordErrorRate, CharErrorRate,
// MatchErrorRate, WordInfoLost, WordInfoPreserved and EditDistance with their functional forms. Strings are UTF-8,
// decoded to code points like Python str (CER and EditDistance count code points); words are split like str.split()
// (runs of Unicode whitespace). Rates are float32 0-d tensors computed in torchmetrics' float32 order. A single string
// is `{s}` in the functionals and update(); forward() / operator() are templates, so pass them a named
// std::vector<std::string> (a braced list does not deduce).
#include "../_audio/_functional.hpp"
#include "_common.hpp"

namespace torchmedia::metric::text {
    namespace detail {
        using metric::detail::fail;
        using metric::detail::invalid;

        // UTF-8 -> code points; false on an invalid sequence (bad lead / continuation byte, truncated, overlong,
        // surrogate, above U+10FFFF).
        inline auto decode_utf8(const std::string &s, std::u32string &out) -> bool {
            static constexpr char32_t min_cp[] = {0, 0, 0x80, 0x800, 0x10000};
            out.clear();
            for (size_t i = 0; i < s.size();) {
                const auto c = static_cast<unsigned char>(s[i]);
                const int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
                if (n == 0 || i + n > s.size()) {
                    return false;
                }
                char32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
                for (int k = 1; k < n; ++k) {
                    const auto cc = static_cast<unsigned char>(s[i + k]);
                    if ((cc >> 6) != 0x2) {
                        return false;
                    }
                    cp = (cp << 6) | (cc & 0x3F);
                }
                if (cp < min_cp[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
                    return false;
                }
                out.push_back(cp);
                i += n;
            }
            return true;
        }

        // Python str.isspace(), the separators of str.split().
        inline auto is_space(char32_t c) -> bool {
            return (c >= 0x9 && c <= 0xD) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
                   (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
                   c == 0x3000;
        }

        inline auto split(const std::u32string &s) -> std::vector<std::u32string> {
            std::vector<std::u32string> words;
            std::u32string w;
            for (const auto c: s) {
                if (!is_space(c)) {
                    w.push_back(c);
                } else if (!w.empty()) {
                    words.push_back(std::move(w));
                    w.clear();
                }
            }
            if (!w.empty()) {
                words.push_back(std::move(w));
            }
            return words;
        }

        // Decoded (preds, target) pairs, or the error message.
        inline auto decode(const char *fn, const std::vector<std::string> &preds,
                           const std::vector<std::string> &target,
                           std::vector<std::pair<std::u32string, std::u32string>> &out) -> std::string {
            if (preds.size() != target.size()) {
                return std::format("{}: expected `preds` and `target` to have the same length, but got {} and {}.", fn,
                                   preds.size(), target.size());
            }
            out.assign(preds.size(), {});
            for (size_t i = 0; i < preds.size(); ++i) {
                if (!decode_utf8(preds[i], out[i].first) || !decode_utf8(target[i], out[i].second)) {
                    return std::format("{}: pair {} is not valid UTF-8.", fn, i);
                }
            }
            return {};
        }

        enum class word_kind { wer, cer, mer, wil, wip };

        inline auto name(word_kind k) -> const char * {
            constexpr const char *names[] = {"word_error_rate", "char_error_rate", "match_error_rate",
                                             "word_information_lost", "word_information_preserved"};
            return names[static_cast<int>(k)];
        }

        // torchmetrics' float32 update sums: errors (WIL / WIP: errors - sum of max lengths), target / preds token
        // counts, max lengths.
        struct word_counts {
            float errors = 0, target_total = 0, preds_total = 0, max_total = 0;
        };

        inline auto count(word_kind k, const std::vector<std::string> &preds, const std::vector<std::string> &target,
                          word_counts &c) -> std::string {
            std::vector<std::pair<std::u32string, std::u32string>> pairs;
            if (auto e = decode(name(k), preds, target, pairs); !e.empty()) {
                return e;
            }
            for (const auto &[p, t]: pairs) {
                const auto add = [&](const auto &pt, const auto &tt) {
                    c.errors += static_cast<float>(::torchmedia::audio::functional::edit_distance(pt, tt));
                    c.target_total += static_cast<float>(tt.size());
                    c.preds_total += static_cast<float>(pt.size());
                    c.max_total += static_cast<float>(std::max(tt.size(), pt.size()));
                };
                k == word_kind::cer ? add(p, t) : add(split(p), split(t));
            }
            if (k == word_kind::wil || k == word_kind::wip) {
                c.errors = c.errors - c.max_total;
            }
            return {};
        }

        inline auto rate(word_kind k, const word_counts &c) -> tensor_t {
            const auto f = [](float v) { return torch::tensor(v, torch::kFloat32); };
            switch (k) {
                case word_kind::wer:
                case word_kind::cer:
                    return f(c.errors) / f(c.target_total);
                case word_kind::mer:
                    return f(c.errors) / f(c.max_total);
                default: {
                    const auto wip = (f(c.errors) / f(c.target_total)) * (f(c.errors) / f(c.preds_total));
                    return k == word_kind::wip ? wip : 1 - wip;
                }
            }
        }

        inline auto word_rate(word_kind k, const std::vector<std::string> &preds,
                              const std::vector<std::string> &target) -> tensor_t {
            word_counts c;
            if (auto e = count(k, preds, target, c); !e.empty()) {
                return invalid(std::move(e));
            }
            return rate(k, c);
        }

        // Accumulated counts of the rate classes (per update, like torchmetrics' float32 states).
        template<word_kind K>
        class word_rate_metric : public metric::detail::metric_base<word_rate_metric<K>> {
        public:
            auto update(const std::vector<std::string> &preds, const std::vector<std::string> &target) -> void {
                word_counts c;
                if (auto e = count(K, preds, target, c); !e.empty()) {
                    return fail(std::move(e));
                }
                _c.errors += c.errors;
                _c.target_total += c.target_total;
                _c.preds_total += c.preds_total;
                _c.max_total += c.max_total;
            }
            auto compute() const -> tensor_t { return rate(K, _c); }

        private:
            word_counts _c;
        };
    } // namespace detail

    // torchmetrics word_error_rate: sum of word edit distances / number of target words.
    inline auto word_error_rate(const std::vector<std::string> &preds, const std::vector<std::string> &target)
            -> tensor_t {
        return detail::word_rate(detail::word_kind::wer, preds, target);
    }
    // torchmetrics char_error_rate: sum of code-point edit distances / number of target code points (whitespace
    // included).
    inline auto char_error_rate(const std::vector<std::string> &preds, const std::vector<std::string> &target)
            -> tensor_t {
        return detail::word_rate(detail::word_kind::cer, preds, target);
    }
    // torchmetrics match_error_rate: sum of word edit distances / sum of max(target words, preds words).
    inline auto match_error_rate(const std::vector<std::string> &preds, const std::vector<std::string> &target)
            -> tensor_t {
        return detail::word_rate(detail::word_kind::mer, preds, target);
    }
    // torchmetrics word_information_preserved: (H / N_target) (H / N_preds) with H = sum of max lengths - errors.
    inline auto word_information_preserved(const std::vector<std::string> &preds,
                                           const std::vector<std::string> &target) -> tensor_t {
        return detail::word_rate(detail::word_kind::wip, preds, target);
    }
    // torchmetrics word_information_lost: 1 - word_information_preserved.
    inline auto word_information_lost(const std::vector<std::string> &preds, const std::vector<std::string> &target)
            -> tensor_t {
        return detail::word_rate(detail::word_kind::wil, preds, target);
    }

    using WordErrorRate = detail::word_rate_metric<detail::word_kind::wer>;
    using CharErrorRate = detail::word_rate_metric<detail::word_kind::cer>;
    using MatchErrorRate = detail::word_rate_metric<detail::word_kind::mer>;
    using WordInfoLost = detail::word_rate_metric<detail::word_kind::wil>;
    using WordInfoPreserved = detail::word_rate_metric<detail::word_kind::wip>;

    // ---------------- EditDistance ----------------
    typedef struct edit_distance_option {
        int64_t _substitution_cost = 1; // >= 0; insertions and deletions cost 1
        reduction_mode _reduction = reduction_mode::elementwise_mean; // torchmetrics' "mean" / "sum" / "none"

        auto substitution_cost(int64_t v) -> edit_distance_option & {
            _substitution_cost = v;
            return *this;
        }
        auto reduction(reduction_mode v) -> edit_distance_option & {
            _reduction = v;
            return *this;
        }
    } edit_distance_option_t;

    namespace detail {
        // torchmetrics _LevenshteinEditDistance(reference = target, op_substitute)(prediction): sacrebleu's TER DP that
        // only fills a band of 25 (more for long references) around the length-ratio diagonal in every row but the
        // last, so distances of very different strings can exceed the true Levenshtein distance.
        inline auto beam_edit_distance(const std::u32string &pred, const std::u32string &ref, int64_t sub_cost)
                -> int64_t {
            constexpr int64_t undefined = 10000000000000000, beam_width = 25;
            // Cells never exceed `undefined`; capping the cost above int32 keeps every distance that fits int32 and
            // leaves the others above it (Python ints there), without int64 overflow.
            sub_cost = std::min<int64_t>(sub_cost, int64_t{1} << 31);
            const auto n = static_cast<int64_t>(pred.size()), m = static_cast<int64_t>(ref.size());
            std::vector<int64_t> prev(m + 1), cur(m + 1);
            std::iota(prev.begin(), prev.end(), 0);
            const double ratio = n > 0 ? static_cast<double>(m) / static_cast<double>(n) : 1.0;
            const auto beam =
                    ratio / 2 > beam_width ? static_cast<int64_t>(std::ceil(ratio / 2 + beam_width)) : beam_width;
            for (int64_t i = 1; i <= n; ++i) {
                std::fill(cur.begin(), cur.end(), undefined);
                const auto diag = static_cast<int64_t>(std::floor(static_cast<double>(i) * ratio));
                const auto hi = i == n ? m + 1 : std::min(m + 1, diag + beam);
                for (int64_t j = std::max<int64_t>(0, diag - beam); j < hi; ++j) {
                    if (j == 0) {
                        cur[0] = prev[0] + 1;
                        continue;
                    }
                    for (const auto c:
                         {prev[j - 1] + (pred[i - 1] == ref[j - 1] ? 0 : sub_cost), prev[j] + 1, cur[j - 1] + 1}) {
                        cur[j] = std::min(cur[j], c);
                    }
                }
                std::swap(prev, cur);
            }
            return prev[m];
        }

        // Per-pair distances (int32 [N]), or the error message.
        inline auto edit_distances(const std::vector<std::string> &preds, const std::vector<std::string> &target,
                                   const edit_distance_option &opt, tensor_t &out) -> std::string {
            if (opt._substitution_cost < 0) {
                return std::format("edit_distance: expected argument `substitution_cost` to be a non-negative "
                                   "integer, but got {}",
                                   opt._substitution_cost);
            }
            std::vector<std::pair<std::u32string, std::u32string>> pairs;
            if (auto e = decode("edit_distance", preds, target, pairs); !e.empty()) {
                return e;
            }
            std::vector<int32_t> d;
            for (const auto &[p, t]: pairs) {
                const auto v = beam_edit_distance(p, t, opt._substitution_cost);
                if (v > std::numeric_limits<int32_t>::max()) { // torchmetrics: int32 overflow / undefined-cell error
                    return std::format("edit_distance: the distance of pair {} does not fit int32.", d.size());
                }
                d.push_back(static_cast<int32_t>(v));
            }
            out = torch::tensor(d, torch::kInt32);
            return {};
        }

        // _edit_distance_compute
        inline auto edit_compute(const tensor_t &scores, const tensor_t &num_elements, reduction_mode r) -> tensor_t {
            if (scores.numel() == 0) {
                return torch::tensor(0, torch::kInt32);
            }
            return r == reduction_mode::elementwise_mean ? scores.sum() / num_elements
                   : r == reduction_mode::sum            ? scores.sum()
                                                         : scores;
        }
    } // namespace detail

    // torchmetrics edit_distance: code-point edit distance per pair (int32 [N] with reduction none), their sum (int64)
    // or mean (float32); no pair gives int32 0.
    inline auto edit_distance(const std::vector<std::string> &preds, const std::vector<std::string> &target,
                              const edit_distance_option &opt = {}) -> tensor_t {
        tensor_t d;
        if (auto e = detail::edit_distances(preds, target, opt, d); !e.empty()) {
            return detail::invalid(std::move(e));
        }
        return detail::edit_compute(d, torch::tensor(d.numel()), opt._reduction);
    }

    class EditDistance : public metric::detail::metric_base<EditDistance> {
    public:
        explicit EditDistance(edit_distance_option opt = {}) : _opt(opt) {
            if (_opt._substitution_cost < 0) {
                const auto c = _opt._substitution_cost;
                _opt = {};
                detail::fail(std::format("edit_distance: expected argument `substitution_cost` to be a non-negative "
                                         "integer, but got {}",
                                         c));
            }
        }

        auto update(const std::vector<std::string> &preds, const std::vector<std::string> &target) -> void {
            tensor_t d;
            if (auto e = detail::edit_distances(preds, target, _opt, d); !e.empty()) {
                return detail::fail(std::move(e));
            }
            if (_opt._reduction == reduction_mode::none) {
                _scores_list.push_back(d);
            } else {
                _scores = _scores + d.sum();
                _num = _num + d.size(0);
            }
        }

        // reduction none: the concatenated per-pair distances (torch.cat raises before any update), 0-d for one pair
        // (Metric.compute's _squeeze_if_scalar).
        auto compute() const -> tensor_t {
            return metric::detail::squeeze_if_scalar(
                    _opt._reduction == reduction_mode::none
                            ? detail::edit_compute(torch::cat(_scores_list), torch::tensor(1), _opt._reduction)
                            : detail::edit_compute(_scores, _num, _opt._reduction));
        }
        auto options() const -> const edit_distance_option & { return _opt; }

    private:
        edit_distance_option _opt;
        tensor_t _scores = torch::tensor(0, torch::kInt64), _num = torch::tensor(0, torch::kInt64);
        std::vector<tensor_t> _scores_list;
    };
} // namespace torchmedia::metric::text
#endif // LIB_TORCH_MEDIA_METRIC_TEXT_HPP
