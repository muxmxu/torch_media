#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
#include <limits>
#include <utility>
#include <vector>
#include "../globel_include.hpp"

// CTC forced alignment (mirrors torchaudio.functional.forced_align / merge_tokens / TokenSpan).
// Upstream dispatches to a compiled op; here it is reimplemented as a header-only Viterbi DP over the
// CTC trellis (batch size 1, the only size torchaudio supports).
namespace torchmedia::audio::functional {
    // Value type for one aligned token (mirrors torchaudio's TokenSpan dataclass).
    struct token_span {
        int64_t token;
        int64_t start;
        int64_t end;
        double score;
        auto length() const -> int64_t { return end - start; }
    };

    // Viterbi forced alignment. log_probs: (1, T, C) log-probabilities; targets: (1, L).
    // Returns (paths, scores): paths (1, T) chosen label per frame; scores (1, T) the log-prob of it.
    inline auto forced_align(const_tensor_lref_t log_probs, const_tensor_lref_t targets,
                             const c10::optional<tensor_t> & /*input_lengths*/ = c10::nullopt,
                             const c10::optional<tensor_t> & /*target_lengths*/ = c10::nullopt, int64_t blank = 0)
            -> std::pair<tensor_t, tensor_t> {
        const int64_t T = log_probs.size(-2);
        const int64_t C = log_probs.size(-1);
        const auto tgt = targets.reshape({-1}).to(torch::kLong);
        const int64_t L = tgt.size(0);

        // Validation (mirrors the Python guard).
        if (L > 0) {
            if ((tgt == blank).any().item<bool>()) {
                return {handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}),
                                                                           "targets must not contain the blank index."),
                        torch::empty({1})};
            }
            if (tgt.max().item<int64_t>() >= C) {
                return {handle_exceptions<tensor_t, std::invalid_argument>(
                                torch::empty({1}), "max(targets) must be < the number of classes."),
                        torch::empty({1})};
            }
        }

        const auto tgt_acc = tgt.accessor<int64_t, 1>();
        auto label = [&](int64_t s) -> int64_t { return (s % 2 == 0) ? blank : tgt_acc[(s - 1) / 2]; };

        const int64_t S = 2 * L + 1;
        const auto lp = log_probs.reshape({T, C}).to(torch::kFloat).contiguous();
        const auto acc = lp.accessor<float, 2>();
        constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

        std::vector<double> prev(S, NEG_INF), cur(S, NEG_INF);
        std::vector<std::vector<int64_t>> backptr(T, std::vector<int64_t>(S, 0));

        prev[0] = acc[0][label(0)];
        if (S > 1) {
            prev[1] = acc[0][label(1)];
        }
        for (int64_t t = 1; t < T; ++t) {
            for (int64_t s = 0; s < S; ++s) {
                const int64_t lab = label(s);
                double best = prev[s];
                int64_t bp = s;
                if (s - 1 >= 0 && prev[s - 1] > best) {
                    best = prev[s - 1];
                    bp = s - 1;
                }
                const bool can_skip = s - 2 >= 0 && lab != blank && lab != label(s - 2);
                if (can_skip && prev[s - 2] > best) {
                    best = prev[s - 2];
                    bp = s - 2;
                }
                cur[s] = best + acc[t][lab];
                backptr[t][s] = bp;
            }
            std::swap(prev, cur);
        }

        // Pick the better terminal state (last blank vs last token) and backtrack.
        int64_t s = S - 1;
        if (S > 1 && prev[S - 2] > prev[S - 1]) {
            s = S - 2;
        }
        std::vector<int64_t> state_path(T, 0);
        state_path[T - 1] = s;
        for (int64_t t = T - 1; t >= 1; --t) {
            state_path[t - 1] = backptr[t][state_path[t]];
        }

        std::vector<int64_t> path_labels(T);
        for (int64_t t = 0; t < T; ++t) {
            path_labels[t] = label(state_path[t]);
        }
        const auto labels = torch::tensor(path_labels, torch::kLong);
        const auto scores_row = log_probs.reshape({T, C}).index({torch::arange(T), labels});
        return {labels.unsqueeze(0), scores_row.unsqueeze(0)};
    }

    // Collapse a forced-alignment frame sequence into token spans (mirrors merge_tokens).
    inline auto merge_tokens(const_tensor_lref_t tokens, const_tensor_lref_t scores, int64_t blank = 0)
            -> std::vector<token_span> {
        const auto tok = tokens.reshape({-1}).to(torch::kLong);
        const int64_t T = tok.size(0);
        const auto acc = tok.accessor<int64_t, 1>();
        std::vector<token_span> spans;
        int64_t i = 0;
        while (i < T) {
            const int64_t cur = acc[i];
            int64_t j = i;
            while (j < T && acc[j] == cur) {
                ++j;
            }
            if (cur != blank) {
                const double sc = scores.reshape({-1}).slice(0, i, j).mean().item<double>();
                spans.push_back(token_span{cur, i, j, sc});
            }
            i = j;
        }
        return spans;
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
