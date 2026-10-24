#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
#define LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
#include <algorithm>
#include <limits>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "../global_include.hpp"

// CTC forced alignment (mirrors torchaudio.functional.forced_align / merge_tokens / TokenSpan).
// Upstream dispatches to a compiled op; here its CPU kernel is reimplemented as a header-only Viterbi DP over the
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

    // Viterbi forced alignment (mirrors torchaudio.functional.forced_align and its compiled CPU kernel, compute.cpp:
    // windowed trellis, tie rules, terminal rule, accumulation in log_probs' dtype, the same checks).
    // log_probs: (1, T, C) float32/float64/float16 log-probabilities; targets: (1, L) int32/int64.
    // Returns (paths, scores), both (1, T): the label per frame (targets' dtype) and its log-prob.
    // Relaxed: non-contiguous or non-CPU log_probs are copied (torchaudio's CPU op raises). Stricter: negative targets
    // raise (torchaudio reads out of bounds).
    inline auto forced_align(const_tensor_lref_t log_probs, const_tensor_lref_t targets,
                             const c10::optional<tensor_t> &input_lengths = c10::nullopt,
                             const c10::optional<tensor_t> &target_lengths = c10::nullopt, int64_t blank = 0)
            -> std::pair<tensor_t, tensor_t> {
        const auto fail = [](const str_t &msg) -> std::pair<tensor_t, tensor_t> {
            return {handle_exceptions<tensor_t, std::invalid_argument>(torch::empty({1}), msg), torch::empty({1})};
        };
        const auto lp_type = log_probs.scalar_type();
        const auto tg_type = targets.scalar_type();
        if (log_probs.dim() != 3 || targets.dim() != 2 || log_probs.size(0) != 1 || targets.size(0) != 1) {
            return fail("log_probs must be (1, T, C) and targets (1, L): only batch size 1 is supported.");
        }
        if (!(lp_type == torch::kFloat || lp_type == torch::kDouble || lp_type == torch::kHalf) ||
            !(tg_type == torch::kInt || tg_type == torch::kLong)) {
            return fail("log_probs must be float64, float32 or float16, and targets int32 or int64.");
        }
        const int64_t T = log_probs.size(1), C = log_probs.size(2), L = targets.size(1);
        if (L == 0) { // torchaudio: torch.max on the empty targets raises
            return fail("targets must not be empty.");
        }
        if (blank < 0 || blank >= C) {
            return fail("blank must be within [0, num classes).");
        }
        const auto tgt = targets[0].to(torch::kCPU, torch::kLong).contiguous();
        if ((tgt == blank).any().item<bool>()) {
            return fail("targets must not contain the blank index.");
        }
        if (tgt.max().item<int64_t>() >= C || tgt.min().item<int64_t>() < 0) {
            return fail("targets values must be in [0, num classes).");
        }
        const auto length_is = [](const c10::optional<tensor_t> &lengths, int64_t expected) {
            return !lengths.has_value() ||
                   (lengths->dim() == 1 && lengths->numel() > 0 && lengths->max().item<int64_t>() == expected);
        };
        if (!length_is(input_lengths, T) || !length_is(target_lengths, L)) {
            return fail("input/target length mismatch: max(lengths) must equal log_probs.size(1) / targets.size(1).");
        }
        const auto tg = tgt.accessor<int64_t, 1>();
        int64_t R = 0; // repeated adjacent targets, which need a blank in between
        for (int64_t i = 1; i < L; ++i) {
            R += tg[i] == tg[i - 1] ? 1 : 0;
        }
        if (T < L + R) {
            return fail("targets length is too long for CTC: need log_probs length >= targets length + repeats.");
        }

        const int64_t S = 2 * L + 1;
        const auto label = [&](int64_t s) -> int64_t { return s % 2 == 0 ? blank : tg[s / 2]; };
        const auto lp = log_probs[0].to(torch::kCPU).contiguous(); // (T, C)
        std::vector<int64_t> path(T);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(lp_type, "forced_align", [&] {
            const auto lpa = lp.accessor<scalar_t, 2>();
            const scalar_t ninf = -std::numeric_limits<scalar_t>::infinity();
            std::vector<scalar_t> alphas(2 * S, ninf); // rows t % 2
            std::vector<int8_t> back(T * S, -1); // 0: stay, 1: from s - 1, 2: skip from s - 2
            int64_t start = T - (L + R) > 0 ? 0 : 1, end = 2; // states outside [start, end) cannot be on a path
            for (int64_t i = start; i < end; ++i) {
                alphas[i] = lpa[0][label(i)];
            }
            for (int64_t t = 1; t < T; ++t) {
                if (T - t <= L + R) {
                    if (start % 2 == 1 && start / 2 + 1 < L && tg[start / 2] != tg[start / 2 + 1]) {
                        ++start;
                    }
                    ++start;
                }
                if (t <= L + R) {
                    if (end % 2 == 0 && end < 2 * L && tg[end / 2 - 1] != tg[end / 2]) {
                        ++end;
                    }
                    ++end;
                }
                scalar_t *cur = alphas.data() + (t % 2) * S;
                const scalar_t *prev = alphas.data() + ((t - 1) % 2) * S;
                std::fill(cur, cur + S, ninf);
                int64_t first = start;
                if (start == 0) {
                    cur[0] = prev[0] + lpa[t][blank];
                    back[t * S] = 0;
                    first = 1;
                }
                for (int64_t i = first; i < end; ++i) {
                    const scalar_t x0 = prev[i], x1 = prev[i - 1];
                    const scalar_t x2 = i % 2 != 0 && i != 1 && tg[i / 2] != tg[i / 2 - 1] ? prev[i - 2] : ninf;
                    // torchaudio's tie rule: move only on a strictly larger score.
                    const int8_t b = x2 > x1 && x2 > x0 ? 2 : (x1 > x0 && x1 > x2 ? 1 : 0);
                    back[t * S + i] = b;
                    cur[i] = (b == 2 ? x2 : (b == 1 ? x1 : x0)) + lpa[t][label(i)];
                }
            }
            const scalar_t *last = alphas.data() + ((T - 1) % 2) * S;
            int64_t s = last[S - 1] > last[S - 2] ? S - 1 : S - 2; // a tie ends on the last token
            for (int64_t t = T - 1; t >= 0; --t) {
                path[t] = label(s);
                s -= back[t * S + s];
            }
        });
        const auto labels = torch::tensor(path, torch::kLong);
        const auto scores = lp.index({torch::arange(T), labels}).unsqueeze(0);
        return {labels.to(targets.device(), tg_type).unsqueeze(0), scores.to(log_probs.device())};
    }

    // Collapse a forced-alignment frame sequence into token spans (mirrors merge_tokens). tokens and scores must be
    // 1-D and of equal length, as in torchaudio (pass paths[0] / scores[0] of forced_align). Breaking change: a
    // (1, T) input used to be flattened, which silently merged spans across the rows of a (B, T) input.
    inline auto merge_tokens(const_tensor_lref_t tokens, const_tensor_lref_t scores, int64_t blank = 0)
            -> std::vector<token_span> {
        if (tokens.dim() != 1 || scores.dim() != 1 || tokens.size(0) != scores.size(0)) {
            return handle_exceptions<std::vector<token_span>, std::invalid_argument>(
                    "`tokens` and `scores` must be 1D Tensors of the same length.");
        }
        const auto tok = tokens.to(torch::kCPU, torch::kLong).contiguous();
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
                const double sc = scores.slice(0, i, j).mean().item<double>();
                spans.push_back(token_span{cur, i, j, sc});
            }
            i = j;
        }
        return spans;
    }
} // namespace torchmedia::audio::functional
#endif // LIB_TORCH_MEDIA_AUDIO_FUNCTIONAL_ALIGNMENT_HPP
