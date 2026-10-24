#pragma once
#ifndef LIB_TORCH_MEDIA_METRIC_COMMON_HPP
#define LIB_TORCH_MEDIA_METRIC_COMMON_HPP
// Shared pieces of torchmedia::metric (torchmetrics 1.9 parity, develop_log/2026-10-23/progress11): the stateful-class
// base, torchmetrics' reduce(), input checks and numpy's float64 summation order.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "../global_include.hpp"

// Scalar math that mirrors Python / C float arithmetic must not be fused into FMAs (same definition as in
// _vision/_functional.hpp; first statement of such function bodies).
#ifndef TORCHMEDIA_FP_CONTRACT_OFF
#if defined(__clang__)
#define TORCHMEDIA_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define TORCHMEDIA_FP_CONTRACT_OFF
#endif
#endif

namespace torchmedia::metric {
    // torchmetrics' reduction argument ("elementwise_mean" / "sum" / "none"; TotalVariation, SCC and VIF call the mean
    // "mean").
    enum class reduction_mode { elementwise_mean, sum, none };

    namespace detail {
        inline auto fail(std::string msg) -> void {
            (void) handle_exceptions<int, std::invalid_argument>(std::move(msg));
        }

        // `return invalid(msg);` throws std::invalid_argument, or returns T{} with LIB_TORCH_MEDIA_NO_EXCEPTIONS.
        template<class T = tensor_t>
        inline auto invalid(std::string msg) -> T {
            return handle_exceptions<T, std::invalid_argument>(std::move(msg));
        }

        // torchmetrics.utilities.distributed.reduce
        inline auto reduce(const tensor_t &x, reduction_mode r) -> tensor_t {
            switch (r) {
                case reduction_mode::elementwise_mean:
                    return torch::mean(x);
                case reduction_mode::sum:
                    return torch::sum(x);
                default:
                    return x;
            }
        }

        // torchmetrics _check_same_shape (a RuntimeError there): the error message, empty when the shapes agree.
        inline auto shape_error(const char *fn, const tensor_t &preds, const tensor_t &target) -> std::string {
            if (preds.sizes() == target.sizes()) {
                return {};
            }
            return std::format("{}: predictions and targets are expected to have the same shape, but got {} and {}.",
                               fn, c10::str(preds.sizes()), c10::str(target.sizes()));
        }

        // torchmetrics _safe_divide (float32 for integer inputs); zero_division: the value of x / 0, NaN = plain
        // true division.
        inline auto safe_divide(const tensor_t &num, const tensor_t &denom, double zero_division = 0.0) -> tensor_t {
            const auto n = num.is_floating_point() ? num : num.to(torch::kFloat32);
            const auto d = denom.is_floating_point() ? denom : denom.to(torch::kFloat32);
            if (std::isnan(zero_division)) {
                return torch::true_divide(n, d);
            }
            return torch::where(d != 0, n / d, torch::full({}, zero_division, n.options()));
        }

        // Metric.compute() squeezes a one-element result to 0-d (torchmetrics _squeeze_if_scalar).
        inline auto squeeze_if_scalar(const tensor_t &x) -> tensor_t { return x.numel() == 1 ? x.squeeze() : x; }

        // numpy's float64 add.reduce of a contiguous array (np.sum / np.mean): pairwise summation (8 accumulators up to
        // 128 elements) inside each 8192-element buffer chunk, chunks added in order (measured on numpy 1.26).
        inline auto pairwise_sum(const double *a, int64_t n) -> double {
            if (n < 8) {
                double res = -0.0;
                for (int64_t i = 0; i < n; ++i) {
                    res += a[i];
                }
                return res;
            }
            if (n <= 128) {
                std::array<double, 8> r{};
                std::copy(a, a + 8, r.begin());
                int64_t i = 8;
                for (; i < n - (n % 8); i += 8) {
                    for (int j = 0; j < 8; ++j) {
                        r[j] += a[i + j];
                    }
                }
                double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; ++i) {
                    res += a[i];
                }
                return res;
            }
            int64_t n2 = n / 2;
            n2 -= n2 % 8;
            return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
        }

        inline auto np_sum(const std::vector<double> &a) -> double {
            double res = 0.0;
            const auto n = static_cast<int64_t>(a.size());
            for (int64_t lo = 0; lo < n; lo += 8192) {
                res += pairwise_sum(a.data() + lo, std::min<int64_t>(8192, n - lo));
            }
            return res;
        }

        // Runs f; true when it failed in the LIB_TORCH_MEDIA_NO_EXCEPTIONS build (last_error() then holds its
        // message, else the earlier message is kept). The throwing build just runs f (a failure throws).
        template<class F>
        inline auto silently_failed(F &&f) -> bool {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
            auto &slot = torchmedia::detail::last_error_slot();
            auto earlier = std::exchange(slot, std::string{});
            f();
            if (!slot.empty()) {
                return true;
            }
            slot = std::move(earlier);
#else
            f();
#endif
            return false;
        }

        // Stateful-class base (torchmetrics Metric): forward(args) / operator()(args) update the accumulated state and
        // return the metric of this batch alone; a rejected batch (LIB_TORCH_MEDIA_NO_EXCEPTIONS) returns the
        // value-initialised result. Derived provides update, compute and options() (its option struct, from which a
        // fresh instance is built; a class without options is default-constructed).
        template<class Derived>
        class metric_base {
        public:
            template<class... Args>
            auto forward(const Args &...args) {
                auto &self = static_cast<Derived &>(*this);
                Derived batch = fresh(self);
                if (silently_failed([&] { batch.update(args...); }) || silently_failed([&] { self.update(args...); })) {
                    return decltype(batch.compute()){};
                }
                return batch.compute();
            }
            template<class... Args>
            auto operator()(const Args &...args) {
                return forward(args...);
            }
            auto reset() -> void {
                auto &self = static_cast<Derived &>(*this);
                self = fresh(self);
            }

        private:
            static auto fresh(const Derived &self) -> Derived {
                if constexpr (requires { self.options(); }) {
                    return Derived(self.options());
                } else {
                    return Derived();
                }
            }
        };
    } // namespace detail
} // namespace torchmedia::metric
#endif // LIB_TORCH_MEDIA_METRIC_COMMON_HPP
