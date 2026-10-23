#pragma once
// Minimal header-only assertion utilities for TorchMedia unit tests (no third-party test library).
// Usage: a test executable calls TM_CHECK* macros, then `return tm_test::summary("name");` from main().
// A non-zero return code marks the ctest case as failed.
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <torch/torch.h>

namespace tm_test {
    inline int g_checks = 0;
    inline int g_failures = 0;

    inline void record(bool ok, const char *expr, const char *file, int line, const std::string &detail = "") {
        ++g_checks;
        if (!ok) {
            ++g_failures;
            std::fprintf(stderr, "  [FAIL] %s:%d  %s%s%s\n", file, line, expr, detail.empty() ? "" : "  -> ",
                         detail.c_str());
        }
    }

    inline bool tensor_close(const torch::Tensor &a, const torch::Tensor &b, double atol, double rtol,
                             std::string &detail) {
        if (a.sizes() != b.sizes()) {
            detail = "shape mismatch";
            return false;
        }
        // to CPU first: some devices (MPS) cannot hold float64.
        const auto af = a.detach().to(torch::kCPU).to(torch::kDouble);
        const auto bf = b.detach().to(torch::kCPU).to(torch::kDouble);
        if (torch::allclose(af, bf, rtol, atol))
            return true;
        const double maxdiff = (af - bf).abs().max().template item<double>();
        detail = "max|delta|=" + std::to_string(maxdiff) + " (atol=" + std::to_string(atol) + ")";
        return false;
    }

    // Run one test function; an unexpected exception becomes a recorded failure instead of aborting the binary.
    template<class Fn>
    inline void run(const char *name, Fn &&fn) {
        try {
            fn();
        } catch (const std::exception &e) {
            const std::string what = e.what();
            record(false, name, __FILE__, __LINE__, "unexpected exception: " + what.substr(0, what.find('\n')));
        }
    }

    inline int summary(const char *name) {
        std::fprintf(stderr, "%s: %d/%d checks passed%s\n", name, g_checks - g_failures, g_checks,
                     g_failures ? "  <<< FAILED" : "  OK");
        return g_failures == 0 ? 0 : 1;
    }
} // namespace tm_test

#define TM_CHECK(cond) ::tm_test::record((cond), #cond, __FILE__, __LINE__)

#define TM_RUN(test_fn) ::tm_test::run(#test_fn, test_fn)

#define TM_CHECK_CLOSE(a, b, atol)                                                                                     \
    ::tm_test::record(std::abs(static_cast<double>(a) - static_cast<double>(b)) <= (atol), #a " ~= " #b, __FILE__,     \
                      __LINE__,                                                                                        \
                      "|delta|=" + std::to_string(std::abs(static_cast<double>(a) - static_cast<double>(b))))

#define TM_CHECK_TENSOR_CLOSE(a, b, atol, rtol)                                                                        \
    do {                                                                                                               \
        std::string _tm_detail;                                                                                        \
        const bool _tm_ok = ::tm_test::tensor_close((a), (b), (atol), (rtol), _tm_detail);                             \
        ::tm_test::record(_tm_ok, #a " ~= " #b, __FILE__, __LINE__, _tm_detail);                                       \
    } while (0)
