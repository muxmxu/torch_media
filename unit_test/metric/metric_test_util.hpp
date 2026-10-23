#pragma once
// Shared helpers of the torchmedia::metric tests: golden entries as tensors and the splitmix64 input generators that
// mirror gen_golden.py (image_pair / seg_pair). Include after a golden_*.hpp.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <vector>
#include "test_util.hpp"

// The goldens come from an arm64 run, where every op here matches bit for bit. Other CPUs may take other torch
// kernels (vectorised reductions / convolutions), so elsewhere allow a float32-level difference.
#if defined(__aarch64__)
constexpr double kTol = 0.0;
#else
constexpr double kTol = 1e-5;
#endif

template<class E>
static auto G(const E &e) -> torch::Tensor {
    return torch::tensor(e.data, torch::kFloat64).reshape(e.shape);
}

static auto dtype_name(torch::Dtype d) -> std::string {
    const std::string s = c10::toString(d);
    for (const auto &[c10_name, torch_name]: std::vector<std::pair<std::string, std::string>>{{"Float", "float32"},
                                                                                              {"Double", "float64"},
                                                                                              {"Half", "float16"},
                                                                                              {"Long", "int64"},
                                                                                              {"Int", "int32"},
                                                                                              {"Short", "int16"},
                                                                                              {"Byte", "uint8"},
                                                                                              {"Bool", "bool"}}) {
        if (s == c10_name) {
            return torch_name;
        }
    }
    return s;
}

// out must match the golden entry in dtype (when checked), shape and values (NaN == NaN).
template<class E>
static void check_golden(const torch::Tensor &out, const E &e, const std::string &name, int line, double tol = kTol,
                         bool check_dtype = true) {
    if (check_dtype) {
        const auto dt = dtype_name(out.scalar_type());
        tm_test::record(dt == e.dtype, name.c_str(), __FILE__, line, "dtype " + dt + " != " + e.dtype);
    }
    const auto a = out.detach().to(torch::kCPU).to(torch::kFloat64);
    // 9-digit float32 goldens round-trip only once rounded to float32 again.
    const torch::Tensor b = G(e).to(out.is_floating_point() ? out.scalar_type() : torch::kFloat64).to(torch::kFloat64);
    if (a.sizes() != b.sizes()) {
        tm_test::record(false, name.c_str(), __FILE__, line,
                        "shape " + c10::str(a.sizes()) + " != " + c10::str(b.sizes()));
        return;
    }
    const bool ok = torch::allclose(a, b, 0.0, tol, true);
    const torch::Tensor finite = torch::isfinite(a) & torch::isfinite(b);
    const double d = finite.any().item<bool>() ? (a - b).abs().masked_select(finite).max().item<double>() : 0.0;
    tm_test::record(ok, name.c_str(), __FILE__, line, "max|delta|=" + std::to_string(d));
}
#define CHECK_GOLDEN(out, name) check_golden((out), golden::name, #name, __LINE__)
#define CHECK_GOLDEN_TOL(out, name, tol) check_golden((out), golden::name, #name, __LINE__, (tol))
#define CHECK_GOLDEN_VALUES(out, name) check_golden((out), golden::name, #name, __LINE__, kTol, false)

template<class Fn>
static auto throws_invalid(Fn &&fn) -> bool {
    try {
        (void) fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

template<class Fn>
static auto throws_any(Fn &&fn) -> bool {
    try {
        (void) fn();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

// ---------------- splitmix64 inputs (gen_golden.py) ----------------
static auto splitmix_u(int64_t n, uint64_t seed) -> std::vector<double> {
    std::vector<double> u(n);
    for (int64_t i = 0; i < n; ++i) {
        uint64_t z = seed + static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        u[i] = static_cast<double>(z >> 11) * 0x1p-53;
    }
    return u;
}

// (preds, target) float64: target = 0.6 * (8 x 8 block noise) + 0.3 * noise + 0.05, preds = clip(target + 0.15 *
// (noise - 0.5), 0, 1), evaluated in numpy's order without FMA contraction.
inline auto image_pair(const std::vector<int64_t> &shape, uint64_t seed) -> std::pair<torch::Tensor, torch::Tensor> {
    TORCHMEDIA_FP_CONTRACT_OFF
    int64_t n = 1;
    for (const auto s: shape) {
        n *= s;
    }
    const int64_t h = shape[shape.size() - 2], w = shape.back();
    const auto fine = splitmix_u(n, seed), coarse = splitmix_u(n, seed + 1), noise = splitmix_u(n, seed + 2);
    std::vector<double> target(n), preds(n);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t plane = i / (h * w), y = i / w % h, x = i % w;
        const double block = coarse[plane * h * w + (y / 8) * 8 * w + (x / 8) * 8];
        const double a = 0.6 * block;
        const double b = 0.3 * fine[i];
        target[i] = a + b + 0.05;
        const double c = 0.15 * (noise[i] - 0.5);
        preds[i] = std::min(std::max(target[i] + c, 0.0), 1.0);
    }
    return {torch::tensor(preds, torch::kFloat64).reshape(shape),
            torch::tensor(target, torch::kFloat64).reshape(shape)};
}

// index (preds, target) int64 [n, h, w]; sample s draws classes 0 .. k_s - 1.
inline auto seg_pair(int64_t n, int64_t c, int64_t h, int64_t w, uint64_t seed)
        -> std::pair<torch::Tensor, torch::Tensor> {
    const std::vector<int64_t> pattern{c, std::max<int64_t>(1, c - 2), std::max<int64_t>(1, c - 1),
                                       std::max<int64_t>(1, c - 3)};
    const auto u1 = splitmix_u(n * h * w, seed), u2 = splitmix_u(n * h * w, seed + 1),
               u3 = splitmix_u(n * h * w, seed + 2);
    std::vector<int64_t> preds(n * h * w), target(n * h * w);
    for (int64_t i = 0; i < n * h * w; ++i) {
        const auto k = static_cast<double>(pattern[(i / (h * w)) % 4]);
        target[i] = static_cast<int64_t>(std::floor(u1[i] * k));
        preds[i] = u2[i] < 0.6 ? target[i] : static_cast<int64_t>(std::floor(u3[i] * k));
    }
    return {torch::tensor(preds).reshape({n, h, w}), torch::tensor(target).reshape({n, h, w})};
}

// gen_golden.py checksum: {sum, x[0], x[n / 2], x[-1]} of the flattened float64 values.
template<class E>
static void check_checksum(const torch::Tensor &x, const E &e, const char *name, int line) {
    const auto f = x.to(torch::kFloat64).flatten();
    const auto got = torch::stack({f.sum(), f[0], f[f.numel() / 2], f[-1]});
    const torch::Tensor want = G(e);
    tm_test::record(torch::allclose(got.slice(0, 1), want.slice(0, 1), 0.0, 0.0) &&
                            std::abs(got[0].item<double>() - want[0].item<double>()) <= 1e-9,
                    name, __FILE__, line, "generator mismatch");
}
#define CHECK_CHECKSUM(x, name) check_checksum((x), golden::name, #name, __LINE__)
