// Assertion-based tests for the vision functional layer against torchvision 0.20.1 transforms.v2.functional
// goldens (gen_golden.py -> golden.hpp: the seeded inputs and every expected output, bit for bit).
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "golden.hpp"
#include "test_util.hpp"

using namespace torchmedia::vision::functional;
namespace F = torchmedia::vision::functional; // F::resize / F::pad: ADL also finds at::resize / at::pad
using IM = interpolation_mode;
using PM = padding_mode;

// The goldens come from an arm64 run, where every op here is bit-exact. Other CPUs may take other torch kernels
// (x86 AVX2 resizes uint8 natively, FMA contraction differs), so elsewhere allow one integer step / 1e-5.
#if defined(__aarch64__)
constexpr double kTolInt = 0.0;
constexpr double kTolFloat = 0.0;
#else
constexpr double kTolInt = 1.0;
constexpr double kTolFloat = 1e-5;
#endif

static auto dtype_name(torch::Dtype d) -> std::string {
    switch (d) {
        case torch::kUInt8:
            return "uint8";
        case torch::kInt8:
            return "int8";
        case torch::kInt16:
            return "int16";
        case torch::kInt32:
            return "int32";
        case torch::kInt64:
            return "int64";
        case torch::kFloat32:
            return "float32";
        case torch::kFloat64:
            return "float64";
        default:
            return c10::toString(d);
    }
}

static auto G(const golden::entry &e) -> torch::Tensor {
    return torch::tensor(e.data, torch::kFloat64).reshape(e.shape);
}

// out must match the golden entry in dtype, shape and values.
static void check_golden(const torch::Tensor &out, const golden::entry &e, const std::string &name, int line) {
    const auto dt = dtype_name(out.scalar_type());
    tm_test::record(dt == e.dtype, name.c_str(), __FILE__, line, "dtype " + dt + " != " + e.dtype);
    std::string detail;
    const double tol = out.is_floating_point() ? kTolFloat : kTolInt;
    // 9-digit float32 goldens round-trip only once rounded to float32 again.
    const auto expected = G(e).to(out.scalar_type());
    tm_test::record(tm_test::tensor_close(out, expected, tol, 0.0, detail), name.c_str(), __FILE__, line, detail);
}
#define CHECK_GOLDEN(out, name) check_golden((out), golden::name, #name, __LINE__)

template<class Fn>
static auto throws_invalid(Fn &&fn) -> bool {
    try {
        (void) fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static const auto S = G(golden::in_s).to(torch::kUInt8); // [3, 5, 7]
static const auto L = G(golden::in_l).to(torch::kUInt8); // [3, 11, 13]
static const auto B = G(golden::in_b).to(torch::kUInt8); // [2, 3, 6, 10]
static auto f32(const torch::Tensor &x) -> torch::Tensor { return x.to(torch::kFloat32) / 255; }
static auto f64(const torch::Tensor &x) -> torch::Tensor { return x.to(torch::kFloat64) / 255; }

static void test_resize_modes() {
    struct mode_case {
        const char *name;
        IM mode;
        bool antialias;
        const golden::entry &up_u8, &down_u8, &up_f32, &down_f32;
    };
    using namespace golden;
    const mode_case cases[] = {
            {"nearest", IM::nearest, true, resize_nearest_up_u8, resize_nearest_down_u8, resize_nearest_up_f32,
             resize_nearest_down_f32},
            {"nearest_exact", IM::nearest_exact, true, resize_nearest_exact_up_u8, resize_nearest_exact_down_u8,
             resize_nearest_exact_up_f32, resize_nearest_exact_down_f32},
            {"bilinear", IM::bilinear, true, resize_bilinear_up_u8, resize_bilinear_down_u8, resize_bilinear_up_f32,
             resize_bilinear_down_f32},
            {"bilinear_noaa", IM::bilinear, false, resize_bilinear_noaa_up_u8, resize_bilinear_noaa_down_u8,
             resize_bilinear_noaa_up_f32, resize_bilinear_noaa_down_f32},
            {"bicubic", IM::bicubic, true, resize_bicubic_up_u8, resize_bicubic_down_u8, resize_bicubic_up_f32,
             resize_bicubic_down_f32},
            {"bicubic_noaa", IM::bicubic, false, resize_bicubic_noaa_up_u8, resize_bicubic_noaa_down_u8,
             resize_bicubic_noaa_up_f32, resize_bicubic_noaa_down_f32},
    };
    for (const auto &c: cases) {
        const std::string n = std::string("resize_") + c.name;
        const auto r = [&](const torch::Tensor &x, std::vector<int64_t> size) {
            return F::resize(x, size, c.mode, std::nullopt, c.antialias);
        };
        check_golden(r(S, {8, 11}), c.up_u8, n + "_up_u8", __LINE__);
        check_golden(r(L, {4, 5}), c.down_u8, n + "_down_u8", __LINE__);
        check_golden(r(f32(S), {8, 11}), c.up_f32, n + "_up_f32", __LINE__);
        check_golden(r(f32(L), {4, 5}), c.down_f32, n + "_down_f32", __LINE__);
    }
    CHECK_GOLDEN(F::resize(f64(L), {4, 5}, IM::bilinear), resize_bilinear_down_f64);
    CHECK_GOLDEN(F::resize(f64(L), {4, 5}, IM::bicubic), resize_bicubic_down_f64);
    CHECK_GOLDEN(F::resize(f64(S), {8, 11}, IM::bilinear, std::nullopt, false), resize_bilinear_noaa_up_f64);
    CHECK_GOLDEN(F::resize(f64(S), {8, 11}, IM::bicubic, std::nullopt, false), resize_bicubic_noaa_up_f64);
}

static void test_resize_batch_and_sizes() {
    CHECK_GOLDEN(F::resize(B, {4, 7}), resize_batch_bilinear_u8);
    CHECK_GOLDEN(F::resize(f32(B), {9, 4}, IM::bicubic), resize_batch_bicubic_f32);
    // more leading dims: the same pixels, reshaped
    const auto b5 = F::resize(f32(B).reshape({2, 1, 3, 6, 10}), {9, 4}, IM::bicubic);
    TM_CHECK(b5.sizes() == torch::IntArrayRef({2, 1, 3, 9, 4}));
    TM_CHECK(torch::equal(b5.reshape({2, 3, 9, 4}), F::resize(f32(B), {9, 4}, IM::bicubic)));

    const auto tall = f32(L).slice(-1, 0, 5); // [3, 11, 5]
    CHECK_GOLDEN(F::resize(f32(L), {6}), resize_short_f32);
    CHECK_GOLDEN(F::resize(tall, {4}, IM::bilinear, 6), resize_max_size_f32);
    CHECK_GOLDEN(F::resize(tall, {}, IM::bilinear, 6), resize_size_none_f32);
    TM_CHECK(F::resize(tall, {4}, IM::bilinear, 9).sizes() == torch::IntArrayRef({3, 8, 4})); // under the cap

    // same size: the input itself; empty batch: reshaped without interpolating
    const auto s = f32(S);
    TM_CHECK(F::resize(s, {5, 7}).is_same(s));
    TM_CHECK(F::resize(torch::zeros({0, 3, 5, 7}), {8, 11}).sizes() == torch::IntArrayRef({0, 3, 8, 11}));
    // integer dtypes other than uint8 go through float32 and are rounded back (torchvision need_cast)
    const auto i16 = F::resize(S.to(torch::kInt16), {8, 11}, IM::bicubic, std::nullopt, false);
    TM_CHECK(i16.scalar_type() == torch::kInt16);
    TM_CHECK(torch::equal(
            i16,
            F::resize(S.to(torch::kFloat32), {8, 11}, IM::bicubic, std::nullopt, false).round().to(torch::kInt16)));
    CHECK_GOLDEN(F::resize(S.bitwise_right_shift(1).to(torch::kInt8) - 64, {8, 11}, IM::bicubic, std::nullopt, false),
                 resize_bicubic_noaa_up_i8);
    CHECK_GOLDEN(F::resize(S.to(torch::kInt32) * 1000 - 128000, {8, 11}, IM::bicubic, std::nullopt, false),
                 resize_bicubic_noaa_up_i32);

    // validation (torchvision ValueError / ZeroDivisionError)
    TM_CHECK(throws_invalid([&] { return F::resize(torch::zeros({5, 7}), {3}); }));
    TM_CHECK(throws_invalid([&] { return F::resize(s, {3, 4, 5}); }));
    TM_CHECK(throws_invalid([&] { return F::resize(s, {3, 4}, IM::bilinear, 9); }));
    TM_CHECK(throws_invalid([&] { return F::resize(s, {4}, IM::bilinear, 4); }));
    TM_CHECK(throws_invalid([&] { return F::resize(s, {}); }));
    TM_CHECK(throws_invalid([&] { return F::resize(torch::zeros({3, 0, 5}), {4}); }));
    TM_CHECK(throws_invalid([&] { return F::resize(torch::zeros({3, 0, 0}), {}, IM::bilinear, 4); }));
    // a zero output size is left to torch, like torchvision
    bool c10_error = false;
    try {
        (void) F::resize(s, {0});
    } catch (const c10::Error &) {
        c10_error = true;
    }
    TM_CHECK(c10_error);
}

static void test_resize_devices() {
    // torchvision only interpolates uint8 natively on the CPU (bicubic) -- elsewhere it goes through float32.
    // MPS has no bicubic / antialias kernels in torch 2.5.1, so the bicubic clamp path runs on the meta device.
    const auto meta = F::resize(S.to(torch::kMeta), {8, 11}, IM::bicubic);
    TM_CHECK(meta.device().is_meta() && meta.scalar_type() == torch::kUInt8);
    TM_CHECK(meta.sizes() == torch::IntArrayRef({3, 8, 11}));
#if GOLDEN_HAS_MPS // the generator only writes the MPS goldens on a host with MPS
    if (torch::mps::is_available()) {
        const auto u8 = F::resize(S.to(torch::kMPS), {8, 11}, IM::bilinear, std::nullopt, false);
        TM_CHECK(u8.device().is_mps());
        CHECK_GOLDEN(u8.cpu(), resize_mps_bilinear_noaa_u8);
        CHECK_GOLDEN(F::resize(f32(S).to(torch::kMPS), {8, 11}, IM::bilinear, std::nullopt, false).cpu(),
                     resize_mps_bilinear_noaa_f32);
    }
#endif
}

static void test_crop() {
    CHECK_GOLDEN(crop(f32(S), 1, 2, 3, 4), crop_in_f32);
    CHECK_GOLDEN(crop(S, -2, -1, 5, 6), crop_oob_tl_u8);
    CHECK_GOLDEN(crop(f32(S), 3, 4, 4, 6), crop_oob_br_f32);
    CHECK_GOLDEN(crop(S[0], 1, 1, 2, 3), crop_2d_u8);
    TM_CHECK(throws_invalid([&] { return crop(torch::zeros({5}), 0, 0, 1, 1); }));
    TM_CHECK(throws_invalid([&] { return crop(torch::zeros({5, 5}), -1, 0, 3, 3); })); // padding needs [..., C, H, W]

    CHECK_GOLDEN(center_crop(S, {2, 2}), center_crop_2x2_u8);
    CHECK_GOLDEN(center_crop(S, {3, 4}), center_crop_3x4_u8);
    CHECK_GOLDEN(center_crop(f32(L), {4}), center_crop_one_f32);
    CHECK_GOLDEN(center_crop(S, {8, 10}), center_crop_pad_both_u8);
    CHECK_GOLDEN(center_crop(f32(S), {6, 7}), center_crop_pad_h_f32);
    CHECK_GOLDEN(center_crop(S, {8, 3}), center_crop_pad_h_crop_w_u8);
    TM_CHECK(center_crop(torch::zeros({0, 5, 7}), {2, 3}).sizes() == torch::IntArrayRef({0, 2, 3}));
    TM_CHECK(throws_invalid([&] { return center_crop(S, {}); }));
    TM_CHECK(throws_invalid([&] { return center_crop(S, {1, 2, 3}); }));
    TM_CHECK(throws_invalid([&] { return center_crop(torch::zeros({5}), {2}); }));

    CHECK_GOLDEN(resized_crop(f32(L), 1, 2, 7, 9, {5, 4}), resized_crop_f32);
    CHECK_GOLDEN(resized_crop(L, 2, 1, 8, 10, {13, 6}, IM::bicubic, false), resized_crop_bicubic_u8);
    TM_CHECK(throws_invalid([&] { return resized_crop(torch::zeros({5, 5}), 0, 0, 2, 2, {3}); }));
}

static void test_five_ten_crop() {
    const auto five = five_crop(S, {3, 4});
    TM_CHECK(five.size() == 5);
    CHECK_GOLDEN(torch::stack(five), five_crop_u8);
    const auto ten = ten_crop(S, {3, 4});
    TM_CHECK(ten.size() == 10);
    CHECK_GOLDEN(torch::stack(ten), ten_crop_u8);
    CHECK_GOLDEN(torch::stack(ten_crop(f32(S), {4}, true)), ten_crop_vflip_f32);
    for (const auto &bad: {std::vector<int64_t>{}, std::vector<int64_t>{1, 2, 3}, std::vector<int64_t>{6, 2},
                           std::vector<int64_t>{2, 8}}) {
        TM_CHECK(throws_invalid([&] { return five_crop(S, bad); }));
        TM_CHECK(throws_invalid([&] { return ten_crop(S, bad); }));
    }
    TM_CHECK(throws_invalid([&] { return five_crop(torch::zeros({5}), {1}); }));
}

static void test_pad() {
    const std::vector<int64_t> p4{1, 2, 3, 4};
    CHECK_GOLDEN(F::pad(S, p4, {7}), pad_constant_u8);
    CHECK_GOLDEN(F::pad(f32(S), p4, {0.5}), pad_constant_f32);
    CHECK_GOLDEN(F::pad(f32(S), {2}), pad_one_f32);
    CHECK_GOLDEN(F::pad(S, {1, 3}, {200}), pad_two_u8);
    CHECK_GOLDEN(F::pad(S, p4, {10, 20, 30}), pad_vector_u8);
    CHECK_GOLDEN(F::pad(f64(S), p4, {0.1, 0.2, 0.3}), pad_vector_f64);
    CHECK_GOLDEN(F::pad(f32(S), {-1, 2, 1, -2}, {0.25, 0.5, 0.75}), pad_vector_neg_f32);
    CHECK_GOLDEN(F::pad(S, p4, {}, PM::edge), pad_edge_u8);
    CHECK_GOLDEN(F::pad(f32(S), p4, {}, PM::edge), pad_edge_f32);
    CHECK_GOLDEN(F::pad(S, p4, {}, PM::reflect), pad_reflect_u8);
    CHECK_GOLDEN(F::pad(f32(S), p4, {}, PM::reflect), pad_reflect_f32);
    CHECK_GOLDEN(F::pad(S, p4, {}, PM::symmetric), pad_symmetric_u8);
    CHECK_GOLDEN(F::pad(f32(S), {-1, 2, 1, -2}, {}, PM::symmetric), pad_symmetric_neg_f32);
    CHECK_GOLDEN(F::pad(S, {-1, 2, 1, -2}), pad_constant_neg_u8);
    CHECK_GOLDEN(F::pad(f32(B), {2, 1}, {}, PM::reflect), pad_batch_reflect_f32);
    TM_CHECK(throws_invalid([&] { return F::pad(S, {1, 2, 3}); }));
    TM_CHECK(throws_invalid([&] { return F::pad(S, {1}, {1, 2, 3}, PM::edge); }));
    TM_CHECK(throws_invalid([&] { return F::pad(torch::zeros({5, 7}), {1}); }));
    // a symmetric pad wider than the image raises IndexError on every device (MPS would gather out of bounds)
    CHECK_GOLDEN(F::pad(S, {7, 5}, {}, PM::symmetric), pad_symmetric_full_u8);
    const auto devices = torch::mps::is_available() ? std::vector{torch::kCPU, torch::kMPS} : std::vector{torch::kCPU};
    for (const auto &dev: devices) {
        for (const auto &p: std::vector<std::vector<int64_t>>{{8, 0}, {0, 6}, {0, 0, 8, 0}, {0, 0, 0, 6}}) {
            bool index_error = false;
            try {
                (void) F::pad(f32(S).to(dev), p, {}, PM::symmetric);
            } catch (const c10::IndexError &) {
                index_error = true;
            }
            TM_CHECK(index_error);
        }
    }
    // torch only bounds-checks a non-empty gather: an empty batch or no channels pads like torchvision
    TM_CHECK(F::pad(torch::zeros({0, 3, 4, 4}), {10}, {}, PM::symmetric).sizes() == torch::IntArrayRef({0, 3, 24, 24}));
    TM_CHECK(F::pad(torch::zeros({1, 0, 4, 4}), {10}, {}, PM::symmetric).sizes() == torch::IntArrayRef({1, 0, 24, 24}));
}

static void test_to_dtype() {
    CHECK_GOLDEN(to_dtype(S, torch::kFloat32, true), to_dtype_u8_f32);
    CHECK_GOLDEN(to_dtype(S, torch::kFloat64, true), to_dtype_u8_f64);
    CHECK_GOLDEN(to_dtype(f32(L), torch::kUInt8, true), to_dtype_f32_u8);
    CHECK_GOLDEN(to_dtype(f64(L), torch::kUInt8, true), to_dtype_f64_u8);
    CHECK_GOLDEN(to_dtype(f32(S), torch::kInt16, true), to_dtype_f32_i16);
    CHECK_GOLDEN(to_dtype(f32(S), torch::kUInt16, true).to(torch::kInt32), to_dtype_f32_u16);
    const auto i16 = to_dtype(S, torch::kInt16, true);
    CHECK_GOLDEN(i16, to_dtype_u8_i16);
    CHECK_GOLDEN(to_dtype(i16 + 77, torch::kUInt8, true), to_dtype_i16_u8);
    CHECK_GOLDEN(to_dtype(S, torch::kUInt16, true).to(torch::kInt32), to_dtype_u8_u16);
    CHECK_GOLDEN(to_dtype((S.to(torch::kInt32) * 257).to(torch::kUInt16), torch::kUInt8, true), to_dtype_u16_u8);
    CHECK_GOLDEN(to_dtype(f32(S) * 250, torch::kUInt8), to_dtype_noscale_f32);
    // half / bfloat16 (goldens stored as float32): the 1 / max scale must not be rounded to the output dtype first
    const auto f16 = to_dtype(S, torch::kHalf, true);
    const auto bf16 = to_dtype(S, torch::kBFloat16, true);
    TM_CHECK(f16.scalar_type() == torch::kHalf && bf16.scalar_type() == torch::kBFloat16);
    CHECK_GOLDEN(f16.to(torch::kFloat32), to_dtype_u8_f16);
    CHECK_GOLDEN(bf16.to(torch::kFloat32), to_dtype_u8_bf16);
    CHECK_GOLDEN(to_dtype(S.to(torch::kInt32) * 200, torch::kHalf, true).to(torch::kFloat32), to_dtype_i32_f16);
    const auto s = f32(S);
    TM_CHECK(to_dtype(s, torch::kFloat32, true).is_same(s));
    TM_CHECK(torch::equal(to_dtype(s, torch::kFloat64, true), s.to(torch::kFloat64)));
    TM_CHECK(throws_invalid([&] { return to_dtype(s, torch::kInt32, true); }));
    TM_CHECK(throws_invalid([&] { return to_dtype(s, torch::kInt64, true); }));
    TM_CHECK(throws_invalid([&] { return to_dtype(f64(S), torch::kInt64, true); }));
    TM_CHECK(throws_invalid([&] { return to_dtype(S.to(torch::kBool), torch::kUInt8, true); }));
}

static void test_color() {
    CHECK_GOLDEN(rgb_to_grayscale(S), gray_u8);
    CHECK_GOLDEN(rgb_to_grayscale(f32(S), 3), gray3_f32);
    CHECK_GOLDEN(rgb_to_grayscale(f64(B)), gray_f64);
    if (torch::mps::is_available()) { // torch 2.5.1's MPS mul(uint8, float scalar) aborts the process
        CHECK_GOLDEN(rgb_to_grayscale(S.to(torch::kMPS)).cpu(), gray_u8);
    }
    const auto one = S.slice(-3, 0, 1);
    const auto g1 = rgb_to_grayscale(one);
    TM_CHECK(torch::equal(g1, one) && g1.data_ptr() != one.data_ptr()); // a copy
    TM_CHECK(torch::equal(rgb_to_grayscale(one, 3), one.repeat({3, 1, 1})));
    TM_CHECK(throws_invalid([&] { return rgb_to_grayscale(S, 2); }));
    TM_CHECK(throws_invalid([&] { return rgb_to_grayscale(B.slice(-3, 0, 2)); }));
    TM_CHECK(throws_invalid([&] { return rgb_to_grayscale(torch::zeros({5, 7})); }));

    CHECK_GOLDEN(adjust_brightness(S, 1.7), brightness_u8);
    CHECK_GOLDEN(adjust_brightness(f32(S), 1.7), brightness_f32);
    CHECK_GOLDEN(adjust_brightness(f64(S), 0.6), brightness_f64);
    TM_CHECK(throws_invalid([&] { return adjust_brightness(S, -0.1); }));
    TM_CHECK(throws_invalid([&] { return adjust_brightness(B.slice(-3, 0, 2), 1.0); }));

    CHECK_GOLDEN(invert(S), invert_u8);
    CHECK_GOLDEN(invert(f32(S)), invert_f32);
    CHECK_GOLDEN(invert(S.to(torch::kInt16) * 100 - 9000), invert_i16);
    CHECK_GOLDEN(invert(S.to(torch::kInt8)), invert_i8);
    CHECK_GOLDEN(invert(S.to(torch::kInt32) * 1000003 - 99999999), invert_i32);
    const auto i64 = S.to(torch::kInt64) * 1000; // near the int64 extremes: x ^ INT64_MAX stays exact as a double
    constexpr auto i64_max = std::numeric_limits<int64_t>::max();
    constexpr auto i64_min = std::numeric_limits<int64_t>::min();
    CHECK_GOLDEN(invert(torch::where(S.remainder(2) == 0, i64_max - i64, i64_min + i64)), invert_i64);
    TM_CHECK(throws_invalid([&] { return invert(S.to(torch::kBool)); }));

    const std::vector<double> mean{0.485, 0.456, 0.406}, stddev{0.229, 0.224, 0.225};
    CHECK_GOLDEN(normalize(f32(S), mean, stddev), normalize_f32);
    CHECK_GOLDEN(normalize(f64(B), mean, stddev), normalize_f64);
    CHECK_GOLDEN(normalize(f32(S), {0.5}, {0.25}), normalize_scalar_f32);
    auto x = f32(S);
    const auto y = normalize(x, mean, stddev, true);
    TM_CHECK(y.is_same(x)); // in place
    CHECK_GOLDEN(x, normalize_f32);
    TM_CHECK(throws_invalid([&] { return normalize(S, mean, stddev); }));
    TM_CHECK(throws_invalid([&] { return normalize(torch::zeros({5, 7}), {0.5}, {0.5}); }));
    TM_CHECK(throws_invalid([&] { return normalize(f32(S), mean, {0.2, 0.0, 0.2}); }));
}

static void test_flip() {
    // torchvision: flip(-1) / flip(-2) on any leading dims.
    const auto img = torch::arange(48, torch::kFloat32).reshape({3, 4, 4}) / 47.0;
    TM_CHECK_CLOSE(hflip(img)[0][0][0].item<double>(), 3.0 / 47.0, 1e-7);
    TM_CHECK_CLOSE(hflip(img)[0][0][3].item<double>(), 0.0, 0.0);
    TM_CHECK_CLOSE(vflip(img)[0][0][0].item<double>(), 12.0 / 47.0, 1e-7);
    TM_CHECK(torch::equal(vflip(B)[1][2][0], B[1][2][5]));
}

// ---------------- progress09: color ops ----------------
static void test_color_adjust() {
    const auto g1 = S.slice(-3, 0, 1);
    CHECK_GOLDEN(adjust_contrast(S, 0.6), contrast_u8);
    CHECK_GOLDEN(adjust_contrast(f32(S), 1.7), contrast_f32);
    CHECK_GOLDEN(adjust_contrast(f64(B), 0.4), contrast_f64);
    CHECK_GOLDEN(adjust_contrast(g1, 1.5), contrast_gray_u8);
    CHECK_GOLDEN(adjust_contrast(f32(g1), 0.5), contrast_gray_f32);
    CHECK_GOLDEN(adjust_saturation(S, 0.3), saturation_u8);
    CHECK_GOLDEN(adjust_saturation(f32(L), 1.8), saturation_f32);
    CHECK_GOLDEN(adjust_saturation(f64(B), 0.0), saturation_f64);
    TM_CHECK(adjust_saturation(g1, 0.5).is_same(g1)); // 1 channel: returned as is
    CHECK_GOLDEN(adjust_sharpness(L, 2.0), sharpness_u8);
    CHECK_GOLDEN(adjust_sharpness(f32(S), 0.4), sharpness_f32);
    CHECK_GOLDEN(adjust_sharpness(f64(L), 1.5), sharpness_f64);
    CHECK_GOLDEN(adjust_sharpness(f32(B).reshape({2, 1, 3, 6, 10}), 3.0), sharpness_5d_f32);
    const auto thin = S.slice(-2, 0, 2); // H <= 2: nothing to sharpen
    TM_CHECK(adjust_sharpness(thin, 2.0).is_same(thin));
    CHECK_GOLDEN(adjust_hue(S, 0.25), hue_u8);
    CHECK_GOLDEN(adjust_hue(f32(L), -0.4), hue_f32);
    CHECK_GOLDEN(adjust_hue(f64(B), 0.1), hue_f64);
    CHECK_GOLDEN(adjust_hue(B, -0.5), hue_batch_u8);
    TM_CHECK(adjust_hue(g1, 0.2).is_same(g1));
    const auto empty = torch::zeros({0, 3, 4, 4});
    TM_CHECK(adjust_hue(empty, 0.2).is_same(empty));
    CHECK_GOLDEN(adjust_gamma(S, 0.5), gamma_u8);
    CHECK_GOLDEN(adjust_gamma(S, 2.2, 1.3), gamma_gain_u8);
    CHECK_GOLDEN(adjust_gamma(f32(L), 1.8), gamma_f32);
    CHECK_GOLDEN(adjust_gamma(f64(B), 0.7, 0.8), gamma_gain_f64);
    // half / bfloat16 with a gain: an in-place mul_(Scalar) would round the gain to the image dtype first
    CHECK_GOLDEN(adjust_gamma(f32(S).to(torch::kHalf), 2.2, 0.7).to(torch::kFloat32), gamma_gain_f16);
    CHECK_GOLDEN(adjust_gamma(f32(L).to(torch::kBFloat16), 1.0, 1.2).to(torch::kFloat32), gamma_gain_bf16);

    TM_CHECK(throws_invalid([&] { return adjust_contrast(S, -0.1); }));
    TM_CHECK(throws_invalid([&] { return adjust_contrast(B.slice(-3, 0, 2), 1.0); }));
    TM_CHECK(throws_invalid([&] { return adjust_saturation(S, -0.1); }));
    TM_CHECK(throws_invalid([&] { return adjust_saturation(torch::zeros({5, 7}), 1.0); }));
    TM_CHECK(throws_invalid([&] { return adjust_sharpness(S, -0.1); }));
    TM_CHECK(throws_invalid([&] { return adjust_sharpness(B.slice(-3, 0, 2), 1.0); }));
    TM_CHECK(throws_invalid([&] { return adjust_hue(S, 0.51); }));
    TM_CHECK(throws_invalid([&] { return adjust_hue(S, std::nan("")); }));
    TM_CHECK(throws_invalid([&] { return adjust_hue(B.slice(-3, 0, 2), 0.1); }));
    TM_CHECK(throws_invalid([&] { return adjust_gamma(S, -1.0); }));
}

static void test_color_tone() {
    CHECK_GOLDEN(posterize(S, 3), posterize_u8);
    CHECK_GOLDEN(posterize(f32(S), 2), posterize_f32);
    CHECK_GOLDEN(posterize(torch::cat({f32(S), torch::ones({3, 1, 7})}, -2), 3), posterize_white_f32);
    CHECK_GOLDEN(posterize(S.to(torch::kInt16) * 100, 5), posterize_i16);
    TM_CHECK(posterize(S, 8).is_same(S)); // bits >= the dtype's value bits: unchanged
    TM_CHECK(throws_invalid([&] { return posterize(S, -1); }));
    TM_CHECK(throws_invalid([&] { return posterize(f32(S), 63); }));
    TM_CHECK(throws_invalid([&] { return posterize(S.to(torch::kBool), 1); }));

    CHECK_GOLDEN(solarize(S, 128.0), solarize_u8);
    CHECK_GOLDEN(solarize(f32(L), 0.4), solarize_f32);
    TM_CHECK(throws_invalid([&] { return solarize(S, 255.5); }));
    TM_CHECK(throws_invalid([&] { return solarize(f32(S), 1.5); }));

    const auto low = S.div(3, "floor") + 60;
    const auto constant = torch::cat({S.slice(0, 0, 2), torch::full_like(S.slice(0, 0, 1), 99)});
    CHECK_GOLDEN(autocontrast(low), autocontrast_u8);
    CHECK_GOLDEN(autocontrast(f32(L) * 0.5 + 0.2), autocontrast_f32);
    CHECK_GOLDEN(autocontrast(constant), autocontrast_const_u8);
    const auto empty = torch::zeros({0, 3, 4, 4}, torch::kUInt8);
    TM_CHECK(autocontrast(empty).is_same(empty));
    TM_CHECK(throws_invalid([&] { return autocontrast(B.slice(-3, 0, 2)); }));

    CHECK_GOLDEN(equalize(S), equalize_u8);
    CHECK_GOLDEN(equalize(B), equalize_batch_u8);
    CHECK_GOLDEN(equalize(f32(L)), equalize_f32);
    CHECK_GOLDEN(equalize(constant), equalize_const_u8);
    TM_CHECK(equalize(empty).is_same(empty));
    TM_CHECK(throws_invalid([&] { return equalize(torch::ones({5}, torch::kUInt8)); }));
}

static void test_blur_erase() {
    CHECK_GOLDEN(gaussian_blur(L, {3}), blur_u8);
    CHECK_GOLDEN(gaussian_blur(f32(L), {5, 3}, {1.2, 0.7}), blur_f32);
    CHECK_GOLDEN(gaussian_blur(f64(B), {3}, {0.9}), blur_f64);
    CHECK_GOLDEN(gaussian_blur(f32(B).reshape({2, 1, 3, 6, 10}), {3, 5}), blur_5d_f32);
    CHECK_GOLDEN(gaussian_blur(f64(L), {5, 9}), blur_default_sigma_f64);
    // a NaN sigma passes torchvision's `s <= 0` check and blurs to NaN
    TM_CHECK(gaussian_blur(f32(S), {3}, {std::nan(""), 1.0}).isnan().all().item<bool>());
    const auto empty = torch::zeros({0, 3, 4, 4});
    TM_CHECK(gaussian_blur(empty, {3}).is_same(empty));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {3, 3, 3}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {4}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {-3}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {3}, {0.0}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(L, {3}, {1.0, 2.0, 3.0}); }));
    TM_CHECK(throws_invalid([&] { return gaussian_blur(torch::zeros({5, 7}), {3}); }));

    CHECK_GOLDEN(erase(S, 1, 2, 3, 4, torch::tensor({7, 8, 9}, torch::kUInt8).view({3, 1, 1})), erase_u8);
    auto x = f32(B);
    const auto y = erase(x, -3, -4, 2, 9, torch::tensor(0.25), true);
    TM_CHECK(y.is_same(x)); // in place
    CHECK_GOLDEN(x, erase_f32);
    const auto s = f32(S);
    const auto copy = erase(s, 0, 0, 1, 1, torch::tensor(1.0));
    TM_CHECK(!copy.is_same(s) && s[0][0][0].item<float>() != 1.0f && copy[0][0][0].item<float>() == 1.0f);
}

// ---------------- progress09: geometry ----------------
static void test_rotate() {
    CHECK_GOLDEN(rotate(S, 30.0), rotate_30_u8);
    CHECK_GOLDEN(rotate(f32(L), 30.0, IM::bilinear, false, {}, {0.1, 0.5, 0.9}), rotate_30_fill_f32);
    CHECK_GOLDEN(rotate(L, 37.5, IM::bilinear, true, {}, {200}), rotate_expand_u8);
    CHECK_GOLDEN(rotate(f32(S), -110.0, IM::nearest, true, {}, {0.0}), rotate_expand_nearest_f32);
    CHECK_GOLDEN(rotate(f64(S), -20.0, IM::bilinear, false, {1.0, 2.0}), rotate_center_f64);
    CHECK_GOLDEN(rotate(f32(B), 45.0, IM::nearest, false, {}, {0.3}), rotate_batch_f32);
    CHECK_GOLDEN(rotate(S, 90.0), rotate_90_rect_u8);
    // multiples of 90 degrees about the center are exact transposes (torchvision fast path)
    const auto sq = L.slice(-1, 0, 11); // 11 x 11
    const auto zero = rotate(S, 360.0);
    TM_CHECK(torch::equal(zero, S) && !zero.is_same(S));
    TM_CHECK(torch::equal(rotate(S, -180.0), torch::rot90(S, 2, {-2, -1})));
    TM_CHECK(torch::equal(rotate(sq, 90.0), torch::rot90(sq, 1, {-2, -1})));
    TM_CHECK(torch::equal(rotate(S, 450.0, IM::nearest, true), torch::rot90(S, 1, {-2, -1})));
    TM_CHECK(torch::equal(rotate(S, -90.0, IM::nearest, true), torch::rot90(S, 3, {-2, -1})));
    // float64 bilinear tells the fast path from the grid path (which is ~1e-16 off: sin(180 deg) != 0)
    TM_CHECK(torch::equal(rotate(f64(S), 180.0, IM::bilinear), torch::rot90(f64(S), 2, {-2, -1})));
    TM_CHECK(torch::equal(rotate(f64(sq), 270.0, IM::bilinear), torch::rot90(f64(sq), 3, {-2, -1})));
    TM_CHECK(rotate(S, 270.0).sizes() == S.sizes()); // non-square, no expand: grid path keeps the size
    const auto empty = torch::zeros({0, 3, 5, 7});
    TM_CHECK(rotate(empty, 33.0, IM::nearest, true).sizes() == torch::IntArrayRef({0, 3, 9, 9}));
    TM_CHECK(throws_invalid([&] { return rotate(S, 30.0, IM::bicubic); }));
    TM_CHECK(throws_invalid([&] { return rotate(S, 30.0, IM::nearest_exact); }));
    TM_CHECK(throws_invalid([&] { return rotate(S, 30.0, IM::nearest, false, {}, {1.0, 2.0}); }));
    TM_CHECK(throws_invalid([&] { return rotate(S, 30.0, IM::nearest, false, {1.0}); }));
    TM_CHECK(throws_invalid([&] { return rotate(S, 90.0, IM::nearest, false, {1.0, 2.0, 3.0}); }));
    TM_CHECK(throws_invalid([&] { return rotate(torch::zeros({5, 7}), 30.0); }));
}

static void test_affine_perspective_elastic() {
    CHECK_GOLDEN(affine(S, 25.0, {1.0, -2.0}, 0.9, {10.0, -5.0}), affine_u8);
    CHECK_GOLDEN(affine(f32(L), -40.0, {0.5, 1.5}, 1.2, {15.0}, IM::bilinear, {0.2, 0.4, 0.6}, {3.0, 4.0}),
                 affine_fill_f32);
    CHECK_GOLDEN(affine(f64(B), 10.0, {0.0, 0.0}, 1.1, {0.0, 20.0}, IM::bilinear, {0.0}), affine_f64);
    CHECK_GOLDEN(affine(L, 123.0, {-2.0, 0.0}, 0.75, {-8.0, 3.0}, IM::bilinear, {255, 0, 128}), affine_u8_bilinear);
    TM_CHECK(throws_invalid([&] { return affine(S, 0.0, {1.0}, 1.0, {0.0}); }));
    TM_CHECK(throws_invalid([&] { return affine(S, 0.0, {0.0, 0.0}, 0.0, {0.0}); }));
    TM_CHECK(throws_invalid([&] { return affine(S, 0.0, {0.0, 0.0}, 1.0, {}); }));
    TM_CHECK(throws_invalid([&] { return affine(S, 0.0, {0.0, 0.0}, 1.0, {0.0}, IM::bicubic); }));
    TM_CHECK(throws_invalid([&] { return affine(S, 0.0, {0.0, 0.0}, 1.0, {0.0}, IM::nearest, {}, {1.0}); }));

    using pts = std::vector<std::array<double, 2>>;
    const pts sp{{0, 0}, {12, 0}, {12, 10}, {0, 10}};
    CHECK_GOLDEN(perspective(L, sp, pts{{1, 0}, {11, 2}, {12, 9}, {0, 10}}), perspective_u8);
    CHECK_GOLDEN(
            perspective(f32(L), sp, pts{{0.5, 0.25}, {11.5, 0.0}, {12.0, 8.25}, {1.25, 10.0}}, IM::bilinear, {0.7}),
            perspective_float_f32);
    CHECK_GOLDEN(perspective(f32(L), sp, pts{{0.1, 0.3}, {11.7, 0.2}, {12, 9.9}, {0.4, 10}}, IM::bilinear),
                 perspective_frac_f32);
    CHECK_GOLDEN(perspective(f64(L), pts{{0.1, 0.2}, {12.3, 0}, {12, 10}, {0, 10}},
                             pts{{1, 0}, {11, 2}, {12, 9}, {0, 10}}, IM::bilinear),
                 perspective_frac_f64);
    const std::vector<double> coeffs{1.1, 0.05, -0.5, -0.02, 0.95, 0.7, 0.001, -0.002};
    CHECK_GOLDEN(perspective(f64(B), {}, {}, IM::nearest, {}, coeffs), perspective_coeffs_f64);
    CHECK_GOLDEN(perspective(S, pts{{0, 0}, {6, 0}, {6, 4}, {0, 4}}, pts{{1, 1}, {5, 0}, {6, 4}, {0, 3}}, IM::nearest,
                             {9, 99, 199}),
                 perspective_nearest_fill_u8);
    // a zero denominator on x + y = 7 gives inf grid entries (NaN at x 6, y 1): torchvision's nearest takes the fill
    // there, its bilinear grid_sample dies with SIGBUS; here those pixels go off the image (the fill, or 0 without one)
    const std::vector<double> zero_den{1.0, 0.0, 0.0, 0.0, 1.0, -1.5, -0.125, -0.125};
    CHECK_GOLDEN(perspective(f32(S), {}, {}, IM::nearest, {0.25, 0.5, 0.75}, zero_den),
                 perspective_zero_den_nearest_f32);
    const auto zd = perspective(f32(S), {}, {}, IM::bilinear, {0.25, 0.5, 0.75}, zero_den);
    TM_CHECK(zd.isfinite().all().item<bool>());
    TM_CHECK(torch::equal(zd.select(-2, 1).select(-1, 6), torch::tensor({0.25f, 0.5f, 0.75f})));
    TM_CHECK(torch::equal(zd.select(-2, 4).select(-1, 3), torch::tensor({0.25f, 0.5f, 0.75f})));
    TM_CHECK(perspective(S, {}, {}, IM::bilinear, {}, zero_den).select(-2, 1).select(-1, 6).eq(0).all().item<bool>());
    TM_CHECK(throws_invalid([&] { return perspective(S, sp, sp, IM::bilinear, {}, coeffs); }));
    TM_CHECK(throws_invalid([&] { return perspective(S, {}, {}, IM::bilinear, {}, {1.0, 2.0}); }));
    TM_CHECK(throws_invalid([&] { return perspective(S, sp, {}); }));
    TM_CHECK(throws_invalid([&] { return perspective(S, pts{{0, 0}, {1, 1}, {2, 2}}, sp); }));
    TM_CHECK(throws_invalid([&] { return perspective(S, sp, sp, IM::bicubic); }));
    TM_CHECK(throws_invalid([&] { return perspective(torch::zeros({5, 7}), sp, sp); }));

    const auto ds = G(golden::disp_s).to(torch::kFloat32);
    const auto dl = G(golden::disp_l).to(torch::kFloat32);
    CHECK_GOLDEN(elastic(S, ds), elastic_u8);
    CHECK_GOLDEN(elastic(f32(L), dl, IM::bicubic), elastic_bicubic_f32);
    CHECK_GOLDEN(elastic(f64(S), ds, IM::nearest, {0.5}), elastic_nearest_fill_f64);
    CHECK_GOLDEN(elastic(L, dl, IM::bilinear, {10, 20, 30}), elastic_fill_u8);
    CHECK_GOLDEN(elastic(f32(S), ds, IM::bicubic, {0.3}), elastic_bicubic_fill_f32);
    // CPU float16 is warped in float32 and cast back (torchvision patch)
    const auto half = elastic(f32(S).to(torch::kHalf), ds);
    TM_CHECK(half.scalar_type() == torch::kHalf);
    CHECK_GOLDEN(half.to(torch::kFloat32), elastic_half);
    TM_CHECK(throws_invalid([&] { return elastic(S, dl); }));
    TM_CHECK(throws_invalid([&] { return elastic(S, ds, IM::nearest_exact); }));
    TM_CHECK(throws_invalid([&] { return elastic(torch::zeros({5, 7}), ds); }));
}

static void test_new_ops_devices() {
#if GOLDEN_HAS_MPS
    if (torch::mps::is_available()) {
        const auto mps = [](const torch::Tensor &x) { return x.to(torch::kMPS); };
        CHECK_GOLDEN(gaussian_blur(mps(f32(L)), {5, 3}, {1.2, 0.7}).cpu(), blur_mps_f32);
        CHECK_GOLDEN(adjust_sharpness(mps(f32(S)), 0.4).cpu(), sharpness_mps_f32);
        CHECK_GOLDEN(rotate(mps(f32(L)), 30.0, IM::bilinear, false, {}, {0.1, 0.5, 0.9}).cpu(), rotate_mps_f32);
        const std::vector<std::array<double, 2>> sp{{0, 0}, {12, 0}, {12, 10}, {0, 10}};
        CHECK_GOLDEN(perspective(mps(f32(L)), sp, {{1, 0}, {11, 2}, {12, 9}, {0, 10}}).cpu(), perspective_mps_f32);
        CHECK_GOLDEN(equalize(mps(B)).cpu(), equalize_mps_u8);
        // torchvision's MPS adjust_hue reads a freed non_blocking copy; ours equals the CPU golden
        CHECK_GOLDEN(adjust_hue(mps(f32(L)), -0.4).cpu(), hue_f32);
        // torchvision aborts here (MPSGraph mul(uint8, float)); the float32 luma path equals the CPU golden
        CHECK_GOLDEN(adjust_contrast(mps(S), 0.6).cpu(), contrast_u8);
    }
#endif
}

// permute_channels / grayscale_to_rgb / gaussian_noise (progress10): the kernels behind RandomChannelPermutation,
// RandomPhotometricDistort, RGB and GaussianNoise.
static void test_channels_noise() {
    CHECK_GOLDEN(permute_channels(S, {2, 0, 1}), permute_channels_u8);
    CHECK_GOLDEN(permute_channels(f32(B), {1, 2, 0}), permute_channels_batch_f32);
    CHECK_GOLDEN(permute_channels(S, {-1, 0, 1}), permute_channels_neg_u8);
    const auto empty = torch::zeros({0, 3, 4, 5});
    TM_CHECK(permute_channels(empty, {0, 1, 2}).is_same(empty));
    TM_CHECK(throws_invalid([] { return permute_channels(S, {0, 1}); }));
    TM_CHECK(throws_invalid([] { return permute_channels(torch::zeros({4, 5}), {0}); }));

    const auto gray = S.slice(0, 0, 1);
    TM_CHECK(torch::equal(grayscale_to_rgb(gray), gray.repeat({3, 1, 1})));
    TM_CHECK(grayscale_to_rgb(S).is_same(S) && grayscale_to_rgb(B).is_same(B));
    TM_CHECK(throws_invalid([] { return grayscale_to_rgb(torch::zeros({2, 4, 5})); }));
    TM_CHECK(throws_invalid([] { return grayscale_to_rgb(torch::zeros({4, 5})); }));

    torch::manual_seed(77);
    CHECK_GOLDEN(gaussian_noise(f32(S)), gaussian_noise_f32);
    torch::manual_seed(78);
    CHECK_GOLDEN(gaussian_noise(f64(S), 0.1, 0.3, false), gaussian_noise_f64);
    torch::manual_seed(79);
    CHECK_GOLDEN(gaussian_noise(f32(S).to(torch::kHalf), 0.05, 0.2).to(torch::kFloat32), gaussian_noise_half);
    TM_CHECK(throws_invalid([] { return gaussian_noise(S); }));
    TM_CHECK(throws_invalid([] { return gaussian_noise(f32(S), 0.0, -0.1); }));
}

int main() {
    TM_RUN(test_resize_modes);
    TM_RUN(test_resize_batch_and_sizes);
    TM_RUN(test_resize_devices);
    TM_RUN(test_crop);
    TM_RUN(test_five_ten_crop);
    TM_RUN(test_pad);
    TM_RUN(test_to_dtype);
    TM_RUN(test_color);
    TM_RUN(test_flip);
    TM_RUN(test_color_adjust);
    TM_RUN(test_color_tone);
    TM_RUN(test_blur_erase);
    TM_RUN(test_rotate);
    TM_RUN(test_affine_perspective_elastic);
    TM_RUN(test_new_ops_devices);
    TM_RUN(test_channels_noise);
    return tm_test::summary("vision_test_functional");
}
