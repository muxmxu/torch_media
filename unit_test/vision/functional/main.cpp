// Assertion-based tests for the vision functional layer against torchvision 0.20.1 transforms.v2.functional
// goldens (gen_golden.py -> golden.hpp: the seeded inputs and every expected output, bit for bit).
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
    return tm_test::summary("vision_test_functional");
}
