// Assertion-based tests for the vision transform classes: each class equals its golden-verified functional op, the
// ImageNet recipe Resize(256) + CenterCrop(224) + ToDtype(float32, scale) + Normalize matches torchvision 0.20.1
// end to end, and the random flips draw like torchvision v2 (gen_golden.py -> golden.hpp). The random classes
// (progress10) replay torchvision's seeded calls: every output, whether it is the input itself, and the next draw.
#include <cmath>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "golden.hpp"
#include "test_util.hpp"

namespace F = torchmedia::vision::functional;
namespace T = torchmedia::vision::transform;
using IM = F::interpolation_mode;
using PM = F::padding_mode;

// Bit-exact on arm64, where the goldens were generated. Elsewhere torch may resize uint8 natively (x86 AVX2, one
// uint8 step = 1 / 255 / 0.224 after Normalize) or contract FMAs differently.
#if defined(__aarch64__)
constexpr double kTolPixel = 0.0;
constexpr double kTolSumRel = 0.0;
constexpr double kTolInt = 0.0;
constexpr double kTolFloat = 0.0;
#else
constexpr double kTolPixel = 0.02;
constexpr double kTolSumRel = 1e-4;
constexpr double kTolInt = 1.0;
constexpr double kTolFloat = 1e-5;
#endif

template<class Fn>
static auto throws_invalid(Fn &&fn) -> bool {
    try {
        (void) fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

// gen_golden.py synthetic_image(): gradients + a disc + stripes + hashed noise from integer ops only.
static auto synthetic_image(int64_t h = 375, int64_t w = 500) -> torch::Tensor {
    const auto y = torch::arange(h).view({h, 1});
    const auto x = torch::arange(w).view({1, w});
    std::vector<torch::Tensor> channels;
    for (int64_t c = 0; c < 3; ++c) {
        auto base = torch::div(x * (200 + 20 * c), w, "floor") + torch::div(y * (150 - 30 * c), h, "floor");
        const auto disc = (x - 330).pow(2) + (y - 150).pow(2) < 80 * 80;
        base = torch::where(disc, 220 - 70 * c, base);
        const auto stripes = (y >= 260) & (y < 330) & (torch::div(x, 7, "floor").remainder(2) == 1);
        base = torch::where(stripes, 30 + 90 * c, base);
        const auto noise =
                ((x * 7919 + y * 104729 + c * 15485863) * int64_t{2654435761}).remainder(1000003).remainder(17) - 8;
        channels.push_back((base + noise).clamp(0, 255));
    }
    return torch::stack(channels).to(torch::kUInt8);
}

static void check_recipe(const torch::Tensor &out, const std::vector<double> &grid, double sum, double sum_sq) {
    TM_CHECK(out.scalar_type() == torch::kFloat32);
    TM_CHECK(out.sizes() == torch::IntArrayRef({3, 224, 224}));
    const auto g = torch::tensor(grid, torch::kFloat64).to(torch::kFloat32).reshape({3, 14, 14});
    TM_CHECK_TENSOR_CLOSE(out.slice(1, 0, 224, 16).slice(2, 0, 224, 16), g, kTolPixel, 0.0);
    const auto d = out.to(torch::kFloat64);
    TM_CHECK_CLOSE(d.sum().item<double>(), sum, kTolSumRel * std::abs(sum));
    TM_CHECK_CLOSE(d.square().sum().item<double>(), sum_sq, kTolSumRel * sum_sq);
}

static void test_imagenet_recipe() {
    const auto img = synthetic_image();
    const auto flat = img.flatten().to(torch::kInt64);
    TM_CHECK(flat.sum().item<int64_t>() == golden::image_sum);
    TM_CHECK((flat * (torch::arange(flat.numel()).remainder(251) + 1)).sum().item<int64_t>() ==
             golden::image_weighted_sum);

    const T::Resize resize(T::resize_option().size({256}));
    const T::CenterCrop center_crop(T::center_crop_option().size({224}));
    const T::ToDtype to_float(T::to_dtype_option().dtype(torch::kFloat32).scale(true));
    const T::Normalize normalize(T::normalize_option().mean({0.485, 0.456, 0.406}).std({0.229, 0.224, 0.225}));

    const auto resized = resize(img);
    TM_CHECK(resized.sizes() == torch::IntArrayRef({3, 256, 341}) && resized.scalar_type() == torch::kUInt8);
    check_recipe(normalize(to_float(center_crop(resized))), golden::recipe_u8_grid, golden::recipe_u8_sum,
                 golden::recipe_u8_sum_sq);
    check_recipe(normalize(center_crop(resize(to_float(img)))), golden::recipe_f32_grid, golden::recipe_f32_sum,
                 golden::recipe_f32_sum_sq);
}

static void test_classes_match_functional() {
    torch::manual_seed(3);
    const auto u8 = torch::randint(0, 256, {2, 3, 9, 12}, torch::kUInt8);
    const auto x = u8.to(torch::kFloat32) / 255;

    const T::Resize resize(
            T::resize_option().size({5}).interpolation(F::interpolation_mode::bicubic).max_size(7).antialias(false));
    TM_CHECK(torch::equal(resize(x), F::resize(x, {5}, F::interpolation_mode::bicubic, 7, false)));
    TM_CHECK(torch::equal(resize.forward(u8), F::resize(u8, {5}, F::interpolation_mode::bicubic, 7, false)));
    const T::Resize longest(T::resize_option().max_size(6)); // size None: longer edge = max_size
    TM_CHECK(longest(x).sizes() == torch::IntArrayRef({2, 3, 4, 6}));

    const T::CenterCrop cc(T::center_crop_option().size({4, 15}));
    TM_CHECK(torch::equal(cc(u8), F::center_crop(u8, {4, 15})) && torch::equal(cc.forward(u8), cc(u8)));

    const T::Pad pad(T::pad_option().padding({1, 2, 3, 4}).fill({0.1, 0.2, 0.3}));
    TM_CHECK(torch::equal(pad(x), F::pad(x, {1, 2, 3, 4}, {0.1, 0.2, 0.3})) && torch::equal(pad.forward(x), pad(x)));
    const T::Pad sym(T::pad_option().padding({2, 1}).padding_mode(F::padding_mode::symmetric));
    TM_CHECK(torch::equal(sym(u8), F::pad(u8, {2, 1}, {}, F::padding_mode::symmetric)));

    const std::vector<double> mean{0.5, 0.4, 0.3}, stddev{0.2, 0.25, 0.3};
    const T::Normalize norm(T::normalize_option().mean(mean).std(stddev));
    TM_CHECK(torch::equal(norm(x), F::normalize(x, mean, stddev)) && torch::equal(norm.forward(x), norm(x)));
    auto y = x.clone();
    const T::Normalize norm_inplace(T::normalize_option().mean(mean).std(stddev).inplace(true));
    TM_CHECK(norm_inplace(y).is_same(y) && torch::equal(y, norm(x)));

    const T::ToDtype to_u8(T::to_dtype_option().dtype(torch::kUInt8).scale(true));
    TM_CHECK(torch::equal(to_u8(x), F::to_dtype(x, torch::kUInt8, true)) && torch::equal(to_u8.forward(x), to_u8(x)));
    TM_CHECK(T::ToDtype()(u8).scalar_type() == torch::kFloat32); // default: float32, no scaling
    TM_CHECK(torch::equal(T::ToDtype()(u8), u8.to(torch::kFloat32)));

    const T::Grayscale gray(T::grayscale_option().num_output_channels(3));
    TM_CHECK(torch::equal(gray(u8), F::rgb_to_grayscale(u8, 3)) && torch::equal(gray.forward(u8), gray(u8)));
    TM_CHECK(torch::equal(T::Grayscale()(x), F::rgb_to_grayscale(x)));

    const T::FiveCrop five(T::five_crop_option().size({4, 5}));
    const auto f1 = five(u8);
    const auto f2 = F::five_crop(u8, {4, 5});
    TM_CHECK(f1.size() == 5 && torch::equal(torch::stack(f1), torch::stack(f2)));
    TM_CHECK(torch::equal(torch::stack(five.forward(u8)), torch::stack(f2)));
    const T::TenCrop ten(T::ten_crop_option().size({4}).vertical_flip(true));
    const auto t1 = ten(x);
    TM_CHECK(t1.size() == 10 && torch::equal(torch::stack(t1), torch::stack(F::ten_crop(x, {4}, true))));
    TM_CHECK(torch::equal(torch::stack(ten.forward(x)), torch::stack(t1)));

    // no cached tensors: the output follows the input's device
    if (torch::mps::is_available()) {
        const auto xm = x.to(torch::kMPS);
        TM_CHECK(norm(xm).device().is_mps() && cc(xm).device().is_mps() && to_u8(xm).device().is_mps());
        TM_CHECK_TENSOR_CLOSE(norm(xm).cpu(), norm(x), 1e-6, 1e-6);
    }
}

static void test_random_flips() {
    const auto x = torch::arange(6, torch::kFloat32).reshape({1, 2, 3});
    torch::manual_seed(7);
    const T::RandomHorizontalFlip hflip; // p = 0.5
    for (const double expected: golden::hflip_flips) {
        const auto out = hflip(x);
        TM_CHECK(expected == 1.0 ? torch::equal(out, x.flip(-1)) : out.is_same(x));
    }
    TM_CHECK(torch::rand({1}).item<double>() == golden::hflip_next_draw); // exactly one draw per call

    torch::manual_seed(11);
    const T::RandomVerticalFlip vflip(T::random_flip_option().p(0.3));
    for (const double expected: golden::vflip_flips) {
        const auto out = vflip.forward(x);
        TM_CHECK(expected == 1.0 ? torch::equal(out, x.flip(-2)) : out.is_same(x));
    }
    TM_CHECK(torch::rand({1}).item<double>() == golden::vflip_next_draw);

    const T::RandomHorizontalFlip never(T::random_flip_option().p(0.0));
    const T::RandomVerticalFlip always(T::random_flip_option().p(1.0));
    TM_CHECK(never(x).is_same(x) && never.forward(x).is_same(x));
    TM_CHECK(torch::equal(always(x), x.flip(-2)));
}

static void test_validation() {
    TM_CHECK(throws_invalid([] { return T::Resize(T::resize_option()); }));
    TM_CHECK(throws_invalid([] { return T::Resize(T::resize_option().size({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::CenterCrop(T::center_crop_option()); }));
    TM_CHECK(throws_invalid([] { return T::CenterCrop(T::center_crop_option().size({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::FiveCrop(T::five_crop_option()); }));
    TM_CHECK(throws_invalid([] { return T::TenCrop(T::ten_crop_option().size({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::Pad(T::pad_option().padding({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::Pad(T::pad_option()); }));
    TM_CHECK(throws_invalid([] { return T::RandomHorizontalFlip(T::random_flip_option().p(1.5)); }));
    TM_CHECK(throws_invalid([] { return T::RandomVerticalFlip(T::random_flip_option().p(-0.1)); }));
    TM_CHECK(throws_invalid([] { return T::RandomVerticalFlip(T::random_flip_option().p(std::nan(""))); }));
    // errors torchvision raises at call time stay there
    const T::Grayscale bad_gray(T::grayscale_option().num_output_channels(2));
    TM_CHECK(throws_invalid([&] { return bad_gray(torch::zeros({3, 4, 4})); }));
}

static void check_values(const torch::Tensor &out, const std::vector<double> &expected) {
    const auto e = torch::tensor(expected, torch::kFloat64).to(out.scalar_type()).reshape(out.sizes());
    TM_CHECK_TENSOR_CLOSE(out, e, kTolPixel, 0.0);
}

static void test_gaussian_blur() {
    const auto x = torch::arange(3 * 9 * 11).remainder(37).to(torch::kFloat32).reshape({3, 9, 11}) / 36;
    torch::manual_seed(5);
    const T::GaussianBlur blur(T::gaussian_blur_option().kernel_size({5}).sigma({0.5, 2.0}));
    check_values(blur(x), golden::blur_out0);
    check_values(blur.forward(x), golden::blur_out1);
    check_values(blur(x), golden::blur_out2);
    torch::manual_seed(6);
    const T::GaussianBlur fixed(T::gaussian_blur_option().kernel_size({3, 5}).sigma({0.8}));
    check_values(fixed(x), golden::blur_fixed_out);
    TM_CHECK(torch::rand({1}).item<double>() == golden::blur_fixed_next_draw); // a fixed sigma still draws once
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option()); }));
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({4})); }));
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({3, 0})); }));
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({3, 3, 3})); }));
    TM_CHECK(throws_invalid(
            [] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({3}).sigma({2.0, 1.0})); }));
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({3}).sigma({0.0})); }));
    TM_CHECK(throws_invalid([] { return T::GaussianBlur(T::gaussian_blur_option().kernel_size({3}).sigma({})); }));
}

// Each p-gated class draws like torchvision v2 and, when applied, equals its functional op.
template<class Cls, class Op>
static void check_random_apply(const Cls &t, Op op, uint64_t seed, const std::vector<double> &applied,
                               double next_draw) {
    const auto x = (torch::arange(3 * 4 * 5) * 37).remainder(256).to(torch::kUInt8).reshape({3, 4, 5});
    torch::manual_seed(seed);
    for (size_t i = 0; i < applied.size(); ++i) {
        const auto out = i % 2 ? t.forward(x) : t(x);
        TM_CHECK(applied[i] == 1.0 ? (!out.is_same(x) && torch::equal(out, op(x))) : out.is_same(x));
    }
    TM_CHECK(torch::rand({1}).item<double>() == next_draw);
}

static void test_random_color() {
    using namespace golden;
    check_random_apply(T::RandomInvert(), F::invert, 21, invert_applied, invert_next_draw);
    check_random_apply(T::RandomAutocontrast(T::random_apply_option().p(0.7)), F::autocontrast, 22,
                       autocontrast_applied, autocontrast_next_draw);
    check_random_apply(T::RandomEqualize(T::random_apply_option().p(0.4)), F::equalize, 23, equalize_applied,
                       equalize_next_draw);
    check_random_apply(
            T::RandomGrayscale(), [](const torch::Tensor &x) { return F::rgb_to_grayscale(x, 3); }, 24,
            grayscale_applied, grayscale_next_draw);
    check_random_apply(
            T::RandomPosterize(T::random_posterize_option().bits(3).p(0.6)),
            [](const torch::Tensor &x) { return F::posterize(x, 3); }, 25, posterize_applied, posterize_next_draw);
    check_random_apply(
            T::RandomSolarize(T::random_solarize_option().threshold(100.0)),
            [](const torch::Tensor &x) { return F::solarize(x, 100.0); }, 26, solarize_applied, solarize_next_draw);
    check_random_apply(
            T::RandomAdjustSharpness(T::random_adjust_sharpness_option().sharpness_factor(2.0).p(0.3)),
            [](const torch::Tensor &x) { return F::adjust_sharpness(x, 2.0); }, 27, sharpness_applied,
            sharpness_next_draw);
    // RandomGrayscale keeps the channel count; a 2-D input reaches the functional op (which rejects it)
    const T::RandomGrayscale always_gray(T::random_apply_option().p(1.0));
    const auto one = torch::zeros({1, 4, 4}, torch::kUInt8);
    TM_CHECK(always_gray(one).sizes() == one.sizes());
    TM_CHECK(throws_invalid([&] { return always_gray(torch::zeros({4, 4})); }));
    // the neutral option defaults change nothing on uint8 (torchvision has no defaults for these)
    const auto x = torch::randint(0, 256, {3, 4, 5}, torch::kUInt8);
    const T::RandomPosterize post(T::random_posterize_option().p(1.0));
    const T::RandomSolarize sol(T::random_solarize_option().p(1.0));
    const T::RandomAdjustSharpness sharp(T::random_adjust_sharpness_option().p(1.0));
    TM_CHECK(torch::equal(post(x), x) && torch::equal(sol(x), x) && torch::equal(sharp(x), x));
    TM_CHECK(throws_invalid([] { return T::RandomInvert(T::random_apply_option().p(1.1)); }));
    TM_CHECK(throws_invalid([] { return T::RandomPosterize(T::random_posterize_option().p(-0.1)); }));
}

// ======== random transform classes (progress10) ========
static auto G(const golden::entry &e) -> torch::Tensor {
    return torch::tensor(e.data, torch::kFloat64).reshape(e.shape);
}
static const auto S = G(golden::in_s).to(torch::kUInt8); // [3, 9, 11]
static const auto B = G(golden::in_b).to(torch::kUInt8); // [2, 3, 6, 10]
static const auto TI = G(golden::in_t).to(torch::kUInt8); // [3, 4, 5]
static const auto E = G(golden::in_e).to(torch::kUInt8); // [1, 24, 26]
static auto f32(const torch::Tensor &x) -> torch::Tensor { return x.to(torch::kFloat32) / 255; }
static auto f64(const torch::Tensor &x) -> torch::Tensor { return x.to(torch::kFloat64) / 255; }

static auto dtype_name(torch::Dtype d) -> std::string {
    return d == torch::kUInt8     ? "uint8"
           : d == torch::kFloat32 ? "float32"
           : d == torch::kFloat64 ? "float64"
                                  : c10::toString(d);
}

// out must match the golden entry in dtype, shape and values.
static void check_entry(const torch::Tensor &out, const golden::entry &e, const std::string &name, int line) {
    const auto dt = dtype_name(out.scalar_type());
    tm_test::record(dt == e.dtype, name.c_str(), __FILE__, line, "dtype " + dt + " != " + e.dtype);
    std::string detail;
    const double tol = out.is_floating_point() ? kTolFloat : kTolInt;
    const auto expected = G(e).to(out.scalar_type()); // 9-digit float32 goldens round-trip once rounded to float32
    tm_test::record(tm_test::tensor_close(out, expected, tol, 0.0, detail), name.c_str(), __FILE__, line, detail);
}
#define CHECK_ENTRY(out, name) check_entry((out), golden::name, #name, __LINE__)

// Replays a seeded case: same outputs, the same "returns the input itself" calls, and the same next draw (so the
// same number of draws). Odd calls go through forward().
template<class Tr>
static void check_case(const Tr &t, const torch::Tensor &x, const golden::seeded_case &c, const std::string &name,
                       int line) {
    torch::manual_seed(c.seed);
    for (size_t k = 0; k < c.outs.size(); ++k) {
        const auto out = k % 2 ? t.forward(x) : t(x);
        const auto call = name + "[" + std::to_string(k) + "]";
        check_entry(out, c.outs[k], call, line);
        tm_test::record(out.is_same(x) == (c.same[k] != 0), (call + " is input").c_str(), __FILE__, line);
    }
    tm_test::record(torch::rand({1}).item<double>() == c.next_draw, (name + " next draw").c_str(), __FILE__, line);
}
#define CHECK_CASE(t, x, name) check_case((t), (x), golden::name, #name, __LINE__)

static void test_random_crop() {
    using O = T::random_crop_option;
    CHECK_CASE(T::RandomCrop(O().size({5, 7})), S, crop_u8);
    CHECK_CASE(T::RandomCrop(O().size({6}).padding({2}).fill({0.5})), f32(S), crop_pad_f32);
    CHECK_CASE(T::RandomCrop(O().size({7, 9}).padding({1, 3}).padding_mode(PM::reflect)), B, crop_pad2_reflect_u8);
    CHECK_CASE(T::RandomCrop(O().size({8, 12}).padding({1, 2, 3, 4}).padding_mode(PM::symmetric)), f32(B),
               crop_pad4_symmetric_f32);
    CHECK_CASE(T::RandomCrop(O().size({12, 14}).pad_if_needed(true).fill({10, 20, 30})), S, crop_pad_if_needed_u8);
    CHECK_CASE(T::RandomCrop(O().size({12, 5}).pad_if_needed(true).padding_mode(PM::edge)), f32(S),
               crop_pad_if_needed_edge_f32);
    CHECK_CASE(T::RandomCrop(O().size({9, 11})), S, crop_full_u8);
    CHECK_CASE(T::RandomCrop(O().size({3, 4})), S.select(0, 0), crop_2d_u8);
    CHECK_CASE(T::RandomCrop(O().size({9, 5})), S, crop_rows_u8);
    TM_CHECK(throws_invalid([] { return T::RandomCrop(O()); }));
    TM_CHECK(throws_invalid([] { return T::RandomCrop(O().size({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::RandomCrop(O().size({3}).padding({1, 2, 3})); }));
    const T::RandomCrop too_big(O().size({12, 14}));
    const T::RandomCrop too_big_padded(O().size({12, 14}).padding({1}));
    TM_CHECK(throws_invalid([&] { return too_big(S); }) && throws_invalid([&] { return too_big_padded(S); }));
    TM_CHECK(throws_invalid([&] { return too_big(torch::zeros({5})); })); // needs [..., H, W]
}

static void test_random_resized_crop() {
    using O = T::random_resized_crop_option;
    CHECK_CASE(T::RandomResizedCrop(O().size({5, 6})), S, rrc_u8);
    CHECK_CASE(T::RandomResizedCrop(O().size({7}).interpolation(IM::bicubic).antialias(false)), f32(S),
               rrc_bicubic_f32);
    CHECK_CASE(T::RandomResizedCrop(O().size({4, 5}).scale({0.3, 0.6}).ratio({0.5, 2.0})), B, rrc_batch_u8);
    // no attempt fits: the whole image, center-cropped to the nearest ratio bound
    CHECK_CASE(T::RandomResizedCrop(O().size({5, 6}).scale({0.9, 1.0}).ratio({0.2, 0.25})), f32(S),
               rrc_fallback_wide_f32);
    CHECK_CASE(T::RandomResizedCrop(O().size({5, 6}).scale({0.9, 1.0}).ratio({5.0, 6.0})), f32(S),
               rrc_fallback_tall_f32);
    CHECK_CASE(T::RandomResizedCrop(O().size({5, 6}).scale({2.0, 3.0})), S, rrc_fallback_whole_u8);
    TM_CHECK(throws_invalid([] { return T::RandomResizedCrop(O()); }));
}

static void test_random_resize() {
    CHECK_CASE(T::RandomResize(T::random_resize_option().min_size(4).max_size(9)), S, rresize_u8);
    CHECK_CASE(T::RandomResize(
                       T::random_resize_option().min_size(3).max_size(8).interpolation(IM::nearest).antialias(false)),
               f32(B), rresize_nearest_f32);
    using SO = T::random_shortest_size_option;
    CHECK_CASE(T::RandomShortestSize(SO().min_size({4, 6, 8}).max_size(9)), S, shortest_u8);
    CHECK_CASE(T::RandomShortestSize(SO().min_size({4, 7}).interpolation(IM::bilinear).antialias(true)), f32(B),
               shortest_f32);
    using JO = T::scale_jitter_option;
    CHECK_CASE(T::ScaleJitter(JO().target_size({8, 10}).scale_range({0.5, 1.5})), S, jitter_scale_u8);
    CHECK_CASE(T::ScaleJitter(
                       JO().target_size({12, 10}).scale_range({0.5, 1.2}).interpolation(IM::bicubic).antialias(true)),
               f32(S), jitter_scale_bicubic_f32);
    CHECK_CASE(T::ScaleJitter(JO().target_size({13, 12}).scale_range({1.0, 1.0})), S, jitter_scale_float32_u8);
    CHECK_CASE(T::ScaleJitter(JO().target_size({8, 10}).scale_range({0.5, 1.5})), f32(B), jitter_scale_batch_f32);
}

static void test_color_jitter() {
    using O = T::color_jitter_option;
    CHECK_CASE(T::ColorJitter(O().brightness({0.4}).contrast({0.4}).saturation({0.4})), S, cj_u8);
    CHECK_CASE(T::ColorJitter(O().brightness({0.4}).contrast({0.4}).saturation({0.4}).hue({0.1})), f32(S), cj_hue_f32);
    CHECK_CASE(T::ColorJitter(O().brightness({0.5, 1.5}).hue({-0.2, 0.3})), B, cj_batch_u8);
    CHECK_CASE(T::ColorJitter(), S, cj_none_u8);
    CHECK_CASE(T::ColorJitter(O().brightness({0}).contrast({1, 1}).saturation({0.5}).hue({0})), f32(S),
               cj_identity_ranges_f32);
    CHECK_CASE(T::ColorJitter(O().brightness({0.3}).contrast({0.2, 0.9}).saturation({0.0, 2.0}).hue({-0.5, 0.5})),
               f64(S), cj_f64);
    CHECK_CASE(T::ColorJitter(O().brightness({1.5}).contrast({1.2})), S, cj_clip_u8);
    TM_CHECK(throws_invalid([] { return T::ColorJitter(O().brightness({-0.1})); }));
    TM_CHECK(throws_invalid([] { return T::ColorJitter(O().contrast({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::ColorJitter(O().saturation({1.5, 0.5})); }));
    TM_CHECK(throws_invalid([] { return T::ColorJitter(O().hue({0.6})); }));
    TM_CHECK(throws_invalid([] { return T::ColorJitter(O().brightness({-0.5, 1.0})); }));

    using PO = T::random_photometric_distort_option;
    CHECK_CASE(T::RandomPhotometricDistort(), S, pd_u8);
    CHECK_CASE(T::RandomPhotometricDistort(PO().brightness({0.875, 1.125})
                                                   .contrast({0.5, 1.5})
                                                   .saturation({0.5, 1.5})
                                                   .hue({-0.05, 0.05})
                                                   .p(1.0)),
               f32(S), pd_p1_f32);
    CHECK_CASE(T::RandomPhotometricDistort(PO().p(0.0)), S, pd_p0_u8);
    CHECK_CASE(T::RandomPhotometricDistort(PO().p(1.0)), f32(B), pd_batch_f32);
    CHECK_CASE(T::RandomChannelPermutation(), S, perm_u8);
    CHECK_CASE(T::RandomChannelPermutation(), f32(B), perm_batch_f32);

    const T::RGB rgb;
    const auto gray = S.slice(0, 0, 1);
    TM_CHECK(torch::equal(rgb(gray), gray.repeat({3, 1, 1})) && rgb.forward(S).is_same(S));
}

static void test_random_warps() {
    using RO = T::random_rotation_option;
    CHECK_CASE(T::RandomRotation(RO().degrees({30})), S, rot_u8);
    CHECK_CASE(
            T::RandomRotation(RO().degrees({-10, 40}).interpolation(IM::bilinear).expand(true).fill({0.2, 0.4, 0.6})),
            f32(S), rot_expand_f32);
    CHECK_CASE(T::RandomRotation(RO().degrees({25}).interpolation(IM::bilinear).center({3, 4}).fill({})), f32(B),
               rot_center_none_f32);
    TM_CHECK(throws_invalid([] { return T::RandomRotation(RO()); }));
    TM_CHECK(throws_invalid([] { return T::RandomRotation(RO().degrees({-5})); }));
    TM_CHECK(throws_invalid([] { return T::RandomRotation(RO().degrees({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::RandomRotation(RO().degrees({5}).center({1})); }));

    using AO = T::random_affine_option;
    CHECK_CASE(T::RandomAffine(AO().degrees({20}).translate({0.2, 0.3}).scale({0.8, 1.2}).shear({10})), S, aff_u8);
    CHECK_CASE(T::RandomAffine(AO().degrees({-45, 10}).shear({-5, 5, -20, 3}).interpolation(IM::bilinear).fill({0.25})),
               f32(S), aff_shear4_f32);
    CHECK_CASE(T::RandomAffine(
                       AO().degrees({15}).translate({0.1, 0.1}).interpolation(IM::bilinear).fill({}).center({2.5, 3})),
               f32(B), aff_center_none_f32);
    CHECK_CASE(T::RandomAffine(AO().degrees({0})), S, aff_zero_u8);
    CHECK_CASE(T::RandomRotation(RO().degrees({0})), S, rot_zero_u8);
    CHECK_CASE(T::RandomRotation(RO().degrees({-60, 150}).interpolation(IM::bilinear).center({2, 6}).fill({})), f64(S),
               rot_center_f64);
    CHECK_CASE(T::RandomAffine(AO().degrees({30})
                                       .translate({0.2, 0.1})
                                       .shear({-10, 15, 5, 20})
                                       .interpolation(IM::bilinear)
                                       .fill({0.5})),
               f64(S), aff_shear_f64);
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO()); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).translate({0.1})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).translate({0.1, 1.5})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).scale({1.0})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).scale({0.0, 1.0})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).shear({-1})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).shear({1, 2, 3})); }));
    TM_CHECK(throws_invalid([] { return T::RandomAffine(AO().degrees({5}).center({1, 2, 3})); }));
    TM_CHECK(!throws_invalid([] { return T::RandomAffine(AO().degrees({5}).translate({1.0, 1.0})); })); // inclusive

    using PO = T::random_perspective_option;
    CHECK_CASE(T::RandomPerspective(), S, persp_u8);
    CHECK_CASE(T::RandomPerspective(PO().distortion_scale(0.8).p(1.0).interpolation(IM::nearest).fill({0.5, 0.1, 0.9})),
               f32(S), persp_nearest_fill_f32);
    CHECK_CASE(T::RandomPerspective(PO().distortion_scale(0.3).p(1.0).fill({})), f32(B), persp_none_batch_f32);
    CHECK_CASE(T::RandomPerspective(PO().distortion_scale(0.6).p(1.0)),
               f32(S.slice(0, 0, 1).slice(1, 0, 7).slice(2, 0, 6)), persp_odd_f32);
    // distortion_scale 1 can cross corners: a zero denominator (inf / NaN grid) takes the fill instead of crashing
    // grid_sample (torchvision: SIGBUS at this seed)
    const T::RandomPerspective full(PO().distortion_scale(1.0).p(1.0));
    const auto x64 = torch::arange(64 * 64, torch::kFloat32).reshape({1, 64, 64}) / 4096;
    torch::manual_seed(13);
    for (int k = 0; k < 4; ++k) {
        TM_CHECK(full(x64).isfinite().all().item<bool>());
    }
    TM_CHECK(throws_invalid([] { return T::RandomPerspective(PO().distortion_scale(1.5)); }));
    TM_CHECK(throws_invalid([] { return T::RandomPerspective(PO().p(-1.0)); }));
    TM_CHECK(!throws_invalid([] { return T::RandomPerspective(PO().distortion_scale(1.0)); })); // inclusive

    using EO = T::elastic_transform_option;
    CHECK_CASE(T::ElasticTransform(), f32(E), elastic_default_f32);
    CHECK_CASE(T::ElasticTransform(EO().alpha({20.0, 30.0}).sigma({0.625, 0.5}).interpolation(IM::nearest)), S,
               elastic_nearest_u8);
    CHECK_CASE(T::ElasticTransform(EO().alpha({10.0}).sigma({0.0}).interpolation(IM::bicubic).fill({})), f32(S),
               elastic_nosigma_bicubic_f32);
    CHECK_CASE(T::ElasticTransform(EO().alpha({20.0, 30.0}).sigma({0.625, 0.5})), f32(S), elastic_even_kernel_f32);
    CHECK_CASE(T::ElasticTransform(EO().alpha({20.0}).sigma({0.5})), f32(B), elastic_batch_f32);
    TM_CHECK(throws_invalid([] { return T::ElasticTransform(EO().alpha({})); }));
    TM_CHECK(throws_invalid([] { return T::ElasticTransform(EO().sigma({1, 2, 3})); }));
    const T::ElasticTransform zero_sigma_y(EO().sigma({1.0, 0.0})); // dx is blurred with both sigmas: 0 is rejected
    TM_CHECK(throws_invalid([&] { return zero_sigma_y(f32(S)); }));
}

static void test_erasing_zoom_noise() {
    using O = T::random_erasing_option;
    CHECK_CASE(T::RandomErasing(), S, erase_u8);
    CHECK_CASE(T::RandomErasing(O().p(1.0).value({})), f32(S), erase_random_f32);
    CHECK_CASE(T::RandomErasing(O().p(1.0).value({0.1, 0.2, 0.3})), f32(B), erase_channel_f32);
    const auto x = S.clone();
    CHECK_CASE(T::RandomErasing(O().p(1.0).value({7}).inplace(true)), x, erase_inplace_u8);
    CHECK_CASE(T::RandomErasing(O().p(1.0).scale({0.99, 1.0}).ratio({1.0, 1.0})), f32(S), erase_fallback_f32);
    CHECK_CASE(T::RandomErasing(O().p(1.0).scale({0.99, 1.0}).ratio({1.0, 1.0})), f32(S).slice(2, 0, 9),
               erase_exact_fit_f32);
    TM_CHECK(throws_invalid([] { return T::RandomErasing(O().p(2.0)); }));
    TM_CHECK(throws_invalid([] { return T::RandomErasing(O().scale({-0.1, 0.5})); }));
    TM_CHECK(throws_invalid([] { return T::RandomErasing(O().scale({0.1, 1.5})); }));
    TM_CHECK(!throws_invalid([] { return T::RandomErasing(O().scale({0.0, 0.33})); })); // inclusive
    const T::RandomErasing two_values(O().p(1.0).value({1, 2}));
    TM_CHECK(throws_invalid([&] { return two_values(S); }));

    using ZO = T::random_zoom_out_option;
    CHECK_CASE(T::RandomZoomOut(), TI, zoom_u8);
    CHECK_CASE(T::RandomZoomOut(ZO().fill({0.1, 0.5, 0.9}).side_range({1.5, 2.0}).p(1.0)), f32(TI), zoom_fill_f32);
    CHECK_CASE(T::RandomZoomOut(ZO().p(0.0)), S, zoom_p0_u8);
    CHECK_CASE(T::RandomZoomOut(ZO().side_range({1.0, 1.5}).p(1.0)), B, zoom_batch_u8);
    TM_CHECK(throws_invalid([] { return T::RandomZoomOut(ZO().side_range({0.5, 2.0})); }));
    TM_CHECK(throws_invalid([] { return T::RandomZoomOut(ZO().side_range({2.0, 1.5})); }));
    TM_CHECK(throws_invalid([] { return T::RandomZoomOut(ZO().p(1.5)); }));

    CHECK_CASE(T::GaussianNoise(), f32(S), noise_f32);
    CHECK_CASE(T::GaussianNoise(T::gaussian_noise_option().mean(0.1).sigma(0.3).clip(false)), f64(S), noise_noclip_f64);
    CHECK_CASE(T::GaussianNoise(T::gaussian_noise_option().sigma(0.2)), f32(B), noise_batch_f32);
    TM_CHECK(throws_invalid([] { return T::GaussianNoise()(S); })); // floating images only
}

static void test_containers() {
    const auto jitter = [](double b, double c) {
        return T::ColorJitter(T::color_jitter_option().brightness({b}).contrast({c}));
    };
    const T::RandomRotation rot20(T::random_rotation_option().degrees({20}));
    const T::RandomHorizontalFlip hflip1(T::random_flip_option().p(1.0));
    const T::RandomVerticalFlip vflip1(T::random_flip_option().p(1.0));
    CHECK_CASE(T::RandomApply({jitter(0.3, 0.3), rot20}, 0.6), S, apply_u8);
    CHECK_CASE(T::RandomApply({jitter(0.3, 0.3), rot20}, 0.6), f32(B), apply_batch_f32);
    CHECK_CASE(T::RandomChoice({hflip1, vflip1, rot20}, {1, 2, 3}), S, choice_u8);
    CHECK_CASE(T::RandomChoice({hflip1, vflip1}), S, choice_equal_u8);
    const T::ColorJitter brightness(T::color_jitter_option().brightness({0.3}));
    CHECK_CASE(T::RandomOrder({brightness, rot20, T::Grayscale(T::grayscale_option().num_output_channels(3))}), S,
               order_u8);
    CHECK_CASE(T::RandomOrder({brightness, rot20, T::Grayscale(T::grayscale_option().num_output_channels(3))}), f32(B),
               order_batch_f32);

    // Compose chains any callables in order (a lambda included)
    const T::Compose chain({T::Resize(T::resize_option().size({6})), T::CenterCrop(T::center_crop_option().size({5})),
                            [](const torch::Tensor &t) { return t.flip(-1); }});
    TM_CHECK(torch::equal(chain(S), F::center_crop(F::resize(S, {6}), {5}).flip(-1)));
    TM_CHECK(torch::equal(chain.forward(S), chain(S)));
    // empty RandomApply / RandomOrder return the input (torchvision raises UnboundLocalError)
    TM_CHECK(T::RandomApply({}, 1.0)(S).is_same(S) && T::RandomOrder({})(S).is_same(S));
    TM_CHECK(T::RandomApply({rot20}, 0.0).forward(S).is_same(S));
    TM_CHECK(throws_invalid([] { return T::Compose({}); }));
    TM_CHECK(throws_invalid([&] { return T::RandomApply({rot20}, 1.5); }));
    TM_CHECK(throws_invalid([&] { return T::RandomChoice({rot20, hflip1}, {1.0}); }));
    TM_CHECK(throws_invalid([&] { return T::RandomChoice({rot20, hflip1}, {0.0, 0.0}); }));
}

static void test_deterministic_misc() {
    const T::Identity id;
    TM_CHECK(id(S).is_same(S) && id.forward(S).is_same(S));
    const T::ConvertImageDtype to_float;
    TM_CHECK(torch::equal(to_float(S), F::to_dtype(S, torch::kFloat32, true)));
    const T::ConvertImageDtype to_u8(T::convert_image_dtype_option().dtype(torch::kUInt8));
    TM_CHECK(torch::equal(to_u8.forward(f32(S)), F::to_dtype(f32(S), torch::kUInt8, true)));

    const auto m = ((torch::arange(144) * 7).remainder(11) - 5).to(torch::kFloat32).reshape({12, 12}) / 8;
    const auto mean = torch::arange(12).remainder(5).to(torch::kFloat32) / 10;
    using LO = T::linear_transformation_option;
    const T::LinearTransformation lt(LO().transformation_matrix(m).mean_vector(mean));
    CHECK_ENTRY(lt(f32(S).slice(1, 0, 2).slice(2, 0, 2)), lt_f32);
    CHECK_ENTRY(lt.forward(B.slice(2, 0, 2).slice(3, 0, 2)), lt_batch_u8); // uint8 - float mean: float32 output
    CHECK_ENTRY(lt(f64(S).slice(1, 0, 2).slice(2, 0, 2)), lt_f64);
    TM_CHECK(throws_invalid([&] { return lt(f32(S)); })); // 3 * 9 * 11 != 12
    TM_CHECK(throws_invalid([&] { return lt(torch::zeros({3, 2, 2}, torch::kMeta)); })); // device type
    TM_CHECK(throws_invalid(
            [&] { return T::LinearTransformation(LO().transformation_matrix(m.slice(0, 0, 6)).mean_vector(mean)); }));
    TM_CHECK(throws_invalid(
            [&] { return T::LinearTransformation(LO().transformation_matrix(m).mean_vector(mean.slice(0, 0, 6))); }));
    TM_CHECK(throws_invalid(
            [&] { return T::LinearTransformation(LO().transformation_matrix(m).mean_vector(mean.to(torch::kMeta))); }));
    TM_CHECK(throws_invalid([&] {
        return T::LinearTransformation(LO().transformation_matrix(m).mean_vector(mean.to(torch::kFloat64)));
    }));
}

// The ImageNet training recipe: RandomResizedCrop(224) + RandomHorizontalFlip + ColorJitter(0.4, 0.4, 0.4) +
// ToDtype(float32, scale) + Normalize, seeded end to end through Compose.
static void test_train_recipe() {
    const auto img = synthetic_image();
    const T::Compose train(
            {T::RandomResizedCrop(T::random_resized_crop_option().size({224})), T::RandomHorizontalFlip(),
             T::ColorJitter(T::color_jitter_option().brightness({0.4}).contrast({0.4}).saturation({0.4})),
             T::ToDtype(T::to_dtype_option().dtype(torch::kFloat32).scale(true)),
             T::Normalize(T::normalize_option().mean({0.485, 0.456, 0.406}).std({0.229, 0.224, 0.225}))});
    using namespace golden;
    const std::array<const std::vector<double> *, 3> grids{&train0_grid, &train1_grid, &train2_grid};
    const std::array<double, 3> sums{train0_sum, train1_sum, train2_sum};
    const std::array<double, 3> sums_sq{train0_sum_sq, train1_sum_sq, train2_sum_sq};
    const std::array<double, 3> next{train0_next_draw, train1_next_draw, train2_next_draw};
    for (int seed = 0; seed < 3; ++seed) {
        torch::manual_seed(seed);
        check_recipe(train(img), *grids[seed], sums[seed], sums_sq[seed]);
        TM_CHECK(torch::rand({1}).item<double>() == next[seed]);
    }
}

int main() {
    TM_RUN(test_imagenet_recipe);
    TM_RUN(test_classes_match_functional);
    TM_RUN(test_random_flips);
    TM_RUN(test_validation);
    TM_RUN(test_gaussian_blur);
    TM_RUN(test_random_color);
    TM_RUN(test_random_crop);
    TM_RUN(test_random_resized_crop);
    TM_RUN(test_random_resize);
    TM_RUN(test_color_jitter);
    TM_RUN(test_random_warps);
    TM_RUN(test_erasing_zoom_noise);
    TM_RUN(test_containers);
    TM_RUN(test_deterministic_misc);
    TM_RUN(test_train_recipe);
    return tm_test::summary("vision_test_transform");
}
