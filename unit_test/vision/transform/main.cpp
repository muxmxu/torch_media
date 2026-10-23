// Assertion-based tests for the vision transform classes: each class equals its golden-verified functional op, the
// ImageNet recipe Resize(256) + CenterCrop(224) + ToDtype(float32, scale) + Normalize matches torchvision 0.20.1
// end to end, and the random flips draw like torchvision v2 (gen_golden.py -> golden.hpp).
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "golden.hpp"
#include "test_util.hpp"

namespace F = torchmedia::vision::functional;
namespace T = torchmedia::vision::transform;

// Bit-exact on arm64, where the goldens were generated. Elsewhere torch may resize uint8 natively (x86 AVX2, one
// uint8 step = 1 / 255 / 0.224 after Normalize) or contract FMAs differently.
#if defined(__aarch64__)
constexpr double kTolPixel = 0.0;
constexpr double kTolSumRel = 0.0;
#else
constexpr double kTolPixel = 0.02;
constexpr double kTolSumRel = 1e-4;
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

int main() {
    TM_RUN(test_imagenet_recipe);
    TM_RUN(test_classes_match_functional);
    TM_RUN(test_random_flips);
    TM_RUN(test_validation);
    TM_RUN(test_gaussian_blur);
    TM_RUN(test_random_color);
    return tm_test::summary("vision_test_transform");
}
