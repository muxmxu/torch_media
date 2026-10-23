// Image I/O tests (stb_image): float load / save round trip, error / overload branches, and torchvision.io-style
// decode_image / read_image against torchvision 0.20.1 goldens (gen_golden.py -> golden.hpp): PNG bit-exact in every
// read mode, JPEG within the measured stb_image vs libjpeg-turbo bound (develop_log/2026-10-23/progress09).
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "golden.hpp"
#include "test_util.hpp"

using namespace torchmedia::vision;

template<class E, class Fn>
static auto throws(Fn &&fn) -> bool {
    try {
        (void) fn();
    } catch (const E &) {
        return true;
    }
    return false;
}

static void test_roundtrip() {
    auto img = torch::arange(3 * 8 * 8, torch::kFloat32).reshape({3, 8, 8}) / static_cast<double>(3 * 8 * 8);
    const std::string path = "roundtrip_img.png";
    TM_CHECK(io::save_image(img, path));
    auto loaded = io::load_image(path);
    TM_CHECK(loaded.size(0) == 3 && loaded.size(1) == 8 && loaded.size(2) == 8);
    const float max_err = (loaded - img).abs().max().item<float>();
    TM_CHECK(max_err < (1.0f / 255.0f) + 1e-4f); // uint8 quantization
}

static void test_io_branches() {
    // non-existent file -> throws
    bool threw_missing = false;
    try {
        (void) io::load_image("nonexistent_xyz.png");
    } catch (const std::runtime_error &) {
        threw_missing = true;
    }
    TM_CHECK(threw_missing);

    // non-image file -> stbi_load returns null -> throws
    { std::ofstream f("not_an_image.png"); f << "not a png"; }
    bool threw_bad = false;
    try {
        (void) io::load_image("not_an_image.png");
    } catch (const std::runtime_error &) {
        threw_bad = true;
    }
    TM_CHECK(threw_bad);

    // wrong dim -> false
    TM_CHECK(io::save_image(torch::zeros({8}), "bad.png") == false);

    // const char* and std::filesystem::path overloads
    auto img = torch::rand({3, 4, 4});
    TM_CHECK(io::save_image(img, std::filesystem::path("rt_path.png")));
    TM_CHECK(io::load_image(std::filesystem::path("rt_path.png")).size(0) == 3);
    TM_CHECK(io::save_image(img, "rt_cstr.png"));
    TM_CHECK(io::load_image("rt_cstr.png").size(0) == 3);
}

using mode = io::image_read_mode;

static auto bytes(const golden::entry &e) -> torch::Tensor { return torch::tensor(e.data).to(torch::kUInt8); }

// The decode of `file` in `m` must equal the golden in dtype, shape and values (uint16 goldens are stored as int32),
// or stay within max_tol / mean_tol levels (JPEG).
static void check_decode(const golden::entry &file, mode m, const golden::entry &want, const char *name, int line,
                         double max_tol = 0, double mean_tol = 0) {
    auto out = io::decode_image(bytes(file), m);
    const bool u16 = out.scalar_type() == torch::kUInt16;
    if (u16) {
        out = out.to(torch::kInt32);
    }
    const bool dtype_ok = want.dtype == (u16 ? "int32" : "uint8") && (u16 || out.scalar_type() == torch::kUInt8);
    tm_test::record(dtype_ok, name, __FILE__, line, "dtype");
    const auto expected = torch::tensor(want.data).reshape(want.shape);
    std::string detail;
    tm_test::record(tm_test::tensor_close(out, expected, max_tol, 0.0, detail), name, __FILE__, line, detail);
    if (out.sizes() == expected.sizes() && mean_tol > 0) {
        const auto mean = (out.to(torch::kFloat64) - expected.to(torch::kFloat64)).abs().mean().item<double>();
        tm_test::record(mean <= mean_tol, name, __FILE__, line, "mean|delta|=" + std::to_string(mean));
    }
}
#define CHECK_DECODE(file, m) check_decode(golden::file, mode::m, golden::file##_##m, #file "_" #m, __LINE__)
// stb_image (patched) vs libjpeg-turbo, measured over 255 JPEGs (progress09 D6): max |delta| 3 (1 in gray mode), mean
// |delta| per file <= 0.06 for 4:4:4 / 4:2:2 / gray and <= 0.12 for 4:2:0
#define CHECK_JPEG(file, m, max_tol, mean_tol)                                                                         \
    check_decode(golden::file, mode::m, golden::file##_##m, #file "_" #m, __LINE__, max_tol, mean_tol)

static void test_decode_png() {
    CHECK_DECODE(png_l, unchanged);
    CHECK_DECODE(png_l, gray_alpha);
    CHECK_DECODE(png_l, rgb);
    CHECK_DECODE(png_l, rgb_alpha);
    CHECK_DECODE(png_la, unchanged);
    CHECK_DECODE(png_la, gray);
    CHECK_DECODE(png_la, rgb_alpha);
    CHECK_DECODE(png_rgb, unchanged);
    CHECK_DECODE(png_rgb, gray); // libpng's truncating 15-bit weights, not stb's
    CHECK_DECODE(png_rgb, gray_alpha);
    CHECK_DECODE(png_rgba, unchanged);
    CHECK_DECODE(png_rgba, gray);
    CHECK_DECODE(png_rgba, rgb);
    CHECK_DECODE(png_1bit, unchanged);
    CHECK_DECODE(png_1bit, gray_alpha); // alpha = (1 << bit depth) - 1, as torchvision passes the filler
    CHECK_DECODE(png_palette, gray);
    CHECK_DECODE(png_palette, rgb);
    CHECK_DECODE(png_palette_trns, gray_alpha);
    CHECK_DECODE(png_palette_trns, rgb_alpha);
    CHECK_DECODE(png_rgb_trns, unchanged); // tRNS of an RGB image is ignored, like libpng
    CHECK_DECODE(png_rgb_trns, rgb_alpha);
    CHECK_DECODE(png_rgb_gama, gray); // linear-light gray through libpng's gamma tables
    CHECK_DECODE(png_rgb_gama, gray_alpha);
    CHECK_DECODE(png_rgba_srgb, gray);
    CHECK_DECODE(png_rgb_gama1, gray); // gamma 1.0: not significant, plain weights
    CHECK_DECODE(png16_rgb, unchanged);
    CHECK_DECODE(png16_rgb, gray);
    CHECK_DECODE(png16_rgb, rgb_alpha);
    CHECK_DECODE(png16_rgb_gama, gray);
    CHECK_DECODE(png16_l, gray_alpha);
    // gAMA / sRGB kept as libpng keeps them
    CHECK_DECODE(png_rgb_two_gama, gray); // the first gAMA wins
    CHECK_DECODE(png_rgb_srgb_gama1, gray); // sRGB wins over a later gAMA
    CHECK_DECODE(png_rgb_srgb_len2, gray); // bad length: skipped
    CHECK_DECODE(png_rgb_srgb_intent7, gray); // invalid intent: no gamma
    CHECK_DECODE(png_rgb_gama_tiny_srgb, gray); // out-of-range gAMA invalidates the later sRGB too
    CHECK_DECODE(png_rgb_gama097, gray); // within 5% of 1.0: not significant
    CHECK_DECODE(png_palette_gama_after_plte, gray); // gAMA after PLTE: skipped
    // deviation: a palette PNG decodes to its colours in unchanged mode (torchvision returns the palette indices)
    const auto pal = bytes(golden::png_palette);
    TM_CHECK(torch::equal(io::decode_image(pal), io::decode_image(pal, mode::rgb)));
}

static void test_decode_jpeg() {
    CHECK_JPEG(jpeg_444, unchanged, 3, 0.06);
    CHECK_JPEG(jpeg_444, gray, 1, 0.06);
    CHECK_JPEG(jpeg_422, unchanged, 3, 0.06); // stb h2v1 patches: max 121 / mean 0.17 without them
    CHECK_JPEG(jpeg_422, gray, 1, 0.06);
    CHECK_JPEG(jpeg_420, unchanged, 3, 0.12);
    CHECK_JPEG(jpeg_420, gray, 1, 0.06);
    CHECK_JPEG(jpeg_gray, unchanged, 1, 0.06);
    CHECK_JPEG(jpeg_gray, rgb, 1, 0.06);
    CHECK_JPEG(jpeg_narrow, unchanged, 3, 0.12); // chroma 2 samples wide: replicated (max 61 without the stb patch)
    CHECK_JPEG(jpeg_rst, unchanged, 3, 0.12); // restart markers
    const auto jpg = bytes(golden::jpeg_444);
    TM_CHECK(torch::equal(io::decode_image(jpg, mode::rgb), io::decode_image(jpg)));
    TM_CHECK(throws<std::invalid_argument>([&] { return io::decode_image(jpg, mode::gray_alpha); }));
    TM_CHECK(throws<std::invalid_argument>([&] { return io::decode_image(jpg, mode::rgb_alpha); }));

    // truncated data (no EOI marker) raises like torchvision ("Image is incomplete or truncated"); bytes after the EOI
    // are ignored
    const auto n = jpg.numel();
    TM_CHECK(throws<std::runtime_error>([&] { return io::decode_image(jpg.slice(0, 0, n * 6 / 10)); }));
    TM_CHECK(throws<std::runtime_error>([&] { return io::decode_image(jpg.slice(0, 0, n - 2)); }));
    TM_CHECK(throws<std::runtime_error>([&] { return io::decode_image(jpg.slice(0, 0, n - 1)); }));
    TM_CHECK(torch::equal(io::decode_image(torch::cat({jpg, torch::full({5}, 0xFF, torch::kUInt8)})),
                          io::decode_image(jpg)));
    // 0xFF fill bytes before a marker are legal (torchvision decodes them)
    const auto fill = torch::cat({jpg.slice(0, 0, n - 2), torch::full({2}, 0xFF, torch::kUInt8), jpg.slice(0, n - 2)});
    TM_CHECK(torch::equal(io::decode_image(fill), io::decode_image(jpg)));
}

static void test_read_image() {
    const auto data = bytes(golden::png_rgba);
    const std::string path = "decode_rgba.png";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(data.data_ptr<uint8_t>()), data.numel());
    }
    TM_CHECK(torch::equal(io::read_file(path), data));
    TM_CHECK(torch::equal(io::read_image(path), io::decode_image(data)));
    TM_CHECK(torch::equal(io::read_image(std::filesystem::path(path), mode::rgb), io::decode_image(data, mode::rgb)));
    TM_CHECK(throws<std::runtime_error>([] { return io::read_file("nonexistent_xyz.png"); }));
    TM_CHECK(throws<std::runtime_error>([] { return io::read_image("nonexistent_xyz.png"); }));
    // a directory or an empty file raises like torchvision (no uninitialised bytes)
    TM_CHECK(throws<std::runtime_error>([] { return io::read_file(std::filesystem::current_path()); }));
    std::ofstream("empty_file.bin").close();
    TM_CHECK(throws<std::runtime_error>([] { return io::read_file("empty_file.bin"); }));

    // argument checks (torchvision: RuntimeError) and undecodable data
    TM_CHECK(throws<std::invalid_argument>([] { return io::decode_image(torch::zeros({0}, torch::kUInt8)); }));
    TM_CHECK(throws<std::invalid_argument>([&] { return io::decode_image(data.to(torch::kInt32)); }));
    TM_CHECK(throws<std::invalid_argument>([&] { return io::decode_image(data.view({1, -1})); }));
    TM_CHECK(throws<std::runtime_error>(
            [] { return io::decode_image(torch::tensor({71, 73, 70, 56, 57, 97}).to(torch::kUInt8)); }));
    auto truncated = data.slice(0, 0, 40).clone(); // valid signature and IHDR, no image data
    TM_CHECK(throws<std::runtime_error>([&] { return io::decode_image(truncated); }));
    TM_CHECK(throws<std::runtime_error>(
            [] { return io::decode_image(torch::tensor({255, 216, 255, 0}).to(torch::kUInt8)); }));
}

int main() {
    test_roundtrip();
    test_io_branches();
    TM_RUN(test_decode_png);
    TM_RUN(test_decode_jpeg);
    TM_RUN(test_read_image);
    return tm_test::summary("vision_test_io");
}
