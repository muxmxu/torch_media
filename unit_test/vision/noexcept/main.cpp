// LIB_TORCH_MEDIA_NO_EXCEPTIONS build of vision functional / io: invalid input returns the silent fallback (an
// undefined tensor, an empty vector) and records the message for torchmedia::last_error()
// (develop_log/2026-10-24/progress05). One check per error site, so a defined fallback at any site fails here.
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <string>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <unistd.h>
#include "test_util.hpp"

namespace vf = torchmedia::vision::functional;
namespace vio = torchmedia::vision::io;
namespace fs = std::filesystem;

// The fallback plus a recorded message containing `expect` (consumed, so the next check needs its own).
static auto failed(bool fallback, const std::string &expect) -> bool {
    const auto msg = torchmedia::last_error();
    torchmedia::clear_last_error();
    if (!fallback || msg.find(expect) == std::string::npos) {
        std::fprintf(stderr, "    fallback %d, last_error() = \"%s\", expected \"%s\"\n", fallback, msg.c_str(),
                     expect.c_str());
        return false;
    }
    return true;
}
static auto undef(const torch::Tensor &t) -> bool { return !t.defined(); }

static void test_functional_sites() {
    const auto img = torch::rand({3, 4, 4});
    const auto two = torch::rand({2, 4, 4}); // neither 1 nor 3 channels
    const auto hw = torch::rand({4, 4});
    const auto line = torch::rand({4});
    const auto u8 = torch::zeros({1, 2, 2}, torch::kUInt8);
    const auto b = torch::ones({1, 2, 2}, torch::kBool);
    using im = vf::interpolation_mode;

    TM_CHECK(failed(undef(vf::rgb_to_grayscale(img, 2)), "num_output_channels must be 1 or 3"));
    TM_CHECK(failed(undef(vf::rgb_to_grayscale(two)), "rgb_to_grayscale: expected"));
    TM_CHECK(failed(undef(vf::permute_channels(hw, {0})), "permute_channels"));
    TM_CHECK(failed(undef(vf::permute_channels(img, {0, 1})), "length of permutation"));
    TM_CHECK(failed(undef(vf::normalize(u8, {0.5}, {0.5})), "should be a float tensor"));
    TM_CHECK(failed(undef(vf::normalize(hw, {0.5}, {0.5})), "normalize"));
    TM_CHECK(failed(undef(vf::normalize(img, {0.5}, {0.0})), "std evaluated to zero"));
    TM_CHECK(failed(undef(vf::crop(line, 0, 0, 1, 1)), "crop"));
    TM_CHECK(failed(undef(vf::crop(hw, -1, 0, 2, 2)), "crop")); // padding needs a channel dim
    TM_CHECK(failed(undef(vf::center_crop(img, {})), "output_size must have 1 or 2 elements"));
    TM_CHECK(failed(undef(vf::center_crop(line, {2})), "center_crop"));
    TM_CHECK(failed(undef(vf::pad(img, {1, 2, 3})), "not a 3 element tuple"));
    TM_CHECK(failed(undef(vf::pad(img, {1}, {0.1, 0.2, 0.3}, vf::padding_mode::reflect)), "non-scalar fill"));
    TM_CHECK(failed(undef(vf::pad(hw, {1})), "pad"));
    TM_CHECK(failed(undef(vf::resize(hw, {2})), "resize"));
    TM_CHECK(failed(undef(vf::resize(img, {1, 2, 3})), "resize: size must have 1 or 2 elements"));
    TM_CHECK(failed(undef(vf::resized_crop(hw, 0, 0, 2, 2, {2})), "resized_crop"));
    TM_CHECK(failed(vf::five_crop(img, {8}).empty(), "five_crop: Requested crop size"));
    TM_CHECK(failed(vf::ten_crop(img, {8}).empty(), "ten_crop: Requested crop size"));
    TM_CHECK(failed(undef(vf::to_dtype(img, torch::kInt32, true)), "cannot be performed safely"));
    TM_CHECK(failed(undef(vf::to_dtype(b, torch::kUInt8, true)), "only defined for integer dtypes"));
    TM_CHECK(failed(undef(vf::adjust_brightness(img, -1.0)), "brightness_factor"));
    TM_CHECK(failed(undef(vf::adjust_brightness(two, 0.5)), "adjust_brightness: expected"));
    TM_CHECK(failed(undef(vf::invert(b)), "invert: only floating and integer"));
    TM_CHECK(failed(undef(vf::adjust_contrast(img, -1.0)), "contrast_factor"));
    TM_CHECK(failed(undef(vf::adjust_contrast(two, 0.5)), "adjust_contrast: expected"));
    TM_CHECK(failed(undef(vf::adjust_saturation(img, -1.0)), "saturation_factor"));
    TM_CHECK(failed(undef(vf::adjust_saturation(two, 0.5)), "adjust_saturation: expected"));
    TM_CHECK(failed(undef(vf::adjust_sharpness(two, 0.5)), "adjust_sharpness: expected"));
    TM_CHECK(failed(undef(vf::adjust_sharpness(img, -1.0)), "sharpness_factor"));
    TM_CHECK(failed(undef(vf::adjust_hue(img, 0.6)), "hue_factor"));
    TM_CHECK(failed(undef(vf::adjust_hue(two, 0.1)), "adjust_hue: expected"));
    TM_CHECK(failed(undef(vf::adjust_gamma(img, -1.0)), "gamma should be a non-negative"));
    TM_CHECK(failed(undef(vf::posterize(img, 63)), "bits must be in [0, 62]"));
    TM_CHECK(failed(undef(vf::posterize(u8, -1)), "bits >= 0"));
    TM_CHECK(failed(undef(vf::solarize(img, 2.0)), "solarize: threshold"));
    TM_CHECK(failed(undef(vf::autocontrast(two)), "autocontrast: expected"));
    TM_CHECK(failed(undef(vf::equalize(line)), "equalize"));
    TM_CHECK(failed(undef(vf::gaussian_blur(img, {3, 3, 3})), "kernel_size must have 1 or 2 elements"));
    TM_CHECK(failed(undef(vf::gaussian_blur(img, {2})), "odd and positive"));
    TM_CHECK(failed(undef(vf::gaussian_blur(img, {3}, {-1.0})), "sigma should have 1 or 2 positive values"));
    TM_CHECK(failed(undef(vf::gaussian_blur(hw, {3})), "gaussian_blur"));
    TM_CHECK(failed(undef(vf::gaussian_noise(u8)), "expected to be in float dtype"));
    TM_CHECK(failed(undef(vf::gaussian_noise(img, 0.0, -1.0)), "sigma shouldn't be negative"));
    TM_CHECK(failed(undef(vf::rotate(img, 10.0, im::bicubic)), "rotate: only nearest and bilinear"));
    TM_CHECK(failed(undef(vf::affine(img, 0.0, {0.0}, 1.0, {0.0})), "translate should be a sequence of length 2"));
    TM_CHECK(failed(undef(vf::affine(img, 0.0, {0.0, 0.0}, 0.0, {0.0})), "scale should be positive"));
    TM_CHECK(failed(undef(vf::affine(img, 0.0, {0.0, 0.0}, 1.0, {0.0, 0.0, 0.0})), "shear should contain"));
    TM_CHECK(failed(undef(vf::affine(hw, 0.0, {0.0, 0.0}, 1.0, {0.0})), "affine"));
    TM_CHECK(failed(undef(vf::perspective(img, {}, {})), "perspective: either the startpoints/endpoints"));
    TM_CHECK(failed(undef(vf::perspective(hw, {}, {}, im::bilinear, {}, {1, 0, 0, 0, 1, 0, 0, 0})), "perspective"));
    const auto disp = torch::zeros({1, 4, 4, 2});
    TM_CHECK(failed(undef(vf::elastic(hw, disp)), "elastic"));
    TM_CHECK(failed(undef(vf::elastic(img, disp, im::nearest_exact)), "nearest_exact interpolation"));
    TM_CHECK(failed(undef(vf::elastic(img, torch::zeros({1, 3, 3, 2}))), "displacement shape"));
}

static auto bytes(const std::string &s) -> torch::Tensor {
    return torch::from_blob(const_cast<char *>(s.data()), {static_cast<int64_t>(s.size())}, torch::kUInt8).clone();
}

static void test_io_sites() {
    const auto dir = fs::temp_directory_path() / std::format("torchmedia_vision_noexcept_{}", ::getpid());
    fs::create_directories(dir);
    const auto bad_png = std::string("\x89PNG\r\n\x1a\n", 8) + std::string(40, 'x');
    std::ofstream(dir / "bad.png") << bad_png;
    std::ofstream(dir / "bad.pgm") << "P2 1 1 255\nx";
    TM_CHECK(failed(undef(vio::load_image((dir / "none.png").string())), "File does not exist"));
    TM_CHECK(failed(undef(vio::load_image((dir / "bad.pgm").string())), "PNM: expected a decimal integer"));
    TM_CHECK(failed(undef(vio::load_image((dir / "bad.png").string())), "Could not load image"));
    TM_CHECK(failed(undef(vio::decode_image(torch::empty({0}, torch::kUInt8))), "expected a non-empty 1-D uint8"));
    const auto jpeg_eoi = bytes(std::string("\xFF\xD8\xFF\xD9", 4)); // an EOI and nothing to decode
    TM_CHECK(failed(undef(vio::decode_image(jpeg_eoi, vio::image_read_mode::rgb_alpha)), "not supported for JPEG"));
    TM_CHECK(failed(undef(vio::decode_image(bytes(std::string("\xFF\xD8\xFF\xE0", 4)))), "incomplete or truncated"));
    TM_CHECK(failed(undef(vio::decode_image(jpeg_eoi)), "decode_image: could not decode"));
    TM_CHECK(failed(undef(vio::decode_image(bytes(bad_png), vio::image_read_mode::gray)), "could not decode"));
    TM_CHECK(failed(undef(vio::decode_image(bytes("GIF89a......"))), "only JPEG and PNG are supported"));
    TM_CHECK(failed(undef(vio::read_file(dir / "none.png")), "read_file: could not read"));
    fs::remove_all(dir);
}

int main() {
    TM_RUN(test_functional_sites);
    TM_RUN(test_io_sites);
    return tm_test::summary("vision_test_noexcept");
}
