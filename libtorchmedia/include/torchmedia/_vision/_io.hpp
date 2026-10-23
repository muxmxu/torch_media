#pragma once
#ifndef LIB_TORCH_MEDIA_VISION_IO_HPP
#define LIB_TORCH_MEDIA_VISION_IO_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#include "../globel_include.hpp"

// stb_image / stb_image_write: header-only image codecs (public domain / MIT). The implementation is
// emitted in the one TU that defines TORCHMEDIA_IO_IMPLEMENTATION (same macro as dr_wav), keeping the
// library header-only while avoiding ODR violations. See develop_log/2026-05-31/progress04.
#ifdef TORCHMEDIA_IO_IMPLEMENTATION
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#endif
#include "_vendor/stb_image.h"
#include "_vendor/stb_image_write.h"

namespace torchmedia::vision::io {

    // Load an image into a [C, H, W] float32 tensor in [0, 1].
    inline auto load_image(const std::string &path) -> tensor_t {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }
        int w = 0, h = 0, c = 0;
        unsigned char *data = stbi_load(path.c_str(), &w, &h, &c, 0);
        if (data == nullptr) {
            throw std::runtime_error("Could not load image: " + path);
        }
        // stb gives [H, W, C] uint8 row-major; -> float [0,1] -> [C, H, W]. permute+contiguous copies out
        // of the stb buffer, so it is safe to free immediately afterwards.
        const auto hwc = torch::from_blob(data, {h, w, c}, torch::kUInt8).to(torch::kFloat32) / 255.0;
        auto chw = hwc.permute({2, 0, 1}).contiguous();
        stbi_image_free(data);
        return chw;
    }

    // Save a [C, H, W] float tensor (values in [0,1]) as a PNG. Returns false on bad input / IO error.
    inline auto save_image(const tensor_t &img, const std::string &path) -> bool {
        if (img.dim() != 3) {
            return false;
        }
        const auto c = img.size(0);
        const auto h = img.size(1);
        const auto w = img.size(2);
        // [C,H,W] float [0,1] -> [H,W,C] uint8 (clamp, scale, round).
        const auto hwc = (img.detach().to(torch::kCPU).clamp(0.0, 1.0) * 255.0 + 0.5)
                                 .to(torch::kUInt8)
                                 .permute({1, 2, 0})
                                 .contiguous();
        const int stride = static_cast<int>(w * c);
        const int ok = stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h),
                                      static_cast<int>(c), hwc.data_ptr<uint8_t>(), stride);
        return ok != 0;
    }

    inline auto load_image(const char *path) -> tensor_t { return load_image(std::string(path)); }
    inline auto load_image(const std::filesystem::path &path) -> tensor_t { return load_image(path.string()); }
    inline auto save_image(const tensor_t &img, const char *path) -> bool {
        return save_image(img, std::string(path));
    }
    inline auto save_image(const tensor_t &img, const std::filesystem::path &path) -> bool {
        return save_image(img, path.string());
    }

    // ---------------- torchvision.io-style decoding: uint8 [C, H, W] (uint16 for 16-bit PNG) ----------------
    // torchvision ImageReadMode. JPEG supports unchanged / gray / rgb only (torchvision raises for the alpha modes).
    enum class image_read_mode { unchanged, gray, gray_alpha, rgb, rgb_alpha };

    namespace detail {
        struct png_header {
            int bit_depth = 8;
            int color_type = -1;
            int64_t gamma = 0; // libpng fixed point (1/2.2 = 45455); 0 = no gAMA / sRGB chunk
        };

        // IHDR depth / color type and the gamma libpng keeps from the gAMA / sRGB chunks (png_handle_gAMA / _sRGB,
        // png_colorspace_set_gamma / _sRGB): a chunk with a bad length or after PLTE is skipped; an out-of-range or
        // second gAMA, a second sRGB or an sRGB intent >= 4 marks the colour space invalid, which keeps the gamma
        // stored so far and skips every later gAMA / sRGB; sRGB (45455) wins over gAMA in either order.
        inline auto read_png_header(const uint8_t *p, int64_t n) -> png_header {
            png_header hdr;
            const auto be32 = [&](int64_t o) {
                return (int64_t{p[o]} << 24) | (int64_t{p[o + 1]} << 16) | (int64_t{p[o + 2]} << 8) | p[o + 3];
            };
            bool plte = false, invalid = false, gama = false, srgb = false;
            for (int64_t off = 8; off + 8 <= n;) {
                const auto len = be32(off);
                const auto *type = p + off + 4;
                if (off + 12 + len > n || std::memcmp(type, "IDAT", 4) == 0) {
                    break;
                }
                const bool colorspace = !plte && !invalid;
                if (std::memcmp(type, "IHDR", 4) == 0 && len >= 10) {
                    hdr.bit_depth = p[off + 16];
                    hdr.color_type = p[off + 17];
                } else if (std::memcmp(type, "PLTE", 4) == 0) {
                    plte = true;
                } else if (colorspace && std::memcmp(type, "gAMA", 4) == 0 && len == 4) {
                    const auto g = be32(off + 8);
                    invalid = g < 16 || g > 625000000 || gama;
                    gama = true;
                    hdr.gamma = invalid || srgb ? hdr.gamma : g;
                } else if (colorspace && std::memcmp(type, "sRGB", 4) == 0 && len == 1) {
                    invalid = srgb || p[off + 8] >= 4;
                    srgb = true;
                    hdr.gamma = invalid ? hdr.gamma : 45455; // PNG_GAMMA_sRGB_INVERSE
                }
                off += 12 + len;
            }
            return hdr;
        }

        // libpng png_reciprocal / png_gamma_significant / png_build_8bit_table / png_build_16bit_table (floating-point
        // build, no sBIT shift): table[v] = floor(max * (v / max)^(g / 1e5) + .5).
        inline auto png_reciprocal(int64_t a) -> int64_t {
            const double r = std::floor(1e10 / static_cast<double>(a) + .5);
            return r <= 2147483647. && r >= -2147483648. ? static_cast<int64_t>(r) : 0;
        }
        inline auto png_gamma_significant(int64_t g) -> bool { return g < 95000 || g > 105000; }
        inline auto png_gamma_table(int64_t g, bool sixteen) -> tensor_t {
            auto table = torch::arange(sixteen ? 65536 : 256, torch::kInt64);
            if (png_gamma_significant(g)) {
                auto *t = table.data_ptr<int64_t>();
                const double e = static_cast<double>(g) * .00001;
                const double fmax = 1.0 / 65535;
                for (int64_t i = sixteen ? 0 : 1; i < (sixteen ? 65536 : 255); ++i) {
                    // separate statements: no FMA contraction of max * pow + .5
                    const double v = sixteen ? 65535. * std::pow(static_cast<double>(i) * fmax, e)
                                             : 255 * std::pow(static_cast<double>(i) / 255., e);
                    t[i] = static_cast<int64_t>(std::floor(v + .5));
                }
            }
            return table;
        }

        // libpng png_do_rgb_to_gray with the weights torchvision sets (png_set_rgb_to_gray(png, 1, 0.2989, 0.587):
        // 9794, 19234, 3740 / 32768). 8-bit rows truncate and 16-bit rows round; with a significant file gamma both
        // round in linear light (through the gamma tables), gray pixels passing through unchanged.
        inline auto png_rgb_to_gray(const tensor_t &rgb, bool sixteen, int64_t file_gamma) -> tensor_t {
            const auto c = rgb.unbind(0);
            const auto weigh = [](const tensor_t &r, const tensor_t &g, const tensor_t &b, int64_t round) {
                return (r * 9794 + g * 19234 + b * 3740 + round).bitwise_right_shift(15);
            };
            const auto screen = file_gamma != 0 ? png_reciprocal(file_gamma) : 0;
            if (file_gamma == 0 || !(png_gamma_significant(file_gamma) || png_gamma_significant(screen))) {
                return weigh(c[0], c[1], c[2], sixteen ? 16384 : 0).unsqueeze(0);
            }
            const auto to_1 = png_gamma_table(png_reciprocal(file_gamma), sixteen);
            const auto from_1 = png_gamma_table(png_reciprocal(screen), sixteen);
            const auto lin = weigh(to_1.index({c[0]}), to_1.index({c[1]}), to_1.index({c[2]}), 16384);
            return torch::where((c[0] == c[1]) & (c[0] == c[2]), c[0], from_1.index({lin})).unsqueeze(0);
        }

        // libjpeg-turbo reads up to the EOI marker and torchvision raises when the data ends first; stb_image pads a
        // truncated stream instead. Marker segments carry their length; in entropy-coded data 0xFF is followed only by
        // 0x00 (stuffing), 0xFF (fill) or a marker, so an FF D9 found this way is the EOI.
        inline auto jpeg_has_eoi(const uint8_t *p, int64_t n) -> bool {
            for (int64_t i = 2; i + 1 < n;) {
                const auto m = p[i + 1];
                if (p[i] != 0xFF || m == 0xFF) {
                    ++i;
                } else if (m == 0xD9) {
                    return true;
                } else if (m == 0x00 || m == 0x01 || (m >= 0xD0 && m <= 0xD8)) {
                    i += 2; // stuffing, TEM, RSTn, SOI: no length
                } else {
                    i = i + 3 < n ? i + 2 + ((p[i + 2] << 8) | p[i + 3]) : n;
                }
            }
            return false;
        }

        // stb_image decode to [C, H, W]; req = 0 keeps the file's channels.
        inline auto stb_decode(const uint8_t *p, int64_t n, int req, bool sixteen) -> tensor_t {
            int w = 0, h = 0, c = 0;
            const int len = static_cast<int>(n);
            void *data = sixteen ? static_cast<void *>(stbi_load_16_from_memory(p, len, &w, &h, &c, req))
                                 : static_cast<void *>(stbi_load_from_memory(p, len, &w, &h, &c, req));
            if (data == nullptr) {
                throw std::runtime_error(std::string("decode_image: could not decode the image: ") +
                                         stbi_failure_reason());
            }
            const auto out =
                    torch::from_blob(data, {h, w, req != 0 ? req : c}, sixteen ? torch::kUInt16 : torch::kUInt8)
                            .permute({2, 0, 1})
                            .clone(torch::MemoryFormat::Contiguous);
            stbi_image_free(data);
            return out;
        }
    } // namespace detail

    // torchvision.io.decode_image: encoded JPEG / PNG bytes (1-D uint8 CPU tensor) -> [C, H, W] uint8 (uint16 for
    // 16-bit PNG). PNG is bit-exact with torchvision (libpng), mode conversions included; JPEG goes through stb_image,
    // whose IDCT / colour conversion / upsampling rounding differ from libjpeg-turbo by a few levels (progress09 D6).
    inline auto decode_image(const tensor_t &data, image_read_mode mode = image_read_mode::unchanged) -> tensor_t {
        if (!data.device().is_cpu() || data.scalar_type() != torch::kUInt8 || data.dim() != 1 || data.numel() == 0 ||
            data.numel() > std::numeric_limits<int>::max()) {
            return handle_exceptions<tensor_t, std::invalid_argument>(
                    "decode_image: expected a non-empty 1-D uint8 CPU tensor of encoded bytes.");
        }
        const auto bytes = data.contiguous();
        const auto *p = bytes.data_ptr<uint8_t>();
        const auto n = bytes.numel();
        if (n >= 3 && p[0] == 0xFF && p[1] == 0xD8 && p[2] == 0xFF) {
            if (mode == image_read_mode::gray_alpha || mode == image_read_mode::rgb_alpha) {
                return handle_exceptions<tensor_t, std::invalid_argument>(
                        "decode_image: the provided mode is not supported for JPEG files.");
            }
            if (!detail::jpeg_has_eoi(p, n)) {
                throw std::runtime_error("decode_image: image is incomplete or truncated.");
            }
            // stb decodes only the Y plane of a YCbCr JPEG for gray, like libjpeg's JCS_GRAYSCALE output
            return detail::stb_decode(p, n,
                                      mode == image_read_mode::gray  ? 1
                                      : mode == image_read_mode::rgb ? 3
                                                                     : 0,
                                      false);
        }
        if (n < 8 || std::memcmp(p, "\x89PNG", 4) != 0) {
            throw std::runtime_error("decode_image: unsupported image file, only JPEG and PNG are supported.");
        }
        const auto hdr = detail::read_png_header(p, n);
        const bool sixteen = hdr.bit_depth == 16;
        auto x = detail::stb_decode(p, n, 0, sixteen);
        // stb turns the tRNS chunk of a gray / RGB image into an alpha channel; libpng (torchvision) ignores it
        const int64_t native = hdr.color_type == 0 ? 1 : hdr.color_type == 2 ? 3 : hdr.color_type == 4 ? 2 : 4;
        if (hdr.color_type != 3 && x.size(0) > native) {
            x = x.slice(0, 0, native);
        }
        if (mode == image_read_mode::unchanged) {
            return x;
        }
        const auto v = x.to(torch::kInt64);
        const auto c = v.size(0);
        // libpng png_set_add_alpha filler as torchvision passes it: (1 << file bit depth) - 1, so 1 / 2 / 4-bit gray
        // files get an alpha of 1 / 3 / 15. A palette without tRNS: torchvision packs RGB / gray bytes into the 4- or
        // 2-channel buffer (every channel scrambled, uninitialised tail); here colours are kept and alpha is 255.
        const int64_t filler = hdr.color_type == 3 ? 255 : (int64_t{1} << hdr.bit_depth) - 1;
        const auto alpha =
                c % 2 == 0 ? v.slice(0, c - 1) : torch::full({1, v.size(1), v.size(2)}, filler, torch::kInt64);
        tensor_t out;
        if (mode == image_read_mode::gray || mode == image_read_mode::gray_alpha) {
            const auto g = c >= 3 ? detail::png_rgb_to_gray(v.slice(0, 0, 3), sixteen, hdr.gamma) : v.slice(0, 0, 1);
            out = mode == image_read_mode::gray ? g : torch::cat({g, alpha});
        } else {
            const auto rgb = c >= 3 ? v.slice(0, 0, 3) : v.slice(0, 0, 1).expand({3, -1, -1});
            out = mode == image_read_mode::rgb ? rgb : torch::cat({rgb, alpha});
        }
        return out.to(x.scalar_type()).contiguous();
    }

    // torchvision.io.read_file: the file's bytes as a 1-D uint8 tensor.
    inline auto read_file(const std::filesystem::path &path) -> tensor_t {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        const auto size = f ? static_cast<int64_t>(f.tellg()) : -1;
        auto out = torch::empty({std::max<int64_t>(size, 0)}, torch::kUInt8);
        // a directory opens and reports a size on some platforms, then fails to read; torchvision also rejects empty
        if (size <= 0 || !f.seekg(0) || !f.read(reinterpret_cast<char *>(out.data_ptr<uint8_t>()), size)) {
            throw std::runtime_error("read_file: could not read " + path.string() + " (missing, unreadable or empty)");
        }
        return out;
    }

    // torchvision.io.read_image / decode_image(path): read_file + decode_image.
    inline auto read_image(const std::filesystem::path &path, image_read_mode mode = image_read_mode::unchanged)
            -> tensor_t {
        return decode_image(read_file(path), mode);
    }
} // namespace torchmedia::vision::io

#endif // LIB_TORCH_MEDIA_VISION_IO_HPP
