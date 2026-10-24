#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_IO_HPP
#define LIB_TORCH_MEDIA_AUDIO_IO_HPP

#include <algorithm>
#include <bit>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <vector>

#include "../global_include.hpp"

// Vendored single-header codecs (provenance and licenses: THIRD_PARTY_LICENSES): dr_wav (WAV read + write), dr_flac,
// dr_mp3 (public domain / MIT-0) and stb_vorbis (public domain / MIT). Their implementations are emitted in exactly ONE
// translation unit, the one that defines TORCHMEDIA_IO_IMPLEMENTATION before including torchmedia; every other TU sees
// the declarations only. This keeps TorchMedia header-only without ODR / multiple-definition errors
// (develop_log/2026-05-30/progress01, D4). A project that already compiles one of these libraries itself defines
// TORCHMEDIA_NO_DR_WAV_IMPLEMENTATION, TORCHMEDIA_NO_DR_FLAC_IMPLEMENTATION, TORCHMEDIA_NO_DR_MP3_IMPLEMENTATION or
// TORCHMEDIA_NO_STB_VORBIS_IMPLEMENTATION next to TORCHMEDIA_IO_IMPLEMENTATION (its copy must be the same version).
// Trusted input only: these are small C decoders, and stb_vorbis has the largest fuzzing / CVE record of them.
#ifdef TORCHMEDIA_IO_IMPLEMENTATION
#if !defined(DR_WAV_IMPLEMENTATION) && !defined(TORCHMEDIA_NO_DR_WAV_IMPLEMENTATION)
#define DR_WAV_IMPLEMENTATION
#endif
#if !defined(DR_FLAC_IMPLEMENTATION) && !defined(TORCHMEDIA_NO_DR_FLAC_IMPLEMENTATION)
#define DR_FLAC_IMPLEMENTATION
#endif
#if !defined(DR_MP3_IMPLEMENTATION) && !defined(TORCHMEDIA_NO_DR_MP3_IMPLEMENTATION)
#define DR_MP3_IMPLEMENTATION
#endif
#ifndef DR_MP3_FLOAT_OUTPUT
#define DR_MP3_FLOAT_OUTPUT // synthesize MP3 in float: no int16 rounding of the decoded samples
#endif
#endif
#include "_vendor/dr_flac.h"
#include "_vendor/dr_mp3.h"
#include "_vendor/dr_wav.h"
#pragma push_macro("STB_VORBIS_HEADER_ONLY") // keep a consumer's own definition
#undef STB_VORBIS_HEADER_ONLY
#define STB_VORBIS_HEADER_ONLY
#include "_vendor/stb_vorbis.c"
#pragma pop_macro("STB_VORBIS_HEADER_ONLY")
#if defined(TORCHMEDIA_IO_IMPLEMENTATION) && !defined(TORCHMEDIA_NO_STB_VORBIS_IMPLEMENTATION)
#include "_io_stb_vorbis_impl.hpp"
#endif

namespace torchmedia::audio::io {

    struct load_audio_t {
        tensor_t data; // [channels, frames] float32
        int sample_rate;
    };

    // torchaudio.load's frame_offset / num_frames: skip frame_offset frames, then read at most num_frames (-1: to the
    // end). An offset past the end gives [channels, 0]; a negative offset or num_frames 0 / < -1 throws
    // std::invalid_argument, as in torchaudio.
    typedef struct load_option {
        int64_t _frame_offset = 0;
        int64_t _num_frames = -1;

        auto frame_offset(int64_t v) -> load_option & {
            _frame_offset = v;
            return *this;
        }
        auto num_frames(int64_t v) -> load_option & {
            _num_frames = v;
            return *this;
        }
    } load_option_t;

    // torchaudio.info's AudioMetaData. encoding: "PCM_S", "PCM_U", "PCM_F", "ULAW", "ALAW" or "UNKNOWN" (other WAV
    // codecs, e.g. ADPCM), "FLAC", "MP3", "VORBIS". bits_per_sample is 0 for MP3 and Vorbis, as in torchaudio.
    struct audio_info {
        int sample_rate = 0;
        int64_t num_frames = 0;
        int num_channels = 0;
        int bits_per_sample = 0;
        std::string encoding;
    };

    // torchaudio.save's encoding / bits_per_sample for WAV ("PCM_S" 16/24/32, "PCM_U" 8, "PCM_F" 32/64). Unset values
    // are derived like torchaudio: bits alone pick "PCM_U" for 8 and "PCM_S" otherwise; neither picks from the dtype
    // (float32: 32-bit float, int32 / int16: signed 32 / 16, uint8: unsigned 8); "PCM_S" alone is 16-bit, "PCM_U"
    // 8-bit and "PCM_F" 32-bit.
    typedef struct save_option {
        std::string _encoding;
        int _bits_per_sample = 0;

        auto encoding(const std::string &v) -> save_option & {
            _encoding = v;
            return *this;
        }
        auto bits_per_sample(int v) -> save_option & {
            _bits_per_sample = v;
            return *this;
        }
    } save_option_t;

    namespace detail {
        // An encoded stream: the file at `path`, or (path empty) the caller-owned buffer [data, data + size).
        struct source {
            std::string path;
            const void *data = nullptr;
            size_t size = 0;

            auto name() const -> std::string { return path.empty() ? "<memory>" : path; }
        };

        // One open stream. info.num_frames is exact. read() returns at most `frames` interleaved [frames, channels]
        // samples, either float32 or full-scale int32 (x / 2^31).
        struct decoder {
            audio_info info;
            virtual ~decoder() = default;
            virtual auto seek(uint64_t frame) -> bool = 0;
            virtual auto read(int64_t frames) -> tensor_t = 0;
        };

        struct wav_decoder final : decoder {
            drwav wav{};

            explicit wav_decoder(const source &src) {
                if (!(src.path.empty() ? drwav_init_memory(&wav, src.data, src.size, nullptr)
                                       : drwav_init_file(&wav, src.path.c_str(), nullptr))) {
                    throw std::runtime_error("Could not decode WAV: " + src.name());
                }
                const auto tag = wav.translatedFormatTag;
                info = {static_cast<int>(wav.sampleRate), static_cast<int64_t>(wav.totalPCMFrameCount), wav.channels,
                        wav.bitsPerSample,
                        tag == DR_WAVE_FORMAT_PCM          ? (wav.bitsPerSample == 8 ? "PCM_U" : "PCM_S")
                        : tag == DR_WAVE_FORMAT_IEEE_FLOAT ? "PCM_F"
                        : tag == DR_WAVE_FORMAT_ALAW       ? "ALAW"
                        : tag == DR_WAVE_FORMAT_MULAW      ? "ULAW"
                                                           : "UNKNOWN"};
            }
            ~wav_decoder() override { drwav_uninit(&wav); }

            auto seek(uint64_t frame) -> bool override { return drwav_seek_to_pcm_frame(&wav, frame); }
            auto read(int64_t frames) -> tensor_t override {
                // Float files come back as stored. Everything else goes through full-scale int32, which is exact for
                // <= 24-bit PCM and maps 8-bit to (x - 128) / 128 like torchaudio (dr_wav's f32 path: x / 127.5 - 1).
                if (wav.translatedFormatTag == DR_WAVE_FORMAT_IEEE_FLOAT) {
                    auto out = torch::empty({frames, info.num_channels}, torch::kFloat32);
                    return out.narrow(0, 0, drwav_read_pcm_frames_f32(&wav, frames, out.data_ptr<float>()));
                }
                auto out = torch::empty({frames, info.num_channels}, torch::kInt32);
                return out.narrow(0, 0, drwav_read_pcm_frames_s32(&wav, frames, out.data_ptr<int32_t>()));
            }
        };

        struct flac_decoder final : decoder {
            std::vector<char> bytes; // an ID3v2-prefixed file, read whole
            drflac *flac = nullptr;

            // `offset`: the "fLaC" marker's position, past an ID3v2 tag. dr_flac's seeks assume it at byte 0, so the
            // stream is handed over from there (shortcut: such a file is read into memory; use offset read callbacks
            // if large tagged FLAC files become common).
            flac_decoder(const source &src, size_t offset) {
                const char *data = static_cast<const char *>(src.data);
                size_t size = src.size;
                if (!src.path.empty() && offset > 0) {
                    std::ifstream file(src.path, std::ios::binary);
                    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
                    data = bytes.data();
                    size = bytes.size();
                }
                flac = offset > size                    ? nullptr // the file shrank after detection
                       : src.path.empty() || offset > 0 ? drflac_open_memory(data + offset, size - offset, nullptr)
                                                        : drflac_open_file(src.path.c_str(), nullptr);
                if (flac == nullptr || flac->bitsPerSample == 32) {
                    // dr_flac reads the 32-bit frame-header code as reserved and would decode no frames
                    const bool wide = flac != nullptr;
                    drflac_close(flac);
                    throw std::runtime_error(
                            std::string(wide ? "Unsupported 32-bit FLAC: " : "Could not decode FLAC: ") + src.name());
                }
                if (flac->totalPCMFrameCount == 0) {
                    // A streamed encode may leave the length unset in STREAMINFO: count it by decoding once (seeking
                    // clamps to it). torchaudio reports 0 frames for such a file and still loads all of it.
                    flac->totalPCMFrameCount = drflac_read_pcm_frames_s32(flac, UINT64_MAX, nullptr);
                    drflac_seek_to_pcm_frame(flac, 0);
                }
                info = {static_cast<int>(flac->sampleRate), static_cast<int64_t>(flac->totalPCMFrameCount),
                        flac->channels, flac->bitsPerSample, "FLAC"};
            }
            ~flac_decoder() override { drflac_close(flac); }

            auto seek(uint64_t frame) -> bool override { return drflac_seek_to_pcm_frame(flac, frame); }
            auto read(int64_t frames) -> tensor_t override {
                auto out = torch::empty({frames, info.num_channels}, torch::kInt32);
                return out.narrow(0, 0, drflac_read_pcm_frames_s32(flac, frames, out.data_ptr<int32_t>()));
            }
        };

        struct mp3_decoder final : decoder {
            drmp3 mp3{};

            explicit mp3_decoder(const source &src) {
                if (!(src.path.empty() ? drmp3_init_memory(&mp3, src.data, src.size, nullptr)
                                       : drmp3_init_file(&mp3, src.path.c_str(), nullptr))) {
                    throw std::runtime_error("Could not decode MP3: " + src.name());
                }
                // The Xing / LAME tag's length minus encoder delay and padding, else a frame-header scan.
                info = {static_cast<int>(mp3.sampleRate), static_cast<int64_t>(drmp3_get_pcm_frame_count(&mp3)),
                        static_cast<int>(mp3.channels), 0, "MP3"};
            }
            ~mp3_decoder() override { drmp3_uninit(&mp3); }

            auto seek(uint64_t frame) -> bool override { return drmp3_seek_to_pcm_frame(&mp3, frame); }
            auto read(int64_t frames) -> tensor_t override {
                auto out = torch::empty({frames, info.num_channels}, torch::kFloat32);
                return out.narrow(0, 0, drmp3_read_pcm_frames_f32(&mp3, frames, out.data_ptr<float>()));
            }
        };

        struct vorbis_decoder final : decoder {
            stb_vorbis *vorbis;
            bool seekable = true;

            explicit vorbis_decoder(const source &src) :
                vorbis(!src.path.empty()    ? stb_vorbis_open_filename(src.path.c_str(), nullptr, nullptr)
                       : src.size > INT_MAX ? nullptr // stb_vorbis takes an int length
                                            : stb_vorbis_open_memory(static_cast<const unsigned char *>(src.data),
                                                                     static_cast<int>(src.size), nullptr, nullptr)) {
                if (vorbis == nullptr) {
                    throw std::runtime_error("Could not decode Ogg Vorbis: " + src.name());
                }
                const auto vi = stb_vorbis_get_info(vorbis);
                info = {static_cast<int>(vi.sample_rate),
                        static_cast<int64_t>(stb_vorbis_stream_length_in_samples(vorbis)), vi.channels, 0, "VORBIS"};
                if (info.num_frames == 0) {
                    // stb_vorbis only looks for the last page in the final 64 KB (e.g. trailing junk defeats it), and
                    // cannot seek without it: count by decoding once, and seek by decoding from the start.
                    seekable = false;
                    info.num_frames = skip(INT64_MAX);
                    stb_vorbis_seek_start(vorbis);
                }
            }
            ~vorbis_decoder() override { stb_vorbis_close(vorbis); }

            auto skip(int64_t frames) -> int64_t { // decodes and drops up to `frames` frames
                int64_t done = 0;
                for (int64_t n = 1; n > 0 && done < frames; done += n) {
                    n = read(std::min<int64_t>(frames - done, 1 << 16)).size(0);
                }
                return done;
            }
            auto seek(uint64_t frame) -> bool override {
                if (!seekable) {
                    stb_vorbis_seek_start(vorbis);
                    return skip(static_cast<int64_t>(frame)) == static_cast<int64_t>(frame);
                }
                return stb_vorbis_seek(vorbis, static_cast<unsigned int>(frame)) != 0;
            }
            auto read(int64_t frames) -> tensor_t override {
                const int64_t ch = info.num_channels;
                auto out = torch::empty({frames, ch}, torch::kFloat32);
                int64_t got = 0;
                for (int n = 1; n > 0 && got < frames; got += n) { // the float count is an int: read in bounded steps
                    const auto want = static_cast<int>(std::min<int64_t>(frames - got, INT_MAX / ch) * ch);
                    n = stb_vorbis_get_samples_float_interleaved(vorbis, static_cast<int>(ch),
                                                                 out.data_ptr<float>() + got * ch, want);
                }
                return out.narrow(0, 0, got);
            }
        };

        // Opens the decoder for the container named by the leading bytes (never by the file extension).
        inline auto open(const source &src) -> std::unique_ptr<decoder> {
            if (!src.path.empty() && !std::filesystem::exists(src.path)) {
                throw std::runtime_error("File does not exist: " + src.path);
            }
            const auto peek = [&](size_t offset, size_t len) -> std::string { // up to len bytes from offset
                if (src.path.empty()) {
                    return offset < src.size ? std::string(static_cast<const char *>(src.data) + offset,
                                                           std::min(len, src.size - offset))
                                             : std::string();
                }
                std::string out(len, '\0');
                std::ifstream file(src.path, std::ios::binary);
                file.seekg(static_cast<std::streamoff>(offset));
                file.read(out.data(), static_cast<std::streamsize>(len));
                out.resize(static_cast<size_t>(std::max<std::streamsize>(file.gcount(), 0)));
                return out;
            };
            const auto h = peek(0, 512);
            const auto at = [&](size_t pos, std::string_view tag) {
                return std::string_view(h).substr(std::min(pos, h.size()), tag.size()) == tag;
            };
            const auto byte = [&](size_t pos) { return pos < h.size() ? static_cast<unsigned char>(h[pos]) : 0u; };
            if (((at(0, "RIFF") || at(0, "RF64")) && at(8, "WAVE")) || (at(0, "riff") && at(24, "wave"))) { // W64
                return std::make_unique<wav_decoder>(src);
            }
            if (at(0, "fLaC")) {
                return std::make_unique<flac_decoder>(src, 0);
            }
            if (at(0, "OggS") && h.size() > 26 && at(27 + byte(26), "\x01vorbis")) {
                return std::make_unique<vorbis_decoder>(src);
            }
            if (at(0, "ID3") && h.size() >= 10) { // ID3v2: 10-byte header + syncsafe size (+ 10-byte footer)
                const size_t tag =
                        10 + (byte(6) << 21 | byte(7) << 14 | byte(8) << 7 | byte(9)) + (byte(5) & 0x10 ? 10 : 0);
                if (peek(tag, 4) == "fLaC") { // tagged FLAC (sox reads it too); anything else goes to dr_mp3
                    return std::make_unique<flac_decoder>(src, tag);
                }
                return std::make_unique<mp3_decoder>(src);
            }
            // An MPEG audio frame sync whose layer bits are not the reserved 00 (that is AAC ADTS), after any zero
            // padding within the first 512 bytes (dr_mp3 resyncs past it).
            const size_t z = std::min(h.find_first_not_of('\0'), h.size());
            if (byte(z) == 0xFF && (byte(z + 1) & 0xE0) == 0xE0 && (byte(z + 1) & 0x06)) {
                return std::make_unique<mp3_decoder>(src);
            }
            throw std::runtime_error("Unsupported or unrecognized audio format: " + src.name());
        }

        inline auto load(const source &src, const load_option &opt) -> load_audio_t {
            if (opt._frame_offset < 0) {
                throw std::invalid_argument("load_audio: frame_offset must be non-negative, got " +
                                            std::to_string(opt._frame_offset));
            }
            if (opt._num_frames == 0 || opt._num_frames < -1) {
                throw std::invalid_argument("load_audio: num_frames must be -1 or positive, got " +
                                            std::to_string(opt._num_frames));
            }
            const auto dec = open(src);
            const int64_t total = dec->info.num_frames, channels = dec->info.num_channels;
            const int64_t start = std::min(opt._frame_offset, total);
            const int64_t count = opt._num_frames < 0 ? total - start : std::min(opt._num_frames, total - start);
            // Decoded in blocks of ~2^20 samples until the data ends, so memory follows what decodes, not the length
            // the header claims (a corrupt one can claim terabytes). A failed seek means the stream ends before
            // `start` (e.g. a truncated file): nothing to read.
            std::vector<tensor_t> blocks;
            if (count > 0 && dec->seek(static_cast<uint64_t>(start))) {
                const int64_t block = std::max<int64_t>(1, (int64_t{1} << 20) / channels);
                for (int64_t left = count; left > 0;) {
                    auto b = dec->read(std::min(left, block));
                    if (b.size(0) == 0) {
                        break;
                    }
                    left -= b.size(0);
                    blocks.push_back(std::move(b));
                }
            }
            const auto raw = blocks.empty() ? torch::empty({0, channels}, torch::kFloat32) : torch::cat(blocks);
            // [frames, channels] -> contiguous [channels, frames] float32 in one copy (Tensor::to would return the
            // transposed float view as is); int32 samples are full scale.
            auto data = torch::empty({raw.size(1), raw.size(0)}, torch::kFloat32).copy_(raw.t());
            if (raw.scalar_type() == torch::kInt32) {
                data.mul_(1.0 / 2147483648.0);
            }
            return {data, dec->info.sample_rate};
        }

        // One interleaved [frames, channels] block in the file's sample format, as torchaudio.save writes it.
        inline auto encode_wav_block(const tensor_t &block, const std::string &encoding, int bits) -> tensor_t {
            const auto dtype = block.scalar_type();
            if (encoding == "PCM_F" && dtype == torch::kFloat32) {
                // Stored as given (deliberate: torchaudio's sox path clips to [-1, 1], rounds to 2^-24 / 2^-31 steps).
                return block.to(bits == 32 ? torch::kFloat32 : torch::kDouble).contiguous();
            }
            // torchaudio first maps every dtype to a full-scale int32 sample: float x * 2^31 clamped and truncated
            // (NaN as 0), int16 x * 2^16, uint8 (x - 128) * 2^24.
            tensor_t full;
            if (dtype == torch::kFloat32) {
                full = block.to(torch::kDouble).mul_(2147483648.0).nan_to_num_(0.0).clamp_(INT32_MIN, INT32_MAX);
                full = full.to(torch::kInt64);
            } else {
                full = block.to(torch::kInt64);
                full = dtype == torch::kUInt8 ? full.sub_(128).mul_(16777216)
                                              : full.mul_(dtype == torch::kInt16 ? 65536 : 1);
            }
            if (encoding == "PCM_F") {
                const auto f = full.to(torch::kDouble).div_(2147483648.0);
                return f.to(bits == 32 ? torch::kFloat32 : torch::kDouble).contiguous();
            }
            // sox's narrowing: round half up to `bits`, clipping only at the positive end (bit-exact with torchaudio).
            const int shift = 32 - bits;
            const auto q = shift == 0 ? full
                                      : full.add(int64_t{1} << (shift - 1))
                                                .div_(int64_t{1} << shift, "floor")
                                                .clamp_max_((int64_t{1} << (bits - 1)) - 1);
            if (bits == 24) { // the low three bytes of each int32, in host order (dr_wav swaps on big-endian hosts)
                const int64_t lo = std::endian::native == std::endian::little ? 0 : 1;
                return q.to(torch::kInt32).reshape(-1).view(torch::kUInt8).view({-1, 4}).narrow(1, lo, 3).contiguous();
            }
            return (bits == 8 ? q.add(128).to(torch::kUInt8) : q.to(bits == 16 ? torch::kInt16 : torch::kInt32))
                    .contiguous();
        }
    } // namespace detail

    // Load an audio file into a [channels, frames] float32 tensor at its NATIVE sample rate (resampling is a separate
    // op, develop_log/2026-05-30/progress01 D3). The container is detected from the leading bytes: WAV (RIFF / RF64 /
    // W64), FLAC (16 / 24-bit; also behind an ID3v2 tag), Ogg Vorbis, MP3. Integer PCM decodes to x / 2^(bits-1) and
    // 8-bit to (x - 128) / 128 like torchaudio; float WAV, MP3 and Vorbis come back as decoded, including any overshoot
    // beyond [-1, 1] (torchaudio's sox backend clips them). Throws std::runtime_error naming the path for a missing,
    // unsupported (e.g. 32-bit FLAC) or undecodable file.
    inline auto load_audio(const std::string &path, const load_option &opt = {}) -> load_audio_t {
        return detail::load({path}, opt);
    }

    // load_audio from an encoded file image in memory (the buffer is only read during the call).
    inline auto load_audio_from_memory(const void *data, size_t size, const load_option &opt = {}) -> load_audio_t {
        return detail::load({"", data, size}, opt);
    }

    // torchaudio.info: the stream's metadata without decoding it (MP3 without a Xing / LAME tag scans frame headers;
    // Vorbis whose last page stb_vorbis cannot find, and FLAC of unknown length, are decoded once to count). The frame
    // count is the header's: for a corrupt or truncated file load_audio returns what actually decodes.
    inline auto info(const path_t &path) -> audio_info { return detail::open({path.string()})->info; }

    inline auto info_from_memory(const void *data, size_t size) -> audio_info {
        return detail::open({"", data, size})->info;
    }

    // Save a [channels, frames] tensor (float32, int32, int16 or uint8, as torchaudio.save accepts) as a WAV file.
    // Defaults follow torchaudio.save: a float32 tensor becomes a 32-bit float WAV (before 2026-10-23 this wrote
    // 16-bit PCM); integer PCM is quantized bit-exactly like torchaudio. Returns false for invalid arguments (rank,
    // dtype, no or > 65535 channels, sample_rate <= 0, an unsupported encoding / bits pair) or an I/O error.
    inline auto save_audio(const tensor_t &audio, const std::string &path, int sample_rate, const save_option &opt = {})
            -> bool {
        const auto dtype = audio.scalar_type();
        if (audio.dim() != 2 || audio.size(0) < 1 || audio.size(0) > 65535 || sample_rate <= 0 ||
            (dtype != torch::kFloat32 && dtype != torch::kInt32 && dtype != torch::kInt16 && dtype != torch::kUInt8)) {
            return false;
        }
        std::string encoding = opt._encoding;
        int bits = opt._bits_per_sample;
        if (encoding.empty()) {
            encoding = bits == 8 || (bits == 0 && dtype == torch::kUInt8) ? "PCM_U"
                       : bits == 0 && dtype == torch::kFloat32            ? "PCM_F"
                                                                          : "PCM_S";
            bits = bits != 0 ? bits : static_cast<int>(audio.element_size()) * 8;
        } else if (bits == 0) {
            bits = encoding == "PCM_U" ? 8 : encoding == "PCM_S" ? 16 : 32;
        }
        if (!((encoding == "PCM_S" && (bits == 16 || bits == 24 || bits == 32)) || (encoding == "PCM_U" && bits == 8) ||
              (encoding == "PCM_F" && (bits == 32 || bits == 64)))) {
            return false;
        }

        drwav_data_format format{};
        format.container = drwav_container_riff;
        format.format = encoding == "PCM_F" ? DR_WAVE_FORMAT_IEEE_FLOAT : DR_WAVE_FORMAT_PCM;
        format.channels = static_cast<drwav_uint32>(audio.size(0));
        format.sampleRate = static_cast<drwav_uint32>(sample_rate);
        format.bitsPerSample = static_cast<drwav_uint32>(bits);
        drwav wav;
        if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
            return false;
        }
        const auto frames = audio.detach().to(torch::kCPU).t(); // [frames, channels]
        const int64_t block = 1 << 16; // frames per conversion: bounds the int64 / double temporaries
        bool ok = true;
        for (int64_t f = 0; ok && f < frames.size(0); f += block) {
            const auto n = std::min(block, frames.size(0) - f);
            const auto samples = detail::encode_wav_block(frames.narrow(0, f, n), encoding, bits);
            ok = drwav_write_pcm_frames(&wav, static_cast<drwav_uint64>(n), samples.data_ptr()) ==
                 static_cast<drwav_uint64>(n);
        }
        return drwav_uninit(&wav) == DRWAV_SUCCESS && ok;
    }

    // const char* overloads so a bare string literal has a single best match — without them,
    // "file.wav" is ambiguous between the std::string and std::filesystem::path overloads.
    inline auto load_audio(const char *path, const load_option &opt = {}) -> load_audio_t {
        return load_audio(std::string(path), opt);
    }

    inline auto save_audio(const tensor_t &audio, const char *path, int sample_rate, const save_option &opt = {})
            -> bool {
        return save_audio(audio, std::string(path), sample_rate, opt);
    }

    inline auto load_audio(const std::filesystem::path &path, const load_option &opt = {}) -> load_audio_t {
        return load_audio(path.string(), opt);
    }

    inline auto save_audio(const tensor_t &audio, const std::filesystem::path &path, int sample_rate,
                           const save_option &opt = {}) -> bool {
        return save_audio(audio, path.string(), sample_rate, opt);
    }
} // namespace torchmedia::audio::io

#endif // LIB_TORCH_MEDIA_AUDIO_IO_HPP
