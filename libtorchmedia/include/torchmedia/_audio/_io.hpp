#pragma once
#ifndef LIB_TORCH_MEDIA_AUDIO_IO_HPP
#define LIB_TORCH_MEDIA_AUDIO_IO_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>

#include "../globel_include.hpp"

// dr_wav: header-only WAV decoder/encoder (public domain / MIT-0).
// The implementation block is emitted in exactly ONE translation unit, namely the one that
// defines TORCHMEDIA_IO_IMPLEMENTATION before including torchmedia. Every other TU sees only the
// declarations. This keeps TorchMedia header-only while avoiding ODR / multiple-definition errors.
// See develop_log/2026-05-30/progress01-remove-ffmpeg-dr_wav-io.md (decision D4).
#ifdef TORCHMEDIA_IO_IMPLEMENTATION
#ifndef DR_WAV_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#endif
#endif
#include "_vendor/dr_wav.h"

namespace torchmedia::audio::io {

    struct load_audio_t {
        tensor_t data;
        int sample_rate;
    };

    // Load a WAV file into a [channels, samples] float32 tensor at the file's NATIVE sample rate.
    // (Resampling is intentionally not done here — it is a separate torch-native op. See D3.)
    auto inline load_audio(const std::string &path) -> load_audio_t {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }

        unsigned int channels = 0;
        unsigned int sample_rate = 0;
        drwav_uint64 total_frames = 0;
        float *interleaved = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sample_rate,
                                                                     &total_frames, nullptr);
        if (interleaved == nullptr) {
            throw std::runtime_error("Could not open / decode WAV file: " + path);
        }

        const auto n_channels = static_cast<int64_t>(channels);
        const auto n_frames = static_cast<int64_t>(total_frames);

        // dr_wav gives a contiguous interleaved [frames, channels] f32 buffer. clone() always copies it into a
        // tensor-owned buffer; contiguous() would not for 1 channel or 1 frame (the transpose is already
        // contiguous) and the result would alias the allocation freed below.
        auto interleaved_t = torch::from_blob(interleaved, {n_frames, n_channels}, torch::kFloat32);
        auto data = interleaved_t.transpose(0, 1).clone(torch::MemoryFormat::Contiguous); // -> [channels, frames]
        drwav_free(interleaved, nullptr);

        return {data, static_cast<int>(sample_rate)};
    }

    // Save a [channels, samples] float32 tensor as a PCM16 WAV. Returns false on bad input / IO error.
    auto inline save_audio(const tensor_t &audio_tensor, const std::string &path, int sample_rate) -> bool {
        if (audio_tensor.dim() != 2 || audio_tensor.scalar_type() != torch::kFloat32) {
            return false; // expects a [channels, samples] float32 tensor
        }

        const auto n_channels = audio_tensor.size(0);
        const auto n_frames = audio_tensor.size(1);

        // [channels, frames] -> interleaved [frames, channels], contiguous, CPU, f32.
        auto interleaved = audio_tensor.detach().to(torch::kCPU).transpose(0, 1).contiguous();
        const float *src = interleaved.data_ptr<float>();
        const auto sample_count = static_cast<size_t>(n_frames * n_channels);

        std::vector<drwav_int16> pcm16(sample_count);
        drwav_f32_to_s16(pcm16.data(), src, sample_count);

        drwav_data_format format{};
        format.container = drwav_container_riff;
        format.format = DR_WAVE_FORMAT_PCM;
        format.channels = static_cast<drwav_uint32>(n_channels);
        format.sampleRate = static_cast<drwav_uint32>(sample_rate);
        format.bitsPerSample = 16;

        drwav wav;
        if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
            return false;
        }
        const drwav_uint64 written =
                drwav_write_pcm_frames(&wav, static_cast<drwav_uint64>(n_frames), pcm16.data());
        drwav_uninit(&wav);

        return written == static_cast<drwav_uint64>(n_frames);
    }

    // const char* overloads so a bare string literal has a single best match — without them,
    // "file.wav" is ambiguous between the std::string and std::filesystem::path overloads.
    auto inline load_audio(const char *path) -> load_audio_t { return load_audio(std::string(path)); }

    auto inline save_audio(const tensor_t &audio_tensor, const char *path, int sample_rate) -> bool {
        return save_audio(audio_tensor, std::string(path), sample_rate);
    }

    auto inline load_audio(const std::filesystem::path &path) -> load_audio_t { return load_audio(path.string()); }

    auto inline save_audio(const tensor_t &audio_tensor, const std::filesystem::path &path, int sample_rate) -> bool {
        return save_audio(audio_tensor, path.string(), sample_rate);
    }
} // namespace torchmedia::audio::io

#endif // LIB_TORCH_MEDIA_AUDIO_IO_HPP
