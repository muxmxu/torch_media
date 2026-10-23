// Self-contained audio I/O round-trip test: synthesize -> save_audio -> load_audio -> compare.
// Exercises the dr_wav backend with no external WAV fixture and no FFmpeg linkage.
// <torchmedia.hpp> is deliberately the first torch-dependent include: it must compile on its own.
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <torchmedia.hpp>
#include "test_util.hpp"

using namespace torchmedia::audio;

static auto max_err(const torch::Tensor &a, const torch::Tensor &b) -> double {
    return a.sizes() == b.sizes() ? (a - b).abs().max().item<double>() : 1.0;
}

template<class Fn>
static auto throws_runtime(Fn &&fn) -> bool {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int main() {
    const int sample_rate = 16000;
    const int64_t n = sample_rate; // 1 second
    const double pi = std::acos(-1.0);

    // 2-channel signal: 440 Hz and 880 Hz sine waves, in [channels, samples] float32.
    const auto t = torch::arange(n, torch::kFloat32) / static_cast<float>(sample_rate);
    const auto wave = torch::stack({torch::sin(2.0 * pi * 440.0 * t), torch::sin(2.0 * pi * 880.0 * t) * 0.5f}, 0);

    const std::string path = "roundtrip_test.wav";
    TM_CHECK(io::save_audio(wave, path, sample_rate));
    const auto loaded = io::load_audio(path);
    TM_CHECK(loaded.sample_rate == sample_rate);
    // PCM16 quantization step is ~1/32768 ~= 3e-5; allow a small margin.
    TM_CHECK(max_err(loaded.data, wave) <= 1e-3);

    // Mono and single-frame files: transposing [frames, 1] or [1, channels] is already contiguous, so load_audio
    // must still copy out of the decoder buffer it frees (regression: the result aliased freed memory, and the
    // allocator overwrote the first frames of small files). Small sizes + a non-zero first sample expose it.
    const auto mono = (0.5f * torch::cos(2.0 * pi * 440.0 * t.slice(0, 0, 4000))).unsqueeze(0); // [1, 4000]
    const auto one_frame = torch::tensor({{0.5f}, {-0.25f}}); // [2, 1]
    for (const auto &[label, ref]: {std::pair{"mono", mono}, std::pair{"single-frame", one_frame}}) {
        const std::string small_path = std::string("roundtrip_") + label + "_test.wav";
        TM_CHECK(io::save_audio(ref, small_path, sample_rate));
        const auto small = io::load_audio(small_path).data;
        auto churn = torch::randn({64, 64}); // reuse freed heap blocks before reading the loaded data
        TM_CHECK(max_err(small, ref) <= 1e-3);
    }

    // const char* and std::filesystem::path overloads.
    TM_CHECK(io::save_audio(mono, "roundtrip_cstr_test.wav", sample_rate));
    TM_CHECK(max_err(io::load_audio("roundtrip_cstr_test.wav").data, mono) <= 1e-3);
    const std::filesystem::path fs_path = "roundtrip_path_test.wav";
    TM_CHECK(io::save_audio(mono, fs_path, sample_rate));
    TM_CHECK(max_err(io::load_audio(fs_path).data, mono) <= 1e-3);

    // save_audio rejects bad input / unwritable paths with false.
    TM_CHECK(!io::save_audio(mono.flatten(), path, sample_rate)); // not [channels, samples]
    TM_CHECK(!io::save_audio(mono.to(torch::kDouble), path, sample_rate)); // not float32
    TM_CHECK(!io::save_audio(mono, "no_such_dir/x.wav", sample_rate)); // drwav_init_file_write fails

    // load_audio throws for a missing file and for a file that is not a WAV.
    TM_CHECK(throws_runtime([] { (void) io::load_audio("does_not_exist.wav"); }));
    std::ofstream("not_a_wav.wav") << "this is not a RIFF/WAVE file";
    TM_CHECK(throws_runtime([] { (void) io::load_audio("not_a_wav.wav"); }));

    return tm_test::summary("audio_test_io");
}
