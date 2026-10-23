// Self-contained audio I/O round-trip test: synthesize -> save_audio -> load_audio -> compare.
// Exercises the dr_wav backend with no external WAV fixture and no FFmpeg linkage.
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <cmath>
#include <print>
#include <torch/torch.h>
#include <torchmedia.hpp>

int main() {
    using namespace torchmedia::audio;

    const int sample_rate = 16000;
    const int64_t n = sample_rate; // 1 second
    const double pi = std::acos(-1.0);

    // 2-channel signal: 440 Hz and 880 Hz sine waves, in [channels, samples] float32.
    auto t = torch::arange(n, torch::kFloat32) / static_cast<float>(sample_rate);
    auto ch0 = torch::sin(2.0 * pi * 440.0 * t);
    auto ch1 = torch::sin(2.0 * pi * 880.0 * t) * 0.5f;
    auto wave = torch::stack({ch0, ch1}, 0); // [2, n]

    const std::string path = "roundtrip_test.wav";

    if (!io::save_audio(wave, path, sample_rate)) {
        std::print("FAIL: save_audio returned false\n");
        return 1;
    }

    auto loaded = io::load_audio(path);
    std::print("saved  shape: [{}, {}]\n", wave.size(0), wave.size(1));
    std::print("loaded shape: [{}, {}], sample_rate = {}\n", loaded.data.size(0), loaded.data.size(1),
               loaded.sample_rate);

    if (loaded.sample_rate != sample_rate) {
        std::print("FAIL: sample_rate mismatch ({} != {})\n", loaded.sample_rate, sample_rate);
        return 1;
    }
    if (loaded.data.sizes() != wave.sizes()) {
        std::print("FAIL: shape mismatch\n");
        return 1;
    }

    const float max_err = (loaded.data - wave).abs().max().item<float>();
    std::print("max round-trip abs error: {}\n", max_err);

    // PCM16 quantization step is ~1/32768 ≈ 3e-5; allow a small margin.
    if (max_err > 1e-3f) {
        std::print("FAIL: round-trip error too large\n");
        return 1;
    }

    std::print("OK: WAV save/load round-trip via dr_wav passed (no FFmpeg).\n");
    return 0;
}
