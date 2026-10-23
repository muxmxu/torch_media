// Assertion-based tests for torchmedia::audio::compliance::kaldi (target: 100% line coverage of _compliance_kaldi.hpp).
// Goldens come from torchaudio.compliance.kaldi 2.5.1 on the same seeded inputs: gen_golden.py writes golden.hpp, the
// C++ calls included, so the cases cannot drift from the reference. With TM_GOLDEN_DIR set to the directory of
// `gen_golden.py --dump DIR`, every full output is compared as well and the element-wise max |delta| is printed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <type_traits>
#include <utility>
#include <vector>
#include "test_util.hpp"

namespace kd = torchmedia::audio::compliance::kaldi;
namespace kdd = torchmedia::audio::compliance::kaldi::detail;

struct golden {
    std::vector<int64_t> shape;
    bool f64; // output dtype float64 (else float32)
    double sum, chk, maxabs; // chk = sum(Y * Z), Z = randn(Y.shape, float64) under seed 99
    std::vector<std::pair<int64_t, double>> els; // (flat index, value)
};
struct kaldi_case {
    const char *name;
    torch::Tensor (*run)();
    golden g;
};

// gen_golden.py: wave16() / wave8()
static auto wave16() -> torch::Tensor {
    torch::manual_seed(0);
    auto w = torch::randn({2, 8000}) * 3000;
    w[0].slice(0, 2000, 3000).zero_(); // digital silence: frames 13-16 hit the epsilon / energy floors
    return w;
}
static auto wave8() -> torch::Tensor {
    torch::manual_seed(1);
    return torch::randn({1, 4000}) * 3000;
}

#include "golden.hpp"

// Twice torchaudio's own float32 spread (golden.hpp). Its float64 output does not move at all (measured 0), so float64
// cases only leave room for float64 rounding.
static const double rtol32 = 2 * torchaudio_variation_f32;
static const double rtol64 = std::max(2 * torchaudio_variation_f64, 1e-12);

static void check_close(const char *name, const char *what, const double a, const double b, const double atol) {
    tm_test::record(std::abs(a - b) <= atol, name, __FILE__, __LINE__,
                    std::string(what) + " |delta|=" + std::to_string(std::abs(a - b)) +
                            " atol=" + std::to_string(atol));
}

// Shape, dtype, sum, checksum and elements; the sums get numel * the element tolerance (sum |Z| < numel).
static void check_case(const kaldi_case &c, const char *golden_dir) {
    const auto y = c.run();
    const auto &g = c.g;
    tm_test::record(y.sizes() == torch::IntArrayRef(g.shape), c.name, __FILE__, __LINE__, "shape");
    tm_test::record(y.scalar_type() == (g.f64 ? torch::kDouble : torch::kFloat), c.name, __FILE__, __LINE__, "dtype");
    if (y.sizes() != torch::IntArrayRef(g.shape)) {
        return;
    }
    const double tol = (g.f64 ? rtol64 : rtol32) * std::max(1.0, g.maxabs);
    const auto n = static_cast<double>(y.numel());
    const auto yd = y.to(torch::kDouble);
    check_close(c.name, "sum", yd.sum().item<double>(), g.sum, tol * n);
    torch::manual_seed(99);
    check_close(c.name, "chk", (yd * torch::randn(y.sizes(), torch::kDouble)).sum().item<double>(), g.chk, tol * n);
    const auto flat = yd.flatten();
    for (const auto &[i, v]: g.els) {
        check_close(c.name, "element", flat[i].item<double>(), v, tol);
    }
    if (golden_dir == nullptr) {
        return;
    }
    std::ifstream f(std::string(golden_dir) + "/" + c.name + ".bin", std::ios::binary);
    std::vector<char> buf{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    const auto bytes = static_cast<size_t>(y.numel()) * y.element_size();
    tm_test::record(buf.size() == bytes, c.name, __FILE__, __LINE__, "dump size");
    if (buf.size() != bytes) {
        return;
    }
    const auto ref = torch::from_blob(buf.data(), y.sizes(), torch::TensorOptions().dtype(y.scalar_type()));
    const double diff = y.numel() ? (yd - ref.to(torch::kDouble)).abs().max().item<double>() : 0.0;
    std::fprintf(stderr, "  parity %-26s %-9s max|delta| %.3g (%.2g of max|Y| %.4g)\n", c.name,
                 torch::equal(y, ref) ? "bit-exact" : "", diff, diff / std::max(1.0, g.maxabs), g.maxabs);
}

static void test_goldens() {
    const char *dir = std::getenv("TM_GOLDEN_DIR");
    for (const auto &c: kaldi_cases) {
        check_case(c, dir);
    }
}

template<class Fn>
static auto throws_invalid(Fn &&f) -> bool {
    try {
        (void) f();
    } catch (const std::invalid_argument &) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Every torchaudio assert (AssertionError there), plus the inputs torchaudio fails on later (see the header comment).
static void test_validation() {
    const auto w = wave16();
    using fo = kd::fbank_option;
    // waveform and framing
    TM_CHECK(throws_invalid([&] { return kd::fbank(w[0]); })); // torchaudio: IndexError
    TM_CHECK(throws_invalid([&] { return kd::fbank(w.unsqueeze(0)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w.to(torch::kInt)); })); // torchaudio: RuntimeError in mean()
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().channel(2)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w.slice(1, 0, 399)); })); // shorter than one window
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().frame_length(0.1)); })); // window of 1 sample
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().sample_frequency(std::nan(""))); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().frame_shift(0.01)); })); // shift of 0 samples
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().frame_shift(INFINITY)); })); // torchaudio: OverflowError
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().frame_length(25.0625).round_to_power_of_two(false)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().preemphasis_coefficient(1.5)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().preemphasis_coefficient(-0.1)); }));
    TM_CHECK(throws_invalid(
            [&] { return kd::fbank(w, fo().sample_frequency(-16000).frame_length(-25).frame_shift(-10)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().energy_floor(-1.0)); })); // torchaudio: ValueError (log)
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().window_type(static_cast<kd::window_type>(5))); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().window_type(static_cast<kd::window_type>(-1))); }));
    // mel banks, through fbank and directly
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().num_mel_bins(3)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().low_freq(8000)); })); // low >= nyquist
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().low_freq(-1)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().high_freq(9000)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().high_freq(-9000)); })); // nyquist offset still < 0
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().low_freq(4000).high_freq(3000)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.9).vtln_low(10)); })); // vtln_low <= low_freq
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.9).vtln_low(8000)); })); // >= high_freq
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.9).vtln_high(9000)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.9).vtln_high(-9000)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.9).vtln_low(5000).vtln_high(4000)); }));
    TM_CHECK(throws_invalid([&] { return kd::fbank(w, fo().vtln_warp(0.0)); })); // torchaudio: ZeroDivisionError
    TM_CHECK(throws_invalid([&] { return kd::get_mel_banks(23, 511, 16000, 20, 0, 100, -500, 1); })); // odd length
    TM_CHECK(throws_invalid([&] { return kd::get_mel_banks(3, 512, 16000, 20, 0, 100, -500, 1); }));
    // mfcc / spectrogram
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, kd::mfcc_option().num_ceps(24)); }));
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, kd::mfcc_option().low_freq(-1)); }));
    // num_ceps torchaudio only fails on later: arange(-1) for the lifter, then feature[:, 0] with no cepstra
    using mo = kd::mfcc_option;
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, mo().num_ceps(-1)); })); // torchaudio: RuntimeError
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, mo().num_ceps(0).htk_compat(true)); })); // torchaudio: IndexError
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, mo().num_ceps(0).use_energy(true)); }));
    TM_CHECK(throws_invalid([&] { return kd::mfcc(w, mo().num_ceps(-23).cepstral_lifter(0).use_energy(true)); }));
    // spectrogram has no mel-bank check behind these three, so each is the one that rejects the input
    using so = kd::spectrogram_option;
    TM_CHECK(throws_invalid([&] { return kd::spectrogram(w, so().channel(5)); }));
    TM_CHECK(throws_invalid([&] { return kd::spectrogram(w, so().frame_length(0.05).round_to_power_of_two(false)); }));
    TM_CHECK(throws_invalid(
            [&] { return kd::spectrogram(w, so().frame_length(25.0625).round_to_power_of_two(false)); }));
    TM_CHECK(throws_invalid(
            [&] { return kd::spectrogram(w, so().sample_frequency(-16000).frame_length(-25).frame_shift(-10)); }));
    // vtln_warp_freq: each assert, the inflection points with negative cutoffs, and the zero warp
    const auto f = torch::linspace(0, 8000, 9);
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_freq(10, 7500, 20, 8000, 0.9, f); }));
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_freq(100, 8000, 20, 8000, 0.9, f); }));
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_freq(100, 7500, 20, 8000, 0.0, f); }));
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_freq(-5, 7500, -10, 8000, 3.0, f); })); // l = -15 < low
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_freq(100, -2, 20, -1, 0.5, f); })); // h = -1 = high
    TM_CHECK(throws_invalid([&] { return kd::vtln_warp_mel_freq(10, 7500, 20, 8000, 0.9, f); }));
}

// Too short for min_duration: an empty tensor in the input's dtype from all three (torchaudio: fbank only).
static void test_min_duration() {
    const auto w = wave16().to(torch::kDouble); // 0.5 s
    for (const auto &y:
         {kd::fbank(w, kd::fbank_option().min_duration(0.6)), kd::mfcc(w, kd::mfcc_option().min_duration(0.6)),
          kd::spectrogram(w, kd::spectrogram_option().min_duration(0.6))}) {
        TM_CHECK(y.dim() == 1 && y.numel() == 0 && y.scalar_type() == torch::kDouble);
    }
    TM_CHECK(kd::fbank(w, kd::fbank_option().min_duration(0.5)).size(0) == 48); // exactly min_duration is enough
}

static void test_detail() {
    TM_CHECK(kdd::next_power_of_2(0) == 1 && kdd::next_power_of_2(1) == 1 && kdd::next_power_of_2(2) == 2);
    TM_CHECK(kdd::next_power_of_2(3) == 4 && kdd::next_power_of_2(400) == 512 && kdd::next_power_of_2(512) == 512);
    // snip_edges: no frame when the waveform is shorter than a window (unreachable through the public functions)
    TM_CHECK(kdd::get_strided(torch::arange(3.0), 4, 1, true).sizes() == torch::IntArrayRef({0, 0}));
    // otherwise reflected at both ends ([2, 1, 0, 0, 1, 2]), the front trimmed when shift > window (torchaudio values)
    TM_CHECK(torch::equal(kdd::get_strided(torch::arange(10.0), 4, 3, false),
                          torch::tensor({0.f, 0.f, 1.f, 2.f, 2.f, 3.f, 4.f, 5.f, 5.f, 6.f, 7.f, 8.f}).reshape({3, 4})));
    TM_CHECK(torch::equal(kdd::get_strided(torch::arange(10.0), 4, 9, false), torch::tensor({{2.f, 3.f, 4.f, 5.f}})));
    // scalar mel scale: closed form, round trip, and the tensor version agrees
    TM_CHECK_CLOSE(kd::mel_scale_scalar(700.0), 1127.0 * std::log(2.0), 1e-12);
    TM_CHECK_CLOSE(kd::inverse_mel_scale_scalar(kd::mel_scale_scalar(1234.5)), 1234.5, 1e-9);
    TM_CHECK_CLOSE(kd::mel_scale(torch::tensor({1234.5}, torch::kDouble)).item<double>(), kd::mel_scale_scalar(1234.5),
                   1e-12);
}

// The setters return the most derived option, so chains mix base and derived keywords in any order.
static void test_options() {
    static_assert(std::is_same_v<decltype(kd::fbank_option().dither(1.0)), kd::fbank_option &&>);
    static_assert(std::is_same_v<decltype(kd::mfcc_option().channel(1).num_mel_bins(40)), kd::mfcc_option &&>);
    kd::spectrogram_option s;
    static_assert(std::is_same_v<decltype(s.dither(1.0)), kd::spectrogram_option &>);
    TM_CHECK(&s.dither(1.0).channel(1) == &s && s._dither == 1.0 && s._channel == 1);
    const auto f =
            kd::fbank_option().dither(1.0).num_mel_bins(80).use_power(false).window_type(kd::window_type::hamming);
    TM_CHECK(f._dither == 1.0 && f._num_mel_bins == 80 && !f._use_power && f._window_type == kd::window_type::hamming);
    const auto d = kd::mfcc_option(); // torchaudio defaults
    TM_CHECK(d._num_ceps == 13 && d._cepstral_lifter == 22.0 && d._num_mel_bins == 23 && d._low_freq == 20.0);
    TM_CHECK(d._window_type == kd::window_type::povey && d._vtln_high == -500.0 && d._energy_floor == 1.0);
}

// A non-contiguous waveform: torchaudio strides it with the input's strides even after reflecting it (wrong frames
// without snip_edges); here the frames follow the tensor actually strided.
static void test_layout() {
    const auto w = wave16();
    const auto wn = w.t().contiguous().t(); // same values, strides (1, 2)
    TM_CHECK(torch::equal(kd::fbank(w, kd::fbank_option().snip_edges(false)),
                          kd::fbank(wn, kd::fbank_option().snip_edges(false))));
    // with snip_edges the frames view the strided storage: the reductions may round differently (as in torchaudio)
    const auto y = kd::fbank(w);
    TM_CHECK_TENSOR_CLOSE(kd::fbank(wn), y, rtol32 * y.abs().max().item<double>(), 0.0);
}

// The output follows the input's device (MPS when available; the mel banks are built on the CPU and moved).
static void test_device() {
    if (!torch::mps::is_available()) {
        return;
    }
    const auto w = wave16();
    const auto o = kd::mfcc_option().num_mel_bins(40).use_energy(true);
    const auto y = kd::mfcc(w.to(torch::kMPS), o);
    TM_CHECK(y.device().is_mps());
    const auto ref = kd::mfcc(w, o);
    TM_CHECK_TENSOR_CLOSE(y, ref, 1e-4 * ref.abs().max().item<double>(), 0.0);
}

int main() {
    TM_RUN(test_goldens);
    TM_RUN(test_validation);
    TM_RUN(test_min_duration);
    TM_RUN(test_detail);
    TM_RUN(test_options);
    TM_RUN(test_layout);
    TM_RUN(test_device);
    return tm_test::summary("audio_test_compliance");
}
