# TorchMedia

TorchMedia (libtorchmedia) is a header-only C++23 library that reimplements torchaudio 2.5.1 and the core of
torchvision 0.20.1 (transforms v2) on native libtorch (`torch::Tensor`) operators. It is meant for C++ inference
preprocessing that has to compute what torchaudio / torchvision computed at training time: each ported op is compared
point-wise against the Python library, and on the CPU most are bit-exact in float32 and float64. `develop_log/`
records the measurements and every intentional difference.

## What is included

| Namespace | Python counterpart | Contents |
|---|---|---|
| `torchmedia::audio::io` | `torchaudio.load` / `info` / `save` | Decode WAV, FLAC, MP3 and Ogg Vorbis from a file or from memory (with frame offset / count), encode WAV. Returns the native sample rate; resampling is a separate op. |
| `torchmedia::audio::functional` | `torchaudio.functional` | spectrogram / inverse_spectrogram, griffinlim, mel and linear filter banks, create_dct, amplitude_to_DB, resample, speed, pitch_shift, phase_vocoder, lfilter / filtfilt / biquad and the SoX effects (EQ biquads, overdrive, phaser, flanger, dither, contrast, vad), loudness, masking, add_noise, convolve / fftconvolve, detect_pitch_frequency, psd / MVDR / RTF beamforming, forced_align |
| `torchmedia::audio::transform` | `torchaudio.transforms` | Spectrogram, MelSpectrogram, MFCC, LFCC, InverseMelScale, GriffinLim, Resample, Speed, SpeedPerturbation, PitchShift, TimeStretch, SpecAugment, Vol, Vad, Loudness, PSD, MVDR, SoudenMVDR, RTFMVDR, ... |
| `torchmedia::audio::compliance::kaldi` | `torchaudio.compliance.kaldi` | fbank, mfcc, spectrogram |
| `torchmedia::audio::separation` | torchiva 0.1.1 | AuxIVA-IP, OverIVA and ILRMA-IP blind source separation |
| `torchmedia::vision::io` | | image load / save (stb_image) |
| `torchmedia::vision::functional`, `torchmedia::vision::transform` | `torchvision.transforms.v2` | resize, crop, center / five / ten crop, pad, normalize, to_dtype, grayscale, flips, brightness, invert; the Resize, CenterCrop, Pad, Normalize, ToDtype, Grayscale, RandomHorizontalFlip, RandomVerticalFlip, FiveCrop and TenCrop classes |

`<torchmedia.hpp>` includes all of the above. The optional `torchmedia::plot` module (matplot++, needs gnuplot at
runtime) is built with `-DTORCHMEDIA_WITH_PLOT=ON` and included separately with `<torchmedia/plot.hpp>`.

## Requirements

- A C++23 compiler (the headers use `<print>`). Developed and tested on macOS arm64 with Apple clang 21.
- CMake 3.20 or newer.
- libtorch 2.5.1. `script/download_dep` fetches the CPU build for macOS arm64, Linux or Windows.

## Build and test

```sh
python3 script/download_dep
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build
```

`script/download_dep` puts libtorch (and matplot++, used only by the optional plot module) under `dependence/`.
To use an existing copy instead, pass `-DTORCHMEDIA_DEPENDENCE_DIR=<dir containing libtorch/>` to the first `cmake`
call. The unit tests in `unit_test/` double as usage examples.

## Use it in your project

TorchMedia is header-only. Add this repository with `add_subdirectory` and link the `torch_media` interface target,
which carries the include path, C++23 and libtorch:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app CXX)

set(TORCHMEDIA_BUILD_TESTS OFF)
add_subdirectory(torch_media) # this repository (libtorch in torch_media/dependence/libtorch)

add_executable(audio_example audio.cpp)
target_link_libraries(audio_example PRIVATE torch_media)
add_executable(vision_example vision.cpp)
target_link_libraries(vision_example PRIVATE torch_media)
```

The audio and image codecs are vendored single-header C libraries (dr_wav, dr_flac, dr_mp3, stb_vorbis, stb_image).
Their implementation is compiled only in the translation unit that defines `TORCHMEDIA_IO_IMPLEMENTATION` before
including TorchMedia, so define it in exactly one `.cpp` of your program, as the audio example below does. A program
that already compiles one of the audio codecs itself opts out with
`TORCHMEDIA_NO_{DR_WAV,DR_FLAC,DR_MP3,STB_VORBIS}_IMPLEMENTATION`.

## Audio example

Read a file, resample it and compute torchaudio's mel spectrogram and Kaldi filter bank features:

```cpp
#define TORCHMEDIA_IO_IMPLEMENTATION // in exactly one .cpp of your program: emits the vendored codecs
#include <numbers>
#include <print>
#include <torchmedia.hpp>

namespace io = torchmedia::audio::io;
namespace F = torchmedia::audio::functional;
namespace T = torchmedia::audio::transform;
namespace kaldi = torchmedia::audio::compliance::kaldi;

int main() {
    // Write a 1 s, 440 Hz tone at 44.1 kHz as a WAV file, then read it back like torchaudio.load.
    const auto t = torch::arange(44100, torch::kFloat) / 44100;
    io::save_audio(0.5 * torch::sin(2 * std::numbers::pi * 440 * t).unsqueeze(0), "tone.wav", 44100);
    const auto [wave, sample_rate] = io::load_audio("tone.wav"); // [channels, frames] float32, native rate

    const auto wave16k = F::resample(wave, sample_rate, 16000); // torchaudio.functional.resample
    const T::MelSpectrogram mel(T::mel_spectrogram_option().sample_rate(16000).n_fft(400).hop_length(160).n_mels(80));
    const auto mels = mel(wave16k); // torchaudio.transforms.MelSpectrogram: [1, 80, 101]
    const auto fbank = kaldi::fbank(wave16k, kaldi::fbank_option().num_mel_bins(80)); // [98, 80]
    std::println("mel {}x{}x{}, kaldi fbank {}x{}", mels.size(0), mels.size(1), mels.size(2), fbank.size(0),
                 fbank.size(1));
}
```

It prints `mel 1x80x101, kaldi fbank 98x80`.

## Vision example

torchvision's ImageNet preprocessing on a `[C, H, W]` image:

```cpp
#include <print>
#include <torchmedia.hpp>

namespace T = torchmedia::vision::transform;

int main() {
    // A [3, H, W] uint8 image. torchmedia::vision::io::load_image(path) gives [C, H, W] float32 in [0, 1], with C
    // the file's channel count (the 3-channel Normalize below needs an RGB file).
    const auto image = torch::randint(0, 256, {3, 375, 500}, torch::kUInt8);

    // torchvision's ImageNet preprocessing: Resize(256), CenterCrop(224), ToDtype(float32, scale=True), Normalize.
    const T::Resize resize(T::resize_option().size({256}));
    const T::CenterCrop crop(T::center_crop_option().size({224}));
    const T::ToDtype to_float(T::to_dtype_option().dtype(torch::kFloat32).scale(true));
    const T::Normalize normalize(T::normalize_option().mean({0.485, 0.456, 0.406}).std({0.229, 0.224, 0.225}));
    const auto x = normalize(to_float(crop(resize(image)))); // [3, 224, 224] float32
    std::println("{}x{}x{}", x.size(0), x.size(1), x.size(2));
}
```

It prints `3x224x224`.

## Behaviour

- Results stay on the input's device, and floating-point results keep the input's dtype. Parity is measured on the CPU
  (float32 and float64) on macOS arm64. The sequential loops (lfilter, overdrive, phaser, flanger) run as torchaudio's
  scalar loops on the CPU; float32 / float64 input on another device runs them on a host copy. Their parity needs
  floating-point contraction within one expression at most: the `torch_media` target adds `-ffp-contract=on` for GCC
  and clang (GCC's GNU-mode default, `fast`, changes results); pass it yourself without the CMake target.
- Autograd input keeps these loops as tensor ops on its device. On MPS the forward pass is correct, but a backward
  pass through them exhausts GPU memory beyond about 20k samples (torchaudio's own loops fail there too).
- Invalid arguments raise `std::invalid_argument` where torchaudio raises. With `LIB_TORCH_MEDIA_NO_EXCEPTIONS`
  defined, a call with invalid arguments returns a default value instead.
- libtorch 2.5.1's MPS backend lacks some ops (`istft`, `angle`), so GriffinLim, InverseSpectrogram, PitchShift and
  TimeStretch do not run on MPS.

## License

MIT, see [LICENSE](LICENSE). The bundled and ported third-party code and its licenses are listed in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
