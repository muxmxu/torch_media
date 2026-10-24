# TorchMedia

TorchMedia (libtorchmedia) is a header-only C++23 library that reimplements torchaudio 2.5.1 and the core of
torchvision 0.20.1 (transforms v2) on native libtorch (`torch::Tensor`) operators. It is meant for C++ inference
preprocessing that has to compute what torchaudio / torchvision computed at training time: each ported op is compared
point-wise against the Python library, and on the CPU most are bit-exact in float32 and float64. `develop_log/`
records the measurements; [doc/](doc/) summarises how to port Python code, every intentional difference and what
reproducibility depends on.

> **Status: personal hobby project.** It is provided "as is", without warranty of any kind, and without any
> obligation to support, maintain, or fix it. APIs and behaviour may change without notice. Verify its outputs for
> your own use case before relying on them; it is not intended for safety-critical use. The author accepts no
> liability for any damage or loss arising from its use — see [LICENSE](LICENSE).

## What is included

| Namespace | Python counterpart | Contents |
|---|---|---|
| `torchmedia::audio::io` | `torchaudio.load` / `info` / `save` | Decode WAV, FLAC, MP3 and Ogg Vorbis from a file or from memory (with frame offset / count), encode WAV. Returns the native sample rate; resampling is a separate op. |
| `torchmedia::audio::functional` | `torchaudio.functional` | spectrogram / inverse_spectrogram, griffinlim, mel and linear filter banks, create_dct, amplitude_to_DB, resample, speed, pitch_shift, phase_vocoder, lfilter / filtfilt / biquad and the SoX effects (EQ biquads, overdrive, phaser, flanger, dither, contrast, vad), loudness, masking, add_noise, convolve / fftconvolve, detect_pitch_frequency, psd / MVDR / RTF beamforming, forced_align |
| `torchmedia::audio::transform` | `torchaudio.transforms` | Spectrogram, MelSpectrogram, MFCC, LFCC, InverseMelScale, GriffinLim, Resample, Speed, SpeedPerturbation, PitchShift, TimeStretch, SpecAugment, Vol, Vad, Loudness, PSD, MVDR, SoudenMVDR, RTFMVDR, ... |
| `torchmedia::audio::compliance::kaldi` | `torchaudio.compliance.kaldi` | fbank, mfcc, spectrogram |
| `torchmedia::audio::separation` | torchiva 0.1.1 | AuxIVA-IP, OverIVA and ILRMA-IP blind source separation |
| `torchmedia::vision::io` | `torchvision.io.read_file` / `decode_image` / `read_image` | decode_image / read_image: JPEG and PNG to uint8 (uint16 for 16-bit PNG) in torchvision's read modes, PNG bit-exact except palette PNGs in unchanged mode (colours, not indices) and the 16-bit sBIT shift, JPEG within a few levels (stb_image instead of libjpeg-turbo); load_image: float32 [0, 1] from PNG, JPEG, BMP, ... (stb_image) and PNM P1-P6 (sample / maxval); save_image (PNG) |
| `torchmedia::vision::functional`, `torchmedia::vision::transform` | `torchvision.transforms.v2` | geometry (resize, crop, resized / center / five / ten crop, pad, flips, rotate, affine, perspective, elastic), colour (brightness, contrast, saturation, hue, gamma, sharpness, grayscale, invert, posterize, solarize, autocontrast, equalize), normalize, to_dtype, gaussian blur / noise, erase; their classes, the random augmentations (RandomResizedCrop, ColorJitter, RandomAffine, RandomErasing, ScaleJitter, ...) with torchvision's seeded draws, and Compose / RandomApply / RandomChoice / RandomOrder |
| `torchmedia::metric::detection` | `torchmetrics.detection`, `torchvision.ops`, pycocotools | box ops (box_iou, generalized / distance / complete IoU, box_convert), the IoU metrics, MeanAveragePrecision (= pycocotools COCOeval, bbox and segm), VOC average precision |
| `torchmedia::metric::image`, `torchmedia::metric::segmentation` | `torchmetrics.image`, `torchmetrics.segmentation` | PSNR, SSIM, MS-SSIM, UQI, TotalVariation, SAM, ERGAS, RASE, RMSE-SW, SCC, VIF; MeanIoU, DiceScore, GeneralizedDiceScore |
| `torchmedia::metric::audio` | `torchmetrics.audio`, pystoi, mir_eval | SNR, SI-SDR, SI-SNR, C-SI-SNR, SDR, SA-SDR, PIT, STOI (= pystoi), bss_eval_sources (= mir_eval), speaker-verification EER / minDCF |
| `torchmedia::metric::text` | `torchmetrics.text` | WER, CER, MER, WIL, WIP, EditDistance |
| `torchmedia::metric::classification` | `torchmetrics.classification` | binary / multiclass / multilabel stat scores, accuracy, precision, recall, F-scores, specificity, confusion matrix, Jaccard, MCC, Cohen's kappa, ROC / PR curves, AUROC, average precision, EER, fixed-point metrics, calibration error, hinge loss, ranking metrics |
| `torchmedia::data` | `torchvision.datasets`, `torch.utils.data` | datasets for libtorch's `torch::data` loader: DatasetFolder / ImageFolder (torchvision's sample order and errors), AudioFolder, ManifestDataset (Kaldi-style or TSV lists), PadCollate (variable-length batches), Subset / ConcatDataset / random_split (same split as torch for the same seed) |

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

For a libtorch outside `torch_media/dependence/`, set `TORCHMEDIA_DEPENDENCE_DIR` to the directory containing
`libtorch/` before `add_subdirectory` (or pass it with `-D`). If no libtorch is there, `find_package(Torch)` uses any
other libtorch it finds on the system; TorchMedia is written against libtorch 2.5.1.

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

## Metric example

`torchmedia::metric` follows torchmetrics 1.9: each metric is a functional (`word_error_rate(preds, target)`) and a
class with `update` / `compute` / `reset` / `forward`. MeanAveragePrecision reproduces pycocotools' COCOeval in float64;
`develop_log/` lists the parity measured for each family. A COCO mAP over two images and a word error rate:

```cpp
#include <print>
#include <torchmedia.hpp>

namespace detection = torchmedia::metric::detection;
namespace text = torchmedia::metric::text;

int main() {
    // torchmetrics.detection.MeanAveragePrecision (= pycocotools COCOeval): one update per batch of images, each
    // image with xyxy boxes, scores and labels (predictions) or boxes and labels (targets).
    detection::MeanAveragePrecision coco_map;
    coco_map.update({{torch::tensor({{10., 10., 50., 50.}, {60., 60., 90., 95.}}), torch::tensor({0.9, 0.6}),
                      torch::tensor({1, 2})},
                     {torch::tensor({{0., 0., 30., 30.}}), torch::tensor({0.8}), torch::tensor({1})}},
                    {{torch::tensor({{12., 8., 48., 52.}, {58., 62., 92., 90.}}), torch::tensor({1, 2})},
                     {torch::tensor({{5., 5., 35., 35.}, {40., 40., 70., 70.}}), torch::tensor({1, 1})}});
    const auto r = coco_map.compute();
    std::println("map {:.4f}, map_50 {:.4f}, map_75 {:.4f}, mar_100 {:.4f}", r.map, r.map_50, r.map_75, r.mar_100);

    // torchmetrics.text.WordErrorRate: word edit distances over all pairs / number of target words.
    const auto wer = text::word_error_rate({"the cat sat on the mat", "hello world"},
                                           {"the cat sat on a mat", "hello big world"});
    std::println("wer {:.4f}", wer.item<double>());
}
```

It prints `map 0.3842, map_50 0.8317, map_75 0.1683, mar_100 0.3833` and `wer 0.2222`, the values torchmetrics (with
pycocotools) gives for the same input.

## Data example

`torchmedia::data` adds datasets to libtorch's own `torch::data` loader (samplers, worker threads, `Stack`), which it
does not reimplement. An AudioFolder batched with PadCollate:

```cpp
#define TORCHMEDIA_IO_IMPLEMENTATION // in exactly one .cpp of your program: emits the vendored codecs
#include <filesystem>
#include <print>
#include <torchmedia.hpp>

namespace data = torchmedia::data;

int main() {
    // A torchvision-style tree root/<class>/<file>: three noise clips of different lengths in two classes.
    const std::filesystem::path root = "speech";
    const std::vector<std::pair<std::string, int64_t>> files = {
            {"no/a.wav", 12000}, {"yes/b.wav", 16000}, {"yes/c.wav", 9000}};
    for (const auto &[file, frames]: files) {
        std::filesystem::create_directories((root / file).parent_path());
        torchmedia::audio::io::save_audio(0.1 * torch::randn({1, frames}), root / file, 16000);
    }

    // AudioFolder = torchvision's DatasetFolder over load_audio (here mono at 16 kHz); classes are the sorted
    // sub-directory names, samples are {waveform [1, T], class index}.
    data::AudioFolder clips(root, {}, data::audio_folder_option().sample_rate(16000).mono(true));
    std::println("classes {}, {} clips", clips.classes(), clips.size().value());

    // libtorch's own DataLoader; PadCollate pads each batch to its longest clip and keeps the lengths.
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            clips.map(data::PadCollate()), torch::data::DataLoaderOptions(2));
    for (const auto &batch: *loader) { // padded_batch {data [B, 1, T_max], target [B], lengths [B]}
        std::print("batch [{}, {}, {}]:", batch.data.size(0), batch.data.size(1), batch.data.size(2));
        for (int64_t i = 0; i < batch.data.size(0); ++i) {
            std::print(" (class {}, {} frames)", batch.target[i].item<int64_t>(), batch.lengths[i].item<int64_t>());
        }
        std::println();
    }
}
```

It prints

```
classes ["no", "yes"], 3 clips
batch [2, 1, 16000]: (class 0, 12000 frames) (class 1, 16000 frames)
batch [1, 1, 9000]: (class 1, 9000 frames)
```

ImageFolder works the same way: it decodes with `decode_image` (JPEG / PNG as uint8 RGB, batched with
`torch::data::transforms::Stack<>()`) or, with `image_folder_option().load_image(true)`, with `load_image` (float32,
also BMP and PNM). `random_split(dataset, {0.8, 0.2}, seed)` gives the indices of `torch.utils.data.random_split` with
`torch.Generator().manual_seed(seed)`.

## Behaviour

- Results stay on the input's device, and floating-point results keep the input's dtype. Parity is measured on the CPU
  (float32 and float64) on macOS arm64. The sequential loops (lfilter, overdrive, phaser, flanger) run as torchaudio's
  scalar loops on the CPU; float32 / float64 input on another device runs them on a host copy. Their parity needs
  floating-point contraction within one expression at most: the `torch_media` target adds `-ffp-contract=on` for GCC
  and clang (GCC's GNU-mode default, `fast`, changes results); pass it yourself without the CMake target.
- Autograd input keeps these loops as tensor ops on its device. On MPS the forward pass is correct, but a backward
  pass through them exhausts GPU memory beyond about 20k samples (torchaudio's own loops fail there too).
- Invalid arguments raise `std::invalid_argument` where torchaudio raises (`std::runtime_error` for I/O). Define
  `LIB_TORCH_MEDIA_NO_EXCEPTIONS` (for the whole program) to keep the library silent instead: a failing call returns
  an undefined tensor (`!out.defined()`), a value-initialised struct or an empty container, and
  `torchmedia::last_error()` returns its message (per thread; `torchmedia::clear_last_error()` resets it). Errors from
  libtorch itself (`c10::Error`) still throw.
- libtorch 2.5.1's MPS backend lacks some ops (`istft`, `angle`), so GriffinLim, InverseSpectrogram, PitchShift and
  TimeStretch do not run on MPS.

## Documentation

Each function's comment in its header names its Python counterpart and any difference from it. Three pages cover
what no single function documents:

- [doc/porting-from-python.md](doc/porting-from-python.md): how a Python call maps to TorchMedia (namespaces, names,
  option structs, classes, errors, building).
- [doc/deviations.md](doc/deviations.md): every intentional or measured difference from the Python libraries.
- [doc/reproducibility.md](doc/reproducibility.md): when results are bit-identical to Python, what changes them, and
  seeded randomness.

## License

MIT, see [LICENSE](LICENSE). The bundled and ported third-party code and its licenses are listed in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
