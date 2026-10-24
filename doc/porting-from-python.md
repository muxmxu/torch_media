# Porting Python preprocessing to TorchMedia

This page explains how a torchaudio / torchvision / torchmetrics call translates to TorchMedia.

The parameters themselves are explained by the Python documentation, because each port computes what its Python
counterpart computes. The comment above each function in its header names that counterpart and any difference. The
results that differ are collected in [deviations.md](deviations.md).

## Where things are

| Python | C++ namespace | Header (under `libtorchmedia/include/torchmedia/`) |
|---|---|---|
| `torchaudio.load` / `info` / `save` | `torchmedia::audio::io` | `_audio/_io.hpp` |
| `torchaudio.functional` | `torchmedia::audio::functional` | `_audio/_functional*.hpp` |
| `torchaudio.transforms` | `torchmedia::audio::transform` | `_audio/_transform*.hpp` |
| `torchaudio.compliance.kaldi` | `torchmedia::audio::compliance::kaldi` | `_audio/_compliance_kaldi.hpp` |
| torchiva | `torchmedia::audio::separation` | `_audio/_separation.hpp` |
| `torchvision.io` | `torchmedia::vision::io` | `_vision/_io.hpp` |
| `torchvision.transforms.v2.functional` | `torchmedia::vision::functional` | `_vision/_functional.hpp` |
| `torchvision.transforms.v2` | `torchmedia::vision::transform` | `_vision/_transform*.hpp` |
| `torchmetrics.*`, `torchmetrics.functional.*` | `torchmedia::metric::{detection, image, segmentation, audio, text, classification}` | `_metric/*.hpp` |
| `torchvision.datasets`, `torch.utils.data` | `torchmedia::data` | `_data/*.hpp` |

`#include <torchmedia.hpp>` brings in everything except the optional plot module.

## Names

Functions and classes keep their Python names: `resample`, `MelSpectrogram`, `rotate`, `RandomResizedCrop`,
`word_error_rate`, `MeanAveragePrecision`. The exceptions are:

| Python | TorchMedia |
|---|---|
| `torchaudio.functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, ...)` | `functional::mel_filter_bank(n_mels, f_min, f_max, sample_rate, n_stft_bins, ...)` (argument order differs) |
| `torchaudio.functional.DB_to_amplitude` | `functional::db_to_amplitude` |
| `torchvision ... horizontal_flip` / `vertical_flip` / `to_grayscale` / `elastic_transform` (aliases of the right-hand names in torchvision) | `hflip` / `vflip` / `rgb_to_grayscale` / `elastic` |
| `torchvision ... convert_image_dtype` | `to_dtype` (with `scale = true`) |
| `torchiva.AuxIVA_IP`, OverIVA, ILRMA | `separation::auxiva_ip` (OverIVA when `n_src` < channels), `separation::ilrma` |
| A torchmetrics flag that changes the return type (`return_full_image`, `return_rmse_map`) | a second function: `structural_similarity_index_measure_with_image`, `root_mean_squared_error_using_sliding_window_with_map` |

Additions with no Python counterpart:
- `functional::mel_scale`, `functional::melspectrogram` and `functional::mfcc` are the functional forms of the
  transforms.
- `metric::detection` also holds the box ops of `torchvision.ops`, such as `box_iou` and `box_convert`.
- `metric::audio` also holds `bss_eval_sources` (= mir_eval) and the speaker-verification `equal_error_rate` /
  `min_dcf` with their `EqualErrorRate` / `MinDCF` classes.

## Arguments

- **Positional arguments stay positional.** Python keyword arguments go into an option struct that is passed last:
  `resample(waveform, orig_freq, new_freq, opt)`.
- **Option structs.** Most are named `xxx_option` and have one fluent setter per keyword, named like the keyword:
  `mel_spectrogram_option().sample_rate(16000).n_mels(80)`. An unset field keeps the Python default; the exceptions
  are in [deviations.md](deviations.md).
  - Several classes share one option struct; for example `RandomHorizontalFlip` and `RandomVerticalFlip` both take
    `random_flip_option`. The constructor's declaration names the struct.
  - A few audio option structs are plain aggregates without setters. Fill them with designated initializers, as in
    `resample(..., {.lowpass_filter_width = 16})`.
- **Vision functionals** take torchvision's positional order, with C++ default arguments in place of keywords:
  `rotate(img, angle, interpolation, expand, center, fill)`.
- **`None`.** It becomes `std::optional` / `c10::optional`, an empty vector `{}`, or a sentinel (`0` or `-1`). For
  example, `spectrogram_option` `hop_length` 0 means torchaudio's `None`. The field's comment says which.
- **Strings and enums.** Most become `enum class` values: `interpolation_mode::bilinear`, `image_read_mode::rgb`,
  `convolve_mode::same`, `average_mode::macro`. A few audio options keep torchaudio's strings, such as
  `normalize_method "window"` and `create_dct norm "ortho"`.
- **Tuples.** Python tuples come back as `std::pair` / `std::tuple`. Named results come back as a struct:
  - `load_audio` returns `{data, sample_rate}`, ready for `auto [wave, sr] = ...`;
  - `MeanAveragePrecision::compute` returns `.map`, `.map_50`, ...

## Classes

- **Transforms.** A transform class is built from its option struct and is called with `operator()` or `forward()`.
  - Calls are `const`, so one instance can be shared across threads.
  - The exception is `audio::transform::MVDR`, whose online mode carries state from call to call.
  - The random vision transforms draw from the global generator; see [reproducibility.md](reproducibility.md).
- **Metric and dataset objects.** A metric class accumulates state, so give each thread its own instance.
  Datasets are read-only after construction and safe for loader workers.
- **Containers.** `Compose`, `RandomApply`, `RandomOrder` and `RandomChoice` take any `transform_fn`
  (`std::function<tensor_t(const tensor_t &)>`), so a lambda works as well as a class.
- **Metrics.** A metric class has `update`, `compute`, `reset` and `forward`, like a torchmetrics `Metric`. `forward`
  updates the state and returns the metric of that batch alone. The functional form computes the metric in one call.
- **Datasets.** They are `torch::data::datasets::Dataset`s. Use libtorch's own `make_data_loader`, samplers and
  `Stack`; the readme has an example.

## Tensors, dtypes and devices

- **Device and dtype.** Results stay on the input's device. Floating-point results keep the input's dtype. The few
  exceptions are named in the function comment and in [deviations.md](deviations.md), for example detection mAP on
  the CPU in float64.
- **Decoders.** They return CPU tensors: `load_audio` gives float32 `[channels, frames]`, `decode_image` gives uint8
  `[C, H, W]` (uint16 for 16-bit PNG), and `load_image` gives float32 in [0, 1].
- **Resampling.** `load_audio` returns the file's own sample rate. Resample with `functional::resample` or
  `transform::Resample`.

## Errors

- **Exception types.** Where Python raises `ValueError` / `TypeError`, TorchMedia throws `std::invalid_argument`. I/O
  failures throw `std::runtime_error`, and a dataset index past the end throws `std::out_of_range`. Errors that the
  Python library leaves to torch stay torch's `c10::Error`.
- **No-exceptions build.** Define `LIB_TORCH_MEDIA_NO_EXCEPTIONS` for the whole program. A failing call then returns a
  fallback value and records its message for `torchmedia::last_error()`. The fallback is:
  - an undefined tensor (test `!out.defined()`);
  - a value-initialised struct;
  - or an empty container.

  The comment in `global_include.hpp` gives the full contract, including what a rejected constructor leaves behind.

## Building

- **Codec implementations.** Define `TORCHMEDIA_IO_IMPLEMENTATION` in exactly one `.cpp` before including TorchMedia.
  That file then contains the vendored codecs. The readme's "Use it in your project" shows the CMake side.
- **libtorch version.** Any libtorch that keeps the API works; 2.5.1 and 2.14.1 are tested. For results identical to
  your Python pipeline, use the libtorch version of the PyTorch that ran it (see
  [reproducibility.md](reproducibility.md)).
- **Pointing at libtorch.** Set `TORCHMEDIA_DEPENDENCE_DIR` to the directory that holds `libtorch/`, either before
  `add_subdirectory` or with `-D`. If no libtorch is there, `find_package(Torch)` silently takes any other libtorch it
  finds on the system, and the compile errors then come from the version mismatch.
- **Compiler flags.** Do not build with `-ffast-math` or `-ffp-contract=fast` (see
  [reproducibility.md](reproducibility.md)).

## Example

```cpp
#include <print>
#include <torchmedia.hpp>

namespace AF = torchmedia::audio::functional;
namespace VF = torchmedia::vision::functional;
namespace VT = torchmedia::vision::transform;
namespace classification = torchmedia::metric::classification;

int main() {
    const auto wave = torch::randn({1, 16000});

    // torchaudio.functional.resample(wave, 16000, 8000, lowpass_filter_width=16): positional arguments stay
    // positional; keyword arguments go into an option struct (this one is a plain aggregate).
    const auto down = AF::resample(wave, 16000, 8000, {.lowpass_filter_width = 16});

    // torchaudio.functional.spectrogram(..., n_fft=512, hop_length=None, ...): fluent setters named after the
    // keyword; an unset field keeps torchaudio's default (here hop_length 0 = None = win_length // 2).
    const auto spec = AF::spectrogram(down, AF::spectrogram_option().n_fft(512));

    // torchvision.transforms.v2.functional.rotate(img, 30.0, expand=True): vision functionals mirror torchvision's
    // positional order with C++ default arguments; strings and enums become enum classes.
    const auto img = torch::rand({3, 64, 48});
    const auto rotated = VF::rotate(img, 30.0, VF::interpolation_mode::bilinear, /*expand=*/true);

    // torchvision.transforms.v2.RandomHorizontalFlip(p=0.5): a class built from its option struct, called with ().
    const VT::RandomHorizontalFlip flip(VT::random_flip_option().p(0.5));
    const auto flipped = flip(img);

    // torchmetrics.classification.MulticlassAccuracy(num_classes=3): update per batch, compute at the end.
    classification::MulticlassAccuracy acc(classification::multiclass_stat_scores_option().num_classes(3));
    acc.update(torch::tensor({0, 2, 1, 2}), torch::tensor({0, 1, 1, 2}));
    std::println("spec {}, rotated {}, flipped {}, accuracy {:.4f}", spec.sizes(), rotated.sizes(), flipped.sizes(),
                 acc.compute().item<double>());
}
```

It prints `spec [1, 257, 32], rotated [3, 80, 74], flipped [3, 64, 48], accuracy 0.8333`. These are the shapes and the
accuracy that torchaudio, torchvision and torchmetrics give for the same calls.

## Not ported

- **torchaudio.** `apply_codec`, `rnnt_loss` / `RNNTLoss`, the pipelines and pretrained models, and the streaming /
  FFmpeg I/O. Saving is WAV only.
- **torchvision.**
  - The AutoAugment family (`AutoAugment`, `RandAugment`, `TrivialAugmentWide`, `AugMix`);
  - `CutMix` / `MixUp` and `JPEG`;
  - the bounding-box, mask and video transforms;
  - the PIL conversions;
  - `Lambda` (pass a C++ lambda to `Compose` instead);
  - GIF / WebP decoding.
- **torchmetrics.** Only the families in the readme's table. Within them, there is no `plot()`, no `task=` wrappers,
  and no `validate_args` (inputs are always checked).
- **Model-coupled feature extractors** such as the Whisper / HF processors, and **room-impulse-response
  simulation.** Out of scope on purpose.
