# Reproducing Python's numbers

TorchMedia exists to compute in C++ what the Python pipeline computed at training time. This page says when the
results are bit-identical, what can break that, and how to get the same random draws.

## What was measured

- **Where.** Every parity figure in `develop_log/` was measured on macOS arm64 (Apple clang, libtorch 2.5.1), on the
  CPU in float32 and float64. Some ops were also run on MPS. Nothing was measured on x86, Linux, Windows or CUDA.
- **Most CPU ops are bit-exact against the Python library.** They call the same libtorch kernels in the same order,
  and Python scalar arithmetic is reproduced in the same order and precision.
- **The rest have a measured tolerance.** This covers the decoders (JPEG, MP3, Vorbis), STOI, `bss_eval_sources`
  and large OverIVA problems; [deviations.md](deviations.md) gives the figures. The tests assert those tolerances.

## What can change the bits

- **The CPU.** libtorch picks kernels by CPU capability, and so does torchvision. An example is uint8 bilinear
  `resize`: on an AVX2 machine it uses torch's native uint8 kernel, otherwise it goes through float32 (arm64 does
  the latter). Both sides take the same path on the same machine. Compare against Python on the machine you deploy
  on, not against numbers from another one.
- **The device.** MPS runs its own kernels, so its results generally differ from the CPU's in the last bits.
  - `lfilter`, `filtfilt`, `biquad` and the filters built on them, `overdrive`, `phaser` and `flanger` compute on a
    CPU copy, so they match the CPU exactly.
  - Some ops do not run on MPS at all: GriffinLim, InverseSpectrogram, PitchShift and TimeStretch, because libtorch
    2.5.1 lacks `istft` / `angle` there.
  - CUDA has not been tested.
- **Floating-point contraction.** Fusing `a * b + c` into one FMA changes the rounding, so:
  - The `torch_media` CMake target adds `-ffp-contract=on`, which allows fusion only within one expression. GCC's
    GNU-mode default (`fast`) fuses across statements and changes results. Without the CMake target, pass
    `-ffp-contract=on` yourself.
  - The functions that reproduce Python scalar arithmetic switch contraction off inside themselves
    (`TORCHMEDIA_FP_CONTRACT_OFF`), for example torchvision's affine matrix and Python-ordered sums. Without this,
    186 of 4676 rotate / affine / blur cases differed.
  - **Never build with `-ffast-math` or `-ffp-contract=fast`.** They override both mechanisms.
  - On Darwin, clang merges `sin(x)` and `cos(x)` into one `sincos` call that can be 1 ulp off Python's `math.sin`.
    The library routes its own scalar `sin` through `detail::py_sin` to avoid it, so no action is needed.
  - Notes: [progress09](../develop_log/2026-10-23/progress09-vision-color-geometry.md) D4,
    [progress10](../develop_log/2026-10-23/progress10-vision-random-transforms.md) D6.
- **The decoder.** JPEG goes through stb_image, not libjpeg-turbo, so a pipeline that decodes JPEG can be up to 3
  levels off torchvision before any other op runs. Decode with the same library on both sides if the training data is
  JPEG and exactness matters.

## Seeded randomness

libtorch's CPU generator is PyTorch's, so `torch::manual_seed(s)` in C++ produces the draws of `torch.manual_seed(s)`
in Python. The random ops below make the same torch calls as their Python counterparts: the same functions, in the
same order, with the same dtypes. Under the same seed their outputs are therefore identical:

| Op | Python counterpart | Notes |
|---|---|---|
| vision random transforms (`RandomResizedCrop`, `ColorJitter`, `RandomAffine`, ..., `RandomApply` / `RandomChoice` / `RandomOrder`) | `torchvision.transforms.v2` | 35,388 seeded outputs bit-exact on the CPU ([progress10](../develop_log/2026-10-23/progress10-vision-random-transforms.md) D3, D9) |
| `GaussianNoise` on MPS | the same on MPS | `torch::manual_seed` seeds the MPS generator too |
| masking (`mask_along_axis`, `TimeMasking`, `FrequencyMasking`, `SpecAugment`) | `torchaudio` | ([progress04](../develop_log/2026-10-23/progress04-audio-surface-audit.md) D7) |
| `SpeedPerturbation` | `torchaudio.transforms.SpeedPerturbation` | factor draw ([progress08](../develop_log/2026-10-23/progress08-audio-parity-followups.md) D5) |
| `dither` with RPDF | `torchaudio.functional.dither` | TPDF draws nothing; GPDF differs (see [deviations.md](deviations.md)) |
| `separation::ilrma` | torchiva ILRMA | the NMF initialisation ([progress03](../develop_log/2026-10-23/progress03-auxiva-bss.md) D4) |
| `data::random_split(ds, lengths, seed)` | `random_split(..., generator=torch.Generator().manual_seed(seed))` | without a seed it uses the global generator, like Python |

Seeded parity needs the calls in the same order as in Python, one after another:
- **Threads.** The classes are safe to call from several threads, but concurrent calls interleave their draws on the
  one global generator, and the results then depend on scheduling.
- **Data loaders.** With a random transform inside a `torch::data` loader, use `workers(0)` for a seeded run. The
  datasets themselves are deterministic.
- **Other users of the generator.** Any other code that draws from the global generator between two calls shifts
  every later draw, on both sides.
