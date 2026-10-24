# Differences from the Python libraries

TorchMedia ports torchaudio 2.5.1, torchvision 0.20.1, torchmetrics 1.9.0, torchiva 0.1.1, pycocotools, pystoi and
mir_eval op for op. This page lists every place where it gives a different result on purpose or because of a
different decoder. Each entry links to the develop_log note that measured it and gives the reasons. The comment above
a function in its header says the same thing for that function.

Not listed here:
- **Stricter argument checks.** Many functions reject input that the Python library accepts silently or that crashes
  it later, for example mismatched lengths, empty threshold lists or a 0-D tensor. They throw `std::invalid_argument`
  before computing anything. The Python library returns something or raises another error type. The notes linked in
  each section list these checks.
- **Python warnings.** No warnings are printed.
- **Parts that are not ported.** See [porting-from-python.md](porting-from-python.md#not-ported).

## Audio I/O (`torchmedia::audio::io`)

Notes: [progress05](../develop_log/2026-10-23/progress05-audio-io-formats.md) D4, D7.

- **WAV and FLAC values.** 8 / 16 / 24-bit PCM, FLAC, W64, RF64, mu-law, A-law and IMA ADPCM equal `torchaudio.load`
  bit for bit. The other WAV encodings differ:
  - 32-bit PCM is exact here. torchaudio's sox path rounds it, up to 6e-8 off.
  - Float WAV is returned as stored, including values beyond [-1, 1]. torchaudio clips them.
- **MP3 length.** dr_mp3 is gapless. A file with a LAME / Xing tag decodes to exactly the source length. libmad under
  torchaudio adds the encoder and decoder delay (for example 2257 frames at 44.1 kHz) and drops the last frame. On the
  frames both decode, the two agree to about 130 dB SNR.
- **Vorbis values.** They are stb_vorbis's float output. torchaudio goes through 16-bit `ov_read`, so its values lie on
  the 2^-15 grid, at most one 16-bit step from ours.
- **Clipping of MP3 and Vorbis.** Overshoot beyond full scale is kept. torchaudio clips it; `torch::clamp(x, -1, 1)`
  gives torchaudio's values.
- **Saving.** Only WAV can be saved.

## Audio functional and transforms

Notes: [progress04](../develop_log/2026-10-23/progress04-audio-surface-audit.md) D6,
[progress08](../develop_log/2026-10-23/progress08-audio-parity-followups.md) D3, D7,
[progress06 (2026-10-24)](../develop_log/2026-10-24/progress06-sanitizer-sweep.md) D2.

- **`functional::spectrogram` returns the complex STFT by default.** `spectrogram_option` defaults to
  `return_complex = true`, which ignores `power`. Call `.return_complex(false)` to get torchaudio's power spectrogram
  (`power=2.0`). `transform::Spectrogram` defaults to the power spectrogram, like torchaudio. `amplitude_to_DB` and
  `MelScale` accept a complex input, and `amplitude_to_DB` takes its magnitude; torchaudio raises for both. So a
  forgotten `.return_complex(false)` silently gives half the dB. This is an open item (G1-3, G3-13).
  Notes: [progress02](../develop_log/2026-10-23/progress02-audio-audit-followup.md) Open.
- **Mutating ops.** `dcshift` and `dither` never modify their input.
- **GPDF dither.** It sums seven samples of the unmodified input. torchaudio sums into a view of its input, so it
  mutates the caller's tensor and re-reads the grown sample; seeded outputs therefore differ at a few elements. Mono
  GPDF dither works here.
- **Shorter windows.** `detect_pitch_frequency` accepts `win_length` 1-2; torchaudio fails there.
- **Zero means "derive".** In `TimeStretch`, `hop_length` 0 means derive the default; so does `f_max` 0 in `LFCC`.
  torchaudio's explicit 0 cannot be expressed.
- **Defaults torchaudio does not have.**
  - Arguments that torchaudio requires have defaults here: the masking parameters, `Vol` (gain 1.0), `Vad` and
    `Loudness` (sample_rate 16000).
  - `PitchShift` always uses a Hann window.
  - float64 input works directly in `PitchShift`, `Speed`, `LFCC` and `InverseMelScale`.
- **Stricter `frechet_distance`.** It rejects a mean whose length does not match the covariance; torchaudio's chained
  comparison accepts some of these.
- **Recursive filters on other devices.** float32 / float64 input on MPS or CUDA without autograd is computed on a CPU
  copy, so the result equals torchaudio's CPU values; the output stays on the input's device. This covers
  `lfilter`, `filtfilt`, `biquad` and the EQ designers, `deemphasis`, `overdrive`, `phaser` and `flanger`. torchaudio
  runs its tensor loop on the device, which libtorch 2.5.1's MPS backend gets wrong beyond about 64k samples.
- **0-D waveform in `lfilter`.** It raises one `std::invalid_argument` for every coefficient shape. torchaudio raises
  ValueError or IndexError depending on the coefficient shape.

## Kaldi compliance (`torchmedia::audio::compliance::kaldi`)

Notes: [progress06](../develop_log/2026-10-23/progress06-kaldi-compliance.md) D5.

- **Input shorter than one frame.** All three functions return `empty({0})` in the input's dtype and device.
  torchaudio's `fbank` does the same, but its `spectrogram` returns float32 on the CPU and its `mfcc` raises.
- **Non-contiguous waveform without `snip_edges`.** The frames are read correctly. torchaudio takes the strides from
  the input and reads the wrong samples.
- **Very large frame shift.** A shift above 2n + 1 samples is clamped; the frames are the same. torchaudio raises once
  the shift no longer fits int64.

## Source separation (`torchmedia::audio::separation`)

Notes: [progress03](../develop_log/2026-10-23/progress03-auxiva-bss.md) D5, D7.

- **Diagonal load.** The default is 1e-8; torchiva uses 1e-5. With `auxiva_option().load(1e-5)` the result is
  bit-identical to torchiva wherever torchiva's own row-norm clamp does not fire.
- **Projection back.** torchiva clamps the row norms at 1e-5, which costs up to 4 dB SDR on int16-scale input. Here
  the clamp applies only to all-zero rows.
- **OverIVA with projection back.** It returns the per-frequency result, `[..., n_src, n_freq, n_frames]`. torchiva
  broadcasts it a second time.
- **`n_src`.** `-1` means the number of channels; torchiva's `None` default crashes.

## Vision I/O (`torchmedia::vision::io`)

Notes: [progress09](../develop_log/2026-10-23/progress09-vision-color-geometry.md) D5, D6;
[progress04 (2026-10-24)](../develop_log/2026-10-24/progress04-pnm-readme.md) D3.

- **JPEG.** Decoded by stb_image (with three chroma-upsampling patches), not libjpeg-turbo. Measured over 255 files:
  - up to 3 levels off, or 4 for 4:4:0 chroma subsampling;
  - 94.9 % of values exact, mean |difference| 0.066;
  - gray mode within 1 level.
- **JPEG files that are not decoded.**
  - Arithmetic-coded JPEGs raise.
  - CMYK JPEGs come back as 3 channels in unchanged mode; torchvision returns 4.
  - The EXIF orientation is not applied.
- **PNG.** Bit-exact, with three exceptions:
  - A palette PNG in unchanged mode returns its colours; torchvision returns the palette indices.
  - A palette PNG without transparency in a mode with alpha returns its colours and alpha 255. torchvision returns
    scrambled channels there.
  - The 16-bit sBIT shift and ICC-based sRGB detection are not reproduced.
- **Formats.** GIF and WebP are not decoded.
- **`load_image` on PNM.** It returns float32 sample / maxval at full 16-bit precision, through its own netpbm reader.
- **`load_image` on 16-bit PNG.** stb_image reduces it to its high byte. Use `decode_image` for the full 16 bits.

## Vision functional and transforms (`torchmedia::vision`)

Notes: [progress07](../develop_log/2026-10-23/progress07-vision-geometry-core.md) D3, D7,
[progress09](../develop_log/2026-10-23/progress09-vision-color-geometry.md) D8,
[progress10](../develop_log/2026-10-23/progress10-vision-random-transforms.md) D7,
[progress05 (2026-10-24)](../develop_log/2026-10-24/progress05-silent-mode-contract.md) D5.

- **`resize` with more than two sizes.** It rejects them. torchvision's functional uses the first two.
- **Symmetric pad that crops an edge to zero.** It returns the empty image; torchvision raises.
- **`fill` and `brightness_factor` are doubles.** A Python int brightness factor makes torchvision compute in the
  integer dtype and wrap; here it computes in floating point and clamps.
- **Output memory format.** torchvision's channels-last restride in `resize` is skipped. The values are equal.
- **uint8 bilinear resize.** It follows torchvision's dispatch, which depends on the CPU: on a CPU without AVX2, such
  as arm64, it goes through float32. Results therefore match torchvision on the same machine, not across machines.
- **MPS.**
  - `adjust_hue` copies its table synchronously and equals the CPU result. torchvision 0.20.1 reads freed memory
    there.
  - `adjust_contrast` / `adjust_saturation` on uint8 run; torchvision aborts.
- **Argument validation.**
  - `posterize` rejects `bits` < 0, `bits` > 62 for floating images, and bool images.
  - `rotate` / `affine` reject a center that is neither `{}` nor `{x, y}`.
  - The grid ops need `[..., C, H, W]`.
- **Containers.**
  - An empty `RandomApply` / `RandomOrder` returns the input; torchvision raises.
  - `RandomChoice` with probabilities summing to 0 throws `std::invalid_argument`.
  - In the no-exceptions build, a container stops at the first part that fails.
- **Perspective grid with non-finite values.** It gives the fill value. torchvision crashes (bilinear) or reads an
  arbitrary pixel (nearest).

## Metrics (`torchmedia::metric`)

Notes: [progress11](../develop_log/2026-10-23/progress11-metric-vision.md) D1, D5, D7;
[progress01 (2026-10-24)](../develop_log/2026-10-24/progress01-metric-audio-text.md) D7;
[progress02 (2026-10-24)](../develop_log/2026-10-24/progress02-metric-classification.md) D7.

- **States own their data.** The states that keep inputs store copies (`detach().clone()`) or converted values: exact
  curves, calibration, EER / minDCF, and the detection boxes and labels. Refilling an input buffer after `update`
  therefore does not change earlier updates. torchmetrics keeps references
  ([progress02 (2026-10-24)](../develop_log/2026-10-24/progress02-metric-classification.md) Agent log, CR5).
- **float64 states.** Class states are float64 when the values are float64, so `ShortTimeObjectiveIntelligibility`
  returns float64.
- **Detection.**
  - `MeanAveragePrecision` and VOC AP compute in float64 on the CPU whatever the input device, so their results are
    CPU tensors.
  - With no image, `MeanAveragePrecision` returns -1 stats and `map_per_class` `{-1}`.
  - There is no faster_coco_eval backend and no `extended_summary`.
- **Segmentation.**
  - IoU classes with `class_metrics`: an image with predictions but no targets is skipped in the per-class sums.
    torchmetrics raises.
  - A reset `MeanIoU` infers `num_classes` again from the next batch.
- **STOI.**
  - The extended STOI omits pystoi's random dither (2.2e-16). Inputs with near-constant segments, such as dropouts or
    all-zero signals, get the expected value of pystoi's random result.
  - The resampling follows the arm64 scipy wheel's rounding.
- **`bss_eval_sources` with one source.** SIR is inf. mir_eval gives inf or 277-298 dB of rounding noise.
- **Classification.**
  - Classes always validate their inputs. torchmetrics' `BinaryAUROC` / `BinaryLogAUC` classes do not.
  - Multilabel binned curves with `ignore_index` work for non-int64 targets.
  - Micro-averaged multiclass PR / ROC / EER classes keep a `[T, 2, 2]` state where torchmetrics raises.
  - A samplewise `ExactMatch` `forward` accumulates like `update`.
  - Threshold tensors move to the input's device.
  - Not ported: `validate_args`, `plot()`, the `task=` wrappers, the fairness metrics.

## Data (`torchmedia::data`)

Notes: [progress03 (2026-10-24)](../develop_log/2026-10-24/progress03-data-layer.md) D2, D3, D6.

- **Paths and listing.**
  - The root is not `~`-expanded.
  - Extension matching lower-cases ASCII only.
  - An I/O error in the middle of a directory listing throws; `os.walk` skips that directory.
- **ImageFolder.**
  - A 16-bit gray PNG is scaled to 8 bits (>> 8). PIL's `convert("RGB")` clips it at 255.
  - With `load_image(true)`, PNM files are read as sample / maxval; PIL rescales or clips when maxval != 255.
- **`random_split`.** A negative or non-integer length is `std::invalid_argument`.
