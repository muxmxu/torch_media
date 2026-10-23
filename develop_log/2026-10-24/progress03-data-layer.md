# Progress 03 — Data layer: datasets over libtorch's torch::data
id: 2026-10-24/progress03
date: 2026-10-24
author: ai
status: active            # implemented and tested; awaiting Mux's review
refs: [2026-10-23/progress09, 2026-10-23/progress10, 2026-10-23/progress05]
supersedes:
commits: []
files:
  - libtorchmedia/include/torchmedia/_data/_dataset.hpp (NEW)
  - libtorchmedia/include/torchmedia/_data/_folder.hpp (NEW)
  - libtorchmedia/include/torchmedia/_data/_manifest.hpp (NEW)
  - libtorchmedia/include/torchmedia/_data/_collate.hpp (NEW)
  - libtorchmedia/include/torchmedia/data.hpp (NEW), libtorchmedia/include/torchmedia.hpp
  - unit_test/data/{CMakeLists.txt, gen_golden.py, golden.hpp, test_dataset.cpp, test_folder.cpp, test_noexcept.cpp}
    (NEW), unit_test/CMakeLists.txt
  - CLAUDE.md

## Goal
A thin dataset layer, `torchmedia::data`, on top of libtorch's own `torch::data` (Dataset / MapDataset, transforms incl.
Stack, samplers, `make_data_loader` with worker threads): torchvision's DatasetFolder / ImageFolder, an AudioFolder, a
manifest-file dataset, variable-length collation, and torch.utils.data's Subset / ConcatDataset / random_split. The
loader machinery itself is not reimplemented.

## Context / Motivation
Mux asked whether a Dataset layer is possible. libtorch already ships the loader (threads, samplers, batching); what is
missing for media work is the dataset side: directory-tree datasets that decode with the library's own image / audio
I/O, a list-file dataset for speech-style manifests, padding collation for variable-length audio, and the split
utilities. The design below was set by the orchestrator under Mux's delegation.

## Decisions

### D1 — Layout: every dataset is a torch::data Dataset
- Decision: namespace `torchmedia::data`, headers `_data/{_dataset, _folder, _manifest, _collate}.hpp` aggregated by
  `torchmedia/data.hpp` (included by `torchmedia.hpp`). Every dataset derives from
  `torch::data::datasets::Dataset<Self, torch::data::Example<tensor_t, tensor_t>>` (`data::example_t`); Subset /
  ConcatDataset keep the wrapped dataset's ExampleType. So `torch::data::make_data_loader`, its samplers, `.map(...)`
  and `torch::data::transforms::Stack<>()` work unchanged.
- Why: reuse libtorch's loader rather than build a second one; one Example type keeps Stack usable.
- Impact: an example is `{sample tensor, int64 0-dim target}`; `transform_fn` / `loader_fn` / `valid_file_fn` are
  `std::function`s, so the vision / audio transform classes (const call operators) plug in directly.

### D2 — DatasetFolder = torchvision 0.20.1, ordering included
- `find_classes(dir)`: names of the entries that are directories, symlinks followed (`DirEntry.is_dir()`), sorted by
  code point (UTF-8 byte order; hidden directories count, broken symlinks do not); none -> std::runtime_error
  "Couldn't find any class folder in {dir}." (FileNotFoundError), an unreadable dir -> std::runtime_error, and an
  entry that cannot be stat'ed (a self-referential symlink: ELOOP; EACCES) -> std::runtime_error "find_classes: cannot
  stat {entry}: ..." (Python's `DirEntry.is_dir()` swallows only FileNotFoundError, and find_classes, unlike os.walk,
  does not catch OSError).
- `make_dataset(dir, class_to_idx, extensions, is_valid_file, allow_empty)`: classes in sorted key order; for each,
  `sorted(os.walk(dir / cls, followlinks=True))`, which sorts the walk by the **directory path string** (so `sub-x` and
  `sub.y` come before `sub/deeper`), then sorted file names. A non-directory entry is a file name (also a broken symlink
  or a symlinked file); a directory that cannot be listed is skipped (os.walk without onerror) — the same rule ends a
  symlink loop where Python's walk ends (ELOOP / ENAMETOOLONG). `has_file_allowed_extension`: the whole path,
  lower-cased, `endswith` one of the extensions as given (a dot-less "png" matches "xpng"). Errors with torchvision's
  messages: empty class_to_idx and "Both extensions and is_valid_file cannot be None or not None at the same time"
  (ValueError -> std::invalid_argument); "Found no valid file for the classes ... Supported extensions are: ..."
  unless allow_empty (FileNotFoundError -> std::runtime_error; a class name in class_to_idx without a directory counts
  as empty, as in torchvision).
- `DatasetFolder(root, loader, dataset_folder_option)`; options `extensions` / `is_valid_file` (exactly one),
  `transform`, `target_transform`, `allow_empty`. Accessors `classes()`, `class_to_idx()` (std::map), `samples()`
  ((path, index) pairs), `targets()`. `get(i)`: `loader(path)`, then transform / target_transform;
  out of range -> std::out_of_range (IndexError).
- Deviations: root is not `~`-expanded; lower-casing is ASCII-only (Python's str.lower is Unicode-aware); an I/O error
  in the middle of a directory listing throws (os.walk skips that directory; `shortcut:` comment).

### D3 — ImageFolder / AudioFolder and their loaders
- `image_loader(image_folder_option)`: default `vision::io::read_file` + `decode_image(bytes, mode)` with mode rgb
  (ImageFolder's PIL loader converts to RGB) -> uint8 [3, H, W] for every file: a 16-bit PNG is scaled to 8 bits by
  `vision::functional::to_dtype(x, kUInt8, scale=true)` (torchvision v2 to_dtype: >> 8), so folders mixing 8- and
  16-bit PNGs batch. Mode `unchanged` returns decode_image's output as is (file channels, uint16 kept). A decode error
  is re-thrown as std::runtime_error "{path}: {decode_image's message}" (decode_image's messages do not name the file;
  PIL's do), so a bad file is found from `get()` and from a loader worker's WorkerException. `load_image(true)`:
  `vision::io::load_image` (stb_image, more formats; PNM as sample / maxval) -> float32 [3, H, W] in [0, 1], gray
  repeated and alpha dropped like convert("RGB"); `mode` is not used. `image_extensions(opt)`: torchvision's
  IMG_EXTENSIONS that the loader decodes, in its order — {.jpg, .jpeg, .png} for decode_image; {.jpg, .jpeg, .png,
  .ppm, .bmp, .pgm} for load_image (no .tif / .tiff / .webp decoder; .ppm / .pgm since progress04).
- Deviations: a 16-bit **gray** PNG is scaled (PIL opens it as I;16 and convert("RGB") clips every value > 255 to
  255; for 16-bit RGB / RGBA PIL keeps the high byte, the same as >> 8). Until progress04 .ppm / .pgm were not
  defaults (stb_image v2.30 read 16-bit PNM samples without swapping their big-endian bytes and had no ASCII PNM);
  load_image now reads PNM itself (2026-10-24/progress04 D1) as sample / maxval at full 16-bit precision. PIL differs
  when maxval != 255: RGB PNM and gray PNM with maxval < 256 are rescaled to 8 bits; gray PNM with maxval > 255 opens
  as I (rescaled to 0..65535) and convert("RGB") clips every value > 255 to 255, as for 16-bit gray PNG (P2 maxval
  1000, samples 0 / 50 / 100: torchvision [0, 255, 255], here 0 / 0.05 / 0.1). stb reads no RLE or 1-bit BMP (the
  error names the file).
- `audio_loader(audio_folder_option)`: `audio::io::load_audio` (float32 [C, T]), then optional `mono` (mean over
  channels, before resampling: cheaper, and resampling is linear), `sample_rate` (> 0: `audio::functional::resample`
  with its defaults; equal rates return the input), `num_frames` (> 0: the first num_frames, or zero-padded at the end;
  deterministic — a random offset belongs in a transform). Without resampling, `num_frames` is passed to load_audio's
  `load_option().num_frames(n)`, so only those frames are decoded (10-minute WAV: 40.2 ms -> 0.3 ms per `get()`, equal
  output); with resampling the whole file is decoded first (`shortcut:` comment). Negative values ->
  std::invalid_argument. `audio_extensions()`: {.wav, .flac, .mp3, .ogg} (the containers load_audio sniffs).
- `ImageFolder(root, folder_opt, image_opt)` / `AudioFolder(root, folder_opt, audio_opt)` derive from DatasetFolder and
  use the default extensions unless `extensions` or `is_valid_file` is given. The loaders are public so a
  ManifestDataset decodes the same way.

### D4 — ManifestDataset
- `ManifestDataset(manifest, loader, manifest_option)`. Default format: Kaldi-style `key path [label]` per non-blank
  line, fields split on runs of spaces / tabs (paths without spaces; a plain TSV reads the same). `header(true)`: a TSV
  whose first non-blank line names the columns, split on every tab (paths may hold spaces), columns picked by
  `path_column` (default "path") and `label_column` (default "label"; "" = unlabeled); every row must have the header's
  field count. Labels are int64 (`std::from_chars`, whole field) unless `class_labels(true)`: then they are class names
  indexed in sorted order (`classes()` / `class_to_idx()`, as DatasetFolder). A missing / empty label is target -1.
  Relative paths resolve against the manifest's directory; CRLF is accepted. `keys()`: the Kaldi key (with a header,
  the path as written).
- Errors: a missing manifest or a directory (std::ifstream opens one as an empty stream) -> std::runtime_error
  "ManifestDataset: cannot open ..."; a malformed line -> std::invalid_argument "ManifestDataset: {manifest}:{line}:
  ..." (field count, non-integer label, missing header column, an empty TSV path field — it would resolve to the
  manifest's directory and fail later in `get()` without the line).
- Why explicit header / class_labels switches instead of sniffing: a tab inside a Kaldi line, or numeric class names,
  would make auto-detection silently wrong.

### D5 — PadCollate (Stack reused)
- `PadCollate(pad_collate_option)` is a `torch::data::transforms::BatchTransform<std::vector<example_t>, padded_batch>`:
  `dataset.map(PadCollate())` makes the loader yield `padded_batch {data [B, ..., Lmax], target (stacked), lengths [B]
  int64}`; the last dim is padded with `value` (default 0) to the batch maximum rounded up to `multiple` (default 1).
  Samples must share rank (>= 1), leading dims and dtype (std::invalid_argument naming the sample); an empty batch or
  multiple < 1 is std::invalid_argument. Equal-shape samples keep using `torch::data::transforms::Stack<>()`.

### D6 — Subset / ConcatDataset / random_split = torch.utils.data
- `Subset<D>(dataset, indices)`: indices are checked against the dataset's size at construction (Python fails on
  access); `dataset()`, `indices()`. `ConcatDataset<D>(std::vector<D>)`: cumulative sizes, `bisect_right` lookup (empty
  datasets skipped), empty list -> std::invalid_argument; one dataset type (wrap different ones in a common type).
- `random_split(dataset, lengths, generator = global | seed)` returns `std::vector<Subset<D>>`. Fractions when the
  lengths sum to 1 (`math.isclose` and <= 1) with **Python 3.12's compensated `sum()`** (Neumaier; ~1.8% of 2-decimal
  fraction lists sum to 1.0000000000000002 left to right but 1.0 in Python); floor(n * f) each, the remainder added
  round-robin; then `torch::randperm(n, generator)` sliced in order. Integer lengths summing to 1 are fractions, as in
  Python. `random_split(ds, lengths, seed)` uses `at::make_generator<at::CPUGeneratorImpl>(seed)` =
  `torch.Generator().manual_seed(seed)`; without a generator the global one is used (`torch::manual_seed` =
  `torch.manual_seed`).
- `lengths` is one `std::vector<double>`: a braced list binds for both `{3, 7}` and `{0.8, 0.2}` (int64 / double
  overloads would be ambiguous for braced lists); a `std::vector<int64_t>` converts with `{v.begin(), v.end()}`.
- Deviations: a negative or non-integer length is std::invalid_argument (Python slices with negatives and raises
  TypeError in randperm for floats); the zero-length-split warning is not printed.
- A torch::data dataset whose `size()` is nullopt is std::invalid_argument in Subset ("Subset: the dataset has no
  size"), random_split and ConcatDataset ("... the dataset at index {i} has no size"); none of them can index it.

### D7 — Thread safety and sharing
- `StatelessDataLoader` copies the dataset into every worker thread. DatasetFolder / ManifestDataset keep their sample
  list in a `shared_ptr<const sample_list>`, so a copy is a reference-count bump and the shared state is read-only
  after construction; the loader / transform `std::function`s are copied per worker. Datasets are therefore safe for
  `workers(n)`.
- A transform that draws from the global generator (the random vision transforms, progress10) interleaves its draws
  across workers: seeded parity needs `workers(0)` (documented in `_dataset.hpp`).

### D8 — Errors
- `handle_exceptions` everywhere (`detail::invalid<T, E>`): std::invalid_argument for invalid arguments,
  std::runtime_error for filesystem conditions (torchvision's FileNotFoundError), std::out_of_range for `get()` past
  the end. With `LIB_TORCH_MEDIA_NO_EXCEPTIONS` the default value comes back and nothing is computed from the invalid
  input: empty dataset / split / index list, undefined example / batch, PadCollate's multiple falls back to 1
  (`data_test_noexcept`).

### D9 — Tests, goldens, coverage
- `gen_golden.py` (torchvision 0.20.1, torch 2.5.1) builds TREE in a temp directory — upper / lower case classes, a
  hidden class, a non-ASCII class, nested dirs ordered by path string (`sub-x`, `sub.y`, `sub/deeper`), case-insensitive
  and dot-less extensions, a directory named `dir.png`, an unlistable (chmod 000) directory, symlinked class dir,
  symlinked sub-directory, symlinked file, broken file symlink with an image extension, broken class symlink, empty
  classes, a file at the root — and records find_classes, DatasetFolder (three extension / is_valid_file setups),
  ImageFolder, make_dataset with a custom non-contiguous class_to_idx, and every error message; plus random_split
  splits for seeds 0 / 1 / 42 / 2024 over nine (n, lengths) cases with explicit generators and four with the global
  generator, and its two error messages. `golden.hpp` carries TREE, which the C++ test rebuilds (image-named files as
  real PNGs written by `save_image`).
- ctest `data_test_folder` (222 checks; 224 since progress04): the goldens equal (order, targets, class_to_idx, messages); find_classes
  raises for a self-referential root symlink and skips a broken one; ImageFolder decodes RGB / gray / RGBA PNGs to the
  saved values in rgb / unchanged / float modes (float: RGB), the broken symlink fails on `get()`; a folder mixing an
  8-bit RGB PNG, a 16-bit gray PNG (embedded bytes), a 16-bit binary .pgm and an ASCII .ppm (progress04): the default
  loader lists the PNGs and gives uint8 [3, 2, 2] (16-bit = >> 8) that a Stack loader batches, uint16 in mode
  unchanged; load_image(true) gives the same values as floats plus both PNM files (sample / maxval), stacked; a corrupt .png is named in `get()`'s error and in a 2-worker loader's WorkerException, a corrupt .wav
  in AudioFolder's; AudioFolder over float32 WAVs from `save_audio` (exact), resampled (= `functional::resample`),
  mono, crop / pad (partial read), all three (full decode); ManifestDataset Kaldi / class-label / header formats and
  every error (incl. an empty TSV path and a directory as the manifest); transforms (a vision ToDtype class) and
  target transforms; DataLoader: Stack batches 2 + 1, drop_last, PadCollate batches (values, pad value, lengths,
  targets), and 2 workers covering every sample once with and without drop_last.
  `data_test_dataset` (1688 checks): random_split goldens (indices and `get()` through them), the seed / generator /
  global forms, error messages and deviations, Subset / ConcatDataset (empty member skipped), an unsized dataset
  rejected by Subset / random_split / ConcatDataset, and a 2-worker loader over ConcatDataset(random_split(...));
  PadCollate closed forms ([L], [C, L], int dtype, value, multiple) and every validation message.
  `data_test_noexcept` (10 checks): D8.
- A mutation check: sorting the walk by `std::filesystem::path` (element-wise compare) instead of the string, and a
  naive left-to-right sum in random_split, each fail the tests.
- Coverage (Debug + `TORCHMEDIA_COVERAGE`, scratch `data/build-cov`): `_collate.hpp` 56/56 lines (data_test_dataset),
  `_folder.hpp` 259/259 and `_manifest.hpp` 127/127 (data_test_folder), `_dataset.hpp` 114/114 as the per-line union
  over both binaries (progress11 D9's policy for a shared header; lcov `DA` rows, which merge template
  instantiations: data_test_dataset leaves only `detail::join`, lines 42-48, which data_test_folder runs).
  `llvm-cov report` on data_test_dataset alone lists 12 missed lines: those 7 plus lines that only some
  instantiations skip (e.g. `random_split<unsized>` after its early return), each run by another instantiation.

## Tasks
- [x] [task20 — DatasetFolder, ImageFolder, AudioFolder](tasks/task20-data-folder.md)
- [x] [task21 — ManifestDataset, PadCollate, split utilities](tasks/task21-data-manifest-collate-split.md)
- [x] [task22 — goldens, tests, coverage](tasks/task22-data-goldens-tests.md)

## Issues / Gotchas
- `sorted(os.walk(...))` sorts by the walked directory's path string, not a depth-first order; `std::filesystem::path`'s
  `operator<` compares element-wise and puts `sub/deeper` before `sub-x` — compare the strings.
- Python 3.12 changed `sum()` of floats to Neumaier summation; random_split's fraction test depends on it (D6).
- `ImageFolder` / `AudioFolder` derive from DatasetFolder, whose CRTP `Self` is DatasetFolder: `.map(...)` on them
  returns `MapDataset<DatasetFolder, T>`. Harmless (all state lives in the base), but the static type changes.
- The derived constructors pass the loader by copy to both the list builder and the base: with `std::move` in the
  same call the unspecified argument order could hand the builder an empty function.
- The unlistable-directory branch of the walk only runs as a non-root user (root ignores chmod 000); the directory is
  kept empty so the goldens hold either way.
- libc++ `directory_entry::is_directory(ec)` sets `ec` for a broken symlink too (ENOENT) and returns false; only
  other codes (ELOOP, EACCES) are errors for find_classes.
- Review round (parity + robustness reviewers), all fixed with a test that failed before the fix (checked against
  the unfixed headers; P1 / P2 / P4 fixed first so the R1 checks after them ran):
  - P1: the default loader returned uint16 for a 16-bit PNG next to uint8 samples (Stack: "Promotion for uint16 ...
    not supported") -> scaled to uint8 except in mode unchanged (D3).
  - P2: load_image(true) listed .ppm / .pgm; 16-bit PNM decoded to the low bytes (1000 -> 232), ASCII PNM threw ->
    dropped from the defaults (D3); the stb bug itself is in Open. Fixed in 2026-10-24/progress04 (load_image reads
    PNM itself); .ppm / .pgm are load_image defaults again.
  - P3: find_classes skipped a root entry it could not stat; torchvision raises OSError -> std::runtime_error (D2).
  - P4: load_image(true) kept the file's channel count ([1, H, W] next to [3, H, W]) -> RGB like ImageFolder (D3).
  - R1: decode errors from ImageFolder's default loader did not name the file -> "{path}: ..." (D3).
  - R2: a directory as the manifest gave an empty dataset -> "cannot open" (D4).
  - R3: an empty TSV path field became the manifest's directory -> "{manifest}:{line}: empty path" (D4).
  - R4: Subset / random_split dereferenced an empty `size()` (UB: garbage size, empty split, runaway randperm);
    ConcatDataset threw bad_optional_access -> std::invalid_argument in all three (D6).
  - R5: AudioFolder's crop decoded the whole file although load_audio reads a prefix -> partial read without
    resampling (D3). Performance only: the crop / pad equality tests cover the new path; no test fails before it.

## Open / TODO (carry-over)
- ConcatDataset over different dataset types (Python allows it), a random-offset crop option in AudioFolder, and
  decoding only the needed frames when cropping with resampling (the loader then decodes the whole file; `shortcut:`
  comment).
- ~~`vision::io::load_image` returns the low byte of 16-bit PNM samples and cannot read ASCII PNM~~ — resolved by
  2026-10-24/progress04 (a netpbm reader in `_vision/_io.hpp`; .ppm / .pgm back in the load_image defaults).
- The example carries no sample rate: set `sample_rate` for trees with mixed rates.
- ManifestDataset: no quoted TSV fields, no Kaldi pipes (`cmd |`) or ark offsets.
- Python DataLoader features absent from torch::data (pin_memory, persistent workers, worker_init_fn / per-worker
  seeding) are not added; workers are threads, not processes.
- Parity measured on macOS arm64 (APFS) only; the tests use POSIX `getpid` / symlinks / permissions.

## Agent log
- 2026-10-24 [ai] Read torch::data (datasets/base.h, map.h, transforms, dataloader*.h), torchvision 0.20.1
  datasets/folder.py and torch 2.5.1 utils/data/dataset.py. Implemented D1-D8 (4 headers + data.hpp). Goldens from
  gen_golden.py; ctest data_test_folder 204, data_test_dataset 1685, data_test_noexcept 10 checks; coverage 100% (D9);
  mutation check (D9). Full Release build + ctest 21 / 21 green; clang-format clean on the new C++ files (golden.hpp
  left as generated, like the other goldens). Not committed.
- 2026-10-24 [ai] Review fixes P1-P4, R1-R5 (Issues): tests first, run against the unfixed headers (data_test_folder
  206/213 then 217/222 after the image-loader stage; data_test_dataset 1684/1687), then the fixes. data_test_folder
  222, data_test_dataset 1688, data_test_noexcept 10 checks; data_test_folder 10 / 10 repeated runs green. Coverage
  `_folder.hpp` 259/259, `_manifest.hpp` 127/127, `_collate.hpp` 56/56, `_dataset.hpp` 114/114 (union). Reviewer probe
  re-run: AudioFolder num_frames(16000) on a 10-minute WAV 0.3 ms per get() (was 40.2 ms), equal output; a manifest
  directory now throws. Full Release build + ctest 21 / 21 green; clang-format clean. Not committed.
- 2026-10-24 [ai] progress04: load_image reads PNM P1-P6 itself (sample / maxval), so `image_extensions(load_image)`
  lists .ppm / .pgm again (torchvision's order); test_image_mixed_and_corrupt now mixes a 16-bit .pgm and an ASCII .ppm
  into the float folder (red before the fix: 216/219). data_test_folder 224 checks; `_folder.hpp` 259/259 lines.
