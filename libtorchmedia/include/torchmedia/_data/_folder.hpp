#pragma once
#ifndef LIB_TORCH_MEDIA_DATA_FOLDER_HPP
#define LIB_TORCH_MEDIA_DATA_FOLDER_HPP
// torchvision.datasets.DatasetFolder / ImageFolder (torchvision 0.20.1) and AudioFolder over the library's decoders
// (develop_log/2026-10-24/progress03). An example is {loaded sample, int64 0-dim class index}; the transforms run in
// get(), so they run in the loader's worker threads.
#include <cctype>
#include <filesystem>
#include <map>
#include <memory>
#include "../_audio/_functional.hpp"
#include "../_audio/_io.hpp"
#include "../_vision/_functional.hpp"
#include "../_vision/_io.hpp"
#include "_dataset.hpp"

namespace torchmedia::data {
    // Loads one sample from its path.
    using loader_fn = std::function<tensor_t(const path_t &)>;
    // torchvision's is_valid_file: called with each candidate path (root joined with the walked names).
    using valid_file_fn = std::function<bool(const path_t &)>;
    using class_to_idx_t = std::map<std::string, int64_t>;
    using sample_t = std::pair<path_t, int64_t>;

    namespace detail {
        // The read-only state a path dataset shares between its copies (one per loader worker).
        struct sample_list {
            std::vector<std::string> classes;
            class_to_idx_t class_to_idx;
            std::vector<sample_t> samples;
            std::vector<int64_t> targets;
            std::vector<std::string> keys; // ManifestDataset only
        };

        // A dataset of (path, class index) samples: get() loads the path and applies the transforms.
        template<class Self>
        class path_dataset : public torch::data::datasets::Dataset<Self, example_t> {
        public:
            auto get(size_t index) -> example_t override {
                if (index >= _list->samples.size()) {
                    return invalid<example_t, std::out_of_range>(
                            std::format("dataset index {} is out of range for size {}", index, _list->samples.size()));
                }
                const auto &[path, target] = _list->samples[index];
                auto data = _loader(path);
                auto label = torch::scalar_tensor(target, torch::kInt64);
                return {_transform ? _transform(data) : data, _target_transform ? _target_transform(label) : label};
            }
            auto size() const -> std::optional<size_t> override { return _list->samples.size(); }

            // Class names, sorted (torchvision's classes) and their indices.
            auto classes() const -> const std::vector<std::string> & { return _list->classes; }
            auto class_to_idx() const -> const class_to_idx_t & { return _list->class_to_idx; }
            // (path, class index) per sample, in dataset order, and the class indices alone.
            auto samples() const -> const std::vector<sample_t> & { return _list->samples; }
            auto targets() const -> const std::vector<int64_t> & { return _list->targets; }

        protected:
            path_dataset(sample_list list, loader_fn loader, transform_fn transform, transform_fn target_transform) :
                _list(std::make_shared<const sample_list>(std::move(list))), _loader(std::move(loader)),
                _transform(std::move(transform)), _target_transform(std::move(target_transform)) {}

            std::shared_ptr<const sample_list> _list;
            loader_fn _loader;
            transform_fn _transform;
            transform_fn _target_transform;
        };

        // os.walk(top, followlinks=True): (directory, non-directory names) for top and every directory below it.
        // A directory that cannot be listed is skipped, as os.walk does without onerror; that also ends a symlink
        // loop (ELOOP / ENAMETOOLONG) at the depth where Python's walk ends.
        inline auto walk(const path_t &top, std::vector<std::pair<std::string, std::vector<std::string>>> &out)
                -> void {
            std::error_code ec;
            std::filesystem::directory_iterator it(top, ec);
            if (ec) {
                return;
            }
            std::vector<std::string> files;
            std::vector<path_t> dirs;
            for (const auto &entry: it) { // shortcut: an I/O error in mid-listing throws (os.walk skips the directory)
                std::error_code ignored; // is_directory follows symlinks; a broken one is a file name, as in os.walk
                entry.is_directory(ignored) ? dirs.push_back(entry.path())
                                            : files.push_back(entry.path().filename().string());
            }
            out.emplace_back(top.string(), std::move(files));
            for (const auto &d: dirs) {
                walk(d, out);
            }
        }
    } // namespace detail

    // torchvision has_file_allowed_extension: the path, lower-cased (ASCII only), ends with one of `extensions`.
    inline auto has_file_allowed_extension(const path_t &file, const std::vector<std::string> &extensions) -> bool {
        auto name = file.string();
        std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return std::ranges::any_of(extensions, [&](const std::string &e) { return name.ends_with(e); });
    }

    // torchvision find_classes: the names of the directories (symlinks followed) directly under `directory`, sorted by
    // code point, and their indices. Throws std::runtime_error (FileNotFoundError / OSError) for none, an unreadable
    // directory, or an entry that cannot be stat'ed (ELOOP, EACCES; a broken symlink is skipped, as in Python).
    inline auto find_classes(const path_t &directory) -> std::pair<std::vector<std::string>, class_to_idx_t> {
        using R = std::pair<std::vector<std::string>, class_to_idx_t>;
        std::error_code ec;
        std::filesystem::directory_iterator it(directory, ec);
        if (ec) {
            return detail::invalid<R, std::runtime_error>(
                    std::format("find_classes: cannot read {}: {}", directory.string(), ec.message()));
        }
        R out;
        for (const auto &entry: it) {
            const bool is_dir = entry.is_directory(ec);
            // DirEntry.is_dir() swallows FileNotFoundError only; unlike os.walk, find_classes does not catch OSError
            if (ec && ec != std::errc::no_such_file_or_directory) {
                return detail::invalid<R, std::runtime_error>(
                        std::format("find_classes: cannot stat {}: {}", entry.path().string(), ec.message()));
            }
            if (is_dir) {
                out.first.push_back(entry.path().filename().string());
            }
        }
        if (out.first.empty()) {
            return detail::invalid<R, std::runtime_error>(
                    std::format("Couldn't find any class folder in {}.", directory.string()));
        }
        std::ranges::sort(out.first);
        for (size_t i = 0; i < out.first.size(); ++i) {
            out.second.emplace(out.first[i], static_cast<int64_t>(i));
        }
        return out;
    }

    // torchvision make_dataset: for each class (sorted), the files under directory / class in sorted(os.walk(...,
    // followlinks=True)) order — directories by their full path string, so "a-b" comes before "a/x" — and sorted names
    // within one, kept when is_valid_file (or has_file_allowed_extension with `extensions`) accepts the path. Pass
    // exactly one of extensions / is_valid_file (std::invalid_argument otherwise, or for an empty class_to_idx). A
    // class without a valid file throws std::runtime_error (FileNotFoundError) unless allow_empty.
    inline auto make_dataset(const path_t &directory, const class_to_idx_t &class_to_idx,
                             const std::optional<std::vector<std::string>> &extensions,
                             const valid_file_fn &is_valid_file = {}, bool allow_empty = false)
            -> std::vector<sample_t> {
        using R = std::vector<sample_t>;
        if (class_to_idx.empty()) {
            return detail::invalid<R>("'class_to_index' must have at least one entry to collect any samples.");
        }
        if (extensions.has_value() == static_cast<bool>(is_valid_file)) {
            return detail::invalid<R>("Both extensions and is_valid_file cannot be None or not None at the same time");
        }
        R instances;
        std::vector<std::string> empty_classes;
        for (const auto &[target_class, class_index]: class_to_idx) {
            const auto before = instances.size();
            const auto target_dir = directory / target_class;
            std::error_code ec;
            std::vector<std::pair<std::string, std::vector<std::string>>> walked;
            if (std::filesystem::is_directory(target_dir, ec)) {
                detail::walk(target_dir, walked);
            }
            std::ranges::sort(walked, {}, &std::pair<std::string, std::vector<std::string>>::first);
            for (auto &[root, names]: walked) {
                std::ranges::sort(names);
                for (const auto &name: names) {
                    auto path = path_t(root) / name;
                    if (extensions ? has_file_allowed_extension(path, *extensions) : is_valid_file(path)) {
                        instances.emplace_back(std::move(path), class_index);
                    }
                }
            }
            if (instances.size() == before) {
                empty_classes.push_back(target_class);
            }
        }
        if (!empty_classes.empty() && !allow_empty) {
            auto msg = std::format("Found no valid file for the classes {}. ", detail::join(empty_classes));
            if (extensions) {
                msg += std::format("Supported extensions are: {}", detail::join(*extensions));
            }
            return detail::invalid<R, std::runtime_error>(msg);
        }
        return instances;
    }

    // DatasetFolder's keyword arguments: exactly one of extensions / is_valid_file, the transforms and allow_empty.
    typedef struct dataset_folder_option {
        std::optional<std::vector<std::string>> _extensions;
        valid_file_fn _is_valid_file;
        transform_fn _transform;
        transform_fn _target_transform;
        bool _allow_empty = false;

        auto extensions(std::vector<std::string> v) -> dataset_folder_option & {
            _extensions = std::move(v);
            return *this;
        }
        auto is_valid_file(valid_file_fn v) -> dataset_folder_option & {
            _is_valid_file = std::move(v);
            return *this;
        }
        auto transform(transform_fn v) -> dataset_folder_option & {
            _transform = std::move(v);
            return *this;
        }
        auto target_transform(transform_fn v) -> dataset_folder_option & {
            _target_transform = std::move(v);
            return *this;
        }
        auto allow_empty(bool v) -> dataset_folder_option & {
            _allow_empty = v;
            return *this;
        }
    } dataset_folder_option_t;

    namespace detail {
        inline auto folder_list(const path_t &root, const loader_fn &loader, const dataset_folder_option &opt)
                -> sample_list {
            if (!loader) {
                return invalid<sample_list>("DatasetFolder: loader must be callable");
            }
            sample_list list;
            std::tie(list.classes, list.class_to_idx) = find_classes(root);
            list.samples = make_dataset(root, list.class_to_idx, opt._extensions, opt._is_valid_file, opt._allow_empty);
            for (const auto &s: list.samples) {
                list.targets.push_back(s.second);
            }
            return list;
        }
    } // namespace detail

    // torchvision.datasets.DatasetFolder: root/<class>/**/<file> samples (find_classes + make_dataset above), loaded
    // by `loader`. Deviation: `root` is not ~-expanded.
    class DatasetFolder : public detail::path_dataset<DatasetFolder> {
    public:
        DatasetFolder(const path_t &root, const loader_fn &loader, const dataset_folder_option &opt) :
            path_dataset(detail::folder_list(root, loader, opt), loader, opt._transform, opt._target_transform) {}
    };

    // ---------------- ImageFolder ----------------
    // The image loader: torchvision.io.decode_image in `mode` (rgb, like ImageFolder's PIL loader's convert("RGB")) as
    // uint8 [C, H, W]; a 16-bit PNG is scaled to 8 bits as v2.functional.to_dtype(scale=True) does (PIL takes the high
    // byte of 16-bit RGB too, but clips 16-bit gray), except in mode unchanged, which is decode_image's output as is
    // (uint16 kept). With load_image(true): vision::io::load_image (stb_image, more formats) as float32 [3, H, W] in
    // [0, 1], gray repeated and alpha dropped like convert("RGB"); `mode` is not used.
    typedef struct image_folder_option {
        vision::io::image_read_mode _mode = vision::io::image_read_mode::rgb;
        bool _load_image = false;

        auto mode(vision::io::image_read_mode v) -> image_folder_option & {
            _mode = v;
            return *this;
        }
        auto load_image(bool v) -> image_folder_option & {
            _load_image = v;
            return *this;
        }
    } image_folder_option_t;

    // torchvision's IMG_EXTENSIONS that the loader decodes: decode_image reads JPEG / PNG, stb_image also BMP (not RLE
    // / 1-bit). Not .ppm / .pgm: stb_image reads 16-bit PNM with swapped bytes (silently wrong) and no ASCII PNM; pass
    // them in `extensions` for 8-bit binary PNM files.
    inline auto image_extensions(const image_folder_option &opt = {}) -> std::vector<std::string> {
        if (opt._load_image) {
            return {".jpg", ".jpeg", ".png", ".bmp"};
        }
        return {".jpg", ".jpeg", ".png"};
    }

    inline auto image_loader(const image_folder_option &opt = {}) -> loader_fn {
        if (opt._load_image) {
            return [](const path_t &p) {
                const auto x = vision::io::load_image(p);
                return (x.size(0) < 3 ? x.narrow(0, 0, 1).expand({3, -1, -1}) : x.narrow(0, 0, 3)).contiguous();
            };
        }
        return [mode = opt._mode](const path_t &p) {
            const auto bytes = vision::io::read_file(p); // its error names the file
            try {
                const auto x = vision::io::decode_image(bytes, mode);
                return mode == vision::io::image_read_mode::unchanged
                               ? x
                               : vision::functional::to_dtype(x, torch::kUInt8, true);
            } catch (const std::exception &e) { // decode_image's errors do not name the file (PIL's do)
                throw std::runtime_error(p.string() + ": " + e.what());
            }
        };
    }

    namespace detail {
        // ImageFolder / AudioFolder: the default extensions unless extensions or is_valid_file is given.
        inline auto with_extensions(dataset_folder_option opt, std::vector<std::string> extensions)
                -> dataset_folder_option {
            if (!opt._extensions && !opt._is_valid_file) {
                opt._extensions = std::move(extensions);
            }
            return opt;
        }
    } // namespace detail

    // torchvision.datasets.ImageFolder: DatasetFolder with image_loader(img) and image_extensions(img) by default.
    class ImageFolder : public DatasetFolder {
    public:
        explicit ImageFolder(const path_t &root, const dataset_folder_option &opt = {},
                             const image_folder_option &img = {}) :
            DatasetFolder(root, image_loader(img), detail::with_extensions(opt, image_extensions(img))) {}
    };

    // ---------------- AudioFolder ----------------
    // The audio loader: audio::io::load_audio -> float32 [C, T]; then, in this order, an optional mono mix-down (mean
    // over channels -> [1, T]), resampling to sample_rate (audio::functional::resample, sinc_interp_hann defaults; 0
    // keeps each file's native rate) and a fixed length of num_frames (0: as decoded): the first num_frames frames, or
    // the clip zero-padded at the end. The crop is deterministic; draw a random offset in a transform if needed.
    typedef struct audio_folder_option {
        int _sample_rate = 0;
        bool _mono = false;
        int64_t _num_frames = 0;

        auto sample_rate(int v) -> audio_folder_option & {
            _sample_rate = v;
            return *this;
        }
        auto mono(bool v) -> audio_folder_option & {
            _mono = v;
            return *this;
        }
        auto num_frames(int64_t v) -> audio_folder_option & {
            _num_frames = v;
            return *this;
        }
    } audio_folder_option_t;

    // The containers audio::io::load_audio decodes.
    inline auto audio_extensions() -> std::vector<std::string> { return {".wav", ".flac", ".mp3", ".ogg"}; }

    // shortcut: with sample_rate set, the whole file is decoded before the crop; read about num_frames * native rate /
    // sample_rate frames (plus the resampling filter's reach) if long files are resampled and cropped short.
    inline auto audio_loader(const audio_folder_option &opt = {}) -> loader_fn {
        if (opt._sample_rate < 0 || opt._num_frames < 0) {
            return detail::invalid<loader_fn>(
                    std::format("audio_loader: sample_rate ({}) and num_frames ({}) must be non-negative",
                                opt._sample_rate, opt._num_frames));
        }
        return [opt](const path_t &p) {
            auto read = audio::io::load_option();
            // without resampling the crop keeps the first num_frames: decode only those
            if (opt._num_frames > 0 && opt._sample_rate == 0) {
                read.num_frames(opt._num_frames);
            }
            auto [x, sr] = audio::io::load_audio(p, read);
            if (opt._mono) {
                x = x.mean(0, true);
            }
            if (opt._sample_rate > 0) {
                x = audio::functional::resample(x, sr, opt._sample_rate);
            }
            if (opt._num_frames > 0) {
                const auto t = x.size(-1);
                x = t >= opt._num_frames ? x.narrow(-1, 0, opt._num_frames)
                                         : torch::constant_pad_nd(x, {0, opt._num_frames - t});
            }
            return x;
        };
    }

    // DatasetFolder with audio_loader(audio) and audio_extensions() by default (torchaudio has no folder dataset; this
    // is torchvision's layout with the library's audio decoders).
    class AudioFolder : public DatasetFolder {
    public:
        explicit AudioFolder(const path_t &root, const dataset_folder_option &opt = {},
                             const audio_folder_option &audio = {}) :
            DatasetFolder(root, audio_loader(audio), detail::with_extensions(opt, audio_extensions())) {}
    };
} // namespace torchmedia::data

#endif // LIB_TORCH_MEDIA_DATA_FOLDER_HPP
