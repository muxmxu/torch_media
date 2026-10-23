// torchmedia::data folder / manifest datasets (develop_log/2026-10-24/progress03): DatasetFolder, find_classes and
// make_dataset against torchvision 0.20.1 on the golden tree (gen_golden.py -> golden.hpp: ordering, class_to_idx,
// symlinks, extension rules, error messages), ImageFolder / AudioFolder decoding files written by the library's
// save_image / save_audio, ManifestDataset parsing, and DataLoader integration (Stack, PadCollate, drop_last, workers).
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <set>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <unistd.h>
#include "golden.hpp"
#include "test_util.hpp"

namespace fs = std::filesystem;
namespace td = torchmedia::data;
namespace vio = torchmedia::vision::io;
namespace aio = torchmedia::audio::io;
namespace afn = torchmedia::audio::functional;

// The message of the E thrown by fn, or "<no error>".
template<class E, class Fn>
static auto error_of(Fn &&fn) -> std::string {
    try {
        (void) fn();
    } catch (const E &e) {
        return e.what();
    }
    return "<no error>";
}

// A directory under temp_directory_path(), removed (permissions restored first) on scope exit.
struct temp_dir {
    fs::path base;
    std::vector<fs::path> locked;

    explicit temp_dir(const std::string &tag) :
        base(fs::temp_directory_path() / std::format("torchmedia_data_{}_{}", tag, ::getpid())) {
        fs::remove_all(base);
        fs::create_directories(base);
    }
    ~temp_dir() {
        for (const auto &d: locked) {
            fs::permissions(d, fs::perms::owner_all);
        }
        std::error_code ec;
        fs::remove_all(base, ec);
    }
    auto write(const fs::path &rel, const std::string &text) const -> fs::path {
        const auto p = base / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p, std::ios::binary) << text;
        return p;
    }
};

static auto lower(std::string s) -> std::string {
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// golden::tree under t.base: image-named files are 2x2 RGB PNGs (value k / 255 for the k-th file), others hold "x".
static void build_golden_tree(temp_dir &t) {
    int k = 0;
    for (const auto &e: golden::tree) {
        const auto p = t.base / e.path;
        fs::create_directories(p.parent_path());
        const std::string kind = e.kind;
        if (kind == "file") {
            const auto name = lower(p.filename().string());
            if (name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".jpeg")) {
                TM_CHECK(vio::save_image(torch::full({3, 2, 2}, ++k / 255.0), p));
            } else {
                t.write(e.path, "x");
            }
        } else if (kind == "link") {
            fs::create_symlink(e.target, p);
        } else {
            fs::create_directories(p);
            if (kind == "noperm") {
                t.locked.push_back(p);
            }
        }
    }
    for (const auto &d: t.locked) {
        fs::permissions(d, fs::perms::none);
    }
}

static auto relative(const std::vector<td::sample_t> &samples, const fs::path &root) -> golden::samples_t {
    golden::samples_t out;
    for (const auto &[p, target]: samples) {
        out.emplace_back(p.string().substr(root.string().size() + 1), target);
    }
    return out;
}

static auto path_loader(const fs::path &p) -> torch::Tensor {
    return torch::tensor(static_cast<int64_t>(p.string().size()));
}

static void test_golden_tree() {
    temp_dir t("tree");
    build_golden_tree(t);
    const auto root = t.base / "root";
    const auto [classes, class_to_idx] = td::find_classes(root);
    TM_CHECK(classes == golden::classes);
    for (size_t i = 0; i < classes.size(); ++i) {
        TM_CHECK(class_to_idx.at(classes[i]) == static_cast<int64_t>(i));
    }
    TM_CHECK(error_of<std::runtime_error>([&] { return td::find_classes(t.base / "ext"); }) ==
             "Couldn't find any class folder in " + (t.base / "ext").string() + ".");
    TM_CHECK(error_of<std::runtime_error>([&] {
                 return td::find_classes(t.base / "missing");
             }).starts_with("find_classes: cannot read "));

    const std::vector<std::string> ext3 = {".png", ".jpg", ".jpeg"};
    const td::DatasetFolder ds(root, path_loader, td::dataset_folder_option().extensions(ext3).allow_empty(true));
    TM_CHECK(ds.classes() == golden::classes);
    TM_CHECK(ds.class_to_idx() == class_to_idx);
    TM_CHECK(relative(ds.samples(), root) == golden::ext3);
    TM_CHECK(ds.size().value() == golden::ext3.size() && ds.targets().size() == golden::ext3.size());
    for (size_t i = 0; i < golden::ext3.size(); ++i) {
        TM_CHECK(ds.targets()[i] == golden::ext3[i].second);
    }
    const auto copy = ds; // the loader's per-worker copies share the sample list
    TM_CHECK(&copy.samples() == &ds.samples());
    TM_CHECK(error_of<std::runtime_error>([&] {
                 return td::DatasetFolder(root, path_loader, td::dataset_folder_option().extensions(ext3));
             }) == golden::ext3_error);

    const auto nodot =
            td::DatasetFolder(root, path_loader, td::dataset_folder_option().extensions({"png"}).allow_empty(true));
    TM_CHECK(relative(nodot.samples(), root) == golden::nodot);

    const auto prefix = root.string().size();
    const auto sub = [&](const fs::path &p) { return p.string().substr(prefix).find("sub") != std::string::npos; };
    const auto valid =
            td::DatasetFolder(root, path_loader, td::dataset_folder_option().is_valid_file(sub).allow_empty(true));
    TM_CHECK(relative(valid.samples(), root) == golden::valid_sub);
    TM_CHECK(error_of<std::runtime_error>([&] {
                 return td::DatasetFolder(root, path_loader, td::dataset_folder_option().is_valid_file(sub));
             }) == golden::valid_sub_error);

    // ImageFolder: torchvision's IMG_EXTENSIONS minus the formats the decoder lacks; none of those are in the tree.
    const td::ImageFolder images(root, td::dataset_folder_option().allow_empty(true));
    TM_CHECK(relative(images.samples(), root) == golden::image_folder);
    const td::ImageFolder floats(root, td::dataset_folder_option().allow_empty(true),
                                 td::image_folder_option().load_image(true));
    TM_CHECK(relative(floats.samples(), root) == golden::image_folder);
    // extensions / is_valid_file given: they replace the default list
    golden::samples_t jpg;
    std::ranges::copy_if(golden::ext3, std::back_inserter(jpg),
                         [](const auto &s) { return lower(s.first).ends_with(".jpg"); });
    TM_CHECK(
            jpg.size() == 2 &&
            relative(
                    td::ImageFolder(root, td::dataset_folder_option().extensions({".jpg"}).allow_empty(true)).samples(),
                    root) == jpg);
    TM_CHECK(relative(td::ImageFolder(root, td::dataset_folder_option().is_valid_file(sub).allow_empty(true)).samples(),
                      root) == golden::valid_sub);
    TM_CHECK(error_of<std::runtime_error>([&] { return td::ImageFolder(root); })
                     .starts_with("Found no valid file for the classes empty_cls, only_txt. Supported extensions are: "
                                  ".jpg, .jpeg, .png"));
    // decoded samples: the k-th image file holds k / 255; the broken symlink fails to load.
    auto img = images;
    for (size_t i = 0; i < img.samples().size(); ++i) {
        const auto rel = golden::image_folder[i].first;
        if (rel == "banana/broken.png") {
            TM_CHECK(error_of<std::runtime_error>([&] { return img.get(i); }).starts_with("read_file: could not read"));
            continue;
        }
        const auto e = img.get(i);
        TM_CHECK(e.data.sizes() == torch::IntArrayRef({3, 2, 2}) && e.data.scalar_type() == torch::kUInt8);
        TM_CHECK(e.target.dim() == 0 && e.target.scalar_type() == torch::kInt64 &&
                 e.target.item<int64_t>() == golden::image_folder[i].second);
        auto f = floats;
        TM_CHECK_TENSOR_CLOSE(f.get(i).data, e.data.to(torch::kFloat32) / 255.0, 1e-7, 0.0);
    }

    const td::class_to_idx_t custom = {{"apple", 5}, {"missing", 1}, {"Zebra", 0}};
    TM_CHECK(relative(td::make_dataset(root, custom, std::vector<std::string>{".png"}, {}, true), root) ==
             golden::custom);
    TM_CHECK(error_of<std::runtime_error>([&] {
                 return td::make_dataset(root, custom, std::vector<std::string>{".png"});
             }) == golden::custom_error);
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::make_dataset(root, {}, ext3); }) ==
             golden::empty_map_error);
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::make_dataset(root, custom, std::nullopt); }) ==
             golden::none_error);
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::make_dataset(root, custom, ext3, sub); }) ==
             golden::none_error);
    TM_CHECK(error_of<std::invalid_argument>([&] {
                 return td::DatasetFolder(root, {}, td::dataset_folder_option().extensions(ext3));
             }) == "DatasetFolder: loader must be callable");
}

// find_classes: an entry that cannot be stat'ed raises (Python's DirEntry.is_dir() only swallows FileNotFoundError, so
// torchvision raises OSError ELOOP here); a broken symlink (ENOENT) is skipped, as in the golden tree.
static void test_find_classes_stat_error() {
    temp_dir t("stat");
    t.write("a/x.png", "x");
    fs::create_symlink("selfloop", t.base / "selfloop");
    fs::create_symlink("nowhere", t.base / "broken");
    const auto msg = error_of<std::runtime_error>([&] { return td::find_classes(t.base); });
    TM_CHECK(msg.starts_with("find_classes: cannot stat " + (t.base / "selfloop").string() + ": "));
    fs::remove(t.base / "selfloop");
    TM_CHECK(td::find_classes(t.base).first == std::vector<std::string>({"a"}));
}

static void test_has_file_allowed_extension() {
    TM_CHECK(td::has_file_allowed_extension("a/B.PnG", {".png"}));
    TM_CHECK(!td::has_file_allowed_extension("a/b.png", {".PNG"})); // extensions are given in lower case
    TM_CHECK(td::has_file_allowed_extension("A.JPG/x.Jpeg", {".png", ".jpeg"}));
    TM_CHECK(!td::has_file_allowed_extension("a.png/x", {".png"}));
    TM_CHECK(!td::has_file_allowed_extension("a.png", {}));
    // load_image reads PNM (develop_log/2026-10-24/progress04); decode_image, like torchvision's, does not
    TM_CHECK(td::image_extensions() == std::vector<std::string>({".jpg", ".jpeg", ".png"}) &&
             td::image_extensions(td::image_folder_option().load_image(true)) ==
                     std::vector<std::string>({".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"}));
}

// get(): loader, transform and target_transform (a vision transform class as the sample transform); index checks.
static void test_get_transforms() {
    temp_dir t("img");
    // cat: an RGB and a gray image, dog: an RGBA one; pixel values k / 255.
    const auto rgb = torch::arange(18, torch::kFloat32).reshape({3, 2, 3}) / 255.0;
    const auto gray = torch::arange(6, torch::kFloat32).reshape({1, 2, 3}).add(40) / 255.0;
    const auto rgba = torch::arange(24, torch::kFloat32).reshape({4, 2, 3}).add(100) / 255.0;
    fs::create_directories(t.base / "cat");
    fs::create_directories(t.base / "dog");
    TM_CHECK(vio::save_image(rgb, t.base / "cat/0.png"));
    TM_CHECK(vio::save_image(gray, t.base / "cat/1.png"));
    TM_CHECK(vio::save_image(rgba, t.base / "dog/2.png"));
    t.write("dog/notes.txt", "x");
    const auto u8 = [](const torch::Tensor &x) { return x.mul(255.0).round().to(torch::kUInt8); };

    td::ImageFolder ds(t.base);
    TM_CHECK(ds.classes() == std::vector<std::string>({"cat", "dog"}) && ds.size().value() == 3);
    TM_CHECK(torch::equal(ds.get(0).data, u8(rgb)));
    TM_CHECK(torch::equal(ds.get(1).data, u8(gray).expand({3, -1, -1})));
    TM_CHECK(torch::equal(ds.get(2).data, u8(rgba).slice(0, 0, 3)));
    TM_CHECK(ds.get(2).target.item<int64_t>() == 1);
    td::ImageFolder unchanged(t.base, {}, td::image_folder_option().mode(vio::image_read_mode::unchanged));
    TM_CHECK(torch::equal(unchanged.get(2).data, u8(rgba)) && unchanged.get(1).data.size(0) == 1);
    // load_image(true): RGB like ImageFolder (gray repeated, alpha dropped), so mixed folders stack
    td::ImageFolder floats(t.base, {}, td::image_folder_option().load_image(true));
    TM_CHECK_TENSOR_CLOSE(floats.get(0).data, rgb, 1e-6, 0.0);
    TM_CHECK_TENSOR_CLOSE(floats.get(1).data, gray.expand({3, -1, -1}), 1e-6, 0.0);
    TM_CHECK_TENSOR_CLOSE(floats.get(2).data, rgba.slice(0, 0, 3), 1e-6, 0.0);

    namespace vt = torchmedia::vision::transform;
    td::ImageFolder transformed(
            t.base, td::dataset_folder_option()
                            .transform(vt::ToDtype(vt::to_dtype_option().dtype(torch::kFloat32).scale(true)))
                            .target_transform([](const torch::Tensor &y) { return y + 10; }));
    const auto e = transformed.get(0);
    TM_CHECK_TENSOR_CLOSE(e.data, rgb, 1e-6, 0.0);
    TM_CHECK(e.target.item<int64_t>() == 10);
    TM_CHECK(error_of<std::out_of_range>([&] { return ds.get(3); }) == "dataset index 3 is out of range for size 3");

    // Stack collation through the loader: batch sizes, drop_last, targets in order.
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            ds.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions(2));
    std::vector<int64_t> sizes, targets;
    for (const auto &batch: *loader) {
        TM_CHECK(batch.data.sizes().slice(1) == torch::IntArrayRef({3, 2, 3}));
        sizes.push_back(batch.data.size(0));
        for (int64_t i = 0; i < batch.target.size(0); ++i) {
            targets.push_back(batch.target[i].item<int64_t>());
        }
    }
    TM_CHECK(sizes == std::vector<int64_t>({2, 1}) && targets == std::vector<int64_t>({0, 0, 1}));
    auto dropping = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            ds.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions(2).drop_last(true));
    int64_t batches = 0;
    for (const auto &batch: *dropping) {
        batches += batch.data.size(0) == 2;
    }
    TM_CHECK(batches == 1);
}

// A 2x2 16-bit gray PNG holding {0, 1000, 40000, 65535} (written with Python's zlib; save_image is 8-bit only).
static const std::vector<uint8_t> gray16_png = {137, 80,  78,  71,  13, 10,  26,  10,  0,   0,  0,  13,  73,  72,  68,
                                                82,  0,   0,   0,   2,  0,   0,   0,   2,   16, 0,  0,   0,   0,   7,
                                                77,  142, 187, 0,   0,  0,   18,  73,  68,  65, 84, 120, 218, 99,  96,
                                                96,  96,  126, 193, 48, 199, 225, 255, 127, 0,  11, 188, 3,   198, 125,
                                                224, 10,  171, 0,   0,  0,   0,   73,  69,  78, 68, 174, 66,  96,  130};

// ImageFolder samples are 8-bit 3-channel in every loader setup (16-bit PNG scaled like v2 to_dtype(scale=True)), so a
// folder mixing 8-bit / 16-bit / PNM files stacks; a file that does not decode is named in the error, also from a
// loader worker.
static void test_image_mixed_and_corrupt() {
    temp_dir t("mixed");
    const auto rgb = torch::arange(12, torch::kFloat32).reshape({3, 2, 2}).mul(20) / 255.0;
    fs::create_directories(t.base / "c");
    TM_CHECK(vio::save_image(rgb, t.base / "c/a.png"));
    std::ofstream(t.base / "c/b.png", std::ios::binary)
            .write(reinterpret_cast<const char *>(gray16_png.data()), std::ssize(gray16_png));
    using namespace std::string_literals;
    t.write("c/d.pgm", "P5\n2 2\n65535\n\x00\x00\x03\xe8\x9c\x40\xff\xff"s); // 16-bit: 0, 1000, 40000, 65535
    t.write("c/e.ppm", "P3 2 2 255\n0 10 20 30 40 50\n60 70 80 90 100 255\n"); // ASCII
    const auto want16 = torch::tensor({0, 3, 156, 255}, torch::kUInt8).reshape({1, 2, 2}).expand({3, -1, -1});

    td::ImageFolder ds(t.base);
    TM_CHECK(ds.size().value() == 2 && ds.get(0).data.scalar_type() == torch::kUInt8);
    TM_CHECK(torch::equal(ds.get(1).data, want16));
    auto stacked = torch::data::make_data_loader(ds.map(torch::data::transforms::Stack<>()),
                                                 torch::data::DataLoaderOptions(2));
    int64_t batches = 0;
    for (const auto &b: *stacked) {
        batches += b.data.sizes() == torch::IntArrayRef({2, 3, 2, 2}) && b.data.scalar_type() == torch::kUInt8;
    }
    TM_CHECK(batches == 1);
    // unchanged: decode_image's output, uint16 kept
    td::ImageFolder raw(t.base, {}, td::image_folder_option().mode(vio::image_read_mode::unchanged));
    TM_CHECK(raw.get(1).data.scalar_type() == torch::kUInt16 && raw.get(1).data.size(0) == 1);
    // load_image(true): the same values as floats, plus the PNM files (sample / maxval), all stacked
    td::ImageFolder floats(t.base, {}, td::image_folder_option().load_image(true));
    TM_CHECK(floats.size().value() == 4);
    for (size_t i = 0; i < 2; ++i) {
        TM_CHECK_TENSOR_CLOSE(floats.get(i).data, ds.get(i).data.to(torch::kFloat32) / 255.0, 1e-7, 0.0);
    }
    TM_CHECK(torch::equal(
            floats.get(2).data,
            torch::tensor({0.f, 1000.f, 40000.f, 65535.f}).div(65535.0).reshape({1, 2, 2}).expand({3, -1, -1})));
    TM_CHECK(torch::equal(
            floats.get(3).data,
            torch::arange(0, 120, 10, torch::kFloat32).index_put_({11}, 255).reshape({2, 2, 3}).permute({2, 0, 1}) /
                    255.0));
    auto float_loader = torch::data::make_data_loader(floats.map(torch::data::transforms::Stack<>()),
                                                      torch::data::DataLoaderOptions(4));
    int64_t float_batches = 0;
    for (const auto &b: *float_loader) {
        float_batches += b.data.sizes() == torch::IntArrayRef({4, 3, 2, 2});
    }
    TM_CHECK(float_batches == 1);

    // corrupt files: get() and a 2-worker loader name the file
    const auto png = t.write("bad/c/zz.png", "not a png at all");
    td::ImageFolder corrupt(t.base / "bad");
    TM_CHECK(error_of<std::runtime_error>([&] { return corrupt.get(0); }) ==
             png.string() + ": decode_image: unsupported image file, only JPEG and PNG are supported.");
    auto workers = torch::data::make_data_loader(corrupt.map(torch::data::transforms::Stack<>()),
                                                 torch::data::DataLoaderOptions(1).workers(2));
    const auto from_worker = error_of<std::exception>([&] {
        for (const auto &b: *workers) {
            (void) b;
        }
        return 0;
    });
    TM_CHECK(from_worker.find(png.string() + ": decode_image:") != std::string::npos);
    const auto wav = t.write("bad/c/zz.wav", "not a wav at all");
    TM_CHECK(error_of<std::runtime_error>([&] { return td::AudioFolder(t.base / "bad").get(0); }).find(wav.string()) !=
             std::string::npos);
}

// AudioFolder over WAV files written by save_audio (float32 WAV is exact): mono, resampling, fixed length.
static void test_audio_folder() {
    temp_dir t("audio");
    torch::manual_seed(0);
    struct clip {
        std::string rel;
        torch::Tensor x;
        int sr;
    };
    const std::vector<clip> clips = {
            {"no/c0.wav", torch::rand({1, 80}) - 0.5, 16000}, {"no/c1.wav", torch::rand({1, 81}) - 0.5, 16000},
            {"no/c2.wav", torch::rand({1, 82}) - 0.5, 16000}, {"yes/a.wav", torch::rand({2, 100}) - 0.5, 16000},
            {"yes/b.WAV", torch::rand({1, 50}) - 0.5, 8000},  {"yes/e0.wav", torch::rand({1, 90}) - 0.5, 16000},
            {"yes/e1.wav", torch::rand({1, 91}) - 0.5, 16000}};
    for (const auto &c: clips) {
        fs::create_directories((t.base / c.rel).parent_path());
        TM_CHECK(aio::save_audio(c.x, t.base / c.rel, c.sr));
    }
    t.write("yes/readme.txt", "x");

    td::AudioFolder ds(t.base);
    TM_CHECK(ds.size().value() == clips.size() && ds.targets() == std::vector<int64_t>({0, 0, 0, 1, 1, 1, 1}));
    for (size_t i = 0; i < clips.size(); ++i) {
        TM_CHECK(ds.samples()[i].first == t.base / clips[i].rel);
        TM_CHECK(torch::equal(ds.get(i).data, clips[i].x));
    }
    td::AudioFolder resampled(t.base, {}, td::audio_folder_option().sample_rate(16000));
    TM_CHECK(torch::equal(resampled.get(4).data, afn::resample(clips[4].x, 8000, 16000)));
    TM_CHECK(resampled.get(4).data.size(1) == 100 && torch::equal(resampled.get(3).data, clips[3].x));
    td::AudioFolder mono(t.base, {}, td::audio_folder_option().mono(true));
    TM_CHECK(torch::equal(mono.get(3).data, clips[3].x.mean(0, true)));
    td::AudioFolder fixed(t.base, {}, td::audio_folder_option().num_frames(64));
    TM_CHECK(torch::equal(fixed.get(0).data, clips[0].x.narrow(1, 0, 64)));
    TM_CHECK(torch::equal(fixed.get(4).data, torch::cat({clips[4].x, torch::zeros({1, 14})}, 1)));
    td::AudioFolder all(t.base, {}, td::audio_folder_option().sample_rate(16000).mono(true).num_frames(95));
    TM_CHECK(torch::equal(all.get(3).data, clips[3].x.mean(0, true).narrow(1, 0, 95)));
    TM_CHECK(torch::equal(all.get(4).data.narrow(1, 0, 95), afn::resample(clips[4].x, 8000, 16000).narrow(1, 0, 95)));
    TM_CHECK(error_of<std::invalid_argument>([] {
                 return td::audio_loader(td::audio_folder_option().sample_rate(-1));
             }).starts_with("audio_loader: sample_rate (-1)"));
    TM_CHECK(!error_of<std::invalid_argument>([] {
                  return td::AudioFolder("x", {}, td::audio_folder_option().num_frames(-2));
              }).empty());

    // PadCollate through the loader: sequential batches, drop_last, and 2 workers covering every sample once.
    const std::vector<int64_t> lengths = {80, 81, 82, 100, 50, 90, 91};
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            mono.map(td::PadCollate(td::pad_collate_option().value(-2.0))), torch::data::DataLoaderOptions(3));
    std::vector<int64_t> seen;
    int64_t batch_index = 0;
    for (const auto &b: *loader) {
        const auto n = b.data.size(0);
        TM_CHECK(n == (batch_index < 2 ? 3 : 1) && b.data.size(1) == 1);
        TM_CHECK(b.lengths.scalar_type() == torch::kInt64 && b.target.size(0) == n);
        for (int64_t i = 0; i < n; ++i) {
            const auto len = b.lengths[i].item<int64_t>();
            const auto k = static_cast<size_t>(batch_index * 3 + i);
            TM_CHECK(len == lengths[k] && b.target[i].item<int64_t>() == ds.targets()[k]);
            TM_CHECK(torch::equal(b.data[i].narrow(-1, 0, len), clips[k].x.mean(0, true)));
            TM_CHECK((b.data[i].narrow(-1, len, b.data.size(-1) - len) == -2.0).all().item<bool>());
            seen.push_back(len);
        }
        TM_CHECK(b.data.size(-1) == *std::max_element(seen.end() - n, seen.end()));
        ++batch_index;
    }
    TM_CHECK(seen == lengths);
    for (const bool drop: {false, true}) {
        auto workers = torch::data::make_data_loader(mono.map(td::PadCollate()),
                                                     torch::data::DataLoaderOptions(2).workers(2).drop_last(drop));
        std::multiset<int64_t> got;
        int64_t batches = 0;
        for (const auto &b: *workers) {
            ++batches;
            for (int64_t i = 0; i < b.lengths.size(0); ++i) {
                got.insert(b.lengths[i].item<int64_t>());
            }
        }
        TM_CHECK(batches == (drop ? 3 : 4) && got.size() == (drop ? 6u : 7u));
        if (!drop) {
            TM_CHECK(got == std::multiset<int64_t>(lengths.begin(), lengths.end()));
        }
    }
}

static void test_manifest() {
    temp_dir t("manifest");
    const auto dir = t.base / "lists";
    const auto kaldi = t.write("lists/kaldi.scp", "utt1 a.wav 3\n\n  utt2\t/abs/b.wav  \nutt3 sub/c.wav -2\r\n");
    const td::ManifestDataset k(kaldi, path_loader);
    TM_CHECK(k.size().value() == 3 && k.keys() == std::vector<std::string>({"utt1", "utt2", "utt3"}));
    TM_CHECK(k.samples()[0].first == dir / "a.wav" && k.samples()[1].first == "/abs/b.wav" &&
             k.samples()[2].first == dir / "sub/c.wav");
    TM_CHECK(k.targets() == std::vector<int64_t>({3, -1, -2}) && k.classes().empty());

    const auto names = t.write("lists/names.scp", "u1 x.wav dog\nu2 y.wav cat\nu3 z.wav\nu4 w.wav dog\n");
    const td::ManifestDataset c(names, path_loader, td::manifest_option().class_labels(true));
    TM_CHECK(c.classes() == std::vector<std::string>({"cat", "dog"}) && c.class_to_idx().at("dog") == 1);
    TM_CHECK(c.targets() == std::vector<int64_t>({1, 0, -1, 1}));

    const auto tsv =
            t.write("lists/list.tsv", "id\tlabel\tpath\textra\nk1\t2\tmy file.wav\tz\n\nk2\t\t/abs/q.wav\tz\n");
    const td::ManifestDataset h(tsv, path_loader, td::manifest_option().header(true));
    TM_CHECK(h.samples()[0].first == dir / "my file.wav" && h.samples()[1].first == "/abs/q.wav");
    TM_CHECK(h.targets() == std::vector<int64_t>({2, -1}) &&
             h.keys() == std::vector<std::string>({"my file.wav", "/abs/q.wav"}));
    const td::ManifestDataset ids(tsv, path_loader,
                                  td::manifest_option().header(true).path_column("id").label_column(""));
    TM_CHECK(ids.samples()[1].first == dir / "k2" && ids.targets() == std::vector<int64_t>({-1, -1}));
    const td::ManifestDataset empty(t.write("lists/empty.tsv", "\n \n"), path_loader);
    TM_CHECK(empty.size().value() == 0);

    // errors name the manifest line
    const auto err = [&](const std::string &text, const td::manifest_option &opt = {}) {
        const auto p = t.write("lists/bad.txt", text);
        return error_of<std::invalid_argument>([&] { return td::ManifestDataset(p, path_loader, opt); });
    };
    const auto at = "ManifestDataset: " + (dir / "bad.txt").string() + ":";
    TM_CHECK(err("u1 a.wav\nonly\n") == at + "2: expected 'key path [label]', got 1 fields");
    TM_CHECK(err("u1 a.wav 1 2\n") == at + "1: expected 'key path [label]', got 4 fields");
    TM_CHECK(err("u1 a.wav 1\nu2 b.wav 3x\n") ==
             at + "2: label '3x' is not an int64 (class_labels(true) reads class names)");
    TM_CHECK(err("\nfile\tlabel\n", td::manifest_option().header(true)) == at + "2: the header has no column 'path'");
    TM_CHECK(err("path\tclass\n", td::manifest_option().header(true)) == at + "1: the header has no column 'label'");
    TM_CHECK(err("path\tlabel\na.wav\t1\t2\n", td::manifest_option().header(true)) ==
             at + "2: expected 2 fields like the header, got 3");
    TM_CHECK(err("path\tlabel\n\t3\n", td::manifest_option().header(true)) == at + "2: empty path");
    TM_CHECK(error_of<std::runtime_error>([&] { return td::ManifestDataset(dir / "none.scp", path_loader); }) ==
             "ManifestDataset: cannot open " + (dir / "none.scp").string());
    // a directory opens as an empty stream
    TM_CHECK(error_of<std::runtime_error>([&] { return td::ManifestDataset(dir, path_loader); }) ==
             "ManifestDataset: cannot open " + dir.string());
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::ManifestDataset(kaldi, {}); }) ==
             "ManifestDataset: loader must be callable");

    // loading through the manifest: audio_loader + transforms
    torch::manual_seed(1);
    const auto x = torch::rand({2, 40});
    TM_CHECK(aio::save_audio(x, dir / "a.wav", 16000));
    td::ManifestDataset audio(kaldi, td::audio_loader(td::audio_folder_option().mono(true)),
                              td::manifest_option()
                                      .transform([](const torch::Tensor &y) { return y * 2; })
                                      .target_transform([](const torch::Tensor &y) { return y.to(torch::kFloat32); }));
    const auto e = audio.get(0);
    TM_CHECK(torch::allclose(e.data, x.mean(0, true) * 2) && e.target.item<float>() == 3.0f);
}

int main() {
    TM_RUN(test_golden_tree);
    TM_RUN(test_find_classes_stat_error);
    TM_RUN(test_has_file_allowed_extension);
    TM_RUN(test_get_transforms);
    TM_RUN(test_image_mixed_and_corrupt);
    TM_RUN(test_audio_folder);
    TM_RUN(test_manifest);
    return tm_test::summary("data_test_folder");
}
