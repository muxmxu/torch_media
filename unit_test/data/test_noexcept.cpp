// torchmedia::data built with LIB_TORCH_MEDIA_NO_EXCEPTIONS: invalid input gives the silent fallback (an empty dataset
// / split / index list, an undefined example or batch) and records the message for torchmedia::last_error() instead of
// throwing, and nothing is computed from it (develop_log/2026-10-24/progress03, 2026-10-24/progress05).
#define TORCHMEDIA_IO_IMPLEMENTATION
#include <filesystem>
#include <fstream>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include <unistd.h>
#include "test_util.hpp"

namespace fs = std::filesystem;
namespace td = torchmedia::data;
using tensor_dataset = torch::data::datasets::TensorDataset;

// The last call failed with a message containing `expect` (consumed).
static auto recorded(const std::string &expect) -> bool {
    const auto msg = torchmedia::last_error();
    torchmedia::clear_last_error();
    return msg.find(expect) != std::string::npos;
}

static void test_defaults() {
    const auto base = fs::temp_directory_path() / std::format("torchmedia_data_noexcept_{}", ::getpid());
    fs::create_directories(base / "cls");
    std::ofstream(base / "bad.scp") << "u1 a.wav\nonly\n";
    td::ImageFolder missing(base / "missing");
    TM_CHECK(missing.size().value() == 0 && missing.classes().empty() && recorded("find_classes: cannot read"));
    TM_CHECK(!missing.get(0).data.defined() && recorded("dataset index 0 is out of range"));
    TM_CHECK(td::ImageFolder(base).size().value() == 0 && recorded("Found no valid file")); // "cls" has no image
    TM_CHECK(td::AudioFolder(base, {}, td::audio_folder_option().sample_rate(-1)).size().value() == 0);
    TM_CHECK(td::ManifestDataset(base / "none.scp", td::audio_loader()).size().value() == 0 && recorded("cannot open"));
    TM_CHECK(td::ManifestDataset(base / "bad.scp", td::audio_loader()).size().value() == 0 && recorded("bad.scp:2"));
    fs::remove_all(base);

    const tensor_dataset ten(torch::arange(10));
    TM_CHECK(td::random_split(ten, {3, 3}).empty() && recorded("random_split: Sum of input lengths"));
    TM_CHECK(td::random_split(ten, {1.5, -0.5}).empty() && recorded("random_split"));
    td::Subset<tensor_dataset> sub(ten, {1, 10});
    TM_CHECK(sub.size().value() == 0 && recorded("Subset: index 10 is out of range"));
    TM_CHECK(!sub.get(0).data.defined() && recorded("Subset: index 0 is out of range for size 0"));
    td::ConcatDataset<tensor_dataset> cat({});
    TM_CHECK(cat.size().value() == 0 && !cat.get(0).data.defined() && recorded("ConcatDataset"));
    td::PadCollate pad(td::pad_collate_option().multiple(0)); // falls back to 1
    TM_CHECK(recorded("PadCollate: multiple must be >= 1"));
    TM_CHECK(pad.apply_batch({{torch::ones({3}), torch::scalar_tensor(0)}}).data.size(-1) == 3);
    TM_CHECK(!pad.apply_batch({}).data.defined() && recorded("PadCollate: empty batch"));
}

int main() {
    TM_RUN(test_defaults);
    return tm_test::summary("data_test_noexcept");
}
