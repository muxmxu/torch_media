// torchmedia::data dataset utilities (develop_log/2026-10-24/progress03): random_split against torch 2.5.1
// torch.utils.data.random_split goldens (gen_golden.py: seeded generators, the global generator, fractions summed like
// Python 3.12), Subset, ConcatDataset (with the DataLoader) and PadCollate closed-form batches and validation.
#include <set>
#include <torch/torch.h>
#include <torchmedia.hpp>
#include "golden.hpp"
#include "test_util.hpp"

namespace td = torchmedia::data;
using tensor_dataset = torch::data::datasets::TensorDataset;

template<class E, class Fn>
static auto error_of(Fn &&fn) -> std::string {
    try {
        (void) fn();
    } catch (const E &e) {
        return e.what();
    }
    return "<no error>";
}

static auto indices_of(std::vector<td::Subset<tensor_dataset>> &parts) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> out;
    for (auto &p: parts) {
        out.push_back(p.indices());
        for (size_t i = 0; i < p.indices().size(); ++i) { // get() reads through the index map
            TM_CHECK(p.get(i).data.item<int64_t>() == static_cast<int64_t>(p.indices()[i]));
        }
    }
    return out;
}

static void test_random_split_goldens() {
    for (const auto &c: golden::splits) {
        const tensor_dataset ds(torch::arange(c.n));
        std::vector<td::Subset<tensor_dataset>> parts;
        if (c.global) {
            torch::manual_seed(c.seed);
            parts = td::random_split(ds, c.lengths);
        } else {
            parts = td::random_split(ds, c.lengths, c.seed);
        }
        TM_CHECK(indices_of(parts) == c.splits);
        auto with_generator = td::random_split(ds, c.lengths, at::make_generator<at::CPUGeneratorImpl>(c.seed));
        TM_CHECK(c.global || indices_of(with_generator) == c.splits);
    }
    const tensor_dataset ten(torch::arange(10));
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::random_split(ten, {1.5, -0.5}); }) ==
             "random_split: " + golden::split_errors[0]);
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::random_split(ten, {3, 3}); }) ==
             "random_split: " + golden::split_errors[1]);
    // deviations: Python slices with negative lengths and raises TypeError for non-integer ones
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::random_split(ten, {-1, 11}); }) ==
             "random_split: length -1 at index 0 is not a non-negative integer");
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::random_split(ten, {2.5, 7.5}); }) ==
             "random_split: length 2.5 at index 0 is not a non-negative integer");
    TM_CHECK(error_of<std::invalid_argument>([&] {
                 return td::random_split(ten, {std::numeric_limits<double>::infinity(), 1});
             }).starts_with("random_split: length inf"));
    TM_CHECK(td::random_split(tensor_dataset(torch::arange(0)), {}).empty());
    // Python 3.12 sum(): fractions whose left-to-right sum is 1.0000000000000002
    TM_CHECK(td::detail::py_sum({0.4, 0.05, 0.13, 0.34, 0.08}) == 1.0 && 0.4 + 0.05 + 0.13 + 0.34 + 0.08 > 1.0);
}

static void test_subset_concat() {
    const tensor_dataset ds(torch::arange(10, 20));
    td::Subset<tensor_dataset> sub(ds, {4, 0, 9});
    TM_CHECK(sub.size().value() == 3 && sub.get(0).data.item<int64_t>() == 14 && sub.get(2).data.item<int64_t>() == 19);
    TM_CHECK(sub.dataset().size().value() == 10 && sub.indices() == std::vector<size_t>({4, 0, 9}));
    TM_CHECK(error_of<std::out_of_range>([&] { return sub.get(3); }) == "Subset: index 3 is out of range for size 3");
    TM_CHECK(error_of<std::invalid_argument>([&] { return td::Subset<tensor_dataset>(ds, {1, 10}); }) ==
             "Subset: index 10 is out of range for a dataset of size 10");

    td::ConcatDataset<tensor_dataset> cat(
            {tensor_dataset(torch::arange(3)), tensor_dataset(torch::arange(0)), tensor_dataset(torch::arange(5, 7))});
    TM_CHECK(cat.size().value() == 5 && cat.datasets().size() == 3);
    TM_CHECK(cat.cumulative_sizes() == std::vector<size_t>({3, 3, 5}));
    std::vector<int64_t> values;
    for (size_t i = 0; i < 5; ++i) {
        values.push_back(cat.get(i).data.item<int64_t>());
    }
    TM_CHECK(values == std::vector<int64_t>({0, 1, 2, 5, 6})); // the empty dataset is skipped
    TM_CHECK(error_of<std::out_of_range>([&] { return cat.get(5); }) ==
             "ConcatDataset: index 5 is out of range for size 5");
    TM_CHECK(error_of<std::invalid_argument>([] { return td::ConcatDataset<tensor_dataset>({}); }) ==
             "ConcatDataset: datasets should not be an empty iterable");

    // a dataset without a size (size() == nullopt) cannot be indexed into
    struct unsized : torch::data::datasets::Dataset<unsized, td::example_t> {
        auto get(size_t) -> td::example_t override { return {}; }
        auto size() const -> std::optional<size_t> override { return std::nullopt; }
    };
    TM_CHECK(error_of<std::invalid_argument>([] { return td::Subset<unsized>(unsized(), {0}); }) ==
             "Subset: the dataset has no size");
    TM_CHECK(error_of<std::invalid_argument>([] { return td::random_split(unsized(), {0.5, 0.5}, 0); }) ==
             "random_split: the dataset has no size");
    TM_CHECK(error_of<std::invalid_argument>([] { return td::ConcatDataset<unsized>({unsized()}); }) ==
             "ConcatDataset: the dataset at index 0 has no size");

    // random_split -> ConcatDataset of the parts -> DataLoader with 2 workers: every sample exactly once.
    auto parts = td::random_split(tensor_dataset(torch::arange(23)), {0.5, 0.3, 0.2}, 7);
    auto loader = torch::data::make_data_loader(td::ConcatDataset<td::Subset<tensor_dataset>>(parts).map(
                                                        torch::data::transforms::Stack<torch::data::TensorExample>()),
                                                torch::data::DataLoaderOptions(4).workers(2));
    std::multiset<int64_t> seen;
    for (const auto &batch: *loader) {
        for (int64_t i = 0; i < batch.data.size(0); ++i) {
            seen.insert(batch.data[i].item<int64_t>());
        }
    }
    std::multiset<int64_t> all;
    for (int64_t i = 0; i < 23; ++i) {
        all.insert(i);
    }
    TM_CHECK(seen == all && parts[0].size().value() == 12 && parts[1].size().value() == 7);
}

static auto ex(const torch::Tensor &x, int64_t y) -> td::example_t {
    return {x, torch::scalar_tensor(y, torch::kInt64)};
}

static void test_pad_collate() {
    td::PadCollate pad;
    auto b = pad.apply_batch({ex(torch::tensor({1.f, 2.f, 3.f}), 4), ex(torch::tensor({5.f}), 7)});
    TM_CHECK(torch::equal(b.data, torch::tensor({{1.f, 2.f, 3.f}, {5.f, 0.f, 0.f}})));
    TM_CHECK(torch::equal(b.lengths, torch::tensor({3, 1}, torch::kInt64)));
    TM_CHECK(torch::equal(b.target, torch::tensor({4, 7}, torch::kInt64)));

    // [C, L] samples, pad value and pad-to-multiple; integer dtype kept
    td::PadCollate pad4(td::pad_collate_option().value(-1).multiple(4));
    const auto x = torch::arange(10, torch::kInt32).reshape({2, 5});
    const auto y = torch::arange(4, torch::kInt32).reshape({2, 2});
    b = pad4.apply_batch({ex(x, 0), ex(y, 1)});
    auto expected = torch::full({2, 2, 8}, -1, torch::kInt32);
    expected[0].narrow(1, 0, 5).copy_(x);
    expected[1].narrow(1, 0, 2).copy_(y);
    TM_CHECK(torch::equal(b.data, expected) && torch::equal(b.lengths, torch::tensor({5, 2}, torch::kInt64)));
    b = pad4.apply_batch({ex(torch::ones({1, 8}), 0)});
    TM_CHECK(b.data.sizes() == torch::IntArrayRef({1, 1, 8})); // already a multiple

    const auto err = [&](std::vector<td::example_t> batch) {
        return error_of<std::invalid_argument>([&] { return pad.apply_batch(std::move(batch)); });
    };
    TM_CHECK(err({}) == "PadCollate: empty batch");
    const auto tail = ": expected tensors of >= 1 dim and one dtype that differ in the last dim only";
    TM_CHECK(err({ex(torch::ones({2, 3}), 0), ex(torch::ones({3, 3}), 0)}) ==
             std::string("PadCollate: sample 1 is [3, 3] Float, sample 0 is [2, 3] Float") + tail);
    TM_CHECK(err({ex(torch::ones({3}), 0), ex(torch::ones({3}, torch::kDouble), 0)}) ==
             std::string("PadCollate: sample 1 is [3] Double, sample 0 is [3] Float") + tail);
    TM_CHECK(err({ex(torch::ones({3}), 0), ex(torch::ones({1, 3}), 0)}) ==
             std::string("PadCollate: sample 1 is [1, 3] Float, sample 0 is [3] Float") + tail);
    TM_CHECK(err({ex(torch::tensor(1.f), 0)}) ==
             std::string("PadCollate: sample 0 is [] Float, sample 0 is [] Float") + tail);
    TM_CHECK(err({td::example_t{}}) == std::string("PadCollate: sample 0 is undefined, sample 0 is undefined") + tail);
    TM_CHECK(error_of<std::invalid_argument>([] { return td::PadCollate(td::pad_collate_option().multiple(0)); }) ==
             "PadCollate: multiple must be >= 1, got 0");
}

int main() {
    TM_RUN(test_random_split_goldens);
    TM_RUN(test_subset_concat);
    TM_RUN(test_pad_collate);
    return tm_test::summary("data_test_dataset");
}
