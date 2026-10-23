#pragma once
#ifndef LIB_TORCH_MEDIA_DATA_DATASET_HPP
#define LIB_TORCH_MEDIA_DATA_DATASET_HPP
// torchmedia::data (develop_log/2026-10-24/progress03): a thin dataset layer over libtorch's torch::data. Every dataset
// is a torch::data::datasets::Dataset, so torch::data::make_data_loader, its samplers, .map(...) and
// torch::data::transforms::Stack work unchanged. This header: the shared types and torch.utils.data's Subset,
// ConcatDataset and random_split.
//
// Thread safety: make_data_loader copies the dataset into each worker thread. The datasets are read-only after
// construction (DatasetFolder / ManifestDataset share their sample list through a shared_ptr to const), so the copies
// are cheap and safe to use concurrently. A transform that draws from the global generator (the random vision
// transforms) still interleaves its draws across workers: seeded parity needs workers(0).
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "../globel_include.hpp"

namespace torchmedia::data {
    // One sample: the loaded (and transformed) tensor and its target.
    using example_t = torch::data::Example<tensor_t, tensor_t>;
    // A sample / target transform: any torchmedia transform class or callable (const tensor_t &) -> tensor_t.
    using transform_fn = std::function<tensor_t(const_tensor_lref_t)>;

    namespace detail {
        // `return invalid<T>(msg);` throws E, or returns T{} with LIB_TORCH_MEDIA_NO_EXCEPTIONS.
        template<class T, class E = std::invalid_argument>
        inline auto invalid(std::string msg) -> T {
            return handle_exceptions<T, E>(std::move(msg));
        }

        // Python's ", ".join(parts)
        inline auto join(const std::vector<std::string> &parts) -> std::string {
            std::string out;
            for (size_t i = 0; i < parts.size(); ++i) {
                out += (i == 0 ? "" : ", ") + parts[i];
            }
            return out;
        }

        // Python >= 3.12 builtin sum() of floats (Neumaier compensated), as random_split sums its fractions.
        inline auto py_sum(const std::vector<double> &values) -> double {
            double sum = 0.0, c = 0.0;
            for (const double x: values) {
                const double t = sum + x;
                c += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
                sum = t;
            }
            return c != 0.0 && std::isfinite(c) ? sum + c : sum;
        }

        // A torch::data Dataset may have no size (size() == nullopt): Subset / ConcatDataset / random_split reject it.
        inline auto checked_indices(std::vector<size_t> indices, std::optional<size_t> size) -> std::vector<size_t> {
            if (!size) {
                return invalid<std::vector<size_t>>("Subset: the dataset has no size");
            }
            for (const auto i: indices) {
                if (i >= *size) {
                    return invalid<std::vector<size_t>>(
                            std::format("Subset: index {} is out of range for a dataset of size {}", i, *size));
                }
            }
            return indices;
        }

        template<class D>
        inline auto cumulative_sizes(const std::vector<D> &datasets) -> std::vector<size_t> {
            if (datasets.empty()) {
                return invalid<std::vector<size_t>>("ConcatDataset: datasets should not be an empty iterable");
            }
            std::vector<size_t> out;
            for (size_t i = 0; i < datasets.size(); ++i) {
                const auto n = datasets[i].size();
                if (!n) {
                    return invalid<std::vector<size_t>>(
                            std::format("ConcatDataset: the dataset at index {} has no size", i));
                }
                out.push_back((out.empty() ? 0 : out.back()) + *n);
            }
            return out;
        }
    } // namespace detail

    // torch.utils.data.Subset: the samples of `dataset` at `indices`. Indices are checked against the dataset's size
    // here (Python fails on access); a bad one throws std::invalid_argument.
    template<class D>
    class Subset : public torch::data::datasets::Dataset<Subset<D>, typename D::ExampleType> {
    public:
        using ExampleType = typename D::ExampleType;

        Subset(D dataset, std::vector<size_t> indices) :
            _dataset(std::move(dataset)), _indices(detail::checked_indices(std::move(indices), _dataset.size())) {}

        auto get(size_t index) -> ExampleType override {
            if (index >= _indices.size()) {
                return detail::invalid<ExampleType, std::out_of_range>(
                        std::format("Subset: index {} is out of range for size {}", index, _indices.size()));
            }
            return _dataset.get(_indices[index]);
        }
        auto size() const -> std::optional<size_t> override { return _indices.size(); }

        auto dataset() const -> const D & { return _dataset; }
        auto indices() const -> const std::vector<size_t> & { return _indices; }

    private:
        D _dataset;
        std::vector<size_t> _indices;
    };

    // torch.utils.data.ConcatDataset over datasets of one type (wrap different types in a common one first).
    template<class D>
    class ConcatDataset : public torch::data::datasets::Dataset<ConcatDataset<D>, typename D::ExampleType> {
    public:
        using ExampleType = typename D::ExampleType;

        explicit ConcatDataset(std::vector<D> datasets) :
            _datasets(std::move(datasets)), _cumulative(detail::cumulative_sizes(_datasets)) {}

        auto get(size_t index) -> ExampleType override {
            if (index >= *size()) {
                return detail::invalid<ExampleType, std::out_of_range>(
                        std::format("ConcatDataset: index {} is out of range for size {}", index, *size()));
            }
            // bisect_right over the cumulative sizes skips empty datasets, as in torch
            const auto k = static_cast<size_t>(std::ranges::upper_bound(_cumulative, index) - _cumulative.begin());
            return _datasets[k].get(index - (k == 0 ? 0 : _cumulative[k - 1]));
        }
        auto size() const -> std::optional<size_t> override { return _cumulative.empty() ? 0 : _cumulative.back(); }

        auto datasets() const -> const std::vector<D> & { return _datasets; }
        auto cumulative_sizes() const -> const std::vector<size_t> & { return _cumulative; }

    private:
        std::vector<D> _datasets;
        std::vector<size_t> _cumulative;
    };

    // torch.utils.data.random_split: `lengths` are split sizes, or fractions when they sum to 1 (math.isclose and <= 1,
    // summed like Python 3.12's sum()): floor(n * f) each, the remainder added round-robin. The permutation is
    // torch::randperm(n, generator), so a generator seeded like Python's (or the global one after the same
    // manual_seed) gives torch's split. Deviations: a negative or non-integer length throws std::invalid_argument
    // (Python slices with it / raises TypeError); the zero-length-split warning is not printed. A std::vector<int64_t>
    // of lengths converts with {v.begin(), v.end()}.
    template<class D>
    inline auto random_split(D dataset, const std::vector<double> &lengths,
                             std::optional<at::Generator> generator = std::nullopt) -> std::vector<Subset<D>> {
        using R = std::vector<Subset<D>>;
        if (!dataset.size()) {
            return detail::invalid<R>("random_split: the dataset has no size");
        }
        const auto n = static_cast<int64_t>(*dataset.size());
        const double total = detail::py_sum(lengths);
        std::vector<int64_t> sizes;
        if (std::abs(total - 1.0) <= 1e-9 * std::max(std::abs(total), 1.0) && total <= 1.0) {
            for (size_t i = 0; i < lengths.size(); ++i) {
                if (lengths[i] < 0.0 || lengths[i] > 1.0) {
                    return detail::invalid<R>(
                            std::format("random_split: Fraction at index {} is not between 0 and 1", i));
                }
                sizes.push_back(static_cast<int64_t>(std::floor(static_cast<double>(n) * lengths[i])));
            }
            const auto remainder = n - std::reduce(sizes.begin(), sizes.end(), int64_t{0});
            for (int64_t i = 0; i < remainder; ++i) {
                sizes[static_cast<size_t>(i) % sizes.size()] += 1;
            }
        } else {
            for (size_t i = 0; i < lengths.size(); ++i) {
                if (!(lengths[i] >= 0.0 && lengths[i] <= 9007199254740992.0 && lengths[i] == std::floor(lengths[i]))) {
                    return detail::invalid<R>(std::format(
                            "random_split: length {} at index {} is not a non-negative integer", lengths[i], i));
                }
                sizes.push_back(static_cast<int64_t>(lengths[i]));
            }
        }
        if (std::reduce(sizes.begin(), sizes.end(), int64_t{0}) != n) {
            return detail::invalid<R>(
                    "random_split: Sum of input lengths does not equal the length of the input dataset!");
        }
        const auto perm = torch::randperm(n, generator, torch::kLong);
        const auto *p = perm.data_ptr<int64_t>();
        R out;
        for (int64_t offset = 0; const auto len: sizes) {
            out.emplace_back(dataset, std::vector<size_t>(p + offset, p + offset + len));
            offset += len;
        }
        return out;
    }

    // random_split with torch.Generator().manual_seed(seed).
    template<class D>
    inline auto random_split(D dataset, const std::vector<double> &lengths, uint64_t seed) -> std::vector<Subset<D>> {
        return random_split(std::move(dataset), lengths, at::make_generator<at::CPUGeneratorImpl>(seed));
    }
} // namespace torchmedia::data

#endif // LIB_TORCH_MEDIA_DATA_DATASET_HPP
