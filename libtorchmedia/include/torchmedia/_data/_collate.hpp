#pragma once
#ifndef LIB_TORCH_MEDIA_DATA_COLLATE_HPP
#define LIB_TORCH_MEDIA_DATA_COLLATE_HPP
// Batch collation for variable-length samples (develop_log/2026-10-24/progress03). Equal-shape samples use libtorch's
// torch::data::transforms::Stack<>() as is.
#include "_dataset.hpp"

namespace torchmedia::data {
    // A padded batch: data [B, ..., L] (L = the longest sample, rounded up to the multiple), the original last-dim
    // lengths [B] (int64) and the stacked targets.
    struct padded_batch {
        tensor_t data;
        tensor_t target;
        tensor_t lengths;
    };

    typedef struct pad_collate_option {
        double _value = 0.0;
        int64_t _multiple = 1;

        auto value(double v) -> pad_collate_option & {
            _value = v;
            return *this;
        }
        auto multiple(int64_t v) -> pad_collate_option & {
            _multiple = v;
            return *this;
        }
    } pad_collate_option_t;

    // Pads every sample's last dim with `value` to the batch maximum (rounded up to a multiple of `multiple`) and
    // stacks them; the other dims and the dtype must agree (std::invalid_argument otherwise). Use it as the loader's
    // collation: torch::data::make_data_loader(dataset.map(PadCollate()), options) yields padded_batch.
    class PadCollate : public torch::data::transforms::BatchTransform<std::vector<example_t>, padded_batch> {
    public:
        explicit PadCollate(const pad_collate_option &opt = {}) : _value(opt._value), _multiple(opt._multiple) {
            if (_multiple < 1) {
                _multiple = 1;
                (void) detail::invalid<int>(std::format("PadCollate: multiple must be >= 1, got {}", opt._multiple));
            }
        }

        auto apply_batch(std::vector<example_t> batch) -> padded_batch override {
            if (const auto err = check(batch); !err.empty()) {
                return detail::invalid<padded_batch>(err);
            }
            std::vector<int64_t> lengths;
            std::vector<tensor_t> targets;
            for (const auto &e: batch) {
                lengths.push_back(e.data.size(-1));
                targets.push_back(e.target);
            }
            const auto longest = (std::ranges::max(lengths) + _multiple - 1) / _multiple * _multiple;
            auto shape = batch.front().data.sizes().vec();
            shape.back() = longest;
            shape.insert(shape.begin(), std::ssize(batch));
            auto data = torch::full(shape, _value, batch.front().data.options());
            for (size_t i = 0; i < batch.size(); ++i) {
                data[static_cast<int64_t>(i)].narrow(-1, 0, lengths[i]).copy_(batch[i].data);
            }
            return {data, torch::stack(targets), torch::tensor(lengths, torch::kInt64)};
        }

    private:
        static auto check(const std::vector<example_t> &batch) -> std::string {
            if (batch.empty()) {
                return "PadCollate: empty batch";
            }
            const auto describe = [](const tensor_t &x) {
                return x.defined() ? c10::str(x.sizes(), " ", x.scalar_type()) : std::string("undefined");
            };
            const auto &first = batch.front().data;
            for (size_t i = 0; i < batch.size(); ++i) {
                const auto &x = batch[i].data; // sample 0 is checked first, so `first` is valid past it
                if (!x.defined() || x.dim() == 0 || x.dim() != first.dim() || x.scalar_type() != first.scalar_type() ||
                    x.sizes().slice(0, x.dim() - 1) != first.sizes().slice(0, first.dim() - 1)) {
                    return std::format("PadCollate: sample {} is {}, sample 0 is {}: expected tensors of >= 1 dim and "
                                       "one dtype that differ in the last dim only",
                                       i, describe(x), describe(first));
                }
            }
            return {};
        }

        double _value;
        int64_t _multiple;
    };
} // namespace torchmedia::data

#endif // LIB_TORCH_MEDIA_DATA_COLLATE_HPP
