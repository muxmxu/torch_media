#pragma once
#include <ostream>
#include <print>
#include <ranges>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include "torchmedia/global_include.hpp"
#ifndef LIBTORCH_MEDIA_BASIC
#define LIBTORCH_MEDIA_BASIC

namespace torchmedia::util {
    auto inline to_device(tensor_t tensor, const std::string &device) -> tensor_t {
        return tensor.to(tensor_options_t().device(device));
    };

    auto inline to_string(const tensor_t &obj) -> std::string {
        std::stringstream ss;
        ss << obj << std::flush;
        return ss.str();
    }

    auto inline to_string(const torch::IntArrayRef &t) -> std::string {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < static_cast<int>(t.size()); ++i) {
            ss << t[i];
            if (i + 1 < static_cast<int>(t.size())) {
                ss << ", ";
            }
        }
        ss << "]" << std::flush;
        return ss.str();
    }

    auto inline print_tensor(const tensor_t &t) -> void { std::print("{}", to_string(t)); }

    auto inline tensor_wise_equals(const tensor_t &left, const tensor_t &right) { return torch::all(left == right); }
} // namespace torchmedia::util
#endif
