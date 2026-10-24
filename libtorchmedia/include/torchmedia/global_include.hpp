#ifndef LIB_TORCH_MEDIA_GLOBAL_INCLUDE_HPP
#define LIB_TORCH_MEDIA_GLOBAL_INCLUDE_HPP
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "util.hpp"


namespace torchmedia {
    using tensor_t = torch::Tensor;
    using const_tensor_lref_t = const tensor_t &;
    using tensor_rref_t = tensor_t &&;
    using tensor_options_t = torch::TensorOptions;
    using str_t = std::string;
    using path_t = std::filesystem::path;


    // Error policy: throw ExceptionType(msg); with LIB_TORCH_MEDIA_NO_EXCEPTIONS, return the default value instead.
    // Callers `return handle_exceptions<...>(...)`, and composites check their arguments before calling their parts,
    // so invalid input does not reach the computation in either build (known gaps: 2026-10-23/progress02 Open).
    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions([[maybe_unused]] str_t msg = "") -> DefaultReturnType {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        return DefaultReturnType{};
#else
        throw ExceptionType(msg);
#endif
    }
    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions([[maybe_unused]] DefaultReturnType &&ret, [[maybe_unused]] str_t msg = "")
            -> DefaultReturnType {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        return std::move(ret);
#else
        throw ExceptionType(msg);
#endif
    }


} // namespace torchmedia
#endif // LIB_TORCH_MEDIA_GLOBAL_INCLUDE_HPP
