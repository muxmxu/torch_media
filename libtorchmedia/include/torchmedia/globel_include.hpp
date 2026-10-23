#ifndef LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
#define LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
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


    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions(str_t msg = "") -> auto {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        return DefaultReturnType{};
#else
        throw ExceptionType(msg);
#endif
    }
    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions(DefaultReturnType &&ret, str_t msg = "") -> auto {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        return ret;
#else
        throw ExceptionType(msg);
#endif
    }


} // namespace torchmedia
#endif // LIB_TORCH_MEDIA_GLOBEL_INCLUDE_HPP
