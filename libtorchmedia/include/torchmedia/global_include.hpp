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


    // Error policy -- the user picks: throw or stay silent (develop_log/2026-10-24/progress05):
    // - default: invalid input throws ExceptionType(msg) (std::invalid_argument; std::runtime_error for I/O).
    // - LIB_TORCH_MEDIA_NO_EXCEPTIONS (define it for the whole program): silent. The call records msg for last_error()
    //   and returns a documented fallback: an undefined tensor (test `!out.defined()`), a value-initialised struct /
    //   pair / tuple (its tensors undefined) or an empty container; a void call (metric update) changes nothing; a
    //   rejected constructor leaves its object callable (documented per class). Results that already report failure
    //   in both builds (save_audio / save_image: false) leave last_error() alone.
    // libtorch's own c10::Error is not ours to silence and still throws. Callers `return handle_exceptions<...>(...)`;
    // composites validate first or stop at a part's fallback, so nothing is computed from invalid input.
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
    namespace detail {
        inline auto last_error_slot() -> str_t & {
            thread_local str_t msg;
            return msg;
        }
    } // namespace detail

    // The message of this thread's most recent failure ("" if none since clear_last_error(); success keeps it).
    inline auto last_error() -> str_t { return detail::last_error_slot(); }
    inline auto clear_last_error() -> void { detail::last_error_slot().clear(); }
#else
    // Failures throw in this build: nothing is recorded.
    inline auto last_error() -> str_t { return {}; }
    inline auto clear_last_error() -> void {}
#endif

    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions(str_t msg = "") -> DefaultReturnType {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        detail::last_error_slot() = std::move(msg);
        return DefaultReturnType{};
#else
        throw ExceptionType(msg);
#endif
    }
    // An explicit fallback value; msg has no default, so a lone message never binds here (a string literal would
    // convert to a bool fallback).
    template<class DefaultReturnType, class ExceptionType>
    auto handle_exceptions([[maybe_unused]] DefaultReturnType &&ret, str_t msg) -> DefaultReturnType {
#ifdef LIB_TORCH_MEDIA_NO_EXCEPTIONS
        detail::last_error_slot() = std::move(msg);
        return std::move(ret);
#else
        throw ExceptionType(msg);
#endif
    }


} // namespace torchmedia
#endif // LIB_TORCH_MEDIA_GLOBAL_INCLUDE_HPP
