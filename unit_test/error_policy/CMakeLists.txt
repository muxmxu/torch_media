# unit_test/error_policy -- the same calls built with and without LIB_TORCH_MEDIA_NO_EXCEPTIONS
# (develop_log/2026-10-24/progress05): the default build throws, the silent build returns the documented fallback and
# records torchmedia::last_error().
add_executable(error_policy_test_throw main.cpp)
target_link_libraries(error_policy_test_throw PRIVATE torch_media)
add_test(NAME error_policy_test_throw COMMAND error_policy_test_throw)

add_executable(error_policy_test_silent main.cpp)
target_compile_definitions(error_policy_test_silent PRIVATE LIB_TORCH_MEDIA_NO_EXCEPTIONS)
target_link_libraries(error_policy_test_silent PRIVATE torch_media)
add_test(NAME error_policy_test_silent COMMAND error_policy_test_silent)
