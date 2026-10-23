# unit_test/data/CMakeLists.txt -- torchmedia::data (develop_log/2026-10-24/progress03; goldens: gen_golden.py).
foreach(area dataset folder)
    add_executable(data_test_${area} test_${area}.cpp)
    target_link_libraries(data_test_${area} PRIVATE torch_media)
    add_test(NAME data_test_${area} COMMAND data_test_${area})
endforeach()
# The same layer built with LIB_TORCH_MEDIA_NO_EXCEPTIONS: invalid input returns handle_exceptions' default value.
add_executable(data_test_noexcept test_noexcept.cpp)
target_compile_definitions(data_test_noexcept PRIVATE LIB_TORCH_MEDIA_NO_EXCEPTIONS)
target_link_libraries(data_test_noexcept PRIVATE torch_media)
add_test(NAME data_test_noexcept COMMAND data_test_noexcept)
