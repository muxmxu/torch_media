# unit_test/vision/noexcept/CMakeLists.txt -- vision functional / io built with LIB_TORCH_MEDIA_NO_EXCEPTIONS.
add_executable(vision_test_noexcept main.cpp)
target_compile_definitions(vision_test_noexcept PRIVATE LIB_TORCH_MEDIA_NO_EXCEPTIONS)
target_link_libraries(vision_test_noexcept PRIVATE torch_media)
add_test(NAME vision_test_noexcept COMMAND vision_test_noexcept)
