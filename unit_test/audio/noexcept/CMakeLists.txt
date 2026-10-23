# unit_test/audio/noexcept/CMakeLists.txt -- the functional layer built with LIB_TORCH_MEDIA_NO_EXCEPTIONS.
add_executable(audio_test_noexcept main.cpp)
target_compile_definitions(audio_test_noexcept PRIVATE LIB_TORCH_MEDIA_NO_EXCEPTIONS)
target_link_libraries(audio_test_noexcept PRIVATE torch_media)
add_test(NAME audio_test_noexcept COMMAND audio_test_noexcept)
