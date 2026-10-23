# unit_test/audio/compliance/CMakeLists.txt
add_executable(audio_test_compliance main.cpp)

target_link_libraries(audio_test_compliance PRIVATE torch_media)
add_test(NAME audio_test_compliance COMMAND audio_test_compliance)
