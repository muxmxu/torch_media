# unit_test/audio/separation/CMakeLists.txt
add_executable(audio_test_separation main.cpp)

target_link_libraries(audio_test_separation PRIVATE torch_media)
add_test(NAME audio_test_separation COMMAND audio_test_separation)
