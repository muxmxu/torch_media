add_executable(audio_test_io main.cpp)
target_link_libraries(audio_test_io PRIVATE torch_media)
add_test(NAME audio_test_io COMMAND audio_test_io)

# TORCHMEDIA_NO_*_IMPLEMENTATION opt-outs: the consumer compiles dr_wav / stb_vorbis itself
add_executable(audio_test_io_optout optout_main.cpp user_codecs.cpp)
target_link_libraries(audio_test_io_optout PRIVATE torch_media)
add_test(NAME audio_test_io_optout COMMAND audio_test_io_optout)
