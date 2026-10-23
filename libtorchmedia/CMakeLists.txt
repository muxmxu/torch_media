# Header-only library: torch_media
add_library(torch_media INTERFACE)

target_include_directories(torch_media INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_features(torch_media INTERFACE cxx_std_23)
target_link_libraries(torch_media INTERFACE ${TORCH_LIBRARIES})
# torchaudio parity of the scalar loops (phaser, flanger, vad, overdrive, the deemph / riaa designs) needs contraction
# within one expression at most: GCC's GNU-mode default (-ffp-contract=fast) fuses across statements.
target_compile_options(torch_media INTERFACE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=on>)

# Optional plotting module: links matplot++ (gnuplot at runtime). Core stays header-only without this.
if(TORCHMEDIA_WITH_PLOT)
    add_library(torch_media_plot INTERFACE)
    target_link_libraries(torch_media_plot INTERFACE torch_media Matplot++::matplot)
    target_compile_definitions(torch_media_plot INTERFACE TORCHMEDIA_WITH_PLOT)
endif()
