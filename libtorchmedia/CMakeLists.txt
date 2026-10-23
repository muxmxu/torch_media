# Header-only library: torch_media
add_library(torch_media INTERFACE)

target_include_directories(torch_media INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_features(torch_media INTERFACE cxx_std_23)
target_link_libraries(torch_media INTERFACE ${TORCH_LIBRARIES})

# Optional plotting module: links matplot++ (gnuplot at runtime). Core stays header-only without this.
if(TORCHMEDIA_WITH_PLOT)
    add_library(torch_media_plot INTERFACE)
    target_link_libraries(torch_media_plot INTERFACE torch_media Matplot++::matplot)
    target_compile_definitions(torch_media_plot INTERFACE TORCHMEDIA_WITH_PLOT)
endif()
