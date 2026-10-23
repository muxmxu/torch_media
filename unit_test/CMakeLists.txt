# unit_test/CMakeLists.txt
# test_util.hpp lives here; make it includable as "test_util.hpp" from any test target.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(audio)
add_subdirectory(vision)
add_subdirectory(metric)
if(TORCHMEDIA_WITH_PLOT)
    add_subdirectory(plot)
endif()
