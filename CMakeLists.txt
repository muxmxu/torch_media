cmake_minimum_required(VERSION 3.20)
project(torch_media LANGUAGES CXX)

# --- language standard (C++23: the headers use <print> / <format>) ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Bundled dependencies live under dependence/ (fetched by script/download_dep).
if(POLICY CMP0126)
    cmake_policy(SET CMP0126 NEW) # a parent project's plain set(TORCHMEDIA_DEPENDENCE_DIR ...) beats the cache default
endif()
set(TORCHMEDIA_DEPENDENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependence"
    CACHE PATH "Directory holding bundled dependencies (libtorch, matplotplusplus)")

# --- libtorch (bundled under dependence/libtorch) ---
list(PREPEND CMAKE_PREFIX_PATH "${TORCHMEDIA_DEPENDENCE_DIR}/libtorch")
find_package(Torch REQUIRED)
if(TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# Make the bundled libtorch dylibs discoverable at runtime without setting DYLD/LD_LIBRARY_PATH.
list(APPEND CMAKE_BUILD_RPATH "${TORCHMEDIA_DEPENDENCE_DIR}/libtorch/lib")

# --- coverage (LLVM source-based; opt-in) ---
option(TORCHMEDIA_COVERAGE "Enable LLVM source-based coverage instrumentation" OFF)
if(TORCHMEDIA_COVERAGE)
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

# --- optional plotting backend (matplot++ / gnuplot; opt-in) ---
option(TORCHMEDIA_WITH_PLOT "Build the optional torchmedia::plot module (matplot++ / gnuplot)" OFF)
if(TORCHMEDIA_WITH_PLOT)
    add_subdirectory(${TORCHMEDIA_DEPENDENCE_DIR}/matplotplusplus matplotplusplus)
endif()

# --- header-only library target ---
add_subdirectory(libtorchmedia)

# --- unit tests (double as usage examples) ---
option(TORCHMEDIA_BUILD_TESTS "Build unit tests" ON)
if(TORCHMEDIA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(unit_test)
endif()
