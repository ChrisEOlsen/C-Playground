cmake_minimum_required(VERSION 3.10)

# Project Name - Start with CXX only, enable CUDA conditionally
project(MonteCarloPi LANGUAGES CXX)

# Generate compile_commands.json for clangd/IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release mode if no build type is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Force C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options to control build targets
option(BUILD_CPU "Build the CPU version (AVX2)" ON)
option(BUILD_CUDA "Build the CUDA version" ON)

# --- CPU Version ---
if(BUILD_CPU)
    message(STATUS "Configuring CPU build...")
    find_package(Threads REQUIRED)
    add_executable(pi_cpu main.cpp)
    target_link_libraries(pi_cpu PRIVATE Threads::Threads)

    # Architecture flags for CPU (AVX2)
    if(MSVC)
        target_compile_options(pi_cpu PRIVATE /arch:AVX2)
        target_compile_options(pi_cpu PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(pi_cpu PRIVATE -march=native -O3)
    endif()
endif()

# --- GPU Version ---
if(BUILD_CUDA)
    message(STATUS "Configuring CUDA build...")
    enable_language(CUDA)
    add_executable(pi_cuda main.cu)

    # Target RTX 40 series (Compute Capability 8.9)
    set_property(TARGET pi_cuda PROPERTY CUDA_ARCHITECTURES 89)

    # Optimization flags for CUDA
    if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
        target_compile_options(pi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    endif()
endif()