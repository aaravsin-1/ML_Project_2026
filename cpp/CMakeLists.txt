cmake_minimum_required(VERSION 3.20)
project(HighFrequencyLOBEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for ultra-low latency & SIMD
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

include_directories(include)

add_executable(lob_benchmark src/main_benchmark.cpp)
