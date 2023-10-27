cmake_minimum_required(VERSION 3.26)
project(BLASComp LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_C_COMPILER "/home/linuxbrew/.linuxbrew/bin/gcc-12")

add_subdirectory(myblas)
add_subdirectory(benchmark)