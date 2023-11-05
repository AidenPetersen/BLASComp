cmake_minimum_required(VERSION 3.22)
project(BLASComp LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(myblas)
add_subdirectory(blas_apis)
add_subdirectory(benchmark)