project(level2)

add_library(${PROJECT_NAME} 
    src/level2.cu
)

add_library(myblas::level2 ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


enable_testing()
add_subdirectory(tests)