project(blas_apis)

add_library(${PROJECT_NAME} src/cublas.cu src/myblas.cu src/openblas.cu)
target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
    myblas::level1
    myblas::level2
    myblas::level3
)

target_link_libraries(${PROJECT_NAME} -lcublas -lcurand)

find_package(BLAS)

if (BLAS_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE ${BLAS_LINKER_FLAGS})
    target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})
endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
