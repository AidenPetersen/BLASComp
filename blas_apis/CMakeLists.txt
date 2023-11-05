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
# set(BLA_VENDER OpenBLAS)
find_package(BLAS)


