project(benchmark)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME}
    blas_apis
)