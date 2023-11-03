project(benchmark)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME}
    myblas::level1
    myblas::level2
    # myblas::level3
)