project(benchmark)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    myblas::level1
    # myblas::level2
    # myblas::level3
)