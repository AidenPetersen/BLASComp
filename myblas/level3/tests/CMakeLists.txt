project(tests)

# all test declarations
list(APPEND TESTS test-sgemm)
list(APPEND TESTS test-strsm)


foreach(test ${TESTS})
    add_executable(${test} ${test}.cu)
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${test} PRIVATE myblas::level3)
    add_test(NAME ${test} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${test})
endforeach ()