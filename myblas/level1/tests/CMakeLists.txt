project(tests)

# all test declarations
list(APPEND TESTS test-sscal)
list(APPEND TESTS test-srotg)
list(APPEND TESTS test-sswap)
list(APPEND TESTS test-saxpy)
list(APPEND TESTS test-sdot)
list(APPEND TESTS test-snrm2)
list(APPEND TESTS test-sasum)
list(APPEND TESTS test-samax)
list(APPEND TESTS test-samin)


foreach(test ${TESTS})
    add_executable(${test} ${test}.cu)
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${test} PRIVATE myblas::level1)
    add_test(NAME ${test} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${test})
endforeach ()