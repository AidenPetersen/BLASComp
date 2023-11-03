project(tests)

# all test declarations
list(APPEND TESTS test-sgemv)
list(APPEND TESTS test-sgbmv)
list(APPEND TESTS test-ssymv)
list(APPEND TESTS test-ssbmv)
list(APPEND TESTS test-stbmv)


foreach(test ${TESTS})
    add_executable(${test} ${test}.cu)
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${test} PRIVATE myblas::level2)
    add_test(NAME ${test} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${test})
endforeach ()