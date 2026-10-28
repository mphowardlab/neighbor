# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2026, Auburn University
# This file is released under the Modified BSD License.

set(TEST_LIST
    approx_math_test.cu
    lbvh_test.cu
    )

# tests require UPP11
find_package(UPP11 QUIET)
if(NOT UPP11_Found)
    message(STATUS "Downloading UPP11...")
    set(url https://raw.githubusercontent.com/DronMDF/upp11/master/upp11.h)
    file(DOWNLOAD ${url} "${CMAKE_CURRENT_BINARY_DIR}/upp11/upp11.h" STATUS status)
    list(GET status 0 error)
    if(error)
        message(FATAL_ERROR "... could not download from ${url}.")
    endif()
    set(UPP11_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
    find_package(UPP11 REQUIRED)
endif()

foreach(TEST_SRC ${TEST_LIST})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE neighbor::neighbor UPP11::UPP11)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # also test without intrinsics
    add_executable(${TEST_NAME}_nointrinsic ${TEST_SRC})
    target_compile_definitions(${TEST_NAME}_nointrinsic PRIVATE NEIGHBOR_NO_INTRINSIC_ROUND)
    target_link_libraries(${TEST_NAME}_nointrinsic PRIVATE neighbor::neighbor UPP11::UPP11)
    add_test(NAME ${TEST_NAME}_nointrinsic COMMAND ${TEST_NAME}_nointrinsic)
endforeach()
