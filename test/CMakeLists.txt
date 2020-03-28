# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(TEST_LIST
    approx_math_test.cu
    lbvh_test.cu
    )

# tests require CUB and UPP11 to compile
find_package(UPP11 REQUIRED)

foreach(TEST_SRC ${TEST_LIST})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE neighbor::neighbor UPP11::UPP11)
    if(NEIGHBOR_CUDA_ARCH)
        target_compile_options(${TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${NEIGHBOR_CUDA_ARCH},code=compute_${NEIGHBOR_CUDA_ARCH}>)
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
