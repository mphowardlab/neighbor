# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(TEST_LIST
    approx_math_test.cu
    lbvh_test.cu
    )

# setup language
if(NEIGHBOR_HIP)
    include(EnableHIP)
else()
    enable_language(CUDA)
endif()

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

    if(NEIGHBOR_HIP)
        set(TEST_NVCC_OPTIONS )
        if(NEIGHBOR_CUDA_ARCH)
            set(TEST_NVCC_OPTIONS "-gencode=arch=compute_${NEIGHBOR_CUDA_ARCH},code=compute_${NEIGHBOR_CUDA_ARCH}")
        endif()
        set(TEST_HIPCC_OPTIONS )
        foreach(_arch ${NEIGHBOR_AMD_ARCH})
            set(TEST_HIPCC_OPTIONS "--amdgpu-target=${_arch}")
        endforeach()
        hip_add_executable(${TEST_NAME} ${TEST_SRC} HIPCC_OPTIONS ${TEST_HIPCC_OPTIONS} NVCC_OPTIONS ${TEST_NVCC_OPTIONS})
    else()
        add_executable(${TEST_NAME} ${TEST_SRC})
        if(NEIGHBOR_CUDA_ARCH)
            target_compile_options(${TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${NEIGHBOR_CUDA_ARCH},code=compute_${NEIGHBOR_CUDA_ARCH}>)
        endif()
    endif()
    target_link_libraries(${TEST_NAME} PRIVATE neighbor::neighbor UPP11::UPP11)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
