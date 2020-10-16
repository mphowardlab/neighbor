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

# configure CUDA options
set(TEST_NVCC_OPTIONS )
if(NEIGHBOR_CUDA_ARCH)
    # sorted arch list
    set(_sorted_arches ${NEIGHBOR_CUDA_ARCH})
    list(SORT _sorted_arches)
    list(REVERSE _sorted_arches)
    list(GET _sorted_arches 0 _max_arch)

    # on modern cmake with CUDA, use the simpler way of setting the archs
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18 AND NOT NEIGHBOR_HIP)
        set(CMAKE_CUDA_ARCHITECTURES "")
        foreach(_arch ${NEIGHBOR_CUDA_ARCH})
            list(APPEND CMAKE_CUDA_ARCHITECTURES "${_arch}-real")
        endforeach()
        list(APPEND CMAKE_CUDA_ARCHITECTURES "${_max_arch}-virtual")
    else()
        foreach(_arch ${NEIGHBOR_CUDA_ARCH})
            set(TEST_NVCC_OPTIONS "${TEST_NVCC_OPTIONS} -gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        set(TEST_NVCC_OPTIONS "${TEST_NVCC_OPTIONS} -gencode=arch=compute_${_max_arch},code=compute_${_max_arch}")
        if(NOT NEIGHBOR_HIP)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TEST_NVCC_OPTIONS}")
        endif()
    endif()
endif()

# configure AMD options
set(TEST_HIPCC_OPTIONS )
if(NEIGHBOR_AMD_ARCH)
    foreach(_arch ${NEIGHBOR_AMD_ARCH})
        set(TEST_HIPCC_OPTIONS "--amdgpu-target=${_arch}")
    endforeach()
endif()

foreach(TEST_SRC ${TEST_LIST})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    # test compiled under HIP or CUDA
    if(NEIGHBOR_HIP)
        hip_add_executable(${TEST_NAME} ${TEST_SRC} HIPCC_OPTIONS ${TEST_HIPCC_OPTIONS} NVCC_OPTIONS ${TEST_NVCC_OPTIONS})
    else()
        add_executable(${TEST_NAME} ${TEST_SRC})
    endif()
    target_link_libraries(${TEST_NAME} PRIVATE neighbor::neighbor UPP11::UPP11)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    # under CUDA, also force the test without intrinsics
    if(NOT NEIGHBOR_HIP)
        add_executable(${TEST_NAME}_nointrinsic ${TEST_SRC})
        target_compile_definitions(${TEST_NAME}_nointrinsic PRIVATE NEIGHBOR_NO_INTRINSIC_ROUND)
        target_link_libraries(${TEST_NAME}_nointrinsic PRIVATE neighbor::neighbor UPP11::UPP11)
        add_test(NAME ${TEST_NAME}_nointrinsic COMMAND ${TEST_NAME}_nointrinsic)
    endif()
endforeach()
