# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2026, Auburn University
# This file is released under the Modified BSD License.

cmake_minimum_required(VERSION 3.21)

project(neighbor VERSION 0.3.2 LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# language requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# if neighbor is being used directly, configure build options
set(NEIGHBOR_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NEIGHBOR_MASTER_PROJECT ON)

    # c++ build options
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        string(APPEND CMAKE_CXX_FLAGS " -Wall -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result -Wconversion")
        if (CMAKE_COMPILER_IS_GNUCXX)
            string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
        else()
            string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
        endif()
    endif()

    # cuda build options
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "Target CUDA architectures.")
    string(APPEND CMAKE_CUDA_FLAGS " -Wno-unused-result -diag-suppress 2810")
endif()
option(NEIGHBOR_INSTALL "Install neighbor header files?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_TEST "Build neighbor test suite?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_BENCHMARK "Build neighbor benchmarks?" OFF)

# setup the neighbor header-only library
include(GNUInstallDirs)
add_library(neighbor INTERFACE)
add_library(neighbor::neighbor ALIAS neighbor)
target_include_directories(neighbor INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# add tests
if(NEIGHBOR_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# add benchmark
if(NEIGHBOR_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# install headers & cmake package
include(CMakePackageConfigHelpers)
if(NEIGHBOR_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/neighbor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    configure_package_config_file(cmake/neighborConfig.cmake.in
                                  "${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake"
                                  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
                                     VERSION ${neighbor_VERSION}
                                     COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    install(TARGETS neighbor EXPORT neighborTargets)

    install(EXPORT neighborTargets
            NAMESPACE neighbor::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)
endif()
