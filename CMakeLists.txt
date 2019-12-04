# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

cmake_minimum_required(VERSION 3.9)

project(neighbor VERSION 0.3.0 LANGUAGES CXX CUDA)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# check if neighbor is being used directly or via add_subdirectory
# this sets the default options for whether headers, tests, and benchmarks should be used
set(NEIGHBOR_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NEIGHBOR_MASTER_PROJECT ON)
endif()
option(NEIGHBOR_INSTALL "Install neighbor header files?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_TEST "Build neighbor test suite?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_BENCHMARK "Build neighbor benchmarks?" ${NEIGHBOR_MASTER_PROJECT})
set(NEIGHBOR_CUDA_ARCH "35" CACHE STRING "Target CUDA architecture for compiled objects.")

# find CUB and make it a dependency
find_package(CUB REQUIRED)

# setup the neighbor header-only library
include(GNUInstallDirs)
set(NEIGHBOR_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
add_library(neighbor INTERFACE)
add_library(neighbor::neighbor ALIAS neighbor)
target_include_directories(neighbor INTERFACE $<BUILD_INTERFACE:${NEIGHBOR_INCLUDE_DIR}>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(neighbor INTERFACE CUB::CUB)

# add tests
if(NEIGHBOR_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# add benchmark
#if(NEIGHBOR_BENCHMARK)
#    add_subdirectory(benchmark)
#endif()

# install headers & cmake package
include(CMakePackageConfigHelpers)
if(NEIGHBOR_INSTALL)
    install(DIRECTORY ${NEIGHBOR_INCLUDE_DIR}/neighbor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    configure_package_config_file(cmake/neighborConfig.cmake.in
                                  "${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake"
                                  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
                                     VERSION ${neighbor_VERSION}
                                     COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
                  cmake/FindCUB.cmake
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    install(TARGETS neighbor EXPORT neighborTargets)

    install(EXPORT neighborTargets
            NAMESPACE neighbor::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)
endif()
