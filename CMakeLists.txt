# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

cmake_minimum_required(VERSION 3.9)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    cmake_policy(SET CMP0074 NEW)
endif()

project(neighbor VERSION 0.3.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# decide whether to use HIP layer or CUDA (default)
option(NEIGHBOR_HIP "Use HIP layer?" OFF)

# check if neighbor is being used directly or via add_subdirectory
set(NEIGHBOR_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NEIGHBOR_MASTER_PROJECT ON)
    set(NEIGHBOR_CUDA_ARCH "35;60;70" CACHE STRING "Target CUDA architecture.")
    set(NEIGHBOR_AMD_ARCH "gfx900;gfx906;gfx908" CACHE STRING "Target AMD architecture.")
endif()
option(NEIGHBOR_INSTALL "Install neighbor header files?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_TEST "Build neighbor test suite?" ${NEIGHBOR_MASTER_PROJECT})
option(NEIGHBOR_BENCHMARK "Build neighbor benchmarks?" ${NEIGHBOR_MASTER_PROJECT})

# setup the neighbor header-only library
include(GNUInstallDirs)
add_library(neighbor INTERFACE)
add_library(neighbor::neighbor ALIAS neighbor)
target_include_directories(neighbor INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# hipper dependency
option(NEIGHBOR_INTERNAL_HIPPER "Use internal copy of hipper?" ON)
add_library(hipper INTERFACE)
add_library(neighbor::hipper ALIAS hipper)
if(NOT NEIGHBOR_INTERNAL_HIPPER)
    if(NOT TARGET hipper::hipper)
        find_package(hipper REQUIRED)
    endif()
    target_link_libraries(hipper INTERFACE hipper::hipper)
else()
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/.git")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule --quiet sync
                        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    else()
        message(FATAL_ERROR "Unable to checkout internal copy of hipper using git submodule.")
    endif()
    target_include_directories(hipper INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/extern/hipper/include>
                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/neighbor>)
endif()
target_link_libraries(neighbor INTERFACE neighbor::hipper)

# cub dependency
if(NEIGHBOR_HIP)
    find_package(HIPCUB REQUIRED)
    target_link_libraries(neighbor INTERFACE HIP::CUB)
    target_compile_definitions(neighbor INTERFACE HIPPER_HIP)
else()
    find_package(CUB REQUIRED)
    target_link_libraries(neighbor INTERFACE CUDA::CUB)
    target_compile_definitions(neighbor INTERFACE HIPPER_CUDA)
endif()

# add tests
if(NEIGHBOR_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# add benchmark
if(NOT NEIGHBOR_HIP AND NEIGHBOR_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# install headers & cmake package
include(CMakePackageConfigHelpers)
if(NEIGHBOR_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/neighbor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    if(NEIGHBOR_INTERNAL_HIPPER)
        install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/extern/hipper/include/hipper DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neighbor)
    endif()

    configure_package_config_file(cmake/neighborConfig.cmake.in
                                  "${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake"
                                  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
                                     VERSION ${neighbor_VERSION}
                                     COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/neighborConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/neighborConfigVersion.cmake
                  cmake/FindCUB.cmake
                  cmake/FindHIPCUB.cmake
                  cmake/Findhipper.cmake
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    install(TARGETS neighbor hipper EXPORT neighborTargets)

    install(EXPORT neighborTargets
            NAMESPACE neighbor::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)
endif()
