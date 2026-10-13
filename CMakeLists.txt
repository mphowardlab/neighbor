# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2026, Auburn University
# This file is released under the Modified BSD License.

cmake_minimum_required(VERSION 3.21)

project(neighbor VERSION 0.3.2 LANGUAGES CXX C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# configure C++ compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result -Wconversion")
    if (CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    endif()
endif()

# decide whether to use HIP layer or CUDA (default)
option(NEIGHBOR_HIP "Use HIP layer?" OFF)

# check if neighbor is being used directly or via add_subdirectory
set(NEIGHBOR_MASTER_PROJECT OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NEIGHBOR_MASTER_PROJECT ON)
    if (NEIGHBOR_HIP)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING "Target AMD architectures.")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wno-unused-result -diag-suppress 2810")
        enable_language(HIP)
    else()
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "Target CUDA architectures.")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -diag-suppress 2810")
        enable_language(CUDA)
    endif()
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

# hipper dependency
option(NEIGHBOR_INTERNAL_HIPPER "Use internal copy of hipper?" ON)
if(NOT NEIGHBOR_INTERNAL_HIPPER)
    if(NOT TARGET hipper::hipper)
        find_package(hipper REQUIRED)
    endif()
    target_link_libraries(neighbor INTERFACE hipper::hipper)
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
    add_library(hipper INTERFACE)
    add_library(neighbor::hipper ALIAS hipper)
    target_include_directories(hipper INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/extern/hipper/include>
                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/neighbor>)
    target_link_libraries(neighbor INTERFACE neighbor::hipper)
endif()

# configure HIP CUB dependency
if(NEIGHBOR_HIP)
    find_package(HIPCUB REQUIRED)
    target_link_libraries(neighbor INTERFACE HIP::CUB)
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
                  cmake/FindHIPCUB.cmake
                  cmake/Findhipper.cmake
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)

    install(TARGETS neighbor EXPORT neighborTargets)
    if(NEIGHBOR_INTERNAL_HIPPER)
        install(TARGETS hipper EXPORT neighborTargets)
    endif()

    install(EXPORT neighborTargets
            NAMESPACE neighbor::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/neighbor)
endif()
