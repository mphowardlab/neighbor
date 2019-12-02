# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

IF(NOT NEIGHBOR_HOOMD_INTERNAL)
    # check for internal hoomd project
    set(HOOMD_INTERNAL OFF)

    # Name the plugin project
    project(neighbor LANGUAGES C CXX)

    # Find the installed HOOMD
    find_package(HOOMD 2.6.0 REQUIRED)

    message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

    # Force installation to the HOOMD installation location
    set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

    # setup unit tests
    enable_testing()
    option(BUILD_TESTING "Build unit tests" ON)
    if(BUILD_TESTING)
        add_custom_target(test_all ALL)
    endif(BUILD_TESTING)
endif()

# CUDA is required to build this module
if(NOT ENABLE_CUDA)
    message(FATAL_ERROR "Plugins require CUDA-enabled hoomd")
endif()

# add the base directory as an include source (for executables / tests)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add subdirectories
add_subdirectory(neighbor)

#if (NOT NEIGHBOR_HOOMD_INTERNAL)
#    add_subdirectory(benchmark)
#endif()
