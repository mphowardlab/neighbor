# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

# check for internal hoomd project
set(HOOMD_INTERNAL OFF)
if (PROJECT_NAME STREQUAL "HOOMD")
    set(HOOMD_INTERNAL ON)
else()
    project(neighbor)

    # set cmake policies consistent with HOOMD
    CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)
    if(CMAKE_MAJOR_VERSION VERSION_GREATER 2)
        if(COMMAND cmake_policy)
        cmake_policy(SET CMP0003 NEW)
        cmake_policy(SET CMP0042 NEW)
        endif(COMMAND cmake_policy)
    endif(CMAKE_MAJOR_VERSION VERSION_GREATER 2)

    # find HOOMD
    include(FindHOOMD.cmake)
    # try to deduce the hoomd version, since FindHOOMD does not support versioning now.
    set(find_hoomd_version "
from __future__ import print_function
import sys,os
sys.stdout = open(os.devnull, 'w')
import hoomd
print(hoomd.__version__, file=sys.stderr, end='')")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_hoomd_version}" ERROR_VARIABLE hoomd_version)
    if(hoomd_version MATCHES "^Traceback*")
        message(STATUS "HOOMD version: not deduced, ensure that it is >= 2.4.1.")
    else()
        message(STATUS "HOOMD version: " ${hoomd_version})
        if(hoomd_version VERSION_LESS "2.4.1")
        message(FATAL_ERROR "HOOMD version must be >= 2.4.1")
        endif(hoomd_version VERSION_LESS "2.4.1")
    endif(hoomd_version MATCHES "^Traceback*")

    # plugins must be built as shared libraries
    if(ENABLE_STATIC)
        message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
    endif(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS on)

    # CUDA is required to build this module
    if(NOT ENABLE_CUDA)
        message(FATAL_ERROR "Plugins require CUDA-enabled hoomd")
    endif(NOT ENABLE_CUDA)

    # setup unit tests
    enable_testing()
    option(BUILD_TESTING "Build unit tests" ON)
    if(BUILD_TESTING)
        add_custom_target(test_all ALL)
    endif(BUILD_TESTING)

    # use full rpath on targets, regardless of installation location
    # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif("${isSystemDir}" STREQUAL "-1")
endif ()

# add the base directory as an include source (for executables / tests)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add subdirectories
add_subdirectory(neighbor)
if (NOT HOOMD_INTERNAL)
    add_subdirectory(benchmark)
endif()
