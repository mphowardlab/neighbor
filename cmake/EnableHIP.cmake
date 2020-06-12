# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

# don't search if HIP has already been found
if(HIP_FOUND)
    return()
endif()

find_package(HIP QUIET)

if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable ROCM_PATH is set to point to the right location.")
endif()
