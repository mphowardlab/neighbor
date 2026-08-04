# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2026, Auburn University
# This file is released under the Modified BSD License.

# Find the HIP CUB library
#
# The following variables are set if hipCUB is found.
#
# HIPCUB_FOUND - Set True if hipCUB is found.
# HIPCUB_INCLUDE_DIRS - Set to hipCUB include directory if found.
#
# Also creates the HIP::CUB target that can be linked against.
#

find_path(HIPCUB_INCLUDE_DIR
          NAMES hipcub/hipcub.hpp
          PATHS ${HIP_ROOT_DIR}
                ENV ROCM_PATH
                /opt/rocm
          PATH_SUFFIXES include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIPCUB REQUIRED_VARS HIPCUB_INCLUDE_DIR)
mark_as_advanced(HIPCUB_FOUND HIPCUB_INCLUDE_DIR)

if(HIPCUB_FOUND)
    set(HIPCUB_INCLUDE_DIRS ${HIPCUB_INCLUDE_DIR})

    if(NOT TARGET HIP::CUB)
        add_library(HIP::CUB INTERFACE IMPORTED)
        set_target_properties(HIP::CUB PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HIPCUB_INCLUDE_DIR}")
    endif()
endif()
