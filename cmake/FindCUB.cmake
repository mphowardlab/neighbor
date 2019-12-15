# Find the NVIDIA CUB library
#
# The following variables are set if CUB is found.
#
# CUB_FOUND - Set True if CUB is found.
# CUB_INCLUDE_DIRS - Set to CUB include directory if found.
#
# Also creates the CUB::CUB target that can be linked against.
#

find_path(CUB_INCLUDE_DIR
          NAMES cub/cub.cuh
          PATHS ${CUB_ROOT} $ENV{CUB_ROOT}
          )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB REQUIRED_VARS CUB_INCLUDE_DIR)
mark_as_advanced(CUB_FOUND CUB_INCLUDE_DIR)

if(CUB_FOUND)
    set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

    if(NOT TARGET CUB::CUB)
        add_library(CUB::CUB INTERFACE IMPORTED)
        set_target_properties(CUB::CUB PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUB_INCLUDE_DIR}")
    endif()
endif()
