# Find the HIP CUB library
#
# The following variables are set if CUB is found.
#
# CUB_FOUND - Set True if CUB is found.
# CUB_INCLUDE_DIRS - Set to CUB include directory if found.
#
# Also creates the HIP::CUB target that can be linked against.
#

# On older CMake, manually search CUB_ROOT first
if(CMAKE_VERSION VERSION_LESS 3.12)
    find_path(CUB_INCLUDE_DIR
              NAMES hipcub/hipcub.hpp
              PATHS ${CUB_ROOT} $ENV{CUB_ROOT}
              NO_DEFAULT_PATH
              )
endif()

# now look in the system directories if something hasn't been found
# For CMake >= 3.12, this will also check CUB_ROOT and ENV{CUB_ROOT}
find_path(CUB_INCLUDE_DIR NAMES hipcub/hipcub.hpp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB REQUIRED_VARS CUB_INCLUDE_DIR)
mark_as_advanced(CUB_FOUND CUB_INCLUDE_DIR)

if(CUB_FOUND)
    set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

    if(NOT TARGET HIP::CUB)
        add_library(HIP::CUB INTERFACE IMPORTED)
        set_target_properties(HIP::CUB PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUB_INCLUDE_DIR}")
    endif()
endif()
