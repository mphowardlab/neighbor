# Find the HIP CUB library
#
# The following variables are set if hipCUB is found.
#
# HIPCUB_FOUND - Set True if hipCUB is found.
# HIPCUB_INCLUDE_DIRS - Set to hipCUB include directory if found.
#
# Also creates the HIP::CUB target that can be linked against.
#

# On older CMake, manually search HIPCUB_ROOT first
if(CMAKE_VERSION VERSION_LESS 3.12)
    find_path(HIPCUB_INCLUDE_DIR
              NAMES hipcub/hipcub.hpp
              PATHS ${HIPCUB_ROOT} $ENV{HIPCUB_ROOT}
              NO_DEFAULT_PATH
              )
endif()

# now look in the system directories if something hasn't been found
# For CMake >= 3.12, this will also check CUB_ROOT and ENV{CUB_ROOT}
find_path(HIPCUB_INCLUDE_DIR NAMES hipcub/hipcub.hpp)

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
