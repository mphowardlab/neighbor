# Find the UPP11 test library
#
# The following variables are set if UPP11 is found.
#
# UPP11_FOUND - Set True if UPP11 is found.
# UPP11_INCLUDE_DIRS - Set to UPP11 include directory if found.
#
# Also creates the UPP11::UPP11 target that can be linked against.
#

find_path(UPP11_INCLUDE_DIR NAMES upp11/upp11.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UPP11 REQUIRED_VARS UPP11_INCLUDE_DIR)
mark_as_advanced(UPP11_FOUND CUB_INCLUDE_DIR)

if(UPP11_FOUND)
    set(UPP11_INCLUDE_DIRS ${UPP11_INCLUDE_DIR})

    if(NOT TARGET UPP11::UPP11)
        add_library(UPP11::UPP11 INTERFACE IMPORTED)
        set_target_properties(UPP11::UPP11 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UPP11_INCLUDE_DIR}")
    endif()
endif()
