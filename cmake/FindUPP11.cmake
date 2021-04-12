# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is released under the Modified BSD License.

# Find the UPP11 test library
#
# The following variables are set if UPP11 is found.
#
# UPP11_FOUND - Set True if UPP11 is found.
# UPP11_INCLUDE_DIRS - Set to UPP11 include directory if found.
#
# Also creates the UPP11::UPP11 target that can be linked against.
#

# On older CMake, manually search UPP11_ROOT first
if(CMAKE_VERSION VERSION_LESS 3.12)
    find_path(UPP11_INCLUDE_DIR
              NAMES upp11/upp11.h
              PATHS ${UPP11_ROOT} $ENV{UPP11_ROOT}
              NO_DEFAULT_PATH
              )
endif()

# now look in the system directories if something hasn't been found
# For CMake >= 3.12, this should check UPP11_ROOT and ENV{UPP11_ROOT} first
find_path(UPP11_INCLUDE_DIR NAMES upp11/upp11.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UPP11 REQUIRED_VARS UPP11_INCLUDE_DIR)
mark_as_advanced(UPP11_FOUND UPP11_INCLUDE_DIR)

if(UPP11_FOUND)
    set(UPP11_INCLUDE_DIRS ${UPP11_INCLUDE_DIR})

    if(NOT TARGET UPP11::UPP11)
        add_library(UPP11::UPP11 INTERFACE IMPORTED)
        set_target_properties(UPP11::UPP11 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UPP11_INCLUDE_DIR}")
    endif()
endif()
