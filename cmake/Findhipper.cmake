# Find the hipper header
#
# The following variables are set if hipper is found.
#
# hipper_FOUND - Set True if hipper is found.
# hipper_INCLUDE_DIRS - Set to hipper include directory if found.
#
# Also creates the hipper::hipper target that can be linked against.
#

# On older CMake, manually search hipper_ROOT first
if(CMAKE_VERSION VERSION_LESS 3.12)
    find_path(hipper_INCLUDE_DIR
              NAMES hipper/hipper_runtime.h
              PATHS ${HIPPER_ROOT} ENV HIPPER_ROOT
              NO_DEFAULT_PATH
              )
endif()

# now look in the system directories if something hasn't been found
# For CMake >= 3.12, this will also check HIPPER_ROOT and ENV{HIPPER_ROOT}
find_path(hipper_INCLUDE_DIR NAMES hipper/hipper_runtime.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hipper REQUIRED_VARS hipper_INCLUDE_DIR)
mark_as_advanced(hipper_FOUND hipper_INCLUDE_DIR)

if(hipper_FOUND)
    set(hipper_INCLUDE_DIRS ${hipper_INCLUDE_DIR})

    if(NOT TARGET hipper::hipper)
        add_library(hipper::hipper INTERFACE IMPORTED)
        set_target_properties(hipper::hipper PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${hipper_INCLUDE_DIR}")
    endif()
endif()
