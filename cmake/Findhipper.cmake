# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is released under the Modified BSD License.

#[=======================================================================[.rst:
Findhipper
----------

Find the hipper GPU runtime library interface.

The location of the library can be hinted using ``hipper_ROOT`` or ``ENV{hipper_ROOT}``.

The following variables are set:

``hipper_FOUND``
  Variable indicating if hipper is found.
``hipper_INCLUDE_DIRS``
  Include path(s) for hipper header.

The following :prop_tgt:`IMPORTED` targets are defined:

``hipper::hipper``
  Target for hipper library.

#]=======================================================================]

if(hipper_FOUND)
    return()
endif()

# try to find hipper runtime header
find_path(hipper_INCLUDE_DIR
          NAMES hipper/hipper_runtime.h
          PATHS ${hipper_ROOT} ENV hipper_ROOT
          PATH_SUFFIXES include
          NO_DEFAULT_PATH
          )
find_path(hipper_INCLUDE_DIR
          NAMES hipper/hipper_runtime.h
          PATH_SUFFIXES include)
mark_as_advanced(hipper_INCLUDE_DIR)

# process package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hipper REQUIRED_VARS hipper_INCLUDE_DIR)
set(hipper_INCLUDE_DIRS ${hipper_INCLUDE_DIR})

# make an imported target available
if(hipper_FOUND AND NOT TARGET hipper::hipper)
    add_library(hipper::hipper INTERFACE IMPORTED)
    set_target_properties(hipper::hipper PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${hipper_INCLUDE_DIRS}")
endif()
