# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(COMPONENT_NAME neighbor)

##############################
# List of all source files
# c++ source files
set(_cc_sources
    LBVH.cc
    LBVHTraverser.cc
    UniformGrid.cc
    UniformGridTraverser.cc
    )
# cuda source files
set(_cu_sources
    LBVH.cu
    LBVHTraverser.cu
    UniformGrid.cu
    UniformGridTraverser.cu
    )
# headers
set(_h_sources
    BoundingVolumes.h
    InsertOps.h
    LBVH.cuh
    LBVH.h
    LBVHTraverser.cuh
    LBVHTraverser.h
    OutputOps.h
    QueryOps.h
    TransformOps.h
    UniformGrid.cuh
    UniformGrid.h
    UniformGridTraverser.cuh
    UniformGridTraverser.h
    )
###########################

# setup library with cc sources
pybind11_add_module(_${COMPONENT_NAME} SHARED ${_cu_sources} ${_cc_sources} ${_h_sources} NO_EXTRAS)

# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd pybind11::pybind11)

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)

fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}
        )

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
