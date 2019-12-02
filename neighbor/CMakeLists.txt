# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

set(COMPONENT_NAME neighbor)

##############################
# List of all source files
# c++ source files
set(_cc_sources
    )
#    UniformGrid.cc
#    UniformGridTraverser.cc
# cuda source files
set(_cu_sources
    )
#    UniformGrid.cu
#    UniformGridTraverser.cu
# headers
set(_h_sources
    BoundingVolumes.h
    InsertOps.h
    kernels/LBVHKernels.cuh
    LBVH.h
    kernels/LBVHTraverserKernels.cuh
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

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
