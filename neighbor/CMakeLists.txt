# Copyright (c) 2018-2019, Michael P. Howard.
# This file is released under the Modified BSD License.

# Maintainer: mphoward

##############################
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
