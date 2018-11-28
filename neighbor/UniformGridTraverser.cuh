// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_
#define NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_

#include "UniformGrid.cuh"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace neighbor
{
namespace gpu
{

//! Traverse the UniformGrid
void uniform_grid_traverse(unsigned int *d_out,
                           const Scalar4 *d_spheres,
                           const UniformGridData grid,
                           const int3 *d_stencil,
                           const unsigned int num_stencil,
                           const BoxDim& box,
                           const unsigned int N,
                           const unsigned int threads,
                           const unsigned int block_size);

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_
