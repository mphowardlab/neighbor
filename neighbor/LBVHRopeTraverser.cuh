// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_ROPE_TRAVERSER_CUH_
#define NEIGHBOR_LBVH_ROPE_TRAVERSER_CUH_

#include "hoomd/HOOMDMath.h"
#include "LBVH.cuh"

namespace neighbor
{
namespace gpu
{

//! Compress LBVH for rope traversal.
void lbvh_compress_ropes(int4 *d_data,
                         float3 *d_lbvh_lo,
                         float3 *d_lbvh_hi,
                         float3 *d_bins,
                         const LBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size);

//! Traverse the LBVH using ropes.
void lbvh_traverse_ropes(unsigned int *d_out,
                         // compressed tree
                         int root,
                         const int4 *d_data,
                         const float3 *d_lbvh_lo,
                         const float3 *d_lbvh_hi,
                         const float3 *d_bins,
                         // traversal spheres
                         const Scalar4 *d_spheres,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int N,
                         unsigned int block_size);

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_ROPE_TRAVERSER_CUH_
