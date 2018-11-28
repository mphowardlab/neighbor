// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_CUH_
#define NEIGHBOR_LBVH_CUH_

#include "hoomd/HOOMDMath.h"

namespace neighbor
{
namespace gpu
{

// LBVH sentinel has value of max signed int (~2 billion)
const int LBVHSentinel=0x7fffffff;

//! Linear bounding volume hierarchy raw data
/*!
 * LBVHData is a lightweight struct representation of the LBVH. It is useful for passing tree data
 * to a CUDA kernel. It is valid to set a pointer to NULL if it is not required, but the caller
 * for doing so responsibly.
 */
struct LBVHData
    {
    int* parent;                        //!< Parent node
    int* left;                          //!< Left child
    int* right;                         //!< Right child
    const unsigned int* primitive;      //!< Primitives
    float3* lo;                         //!< Lower bound of AABB
    float3* hi;                         //!< Upper bound of AABB
    int root;                           //!< Root index
    };

//! Generate the Morton codes
void lbvh_gen_codes(unsigned int *d_codes,
                    unsigned int *d_indexes,
                    const Scalar4 *d_points,
                    const Scalar3 lo,
                    const Scalar3 hi,
                    const unsigned int N,
                    const unsigned int block_size);

//! Sort the Morton codes.
uchar2 lbvh_sort_codes(void *d_tmp,
                       size_t &tmp_bytes,
                       unsigned int *d_codes,
                       unsigned int *d_sorted_codes,
                       unsigned int *d_indexes,
                       unsigned int *d_sorted_indexes,
                       const unsigned int N);

//! Generate the tree hierarchy
void lbvh_gen_tree(const LBVHData tree,
                   const unsigned int *d_codes,
                   const unsigned int N,
                   const unsigned int block_size);

//! Bubble the bounding boxes up the tree hierarchy.
void lbvh_bubble_aabbs(const LBVHData tree,
                       unsigned int *d_locks,
                       const Scalar4 *d_points,
                       const unsigned int N,
                       const unsigned int block_size);


} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_CUH_
