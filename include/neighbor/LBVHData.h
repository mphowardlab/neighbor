// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_DATA_H_
#define NEIGHBOR_LBVH_DATA_H_

#include "hipper_runtime.h"

namespace neighbor
{

// LBVH sentinel has value of max signed int (~2 billion)
const int LBVHSentinel=0x7fffffff;

//! Linear bounding volume hierarchy raw data, writeable.
/*!
 * LBVHData is a lightweight struct representation of the LBVH. It is useful for passing tree data
 * to a CUDA kernel. It is valid to set a pointer to NULL if it is not required, but the caller
 * for doing so responsibly.
 */
struct LBVHData
    {
    int* parent;                //!< Parent node
    int* left;                  //!< Left child
    int* right;                 //!< Right child
    unsigned int* primitive;    //!< Primitives
    float3* lo;                 //!< Lower bound of AABB
    float3* hi;                 //!< Upper bound of AABB
    int root;                   //!< Root index
    };

//! Linear bounding volume hierarchy raw data, read only.
struct ConstLBVHData
    {
    const int* parent;              //!< Parent node
    const int* left;                //!< Left child
    const int* right;               //!< Right child
    const unsigned int* primitive;  //!< Primitives
    const float3* lo;               //!< Lower bound of AABB
    const float3* hi;               //!< Upper bound of AABB
    int root;                       //!< Root index
    };

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_DATA_H_
