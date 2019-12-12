// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_DATA_H_
#define NEIGHBOR_LBVH_TRAVERSER_DATA_H_

#include <cuda_runtime.h>

namespace neighbor
{

//! Lightweight data structure to hold the compressed LBVH.
struct LBVHCompressedData
    {
    int root;       //!< Root index of the LBVH
    int4* data;     //!< Compressed LBVH data.
    float3* lo;     //!< Lower bound used in compression.
    float3* hi;     //!< Upper bound used in compression.
    float3* bins;   //!< Bin spacing used in compression.
    };

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_DATA_H_
