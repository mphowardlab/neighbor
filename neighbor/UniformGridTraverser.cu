// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGridTraverser.cuh"
#include "OutputOps.h"
#include "QueryOps.h"
#include "TransformOps.h"

namespace neighbor
{
namespace gpu
{

// template declaration for compressing without transforming primitives
template void uniform_grid_compress(const UniformGridCompressedData&,
                                    const NullTransformOp&,
                                    const UniformGridData&,
                                    const unsigned int,
                                    const unsigned int,
                                    const unsigned int,
                                    cudaStream_t);

// template declaration for compressing with map transformation of primitives
template void uniform_grid_compress(const UniformGridCompressedData&,
                                    const MapTransformOp&,
                                    const UniformGridData&,
                                    const unsigned int,
                                    const unsigned int,
                                    const unsigned int,
                                    cudaStream_t);

// template declaration to count neighbors
template void uniform_grid_traverse(const CountNeighborsOp& out,
                                    const UniformGridCompressedData& lbvh,
                                    const SphereQueryOp& query,
                                    const Scalar3 *d_images,
                                    unsigned int Nimages,
                                    unsigned int block_size,
                                    cudaStream_t stream);

// template declaration to generate neighbor list
template void uniform_grid_traverse(const NeighborListOp& out,
                                    const UniformGridCompressedData& lbvh,
                                    const SphereQueryOp& query,
                                    const Scalar3 *d_images,
                                    unsigned int Nimages,
                                    unsigned int block_size,
                                    cudaStream_t stream);

} // end namespace gpu
} // end namespace neighbor
