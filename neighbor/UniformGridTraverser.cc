// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGridTraverser.h"
#include "OutputOps.h"
#include "QueryOps.h"

namespace neighbor
{

template void UniformGridTraverser::traverse(CountNeighborsOp&,
                                             const SphereQueryOp&,
                                             const UniformGrid&,
                                             const GlobalArray<Scalar3>&,
                                             cudaStream_t);

template void UniformGridTraverser::traverse(NeighborListOp&,
                                             const SphereQueryOp&,
                                             const UniformGrid&,
                                             const GlobalArray<Scalar3>&,
                                             cudaStream_t);

} // end namespace neighbor
