// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHTraverser.h"
#include "OutputOps.h"
#include "QueryOps.h"

namespace neighbor
{

template void LBVHTraverser::traverse(CountNeighborsOp& out,
                                      const SphereQueryOp& query,
                                      const LBVH& lbvh,
                                      const GlobalArray<Scalar3>& images);

template void LBVHTraverser::traverse(NeighborListOp& out,
                                      const SphereQueryOp& query,
                                      const LBVH& lbvh,
                                      const GlobalArray<Scalar3>& images);

} // end namespace neighbor
