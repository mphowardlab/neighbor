// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHTraverser.cuh"
#include "OutputOps.h"
#include "QueryOps.h"

namespace neighbor
{
namespace gpu
{
namespace kernel
{

//! Kernel to compress LBVH for rope traversal
/*!
 * \param ctree Compressed LBVH.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 *
 * The bounding boxes and hierarchy of the LBVH are compressed into
 * (1) int4 / node. Each node holds the compressed bounds (2 ints),
 * the left child of the node (or primitive), and the rope to advance ahead.
 * The ropes are generated in this kernel by backtracking. The compression
 * converts the float bounds of the box into a 10-bit integer for each
 * component. The output \a bins size for the compression is done in a
 * conservative way so that on decompression, the bounds of the nodes are
 * never underestimated.
 */
__global__ void lbvh_compress_ropes(LBVHCompressedData ctree,
                                    const LBVHData tree,
                                    const unsigned int N_internal,
                                    const unsigned int N_nodes)
    {
    // one thread per node
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (int)N_nodes)
        return;

    // load the tree extent for meshing
    __shared__ float3 tree_lo, tree_hi, tree_bininv;
    if (threadIdx.x == 0)
        {
        tree_lo = tree.lo[tree.root];
        tree_hi = tree.hi[tree.root];
        // compute box size, rounding up to ensure fully covered
        float3 L = make_float3(__fsub_ru(tree_hi.x, tree_lo.x),
                               __fsub_ru(tree_hi.y, tree_lo.y),
                               __fsub_ru(tree_hi.z, tree_lo.z));
        if (L.x <= 0.f) L.x = 1.0f;
        if (L.y <= 0.f) L.y = 1.0f;
        if (L.z <= 0.f) L.z = 1.0f;

        // round down the bin scale factor so that it always *underestimates* the offset
        tree_bininv = make_float3(__fdiv_rd(1023.f,L.x),
                                  __fdiv_rd(1023.f,L.y),
                                  __fdiv_rd(1023.f,L.z));
        }
    __syncthreads();

    // backtrack tree to find the first right ancestor of this node
    int rope = LBVHSentinel;
    int current = idx;
    while (current != tree.root && rope == LBVHSentinel)
        {
        int parent = tree.parent[current];
        int left = tree.left[parent];
        if (left == current)
            {
            // if this is a left node, then rope is determined to the right
            rope = tree.right[parent];
            }
        else
            {
            // otherwise, keep ascending the tree
            current = parent;
            }
        }

    // compress node data into one byte per box dim
    // low bounds are encoded relative to the low of the box, always rounding down
    const float3 lo = tree.lo[idx];
    const uint3 lo_bin = make_uint3((unsigned int)floorf(__fmul_rd(__fsub_rd(lo.x,tree_lo.x),tree_bininv.x)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(lo.y,tree_lo.y),tree_bininv.y)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(lo.z,tree_lo.z),tree_bininv.z)));
    const unsigned int lo_bin3 = (lo_bin.x << 20) +  (lo_bin.y << 10) + lo_bin.z;

    // high bounds are encoded relative to the high of the box, always rounding down
    const float3 hi = tree.hi[idx];
    const uint3 hi_bin = make_uint3((unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.x,hi.x),tree_bininv.x)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.y,hi.y),tree_bininv.y)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.z,hi.z),tree_bininv.z)));
    const unsigned int hi_bin3 = (hi_bin.x << 20) + (hi_bin.y << 10) + hi_bin.z;

    // node holds left child for internal nodes (>= 0) or primitive for leaf (< 0)
    int left_flag = (idx < N_internal) ? tree.left[idx] : ~tree.primitive[idx-N_internal];

    // stash all the data into one int4
    ctree.data[idx] = make_int4(lo_bin3, hi_bin3, left_flag, rope);

    // first thread writes out the compression values, rounding down bin size to ensure box bounds always expand even with floats
    if (idx == 0)
        {
        *ctree.lo = tree_lo;
        *ctree.hi = tree_hi;
        *ctree.bins = make_float3(__frcp_rd(tree_bininv.x),__frcp_rd(tree_bininv.y),__frcp_rd(tree_bininv.z));
        }
    }

} // end namespace kernel

/*!
 * \param ctree Compressed LBVH.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::lbvh_compress_ropes
 */
void lbvh_compress_ropes(LBVHCompressedData ctree,
                         const LBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_compress_ropes);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N_nodes + run_block_size - 1)/run_block_size;
    kernel::lbvh_compress_ropes<<<num_blocks, run_block_size>>>
        (ctree, tree, N_internal, N_nodes);
    }

// template declaration to count neighbors
template void lbvh_traverse_ropes(CountNeighborsOp& out,
                                  const LBVHCompressedData& lbvh,
                                  const SphereQueryOp& query,
                                  const Scalar3 *d_images,
                                  unsigned int Nimages,
                                  unsigned int block_size);

// template declaration to generate neighbor list
template void lbvh_traverse_ropes(NeighborListOp& out,
                                  const LBVHCompressedData& lbvh,
                                  const SphereQueryOp& query,
                                  const Scalar3 *d_images,
                                  unsigned int Nimages,
                                  unsigned int block_size);

} // end namespace gpu
} // end namespace neighbor
