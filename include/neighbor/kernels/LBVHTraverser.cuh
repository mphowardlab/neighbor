// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_CUH_
#define NEIGHBOR_LBVH_TRAVERSER_CUH_

#include <hipper/hipper_runtime.h>

#include "../ApproximateMath.h"
#include "../LBVHData.h"
#include "../LBVHTraverserData.h"
#include "../BoundingVolumes.h"

namespace neighbor
{
namespace gpu
{
namespace kernel
{
//! Kernel to compress LBVH for rope traversal
/*!
 * \param ctree Compressed LBVH.
 * \param transform Transformation operation.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 *
 * \tparam TransformOpT Type of operation for transforming cached primitive index.
 *
 * The bounding boxes and hierarchy of the LBVH are compressed into
 * (1) int4 / node. Each node holds the compressed bounds (2 ints),
 * the left child of the node (or primitive), and the rope to advance ahead.
 * The ropes are generated in this kernel by backtracking. The compression
 * converts the float bounds of the box into a 10-bit integer for each
 * component. The output \a bins size for the compression is done in a
 * conservative way so that on decompression, the bounds of the nodes are
 * never underestimated.
 *
 * The stored primitive may be transformed to a new value for more efficient caching for traversal.
 * The transformation is implemented by \a transform.
 */
template<class TransformOpT>
__global__ void lbvh_compress_ropes(const LBVHCompressedData ctree,
                                    const TransformOpT transform,
                                    const ConstLBVHData tree,
                                    const unsigned int N_internal,
                                    const unsigned int N_nodes)
    {
    // one thread per node
    const int idx = hipper::threadRank<1,1>();
    if (idx >= (int)N_nodes)
        return;

    // load the tree extent for meshing
    __shared__ float3 tree_lo, tree_hi, tree_bininv;
    if (threadIdx.x == 0)
        {
        tree_lo = tree.lo[tree.root];
        tree_hi = tree.hi[tree.root];
        // compute box size, rounding up to ensure fully covered
        float3 L = make_float3(approx::fsub_ru(tree_hi.x, tree_lo.x),
                               approx::fsub_ru(tree_hi.y, tree_lo.y),
                               approx::fsub_ru(tree_hi.z, tree_lo.z));
        if (L.x <= 0.f) L.x = 1.0f;
        if (L.y <= 0.f) L.y = 1.0f;
        if (L.z <= 0.f) L.z = 1.0f;

        // round down the bin scale factor so that it always *underestimates* the offset
        tree_bininv = make_float3(approx::fdiv_rd(1023.f,L.x),
                                  approx::fdiv_rd(1023.f,L.y),
                                  approx::fdiv_rd(1023.f,L.z));
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
    const uint3 lo_bin = make_uint3((unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(lo.x,tree_lo.x),tree_bininv.x)),
                                    (unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(lo.y,tree_lo.y),tree_bininv.y)),
                                    (unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(lo.z,tree_lo.z),tree_bininv.z)));
    const unsigned int lo_bin3 = (lo_bin.x << 20) +  (lo_bin.y << 10) + lo_bin.z;

    // high bounds are encoded relative to the high of the box, always rounding down
    const float3 hi = tree.hi[idx];
    const uint3 hi_bin = make_uint3((unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(tree_hi.x,hi.x),tree_bininv.x)),
                                    (unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(tree_hi.y,hi.y),tree_bininv.y)),
                                    (unsigned int)floorf(approx::fmul_rd(approx::fsub_rd(tree_hi.z,hi.z),tree_bininv.z)));
    const unsigned int hi_bin3 = (hi_bin.x << 20) + (hi_bin.y << 10) + hi_bin.z;

    // node holds left child for internal nodes (>= 0) or primitive for leaf (< 0)
    int left_flag = (idx < N_internal) ? tree.left[idx] : ~transform(tree.primitive[idx-N_internal]);

    // stash all the data into one int4
    ctree.data[idx] = make_int4(lo_bin3, hi_bin3, left_flag, rope);

    // first thread writes out the compression values, rounding down bin size to ensure box bounds always expand even with floats
    if (idx == 0)
        {
        *ctree.lo = tree_lo;
        *ctree.hi = tree_hi;
        *ctree.bins = make_float3(approx::frcp_rd(tree_bininv.x),approx::frcp_rd(tree_bininv.y),approx::frcp_rd(tree_bininv.z));
        }
    }

//! Kernel to traverse the LBVH using ropes.
/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param query Query operation.
 * \param images Translation operation.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
 * \tparam TranslateOpT The type of translation operation.
 *
 * The LBVH is traversed using the rope scheme. In this method, the
 * test sphere always descends to the left child of an intersected node,
 * and advances to the next viable branch of the tree (along the rope) when
 * no overlap occurs. This is a stackless traversal scheme.
 *
 * The query volume for the traversal can be constructed using the \a query operation.
 * This operation is responsible for constructing the query volume, translating it,
 * and performing overlap operations with the BoundingBox volumes in the LBVH.
 *
 * Each query volume can optionally be translated using a set of \a images. Before
 * entering the traversal loop, each volume is translated and intersected against
 * the tree root. A set of bitflags is encoded for which images possibly overlap the
 * tree. (Some may only intersect in the self-image, while others may intersect multiple
 * times.) This is done first to avoid divergence within the traversal loop.
 * During traversal, an image processes the entire tree, and then advances to the next
 * image once traversal terminates. A maximum of 32 images is supported.
 */
template<class OutputOpT, class QueryOpT, class TranslateOpT>
__global__ void lbvh_traverse_ropes(const OutputOpT out,
                                    const LBVHCompressedData lbvh,
                                    const QueryOpT query,
                                    const TranslateOpT images)
    {
    // one thread per test
    const unsigned int idx = hipper::threadRank<1,1>();
    if (idx >= query.size())
        return;

    // load tree compression sizes into shared memory
    __shared__ BoundingBox tree_box;
    __shared__ float3 tree_bins;
    if (threadIdx.x == 0)
        {
        tree_box = BoundingBox(*lbvh.lo, *lbvh.hi);
        tree_bins = *lbvh.bins;
        }
    __syncthreads();

    // query thread data
    const typename QueryOpT::ThreadData qdata = query.setup(idx);
    typename OutputOpT::ThreadData result = out.setup(idx, qdata);

    // find image flags against root before divergence
    unsigned int flags = 0;
    const int nbits = (images.size() <= 32u) ? images.size() : 32;
    for (unsigned int i=0; i < nbits; ++i)
        {
        const typename TranslateOpT::type image = images.get(i);
        const typename QueryOpT::Volume q = query.get(qdata,image);
        if (query.overlap(q,tree_box)) flags |= 1u << i;
        }

    // stackless search
    do
        {
        // look for the next image
        int image_bit = __ffs(flags);
        if (image_bit)
            {
            // shift the lsb by 1 to get the image index
            --image_bit;

            // unset the bit from this image
            flags &= ~(1u << image_bit);
            }
        else
            {
            // no more images, quit
            break;
            }

        // move the sphere to the next image
        const typename TranslateOpT::type image = images.get(image_bit);
        typename QueryOpT::Volume q = query.get(qdata, image);

        int node = lbvh.root;
        while (node != LBVHSentinel)
            {
            // load node and decompress bounds so that they always *expand*
            const int4 aabb = __ldg(lbvh.data + node);
            const unsigned int lo = aabb.x;
            const float3 lof = make_float3(approx::fadd_rd(tree_box.lo.x, approx::fmul_rd((lo >> 20) & 0x3ffu,tree_bins.x)),
                                           approx::fadd_rd(tree_box.lo.y, approx::fmul_rd((lo >> 10) & 0x3ffu,tree_bins.y)),
                                           approx::fadd_rd(tree_box.lo.z, approx::fmul_rd((lo      ) & 0x3ffu,tree_bins.z)));

            const unsigned int hi = aabb.y;
            const float3 hif = make_float3(approx::fsub_ru(tree_box.hi.x, approx::fmul_rd((hi >> 20) & 0x3ffu,tree_bins.x)),
                                           approx::fsub_ru(tree_box.hi.y, approx::fmul_rd((hi >> 10) & 0x3ffu,tree_bins.y)),
                                           approx::fsub_ru(tree_box.hi.z, approx::fmul_rd((hi      ) & 0x3ffu,tree_bins.z)));
            const int left = aabb.z;

            // advance to rope as a preliminary
            node = aabb.w;

            // if overlap, do work with primitive. otherwise, rope ahead
            if (query.overlap(q, BoundingBox(lof,hif)))
                {
                if(left < 0)
                    {
                    const int primitive = ~left;
                    if (query.refine(qdata,primitive))
                        out.process(result,primitive);
                    // leaf nodes always move to their rope
                    }
                else
                    {
                    // internal node takes left child
                    node = left;
                    }
                }
            } // end stackless search
        } while(true);

    out.finalize(result);
    }
} // end namespace kernel

//! Compress LBVH for rope traversal.
/*!
 * \param ctree Compressed LBVH.
 * \param transform Transformation operation.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam TransformOpT Type of operation for transforming cached primitive index.
 *
 * \sa kernel::lbvh_compress_ropes
 */
template<class TransformOpT>
void lbvh_compress_ropes(const LBVHCompressedData& ctree,
                         const TransformOpT& transform,
                         const ConstLBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size,
                         hipper::stream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipper::funcAttributes_t attr;
        hipper::funcGetAttributes(&attr, reinterpret_cast<const void*>(kernel::lbvh_compress_ropes<TransformOpT>));
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
    const unsigned int num_blocks = (N_nodes + run_block_size - 1)/run_block_size;

    hipper::KernelLauncher launcher(num_blocks, run_block_size, stream);
    launcher(kernel::lbvh_compress_ropes<TransformOpT>, ctree, transform, tree, N_internal, N_nodes);
    }

//! Traverse the LBVH using ropes.
/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param query Query operation.
 * \param images Translation operation.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
 * \tparam TranslateOpT The type of translation operation.
 *
 * \sa kernel::lbvh_traverse_ropes
 */
template<class OutputOpT, class QueryOpT, class TranslateOpT>
void lbvh_traverse_ropes(const OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         const QueryOpT& query,
                         const TranslateOpT& images,
                         unsigned int block_size,
                         hipper::stream_t stream)
    {
    // quit if there are no images
    if (query.size() == 0 || images.size() == 0)
        return;

    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        hipper::funcAttributes_t attr;
        hipper::funcGetAttributes(&attr, reinterpret_cast<const void*>(kernel::lbvh_traverse_ropes<OutputOpT,QueryOpT,TranslateOpT>));
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;
    const unsigned int num_blocks = (query.size() + run_block_size - 1)/run_block_size;

    hipper::KernelLauncher launcher(num_blocks, run_block_size, stream);
    launcher(kernel::lbvh_traverse_ropes<OutputOpT,QueryOpT,TranslateOpT>, out, lbvh, query, images);
    }

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_CUH_
