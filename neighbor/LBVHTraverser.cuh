// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_CUH_
#define NEIGHBOR_LBVH_TRAVERSER_CUH_

#include "hoomd/HOOMDMath.h"
#include "LBVH.cuh"
#include "BoundingVolumes.h"

namespace neighbor
{
namespace gpu
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

//! Compress LBVH for rope traversal.
void lbvh_compress_ropes(LBVHCompressedData ctree,
                         const LBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size);

//! Traverse the LBVH using ropes.
template<class OutputOpT, class QueryOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         const QueryOpT& query,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int block_size);

/*
 * Templated function definitions should only be available in NVCC.
 */
#ifdef NVCC
namespace kernel
{
//! Kernel to traverse the LBVH using ropes.
/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param query Query operation.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
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
 * Each query volume can optionally be translated by a set of \a d_images. The self-image
 * is automatically traversed and should not be included in \a d_images. Before
 * entering the traversal loop, each volume is translated and intersected against
 * the tree root. A set of bitflags is encoded for which images possibly overlap the
 * tree. (Some may only intersect in the self-image, while others may intersect multiple
 * times.) This is done first to avoid divergence within the traversal loop.
 * During traversal, an image processes the entire tree, and then advances to the next
 * image once traversal terminates. A maximum of 32 images is supported.
 */
template<class OutputOpT, class QueryOpT>
__global__ void lbvh_traverse_ropes(OutputOpT out,
                                    const LBVHCompressedData lbvh,
                                    const QueryOpT query,
                                    const Scalar3 *d_images,
                                    const unsigned int Nimages)
    {
    // one thread per test
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= query.size())
        return;
    typename OutputOpT::ThreadData result = out.setup(idx);

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

    // find image flags against root before divergence
    unsigned int flags = 0;
    const int nbits = ((int)Nimages <= 32) ? Nimages : 32;
    for (unsigned int i=0; i < nbits; ++i)
        {
        const Scalar3 image = d_images[i];
        const typename QueryOpT::Volume q = query.get(qdata,image);
        if (query.overlap(q,tree_box)) flags |= 1u << i;
        }

    // stackless search
    typename QueryOpT::Volume q = query.get(qdata, make_scalar3(0,0,0));
    int node = lbvh.root;
    do
        {
        while (node != LBVHSentinel)
            {
            // load node and decompress bounds so that they always *expand*
            const int4 aabb = __ldg(lbvh.data + node);
            const unsigned int lo = aabb.x;
            const float3 lof = make_float3(__fadd_rd(tree_box.lo.x, __fmul_rd((lo >> 20) & 0x3ffu,tree_bins.x)),
                                           __fadd_rd(tree_box.lo.y, __fmul_rd((lo >> 10) & 0x3ffu,tree_bins.y)),
                                           __fadd_rd(tree_box.lo.z, __fmul_rd((lo      ) & 0x3ffu,tree_bins.z)));

            const unsigned int hi = aabb.y;
            const float3 hif = make_float3(__fsub_ru(tree_box.hi.x, __fmul_rd((hi >> 20) & 0x3ffu,tree_bins.x)),
                                           __fsub_ru(tree_box.hi.y, __fmul_rd((hi >> 10) & 0x3ffu,tree_bins.y)),
                                           __fsub_ru(tree_box.hi.z, __fmul_rd((hi      ) & 0x3ffu,tree_bins.z)));
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

        // look for the next image
        int image_bit = __ffs(flags);
        if (image_bit)
            {
            // shift the lsb by 1 to get the image index
            --image_bit;

            // move the sphere to the next image
            const Scalar3 image = d_images[image_bit];
            q = query.get(qdata, image);
            node = lbvh.root;

            // unset the bit from this image
            flags &= ~(1u << image_bit);
            }
        else
            {
            // no more images, quit
            break;
            }
        } while(true);

    out.finalize(result);
    }
} // end namespace kernel

/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param d_spheres Test spheres to intersect with LBVH.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 * \param block_size Number of CUDA threads per block.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
 *
 * \sa kernel::lbvh_traverse_ropes
 */
template<class OutputOpT, class QueryOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         const QueryOpT& query,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int block_size)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_traverse_ropes<OutputOpT,QueryOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (query.size() + run_block_size - 1)/run_block_size;
    kernel::lbvh_traverse_ropes<<<num_blocks, run_block_size>>>
        (out, lbvh, query, d_images, Nimages);
    }
#endif

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_CUH_
