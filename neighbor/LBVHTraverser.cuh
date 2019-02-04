// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_CUH_
#define NEIGHBOR_LBVH_TRAVERSER_CUH_

#include "hoomd/HOOMDMath.h"
#include "LBVH.cuh"

namespace neighbor
{
namespace gpu
{

//! Lightweight data structure to hold the compressed LBVH.
struct LBVHCompressedData
    {
    int root;           //!< Root index of the LBVH
    const int4* data;   //!< Compressed LBVH data.
    const float3* lo;   //!< Lower bound used in compression.
    const float3* hi;   //!< Upper bound used in compression.
    const float3* bins; //!< Bin spacing used in compression.
    };

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
template<class OutputOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         // traversal spheres
                         const Scalar4 *d_spheres,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int N,
                         unsigned int block_size);

/*
 * Templated methods should only be compiled by NVCC
 */
#ifdef NVCC
namespace kernel
{
//! Test for overlap between a sphere and an AABB
/*!
 * \param o Center of sphere.
 * \param rsq Squared radius of sphere.
 * \param lo Lower bound of AABB.
 * \param hi Upper bound of AABB.
 *
 * \returns True if the sphere and AABB overlap.
 *
 * The intersection test is performed by finding the closest point to \a o
 * that is inside the AABB using a sequence of min and max ops. The distance
 * to this point from \a o is then computed in round down mode. If the squared
 * distance between the point and \a o is less than \a rsq, then the two
 * objects intersect.
 */
__device__ __forceinline__ bool boxSphereOverlap(const float3& o,
                                                 const float rsq,
                                                 const float3& lo,
                                                 const float3& hi)
    {
    const float3 dr = make_float3(__fsub_rd(fminf(fmaxf(o.x, lo.x), hi.x), o.x),
                                  __fsub_rd(fminf(fmaxf(o.y, lo.y), hi.y), o.y),
                                  __fsub_rd(fminf(fmaxf(o.z, lo.z), hi.z), o.z));
    const float dr2 = __fmaf_rd(dr.x, dr.x, __fmaf_rd(dr.y, dr.y, __fmul_rd(dr.z,dr.z)));

    return (dr2 <= rsq);
    }

//! Fit a sphere (in Scalar precision) to a float sphere
/*!
 * \param o Center of sphere.
 * \param rsq Squared radius of sphere.
 * \param sphere Original sphere.
 *
 * This method is a convenience wrapper for handling casting of a sphere from Scalar
 * precision to float precision. Spheres are originally translated around in Scalar
 * precision to avoid round-off errors. The translated sphere can then be cast into
 * float precision, which requires padding the sphere radius to account for uncertainty
 * due to rounding. The fitted sphere always encloses the original sphere.
 *
 * When Scalar is float, this method simply returns \a sphere.
 */
__device__ __forceinline__ void fitSphere(float3& o, float& rsq, const Scalar4& sphere)
    {
    #ifndef SINGLE_PRECISION
    const float3 lo = make_float3(__double2float_rd(sphere.x),
                                  __double2float_rd(sphere.y),
                                  __double2float_rd(sphere.z));
    const float3 hi = make_float3(__double2float_ru(sphere.x),
                                  __double2float_ru(sphere.y),
                                  __double2float_ru(sphere.z));
    const float delta = fmaxf(fmaxf(__fsub_ru(hi.x,lo.x),__fsub_ru(hi.y,lo.y)),__fsub_ru(hi.z,lo.z));
    const float R = __fadd_ru(__double2float_ru(sphere.w),delta);
    o = make_float3(lo.x, lo.y, lo.z);
    rsq = __fmul_ru(R,R);
    #else
    o = make_float3(sphere.x, sphere.y, sphere.z);
    rsq = __fmul_ru(sphere.w,sphere.w);
    #endif
    }

//! Kernel to traverse the LBVH using ropes.
/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param d_spheres Test spheres to intersect with LBVH.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 *
 * \tparam OutputOpT The type of output operation.
 *
 * The LBVH is traversed using the rope scheme. In this method, the
 * test sphere always descends to the left child of an intersected node,
 * and advances to the next viable branch of the tree (along the rope) when
 * no overlap occurs. This is a stackless traversal scheme.
 *
 * Each sphere can optionally be translated by a set of \a d_images. The self-image
 * is automatically traversed and should not be included in \a d_images. Before
 * entering the traversal loop, each sphere is translated and intersected against
 * the tree root. A set of bitflags is encoded for which images possibly overlap the
 * tree. (Some may only intersect in the self-image, while others may intersect multiple
 * times.) This is done first to avoid divergence within the traversal loop.
 * During traversal, an image processes the entire tree, and then advances to the next
 * image once traversal terminates. A maximum of 32 images is supported.
 */
template<class OutputOpT>
__global__ void lbvh_traverse_ropes(OutputOpT out,
                                    const int root,
                                    const int4 *d_data,
                                    const float3 *d_lbvh_lo,
                                    const float3 *d_lbvh_hi,
                                    const float3 *d_bins,
                                    const Scalar4 *d_spheres,
                                    const Scalar3 *d_images,
                                    const unsigned int Nimages,
                                    const unsigned int N)
    {
    // one thread per test
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    typename OutputOpT::ThreadData result = out.setup(idx);

    // load tree compression sizes into shared memory
    __shared__ float3 tree_lo, tree_hi, tree_bins;
    if (threadIdx.x == 0)
        {
        tree_lo = *d_lbvh_lo;
        tree_hi = *d_lbvh_hi;
        tree_bins = *d_bins;
        }
    __syncthreads();

    // test sphere
    const Scalar4 sphere = d_spheres[idx];

    // find image flags against root before divergence
    unsigned int flags = 0;
    const int nbits = ((int)Nimages <= 32) ? Nimages : 32;
    for (unsigned int i=0; i < nbits; ++i)
        {
        const Scalar3 image = d_images[i];
        const Scalar4 test = make_scalar4(sphere.x+image.x, sphere.y+image.y, sphere.z+image.z, sphere.w);
        float3 t; float rsq;
        fitSphere(t, rsq, test);

        // box-sphere overlap test
        if (boxSphereOverlap(t, rsq, tree_lo, tree_hi)) flags |= 1u << i;
        }

    // stackless search
    float3 t; float rsq;
    fitSphere(t, rsq, sphere);
    int node = root;
    do
        {
        while (node != LBVHSentinel)
            {
            // load node and decompress bounds so that they always *expand*
            const int4 aabb = __ldg(d_data + node);
            const unsigned int lo = aabb.x;
            const float3 lof = make_float3(__fadd_rd(tree_lo.x, __fmul_rd((lo >> 20) & 0x3ffu,tree_bins.x)),
                                           __fadd_rd(tree_lo.y, __fmul_rd((lo >> 10) & 0x3ffu,tree_bins.y)),
                                           __fadd_rd(tree_lo.z, __fmul_rd((lo      ) & 0x3ffu,tree_bins.z)));

            const unsigned int hi = aabb.y;
            const float3 hif = make_float3(__fsub_ru(tree_hi.x, __fmul_rd((hi >> 20) & 0x3ffu,tree_bins.x)),
                                           __fsub_ru(tree_hi.y, __fmul_rd((hi >> 10) & 0x3ffu,tree_bins.y)),
                                           __fsub_ru(tree_hi.z, __fmul_rd((hi      ) & 0x3ffu,tree_bins.z)));
            const int left = aabb.z;

            // advance to rope as a preliminary
            node = aabb.w;

            // if overlap, do work with primitive. otherwise, rope ahead
            if (boxSphereOverlap(t, rsq, lof, hif))
                {
                if(left < 0)
                    {
                    const int primitive = ~left;
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
            const Scalar4 new_sphere = make_scalar4(sphere.x+image.x, sphere.y+image.y, sphere.z+image.z, sphere.w);
            fitSphere(t, rsq, new_sphere);
            node = root;

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
 *
 * \sa kernel::lbvh_traverse_ropes
 */
template<class OutputOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         // traversal spheres
                         const Scalar4 *d_spheres,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int N,
                         unsigned int block_size)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_traverse_ropes<OutputOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;
    kernel::lbvh_traverse_ropes<OutputOpT><<<num_blocks, run_block_size>>>
        (out, lbvh.root, lbvh.data, lbvh.lo, lbvh.hi, lbvh.bins, d_spheres, d_images, Nimages, N);
    }
#endif

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_CUH_
