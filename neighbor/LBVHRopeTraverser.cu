// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVHRopeTraverser.cuh"

namespace neighbor
{
namespace gpu
{
namespace kernel
{

//! Kernel to compress LBVH for rope traversal
/*!
 * \param d_data Compressed LBVH.
 * \param d_lbvh_lo Lower bound of LBVH for (de)compression.
 * \param d_lbvh_hi Upper bound of LBVH for (de)compression.
 * \param d_bins Size of compressed bins.
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
__global__ void lbvh_compress_ropes(int4 *d_data,
                                    float3 *d_lbvh_lo,
                                    float3 *d_lbvh_hi,
                                    float3 *d_bins,
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
    d_data[idx] = make_int4(lo_bin3, hi_bin3, left_flag, rope);

    // first thread writes out the compression values, rounding down bin size to ensure box bounds always expand even with floats
    if (idx == 0)
        {
        *d_lbvh_lo = tree_lo;
        *d_lbvh_hi = tree_hi;
        *d_bins = make_float3(__frcp_rd(tree_bininv.x),__frcp_rd(tree_bininv.y),__frcp_rd(tree_bininv.z));
        }
    }

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
 * \param d_out Number of primitives overlapped by test spheres.
 * \param root Root node of the LBVH.
 * \param d_data Compressed LBVH data.
 * \param d_lbvh_lo Lower bound of the LBVH.
 * \param d_lbvh_hi Upper bound of the LBVH.
 * \param d_bins Bin size for decompression of \a d_data.
 * \param d_spheres Test spheres to intersect with LBVH.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
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
__global__ void lbvh_traverse_ropes(unsigned int *d_out,
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
    unsigned int n_hit = 0;
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
                    // use this if test (always evaluates to true) to shut up compiler warning about primitive
                    if (primitive != LBVHSentinel)
                        ++n_hit;
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

    // write number of neighbors
    d_out[idx] = n_hit;
    }

} // end namespace kernel

/*!
 * \param d_data Compressed LBVH.
 * \param d_lbvh_lo Lower bound of LBVH for (de)compression.
 * \param d_lbvh_hi Upper bound of LBVH for (de)compression.
 * \param d_bins Size of compressed bins.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::lbvh_compress_ropes
 */
void lbvh_compress_ropes(int4 *d_data,
                         float3 *d_lbvh_lo,
                         float3 *d_lbvh_hi,
                         float3 *d_bins,
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
        (d_data, d_lbvh_lo, d_lbvh_hi, d_bins, tree, N_internal, N_nodes);
    }

/*!
 * \param d_out Number of primitives overlapped by test spheres.
 * \param root Root node of the LBVH.
 * \param d_data Compressed LBVH data.
 * \param d_lbvh_lo Lower bound of the LBVH.
 * \param d_lbvh_hi Upper bound of the LBVH.
 * \param d_bins Bin size for decompression of \a d_data.
 * \param d_spheres Test spheres to intersect with LBVH.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::lbvh_traverse_ropes
 */
void lbvh_traverse_ropes(unsigned int *d_out,
                         int root,
                         const int4 *d_data,
                         const float3 *d_lbvh_lo,
                         const float3 *d_lbvh_hi,
                         const float3 *d_bins,
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
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_traverse_ropes);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;
    kernel::lbvh_traverse_ropes<<<num_blocks, run_block_size>>>
        (d_out, root, d_data, d_lbvh_lo, d_lbvh_hi, d_bins, d_spheres, d_images, Nimages, N);
    }

} // end namespace gpu
} // end namespace neighbor
