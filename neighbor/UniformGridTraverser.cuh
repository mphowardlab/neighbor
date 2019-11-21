// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_
#define NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_

#include "UniformGrid.cuh"
#include "BoundingVolumes.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace neighbor
{
namespace gpu
{

struct UniformGridCompressedData
    {
    Scalar4* data;
    uint2* range;
    Scalar3 lo;             //!< Lower bound of grid
    Scalar3 hi;             //!< Upper bound of grid
    Scalar3 L;              //!< Size of grid
    Scalar3 width;          //!< Width of bins
    Index3D indexer;        //!< 3D indexer into the grid memory

    #ifdef NVCC
    __device__ __forceinline__ int3 toCell(const float3& r) const
        {
        // convert position into fraction, then bin
        const Scalar3 f = make_scalar3(r.x-lo.x, r.y-lo.y, r.z-lo.z)/L;
        int3 bin = make_int3(static_cast<int>(f.x * indexer.getW()),
                             static_cast<int>(f.y * indexer.getH()),
                             static_cast<int>(f.z * indexer.getD()));
        return bin;
        }
    #endif // NVCC
    };

template<class TransformOpT>
void uniform_grid_compress(const UniformGridCompressedData& cgrid,
                           const TransformOpT& transform,
                           const UniformGridData& grid,
                           const unsigned int N,
                           const unsigned int Ncell,
                           const unsigned int block_size,
                           cudaStream_t stream = 0);

//! Traverse the UniformGrid
template<class OutputOpT, class QueryOpT>
void uniform_grid_traverse(const OutputOpT& out,
                           const UniformGridCompressedData& grid,
                           const QueryOpT& query,
                           const Scalar3 *d_images,
                           unsigned int Nimages,
                           const unsigned int block_size,
                           cudaStream_t stream = 0);

#ifdef NVCC
namespace kernel
{
template<class TransformOpT>
__global__ void uniform_grid_compress(const UniformGridCompressedData cgrid,
                                      const TransformOpT transform,
                                      const UniformGridData grid,
                                      const unsigned int N,
                                      const unsigned int Ncell)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // transform primitive data
    if (idx < N)
        {
        // load point and transform the tag
        Scalar4 p = grid.points[idx];
        const unsigned int tag = __scalar_as_int(p.w);
        p.w = __int_as_scalar(transform(tag));
        cgrid.data[idx] = p;
        }

    // compress cell data into cache-friendly form
    if (idx < Ncell)
        {
        const unsigned int first = __ldg(grid.first + idx);
        const unsigned int size = __ldg(grid.size + idx);
        cgrid.range[idx] = make_uint2(first, first + size);
        }
    }

//! Kernel to traverse the UniformGrid
/*!
 * \param d_out Number of primitive intersections per sphere.
 * \param d_spheres Test spheres to intersect with UniformGrid.
 * \param grid UniformGrid to traverse.
 * \param d_stencil Traversal stencil for spheres.
 * \param num_stencil Number of images in the stencil.
 * \param box HOOMD-blue BoxDim for wrapping periodic images.
 * \param N Number of traversal spheres.
 * \tparam threads Number of CUDA threads assigned per sphere.
 *
 * This kernel is based on NeighborListStencilGPU in HOOMD-blue.
 * \a threads CUDA threads are assigned per traversal sphere.
 * They concurrently process each sphere using appropriate warp-level
 * syncrhonization and reduction. The threads advance through the
 * list of stencils, processing primitives from adjacents bins that
 * have not yet been checked. The result is accumulated, and the first
 * thread in each bundle writes the number of intersections to \a d_out.
 * All threads remain active until all have completed their work, at which
 * point they can exit.
 *
 * Intersection tests between the spheres and the points are done in Scalar
 * precision to avoid round off errors and are also subject to the minimum
 * image convention through \a box. This differs from the approach of the
 * LBVH, which traverses images explicitly. This is not a good idea for the
 * UniformGrid because of the overhead in loading new cells and because some
 * images may fall in bins outside the grid.
 */
template<class OutputOpT, class QueryOpT>
__global__ void uniform_grid_traverse(const OutputOpT out,
                                      const UniformGridCompressedData grid,
                                      const QueryOpT query,
                                      const Scalar3 *d_images,
                                      const unsigned int Nimages)
    {
    // one thread per test
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= query.size())
        return;

    // load grid size into shared memory
    __shared__ BoundingBox grid_box;
    if (threadIdx.x == 0)
        {
        grid_box = BoundingBox(grid.lo, grid.hi);
        }
    __syncthreads();

    // query thread data
    const typename QueryOpT::ThreadData qdata = query.setup(idx);
    typename OutputOpT::ThreadData result = out.setup(idx, qdata);

    // find image flags against grid box before divergence
    unsigned int flags = 0;
    const int nbits = ((int)Nimages <= 32) ? Nimages : 32;
    for (int i=0; i < nbits; ++i)
        {
        const Scalar3 image = d_images[i];
        const typename QueryOpT::Volume q = query.get(qdata,image);
        if (query.overlap(q,grid_box)) flags |= 1u << i;
        }

    // stackless search
    typename QueryOpT::Volume q = query.get(qdata, make_scalar3(0,0,0));
    do
        {
        // get bin of this bounding box
        const BoundingBox b = q.asBox();

        // clamp lo to grid
        int3 lo = grid.toCell(b.lo);
        if (lo.x < 0) lo.x = 0;
        if (lo.y < 0) lo.y = 0;
        if (lo.z < 0) lo.z = 0;

        // clamp hi to grid
        int3 hi = grid.toCell(b.hi);
        if (hi.x >= grid.indexer.getW()) hi.x = grid.indexer.getW()-1;
        if (hi.y >= grid.indexer.getH()) hi.y = grid.indexer.getH()-1;
        if (hi.z >= grid.indexer.getD()) hi.z = grid.indexer.getD()-1;

        // loop on cells
        for (int k=lo.z; k <= hi.z; ++k)
            {
            for (int j=lo.y; j <= hi.y; ++j)
                {
                for (int i=lo.x; i <= hi.x; ++i)
                    {
                    const uint2 range = grid.range[grid.indexer(i,j,k)];

                    // loop over primitives
                    for (unsigned int n=range.x; n < range.y; ++n)
                        {
                        const Scalar4 p = grid.data[n];
                        const Scalar3 r = make_scalar3(p.x, p.y, p.z);
                        const unsigned int primitive = __scalar_as_int(p.w);

                        if (query.overlap(q, BoundingBox(r,r)))
                            {
                            if (query.refine(qdata,primitive))
                                out.process(result,primitive);
                            }
                        }
                    }
                }
            }

        // look for the next image
        int image_bit = __ffs(flags);
        if (image_bit)
            {
            // shift the lsb by 1 to get the image index
            --image_bit;

            // move the sphere to the next image
            const Scalar3 image = d_images[image_bit];
            q = query.get(qdata, image);

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

template<class TransformOpT>
void uniform_grid_compress(const UniformGridCompressedData& cgrid,
                           const TransformOpT& transform,
                           const UniformGridData& grid,
                           const unsigned int N,
                           const unsigned int Ncell,
                           const unsigned int block_size,
                           cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_compress<TransformOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int Nthread = (N > Ncell) ? N : Ncell;
    const unsigned int num_blocks = (Nthread + run_block_size - 1)/run_block_size;
    kernel::uniform_grid_compress<<<num_blocks, run_block_size, 0, stream>>>
        (cgrid, transform, grid, N, Ncell);
    }

//! Traverse the UniformGrid
template<class OutputOpT, class QueryOpT>
void uniform_grid_traverse(const OutputOpT& out,
                           const UniformGridCompressedData& grid,
                           const QueryOpT& query,
                           const Scalar3 *d_images,
                           unsigned int Nimages,
                           const unsigned int block_size,
                           cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_traverse<OutputOpT,QueryOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (query.size() + run_block_size - 1)/run_block_size;
    kernel::uniform_grid_traverse<<<num_blocks, run_block_size, 0, stream>>>
        (out, grid, query, d_images, Nimages);
    }
#endif

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_TRAVERSER_CUH_
