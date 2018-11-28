// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGridTraverser.cuh"
#include "hoomd/WarpTools.cuh"

namespace neighbor
{
namespace gpu
{
namespace kernel
{

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
template<unsigned int threads>
__global__ void uniform_grid_traverse(unsigned int *d_out,
                                      const Scalar4 *d_spheres,
                                      const UniformGridData grid,
                                      const int3 *d_stencil,
                                      const unsigned int num_stencil,
                                      const BoxDim box,
                                      const unsigned int N)
    {
    // *threads* per primitive
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*threads)
        return;

    // load data for the test sphere
    const unsigned int idx = tid / threads;
    const Scalar4 sphere = d_spheres[idx];
    const Scalar3 ri = make_scalar3(sphere.x, sphere.y, sphere.z);
    const Scalar rcutsq = sphere.w*sphere.w; // assume this is <= stencil radius^2
    const int3 bin = grid.toCell(ri);

    // variables for iterating through with multiple threads
    unsigned int offset = tid % threads;
    int s = -1;
    unsigned int cell = gpu::UniformGridSentinel; // dummy
    unsigned int first = gpu::UniformGridSentinel; // dummy
    unsigned int size = 0; // dummy, 0 at first to take first stencil

    bool done = false;
    unsigned int n_hit = 0;
    while(!done)
        {
        while(offset >= size && !done)
            {
            // past end of current cell, move to next stencil
            offset -= size;
            ++s;

            // if stencil index is valid
            if(s < (int)num_stencil)
                {
                const int3 stencil = d_stencil[s];
                const int3 neigh_bin = grid.wrap(make_int3(bin.x+stencil.x, bin.y+stencil.y, bin.z+stencil.z));
                cell = grid.indexer(neigh_bin.x, neigh_bin.y, neigh_bin.z);
                size = __ldg(grid.size + cell);
                if (size > 0) first = __ldg(grid.first + cell);
                }
            else
                {
                done = true;
                }
            }

        // process the point
        unsigned int primitive = UniformGridSentinel;
        if (!done)
            {
            const Scalar4 neigh = grid.point[first + offset];
            const Scalar3 rj = make_scalar3(neigh.x, neigh.y, neigh.z);

            const Scalar3 rij = box.minImage(rj - ri);
            const Scalar dr2 = dot(rij, rij);
            if (dr2 <= rcutsq)
                {
                primitive = __scalar_as_int(neigh.w);
                }

            offset += threads;
            }

        // all threads are done if thread 0 is done. otherwise, process the hits
        if (threads > 1)
            done = hoomd::detail::WarpScan<bool,threads>().Broadcast(done, 0);

        if (!done)
            {
            unsigned char hit = (primitive != UniformGridSentinel);
            unsigned char k(0), n(hit);
            if (threads > 1)
                hoomd::detail::WarpScan<unsigned char, threads>().ExclusiveSum(hit, k, n);
            n_hit += n;
            }
        }

    // only first thread writes out
    if (tid % threads == 0)
        {
        d_out[idx] = n_hit;
        }
    }

} // end namespace kernel

//! Helper to clamp the block size of the templated kernel
/*!
 * \param block_size Requested number of CUDA threads per block.
 * \tparam threads Number of CUDA threads per traversal sphere.
 * \returns The valid number of CUDA threads per block.
 *
 * Each kernel may have different limits on the block size, and so this
 * needs to be computed for each templated parameter. This helper function avoids
 * alot of copy-paste.
 */
template<unsigned int threads>
static unsigned int clamp(unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_traverse<threads>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    return (block_size < max_block_size) ? block_size : max_block_size;
    }

/*!
 * \param d_out Number of primitive intersections per sphere.
 * \param d_spheres Test spheres to intersect with UniformGrid.
 * \param grid UniformGrid to traverse.
 * \param d_stencil Traversal stencil for spheres.
 * \param num_stencil Number of images in the stencil.
 * \param box HOOMD-blue BoxDim for wrapping periodic images.
 * \param N Number of traversal spheres.
 * \param threads Number of CUDA threads assigned per sphere.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::uniform_grid_traverse
 *
 * The appropriate templated invocation of the kernel is selected through a series of
 * if statements. If an invalid \a threads is specified, nothing happens.
 */
void uniform_grid_traverse(unsigned int *d_out,
                           const Scalar4 *d_spheres,
                           const UniformGridData grid,
                           const int3 *d_stencil,
                           const unsigned int num_stencil,
                           const BoxDim& box,
                           const unsigned int N,
                           const unsigned int threads,
                           const unsigned int block_size)
    {
    if (threads == 1)
        {
        const unsigned int run_block_size = clamp<1>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_traverse<1><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else if (threads == 2)
        {
        const unsigned int run_block_size = clamp<2>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_traverse<2><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else if (threads == 4)
        {
        const unsigned int run_block_size = clamp<4>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;

        kernel::uniform_grid_traverse<4><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else if (threads == 8)
        {
        const unsigned int run_block_size = clamp<8>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_traverse<8><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else if (threads == 16)
        {
        const unsigned int run_block_size = clamp<16>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_traverse<16><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else if (threads == 32)
        {
        const unsigned int run_block_size = clamp<32>(block_size);
        const unsigned int num_blocks = (N*threads + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_traverse<32><<<num_blocks, run_block_size>>>(d_out, d_spheres, grid, d_stencil, num_stencil, box, N);
        }
    else
        {
        // do nothing
        }
    }

} // end namespace gpu
} // end namespace neighbor
