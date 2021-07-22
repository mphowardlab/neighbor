// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGrid.cuh"
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include "hoomd/extern/cub/cub/cub.cuh"
#endif

namespace neighbor
{
namespace gpu
{
namespace kernel
{
//! Kernel to bin points into the grid
/*!
 * \param d_cells Bin assignment per point.
 * \param d_primitives Unsorted index of each point.
 * \param d_points The points to bin.
 * \param grid Uniform grid to bin into.
 * \param N Number of points to bin.
 *
 * One thread is used to bin each point onto an orthorhombic grid.
 * Points lying outside the grid are clamped onto the boundaries of the
 * grid. It is the caller's duty to ensure this is reasonable behavior
 * (e.g., fixing roundoff error) and not because points are not within the
 * right bounds.
 */
__global__ void uniform_grid_bin_points(unsigned int *d_cells,
                                        unsigned int *d_primitives,
                                        const GridPointOp insert,
                                        const UniformGridData grid)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= insert.size())
        return;

    const Scalar3 r = insert.get(idx);
    const int3 bin = grid.toCell(r);

    d_cells[idx] = grid.indexer(bin.x, bin.y, bin.z);
    d_primitives[idx] = idx;
    }

//! Kernel to shuffle sorted point data
/*!
 * \param d_sorted_points Sorted and packed points.
 * \param d_points Unsorted (original) points.
 * \param d_sorted_indexes The indexes of points in sorted order.
 * \param N the number of points.
 *
 * One thread is used for each point to rearrange the point data. The
 * w component of the sorted points stores the original index of the point
 * for cache-friendly loads by traversal schemes.
 */
__global__ void uniform_grid_move_points(Scalar4 *d_sorted_points,
                                         const GridPointOp insert,
                                         const unsigned int *d_sorted_indexes)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= insert.size())
        return;

    const unsigned int tag = d_sorted_indexes[idx];
    const Scalar3 r = insert.get(tag);

    d_sorted_points[idx] = make_scalar4(r.x, r.y, r.z, __int_as_scalar(tag));
    }

//! Kernel to find the first point and last point in each bin.
/*!
 * \param d_first Index of first point in each bin.
 * \param d_size Number of points in each bin.
 * \param d_cells The bin assignment of each point.
 * \param N Total number of points.
 *
 * One thread is used for each point in the grid. The point looks at the bin
 * assignment of its left and right neighbors (with appropriate clamping for
 * the first / last points). If the left neighbor has a different bin, then
 * this particle is the *first*. If the right neighbor has a different bin, this
 * particle is the *last*. There is only \b one *first* or *last* particle for
 * each cell, and so there is no race condition writing this result.
 *
 * The \a d_size array does not yet hold the size of the cell list when the kernel
 * completes. A second kernel (or a grid synchronization call) is needed to subtract
 * *last* from *first* to find the effective size (see kernel::uniform_grid_size_cells).
 */
__global__ void uniform_grid_find_ends(unsigned int *d_first,
                                       unsigned int *d_size,
                                       const unsigned int *d_cells,
                                       const unsigned int N)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    // current cell
    const unsigned int cell = d_cells[idx];
    // look to left if not first
    const unsigned int left = (idx > 0) ? d_cells[idx-1] : UniformGridSentinel;
    // look to right if not last
    const unsigned int right = (idx < N-1) ? d_cells[idx+1] : UniformGridSentinel;

    // if left is not same as self (or idx == 0 by use of sentinel), this is the first index in the cell
    if (left != cell)
        {
        d_first[cell] = idx;
        }

    // if right is not the same as self (or idx == N-1 by use of sentinel), this is the last index in the cell
    if (right != cell)
        {
        d_size[cell] = idx + 1;
        }
    }

//! Kernel to find the size of each bin.
/*!
 * \param d_size Last point in each cell (input), size of each bin (output).
 * \param d_first First point in each bin.
 * \param Ncells Number of cells.
 *
 * One thread is used per cell, and the difference in last/first is subtracted (if the cell is actually set)
 * to find its size. Otherwise, nothing needs to be done because a separate memset has already filled the
 * correct default size (0).
 */
__global__ void uniform_grid_size_cells(unsigned int *d_size,
                                        const unsigned int *d_first,
                                        const unsigned int Ncells)
    {
    // one thread per cell
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;

    const unsigned int first = d_first[idx];
    if (first != UniformGridSentinel)
        {
        const unsigned int last = d_size[idx];
        d_size[idx] = last-first;
        }
    }

} // end namespace kernel

/*!
 * \param d_cells Bin assignment per point.
 * \param d_primitives Unsorted index of each point.
 * \param d_points The points to bin.
 * \param grid Uniform grid to bin into.
 * \param N Number of points to bin.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::uniform_grid_bin_points
 */
void uniform_grid_bin_points(unsigned int *d_cells,
                             unsigned int *d_primitives,
                             const GridPointOp& insert,
                             const UniformGridData grid,
                             const unsigned int block_size,
                             cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_bin_points);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (insert.size() + run_block_size - 1)/run_block_size;
    kernel::uniform_grid_bin_points<<<num_blocks, run_block_size, 0, stream>>>(d_cells, d_primitives, insert, grid);
    }

/*!
 * \param d_tmp Temporary storage for CUB.
 * \param tmp_bytes Temporary storage size (B) for CUB.
 * \param d_cells Unsorted bin assignment.
 * \param d_sorted_codes Sorted bin assignment.
 * \param d_indexes Unsorted primitive indexes.
 * \param d_sorted_indexes Sorted primitive indexes.
 * \param d_points Unsorted primitives.
 * \param d_sorted_points Sorted primitives.
 * \param N Number of primitives.
 *
 * \returns Two flags (swap) with the location of the sorted bins and indexes. If swap.x
 *          is 1, then the sorted bins are in \a d_cells and need to be swapped. Similarly,
 *          if swap.y is 1, then the sorted indexes are in \a d_indexes.
 *
 * The bin assignments are sorted in ascending order using radix sort in the CUB library.
 * This function must be called twice in order for the sort to occur. When \a d_tmp is NULL
 * on the first call, CUB sizes the temporary storage that is required and sets it in \a tmp_bytes.
 * Usually, this is a small amount and can be allocated from a buffer (e.g., a HOOMD-blue
 * CachedAllocator). Some versions of CUB were buggy and required \a d_tmp be allocated even
 * when \a tmp_bytes was 0. To bypass this, allocate a small amount (say, 4B) when \a tmp_bytes is 0.
 * The second call will then sort the bins and indexes. The sorted data will be in the
 * appropriate buffer, which can be determined by the returned flags.
 */
uchar2 uniform_grid_sort_points(void *d_tmp,
                                size_t &tmp_bytes,
                                unsigned int *d_cells,
                                unsigned int *d_sorted_cells,
                                unsigned int *d_indexes,
                                unsigned int *d_sorted_indexes,
                                const unsigned int N,
                                cudaStream_t stream)
    {

    HOOMD_CUB::DoubleBuffer<unsigned int> d_keys(d_cells, d_sorted_cells);
    HOOMD_CUB::DoubleBuffer<unsigned int> d_vals(d_indexes, d_sorted_indexes);

    HOOMD_CUB::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, 8*sizeof(unsigned int), stream);

    uchar2 swap = make_uchar2(0,0);
    if (d_tmp != NULL)
        {
        // synchronize first to make sure active selection is known
        cudaStreamSynchronize(stream);

        // mark that the gpu arrays should be flipped if the final result is not in the sorted array (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

void uniform_grid_move_points(Scalar4 *d_sorted_points,
                              const GridPointOp& insert,
                              const unsigned int *d_sorted_indexes,
                              const unsigned int block_size,
                              cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_move_points);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (insert.size() + run_block_size - 1)/run_block_size;
    kernel::uniform_grid_move_points<<<num_blocks, run_block_size, 0, stream>>>(d_sorted_points, insert, d_sorted_indexes);
    }

/*!
 * \param d_first Index of first point in each bin.
 * \param d_size Number of points in each bin.
 * \param d_cells The bin assignment of each point.
 * \param N Total number of points.
 * \param Ncells Number of bins.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa kernel::uniform_grid_find_ends
 * \sa kernel::uniform_grid_size_cells
 */
void uniform_grid_find_cells(unsigned int *d_first,
                             unsigned int *d_size,
                             const unsigned int *d_cells,
                             const unsigned int N,
                             const unsigned int Ncells,
                             const unsigned int block_size,
                             cudaStream_t stream)
    {
    // initially, fill all cells as empty
    HOOMD_THRUST::fill(HOOMD_THRUST::cuda::par.on(stream), d_first, d_first+Ncells, UniformGridSentinel);
    cudaMemsetAsync(d_size, 0, sizeof(unsigned int)*Ncells, stream);

    // get the range of primitives covered by each cell
        {
        // clamp block size
        static unsigned int max_block_size_find = UINT_MAX;
        if (max_block_size_find == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_find_ends);
            max_block_size_find = attr.maxThreadsPerBlock;
            }
        const unsigned int run_block_size = (block_size < max_block_size_find) ? block_size : max_block_size_find;

        const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_find_ends<<<num_blocks, run_block_size, 0, stream>>>(d_first, d_size, d_cells, N);
        }

    // compute the number of primitives in each cell
        {
        // clamp block size
        static unsigned int max_block_size_size = UINT_MAX;
        if (max_block_size_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (const void*)kernel::uniform_grid_size_cells);
            max_block_size_size = attr.maxThreadsPerBlock;
            }
        const unsigned int run_block_size = (block_size < max_block_size_size) ? block_size : max_block_size_size;

        const unsigned int num_blocks = (Ncells + run_block_size - 1)/run_block_size;
        kernel::uniform_grid_size_cells<<<num_blocks, run_block_size, 0, stream>>>(d_size, d_first, Ncells);
        }
    }
} // end namespace gpu
} // end namespace neighbor
