// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_CUH_
#define NEIGHBOR_UNIFORM_GRID_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace neighbor
{
namespace gpu
{

// UniformGrid sentinel has value of max unsigned int
const unsigned int UniformGridSentinel=0xffffffff;

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define HOSTDEVICE inline
#endif

//! Uniform grid raw data
/*!
 * UniformGridData is a lightweight struct representation of the UniformGrid.
 * It is useful for passing grid data to a CUDA kernel. It is valid to set a pointer to
 * NULL if it is not required, but the caller must do so responsibly.
 *
 * UniformGridData also supplies two methods for converting a point to a bin within the
 * grid and for wrapping a bin index back into the grid.
 */
struct UniformGridData
    {
    unsigned int* first;    //!< First primitive index for the cell
    unsigned int* size;     //!< Number of primitives in the cell
    Scalar4* point;         //!< Sorted points
    Scalar3 lo;             //!< Lower bound of grid
    Scalar3 L;              //!< Size of grid
    Scalar3 width;          //!< Width of bins
    Index3D indexer;        //!< 3D indexer into the grid memory

    HOSTDEVICE int3 toCell(const Scalar3& r) const
        {
        // convert position into fraction, then bin
        const Scalar3 f = (r-lo)/L;
        int3 bin = make_int3(static_cast<int>(f.x * indexer.getW()),
                             static_cast<int>(f.y * indexer.getH()),
                             static_cast<int>(f.z * indexer.getD()));

        // this is forced binning on the edges to handle roundoff, it
        // is the caller's responsibility to make sure this is reasonable.
        if (bin.x >= (int)indexer.getW()) bin.x = indexer.getW() - 1;
        if (bin.x < 0) bin.x = 0;
        if (bin.y >= (int)indexer.getH()) bin.y = indexer.getH() - 1;
        if (bin.y < 0) bin.y = 0;
        if (bin.z >= (int)indexer.getD()) bin.z = indexer.getD() - 1;
        if (bin.z < 0) bin.z = 0;

        return bin;
        }

    HOSTDEVICE int3 wrap(const int3& bin) const
        {
        int3 wrap = bin;
        if (wrap.x >= (int)indexer.getW()) wrap.x -= indexer.getW();
        if (wrap.x < 0) wrap.x += indexer.getW();
        if (wrap.y >= (int)indexer.getH()) wrap.y -= indexer.getH();
        if (wrap.y < 0) wrap.y += indexer.getH();
        if (wrap.z >= (int)indexer.getD()) wrap.z -= indexer.getD();
        if (wrap.z < 0) wrap.z += indexer.getD();

        return wrap;
        }
    };
#undef HOSTDEVICE

//! Bin points into the grid
void uniform_grid_bin_points(unsigned int *d_cells,
                             unsigned int *d_primitives,
                             const Scalar4 *d_points,
                             const UniformGridData grid,
                             const unsigned int N,
                             const unsigned int block_size);

//! Sort points by bin assignment
uchar2 uniform_grid_sort_points(void *d_tmp,
                                size_t &tmp_bytes,
                                unsigned int *d_cells,
                                unsigned int *d_sorted_cells,
                                unsigned int *d_indexes,
                                unsigned int *d_sorted_indexes,
                                const Scalar4 *d_points,
                                Scalar4 *d_sorted_points,
                                const unsigned int N);

//! Find the first point and size of each bin
void uniform_grid_find_cells(unsigned int *d_first,
                             unsigned int *d_size,
                             const unsigned int *d_cells,
                             const unsigned int N,
                             const unsigned int Ncells,
                             const unsigned int block_size);

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_CUH_
