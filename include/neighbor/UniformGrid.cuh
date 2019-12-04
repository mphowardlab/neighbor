// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_CUH_
#define NEIGHBOR_UNIFORM_GRID_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define HOSTDEVICE inline
#endif

namespace neighbor
{

struct GridPointOp
    {
    //! Constructor
    /*!
     * \param points_ Points array (x,y,z,_)
     * \param N_ The number of points
     */
    GridPointOp(const Scalar4 *points_, unsigned int N_)
        : points(points_), N(N_)
        {}

    //! Get the point
    /*!
     * \param idx the index of the primitive
     *
     * \returns The point
     */
    HOSTDEVICE Scalar3 get(const unsigned int idx) const
        {
        const Scalar4 point = points[idx];
        return make_scalar3(point.x, point.y, point.z);
        }

    //! Get the number of leaf node bounding volumes
    /*!
     * \returns The initial number of leaf nodes
     */
    HOSTDEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4 *points;
    unsigned int N;
    };

namespace gpu
{

// UniformGrid sentinel has value of max unsigned int
const unsigned int UniformGridSentinel=0xffffffff;

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
    Scalar4* points;        //!< Points data
    Scalar3 lo;             //!< Lower bound of grid
    Scalar3 hi;             //!< Upper bound of grid
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
        return bin;
        }
    };

#undef HOSTDEVICE

//! Bin points into the grid
void uniform_grid_bin_points(unsigned int *d_cells,
                             unsigned int *d_primitives,
                             const GridPointOp& insert,
                             const UniformGridData grid,
                             const unsigned int block_size,
                             cudaStream_t stream = 0);

//! Sort points by bin assignment
uchar2 uniform_grid_sort_points(void *d_tmp,
                                size_t &tmp_bytes,
                                unsigned int *d_cells,
                                unsigned int *d_sorted_cells,
                                unsigned int *d_indexes,
                                unsigned int *d_sorted_indexes,
                                const unsigned int N,
                                cudaStream_t stream = 0);

void uniform_grid_move_points(Scalar4 *d_sorted_points,
                              const GridPointOp& insert,
                              const unsigned int *d_sorted_indexes,
                              const unsigned int block_size,
                              cudaStream_t stream = 0);

//! Find the first point and size of each bin
void uniform_grid_find_cells(unsigned int *d_first,
                             unsigned int *d_size,
                             const unsigned int *d_cells,
                             const unsigned int N,
                             const unsigned int Ncells,
                             const unsigned int block_size,
                             cudaStream_t stream = 0);

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_CUH_
