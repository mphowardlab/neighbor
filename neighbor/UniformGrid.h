// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_H_
#define NEIGHBOR_UNIFORM_GRID_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "UniformGrid.cuh"

#include "hoomd/HOOMDMath.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/Autotuner.h"

namespace neighbor
{

//! Uniform grid (cell list)
/*!
 * A uniform grid, also called a cell list, divides a volume into equal sized bins (cells).
 * Each primitive is assigned into a bin based on its position. The uniform grid can then be
 * searched by assigning each query object into a bin, and looking in the adjacent bins for
 * primitives that intersect. If the size of a bin is greater than or equal to the search radius,
 * then only 27 bins are searched. The caller requests a nominal width for the bins, and the bin
 * size is expanded to ensure this nominal width is covered by a single bin.
 *
 * As for the LBVH, this class constructs the uniform grid, and the search method is deferred
 * to a separate class. In order to limit memory usage and ensure the uniform grid is constructed
 * deterministically, radix sort is used to determine which primitives belong to each bin. The
 * number of primitives in each bin can be computed from the sorted primitives. Currently, only
 * points can be assigned into the uniform grid.
 *
 * For processing the UniformGrid in GPU kernels, it may be useful to obtain an object containing
 * only the raw pointers to the tree data (see UniformGridData in UniformGrid.cuh). The caller must
 * construct such an object due to the multitude of different access modes that are possible
 * for the GPU data.
 */
class UniformGrid
    {
    public:
        //! Setup a UniformGrid
        UniformGrid(std::shared_ptr<const ExecutionConfiguration> exec_conf, Scalar width);

        //! Destroy a UniformGrid
        ~UniformGrid();

        //! Build the UniformGrid
        void build(const GridPointOp& insert, const Scalar3 lo, const Scalar3 hi, cudaStream_t stream = 0);

        //! Pre-setup function
        void setup(unsigned int N, const Scalar3 lo, const Scalar3 hi)
            {
            sizeGrid(lo,hi);
            allocate(N);
            }

        //! Get lower bound of grid
        Scalar3 getLo() const
            {
            return m_lo;
            }

        //! Get upper bound of grid
        Scalar3 getHi() const
            {
            return m_hi;
            }

        //! Get extent of grid
        Scalar3 getL() const
            {
            return m_L;
            }

        //! Get width of bins in grid
        Scalar3 getWidth() const
            {
            return m_width;
            }

        //! Get number of bins in grid
        uint3 getDimensions() const
            {
            return m_dim;
            }

        //! Get 3D indexer into the grid memory
        Index3D getIndexer() const
            {
            return m_indexer;
            }

        //! Get first index of first particle in cell
        const GlobalArray<unsigned int>& getFirsts() const
            {
            return m_first;
            }

        //! Get number of particles in cell
        const GlobalArray<unsigned int>& getSizes() const
            {
            return m_size;
            }

        //! Get number of primitives
        unsigned int getN() const
            {
            return m_N;
            }

        //! Get cell index for each primitive in sorted order
        const GlobalArray<unsigned int>& getCells() const
            {
            return m_sorted_cells;
            }

        //! Get original primitive indexes in sorted order
        const GlobalArray<unsigned int>& getPrimitives() const
            {
            return m_sorted_indexes;
            }

        //! Get points for primitives in sorted order
        /*!
         * It is convenient to obtain the points in bin-sorted order.
         * For performance, the original index of the primitive is stored
         * in the w component.
         */
        const GlobalArray<Scalar4>& getPoints() const
            {
            return m_points;
            }

        //! Set the kernel autotuner parameters
        /*!
         * \param enable If true, run the autotuners. If false, disable them.
         * \param period Number of builds between running the autotuners.
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tune_bin->setEnabled(enable);
            m_tune_bin->setPeriod(period);

            m_tune_move->setEnabled(enable);
            m_tune_move->setPeriod(period);

            m_tune_cells->setEnabled(enable);
            m_tune_cells->setPeriod(period);
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

        Scalar3 m_lo;           //!< Lower bound of grid
        Scalar3 m_hi;           //!< Upper bound of grid
        Scalar3 m_L;            //!< Length of grid
        Scalar m_min_width;     //!< Minimum width for grid cells
        Scalar3 m_width;        //!< Actual bin width in each dimension
        uint3 m_dim;            //!< Number of bins in each dimension
        Index3D m_indexer;      //!< Indexer into the cell list
        GlobalArray<unsigned int> m_first;     //!< First particle in each cell
        GlobalArray<unsigned int> m_size;      //!< Number of particles in each cell

        unsigned int m_N;                           //!< Number of points
        GlobalArray<unsigned int> m_cells;             //!< Cells for each point
        GlobalArray<unsigned int> m_sorted_cells;      //!< Cells for each point in sorted order
        GlobalArray<unsigned int> m_indexes;           //!< Particles in cell list order
        GlobalArray<unsigned int> m_sorted_indexes;    //!< Primitives sorted in cell order
        GlobalArray<Scalar4> m_points;                 //!< Sorted points

        std::unique_ptr<Autotuner> m_tune_bin;      //!< Autotuner for binning kernel
        std::unique_ptr<Autotuner> m_tune_move;     //!< Autotuner for moving kernel
        std::unique_ptr<Autotuner> m_tune_cells;    //!< Autotuner for cell search kernel

        //! Size the grid
        void sizeGrid(const Scalar3 lo, const Scalar3 hi);

        //! Allocate UniformGrid
        void allocate(unsigned int N);
    };
} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_H_
