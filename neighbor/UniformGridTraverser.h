// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_
#define NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_

#include "UniformGrid.h"
#include "hoomd/BoxDim.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/Autotuner.h"

namespace neighbor
{

//! Uniform grid traversal method
/*!
 * Basic implementation of a grid traverser. Search spheres are dropped down onto the grid.
 * Each sphere intersects against points in its bin and the 26 neighbor bins, assuming that
 * (1) the grid is periodic in 3D and (2) the search radius for the sphere is smaller than
 * the grid bin width. These assumptions could be relaxed by a more sophisticated traversal
 * scheme. However, this traverser primarily exists as a benchmark for the LBVH traverser,
 * and so this functionality has not yet been implemented.
 *
 * The algorithm for finding neighboring cells is essentially the stencil scheme described
 * elsewhere. The stencil is constructed once based on the size of the grid and should account
 * for periodic boundary conditions appropriately, even in small boxes.
 *
 * For systems with small numbers of test spheres, parallelism can be enhanced by searching
 * the grid using multiple threads per test sphere using ::setThreads. By default, only 1
 * thread is used per test sphere.
 */
class UniformGridTraverser
    {
    public:
        //! Create uniform grid traverser
        UniformGridTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Destroy uniform grid traverser
        virtual ~UniformGridTraverser();

        //! Traverse the uniform grid using the primitives it contains (simple stencil)
        void traverse(const GlobalArray<unsigned int>& out,
                      const GlobalArray<Scalar4> spheres,
                      unsigned int N,
                      const UniformGrid& grid,
                      const BoxDim& box);

        //! Get the number of threads per traversal sphere
        unsigned char getThreads() const
            {
            return m_threads;
            }

        //! Set the number of threads per traversal sphere
        void setThreads(unsigned char threads);

        //! Set the kernel autotuner parameters
        /*!
         * \param enable If true, run the autotuners. If false, disable them.
         * \param period Number of traversals between running the autotuners.
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tune_traverse->setEnabled(enable);
            m_tune_traverse->setPeriod(period);
            }

    protected:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration

        unsigned char m_threads;    //!< Number of threads per particle
        unsigned int m_num_stencil; //!< Number of stencils
        uint3 m_stencil_dim;        //!< Dimension stencil was made for
        GlobalArray<int3> m_stencil;   //!< Stencil for traversal

        std::unique_ptr<Autotuner> m_tune_traverse; //!< Autotuner for traversal kernel

        //! Construct the traversal stencil for the grid
        void createStencil(const UniformGrid& grid);
    };

} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_
