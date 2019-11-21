// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_
#define NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_

#include "UniformGrid.h"
#include "UniformGridTraverser.cuh"
#include "TransformOps.h"

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
        UniformGridTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf)
            : m_exec_conf(exec_conf), m_replay(false)
            {
            m_exec_conf->msg->notice(4) << "Constructing UniformGridTraverser" << std::endl;

            m_tune_compress.reset(new Autotuner(32, 1024, 32, 5, 100000, "grid_compress", m_exec_conf));
            m_tune_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "grid_traverse", m_exec_conf));
            }

        //! Destroy uniform grid traverser
        ~UniformGridTraverser()
            {
            m_exec_conf->msg->notice(4) << "Destroying UniformGridTraverser" << std::endl;
            }

        //! Setup uniform grid for traversal
        template<class TransformOpT>
        void setup(const TransformOpT& transform, const UniformGrid& grid, cudaStream_t stream = 0);

        //! Setup uniform grid for traversal
        void setup(const UniformGrid& grid, cudaStream_t stream = 0)
            {
            setup(NullTransformOp(), grid, stream);
            }

        //! Reset (nullify) the setup
        void reset()
            {
            m_replay = false;
            }

        //! Traverse the uniform grid
        template<class OutputOpT, class QueryOpT, class TransformOpT>
        void traverse(OutputOpT& out,
                      const QueryOpT& query,
                      const TransformOpT& transform,
                      const UniformGrid& grid,
                      const GlobalArray<Scalar3>& images = GlobalArray<Scalar3>(),
                      cudaStream_t stream = 0);

        template<class OutputOpT, class QueryOpT>
        void traverse(OutputOpT& out,
                      const QueryOpT& query,
                      const UniformGrid& grid,
                      const GlobalArray<Scalar3>& images = GlobalArray<Scalar3>(),
                      cudaStream_t stream = 0)
            {
            traverse(out, query, NullTransformOp(), grid, images, stream);
            }

        //! Set the kernel autotuner parameters
        /*!
         * \param enable If true, run the autotuners. If false, disable them.
         * \param period Number of traversals between running the autotuners.
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tune_compress->setEnabled(enable);
            m_tune_compress->setPeriod(period);

            m_tune_traverse->setEnabled(enable);
            m_tune_traverse->setPeriod(period);
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration

        GlobalArray<Scalar4> m_data;    //!< Internal representation of primitive data
        GlobalArray<uint2> m_range;     //!< Internal representation of cell ranges

        std::unique_ptr<Autotuner> m_tune_compress; //!< Autotuner for compression kernel
        std::unique_ptr<Autotuner> m_tune_traverse; //!< Autotuner for traversal kernel

        template<class TransformOpT>
        void compress(const UniformGrid& grid, const TransformOpT& transform, cudaStream_t stream);

        bool m_replay;  //!< If true, the compressed structure has already been set explicitly
    };

/*!
 * \param transform Transformation operation for cached primitive indexes.
 * \param grid Grid to traverse.
 *
 * \tparam TransformOpT The type of transformation operation.
 *
 * This method just calls the compress method on the grid, and marks that this has been done
 * internally so that subsequent calls to traverse do not compress. This is useful if the same
 * grid is going to be traversed multiple times. It is the caller's responsibility to ensure
 * that the transform op and grid do not change between setup and traversal, or the result will
 * be incorrect.
 *
 * To clear a setup, call reset().
 */
template<class TransformOpT>
void UniformGridTraverser::setup(const TransformOpT& transform, const UniformGrid& grid, cudaStream_t stream)
    {
    if (grid.getN() == 0) return;

    compress(grid, transform, stream);
    m_replay = true;
    }

/*!
 * \param out Number of overlaps per sphere.
 * \param spheres Test spheres.
 * \param N Number of test spheres.
 * \param grid UniformGrid to traverse.
 * \param box HOOMD-blue BoxDim corresponding giving the periodicity of the system.
 *
 * The format for a \a sphere is (x,y,z,R), where R is the radius of the sphere.
 *
 * The caller must ensure that all \a spheres lie within the bounds of the \a grid, or
 * traversal will produce incorrect results.
 */
//! Traverse the uniform grid
template<class OutputOpT, class QueryOpT, class TransformOpT>
void UniformGridTraverser::traverse(OutputOpT& out,
                                    const QueryOpT& query,
                                    const TransformOpT& transform,
                                    const UniformGrid& grid,
                                    const GlobalArray<Scalar3>& images,
                                    cudaStream_t stream)
    {
    // don't traverse empty grid
    if (grid.getN() == 0) return;

    // kernel uses int32 bitflags for the images, so limit to 32 images
    const unsigned int Nimages = images.getNumElements();
    if (Nimages > 32)
        {
        m_exec_conf->msg->error() << "A maximum of 32 image vectors are supported by LBVH traversers." << std::endl;
        throw std::runtime_error("Too many images (>32) in LBVH traverser.");
        }

    // setup if this is not a replay
    if (!m_replay)
        setup(transform, grid, stream);

    // cell list data
    ArrayHandle<unsigned int> d_first(grid.getFirsts(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_size(grid.getSizes(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_data(m_data, access_location::device, access_mode::read);
    ArrayHandle<uint2> d_range(m_range, access_location::device, access_mode::read);
    // uniform grid data
    gpu::UniformGridCompressedData g;
    g.data = d_data.data;
    g.range = d_range.data;
    g.lo = grid.getLo();
    g.hi = grid.getHi();
    g.L = grid.getL();
    g.width = grid.getWidth();
    g.indexer = grid.getIndexer();

    ArrayHandle<Scalar3> d_images(images, access_location::device, access_mode::read);

    m_tune_traverse->begin();
    gpu::uniform_grid_traverse(out,
                               g,
                               query,
                               d_images.data,
                               Nimages,
                               m_tune_traverse->getParam(),
                               stream);
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_traverse->end();
    }

template<class TransformOpT>
void UniformGridTraverser::compress(const UniformGrid& grid, const TransformOpT& transform, cudaStream_t stream)
    {
    if (grid.getN() > m_data.getNumElements())
        {
        GlobalArray<Scalar4> tmp(grid.getN(), m_exec_conf);
        m_data.swap(tmp);
        }

    const unsigned int Ncell = grid.getIndexer().getNumElements();
    if (Ncell > m_range.getNumElements())
        {
        GlobalArray<uint2> tmp(Ncell, m_exec_conf);
        m_range.swap(tmp);
        }

    // minimum cell list data to compress / transform (other values are undefined!)
    ArrayHandle<unsigned int> d_first(grid.getFirsts(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_size(grid.getSizes(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_points(grid.getPoints(), access_location::device, access_mode::read);
    gpu::UniformGridData g;
    g.first = d_first.data;
    g.size = d_size.data;
    g.points = d_points.data;

    // minimum compressed data (other values are undefined!)
    ArrayHandle<Scalar4> d_data(m_data, access_location::device, access_mode::overwrite);
    ArrayHandle<uint2> d_range(m_range, access_location::device, access_mode::overwrite);
    gpu::UniformGridCompressedData cgrid;
    cgrid.data = d_data.data;
    cgrid.range = d_range.data;

    m_tune_compress->begin();
    gpu::uniform_grid_compress(cgrid,
                               transform,
                               g,
                               grid.getN(),
                               Ncell,
                               m_tune_compress->getParam(),
                               stream);
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_compress->end();
    }

} // end namespace neighbor

#endif // NEIGHBOR_UNIFORM_GRID_TRAVERSER_H_
