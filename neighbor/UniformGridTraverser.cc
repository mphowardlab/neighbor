// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGridTraverser.h"
#include "UniformGridTraverser.cuh"

namespace neighbor
{

/*!
 * \param exec_conf HOOMD-blue execution configuration.
 */
UniformGridTraverser::UniformGridTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_exec_conf(exec_conf), m_threads(1), m_num_stencil(0)
    {
    m_exec_conf->msg->notice(4) << "Constructing UniformGridTraverser" << std::endl;

    m_tune_traverse.reset(new Autotuner(32, 1024, 32, 5, 100000, "grid_traverse", m_exec_conf));

    m_stencil_dim = make_uint3(0,0,0);
    }

UniformGridTraverser::~UniformGridTraverser()
    {
    m_exec_conf->msg->notice(4) << "Destroying UniformGridTraverser" << std::endl;
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
void UniformGridTraverser::traverse(const GPUArray<unsigned int>& out,
                                    const GPUArray<Scalar4> spheres,
                                    unsigned int N,
                                    const UniformGrid& grid,
                                    const BoxDim& box)
    {
    // box must be periodic in all directions
    if (!box.getPeriodic().x || !box.getPeriodic().y || !box.getPeriodic().z)
        {
        m_exec_conf->msg->error() << "Box must be periodic in all dimensions for now" << std::endl;
        throw std::runtime_error("Box must be periodic");
        }

    // create stencil if grid dimension has changed
    const uint3 dim = grid.getDimensions();
    if (dim.x != m_stencil_dim.x || dim.y != m_stencil_dim.y || dim.z != m_stencil_dim.z)
        {
        createStencil(grid);
        m_stencil_dim = grid.getDimensions();
        }

    // counts
    ArrayHandle<unsigned int> d_out(out, access_location::device, access_mode::overwrite);

    // primitives
    ArrayHandle<Scalar4> d_spheres(spheres, access_location::device, access_mode::read);

    // cell list data
    ArrayHandle<unsigned int> d_first(grid.getFirsts(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_size(grid.getSizes(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_primitives(grid.getPoints(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_stencil(m_stencil, access_location::device, access_mode::read);
    // uniform grid data
    gpu::UniformGridData g;
    g.first = d_first.data;
    g.size = d_size.data;
    g.point = d_primitives.data;
    g.lo = grid.getLo();
    g.L = grid.getL();
    g.width = grid.getWidth();
    g.indexer = grid.getIndexer();

    m_tune_traverse->begin();
    gpu::uniform_grid_traverse(d_out.data,
                               // test point traversal
                               d_spheres.data,
                               g,
                               d_stencil.data,
                               m_num_stencil,
                               box,
                               // block config
                               N,
                               m_threads,
                               m_tune_traverse->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tune_traverse->end();
    }

/*!
 * \param grid UniformGrid to create traversal stencil for
 *
 * The standard 27-point stencil is constructed, while ensuring that
 * small numbers of cells in each dimension are handled correctly.
 */
void UniformGridTraverser::createStencil(const UniformGrid& grid)
    {
    // size stencil assuming standard 27 cells (if at least 3 cells per dim) and allocate
    const uint3 dim = grid.getDimensions();
    const char3 min = make_char3(-(dim.x > 2), -(dim.y > 2), -(dim.z > 2));
    const char3 max = make_char3( (dim.x > 1),  (dim.y > 1),  (dim.z > 1));
    m_num_stencil = (max.x-min.x+1)*(max.y-min.y+1)*(max.z-min.z+1);
    if (m_stencil.getNumElements() < m_num_stencil)
        {
        GPUArray<int3> tmp(m_num_stencil, m_exec_conf);
        m_stencil.swap(tmp);
        }

    // push stencils into the list
    unsigned int idx = 0;
    ArrayHandle<int3> h_stencil(m_stencil, access_location::host, access_mode::overwrite);
    for (char i=min.x; i <= max.x; ++i)
        {
        for (char j=min.y; j <= max.y; ++j)
            {
            for (char k=min.z; k <= max.z; ++k)
                {
                h_stencil.data[idx++] = make_int3(i, j, k);
                }
            }
        }
    }

/*!
 * \param threads Number of threads per traversal sphere.
 *
 * The \a threads must be a power of 2 that is greater than or equal to
 * 1 and less than or equal to 32. The reason for this restriction is that
 * the CUDA kernels use warp scan/reduction to process the particles, and so
 * the number of threads is set by the CUDA warp size.
 */
void UniformGridTraverser::setThreads(unsigned char threads)
    {
    if (threads > 0)
        {
        if  (threads <= 32 && (threads & (threads - 1)) == 0)
            {
            m_threads = threads;
            }
        else
            {
            m_exec_conf->msg->error() << "Threads must be a power of 2 >= 1 and <= 32" << std::endl;
            throw std::runtime_error("Threads must be a power of 2 >= 1 and <= 32");
            }
        }
    else
        {
        m_exec_conf->msg->error() << "Threads must be a power of 2 >= 1 and <= 32" << std::endl;
        throw std::runtime_error("Threads must be a power of 2 >=1 and <= 32");
        }
    }

} // end namespace neighbor
