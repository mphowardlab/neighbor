// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "UniformGrid.h"
#include "UniformGrid.cuh"
#include "hoomd/CachedAllocator.h"

namespace neighbor
{

/*!
 * \param exec_conf HOOMD-blue execution configuration
 * \param lo The lower bound of the grid.
 * \param hi The upper bound of the grid.
 * \param width The nominal width of a bin.
 *
 * The grid is sized at construction, and so \a lo and \a hi must be big enough to include
 * all primitives that will be placed into the grid. The nominal \a width is respected when
 * sizing bins so that a bin is always at least as wide as \a width. Typically, \a width
 * should be set equal to the traversal radius so that only 27 neighbors are searched.
 *
 * Some low-level memory allocations occur on construction, but most are deferred until the
 * first call to ::build.
 */
UniformGrid::UniformGrid(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                         const Scalar3 lo,
                         const Scalar3 hi,
                         Scalar width)
    : m_exec_conf(exec_conf)
    {
    m_exec_conf->msg->notice(4) << "Constructing UniformGrid" << std::endl;

    m_tune_bin.reset(new Autotuner(32, 1024, 32, 5, 100000, "grid_bin", m_exec_conf));
    m_tune_cells.reset(new Autotuner(32, 1024, 32, 5, 100000, "grid_find_cells", m_exec_conf));

    sizeGrid(lo, hi, width);
    }

UniformGrid::~UniformGrid()
    {
    m_exec_conf->msg->notice(4) << "Destroying UniformGrid" << std::endl;
    }

/*!
 * \param points Point primitives
 * \param N Number of primitives
 *
 * The \a points are placed into the grid using a radix sort approach that consumes O(N) memory.
 * This approach is good because it is (1) deterministic and (2) has well-behaved memory
 * requirements that do not explode too quickly in sparse or non-uniform primitive distributions.
 *
 * It is the responsibility of the caller to ensure that all \a points lie between \a lo and \a hi
 * used to construct the UniformGrid. An error will not be raised. Instead, points will simply be
 * clamped into the box.
 */
void UniformGrid::build(const GlobalArray<Scalar4>& points, unsigned int N)
    {
    allocate(N);

    // assign points into bins
        {
        ArrayHandle<unsigned int> d_cells(m_cells, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);

        // uniform grid data
        gpu::UniformGridData grid;
        grid.first = NULL;
        grid.size = NULL;
        grid.point = NULL;
        grid.lo = m_lo;
        grid.L = m_L;
        grid.width = m_width;
        grid.indexer = m_indexer;

        m_tune_bin->begin();
        gpu::uniform_grid_bin_points(d_cells.data, d_indexes.data, d_points.data, grid, N, m_tune_bin->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_bin->end();
        }

    // sort points
        {
        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_cells(m_cells, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_cells(m_sorted_cells, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_sorted_points(m_points, access_location::device, access_mode::overwrite);

            void *d_tmp = NULL;
            size_t tmp_bytes = 0;
            gpu::uniform_grid_sort_points(d_tmp,
                                          tmp_bytes,
                                          d_cells.data,
                                          d_sorted_cells.data,
                                          d_indexes.data,
                                          d_sorted_indexes.data,
                                          d_points.data,
                                          d_sorted_points.data,
                                          m_N);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void *)d_alloc();

            swap = gpu::uniform_grid_sort_points(d_tmp,
                                                 tmp_bytes,
                                                 d_cells.data,
                                                 d_sorted_cells.data,
                                                 d_indexes.data,
                                                 d_sorted_indexes.data,
                                                 d_points.data,
                                                 d_sorted_points.data,
                                                 m_N);
            }
        if (swap.x) m_sorted_cells.swap(m_cells);
        if (swap.y) m_sorted_indexes.swap(m_indexes);
        }

    // find beginning of cells in primitives list + number in each cell
        {
        ArrayHandle<unsigned int> d_first(m_first, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_size(m_size, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_cells(m_sorted_cells, access_location::device, access_mode::read);

        m_tune_cells->begin();
        gpu::uniform_grid_find_cells(d_first.data,
                                     d_size.data,
                                     d_sorted_cells.data,
                                     N,
                                     m_indexer.getNumElements(),
                                     m_tune_cells->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_cells->end();
        }
    }

/*!
 * \param lo Lower bound of grid.
 * \param hi Upper bound of grid.
 * \param width Nominal width of a cell.
 *
 * The bins are sized, ensuring there is at least 1 bin in each dimension and that
 * all bins are at least as large as \a width.
 */
void UniformGrid::sizeGrid(const Scalar3 lo, const Scalar3 hi, Scalar width)
    {
    m_lo = lo;
    m_L = hi-lo;
    // round down the number of bins to get the nominal number in each dimension
    m_dim = make_uint3(static_cast<unsigned int>(m_L.x/width),
                       static_cast<unsigned int>(m_L.y/width),
                       static_cast<unsigned int>(m_L.z/width));
    if (m_dim.x == 0) m_dim.x = 1;
    if (m_dim.y == 0) m_dim.y = 1;
    if (m_dim.z == 0) m_dim.z = 1;
    m_indexer = Index3D(m_dim.x, m_dim.y, m_dim.z);

    // true bin width based on size
    m_width = make_scalar3(m_L.x/m_dim.x, m_L.y/m_dim.y, m_L.z/m_dim.z);

    // allocate memory per grid cell
    GlobalArray<unsigned int> first(m_indexer.getNumElements(), m_exec_conf);
    m_first.swap(first);
    GlobalArray<unsigned int> size(m_indexer.getNumElements(), m_exec_conf);
    m_size.swap(size);
    }

/*!
 * \param N Number of primitives.
 *
 * Per-primitive memory is allocated, which usually occurs at build time.
 */
void UniformGrid::allocate(unsigned int N)
    {
    // don't do anything if already allocated at this size
    if (N == m_N) return;

    // per-particle memory
    m_N = N;

    GlobalArray<unsigned int> cells(N, m_exec_conf);
    m_cells.swap(cells);

    GlobalArray<unsigned int> sorted_cells(N, m_exec_conf);
    m_sorted_cells.swap(sorted_cells);

    GlobalArray<unsigned int> indexes(N, m_exec_conf);
    m_indexes.swap(indexes);

    GlobalArray<unsigned int> sorted_indexes(N, m_exec_conf);
    m_sorted_indexes.swap(sorted_indexes);

    GlobalArray<Scalar4> points(N, m_exec_conf);
    m_points.swap(points);
    }
} // end namespace neighbor
