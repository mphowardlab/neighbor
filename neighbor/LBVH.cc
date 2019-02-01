// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVH.h"
#include "LBVH.cuh"
#include "hoomd/CachedAllocator.h"

namespace neighbor
{

/*!
 * \param exec_conf HOOMD-blue execution configuration
 *
 * The constructor defers memory initialization to the first call to ::build.
 */
LBVH::LBVH(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_exec_conf(exec_conf), m_root(gpu::LBVHSentinel), m_N(0), m_N_internal(0), m_N_nodes(0)
    {
    m_exec_conf->msg->notice(4) << "Constructing LBVH" << std::endl;

    m_tune_gen_codes.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_gen_codes", m_exec_conf));
    m_tune_gen_tree.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_gen_tree", m_exec_conf));
    m_tune_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000, "lbvh_bubble", m_exec_conf));
    }

LBVH::~LBVH()
    {
    m_exec_conf->msg->notice(4) << "Destroying LBVH" << std::endl;
    }

/*!
 * \param N Number of primitives
 *
 * Initializes the memory for an LBVH holding \a N primitives. The memory
 * requirements are O(N). Every node is allocated 1 integer (4B) holding the parent
 * node and 2 float3s (24B) holding the bounding box. Each internal node additional
 * is allocated 2 integers (8B) holding their children and 1 integer (4B) holding a
 * flag used to backpropagate the bounding boxes.
 *
 * Primitive sorting requires 4N integers of storage, which is allocated persistently
 * to avoid the overhead of repeated malloc / free calls.
 *
 * \note
 * Additional calls to allocate are ignored if \a N has not changed from
 * the previous call.
 *
 */
void LBVH::allocate(unsigned int N)
    {
    // don't do anything if already allocated
    if (N == m_N)
        {
        m_root = 0;
        return;
        }

    m_root = 0;
    m_N = N;
    m_N_internal = (m_N > 0) ? m_N - 1 : 0;
    m_N_nodes = m_N + m_N_internal;

    // tree node memory
    GlobalArray<int> parent(m_N_nodes, m_exec_conf);
    m_parent.swap(parent);

    GlobalArray<int> left(m_N_internal, m_exec_conf);
    m_left.swap(left);

    GlobalArray<int> right(m_N_internal, m_exec_conf);
    m_right.swap(right);

    GlobalArray<float3> lo(m_N_nodes, m_exec_conf);
    m_lo.swap(lo);

    GlobalArray<float3> hi(m_N_nodes, m_exec_conf);
    m_hi.swap(hi);

    // morton code generation / sorting memory
    GlobalArray<unsigned int> codes(m_N, m_exec_conf);
    m_codes.swap(codes);

    GlobalArray<unsigned int> indexes(m_N, m_exec_conf);
    m_indexes.swap(indexes);

    GlobalArray<unsigned int> sorted_codes(m_N, m_exec_conf);
    m_sorted_codes.swap(sorted_codes);

    GlobalArray<unsigned int> sorted_indexes(m_N, m_exec_conf);
    m_sorted_indexes.swap(sorted_indexes);

    GlobalArray<unsigned int> locks(m_N_internal, m_exec_conf);
    m_locks.swap(locks);
    }

/*!
 * \param points Point primitives
 * \param N Number of primitives
 * \param lo Lower bound of the scene
 * \param hi Upper bound of the scene
 *
 * The LBVH is constructed using the algorithm due to Karras using 30-bit Morton codes.
 * The caller should ensure that all \a points lie within \a lo and \a hi for best performance.
 * Points lying outside this range are clamped to it during the Morton code calculation, which
 * may lead to a low quality LBVH.
 *
 * \note
 * Currently, small LBVHs (`N` <= 2) are not implemented, and an error will be raised.
 */
void LBVH::build(const GlobalArray<Scalar4>& points, unsigned int N, const Scalar3 lo, const Scalar3 hi)
    {
    if (N < 2)
        {
        m_exec_conf->msg->error() << "Small LBVHs (N=0,1,2) are currently not implemented." << std::endl;
        throw std::runtime_error("Small LBVHs are not implemented.");
        }
    // resize memory for the tree
    allocate(N);

    // calculate morton codes
        {
        ArrayHandle<unsigned int> d_codes(m_codes, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);

        m_tune_gen_codes->begin();
        gpu::lbvh_gen_codes(d_codes.data, d_indexes.data, d_points.data, lo, hi, m_N, m_tune_gen_codes->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_gen_codes->end();
        }

    // sort morton codes
        {
        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_codes(m_codes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_codes(m_sorted_codes, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);

            void *d_tmp = NULL;
            size_t tmp_bytes = 0;
            gpu::lbvh_sort_codes(d_tmp,
                                 tmp_bytes,
                                 d_codes.data,
                                 d_sorted_codes.data,
                                 d_indexes.data,
                                 d_sorted_indexes.data,
                                 m_N);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void *)d_alloc();

            swap = gpu::lbvh_sort_codes(d_tmp,
                                        tmp_bytes,
                                        d_codes.data,
                                        d_sorted_codes.data,
                                        d_indexes.data,
                                        d_sorted_indexes.data,
                                        m_N);
            }
        if (swap.x) m_sorted_codes.swap(m_codes);
        if (swap.y) m_sorted_indexes.swap(m_indexes);
        }

    // process hierarchy and bubble aabbs
        {
        ArrayHandle<int> d_parent(m_parent, access_location::device, access_mode::overwrite);
        ArrayHandle<int> d_left(m_left, access_location::device, access_mode::overwrite);
        ArrayHandle<int> d_right(m_right, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);
        ArrayHandle<float3> d_lo(m_lo, access_location::device, access_mode::overwrite);
        ArrayHandle<float3> d_hi(m_hi, access_location::device, access_mode::overwrite);

        gpu::LBVHData tree;
        tree.parent = d_parent.data;
        tree.left = d_left.data;
        tree.right = d_right.data;
        tree.primitive = d_sorted_indexes.data;
        tree.lo = d_lo.data;
        tree.hi = d_hi.data;
        tree.root = m_root;

        // generate the tree hierarchy
        ArrayHandle<unsigned int> d_sorted_codes(m_sorted_codes, access_location::device, access_mode::read);

        m_tune_gen_tree->begin();
        gpu::lbvh_gen_tree(tree, d_sorted_codes.data, m_N, m_tune_gen_tree->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_gen_tree->end();

        // bubble up the aabbs
        ArrayHandle<unsigned int> d_locks(m_locks, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_points(points, access_location::device, access_mode::read);

        m_tune_bubble->begin();
        gpu::lbvh_bubble_aabbs(tree, d_locks.data, d_points.data, m_N, m_tune_bubble->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_bubble->end();
        }
    }
} // end namespace neighbor
