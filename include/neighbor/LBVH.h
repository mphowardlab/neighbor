// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_H_
#define NEIGHBOR_LBVH_H_

#include <thrust/device_vector.h>

#include "Autotuner.h"
#include "kernels/LBVHKernels.cuh"

namespace neighbor
{
//! Linear bounding volume hierarchy
/*!
 * A linear bounding hierarchy (LBVH) is a binary tree structure that can be used for overlap
 * or collision detection. Briefly, a leaf node in the tree encloses a single primitive
 * object (currently assumed to be a point). The next layer of (internal) nodes in the tree
 * encloses two leaf nodes, while higher layers enclose two internal nodes. The
 * tree terminates in a single root node. The nodes are fit by axis-aligned bounding boxes.
 * The bounding boxes have lower and upper bounds in Cartesian coordinates that enclose
 * all primitives that are children of a node. The LBVH can be traversed by testing for
 * intersections of some test object against a node and then descending to the children of
 * intersected nodes until the primitives are reached.
 *
 * The LBVH class constructs such a hierarchy on the GPU using the ::build method. The
 * point primitives must be supplied as Scalar4s (defined in HOOMD-blue precision model).
 * Regardless of the precision of the primitives, the bounding boxes are stored in
 * single-precision in a way that preserves correctness of the tree. The build algorithm
 * is due to Karras with 30-bit Morton codes to sort primitives.
 *
 * The data needed for tree traversal can be accessed by the appropriate methods. It is
 * recommended to use the sorted primitive order for traversal for best performance
 * (see ::getPrimitives). Since different traversal schemes can be prescribed or additional
 * tree processing may occur, the traversal is delegated to a LBVHTraverser object.
 * The memory layout of the data arrays is such that all internal nodes precede all
 * leaf nodes, and the root node is node 0.
 *
 * For processing the LBVH in GPU kernels, it may be useful to obtain an object containing
 * only the raw pointers to the tree data (see LBVHData in LBVH.cuh). The caller must
 * construct such an object due to the multitude of different access modes that are possible
 * for the GPU data.
 */
class LBVH
    {
    public:
        //! Setup an unallocated LBVH
        LBVH();

        //! Build the LBVH
        template<class InsertOpT>
        void build(const InsertOpT& insert, const float3 lo, const float3 hi, cudaStream_t stream = 0);

        //! Pre-setup function
        void setup(unsigned int N)
            {
            allocate(N);
            }

        //! Get the LBVH root node
        int getRoot() const
            {
            return m_root;
            }

        //! Get the number of primitives
        unsigned int getN() const
            {
            return m_N;
            }

        //! Get the number of internal nodes
        unsigned int getNInternal() const
            {
            return m_N_internal;
            }

        //! Get the total number of nodes
        unsigned int getNNodes() const
            {
            return m_N_nodes;
            }

        //! Get the array of parents of a given node
        const thrust::device_vector<int>& getParents() const
            {
            return m_parent;
            }

        //! Get the array of left children of a given node
        const thrust::device_vector<int>& getLeftChildren() const
            {
            return m_left;
            }

        //! Get the array of right children of a given node
        const thrust::device_vector<int>& getRightChildren() const
            {
            return m_right;
            }

        //! Get the lower bounds of the boxes enclosing a node
        const thrust::device_vector<float3>& getLowerBounds() const
            {
            return m_lo;
            }

        //! Get the upper bounds of the boxes enclosing a node
        const thrust::device_vector<float3>& getUpperBounds() const
            {
            return m_hi;
            }

        //! Get the original indexes of the primitives in each leaf node
        const thrust::device_vector<unsigned int>& getPrimitives() const
            {
            return m_sorted_indexes;
            }

        const gpu::LBVHData data()
            {
            gpu::LBVHData tree;

            tree.parent = thrust::raw_pointer_cast(m_parent.data());
            tree.left = thrust::raw_pointer_cast(m_left.data());
            tree.right = thrust::raw_pointer_cast(m_right.data());
            tree.primitive = thrust::raw_pointer_cast(m_sorted_indexes.data());
            tree.lo = thrust::raw_pointer_cast(m_lo.data());
            tree.hi = thrust::raw_pointer_cast(m_hi.data());
            tree.root = m_root;

            return tree;
            }

        //! Set the kernel autotuner parameters
        /*!
         * \param enable If true, run the autotuners. If false, disable them.
         * \param period Number of builds between running the autotuners.
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tune_gen_codes->setEnabled(enable);
            m_tune_gen_codes->setPeriod(period);

            m_tune_gen_tree->setEnabled(enable);
            m_tune_gen_tree->setPeriod(period);

            m_tune_bubble->setEnabled(enable);
            m_tune_bubble->setPeriod(period);
            }

    private:
        int m_root;                 //!< Root index
        unsigned int m_N;           //!< Number of primitives in the tree
        unsigned int m_N_internal;  //!< Number of internal nodes in tree
        unsigned int m_N_nodes;     //!< Number of nodes in the tree

        thrust::device_vector<int> m_parent; //!< Parent node
        thrust::device_vector<int> m_left;   //!< Left child
        thrust::device_vector<int> m_right;  //!< Right child
        thrust::device_vector<float3> m_lo;  //!< Lower bound of AABB
        thrust::device_vector<float3> m_hi;  //!< Upper bound of AABB

        thrust::device_vector<unsigned int> m_codes;             //!< Morton codes
        thrust::device_vector<unsigned int> m_indexes;           //!< Primitive indexes
        thrust::device_vector<unsigned int> m_sorted_codes;      //!< Sorted morton codes
        thrust::device_vector<unsigned int> m_sorted_indexes;    //!< Sorted primitive indexes

        thrust::device_vector<unsigned int> m_locks; //!< Node locks for generating aabb hierarchy

        std::unique_ptr<Autotuner> m_tune_gen_codes;    //!< Autotuner for generating Morton codes kernel
        std::unique_ptr<Autotuner> m_tune_gen_tree;     //!< Autotuner for generating tree hierarchy kernel
        std::unique_ptr<Autotuner> m_tune_bubble;       //!< Autotuner for AABB bubble kernel

        //! Allocate
        void allocate(unsigned int N);
    };

/*!
 * The constructor defers memory initialization to the first call to ::build.
 */
LBVH::LBVH()
    : m_root(gpu::LBVHSentinel), m_N(0), m_N_internal(0), m_N_nodes(0)
    {
    m_tune_gen_codes.reset(new Autotuner(32, 1024, 32, 5, 100000));
    m_tune_gen_tree.reset(new Autotuner(32, 1024, 32, 5, 100000));
    m_tune_bubble.reset(new Autotuner(32, 1024, 32, 5, 100000));
    }

/*!
 * \param insert The insert operation determining AABB extents of primitives
 * \param N Number of primitives
 * \param lo Lower bound of the scene
 * \param hi Upper bound of the scene
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * The LBVH is constructed using the algorithm due to Karras using 30-bit Morton codes.
 * The caller should ensure that all \a points lie within \a lo and \a hi for best performance.
 * Points lying outside this range are clamped to it during the Morton code calculation, which
 * may lead to a low quality LBVH.
 *
 * \note
 * Currently, small LBVHs (`N` <= 2) are not implemented, and an error will be raised.
 */
template<class InsertOpT>
void LBVH::build(const InsertOpT& insert, const float3 lo, const float3 hi, cudaStream_t stream)
    {
    const unsigned int N = insert.size();

    // resize memory for the tree
    allocate(N);

    // if N = 0, don't do anything and quit, since this is an empty lbvh
    if (N == 0) return;

    // single-particle just needs a small amount of data
    if (N == 1)
        {
        gpu::LBVHData tree = data();
        gpu::lbvh_one_primitive(tree, insert, stream);
        return;
        }

    // calculate morton codes
    m_tune_gen_codes->begin();
    gpu::lbvh_gen_codes(thrust::raw_pointer_cast(m_codes.data()),
                        thrust::raw_pointer_cast(m_indexes.data()),
                        insert,
                        lo,
                        hi,
                        m_N,
                        m_tune_gen_codes->getParam(),
                        stream);
     m_tune_gen_codes->end();

    // sort morton codes
        {
        uchar2 swap;

        size_t tmp_bytes = 0;
        gpu::lbvh_sort_codes(NULL,
                             tmp_bytes,
                             thrust::raw_pointer_cast(m_codes.data()),
                             thrust::raw_pointer_cast(m_sorted_codes.data()),
                             thrust::raw_pointer_cast(m_indexes.data()),
                             thrust::raw_pointer_cast(m_sorted_indexes.data()),
                             m_N,
                             stream);

        // make requested temporary allocation (1 char = 1B)
        size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
        thrust::device_vector<unsigned char> tmp(alloc_size);

        swap = neighbor::gpu::lbvh_sort_codes((void*)thrust::raw_pointer_cast(tmp.data()),
                                              tmp_bytes,
                                              thrust::raw_pointer_cast(m_codes.data()),
                                              thrust::raw_pointer_cast(m_sorted_codes.data()),
                                              thrust::raw_pointer_cast(m_indexes.data()),
                                              thrust::raw_pointer_cast(m_sorted_indexes.data()),
                                              m_N,
                                              stream);

        // sorting will synchronize the stream before returning, so this unfortunately blocks concurrent execution of builds
        if (swap.x) m_sorted_codes.swap(m_codes);
        if (swap.y) m_sorted_indexes.swap(m_indexes);
        }

    // process hierarchy and bubble aabbs
    gpu::LBVHData tree = data();

    m_tune_gen_tree->begin();
    gpu::lbvh_gen_tree(tree,
                       thrust::raw_pointer_cast(m_sorted_codes.data()),
                       m_N,
                       m_tune_gen_tree->getParam(),
                       stream);
    m_tune_gen_tree->end();

    m_tune_bubble->begin();
    gpu::lbvh_bubble_aabbs(tree,
                           insert,
                           thrust::raw_pointer_cast(m_locks.data()),
                           m_N,
                           m_tune_bubble->getParam(),
                           stream);
    m_tune_bubble->end();
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
 */
void LBVH::allocate(unsigned int N)
    {
    m_root = 0;
    m_N = N;
    m_N_internal = (m_N > 0) ? m_N - 1 : 0;
    m_N_nodes = m_N + m_N_internal;

    if (m_N_nodes > m_parent.size())
        {
        thrust::device_vector<int> parent(m_N_nodes);
        m_parent.swap(parent);
        }

    if (m_N_internal > m_left.size())
        {
        thrust::device_vector<int> left(m_N_internal);
        m_left.swap(left);

        thrust::device_vector<int> right(m_N_internal);
        m_right.swap(right);

        thrust::device_vector<unsigned int> locks(m_N_internal);
        m_locks.swap(locks);
        }

    if (m_N_nodes > m_lo.size())
        {
        thrust::device_vector<float3> lo(m_N_nodes);
        m_lo.swap(lo);

        thrust::device_vector<float3> hi(m_N_nodes);
        m_hi.swap(hi);
        }

    if (m_N > m_codes.size())
        {
        thrust::device_vector<unsigned int> codes(m_N);
        m_codes.swap(codes);

        thrust::device_vector<unsigned int> indexes(m_N);
        m_indexes.swap(indexes);

        thrust::device_vector<unsigned int> sorted_codes(m_N);
        m_sorted_codes.swap(sorted_codes);

        thrust::device_vector<unsigned int> sorted_indexes(m_N);
        m_sorted_indexes.swap(sorted_indexes);
        }
    }

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_H_
