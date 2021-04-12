// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is released under the Modified BSD License.

#ifndef NEIGHBOR_LBVH_H_
#define NEIGHBOR_LBVH_H_

#include <assert.h>
#include <hipper/hipper_runtime.h>

#include "Memory.h"
#include "Tunable.h"

#include "LBVHData.h"
#include "kernels/LBVH.cuh"

namespace neighbor
{
//! Linear bounding volume hierarchy.
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
 * primitives can be injected using a templated insert operation, which converts them to bounding boxes.
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
 * only the raw pointers to the tree data using ::data (see LBVHData).
 */
class LBVH : public Tunable<unsigned int>
    {
    public:
        //! Setup an unallocated LBVH.
        LBVH();

        //! Setup LBVH memory for building.
        template<class InsertOpT>
        void setup(const LaunchParameters& params, const InsertOpT& insert)
            {
            allocate(params, insert.size());
            }

        //! Setup LBVH memory for building.
        template<class InsertOpT>
        void setup(hipper::stream_t stream, const InsertOpT& insert)
            {
            setup(LaunchParameters(32, stream), insert);
            }

        //! Setup LBVH memory for building.
        template<class InsertOpT>
        void setup(const InsertOpT& insert)
            {
            setup(0, insert);
            }

        //! Build the LBVH in a stream with tunable parameters.
        template<class InsertOpT>
        void build(const LaunchParameters& params, const InsertOpT& insert, const float3& lo, const float3& hi);

        //! Build the LBVH in a stream.
        /*!
         * \param stream CUDA stream for kernel execution.
         * \param insert The insert operation holding the primitives.
         * \param lo Lower bound of the scene.
         * \param hi Upper bound of the scene.
         *
         * \tparam InsertOpT The kind of insert operation.
         *
         * The tunable block size defaults to 32 threads per block.
         */
        template<class InsertOpT>
        void build(hipper::stream_t stream, const InsertOpT& insert, const float3& lo, const float3& hi)
            {
            build(LaunchParameters(32,stream), insert, lo, hi);
            }

        //! Build the LBVH.
        /*!
         * \param insert The insert operation holding the primitives.
         * \param lo Lower bound of the scene.
         * \param hi Upper bound of the scene.
         *
         * \tparam InsertOpT The kind of insert operation.
         *
         * The tunable block size defaults to 32 threads per block, and the kernel executes in the default stream.
         */
        template<class InsertOpT>
        void build(const InsertOpT& insert, const float3& lo, const float3& hi)
            {
            build(0, insert, lo, hi);
            }

        //! Get the LBVH root node.
        int getRoot() const
            {
            return m_root;
            }

        //! Get the number of primitives.
        unsigned int getN() const
            {
            return m_N;
            }

        //! Get the number of internal nodes.
        unsigned int getNInternal() const
            {
            return m_N_internal;
            }

        //! Get the total number of nodes.
        unsigned int getNNodes() const
            {
            return m_N_nodes;
            }

        //! Get the array of parents of a given node.
        const shared_array<int>& getParents() const
            {
            return m_parent;
            }

        //! Get the array of left children of a given node.
        const shared_array<int>& getLeftChildren() const
            {
            return m_left;
            }

        //! Get the array of right children of a given node.
        const shared_array<int>& getRightChildren() const
            {
            return m_right;
            }

        //! Get the lower bounds of the boxes enclosing a node.
        const shared_array<float3>& getLowerBounds() const
            {
            return m_lo;
            }

        //! Get the upper bounds of the boxes enclosing a node.
        const shared_array<float3>& getUpperBounds() const
            {
            return m_hi;
            }

        //! Get the original indexes of the primitives in each leaf node.
        const shared_array<unsigned int>& getPrimitives() const
            {
            return m_indexes.current();
            }

        //! Get the pointer version of the read-only data in the tree.
        const ConstLBVHData data() const
            {
            ConstLBVHData tree;
            tree.parent = m_parent.get();
            tree.left = m_left.get();
            tree.right = m_right.get();
            tree.primitive = m_indexes.current().get();
            tree.lo = m_lo.get();
            tree.hi = m_hi.get();
            tree.root = m_root;
            return tree;
            }

    private:
        int m_root;                 //!< Root index
        unsigned int m_N;           //!< Number of primitives in the tree
        unsigned int m_N_internal;  //!< Number of internal nodes in tree
        unsigned int m_N_nodes;     //!< Number of nodes in the tree

        shared_array<int> m_parent; //!< Parent node
        shared_array<int> m_left;   //!< Left child
        shared_array<int> m_right;  //!< Right child
        shared_array<float3> m_lo;  //!< Lower bound of AABB
        shared_array<float3> m_hi;  //!< Upper bound of AABB

        buffered_array<unsigned int> m_codes;   //!< Morton codes
        buffered_array<unsigned int> m_indexes; //!< Primitive indexes
        shared_array<unsigned char> m_tmp;      //!< Temporary memory for sorting

        shared_array<unsigned int> m_locks; //!< Node locks for generating aabb hierarchy

        //! Allocate.
        void allocate(const LaunchParameters& params, unsigned int N);

        //! Get the pointer version of the data in the tree.
        const LBVHData data()
            {
            LBVHData tree;
            tree.parent = m_parent.get();
            tree.left = m_left.get();
            tree.right = m_right.get();
            tree.primitive = m_indexes.current().get();
            tree.lo = m_lo.get();
            tree.hi = m_hi.get();
            tree.root = m_root;
            return tree;
            }
    };

/*!
 * The constructor defers memory initialization to the first call to ::build.
 */
LBVH::LBVH()
    : Tunable<unsigned int>(32, 1024, 32),
      m_root(LBVHSentinel), m_N(0), m_N_internal(0), m_N_nodes(0)
    {}

/*!
 * \param params Launch parameters for kernel execution, including tunable block size and stream.
 * \param insert The insert operation holding the primitives.
 * \param lo Lower bound of the scene.
 * \param hi Upper bound of the scene.
 *
 * \tparam InsertOpT The kind of insert operation.
 *
 * The LBVH is constructed using the algorithm due to Karras using 30-bit Morton codes.
 * The caller should ensure that all \a points lie within \a lo and \a hi for best performance.
 * Points lying outside this range are clamped to it during the Morton code calculation, which
 * may lead to a low quality LBVH.
 */
template<class InsertOpT>
void LBVH::build(const LaunchParameters& params, const InsertOpT& insert, const float3& lo, const float3& hi)
    {
    // resize memory for the tree (will do nothing if setup already called)
    setup(params, insert);

    // if N = 0, don't do anything and quit, since this is an empty lbvh
    if (m_N == 0) return;

    // single-particle just needs a small amount of data
    if (m_N == 1)
        {
        LBVHData tree = data();
        gpu::lbvh_one_primitive(tree, insert, params.stream);
        return;
        }

    // check tuning parameter first
    checkParameter(params);

    // calculate morton codes
    gpu::lbvh_gen_codes(m_codes.current().get(),
                        m_indexes.current().get(),
                        insert,
                        lo,
                        hi,
                        m_N,
                        params.tunable,
                        params.stream);

    // sort morton codes
        {
        uchar2 swap;

        size_t tmp_bytes = 0;
        gpu::lbvh_sort_codes(NULL,
                             tmp_bytes,
                             m_codes.current().get(),
                             m_codes.alternate().get(),
                             m_indexes.current().get(),
                             m_indexes.alternate().get(),
                             m_N,
                             params.stream);

        // allocation already be taken care of by setup() call above, so assume here.
        assert(m_tmp.size() >= tmp_bytes);

        swap = gpu::lbvh_sort_codes((void*)m_tmp.get(),
                                    tmp_bytes,
                                    m_codes.current().get(),
                                    m_codes.alternate().get(),
                                    m_indexes.current().get(),
                                    m_indexes.alternate().get(),
                                    m_N,
                                    params.stream);

        // flip the buffer selector if the sorted codes or indexes are in the alternate array
        if (swap.x) m_codes.flip();
        if (swap.y) m_indexes.flip();
        }

    // process hierarchy and bubble aabbs
    LBVHData tree = data();

    gpu::lbvh_gen_tree(tree,
                       m_codes.current().get(),
                       m_N,
                       params.tunable,
                       params.stream);

    gpu::lbvh_bubble_aabbs(tree,
                           insert,
                           m_locks.get(),
                           m_N,
                           params.tunable,
                           params.stream);
    }

/*!
 * \param params Kernel launch parameters (only used for stream).
 * \param N Number of primitives.
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
 * Additional calls to allocate are ignored if \a N has not changed from the previous call.
 */
void LBVH::allocate(const LaunchParameters& params, unsigned int N)
    {
    // do nothing if N has not changed
    if (N == m_N) return;

    m_root = 0;
    m_N = N;
    m_N_internal = (m_N > 0) ? m_N - 1 : 0;
    m_N_nodes = m_N + m_N_internal;

    // all nodes
    if (m_N_nodes > m_parent.size())
        {
        shared_array<int> parent(m_N_nodes);
        m_parent.swap(parent);
        }

    // internal nodes
    if (m_N_internal > m_left.size())
        {
        shared_array<int> left(m_N_internal);
        m_left.swap(left);

        shared_array<int> right(m_N_internal);
        m_right.swap(right);

        shared_array<unsigned int> locks(m_N_internal);
        m_locks.swap(locks);
        }

    // node bounds
    if (m_N_nodes > m_lo.size())
        {
        shared_array<float3> lo(m_N_nodes);
        m_lo.swap(lo);

        shared_array<float3> hi(m_N_nodes);
        m_hi.swap(hi);
        }

    // sorting arrays
    if (m_N > m_codes.size())
        {
        buffered_array<unsigned int> codes(m_N);
        m_codes.swap(codes);

        buffered_array<unsigned int> indexes(m_N);
        m_indexes.swap(indexes);
        }

    // check for required size of CUB allocation
    size_t tmp_bytes = 0;
    gpu::lbvh_sort_codes(NULL,
                         tmp_bytes,
                         m_codes.current().get(),
                         m_codes.alternate().get(),
                         m_indexes.current().get(),
                         m_indexes.alternate().get(),
                         m_N,
                         params.stream);
    if (tmp_bytes == 0) tmp_bytes = 4; // make at least 4 bytes (old workaround)
    if (tmp_bytes > m_tmp.size())
        {
        shared_array<unsigned char> tmp(tmp_bytes);
        m_tmp.swap(tmp);
        }
    }

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_H_
