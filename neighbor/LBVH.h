// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_H_
#define NEIGHBOR_LBVH_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Autotuner.h"

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
        LBVH(std::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Destroy an LBVH
        ~LBVH();

        //! Build the LBVH
        void build(const GPUArray<Scalar4>& points, unsigned int N, const Scalar3 lo, const Scalar3 hi);

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
        const GPUArray<int>& getParents() const
            {
            return m_parent;
            }

        //! Get the array of left children of a given node
        const GPUArray<int>& getLeftChildren() const
            {
            return m_left;
            }

        //! Get the array of right children of a given node
        const GPUArray<int>& getRightChildren() const
            {
            return m_right;
            }

        //! Get the lower bounds of the boxes enclosing a node
        const GPUArray<float3>& getLowerBounds() const
            {
            return m_lo;
            }

        //! Get the upper bounds of the boxes enclosing a node
        const GPUArray<float3>& getUpperBounds() const
            {
            return m_hi;
            }

        //! Get the original indexes of the primitives in each leaf node
        const GPUArray<unsigned int>& getPrimitives() const
            {
            return m_sorted_indexes;
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
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

        int m_root;                 //!< Root index
        unsigned int m_N;           //!< Number of primitives in the tree
        unsigned int m_N_internal;  //!< Number of internal nodes in tree
        unsigned int m_N_nodes;     //!< Number of nodes in the tree

        GPUArray<int> m_parent; //!< Parent node
        GPUArray<int> m_left;   //!< Left child
        GPUArray<int> m_right;  //!< Right child
        GPUArray<float3> m_lo;  //!< Lower bound of AABB
        GPUArray<float3> m_hi;  //!< Upper bound of AABB

        GPUArray<unsigned int> m_codes;             //!< Morton codes
        GPUArray<unsigned int> m_indexes;           //!< Primitive indexes
        GPUArray<unsigned int> m_sorted_codes;      //!< Sorted morton codes
        GPUArray<unsigned int> m_sorted_indexes;    //!< Sorted primitive indexes

        GPUArray<unsigned int> m_locks; //!< Node locks for generating aabb hierarchy

        std::unique_ptr<Autotuner> m_tune_gen_codes;    //!< Autotuner for generating Morton codes kernel
        std::unique_ptr<Autotuner> m_tune_gen_tree;     //!< Autotuner for generating tree hierarchy kernel
        std::unique_ptr<Autotuner> m_tune_bubble;       //!< Autotuner for AABB bubble kernel

        //! Allocate
        void allocate(unsigned int N);
    };

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_H_
