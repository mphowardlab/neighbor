// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_H_
#define NEIGHBOR_LBVH_TRAVERSER_H_

#include "LBVH.h"
#include "hoomd/ExecutionConfiguration.h"

namespace neighbor
{

//! Linear bounding volume hierarchy traverser.
/*!
 * A LBVHTraverser implements a scheme to traverse a LBVH. For example, two options
 * are using a stack-based traversal or using a stackless rope traversal scheme.
 * A LBVHTraverser will typically take the data from the LBVH and compress it into a
 * format that is efficient for traversal. During this step, the LBVHTraverser is also
 * permitted to modify the LBVH before compression, if it is useful for traversal (e.g.,
 * performing subtree collapse).
 *
 * The LBVHTraverser class supplies an abstract base for possible traversal schemes, all of
 * which must implement a ::traverse method. Currently, traversal is specialized to sphere
 * overlaps, but it will be generalized to other shapes in the future. Additionally, the
 * traversal scheme currently only *counts* overlaps of the test spheres with the tree, but
 * will later be generalized to support, e.g., generation of a list of overlaps.
 *
 * The LBVH is not aware of periodic boundary conditions of a scene, and so by default the
 * LBVHTraverser only intersects the sphere directly against the LBVH. However, an additional
 * image list can be specified for ::traverse. The image list specifies *additional* translations
 * of the particle to consider, beyond the original sphere.
 */
class LBVHTraverser
    {
    public:
        //! Simple constructor for a LBVHTraverser.
        /*!
         * \param exec_conf HOOMD-blue execution configuration.
         */
        LBVHTraverser(std::shared_ptr<const ExecutionConfiguration> exec_conf)
            : m_exec_conf(exec_conf)
        {}

        //! Traverse the LBVH.
        /*!
         * \param out Number of overlaps per sphere.
         * \param spheres Test spheres.
         * \param N Number of test spheres.
         * \param lbvh LBVH to traverse.
         * \param images Additional images of \a spheres to test.
         *
         * The format for a \a sphere is (x,y,z,R), where R is the radius of the sphere.
         */
        virtual void traverse(const GPUArray<unsigned int>& out,
                              const GPUArray<Scalar4>& spheres,
                              unsigned int N,
                              const LBVH& lbvh,
                              const GPUArray<Scalar3>& images = GPUArray<Scalar3>()) = 0;

    protected:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
    };

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_H_
