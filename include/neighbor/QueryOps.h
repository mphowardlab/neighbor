// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_QUERY_OPS_H_
#define NEIGHBOR_QUERY_OPS_H_

#include <hipper/hipper_runtime.h>

#include "BoundingVolumes.h"

namespace neighbor
{

//! Reference implementation of a query operation based on spherical volumes.
/*!
 * A query operation defines a procedure for traversing an LBVH. It must include a few key methods:
 *  1. setup(): Initializes thread-local data for the query.
 *  2. get(): Constructs the bounding volume for a given image.
 *  3. overlap(): Tests if the bounding volume overlaps a bounding box.
 *  4. refine(): Takes a primitive that is overlapped by the bounding volume, and refines the intersection.
 *  5. size(): Return the number of query volumes to test.
 *
 * Each query operation additionally should specify (by typedef, etc.) a \a ThreadData type for its internal
 * data and a \a Volume for the type of query volume used. It is helpful to use a built-in BoundingVolume, which
 * already have overlap tests defined.
 *
 * For accuracy, this reference implementation takes a point defining a spherical volume with radius stored
 * internally as (x,y,z,R). The precision is Scalar, and for accuracy, translations are also performed in Scalar
 * precision. The bounding volume is dropped into appropriate (lower) precision by the BoundingSphere.
 */
struct SphereQueryOp
    {
    //! Constructor
    /*!
     * \param spheres_ Sphere data storing (x,y,z,R) for each query volume.
     * \param N_ The number of spheres.
     */
    SphereQueryOp(float4 *spheres_, unsigned int N_)
        : spheres(spheres_), N(N_)
        {}

    typedef float4 ThreadData;
    typedef BoundingSphere Volume;

    //! Setup the thread data.
    /*!
     * \param idx The thread index for the query.
     *
     * The reference position for the sphere is simply loaded into thread-local memory.
     */
    __device__ __forceinline__ ThreadData setup(const unsigned int idx) const
        {
        return spheres[idx];
        }

    //! Get the bounding volume for a given translation image.
    /*!
     * \param q Thread data.
     * \param image Translation vector for volume from reference position.
     *
     * \returns The enclosing BoundingSphere at \a image.
     */
    __device__ __forceinline__ Volume get(const ThreadData& q, const float3& image) const
        {
        const float3 t = make_float3(q.x + image.x, q.y + image.y, q.z + image.z);
        return BoundingSphere(t,q.w);
        }

    //! Test for overlap between bounding volume and box.
    /*!
     * \param v Bounding volume being queried.
     * \param box Bounding box from BVH.
     *
     * \returns True if \a v and \a box overlap.
     *
     * A custom overlap protocol could be defined, but here we just use the one
     * already implemented. This overlap test should be *fast*, since it will be
     * applied against all internal nodes of the BVH.
     */
    __device__ __forceinline__ bool overlap(const Volume& v, const BoundingBox& box) const
        {
        return v.overlap(box);
        }

    //! Refine the overlap with a primitive.
    /*!
     * \param q Thread data.
     * \param primitive Overlapped primitive.
     *
     * \returns True if the primitive truly overlaps.
     *
     * Some query operations need to refine the overlap test with a given primitive,
     * sometimes called "narrow phase" detection. One example would be to perform
     * an exact distance check between two points to see if they lie within a certain
     * distance of each other.
     *
     * This method may be called deep within the traversal kernel, so avoid doing too
     * many calculations, loads, etc. if possible.
     *
     * In this reference implementation, we do not need any additional refinement, and
     * so we simply return true for all overlapped primitives.
     */
    __device__ __forceinline__ bool refine(const ThreadData& q, const int primitive) const
        {
        return true;
        }

    //! Get the number of query volumes.
    /*!
     * \returns The number of query volumes.
     */
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return N;
        }

    const float4* spheres;
    const unsigned int N;
    };

} // end namespace neighbor

#endif // NEIGHBOR_QUERY_OPS_H_
