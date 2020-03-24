// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_INSERT_OPS_H_
#define NEIGHBOR_INSERT_OPS_H_

#include "hipper_runtime.h"

#include "BoundingVolumes.h"

namespace neighbor
{
//! Reference implementation of an (almost trivial) tree insertion operation for points
/*!
 * The get() method returns a BoundingBox object, which will be used to instantiate a leaf node.
 */
struct PointInsertOp
    {
    //! Constructor
    /*!
     * \param points_ Points array (x,y,z)
     * \param N_ The number of points
     */
    PointInsertOp(const float3* points_, unsigned int N_)
        : points(points_), N(N_)
        {}

    //! Get the bounding volume for a given primitive
    /*!
     * \param idx the index of the primitive
     *
     * \returns The enclosing BoundingBox
     */
    __device__ __forceinline__ BoundingBox get(const unsigned int idx) const
        {
        const float3 p = points[idx];

        // construct the bounding box for a point
        return BoundingBox(p,p);
        }

    //! Get the number of leaf node bounding volumes
    /*!
     * \returns The initial number of leaf nodes
     */
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return N;
        }

    const float3* points;
    const unsigned int N;
    };

//! An insertion operation for spheres of constant radius
struct SphereInsertOp
    {
    //! Constructor
    /*!
     * \param points_ Sphere centers (x,y,z)
     * \param r_ Constant sphere radius
     * \param N_ The number of points
     */
    SphereInsertOp(const float3 *points_, const float r_, unsigned int N_)
        : points(points_), r(r_), N(N_)
        {}

    //! Get the bounding volume for a given primitive
    /*!
     * \param idx the index of the primitive
     *
     * \returns The enclosing BoundingBox
     */
    __device__ __forceinline__ BoundingBox get(unsigned int idx) const
        {
        const float3 point = points[idx];
        const float3 lo = make_float3(point.x-r, point.y-r, point.z-r);
        const float3 hi = make_float3(point.x+r, point.y+r, point.z+r);

        return BoundingBox(lo,hi);
        }

    //! Get the number of leaf node bounding volumes
    /*!
     * \returns The initial number of leaf nodes
     */
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return N;
        }

    const float3* points;  //!< Sphere centers
    const float r;         //!< Constant sphere radius
    const unsigned int N;  //!< Number of spheres
    };

} // end namespace neighbor

#endif // NEIGHBOR_INSERT_OPS_H_
