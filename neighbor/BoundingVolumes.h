// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_BOUNDING_VOLUMES_H_
#define NEIGHBOR_BOUNDING_VOLUMES_H_

#include "MixedPrecision.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace neighbor
{

//! Axis-aligned bounding box
/*!
 * A bounding box is defined by a lower and upper bound that should fully
 * enclose objects inside it. Internally, the bounds are stored using
 * single-precision floating-point values. If the bounds are given in double precision,
 * they are appropriately rounded down or up, respectively, to fully enclose the
 * given bounds.
 *
 * The BoundingBox also serves as an example of a general bounding volume. Every bounding
 * volume must implement constructors for both single and double precision specifiers.
 * They must also implement an overlap method with as many other bounding volumes as is
 * practical or required. At minimum, they must implement an overlap method with a
 * BoundingBox.
 */
struct BoundingBox
    {
    //! Default constructor
    BoundingBox() {}

    //! Single-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     */
    HOSTDEVICE BoundingBox(const NeighborReal3& lo_, const NeighborReal3& hi_)
        : lo(lo_), hi(hi_)
        {}

    #ifdef __HIPCC__
    //! Double-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     *
     * \a lo_ is rounded down and \a hi_ is rounded up to the nearest fp32 representable value.
     *
     * \todo This needs a __host__ implementation that does not rely on CUDA intrinsics.
     */
    DEVICE BoundingBox(const double3& lo_, const double3& hi_)
        {
        lo = make_neighbor_real3(DOUBLE2REAL_RD(lo_.x), DOUBLE2REAL_RD(lo_.y), DOUBLE2REAL_RD(lo_.z));
        hi = make_neighbor_real3(DOUBLE2REAL_RU(hi_.x), DOUBLE2REAL_RU(hi_.y), DOUBLE2REAL_RU(hi_.z));
        }
    #endif

    DEVICE NeighborReal3 getCenter() const
        {
        NeighborReal3 c;
        c.x = 0.5f*(lo.x+hi.x);
        c.y = 0.5f*(lo.y+hi.y);
        c.z = 0.5f*(lo.z+hi.z);

        return c;
        }

    //! Test for overlap between two bounding boxes.
    /*!
     * \param box Bounding box.
     *
     * \returns True if this box overlaps \a box.
     *
     * The overlap test is performed using cheap comparison operators.
     * The two overlap if none of the dimensions of the box overlap.
     */
    HOSTDEVICE bool overlap(const BoundingBox& box) const
        {
        return !(hi.x < box.lo.x || lo.x > box.hi.x ||
                 hi.y < box.lo.y || lo.y > box.hi.y ||
                 hi.z < box.lo.z || lo.z > box.hi.z);
        }

    HOSTDEVICE BoundingBox asBox() const
        {
        return *this;
        }

    NeighborReal3 lo;  //!< Lower bound of box
    NeighborReal3 hi;  //!< Upper bound of box
    };

//! Bounding sphere
/*!
 * Implements a spherical volume with a given origin and radius that fully encloses
 * its objects. The sphere data is stored internally using single-precision values,
 * and its radius is padded to account for uncertainty due to rounding. Note that as
 * a result, the origin of the sphere may not be identical to its specified value if
 * double-precision was used.
 */
struct BoundingSphere
    {
    BoundingSphere() {}

    #ifdef __HIPCC__
    //! Single-precision constructor.
    /*!
     * \param o Center of sphere.
     * \param rsq Squared radius of sphere.
     *
     * \a r is rounded up to ensure it fully encloses all data.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE BoundingSphere(const NeighborReal3& o, const NeighborReal r)
        {
        origin = o;
        Rsq = REAL_MUL_RU(r,r);
        }

    //! Double-precision constructor.
    /*!
     * \param o Center of sphere.
     * \param rsq Squared radius of sphere.
     *
     * \a o is rounded down and \a r is padded to ensure the sphere
     * encloses all data.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE BoundingSphere(const double3& o, const double r)
        {
        const NeighborReal3 lo = make_neighbor_real3(DOUBLE2REAL_RD(o.x),
                                      DOUBLE2REAL_RD(o.y),
                                      DOUBLE2REAL_RD(o.z));
        const NeighborReal3 hi = make_neighbor_real3(DOUBLE2REAL_RU(o.x),
                                      DOUBLE2REAL_RU(o.y),
                                      DOUBLE2REAL_RU(o.z));
        const NeighborReal delta = REAL_MAX(REAL_MAX(REAL_SUB_RU(hi.x,lo.x),REAL_SUB_RU(hi.y,lo.y)),REAL_SUB_RU(hi.z,lo.z));
        const NeighborReal R = REAL_ADD_RU(DOUBLE2REAL_RU(r),delta);
        origin = make_neighbor_real3(lo.x, lo.y, lo.z);
        Rsq = REAL_MUL_RU(R,R);
        }

    //! Test for overlap between a sphere and a BoundingBox.
    /*!
     * \param box Bounding box.
     *
     * \returns True if the sphere overlaps \a box.
     *
     * The intersection test is performed by finding the closest point to \a o
     * that is inside the box using a sequence of min and max ops. The distance
     * to this point from \a o is then computed in round down mode. If the squared
     * distance between the point and \a o is less than \a Rsq, then the two
     * objects intersect.
     *
     * \todo This needs a __host__ implementation.
     */
    DEVICE bool overlap(const BoundingBox& box) const
        {
        const NeighborReal3 dr = make_neighbor_real3(REAL_SUB_RD(REAL_MIN(REAL_MAX(origin.x, box.lo.x), box.hi.x), origin.x),
                                      REAL_SUB_RD(REAL_MIN(REAL_MAX(origin.y, box.lo.y), box.hi.y), origin.y),
                                      REAL_SUB_RD(REAL_MIN(REAL_MAX(origin.z, box.lo.z), box.hi.z), origin.z));
        const NeighborReal dr2 = REAL_MAF_RD(dr.x, dr.x, REAL_MAF_RD(dr.y, dr.y, REAL_MUL_RD(dr.z,dr.z)));

        return (dr2 <= Rsq);
        }

    DEVICE BoundingBox asBox() const
        {
        const NeighborReal R = REAL_SQRT_RU(Rsq);
        const NeighborReal3 lo = make_neighbor_real3(REAL_SUB_RD(origin.x,R),REAL_SUB_RD(origin.y,R),REAL_SUB_RD(origin.z,R));
        const NeighborReal3 hi = make_neighbor_real3(REAL_ADD_RU(origin.x,R),REAL_ADD_RU(origin.y,R),REAL_ADD_RU(origin.z,R));
        return BoundingBox(lo,hi);
        }
    #endif

    NeighborReal3 origin;  //!< Center of the sphere
    NeighborReal Rsq;      //!< Squared radius of the sphere
    };

} // end namespace neighbor

#undef DEVICE
#undef HOSTDEVICE

#endif // NEIGHBOR_BOUNDING_VOLUMES_H_
