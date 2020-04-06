// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_BOUNDING_VOLUMES_H_
#define NEIGHBOR_BOUNDING_VOLUMES_H_

#include "hipper_runtime.h"
#include "ApproximateMath.h"

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
    /*!
     * This constructor may not assign anything, as it causes issues inside kernels.
     */
    __device__ BoundingBox() {}

    //! Single-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     */
    __device__ BoundingBox(const float3& lo_, const float3& hi_)
        : lo(lo_), hi(hi_)
        {}

    //! Double-precision constructor
    /*!
     * \param lo_ Lower bound of box.
     * \param hi_ Upper bound of box.
     *
     * \a lo_ is rounded down and \a hi_ is rounded up to the nearest fp32 representable value.
     *
     * \todo This needs a __host__ implementation that does not rely on CUDA intrinsics.
     */
    __device__ BoundingBox(const double3& lo_, const double3& hi_)
        {
        lo = make_float3(approx::double2float_rd(lo_.x), approx::double2float_rd(lo_.y), approx::double2float_rd(lo_.z));
        hi = make_float3(approx::double2float_ru(hi_.x), approx::double2float_ru(hi_.y), approx::double2float_ru(hi_.z));
        }

    //! Get the center of the box
    /*!
     * \returns The center of the box, which is the arithmetic mean of the bounds.
     */
    __device__ __forceinline__ float3 getCenter() const
        {
        float3 c;
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
    __device__ __forceinline__ bool overlap(const BoundingBox& box) const
        {
        return !(hi.x < box.lo.x || lo.x > box.hi.x ||
                 hi.y < box.lo.y || lo.y > box.hi.y ||
                 hi.z < box.lo.z || lo.z > box.hi.z);
        }

    float3 lo;  //!< Lower bound of box
    float3 hi;  //!< Upper bound of box
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
    //! Default constructor
    /*!
     * This constructor may not assign anything, as it causes issues inside kernels.
     */
    __device__ BoundingSphere() {}

    //! Single-precision constructor.
    /*!
     * \param o Center of sphere.
     * \param rsq Squared radius of sphere.
     *
     * \a r is rounded up to ensure it fully encloses all data.
     *
     * \todo This needs a __host__ implementation.
     */
    __device__ BoundingSphere(const float3& o, const float r)
        {
        origin = o;
        Rsq = approx::fmul_ru(r,r);
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
    __device__ BoundingSphere(const double3& o, const double r)
        {
        const float3 lo = make_float3(approx::double2float_rd(o.x),
                                      approx::double2float_rd(o.y),
                                      approx::double2float_rd(o.z));
        const float3 hi = make_float3(approx::double2float_ru(o.x),
                                      approx::double2float_ru(o.y),
                                      approx::double2float_ru(o.z));
        const float delta = fmaxf(fmaxf(approx::fsub_ru(hi.x,lo.x),approx::fsub_ru(hi.y,lo.y)),approx::fsub_ru(hi.z,lo.z));
        const float R = approx::fadd_ru(approx::double2float_ru(r),delta);
        origin = make_float3(lo.x, lo.y, lo.z);
        Rsq = approx::fmul_ru(R,R);
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
    __device__ __forceinline__ bool overlap(const BoundingBox& box) const
        {
        const float3 dr = make_float3(approx::fsub_rd(fminf(fmaxf(origin.x, box.lo.x), box.hi.x), origin.x),
                                      approx::fsub_rd(fminf(fmaxf(origin.y, box.lo.y), box.hi.y), origin.y),
                                      approx::fsub_rd(fminf(fmaxf(origin.z, box.lo.z), box.hi.z), origin.z));
        const float dr2 = approx::fmaf_rd(dr.x, dr.x, approx::fmaf_rd(dr.y, dr.y, approx::fmul_rd(dr.z,dr.z)));

        return (dr2 <= Rsq);
        }

    float3 origin;  //!< Center of the sphere
    float Rsq;      //!< Squared radius of the sphere
    };

} // end namespace neighbor

#endif // NEIGHBOR_BOUNDING_VOLUMES_H_
