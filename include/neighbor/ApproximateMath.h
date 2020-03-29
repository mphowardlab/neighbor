// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_APPROXIMATE_MATH_H_
#define NEIGHBOR_APPROXIMATE_MATH_H_

#include "hipper_runtime.h"
#ifndef HIPPER_PLATFORM_NVCC
#include <cfloat>
#endif

#define DEVICE __device__ __forceinline__

namespace neighbor
{

//! Approximate math operations.
/*!
 * This namespace defines inline functions that wrap around "approximate" math operations.
 * Approximate operations are intended to mimic single-precision floating-point math having
 * different IEEE-754 rounding conventions. These operators are implemented as CUDA intrinsics,
 * requiring emulation or other names on the host or other architectures.
 *
 * In neighbor, we never actually need *exactly* the IEEE-754 result. Instead, we want an fp32
 * result that is at least below or above this result (and as close as possible) to produce an
 * efficient, watertight calculation. It is this level of approximation that is guaranteed by
 * these functions.
 */
namespace approx
{
//! Convert a double to a float, rounding down.
/*!
 * \param x The double to convert.
 * \return A float that is guaranteed to be below the value of \a x.
 *
 * This function guarantees that the returned value will be less than the value of \a x.
 * It tries to return the closest representable float to \a x that satisfies this condition
 * without incurring significant overhead.
 *
 * The conversion is done using a device intrinsic.
 */
DEVICE float double2float_rd(double x)
    {
    // CUDA and HIP both support this intrinsic.
    return __double2float_rd(x);
    }

//! Convert a double to a float, rounding up.
/*!
 * \param x The double to convert.
 * \return A float that is guaranteed to be above the value of \a x.
 *
 * This function guarantees that the returned value will be less than the value of \a x.
 * It tries to return the closest representable float to \a x that satisfies this condition
 * without incurring significant overhead.
 *
 * The conversion is done using a device intrinsic.
 */
DEVICE float double2float_ru(double x)
    {
    // CUDA and HIP both support this intrinsic.
    return __double2float_ru(x);
    }

//! Add two floats, rounding the result down.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The sum \f$x+y\f$.
 *
 * This function guarantees that the resulting sum will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float fadd_rd(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fadd_rd(x,y);
    #else
    return nextafterf(x+y, -FLT_MAX);
    #endif
    }

//! Add two floats, rounding the result up.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The sum \f$x+y\f$.
 *
 * This function guarantees that the resulting sum will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float fadd_ru(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fadd_ru(x,y);
    #else
    return nextafterf(x+y, FLT_MAX);
    #endif
    }

//! Subtract two floats, rounding the result down.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The difference \f$x-y\f$.
 *
 * This function guarantees that the resulting subtraction will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float fsub_rd(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fsub_rd(x,y);
    #else
    return nextafterf(x-y, -FLT_MAX);
    #endif
    }

//! Subtract two floats, rounding the result up.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The difference \f$x-y\f$.
 *
 * This function guarantees that the resulting subtraction will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float fsub_ru(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fsub_ru(x,y);
    #else
    return nextafterf(x-y, FLT_MAX);
    #endif
    }

//! Multiply two floats, rounding the result down.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The product \f$xy\f$.
 *
 * This function guarantees that the resulting product will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float fmul_rd(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fmul_rd(x,y);
    #else
    return nextafterf(x*y, -FLT_MAX);
    #endif
    }

//! Multiply two floats, rounding the result up.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The product \f$xy\f$.
 *
 * This function guarantees that the resulting product will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float fmul_ru(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fmul_ru(x,y);
    #else
    return nextafterf(x*y, FLT_MAX);
    #endif
    }

//! Divide two floats, rounding the result down.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The product \f$x/y\f$.
 *
 * This function guarantees that the resulting quotient will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float fdiv_rd(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fdiv_rd(x,y);
    #else
    return nextafterf(x/y, -FLT_MAX);
    #endif
    }

//! Divide two floats, rounding the result up.
/*!
 * \param x First value.
 * \param y Second value.
 * \returns The product \f$x/y\f$.
 *
 * This function guarantees that the resulting quotient will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float fdiv_ru(float x, float y)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fdiv_ru(x,y);
    #else
    return nextafterf(x/y, FLT_MAX);
    #endif
    }

//! Take the reciprocal of a float, rounding the result down.
/*!
 * \param x Value.
 * \returns The reciprocal \f$1/x\f$.
 *
 * This function guarantees that the resulting reciprocal will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float frcp_rd(float x)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __frcp_rd(x);
    #else
    return nextafterf(1.0f/x, -FLT_MAX);
    #endif
    }

//! Take the reciprocal of a float, rounding the result up.
/*!
 * \param x Value.
 * \returns The reciprocal \f$1/x\f$.
 *
 * This function guarantees that the resulting reciprocal will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float frcp_ru(float x)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __frcp_ru(x);
    #else
    return nextafterf(1.0f/x, FLT_MAX);
    #endif
    }

//! Fused multiply-add, rounding the result down.
/*!
 * \param x First value.
 * \param y Second value.
 * \param z Third value.
 * \returns The single operation \f$xy+z\f$.
 *
 * This function guarantees that the resulting value will be less than or equal to
 * the IEEE-754 result in round-down mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward -FLT_MAX, which may be the
 * same value or the next smaller value.
 */
DEVICE float fmaf_rd(float x, float y, float z)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fmaf_rd(x,y,z);
    #else
    return nextafterf(fmaf(x,y,z), -FLT_MAX);
    #endif
    }

//! Fused multiply-add, rounding the result up.
/*!
 * \param x First value.
 * \param y Second value.
 * \param z Third value.
 * \returns The single operation \f$xy+z\f$.
 *
 * This function guarantees that the resulting value will be greater than or equal to
 * the IEEE-754 result in round-up mode.
 *
 * On CUDA devices, this is exactly satisfied using a device intrinsic.
 * Otherwise, this returns the nextafter float toward FLT_MAX, which may be the
 * same value or the next greater value.
 */
DEVICE float fmaf_ru(float x, float y, float z)
    {
    #ifdef HIPPER_PLATFORM_NVCC
    return __fmaf_ru(x,y,z);
    #else
    return nextafterf(fmaf(x,y,z), FLT_MAX);
    #endif
    }
} // end namespace approx
} // end namespace neighbor

#undef DEVICE
#undef HIPPER_PLATFORM_NVCC

#endif // NEIGHBOR_APPROXIMATE_MATH_H_
