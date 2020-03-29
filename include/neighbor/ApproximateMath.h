// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_APPROXIMATE_MATH_H_
#define NEIGHBOR_APPROXIMATE_MATH_H_

#include <cuda_runtime.h>

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
 * On CUDA devices, the conversion is done using a device intrinsic.
 */
DEVICE float double2float_rd(double x)
    {
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
 * On CUDA devices, the conversion is done using a device intrinsic.
 */
DEVICE float double2float_ru(double x)
    {
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
 */
DEVICE float fadd_rd(float x, float y)
    {
    return __fadd_rd(x,y);
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
 */
DEVICE float fadd_ru(float x, float y)
    {
    return __fadd_ru(x,y);
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
 */
DEVICE float fsub_rd(float x, float y)
    {
    return __fsub_rd(x,y);
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
 */
DEVICE float fsub_ru(float x, float y)
    {
    return __fsub_ru(x,y);
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
 */
DEVICE float fmul_rd(float x, float y)
    {
    return __fmul_rd(x,y);
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
 */
DEVICE float fmul_ru(float x, float y)
    {
    return __fmul_ru(x,y);
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
 */
DEVICE float fdiv_rd(float x, float y)
    {
    return __fdiv_rd(x,y);
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
 */
DEVICE float fdiv_ru(float a, float b)
    {
    return __fdiv_ru(a,b);
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
 */
DEVICE float frcp_rd(float x)
    {
    return __frcp_rd(x);
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
 */
DEVICE float frcp_ru(float x)
    {
    return __frcp_ru(x);
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
 */
DEVICE float fmaf_rd(float x, float y, float z)
    {
    return __fmaf_rd(x,y,z);
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
 */
DEVICE float fmaf_ru(float x, float y, float z)
    {
    return __fmaf_ru(x,y,z);
    }
} // end namespace approx
} // end namespace neighbor

#undef DEVICE

#endif // NEIGHBOR_APPROXIMATE_MATH_H_
