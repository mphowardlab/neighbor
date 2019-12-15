// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_TRANSLATE_OPS_H_
#define NEIGHBOR_TRANSLATE_OPS_H_

#include <cuda_runtime.h>

namespace neighbor
{

//! Self-image operator
/*!
 * The self-image operator returns only the zero vector as a float3.
 */
struct SelfOp
    {
    typedef float3 type;

    SelfOp() {}

    //! Get the image vector
    /*!
     * \param idx Image vector index.
     * \returns The \a idx -th image vector.
     *
     * No check is done to ensure that index does not run past the size of the vector.
     * This method always returns a zero vector.
     */
    __device__ __forceinline__ float3 get(const unsigned int idx) const
        {
        return make_float3(0.f,0.f,0.f);
        }

    //! Get the number of images
    /*!
     * \returns Always returns 1
     */
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return 1;
        }
    };

//! Image list operator
/*!
 * \tparam Real3 The data type of the image.
 *
 * Supply the images as an array that can be read.
 */
template<typename Real3>
struct ImageListOp
    {
    typedef Real3 type;

    ImageListOp() : images(NULL), N(0) {}

    //! Constructor
    /*!
     * \param images_ Array of images.
     * \param N_ Number of images
     */
    ImageListOp(const Real3* images_, unsigned int N_)
        : images(images_), N(N_)
        {}

    //! Get the image vector
    /*!
     * \param idx Image vector index.
     * \returns The \a idx -th image vector.
     *
     * No check is done to ensure that index does not run past the size of the vector.
     * This behavior is undefined.
     */
    __device__ __forceinline__ Real3 get(const unsigned int idx) const
        {
        return images[idx];
        }

    //! Get the number of images
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return N;
        }

    const Real3* images;
    const unsigned int N;
    };

} // end namespace neighbor

#endif // NEIGHBOR_TRANSLATE_OPS_H_
