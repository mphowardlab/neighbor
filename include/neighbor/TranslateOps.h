// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_TRANSLATE_OPS_H_
#define NEIGHBOR_TRANSLATE_OPS_H_

namespace neighbor
{

struct SelfOp
    {
    typedef float3 type;

    SelfOp() {}

    __device__ __forceinline__ float3 get(const unsigned int idx) const
        {
        return make_float3(0.f,0.f,0.f);
        }

    //! Get the number of images
    __host__ __device__ __forceinline__ unsigned int size() const
        {
        return 1;
        }
    };

template<typename Real3>
struct ImageListOp
    {
    typedef Real3 type;

    ImageListOp() : images(NULL), N(0) {}

    ImageListOp(const Real3* images_, unsigned int N_)
        : images(images_), N(N_)
        {}

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
