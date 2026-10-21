// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is released under the Modified BSD License.

#include <cuda_runtime.h>
#include "neighbor/Memory.h"

#include <memory>
#include <vector>

#ifdef __NVCC__
#include "neighbor/BoundingVolumes.h"
#endif

// forward declaration
namespace neighbor
    {
    class LBVH;
    class LBVHTraverser;
    }

//! Particle insert operation
struct ParticleInsertOp
    {
    ParticleInsertOp(const float3* particles_, unsigned int N_)
        : particles(particles_), N(N_)
        {}

    #ifdef __NVCC__
    __device__ __forceinline__
    neighbor::BoundingBox get(const unsigned int idx) const
        {
        const float3 p = particles[idx];
        return neighbor::BoundingBox(p,p);
        }

    __host__ __device__ __forceinline__
    #endif
    unsigned int size() const
        {
        return N;
        }

    const float3* particles;
    const unsigned int N;
    };

//! Particle query op using spheres
struct ParticleQueryOp
    {
    ParticleQueryOp(const float4* spheres_, unsigned int N_)
        : spheres(spheres_), N(N_)
        {}

    #ifdef __NVCC__
    typedef float4 ThreadData;
    typedef neighbor::BoundingSphere Volume;

    __device__ __forceinline__ ThreadData setup(const unsigned int idx) const
        {
        return spheres[idx];
        }

    __device__ __forceinline__ Volume get(const ThreadData& q, const float3 image) const
        {
        const float3 t = make_float3(q.x + image.x, q.y + image.y, q.z + image.z);
        return neighbor::BoundingSphere(t,q.w);
        }

    __device__ __forceinline__ bool overlap(const Volume& v, const neighbor::BoundingBox& box) const
        {
        return v.overlap(box);
        }

    __device__ __forceinline__ bool refine(const ThreadData& q, const int primitive) const
        {
        return true;
        }

    __host__ __device__ __forceinline__
    #endif
    unsigned int size() const
        {
        return N;
        }

    const float4* spheres;
    const unsigned int N;
    };

//! Wrapper around the neighbor::LBVH that can be called from a C++ program
class LBVHWrapper
    {
    public:
        //! Constructor
        LBVHWrapper();

        //! Build using pointers and floats
        void build(const float3* pos, unsigned int N, const float3& lo, const float3& hi, unsigned int param);

        //! Get the underlying neighbor::LBVH object
        std::shared_ptr<neighbor::LBVH> get()
            {
            return lbvh_;
            }

        //! Get the vector holding the LBVH primitives
        const neighbor::shared_array<unsigned int>& getPrimitives() const;

        //! Set the autotuner parameters
        std::vector<unsigned int> getTunableParameters() const;

    private:
        std::shared_ptr<neighbor::LBVH> lbvh_;  //!< Underlying LBVH
    };

//! Wrapper around the neighbor::LBVHTraverser that can be called from a C++ program
class LBVHTraverserWrapper
    {
    public:
        //! Constructor
        LBVHTraverserWrapper();

        //! Traversal method using pointers and floats
        void traverse(unsigned int* hits,
                      const float4* spheres,
                      unsigned int N,
                      std::shared_ptr<neighbor::LBVH> lbvh,
                      const float3* images,
                      unsigned int Nimages,
                      unsigned int param);

        //! Get the underlying neighbor::LBVHTraverser object
        std::shared_ptr<neighbor::LBVHTraverser> get()
            {
            return trav_;
            }

        //! Set the autotuner parameters
        std::vector<unsigned int> getTunableParameters() const;

    private:
        std::shared_ptr<neighbor::LBVHTraverser> trav_; //!< Underlying traverser
    };
