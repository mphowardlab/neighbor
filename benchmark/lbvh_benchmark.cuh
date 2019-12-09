
#include <cuda_runtime.h>
#include <memory>
#include <thrust/device_vector.h>

#include "hoomd/HOOMDMath.h"
#ifdef NVCC
#include "neighbor/BoundingVolumes.h"
#endif

// forward declaration
namespace neighbor
    {
    class LBVH;
    class LBVHTraverser;
    }

struct ParticleInsertOp
    {
    ParticleInsertOp(const Scalar4* particles_, unsigned int N_)
        : particles(particles_), N(N_)
        {}

    #ifdef NVCC
    __device__ __forceinline__
    neighbor::BoundingBox get(const unsigned int idx) const
        {
        const Scalar4 particle = particles[idx];
        const Scalar3 p = make_scalar3(particle.x, particle.y, particle.z);
        return neighbor::BoundingBox(p,p);
        }
    #endif

    #ifdef NVCC
    __host__ __device__ __forceinline__
    #endif
    unsigned int size() const
        {
        return N;
        }

    const Scalar4* particles;
    unsigned int N;
    };

struct ParticleQueryOp
    {
    ParticleQueryOp(const Scalar4* spheres_, unsigned int N_)
        : spheres(spheres_), N(N_)
        {}

    #ifdef NVCC
    typedef Scalar4 ThreadData;
    typedef neighbor::BoundingSphere Volume;

    __device__ __forceinline__ ThreadData setup(const unsigned int idx) const
        {
        return spheres[idx];
        }

    __device__ __forceinline__ Volume get(const ThreadData& q, const Scalar3 image) const
        {
        const Scalar3 t = make_scalar3(q.x + image.x, q.y + image.y, q.z + image.z);
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

    const Scalar4* spheres;
    unsigned int N;
    };

class LBVHWrapper
    {
    public:
        LBVHWrapper();

        void build(const Scalar4* pos, unsigned int N, const Scalar3& lo, const Scalar3& hi);

        std::shared_ptr<neighbor::LBVH> get()
            {
            return lbvh_;
            }

        const thrust::device_vector<unsigned int>& getPrimitives() const;

        void setAutotunerParams(bool enable, unsigned int period);

    private:
        std::shared_ptr<neighbor::LBVH> lbvh_;
    };

class LBVHTraverserWrapper
    {
    public:
        LBVHTraverserWrapper();

        void traverse(unsigned int* hits,
                      const Scalar4* spheres,
                      unsigned int N,
                      std::shared_ptr<neighbor::LBVH> lbvh,
                      const Scalar3* images,
                      unsigned int Nimages);

        std::shared_ptr<neighbor::LBVHTraverser> get()
            {
            return trav_;
            }

        void setAutotunerParams(bool enable, unsigned int period);

    private:
        std::shared_ptr<neighbor::LBVHTraverser> trav_;
    };
