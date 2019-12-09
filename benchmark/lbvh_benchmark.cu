#include "lbvh_benchmark.cuh"

#include "neighbor/LBVH.h"
#include "neighbor/LBVHTraverser.h"
#include "neighbor/BoundingVolumes.h"
#include "neighbor/OutputOps.h"
#include "neighbor/TranslateOps.h"

__host__ float double2float_rd(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) > x)
        {
        xf = std::nextafterf(xf, -std::numeric_limits<float>::infinity());
        }
    return xf;
    }

__host__ float double2float_ru(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) < x)
        {
        xf = std::nextafterf(xf, std::numeric_limits<float>::infinity());
        }
    return xf;
    }

LBVHWrapper::LBVHWrapper()
    {
    lbvh_ = std::make_shared<neighbor::LBVH>();
    }

void LBVHWrapper::build(const Scalar4* pos, unsigned int N, const Scalar3& lo, const Scalar3& hi)
    {
    float3 lof = make_float3(double2float_rd(lo.x), double2float_rd(lo.y), double2float_rd(lo.z));
    float3 hif = make_float3(double2float_ru(hi.x), double2float_ru(hi.y), double2float_ru(hi.z));

    lbvh_->build(ParticleInsertOp(pos, N), lof, hif);
    }

const thrust::device_vector<unsigned int>& LBVHWrapper::getPrimitives() const
    {
    return lbvh_->getPrimitives();
    }

void LBVHWrapper::setAutotunerParams(bool enable, unsigned int period)
    {
    lbvh_->setAutotunerParams(enable, period);
    }

LBVHTraverserWrapper::LBVHTraverserWrapper()
    {
    trav_ = std::make_shared<neighbor::LBVHTraverser>();
    };

void LBVHTraverserWrapper::traverse(unsigned int* hits,
                                    const Scalar4* spheres,
                                    unsigned int N,
                                    std::shared_ptr<neighbor::LBVH> lbvh,
                                    const Scalar3* images,
                                    unsigned int Nimages)
    {
    neighbor::CountNeighborsOp count(hits);
    ParticleQueryOp query(spheres, N);
    neighbor::ImageListOp<Scalar3> translate(images, Nimages);
    trav_->traverse(count, query, *lbvh, translate);
    };

void LBVHTraverserWrapper::setAutotunerParams(bool enable, unsigned int period)
    {
    trav_->setAutotunerParams(enable, period);
    }
