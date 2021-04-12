// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is released under the Modified BSD License.

#include "lbvh_benchmark.cuh"

#include "neighbor/neighbor.h"

//! Host function approximating double2float round-down
static float double2float_rd(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) > x)
        {
        xf = std::nextafterf(xf, -std::numeric_limits<float>::infinity());
        }
    return xf;
    }

//! Host function approximating double2float round-up
static float double2float_ru(double x)
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

void LBVHWrapper::build(const Scalar4* pos, unsigned int N, const Scalar3& lo, const Scalar3& hi, unsigned int param)
    {
    float3 lof = make_float3(double2float_rd(lo.x), double2float_rd(lo.y), double2float_rd(lo.z));
    float3 hif = make_float3(double2float_ru(hi.x), double2float_ru(hi.y), double2float_ru(hi.z));

    lbvh_->build(neighbor::LBVH::LaunchParameters(param), ParticleInsertOp(pos, N), lof, hif);
    }

const neighbor::shared_array<unsigned int>& LBVHWrapper::getPrimitives() const
    {
    return lbvh_->getPrimitives();
    }

std::vector<unsigned int> LBVHWrapper::getTunableParameters() const
    {
    return lbvh_->getTunableParameters();
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
                                    unsigned int Nimages,
                                    unsigned int param)
    {
    neighbor::CountNeighborsOp count(hits);
    ParticleQueryOp query(spheres, N);
    neighbor::ImageListOp<Scalar3> translate(images, Nimages);
    trav_->traverse(neighbor::LBVHTraverser::LaunchParameters(param), *lbvh, query, count, translate);
    };

std::vector<unsigned int> LBVHTraverserWrapper::getTunableParameters() const
    {
    return trav_->getTunableParameters();
    }
