// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

#ifndef NEIGHBOR_MIXED_PRECISION_H_
#define NEIGHBOR_MIXED_PRECISION_H_

#include "hoomd/HOOMDMath.h"

#ifndef NEIGHBOR_DISABLE_MIXED
typedef float NeighborReal;
typedef float3 NeighborReal3;

#define make_neighbor_real3(x,y,z) make_float3(x,y,z)

#define DOUBLE2REAL_RD(x) (__double2float_rd(x))
#define DOUBLE2REAL_RU(x) (__double2float_ru(x))
#define REAL_ADD_RD(x,y) (__fadd_rd(x,y))
#define REAL_ADD_RU(x,y) (__fadd_ru(x,y))
#define REAL_SUB_RD(x,y) (__fsub_rd(x,y))
#define REAL_SUB_RU(x,y) (__fsub_ru(x,y))
#define REAL_MUL_RD(x,y) (__fmul_rd(x,y))
#define REAL_MUL_RU(x,y) (__fmul_ru(x,y))
#define REAL_DIV_RD(x,y) (__fdiv_rd(x,y))
#define REAL_RCP_RD(x) (__frcp_rd(x))
#define REAL_MIN(x,y) (fminf(x,y))
#define REAL_MAX(x,y) (fmaxf(xy))
#define REAL_MAF_RD(x,y,z) (__fmaf_rd(x,y,z))
#define REAL_SQRT_RU(x) (__fsqrt_ru(x))
#define REAL_FLOOR(x) (floorf(x))
#else

// No need to have higher precision than the input data types
typedef Scalar NeigbhorReal;
typedef Scalar3 NeighborReal3;

#define make_neighbor_real3(x,y,z) make_scalar3(x,y,z)

#define DOUBLE2REAL_RD(x) (x)
#define DOUBLE2REAL_RU(x) (x)
#define REAL_ADD_RD(x,y) (x+y)
#define REAL_ADD_RU(x,y) (x+y)
#define REAL_SUB_RD(x,y) (x-y)
#define REAL_SUB_RU(x,y) (x-y)
#define REAL_MUL_RD(x,y) (x*y)
#define REAL_MUL_RU(x,y) (x*y)
#define REAL_DIV_RD(x,y) (x*y)
#define REAL_RCP_RD(x) (Scalar(1.0)/x)

#ifdef SINGLE_PRECISION
#define REAL_MIN(x) (fminf(x,y)) 
#define REAL_MAX(x) (fmaxf(x,y))
#define REAL_FLOOR(x) (floorf(x))
#else
#define REAL_MIN(x) (fmin(x,y)) 
#define REAL_MAX(x) (fmax(x,y))
#define REAL_FLOOR(x) (floor(x))
#endif

#define REAL_MAF_RD(x,y,z) (x*y+z)
#define REAL_SQRT_RU(x) (fast::sqrt(x))
#endif

#endif // NEIGHBOR_MIXED_PRECISION_H_

