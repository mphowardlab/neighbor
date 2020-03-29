// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "neighbor/hipper_runtime.h"

#include "neighbor/neighbor.h"

#include "upp11_config.h"
UP_MAIN();

//! Kernel to execute math tests (single thread)
__global__ void approx_math_kernel(float* result)
    {
    using namespace neighbor;

    // one thread only
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx != 0)
        return;

    // casting
        {
        const double a = 1.000000000001;
        result[0] = approx::double2float_rd(a); // = 1.0f
        result[1] = approx::double2float_ru(a); // > 1.0f
        result[2] = approx::double2float_rd(-a); // < -1.0f
        result[3] = approx::double2float_ru(-a); // = -1.0f
        }
    // addition
        {
        const float a = 1.0f;
        const float b = 1.e-12f;
        result[4] = approx::fadd_rd(a,b); // = 1.0f
        result[5] = approx::fadd_ru(a,b); // > 1.0f
        result[6] = approx::fadd_rd(-a,-b); // < -1.0f
        result[7] = approx::fadd_ru(-a,-b); // = -1.0f
        }
    // subtraction
        {
        const float a = 1.0f;
        const float b = 1.e-12f;
        result[8] = approx::fsub_rd(a,b); // < 1.0f
        result[9] = approx::fsub_ru(a,b); // = 1.0f
        result[10] = approx::fsub_rd(-a,-b); // = -1.0f
        result[11] = approx::fsub_ru(-a,-b); // > -1.0f
        }
    // multiplication
        {
        const float a = 0.1f;
        const float b = 0.2f;
        result[12] = approx::fmul_rd(a,b); // = 0.02f
        result[13] = approx::fmul_ru(a,b); // > 0.02f
        result[14] = approx::fmul_rd(-a,b); // < -0.02f
        result[15] = approx::fmul_ru(-a,b);; // = -0.02f
        }
    // division
        {
        const float a = 1.0f;
        const float b = 3.0f;
        result[16] = approx::fdiv_rd(a,b); // < 1.0/3.0
        result[17] = approx::fdiv_ru(a,b); // > 1.0/3.0
        result[18] = approx::fdiv_rd(-a,b); // < -1.0/3.0
        result[19] = approx::fdiv_ru(-a,b); // > -1.0/3.0
        }
    // reciprocal
        {
        const float a = 3.0f;
        result[20] = approx::frcp_rd(a); // < 1.0/3.0
        result[21] = approx::frcp_ru(a); // > 1.0/3.0
        result[22] = approx::frcp_rd(-a); // < -1.0/3.0
        result[23] = approx::frcp_ru(-a); // > -1.0/3.0
        }
    // fma (just tests basic ops, would be to better to use a real fma)
        {
        const float a = 2.0f;
        const float b = 1.0f;
        const float c = 1.e-12f;
        result[24] = approx::fmaf_rd(a,b,c); // = 2.0f
        result[25] = approx::fmaf_ru(a,b,c); // > 2.0f
        result[26] = approx::fmaf_rd(-a,b,-c); // < -2.0f
        result[27] = approx::fmaf_ru(-a,b,-c); // = -2.0f
        }
    }

UP_TEST( approx_math_test )
    {
    // test the 7 functions in ApproximateMath.h in round down and round up modes, 2 tests each
    neighbor::shared_array<float> result(7*2*2);

    hipper::KernelLauncher launcher(1,1);
    launcher(approx_math_kernel, result.get());
    hipper::deviceSynchronize();

    // casting
    UP_ASSERT_EQUAL(result[0], 1.0f);
    UP_ASSERT_GREATER(result[1], 1.0f);
    UP_ASSERT_LESS(result[2], -1.0f);
    UP_ASSERT_EQUAL(result[3], -1.0f);
    // addition
    UP_ASSERT_LESS_EQUAL(result[4], 1.0f);
    UP_ASSERT_GREATER(result[5], 1.0f);
    UP_ASSERT_LESS(result[6], -1.0f);
    UP_ASSERT_GREATER_EQUAL(result[7], -1.0f);
    // subtraction
    UP_ASSERT_LESS(result[8], 1.0f);
    UP_ASSERT_GREATER_EQUAL(result[9], 1.0f);
    UP_ASSERT_LESS_EQUAL(result[10], -1.0f);
    UP_ASSERT_GREATER(result[11], -1.0f);
    // multiplication
    UP_ASSERT_LESS_EQUAL(result[12], 0.02f);
    UP_ASSERT_GREATER(result[13], 0.02f);
    UP_ASSERT_LESS(result[14], -0.02f);
    UP_ASSERT_GREATER_EQUAL(result[15], -0.02f);
    // division (loose test)
    UP_ASSERT_LESS(result[16], 1.0/3.0);
    UP_ASSERT_GREATER(result[17], 1.0/3.0);
    UP_ASSERT_LESS(result[18], -1.0/3.0);
    UP_ASSERT_GREATER(result[19], -1.0/3.0);
    // reciprocal (loose test)
    UP_ASSERT_LESS(result[20], 1.0/3.0);
    UP_ASSERT_GREATER(result[21], 1.0/3.0);
    UP_ASSERT_LESS(result[22], -1.0/3.0);
    UP_ASSERT_GREATER(result[23], -1.0/3.0);
    // fma
    UP_ASSERT_LESS_EQUAL(result[24], 2.0f);
    UP_ASSERT_GREATER(result[25], 2.0f);
    UP_ASSERT_LESS(result[26], -2.0f);
    UP_ASSERT_GREATER_EQUAL(result[27], -2.0f);
    }
