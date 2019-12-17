// Copyright (c) 2018-2019, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_RUNTIME_H_
#define NEIGHBOR_RUNTIME_H_

#include <iostream>
#include <functional>
#include <utility>

#include <cuda_runtime.h>
#ifdef __CUDACC__
#define GPUCC
#endif

namespace neighbor
{
namespace gpu
{

/* errors */
typedef cudaError_t error_t;
enum error
    {
    success=cudaSuccess
    };
//! Coerce comparison of gpu::error with the native error as int
inline bool operator==(error a, error_t b)
    {
    return (static_cast<int>(a) == static_cast<int>(b));
    }
//! Coerce comparison of gpu::error with the native error as int
inline bool operator==(error_t a, error b)
    {
    return (static_cast<int>(a) == static_cast<int>(b));
    }

/* streams */
typedef cudaStream_t stream_t;
//! Create a GPU stream
inline error_t streamCreate(stream_t* stream)
    {
    return cudaStreamCreate(stream);
    }
//! Synchronize a GPU stream
inline error_t streamSynchronize(stream_t stream)
    {
    return cudaStreamSynchronize(stream);
    }
//! Destroy a GPU stream
inline error_t streamDestroy(stream_t stream)
    {
    return cudaStreamDestroy(stream);
    }

/* memory */
static const int memAttachGlobal = cudaMemAttachGlobal;
static const int memAttachHost = cudaMemAttachHost;
//! Allocate GPU memory
inline error_t malloc(void** ptr, size_t size)
    {
    return cudaMalloc(ptr, size);
    }
//! Allocate managed GPU memory
inline error_t mallocManaged(void** ptr, size_t size, unsigned int flags = memAttachGlobal)
    {
    return cudaMallocManaged(ptr, size, flags);
    }
//! Free GPU memory
inline error_t free(void* ptr)
    {
    return cudaFree(ptr);
    }
//! Set GPU memory to a value
inline error_t memset(void* ptr, int value, size_t count)
    {
    return cudaMemset(ptr, value, count);
    }
//! Asynchronously set GPU memory to a value
inline error_t memsetAsync(void* ptr, int value, size_t count, stream_t stream=0)
    {
    return cudaMemsetAsync(ptr, value, count, stream);
    }

/* kernels */
typedef cudaFuncAttributes funcAttributes;
//! Get the GPU function attributes
inline error_t funcGetAttributes(funcAttributes* attr, const void* func)
    {
    return cudaFuncGetAttributes(attr, func);
    }

#ifdef GPUCC
//! Launch a compute kernel on the GPU
class KernelLauncher
    {
    public:
        KernelLauncher(int blocks, int threadsPerBlock, size_t sharedBytes, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(stream)
            {}

        KernelLauncher(int blocks, int threadsPerBlock, size_t sharedBytes)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(0)
            {}

        KernelLauncher(int blocks, int threadsPerBlock, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(stream)
            {}

        KernelLauncher(int blocks, int threadsPerBlock)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(0)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, size_t sharedBytes, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(stream)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, size_t sharedBytes)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(sharedBytes),
              stream_(0)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock, stream_t stream)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(stream)
            {}

        KernelLauncher(dim3 blocks, dim3 threadsPerBlock)
            : blocks_(blocks),
              threadsPerBlock_(threadsPerBlock),
              sharedBytes_(0),
              stream_(0)
            {}

        template<class Kernel, class ...Args>
        void operator()(const Kernel& kernel, Args&&... args)
            {
            kernel<<<blocks_,threadsPerBlock_,sharedBytes_,stream_>>>(std::forward<Args>(args)...);
            }

    private:
        dim3 blocks_;
        dim3 threadsPerBlock_;
        size_t sharedBytes_;
        stream_t stream_;
    };
#endif
} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_RUNTIME_H_
