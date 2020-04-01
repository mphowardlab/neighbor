/*
Copyright (c) 2020, Michael P. Howard. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

// include appropriate runtime
#if defined(HIPPER_CUDA) && !defined(HIPPER_HIP)
#include <cuda_runtime.h>
#define HIPPER(name) cuda##name
#elif !defined(HIPPER_CUDA) && defined(HIPPER_HIP)
#include <hip/hip_runtime.h>
#define HIPPER(name) hip##name
#else
#error "Device platform must be either HIPPER_CUDA or HIPPER_HIP."
#endif

#include <utility>

// set platform based on what mode is being used
#if (defined(HIPPER_CUDA) && defined(__NVCC__)) || (defined(HIPPER_HIP) && defined(__HIP_PLATFORM_NVCC__))
#define HIPPER_PLATFORM_NVCC
#elif (defined(HIPPER_HIP) && defined(__HCC__))
#define HIPPER_PLATFORM_HCC
#endif

// set device compilation flag based on CUDA or HIP flags (using HIP criteria)
#if (defined(HIPPER_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0) || (defined(HIPPER_HIP) && defined(__HIP_DEVICE_COMPILE__))
#define HIPPER_DEVICE_COMPILE 1
#endif

namespace hipper
{
/*!
 * \defgroup errors Error handling
 * @{
 */
typedef HIPPER(Error_t) error_t;

enum error
    {
    success = HIPPER(Success),
    errorInvalidValue = HIPPER(ErrorInvalidValue),
    #if defined(HIPPER_CUDA)
    errorOutOfMemory = cudaErrorMemoryAllocation,
    errorNotInitialized = cudaErrorInitializationError,
    errorDeinitialized = cudaErrorCudartUnloading,
    #elif defined(HIPPER_HIP)
    errorOutOfMemory = hipErrorOutOfMemory,
    errorNotInitialized = hipErrorNotInitialized,
    errorDeinitialized = hipErrorDeinitialized,
    #endif
    errorProfilerNotInitialized = HIPPER(ErrorProfilerNotInitialized),
    errorProfilerAlreadyStarted = HIPPER(ErrorProfilerAlreadyStarted),
    errorProfilerAlreadyStopped = HIPPER(ErrorProfilerAlreadyStopped),
    errorInvalidConfiguration = HIPPER(ErrorInvalidConfiguration),
    errorInvalidSymbol = HIPPER(ErrorInvalidSymbol),
    errorInvalidDevicePointer = HIPPER(ErrorInvalidDevicePointer),
    errorInvalidMemcpyDirection = HIPPER(ErrorInvalidMemcpyDirection),
    errorInsufficientDriver = HIPPER(ErrorInsufficientDriver),
    errorMissingConfiguration = HIPPER(ErrorMissingConfiguration),
    errorPriorLaunchFailure = HIPPER(ErrorPriorLaunchFailure),
    errorInvalidDeviceFunction = HIPPER(ErrorInvalidDeviceFunction),
    errorNoDevice = HIPPER(ErrorNoDevice),
    errorInvalidDevice = HIPPER(ErrorInvalidDevice),
    #if defined(HIPPER_CUDA)
    errorInvalidImage = cudaErrorInvalidKernelImage,
    #if CUDART_VERSION >= 10020
    errorInvalidContext = cudaErrorDeviceUninitialized,
    #else // there was a typo in older CUDA
    errorInvalidContext = cudaErrorDeviceUninitilialized,
    #endif
    errorMapFailed = cudaErrorMapBufferObjectFailed,
    errorUnmapFailed = cudaErrorUnmapBufferObjectFailed,
    errorNoBinaryForGPU = cudaErrorNoKernelImageForDevice,
    errorECCNotCorrectable = cudaErrorECCUncorrectable,
    #elif defined(HIPPER_HIP)
    errorInvalidImage = hipErrorInvalidImage,
    errorInvalidContext = hipErrorInvalidContext,
    errorMapFailed = hipErrorMapFailed,
    errorUnmapFailed = hipErrorUnmapFailed,
    errorNoBinaryForGPU = hipErrorNoBinaryForGpu,
    errorECCNotCorrectable = hipErrorECCNotCorrectable,
    #endif
    errorUnsupportedLimit = HIPPER(ErrorUnsupportedLimit),
    errorPeerAccessUnsupported = HIPPER(ErrorPeerAccessUnsupported),
    #if defined(HIPPER_CUDA)
    errorInvalidKernelFile = cudaErrorInvalidPtx,
    #elif defined(HIPPER_HIP)
    errorInvalidKernelFile = hipErrorInvalidKernelFile,
    #endif
    errorInvalidGraphicsContext = HIPPER(ErrorInvalidGraphicsContext),
    /* CUDA 10.1 only
    errorInvalidSource = HIPPER(ErrorInvalidSource),
    errorFileNotFound = HIPPER(ErrorFileNotFound),
    */
    errorSharedObjectSymbolNotFound = HIPPER(ErrorSharedObjectSymbolNotFound),
    errorSharedObjectInitFailed = HIPPER(ErrorSharedObjectInitFailed),
    errorOperatingSystem = HIPPER(ErrorOperatingSystem),
    #if defined(HIPPER_CUDA)
    errorInvalidHandle = cudaErrorInvalidResourceHandle,
    /* CUDA 10.1 only
    errorNotFound = cudaErrorSymbolNotFound,
    */
    #elif defined(HIPPER_HIP)
    errorInvalidHandle = hipErrorInvalidHandle,
    /* CUDA 10.1 only
    errorNotFound = hipErrorNotFound,
    */
    #endif
    errorNotReady = HIPPER(ErrorNotReady),
    errorIllegalAddress = HIPPER(ErrorIllegalAddress),
    errorLaunchOutOfResources = HIPPER(ErrorLaunchOutOfResources),
    #if defined(HIPPER_CUDA)
    errorLaunchTimeOut = cudaErrorLaunchTimeout,
    #elif defined(HIPPER_HIP)
    errorLaunchTimeOut = hipErrorLaunchTimeOut,
    #endif
    errorPeerAccessAlreadyEnabled = HIPPER(ErrorPeerAccessAlreadyEnabled),
    errorPeerAccessNotEnabled = HIPPER(ErrorPeerAccessNotEnabled),
    errorSetOnActiveProcess = HIPPER(ErrorSetOnActiveProcess),
    errorAssert = HIPPER(ErrorAssert),
    errorHostMemoryAlreadyRegistered = HIPPER(ErrorHostMemoryAlreadyRegistered),
    errorHostMemoryNotRegistered = HIPPER(ErrorHostMemoryNotRegistered),
    errorLaunchFailure = HIPPER(ErrorLaunchFailure),
    /* CUDA 9.0 only
    errorCooperativeLaunchTooLarge = HIPPER(ErrorCooperativeLaunchTooLarge),
    */
    errorNotSupported = HIPPER(ErrorNotSupported),
    errorUnknown = HIPPER(ErrorUnknown)
    };

//! Coerce equal comparison of hipper::error with the native error as int.
inline bool operator==(error a, error_t b)
    {
    return (static_cast<int>(a) == static_cast<int>(b));
    }
inline bool operator==(error_t a, error b)
    {
    return (b == a);
    }

//! Coerce not-equal comparison of hipper::error with the native error as int
inline bool operator!=(error a, error_t b)
    {
    return (static_cast<int>(a) != static_cast<int>(b));
    }
inline bool operator!=(error_t a, error b)
    {
    return (b != a);
    }

//! Returns the string representation of an error code enum name.
inline const char* getErrorName(error_t error)
    {
    return HIPPER(GetErrorName)(error);
    }

//! Returns the description string for an error code.
inline const char* getErrorString(error_t error)
    {
    return HIPPER(GetErrorString)(error);
    }

//! Returns the last error from a runtime call.
inline error_t getLastError(void)
    {
    return HIPPER(GetLastError)();
    }

//! Returns the last error from a runtime call.
inline error_t peekAtLastError(void)
    {
    return HIPPER(PeekAtLastError)();
    }
/*! @} */

/*!
 * \defgroup devices Device Management
 * @{
 */
#if defined(HIPPER_CUDA)
typedef cudaDeviceProp deviceProp_t;
typedef cudaFuncCache funcCache;
typedef cudaLimit limit;
#elif defined(HIPPER_HIP)
typedef hipDeviceProp_t deviceProp_t;
typedef hipFuncCache_t funcCache;
typedef hipLimit_t limit;
#endif
typedef HIPPER(SharedMemConfig) sharedMemConfig;
enum funcCache_t
    {
    funcCachePreferNone = HIPPER(FuncCachePreferNone),
    funcCachePreferShared = HIPPER(FuncCachePreferShared),
    funcCachePreferL1 = HIPPER(FuncCachePreferL1),
    funcCachePreferEqual = HIPPER(FuncCachePreferEqual)
    };
inline funcCache castFuncCache(funcCache_t cacheConfig)
    {
    funcCache result;
    switch(cacheConfig)
        {
        case funcCachePreferNone:
            result = HIPPER(FuncCachePreferNone);
            break;
        case funcCachePreferShared:
            result = HIPPER(FuncCachePreferShared);
            break;
        case funcCachePreferL1:
            result = HIPPER(FuncCachePreferL1);
            break;
        case funcCachePreferEqual:
            result = HIPPER(FuncCachePreferEqual);
            break;
        }
    return result;
    }
enum limit_t
    {
    limitMallocHeapSize = HIPPER(LimitMallocHeapSize)
    };
inline limit castLimit(limit_t lim)
    {
    limit result;
    switch(lim)
        {
        case limitMallocHeapSize:
            result = HIPPER(LimitMallocHeapSize);
            break;
        }
    return result;
    }
enum sharedMemConfig_t
    {
    sharedMemBankSizeDefault = HIPPER(SharedMemBankSizeDefault),
    sharedMemBankSizeFourByte = HIPPER(SharedMemBankSizeFourByte),
    sharedMemBankSizeEightByte = HIPPER(SharedMemBankSizeEightByte)
    };
inline sharedMemConfig castSharedMemConfig(sharedMemConfig_t config)
    {
    sharedMemConfig result;
    switch(config)
        {
        case sharedMemBankSizeFourByte:
            result = HIPPER(SharedMemBankSizeFourByte);
            break;
        case sharedMemBankSizeEightByte:
            result = HIPPER(SharedMemBankSizeEightByte);
            break;
        default:
            result = HIPPER(SharedMemBankSizeDefault);
        }
    return result;
    }
static const int deviceScheduleAuto = HIPPER(DeviceScheduleAuto);
static const int deviceScheduleSpin = HIPPER(DeviceScheduleSpin);
static const int deviceScheduleYield = HIPPER(DeviceScheduleYield);
static const int deviceScheduleBlockingSync = HIPPER(DeviceScheduleBlockingSync);
static const int deviceMapHost = HIPPER(DeviceMapHost);
static const int deviceLmemReizeToMax = HIPPER(DeviceLmemResizeToMax);

//! Select compute-device which best matches criteria.
inline error_t chooseDevice(int* device, deviceProp_t* prop)
    {
    return HIPPER(ChooseDevice)(device, prop);
    }

//! Returns a handle to a compute device.
inline error_t deviceGetByPCIBusId(int* device, const char* pciBusId)
    {
    return HIPPER(DeviceGetByPCIBusId)(device,pciBusId);
    }

//! Returns resource limits.
inline error_t deviceGetLimit(size_t* value, limit_t lim)
    {
    return HIPPER(DeviceGetLimit)(value, castLimit(lim));
    }

//! Returns a PCI Bus Id string for the device.
inline error_t deviceGetPCIBusId(char* pciBusId, int len, int device)
    {
    return HIPPER(DeviceGetPCIBusId)(pciBusId, len, device);
    }

//! Returns numerical values that correspond to the least and greatest stream priorities.
inline error_t deviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
    {
    return HIPPER(DeviceGetStreamPriorityRange)(leastPriority, greatestPriority);
    }

//! Destroy all allocations and reset all state on the current device in the current process.
inline error_t deviceReset(void)
    {
    return HIPPER(DeviceReset)();
    }

//! Sets the preferred cache configuration for the current device.
inline error_t deviceSetCacheConfig(funcCache_t cacheConfig)
    {
    return HIPPER(DeviceSetCacheConfig)(castFuncCache(cacheConfig));
    }

#if 0 // not currently supported in HIP, although it is supposed to be
//! Set resource limits.
inline error_t deviceSetLimit(limit_t lim, size_t value)
    {
    return HIPPER(DeviceSetLimit)(castLimit(lim), value);
    }
#endif

//! Sets the shared memory configuration for the current device.
inline error_t deviceSetSharedMemConfig(sharedMemConfig_t config)
    {
    return HIPPER(DeviceSetSharedMemConfig)(castSharedMemConfig(config));
    }

//! Wait for compute device to finish.
inline error_t deviceSynchronize(void)
    {
    return HIPPER(DeviceSynchronize)();
    }

//! Returns which device is currently being used.
inline error_t getDevice(int* device)
    {
    return HIPPER(GetDevice)(device);
    }

//! Returns the number of compute-capable devices.
inline error_t getDeviceCount(int* count)
    {
    return HIPPER(GetDeviceCount)(count);
    }

#if HIPPER_USE_DEPRECATED // hipCtxGetFlags is deprecated
//! Gets the flags for the current device.
inline error_t getDeviceFlags(unsigned int* flags)
    {
    #if defined(HIPPER_CUDA)
    return cudaGetDeviceFlags(flags);
    #elif defined(HIPPER_HIP)
    return hipCtxGetFlags(flags);
    #endif
    }
#endif

//! Returns information about the compute-device.
inline error_t getDeviceProperties(deviceProp_t* prop, int device)
    {
    return HIPPER(GetDeviceProperties)(prop, device);
    }

//! Set device to be used for GPU executions.
inline error_t setDevice(int device)
    {
    return HIPPER(SetDevice)(device);
    }

//! Sets flags to be used for device executions.
inline error_t setDeviceFlags(unsigned int flags)
    {
    return HIPPER(SetDeviceFlags)(flags);
    }
/*! @} */

/*!
 * \defgroup streams Stream Management
 * @{
 */
typedef HIPPER(Stream_t) stream_t;
typedef HIPPER(Event_t) event_t;

static const int streamDefault = HIPPER(StreamDefault);
static const int streamNonBlocking = HIPPER(StreamNonBlocking);

//! Create an asynchronous stream.
inline error_t streamCreate(stream_t* stream)
    {
    return HIPPER(StreamCreate)(stream);
    }

//! Create an asynchronous stream with flags.
inline error_t streamCreateWithFlags(stream_t* stream, unsigned int flags)
    {
    return HIPPER(StreamCreateWithFlags)(stream, flags);
    }

//! Destroys and cleans up an asynchronous stream.
inline error_t streamDestroy(stream_t stream)
    {
    return HIPPER(StreamDestroy)(stream);
    }

//! Queries an asynchronous stream for completion status.
inline error_t streamQuery(stream_t stream)
    {
    return HIPPER(StreamQuery)(stream);
    }

//! Waits for stream tasks to complete.
inline error_t streamSynchronize(stream_t stream)
    {
    return HIPPER(StreamSynchronize)(stream);
    }

//!Cause compute stream to wait for an event
inline error_t streamWaitEvent(stream_t stream, event_t event, unsigned int flags)
    {
    return HIPPER(StreamWaitEvent)(stream, event, flags);
    }
/*! @} */

/*!
 * \defgroup events Event Management
 * @{
 */
static const int eventDefault = HIPPER(EventDefault);
static const int eventBlockingSync = HIPPER(EventBlockingSync);
static const int eventDisableTiming = HIPPER(EventDisableTiming);
static const int eventInterprocess = HIPPER(EventInterprocess);

//! Creates an event object.
inline error_t eventCreate(event_t* event)
    {
    return HIPPER(EventCreate)(event);
    }

//! Creates an event object with the specified flags.
inline error_t eventCreateWithFlags(event_t* event, unsigned int flags)
    {
    return HIPPER(EventCreateWithFlags)(event, flags);
    }

//! Destroys an event object.
inline error_t eventDestroy(event_t event)
    {
    return HIPPER(EventDestroy)(event);
    }

//! Computes the elapsed time between events.
inline error_t eventElaspedTime(float* ms, event_t start, event_t end)
    {
    return HIPPER(EventElapsedTime)(ms, start, end);
    }

//! Queries an event's status.
inline error_t eventQuery(event_t event)
    {
    return HIPPER(EventQuery)(event);
    }

//! Records an event.
inline error_t eventRecord(event_t event, stream_t stream = 0)
    {
    return HIPPER(EventRecord)(event, stream);
    }

//! Waits for an event to complete.
inline error_t eventSynchronize(event_t event)
    {
    return HIPPER(EventSynchronize)(event);
    }
/*! @} */

/*!
 * \defgroup memory Memory Management
 * @{
 */
#if defined(HIPPER_CUDA)
static const int hostMallocDefault = cudaHostAllocDefault;
static const int hostMallocMapped = cudaHostAllocMapped;
static const int hostMallocPortable = cudaHostAllocPortable;
static const int hostMallocWriteCombined = cudaHostAllocWriteCombined;
#elif defined(HIPPER_HIP)
static const int hostMallocDefault = hipHostMallocDefault;
static const int hostMallocMapped = hipHostMallocMapped;
static const int hostMallocPortable = hipHostMallocPortable;
static const int hostMallocWriteCombined = hipHostMallocWriteCombined;
#endif
static const int hostRegisterDefault = HIPPER(HostRegisterDefault);
static const int hostRegisterMapped = HIPPER(HostRegisterMapped);
static const int hostRegisterPointer = HIPPER(HostRegisterPortable);
static const int hostRegisterIoMemory = HIPPER(HostRegisterIoMemory);

static const int memAttachGlobal = HIPPER(MemAttachGlobal);
static const int memAttachHost = HIPPER(MemAttachHost);

typedef HIPPER(MemcpyKind) memcpyKind;
enum memcpyKind_t
    {
    memcpyHostToHost = HIPPER(MemcpyHostToHost),
    memcpyHostToDevice = HIPPER(MemcpyHostToDevice),
    memcpyDeviceToHost = HIPPER(MemcpyDeviceToHost),
    memcpyDeviceToDevice = HIPPER(MemcpyDeviceToDevice),
    memcpyDefault = HIPPER(MemcpyDefault)
    };
inline memcpyKind castMemcpyKind(memcpyKind_t kind)
    {
    memcpyKind result;
    switch(kind)
        {
        case memcpyHostToHost:
            result = HIPPER(MemcpyHostToHost);
            break;
        case memcpyHostToDevice:
            result = HIPPER(MemcpyHostToDevice);
            break;
        case memcpyDeviceToHost:
            result = HIPPER(MemcpyDeviceToHost);
            break;
        default:
            result = HIPPER(MemcpyDefault);
        }
    return result;
    }

//! Frees memory on the device.
inline error_t free(void* ptr)
    {
    return HIPPER(Free)(ptr);
    }

//! Frees page-locked memory.
inline error_t hostFree(void* ptr)
    {
    #if defined(HIPPER_CUDA)
    return cudaFreeHost(ptr);
    #elif defined(HIPPER_HIP)
    return hipHostFree(ptr);
    #endif
    }

//! Passes back device pointer of mapped host memory, which is registered in hostRegister.
inline error_t hostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
    {
    return HIPPER(HostGetDevicePointer)(pDevice, pHost, flags);
    }

//! Registers an existing host memory range for use by CUDA.
inline error_t hostRegister(void* ptr, size_t size, unsigned int flags)
    {
    return HIPPER(HostRegister)(ptr, size, flags);
    }

//! Unregisters a memory range that was registered with hostRegister.
inline error_t hostUnregister(void *ptr)
    {
    return HIPPER(HostUnregister)(ptr);
    }

//! Allocate memory on the device.
inline error_t malloc(void** ptr, size_t size)
    {
    return HIPPER(Malloc)(ptr, size);
    }

//! Allocate managed GPU memory
inline error_t mallocManaged(void** ptr, size_t size, unsigned int flags = memAttachGlobal)
    {
    return HIPPER(MallocManaged)(ptr, size, flags);
    }

//! Copies data between host and device
inline error_t memcpy(void* dst, const void* src, size_t count, memcpyKind_t kind)
    {
    return HIPPER(Memcpy)(dst, src, count, castMemcpyKind(kind));
    }

//! Asynchronously copies memory between host and device
inline error_t memcpyAsync(void* dst, const void* src, size_t count, memcpyKind_t kind, stream_t stream = 0)
    {
    return HIPPER(MemcpyAsync)(dst, src, count, castMemcpyKind(kind), stream);
    }

//! Initializes or sets device memory to a value.
inline error_t memset(void* ptr, int value, size_t count)
    {
    return HIPPER(Memset)(ptr, value, count);
    }

//! Initializes or sets device memory to a value asynchronously.
inline error_t memsetAsync(void* ptr, int value, size_t count, stream_t stream = 0)
    {
    return HIPPER(MemsetAsync)(ptr, value, count, stream);
    }
/*! @} */

/*!
 * \defgroup profile Profiler Control
 * @{
 */
#if HIPPER_USE_DEPRECATED // hip profiling is deprecated for roctracer
//! Enable profiling.
inline error_t profilerStart(void)
    {
    return HIPPER(ProfilerStart)();
    }
//! Disable profiling.
inline error_t profilerStop(void)
    {
    return HIPPER(ProfilerStop)();
    }
#endif
/*! @} */

/*!
 * \defgroup exec Execution Control
 * @{
 */
typedef HIPPER(FuncAttributes) funcAttributes_t;

//! Find out attributes for a given function.
inline error_t funcGetAttributes(funcAttributes_t* attr, const void* func)
    {
    return HIPPER(FuncGetAttributes)(attr, func);
    }

//! Sets the preferred cache configuration for a device function.
inline error_t funcSetCacheConfig(const void* func, funcCache_t cacheConfig)
    {
    return HIPPER(FuncSetCacheConfig)(func, castFuncCache(cacheConfig));
    }

//! Thread index in block.
__device__ inline dim3 threadIndex()
    {
    #if defined(HIPPER_CUDA)
    return threadIdx;
    #elif defined(HIPPER_HIP)
    return dim3(hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z);
    #endif
    }

//! Number of threads in block.
__device__ inline dim3 blockSize()
    {
    #if defined(HIPPER_CUDA)
    return blockDim;
    #elif defined(HIPPER_HIP)
    return dim3(hipBlockDim_x, hipBlockDim_y, hipBlockDim_z);
    #endif
    }

//! Block index in grid.
__device__ inline dim3 blockIndex()
    {
    #if defined(HIPPER_CUDA)
    return blockIdx;
    #elif defined(HIPPER_HIP)
    return dim3(hipBlockIdx_x, hipBlockIdx_y, hipBlockIdx_z);
    #endif
    }

//! Number of blocks in grid.
__device__ inline dim3 gridSize()
    {
    #if defined(HIPPER_CUDA)
    return gridDim;
    #elif defined(HIPPER_HIP)
    return dim3(hipGridDim_x, hipGridDim_y, hipGridDim_z);
    #endif
    }

//! Map the launch dimensions to a one-dimensional rank.
template<char GridDim=1, char BlockDim=1>
__device__ int threadRank() = delete;
template<>
__device__ int threadRank<1,1>()
    {
    return blockIndex().x*blockSize().x + threadIndex().x;
    }
template<>
__device__ int threadRank<1,2>()
    {
    const dim3 bDim = blockSize();
    const dim3 tIdx = threadIndex();
    return blockIndex().x*bDim.x*(bDim.y + tIdx.y) + tIdx.x;
    }
template<>
__device__ int threadRank<1,3>()
    {
    const dim3 bDim = blockSize();
    const dim3 tIdx = threadIndex();
    return blockIndex().x*bDim.x*(bDim.y*(bDim.z + tIdx.z) + tIdx.y) + tIdx.x;
    }

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
            #if defined(HIPPER_CUDA)
            kernel<<<blocks_,threadsPerBlock_,sharedBytes_,stream_>>>(std::forward<Args>(args)...);
            #elif defined(HIPPER_HIP)
            hipLaunchKernelGGL(kernel,blocks_,threadsPerBlock_,sharedBytes_,stream_,std::forward<Args>(args)...);
            #endif
            }

    private:
        dim3 blocks_;
        dim3 threadsPerBlock_;
        size_t sharedBytes_;
        stream_t stream_;
    };
/*! @} */
} // end namespace hipper
