/*
 * cuda_runtime.h - CUDA Runtime API header for cuvk
 *
 * Provides the public types and function declarations for the CUDA Runtime API.
 * This is a replacement header used when compiling against our libcudart.so
 * instead of NVIDIA's.
 */

#ifndef CUVK_CUDA_RUNTIME_H
#define CUVK_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error types
 * ============================================================================ */

typedef enum cudaError {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorCudartUnloading       = 4,
    cudaErrorInvalidDevicePointer  = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidDevice         = 101,
    cudaErrorInvalidKernelImage    = 200,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorNotReady              = 600,
    cudaErrorLaunchFailure         = 719,
    cudaErrorUnknown               = 999,
} cudaError_t;

/* ============================================================================
 * Memory copy direction
 * ============================================================================ */

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} cudaMemcpyKind;

/* ============================================================================
 * Opaque handle types
 * ============================================================================ */

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st  *cudaEvent_t;

/* ============================================================================
 * dim3
 * ============================================================================ */

typedef struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
#endif
} dim3;

/* ============================================================================
 * cudaDeviceProp (basic fields)
 * ============================================================================ */

typedef struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    major;
    int    minor;
    int    multiProcessorCount;
    int    clockRate;
    int    memoryClockRate;
    int    memoryBusWidth;
    int    l2CacheSize;
    int    computeMode;
    int    concurrentKernels;
    int    asyncEngineCount;
    int    unifiedAddressing;
    int    managedMemory;
} cudaDeviceProp;

/* ============================================================================
 * Memory management
 * ============================================================================ */

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                        cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);

/* ============================================================================
 * Device management
 * ============================================================================ */

cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);

/* ============================================================================
 * Synchronization
 * ============================================================================ */

cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);

/* ============================================================================
 * Error handling
 * ============================================================================ */

cudaError_t  cudaGetLastError(void);
cudaError_t  cudaPeekAtLastError(void);
const char  *cudaGetErrorString(cudaError_t error);
const char  *cudaGetErrorName(cudaError_t error);

/* ============================================================================
 * Stream management
 * ============================================================================ */

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

/* ============================================================================
 * Event management
 * ============================================================================ */

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/* ============================================================================
 * Kernel launch
 * ============================================================================ */

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                              void **args, size_t sharedMem,
                              cudaStream_t stream);

/* ============================================================================
 * Symbol memory operations
 * ============================================================================ */

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                size_t count, size_t offset,
                                cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                  size_t count, size_t offset,
                                  cudaMemcpyKind kind);

/* ============================================================================
 * nvcc codegen support (not user-facing, called by compiler-generated code)
 * ============================================================================ */

void      **__cudaRegisterFatBinary(void *fatCubin);
void        __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void        __cudaUnregisterFatBinary(void **fatCubinHandle);
void        __cudaRegisterFunction(void **fatCubinHandle,
                                   const char *hostFun,
                                   char *deviceFun,
                                   const char *deviceName,
                                   int thread_limit,
                                   void *tid, void *bid, dim3 *bDim,
                                   dim3 *gDim, int *wSize);
void        __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                              const char *deviceAddress,
                              const char *deviceName, int ext, size_t size,
                              int constant, int global);

unsigned    __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                        size_t sharedMem,
                                        cudaStream_t stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                        size_t *sharedMem,
                                        cudaStream_t *stream);

/* ============================================================================
 * CUDA Runtime version
 * ============================================================================ */

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
cudaError_t cudaDriverGetVersion(int *driverVersion);

#ifdef __cplusplus
}
#endif

#endif /* CUVK_CUDA_RUNTIME_H */
