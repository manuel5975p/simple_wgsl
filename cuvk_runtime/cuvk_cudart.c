/*
 * cuvk_cudart.c - CUDA Runtime API implementation
 *
 * Thin wrappers around the CUDA Driver API (cuvk_runtime) to provide
 * libcudart.so functionality. Handles:
 *   - Memory management (cudaMalloc, cudaFree, cudaMemcpy, cudaMemset)
 *   - Device management (cudaGetDevice, cudaSetDevice, etc.)
 *   - Stream/event management
 *   - Error handling
 *   - nvcc codegen support (fatbin registration, launch configuration)
 *   - Kernel launch (cudaLaunchKernel)
 */

#include "cuvk_internal.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* ============================================================================
 * Internal state
 * ============================================================================ */

static __thread cudaError_t g_last_error = cudaSuccess;
static __thread int g_current_device = 0;
static int g_cudart_initialized = 0;

/* CUresult -> cudaError_t mapping */
static cudaError_t cu_to_cudart(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS:              return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:  return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY:  return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED:return cudaErrorInitializationError;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_IMAGE:  return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_NOT_READY:      return cudaErrorNotReady;
    default:                        return cudaErrorUnknown;
    }
}

/* Set last error and return it */
static cudaError_t set_error(cudaError_t e) {
    g_last_error = e;
    return e;
}

/* Lazy init: ensure driver + context are ready */
static cudaError_t ensure_init(void) {
    if (g_cuvk.initialized && g_cuvk.current_ctx)
        return cudaSuccess;

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));

    if (!g_cuvk.current_ctx) {
        CUcontext ctx;
        r = cuDevicePrimaryCtxRetain(&ctx, g_current_device);
        if (r != CUDA_SUCCESS)
            return set_error(cu_to_cudart(r));
    }

    g_cudart_initialized = 1;
    return cudaSuccess;
}

/* ============================================================================
 * Error handling
 * ============================================================================ */

cudaError_t cudaGetLastError(void) {
    cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError(void) {
    return g_last_error;
}

const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
    case cudaSuccess:                    return "no error";
    case cudaErrorInvalidValue:          return "invalid argument";
    case cudaErrorMemoryAllocation:      return "out of memory";
    case cudaErrorInitializationError:   return "initialization error";
    case cudaErrorCudartUnloading:       return "driver shutting down";
    case cudaErrorInvalidDevicePointer:  return "invalid device pointer";
    case cudaErrorInvalidMemcpyDirection:return "invalid memcpy direction";
    case cudaErrorInvalidDevice:         return "invalid device ordinal";
    case cudaErrorInvalidKernelImage:    return "invalid kernel image";
    case cudaErrorNoKernelImageForDevice:return "no kernel image for device";
    case cudaErrorNotReady:              return "device not ready";
    case cudaErrorLaunchFailure:         return "launch failure";
    default:                             return "unknown error";
    }
}

const char *cudaGetErrorName(cudaError_t error) {
    switch (error) {
    case cudaSuccess:                    return "cudaSuccess";
    case cudaErrorInvalidValue:          return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:   return "cudaErrorInitializationError";
    case cudaErrorCudartUnloading:       return "cudaErrorCudartUnloading";
    case cudaErrorInvalidDevicePointer:  return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidMemcpyDirection:return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorInvalidDevice:         return "cudaErrorInvalidDevice";
    case cudaErrorInvalidKernelImage:    return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice:return "cudaErrorNoKernelImageForDevice";
    case cudaErrorNotReady:              return "cudaErrorNotReady";
    case cudaErrorLaunchFailure:         return "cudaErrorLaunchFailure";
    default:                             return "cudaErrorUnknown";
    }
}

/* ============================================================================
 * Device management
 * ============================================================================ */

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return set_error(cudaErrorInvalidValue);
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;
    *count = (int)g_cuvk.physical_device_count;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return set_error(cudaErrorInvalidValue);
    *device = g_current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;
    if (device < 0 || (uint32_t)device >= g_cuvk.physical_device_count)
        return set_error(cudaErrorInvalidDevice);
    g_current_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) return set_error(cudaErrorInvalidValue);
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;
    if (device < 0 || (uint32_t)device >= g_cuvk.physical_device_count)
        return set_error(cudaErrorInvalidDevice);

    memset(prop, 0, sizeof(*prop));

    CUdevice dev = (CUdevice)device;
    cuDeviceGetName(prop->name, sizeof(prop->name), dev);

    size_t total = 0;
    cuDeviceTotalMem_v2(&total, dev);
    prop->totalGlobalMem = total;

    int val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    prop->maxThreadsPerBlock = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
    prop->maxThreadsDim[0] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
    prop->maxThreadsDim[1] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
    prop->maxThreadsDim[2] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
    prop->maxGridSize[0] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
    prop->maxGridSize[1] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
    prop->maxGridSize[2] = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    prop->warpSize = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    prop->sharedMemPerBlock = (size_t)val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
    prop->regsPerBlock = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    prop->multiProcessorCount = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    prop->clockRate = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    prop->major = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    prop->minor = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    prop->unifiedAddressing = val;
    cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
    prop->managedMemory = val;

    return cudaSuccess;
}

/* ============================================================================
 * Memory management
 * ============================================================================ */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return set_error(cudaErrorInvalidValue);
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;
    if (size == 0) { *devPtr = NULL; return cudaSuccess; }

    CUdeviceptr dptr;
    CUresult r = cuMemAlloc_v2(&dptr, size);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    *devPtr = (void *)dptr;
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return cudaSuccess;
    CUresult r = cuMemFree_v2((CUdeviceptr)devPtr);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                        cudaMemcpyKind kind) {
    if (count == 0) return cudaSuccess;
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUresult r;
    switch (kind) {
    case cudaMemcpyHostToDevice:
        r = cuMemcpyHtoD_v2((CUdeviceptr)dst, src, count);
        break;
    case cudaMemcpyDeviceToHost:
        r = cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyDeviceToDevice:
        r = cuMemcpyDtoD_v2((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        return cudaSuccess;
    case cudaMemcpyDefault:
        /* TODO: auto-detect direction from pointer attributes */
        r = cuMemcpyHtoD_v2((CUdeviceptr)dst, src, count);
        break;
    default:
        return set_error(cudaErrorInvalidMemcpyDirection);
    }
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUresult r;
    switch (kind) {
    case cudaMemcpyHostToDevice:
        r = cuMemcpyHtoDAsync_v2((CUdeviceptr)dst, src, count, (CUstream)stream);
        break;
    case cudaMemcpyDeviceToHost:
        r = cuMemcpyDtoHAsync_v2(dst, (CUdeviceptr)src, count, (CUstream)stream);
        break;
    case cudaMemcpyDeviceToDevice:
        /* DtoD async not available in driver API; fall back to synchronous */
        r = cuMemcpyDtoD_v2((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyHostToHost:
        memcpy(dst, src, count);
        return cudaSuccess;
    case cudaMemcpyDefault:
        r = cuMemcpyHtoDAsync_v2((CUdeviceptr)dst, src, count, (CUstream)stream);
        break;
    default:
        return set_error(cudaErrorInvalidMemcpyDirection);
    }
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    if (count == 0) return cudaSuccess;
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUresult r = cuMemsetD8_v2((CUdeviceptr)devPtr, (unsigned char)value, count);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) return set_error(cudaErrorInvalidValue);
    *ptr = malloc(size);
    if (!*ptr) return set_error(cudaErrorMemoryAllocation);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

/* ============================================================================
 * Synchronization
 * ============================================================================ */

cudaError_t cudaDeviceSynchronize(void) {
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;
    CUresult r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    return cudaDeviceSynchronize();
}

/* ============================================================================
 * Stream management
 * ============================================================================ */

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) return set_error(cudaErrorInvalidValue);
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUresult r = cuStreamCreate((CUstream *)pStream, 0);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    CUresult r = cuStreamDestroy_v2((CUstream)stream);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    CUresult r = cuStreamSynchronize((CUstream)stream);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

/* ============================================================================
 * Event management
 * ============================================================================ */

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) return set_error(cudaErrorInvalidValue);
    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUresult r = cuEventCreate((CUevent *)event, 0);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    CUresult r = cuEventDestroy_v2((CUevent)event);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    CUresult r = cuEventRecord((CUevent)event, (CUstream)stream);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    CUresult r = cuEventSynchronize((CUevent)event);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return set_error(cudaErrorInvalidValue);
    CUresult r = cuEventElapsedTime_v2(ms, (CUevent)start, (CUevent)end);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}

/* ============================================================================
 * Version queries
 * ============================================================================ */

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) return set_error(cudaErrorInvalidValue);
    *runtimeVersion = 12020;  /* CUDA 12.2 */
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return set_error(cudaErrorInvalidValue);
    int ver;
    CUresult r = cuDriverGetVersion(&ver);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    *driverVersion = ver;
    return cudaSuccess;
}

/* ============================================================================
 * Symbol operations (stubs - need module-level global variable tracking)
 * ============================================================================ */

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                size_t count, size_t offset,
                                cudaMemcpyKind kind) {
    (void)symbol; (void)src; (void)count; (void)offset; (void)kind;
    return set_error(cudaErrorUnknown);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                  size_t count, size_t offset,
                                  cudaMemcpyKind kind) {
    (void)dst; (void)symbol; (void)count; (void)offset; (void)kind;
    return set_error(cudaErrorUnknown);
}

/* ============================================================================
 * Fatbin registration and kernel launch support
 * ============================================================================ */

/* Registry entry: maps a fatbin handle to its loaded CUmodule */
typedef struct FatbinEntry {
    void       **handle;    /* the handle returned to nvcc codegen */
    CUmodule     module;    /* loaded driver API module */
} FatbinEntry;

/* Registry entry: maps host function pointer -> (module, kernel name) */
typedef struct FuncEntry {
    const char  *host_fun;     /* host-side function pointer (as address) */
    CUmodule     module;
    char        *device_name;  /* kernel name string (owned) */
    CUfunction   func;         /* resolved CUfunction (lazy) */
} FuncEntry;

/* Registry entry: maps host variable pointer -> (module, var name, device ptr) */
typedef struct VarEntry {
    const char  *host_var;
    CUmodule     module;
    char        *device_name;
    CUdeviceptr  dptr;
    size_t       size;
} VarEntry;

#define MAX_FATBINS 64
#define MAX_FUNCS   256
#define MAX_VARS    256

static FatbinEntry  g_fatbins[MAX_FATBINS];
static int          g_fatbin_count = 0;

static FuncEntry    g_funcs[MAX_FUNCS];
static int          g_func_count = 0;

static VarEntry     g_vars[MAX_VARS];
static int          g_var_count = 0;

static pthread_mutex_t g_registry_lock = PTHREAD_MUTEX_INITIALIZER;

/* FatbincWrapper from CUDA internals */
#define FATBINC_MAGIC 0x466243B1u

typedef struct {
    unsigned int magic;
    unsigned int version;
    const void  *data;
    const void  *filename_or_fatbins;
} FatbincWrapper;

void **__cudaRegisterFatBinary(void *fatCubin) {
    CUVK_LOG("[cudart] __cudaRegisterFatBinary(%p)\n", fatCubin);

    /* Lazy init driver */
    ensure_init();

    pthread_mutex_lock(&g_registry_lock);

    if (g_fatbin_count >= MAX_FATBINS) {
        CUVK_LOG("[cudart] ERROR: fatbin registry full\n");
        pthread_mutex_unlock(&g_registry_lock);
        return NULL;
    }

    /* Load the module via the driver API */
    CUmodule module = NULL;
    const FatbincWrapper *wrapper = (const FatbincWrapper *)fatCubin;
    CUresult r;
    if (wrapper->magic == FATBINC_MAGIC && wrapper->data) {
        r = cuModuleLoadData(&module, wrapper->data);
    } else {
        r = cuModuleLoadData(&module, fatCubin);
    }

    if (r != CUDA_SUCCESS) {
        CUVK_LOG("[cudart] WARNING: cuModuleLoadData failed (%d), deferring\n", r);
        /* Store NULL module - will be loaded lazily when context is ready */
    }

    int idx = g_fatbin_count++;
    g_fatbins[idx].handle = (void **)fatCubin;
    g_fatbins[idx].module = module;

    pthread_mutex_unlock(&g_registry_lock);

    CUVK_LOG("[cudart]   -> handle=%p module=%p\n", fatCubin, (void *)module);
    return (void **)&g_fatbins[idx];
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    (void)fatCubinHandle;
    CUVK_LOG("[cudart] __cudaRegisterFatBinaryEnd(%p)\n", (void *)fatCubinHandle);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    CUVK_LOG("[cudart] __cudaUnregisterFatBinary(%p)\n", (void *)fatCubinHandle);

    pthread_mutex_lock(&g_registry_lock);

    /* Find and unload the module */
    FatbinEntry *entry = (FatbinEntry *)fatCubinHandle;
    if (entry >= g_fatbins && entry < g_fatbins + g_fatbin_count) {
        if (entry->module) {
            cuModuleUnload(entry->module);
            entry->module = NULL;
        }
    }

    pthread_mutex_unlock(&g_registry_lock);
}

void __cudaRegisterFunction(void **fatCubinHandle,
                            const char *hostFun,
                            char *deviceFun,
                            const char *deviceName,
                            int thread_limit,
                            void *tid, void *bid, dim3 *bDim,
                            dim3 *gDim, int *wSize) {
    (void)thread_limit; (void)tid; (void)bid; (void)bDim;
    (void)gDim; (void)wSize;

    CUVK_LOG("[cudart] __cudaRegisterFunction(handle=%p, host=%p, dev=\"%s\")\n",
            (void *)fatCubinHandle, (const void *)hostFun,
            deviceName ? deviceName : deviceFun);

    pthread_mutex_lock(&g_registry_lock);

    if (g_func_count >= MAX_FUNCS) {
        CUVK_LOG("[cudart] ERROR: function registry full\n");
        pthread_mutex_unlock(&g_registry_lock);
        return;
    }

    FatbinEntry *fb = (FatbinEntry *)fatCubinHandle;
    CUmodule module = NULL;
    if (fb >= g_fatbins && fb < g_fatbins + g_fatbin_count)
        module = fb->module;

    const char *name = deviceName ? deviceName : deviceFun;
    int idx = g_func_count++;
    g_funcs[idx].host_fun = hostFun;
    g_funcs[idx].module = module;
    g_funcs[idx].device_name = strdup(name);
    g_funcs[idx].func = NULL;  /* resolve lazily */

    pthread_mutex_unlock(&g_registry_lock);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       const char *deviceAddress,
                       const char *deviceName, int ext, size_t size,
                       int constant, int global) {
    (void)deviceAddress; (void)ext; (void)constant; (void)global;

    CUVK_LOG("[cudart] __cudaRegisterVar(handle=%p, host=%p, name=\"%s\", size=%zu)\n",
            (void *)fatCubinHandle, (void *)hostVar, deviceName, size);

    pthread_mutex_lock(&g_registry_lock);

    if (g_var_count >= MAX_VARS) {
        pthread_mutex_unlock(&g_registry_lock);
        return;
    }

    FatbinEntry *fb = (FatbinEntry *)fatCubinHandle;
    CUmodule module = NULL;
    if (fb >= g_fatbins && fb < g_fatbins + g_fatbin_count)
        module = fb->module;

    int idx = g_var_count++;
    g_vars[idx].host_var = hostVar;
    g_vars[idx].module = module;
    g_vars[idx].device_name = strdup(deviceName);
    g_vars[idx].dptr = 0;
    g_vars[idx].size = size;

    pthread_mutex_unlock(&g_registry_lock);
}

/* Resolve a host function pointer to a CUfunction */
static CUfunction resolve_func(const void *hostFun) {
    pthread_mutex_lock(&g_registry_lock);

    for (int i = 0; i < g_func_count; i++) {
        if (g_funcs[i].host_fun == (const char *)hostFun) {
            if (!g_funcs[i].func && g_funcs[i].module) {
                /* Lazy resolve */
                cuModuleGetFunction(&g_funcs[i].func, g_funcs[i].module,
                                    g_funcs[i].device_name);
                CUVK_LOG("[cudart] resolved \"%s\" -> %p\n",
                        g_funcs[i].device_name, (void *)g_funcs[i].func);
            }
            CUfunction f = g_funcs[i].func;
            pthread_mutex_unlock(&g_registry_lock);
            return f;
        }
    }

    pthread_mutex_unlock(&g_registry_lock);
    CUVK_LOG("[cudart] WARNING: host function %p not found in registry\n", hostFun);
    return NULL;
}

/* ============================================================================
 * Launch configuration stack (for <<<>>> syntax)
 * ============================================================================ */

typedef struct LaunchConfig {
    dim3         gridDim;
    dim3         blockDim;
    size_t       sharedMem;
    cudaStream_t stream;
} LaunchConfig;

#define MAX_LAUNCH_STACK 32
static __thread LaunchConfig g_launch_stack[MAX_LAUNCH_STACK];
static __thread int          g_launch_stack_top = 0;

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem,
                                      cudaStream_t stream) {
    CUVK_LOG("[cudart] __cudaPushCallConfiguration(grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu)\n",
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z, sharedMem);

    if (g_launch_stack_top >= MAX_LAUNCH_STACK) {
        CUVK_LOG("[cudart] ERROR: launch config stack overflow\n");
        return 1;
    }

    LaunchConfig *cfg = &g_launch_stack[g_launch_stack_top++];
    cfg->gridDim = gridDim;
    cfg->blockDim = blockDim;
    cfg->sharedMem = sharedMem;
    cfg->stream = stream;
    return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                        size_t *sharedMem,
                                        cudaStream_t *stream) {
    if (g_launch_stack_top <= 0) {
        CUVK_LOG("[cudart] ERROR: launch config stack underflow\n");
        return set_error(cudaErrorInvalidValue);
    }

    LaunchConfig *cfg = &g_launch_stack[--g_launch_stack_top];
    if (gridDim)   *gridDim = cfg->gridDim;
    if (blockDim)  *blockDim = cfg->blockDim;
    if (sharedMem) *sharedMem = cfg->sharedMem;
    if (stream)    *stream = cfg->stream;
    return cudaSuccess;
}

/* ============================================================================
 * Kernel launch
 * ============================================================================ */

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                              void **args, size_t sharedMem,
                              cudaStream_t stream) {
    CUVK_LOG("[cudart] cudaLaunchKernel(func=%p grid=(%u,%u,%u) block=(%u,%u,%u))\n",
            func, gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z);

    cudaError_t e = ensure_init();
    if (e != cudaSuccess) return e;

    CUfunction cu_func = resolve_func(func);
    if (!cu_func) {
        CUVK_LOG("[cudart] ERROR: could not resolve function %p\n", func);
        return set_error(cudaErrorInvalidKernelImage);
    }

    CUresult r = cuLaunchKernel(cu_func,
                                gridDim.x, gridDim.y, gridDim.z,
                                blockDim.x, blockDim.y, blockDim.z,
                                (unsigned int)sharedMem,
                                (CUstream)stream,
                                args, NULL);
    if (r != CUDA_SUCCESS)
        return set_error(cu_to_cudart(r));
    return cudaSuccess;
}
