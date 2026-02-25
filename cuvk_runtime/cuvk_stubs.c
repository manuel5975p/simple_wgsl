/*
 * cuvk_stubs.c - Stub implementations for unimplemented CUDA driver API functions
 *
 * Each stub returns CUDA_ERROR_NOT_SUPPORTED (unless otherwise noted).
 * This file ensures that linking against libcuvk_runtime doesn't fail for
 * commonly used CUDA API functions.
 *
 * Functions already implemented elsewhere:
 *   cuvk_init.c:   cuInit, cuDriverGetVersion, cuDeviceGet, cuDeviceGetCount,
 *                  cuDeviceGetName, cuDeviceTotalMem_v2, cuDeviceGetAttribute,
 *                  cuCtxCreate_v4, cuCtxDestroy_v2, cuCtxGetCurrent,
 *                  cuCtxSetCurrent, cuCtxSynchronize
 *   cuvk_memory.c: cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2,
 *                  cuMemcpyDtoH_v2, cuMemcpyDtoD_v2, cuMemcpyHtoDAsync_v2,
 *                  cuMemcpyDtoHAsync_v2, cuMemsetD8_v2, cuMemsetD16_v2,
 *                  cuMemsetD32_v2
 *   cuvk_module.c: cuModuleLoadData, cuModuleLoadDataEx, cuModuleUnload,
 *                  cuModuleGetFunction, cuFuncGetAttribute
 *   cuvk_launch.c: cuLaunchKernel
 *   cuvk_stream.c: cuStreamCreate, cuStreamCreateWithPriority,
 *                  cuStreamDestroy_v2, cuStreamSynchronize, cuStreamQuery,
 *                  cuEventCreate, cuEventDestroy_v2, cuEventRecord,
 *                  cuEventSynchronize, cuEventElapsedTime
 */

#include "cuvk_internal.h"

#include <stdio.h>
#include <string.h>

#define CUVK_MAX_DEVICES 8
static CUcontext g_primary_ctx[CUVK_MAX_DEVICES] = {0};
static int g_primary_ctx_refcount[CUVK_MAX_DEVICES] = {0};

/* ============================================================================
 * Error string helpers (proper implementations, not stubs)
 * ============================================================================ */

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    switch (error) {
    case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
    return cuGetErrorString(error, pStr); /* same for our purposes */
}

/* ============================================================================
 * Device management extras
 * ============================================================================ */

/* cuDeviceGetUuid: macro maps to cuDeviceGetUuid_v2 */
CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    if (!g_cuvk.initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || (uint32_t)dev >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;
    memset(uuid->bytes, 0, sizeof(uuid->bytes));
    uuid->bytes[0] = 0xCU;
    uuid->bytes[1] = 0xDA;
    uuid->bytes[2] = (unsigned char)(dev & 0xFF);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                                 CUdevice dev) {
    (void)luid; (void)deviceNodeMask; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    CUVK_LOG("[cuvk] cuDevicePrimaryCtxRetain(dev=%d)\n", dev);
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= CUVK_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;

    if (!g_primary_ctx[dev]) {
        CUresult res = cuCtxCreate_v4(&g_primary_ctx[dev], NULL, 0, dev);
        CUVK_LOG("[cuvk]   cuCtxCreate_v4(dev=%d) returned %d\n", dev, res);
        if (res != CUDA_SUCCESS)
            return res;
    }
    g_primary_ctx_refcount[dev]++;
    g_cuvk.current_ctx = g_primary_ctx[dev];
    *pctx = g_primary_ctx[dev];
    CUVK_LOG("[cuvk]   primary ctx[%d]=%p\n", dev, (void *)g_primary_ctx[dev]);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    if (dev < 0 || dev >= CUVK_MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE;
    if (!g_primary_ctx[dev])
        return CUDA_SUCCESS;
    if (--g_primary_ctx_refcount[dev] <= 0) {
        cuCtxDestroy_v2(g_primary_ctx[dev]);
        g_primary_ctx[dev] = NULL;
        g_primary_ctx_refcount[dev] = 0;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
    if (dev < 0 || dev >= CUVK_MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE;
    if (g_primary_ctx[dev]) {
        cuCtxDestroy_v2(g_primary_ctx[dev]);
        g_primary_ctx[dev] = NULL;
        g_primary_ctx_refcount[dev] = 0;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
    if (dev < 0 || dev >= CUVK_MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE;
    if (flags) *flags = 0;
    if (active) *active = (g_primary_ctx[dev] != NULL) ? 1 : 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    (void)dev; (void)flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(
    size_t *maxWidthInElements, CUarray_format format,
    unsigned numChannels, CUdevice dev) {
    (void)maxWidthInElements; (void)format; (void)numChannels; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
    (void)canAccessPeer; (void)dev; (void)peerDev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
    (void)value; (void)attrib; (void)srcDevice; (void)dstDevice;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    (void)dev; (void)pciBusId;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    (void)pciBusId; (void)len; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Context extras
 * ============================================================================ */

/* cuCtxPushCurrent: macro maps to cuCtxPushCurrent_v2 */
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
    g_cuvk.current_ctx = ctx;
    return CUDA_SUCCESS;
}

/* cuCtxPopCurrent: macro maps to cuCtxPopCurrent_v2 */
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
    if (pctx) *pctx = g_cuvk.current_ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (!g_cuvk.current_ctx) return CUDA_ERROR_INVALID_CONTEXT;
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx;
    if (!version) return CUDA_ERROR_INVALID_VALUE;
    *version = 13010;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
    (void)limit; (void)value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    (void)limit;
    if (!pvalue) return CUDA_ERROR_INVALID_VALUE;
    *pvalue = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
    if (!pconfig) return CUDA_ERROR_INVALID_VALUE;
    *pconfig = CU_FUNC_CACHE_PREFER_NONE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
    (void)config;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
    (void)peerContext; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
    (void)peerContext;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Memory extras
 * ============================================================================ */

/* cuMemGetInfo: macro maps to cuMemGetInfo_v2 */
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
    CUVK_LOG("[cuvk] cuMemGetInfo called\n");
    fflush(stderr);
    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mem);
    size_t total_mem = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total_mem += mem.memoryHeaps[i].size;
    }
    if (total) *total = total_mem;
    if (free) *free = total_mem;
    return CUDA_SUCCESS;
}

/* cuMemAllocPitch: macro maps to cuMemAllocPitch_v2 */
CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
    (void)dptr; (void)pPitch; (void)WidthInBytes; (void)Height;
    (void)ElementSizeBytes;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemAllocHost: macro maps to cuMemAllocHost_v2 */
CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
    (void)pp; (void)bytesize;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
    (void)p;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
    (void)pp; (void)bytesize; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemHostGetDevicePointer: macro maps to cuMemHostGetDevicePointer_v2 */
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
    (void)pdptr; (void)p; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    (void)pFlags; (void)p;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
    (void)dptr; (void)bytesize; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemHostRegister: macro maps to cuMemHostRegister_v2 */
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
    (void)p; (void)bytesize; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemHostUnregister(void *p) {
    (void)p;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src,
                          size_t ByteCount) {
    (void)dst; (void)src; (void)ByteCount;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
    (void)dst; (void)src; (void)ByteCount; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext;
    (void)ByteCount;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext;
    (void)ByteCount; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemAdvise: macro maps to cuMemAdvise_v2 */
CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUmemLocation location) {
    (void)devPtr; (void)count; (void)advice; (void)location;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemPrefetchAsync: macro maps via __CUDA_API_PTSZ to cuMemPrefetchAsync_v2 */
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUmemLocation location,
                                    unsigned int flags, CUstream hStream) {
    (void)devPtr; (void)count; (void)location; (void)flags; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
    (void)data; (void)attribute; (void)ptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
    (void)numAttributes; (void)attributes; (void)data; (void)ptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
    (void)value; (void)attribute; (void)ptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuMemGetAddressRange: macro maps to cuMemGetAddressRange_v2 */
CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
    (void)pbase; (void)psize; (void)dptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
    (void)pCopy;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
    (void)pCopy; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle,
                                     CUevent event) {
    (void)pHandle; (void)event;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
    (void)phEvent; (void)handle;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle,
                                   CUdeviceptr dptr) {
    (void)pHandle; (void)dptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuIpcOpenMemHandle: macro maps to cuIpcOpenMemHandle_v2 */
CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
    (void)pdptr; (void)handle; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize,
                                 CUstream hStream) {
    (void)dptr; (void)bytesize; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    (void)dptr; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Module/function extras
 * ============================================================================ */

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    (void)module; (void)fname;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module,
                                       const void *fatCubin) {
    return cuModuleLoadData(module, fatCubin);
}

/* cuModuleGetGlobal: macro maps to cuModuleGetGlobal_v2 */
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
    (void)pTexRef; (void)hmod; (void)name;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
    (void)pSurfRef; (void)hmod; (void)name;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = CU_MODULE_EAGER_LOADING;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
    (void)hfunc; (void)attrib; (void)value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc,
                                      CUfunc_cache config) {
    (void)hfunc; (void)config;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
    (void)hfunc; (void)config;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
    (void)hmod; (void)hfunc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncGetName(const char **name, CUfunction hfunc) {
    (void)name; (void)hfunc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncIsLoaded(CUfunctionLoadingState *state,
                                CUfunction function) {
    (void)function;
    if (state) *state = CU_FUNCTION_LOADING_STATE_LOADED;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncLoad(CUfunction function) {
    (void)function;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Library API (CUDA 12+) - wrappers around module API
 * CUlibrary is internally CUmodule, CUkernel is internally CUfunction.
 * ============================================================================ */

CUresult CUDAAPI cuLibraryLoadData(CUlibrary *library, const void *code,
                                   CUjit_option *jitOptions,
                                   void **jitOptionsValues,
                                   unsigned int numJitOptions,
                                   CUlibraryOption *libraryOptions,
                                   void **libraryOptionValues,
                                   unsigned int numLibraryOptions) {
    (void)jitOptions; (void)jitOptionsValues; (void)numJitOptions;
    (void)libraryOptions; (void)libraryOptionValues; (void)numLibraryOptions;
    CUVK_LOG("[cuvk] cuLibraryLoadData: library=%p code=%p\n",
            (void *)library, code);
    fflush(stderr);
    if (!library || !code)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData((CUmodule *)library, code);
}

CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                                       CUjit_option *jitOptions,
                                       void **jitOptionsValues,
                                       unsigned int numJitOptions,
                                       CUlibraryOption *libraryOptions,
                                       void **libraryOptionValues,
                                       unsigned int numLibraryOptions) {
    (void)library; (void)fileName;
    (void)jitOptions; (void)jitOptionsValues; (void)numJitOptions;
    (void)libraryOptions; (void)libraryOptionValues; (void)numLibraryOptions;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLibraryUnload(CUlibrary library) {
    return cuModuleUnload((CUmodule)library);
}

CUresult CUDAAPI cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                                    const char *name) {
    CUVK_LOG("[cuvk] cuLibraryGetKernel: name=%s\n", name ? name : "(null)");
    fflush(stderr);
    return cuModuleGetFunction((CUfunction *)pKernel,
                               (CUmodule)library, name);
}

CUresult CUDAAPI cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
    if (!pMod) return CUDA_ERROR_INVALID_VALUE;
    *pMod = (CUmodule)library;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
    if (!pFunc) return CUDA_ERROR_INVALID_VALUE;
    *pFunc = (CUfunction)kernel;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel) {
    if (!pLib || !kernel) return CUDA_ERROR_INVALID_VALUE;
    *pLib = (CUlibrary)((CUfunction)kernel)->module;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                    CUlibrary library, const char *name) {
    return cuModuleGetGlobal(dptr, bytes, (CUmodule)library, name);
}

CUresult CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                                     CUlibrary library, const char *name) {
    (void)dptr; (void)bytes; (void)library; (void)name;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library,
                                             const char *symbol) {
    (void)fptr; (void)library; (void)symbol;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLibraryGetKernelCount(unsigned int *count,
                                         CUlibrary lib) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    CUmodule mod = (CUmodule)lib;
    *count = mod ? mod->function_count : 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryEnumerateKernels(CUkernel *kernels,
                                           unsigned int numKernels,
                                           CUlibrary lib) {
    if (!kernels) return CUDA_ERROR_INVALID_VALUE;
    CUmodule mod = (CUmodule)lib;
    if (!mod) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < numKernels && i < mod->function_count; i++)
        kernels[i] = (CUkernel)&mod->functions[i];
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                                      CUkernel kernel, CUdevice dev) {
    (void)dev;
    return cuFuncGetAttribute(pi, attrib, (CUfunction)kernel);
}

CUresult CUDAAPI cuKernelSetAttribute(CUfunction_attribute attrib, int val,
                                      CUkernel kernel, CUdevice dev) {
    (void)attrib; (void)val; (void)kernel; (void)dev;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config,
                                        CUdevice dev) {
    (void)kernel; (void)config; (void)dev;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetName(const char **name, CUkernel kernel) {
    if (!name || !kernel) return CUDA_ERROR_INVALID_VALUE;
    *name = ((CUfunction)kernel)->name;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex,
                                      size_t *paramOffset, size_t *paramSize) {
    (void)kernel; (void)paramIndex;
    (void)paramOffset; (void)paramSize;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Launch extras
 * ============================================================================ */

CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f,
                                           unsigned int gridDimX,
                                           unsigned int gridDimY,
                                           unsigned int gridDimZ,
                                           unsigned int blockDimX,
                                           unsigned int blockDimY,
                                           unsigned int blockDimZ,
                                           unsigned int sharedMemBytes,
                                           CUstream hStream,
                                           void **kernelParams) {
    (void)f; (void)gridDimX; (void)gridDimY; (void)gridDimZ;
    (void)blockDimX; (void)blockDimY; (void)blockDimZ;
    (void)sharedMemBytes; (void)hStream; (void)kernelParams;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                  void *userData) {
    (void)hStream; (void)fn; (void)userData;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Stream extras
 * ============================================================================ */

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
    (void)hStream; (void)hEvent; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback,
                                     void *userData, unsigned int flags) {
    (void)hStream; (void)callback; (void)userData; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
    (void)hStream; (void)dptr; (void)length; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
    (void)hStream;
    if (!priority) return CUDA_ERROR_INVALID_VALUE;
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream;
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    (void)hStream;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = g_cuvk.current_ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device) {
    (void)hStream;
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Event extras
 * ============================================================================ */

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    (void)hEvent;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                        unsigned int flags) {
    (void)hEvent; (void)hStream; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Texture/surface objects
 * ============================================================================ */

CUresult CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject,
                                   const CUDA_RESOURCE_DESC *pResDesc,
                                   const CUDA_TEXTURE_DESC *pTexDesc,
                                   const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
    (void)pTexObject; (void)pResDesc; (void)pTexDesc; (void)pResViewDesc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
    (void)texObject;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
    (void)pResDesc; (void)texObject;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
    (void)pTexDesc; (void)texObject;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
    (void)pResViewDesc; (void)texObject;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
    (void)pSurfObject; (void)pResDesc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
    (void)surfObject;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Deprecated texture reference API
 * ============================================================================ */

/* cuTexRefSetAddress: macro maps to cuTexRefSetAddress_v2 */
CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
    (void)ByteOffset; (void)hTexRef; (void)dptr; (void)bytes;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuTexRefSetAddress2D: macro maps to cuTexRefSetAddress2D_v3 */
CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR *desc,
                                      CUdeviceptr dptr, size_t Pitch) {
    (void)hTexRef; (void)desc; (void)dptr; (void)Pitch;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
    (void)hTexRef; (void)fmt; (void)NumPackedComponents;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    (void)hTexRef; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
    (void)hTexRef; (void)hArray; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
    (void)hTexRef; (void)dim; (void)am;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    (void)hTexRef; (void)fm;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuTexRefGetAddress: macro maps to cuTexRefGetAddress_v2 */
CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
    (void)pdptr; (void)hTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    (void)phArray; (void)hTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
    (void)pFormat; (void)pNumChannels; (void)hTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
    (void)pFlags; (void)hTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
    (void)pTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
    (void)hTexRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* Deprecated surface reference API */
CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
    (void)hSurfRef; (void)hArray; (void)Flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
    (void)phArray; (void)hSurfRef;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Array management
 * ============================================================================ */

/* cuArrayCreate: macro maps to cuArrayCreate_v2 */
CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    (void)pHandle; (void)pAllocateArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
    (void)hArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuArrayGetDescriptor: macro maps to cuArrayGetDescriptor_v2 */
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
    (void)pArrayDescriptor; (void)hArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuArray3DCreate: macro maps to cuArray3DCreate_v2 */
CUresult CUDAAPI cuArray3DCreate(CUarray *pHandle,
                                 const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    (void)pHandle; (void)pAllocateArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuArray3DGetDescriptor: macro maps to cuArray3DGetDescriptor_v2 */
CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    (void)pArrayDescriptor; (void)hArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
    (void)pHandle; (void)pMipmappedArrayDesc; (void)numMipmapLevels;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
    (void)pLevelArray; (void)hMipmappedArray; (void)level;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    (void)hMipmappedArray;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Occupancy
 * ============================================================================ */

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize,
    size_t dynamicSMemSize) {
    (void)numBlocks; (void)func; (void)blockSize; (void)dynamicSMemSize;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit) {
    (void)minGridSize; (void)blockSize; (void)func;
    (void)blockSizeToDynamicSMemSize; (void)dynamicSMemSize;
    (void)blockSizeLimit;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize,
    size_t dynamicSMemSize, unsigned int flags) {
    (void)numBlocks; (void)func; (void)blockSize;
    (void)dynamicSMemSize; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
    (void)minGridSize; (void)blockSize; (void)func;
    (void)blockSizeToDynamicSMemSize; (void)dynamicSMemSize;
    (void)blockSizeLimit; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(
    size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
    (void)dynamicSmemSize; (void)func; (void)numBlocks; (void)blockSize;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Graph API (basic stubs)
 * ============================================================================ */

CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    (void)phGraph; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) {
    (void)hGraph;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
    (void)hGraphExec;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuGraphInstantiate: macro maps to cuGraphInstantiate_v3 */
CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                    unsigned long long flags) {
    (void)phGraphExec; (void)hGraph; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Linker
 * ============================================================================ */

/* cuLinkCreate: macro maps to cuLinkCreate_v2 */
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
    (void)numOptions; (void)options; (void)optionValues; (void)stateOut;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuLinkAddData: macro maps to cuLinkAddData_v2 */
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
    (void)state; (void)type; (void)data; (void)size; (void)name;
    (void)numOptions; (void)options; (void)optionValues;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuLinkAddFile: macro maps to cuLinkAddFile_v2 */
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
    (void)state; (void)type; (void)path;
    (void)numOptions; (void)options; (void)optionValues;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
    (void)state; (void)cubinOut; (void)sizeOut;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
    (void)state;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * External memory / semaphore
 * ============================================================================ */

CUresult CUDAAPI cuImportExternalMemory(
    CUexternalMemory *extMem_out,
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    (void)extMem_out; (void)memHandleDesc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory extMem) {
    (void)extMem;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
    (void)extSem_out; (void)semHandleDesc;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    (void)extSem;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    (void)extSemArray; (void)paramsArray; (void)numExtSems; (void)stream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    (void)extSemArray; (void)paramsArray; (void)numExtSems; (void)stream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Graphics interop
 * ============================================================================ */

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    (void)resource;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
    (void)count; (void)resources; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
    (void)count; (void)resources; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuGraphicsResourceGetMappedPointer: macro maps to _v2 */
CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr,
                                                    size_t *pSize,
                                                    CUgraphicsResource resource) {
    (void)pDevPtr; (void)pSize; (void)resource;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuGraphicsResourceSetMapFlags: macro maps to _v2 */
CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
    (void)resource; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Misc
 * ============================================================================ */

/* Forward declarations for functions in other translation units */
CUresult CUDAAPI cuInit(unsigned int Flags);
CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetCount(int *count);
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev);
CUresult CUDAAPI cuCtxCreate_v4(CUcontext *pctx,
                                CUctxCreateParams *ctxCreateParams,
                                unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr);
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n);
CUresult CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n);
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n);
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src,
                                       size_t n, CUstream s);
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src,
                                       size_t n, CUstream s);
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dst, unsigned char uc, size_t n);
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dst, unsigned short us, size_t n);
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dst, unsigned int ui, size_t n);
CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                     unsigned int n, CUjit_option *o,
                                     void **v);
CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                      const char *name);
CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                     CUfunction hfunc);
CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gx, unsigned int gy,
                                unsigned int gz, unsigned int bx,
                                unsigned int by, unsigned int bz,
                                unsigned int smem, CUstream s,
                                void **params, void **extra);
CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *s, unsigned int f,
                                             int p);
CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamQuery(CUstream hStream);
CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuEventElapsedTime_v2(float *ms, CUevent start, CUevent end);

typedef struct {
    const char *name;
    void       *fn;
} CuvkProcEntry;

static const CuvkProcEntry g_proc_table[] = {
    {"cuInit",                        (void *)cuInit},
    {"cuDriverGetVersion",            (void *)cuDriverGetVersion},
    {"cuDeviceGet",                   (void *)cuDeviceGet},
    {"cuDeviceGetCount",              (void *)cuDeviceGetCount},
    {"cuDeviceGetName",               (void *)cuDeviceGetName},
    {"cuDeviceTotalMem",              (void *)cuDeviceTotalMem_v2},
    {"cuDeviceTotalMem_v2",           (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetAttribute",          (void *)cuDeviceGetAttribute},
    {"cuDeviceGetUuid",               (void *)cuDeviceGetUuid},
    {"cuDeviceGetUuid_v2",            (void *)cuDeviceGetUuid},
    {"cuCtxCreate",                   (void *)cuCtxCreate_v4},
    {"cuCtxCreate_v2",                (void *)cuCtxCreate_v4},
    {"cuCtxCreate_v3",                (void *)cuCtxCreate_v4},
    {"cuCtxCreate_v4",                (void *)cuCtxCreate_v4},
    {"cuCtxDestroy",                  (void *)cuCtxDestroy_v2},
    {"cuCtxDestroy_v2",               (void *)cuCtxDestroy_v2},
    {"cuCtxGetCurrent",               (void *)cuCtxGetCurrent},
    {"cuCtxSetCurrent",               (void *)cuCtxSetCurrent},
    {"cuCtxSynchronize",              (void *)cuCtxSynchronize},
    {"cuCtxPushCurrent",              (void *)cuCtxPushCurrent},
    {"cuCtxPushCurrent_v2",           (void *)cuCtxPushCurrent},
    {"cuCtxPopCurrent",               (void *)cuCtxPopCurrent},
    {"cuCtxPopCurrent_v2",            (void *)cuCtxPopCurrent},
    {"cuCtxGetDevice",                (void *)cuCtxGetDevice},
    {"cuCtxGetFlags",                 (void *)cuCtxGetFlags},
    {"cuCtxGetApiVersion",            (void *)cuCtxGetApiVersion},
    {"cuCtxSetLimit",                 (void *)cuCtxSetLimit},
    {"cuCtxGetLimit",                 (void *)cuCtxGetLimit},
    {"cuCtxGetCacheConfig",           (void *)cuCtxGetCacheConfig},
    {"cuCtxSetCacheConfig",           (void *)cuCtxSetCacheConfig},
    {"cuCtxGetStreamPriorityRange",   (void *)cuCtxGetStreamPriorityRange},
    {"cuDevicePrimaryCtxRetain",      (void *)cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease",     (void *)cuDevicePrimaryCtxRelease},
    {"cuDevicePrimaryCtxRelease_v2",  (void *)cuDevicePrimaryCtxRelease},
    {"cuDevicePrimaryCtxReset",       (void *)cuDevicePrimaryCtxReset},
    {"cuDevicePrimaryCtxReset_v2",    (void *)cuDevicePrimaryCtxReset},
    {"cuDevicePrimaryCtxGetState",    (void *)cuDevicePrimaryCtxGetState},
    {"cuDevicePrimaryCtxSetFlags",    (void *)cuDevicePrimaryCtxSetFlags},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags},
    {"cuMemAlloc",                    (void *)cuMemAlloc_v2},
    {"cuMemAlloc_v2",                 (void *)cuMemAlloc_v2},
    {"cuMemFree",                     (void *)cuMemFree_v2},
    {"cuMemFree_v2",                  (void *)cuMemFree_v2},
    {"cuMemcpyHtoD",                  (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyHtoD_v2",              (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyDtoH",                  (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoH_v2",              (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoD",                  (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoD_v2",              (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyHtoDAsync",             (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyHtoDAsync_v2",         (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyDtoHAsync",             (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemcpyDtoHAsync_v2",         (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemsetD8",                    (void *)cuMemsetD8_v2},
    {"cuMemsetD8_v2",                (void *)cuMemsetD8_v2},
    {"cuMemsetD16",                   (void *)cuMemsetD16_v2},
    {"cuMemsetD16_v2",               (void *)cuMemsetD16_v2},
    {"cuMemsetD32",                   (void *)cuMemsetD32_v2},
    {"cuMemsetD32_v2",               (void *)cuMemsetD32_v2},
    {"cuMemGetInfo",                  (void *)cuMemGetInfo},
    {"cuMemGetInfo_v2",              (void *)cuMemGetInfo},
    {"cuModuleLoad",                  (void *)cuModuleLoad},
    {"cuModuleLoadData",              (void *)cuModuleLoadData},
    {"cuModuleLoadDataEx",            (void *)cuModuleLoadDataEx},
    {"cuModuleLoadFatBinary",         (void *)cuModuleLoadFatBinary},
    {"cuModuleUnload",                (void *)cuModuleUnload},
    {"cuModuleGetFunction",           (void *)cuModuleGetFunction},
    {"cuModuleGetGlobal",             (void *)cuModuleGetGlobal},
    {"cuModuleGetGlobal_v2",         (void *)cuModuleGetGlobal},
    {"cuFuncGetAttribute",            (void *)cuFuncGetAttribute},
    {"cuFuncSetAttribute",            (void *)cuFuncSetAttribute},
    {"cuFuncSetCacheConfig",          (void *)cuFuncSetCacheConfig},
    {"cuFuncGetModule",               (void *)cuFuncGetModule},
    {"cuFuncGetName",                 (void *)cuFuncGetName},
    {"cuFuncIsLoaded",                (void *)cuFuncIsLoaded},
    {"cuFuncLoad",                    (void *)cuFuncLoad},
    {"cuLibraryLoadData",             (void *)cuLibraryLoadData},
    {"cuLibraryLoadFromFile",         (void *)cuLibraryLoadFromFile},
    {"cuLibraryUnload",               (void *)cuLibraryUnload},
    {"cuLibraryGetKernel",            (void *)cuLibraryGetKernel},
    {"cuLibraryGetModule",            (void *)cuLibraryGetModule},
    {"cuKernelGetFunction",           (void *)cuKernelGetFunction},
    {"cuKernelGetLibrary",            (void *)cuKernelGetLibrary},
    {"cuLibraryGetGlobal",            (void *)cuLibraryGetGlobal},
    {"cuLibraryGetManaged",           (void *)cuLibraryGetManaged},
    {"cuLibraryGetUnifiedFunction",   (void *)cuLibraryGetUnifiedFunction},
    {"cuLibraryGetKernelCount",       (void *)cuLibraryGetKernelCount},
    {"cuLibraryEnumerateKernels",     (void *)cuLibraryEnumerateKernels},
    {"cuKernelGetAttribute",          (void *)cuKernelGetAttribute},
    {"cuKernelSetAttribute",          (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig",        (void *)cuKernelSetCacheConfig},
    {"cuKernelGetName",               (void *)cuKernelGetName},
    {"cuKernelGetParamInfo",          (void *)cuKernelGetParamInfo},
    {"cuLinkCreate",                  (void *)cuLinkCreate},
    {"cuLinkCreate_v2",               (void *)cuLinkCreate},
    {"cuLinkAddData",                 (void *)cuLinkAddData},
    {"cuLinkAddData_v2",              (void *)cuLinkAddData},
    {"cuLinkAddFile",                 (void *)cuLinkAddFile},
    {"cuLinkAddFile_v2",              (void *)cuLinkAddFile},
    {"cuLinkComplete",                (void *)cuLinkComplete},
    {"cuLinkDestroy",                 (void *)cuLinkDestroy},
    {"cuIpcGetEventHandle",           (void *)cuIpcGetEventHandle},
    {"cuIpcOpenEventHandle",          (void *)cuIpcOpenEventHandle},
    {"cuIpcGetMemHandle",             (void *)cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle",            (void *)cuIpcOpenMemHandle},
    {"cuIpcOpenMemHandle_v2",         (void *)cuIpcOpenMemHandle},
    {"cuIpcCloseMemHandle",           (void *)cuIpcCloseMemHandle},
    {"cuLaunchKernel",                (void *)cuLaunchKernel},
    {"cuStreamCreate",                (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority",    (void *)cuStreamCreateWithPriority},
    {"cuStreamDestroy",               (void *)cuStreamDestroy_v2},
    {"cuStreamDestroy_v2",           (void *)cuStreamDestroy_v2},
    {"cuStreamSynchronize",           (void *)cuStreamSynchronize},
    {"cuStreamQuery",                 (void *)cuStreamQuery},
    {"cuStreamWaitEvent",             (void *)cuStreamWaitEvent},
    {"cuStreamGetPriority",           (void *)cuStreamGetPriority},
    {"cuStreamGetFlags",              (void *)cuStreamGetFlags},
    {"cuStreamGetCtx",                (void *)cuStreamGetCtx},
    {"cuStreamGetCtx_v2",             (void *)cuStreamGetCtx},
    {"cuStreamGetDevice",             (void *)cuStreamGetDevice},
    {"cuEventCreate",                 (void *)cuEventCreate},
    {"cuEventDestroy",                (void *)cuEventDestroy_v2},
    {"cuEventDestroy_v2",            (void *)cuEventDestroy_v2},
    {"cuEventRecord",                 (void *)cuEventRecord},
    {"cuEventSynchronize",            (void *)cuEventSynchronize},
    {"cuEventQuery",                  (void *)cuEventQuery},
    {"cuEventElapsedTime",            (void *)cuEventElapsedTime_v2},
    {"cuEventElapsedTime_v2",        (void *)cuEventElapsedTime_v2},
    {"cuGetErrorString",              (void *)cuGetErrorString},
    {"cuGetErrorName",                (void *)cuGetErrorName},
    {"cuModuleGetLoadingMode",        (void *)cuModuleGetLoadingMode},
    {"cuGetExportTable",              (void *)cuGetExportTable},
    {"cuGetProcAddress",              (void *)cuGetProcAddress},
    {"cuGetProcAddress_v2",           (void *)cuGetProcAddress},
    {NULL, NULL}
};

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  CUdriverProcAddressQueryResult *symbolStatus) {

    if (!symbol || !pfn)
        return CUDA_ERROR_INVALID_VALUE;

    if (symbol[0] == '\0') {
        *pfn = NULL;
        if (symbolStatus)
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        return CUDA_ERROR_NOT_FOUND;
    }

    for (const CuvkProcEntry *e = g_proc_table; e->name; e++) {
        if (strcmp(e->name, symbol) == 0) {
            if (cudaVersion > 13020) {
                CUVK_LOG("[cuvk] cuGetProcAddress: \"%s\" ver=%d > 13020 -> NOT_FOUND\n",
                        symbol, cudaVersion);
                *pfn = NULL;
                if (symbolStatus)
                    *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
                return CUDA_ERROR_NOT_FOUND;
            }
            *pfn = e->fn;
            if (symbolStatus)
                *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
            CUVK_LOG("[cuvk] cuGetProcAddress: FOUND \"%s\" ver=%d flags=%llu\n",
                    symbol, cudaVersion, (unsigned long long)flags);
            return CUDA_SUCCESS;
        }
    }

    CUVK_LOG("[cuvk] cuGetProcAddress: NOT FOUND \"%s\"\n", symbol);
    *pfn = NULL;
    if (symbolStatus)
        *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_ERROR_NOT_FOUND;
}

/* FatbincWrapper: __fatBinC_Wrapper_t from CUDA runtime internals */
#define FATBINC_MAGIC 0x466243B1u

typedef struct {
    unsigned int magic;
    unsigned int version;
    const void  *data;       /* -> FatbinHeader (magic 0xBA55ED50) */
    const void  *filename_or_fatbins;
} FatbincWrapper;

static CUresult CUDAAPI
cudart_get_module_from_cubin(CUmodule *module,
                             const FatbincWrapper *wrapper) {
    CUVK_LOG("[cuvk] get_module_from_cubin: module=%p wrapper=%p\n",
            (void *)module, (const void *)wrapper);
    fflush(stderr);
    if (!module || !wrapper) return CUDA_ERROR_INVALID_VALUE;
    CUVK_LOG("[cuvk]   wrapper magic=0x%08x ver=%u data=%p\n",
            wrapper->magic, wrapper->version, wrapper->data);
    fflush(stderr);
    if (wrapper->magic != FATBINC_MAGIC || !wrapper->data)
        return CUDA_ERROR_INVALID_IMAGE;
    return cuModuleLoadData(module, wrapper->data);
}

static CUresult CUDAAPI
cudart_interface_fn2(CUcontext *pctx, CUdevice dev) {
    CUVK_LOG("[cuvk] cudart_interface_fn2: pctx=%p dev=%d\n",
            (void *)pctx, dev);
    fflush(stderr);
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    CUresult res = cuDevicePrimaryCtxRetain(pctx, dev);
    CUVK_LOG("[cuvk] cudart_interface_fn2 -> %d (ctx=%p)\n",
            res, (void *)*pctx);
    fflush(stderr);
    return res;
}

static CUresult CUDAAPI
cudart_get_module_from_cubin_ext1(CUmodule *module,
                                  const FatbincWrapper *wrapper,
                                  void *arg3, void *arg4, unsigned int arg5) {
    CUVK_LOG("[cuvk] get_module_from_cubin_ext1: "
            "module=%p wrapper=%p\n", (void *)module, (const void *)wrapper);
    (void)arg3; (void)arg4; (void)arg5;
    return cudart_get_module_from_cubin(module, wrapper);
}

static void cudart_interface_fn7(size_t arg1) {
    CUVK_LOG("[cuvk] [slot7] fn7: arg1=%zu\n", arg1);
    (void)arg1;
}

static CUresult CUDAAPI
cudart_get_module_from_cubin_ext2(const void *fatbin_header,
                                  CUmodule *module,
                                  void *arg3, void *arg4, unsigned int arg5) {
    CUVK_LOG("[cuvk] [slot8] get_module_from_cubin_ext2: "
            "header=%p module=%p arg3=%p arg4=%p arg5=%u\n",
            fatbin_header, (void *)module, arg3, arg4, arg5);
    (void)arg3; (void)arg4; (void)arg5;
    if (!module || !fatbin_header) return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData(module, fatbin_header);
}

static CUresult CUDAAPI cudart_load_compilers(void) {
    CUVK_LOG("[cuvk] [slot12] load_compilers\n");
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Context-local storage (CONTEXT_LOCAL_STORAGE_INTERFACE_V0301)
 * ============================================================================ */

static CUresult CUDAAPI
ctx_local_storage_put(CUcontext cu_ctx, void *key, void *value,
                      void (*dtor_cb)(CUcontext, void *, void *)) {
    CUVK_LOG("[cuvk] ctx_storage_put: ctx=%p key=%p value=%p dtor=%p\n",
            (void *)cu_ctx, key, value, (void *)(uintptr_t)dtor_cb);
    fflush(stderr);
    struct CUctx_st *ctx = cu_ctx ? (struct CUctx_st *)cu_ctx : g_cuvk.current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    uintptr_t k = (uintptr_t)key;
    for (uint32_t i = 0; i < ctx->storage_count; i++) {
        if (ctx->storage[i].key == k) {
            ctx->storage[i].value = value;
            ctx->storage[i].dtor_cb = dtor_cb;
            return CUDA_SUCCESS;
        }
    }
    if (ctx->storage_count >= ctx->storage_capacity) {
        uint32_t cap = ctx->storage_capacity ? ctx->storage_capacity * 2 : 8;
        CuvkStorageEntry *s = (CuvkStorageEntry *)realloc(
            ctx->storage, cap * sizeof(CuvkStorageEntry));
        if (!s) return CUDA_ERROR_OUT_OF_MEMORY;
        ctx->storage = s;
        ctx->storage_capacity = cap;
    }
    ctx->storage[ctx->storage_count].key = k;
    ctx->storage[ctx->storage_count].value = value;
    ctx->storage[ctx->storage_count].dtor_cb = dtor_cb;
    ctx->storage_count++;
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI
ctx_local_storage_delete(CUcontext cu_ctx, void *key) {
    struct CUctx_st *ctx = cu_ctx ? (struct CUctx_st *)cu_ctx : g_cuvk.current_ctx;
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    uintptr_t k = (uintptr_t)key;
    for (uint32_t i = 0; i < ctx->storage_count; i++) {
        if (ctx->storage[i].key == k) {
            ctx->storage[i] = ctx->storage[--ctx->storage_count];
            return CUDA_SUCCESS;
        }
    }
    return CUDA_SUCCESS;
}

static CUresult CUDAAPI
ctx_local_storage_get(void **value_out, CUcontext cu_ctx, void *key) {
    CUVK_LOG("[cuvk] ctx_storage_get: value_out=%p ctx=%p key=%p\n",
            (void *)value_out, (void *)cu_ctx, key);
    fflush(stderr);
    struct CUctx_st *ctx = cu_ctx ? (struct CUctx_st *)cu_ctx : g_cuvk.current_ctx;
    if (!ctx) {
        CUVK_LOG("[cuvk]   -> INVALID_CONTEXT\n");
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    uintptr_t k = (uintptr_t)key;
    for (uint32_t i = 0; i < ctx->storage_count; i++) {
        if (ctx->storage[i].key == k) {
            if (value_out) *value_out = ctx->storage[i].value;
            CUVK_LOG("[cuvk]   -> FOUND value=%p\n", ctx->storage[i].value);
            return CUDA_SUCCESS;
        }
    }
    CUVK_LOG("[cuvk]   -> NOT_FOUND (count=%u)\n", ctx->storage_count);
    return CUDA_ERROR_INVALID_HANDLE;
}

static const void *g_ctx_storage_table[4] = {
    (const void *)ctx_local_storage_put,
    (const void *)ctx_local_storage_delete,
    (const void *)ctx_local_storage_get,
    NULL,
};

static CUresult generic_noop_fn(void) { return CUDA_SUCCESS; }

static uint32_t g_tools_buffer1[1024] = {0};
static uint32_t g_tools_buffer2[14] = {0};

static void tools_get_buffer1(void **out_ptr, size_t *out_size) {
    *out_ptr = (void *)g_tools_buffer1;
    *out_size = 1024;
}

static void tools_get_buffer2(void **out_ptr, size_t *out_size) {
    *out_ptr = (void *)g_tools_buffer2;
    *out_size = 14;
}

/* ============================================================================
 * Integrity check (HMAC-MD2) - ported from ZLUDA dark_api/src/lib.rs
 * ============================================================================ */

#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <time.h>

static const uint8_t MIXING_TABLE[256] = {
    0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01,
    0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06, 0x13,
    0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C,
    0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA,
    0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16,
    0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12,
    0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49,
    0xA0, 0xFB, 0xF5, 0x8E, 0xBB, 0x2F, 0xEE, 0x7A,
    0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F,
    0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21,
    0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27,
    0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03,
    0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1,
    0xD7, 0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6,
    0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6,
    0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1,
    0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20,
    0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02,
    0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6,
    0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F,
    0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A,
    0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26,
    0x2C, 0x53, 0x0D, 0x6E, 0x85, 0x28, 0x84, 0x09,
    0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52,
    0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA,
    0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A,
    0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D,
    0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39,
    0xF2, 0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4,
    0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
    0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A,
    0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83, 0x14,
};

static void integrity_single_pass(uint8_t *arg1, uint8_t arg2) {
    uint8_t temp1 = arg1[0x40];
    uint8_t idx = temp1;
    arg1[idx + 0x10] = arg2;
    uint8_t temp3 = (temp1 + 1) & 0x0f;
    arg1[idx + 0x20] = arg1[idx] ^ arg2;
    uint8_t temp4 = MIXING_TABLE[(arg2 ^ arg1[0x41]) & 0xff];
    uint8_t old_checksum = arg1[idx + 0x30];
    arg1[idx + 0x30] = temp4 ^ old_checksum;
    arg1[0x41] = temp4 ^ old_checksum;
    arg1[0x40] = temp3;
    if (temp3 != 0)
        return;
    uint8_t t1 = 0x29;
    uint8_t t5 = 0x00;
    for (;;) {
        t1 = t1 ^ arg1[0];
        arg1[0] = t1;
        for (int i = 1; i < 0x30; i++) {
            t1 = arg1[i] ^ MIXING_TABLE[t1];
            arg1[i] = t1;
        }
        t1 = (uint8_t)(t1 + t5);
        t5 = (uint8_t)(t5 + 1);
        if (t5 == 0x12)
            break;
        t1 = MIXING_TABLE[t1];
    }
}

static void integrity_hash_pass(uint8_t *acc, const uint8_t *data,
                                size_t len, uint8_t xor_mask) {
    for (size_t i = 0; i < len; i++)
        integrity_single_pass(acc, data[i] ^ xor_mask);
}

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_check_table;
    const void *fn_address;
    uint64_t unix_seconds;
} IntegrityPass3Input;

typedef struct {
    CUuuid guid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} DeviceHashInfo;

static void integrity_pass5(uint8_t *result, uint64_t *out) {
    uint8_t temp = (uint8_t)(16u - result[64]);
    for (uint8_t i = 0; i < temp; i++)
        integrity_single_pass(result, temp);
    for (int i = 0x30; i < 0x40; i++)
        integrity_single_pass(result, result[i]);
    memcpy(&out[0], &result[0], 8);
    memcpy(&out[1], &result[8], 8);
}

static void integrity_zero(uint8_t *result) {
    memset(result, 0, 16);
    memset(result + 48, 0, 18);
}

static const void *g_cudart_interface_table[13];

static void integrity_compute(uint32_t version, uint64_t unix_seconds,
                              const void *integrity_table,
                              const void *cudart_table,
                              const void *fn_address,
                              uint64_t *out) {
    if ((version % 10) == 0) {
        out[0] = 0x3341181C03CB675CULL;
        out[1] = 0x8ED383AA1F4CD1E8ULL;
        return;
    }
    if ((version % 10) == 1) {
        out[0] = 0x1841181C03CB675CULL;
        out[1] = 0x8ED383AA1F4CD1E8ULL;
        return;
    }

    static const uint8_t pass1_result[16] = {
        0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
        0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F,
    };

    uint8_t acc[66];
    memset(acc, 0, sizeof(acc));

    integrity_hash_pass(acc, pass1_result, 16, 0x36);

    int driver_ver_int = 0;
    cuDriverGetVersion(&driver_ver_int);
    IntegrityPass3Input p3 = {
        .driver_version = (uint32_t)driver_ver_int,
        .version = version,
        .current_process = (uint32_t)getpid(),
        .current_thread = (uint32_t)(uintptr_t)pthread_self(),
        .cudart_table = cudart_table,
        .integrity_check_table = integrity_table,
        .fn_address = fn_address,
        .unix_seconds = unix_seconds,
    };
    integrity_hash_pass(acc, (const uint8_t *)&p3, sizeof(p3), 0);

    uint32_t dev_count = g_cuvk.physical_device_count;
    if (dev_count == 0) dev_count = 1;
    for (uint32_t d = 0; d < dev_count; d++) {
        DeviceHashInfo info;
        memset(&info, 0, sizeof(info));
        cuDeviceGetUuid(&info.guid, (CUdevice)d);
        int pci_val = 0;
        cuDeviceGetAttribute(&pci_val, 32, (CUdevice)d);
        info.pci_domain = (int32_t)pci_val;
        cuDeviceGetAttribute(&pci_val, 33, (CUdevice)d);
        info.pci_bus = (int32_t)pci_val;
        cuDeviceGetAttribute(&pci_val, 34, (CUdevice)d);
        info.pci_device = (int32_t)pci_val;
        integrity_hash_pass(acc, (const uint8_t *)&info, sizeof(info), 0);
    }

    uint64_t pass5_1[2];
    integrity_pass5(acc, pass5_1);

    integrity_zero(acc);

    integrity_hash_pass(acc, pass1_result, 16, 0x5c);

    integrity_hash_pass(acc, (const uint8_t *)pass5_1, 16, 0);

    integrity_pass5(acc, out);
}

static const void *g_integrity_table[3];

static CUresult CUDAAPI
integrity_check_fn(uint32_t version, uint64_t unix_seconds,
                   uint64_t *result) {
    CUVK_LOG("[cuvk] integrity_check: version=%u ts=%llu\n",
            version, (unsigned long long)unix_seconds);
    if (!result)
        return CUDA_ERROR_INVALID_VALUE;
    integrity_compute(version, unix_seconds,
                      g_integrity_table, g_cudart_interface_table,
                      (const void *)integrity_check_fn, result);
    CUVK_LOG("[cuvk] integrity_check: result=[0x%016llx, 0x%016llx]\n",
            (unsigned long long)result[0], (unsigned long long)result[1]);
    return CUDA_SUCCESS;
}

static const void *g_integrity_table[3] = {
    (const void *)(3 * sizeof(void *)),
    (const void *)integrity_check_fn,
    NULL,
};

static const void *g_tools_tls_table[4] = {
    (const void *)(4 * sizeof(void *)),
    NULL, NULL, NULL,
};

static CUresult CUDAAPI
context_check_fn(CUcontext ctx, uint32_t *result1, const void **result2) {
    (void)ctx; (void)result2;
    if (result1) *result1 = 0;
    return CUDA_SUCCESS;
}

static uint32_t check_fn3(void) { return 0; }

static const void *g_context_checks_table[4] = {
    (const void *)(4 * sizeof(void *)),
    NULL,
    (const void *)context_check_fn,
    (const void *)check_fn3,
};

static const void *g_unknown_table[16] = {
    (const void *)(16 * sizeof(void *)),
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn, (const void *)generic_noop_fn,
    (const void *)generic_noop_fn,
};

static const void *g_tools_runtime_table[7] = {
    (const void *)(7 * sizeof(void *)),
    NULL,
    (const void *)tools_get_buffer1,
    NULL,
    NULL,
    NULL,
    (const void *)tools_get_buffer2,
};

static const void *g_cudart_interface_table[13] = {
    /* [0] SIZE_OF */ (const void *)(13 * sizeof(void *)),
    /* [1] get_module_from_cubin */ (const void *)cudart_get_module_from_cubin,
    /* [2] cudart_interface_fn2  */ (const void *)cudart_interface_fn2,
    /* [3] NULL */ NULL,
    /* [4] NULL */ NULL,
    /* [5] NULL */ NULL,
    /* [6] get_module_from_cubin_ext1 */ (const void *)cudart_get_module_from_cubin_ext1,
    /* [7] cudart_interface_fn7 */ (const void *)cudart_interface_fn7,
    /* [8] get_module_from_cubin_ext2 */ (const void *)cudart_get_module_from_cubin_ext2,
    /* [9]  NULL */ NULL,
    /* [10] NULL */ NULL,
    /* [11] NULL */ NULL,
    /* [12] load_compilers */ (const void *)cudart_load_compilers,
};

static const unsigned char CUDART_INTERFACE_UUID[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9
};

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
    if (!ppExportTable || !pExportTableId) return CUDA_ERROR_INVALID_VALUE;
    const unsigned char *b = (const unsigned char *)pExportTableId->bytes;
    CUVK_LOG("[cuvk] cuGetExportTable: uuid="
            "%02x%02x%02x%02x-%02x%02x-%02x%02x-"
            "%02x%02x-%02x%02x%02x%02x%02x%02x\n",
            b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7],
            b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]);
    if (memcmp(pExportTableId->bytes, CUDART_INTERFACE_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> CUDART_INTERFACE\n");
        *ppExportTable = g_cudart_interface_table;
        return CUDA_SUCCESS;
    }
    /* The f8cff951 UUID shortcuts init if SUCCESS - must return NOT_FOUND */
    static const unsigned char SKIP_INIT_UUID[16] = {
        0xf8, 0xcf, 0xf9, 0x51, 0x21, 0x46, 0x8b, 0x4e,
        0xb9, 0xe2, 0xfb, 0x46, 0x9e, 0x7c, 0x0d, 0xd9
    };
    if (memcmp(pExportTableId->bytes, SKIP_INIT_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> NOT_FOUND (skip-init table)\n");
        *ppExportTable = NULL;
        return CUDA_ERROR_NOT_FOUND;
    }
    static const unsigned char TOOLS_RUNTIME_UUID[16] = {
        0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
        0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66
    };
    if (memcmp(pExportTableId->bytes, TOOLS_RUNTIME_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> TOOLS_RUNTIME\n");
        *ppExportTable = g_tools_runtime_table;
        return CUDA_SUCCESS;
    }
    static const unsigned char INTEGRITY_UUID[16] = {
        0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
        0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2
    };
    if (memcmp(pExportTableId->bytes, INTEGRITY_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> INTEGRITY_CHECK\n");
        *ppExportTable = g_integrity_table;
        return CUDA_SUCCESS;
    }
    static const unsigned char CTX_STORAGE_UUID[16] = {
        0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
        0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93
    };
    if (memcmp(pExportTableId->bytes, CTX_STORAGE_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> CONTEXT_LOCAL_STORAGE\n");
        *ppExportTable = g_ctx_storage_table;
        return CUDA_SUCCESS;
    }
    static const unsigned char TOOLS_TLS_UUID[16] = {
        0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
        0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc
    };
    if (memcmp(pExportTableId->bytes, TOOLS_TLS_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> TOOLS_TLS\n");
        *ppExportTable = g_tools_tls_table;
        return CUDA_SUCCESS;
    }
    static const unsigned char CONTEXT_CHECKS_UUID[16] = {
        0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
        0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e
    };
    if (memcmp(pExportTableId->bytes, CONTEXT_CHECKS_UUID, 16) == 0) {
        CUVK_LOG("[cuvk]   -> CONTEXT_CHECKS\n");
        *ppExportTable = g_context_checks_table;
        return CUDA_SUCCESS;
    }
    /* Other interfaces: return no-op table */
    CUVK_LOG("[cuvk]   -> generic noop table\n");
    *ppExportTable = g_unknown_table;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Memory pool stubs
 * ============================================================================ */

CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *pool,
                                 const CUmemPoolProps *poolProps) {
    (void)pool; (void)poolProps;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) {
    (void)pool;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    (void)pool; (void)minBytesToKeep;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr,
                                       void *value) {
    (void)pool; (void)attr; (void)value;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr,
                                       void *value) {
    (void)pool; (void)attr; (void)value;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                         CUmemoryPool pool,
                                         CUstream hStream) {
    (void)dptr; (void)bytesize; (void)pool; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Virtual memory management stubs
 * ============================================================================ */

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                                     size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags) {
    (void)ptr; (void)size; (void)alignment; (void)addr; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                             const CUmemAllocationProp *prop,
                             unsigned long long flags) {
    (void)handle; (void)size; (void)prop; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    (void)handle;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
    (void)ptr; (void)size; (void)offset; (void)handle; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc *desc, size_t count) {
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags,
                                const CUmemLocation *location,
                                CUdeviceptr ptr) {
    (void)flags; (void)location; (void)ptr;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags option) {
    (void)granularity; (void)prop; (void)option;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
    (void)prop; (void)handle;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Memory range attributes
 * ============================================================================ */

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
    (void)data; (void)dataSize; (void)attribute; (void)devPtr; (void)count;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
    (void)data; (void)dataSizes; (void)attributes; (void)numAttributes;
    (void)devPtr; (void)count;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ============================================================================
 * Stream capture
 * ============================================================================ */

CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream,
                                      CUstreamCaptureMode mode) {
    (void)hStream; (void)mode;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    (void)hStream; (void)phGraph;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream,
                                     CUstreamCaptureStatus *captureStatus) {
    (void)hStream; (void)captureStatus;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(
    CUstreamCaptureMode *mode) {
    (void)mode;
    return CUDA_ERROR_NOT_SUPPORTED;
}
