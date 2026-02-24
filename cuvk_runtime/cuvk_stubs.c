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
    (void)uuid; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                                 CUdevice dev) {
    (void)luid; (void)deviceNodeMask; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuDevicePrimaryCtxRetain: no versioned macro */
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)pctx; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuDevicePrimaryCtxRelease: macro maps to cuDevicePrimaryCtxRelease_v2 */
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuDevicePrimaryCtxReset: macro maps to cuDevicePrimaryCtxReset_v2 */
CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
    (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuDevicePrimaryCtxGetState: no versioned macro */
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
    (void)dev; (void)flags; (void)active;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuDevicePrimaryCtxSetFlags: macro maps to cuDevicePrimaryCtxSetFlags_v2 */
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    (void)dev; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
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
    (void)ctx;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* cuCtxPopCurrent: macro maps to cuCtxPopCurrent_v2 */
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
    (void)pctx;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    (void)device;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx; (void)version;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
    (void)limit; (void)value;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    (void)pvalue; (void)limit;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
    (void)pconfig;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
    (void)config;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
    (void)leastPriority; (void)greatestPriority;
    return CUDA_ERROR_NOT_SUPPORTED;
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
    (void)free; (void)total;
    return CUDA_ERROR_NOT_SUPPORTED;
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
    (void)module; (void)fatCubin;
    return CUDA_ERROR_NOT_SUPPORTED;
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
    (void)mode;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
    (void)hfunc; (void)attrib; (void)value;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc,
                                      CUfunc_cache config) {
    (void)hfunc; (void)config;
    return CUDA_ERROR_NOT_SUPPORTED;
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
    (void)state; (void)function;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuFuncLoad(CUfunction function) {
    (void)function;
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
    (void)hStream; (void)priority;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    (void)hStream; (void)pctx;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device) {
    (void)hStream; (void)device;
    return CUDA_ERROR_NOT_SUPPORTED;
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

/* cuGetProcAddress: macro maps to cuGetProcAddress_v2 */
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  CUdriverProcAddressQueryResult *symbolStatus) {
    (void)symbol; (void)pfn; (void)cudaVersion; (void)flags;
    (void)symbolStatus;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
    (void)ppExportTable; (void)pExportTableId;
    return CUDA_ERROR_NOT_SUPPORTED;
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
