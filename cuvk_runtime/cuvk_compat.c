/*
 * cuvk_compat.c - Unversioned symbol wrappers for CUDA driver API.
 *
 * cuda.h renames functions via macros (e.g. cuMemAlloc -> cuMemAlloc_v2),
 * so our implementations only export the _v2 names. But cuFFT (and other
 * CUDA libraries) resolve functions via dlsym with the unversioned names.
 * This file creates thin wrappers plus stubs for missing functions.
 */

#include "cuda.h"
#include <stddef.h>

/* Macro to declare an extern _v2 function and define a wrapper. */
#define COMPAT_WRAP0(name, ret) \
    extern ret name##_v2(void); \
    ret name(void) { return name##_v2(); }

#define COMPAT_WRAP1(name, ret, t1) \
    extern ret name##_v2(t1); \
    ret name(t1 a1) { return name##_v2(a1); }

#define COMPAT_WRAP2(name, ret, t1, t2) \
    extern ret name##_v2(t1, t2); \
    ret name(t1 a1, t2 a2) { return name##_v2(a1, a2); }

#define COMPAT_WRAP3(name, ret, t1, t2, t3) \
    extern ret name##_v2(t1, t2, t3); \
    ret name(t1 a1, t2 a2, t3 a3) { return name##_v2(a1, a2, a3); }

#define COMPAT_WRAP4(name, ret, t1, t2, t3, t4) \
    extern ret name##_v2(t1, t2, t3, t4); \
    ret name(t1 a1, t2 a2, t3 a3, t4 a4) { return name##_v2(a1, a2, a3, a4); }

#define COMPAT_WRAP5(name, ret, t1, t2, t3, t4, t5) \
    extern ret name##_v2(t1, t2, t3, t4, t5); \
    ret name(t1 a1, t2 a2, t3 a3, t4 a4, t5 a5) { \
        return name##_v2(a1, a2, a3, a4, a5); \
    }

/*
 * First #undef each macro from cuda.h, then provide the wrapper.
 * The wrapper calls the real _v2 function that lives in another TU.
 */

#undef cuDeviceTotalMem
COMPAT_WRAP2(cuDeviceTotalMem, CUresult, size_t *, CUdevice)

#undef cuModuleGetGlobal
extern CUresult cuModuleGetGlobal_v2(CUdeviceptr *, size_t *, CUmodule,
                                     const char *);
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

#undef cuMemGetInfo
COMPAT_WRAP2(cuMemGetInfo, CUresult, size_t *, size_t *)

#undef cuMemAlloc
COMPAT_WRAP2(cuMemAlloc, CUresult, CUdeviceptr *, size_t)

#undef cuMemFree
COMPAT_WRAP1(cuMemFree, CUresult, CUdeviceptr)

#undef cuCtxDestroy
COMPAT_WRAP1(cuCtxDestroy, CUresult, CUcontext)

#undef cuCtxPopCurrent
COMPAT_WRAP1(cuCtxPopCurrent, CUresult, CUcontext *)

#undef cuCtxPushCurrent
COMPAT_WRAP1(cuCtxPushCurrent, CUresult, CUcontext)

#undef cuStreamDestroy
COMPAT_WRAP1(cuStreamDestroy, CUresult, CUstream)

#undef cuEventDestroy
COMPAT_WRAP1(cuEventDestroy, CUresult, CUevent)

#undef cuDevicePrimaryCtxRelease
COMPAT_WRAP1(cuDevicePrimaryCtxRelease, CUresult, CUdevice)

#undef cuDevicePrimaryCtxReset
COMPAT_WRAP1(cuDevicePrimaryCtxReset, CUresult, CUdevice)

#undef cuDevicePrimaryCtxSetFlags
COMPAT_WRAP2(cuDevicePrimaryCtxSetFlags, CUresult, CUdevice, unsigned int)

#undef cuDeviceGetUuid
COMPAT_WRAP2(cuDeviceGetUuid, CUresult, CUuuid *, CUdevice)

#undef cuEventElapsedTime
COMPAT_WRAP3(cuEventElapsedTime, CUresult, float *, CUevent, CUevent)

#undef cuMemAdvise
COMPAT_WRAP4(cuMemAdvise, CUresult, CUdeviceptr, size_t, CUmem_advise,
             CUmemLocation)

#undef cuMemHostGetDevicePointer
COMPAT_WRAP3(cuMemHostGetDevicePointer, CUresult, CUdeviceptr *, void *,
             unsigned int)

#undef cuMemcpyHtoD
COMPAT_WRAP3(cuMemcpyHtoD, CUresult, CUdeviceptr, const void *, size_t)

#undef cuMemcpyDtoH
COMPAT_WRAP3(cuMemcpyDtoH, CUresult, void *, CUdeviceptr, size_t)

#undef cuMemcpyDtoD
COMPAT_WRAP3(cuMemcpyDtoD, CUresult, CUdeviceptr, CUdeviceptr, size_t)

#undef cuMemcpyHtoDAsync
COMPAT_WRAP4(cuMemcpyHtoDAsync, CUresult, CUdeviceptr, const void *, size_t,
             CUstream)

#undef cuMemcpyDtoHAsync
COMPAT_WRAP4(cuMemcpyDtoHAsync, CUresult, void *, CUdeviceptr, size_t,
             CUstream)

#undef cuMemPrefetchAsync
COMPAT_WRAP5(cuMemPrefetchAsync, CUresult, CUdeviceptr, size_t, CUmemLocation,
             unsigned int, CUstream)

#undef cuMemsetD8
COMPAT_WRAP3(cuMemsetD8, CUresult, CUdeviceptr, unsigned char, size_t)

#undef cuGraphicsResourceGetMappedPointer
COMPAT_WRAP3(cuGraphicsResourceGetMappedPointer, CUresult, CUdeviceptr *,
             size_t *, CUgraphicsResource)

#undef cuGraphicsResourceSetMapFlags
COMPAT_WRAP2(cuGraphicsResourceSetMapFlags, CUresult, CUgraphicsResource,
             unsigned int)

/* Graph API wrappers */
#undef cuGraphAddKernelNode
extern CUresult cuGraphAddKernelNode_v2(CUgraphNode *, CUgraph,
                                        const CUgraphNode *, size_t,
                                        const CUDA_KERNEL_NODE_PARAMS *);
CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                              const CUgraphNode *dependencies,
                              size_t numDependencies,
                              const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    return cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies,
                                  numDependencies, nodeParams);
}

#undef cuGraphKernelNodeGetParams
COMPAT_WRAP2(cuGraphKernelNodeGetParams, CUresult, CUgraphNode,
             CUDA_KERNEL_NODE_PARAMS *)

#undef cuGraphKernelNodeSetParams
COMPAT_WRAP2(cuGraphKernelNodeSetParams, CUresult, CUgraphNode,
             const CUDA_KERNEL_NODE_PARAMS *)

#undef cuGraphExecKernelNodeSetParams
COMPAT_WRAP3(cuGraphExecKernelNodeSetParams, CUresult, CUgraphExec,
             CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *)

#undef cuGraphGetEdges
extern CUresult cuGraphGetEdges_v2(CUgraph, CUgraphNode *, CUgraphNode *,
                                   CUgraphEdgeData *, size_t *);
CUresult cuGraphGetEdges(CUgraphNode *from, CUgraphNode *to,
                         CUgraphEdgeData *edgeData, size_t *numEdges) {
    (void)from; (void)to; (void)edgeData; (void)numEdges;
    return CUDA_ERROR_NOT_SUPPORTED;
}

#undef cuGraphNodeGetDependencies
extern CUresult cuGraphNodeGetDependencies_v2(CUgraphNode, CUgraphNode *,
                                              CUgraphEdgeData *, size_t *);
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode,
                                    CUgraphNode *dependencies,
                                    CUgraphEdgeData *edgeData,
                                    size_t *numDependencies) {
    return cuGraphNodeGetDependencies_v2(hNode, dependencies, edgeData,
                                        numDependencies);
}

#undef cuGraphNodeGetDependentNodes
extern CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode, CUgraphNode *,
                                                CUgraphEdgeData *, size_t *);
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                      CUgraphNode *dependentNodes,
                                      CUgraphEdgeData *edgeData,
                                      size_t *numDependentNodes) {
    return cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, edgeData,
                                          numDependentNodes);
}

#undef cuGraphAddDependencies
extern CUresult cuGraphAddDependencies_v2(CUgraph, const CUgraphNode *,
                                          const CUgraphNode *,
                                          const CUgraphEdgeData *, size_t);
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                const CUgraphNode *to,
                                const CUgraphEdgeData *edgeData,
                                size_t numDependencies) {
    return cuGraphAddDependencies_v2(hGraph, from, to, edgeData,
                                    numDependencies);
}

#undef cuGraphRemoveDependencies
extern CUresult cuGraphRemoveDependencies_v2(CUgraph, const CUgraphNode *,
                                             const CUgraphNode *,
                                             const CUgraphEdgeData *, size_t);
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from,
                                   const CUgraphNode *to,
                                   const CUgraphEdgeData *edgeData,
                                   size_t numDependencies) {
    return cuGraphRemoveDependencies_v2(hGraph, from, to, edgeData,
                                       numDependencies);
}

/* cuGraphInstantiate (maps to cuGraphInstantiateWithFlags) */
extern CUresult cuGraphInstantiateWithFlags(CUgraphExec *, CUgraph,
                                            unsigned long long);

#undef cuGraphInstantiate
CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                    unsigned long long flags) {
    return cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
}

/* cuGetProcAddress (used internally, macro renames to _v2) */
#undef cuGetProcAddress
extern CUresult cuGetProcAddress_v2(const char *, void **, int,
                                    cuuint64_t, CUdriverProcAddressQueryResult *);
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags,
                          CUdriverProcAddressQueryResult *symbolStatus) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
}

#ifdef CUVK_NVJITLINK
#undef cuLinkCreate
extern CUresult cuLinkCreate_v2(unsigned int, CUjit_option *, void **,
                                CUlinkState *);
CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                      void **optionValues, CUlinkState *stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

#undef cuLinkAddData
extern CUresult cuLinkAddData_v2(CUlinkState, CUjitInputType, void *, size_t,
                                 const char *, unsigned int, CUjit_option *,
                                 void **);
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                       size_t size, const char *name, unsigned int numOptions,
                       CUjit_option *options, void **optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                            optionValues);
}

#undef cuLinkAddFile
extern CUresult cuLinkAddFile_v2(CUlinkState, CUjitInputType, const char *,
                                 unsigned int, CUjit_option *, void **);
CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type,
                       const char *path, unsigned int numOptions,
                       CUjit_option *options, void **optionValues) {
    return cuLinkAddFile_v2(state, type, path, numOptions, options,
                            optionValues);
}
#endif

/* ---- Missing stubs needed by cuFFT ---- */

#undef cuStreamWaitValue32
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     unsigned int value, unsigned int flags) {
    (void)stream; (void)addr; (void)value; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

#undef cuStreamBatchMemOp
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
    (void)stream; (void)count; (void)paramArray; (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
}

#undef cuMemcpy2D
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
    (void)pCopy;
    return CUDA_ERROR_NOT_SUPPORTED;
}

#undef cuMemcpy2DAsync
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy,
                                 CUstream hStream) {
    (void)pCopy; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* _v2 aliases for the above (cuFFT dlsyms both names) */
__asm__(".globl cuMemcpy2D_v2\ncuMemcpy2D_v2 = cuMemcpy2D");
__asm__(".globl cuMemcpy2DAsync_v2\ncuMemcpy2DAsync_v2 = cuMemcpy2DAsync");

#undef cuMemcpyDtoDAsync
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                                   CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcDevice; (void)ByteCount; (void)hStream;
    return CUDA_ERROR_NOT_SUPPORTED;
}
__asm__(".globl cuMemcpyDtoDAsync_v2\n"
        "cuMemcpyDtoDAsync_v2 = cuMemcpyDtoDAsync");
