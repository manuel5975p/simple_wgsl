/*
 * cuvk_internal.h - Internal header for CUDA-on-Vulkan runtime
 *
 * Shared by all cuvk_runtime .c files. Defines the backing structs for
 * CUDA opaque handles (CUctx_st, CUmod_st, CUfunc_st, CUstream_st,
 * CUevent_st) and helper utilities.
 */

#ifndef CUVK_INTERNAL_H
#define CUVK_INTERNAL_H

#include <vulkan/vulkan.h>
#include "cuda.h"
#include "simple_wgsl.h"

#include <stdbool.h>
#include <stdio.h>

#ifdef CUVK_DEBUG_LOG
#define CUVK_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define CUVK_LOG(...) ((void)0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Vulkan function pointer tables (loaded via dlsym + vkGet*ProcAddr)
 * ============================================================================ */

/* Pre-instance functions (loaded via vkGetInstanceProcAddr(NULL, ...)) */
#define CUVK_GLOBAL_FUNCS(X) \
    X(vkCreateInstance) \
    X(vkEnumerateInstanceExtensionProperties) \
    X(vkEnumerateInstanceLayerProperties)

/* Instance-level functions (loaded after vkCreateInstance) */
#define CUVK_INSTANCE_FUNCS(X) \
    X(vkCreateDevice) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetDeviceProcAddr) \
    X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties)

/* Device-level functions (loaded after vkCreateDevice) */
#define CUVK_DEVICE_FUNCS(X) \
    X(vkAllocateCommandBuffers) \
    X(vkAllocateDescriptorSets) \
    X(vkAllocateMemory) \
    X(vkBeginCommandBuffer) \
    X(vkBindBufferMemory) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdBindPipeline) \
    X(vkCmdCopyBuffer) \
    X(vkCmdDispatch) \
    X(vkCmdFillBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdPushConstants) \
    X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) \
    X(vkCreateBuffer) \
    X(vkCreateCommandPool) \
    X(vkCreateComputePipelines) \
    X(vkCreateDescriptorPool) \
    X(vkCreateDescriptorSetLayout) \
    X(vkCreateFence) \
    X(vkCreatePipelineLayout) \
    X(vkCreateQueryPool) \
    X(vkCreateSemaphore) \
    X(vkCreateShaderModule) \
    X(vkDestroyBuffer) \
    X(vkDestroyCommandPool) \
    X(vkDestroyDescriptorPool) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkDestroyDevice) \
    X(vkDestroyFence) \
    X(vkDestroyPipeline) \
    X(vkDestroyPipelineLayout) \
    X(vkDestroyQueryPool) \
    X(vkDestroySemaphore) \
    X(vkDestroyShaderModule) \
    X(vkDeviceWaitIdle) \
    X(vkEndCommandBuffer) \
    X(vkFlushMappedMemoryRanges) \
    X(vkFreeCommandBuffers) \
    X(vkFreeDescriptorSets) \
    X(vkFreeMemory) \
    X(vkGetBufferMemoryRequirements) \
    X(vkGetDeviceQueue) \
    X(vkGetFenceStatus) \
    X(vkGetQueryPoolResults) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkMapMemory) \
    X(vkQueueSubmit) \
    X(vkResetCommandBuffer) \
    X(vkResetDescriptorPool) \
    X(vkResetFences) \
    X(vkSignalSemaphore) \
    X(vkUpdateDescriptorSets) \
    X(vkWaitForFences) \
    X(vkWaitSemaphores)

typedef struct CuvkVk {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#define CUVK_VK_FIELD(fn) PFN_##fn fn;
    CUVK_GLOBAL_FUNCS(CUVK_VK_FIELD)
    CUVK_INSTANCE_FUNCS(CUVK_VK_FIELD)
    CUVK_DEVICE_FUNCS(CUVK_VK_FIELD)
#undef CUVK_VK_FIELD
} CuvkVk;

/* ============================================================================
 * VkResult -> CUresult mapping
 * ============================================================================ */

static inline CUresult cuvk_vk_to_cu(VkResult vr) {
    switch (vr) {
    case VK_SUCCESS:                    return CUDA_SUCCESS;
    case VK_NOT_READY:                  return CUDA_ERROR_NOT_READY;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return CUDA_ERROR_OUT_OF_MEMORY;
    case VK_ERROR_DEVICE_LOST:
    case VK_ERROR_INITIALIZATION_FAILED:
    default:                            return CUDA_ERROR_UNKNOWN;
    }
}

/* ============================================================================
 * Internal helper structs
 * ============================================================================ */

/* Context-local storage entry (used by cudart for per-context state) */
typedef struct CuvkStorageEntry {
    uintptr_t key;
    void     *value;
    void    (*dtor_cb)(CUcontext, void *key, void *value);
} CuvkStorageEntry;

/* A single Vulkan buffer allocation tracked by the context */
typedef struct CuvkAlloc {
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    VkDeviceAddress device_addr;  /* BDA virtual address, or 0 */
    void           *host_mapped;  /* persistent map for host-visible memory */
} CuvkAlloc;

/* A pinned host allocation (cuMemAllocHost) — VkBuffer in HOST_CACHED memory */
typedef struct CuvkHostAlloc {
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    void           *mapped;       /* persistently mapped pointer returned to user */
    bool            coherent;     /* false → needs flush/invalidate */
} CuvkHostAlloc;

/* Cached compute pipeline keyed by block dimensions */
typedef struct CuvkPipelineEntry {
    uint32_t          block_x;
    uint32_t          block_y;
    uint32_t          block_z;
    VkPipeline        pipeline;
} CuvkPipelineEntry;

/* Per-parameter metadata extracted from SSIR */
typedef struct CuvkParamInfo {
    uint32_t    size;        /* byte size of parameter */
    bool        is_pointer;  /* true if this param is a device pointer */
} CuvkParamInfo;

/* ============================================================================
 * CUDA Graph backing structs
 * ============================================================================ */

/* CUgraphNode */
struct CUgraphNode_st {
    CUgraphNodeType          type;
    union {
        CUDA_KERNEL_NODE_PARAMS  kernel;
        CUDA_MEMCPY3D            memcpy;
        CUDA_MEMSET_NODE_PARAMS  memset;
        CUDA_HOST_NODE_PARAMS    host;
    } params;
    void                   **kernel_params_copy;  /* owned copy of kernel param ptrs */
    size_t                   kernel_params_count;

    /* Graph topology (adjacency lists, indices into graph's node ptr array) */
    uint32_t                *deps;
    uint32_t                 dep_count;
    uint32_t                 dep_capacity;
    uint32_t                *dependents;
    uint32_t                 dependent_count;
    uint32_t                 dependent_capacity;
};

/* CUgraph — mutable graph template.
 * Nodes are individually heap-allocated so CUgraphNode pointers remain stable. */
struct CUgraph_st {
    struct CUgraphNode_st **nodes;   /* array of pointers to heap-allocated nodes */
    uint32_t                node_count;
    uint32_t                node_capacity;
};

/* CUgraphExec — instantiated (executable) graph */
struct CUgraphExec_st {
    struct CUctx_st        *ctx;
    struct CUgraphNode_st  *nodes;        /* flat deep copy, topologically sorted */
    uint32_t                node_count;
    uint64_t               *sem_values;   /* timeline semaphore value per node */
};

/* ============================================================================
 * CUDA opaque handle backing structs
 * ============================================================================ */

/* CUstream */
struct CUstream_st {
    struct CUctx_st    *ctx;
    VkCommandBuffer     cmd_buf;
    VkFence             fence;
    bool                recording;  /* true while commands are being recorded */
};

/* CUevent */
struct CUevent_st {
    struct CUctx_st    *ctx;
    struct CUstream_st *stream;     /* stream the event was recorded on */
    VkQueryPool         query_pool;
    bool                recorded;
};

/* CUcontext */
struct CUctx_st {
    /* Vulkan core objects */
    VkInstance                      instance;
    VkPhysicalDevice                physical_device;
    VkDevice                        device;
    VkQueue                         compute_queue;
    uint32_t                        compute_queue_family;

    /* Command / descriptor pools */
    VkCommandPool                   cmd_pool;
    VkDescriptorPool                desc_pool;

    /* Cached device properties */
    VkPhysicalDeviceMemoryProperties mem_props;
    VkPhysicalDeviceProperties       dev_props;

    /* Buffer Device Address support */
    bool                            has_bda;
    PFN_vkGetBufferDeviceAddress    pfn_get_bda;

    /* Tracked allocations */
    CuvkAlloc                      *allocs;
    uint32_t                        alloc_count;
    uint32_t                        alloc_capacity;
    uint64_t                        next_synthetic_addr; /* for non-BDA fallback */

    /* Timeline semaphore for ordered submission */
    VkSemaphore                     timeline_sem;
    uint64_t                        timeline_value;

    /* Reusable one-shot command buffer + fence (for synchronous ops) */
    VkCommandBuffer                 oneshot_cb;
    VkFence                         oneshot_fence;

    /* Cached staging buffer for uploads (grow-only, persistently mapped) */
    VkBuffer                        staging_buf;
    VkDeviceMemory                  staging_mem;
    VkDeviceSize                    staging_capacity;
    void                           *staging_mapped;

    /* Download staging buffer (HOST_CACHED for fast CPU reads) */
    VkBuffer                        download_buf;
    VkDeviceMemory                  download_mem;
    VkDeviceSize                    download_capacity;
    void                           *download_mapped;
    bool                            download_needs_invalidate;

    /* Pinned host allocations (cuMemAllocHost) */
    CuvkHostAlloc                  *host_allocs;
    uint32_t                        host_alloc_count;
    uint32_t                        host_alloc_capacity;

    /* Default (NULL) stream */
    struct CUstream_st              default_stream;

    /* Deferred cuFFT flush callback — set by cuFFT library, called at
     * sync points (cuCtxSynchronize, cuMemcpy).  NULL if cuFFT not loaded. */
    void                          (*fft_flush_fn)(struct CUctx_st *);

    /* Context-local storage (used by cudart internals) */
    CuvkStorageEntry               *storage;
    uint32_t                        storage_count;
    uint32_t                        storage_capacity;
};

/* CUfunction */
struct CUfunc_st {
    struct CUmod_st        *module;
    char                   *name;
    VkShaderModule          shader_module;
    VkPipelineLayout        pipeline_layout;
    VkDescriptorSetLayout   desc_layout;

    /* Pipeline cache (specialised per block size) */
    CuvkPipelineEntry      *pipeline_cache;
    uint32_t                pipeline_cache_count;
    uint32_t                pipeline_cache_capacity;

    /* Parameter reflection */
    CuvkParamInfo          *params;
    uint32_t                param_count;
    uint32_t                push_constant_size;

    /* BDA mode flag */
    bool                    use_bda;
};

/* A module-level global variable (e.g. .global .u64 lut_sp) */
typedef struct CuvkModuleGlobal {
    char                name[256];
    uint32_t            size;
    uint32_t            binding;
    VkBuffer            buffer;
    VkDeviceMemory      memory;
    uint64_t            device_addr;
} CuvkModuleGlobal;

/* CUmodule */
struct CUmod_st {
    struct CUctx_st    *ctx;

    /* Compiled SPIR-V (owned) */
    uint32_t           *spirv_words;
    uint32_t            spirv_count;   /* word count */

    /* SSIR module (owned, kept for reflection) */
    SsirModule         *ssir;

    /* Functions extracted from the module */
    struct CUfunc_st   *functions;
    uint32_t            function_count;

    /* Module-level globals (descriptor-backed) */
    CuvkModuleGlobal   *globals;
    uint32_t            global_count;
    VkDescriptorSetLayout globals_desc_layout;
    VkDescriptorPool      globals_desc_pool;
    VkDescriptorSet       globals_desc_set;
};

/* ============================================================================
 * Global runtime state (singleton)
 * ============================================================================ */

#define CUVK_MAX_PHYSICAL_DEVICES 16

typedef struct CuvkGlobal {
    bool                initialized;
    bool                has_validation;
    bool                exiting;
    VkInstance          instance;
    VkPhysicalDevice    physical_devices[CUVK_MAX_PHYSICAL_DEVICES];
    uint32_t            physical_device_count;
    struct CUctx_st    *current_ctx;
    VkDebugUtilsMessengerEXT debug_messenger;
    CuvkVk              vk;
} CuvkGlobal;

extern CuvkGlobal g_cuvk;

/* ============================================================================
 * Internal helper declarations
 * ============================================================================ */

/* Find a Vulkan memory type index matching the given filter and properties */
int32_t cuvk_find_memory_type(const VkPhysicalDeviceMemoryProperties *mem_props,
                              uint32_t type_filter,
                              VkMemoryPropertyFlags required);

/* Look up a CuvkAlloc by its device address (BDA) or buffer handle */
CuvkAlloc *cuvk_alloc_lookup(struct CUctx_st *ctx, CUdeviceptr dptr);

/* Extract PTX text from an NVIDIA fatbin container. Returns malloc'd string. */
char *cuvk_fatbin_extract_ptx(const void *fatbin_data, size_t *ptx_len);

/* Submit whatever is recorded on the stream and wait for completion */
CUresult cuvk_stream_submit_and_wait(struct CUstream_st *stream);

/* Resolve a CUstream handle: NULL, CU_STREAM_LEGACY (0x1), and
 * CU_STREAM_PER_THREAD (0x2) all map to the context's default stream. */
static inline struct CUstream_st *cuvk_resolve_stream(CUstream hStream) {
    if ((uintptr_t)hStream <= 2)
        return g_cuvk.current_ctx ? &g_cuvk.current_ctx->default_stream : NULL;
    return hStream;
}

/* Flush deferred cuFFT work (if any).  Calls ctx->fft_flush_fn. */
static inline void cuvk_fft_flush(struct CUctx_st *ctx) {
    if (ctx->fft_flush_fn) ctx->fft_flush_fn(ctx);
}

#ifdef CUVK_NVJITLINK
/* Compile LTO-IR sections from a fatbin via nvJitLink. Returns malloc'd PTX. */
char *cuvk_jitlink_compile_ltoir(const void *fatbin_data, size_t *ptx_len);
/* Compile raw LTO-IR blob via nvJitLink. Returns malloc'd PTX. */
char *cuvk_jitlink_compile_raw(const void *data, size_t size, size_t *ptx_len);
#endif

/* Ensure a stream's command buffer is recording (begin if needed) */
CUresult cuvk_stream_ensure_recording(struct CUstream_st *stream);

/* Begin a one-shot command buffer (for synchronous memory ops, etc.) */
CUresult cuvk_oneshot_begin(struct CUctx_st *ctx, VkCommandBuffer *out_cb);

/* End + submit + wait on a one-shot command buffer */
CUresult cuvk_oneshot_end(struct CUctx_st *ctx, VkCommandBuffer cb);

#ifdef __cplusplus
}
#endif

#endif /* CUVK_INTERNAL_H */
