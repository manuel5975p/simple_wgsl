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

#ifdef __cplusplus
extern "C" {
#endif

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

/* A single Vulkan buffer allocation tracked by the context */
typedef struct CuvkAlloc {
    VkBuffer        buffer;
    VkDeviceMemory  memory;
    VkDeviceSize    size;
    VkDeviceAddress device_addr;  /* BDA virtual address, or 0 */
    void           *host_mapped;  /* persistent map for host-visible memory */
} CuvkAlloc;

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
    VkQueryPool         query_pool;
    bool                recorded;
    uint64_t            timestamp_ns;
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

    /* Default (NULL) stream */
    struct CUstream_st              default_stream;
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
};

/* ============================================================================
 * Global runtime state (singleton)
 * ============================================================================ */

#define CUVK_MAX_PHYSICAL_DEVICES 16

typedef struct CuvkGlobal {
    bool                initialized;
    VkInstance          instance;
    VkPhysicalDevice    physical_devices[CUVK_MAX_PHYSICAL_DEVICES];
    uint32_t            physical_device_count;
    struct CUctx_st    *current_ctx;
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

/* Submit whatever is recorded on the stream and wait for completion */
CUresult cuvk_stream_submit_and_wait(struct CUstream_st *stream);

/* Begin a one-shot command buffer (for synchronous memory ops, etc.) */
CUresult cuvk_oneshot_begin(struct CUctx_st *ctx, VkCommandBuffer *out_cb);

/* End + submit + wait on a one-shot command buffer */
CUresult cuvk_oneshot_end(struct CUctx_st *ctx, VkCommandBuffer cb);

#ifdef __cplusplus
}
#endif

#endif /* CUVK_INTERNAL_H */
