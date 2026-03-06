/*
 * cuvk_memory.c - Memory allocation, deallocation, transfers, and memset
 *
 * Implements the CUDA driver API functions: cuMemAlloc_v2, cuMemFree_v2,
 * cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuMemcpyDtoD_v2, cuMemcpyHtoDAsync_v2,
 * cuMemcpyDtoHAsync_v2, cuMemsetD8_v2, cuMemsetD16_v2, cuMemsetD32_v2.
 *
 * Also provides: cuvk_alloc_lookup (binary search on sorted allocs array).
 */

#include "cuvk_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper: ensure the cached staging buffer is at least `size` bytes.
 * Grow-only: if the current buffer is large enough, reuse it.
 * The buffer is persistently mapped (never unmapped).
 * ============================================================================ */

static CUresult cuvk_ensure_staging(struct CUctx_st *ctx, VkDeviceSize size)
{
    if (ctx->staging_capacity >= size)
        return CUDA_SUCCESS;

    /* Destroy old staging buffer if any */
    if (ctx->staging_buf) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
        ctx->staging_buf = VK_NULL_HANDLE;
    }
    if (ctx->staging_mem) {
        g_cuvk.vk.vkFreeMemory(ctx->device, ctx->staging_mem, NULL);
        ctx->staging_mem = VK_NULL_HANDLE;
    }
    ctx->staging_mapped = NULL;
    ctx->staging_capacity = 0;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkMemoryRequirements mem_reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    vr = g_cuvk.vk.vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    void *mapped = NULL;
    vr = g_cuvk.vk.vkMapMemory(ctx->device, memory, 0, size, 0, &mapped);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    ctx->staging_buf = buffer;
    ctx->staging_mem = memory;
    ctx->staging_capacity = size;
    ctx->staging_mapped = mapped;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Helper: ensure the download staging buffer is at least `size` bytes.
 * Uses HOST_CACHED memory for fast CPU reads (device-to-host path).
 * ============================================================================ */

static CUresult cuvk_ensure_download_staging(struct CUctx_st *ctx,
                                              VkDeviceSize size)
{
    if (ctx->download_capacity >= size)
        return CUDA_SUCCESS;

    /* Destroy old download buffer if any */
    if (ctx->download_buf) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, ctx->download_buf, NULL);
        ctx->download_buf = VK_NULL_HANDLE;
    }
    if (ctx->download_mem) {
        g_cuvk.vk.vkFreeMemory(ctx->device, ctx->download_mem, NULL);
        ctx->download_mem = VK_NULL_HANDLE;
    }
    ctx->download_mapped = NULL;
    ctx->download_capacity = 0;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkMemoryRequirements mem_reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    /* Try HOST_VISIBLE | HOST_CACHED | HOST_COHERENT first */
    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    bool needs_invalidate = false;

    if (mem_type < 0) {
        /* Fall back to HOST_VISIBLE | HOST_CACHED without coherent */
        mem_type = cuvk_find_memory_type(
            &ctx->mem_props, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        needs_invalidate = true;
    }

    if (mem_type < 0) {
        /* Last resort: HOST_VISIBLE | HOST_COHERENT (same as upload staging) */
        mem_type = cuvk_find_memory_type(
            &ctx->mem_props, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        needs_invalidate = false;
    }

    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    vr = g_cuvk.vk.vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    void *mapped = NULL;
    vr = g_cuvk.vk.vkMapMemory(ctx->device, memory, 0, size, 0, &mapped);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    ctx->download_buf = buffer;
    ctx->download_mem = memory;
    ctx->download_capacity = size;
    ctx->download_mapped = mapped;
    ctx->download_needs_invalidate = needs_invalidate;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuvk_alloc_lookup - binary search sorted allocs array
 * ============================================================================ */

CuvkAlloc *cuvk_alloc_lookup(struct CUctx_st *ctx, CUdeviceptr dptr)
{
    if (!ctx || !ctx->allocs || ctx->alloc_count == 0)
        return NULL;

    uint64_t addr = (uint64_t)dptr;
    uint32_t lo = 0;
    uint32_t hi = ctx->alloc_count;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        CuvkAlloc *a = &ctx->allocs[mid];

        if (addr < (uint64_t)a->device_addr) {
            hi = mid;
        } else if (addr >= (uint64_t)a->device_addr + (uint64_t)a->size) {
            lo = mid + 1;
        } else {
            /* alloc->device_addr <= addr < alloc->device_addr + alloc->size */
            return a;
        }
    }

    return NULL;
}

/* ============================================================================
 * Helper: find insertion position in sorted allocs array
 * ============================================================================ */

static uint32_t cuvk_alloc_find_insert_pos(struct CUctx_st *ctx,
                                            uint64_t device_addr)
{
    uint32_t lo = 0;
    uint32_t hi = ctx->alloc_count;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if ((uint64_t)ctx->allocs[mid].device_addr < device_addr)
            lo = mid + 1;
        else
            hi = mid;
    }

    return lo;
}

/* ============================================================================
 * cuMemAlloc_v2
 * ============================================================================ */

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    CUVK_LOG("[cuvk] cuMemAlloc_v2: size=%zu\n", bytesize);
    if (!dptr || bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Create VkBuffer */
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (ctx->has_bda)
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = (VkDeviceSize)bytesize;
    buf_ci.usage = usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Get memory requirements */
    VkMemoryRequirements mem_reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    /* Find device-local memory type */
    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /* Allocate memory */
    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    VkMemoryAllocateFlagsInfo flags_info = {0};
    if (ctx->has_bda) {
        flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        alloc_info.pNext = &flags_info;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    /* Bind memory to buffer */
    vr = g_cuvk.vk.vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    /* Get device address */
    VkDeviceAddress device_addr;
    if (ctx->has_bda && ctx->pfn_get_bda) {
        VkBufferDeviceAddressInfo addr_info = {0};
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = buffer;
        device_addr = ctx->pfn_get_bda(ctx->device, &addr_info);
    } else {
        /* Synthetic monotonic address. Start at 0x100000 to avoid confusion
         * with NULL. Increment by alloc size, aligned to 256. */
        if (ctx->next_synthetic_addr == 0)
            ctx->next_synthetic_addr = 0x100000;
        device_addr = ctx->next_synthetic_addr;
        /* Align next address to 256 bytes after this allocation */
        uint64_t aligned_size = (bytesize + 255) & ~(uint64_t)255;
        ctx->next_synthetic_addr += aligned_size;
    }

    /* Grow allocs array if needed */
    if (ctx->alloc_count >= ctx->alloc_capacity) {
        uint32_t new_cap = ctx->alloc_capacity == 0 ? 16 :
                           ctx->alloc_capacity * 2;
        CuvkAlloc *new_allocs = (CuvkAlloc *)realloc(
            ctx->allocs, new_cap * sizeof(CuvkAlloc));
        if (!new_allocs) {
            g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
            g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        ctx->allocs = new_allocs;
        ctx->alloc_capacity = new_cap;
    }

    /* Find insertion position (keep sorted by device_addr) */
    uint32_t pos = cuvk_alloc_find_insert_pos(ctx, (uint64_t)device_addr);

    /* Shift elements right to make room */
    if (pos < ctx->alloc_count) {
        memmove(&ctx->allocs[pos + 1], &ctx->allocs[pos],
                (ctx->alloc_count - pos) * sizeof(CuvkAlloc));
    }

    /* Insert new allocation */
    ctx->allocs[pos].buffer = buffer;
    ctx->allocs[pos].memory = memory;
    ctx->allocs[pos].size = (VkDeviceSize)bytesize;
    ctx->allocs[pos].device_addr = device_addr;
    ctx->allocs[pos].host_mapped = NULL;
    ctx->alloc_count++;

    *dptr = (CUdeviceptr)device_addr;
    CUVK_LOG("[cuvk] cuMemAlloc_v2: SUCCESS addr=0x%llx size=%zu\n",
            (unsigned long long)device_addr, bytesize);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuMemFree_v2
 * ============================================================================ */

CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr)
{
    if (dptr == 0)
        return CUDA_SUCCESS; /* freeing NULL is a no-op */

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Find the allocation */
    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dptr);
    if (!alloc)
        return CUDA_ERROR_INVALID_VALUE;

    /* Calculate index in allocs array */
    uint32_t idx = (uint32_t)(alloc - ctx->allocs);

    /* Destroy Vulkan resources */
    g_cuvk.vk.vkDestroyBuffer(ctx->device, alloc->buffer, NULL);
    g_cuvk.vk.vkFreeMemory(ctx->device, alloc->memory, NULL);

    /* Remove from array by shifting left */
    if (idx + 1 < ctx->alloc_count) {
        memmove(&ctx->allocs[idx], &ctx->allocs[idx + 1],
                (ctx->alloc_count - idx - 1) * sizeof(CuvkAlloc));
    }
    ctx->alloc_count--;

    return CUDA_SUCCESS;
}

/* ============================================================================
 * Pinned host memory: lookup helpers
 * ============================================================================ */

static CuvkHostAlloc *cuvk_host_alloc_lookup(struct CUctx_st *ctx,
                                              const void *ptr)
{
    if (!ctx || !ctx->host_allocs || ctx->host_alloc_count == 0)
        return NULL;

    uintptr_t addr = (uintptr_t)ptr;
    uint32_t lo = 0, hi = ctx->host_alloc_count;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        CuvkHostAlloc *a = &ctx->host_allocs[mid];
        uintptr_t start = (uintptr_t)a->mapped;

        if (addr < start) {
            hi = mid;
        } else if (addr >= start + (uintptr_t)a->size) {
            lo = mid + 1;
        } else {
            return a;
        }
    }
    return NULL;
}

static uint32_t cuvk_host_alloc_find_insert_pos(struct CUctx_st *ctx,
                                                  uintptr_t addr)
{
    uint32_t lo = 0, hi = ctx->host_alloc_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if ((uintptr_t)ctx->host_allocs[mid].mapped < addr)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* ============================================================================
 * cuMemAllocHost_v2 - Allocate pinned host memory (HOST_CACHED VkBuffer)
 * ============================================================================ */

CUresult CUDAAPI cuMemAllocHost_v2(void **pp, size_t bytesize)
{
    if (!pp || bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = (VkDeviceSize)bytesize;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkMemoryRequirements mem_reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    /* Prefer HOST_VISIBLE | HOST_CACHED | HOST_COHERENT (system RAM, cached) */
    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    bool coherent = true;

    if (mem_type < 0) {
        mem_type = cuvk_find_memory_type(
            &ctx->mem_props, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        coherent = false;
    }

    if (mem_type < 0) {
        mem_type = cuvk_find_memory_type(
            &ctx->mem_props, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        coherent = true;
    }

    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    vr = g_cuvk.vk.vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    void *mapped = NULL;
    vr = g_cuvk.vk.vkMapMemory(ctx->device, memory, 0, bytesize, 0, &mapped);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    /* Track the allocation (sorted by mapped pointer) */
    if (ctx->host_alloc_count >= ctx->host_alloc_capacity) {
        uint32_t new_cap = ctx->host_alloc_capacity == 0 ? 8 :
                           ctx->host_alloc_capacity * 2;
        CuvkHostAlloc *new_allocs = (CuvkHostAlloc *)realloc(
            ctx->host_allocs, new_cap * sizeof(CuvkHostAlloc));
        if (!new_allocs) {
            g_cuvk.vk.vkFreeMemory(ctx->device, memory, NULL);
            g_cuvk.vk.vkDestroyBuffer(ctx->device, buffer, NULL);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        ctx->host_allocs = new_allocs;
        ctx->host_alloc_capacity = new_cap;
    }

    uint32_t pos = cuvk_host_alloc_find_insert_pos(ctx, (uintptr_t)mapped);
    if (pos < ctx->host_alloc_count) {
        memmove(&ctx->host_allocs[pos + 1], &ctx->host_allocs[pos],
                (ctx->host_alloc_count - pos) * sizeof(CuvkHostAlloc));
    }

    ctx->host_allocs[pos].buffer = buffer;
    ctx->host_allocs[pos].memory = memory;
    ctx->host_allocs[pos].size = (VkDeviceSize)bytesize;
    ctx->host_allocs[pos].mapped = mapped;
    ctx->host_allocs[pos].coherent = coherent;
    ctx->host_alloc_count++;

    *pp = mapped;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuMemFreeHost - Free pinned host memory
 * ============================================================================ */

CUresult CUDAAPI cuMemFreeHost(void *p)
{
    if (!p)
        return CUDA_SUCCESS;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    CuvkHostAlloc *alloc = cuvk_host_alloc_lookup(ctx, p);
    if (!alloc)
        return CUDA_ERROR_INVALID_VALUE;

    uint32_t idx = (uint32_t)(alloc - ctx->host_allocs);

    g_cuvk.vk.vkDestroyBuffer(ctx->device, alloc->buffer, NULL);
    g_cuvk.vk.vkFreeMemory(ctx->device, alloc->memory, NULL);

    if (idx + 1 < ctx->host_alloc_count) {
        memmove(&ctx->host_allocs[idx], &ctx->host_allocs[idx + 1],
                (ctx->host_alloc_count - idx - 1) * sizeof(CuvkHostAlloc));
    }
    ctx->host_alloc_count--;

    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuMemcpyHtoD_v2 - Host to Device copy (synchronous)
 *
 * If srcHost is pinned (cuMemAllocHost), DMA directly — no staging memcpy.
 * Otherwise falls back to staging buffer path.
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                  size_t ByteCount)
{
    if (!srcHost || ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Flush default stream to ensure prior dispatches complete */
    cuvk_stream_submit_and_wait(&ctx->default_stream);

    /* Look up destination alloc */
    CuvkAlloc *dst_alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!dst_alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize dst_offset = (VkDeviceSize)((uint64_t)dstDevice -
                                              (uint64_t)dst_alloc->device_addr);

    /* Fast path: srcHost is pinned memory — DMA directly */
    CuvkHostAlloc *host_alloc = cuvk_host_alloc_lookup(ctx, srcHost);
    if (host_alloc) {
        VkDeviceSize src_offset = (VkDeviceSize)(
            (uintptr_t)srcHost - (uintptr_t)host_alloc->mapped);

        /* Flush CPU writes if non-coherent */
        if (!host_alloc->coherent) {
            VkMappedMemoryRange range = {0};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = host_alloc->memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            g_cuvk.vk.vkFlushMappedMemoryRanges(ctx->device, 1, &range);
        }

        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult res = cuvk_oneshot_begin(ctx, &cb);
        if (res != CUDA_SUCCESS)
            return res;

        VkBufferCopy region = {0};
        region.srcOffset = src_offset;
        region.dstOffset = dst_offset;
        region.size = (VkDeviceSize)ByteCount;

        g_cuvk.vk.vkCmdCopyBuffer(cb, host_alloc->buffer, dst_alloc->buffer, 1, &region);
        return cuvk_oneshot_end(ctx, cb);
    }

    /* Slow path: staging buffer + memcpy */
    CUresult res = cuvk_ensure_staging(ctx, (VkDeviceSize)ByteCount);
    if (res != CUDA_SUCCESS)
        return res;

    memcpy(ctx->staging_mapped, srcHost, ByteCount);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size = (VkDeviceSize)ByteCount;

    g_cuvk.vk.vkCmdCopyBuffer(cb, ctx->staging_buf, dst_alloc->buffer, 1, &region);
    return cuvk_oneshot_end(ctx, cb);
}

/* ============================================================================
 * cuMemcpyDtoH_v2 - Device to Host copy (synchronous)
 *
 * If dstHost is pinned (cuMemAllocHost), DMA directly — no staging memcpy.
 * Otherwise falls back to download staging buffer path.
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                  size_t ByteCount)
{
    if (!dstHost || ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Flush default stream to ensure prior dispatches complete */
    cuvk_stream_submit_and_wait(&ctx->default_stream);

    /* Look up source alloc */
    CuvkAlloc *src_alloc = cuvk_alloc_lookup(ctx, srcDevice);
    if (!src_alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize src_offset = (VkDeviceSize)((uint64_t)srcDevice -
                                              (uint64_t)src_alloc->device_addr);

    /* Fast path: dstHost is pinned memory — DMA directly */
    CuvkHostAlloc *host_alloc = cuvk_host_alloc_lookup(ctx, dstHost);
    if (host_alloc) {
        VkDeviceSize dst_offset = (VkDeviceSize)(
            (uintptr_t)dstHost - (uintptr_t)host_alloc->mapped);

        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult res = cuvk_oneshot_begin(ctx, &cb);
        if (res != CUDA_SUCCESS)
            return res;

        VkBufferCopy region = {0};
        region.srcOffset = src_offset;
        region.dstOffset = dst_offset;
        region.size = (VkDeviceSize)ByteCount;

        g_cuvk.vk.vkCmdCopyBuffer(cb, src_alloc->buffer, host_alloc->buffer, 1, &region);

        res = cuvk_oneshot_end(ctx, cb);
        if (res != CUDA_SUCCESS)
            return res;

        /* Invalidate CPU cache if non-coherent */
        if (!host_alloc->coherent) {
            VkMappedMemoryRange range = {0};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = host_alloc->memory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            g_cuvk.vk.vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        }

        return CUDA_SUCCESS;
    }

    /* Slow path: download staging buffer + memcpy */
    CUresult res = cuvk_ensure_download_staging(ctx, (VkDeviceSize)ByteCount);
    if (res != CUDA_SUCCESS)
        return res;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    VkBufferCopy region = {0};
    region.srcOffset = src_offset;
    region.dstOffset = 0;
    region.size = (VkDeviceSize)ByteCount;

    g_cuvk.vk.vkCmdCopyBuffer(cb, src_alloc->buffer, ctx->download_buf, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);
    if (res != CUDA_SUCCESS)
        return res;

    if (ctx->download_needs_invalidate) {
        VkMappedMemoryRange range = {0};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = ctx->download_mem;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        g_cuvk.vk.vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
    }

    memcpy(dstHost, ctx->download_mapped, ByteCount);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuMemcpyDtoD_v2 - Device to Device copy (synchronous)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                  size_t ByteCount)
{
    if (ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Flush default stream to ensure prior dispatches complete */
    cuvk_stream_submit_and_wait(&ctx->default_stream);

    /* Look up both allocs */
    CuvkAlloc *src_alloc = cuvk_alloc_lookup(ctx, srcDevice);
    CuvkAlloc *dst_alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!src_alloc || !dst_alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize src_offset = (VkDeviceSize)((uint64_t)srcDevice -
                                              (uint64_t)src_alloc->device_addr);
    VkDeviceSize dst_offset = (VkDeviceSize)((uint64_t)dstDevice -
                                              (uint64_t)dst_alloc->device_addr);

    /* Record and submit copy command */
    VkCommandBuffer cb = VK_NULL_HANDLE;
    CUresult res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    VkBufferCopy region = {0};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = (VkDeviceSize)ByteCount;

    g_cuvk.vk.vkCmdCopyBuffer(cb, src_alloc->buffer, dst_alloc->buffer, 1, &region);

    return cuvk_oneshot_end(ctx, cb);
}

/* ============================================================================
 * cuMemcpyHtoDAsync_v2 - Host to Device copy (async stub, calls sync version)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                       const void *srcHost,
                                       size_t ByteCount, CUstream hStream)
{
    /* Flush the target stream so any prior recorded work completes first */
    if (hStream) {
        CUresult res = cuvk_stream_submit_and_wait(hStream);
        if (res != CUDA_SUCCESS)
            return res;
    }
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

/* ============================================================================
 * cuMemcpyDtoHAsync_v2 - Device to Host copy (async stub, calls sync version)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                       size_t ByteCount, CUstream hStream)
{
    /* Flush the target stream so any prior recorded work completes first */
    if (hStream) {
        CUresult res = cuvk_stream_submit_and_wait(hStream);
        if (res != CUDA_SUCCESS)
            return res;
    }
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

/* ============================================================================
 * cuMemsetD32_v2 - Fill buffer with 32-bit value
 * ============================================================================ */

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                 size_t N)
{
    if (N == 0)
        return CUDA_SUCCESS;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Flush default stream to ensure prior dispatches complete */
    cuvk_stream_submit_and_wait(&ctx->default_stream);

    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize offset = (VkDeviceSize)((uint64_t)dstDevice -
                                          (uint64_t)alloc->device_addr);
    VkDeviceSize byte_count = (VkDeviceSize)(N * 4);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    CUresult res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    g_cuvk.vk.vkCmdFillBuffer(cb, alloc->buffer, offset, byte_count, ui);

    return cuvk_oneshot_end(ctx, cb);
}

/* ============================================================================
 * cuMemsetD8_v2 - Fill buffer with 8-bit value
 *
 * Uses staging buffer approach: fill staging with the byte pattern,
 * then copy to the device buffer.
 * ============================================================================ */

CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                                size_t N)
{
    if (N == 0)
        return CUDA_SUCCESS;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize offset = (VkDeviceSize)((uint64_t)dstDevice -
                                          (uint64_t)alloc->device_addr);

    /* Try fast path with vkCmdFillBuffer for aligned sizes */
    uint32_t pattern = (uint32_t)uc | ((uint32_t)uc << 8) |
                       ((uint32_t)uc << 16) | ((uint32_t)uc << 24);

    /* vkCmdFillBuffer requires offset and size to be multiples of 4 */
    VkDeviceSize aligned_count = (N / 4) * 4;
    VkDeviceSize remainder = N - aligned_count;

    if (aligned_count > 0 && (offset % 4) == 0) {
        /* Fast path: use vkCmdFillBuffer for the aligned portion */
        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult res = cuvk_oneshot_begin(ctx, &cb);
        if (res != CUDA_SUCCESS)
            return res;

        g_cuvk.vk.vkCmdFillBuffer(cb, alloc->buffer, offset, aligned_count, pattern);

        res = cuvk_oneshot_end(ctx, cb);
        if (res != CUDA_SUCCESS)
            return res;

        if (remainder == 0)
            return CUDA_SUCCESS;

        /* Handle remainder via staging */
        offset += aligned_count;
    } else {
        /* All via staging */
        remainder = N;
        aligned_count = 0;
    }

    /* Staging buffer for remainder (or all if unaligned) */
    VkDeviceSize staging_size = (aligned_count == 0) ? (VkDeviceSize)N :
                                                        remainder;

    CUresult res = cuvk_ensure_staging(ctx, staging_size);
    if (res != CUDA_SUCCESS)
        return res;

    memset(ctx->staging_mapped, uc, (size_t)staging_size);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = (aligned_count == 0) ?
                       (VkDeviceSize)((uint64_t)dstDevice -
                                      (uint64_t)alloc->device_addr) :
                       offset;
    region.size = staging_size;

    g_cuvk.vk.vkCmdCopyBuffer(cb, ctx->staging_buf, alloc->buffer, 1, &region);

    return cuvk_oneshot_end(ctx, cb);
}

/* ============================================================================
 * cuMemsetD16_v2 - Fill buffer with 16-bit value
 *
 * Uses staging buffer approach: fill staging with the 16-bit pattern,
 * then copy to the device buffer.
 * ============================================================================ */

CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                 size_t N)
{
    if (N == 0)
        return CUDA_SUCCESS;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize offset = (VkDeviceSize)((uint64_t)dstDevice -
                                          (uint64_t)alloc->device_addr);

    VkDeviceSize byte_count = (VkDeviceSize)(N * 2);

    /* Try fast path with vkCmdFillBuffer for aligned sizes */
    uint32_t pattern = (uint32_t)us | ((uint32_t)us << 16);

    /* vkCmdFillBuffer requires offset and size to be multiples of 4 */
    VkDeviceSize aligned_bytes = (byte_count / 4) * 4;
    VkDeviceSize remainder_bytes = byte_count - aligned_bytes;

    if (aligned_bytes > 0 && (offset % 4) == 0) {
        /* Fast path: use vkCmdFillBuffer for the aligned portion */
        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult res = cuvk_oneshot_begin(ctx, &cb);
        if (res != CUDA_SUCCESS)
            return res;

        g_cuvk.vk.vkCmdFillBuffer(cb, alloc->buffer, offset, aligned_bytes, pattern);

        res = cuvk_oneshot_end(ctx, cb);
        if (res != CUDA_SUCCESS)
            return res;

        if (remainder_bytes == 0)
            return CUDA_SUCCESS;

        /* Handle remainder via staging */
        offset += aligned_bytes;
    } else {
        /* All via staging */
        remainder_bytes = byte_count;
        aligned_bytes = 0;
    }

    /* Staging buffer for remainder (or all if unaligned) */
    VkDeviceSize staging_size = (aligned_bytes == 0) ? byte_count :
                                                        remainder_bytes;

    CUresult res = cuvk_ensure_staging(ctx, staging_size);
    if (res != CUDA_SUCCESS)
        return res;

    /* Fill with 16-bit pattern */
    unsigned short *p = (unsigned short *)ctx->staging_mapped;
    size_t count = (size_t)(staging_size / 2);
    for (size_t i = 0; i < count; i++)
        p[i] = us;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = (aligned_bytes == 0) ?
                       (VkDeviceSize)((uint64_t)dstDevice -
                                      (uint64_t)alloc->device_addr) :
                       offset;
    region.size = staging_size;

    g_cuvk.vk.vkCmdCopyBuffer(cb, ctx->staging_buf, alloc->buffer, 1, &region);

    return cuvk_oneshot_end(ctx, cb);
}
