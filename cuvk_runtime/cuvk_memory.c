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

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper: create a staging buffer (host-visible, coherent)
 * ============================================================================ */

static CUresult cuvk_create_staging_buffer(struct CUctx_st *ctx,
                                           VkDeviceSize size,
                                           VkBuffer *out_buffer,
                                           VkDeviceMemory *out_memory,
                                           void **out_mapped)
{
    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    vr = vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        vkFreeMemory(ctx->device, memory, NULL);
        vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    if (out_mapped) {
        void *mapped = NULL;
        vr = vkMapMemory(ctx->device, memory, 0, size, 0, &mapped);
        if (vr != VK_SUCCESS) {
            vkFreeMemory(ctx->device, memory, NULL);
            vkDestroyBuffer(ctx->device, buffer, NULL);
            return cuvk_vk_to_cu(vr);
        }
        *out_mapped = mapped;
    }

    *out_buffer = buffer;
    *out_memory = memory;
    return CUDA_SUCCESS;
}

/* Destroy a staging buffer */
static void cuvk_destroy_staging_buffer(struct CUctx_st *ctx,
                                        VkBuffer buffer,
                                        VkDeviceMemory memory)
{
    vkDestroyBuffer(ctx->device, buffer, NULL);
    vkFreeMemory(ctx->device, memory, NULL);
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
    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, &buffer);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Get memory requirements */
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, buffer, &mem_reqs);

    /* Find device-local memory type */
    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
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
    vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, &memory);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
        return cuvk_vk_to_cu(vr);
    }

    /* Bind memory to buffer */
    vr = vkBindBufferMemory(ctx->device, buffer, memory, 0);
    if (vr != VK_SUCCESS) {
        vkFreeMemory(ctx->device, memory, NULL);
        vkDestroyBuffer(ctx->device, buffer, NULL);
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
            vkFreeMemory(ctx->device, memory, NULL);
            vkDestroyBuffer(ctx->device, buffer, NULL);
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
    vkDestroyBuffer(ctx->device, alloc->buffer, NULL);
    vkFreeMemory(ctx->device, alloc->memory, NULL);

    /* Remove from array by shifting left */
    if (idx + 1 < ctx->alloc_count) {
        memmove(&ctx->allocs[idx], &ctx->allocs[idx + 1],
                (ctx->alloc_count - idx - 1) * sizeof(CuvkAlloc));
    }
    ctx->alloc_count--;

    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuMemcpyHtoD_v2 - Host to Device copy (synchronous)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                  size_t ByteCount)
{
    if (!srcHost || ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Look up destination alloc */
    CuvkAlloc *dst_alloc = cuvk_alloc_lookup(ctx, dstDevice);
    if (!dst_alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize offset = (VkDeviceSize)((uint64_t)dstDevice -
                                          (uint64_t)dst_alloc->device_addr);

    /* Create staging buffer */
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void *mapped = NULL;

    CUresult res = cuvk_create_staging_buffer(ctx, (VkDeviceSize)ByteCount,
                                               &staging_buf, &staging_mem,
                                               &mapped);
    if (res != CUDA_SUCCESS)
        return res;

    /* Copy host data into staging */
    memcpy(mapped, srcHost, ByteCount);
    vkUnmapMemory(ctx->device, staging_mem);

    /* Record and submit copy command */
    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return res;
    }

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = offset;
    region.size = (VkDeviceSize)ByteCount;

    vkCmdCopyBuffer(cb, staging_buf, dst_alloc->buffer, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);

    /* Clean up staging */
    cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);

    return res;
}

/* ============================================================================
 * cuMemcpyDtoH_v2 - Device to Host copy (synchronous)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                  size_t ByteCount)
{
    if (!dstHost || ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Look up source alloc */
    CuvkAlloc *src_alloc = cuvk_alloc_lookup(ctx, srcDevice);
    if (!src_alloc)
        return CUDA_ERROR_INVALID_VALUE;

    VkDeviceSize offset = (VkDeviceSize)((uint64_t)srcDevice -
                                          (uint64_t)src_alloc->device_addr);

    /* Create staging buffer */
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;

    CUresult res = cuvk_create_staging_buffer(ctx, (VkDeviceSize)ByteCount,
                                               &staging_buf, &staging_mem,
                                               NULL);
    if (res != CUDA_SUCCESS)
        return res;

    /* Record and submit copy command */
    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return res;
    }

    VkBufferCopy region = {0};
    region.srcOffset = offset;
    region.dstOffset = 0;
    region.size = (VkDeviceSize)ByteCount;

    vkCmdCopyBuffer(cb, src_alloc->buffer, staging_buf, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);
    if (res != CUDA_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return res;
    }

    /* Map staging and copy to host */
    void *mapped = NULL;
    VkResult vr = vkMapMemory(ctx->device, staging_mem, 0,
                               (VkDeviceSize)ByteCount, 0, &mapped);
    if (vr != VK_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return cuvk_vk_to_cu(vr);
    }

    memcpy(dstHost, mapped, ByteCount);
    vkUnmapMemory(ctx->device, staging_mem);

    /* Clean up staging */
    cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);

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

    vkCmdCopyBuffer(cb, src_alloc->buffer, dst_alloc->buffer, 1, &region);

    return cuvk_oneshot_end(ctx, cb);
}

/* ============================================================================
 * cuMemcpyHtoDAsync_v2 - Host to Device copy (async stub, calls sync version)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                       const void *srcHost,
                                       size_t ByteCount, CUstream hStream)
{
    (void)hStream;
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

/* ============================================================================
 * cuMemcpyDtoHAsync_v2 - Device to Host copy (async stub, calls sync version)
 * ============================================================================ */

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                       size_t ByteCount, CUstream hStream)
{
    (void)hStream;
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

    vkCmdFillBuffer(cb, alloc->buffer, offset, byte_count, ui);

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

        vkCmdFillBuffer(cb, alloc->buffer, offset, aligned_count, pattern);

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
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void *mapped = NULL;

    CUresult res = cuvk_create_staging_buffer(ctx, staging_size,
                                               &staging_buf, &staging_mem,
                                               &mapped);
    if (res != CUDA_SUCCESS)
        return res;

    memset(mapped, uc, (size_t)staging_size);
    vkUnmapMemory(ctx->device, staging_mem);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return res;
    }

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = (aligned_count == 0) ?
                       (VkDeviceSize)((uint64_t)dstDevice -
                                      (uint64_t)alloc->device_addr) :
                       offset;
    region.size = staging_size;

    vkCmdCopyBuffer(cb, staging_buf, alloc->buffer, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);
    cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);

    return res;
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

        vkCmdFillBuffer(cb, alloc->buffer, offset, aligned_bytes, pattern);

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
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void *mapped = NULL;

    CUresult res = cuvk_create_staging_buffer(ctx, staging_size,
                                               &staging_buf, &staging_mem,
                                               &mapped);
    if (res != CUDA_SUCCESS)
        return res;

    /* Fill with 16-bit pattern */
    unsigned short *p = (unsigned short *)mapped;
    size_t count = (size_t)(staging_size / 2);
    for (size_t i = 0; i < count; i++)
        p[i] = us;

    vkUnmapMemory(ctx->device, staging_mem);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS) {
        cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);
        return res;
    }

    VkBufferCopy region = {0};
    region.srcOffset = 0;
    region.dstOffset = (aligned_bytes == 0) ?
                       (VkDeviceSize)((uint64_t)dstDevice -
                                      (uint64_t)alloc->device_addr) :
                       offset;
    region.size = staging_size;

    vkCmdCopyBuffer(cb, staging_buf, alloc->buffer, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);
    cuvk_destroy_staging_buffer(ctx, staging_buf, staging_mem);

    return res;
}
