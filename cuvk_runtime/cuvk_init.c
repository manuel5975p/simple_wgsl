/*
 * cuvk_init.c - Vulkan bootstrap, device enumeration, and context management
 *
 * Implements the CUDA driver API functions: cuInit, cuDriverGetVersion,
 * cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDeviceTotalMem_v2,
 * cuDeviceGetAttribute, cuCtxCreate_v4, cuCtxDestroy_v2, cuCtxGetCurrent,
 * cuCtxSetCurrent, cuCtxSynchronize, and the cuvk_find_memory_type helper.
 */

#include "cuvk_internal.h"

#include <stdlib.h>
#include <string.h>

/* Global runtime singleton */
CuvkGlobal g_cuvk = {0};

/* ============================================================================
 * Helper: find a Vulkan memory type matching filter + required property flags
 * ============================================================================ */

int32_t cuvk_find_memory_type(const VkPhysicalDeviceMemoryProperties *mem_props,
                              uint32_t type_filter,
                              VkMemoryPropertyFlags required)
{
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (mem_props->memoryTypes[i].propertyFlags & required) == required) {
            return (int32_t)i;
        }
    }
    return -1;
}

/* ============================================================================
 * cuInit
 * ============================================================================ */

CUresult CUDAAPI cuInit(unsigned int Flags)
{
    (void)Flags;

    if (g_cuvk.initialized)
        return CUDA_SUCCESS;

    /* Create VkInstance with VK_KHR_get_physical_device_properties2 */
    const char *instance_extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CUDA-on-Vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "cuvk_runtime";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = 1;
    create_info.ppEnabledExtensionNames = instance_extensions;

    VkResult vr = vkCreateInstance(&create_info, NULL, &g_cuvk.instance);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Enumerate physical devices */
    uint32_t count = 0;
    vr = vkEnumeratePhysicalDevices(g_cuvk.instance, &count, NULL);
    if (vr != VK_SUCCESS) {
        vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return cuvk_vk_to_cu(vr);
    }

    if (count == 0) {
        vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return CUDA_ERROR_NO_DEVICE;
    }

    if (count > CUVK_MAX_PHYSICAL_DEVICES)
        count = CUVK_MAX_PHYSICAL_DEVICES;

    vr = vkEnumeratePhysicalDevices(g_cuvk.instance, &count,
                                    g_cuvk.physical_devices);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) {
        vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return cuvk_vk_to_cu(vr);
    }

    g_cuvk.physical_device_count = count;
    g_cuvk.initialized = true;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDriverGetVersion
 * ============================================================================ */

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion)
{
    if (!driverVersion)
        return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = 12000; /* Pretend CUDA 12.0 */
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDeviceGet
 * ============================================================================ */

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    if (ordinal < 0 || (uint32_t)ordinal >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDeviceGetCount
 * ============================================================================ */

CUresult CUDAAPI cuDeviceGetCount(int *count)
{
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!count)
        return CUDA_ERROR_INVALID_VALUE;
    *count = (int)g_cuvk.physical_device_count;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDeviceGetName
 * ============================================================================ */

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev)
{
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!name || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || (uint32_t)dev >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_cuvk.physical_devices[dev], &props);

    strncpy(name, props.deviceName, (size_t)len);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDeviceTotalMem_v2
 * ============================================================================ */

CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!bytes)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || (uint32_t)dev >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(g_cuvk.physical_devices[dev],
                                        &mem_props);

    /* Sum all device-local heap sizes */
    size_t total = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += (size_t)mem_props.memoryHeaps[i].size;
        }
    }

    *bytes = total;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDeviceGetAttribute
 * ============================================================================ */

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev)
{
    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pi)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || (uint32_t)dev >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;

    VkPhysicalDevice phys = g_cuvk.physical_devices[dev];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);

    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = (int)props.limits.maxComputeWorkGroupInvocations;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
        *pi = (int)props.limits.maxComputeWorkGroupSize[0];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
        *pi = (int)props.limits.maxComputeWorkGroupSize[1];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
        *pi = (int)props.limits.maxComputeWorkGroupSize[2];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        *pi = (int)props.limits.maxComputeWorkGroupCount[0];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
        *pi = (int)props.limits.maxComputeWorkGroupCount[1];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
        *pi = (int)props.limits.maxComputeWorkGroupCount[2];
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        *pi = (int)props.limits.maxComputeSharedMemorySize;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: {
        /* Query subgroup size via VkPhysicalDeviceSubgroupProperties */
        VkPhysicalDeviceSubgroupProperties subgroup_props = {0};
        subgroup_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceProperties2 props2 = {0};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup_props;

        vkGetPhysicalDeviceProperties2(phys, &props2);
        *pi = (int)subgroup_props.subgroupSize;
        if (*pi == 0)
            *pi = 32; /* default fallback */
        return CUDA_SUCCESS;
    }
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *pi = 1; /* Not directly available in Vulkan */
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *pi = 1000000; /* Placeholder: 1 GHz in kHz */
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

/* ============================================================================
 * cuCtxCreate_v4
 * ============================================================================ */

CUresult CUDAAPI cuCtxCreate_v4(CUcontext *pctx,
                                CUctxCreateParams *ctxCreateParams,
                                unsigned int flags, CUdevice dev)
{
    (void)ctxCreateParams; /* ignored */
    (void)flags;

    if (!g_cuvk.initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || (uint32_t)dev >= g_cuvk.physical_device_count)
        return CUDA_ERROR_INVALID_DEVICE;

    VkPhysicalDevice phys = g_cuvk.physical_devices[dev];

    /* Allocate context struct */
    struct CUctx_st *ctx = (struct CUctx_st *)calloc(1, sizeof(*ctx));
    if (!ctx)
        return CUDA_ERROR_OUT_OF_MEMORY;

    ctx->instance = g_cuvk.instance;
    ctx->physical_device = phys;

    /* Find a queue family with compute support */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, NULL);

    VkQueueFamilyProperties *qf_props = NULL;
    if (qf_count > 0) {
        qf_props = (VkQueueFamilyProperties *)calloc(qf_count, sizeof(*qf_props));
        if (!qf_props) {
            free(ctx);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qf_props);
    }

    int32_t compute_family = -1;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = (int32_t)i;
            break;
        }
    }
    free(qf_props);

    if (compute_family < 0) {
        free(ctx);
        return CUDA_ERROR_NO_DEVICE;
    }

    ctx->compute_queue_family = (uint32_t)compute_family;

    /* Check for BDA support */
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {0};
    bda_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;

    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &bda_features;
    vkGetPhysicalDeviceFeatures2(phys, &features2);

    bool has_bda = (bda_features.bufferDeviceAddress == VK_TRUE);

    /* Check for timeline semaphore support */
    VkPhysicalDeviceTimelineSemaphoreFeatures ts_features = {0};
    ts_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;

    VkPhysicalDeviceFeatures2 features2_ts = {0};
    features2_ts.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2_ts.pNext = &ts_features;
    vkGetPhysicalDeviceFeatures2(phys, &features2_ts);

    bool has_timeline = (ts_features.timelineSemaphore == VK_TRUE);

    /* Build device extension list */
    const char *device_extensions[2];
    uint32_t ext_count = 0;

    if (has_bda)
        device_extensions[ext_count++] = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
    if (has_timeline)
        device_extensions[ext_count++] = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;

    /* Create VkDevice */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {0};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = ctx->compute_queue_family;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    /* Chain enabled features */
    VkPhysicalDeviceBufferDeviceAddressFeatures enable_bda = {0};
    enable_bda.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    enable_bda.bufferDeviceAddress = has_bda ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceTimelineSemaphoreFeatures enable_ts = {0};
    enable_ts.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enable_ts.timelineSemaphore = has_timeline ? VK_TRUE : VK_FALSE;

    /* Chain: device_ci -> bda -> ts */
    enable_bda.pNext = &enable_ts;

    VkDeviceCreateInfo device_ci = {0};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.pNext = &enable_bda;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.enabledExtensionCount = ext_count;
    device_ci.ppEnabledExtensionNames = device_extensions;

    VkResult vr = vkCreateDevice(phys, &device_ci, NULL, &ctx->device);
    if (vr != VK_SUCCESS) {
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Get compute queue */
    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family, 0,
                     &ctx->compute_queue);

    /* Get BDA function pointer */
    ctx->has_bda = has_bda;
    if (has_bda) {
        ctx->pfn_get_bda = (PFN_vkGetBufferDeviceAddress)
            vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddress");
    }

    /* Create command pool */
    VkCommandPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = ctx->compute_queue_family;

    vr = vkCreateCommandPool(ctx->device, &pool_ci, NULL, &ctx->cmd_pool);
    if (vr != VK_SUCCESS) {
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Create descriptor pool */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 },
    };

    VkDescriptorPoolCreateInfo desc_pool_ci = {0};
    desc_pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    desc_pool_ci.maxSets = 256;
    desc_pool_ci.poolSizeCount = 2;
    desc_pool_ci.pPoolSizes = pool_sizes;

    vr = vkCreateDescriptorPool(ctx->device, &desc_pool_ci, NULL,
                                &ctx->desc_pool);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Create timeline semaphore if supported */
    if (has_timeline) {
        VkSemaphoreTypeCreateInfo ts_ci = {0};
        ts_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        ts_ci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        ts_ci.initialValue = 0;

        VkSemaphoreCreateInfo sem_ci = {0};
        sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sem_ci.pNext = &ts_ci;

        vr = vkCreateSemaphore(ctx->device, &sem_ci, NULL,
                               &ctx->timeline_sem);
        if (vr != VK_SUCCESS) {
            vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
            vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
            vkDestroyDevice(ctx->device, NULL);
            free(ctx);
            return cuvk_vk_to_cu(vr);
        }
        ctx->timeline_value = 0;
    }

    /* Init default stream: allocate a command buffer and create a fence */
    VkCommandBufferAllocateInfo cb_ai = {0};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx->cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;

    vr = vkAllocateCommandBuffers(ctx->device, &cb_ai,
                                  &ctx->default_stream.cmd_buf);
    if (vr != VK_SUCCESS) {
        if (ctx->timeline_sem)
            vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    vr = vkCreateFence(ctx->device, &fence_ci, NULL,
                       &ctx->default_stream.fence);
    if (vr != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);
        if (ctx->timeline_sem)
            vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    ctx->default_stream.ctx = ctx;
    ctx->default_stream.recording = false;

    /* Cache memory and device properties */
    vkGetPhysicalDeviceMemoryProperties(phys, &ctx->mem_props);
    vkGetPhysicalDeviceProperties(phys, &ctx->dev_props);

    /* Set as current context */
    g_cuvk.current_ctx = ctx;
    *pctx = ctx;

    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuCtxDestroy_v2
 * ============================================================================ */

CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx)
{
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    vkDeviceWaitIdle(ctx->device);

    /* Destroy default stream resources */
    if (ctx->default_stream.fence)
        vkDestroyFence(ctx->device, ctx->default_stream.fence, NULL);
    if (ctx->default_stream.cmd_buf)
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);

    /* Destroy timeline semaphore */
    if (ctx->timeline_sem)
        vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);

    /* Destroy descriptor pool */
    if (ctx->desc_pool)
        vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);

    /* Destroy command pool */
    if (ctx->cmd_pool)
        vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);

    /* Destroy device */
    if (ctx->device)
        vkDestroyDevice(ctx->device, NULL);

    /* Free tracked allocations array */
    free(ctx->allocs);

    /* Clear current context if this was it */
    if (g_cuvk.current_ctx == ctx)
        g_cuvk.current_ctx = NULL;

    free(ctx);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuCtxGetCurrent / cuCtxSetCurrent
 * ============================================================================ */

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx)
{
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = g_cuvk.current_ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx)
{
    g_cuvk.current_ctx = ctx;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuCtxSynchronize
 * ============================================================================ */

CUresult CUDAAPI cuCtxSynchronize(void)
{
    if (!g_cuvk.current_ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    VkResult vr = vkDeviceWaitIdle(g_cuvk.current_ctx->device);
    return cuvk_vk_to_cu(vr);
}
