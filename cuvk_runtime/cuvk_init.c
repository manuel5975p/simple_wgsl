/*
 * cuvk_init.c - Vulkan bootstrap, device enumeration, and context management
 *
 * Implements the CUDA driver API functions: cuInit, cuDriverGetVersion,
 * cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDeviceTotalMem_v2,
 * cuDeviceGetAttribute, cuCtxCreate_v4, cuCtxDestroy_v2, cuCtxGetCurrent,
 * cuCtxSetCurrent, cuCtxSynchronize, and the cuvk_find_memory_type helper.
 */

#include "cuvk_internal.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global runtime singleton */
CuvkGlobal g_cuvk = {0};

/* ============================================================================
 * Vulkan function pointer loading (bootstrap via dlsym)
 * ============================================================================ */

static VkResult cuvk_load_vk(void)
{
#ifdef _WIN32
    HMODULE module = LoadLibraryA("vulkan-1.dll");
    if (!module)
        return VK_ERROR_INITIALIZATION_FAILED;
    g_cuvk.vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
        GetProcAddress(module, "vkGetInstanceProcAddr");
#elif defined(__APPLE__)
    void *module = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!module)
        module = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!module)
        return VK_ERROR_INITIALIZATION_FAILED;
    g_cuvk.vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
        dlsym(module, "vkGetInstanceProcAddr");
#else
    void *module = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!module)
        module = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!module)
        return VK_ERROR_INITIALIZATION_FAILED;
    g_cuvk.vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
        dlsym(module, "vkGetInstanceProcAddr");
#endif
    if (!g_cuvk.vk.vkGetInstanceProcAddr)
        return VK_ERROR_INITIALIZATION_FAILED;

    /* Load pre-instance (global) functions */
#define CUVK_VK_LOAD_GLOBAL(fn) \
    g_cuvk.vk.fn = (PFN_##fn)g_cuvk.vk.vkGetInstanceProcAddr(NULL, #fn);
    CUVK_GLOBAL_FUNCS(CUVK_VK_LOAD_GLOBAL)
#undef CUVK_VK_LOAD_GLOBAL

    return VK_SUCCESS;
}

static void cuvk_load_instance(VkInstance instance)
{
#define CUVK_VK_LOAD_INST(fn) \
    g_cuvk.vk.fn = (PFN_##fn)g_cuvk.vk.vkGetInstanceProcAddr(instance, #fn);
    CUVK_INSTANCE_FUNCS(CUVK_VK_LOAD_INST)
#undef CUVK_VK_LOAD_INST
}

static void cuvk_load_device(VkDevice device)
{
#define CUVK_VK_LOAD_DEV(fn) \
    g_cuvk.vk.fn = (PFN_##fn)g_cuvk.vk.vkGetDeviceProcAddr(device, #fn);
    CUVK_DEVICE_FUNCS(CUVK_VK_LOAD_DEV)
#undef CUVK_VK_LOAD_DEV
}

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
 * Vulkan debug callback
 * ============================================================================ */

static VKAPI_ATTR VkBool32 VKAPI_CALL cuvk_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *user)
{
    (void)type; (void)user;
    const char *sev = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                          ? "ERROR" : "WARN";
    fprintf(stderr, "[cuvk/vk] %s: %s\n", sev, data->pMessage);
    return VK_FALSE;
}

static void cuvk_atexit_handler(void) { g_cuvk.exiting = true; }

/* ============================================================================
 * cuInit
 * ============================================================================ */

CUresult CUDAAPI cuInit(unsigned int Flags)
{
    (void)Flags;
    CUVK_LOG("[cuvk] cuInit called (initialized=%d)\n",
            g_cuvk.initialized);

    if (g_cuvk.initialized)
        return CUDA_SUCCESS;

    /* Bootstrap Vulkan function pointers via dlsym */
    VkResult vr_load = cuvk_load_vk();
    if (vr_load != VK_SUCCESS)
        return cuvk_vk_to_cu(vr_load);

    bool want_validation = false;
#ifndef NDEBUG
    if (getenv("CUVK_VALIDATION"))
        want_validation = true;
#endif

    const char *instance_extensions[2];
    uint32_t inst_ext_count = 0;
    instance_extensions[inst_ext_count++] =
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
    if (want_validation)
        instance_extensions[inst_ext_count++] =
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    const char *validation_layers[] = {
        "VK_LAYER_KHRONOS_validation",
    };

    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CUDA-on-Vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "cuvk_runtime";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkDebugUtilsMessengerCreateInfoEXT dbg_ci = {0};
    dbg_ci.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg_ci.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg_ci.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg_ci.pfnUserCallback = cuvk_debug_callback;

    const char *log_action = "VK_DBG_LAYER_ACTION_LOG_MSG";
    VkBool32 vk_false = VK_FALSE;
    VkLayerSettingEXT layer_settings[] = {
        {
            .pLayerName   = "VK_LAYER_KHRONOS_validation",
            .pSettingName = "debug_action",
            .type         = VK_LAYER_SETTING_TYPE_STRING_EXT,
            .valueCount   = 1,
            .pValues      = &log_action,
        },
        {
            .pLayerName   = "VK_LAYER_KHRONOS_validation",
            .pSettingName = "thread_safety",
            .type         = VK_LAYER_SETTING_TYPE_BOOL32_EXT,
            .valueCount   = 1,
            .pValues      = &vk_false,
        },
    };
    VkLayerSettingsCreateInfoEXT layer_settings_ci = {0};
    layer_settings_ci.sType =
        VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
    layer_settings_ci.settingCount = 2;
    layer_settings_ci.pSettings = layer_settings;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = inst_ext_count;
    create_info.ppEnabledExtensionNames = instance_extensions;
    if (want_validation) {
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = validation_layers;
        dbg_ci.pNext = &layer_settings_ci;
        create_info.pNext = &dbg_ci;
    }

    VkResult vr = g_cuvk.vk.vkCreateInstance(&create_info, NULL, &g_cuvk.instance);
    if (vr != VK_SUCCESS) {
        CUVK_LOG("[cuvk] cuInit: vkCreateInstance failed vr=%d\n", vr);
        return cuvk_vk_to_cu(vr);
    }

    /* Load instance-level function pointers */
    cuvk_load_instance(g_cuvk.instance);

    g_cuvk.has_validation = want_validation;
    if (want_validation) {
        PFN_vkCreateDebugUtilsMessengerEXT fn =
            (PFN_vkCreateDebugUtilsMessengerEXT)g_cuvk.vk.vkGetInstanceProcAddr(
                g_cuvk.instance, "vkCreateDebugUtilsMessengerEXT");
        if (fn) {
            fn(g_cuvk.instance, &dbg_ci, NULL,
               &g_cuvk.debug_messenger);
        }
    }

    /* Enumerate physical devices */
    uint32_t count = 0;
    vr = g_cuvk.vk.vkEnumeratePhysicalDevices(g_cuvk.instance, &count, NULL);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return cuvk_vk_to_cu(vr);
    }

    if (count == 0) {
        g_cuvk.vk.vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return CUDA_ERROR_NO_DEVICE;
    }

    if (count > CUVK_MAX_PHYSICAL_DEVICES)
        count = CUVK_MAX_PHYSICAL_DEVICES;

    vr = g_cuvk.vk.vkEnumeratePhysicalDevices(g_cuvk.instance, &count,
                                    g_cuvk.physical_devices);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) {
        g_cuvk.vk.vkDestroyInstance(g_cuvk.instance, NULL);
        g_cuvk.instance = VK_NULL_HANDLE;
        return cuvk_vk_to_cu(vr);
    }

    g_cuvk.physical_device_count = count;
    g_cuvk.initialized = true;
    atexit(cuvk_atexit_handler);
    CUVK_LOG("[cuvk] cuInit SUCCESS (devices=%u)\n", count);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuDriverGetVersion
 * ============================================================================ */

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion)
{
    if (!driverVersion)
        return CUDA_ERROR_INVALID_VALUE;
    CUVK_LOG("[cuvk] cuDriverGetVersion called\n");
    *driverVersion = 13020;
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
    CUVK_LOG("[cuvk] cuDeviceGetCount -> %d\n", *count);
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
    g_cuvk.vk.vkGetPhysicalDeviceProperties(g_cuvk.physical_devices[dev], &props);

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
    g_cuvk.vk.vkGetPhysicalDeviceMemoryProperties(g_cuvk.physical_devices[dev],
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
    g_cuvk.vk.vkGetPhysicalDeviceProperties(phys, &props);

    CUVK_LOG("[cuvk] cuDeviceGetAttribute(attrib=%d, dev=%d)\n", attrib, dev);
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
        *pi = 7;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
        *pi = 5;
        return CUDA_SUCCESS;
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

        g_cuvk.vk.vkGetPhysicalDeviceProperties2(phys, &props2);
        *pi = (int)subgroup_props.subgroupSize;
        if (*pi == 0)
            *pi = 32; /* default fallback */
        return CUDA_SUCCESS;
    }
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *pi = 68;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
        *pi = 2048;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
        *pi = 32;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
        *pi = 3;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
        *pi = 65536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
        *pi = 2147483647;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
        *pi = 512;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *pi = 1410000;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
        *pi = 9501000;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
        *pi = 384;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
        *pi = 6291456;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
        *pi = 65536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
        *pi = 163840;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
        *pi = 65536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
        *pi = 163840;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO:
        *pi = 2;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
        *pi = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:
        *pi = 0;
        return CUDA_SUCCESS;
    default:
        *pi = 0;
        return CUDA_SUCCESS;
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
    g_cuvk.vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, NULL);

    VkQueueFamilyProperties *qf_props = NULL;
    if (qf_count > 0) {
        qf_props = (VkQueueFamilyProperties *)calloc(qf_count, sizeof(*qf_props));
        if (!qf_props) {
            free(ctx);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        g_cuvk.vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qf_props);
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
    g_cuvk.vk.vkGetPhysicalDeviceFeatures2(phys, &features2);

    bool has_bda = (bda_features.bufferDeviceAddress == VK_TRUE);

    /* Check for timeline semaphore support */
    VkPhysicalDeviceTimelineSemaphoreFeatures ts_features = {0};
    ts_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;

    VkPhysicalDeviceFeatures2 features2_ts = {0};
    features2_ts.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2_ts.pNext = &ts_features;
    g_cuvk.vk.vkGetPhysicalDeviceFeatures2(phys, &features2_ts);

    bool has_timeline = (ts_features.timelineSemaphore == VK_TRUE);
    bool has_f64 = (features2.features.shaderFloat64 == VK_TRUE);
    bool has_i64 = (features2.features.shaderInt64 == VK_TRUE);

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

    VkPhysicalDeviceFeatures2 enable_features2 = {0};
    enable_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enable_features2.features.shaderFloat64 = has_f64 ? VK_TRUE : VK_FALSE;
    enable_features2.features.shaderInt64 = has_i64 ? VK_TRUE : VK_FALSE;

    /* Chain: device_ci -> features2 -> bda -> ts */
    enable_features2.pNext = &enable_bda;
    enable_bda.pNext = &enable_ts;

    VkDeviceCreateInfo device_ci = {0};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.pNext = &enable_features2;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.enabledExtensionCount = ext_count;
    device_ci.ppEnabledExtensionNames = device_extensions;

    VkResult vr = g_cuvk.vk.vkCreateDevice(phys, &device_ci, NULL, &ctx->device);
    if (vr != VK_SUCCESS) {
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Load device-level function pointers */
    cuvk_load_device(ctx->device);

    /* Get compute queue */
    g_cuvk.vk.vkGetDeviceQueue(ctx->device, ctx->compute_queue_family, 0,
                     &ctx->compute_queue);

    /* Get BDA function pointer */
    ctx->has_bda = has_bda;
    if (has_bda) {
        ctx->pfn_get_bda = (PFN_vkGetBufferDeviceAddress)
            g_cuvk.vk.vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddress");
        if (!ctx->pfn_get_bda) {
            ctx->pfn_get_bda = (PFN_vkGetBufferDeviceAddress)
                g_cuvk.vk.vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddressKHR");
        }
    }

    /* Create command pool */
    VkCommandPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = ctx->compute_queue_family;

    vr = g_cuvk.vk.vkCreateCommandPool(ctx->device, &pool_ci, NULL, &ctx->cmd_pool);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
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

    vr = g_cuvk.vk.vkCreateDescriptorPool(ctx->device, &desc_pool_ci, NULL,
                                &ctx->desc_pool);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
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

        vr = g_cuvk.vk.vkCreateSemaphore(ctx->device, &sem_ci, NULL,
                               &ctx->timeline_sem);
        if (vr != VK_SUCCESS) {
            g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
            g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
            g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
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

    vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cb_ai,
                                  &ctx->default_stream.cmd_buf);
    if (vr != VK_SUCCESS) {
        if (ctx->timeline_sem)
            g_cuvk.vk.vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    vr = g_cuvk.vk.vkCreateFence(ctx->device, &fence_ci, NULL,
                       &ctx->default_stream.fence);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);
        if (ctx->timeline_sem)
            g_cuvk.vk.vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    ctx->default_stream.ctx = ctx;
    ctx->default_stream.recording = false;

    /* Allocate reusable one-shot command buffer */
    VkCommandBufferAllocateInfo oneshot_ai = {0};
    oneshot_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    oneshot_ai.commandPool = ctx->cmd_pool;
    oneshot_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    oneshot_ai.commandBufferCount = 1;

    vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &oneshot_ai, &ctx->oneshot_cb);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyFence(ctx->device, ctx->default_stream.fence, NULL);
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);
        if (ctx->timeline_sem)
            g_cuvk.vk.vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Create reusable one-shot fence (signaled so first reset succeeds) */
    VkFenceCreateInfo oneshot_fence_ci = {0};
    oneshot_fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    oneshot_fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vr = g_cuvk.vk.vkCreateFence(ctx->device, &oneshot_fence_ci, NULL,
                       &ctx->oneshot_fence);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &ctx->oneshot_cb);
        g_cuvk.vk.vkDestroyFence(ctx->device, ctx->default_stream.fence, NULL);
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);
        if (ctx->timeline_sem)
            g_cuvk.vk.vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
        free(ctx);
        return cuvk_vk_to_cu(vr);
    }

    /* Cache memory and device properties */
    g_cuvk.vk.vkGetPhysicalDeviceMemoryProperties(phys, &ctx->mem_props);
    g_cuvk.vk.vkGetPhysicalDeviceProperties(phys, &ctx->dev_props);

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
    if (!ctx->device)
        return CUDA_SUCCESS;

    if (g_cuvk.exiting && g_cuvk.has_validation) {
        ctx->device = VK_NULL_HANDLE;
        if (g_cuvk.current_ctx == ctx)
            g_cuvk.current_ctx = NULL;
        return CUDA_SUCCESS;
    }

    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);

    /* Destroy default stream resources */
    if (ctx->default_stream.fence)
        g_cuvk.vk.vkDestroyFence(ctx->device, ctx->default_stream.fence, NULL);
    if (ctx->default_stream.cmd_buf)
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->default_stream.cmd_buf);

    /* Destroy reusable one-shot resources */
    if (ctx->oneshot_fence)
        g_cuvk.vk.vkDestroyFence(ctx->device, ctx->oneshot_fence, NULL);
    if (ctx->oneshot_cb)
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1,
                             &ctx->oneshot_cb);

    /* Destroy cached staging buffer */
    if (ctx->staging_buf)
        g_cuvk.vk.vkDestroyBuffer(ctx->device, ctx->staging_buf, NULL);
    if (ctx->staging_mem)
        g_cuvk.vk.vkFreeMemory(ctx->device, ctx->staging_mem, NULL);

    /* Destroy download staging buffer */
    if (ctx->download_buf)
        g_cuvk.vk.vkDestroyBuffer(ctx->device, ctx->download_buf, NULL);
    if (ctx->download_mem)
        g_cuvk.vk.vkFreeMemory(ctx->device, ctx->download_mem, NULL);

    /* Destroy timeline semaphore */
    if (ctx->timeline_sem)
        g_cuvk.vk.vkDestroySemaphore(ctx->device, ctx->timeline_sem, NULL);

    /* Destroy descriptor pool */
    if (ctx->desc_pool)
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);

    /* Destroy pinned host allocations */
    for (uint32_t i = 0; i < ctx->host_alloc_count; i++) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, ctx->host_allocs[i].buffer, NULL);
        g_cuvk.vk.vkFreeMemory(ctx->device, ctx->host_allocs[i].memory, NULL);
    }
    free(ctx->host_allocs);
    ctx->host_allocs = NULL;

    /* Destroy command pool */
    if (ctx->cmd_pool)
        g_cuvk.vk.vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);

    /* Destroy device */
    g_cuvk.vk.vkDestroyDevice(ctx->device, NULL);
    ctx->device = VK_NULL_HANDLE;

    /* Free tracked allocations array */
    free(ctx->allocs);
    ctx->allocs = NULL;

    /* Clear current context if this was it */
    if (g_cuvk.current_ctx == ctx)
        g_cuvk.current_ctx = NULL;

    /* NOTE: do not free(ctx) here. Modules hold a ctx pointer and may be
     * unloaded after the context during atexit. We null device above so
     * cuModuleUnload can detect the destroyed context and skip cleanup.
     * The struct leaks at process exit, which is harmless. */
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

    /* Flush default stream (submit any pending commands) */
    cuvk_stream_submit_and_wait(&g_cuvk.current_ctx->default_stream);

    /* Flush deferred cuFFT submissions */
    cuvk_fft_flush(g_cuvk.current_ctx);

    VkResult vr = g_cuvk.vk.vkDeviceWaitIdle(g_cuvk.current_ctx->device);
    return cuvk_vk_to_cu(vr);
}
