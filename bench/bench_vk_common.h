/*
 * bench_vk_common.h - Shared Vulkan boilerplate for benchmarks
 */
#ifndef BENCH_VK_COMMON_H
#define BENCH_VK_COMMON_H

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  VkInstance instance;
  VkPhysicalDevice phys_dev;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorPool desc_pool;
  VkPhysicalDeviceMemoryProperties mem_props;
  VkQueryPool timestamp_pool;
  float timestamp_period; /* ns per tick */
  PFN_vkGetBufferDeviceAddress pfn_get_bda;
} VkCtx;

static inline int32_t find_memory_type(VkPhysicalDeviceMemoryProperties *props,
                                       uint32_t type_bits,
                                       VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) &&
        (props->memoryTypes[i].propertyFlags & flags) == flags)
      return (int32_t)i;
  }
  return -1;
}

static inline int vk_init(VkCtx *ctx, uint32_t max_sets,
                           uint32_t desc_count, int create_timestamp_pool) {
  memset(ctx, 0, sizeof(*ctx));

  VkApplicationInfo app_info = {0};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.apiVersion = VK_API_VERSION_1_2;

  const char *inst_exts[] = {
    "VK_KHR_portability_enumeration",
  };

  VkInstanceCreateInfo ci = {0};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app_info;
  ci.flags = 0x00000001; /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
  ci.enabledExtensionCount = 1;
  ci.ppEnabledExtensionNames = inst_exts;
  if (vkCreateInstance(&ci, NULL, &ctx->instance) != VK_SUCCESS)
    return -1;

  uint32_t dev_count = 0;
  vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL);
  if (dev_count == 0) return -1;

  VkPhysicalDevice *devs = malloc(dev_count * sizeof(VkPhysicalDevice));
  vkEnumeratePhysicalDevices(ctx->instance, &dev_count, devs);

  ctx->phys_dev = devs[0];
  for (uint32_t i = 0; i < dev_count; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[i], &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      ctx->phys_dev = devs[i];
      break;
    }
  }
  free(devs);

  VkPhysicalDeviceProperties dev_props;
  vkGetPhysicalDeviceProperties(ctx->phys_dev, &dev_props);
  ctx->timestamp_period = dev_props.limits.timestampPeriod;
  printf("Device: %s (timestamp period: %.1f ns)\n",
         dev_props.deviceName, ctx->timestamp_period);

  vkGetPhysicalDeviceMemoryProperties(ctx->phys_dev, &ctx->mem_props);

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = malloc(
      qf_count * sizeof(VkQueueFamilyProperties));
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, qf_props);

  ctx->queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        qf_props[i].timestampValidBits > 0) {
      ctx->queue_family = i;
      break;
    }
  }
  free(qf_props);
  if (ctx->queue_family == UINT32_MAX) return -1;

  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci = {0};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = ctx->queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  /* Enable BDA (required for var<device> FFT shaders) */
  VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {0};
  bda_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  bda_features.bufferDeviceAddress = VK_TRUE;

  VkDeviceCreateInfo dci = {0};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = &bda_features;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  if (vkCreateDevice(ctx->phys_dev, &dci, NULL, &ctx->device) != VK_SUCCESS)
    return -1;
  vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

  ctx->pfn_get_bda = (PFN_vkGetBufferDeviceAddress)
      vkGetDeviceProcAddr(ctx->device, "vkGetBufferDeviceAddress");

  VkCommandPoolCreateInfo cpci = {0};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = ctx->queue_family;
  if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool)
      != VK_SUCCESS)
    return -1;

  VkDescriptorPoolSize pool_sz = {0};
  pool_sz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sz.descriptorCount = desc_count;

  VkDescriptorPoolCreateInfo dpci = {0};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = max_sets;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_sz;
  if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool)
      != VK_SUCCESS)
    return -1;

  if (create_timestamp_pool) {
    VkQueryPoolCreateInfo qpci = {0};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    if (vkCreateQueryPool(ctx->device, &qpci, NULL, &ctx->timestamp_pool)
        != VK_SUCCESS)
      return -1;
  }

  return 0;
}

static inline void vk_destroy(VkCtx *ctx) {
  vkDeviceWaitIdle(ctx->device);
  if (ctx->timestamp_pool)
    vkDestroyQueryPool(ctx->device, ctx->timestamp_pool, NULL);
  vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
  vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
  vkDestroyDevice(ctx->device, NULL);
  vkDestroyInstance(ctx->instance, NULL);
}

#endif /* BENCH_VK_COMMON_H */
