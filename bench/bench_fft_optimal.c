/*
 * bench_fft_optimal.c - Benchmark the FFT optimal planner
 *
 * Phase 1: For power-of-2 sizes N=2..128, find optimal CT decomposition.
 * Phase 2: For ALL sizes N=2..100, compare direct DFT vs Bluestein vs
 *          CT-optimal (pow2 only) and find the best strategy.
 *
 * Build: linked against fft_optimal_gen + wgsl_compiler + Vulkan.
 * Usage: ./bench_fft_optimal
 */

#include "fft_optimal_gen.h"
#include "simple_wgsl.h"
#include <vulkan/vulkan.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARMUP_ITERS 3
#define BENCH_ITERS  10

/* ========================================================================== */
/* Vulkan setup                                                               */
/* ========================================================================== */

typedef struct {
  VkInstance instance;
  VkPhysicalDevice phys_dev;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkDescriptorPool desc_pool;
  VkPhysicalDeviceMemoryProperties mem_props;
} VkCtx;

static int32_t find_memory_type(VkPhysicalDeviceMemoryProperties *props,
                                uint32_t type_bits,
                                VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < props->memoryTypeCount; i++) {
    if ((type_bits & (1u << i)) &&
        (props->memoryTypes[i].propertyFlags & flags) == flags)
      return (int32_t)i;
  }
  return -1;
}

static int vk_init(VkCtx *ctx) {
  memset(ctx, 0, sizeof(*ctx));

  VkApplicationInfo app_info = {0};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ci = {0};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app_info;
  if (vkCreateInstance(&ci, NULL, &ctx->instance) != VK_SUCCESS)
    return -1;

  uint32_t dev_count = 0;
  vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL);
  if (dev_count == 0) {
    fprintf(stderr, "No Vulkan devices\n");
    return -1;
  }

  VkPhysicalDevice *devs = (VkPhysicalDevice *)malloc(
      dev_count * sizeof(VkPhysicalDevice));
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
  printf("  Device: %s\n", dev_props.deviceName);

  vkGetPhysicalDeviceMemoryProperties(ctx->phys_dev, &ctx->mem_props);

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      ctx->phys_dev, &qf_count, NULL);
  VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(
      qf_count * sizeof(VkQueueFamilyProperties));
  vkGetPhysicalDeviceQueueFamilyProperties(
      ctx->phys_dev, &qf_count, qf_props);

  ctx->queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
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

  VkDeviceCreateInfo dci = {0};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;

  if (vkCreateDevice(ctx->phys_dev, &dci, NULL, &ctx->device)
      != VK_SUCCESS)
    return -1;
  vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

  VkCommandPoolCreateInfo cpci = {0};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = ctx->queue_family;
  if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool)
      != VK_SUCCESS)
    return -1;

  VkDescriptorPoolSize pool_sz = {0};
  pool_sz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sz.descriptorCount = 256;

  VkDescriptorPoolCreateInfo dpci = {0};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 256;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_sz;
  if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool)
      != VK_SUCCESS)
    return -1;

  return 0;
}

static void vk_destroy(VkCtx *ctx) {
  vkDeviceWaitIdle(ctx->device);
  vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
  vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
  vkDestroyDevice(ctx->device, NULL);
  vkDestroyInstance(ctx->instance, NULL);
}

/* ========================================================================== */
/* Pipeline creation from SPIR-V (1 binding, read_write)                      */
/* ========================================================================== */

typedef struct {
  VkShaderModule shader;
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
  VkPipeline pipeline;
} FftPipeline;

static int create_pipeline(VkCtx *ctx, uint32_t *spirv,
                           size_t spirv_count, FftPipeline *out) {
  memset(out, 0, sizeof(*out));

  VkShaderModuleCreateInfo smci = {0};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spirv_count * sizeof(uint32_t);
  smci.pCode = spirv;
  if (vkCreateShaderModule(ctx->device, &smci, NULL, &out->shader)
      != VK_SUCCESS)
    return -1;

  VkDescriptorSetLayoutBinding binding = {0};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslci = {0};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 1;
  dslci.pBindings = &binding;
  if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                   &out->desc_layout) != VK_SUCCESS)
    return -1;

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &out->desc_layout;
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                              &out->pipe_layout) != VK_SUCCESS)
    return -1;

  VkComputePipelineCreateInfo ccpci = {0};
  ccpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ccpci.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ccpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ccpci.stage.module = out->shader;
  ccpci.stage.pName = "main";
  ccpci.layout = out->pipe_layout;

  if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                &ccpci, NULL, &out->pipeline)
      != VK_SUCCESS)
    return -1;

  return 0;
}

static void destroy_pipeline(VkCtx *ctx, FftPipeline *p) {
  vkDestroyPipeline(ctx->device, p->pipeline, NULL);
  vkDestroyPipelineLayout(ctx->device, p->pipe_layout, NULL);
  vkDestroyDescriptorSetLayout(ctx->device, p->desc_layout, NULL);
  vkDestroyShaderModule(ctx->device, p->shader, NULL);
}

/* ========================================================================== */
/* Buffer helpers                                                             */
/* ========================================================================== */

typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  void *mapped;
} GpuBuffer;

static int create_buffer(VkCtx *ctx, VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags mem_flags,
                         GpuBuffer *out) {
  memset(out, 0, sizeof(*out));
  out->size = size;

  VkBufferCreateInfo bci = {0};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer)
      != VK_SUCCESS)
    return -1;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);

  int32_t mt = find_memory_type(&ctx->mem_props,
                                reqs.memoryTypeBits, mem_flags);
  if (mt < 0) return -1;

  VkMemoryAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = reqs.size;
  ai.memoryTypeIndex = (uint32_t)mt;
  if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory)
      != VK_SUCCESS)
    return -1;

  vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);

  if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    vkMapMemory(ctx->device, out->memory, 0, size, 0, &out->mapped);

  return 0;
}

static void destroy_buffer(VkCtx *ctx, GpuBuffer *b) {
  if (b->mapped) vkUnmapMemory(ctx->device, b->memory);
  vkFreeMemory(ctx->device, b->memory, NULL);
  vkDestroyBuffer(ctx->device, b->buffer, NULL);
}

/* ========================================================================== */
/* SPIR-V compilation via simple_wgsl                                         */
/* ========================================================================== */

static int compile_wgsl_to_spirv(const char *src, uint32_t **out_words,
                                 size_t *out_count) {
  WgslAstNode *ast = wgsl_parse(src);
  if (!ast) return -1;

  WgslResolver *res = wgsl_resolver_build(ast);
  if (!res) { wgsl_free_ast(ast); return -1; }

  WgslLowerOptions opts = {0};
  opts.spirv_version = 0x00010300;
  opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
  opts.packing = WGSL_LOWER_PACK_STD430;
  opts.enable_debug_names = 0;

  WgslLowerResult lr = wgsl_lower_emit_spirv(ast, res, &opts,
                                              out_words, out_count);
  wgsl_resolver_free(res);
  wgsl_free_ast(ast);
  return lr == WGSL_LOWER_OK ? 0 : -1;
}

/* ========================================================================== */
/* CPU reference DFT                                                          */
/* ========================================================================== */

static void cpu_dft(const float *in, float *out, int n) {
  for (int k = 0; k < n; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; j++) {
      double angle = -2.0 * M_PI * k * j / n;
      double wr = cos(angle), wi = sin(angle);
      double xr = in[j * 2], xi = in[j * 2 + 1];
      sr += xr * wr - xi * wi;
      si += xr * wi + xi * wr;
    }
    out[k * 2] = (float)sr;
    out[k * 2 + 1] = (float)si;
  }
}

/* ========================================================================== */
/* Timing                                                                     */
/* ========================================================================== */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
  qsort(arr, (size_t)n, sizeof(double), cmp_double);
  if (n % 2 == 0) return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
  return arr[n / 2];
}

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

/* (ilog2 removed — only needed by old FftOptPlan API) */

/* ========================================================================== */
/* Core benchmark: takes pre-generated WGSL                                   */
/* ========================================================================== */

/*
 * Benchmark a pre-generated WGSL shader for FFT size n.
 * Returns median execution time in ms, or -1.0 on failure.
 * Sets *out_correct to 1 if GPU matches CPU DFT.
 */
static double bench_wgsl(VkCtx *ctx, int n, const char *wgsl,
                          int batch_size, int *out_correct) {
  *out_correct = 0;

  /* Compile to SPIR-V */
  uint32_t *spirv = NULL;
  size_t spirv_count = 0;
  if (compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count) != 0) {
    return -1.0;
  }

  /* Create pipeline */
  FftPipeline pipe;
  if (create_pipeline(ctx, spirv, spirv_count, &pipe) != 0) {
    wgsl_lower_free(spirv);
    return -1.0;
  }
  wgsl_lower_free(spirv);

  /* Allocate buffer */
  VkDeviceSize buf_bytes = (VkDeviceSize)batch_size * n * 2 * sizeof(float);
  GpuBuffer gpu_buf;
  if (create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &gpu_buf) != 0) {
    destroy_pipeline(ctx, &pipe);
    return -1.0;
  }

  /* Fill with random data */
  float *host_data = (float *)gpu_buf.mapped;
  srand(42);
  for (int i = 0; i < batch_size * n * 2; i++)
    host_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  /* Descriptor set */
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.desc_layout;
  VkDescriptorSet ds;
  vkAllocateDescriptorSets(ctx->device, &dsai, &ds);

  VkDescriptorBufferInfo dbi = {0};
  dbi.buffer = gpu_buf.buffer;
  dbi.range = buf_bytes;
  VkWriteDescriptorSet wds = {0};
  wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wds.dstSet = ds;
  wds.dstBinding = 0;
  wds.descriptorCount = 1;
  wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wds.pBufferInfo = &dbi;
  vkUpdateDescriptorSets(ctx->device, 1, &wds, 0, NULL);

  /* Command buffer + fence */
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cb);

  VkCommandBufferBeginInfo cbbi = {0};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(ctx->device, &fci, NULL, &fence);

  VkSubmitInfo submit_info = {0};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;

  /* Verify correctness */
  srand(42);
  for (int i = 0; i < batch_size * n * 2; i++)
    host_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  float *verify_in = (float *)malloc((size_t)n * 2 * sizeof(float));
  float *verify_ref = (float *)malloc((size_t)n * 2 * sizeof(float));
  memcpy(verify_in, host_data, (size_t)n * 2 * sizeof(float));

  vkResetCommandBuffer(cb, 0);
  vkBeginCommandBuffer(cb, &cbbi);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdDispatch(cb, (uint32_t)batch_size, 1, 1);
  vkEndCommandBuffer(cb);
  vkResetFences(ctx->device, 1, &fence);
  vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
  vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

  cpu_dft(verify_in, verify_ref, n);
  float max_err = 0.0f;
  for (int i = 0; i < n * 2; i++) {
    float err = fabsf(host_data[i] - verify_ref[i]);
    if (err > max_err) max_err = err;
  }
  *out_correct = (max_err < 0.5f) ? 1 : 0;

  free(verify_in);
  free(verify_ref);

  if (!*out_correct) {
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &ds);
    destroy_buffer(ctx, &gpu_buf);
    destroy_pipeline(ctx, &pipe);
    return -1.0;
  }

  /* Benchmark */
  srand(42);
  for (int i = 0; i < batch_size * n * 2; i++)
    host_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  for (int i = 0; i < WARMUP_ITERS; i++) {
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, (uint32_t)batch_size, 1, 1);
    vkEndCommandBuffer(cb);
    vkResetFences(ctx->device, 1, &fence);
    vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
  }

  double times[BENCH_ITERS];
  for (int i = 0; i < BENCH_ITERS; i++) {
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, (uint32_t)batch_size, 1, 1);
    vkEndCommandBuffer(cb);

    vkResetFences(ctx->device, 1, &fence);
    double t_start = now_ms();
    vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    times[i] = now_ms() - t_start;
  }

  double median_time = median_d(times, BENCH_ITERS);

  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &ds);
  destroy_buffer(ctx, &gpu_buf);
  destroy_pipeline(ctx, &pipe);

  return median_time;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

static int is_pow2(int n) { return n >= 2 && (n & (n - 1)) == 0; }

static int next_pow2_bench(int n) {
  int p = 1;
  while (p < n) p *= 2;
  return p;
}

static void format_fft_plan(const FftPlan *plans, int n,
                             char *buf, int buflen) {
  FftPlan p = plans[n];
  if (p.type == FFT_PLAN_DFT) {
    snprintf(buf, buflen, "DFT%d", n);
  } else if (p.type == FFT_PLAN_CT) {
    int R = p.radix, M = n / R;
    char rbuf[128], mbuf[128];
    format_fft_plan(plans, R, rbuf, sizeof(rbuf));
    format_fft_plan(plans, M, mbuf, sizeof(mbuf));
    snprintf(buf, buflen, "(%s x %s)", rbuf, mbuf);
  } else {
    snprintf(buf, buflen, "Blue%d", n);
  }
}

static void format_fft_plan_short(const FftPlan *plans, int n,
                                   char *buf, int buflen) {
  FftPlan p = plans[n];
  if (p.type == FFT_PLAN_DFT)
    snprintf(buf, buflen, "DFT-%d", n);
  else if (p.type == FFT_PLAN_CT)
    snprintf(buf, buflen, "CT %dx%d", p.radix, n / p.radix);
  else
    snprintf(buf, buflen, "Blue-%d", n);
}

#define MAX_BENCH_N 256

int main(void) {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);

  printf("========================================================================\n");
  printf("  FFT Optimal Planner — All Sizes N=2..%d\n", MAX_BENCH_N);
  printf("========================================================================\n");

  VkCtx ctx;
  if (vk_init(&ctx) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  FftPlan plans[FFT_PLAN_MAX_N + 1];
  double plan_times[FFT_PLAN_MAX_N + 1];
  memset(plans, 0, sizeof(plans));
  for (int i = 0; i <= FFT_PLAN_MAX_N; i++) plan_times[i] = -1.0;

  /* ====================================================================== */
  /* Hardcoded results from previous benchmark run (N=2..128, pow2 to 512) */
  /* ====================================================================== */
  {
    static const struct { int n, type, radix; double time; } known[] = {
      /* Power-of-2 (Pass 1) */
      {2,FFT_PLAN_DFT,0,5.924}, {4,FFT_PLAN_DFT,0,2.136},
      {8,FFT_PLAN_CT,2,1.760}, {16,FFT_PLAN_CT,2,1.586},
      {32,FFT_PLAN_CT,2,2.119}, {64,FFT_PLAN_CT,16,2.625},
      {128,FFT_PLAN_CT,2,1.671}, {256,FFT_PLAN_CT,128,4.638},
      {512,FFT_PLAN_CT,32,6.859},
      /* Non-pow2, N=3..127 */
      {3,FFT_PLAN_BLUESTEIN,0,3.121}, {5,FFT_PLAN_BLUESTEIN,0,3.627},
      {6,FFT_PLAN_BLUESTEIN,0,3.779}, {7,FFT_PLAN_DFT,0,3.599},
      {9,FFT_PLAN_BLUESTEIN,0,2.800}, {10,FFT_PLAN_CT,5,2.077},
      {11,FFT_PLAN_BLUESTEIN,0,3.226}, {12,FFT_PLAN_BLUESTEIN,0,2.717},
      {13,FFT_PLAN_BLUESTEIN,0,3.721}, {14,FFT_PLAN_CT,7,3.174},
      {15,FFT_PLAN_BLUESTEIN,0,2.650}, {17,FFT_PLAN_DFT,0,3.431},
      {18,FFT_PLAN_CT,6,3.393}, {19,FFT_PLAN_BLUESTEIN,0,4.324},
      {20,FFT_PLAN_CT,4,2.914}, {21,FFT_PLAN_DFT,0,3.694},
      {22,FFT_PLAN_DFT,0,3.502}, {23,FFT_PLAN_DFT,0,3.804},
      {24,FFT_PLAN_CT,8,2.035}, {25,FFT_PLAN_DFT,0,3.249},
      {26,FFT_PLAN_CT,2,3.007}, {27,FFT_PLAN_CT,9,3.011},
      {28,FFT_PLAN_CT,4,3.242}, {29,FFT_PLAN_BLUESTEIN,0,3.492},
      {30,FFT_PLAN_CT,2,2.275}, {31,FFT_PLAN_BLUESTEIN,0,3.144},
      {33,FFT_PLAN_CT,11,2.999}, {34,FFT_PLAN_BLUESTEIN,0,3.461},
      {35,FFT_PLAN_CT,7,2.862}, {36,FFT_PLAN_CT,2,2.609},
      {37,FFT_PLAN_BLUESTEIN,0,4.296}, {38,FFT_PLAN_CT,2,3.294},
      {39,FFT_PLAN_BLUESTEIN,0,2.776}, {40,FFT_PLAN_CT,5,2.280},
      {41,FFT_PLAN_BLUESTEIN,0,3.109}, {42,FFT_PLAN_CT,14,1.993},
      {43,FFT_PLAN_BLUESTEIN,0,3.422}, {44,FFT_PLAN_CT,22,2.608},
      {45,FFT_PLAN_BLUESTEIN,0,3.383}, {46,FFT_PLAN_CT,2,3.424},
      {47,FFT_PLAN_BLUESTEIN,0,2.857}, {48,FFT_PLAN_CT,16,2.303},
      {49,FFT_PLAN_BLUESTEIN,0,4.010}, {50,FFT_PLAN_BLUESTEIN,0,3.178},
      {51,FFT_PLAN_BLUESTEIN,0,3.406}, {52,FFT_PLAN_CT,13,2.967},
      {53,FFT_PLAN_BLUESTEIN,0,3.472}, {54,FFT_PLAN_CT,6,3.670},
      {55,FFT_PLAN_BLUESTEIN,0,2.727}, {56,FFT_PLAN_CT,8,2.886},
      {57,FFT_PLAN_CT,3,4.320}, {58,FFT_PLAN_BLUESTEIN,0,2.736},
      {59,FFT_PLAN_BLUESTEIN,0,3.801}, {60,FFT_PLAN_CT,15,2.749},
      {61,FFT_PLAN_BLUESTEIN,0,3.662}, {62,FFT_PLAN_BLUESTEIN,0,3.368},
      {63,FFT_PLAN_CT,7,3.440}, {65,FFT_PLAN_CT,5,3.580},
      {66,FFT_PLAN_CT,6,3.255}, {67,FFT_PLAN_BLUESTEIN,0,4.591},
      {68,FFT_PLAN_CT,17,2.626}, {69,FFT_PLAN_CT,3,4.266},
      {70,FFT_PLAN_CT,2,3.378}, {71,FFT_PLAN_BLUESTEIN,0,5.659},
      {72,FFT_PLAN_CT,24,3.319}, {73,FFT_PLAN_BLUESTEIN,0,4.759},
      {74,FFT_PLAN_BLUESTEIN,0,5.905}, {75,FFT_PLAN_CT,25,4.445},
      {76,FFT_PLAN_CT,4,3.757}, {77,FFT_PLAN_CT,11,3.958},
      {78,FFT_PLAN_CT,26,3.858}, {79,FFT_PLAN_BLUESTEIN,0,6.832},
      {80,FFT_PLAN_CT,16,3.507}, {81,FFT_PLAN_CT,27,3.554},
      {82,FFT_PLAN_BLUESTEIN,0,4.008}, {83,FFT_PLAN_BLUESTEIN,0,5.705},
      {84,FFT_PLAN_CT,4,3.043}, {85,FFT_PLAN_CT,17,3.358},
      {86,FFT_PLAN_CT,2,4.528}, {87,FFT_PLAN_CT,3,4.069},
      {88,FFT_PLAN_CT,8,3.139}, {89,FFT_PLAN_BLUESTEIN,0,6.041},
      {90,FFT_PLAN_CT,3,3.524}, {91,FFT_PLAN_CT,13,3.700},
      {92,FFT_PLAN_CT,2,3.840}, {93,FFT_PLAN_CT,31,4.173},
      {94,FFT_PLAN_BLUESTEIN,0,4.272}, {95,FFT_PLAN_CT,5,5.648},
      {96,FFT_PLAN_CT,48,2.885}, {97,FFT_PLAN_BLUESTEIN,0,5.390},
      {98,FFT_PLAN_CT,7,3.600}, {99,FFT_PLAN_CT,3,3.909},
      {100,FFT_PLAN_CT,10,3.713}, {101,FFT_PLAN_BLUESTEIN,0,9.192},
      {102,FFT_PLAN_CT,2,4.246}, {103,FFT_PLAN_BLUESTEIN,0,3.938},
      {104,FFT_PLAN_CT,8,3.624}, {105,FFT_PLAN_CT,7,3.495},
      {106,FFT_PLAN_BLUESTEIN,0,4.707}, {107,FFT_PLAN_BLUESTEIN,0,4.982},
      {108,FFT_PLAN_BLUESTEIN,0,4.060}, {109,FFT_PLAN_BLUESTEIN,0,4.856},
      {110,FFT_PLAN_CT,22,3.868}, {111,FFT_PLAN_CT,37,4.219},
      {112,FFT_PLAN_CT,8,2.698}, {113,FFT_PLAN_BLUESTEIN,0,4.403},
      {114,FFT_PLAN_CT,2,3.370}, {115,FFT_PLAN_BLUESTEIN,0,4.312},
      {116,FFT_PLAN_CT,2,4.155}, {117,FFT_PLAN_CT,13,4.763},
      {118,FFT_PLAN_CT,59,4.817}, {119,FFT_PLAN_BLUESTEIN,0,4.161},
      {120,FFT_PLAN_CT,8,2.951}, {121,FFT_PLAN_BLUESTEIN,0,3.502},
      {122,FFT_PLAN_CT,2,4.733}, {123,FFT_PLAN_BLUESTEIN,0,4.182},
      {124,FFT_PLAN_BLUESTEIN,0,3.989}, {125,FFT_PLAN_BLUESTEIN,0,5.072},
      {126,FFT_PLAN_CT,42,3.409}, {127,FFT_PLAN_BLUESTEIN,0,8.817},
    };
    int nknown = sizeof(known)/sizeof(known[0]);
    for (int i = 0; i < nknown; i++) {
      int idx = known[i].n;
      plans[idx] = (FftPlan){known[i].type, known[i].radix};
      plan_times[idx] = known[i].time;
    }
    printf("Loaded %d known optimal plans (N=2..128 + pow2 to 512).\n", nknown);
  }

  /* ====================================================================== */
  /* Pass 2: Remaining sizes 129..256 (mixed-radix CT + DFT + Bluestein)    */
  /* ====================================================================== */

  printf("\n--- Pass 2: Sizes 129..%d (mixed-radix CT + DFT + Bluestein) ---\n",
         MAX_BENCH_N);

  for (int n = 129; n <= MAX_BENCH_N; n++) {
    if (is_pow2(n)) continue;

    int batch_size = 524288 / n;
    if (batch_size < 64) batch_size = 64;

    printf("\nN=%d (batch=%d):\n", n, batch_size);

    FftPlan trial[FFT_PLAN_MAX_N + 1];
    memcpy(trial, plans, sizeof(trial));

    double best_time = 1e30;
    FftPlan best_plan = {FFT_PLAN_DFT, 0};
    int any_ok = 0;

    /* Strategy 1: Direct DFT (only tiny sizes) */
    if (n <= 32) {
      trial[n] = (FftPlan){FFT_PLAN_DFT, 0};
      char *wgsl = gen_fft(n, trial, 1);
      if (wgsl) {
        int correct = 0;
        double t = bench_wgsl(&ctx, n, wgsl, batch_size, &correct);
        printf("  DFT-%-3d        %7.3f ms  %s\n", n, t, correct ? "OK" : "FAIL");
        if (correct && t > 0 && t < best_time) {
          best_time = t;
          best_plan = trial[n];
          any_ok = 1;
        }
        free(wgsl);
      }
    }

    /* Strategy 2: CT factor pairs — try both (R, M) and (M, R) for
     * each divisor R up to sqrt(N) to avoid redundant huge-shader tests */
    for (int R = 2; R * R <= n; R++) {
      if (n % R != 0) continue;
      int M = n / R;

      /* Try R x M */
      if (plan_times[R] >= 0 && plan_times[M] >= 0) {
        trial[n] = (FftPlan){FFT_PLAN_CT, R};
        char *wgsl = gen_fft(n, trial, 1);
        if (wgsl) {
          int correct = 0;
          double t = bench_wgsl(&ctx, n, wgsl, batch_size, &correct);
          printf("  CT  %3dx%-3d    %7.3f ms  %s\n",
                 R, M, t, correct ? "OK" : "FAIL");
          if (correct && t > 0 && t < best_time) {
            best_time = t;
            best_plan = trial[n];
            any_ok = 1;
          }
          free(wgsl);
        }
      }

      /* Try M x R (skip if R == M, i.e. perfect square) */
      if (R != M && plan_times[M] >= 0 && plan_times[R] >= 0) {
        trial[n] = (FftPlan){FFT_PLAN_CT, M};
        char *wgsl = gen_fft(n, trial, 1);
        if (wgsl) {
          int correct = 0;
          double t = bench_wgsl(&ctx, n, wgsl, batch_size, &correct);
          printf("  CT  %3dx%-3d    %7.3f ms  %s\n",
                 M, R, t, correct ? "OK" : "FAIL");
          if (correct && t > 0 && t < best_time) {
            best_time = t;
            best_plan = trial[n];
            any_ok = 1;
          }
          free(wgsl);
        }
      }
    }

    /* Strategy 3: Bluestein (needed for primes; skip if CT already works) */
    if (!any_ok) {
      int M = next_pow2_bench(2 * n - 1);
      if (M <= FFT_PLAN_MAX_N && plan_times[M] > 0) {
        trial[n] = (FftPlan){FFT_PLAN_BLUESTEIN, 0};
        char *wgsl = gen_fft(n, trial, 1);
        if (wgsl) {
          int correct = 0;
          double t = bench_wgsl(&ctx, n, wgsl, batch_size, &correct);
          printf("  Blue (M=%d)   %7.3f ms  %s\n",
                 M, t, correct ? "OK" : "FAIL");
          if (correct && t > 0 && t < best_time) {
            best_time = t;
            best_plan = trial[n];
            any_ok = 1;
          }
          free(wgsl);
        }
      }
    }

    if (any_ok) {
      plans[n] = best_plan;
      plan_times[n] = best_time;
      char ps[64];
      format_fft_plan_short(plans, n, ps, sizeof(ps));
      printf("  -> WINNER: %s (%.3f ms)\n", ps, best_time);
    } else {
      printf("  -> NO WORKING STRATEGY\n");
    }
  }

  /* ====================================================================== */
  /* Summary                                                                */
  /* ====================================================================== */

  printf("\n========================================================================\n");
  printf("  Optimal Plan Table: N=2..%d\n", MAX_BENCH_N);
  printf("========================================================================\n");
  printf("  %-5s  %-14s  %8s  %s\n", "N", "Strategy", "Time(ms)", "Decomposition");
  printf("  %-5s  %-14s  %8s  %s\n", "-----", "--------------", "--------",
         "---------------------------");

  int n_ok = 0, n_fail = 0;
  for (int n = 2; n <= MAX_BENCH_N; n++) {
    char ps[64], full[256];
    if (plan_times[n] > 0) {
      format_fft_plan_short(plans, n, ps, sizeof(ps));
      format_fft_plan(plans, n, full, sizeof(full));
      printf("  %-5d  %-14s  %8.3f  %s\n", n, ps, plan_times[n], full);
      n_ok++;
    } else {
      printf("  %-5d  %-14s  %8s\n", n, "NONE", "---");
      n_fail++;
    }
  }

  printf("\n%d/%d sizes covered.\n", n_ok, n_ok + n_fail);

  /* C initializer */
  printf("\nFftPlan optimal[FFT_PLAN_MAX_N + 1] = {\n");
  for (int n = 0; n <= MAX_BENCH_N; n++) {
    const char *ts = plans[n].type == FFT_PLAN_DFT ? "FFT_PLAN_DFT"
                   : plans[n].type == FFT_PLAN_CT  ? "FFT_PLAN_CT"
                   : "FFT_PLAN_BLUESTEIN";
    printf("  [%3d] = {%s, %d},\n", n, ts, plans[n].radix);
  }
  printf("};\n");

  vk_destroy(&ctx);
  return 0;
}
