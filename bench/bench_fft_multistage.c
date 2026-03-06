/*
 * bench_fft_multistage.c - Benchmark multi-stage Stockham FFT (single submit)
 *
 * For each FFT size N, factorizes into radices, generates + compiles all stage
 * shaders, records a single command buffer with all stages ping-ponged,
 * and benchmarks across batch sizes.
 *
 * Usage: ./bench_fft_multistage
 */

#include "fft_stockham_gen.h"
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

#define MAX_STAGES     20
#define WORKGROUP_SIZE 256
#define MAX_RADIX      32
#define WARMUP_ITERS   3
#define BENCH_ITERS    10

/* ========================================================================== */
/* Helpers                                                                     */
/* ========================================================================== */

static int factorize(int n, int *radices, int max_stages) {
  int count = 0, rem = n;
  while (rem > 1 && count < max_stages) {
    int best = 0;
    for (int r = MAX_RADIX; r >= 2; r--)
      if (rem % r == 0) { best = r; break; }
    if (best == 0) return 0;
    radices[count++] = best;
    rem /= best;
  }
  return (rem == 1) ? count : 0;
}

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

static void cpu_dft(const float *in, float *out, int n) {
  for (int k = 0; k < n; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; j++) {
      double angle = -2.0 * M_PI * (double)k * (double)j / (double)n;
      sr += (double)in[j * 2] * cos(angle) - (double)in[j * 2 + 1] * sin(angle);
      si += (double)in[j * 2] * sin(angle) + (double)in[j * 2 + 1] * cos(angle);
    }
    out[k * 2] = (float)sr;
    out[k * 2 + 1] = (float)si;
  }
}

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
  WgslLowerResult lr = wgsl_lower_emit_spirv(ast, res, &opts,
                                              out_words, out_count);
  wgsl_resolver_free(res);
  wgsl_free_ast(ast);
  return lr == WGSL_LOWER_OK ? 0 : -1;
}

/* ========================================================================== */
/* Vulkan setup (same as bench_fft_stockham.c)                                */
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
  if (vkCreateInstance(&ci, NULL, &ctx->instance) != VK_SUCCESS) return -1;

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
      ctx->phys_dev = devs[i]; break;
    }
  }
  free(devs);

  VkPhysicalDeviceProperties dev_props;
  vkGetPhysicalDeviceProperties(ctx->phys_dev, &dev_props);
  printf("  Device: %s\n", dev_props.deviceName);
  vkGetPhysicalDeviceMemoryProperties(ctx->phys_dev, &ctx->mem_props);

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, NULL);
  VkQueueFamilyProperties *qf = malloc(qf_count * sizeof(*qf));
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys_dev, &qf_count, qf);
  ctx->queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx->queue_family = i; break; }
  free(qf);
  if (ctx->queue_family == UINT32_MAX) return -1;

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {0};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = ctx->queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {0};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  if (vkCreateDevice(ctx->phys_dev, &dci, NULL, &ctx->device) != VK_SUCCESS)
    return -1;
  vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

  VkCommandPoolCreateInfo cpci = {0};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = ctx->queue_family;
  if (vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool) != VK_SUCCESS)
    return -1;

  VkDescriptorPoolSize pool_sz = {0};
  pool_sz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sz.descriptorCount = 1024;
  VkDescriptorPoolCreateInfo dpci = {0};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 512;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_sz;
  if (vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool) != VK_SUCCESS)
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

typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  void *mapped;
} GpuBuffer;

static int create_buffer(VkCtx *ctx, VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags mem_flags, GpuBuffer *out) {
  memset(out, 0, sizeof(*out));
  out->size = size;
  VkBufferCreateInfo bci = {0};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size; bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer) != VK_SUCCESS) return -1;
  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);
  int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits, mem_flags);
  if (mt < 0) return -1;
  VkMemoryAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = reqs.size; ai.memoryTypeIndex = (uint32_t)mt;
  if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory) != VK_SUCCESS) return -1;
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
/* Multi-stage pipeline setup                                                  */
/* ========================================================================== */

typedef struct {
  int n;
  int n_stages;
  int radices[MAX_STAGES];
  uint32_t dispatch_x[MAX_STAGES];

  VkShaderModule shaders[MAX_STAGES];
  VkPipeline pipelines[MAX_STAGES];
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
} FftPlan;

static int plan_create(VkCtx *ctx, int n, FftPlan *plan) {
  memset(plan, 0, sizeof(*plan));
  plan->n = n;
  plan->n_stages = factorize(n, plan->radices, MAX_STAGES);
  if (plan->n_stages == 0) return -1;

  /* Shared descriptor layout: 2 storage buffer bindings */
  VkDescriptorSetLayoutBinding bindings[2] = {{0}};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslci = {0};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2;
  dslci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                   &plan->desc_layout) != VK_SUCCESS)
    return -1;

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &plan->desc_layout;
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                              &plan->pipe_layout) != VK_SUCCESS)
    return -1;

  int stride = 1;
  for (int s = 0; s < plan->n_stages; s++) {
    int radix = plan->radices[s];
    plan->dispatch_x[s] = ((uint32_t)(n / radix) + WORKGROUP_SIZE - 1) /
                           WORKGROUP_SIZE;

    char *wgsl = gen_fft_stockham(radix, stride, n, 1, WORKGROUP_SIZE);
    if (!wgsl) return -1;

    uint32_t *spirv = NULL;
    size_t spirv_count = 0;
    int rc = compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count);
    free(wgsl);
    if (rc != 0) return -1;

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv_count * sizeof(uint32_t);
    smci.pCode = spirv;
    VkResult vr = vkCreateShaderModule(ctx->device, &smci, NULL,
                                        &plan->shaders[s]);
    wgsl_lower_free(spirv);
    if (vr != VK_SUCCESS) return -1;

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = plan->shaders[s];
    cpci.stage.pName = "main";
    cpci.layout = plan->pipe_layout;
    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                   &cpci, NULL, &plan->pipelines[s]);
    if (vr != VK_SUCCESS) return -1;

    stride *= radix;
  }
  return 0;
}

static void plan_destroy(VkCtx *ctx, FftPlan *plan) {
  for (int s = 0; s < plan->n_stages; s++) {
    if (plan->pipelines[s])
      vkDestroyPipeline(ctx->device, plan->pipelines[s], NULL);
    if (plan->shaders[s])
      vkDestroyShaderModule(ctx->device, plan->shaders[s], NULL);
  }
  if (plan->pipe_layout)
    vkDestroyPipelineLayout(ctx->device, plan->pipe_layout, NULL);
  if (plan->desc_layout)
    vkDestroyDescriptorSetLayout(ctx->device, plan->desc_layout, NULL);
}

/* ========================================================================== */
/* Benchmark one FFT size (batch=1)                                            */
/* ========================================================================== */

static int bench_size(VkCtx *ctx, int n, double *out_compile_ms,
                      double *out_exec_ms) {
  double t0 = now_ms();
  FftPlan plan;
  if (plan_create(ctx, n, &plan) != 0) {
    fprintf(stderr, "  N=%-7d PLAN FAILED\n", n);
    return -1;
  }
  *out_compile_ms = now_ms() - t0;
  int ns = plan.n_stages;

  VkDeviceSize buf_bytes = (VkDeviceSize)n * 2 * sizeof(float);
  GpuBuffer src_buf, dst_buf, scratch_buf;
  VkMemoryPropertyFlags mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    mem_flags, &src_buf) != 0 ||
      create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    mem_flags, &dst_buf) != 0 ||
      create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    mem_flags, &scratch_buf) != 0) {
    plan_destroy(ctx, &plan);
    return -1;
  }

  /* Fill src with random data */
  float *src_data = (float *)src_buf.mapped;
  srand(42);
  for (int i = 0; i < n * 2; i++)
    src_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  /* Set up descriptor sets with ping-pong */
  VkDescriptorSet ds[MAX_STAGES];
  VkDescriptorSetLayout layouts[MAX_STAGES];
  for (int i = 0; i < ns; i++) layouts[i] = plan.desc_layout;
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = (uint32_t)ns;
  dsai.pSetLayouts = layouts;
  vkAllocateDescriptorSets(ctx->device, &dsai, ds);

  VkBuffer read_bufs[MAX_STAGES], write_bufs[MAX_STAGES];
  write_bufs[ns - 1] = dst_buf.buffer;
  for (int i = ns - 2; i >= 0; i--)
    write_bufs[i] = (write_bufs[i + 1] == dst_buf.buffer)
                   ? scratch_buf.buffer : dst_buf.buffer;
  read_bufs[0] = src_buf.buffer;
  for (int i = 1; i < ns; i++)
    read_bufs[i] = write_bufs[i - 1];

  for (int i = 0; i < ns; i++) {
    VkDescriptorBufferInfo dbi[2] = {{0}};
    dbi[0].buffer = read_bufs[i]; dbi[0].range = buf_bytes;
    dbi[1].buffer = write_bufs[i]; dbi[1].range = buf_bytes;
    VkWriteDescriptorSet wds[2] = {{0}};
    for (int b = 0; b < 2; b++) {
      wds[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wds[b].dstSet = ds[i]; wds[b].dstBinding = (uint32_t)b;
      wds[b].descriptorCount = 1;
      wds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      wds[b].pBufferInfo = &dbi[b];
    }
    vkUpdateDescriptorSets(ctx->device, 2, wds, 0, NULL);
  }

  /* Record command buffer */
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cb);

  VkCommandBufferBeginInfo cbbi = {0};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cb, &cbbi);
  VkMemoryBarrier bar = {0};
  bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  for (int i = 0; i < ns; i++) {
    if (i > 0)
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &bar, 0, NULL, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      plan.pipelines[i]);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            plan.pipe_layout, 0, 1, &ds[i], 0, NULL);
    vkCmdDispatch(cb, plan.dispatch_x[i], 1, 1);
  }
  vkEndCommandBuffer(cb);

  /* Benchmark */
  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(ctx->device, &fci, NULL, &fence);
  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;

  for (int i = 0; i < WARMUP_ITERS; i++) {
    vkResetFences(ctx->device, 1, &fence);
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
  }

  double times[BENCH_ITERS];
  for (int i = 0; i < BENCH_ITERS; i++) {
    vkResetFences(ctx->device, 1, &fence);
    double t_start = now_ms();
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    times[i] = now_ms() - t_start;
  }
  *out_exec_ms = median_d(times, BENCH_ITERS);

  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, (uint32_t)ns, ds);
  destroy_buffer(ctx, &src_buf);
  destroy_buffer(ctx, &dst_buf);
  destroy_buffer(ctx, &scratch_buf);
  plan_destroy(ctx, &plan);
  return 0;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(void) {
  printf("========================================================================\n");
  printf("  Stockham FFT Benchmark — single C2C, sizes 2..2^20\n");
  printf("========================================================================\n");

  VkCtx ctx;
  if (vk_init(&ctx) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  printf("\n  %-10s %3s  %-16s  %8s  %8s\n",
         "N", "stg", "factorization", "compile", "exec");
  printf("  %-10s %3s  %-16s  %8s  %8s\n",
         "----------", "---", "----------------", "--------", "--------");

  for (int exp = 1; exp <= 20; exp++) {
    int n = 1 << exp;
    int radices[MAX_STAGES], nstg = factorize(n, radices, MAX_STAGES);
    if (nstg == 0) {
      printf("  %-10d  -   (unfactorizable)\n", n);
      continue;
    }

    double compile_ms = 0, exec_ms = 0;
    if (bench_size(&ctx, n, &compile_ms, &exec_ms) != 0) continue;

    /* Format factorization string */
    char fact[64]; int pos = 0;
    for (int s = 0; s < nstg && pos < 60; s++)
      pos += snprintf(fact + pos, sizeof(fact) - (size_t)pos,
                      "%s%d", s ? "x" : "", radices[s]);

    printf("  %-10d %3d  %-16s  %6.1f ms  %6.3f ms\n",
           n, nstg, fact, compile_ms, exec_ms);
  }

  printf("\n");
  vk_destroy(&ctx);
  return 0;
}
