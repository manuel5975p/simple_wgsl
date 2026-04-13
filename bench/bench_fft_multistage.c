/*
 * bench_fft_multistage.c - Benchmark multi-stage Stockham FFT with parameter sweep
 *
 * For each FFT size N, sweeps (max_radix, workgroup_size) configurations:
 * factorizes into radices, generates + compiles all stage shaders, records a
 * single command buffer with all stages + barriers, and benchmarks with GPU
 * timestamps.
 *
 * Usage: ./bench_fft_multistage
 */

#include "fft_stockham_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARMUP_ITERS 3
#define BENCH_ITERS  10

/* ========================================================================== */
/* Helpers                                                                     */
/* ========================================================================== */

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
  qsort(arr, (size_t)n, sizeof(double), cmp_double);
  if (n % 2 == 0) return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
  return arr[n / 2];
}

static int compile_wgsl_to_spirv(const char *src, uint32_t **out_words,
                                  size_t *out_count) {
  WgslParseResult pr = wgsl_parse(src);
  WgslAstNode *ast = pr.value;
  if (!ast) { wgsl_diagnostic_list_free(pr.diags); return -1; }
  WgslResolveResult rr = wgsl_resolver_build(ast);
  WgslResolver *res = rr.value;
  if (!res) { wgsl_free_ast(ast); wgsl_diagnostic_list_free(pr.diags); wgsl_diagnostic_list_free(rr.diags); return -1; }
  WgslLowerOptions opts = {0};
  opts.spirv_version = 0x00010300;
  opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
  opts.packing = WGSL_LOWER_PACK_STD430;
  WgslLowerSpirvResult lsr = wgsl_lower_emit_spirv(ast, res, &opts);
  wgsl_resolver_free(res);
  wgsl_diagnostic_list_free(rr.diags);
  wgsl_free_ast(ast);
  wgsl_diagnostic_list_free(pr.diags);
  if (lsr.code != SW_OK) {
    wgsl_diagnostic_list_free(lsr.diags);
    if (lsr.words) wgsl_lower_free(lsr.words);
    return -1;
  }
  *out_words = lsr.words;
  *out_count = lsr.word_count;
  wgsl_diagnostic_list_free(lsr.diags);
  return 0;
}

/* ========================================================================== */
/* GPU buffer helpers                                                         */
/* ========================================================================== */

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
  if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer) != VK_SUCCESS)
    return -1;
  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);
  int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits,
                                mem_flags);
  if (mt < 0) return -1;
  VkMemoryAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = reqs.size; ai.memoryTypeIndex = (uint32_t)mt;
  if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory) != VK_SUCCESS)
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

static void staged_copy(VkCtx *ctx, VkBuffer src, VkBuffer dst,
                        VkDeviceSize size) {
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cmd);
  VkCommandBufferBeginInfo bi = {0};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  VkBufferCopy region = {0, 0, size};
  vkCmdCopyBuffer(cmd, src, dst, 1, &region);
  vkEndCommandBuffer(cmd);
  VkFence fence;
  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(ctx->device, &fci, NULL, &fence);
  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
  vkQueueSubmit(ctx->queue, 1, &si, fence);
  vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd);
}

/* ========================================================================== */
/* Multi-stage pipeline setup                                                  */
/* ========================================================================== */

typedef struct {
  int n;
  int n_stages;
  int radices[FFT_STOCKHAM_MAX_STAGES];
  uint32_t dispatch_x[FFT_STOCKHAM_MAX_STAGES];
  int workgroup_size;

  VkShaderModule shaders[FFT_STOCKHAM_MAX_STAGES];
  VkPipeline pipelines[FFT_STOCKHAM_MAX_STAGES];
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
} FftPlan;

static int plan_create(VkCtx *ctx, int n, int max_radix, int wg_size,
                       FftPlan *plan) {
  memset(plan, 0, sizeof(*plan));
  plan->n = n;
  plan->workgroup_size = wg_size;
  plan->n_stages = fft_stockham_factorize(n, max_radix, plan->radices);
  if (plan->n_stages == 0) return -1;

  /* Shared descriptor layout */
  VkDescriptorSetLayoutBinding bindings[2] = {{0}};
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslci = {0};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2; dslci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                   &plan->desc_layout) != VK_SUCCESS)
    return -1;

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1; plci.pSetLayouts = &plan->desc_layout;
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                              &plan->pipe_layout) != VK_SUCCESS)
    return -1;

  int stride = 1;
  for (int s = 0; s < plan->n_stages; s++) {
    int radix = plan->radices[s];
    plan->dispatch_x[s] = (uint32_t)fft_stockham_dispatch_x(n, radix,
                                                             wg_size);

    char *wgsl = gen_fft_stockham(radix, stride, n, 1, wg_size);
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
/* Benchmark one (N, max_radix, workgroup_size) configuration                  */
/* ========================================================================== */

static int bench_one(VkCtx *ctx, int n, int max_radix, int wg_size,
                     int batch, int repeats,
                     double *out_ms, float *out_err) {
  FftPlan plan;
  if (plan_create(ctx, n, max_radix, wg_size, &plan) != 0)
    return -1;
  int ns = plan.n_stages;

  /* Device-local buffers */
  VkDeviceSize buf_bytes = (VkDeviceSize)n * batch * 2 * sizeof(float);
  GpuBuffer src_buf, dst_buf, scratch_buf, staging;

  VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (create_buffer(ctx, buf_bytes, dev_usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &src_buf) != 0 ||
      create_buffer(ctx, buf_bytes, dev_usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &dst_buf) != 0 ||
      create_buffer(ctx, buf_bytes, dev_usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &scratch_buf) != 0 ||
      create_buffer(ctx, buf_bytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging) != 0) {
    plan_destroy(ctx, &plan);
    return -1;
  }

  /* Upload impulse: (1,0, 0,0, ...) */
  float *host = (float *)staging.mapped;
  memset(host, 0, (size_t)buf_bytes);
  host[0] = 1.0f;
  staged_copy(ctx, staging.buffer, src_buf.buffer, buf_bytes);

  /* Descriptor sets with ping-pong */
  VkDescriptorSet ds[FFT_STOCKHAM_MAX_STAGES];
  VkDescriptorSetLayout layouts[FFT_STOCKHAM_MAX_STAGES];
  for (int i = 0; i < ns; i++) layouts[i] = plan.desc_layout;
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = (uint32_t)ns;
  dsai.pSetLayouts = layouts;
  if (vkAllocateDescriptorSets(ctx->device, &dsai, ds) != VK_SUCCESS) {
    plan_destroy(ctx, &plan);
    return -1;
  }

  VkBuffer read_bufs[FFT_STOCKHAM_MAX_STAGES];
  VkBuffer write_bufs[FFT_STOCKHAM_MAX_STAGES];
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

  /* Command buffer: [timestamp0] stages+barriers * repeats [timestamp1] */
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cb);

  VkFence fence;
  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(ctx->device, &fci, NULL, &fence);

  VkCommandBufferBeginInfo cbbi = {0};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  VkMemoryBarrier bar = {0};
  bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  /* Warmup */
  for (int w = 0; w < WARMUP_ITERS; w++) {
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    for (int i = 0; i < ns; i++) {
      if (i > 0)
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &bar, 0, NULL, 0, NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                        plan.pipelines[i]);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              plan.pipe_layout, 0, 1, &ds[i], 0, NULL);
      vkCmdDispatch(cb, plan.dispatch_x[i], (uint32_t)batch, 1);
    }
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);
  }

  /* Verify correctness: readback dst */
  staged_copy(ctx, dst_buf.buffer, staging.buffer, buf_bytes);
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err_re = fabsf(host[i * 2] - 1.0f);
    float err_im = fabsf(host[i * 2 + 1]);
    if (err_re > max_err) max_err = err_re;
    if (err_im > max_err) max_err = err_im;
  }
  *out_err = max_err;

  /* Timed runs with GPU timestamps */
  double times[BENCH_ITERS];
  for (int it = 0; it < BENCH_ITERS; it++) {
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdResetQueryPool(cb, ctx->timestamp_pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 0);
    for (int r = 0; r < repeats; r++) {
      for (int i = 0; i < ns; i++) {
        if (i > 0 || r > 0)
          vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, 1, &bar, 0, NULL, 0, NULL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          plan.pipelines[i]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                plan.pipe_layout, 0, 1, &ds[i], 0, NULL);
        vkCmdDispatch(cb, plan.dispatch_x[i], (uint32_t)batch, 1);
      }
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 1);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                          sizeof(timestamps), timestamps,
                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    double dt_ns = (double)(timestamps[1] - timestamps[0]) *
                   ctx->timestamp_period;
    times[it] = dt_ns / 1e6 / repeats;
  }
  *out_ms = median_d(times, BENCH_ITERS);

  /* Cleanup */
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, (uint32_t)ns, ds);
  destroy_buffer(ctx, &staging);
  destroy_buffer(ctx, &scratch_buf);
  destroy_buffer(ctx, &dst_buf);
  destroy_buffer(ctx, &src_buf);
  plan_destroy(ctx, &plan);
  return 0;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(void) {
  VkCtx ctx;
  if (vk_init(&ctx, 512, 1024, 1) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  /* Sizes: 2^10 to 2^20 */
  printf("\n%-10s  %3s  %4s  %3s  %-20s  %6s  %6s  %10s  %6s\n",
         "N", "mr", "wg", "stg", "factorization", "batch", "reps",
         "ms/fft", "err");
  printf("---------  ---  ----  ---  --------------------  "
         "------  ------  ----------  ------\n");

  /* max_radix values to sweep (0 = default 32) */
  int mr_values[] = {0, 4, 8, 16};
  int n_mr = sizeof(mr_values) / sizeof(mr_values[0]);

  /* workgroup_size values to sweep */
  int wg_values[] = {64, 128, 256};
  int n_wg = sizeof(wg_values) / sizeof(wg_values[0]);

  for (int exp = 10; exp <= 20; exp++) {
    int n = 1 << exp;
    int batch = 1;

    for (int mi = 0; mi < n_mr; mi++) {
      int mr = mr_values[mi];

      for (int wi = 0; wi < n_wg; wi++) {
        int wg = wg_values[wi];

        int radices[FFT_STOCKHAM_MAX_STAGES];
        int nstg = fft_stockham_factorize(n, mr, radices);
        if (nstg == 0) continue;

        /* Correctness + pilot */
        double ms;
        float err;
        if (bench_one(&ctx, n, mr, wg, batch, 1, &ms, &err) != 0)
          continue;
        if (err > 1e-3f) continue;

        /* Scale repeats to ~50ms */
        int reps = (ms > 0.001) ? (int)(50.0 / ms) : 50000;
        if (reps < 10) reps = 10;
        if (reps > 50000) reps = 50000;

        double ms_final;
        float err_final;
        if (bench_one(&ctx, n, mr, wg, batch, reps, &ms_final, &err_final)
            != 0)
          continue;

        /* Format factorization */
        char fact[64]; int pos = 0;
        for (int s = 0; s < nstg && pos < 58; s++)
          pos += snprintf(fact + pos, sizeof(fact) - (size_t)pos,
                          "%s%d", s ? "x" : "", radices[s]);

        printf("%-10d  %3d  %4d  %3d  %-20s  %6d  %6d  %10.4f  %6.1e\n",
               n, mr, wg, nstg, fact, batch, reps, ms_final, err);
        fflush(stdout);
      }
    }
  }

  printf("\n");
  vk_destroy(&ctx);
  return 0;
}
