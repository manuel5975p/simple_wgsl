/*
 * bench_fft_fused.c - Benchmark fused single-dispatch FFT
 *
 * For each FFT size, generates a fused WGSL shader, compiles to SPIR-V,
 * runs on GPU, verifies correctness, and measures latency.
 *
 * Build: linked against fft_fused_gen + wgsl_compiler + Vulkan.
 * Usage: ./bench_fft_fused
 */

#include "fft_fused_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARMUP_ITERS 5
#define BENCH_ITERS  20


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
  bci.size = size;
  bci.usage = usage;
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
  ai.allocationSize = reqs.size;
  ai.memoryTypeIndex = (uint32_t)mt;
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

/* ========================================================================== */
/* Pipeline creation (2 storage bindings: src read, dst read_write)            */
/* ========================================================================== */

typedef struct {
  VkShaderModule shader;
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
  VkPipeline pipeline;
} FftPipeline;

static int create_pipeline(VkCtx *ctx, uint32_t *spirv, size_t spirv_count,
                           int num_bindings, FftPipeline *out) {
  memset(out, 0, sizeof(*out));

  VkShaderModuleCreateInfo smci = {0};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spirv_count * sizeof(uint32_t);
  smci.pCode = spirv;
  if (vkCreateShaderModule(ctx->device, &smci, NULL, &out->shader)
      != VK_SUCCESS)
    return -1;

  VkDescriptorSetLayoutBinding bindings[3] = {0};
  for (int i = 0; i < num_bindings; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci = {0};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = num_bindings;
  dslci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL,
                                   &out->desc_layout) != VK_SUCCESS)
    return -1;

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &out->desc_layout;
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL, &out->pipe_layout)
      != VK_SUCCESS)
    return -1;

  VkComputePipelineCreateInfo ccpci = {0};
  ccpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ccpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ccpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ccpci.stage.module = out->shader;
  ccpci.stage.pName = "main";
  ccpci.layout = out->pipe_layout;

  if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &ccpci,
                                NULL, &out->pipeline) != VK_SUCCESS)
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
/* WGSL -> SPIR-V compilation                                                 */
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
/* Sorting / statistics                                                       */
/* ========================================================================== */

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
  qsort(arr, n, sizeof(double), cmp_double);
  if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
  return arr[n/2];
}

/* ========================================================================== */
/* Bench one FFT size                                                         */
/* ========================================================================== */

/* Helper: record a buffer copy command and submit+wait */
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
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(ctx->queue, 1, &si, fence);
  vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd);
}

static int bench_one(VkCtx *ctx, int n, int max_radix, int wg_limit,
                     int batch, int repeats,
                     double *out_ms, float *out_err) {

  /* Generate WGSL */
  char *wgsl = gen_fft_fused_ex(n, 1, max_radix, wg_limit);
  if (!wgsl) {
    fprintf(stderr, "  N=%d mr=%d wg=%d: gen_fft_fused_ex failed\n",
            n, max_radix, wg_limit);
    return -1;
  }

  /* Compile */
  uint32_t *spirv = NULL;
  size_t spirv_count = 0;
  if (compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count) != 0) {
    fprintf(stderr, "  N=%d: WGSL compilation failed\n", n);
    free(wgsl);
    return -1;
  }
  free(wgsl);

  /* LUT */
  int lut_count = fft_fused_lut_size_ex(n, 1, max_radix); /* wg_limit doesn't affect LUT */
  int num_bindings = (lut_count > 0) ? 3 : 2;

  /* Create pipeline */
  FftPipeline pipe;
  if (create_pipeline(ctx, spirv, spirv_count, num_bindings, &pipe) != 0) {
    fprintf(stderr, "  N=%d: pipeline creation failed\n", n);
    wgsl_lower_free(spirv);
    return -1;
  }
  wgsl_lower_free(spirv);

  /* Batching: pad batch count to multiple of B */
  int B = fft_fused_batch_per_wg_ex(n, max_radix, wg_limit);
  if (B < 1) B = 1;
  int padded_batch = ((batch + B - 1) / B) * B;
  int dispatch_count = padded_batch / B;

  /* Device-local buffers for compute (VRAM) */
  VkDeviceSize buf_size = (VkDeviceSize)n * padded_batch * 2 * sizeof(float);
  GpuBuffer src_buf, dst_buf, lut_buf, staging;
  memset(&lut_buf, 0, sizeof(lut_buf));

  if (create_buffer(ctx, buf_size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &src_buf) != 0)
    return -1;
  if (create_buffer(ctx, buf_size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &dst_buf) != 0)
    return -1;

  /* Staging buffer (host-visible) for upload/readback */
  VkDeviceSize staging_size = buf_size;
  if (lut_count > 0) {
    VkDeviceSize lut_bytes = (VkDeviceSize)lut_count * 2 * sizeof(float);
    if (lut_bytes > staging_size) staging_size = lut_bytes;
  }
  if (create_buffer(ctx, staging_size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging) != 0)
    return -1;

  /* Upload impulse: (1,0,0,...) for first FFT in batch */
  float *src_data = (float *)staging.mapped;
  memset(src_data, 0, buf_size);
  src_data[0] = 1.0f;
  staged_copy(ctx, staging.buffer, src_buf.buffer, buf_size);

  /* Upload twiddle LUT */
  if (lut_count > 0) {
    float *lut_data = fft_fused_compute_lut_ex(n, 1, max_radix);
    VkDeviceSize lut_size = (VkDeviceSize)lut_count * 2 * sizeof(float);
    if (create_buffer(ctx, lut_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &lut_buf) != 0) {
      free(lut_data);
      return -1;
    }
    memcpy(staging.mapped, lut_data, lut_size);
    staged_copy(ctx, staging.buffer, lut_buf.buffer, lut_size);
    free(lut_data);
  }

  /* Descriptor set */
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.desc_layout;
  VkDescriptorSet desc_set;
  vkAllocateDescriptorSets(ctx->device, &dsai, &desc_set);

  VkDescriptorBufferInfo buf_infos[3] = {0};
  buf_infos[0].buffer = src_buf.buffer;
  buf_infos[0].range = buf_size;
  buf_infos[1].buffer = dst_buf.buffer;
  buf_infos[1].range = buf_size;
  if (lut_count > 0) {
    buf_infos[2].buffer = lut_buf.buffer;
    buf_infos[2].range = (VkDeviceSize)lut_count * 2 * sizeof(float);
  }

  VkWriteDescriptorSet writes[3] = {0};
  for (int i = 0; i < num_bindings; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = desc_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buf_infos[i];
  }
  vkUpdateDescriptorSets(ctx->device, num_bindings, writes, 0, NULL);

  /* Command buffer */
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cmd);

  VkFence fence;
  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(ctx->device, &fci, NULL, &fence);

  VkCommandBufferBeginInfo cbbi = {0};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  /* Warmup */
  for (int i = 0; i < WARMUP_ITERS; i++) {
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdDispatch(cmd, dispatch_count, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);
  }

  /* Verify correctness: read back dst via staging */
  staged_copy(ctx, dst_buf.buffer, staging.buffer, buf_size);
  float *dst_data = (float *)staging.mapped;
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err_re = fabsf(dst_data[i * 2] - 1.0f);
    float err_im = fabsf(dst_data[i * 2 + 1]);
    if (err_re > max_err) max_err = err_re;
    if (err_im > max_err) max_err = err_im;
  }
  *out_err = max_err;

  /* Timed runs with GPU timestamps — record `repeats` dispatches per submit */
  double times[BENCH_ITERS];
  for (int i = 0; i < BENCH_ITERS; i++) {
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdResetQueryPool(cmd, ctx->timestamp_pool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &desc_set, 0, NULL);
    for (int r = 0; r < repeats; r++)
      vkCmdDispatch(cmd, dispatch_count, 1, 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                          sizeof(timestamps), timestamps,
                          sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT);
    double dt_ns = (double)(timestamps[1] - timestamps[0]) *
                   ctx->timestamp_period;
    times[i] = dt_ns / 1e6 / repeats; /* ms per single dispatch */
  }

  *out_ms = median_d(times, BENCH_ITERS);

  /* Cleanup */
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
  if (lut_buf.buffer) destroy_buffer(ctx, &lut_buf);
  destroy_buffer(ctx, &staging);
  destroy_buffer(ctx, &dst_buf);
  destroy_buffer(ctx, &src_buf);
  destroy_pipeline(ctx, &pipe);

  return 0;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(void) {
  VkCtx ctx;
  if (vk_init(&ctx, 16, 32, 1) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  /* Test sizes */
  int sizes[] = {
    2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
  };
  int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

  /* max_radix values to sweep (0 = auto) */
  int mr_values[] = {0, 2, 4, 8, 16};
  int n_mr = sizeof(mr_values) / sizeof(mr_values[0]);

  /* wg_limit values to sweep (0 = default 256) */
  int wgl_values[] = {0, 512, 1024};
  int n_wgl = sizeof(wgl_values) / sizeof(wgl_values[0]);

  printf("\n%-8s  %3s  %5s  %4s  %5s  %6s  %6s  %8s  %6s\n",
         "N", "mr", "wglim", "B", "wg", "batch", "reps", "us/fft", "err");
  printf("-------  ---  -----  ----  -----  ------  ------  --------  ------\n");

  for (int i = 0; i < n_sizes; i++) {
    int n = sizes[i];
    int batch = (n <= 64) ? 65536 : (n <= 512) ? 16384 : (n <= 2048) ? 4096 : 256;

    for (int mi = 0; mi < n_mr; mi++) {
      int mr = mr_values[mi];

      for (int wi = 0; wi < n_wgl; wi++) {
        int wgl = wgl_values[wi];

        int wg = fft_fused_workgroup_size_ex(n, mr, wgl);
        int B = fft_fused_batch_per_wg_ex(n, mr, wgl);
        if (wg == 0) continue;

        /* Verify correctness */
        double ms;
        float err;
        if (bench_one(&ctx, n, mr, wgl, 1, 1, &ms, &err) != 0)
          continue;
        if (err > 1e-4f) continue;

        /* Pilot + sustained benchmark */
        double pilot_ms;
        float pilot_err;
        if (bench_one(&ctx, n, mr, wgl, batch, 1, &pilot_ms, &pilot_err) != 0)
          continue;
        int reps = (pilot_ms > 0.001) ? (int)(100.0 / pilot_ms) : 100000;
        if (reps < 1) reps = 1;
        if (reps > 200000) reps = 200000;
        double ms_batch;
        float err_batch;
        if (bench_one(&ctx, n, mr, wgl, batch, reps, &ms_batch, &err_batch) != 0)
          continue;
        double us_per_fft = (ms_batch > 0) ? ms_batch * 1000.0 / batch : -1.0;

        printf("%-8d  %3d  %5d  %4d  %5d  %6d  %6d  %8.4f  %6.1e\n",
               n, mr, wgl, B, wg, batch, reps, us_per_fft, err);
        fflush(stdout);
      }
    }
  }

  vk_destroy(&ctx);
  return 0;
}
