/*
 * bench_fft_stockham.c - Overhead-less benchmark for Stockham FFT generator
 *
 * For each radix N, generates a single-stage Stockham FFT shader via
 * gen_fft_stockham(), compiles it once, creates the pipeline once, then
 * benchmarks across multiple batch sizes with minimal overhead (one fence,
 * one command buffer, reused throughout).
 *
 * Build: linked against wgsl_compiler + Vulkan + fft_stockham_gen.
 * Usage: ./bench_fft_stockham
 */

#include "fft_stockham_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_BATCH    1000000
#define WARMUP_ITERS 5
#define BENCH_ITERS  20

/* ========================================================================== */
/* FFT helpers                                                                */
/* ========================================================================== */

static int is_po2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static int ilog2(int n) {
  int r = 0;
  while (n > 1) { n >>= 1; r++; }
  return r;
}

/* ========================================================================== */
/* CPU reference DFT                                                          */
/* ========================================================================== */

static void cpu_dft(const float *in, float *out, int n) {
  for (int k = 0; k < n; k++) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; j++) {
      double angle = -2.0 * M_PI * k * j / n;
      sr += in[j*2] * cos(angle) - in[j*2+1] * sin(angle);
      si += in[j*2] * sin(angle) + in[j*2+1] * cos(angle);
    }
    out[k*2] = (float)sr;
    out[k*2+1] = (float)si;
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
/* SPIR-V compilation via simple_wgsl                                         */
/* ========================================================================== */

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
  opts.enable_debug_names = 0;

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
/* Pipeline creation (2 storage buffer bindings)                              */
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

  VkDescriptorSetLayoutBinding bindings[2];
  memset(bindings, 0, sizeof(bindings));
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
                                   &out->desc_layout) != VK_SUCCESS)
    return -1;

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &out->desc_layout;
  if (vkCreatePipelineLayout(ctx->device, &plci, NULL,
                              &out->pipe_layout) != VK_SUCCESS)
    return -1;

  VkComputePipelineCreateInfo cpci = {0};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = out->shader;
  cpci.stage.pName = "main";
  cpci.layout = out->pipe_layout;

  if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                &cpci, NULL, &out->pipeline)
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
/* Benchmark one radix across all batch sizes                                 */
/* ========================================================================== */

static const int batch_sizes[] = {64, 1024, 16384, 262144, 1000000};
#define N_BATCHES (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]))

static int bench_radix(VkCtx *ctx, int radix, double *out_compile_ms) {
  /* 1. Generate WGSL */
  char *wgsl = gen_fft_stockham(radix, 1, radix, 1, 1);
  if (!wgsl) {
    fprintf(stderr, "  N=%-3d  GENERATE FAILED\n", radix);
    return -1;
  }

  /* 2. Compile to SPIR-V */
  uint32_t *spirv = NULL;
  size_t spirv_count = 0;
  double t0 = now_ms();
  int rc = compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count);
  *out_compile_ms = now_ms() - t0;

  if (rc != 0) {
    fprintf(stderr, "  N=%-3d  COMPILE FAILED\n", radix);
    free(wgsl);
    return -1;
  }

  /* 3. Create pipeline */
  FftPipeline pipe;
  rc = create_pipeline(ctx, spirv, spirv_count, &pipe);
  wgsl_lower_free(spirv);
  if (rc != 0) {
    fprintf(stderr, "  N=%-3d  PIPELINE FAILED\n", radix);
    free(wgsl);
    return -1;
  }

  /* 4. Allocate src + dst buffers for MAX_BATCH */
  VkDeviceSize buf_bytes =
      (VkDeviceSize)MAX_BATCH * radix * 2 * sizeof(float);
  GpuBuffer src_buf, dst_buf;
  VkMemoryPropertyFlags mem_flags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  rc = create_buffer(ctx, buf_bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     mem_flags, &src_buf);
  if (rc != 0) {
    destroy_pipeline(ctx, &pipe);
    free(wgsl);
    return -1;
  }
  rc = create_buffer(ctx, buf_bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     mem_flags, &dst_buf);
  if (rc != 0) {
    destroy_buffer(ctx, &src_buf);
    destroy_pipeline(ctx, &pipe);
    free(wgsl);
    return -1;
  }

  /* 5. Fill src with random test data */
  float *src_data = (float *)src_buf.mapped;
  srand(42);
  for (long i = 0; i < (long)MAX_BATCH * radix * 2; i++)
    src_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  /* 6. Create descriptor set (binding 0 = src, binding 1 = dst) */
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.desc_layout;
  VkDescriptorSet ds;
  vkAllocateDescriptorSets(ctx->device, &dsai, &ds);

  VkDescriptorBufferInfo dbi[2];
  memset(dbi, 0, sizeof(dbi));
  dbi[0].buffer = src_buf.buffer;
  dbi[0].range = buf_bytes;
  dbi[1].buffer = dst_buf.buffer;
  dbi[1].range = buf_bytes;

  VkWriteDescriptorSet wds[2];
  memset(wds, 0, sizeof(wds));
  wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wds[0].dstSet = ds;
  wds[0].dstBinding = 0;
  wds[0].descriptorCount = 1;
  wds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wds[0].pBufferInfo = &dbi[0];
  wds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wds[1].dstSet = ds;
  wds[1].dstBinding = 1;
  wds[1].descriptorCount = 1;
  wds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wds[1].pBufferInfo = &dbi[1];
  vkUpdateDescriptorSets(ctx->device, 2, wds, 0, NULL);

  /* Allocate command buffer */
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = ctx->cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(ctx->device, &cbai, &cb);

  VkCommandBufferBeginInfo cbbi = {0};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  /* 7. Create ONE fence, reuse it */
  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(ctx->device, &fci, NULL, &fence);

  VkSubmitInfo submit_info = {0};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;

  const char *tag = is_po2(radix) ? "CT " : "DFT";
  double flops_per_fft = is_po2(radix)
      ? 5.0 * radix * ilog2(radix)
      : 8.0 * radix * radix;

  /* 9. Verify correctness (batch_size=1) */
  int verify_ok = 0;
  {
    /* Re-seed and fill just one FFT worth of src data */
    srand(42);
    for (int i = 0; i < radix * 2; i++)
      src_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

    /* Clear dst */
    float *dst_data = (float *)dst_buf.mapped;
    memset(dst_data, 0, (size_t)radix * 2 * sizeof(float));

    /* Dispatch batch_size=1 */
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipe.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    vkEndCommandBuffer(cb);
    vkResetFences(ctx->device, 1, &fence);
    vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

    /* CPU reference */
    float *verify_ref = (float *)malloc((size_t)radix * 2 * sizeof(float));
    cpu_dft(src_data, verify_ref, radix);

    float max_err = 0.0f;
    for (int i = 0; i < radix * 2; i++) {
      float err = fabsf(dst_data[i] - verify_ref[i]);
      if (err > max_err) max_err = err;
    }
    verify_ok = (max_err < 0.5f);

    free(verify_ref);
  }

  /* 8. For each batch_size, benchmark */
  for (int bi = 0; bi < N_BATCHES; bi++) {
    int batch_size = batch_sizes[bi];

    /* a. Record command buffer */
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipe.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe.pipe_layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, (uint32_t)batch_size, 1);
    vkEndCommandBuffer(cb);

    /* b. Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
      vkResetFences(ctx->device, 1, &fence);
      vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
      vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    /* c. Measure */
    double times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; i++) {
      vkResetFences(ctx->device, 1, &fence);
      double t_start = now_ms();
      vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
      vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
      times[i] = now_ms() - t_start;
    }

    /* d. Report */
    double med = median_d(times, BENCH_ITERS);
    double ffts_per_sec = batch_size / (med / 1000.0);
    double gflops = ffts_per_sec * flops_per_fft / 1e9;

    if (bi == 0) {
      printf("  N=%-3d [%s]  batch=%-9d %7.3f ms  %10.0f FFT/s  "
             "%6.2f GFLOP/s  %s\n",
             radix, tag, batch_size, med, ffts_per_sec, gflops,
             verify_ok ? "OK" : "FAIL");
    } else {
      printf("  N=%-3d [%s]  batch=%-9d %7.3f ms  %10.0f FFT/s  "
             "%6.2f GFLOP/s\n",
             radix, tag, batch_size, med, ffts_per_sec, gflops);
    }
  }

  /* 10. Cleanup */
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &ds);
  destroy_buffer(ctx, &dst_buf);
  destroy_buffer(ctx, &src_buf);
  destroy_pipeline(ctx, &pipe);
  free(wgsl);

  return verify_ok ? 0 : -1;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  static const int radices[] = {
    2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32
  };
  int n_radices = (int)(sizeof(radices) / sizeof(radices[0]));

  printf("============================================"
         "============================\n");
  printf("  WGSL Stockham FFT Benchmark\n");
  printf("============================================"
         "============================\n");

  VkCtx ctx;
  if (vk_init(&ctx, 256, 256, 0) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  int passed = 0, failed = 0;
  double compile_times[32];

  printf("\n");
  for (int ri = 0; ri < n_radices; ri++) {
    int radix = radices[ri];
    double compile_ms = 0.0;
    int rc = bench_radix(&ctx, radix, &compile_ms);
    compile_times[ri] = compile_ms;
    if (rc == 0) passed++;
    else failed++;
  }

  printf("\n  Compile times:\n");
  for (int ri = 0; ri < n_radices; ri++) {
    printf("  N=%-3d  %5.1f ms\n", radices[ri], compile_times[ri]);
  }

  printf("\n  Results: %d passed, %d failed\n\n",
         passed, failed);

  vk_destroy(&ctx);
  return failed > 0 ? 1 : 0;
}
