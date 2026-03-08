/*
 * bench_fft_base.c - Benchmark hardcoded WGSL FFT base cases (N=2..128)
 *
 * For each size N, generates a self-contained WGSL compute shader that
 * performs a complete FFT of N complex numbers per workgroup invocation.
 * Powers of 2 use unrolled Cooley-Tukey; others use direct DFT with
 * precomputed twiddle tables.
 *
 * Build: linked against wgsl_compiler + Vulkan.
 * Usage: ./bench_fft_base [min_n] [max_n]
 */

#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <stdarg.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================== */
/* String builder                                                             */
/* ========================================================================== */

typedef struct {
  char *buf;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
  sb->cap = 4096;
  sb->buf = (char *)malloc(sb->cap);
  sb->buf[0] = '\0';
  sb->len = 0;
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (needed < 0) return;
  while (sb->len + (size_t)needed + 1 > sb->cap) {
    sb->cap *= 2;
    sb->buf = (char *)realloc(sb->buf, sb->cap);
  }
  va_start(ap, fmt);
  vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, ap);
  va_end(ap);
  sb->len += (size_t)needed;
}

static char *sb_finish(StrBuf *sb) { return sb->buf; }

/* ========================================================================== */
/* FFT helpers                                                                */
/* ========================================================================== */

static int is_po2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static int ilog2(int n) {
  int r = 0;
  while (n > 1) { n >>= 1; r++; }
  return r;
}

static int bit_reverse(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) { r = (r << 1) | (v & 1); v >>= 1; }
  return r;
}

/* ========================================================================== */
/* Float formatting: %.17g can produce "1" or "0" which WGSL parses as        */
/* integer literals. sb_float ensures a decimal point is always present.       */
/* ========================================================================== */

static void sb_float(StrBuf *sb, double v) {
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.17g", v);
  sb_printf(sb, "%s", tmp);
  /* If no '.' or 'e'/'E', it looks like an integer literal to WGSL */
  if (!strchr(tmp, '.') && !strchr(tmp, 'e') && !strchr(tmp, 'E'))
    sb_printf(sb, ".0");
}

/* ========================================================================== */
/* WGSL generation: power-of-2 Cooley-Tukey (fully unrolled)                  */
/* ========================================================================== */

static char *gen_fft_po2(int n) {
  int log2n = ilog2(n);
  StrBuf sb;
  sb_init(&sb);

  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb,
    "@group(0) @binding(0) var<storage, read_write> data: Buf;\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb,
    "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  for (int i = 0; i < n; i++) {
    int br = bit_reverse(i, log2n);
    sb_printf(&sb,
      "  var r%d: f32 = data.d[base + %uu];\n", i, br * 2);
    sb_printf(&sb,
      "  var i%d: f32 = data.d[base + %uu];\n", i, br * 2 + 1);
  }

  int tmp_id = 0;
  for (int stage = 0; stage < log2n; stage++) {
    int half_size = 1 << stage;
    int group_size = half_size * 2;
    int n_groups = n / group_size;

    for (int g = 0; g < n_groups; g++) {
      for (int b = 0; b < half_size; b++) {
        int even = g * group_size + b;
        int odd = even + half_size;
        int tw_k = b * (n / group_size);
        double angle = -2.0 * M_PI * tw_k / n;
        double tw_re = cos(angle);
        double tw_im = sin(angle);

        if (fabs(tw_re) < 1e-15) tw_re = 0.0;
        if (fabs(tw_im) < 1e-15) tw_im = 0.0;

        sb_printf(&sb, "  let tr%d: f32 = ", tmp_id);
        sb_float(&sb, tw_re);
        sb_printf(&sb, " * r%d - (", odd);
        sb_float(&sb, tw_im);
        sb_printf(&sb, ") * i%d;\n", odd);
        sb_printf(&sb, "  let ti%d: f32 = ", tmp_id);
        sb_float(&sb, tw_re);
        sb_printf(&sb, " * i%d + (", odd);
        sb_float(&sb, tw_im);
        sb_printf(&sb, ") * r%d;\n", odd);
        sb_printf(&sb,
          "  let er%d: f32 = r%d; let ei%d: f32 = i%d;\n",
          tmp_id, even, tmp_id, even);
        sb_printf(&sb,
          "  r%d = er%d + tr%d; i%d = ei%d + ti%d;\n",
          even, tmp_id, tmp_id, even, tmp_id, tmp_id);
        sb_printf(&sb,
          "  r%d = er%d - tr%d; i%d = ei%d - ti%d;\n",
          odd, tmp_id, tmp_id, odd, tmp_id, tmp_id);
        tmp_id++;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    sb_printf(&sb,
      "  data.d[base + %uu] = r%d;\n", i * 2, i);
    sb_printf(&sb,
      "  data.d[base + %uu] = i%d;\n", i * 2 + 1, i);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

/* ========================================================================== */
/* WGSL generation: direct DFT (any N)                                        */
/* ========================================================================== */

static char *gen_fft_dft(int n) {
  StrBuf sb;
  sb_init(&sb);

  sb_printf(&sb, "struct Buf { d: array<f32> };\n");
  sb_printf(&sb,
    "@group(0) @binding(0) var<storage, read_write> data: Buf;\n");
  sb_printf(&sb, "@compute @workgroup_size(1)\n");
  sb_printf(&sb,
    "fn main(@builtin(workgroup_id) wid: vec3<u32>) {\n");
  sb_printf(&sb, "  let base: u32 = wid.x * %uu;\n", n * 2);

  for (int i = 0; i < n; i++) {
    sb_printf(&sb,
      "  let xr%d: f32 = data.d[base + %uu];\n", i, i * 2);
    sb_printf(&sb,
      "  let xi%d: f32 = data.d[base + %uu];\n", i, i * 2 + 1);
  }

  for (int k = 0; k < n; k++) {
    sb_printf(&sb, "  var or%d: f32 = 0.0;\n", k);
    sb_printf(&sb, "  var oi%d: f32 = 0.0;\n", k);
    for (int j = 0; j < n; j++) {
      int tw_idx = (k * j) % n;
      double angle = -2.0 * M_PI * tw_idx / n;
      double wr = cos(angle);
      double wi = sin(angle);
      if (fabs(wr) < 1e-15) wr = 0.0;
      if (fabs(wi) < 1e-15) wi = 0.0;

      if (wr == 1.0 && wi == 0.0) {
        sb_printf(&sb,
          "  or%d = or%d + xr%d; oi%d = oi%d + xi%d;\n",
          k, k, j, k, k, j);
      } else if (wr == 0.0 && wi == -1.0) {
        sb_printf(&sb,
          "  or%d = or%d + xi%d; oi%d = oi%d - xr%d;\n",
          k, k, j, k, k, j);
      } else if (wr == -1.0 && wi == 0.0) {
        sb_printf(&sb,
          "  or%d = or%d - xr%d; oi%d = oi%d - xi%d;\n",
          k, k, j, k, k, j);
      } else if (wr == 0.0 && wi == 1.0) {
        sb_printf(&sb,
          "  or%d = or%d - xi%d; oi%d = oi%d + xr%d;\n",
          k, k, j, k, k, j);
      } else {
        sb_printf(&sb, "  or%d = or%d + ", k, k);
        sb_float(&sb, wr);
        sb_printf(&sb, " * xr%d - (", j);
        sb_float(&sb, wi);
        sb_printf(&sb, ") * xi%d;\n", j);
        sb_printf(&sb, "  oi%d = oi%d + ", k, k);
        sb_float(&sb, wr);
        sb_printf(&sb, " * xi%d + (", j);
        sb_float(&sb, wi);
        sb_printf(&sb, ") * xr%d;\n", j);
      }
    }
  }

  for (int k = 0; k < n; k++) {
    sb_printf(&sb,
      "  data.d[base + %uu] = or%d;\n", k * 2, k);
    sb_printf(&sb,
      "  data.d[base + %uu] = oi%d;\n", k * 2 + 1, k);
  }

  sb_printf(&sb, "}\n");
  return sb_finish(&sb);
}

static char *gen_fft_wgsl(int n) {
  if (is_po2(n)) return gen_fft_po2(n);
  return gen_fft_dft(n);
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
/* Pipeline creation from SPIR-V                                              */
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
/* Benchmark one FFT size                                                     */
/* ========================================================================== */

#define WARMUP_ITERS 3
#define BENCH_ITERS 10

static int bench_one(VkCtx *ctx, int n, int batch_size) {
  char *wgsl = gen_fft_wgsl(n);
  if (!wgsl) return -1;

  uint32_t *spirv = NULL;
  size_t spirv_count = 0;
  double t0 = now_ms();
  int rc = compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count);
  double compile_ms = now_ms() - t0;

  if (rc != 0) {
    fprintf(stderr, "  N=%-3d  COMPILE FAILED\n", n);
    free(wgsl);
    return -1;
  }

  FftPipeline pipe;
  rc = create_pipeline(ctx, spirv, spirv_count, &pipe);
  wgsl_lower_free(spirv);
  if (rc != 0) {
    fprintf(stderr, "  N=%-3d  PIPELINE FAILED\n", n);
    free(wgsl);
    return -1;
  }

  VkDeviceSize buf_bytes =
      (VkDeviceSize)batch_size * n * 2 * sizeof(float);
  GpuBuffer gpu_buf;
  rc = create_buffer(ctx, buf_bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &gpu_buf);
  if (rc != 0) {
    destroy_pipeline(ctx, &pipe);
    free(wgsl);
    return -1;
  }

  float *host_data = (float *)gpu_buf.mapped;
  srand(42);
  for (int i = 0; i < batch_size * n * 2; i++)
    host_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

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

  for (int i = 0; i < WARMUP_ITERS; i++) {
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipe.pipeline);
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
    srand(42);
    for (int j = 0; j < batch_size * n * 2; j++)
      host_data[j] = (float)(rand() % 1000) / 500.0f - 1.0f;

    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipe.pipeline);
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

  double med = median_d(times, BENCH_ITERS);
  double ffts_per_sec = batch_size / (med / 1000.0);
  double flops_per_fft = is_po2(n)
      ? 5.0 * n * ilog2(n)
      : 8.0 * n * n;
  double gflops = ffts_per_sec * flops_per_fft / 1e9;

  srand(42);
  for (int j = 0; j < batch_size * n * 2; j++)
    host_data[j] = (float)(rand() % 1000) / 500.0f - 1.0f;

  float *verify_in = (float *)malloc((size_t)n * 2 * sizeof(float));
  float *verify_ref = (float *)malloc((size_t)n * 2 * sizeof(float));
  memcpy(verify_in, host_data, (size_t)n * 2 * sizeof(float));

  vkResetCommandBuffer(cb, 0);
  vkBeginCommandBuffer(cb, &cbbi);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipe.pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdDispatch(cb, (uint32_t)batch_size, 1, 1);
  vkEndCommandBuffer(cb);
  vkResetFences(ctx->device, 1, &fence);
  vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
  vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

  cpu_dft(verify_in, verify_ref, n);
  float *gpu_out = host_data;
  float max_err = 0.0f;
  for (int i = 0; i < n * 2; i++) {
    float err = fabsf(gpu_out[i] - verify_ref[i]);
    if (err > max_err) max_err = err;
  }

  const char *tag = is_po2(n) ? "CT" : "DFT";
  const char *status = max_err < 0.5f ? "OK" : "FAIL";
  printf("  N=%-3d [%-3s] %s  %7.3f ms  %10.0f FFT/s  "
         "%6.2f GFLOP/s  compile=%5.1fms  err=%.2e\n",
         n, tag, status, med, ffts_per_sec, gflops,
         compile_ms, max_err);

  free(verify_in);
  free(verify_ref);
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &ds);
  destroy_buffer(ctx, &gpu_buf);
  destroy_pipeline(ctx, &pipe);
  free(wgsl);

  return max_err < 0.5f ? 0 : -1;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char **argv) {
  int min_n = 2, max_n = 128;
  if (argc > 1) min_n = atoi(argv[1]);
  if (argc > 2) max_n = atoi(argv[2]);
  if (min_n < 2) min_n = 2;
  if (max_n > 128) max_n = 128;

  printf("============================================"
         "============================\n");
  printf("  WGSL FFT Base Case Benchmark (N=%d..%d)\n",
         min_n, max_n);
  printf("============================================"
         "============================\n");

  VkCtx ctx;
  if (vk_init(&ctx, 256, 256, 0) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  int batch_size = 4096;
  int passed = 0, failed = 0;

  printf("\n");
  for (int n = min_n; n <= max_n; n++) {
    int rc = bench_one(&ctx, n, batch_size);
    if (rc == 0) passed++;
    else failed++;
  }

  printf("\n  Results: %d passed, %d failed out of %d\n",
         passed, failed, passed + failed);

  vk_destroy(&ctx);
  return failed > 0 ? 1 : 0;
}
