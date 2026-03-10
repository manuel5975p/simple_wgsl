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
#include "bench_vk_common.h"

#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARMUP_ITERS 3
#define BENCH_ITERS  10


/* ========================================================================== */
/* Pipeline creation from SPIR-V (1 binding, read_write)                      */
/* ========================================================================== */

typedef struct {
  VkShaderModule shader;
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

  /* Push constant layout: 1 x u64 (data buffer BDA address) */
  VkPushConstantRange pc_range = {0};
  pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc_range.offset = 0;
  pc_range.size = 8; /* sizeof(u64) */

  VkPipelineLayoutCreateInfo plci = {0};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc_range;
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
  VkDeviceAddress bda;
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
  bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer)
      != VK_SUCCESS)
    return -1;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);

  int32_t mt = find_memory_type(&ctx->mem_props,
                                reqs.memoryTypeBits, mem_flags);
  if (mt < 0) return -1;

  VkMemoryAllocateFlagsInfo flags_info = {0};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.pNext = &flags_info;
  ai.allocationSize = reqs.size;
  ai.memoryTypeIndex = (uint32_t)mt;
  if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory)
      != VK_SUCCESS)
    return -1;

  vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);

  if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    vkMapMemory(ctx->device, out->memory, 0, size, 0, &out->mapped);

  if (ctx->pfn_get_bda) {
    VkBufferDeviceAddressInfo addr_info = {0};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = out->buffer;
    out->bda = ctx->pfn_get_bda(ctx->device, &addr_info);
  }

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

  /* BDA address for push constants */
  uint64_t buf_bda = gpu_buf.bda;

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
  vkCmdPushConstants(cb, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, 8, &buf_bda);
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
    vkCmdPushConstants(cb, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, 8, &buf_bda);
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
    vkCmdPushConstants(cb, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, 8, &buf_bda);
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
  destroy_buffer(ctx, &gpu_buf);
  destroy_pipeline(ctx, &pipe);

  return median_time;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

static int is_prime_bench(int n) {
  if (n < 2) return 0;
  if (n < 4) return 1;
  if (n % 2 == 0 || n % 3 == 0) return 0;
  for (int i = 5; i * i <= n; i += 6)
    if (n % i == 0 || n % (i + 2) == 0) return 0;
  return 1;
}

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
  } else if (p.type == FFT_PLAN_RADER) {
    char mbuf[128];
    format_fft_plan(plans, n - 1, mbuf, sizeof(mbuf));
    snprintf(buf, buflen, "Rader%d(%s)", n, mbuf);
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
  else if (p.type == FFT_PLAN_RADER)
    snprintf(buf, buflen, "Rader-%d", n);
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
  if (vk_init(&ctx, 256, 256, 0) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  FftPlan plans[FFT_PLAN_MAX_N + 1];
  double plan_times[FFT_PLAN_MAX_N + 1];
  memset(plans, 0, sizeof(plans));
  for (int i = 0; i <= FFT_PLAN_MAX_N; i++) plan_times[i] = -1.0;

  /* ====================================================================== */
  /* Benchmark all sizes N=2..MAX_BENCH_N incrementally                     */
  /* ====================================================================== */

  printf("\n--- Benchmarking all sizes N=2..%d ---\n", MAX_BENCH_N);

  for (int n = 2; n <= MAX_BENCH_N; n++) {

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

    /* Strategy 3: Rader (for primes where p-1 has a known plan) */
    if (is_prime_bench(n) && plan_times[n - 1] >= 0) {
      trial[n] = (FftPlan){FFT_PLAN_RADER, 0};
      char *wgsl = gen_fft(n, trial, 1);
      if (wgsl) {
        int correct = 0;
        double t = bench_wgsl(&ctx, n, wgsl, batch_size, &correct);
        printf("  Rader (M=%d)  %7.3f ms  %s\n",
               n - 1, t, correct ? "OK" : "FAIL");
        if (correct && t > 0 && t < best_time) {
          best_time = t;
          best_plan = trial[n];
          any_ok = 1;
        }
        free(wgsl);
      }
    }

    /* Strategy 4: Bluestein (always try — competes with CT and Rader) */
    {
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
                   : plans[n].type == FFT_PLAN_RADER ? "FFT_PLAN_RADER"
                   : "FFT_PLAN_BLUESTEIN";
    printf("  [%3d] = {%s, %d},\n", n, ts, plans[n].radix);
  }
  printf("};\n");

  vk_destroy(&ctx);
  return 0;
}
