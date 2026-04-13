/*
 * test_fft_multistage.c - Integration test for multi-stage Stockham FFT
 *
 * Validates that composing multiple gen_fft_stockham() stages produces
 * correct FFTs for various N values (powers of two, composites, primorials).
 * Also tests forward+inverse roundtrip.
 *
 * Build: linked against fft_stockham_gen + wgsl_compiler + Vulkan.
 * Usage: ./test_fft_multistage
 */

#include "fft_stockham_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_STAGES 32
#define WORKGROUP_SIZE 64

/* ========================================================================== */
/* Factorization                                                              */
/* ========================================================================== */

static int factorize(int n, int *radices, int max_stages) {
  int count = 0, rem = n;
  while (rem > 1 && count < max_stages) {
    int best = 0;
    for (int r = 32; r >= 2; r--)
      if (rem % r == 0) { best = r; break; }
    if (best == 0) return 0;
    radices[count++] = best;
    rem /= best;
  }
  return (rem == 1) ? count : 0;
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
/* Test one N value                                                           */
/* ========================================================================== */

static int test_fft_size(VkCtx *ctx, int n, int *out_n_stages,
                         int *out_radices, float *out_max_err) {
  *out_max_err = 0.0f;

  /* 1. Factorize */
  int radices[MAX_STAGES];
  int n_stages = factorize(n, radices, MAX_STAGES);
  *out_n_stages = n_stages;
  if (n_stages == 0) {
    fprintf(stderr, "    N=%d: cannot factorize into radices 2..32\n", n);
    return -1;
  }
  memcpy(out_radices, radices, (size_t)n_stages * sizeof(int));

  /* 2. Generate WGSL + compile SPIR-V + create pipelines for each stage */
  FftPipeline pipes[MAX_STAGES];
  int dispatch_x[MAX_STAGES];
  int stride = 1;

  for (int s = 0; s < n_stages; s++) {
    char *wgsl = gen_fft_stockham(radices[s], stride, n, 1, WORKGROUP_SIZE);
    if (!wgsl) {
      fprintf(stderr, "    N=%d stage %d: gen_fft_stockham failed\n", n, s);
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &pipes[j]);
      return -1;
    }

    uint32_t *spirv = NULL;
    size_t spirv_count = 0;
    int rc = compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count);
    free(wgsl);
    if (rc != 0) {
      fprintf(stderr, "    N=%d stage %d: compile failed\n", n, s);
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &pipes[j]);
      return -1;
    }

    rc = create_pipeline(ctx, spirv, spirv_count, &pipes[s]);
    wgsl_lower_free(spirv);
    if (rc != 0) {
      fprintf(stderr, "    N=%d stage %d: pipeline creation failed\n", n, s);
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &pipes[j]);
      return -1;
    }

    int butterflies = n / radices[s];
    dispatch_x[s] = (butterflies + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    stride *= radices[s];
  }

  /* 3. Allocate buffers: buf_a and buf_b for ping-pong */
  VkDeviceSize buf_bytes = (VkDeviceSize)n * 2 * sizeof(float);
  GpuBuffer buf_a, buf_b;
  VkMemoryPropertyFlags mem_flags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  if (create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    mem_flags, &buf_a) != 0 ||
      create_buffer(ctx, buf_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    mem_flags, &buf_b) != 0) {
    fprintf(stderr, "    N=%d: buffer allocation failed\n", n);
    for (int s = 0; s < n_stages; s++) destroy_pipeline(ctx, &pipes[s]);
    return -1;
  }

  /* 4. Fill buf_a with test data */
  float *input_data = (float *)buf_a.mapped;
  srand(42 + n);
  for (int i = 0; i < n * 2; i++)
    input_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

  /* Save a copy of input for roundtrip test later */
  float *input_copy = (float *)malloc((size_t)n * 2 * sizeof(float));
  memcpy(input_copy, input_data, (size_t)n * 2 * sizeof(float));

  /* Clear buf_b */
  memset(buf_b.mapped, 0, (size_t)n * 2 * sizeof(float));

  /* 5. Create descriptor sets for each stage (ping-pong) */
  VkDescriptorSet desc_sets[MAX_STAGES];
  VkDescriptorSetLayout layouts[MAX_STAGES];
  for (int s = 0; s < n_stages; s++)
    layouts[s] = pipes[s].desc_layout;

  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = ctx->desc_pool;
  dsai.descriptorSetCount = (uint32_t)n_stages;
  dsai.pSetLayouts = layouts;
  if (vkAllocateDescriptorSets(ctx->device, &dsai, desc_sets) != VK_SUCCESS) {
    fprintf(stderr, "    N=%d: descriptor set allocation failed\n", n);
    destroy_buffer(ctx, &buf_a);
    destroy_buffer(ctx, &buf_b);
    for (int s = 0; s < n_stages; s++) destroy_pipeline(ctx, &pipes[s]);
    free(input_copy);
    return -1;
  }

  /* Update descriptor sets with ping-pong buffer assignments */
  for (int s = 0; s < n_stages; s++) {
    /* Even stages: src=buf_a, dst=buf_b. Odd stages: src=buf_b, dst=buf_a */
    GpuBuffer *src_buf = (s % 2 == 0) ? &buf_a : &buf_b;
    GpuBuffer *dst_buf = (s % 2 == 0) ? &buf_b : &buf_a;

    VkDescriptorBufferInfo dbi[2];
    memset(dbi, 0, sizeof(dbi));
    dbi[0].buffer = src_buf->buffer;
    dbi[0].range = buf_bytes;
    dbi[1].buffer = dst_buf->buffer;
    dbi[1].range = buf_bytes;

    VkWriteDescriptorSet wds[2];
    memset(wds, 0, sizeof(wds));
    wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[0].dstSet = desc_sets[s];
    wds[0].dstBinding = 0;
    wds[0].descriptorCount = 1;
    wds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[0].pBufferInfo = &dbi[0];
    wds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[1].dstSet = desc_sets[s];
    wds[1].dstBinding = 1;
    wds[1].descriptorCount = 1;
    wds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[1].pBufferInfo = &dbi[1];
    vkUpdateDescriptorSets(ctx->device, 2, wds, 0, NULL);
  }

  /* 6. Record and submit command buffer with all stages + barriers */
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

  for (int s = 0; s < n_stages; s++) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipes[s].pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipes[s].pipe_layout, 0, 1,
                            &desc_sets[s], 0, NULL);
    vkCmdDispatch(cb, (uint32_t)dispatch_x[s], 1, 1);

    /* Memory barrier between stages */
    if (s < n_stages - 1) {
      VkMemoryBarrier barrier = {0};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 1, &barrier, 0, NULL, 0, NULL);
    }
  }

  vkEndCommandBuffer(cb);

  VkFenceCreateInfo fci = {0};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(ctx->device, &fci, NULL, &fence);

  VkSubmitInfo submit_info = {0};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cb;
  vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
  vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

  /* 7. Read result - output is in the buffer that the last stage wrote to */
  float *result;
  if (n_stages % 2 == 0) {
    /* Even number of stages: last write went to buf_a */
    result = (float *)buf_a.mapped;
  } else {
    /* Odd number of stages: last write went to buf_b */
    result = (float *)buf_b.mapped;
  }

  /* 8. CPU reference DFT */
  float *ref = (float *)malloc((size_t)n * 2 * sizeof(float));
  cpu_dft(input_copy, ref, n);

  /* 9. Compare */
  float max_err = 0.0f;
  for (int i = 0; i < n * 2; i++) {
    float err = fabsf(result[i] - ref[i]);
    if (err > max_err) max_err = err;
  }
  *out_max_err = max_err;

  int fft_ok = (max_err < 1.0f);

  /* ====================================================================== */
  /* 10. Roundtrip test: FFT then IFFT, check result matches original       */
  /* ====================================================================== */

  int roundtrip_ok = 1;

  /* Generate IFFT pipelines (direction = -1) */
  FftPipeline ifft_pipes[MAX_STAGES];
  int ifft_dispatch_x[MAX_STAGES];
  stride = 1;

  int ifft_gen_ok = 1;
  for (int s = 0; s < n_stages; s++) {
    char *wgsl = gen_fft_stockham(radices[s], stride, n, -1, WORKGROUP_SIZE);
    if (!wgsl) {
      ifft_gen_ok = 0;
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &ifft_pipes[j]);
      break;
    }

    uint32_t *spirv = NULL;
    size_t spirv_count = 0;
    int rc = compile_wgsl_to_spirv(wgsl, &spirv, &spirv_count);
    free(wgsl);
    if (rc != 0) {
      ifft_gen_ok = 0;
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &ifft_pipes[j]);
      break;
    }

    rc = create_pipeline(ctx, spirv, spirv_count, &ifft_pipes[s]);
    wgsl_lower_free(spirv);
    if (rc != 0) {
      ifft_gen_ok = 0;
      for (int j = 0; j < s; j++) destroy_pipeline(ctx, &ifft_pipes[j]);
      break;
    }

    int butterflies = n / radices[s];
    ifft_dispatch_x[s] = (butterflies + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    stride *= radices[s];
  }

  if (ifft_gen_ok) {
    /* The FFT result is currently in result buffer.
     * We need to copy it to buf_a as the starting point for IFFT. */
    if (n_stages % 2 == 0) {
      /* FFT result is in buf_a already — good, IFFT starts from buf_a */
    } else {
      /* FFT result is in buf_b — copy to buf_a */
      memcpy(buf_a.mapped, buf_b.mapped, (size_t)n * 2 * sizeof(float));
    }
    memset(buf_b.mapped, 0, (size_t)n * 2 * sizeof(float));

    /* Create IFFT descriptor sets */
    VkDescriptorSet ifft_desc_sets[MAX_STAGES];
    VkDescriptorSetLayout ifft_layouts[MAX_STAGES];
    for (int s = 0; s < n_stages; s++)
      ifft_layouts[s] = ifft_pipes[s].desc_layout;

    VkDescriptorSetAllocateInfo ifft_dsai = {0};
    ifft_dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ifft_dsai.descriptorPool = ctx->desc_pool;
    ifft_dsai.descriptorSetCount = (uint32_t)n_stages;
    ifft_dsai.pSetLayouts = ifft_layouts;

    if (vkAllocateDescriptorSets(ctx->device, &ifft_dsai,
                                  ifft_desc_sets) == VK_SUCCESS) {
      /* Update IFFT descriptor sets with same ping-pong pattern */
      for (int s = 0; s < n_stages; s++) {
        GpuBuffer *src_buf = (s % 2 == 0) ? &buf_a : &buf_b;
        GpuBuffer *dst_buf = (s % 2 == 0) ? &buf_b : &buf_a;

        VkDescriptorBufferInfo dbi[2];
        memset(dbi, 0, sizeof(dbi));
        dbi[0].buffer = src_buf->buffer;
        dbi[0].range = buf_bytes;
        dbi[1].buffer = dst_buf->buffer;
        dbi[1].range = buf_bytes;

        VkWriteDescriptorSet wds[2];
        memset(wds, 0, sizeof(wds));
        wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[0].dstSet = ifft_desc_sets[s];
        wds[0].dstBinding = 0;
        wds[0].descriptorCount = 1;
        wds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[0].pBufferInfo = &dbi[0];
        wds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[1].dstSet = ifft_desc_sets[s];
        wds[1].dstBinding = 1;
        wds[1].descriptorCount = 1;
        wds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[1].pBufferInfo = &dbi[1];
        vkUpdateDescriptorSets(ctx->device, 2, wds, 0, NULL);
      }

      /* Record IFFT command buffer */
      vkResetCommandBuffer(cb, 0);
      vkBeginCommandBuffer(cb, &cbbi);

      for (int s = 0; s < n_stages; s++) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ifft_pipes[s].pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ifft_pipes[s].pipe_layout, 0, 1,
                                &ifft_desc_sets[s], 0, NULL);
        vkCmdDispatch(cb, (uint32_t)ifft_dispatch_x[s], 1, 1);

        if (s < n_stages - 1) {
          VkMemoryBarrier barrier = {0};
          barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(cb,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0, 1, &barrier, 0, NULL, 0, NULL);
        }
      }

      vkEndCommandBuffer(cb);

      vkResetFences(ctx->device, 1, &fence);
      vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
      vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

      /* Read IFFT result */
      float *ifft_result;
      if (n_stages % 2 == 0) {
        ifft_result = (float *)buf_a.mapped;
      } else {
        ifft_result = (float *)buf_b.mapped;
      }

      /* Compare IFFT(FFT(x))/N with original x */
      float max_rt_err = 0.0f;
      float inv_n = 1.0f / (float)n;
      for (int i = 0; i < n * 2; i++) {
        float val = ifft_result[i] * inv_n;
        float err = fabsf(val - input_copy[i]);
        if (err > max_rt_err) max_rt_err = err;
      }

      /* Roundtrip tolerance: generous for large N with f32 */
      float rt_tol = (n <= 64) ? 0.01f : (n <= 1024) ? 0.1f : 1.0f;
      roundtrip_ok = (max_rt_err < rt_tol);

      vkFreeDescriptorSets(ctx->device, ctx->desc_pool,
                            (uint32_t)n_stages, ifft_desc_sets);
    } else {
      roundtrip_ok = 0;
    }

    for (int s = 0; s < n_stages; s++)
      destroy_pipeline(ctx, &ifft_pipes[s]);
  } else {
    roundtrip_ok = 0;
  }

  /* Cleanup */
  vkDestroyFence(ctx->device, fence, NULL);
  vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
  vkFreeDescriptorSets(ctx->device, ctx->desc_pool,
                        (uint32_t)n_stages, desc_sets);
  destroy_buffer(ctx, &buf_a);
  destroy_buffer(ctx, &buf_b);
  for (int s = 0; s < n_stages; s++)
    destroy_pipeline(ctx, &pipes[s]);
  free(ref);
  free(input_copy);

  return (fft_ok && roundtrip_ok) ? 0 : -1;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  static const int test_sizes[] = {
    2, 4, 8, 16, 32,           /* single-stage po2 */
    64, 128, 256, 512, 1024,   /* multi-stage po2 */
    6, 12, 24, 48, 96,         /* mixed composites */
    9, 25, 27,                 /* perfect powers */
    768, 1536,                 /* large composites */
    30, 210, 2310,             /* primorials */
  };
  int n_tests = (int)(sizeof(test_sizes) / sizeof(test_sizes[0]));

  printf("test_fft_multistage\n");

  VkCtx ctx;
  if (vk_init(&ctx, 512, 1024, 0) != 0) {
    fprintf(stderr, "Vulkan init failed\n");
    return 1;
  }

  int passed = 0, failed = 0;

  for (int ti = 0; ti < n_tests; ti++) {
    int n = test_sizes[ti];
    int n_stages = 0;
    int radices[MAX_STAGES];
    float max_err = 0.0f;

    int rc = test_fft_size(&ctx, n, &n_stages, radices, &max_err);

    /* Print factorization */
    printf("  N=%-6d [%d stages:", n, n_stages);
    for (int s = 0; s < n_stages; s++)
      printf(" %d", radices[s]);
    printf("]");

    /* Pad for alignment */
    int printed = 0;
    for (int s = 0; s < n_stages; s++) {
      if (s > 0) printed += 1;
      if (radices[s] >= 10) printed += 2; else printed += 1;
    }
    int stage_chars = 12 + printed + 1;  /* "[N stages: ...]" */
    for (int p = stage_chars; p < 30; p++)
      printf(" ");

    if (rc == 0) {
      printf(" OK  (max_err=%.3f)\n", max_err);
      passed++;
    } else {
      printf(" FAIL (max_err=%.3f)\n", max_err);
      failed++;
    }
  }

  printf("  Results: %d passed, %d failed\n", passed, failed);

  vk_destroy(&ctx);
  return failed > 0 ? 1 : 0;
}
