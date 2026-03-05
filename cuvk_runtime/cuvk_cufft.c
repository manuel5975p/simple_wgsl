/*
 * cuvk_cufft.c - cuFFT implementation for the CUDA-on-Vulkan runtime
 *
 * GPU-accelerated 1D FFT via Vulkan compute shaders. Shaders are compiled
 * from WGSL source at plan creation time using the project's WGSL compiler.
 * Cooley-Tukey radix-2 DIT with separate dispatches per butterfly stage.
 */

#include "cuvk_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * cuFFT types (must match cufft.h)
 * ============================================================================ */

typedef enum {
    CUFFT_SUCCESS            = 0,
    CUFFT_INVALID_PLAN       = 1,
    CUFFT_ALLOC_FAILED       = 2,
    CUFFT_INVALID_TYPE       = 3,
    CUFFT_INVALID_VALUE      = 4,
    CUFFT_INTERNAL_ERROR     = 5,
    CUFFT_EXEC_FAILED        = 6,
    CUFFT_SETUP_FAILED       = 7,
    CUFFT_INVALID_SIZE       = 8,
    CUFFT_UNALIGNED_DATA     = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE     = 11,
    CUFFT_PARSE_ERROR        = 12,
    CUFFT_NO_WORKSPACE       = 13,
    CUFFT_NOT_IMPLEMENTED    = 14,
    CUFFT_LICENSE_ERROR      = 15,
    CUFFT_NOT_SUPPORTED      = 16,
} cufftResult;

typedef enum {
    CUFFT_R2C = 0x2a,
    CUFFT_C2R = 0x2c,
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69,
} cufftType;

#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE  (1)

typedef int cufftHandle;
typedef float cufftReal;
typedef struct { float x, y; } cufftComplex;
typedef double cufftDoubleReal;
typedef struct { double x, y; } cufftDoubleComplex;

typedef enum {
    CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01,
} cufftCompatibility;

/* ============================================================================
 * Internal plan structure
 * ============================================================================ */

#define MAX_CUFFT_PLANS 64
#define FFT_WORKGROUP_SIZE 256

typedef struct {
    struct CUctx_st *ctx;
    int nx;
    int log2n;
    cufftType type;
    int valid;

    VkShaderModule butterfly_shader;
    VkPipelineLayout butterfly_layout;
    VkDescriptorSetLayout butterfly_desc_layout;
    VkPipeline butterfly_pipeline;

    VkShaderModule bitrev_shader;
    VkPipelineLayout bitrev_layout;
    VkDescriptorSetLayout bitrev_desc_layout;
    VkPipeline bitrev_pipeline;

    VkBuffer twiddle_buf;
    VkDeviceMemory twiddle_mem;

    VkBuffer temp_buf;
    VkDeviceMemory temp_mem;
    VkDeviceSize temp_size;

    VkDescriptorPool desc_pool;
} CufftPlan;

static CufftPlan g_cufft_plans[MAX_CUFFT_PLANS];
static int g_cufft_next_handle = 0;

/* ============================================================================
 * WGSL shader sources
 * ============================================================================ */

static const char *k_butterfly_wgsl =
    "struct Params { stage: u32, direction: i32, n: u32 };\n"
    "struct DataBuf { d: array<f32> };\n"
    "struct TwBuf { d: array<f32> };\n"
    "@group(0) @binding(0) var<storage, read_write> data: DataBuf;\n"
    "@group(0) @binding(1) var<storage, read> twiddles: TwBuf;\n"
    "@group(0) @binding(2) var<uniform> params: Params;\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let tid: u32 = gid.x;\n"
    "  let half_size: u32 = 1u << params.stage;\n"
    "  let n_half: u32 = params.n >> 1u;\n"
    "  if (tid >= n_half) { return; }\n"
    "  let grp: u32 = tid / half_size;\n"
    "  let idx_in_grp: u32 = tid - grp * half_size;\n"
    "  let even: u32 = grp * half_size * 2u + idx_in_grp;\n"
    "  let odd: u32 = even + half_size;\n"
    "  let tw_idx: u32 = idx_in_grp * (params.n / (2u * half_size));\n"
    "  let tw_re: f32 = twiddles.d[tw_idx * 2u];\n"
    "  let tw_im_base: f32 = twiddles.d[tw_idx * 2u + 1u];\n"
    "  let dir_f: f32 = -f32(params.direction);\n"
    "  let tw_im: f32 = tw_im_base * dir_f;\n"
    "  let ei: u32 = even * 2u;\n"
    "  let oi: u32 = odd * 2u;\n"
    "  let e_re: f32 = data.d[ei];\n"
    "  let e_im: f32 = data.d[ei + 1u];\n"
    "  let o_re: f32 = data.d[oi];\n"
    "  let o_im: f32 = data.d[oi + 1u];\n"
    "  let t_re: f32 = tw_re * o_re - tw_im * o_im;\n"
    "  let t_im: f32 = tw_re * o_im + tw_im * o_re;\n"
    "  data.d[ei]      = e_re + t_re;\n"
    "  data.d[ei + 1u] = e_im + t_im;\n"
    "  data.d[oi]      = e_re - t_re;\n"
    "  data.d[oi + 1u] = e_im - t_im;\n"
    "}\n";

static const char *k_bitrev_wgsl =
    "struct Params { log2n: u32, n: u32, pad0: u32 };\n"
    "struct DstBuf { d: array<f32> };\n"
    "struct SrcBuf { d: array<f32> };\n"
    "@group(0) @binding(0) var<storage, read_write> dst: DstBuf;\n"
    "@group(0) @binding(1) var<storage, read> src: SrcBuf;\n"
    "@group(0) @binding(2) var<uniform> params: Params;\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let i: u32 = gid.x;\n"
    "  if (i >= params.n) { return; }\n"
    "  var rev: u32 = 0u;\n"
    "  var v: u32 = 0u;\n"
    "  v = i;\n"
    "  for (var b: u32 = 0u; b < params.log2n; b = b + 1u) {\n"
    "    rev = (rev << 1u) | (v & 1u);\n"
    "    v = v >> 1u;\n"
    "  }\n"
    "  dst.d[rev * 2u]      = src.d[i * 2u];\n"
    "  dst.d[rev * 2u + 1u] = src.d[i * 2u + 1u];\n"
    "}\n";

/* ============================================================================
 * Helper: compile WGSL source to SPIR-V
 * ============================================================================ */

static cufftResult compile_wgsl(const char *wgsl_source,
                                uint32_t **out_words,
                                size_t *out_count)
{
    WgslAstNode *ast = wgsl_parse(wgsl_source);
    if (!ast) {
        CUVK_LOG("[cufft] wgsl_parse failed\n");
        return CUFFT_INTERNAL_ERROR;
    }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        CUVK_LOG("[cufft] wgsl_resolver_build failed\n");
        wgsl_free_ast(ast);
        return CUFFT_INTERNAL_ERROR;
    }

    WgslLowerOptions opts = {0};
    opts.spirv_version = 0x00010300;
    opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
    opts.packing = WGSL_LOWER_PACK_STD430;
    opts.enable_debug_names = 0;

    WgslLowerResult lr = wgsl_lower_emit_spirv(ast, resolver, &opts,
                                                out_words, out_count);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lr != WGSL_LOWER_OK) {
        CUVK_LOG("[cufft] wgsl_lower_emit_spirv failed: %d\n", lr);
        return CUFFT_INTERNAL_ERROR;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: create a Vulkan buffer with device-local memory
 * ============================================================================ */

static cufftResult create_device_buffer(struct CUctx_st *ctx,
                                        VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkBuffer *out_buf,
                                        VkDeviceMemory *out_mem)
{
    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, out_buf);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, *out_buf, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, out_mem);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    vr = vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0);
    if (vr != VK_SUCCESS) {
        vkFreeMemory(ctx->device, *out_mem, NULL);
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: create compute pipeline from SPIR-V
 * ============================================================================ */

static void patch_local_size(uint32_t *spirv, size_t count,
                             uint32_t x, uint32_t y, uint32_t z)
{
    for (size_t i = 5; i < count; ) {
        uint32_t word = spirv[i];
        uint16_t opcode = word & 0xFFFF;
        uint16_t wc = word >> 16;
        if (wc == 0) break;
        if (opcode == 16 && wc == 6 && i + 5 < count &&
            spirv[i + 2] == 17) {
            CUVK_LOG("[cufft] patch_local_size: old=(%u,%u,%u) new=(%u,%u,%u)\n",
                     spirv[i + 3], spirv[i + 4], spirv[i + 5], x, y, z);
            spirv[i + 3] = x;
            spirv[i + 4] = y;
            spirv[i + 5] = z;
            return;
        }
        i += wc;
    }
}

static cufftResult create_compute_pipeline(
    struct CUctx_st *ctx,
    uint32_t *spirv_words, size_t spirv_count,
    uint32_t local_size_x,
    VkShaderModule *out_shader,
    VkDescriptorSetLayout *out_desc_layout,
    VkPipelineLayout *out_pipe_layout,
    VkPipeline *out_pipeline)
{
    patch_local_size(spirv_words, spirv_count, local_size_x, 1, 1);

    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spirv_count * sizeof(uint32_t);
    sm_ci.pCode = spirv_words;

    VkResult vr = vkCreateShaderModule(ctx->device, &sm_ci, NULL, out_shader);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    /* 3 bindings: storage RW (data/dst), storage R (twiddles/src), uniform (params) */
    VkDescriptorSetLayoutBinding bindings[3] = {{0}};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 3;
    dsl_ci.pBindings = bindings;
    vr = vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL,
                                      out_desc_layout);
    if (vr != VK_SUCCESS) {
        vkDestroyShaderModule(ctx->device, *out_shader, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = out_desc_layout;

    vr = vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, out_pipe_layout);
    if (vr != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx->device, *out_desc_layout, NULL);
        vkDestroyShaderModule(ctx->device, *out_shader, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = *out_shader;
    pipe_ci.stage.pName = "main";
    pipe_ci.layout = *out_pipe_layout;

    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE,
                                  1, &pipe_ci, NULL, out_pipeline);
    if (vr != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx->device, *out_pipe_layout, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, *out_desc_layout, NULL);
        vkDestroyShaderModule(ctx->device, *out_shader, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: upload data to a device buffer via staging
 * ============================================================================ */

static cufftResult upload_to_buffer(struct CUctx_st *ctx,
                                    VkBuffer dst_buf,
                                    const void *data, size_t size)
{
    /* Create staging buffer */
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;

    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = (VkDeviceSize)size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, &staging);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, staging, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, staging, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, &staging_mem);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, staging, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    vkBindBufferMemory(ctx->device, staging, staging_mem, 0);

    void *mapped = NULL;
    vkMapMemory(ctx->device, staging_mem, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(ctx->device, staging_mem);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    CUresult res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS) {
        vkFreeMemory(ctx->device, staging_mem, NULL);
        vkDestroyBuffer(ctx->device, staging, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    VkBufferCopy region = {0};
    region.size = (VkDeviceSize)size;
    vkCmdCopyBuffer(cb, staging, dst_buf, 1, &region);

    res = cuvk_oneshot_end(ctx, cb);

    vkFreeMemory(ctx->device, staging_mem, NULL);
    vkDestroyBuffer(ctx->device, staging, NULL);

    return (res == CUDA_SUCCESS) ? CUFFT_SUCCESS : CUFFT_INTERNAL_ERROR;
}

/* ============================================================================
 * Helper: create uniform buffer with host-visible memory
 * ============================================================================ */

static cufftResult create_uniform_buffer(struct CUctx_st *ctx,
                                         VkDeviceSize size,
                                         VkBuffer *out_buf,
                                         VkDeviceMemory *out_mem,
                                         void **out_mapped)
{
    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, out_buf);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, *out_buf, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, out_mem);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0);
    vkMapMemory(ctx->device, *out_mem, 0, size, 0, out_mapped);

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * cufftPlan1d
 * ============================================================================ */

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
{
    CUVK_LOG("[cufft] cufftPlan1d: nx=%d type=0x%x batch=%d\n", nx, type, batch);

    if (!plan || nx <= 0)
        return CUFFT_INVALID_VALUE;

    /* Must be power of two */
    if ((nx & (nx - 1)) != 0)
        return CUFFT_INVALID_SIZE;

    if (type != CUFFT_C2C && type != CUFFT_R2C)
        return CUFFT_INVALID_TYPE;

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return CUFFT_SETUP_FAILED;

    if (!g_cuvk.current_ctx) {
        CUcontext ctx;
        r = cuDevicePrimaryCtxRetain(&ctx, 0);
        if (r != CUDA_SUCCESS) return CUFFT_SETUP_FAILED;
    }

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) return CUFFT_SETUP_FAILED;

    if (g_cufft_next_handle >= MAX_CUFFT_PLANS)
        return CUFFT_ALLOC_FAILED;

    int handle = g_cufft_next_handle++;
    CufftPlan *p = &g_cufft_plans[handle];
    memset(p, 0, sizeof(*p));
    p->ctx = ctx;
    p->nx = nx;
    p->type = type;

    int log2n = 0;
    { int tmp = nx; while (tmp > 1) { tmp >>= 1; log2n++; } }
    p->log2n = log2n;

    /* Compile butterfly shader */
    uint32_t *butterfly_spirv = NULL;
    size_t butterfly_count = 0;
    cufftResult cr = compile_wgsl(k_butterfly_wgsl,
                                   &butterfly_spirv, &butterfly_count);
    if (cr != CUFFT_SUCCESS) return cr;

    cr = create_compute_pipeline(ctx, butterfly_spirv, butterfly_count,
                                  FFT_WORKGROUP_SIZE,
                                  &p->butterfly_shader,
                                  &p->butterfly_desc_layout,
                                  &p->butterfly_layout,
                                  &p->butterfly_pipeline);
    wgsl_lower_free(butterfly_spirv);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Compile bit-reversal shader */
    uint32_t *bitrev_spirv = NULL;
    size_t bitrev_count = 0;
    cr = compile_wgsl(k_bitrev_wgsl, &bitrev_spirv, &bitrev_count);
    if (cr != CUFFT_SUCCESS) return cr;

    cr = create_compute_pipeline(ctx, bitrev_spirv, bitrev_count,
                                  FFT_WORKGROUP_SIZE,
                                  &p->bitrev_shader,
                                  &p->bitrev_desc_layout,
                                  &p->bitrev_layout,
                                  &p->bitrev_pipeline);
    wgsl_lower_free(bitrev_spirv);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Create descriptor pool: need 2 sets (butterfly + bitrev), each with
       2 storage + 1 uniform, but we allocate per-exec so need more */
    VkDescriptorPoolSize pool_sizes[2] = {{0}};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 32;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 16;

    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_ci.maxSets = 16;
    dp_ci.poolSizeCount = 2;
    dp_ci.pPoolSizes = pool_sizes;

    VkResult vr = vkCreateDescriptorPool(ctx->device, &dp_ci, NULL,
                                          &p->desc_pool);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    /* Precompute twiddle factors and upload */
    int n_half = nx / 2;
    size_t twiddle_bytes = (size_t)n_half * 2 * sizeof(float);
    float *twiddles = (float *)malloc(twiddle_bytes);
    if (!twiddles) return CUFFT_ALLOC_FAILED;

    for (int k = 0; k < n_half; k++) {
        double angle = -2.0 * M_PI * k / nx;
        twiddles[(size_t)k * 2]     = (float)cos(angle);
        twiddles[(size_t)k * 2 + 1] = (float)sin(angle);
    }

    cr = create_device_buffer(ctx, (VkDeviceSize)twiddle_bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &p->twiddle_buf, &p->twiddle_mem);
    if (cr != CUFFT_SUCCESS) { free(twiddles); return cr; }

    cr = upload_to_buffer(ctx, p->twiddle_buf, twiddles, twiddle_bytes);
    free(twiddles);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Create temp buffer for bit-reversal (same size as data) */
    size_t data_bytes = (size_t)nx * 2 * sizeof(float);
    cr = create_device_buffer(ctx, (VkDeviceSize)data_bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              &p->temp_buf, &p->temp_mem);
    if (cr != CUFFT_SUCCESS) return cr;
    p->temp_size = (VkDeviceSize)data_bytes;

    p->valid = 1;
    *plan = handle;
    CUVK_LOG("[cufft] cufftPlan1d: SUCCESS handle=%d\n", handle);
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: execute FFT on a data buffer (C2C core)
 * ============================================================================ */

static cufftResult exec_fft(CufftPlan *p, VkBuffer data_buf,
                            VkDeviceSize data_size, int direction)
{
    struct CUctx_st *ctx = p->ctx;
    int nx = p->nx;
    int n_half = nx / 2;
    uint32_t wg_count = ((uint32_t)n_half + FFT_WORKGROUP_SIZE - 1) /
                        FFT_WORKGROUP_SIZE;
    uint32_t wg_count_n = ((uint32_t)nx + FFT_WORKGROUP_SIZE - 1) /
                          FFT_WORKGROUP_SIZE;

    VkBuffer param_buf = VK_NULL_HANDLE;
    VkDeviceMemory param_mem = VK_NULL_HANDLE;
    void *param_mapped = NULL;

    cufftResult cr = create_uniform_buffer(ctx, 16,
                                            &param_buf, &param_mem,
                                            &param_mapped);
    if (cr != CUFFT_SUCCESS) return cr;

    VkMemoryBarrier mem_bar = {0};
    mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;

    /* --- Bitrev: copy data->temp, barrier, compute bitrev temp->data --- */
    {
        uint32_t bitrev_params[4];
        bitrev_params[0] = (uint32_t)p->log2n;
        bitrev_params[1] = (uint32_t)nx;
        bitrev_params[2] = 0;
        bitrev_params[3] = 0;
        memcpy(param_mapped, bitrev_params, 16);

        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo ds_ai = {0};
        ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_ai.descriptorPool = p->desc_pool;
        ds_ai.descriptorSetCount = 1;
        ds_ai.pSetLayouts = &p->bitrev_desc_layout;
        VkResult vr = vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds);
        if (vr != VK_SUCCESS) goto fail;

        VkDescriptorBufferInfo buf_infos[3] = {{0}};
        buf_infos[0].buffer = data_buf;
        buf_infos[0].range = data_size;
        buf_infos[1].buffer = p->temp_buf;
        buf_infos[1].range = data_size;
        buf_infos[2].buffer = param_buf;
        buf_infos[2].range = 16;

        VkWriteDescriptorSet writes[3] = {{0}};
        for (int i = 0; i < 3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i < 2)
                ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult r = cuvk_oneshot_begin(ctx, &cb);
        if (r != CUDA_SUCCESS) {
            vkFreeDescriptorSets(ctx->device, p->desc_pool, 1, &ds);
            goto fail;
        }

        /* Copy data -> temp */
        VkBufferCopy region = {0};
        region.size = data_size;
        vkCmdCopyBuffer(cb, data_buf, p->temp_buf, 1, &region);

        /* Barrier: transfer write -> compute read */
        mem_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mem_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_bar, 0, NULL, 0, NULL);

        /* Dispatch bitrev: reads temp, writes data */
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          p->bitrev_pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                p->bitrev_layout, 0, 1, &ds, 0, NULL);
        vkCmdDispatch(cb, wg_count_n, 1, 1);

        r = cuvk_oneshot_end(ctx, cb);
        vkFreeDescriptorSets(ctx->device, p->desc_pool, 1, &ds);
        if (r != CUDA_SUCCESS) goto fail;
    }

    /* --- Butterfly passes --- */
    {
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo ds_ai = {0};
        ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_ai.descriptorPool = p->desc_pool;
        ds_ai.descriptorSetCount = 1;
        ds_ai.pSetLayouts = &p->butterfly_desc_layout;
        VkResult vr = vkAllocateDescriptorSets(ctx->device, &ds_ai, &ds);
        if (vr != VK_SUCCESS) goto fail;

        VkDescriptorBufferInfo buf_infos[3] = {{0}};
        buf_infos[0].buffer = data_buf;
        buf_infos[0].range = data_size;
        buf_infos[1].buffer = p->twiddle_buf;
        buf_infos[1].range = (VkDeviceSize)n_half * 2 * sizeof(float);
        buf_infos[2].buffer = param_buf;
        buf_infos[2].range = 16;

        VkWriteDescriptorSet writes[3] = {{0}};
        for (int i = 0; i < 3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i < 2)
                ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

        for (int stage = 0; stage < p->log2n; stage++) {
            uint32_t params[4];
            params[0] = (uint32_t)stage;
            memcpy(&params[1], &direction, sizeof(int));
            params[2] = (uint32_t)nx;
            params[3] = 0;
            memcpy(param_mapped, params, 16);

            VkCommandBuffer cb = VK_NULL_HANDLE;
            CUresult r = cuvk_oneshot_begin(ctx, &cb);
            if (r != CUDA_SUCCESS) {
                vkFreeDescriptorSets(ctx->device, p->desc_pool, 1, &ds);
                goto fail;
            }

            /* Barrier: previous compute write -> current compute read */
            mem_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mem_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_bar, 0, NULL, 0, NULL);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              p->butterfly_pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    p->butterfly_layout, 0, 1, &ds,
                                    0, NULL);
            vkCmdDispatch(cb, wg_count, 1, 1);

            r = cuvk_oneshot_end(ctx, cb);
            if (r != CUDA_SUCCESS) {
                vkFreeDescriptorSets(ctx->device, p->desc_pool, 1, &ds);
                goto fail;
            }
        }

        vkFreeDescriptorSets(ctx->device, p->desc_pool, 1, &ds);
    }

    vkUnmapMemory(ctx->device, param_mem);
    vkFreeMemory(ctx->device, param_mem, NULL);
    vkDestroyBuffer(ctx->device, param_buf, NULL);
    return CUFFT_SUCCESS;

fail:
    vkUnmapMemory(ctx->device, param_mem);
    vkFreeMemory(ctx->device, param_mem, NULL);
    vkDestroyBuffer(ctx->device, param_buf, NULL);
    return CUFFT_EXEC_FAILED;
}

/* ============================================================================
 * cufftExecC2C
 * ============================================================================ */

cufftResult cufftExecC2C(cufftHandle plan_handle,
                          cufftComplex *idata,
                          cufftComplex *odata,
                          int direction)
{
    CUVK_LOG("[cufft] cufftExecC2C: plan=%d dir=%d\n", plan_handle, direction);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) ctx = p->ctx;
    CUdeviceptr iptr = (CUdeviceptr)idata;
    CUdeviceptr optr = (CUdeviceptr)odata;

    CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, iptr);
    if (!alloc_in) return CUFFT_INVALID_VALUE;

    VkDeviceSize data_size = (VkDeviceSize)p->nx * 2 * sizeof(float);

    /* If out-of-place, copy input to output first */
    if (iptr != optr) {
        CuvkAlloc *alloc_out = cuvk_alloc_lookup(ctx, optr);
        if (!alloc_out) return CUFFT_INVALID_VALUE;

        VkDeviceSize src_off = (VkDeviceSize)((uint64_t)iptr -
                                (uint64_t)alloc_in->device_addr);
        VkDeviceSize dst_off = (VkDeviceSize)((uint64_t)optr -
                                (uint64_t)alloc_out->device_addr);

        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult r = cuvk_oneshot_begin(ctx, &cb);
        if (r != CUDA_SUCCESS) return CUFFT_EXEC_FAILED;

        VkBufferCopy region = {0};
        region.srcOffset = src_off;
        region.dstOffset = dst_off;
        region.size = data_size;
        vkCmdCopyBuffer(cb, alloc_in->buffer, alloc_out->buffer, 1, &region);
        r = cuvk_oneshot_end(ctx, cb);
        if (r != CUDA_SUCCESS) return CUFFT_EXEC_FAILED;
    }

    /* Work on the output buffer */
    CuvkAlloc *alloc_work = cuvk_alloc_lookup(ctx, optr);
    if (!alloc_work) return CUFFT_INVALID_VALUE;

    return exec_fft(p, alloc_work->buffer, data_size, direction);
}

/* ============================================================================
 * cufftExecR2C
 * ============================================================================ */

cufftResult cufftExecR2C(cufftHandle plan_handle,
                          cufftReal *idata,
                          cufftComplex *odata)
{
    CUVK_LOG("[cufft] cufftExecR2C: plan=%d\n", plan_handle);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = p->ctx;
    int nx = p->nx;
    VkDeviceSize complex_size = (VkDeviceSize)nx * 2 * sizeof(float);

    /* Allocate temporary complex buffer for the full C2C transform */
    VkBuffer work_buf = VK_NULL_HANDLE;
    VkDeviceMemory work_mem = VK_NULL_HANDLE;
    cufftResult cr = create_device_buffer(ctx, complex_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &work_buf, &work_mem);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Zero the work buffer first */
    {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult r = cuvk_oneshot_begin(ctx, &cb);
        if (r != CUDA_SUCCESS) goto r2c_fail;
        vkCmdFillBuffer(cb, work_buf, 0, complex_size, 0);
        r = cuvk_oneshot_end(ctx, cb);
        if (r != CUDA_SUCCESS) goto r2c_fail;
    }

    /* Copy real data into even-indexed floats (real parts) of complex buffer.
       We need to interleave: src[i] -> dst[i*2], dst[i*2+1] = 0.
       Since we zeroed above, we just need to scatter the reals.
       Use a staging approach: download reals, pack as complex, re-upload. */
    {
        CUdeviceptr iptr = (CUdeviceptr)idata;
        CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, iptr);
        if (!alloc_in) goto r2c_fail;

        size_t real_bytes = (size_t)nx * sizeof(float);

        /* Download reals to host */
        float *h_real = (float *)malloc(real_bytes);
        if (!h_real) goto r2c_fail;

        CUresult r = cuMemcpyDtoH_v2(h_real, iptr, real_bytes);
        if (r != CUDA_SUCCESS) { free(h_real); goto r2c_fail; }

        /* Pack as interleaved complex */
        float *h_complex = (float *)calloc((size_t)nx * 2, sizeof(float));
        if (!h_complex) { free(h_real); goto r2c_fail; }
        for (int i = 0; i < nx; i++) {
            h_complex[(size_t)i * 2] = h_real[i];
        }
        free(h_real);

        /* Upload to work buffer */
        cr = upload_to_buffer(ctx, work_buf, h_complex, (size_t)nx * 2 * sizeof(float));
        free(h_complex);
        if (cr != CUFFT_SUCCESS) goto r2c_fail;
    }

    /* Run forward C2C on work buffer */
    cr = exec_fft(p, work_buf, complex_size, CUFFT_FORWARD);
    if (cr != CUFFT_SUCCESS) goto r2c_fail;

    /* Copy first N/2+1 complex values to output */
    {
        CUdeviceptr optr = (CUdeviceptr)odata;
        CuvkAlloc *alloc_out = cuvk_alloc_lookup(ctx, optr);
        if (!alloc_out) goto r2c_fail;

        VkDeviceSize dst_off = (VkDeviceSize)((uint64_t)optr -
                                (uint64_t)alloc_out->device_addr);
        VkDeviceSize copy_size = (VkDeviceSize)(nx / 2 + 1) * 2 * sizeof(float);

        VkCommandBuffer cb = VK_NULL_HANDLE;
        CUresult r = cuvk_oneshot_begin(ctx, &cb);
        if (r != CUDA_SUCCESS) goto r2c_fail;

        VkBufferCopy region = {0};
        region.srcOffset = 0;
        region.dstOffset = dst_off;
        region.size = copy_size;
        vkCmdCopyBuffer(cb, work_buf, alloc_out->buffer, 1, &region);

        r = cuvk_oneshot_end(ctx, cb);
        if (r != CUDA_SUCCESS) goto r2c_fail;
    }

    vkFreeMemory(ctx->device, work_mem, NULL);
    vkDestroyBuffer(ctx->device, work_buf, NULL);
    return CUFFT_SUCCESS;

r2c_fail:
    vkFreeMemory(ctx->device, work_mem, NULL);
    vkDestroyBuffer(ctx->device, work_buf, NULL);
    return CUFFT_EXEC_FAILED;
}

/* ============================================================================
 * cufftDestroy
 * ============================================================================ */

cufftResult cufftDestroy(cufftHandle plan_handle)
{
    CUVK_LOG("[cufft] cufftDestroy: plan=%d\n", plan_handle);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = p->ctx;
    vkDeviceWaitIdle(ctx->device);

    if (p->butterfly_pipeline)
        vkDestroyPipeline(ctx->device, p->butterfly_pipeline, NULL);
    if (p->butterfly_layout)
        vkDestroyPipelineLayout(ctx->device, p->butterfly_layout, NULL);
    if (p->butterfly_desc_layout)
        vkDestroyDescriptorSetLayout(ctx->device,
                                      p->butterfly_desc_layout, NULL);
    if (p->butterfly_shader)
        vkDestroyShaderModule(ctx->device, p->butterfly_shader, NULL);

    if (p->bitrev_pipeline)
        vkDestroyPipeline(ctx->device, p->bitrev_pipeline, NULL);
    if (p->bitrev_layout)
        vkDestroyPipelineLayout(ctx->device, p->bitrev_layout, NULL);
    if (p->bitrev_desc_layout)
        vkDestroyDescriptorSetLayout(ctx->device,
                                      p->bitrev_desc_layout, NULL);
    if (p->bitrev_shader)
        vkDestroyShaderModule(ctx->device, p->bitrev_shader, NULL);

    if (p->twiddle_buf)
        vkDestroyBuffer(ctx->device, p->twiddle_buf, NULL);
    if (p->twiddle_mem)
        vkFreeMemory(ctx->device, p->twiddle_mem, NULL);

    if (p->temp_buf)
        vkDestroyBuffer(ctx->device, p->temp_buf, NULL);
    if (p->temp_mem)
        vkFreeMemory(ctx->device, p->temp_mem, NULL);

    if (p->desc_pool)
        vkDestroyDescriptorPool(ctx->device, p->desc_pool, NULL);

    p->valid = 0;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Stubs: no-op / return success
 * ============================================================================ */

cufftResult cufftSetStream(cufftHandle plan, CUstream stream) {
    (void)plan; (void)stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    (void)plan; (void)autoAllocate;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    (void)plan; (void)workArea;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetVersion(int *version) {
    if (version) *version = 12100;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(int type, int *value) {
    (void)type;
    if (value) *value = 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetCompatibilityMode(cufftHandle plan,
                                       cufftCompatibility mode) {
    (void)plan; (void)mode;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Stubs: not supported
 * ============================================================================ */

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) {
    (void)plan; (void)nx; (void)ny; (void)type;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz,
                         cufftType type) {
    (void)plan; (void)nx; (void)ny; (void)nz; (void)type;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
                           int *inembed, int istride, int idist,
                           int *onembed, int ostride, int odist,
                           cufftType type, int batch) {
    (void)plan; (void)rank; (void)n;
    (void)inembed; (void)istride; (void)idist;
    (void)onembed; (void)ostride; (void)odist;
    (void)type; (void)batch;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata,
                          cufftReal *odata) {
    (void)plan; (void)idata; (void)odata;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
                          cufftDoubleComplex *odata, int direction) {
    (void)plan; (void)idata; (void)odata; (void)direction;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
                          cufftDoubleComplex *odata) {
    (void)plan; (void)idata; (void)odata;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
                          cufftDoubleReal *odata) {
    (void)plan; (void)idata; (void)odata;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftCreate(cufftHandle *handle) {
    if (!handle) return CUFFT_INVALID_VALUE;
    *handle = -1;
    return CUFFT_SUCCESS;
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
                             int batch, size_t *workSize) {
    (void)plan; (void)nx; (void)type; (void)batch; (void)workSize;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftGetSize(cufftHandle handle, size_t *workSize) {
    (void)handle;
    if (workSize) *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate1d(int nx, cufftType type, int batch,
                             size_t *workSize) {
    (void)nx; (void)type; (void)batch;
    if (workSize) *workSize = 0;
    return CUFFT_SUCCESS;
}
