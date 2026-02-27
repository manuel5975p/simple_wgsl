/*
 * cuvk_module.c - PTX module loading, SPIR-V compilation, function extraction
 *
 * Implements the CUDA driver API functions: cuModuleLoadData,
 * cuModuleLoadDataEx, cuModuleUnload, cuModuleGetFunction,
 * cuFuncGetAttribute.
 */

#include "cuvk_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper: compare CuvkParamInfo by binding index for qsort
 * ============================================================================ */

typedef struct {
    uint32_t binding;
    CuvkParamInfo info;
} ParamBindingPair;

static int compare_by_binding(const void *a, const void *b)
{
    const ParamBindingPair *pa = (const ParamBindingPair *)a;
    const ParamBindingPair *pb = (const ParamBindingPair *)b;
    if (pa->binding < pb->binding) return -1;
    if (pa->binding > pb->binding) return 1;
    return 0;
}

/* ============================================================================
 * Helper: extract parameter metadata from SSIR globals
 *
 * For each global with group==0, builtin==0, and pointer-to-storage type,
 * we create a CuvkParamInfo entry (is_pointer=true).
 * Sort by binding index to establish parameter order.
 * ============================================================================ */

static CUresult extract_params(SsirModule *ssir,
                               CuvkParamInfo **out_params,
                               uint32_t *out_count)
{
    /* First pass: count matching globals */
    uint32_t count = 0;
    for (uint32_t i = 0; i < ssir->global_count; i++) {
        SsirGlobalVar *g = &ssir->globals[i];
        if (!g->has_group || g->group != 0)
            continue;
        if (g->builtin != SSIR_BUILTIN_NONE)
            continue;
        /* Check type: must be pointer to storage address space */
        SsirType *t = ssir_get_type(ssir, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR)
            continue;
        if (t->ptr.space != SSIR_ADDR_STORAGE)
            continue;
        count++;
    }

    if (count == 0) {
        *out_params = NULL;
        *out_count = 0;
        return CUDA_SUCCESS;
    }

    /* Collect binding + info pairs */
    ParamBindingPair *pairs = (ParamBindingPair *)calloc(count, sizeof(ParamBindingPair));
    if (!pairs)
        return CUDA_ERROR_OUT_OF_MEMORY;

    uint32_t idx = 0;
    for (uint32_t i = 0; i < ssir->global_count; i++) {
        SsirGlobalVar *g = &ssir->globals[i];
        if (!g->has_group || g->group != 0)
            continue;
        if (g->builtin != SSIR_BUILTIN_NONE)
            continue;
        SsirType *t = ssir_get_type(ssir, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR)
            continue;
        if (t->ptr.space != SSIR_ADDR_STORAGE)
            continue;

        pairs[idx].binding = g->binding;
        pairs[idx].info.is_pointer = true;
        pairs[idx].info.size = sizeof(CUdeviceptr); /* 8 bytes */
        idx++;
    }

    /* Sort by binding index */
    qsort(pairs, count, sizeof(ParamBindingPair), compare_by_binding);

    /* Extract just the CuvkParamInfo array */
    CuvkParamInfo *params = (CuvkParamInfo *)calloc(count, sizeof(CuvkParamInfo));
    if (!params) {
        free(pairs);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < count; i++)
        params[i] = pairs[i].info;

    free(pairs);
    *out_params = params;
    *out_count = count;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Helper: extract parameter metadata from push constant struct (BDA mode)
 *
 * In BDA mode, all kernel params are members of a push constant struct.
 * We iterate the struct members and create CuvkParamInfo entries.
 * ============================================================================ */

static CUresult extract_params_bda(SsirModule *ssir,
                                    CuvkParamInfo **out_params,
                                    uint32_t *out_count,
                                    uint32_t *out_pc_size,
                                    uint32_t ep_index)
{
    /* Find the push constant global for a specific entry point.
     * Each entry point has its own KernelParams struct in its interface. */
    SsirEntryPoint *ep = (ep_index < ssir->entry_point_count)
                          ? &ssir->entry_points[ep_index] : NULL;

    for (uint32_t i = 0; i < ssir->global_count; i++) {
        SsirGlobalVar *g = &ssir->globals[i];
        SsirType *t = ssir_get_type(ssir, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR)
            continue;
        if (t->ptr.space != SSIR_ADDR_PUSH_CONSTANT)
            continue;

        /* If we have entry point info, only accept globals in its interface */
        if (ep) {
            bool in_iface = false;
            for (uint32_t j = 0; j < ep->interface_count; j++) {
                if (ep->interface[j] == g->id) { in_iface = true; break; }
            }
            if (!in_iface) continue;
        }

        /* Found the push constant — its pointee is a struct */
        SsirType *st = ssir_get_type(ssir, t->ptr.pointee);
        if (!st || st->kind != SSIR_TYPE_STRUCT)
            continue;

        uint32_t count = st->struc.member_count;
        CuvkParamInfo *params = (CuvkParamInfo *)calloc(count, sizeof(CuvkParamInfo));
        if (!params)
            return CUDA_ERROR_OUT_OF_MEMORY;

        uint32_t total_size = 0;
        uint32_t user_param_count = count;
        for (uint32_t j = 0; j < count; j++) {
            /* Skip hidden __ntid_* members from user-visible param count */
            if (st->struc.member_names && st->struc.member_names[j] &&
                strncmp(st->struc.member_names[j], "__ntid_", 7) == 0) {
                if (j < user_param_count)
                    user_param_count = j; /* first hidden member index */
            }
            SsirType *mt = ssir_get_type(ssir, st->struc.members[j]);
            if (mt && (mt->kind == SSIR_TYPE_U64 || mt->kind == SSIR_TYPE_I64)) {
                params[j].is_pointer = true;
                params[j].size = 8;
            } else if (mt && (mt->kind == SSIR_TYPE_U32 || mt->kind == SSIR_TYPE_I32 ||
                              mt->kind == SSIR_TYPE_F32)) {
                params[j].is_pointer = false;
                params[j].size = 4;
            } else if (mt && (mt->kind == SSIR_TYPE_F64)) {
                params[j].is_pointer = false;
                params[j].size = 8;
            } else if (mt && (mt->kind == SSIR_TYPE_U16 || mt->kind == SSIR_TYPE_I16 ||
                              mt->kind == SSIR_TYPE_F16)) {
                params[j].is_pointer = false;
                params[j].size = 2;
            } else {
                params[j].is_pointer = false;
                params[j].size = 4;
            }
            /* Use struct offsets to compute total size */
            if (st->struc.offsets)
                total_size = st->struc.offsets[j] + params[j].size;
            else
                total_size += params[j].size;
        }

        *out_params = params;
        *out_count = user_param_count;
        if (out_pc_size)
            *out_pc_size = total_size;
        return CUDA_SUCCESS;
    }

    /* No push constant found — empty */
    *out_params = NULL;
    *out_count = 0;
    if (out_pc_size) *out_pc_size = 0;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuModuleLoadData
 * ============================================================================ */

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image)
{
    if (!module || !image)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    const char *ptx_text = NULL;
    char *fatbin_ptx = NULL;
    uint32_t magic = 0;
    memcpy(&magic, image, sizeof(magic));
    CUVK_LOG("[cuvk] cuModuleLoadData: magic=0x%08X image=%p\n",
            magic, image);

    /* Handle FatbincWrapper (magic 0x466243B1) - unwrap to inner fatbin */
    if (magic == 0x466243B1u) {
        const void *inner;
        memcpy(&inner, (const char *)image + 8, sizeof(inner));
        CUVK_LOG("[cuvk]   FatbincWrapper -> inner=%p\n", inner);
        if (!inner) return CUDA_ERROR_INVALID_IMAGE;
        image = inner;
        memcpy(&magic, image, sizeof(magic));
        CUVK_LOG("[cuvk]   inner magic=0x%08X\n", magic);
    }

    /* Check for fatbin container (magic 0xBA55ED50) */
    if (magic == 0xBA55ED50) {
        fatbin_ptx = cuvk_fatbin_extract_ptx(image, NULL);
        if (!fatbin_ptx)
            return CUDA_ERROR_INVALID_IMAGE;
        ptx_text = fatbin_ptx;
    } else {
        ptx_text = (const char *)image;
    }

    /* Determine if BDA mode should be used */
    bool use_bda = ctx->has_bda;

    /* 1. PTX -> SSIR */
    PtxToSsirOptions ptx_opts = {0};
    ptx_opts.preserve_names = 1;
    ptx_opts.use_bda = use_bda ? 1 : 0;
    SsirModule *ssir = NULL;
    char *error = NULL;

    /* Debug: dump raw PTX */
    const char *ptx_dump = getenv("CUVK_DUMP_PTX");
    if (ptx_dump) {
        FILE *pf = fopen(ptx_dump, "w");
        if (pf) {
            fputs(ptx_text, pf);
            fclose(pf);
            CUVK_LOG("[cuvk] dumped PTX to %s (%zu bytes)\n",
                    ptx_dump, strlen(ptx_text));
        }
    }

    PtxToSsirResult pr = ptx_to_ssir(ptx_text, &ptx_opts, &ssir, &error);
    free(fatbin_ptx);
    if (pr != PTX_TO_SSIR_OK) {
        CUVK_LOG("[cuvk] ptx_to_ssir FAILED: %s\n", error ? error : "unknown");
        ptx_to_ssir_free(error);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    if (error)
        CUVK_LOG("[cuvk] ptx_to_ssir warnings: %s\n", error);
    ptx_to_ssir_free(error);

    /* 2. SSIR -> SPIR-V */
    SsirToSpirvOptions spirv_opts = {0};
    spirv_opts.spirv_version = use_bda ? 0x00010500 : 0x00010300;
    spirv_opts.enable_debug_names = 1;

    uint32_t *words = NULL;
    size_t word_count = 0;

    SsirToSpirvResult sr = ssir_to_spirv(ssir, &spirv_opts, &words, &word_count);
    if (sr != SSIR_TO_SPIRV_OK) {
        CUVK_LOG("[cuvk] ssir_to_spirv FAILED: %d\n", sr);
        ssir_module_destroy(ssir);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    CUVK_LOG("[cuvk] SPIR-V: %zu words generated\n", word_count);

    /* Debug: dump SPIR-V to file if CUVK_DUMP_SPIRV is set */
    const char *dump_path = getenv("CUVK_DUMP_SPIRV");
    if (dump_path) {
        FILE *df = fopen(dump_path, "wb");
        if (df) {
            fwrite(words, sizeof(uint32_t), word_count, df);
            fclose(df);
            CUVK_LOG("[cuvk] dumped %zu SPIR-V words to %s\n",
                    word_count, dump_path);
        }
    }

    /* 3. Allocate CUmod_st */
    struct CUmod_st *mod = (struct CUmod_st *)calloc(1, sizeof(*mod));
    if (!mod) {
        ssir_to_spirv_free(words);
        ssir_module_destroy(ssir);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    mod->ctx = ctx;
    mod->ssir = ssir;

    /* Take ownership of SPIR-V words */
    mod->spirv_words = words;
    mod->spirv_count = (uint32_t)word_count;

    /* 4. For non-BDA mode, extract shared params; BDA extracts per entry point */
    CuvkParamInfo *shared_params = NULL;
    uint32_t shared_param_count = 0;
    CUresult res;
    if (!use_bda) {
        res = extract_params(ssir, &shared_params, &shared_param_count);
        if (res != CUDA_SUCCESS) {
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir);
            free(mod);
            return res;
        }
    }

    /* 5. For each entry point, allocate a CUfunc_st */
    uint32_t ep_count = ssir->entry_point_count;
    if (ep_count == 0) {
        free(shared_params);
        ssir_to_spirv_free(mod->spirv_words);
        ssir_module_destroy(ssir);
        free(mod);
        return CUDA_ERROR_INVALID_IMAGE;
    }

    struct CUfunc_st *funcs = (struct CUfunc_st *)calloc(ep_count, sizeof(*funcs));
    if (!funcs) {
        free(shared_params);
        ssir_to_spirv_free(mod->spirv_words);
        ssir_module_destroy(ssir);
        free(mod);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /* Create single VkShaderModule from the SPIR-V */
    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = (size_t)mod->spirv_count * sizeof(uint32_t);
    sm_ci.pCode = mod->spirv_words;

    VkShaderModule shared_shader = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(ctx->device, &sm_ci, NULL, &shared_shader);
    if (vr != VK_SUCCESS) {
        free(funcs);
        free(shared_params);
        ssir_to_spirv_free(mod->spirv_words);
        ssir_module_destroy(ssir);
        free(mod);
        return cuvk_vk_to_cu(vr);
    }

    for (uint32_t i = 0; i < ep_count; i++) {
        SsirEntryPoint *ep = &ssir->entry_points[i];
        struct CUfunc_st *f = &funcs[i];

        f->module = mod;
        f->use_bda = use_bda;

        /* Copy name */
        f->name = strdup(ep->name ? ep->name : "");

        /* Share the shader module (each function in the same module uses it) */
        f->shader_module = shared_shader;

        /* Extract parameter info per entry point */
        if (use_bda) {
            CuvkParamInfo *ep_params = NULL;
            uint32_t ep_param_count = 0;
            uint32_t ep_pc_size = 0;
            res = extract_params_bda(ssir, &ep_params, &ep_param_count,
                                     &ep_pc_size, i);
            f->param_count = ep_param_count;
            f->push_constant_size = ep_pc_size;
            f->params = ep_params;
        } else {
            f->param_count = shared_param_count;
            f->push_constant_size = 0;
            if (shared_param_count > 0) {
                f->params = (CuvkParamInfo *)calloc(shared_param_count,
                                                     sizeof(CuvkParamInfo));
                if (f->params)
                    memcpy(f->params, shared_params,
                           shared_param_count * sizeof(CuvkParamInfo));
            }
        }

        if (use_bda) {
            /* BDA mode: pipeline layout with push constants, no descriptor sets */
            VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
            dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dsl_ci.bindingCount = 0;
            dsl_ci.pBindings = NULL;
            vr = vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL,
                                             &f->desc_layout);
            if (vr != VK_SUCCESS) {
                for (uint32_t k = 0; k <= i; k++) {
                    free(funcs[k].name);
                    free(funcs[k].params);
                    if (k < i && funcs[k].desc_layout)
                        vkDestroyDescriptorSetLayout(ctx->device, funcs[k].desc_layout, NULL);
                    if (k < i && funcs[k].pipeline_layout)
                        vkDestroyPipelineLayout(ctx->device, funcs[k].pipeline_layout, NULL);
                }
                vkDestroyShaderModule(ctx->device, shared_shader, NULL);
                free(funcs); free(shared_params);
                ssir_to_spirv_free(mod->spirv_words);
                ssir_module_destroy(ssir); free(mod);
                return cuvk_vk_to_cu(vr);
            }

            VkPushConstantRange pc_range = {0};
            pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc_range.offset = 0;
            pc_range.size = f->push_constant_size > 0 ? f->push_constant_size : 4;

            VkPipelineLayoutCreateInfo pl_ci = {0};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 0;
            pl_ci.pSetLayouts = NULL;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &pc_range;

            vr = vkCreatePipelineLayout(ctx->device, &pl_ci, NULL,
                                        &f->pipeline_layout);
        } else {
            /* Descriptor mode: one storage buffer per pointer param */
            VkDescriptorSetLayoutBinding *bindings = NULL;
            if (f->param_count > 0) {
                bindings = (VkDescriptorSetLayoutBinding *)calloc(
                    f->param_count, sizeof(VkDescriptorSetLayoutBinding));
                for (uint32_t j = 0; j < f->param_count; j++) {
                    bindings[j].binding = j;
                    bindings[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    bindings[j].descriptorCount = 1;
                    bindings[j].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                }
            }

            VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
            dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dsl_ci.bindingCount = f->param_count;
            dsl_ci.pBindings = bindings;

            vr = vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL,
                                             &f->desc_layout);
            free(bindings);

            if (vr != VK_SUCCESS) {
                for (uint32_t k = 0; k <= i; k++) {
                    free(funcs[k].name);
                    free(funcs[k].params);
                    if (k < i && funcs[k].desc_layout)
                        vkDestroyDescriptorSetLayout(ctx->device, funcs[k].desc_layout, NULL);
                    if (k < i && funcs[k].pipeline_layout)
                        vkDestroyPipelineLayout(ctx->device, funcs[k].pipeline_layout, NULL);
                }
                vkDestroyShaderModule(ctx->device, shared_shader, NULL);
                free(funcs); free(shared_params);
                ssir_to_spirv_free(mod->spirv_words);
                ssir_module_destroy(ssir); free(mod);
                return cuvk_vk_to_cu(vr);
            }

            VkPipelineLayoutCreateInfo pl_ci = {0};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &f->desc_layout;

            vr = vkCreatePipelineLayout(ctx->device, &pl_ci, NULL,
                                        &f->pipeline_layout);
        }

        if (vr != VK_SUCCESS) {
            if (f->desc_layout)
                vkDestroyDescriptorSetLayout(ctx->device, f->desc_layout, NULL);
            for (uint32_t k = 0; k < i; k++) {
                free(funcs[k].name);
                free(funcs[k].params);
                if (funcs[k].desc_layout)
                    vkDestroyDescriptorSetLayout(ctx->device, funcs[k].desc_layout, NULL);
                if (funcs[k].pipeline_layout)
                    vkDestroyPipelineLayout(ctx->device, funcs[k].pipeline_layout, NULL);
            }
            vkDestroyShaderModule(ctx->device, shared_shader, NULL);
            free(funcs); free(shared_params);
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir); free(mod);
            return cuvk_vk_to_cu(vr);
        }

        /* Pipeline creation is deferred to cuLaunchKernel */
        f->pipeline_cache = NULL;
        f->pipeline_cache_count = 0;
        f->pipeline_cache_capacity = 0;
    }

    free(shared_params);

    mod->functions = funcs;
    mod->function_count = ep_count;

    *module = mod;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuModuleLoadDataEx - delegates to cuModuleLoadData, ignoring options
 * ============================================================================ */

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                     unsigned int numOptions,
                                     CUjit_option *options,
                                     void **optionValues)
{
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return cuModuleLoadData(module, image);
}

/* ============================================================================
 * cuModuleUnload
 * ============================================================================ */

CUresult CUDAAPI cuModuleUnload(CUmodule hmod)
{
    if (!hmod)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = hmod->ctx;

    if (!ctx || !ctx->device ||
        (g_cuvk.exiting && g_cuvk.has_validation)) {
        for (uint32_t i = 0; i < hmod->function_count; i++) {
            free(hmod->functions[i].pipeline_cache);
            free(hmod->functions[i].name);
            free(hmod->functions[i].params);
        }
        free(hmod->functions);
        if (hmod->ssir) ssir_module_destroy(hmod->ssir);
        if (hmod->spirv_words) ssir_to_spirv_free(hmod->spirv_words);
        free(hmod);
        return CUDA_SUCCESS;
    }

    vkDeviceWaitIdle(ctx->device);

    /* Track whether we've already destroyed the shared shader module */
    VkShaderModule shared_shader = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < hmod->function_count; i++) {
        struct CUfunc_st *f = &hmod->functions[i];

        /* Destroy cached pipelines */
        for (uint32_t j = 0; j < f->pipeline_cache_count; j++) {
            if (f->pipeline_cache[j].pipeline)
                vkDestroyPipeline(ctx->device,
                                  f->pipeline_cache[j].pipeline, NULL);
        }
        free(f->pipeline_cache);

        if (f->pipeline_layout)
            vkDestroyPipelineLayout(ctx->device, f->pipeline_layout, NULL);

        if (f->desc_layout)
            vkDestroyDescriptorSetLayout(ctx->device, f->desc_layout, NULL);

        if (f->shader_module && f->shader_module != shared_shader) {
            shared_shader = f->shader_module;
            vkDestroyShaderModule(ctx->device, f->shader_module, NULL);
        }

        free(f->name);
        free(f->params);
    }

    free(hmod->functions);

    /* Destroy SSIR module */
    if (hmod->ssir)
        ssir_module_destroy(hmod->ssir);

    /* Free SPIR-V words */
    if (hmod->spirv_words)
        ssir_to_spirv_free(hmod->spirv_words);

    free(hmod);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuModuleGetFunction
 * ============================================================================ */

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                      const char *name)
{
    if (!hfunc || !hmod || !name)
        return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hmod->function_count; i++) {
        if (hmod->functions[i].name &&
            strcmp(hmod->functions[i].name, name) == 0) {
            *hfunc = &hmod->functions[i];
            return CUDA_SUCCESS;
        }
    }

    return CUDA_ERROR_NOT_FOUND;
}

/* ============================================================================
 * cuFuncGetAttribute
 * ============================================================================ */

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                     CUfunction hfunc)
{
    if (!pi || !hfunc)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = hfunc->module ? hfunc->module->ctx : g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = (int)ctx->dev_props.limits.maxComputeWorkGroupInvocations;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
        *pi = 0; /* No static shared memory in our SPIR-V */
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_NUM_REGS:
        *pi = 32; /* Reasonable default */
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION:
        *pi = 70; /* We target PTX 7.0 */
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
        *pi = 70;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
        *pi = 0;
        return CUDA_SUCCESS;
    default:
        *pi = 0;
        return CUDA_SUCCESS;
    }
}
