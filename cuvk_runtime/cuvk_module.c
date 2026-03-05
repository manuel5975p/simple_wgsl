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
 * Helper: extract and allocate module-level globals (.global without init)
 * ============================================================================ */

static CUresult extract_module_globals(struct CUmod_st *mod)
{
    struct CUctx_st *ctx = mod->ctx;
    SsirModule *ssir = mod->ssir;
    if (!ssir) return CUDA_SUCCESS;

    uint32_t count = 0;
    for (uint32_t i = 0; i < ssir->global_count; i++) {
        SsirGlobalVar *g = &ssir->globals[i];
        if (!g->has_group || !g->has_binding) continue;
        if (g->builtin != SSIR_BUILTIN_NONE) continue;
        SsirType *t = ssir_get_type(ssir, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR) continue;
        if (t->ptr.space != SSIR_ADDR_STORAGE) continue;
        SsirType *pointee = ssir_get_type(ssir, t->ptr.pointee);
        if (!pointee || pointee->kind != SSIR_TYPE_STRUCT) continue;
        if (pointee->struc.member_count != 1) continue;
        count++;
    }
    if (count == 0) return CUDA_SUCCESS;

    CuvkModuleGlobal *globals = (CuvkModuleGlobal *)calloc(
        count, sizeof(CuvkModuleGlobal));
    if (!globals) return CUDA_ERROR_OUT_OF_MEMORY;

    uint32_t idx = 0;
    for (uint32_t i = 0; i < ssir->global_count && idx < count; i++) {
        SsirGlobalVar *g = &ssir->globals[i];
        if (!g->has_group || !g->has_binding) continue;
        if (g->builtin != SSIR_BUILTIN_NONE) continue;
        SsirType *t = ssir_get_type(ssir, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR) continue;
        if (t->ptr.space != SSIR_ADDR_STORAGE) continue;
        SsirType *pointee = ssir_get_type(ssir, t->ptr.pointee);
        if (!pointee || pointee->kind != SSIR_TYPE_STRUCT) continue;
        if (pointee->struc.member_count != 1) continue;

        SsirType *member_type = ssir_get_type(ssir, pointee->struc.members[0]);
        uint32_t byte_size = 4;
        if (member_type) {
            switch (member_type->kind) {
            case SSIR_TYPE_U64: case SSIR_TYPE_I64: case SSIR_TYPE_F64:
                byte_size = 8; break;
            case SSIR_TYPE_U16: case SSIR_TYPE_I16: case SSIR_TYPE_F16:
                byte_size = 2; break;
            default: byte_size = 4; break;
            }
        }

        CuvkModuleGlobal *mg = &globals[idx];
        snprintf(mg->name, sizeof(mg->name), "%s", g->name ? g->name : "");
        mg->size = byte_size;
        mg->binding = g->binding;

        VkDeviceSize buf_size = byte_size < 16 ? 16 : byte_size;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (ctx->has_bda)
            usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        VkBufferCreateInfo buf_ci = {0};
        buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_ci.size = buf_size;
        buf_ci.usage = usage;
        buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult vr = vkCreateBuffer(ctx->device, &buf_ci, NULL, &mg->buffer);
        if (vr != VK_SUCCESS) {
            for (uint32_t k = 0; k < idx; k++) {
                vkDestroyBuffer(ctx->device, globals[k].buffer, NULL);
                vkFreeMemory(ctx->device, globals[k].memory, NULL);
            }
            free(globals);
            return cuvk_vk_to_cu(vr);
        }

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(ctx->device, mg->buffer, &mem_reqs);
        int32_t mem_type = cuvk_find_memory_type(
            &ctx->mem_props, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type < 0) {
            vkDestroyBuffer(ctx->device, mg->buffer, NULL);
            for (uint32_t k = 0; k < idx; k++) {
                vkDestroyBuffer(ctx->device, globals[k].buffer, NULL);
                vkFreeMemory(ctx->device, globals[k].memory, NULL);
            }
            free(globals);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        VkMemoryAllocateInfo alloc_info = {0};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = (uint32_t)mem_type;

        VkMemoryAllocateFlagsInfo flags_info = {0};
        if (ctx->has_bda) {
            flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            alloc_info.pNext = &flags_info;
        }

        vr = vkAllocateMemory(ctx->device, &alloc_info, NULL, &mg->memory);
        if (vr != VK_SUCCESS) {
            vkDestroyBuffer(ctx->device, mg->buffer, NULL);
            for (uint32_t k = 0; k < idx; k++) {
                vkDestroyBuffer(ctx->device, globals[k].buffer, NULL);
                vkFreeMemory(ctx->device, globals[k].memory, NULL);
            }
            free(globals);
            return cuvk_vk_to_cu(vr);
        }

        vr = vkBindBufferMemory(ctx->device, mg->buffer, mg->memory, 0);
        if (vr != VK_SUCCESS) {
            vkFreeMemory(ctx->device, mg->memory, NULL);
            vkDestroyBuffer(ctx->device, mg->buffer, NULL);
            for (uint32_t k = 0; k < idx; k++) {
                vkDestroyBuffer(ctx->device, globals[k].buffer, NULL);
                vkFreeMemory(ctx->device, globals[k].memory, NULL);
            }
            free(globals);
            return cuvk_vk_to_cu(vr);
        }

        if (ctx->has_bda && ctx->pfn_get_bda) {
            VkBufferDeviceAddressInfo addr_info = {0};
            addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addr_info.buffer = mg->buffer;
            mg->device_addr = ctx->pfn_get_bda(ctx->device, &addr_info);
        } else {
            if (ctx->next_synthetic_addr == 0)
                ctx->next_synthetic_addr = 0x100000;
            mg->device_addr = ctx->next_synthetic_addr;
            ctx->next_synthetic_addr += (buf_size + 255) & ~(uint64_t)255;
        }

        if (ctx->alloc_count >= ctx->alloc_capacity) {
            uint32_t new_cap = ctx->alloc_capacity == 0 ? 16 :
                               ctx->alloc_capacity * 2;
            CuvkAlloc *new_allocs = (CuvkAlloc *)realloc(
                ctx->allocs, new_cap * sizeof(CuvkAlloc));
            if (!new_allocs) {
                vkFreeMemory(ctx->device, mg->memory, NULL);
                vkDestroyBuffer(ctx->device, mg->buffer, NULL);
                for (uint32_t k = 0; k < idx; k++) {
                    vkDestroyBuffer(ctx->device, globals[k].buffer, NULL);
                    vkFreeMemory(ctx->device, globals[k].memory, NULL);
                }
                free(globals);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            ctx->allocs = new_allocs;
            ctx->alloc_capacity = new_cap;
        }

        uint32_t lo = 0, hi = ctx->alloc_count;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if ((uint64_t)ctx->allocs[mid].device_addr < mg->device_addr)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < ctx->alloc_count)
            memmove(&ctx->allocs[lo + 1], &ctx->allocs[lo],
                    (ctx->alloc_count - lo) * sizeof(CuvkAlloc));
        ctx->allocs[lo].buffer = mg->buffer;
        ctx->allocs[lo].memory = mg->memory;
        ctx->allocs[lo].size = (VkDeviceSize)buf_size;
        ctx->allocs[lo].device_addr = mg->device_addr;
        ctx->allocs[lo].host_mapped = NULL;
        ctx->alloc_count++;

        CUVK_LOG("[cuvk] module global '%s': binding=%u size=%u addr=0x%llx\n",
                mg->name, mg->binding, mg->size,
                (unsigned long long)mg->device_addr);
        idx++;
    }

    mod->globals = globals;
    mod->global_count = idx;
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
        size_t extract_len = 0;
        fatbin_ptx = cuvk_fatbin_extract_ptx(image, &extract_len);
        /* Validate: real PTX is text (may have leading whitespace) */
        const char *p = fatbin_ptx;
        if (p) while (*p == '\n' || *p == '\r' || *p == ' ' || *p == '\t') p++;
        if (fatbin_ptx && *p != '.' && *p != '/' && *p != '\0') {
            CUVK_LOG("[cuvk] fatbin 'PTX' is binary (LTO-IR), not text\n");
#ifdef CUVK_NVJITLINK
            char *compiled = cuvk_jitlink_compile_raw(
                fatbin_ptx, extract_len, NULL);
            free(fatbin_ptx);
            fatbin_ptx = compiled;
            if (fatbin_ptx) {
                CUVK_LOG("[cuvk] LTO-IR compiled to PTX via nvJitLink\n");
            }
#else
            free(fatbin_ptx);
            fatbin_ptx = NULL;
#endif
        }
        if (!fatbin_ptx) {
#ifdef CUVK_NVJITLINK
            fatbin_ptx = cuvk_jitlink_compile_ltoir(image, NULL);
            if (fatbin_ptx) {
                CUVK_LOG("[cuvk] fatbin LTO-IR compiled to PTX via nvJitLink\n");
            } else
#endif
            {
                CUVK_LOG("[cuvk] fatbin has no PTX, creating empty module\n");
                struct CUmod_st *mod = (struct CUmod_st *)calloc(1, sizeof(*mod));
                if (!mod) return CUDA_ERROR_OUT_OF_MEMORY;
                mod->ctx = ctx;
                *module = mod;
                return CUDA_SUCCESS;
            }
        }
        ptx_text = fatbin_ptx;
    } else {
        ptx_text = (const char *)image;
    }

    /* If fatbin-sourced PTX has no kernel/function entries, create an empty
     * module. This happens with nvcc binaries that have no __global__ kernels
     * (e.g. cuBLAS-only programs). Only applies to fatbin PTX, not raw PTX
     * strings passed directly to cuModuleLoadData. */
    if (fatbin_ptx &&
        !strstr(ptx_text, ".entry") && !strstr(ptx_text, ".func")) {
        free(fatbin_ptx);
        fatbin_ptx = NULL;
        ptx_text = NULL;
#ifdef CUVK_NVJITLINK
        if (magic == 0xBA55ED50) {
            fatbin_ptx = cuvk_jitlink_compile_ltoir(image, NULL);
            if (fatbin_ptx) {
                CUVK_LOG("[cuvk] empty PTX, LTO-IR compiled via nvJitLink\n");
                ptx_text = fatbin_ptx;
            }
        }
#endif
        if (!ptx_text) {
            CUVK_LOG("[cuvk] PTX has no .entry/.func, creating empty module\n");
            struct CUmod_st *mod = (struct CUmod_st *)calloc(1, sizeof(*mod));
            if (!mod) return CUDA_ERROR_OUT_OF_MEMORY;
            mod->ctx = ctx;
            *module = mod;
            return CUDA_SUCCESS;
        }
    }

    /* Determine if BDA mode should be used */
    bool use_bda = ctx->has_bda;
    CUVK_LOG("[cuvk] BDA mode: %s\n", use_bda ? "enabled" : "disabled");

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

    /* 3b. Extract and allocate module-level globals */
    CUresult res = extract_module_globals(mod);
    if (res != CUDA_SUCCESS) {
        ssir_to_spirv_free(mod->spirv_words);
        ssir_module_destroy(ssir);
        free(mod);
        return res;
    }

    /* 3c. Create descriptor set layout/pool/set for module globals */
    VkResult vr;
    if (mod->global_count > 0) {
        VkDescriptorSetLayoutBinding *bindings =
            (VkDescriptorSetLayoutBinding *)calloc(
                mod->global_count, sizeof(VkDescriptorSetLayoutBinding));
        if (!bindings) {
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir);
            free(mod);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        for (uint32_t i = 0; i < mod->global_count; i++) {
            bindings[i].binding = mod->globals[i].binding;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = mod->global_count;
        dsl_ci.pBindings = bindings;
        vr = vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL,
                                          &mod->globals_desc_layout);
        free(bindings);
        if (vr != VK_SUCCESS) {
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir);
            free(mod);
            return cuvk_vk_to_cu(vr);
        }

        VkDescriptorPoolSize pool_size = {0};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = mod->global_count;

        VkDescriptorPoolCreateInfo dp_ci = {0};
        dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp_ci.maxSets = 1;
        dp_ci.poolSizeCount = 1;
        dp_ci.pPoolSizes = &pool_size;
        vr = vkCreateDescriptorPool(ctx->device, &dp_ci, NULL,
                                     &mod->globals_desc_pool);
        if (vr != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(ctx->device,
                                          mod->globals_desc_layout, NULL);
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir);
            free(mod);
            return cuvk_vk_to_cu(vr);
        }

        VkDescriptorSetAllocateInfo ds_ai = {0};
        ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_ai.descriptorPool = mod->globals_desc_pool;
        ds_ai.descriptorSetCount = 1;
        ds_ai.pSetLayouts = &mod->globals_desc_layout;
        vr = vkAllocateDescriptorSets(ctx->device, &ds_ai,
                                       &mod->globals_desc_set);
        if (vr != VK_SUCCESS) {
            vkDestroyDescriptorPool(ctx->device, mod->globals_desc_pool, NULL);
            vkDestroyDescriptorSetLayout(ctx->device,
                                          mod->globals_desc_layout, NULL);
            ssir_to_spirv_free(mod->spirv_words);
            ssir_module_destroy(ssir);
            free(mod);
            return cuvk_vk_to_cu(vr);
        }

        VkWriteDescriptorSet *writes = (VkWriteDescriptorSet *)calloc(
            mod->global_count, sizeof(VkWriteDescriptorSet));
        VkDescriptorBufferInfo *buf_infos = (VkDescriptorBufferInfo *)calloc(
            mod->global_count, sizeof(VkDescriptorBufferInfo));
        for (uint32_t i = 0; i < mod->global_count; i++) {
            buf_infos[i].buffer = mod->globals[i].buffer;
            buf_infos[i].offset = 0;
            buf_infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = mod->globals_desc_set;
            writes[i].dstBinding = mod->globals[i].binding;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(ctx->device, mod->global_count,
                                writes, 0, NULL);
        free(writes);
        free(buf_infos);
    }

    /* 4. For non-BDA mode, extract shared params; BDA extracts per entry point */
    CuvkParamInfo *shared_params = NULL;
    uint32_t shared_param_count = 0;
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
    CUVK_LOG("[cuvk] creating VkShaderModule (%zu bytes)\n", sm_ci.codeSize);
    vr = vkCreateShaderModule(ctx->device, &sm_ci, NULL, &shared_shader);
    CUVK_LOG("[cuvk] vkCreateShaderModule -> %d\n", vr);
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
            if (mod->global_count > 0) {
                pl_ci.setLayoutCount = 1;
                pl_ci.pSetLayouts = &mod->globals_desc_layout;
            } else {
                pl_ci.setLayoutCount = 0;
                pl_ci.pSetLayouts = NULL;
            }
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
    CUVK_LOG("[cuvk] cuModuleLoadData SUCCESS: mod=%p %u functions\n",
            (void *)mod, ep_count);
    for (uint32_t di = 0; di < ep_count; di++)
        CUVK_LOG("[cuvk]   func[%u]: '%s' params=%u\n",
                 di, funcs[di].name, funcs[di].param_count);
    fprintf(stderr, "[cuvk] cuModuleLoadData returning CUDA_SUCCESS (0)\n");
    fflush(stderr);
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
        free(hmod->globals);
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

    /* Destroy module globals */
    for (uint32_t i = 0; i < hmod->global_count; i++) {
        CuvkModuleGlobal *mg = &hmod->globals[i];
        /* Remove from ctx->allocs */
        for (uint32_t j = 0; j < ctx->alloc_count; j++) {
            if (ctx->allocs[j].buffer == mg->buffer) {
                memmove(&ctx->allocs[j], &ctx->allocs[j + 1],
                        (ctx->alloc_count - j - 1) * sizeof(CuvkAlloc));
                ctx->alloc_count--;
                break;
            }
        }
        vkDestroyBuffer(ctx->device, mg->buffer, NULL);
        vkFreeMemory(ctx->device, mg->memory, NULL);
    }
    if (hmod->globals_desc_pool)
        vkDestroyDescriptorPool(ctx->device, hmod->globals_desc_pool, NULL);
    if (hmod->globals_desc_layout)
        vkDestroyDescriptorSetLayout(ctx->device,
                                      hmod->globals_desc_layout, NULL);
    free(hmod->globals);

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
    fprintf(stderr, "[cuvk] cuModuleGetFunction: name='%.80s' mod=%p func_count=%u\n",
             name ? name : "(null)", (void *)hmod,
             hmod ? hmod->function_count : 0);
    if (!hfunc || !hmod || !name)
        return CUDA_ERROR_INVALID_VALUE;

    for (uint32_t i = 0; i < hmod->function_count; i++) {
        if (hmod->functions[i].name &&
            strcmp(hmod->functions[i].name, name) == 0) {
            *hfunc = &hmod->functions[i];
            CUVK_LOG("[cuvk]   -> FOUND func[%u]\n", i);
            return CUDA_SUCCESS;
        }
    }

    CUVK_LOG("[cuvk]   -> NOT_FOUND\n");
    return CUDA_ERROR_NOT_FOUND;
}

/* ============================================================================
 * cuFuncGetAttribute
 * ============================================================================ */

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                     CUfunction hfunc)
{
    fprintf(stderr, "[cuvk] cuFuncGetAttribute: attrib=%d func=%p\n",
            attrib, (void *)hfunc);
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
