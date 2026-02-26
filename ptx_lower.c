#include "simple_wgsl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>

#ifndef PTX_REALLOC
#define PTX_REALLOC(p, sz) realloc(p, sz)
#endif
#ifndef PTX_FREE
#define PTX_FREE(p) free(p)
#endif

static char *ptx_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *d = (char *)malloc(len);
    if (d) memcpy(d, s, len);
    return d;
}

/* ===== Internal Types (mirrors original PtxParser state) ===== */

typedef struct {
    char name[80];
    uint32_t ptr_id;
    uint32_t ptr_type;
    uint32_t val_type;
    uint32_t global_id;
    uint32_t local_ptr_id;
    uint32_t pending_binding;
    bool is_pred;
    bool is_bda_ptr;
} PlReg;

typedef struct {
    char name[80];
    uint32_t block_id;
    bool defined;
} PlLabel;

typedef struct {
    char name[80];
    uint32_t global_id;
    uint32_t type_id;
    bool is_surface;
    SsirTextureDim dim;
} PlTexRef;

typedef struct {
    char name[80];
    uint32_t global_id;
    uint32_t type_id;
} PlSamplerRef;

typedef struct {
    SsirModule *mod;
    PtxToSsirOptions opts;
    bool use_bda;

    uint32_t func_id;
    uint32_t block_id;
    bool block_from_label;
    bool is_entry;
    uint32_t ep_index;

    PlReg *regs;
    int reg_count, reg_cap;

    PlLabel *labels;
    int label_count, label_cap;

    uint32_t *iface;
    int iface_count, iface_cap;

    uint32_t *merge_blocks;
    int merge_block_count, merge_block_cap;

    struct { uint32_t merge_block; uint32_t target_label; bool has_inner_merge; } *construct_stack;
    int construct_depth, construct_cap;

    uint32_t wg_size[3];
    uint32_t next_binding;
    uint32_t module_binding_base;

    uint32_t bda_pc_global;
    uint32_t bda_param_count;

    struct { char name[80]; uint32_t func_id; uint32_t ret_type; } *funcs;
    int func_count, func_cap;

    PlTexRef *texrefs;
    int texref_count, texref_cap;

    PlSamplerRef *samplers;
    int sampler_count, sampler_cap;

    struct { char cond_target[80]; char merge_label[80]; } *precomputed_merges;
    int precomputed_merge_count, precomputed_merge_cap;

    char error[1024];
    int had_error;
} PtxLower;

static void pl_error(PtxLower *p, const char *fmt, ...) {
    if (p->had_error) return;
    p->had_error = 1;
    va_list a;
    va_start(a, fmt);
    vsnprintf(p->error, sizeof(p->error), fmt, a);
    va_end(a);
}

/* ===== Interface helpers ===== */

static void pl_add_iface(PtxLower *p, uint32_t global_id) {
    for (int i = 0; i < p->iface_count; i++)
        if (p->iface[i] == global_id) return;
    if (p->iface_count >= p->iface_cap) {
        p->iface_cap = p->iface_cap ? p->iface_cap * 2 : 16;
        p->iface = (uint32_t *)PTX_REALLOC(p->iface,
            p->iface_cap * sizeof(uint32_t));
    }
    p->iface[p->iface_count++] = global_id;
}

/* ===== Type Mapping ===== */

static uint32_t pl_map_type(PtxLower *p, PtxDataType t) {
    switch (t) {
    case PTX_TYPE_PRED: return ssir_type_bool(p->mod);
    case PTX_TYPE_B8:   return ssir_type_u8(p->mod);
    case PTX_TYPE_B16:  return ssir_type_u16(p->mod);
    case PTX_TYPE_B32:  return ssir_type_u32(p->mod);
    case PTX_TYPE_B64:  return ssir_type_u64(p->mod);
    case PTX_TYPE_U8:   return ssir_type_u8(p->mod);
    case PTX_TYPE_U16:  return ssir_type_u16(p->mod);
    case PTX_TYPE_U32:  return ssir_type_u32(p->mod);
    case PTX_TYPE_U64:  return ssir_type_u64(p->mod);
    case PTX_TYPE_S8:   return ssir_type_i8(p->mod);
    case PTX_TYPE_S16:  return ssir_type_i16(p->mod);
    case PTX_TYPE_S32:  return ssir_type_i32(p->mod);
    case PTX_TYPE_S64:  return ssir_type_i64(p->mod);
    case PTX_TYPE_F16:  return ssir_type_f16(p->mod);
    case PTX_TYPE_F32:  return ssir_type_f32(p->mod);
    case PTX_TYPE_F64:  return ssir_type_f64(p->mod);
    default:            return ssir_type_u32(p->mod);
    }
}

static bool pl_is_signed_type(uint32_t kind) {
    return kind == SSIR_TYPE_I8 || kind == SSIR_TYPE_I16 ||
           kind == SSIR_TYPE_I32 || kind == SSIR_TYPE_I64;
}

static bool pl_is_float_type(uint32_t kind) {
    return kind == SSIR_TYPE_F16 || kind == SSIR_TYPE_F32 ||
           kind == SSIR_TYPE_F64;
}

static int pl_scalar_bit_width(const SsirType *t) {
    switch (t->kind) {
    case SSIR_TYPE_BOOL:  return 1;
    case SSIR_TYPE_U8:  case SSIR_TYPE_I8:  return 8;
    case SSIR_TYPE_U16: case SSIR_TYPE_I16: case SSIR_TYPE_F16: return 16;
    case SSIR_TYPE_U32: case SSIR_TYPE_I32: case SSIR_TYPE_F32: return 32;
    case SSIR_TYPE_U64: case SSIR_TYPE_I64: case SSIR_TYPE_F64: return 64;
    default: return 0;
    }
}

static uint32_t pl_type_byte_size(PtxLower *p, uint32_t type_id) {
    SsirType *t = ssir_get_type(p->mod, type_id);
    if (!t) return 4;
    switch (t->kind) {
    case SSIR_TYPE_U8:  case SSIR_TYPE_I8:  return 1;
    case SSIR_TYPE_U16: case SSIR_TYPE_I16: case SSIR_TYPE_F16: return 2;
    case SSIR_TYPE_U32: case SSIR_TYPE_I32: case SSIR_TYPE_F32: return 4;
    case SSIR_TYPE_U64: case SSIR_TYPE_I64: case SSIR_TYPE_F64: return 8;
    default: return 4;
    }
}

/* ===== Register Management ===== */

static PlReg *pl_find_reg(PtxLower *p, const char *name) {
    for (int i = 0; i < p->reg_count; i++)
        if (strcmp(p->regs[i].name, name) == 0)
            return &p->regs[i];
    return NULL;
}

static PlReg *pl_add_reg(PtxLower *p, const char *name,
                          uint32_t val_type, bool is_pred) {
    PlReg *existing = pl_find_reg(p, name);
    if (existing) return existing;

    if (p->reg_count >= p->reg_cap) {
        p->reg_cap = p->reg_cap ? p->reg_cap * 2 : 64;
        p->regs = (PlReg *)PTX_REALLOC(p->regs, p->reg_cap * sizeof(PlReg));
    }
    PlReg *r = &p->regs[p->reg_count++];
    memset(r, 0, sizeof(*r));
    snprintf(r->name, sizeof(r->name), "%s", name);
    r->val_type = val_type;
    r->is_pred = is_pred;
    r->pending_binding = UINT32_MAX;

    uint32_t ptr_type = ssir_type_ptr(p->mod, val_type, SSIR_ADDR_FUNCTION);
    r->ptr_type = ptr_type;
    r->ptr_id = ssir_function_add_local(p->mod, p->func_id,
        p->opts.preserve_names ? name : NULL, ptr_type);
    return r;
}

static uint32_t pl_load_reg(PtxLower *p, const char *name) {
    PlReg *r = pl_find_reg(p, name);
    if (!r) { pl_error(p, "undefined register '%s'", name); return 0; }
    return ssir_build_load(p->mod, p->func_id, p->block_id,
                           r->val_type, r->ptr_id);
}

static void pl_store_reg_typed(PtxLower *p, const char *name,
                                uint32_t value, uint32_t value_type) {
    PlReg *r = pl_find_reg(p, name);
    if (!r) { pl_error(p, "undefined register '%s'", name); return; }
    if (value_type && value_type != r->val_type) {
        SsirType *vt = ssir_get_type(p->mod, value_type);
        SsirType *rt = ssir_get_type(p->mod, r->val_type);
        if (vt && rt && pl_scalar_bit_width(vt) != pl_scalar_bit_width(rt))
            value = ssir_build_convert(p->mod, p->func_id, p->block_id,
                                       r->val_type, value);
        else
            value = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                       r->val_type, value);
    }
    ssir_build_store(p->mod, p->func_id, p->block_id, r->ptr_id, value);
}

static void pl_store_reg(PtxLower *p, const char *name, uint32_t value) {
    PlReg *r = pl_find_reg(p, name);
    if (!r) { pl_error(p, "undefined register '%s'", name); return; }
    ssir_build_store(p->mod, p->func_id, p->block_id, r->ptr_id, value);
}

static void pl_propagate_buffer(PtxLower *p, const char *dst, const char *src) {
    PlReg *dr = pl_find_reg(p, dst);
    PlReg *sr = pl_find_reg(p, src);
    if (dr && sr) {
        dr->global_id = sr->global_id;
        dr->local_ptr_id = sr->local_ptr_id;
        dr->pending_binding = sr->pending_binding;
        dr->is_bda_ptr = sr->is_bda_ptr;
    }
}

static uint32_t pl_materialize_buffer(PtxLower *p, PlReg *r,
                                       uint32_t elem_type) {
    if (r->global_id != 0) return r->global_id;
    if (r->pending_binding == UINT32_MAX) return 0;

    uint32_t rt_array = ssir_type_runtime_array(p->mod, elem_type);
    uint32_t members[] = { rt_array };
    uint32_t offsets[] = { 0 };
    char struct_name[88];
    snprintf(struct_name, sizeof(struct_name), "buf_%u", r->pending_binding);
    uint32_t struct_type = ssir_type_struct(p->mod, struct_name,
        members, 1, offsets);
    uint32_t ptr_type = ssir_type_ptr(p->mod, struct_type, SSIR_ADDR_STORAGE);
    uint32_t gid = ssir_global_var(p->mod, struct_name, ptr_type);
    ssir_global_set_group(p->mod, gid, 0);
    ssir_global_set_binding(p->mod, gid, r->pending_binding);
    pl_add_iface(p, gid);
    r->global_id = gid;
    for (int i = 0; i < p->reg_count; i++)
        if (p->regs[i].pending_binding == r->pending_binding)
            p->regs[i].global_id = gid;
    return gid;
}

/* ===== Label Management ===== */

static uint32_t pl_get_or_create_label(PtxLower *p, const char *name) {
    for (int i = 0; i < p->label_count; i++)
        if (strcmp(p->labels[i].name, name) == 0)
            return p->labels[i].block_id;
    if (p->label_count >= p->label_cap) {
        p->label_cap = p->label_cap ? p->label_cap * 2 : 32;
        p->labels = (PlLabel *)PTX_REALLOC(p->labels,
            p->label_cap * sizeof(PlLabel));
    }
    PlLabel *l = &p->labels[p->label_count++];
    snprintf(l->name, sizeof(l->name), "%s", name);
    l->block_id = ssir_module_alloc_id(p->mod);
    l->defined = false;
    return l->block_id;
}

static bool pl_is_label_defined(PtxLower *p, const char *name) {
    for (int i = 0; i < p->label_count; i++)
        if (strcmp(p->labels[i].name, name) == 0)
            return p->labels[i].defined;
    return false;
}

static bool pl_is_merge_used(PtxLower *p, uint32_t block_id) {
    for (int i = 0; i < p->merge_block_count; i++)
        if (p->merge_blocks[i] == block_id) return true;
    return false;
}

static void pl_mark_merge(PtxLower *p, uint32_t block_id) {
    if (pl_is_merge_used(p, block_id)) return;
    if (p->merge_block_count >= p->merge_block_cap) {
        p->merge_block_cap = p->merge_block_cap ? p->merge_block_cap * 2 : 16;
        p->merge_blocks = (uint32_t *)PTX_REALLOC(p->merge_blocks,
            p->merge_block_cap * sizeof(uint32_t));
    }
    p->merge_blocks[p->merge_block_count++] = block_id;
}

static void pl_push_construct(PtxLower *p, uint32_t merge_block,
                               uint32_t target_label, bool has_inner_merge) {
    if (p->construct_depth >= p->construct_cap) {
        p->construct_cap = p->construct_cap ? p->construct_cap * 2 : 16;
        p->construct_stack = PTX_REALLOC(p->construct_stack,
            p->construct_cap * sizeof(p->construct_stack[0]));
    }
    p->construct_stack[p->construct_depth].merge_block = merge_block;
    p->construct_stack[p->construct_depth].target_label = target_label;
    p->construct_stack[p->construct_depth].has_inner_merge = has_inner_merge;
    p->construct_depth++;
}

static bool pl_has_deferred_merge_for(PtxLower *p, uint32_t block_id) {
    for (int i = 0; i < p->construct_depth; i++)
        if (p->construct_stack[i].target_label == block_id &&
            p->construct_stack[i].has_inner_merge)
            return true;
    return false;
}

/* ===== Constant Helpers ===== */

static uint32_t pl_const_for_type(PtxLower *p, uint32_t type_id, uint64_t ival,
                                   double fval, bool is_float) {
    SsirType *ty = ssir_get_type(p->mod, type_id);
    if (!ty) return 0;
    switch (ty->kind) {
    case SSIR_TYPE_BOOL:  return ssir_const_bool(p->mod, ival != 0);
    case SSIR_TYPE_I8:    return ssir_const_i8(p->mod, (int8_t)ival);
    case SSIR_TYPE_U8:    return ssir_const_u8(p->mod, (uint8_t)ival);
    case SSIR_TYPE_I16:   return ssir_const_i16(p->mod, (int16_t)ival);
    case SSIR_TYPE_U16:   return ssir_const_u16(p->mod, (uint16_t)ival);
    case SSIR_TYPE_I32:   return ssir_const_i32(p->mod, (int32_t)ival);
    case SSIR_TYPE_U32:   return ssir_const_u32(p->mod, (uint32_t)ival);
    case SSIR_TYPE_I64:   return ssir_const_i64(p->mod, (int64_t)ival);
    case SSIR_TYPE_U64:   return ssir_const_u64(p->mod, ival);
    case SSIR_TYPE_F16:   (void)is_float; (void)fval; return ssir_const_f16(p->mod, 0);
    case SSIR_TYPE_F32:   return ssir_const_f32(p->mod, is_float ? (float)fval : (float)ival);
    case SSIR_TYPE_F64:   return ssir_const_f64(p->mod, is_float ? fval : (double)ival);
    default: return 0;
    }
}

/* ===== Texture/Sampler Helpers ===== */

static PlTexRef *pl_find_texref(PtxLower *p, const char *name) {
    for (int i = 0; i < p->texref_count; i++)
        if (strcmp(p->texrefs[i].name, name) == 0)
            return &p->texrefs[i];
    return NULL;
}

static PlSamplerRef *pl_find_sampler(PtxLower *p, const char *name) {
    for (int i = 0; i < p->sampler_count; i++)
        if (strcmp(p->samplers[i].name, name) == 0)
            return &p->samplers[i];
    return NULL;
}

static PlTexRef *pl_add_texref(PtxLower *p, const char *name,
                                uint32_t global_id, uint32_t type_id,
                                bool is_surface, SsirTextureDim dim) {
    if (p->texref_count >= p->texref_cap) {
        p->texref_cap = p->texref_cap ? p->texref_cap * 2 : 16;
        p->texrefs = (PlTexRef *)PTX_REALLOC(p->texrefs,
            p->texref_cap * sizeof(PlTexRef));
    }
    PlTexRef *tr = &p->texrefs[p->texref_count++];
    memset(tr, 0, sizeof(*tr));
    snprintf(tr->name, sizeof(tr->name), "%s", name);
    tr->global_id = global_id;
    tr->type_id = type_id;
    tr->is_surface = is_surface;
    tr->dim = dim;
    return tr;
}

static PlSamplerRef *pl_add_sampler(PtxLower *p, const char *name,
                                     uint32_t global_id, uint32_t type_id) {
    if (p->sampler_count >= p->sampler_cap) {
        p->sampler_cap = p->sampler_cap ? p->sampler_cap * 2 : 16;
        p->samplers = (PlSamplerRef *)PTX_REALLOC(p->samplers,
            p->sampler_cap * sizeof(PlSamplerRef));
    }
    PlSamplerRef *sr = &p->samplers[p->sampler_count++];
    memset(sr, 0, sizeof(*sr));
    snprintf(sr->name, sizeof(sr->name), "%s", name);
    sr->global_id = global_id;
    sr->type_id = type_id;
    return sr;
}

static PlSamplerRef *pl_get_implicit_sampler(PtxLower *p, const char *tex_name) {
    char sname[80];
    snprintf(sname, sizeof(sname), "_sampler_%s", tex_name);
    PlSamplerRef *existing = pl_find_sampler(p, sname);
    if (existing) return existing;
    uint32_t sampler_t = ssir_type_sampler(p->mod);
    uint32_t ptr_t = ssir_type_ptr(p->mod, sampler_t, SSIR_ADDR_UNIFORM_CONSTANT);
    uint32_t gid = ssir_global_var(p->mod, sname, ptr_t);
    ssir_global_set_group(p->mod, gid, 0);
    ssir_global_set_binding(p->mod, gid, p->next_binding++);
    pl_add_iface(p, gid);
    return pl_add_sampler(p, sname, gid, sampler_t);
}

static uint32_t pl_surface_format_for_type(PtxLower *p, uint32_t elem_type) {
    SsirType *ty = ssir_get_type(p->mod, elem_type);
    if (!ty) return 1;
    if (pl_is_signed_type(ty->kind)) return 21;
    if (pl_is_float_type(ty->kind))  return 1;
    return 30;
}

/* ===== Special Register Handling ===== */

static uint32_t pl_ensure_builtin_global(PtxLower *p, SsirBuiltinVar builtin,
                                          uint32_t val_type, const char *name) {
    for (int i = 0; i < p->iface_count; i++) {
        SsirGlobalVar *g = ssir_get_global(p->mod, p->iface[i]);
        if (g && g->builtin == builtin) return p->iface[i];
    }
    uint32_t ptr_type = ssir_type_ptr(p->mod, val_type, SSIR_ADDR_INPUT);
    uint32_t gid = ssir_global_var(p->mod, name, ptr_type);
    ssir_global_set_builtin(p->mod, gid, builtin);
    pl_add_iface(p, gid);
    return gid;
}

static int pl_component_index(char c) {
    switch (c) {
    case 'x': return 0; case 'y': return 1; case 'z': return 2;
    default: return 0;
    }
}

static uint32_t pl_load_special_reg(PtxLower *p, const char *name,
                                     uint32_t expected_type) {
    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t vec3u_t = ssir_type_vec(p->mod, u32_t, 3);
    SsirBuiltinVar builtin = SSIR_BUILTIN_LOCAL_INVOCATION_ID;
    int comp = 0;
    const char *gname = "gl_LocalInvocationID";

    if (strncmp(name, "%tid.", 5) == 0) {
        builtin = SSIR_BUILTIN_LOCAL_INVOCATION_ID;
        comp = pl_component_index(name[5]);
        gname = "gl_LocalInvocationID";
    } else if (strncmp(name, "%ctaid.", 7) == 0) {
        builtin = SSIR_BUILTIN_WORKGROUP_ID;
        comp = pl_component_index(name[7]);
        gname = "gl_WorkGroupID";
    } else if (strncmp(name, "%nctaid.", 8) == 0) {
        builtin = SSIR_BUILTIN_NUM_WORKGROUPS;
        comp = pl_component_index(name[8]);
        gname = "gl_NumWorkGroups";
    } else if (strncmp(name, "%ntid.", 6) == 0) {
        comp = pl_component_index(name[6]);
        if (p->wg_size[comp] > 0) {
            uint32_t val = ssir_const_u32(p->mod, p->wg_size[comp]);
            if (expected_type != u32_t && expected_type != 0)
                return ssir_build_convert(p->mod, p->func_id, p->block_id,
                                          expected_type, val);
            return val;
        }
        if (p->use_bda && p->bda_pc_global) {
            uint32_t member_idx = p->bda_param_count + (uint32_t)comp;
            uint32_t pc_u32_ptr = ssir_type_ptr(p->mod, u32_t, SSIR_ADDR_PUSH_CONSTANT);
            uint32_t idx_const = ssir_const_u32(p->mod, member_idx);
            uint32_t indices[] = { idx_const };
            uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                pc_u32_ptr, p->bda_pc_global, indices, 1);
            uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                           u32_t, ptr);
            if (expected_type != u32_t && expected_type != 0)
                return ssir_build_convert(p->mod, p->func_id, p->block_id,
                                          expected_type, val);
            return val;
        }
        builtin = SSIR_BUILTIN_NUM_WORKGROUPS;
        gname = "gl_NumWorkGroups";
    } else if (strcmp(name, "%laneid") == 0) {
        builtin = SSIR_BUILTIN_SUBGROUP_INVOCATION_ID;
        uint32_t gid = pl_ensure_builtin_global(p, builtin, u32_t, "gl_SubgroupInvocationID");
        uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, u32_t, gid);
        if (expected_type != u32_t && expected_type != 0)
            return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
        return val;
    } else if (strcmp(name, "%warpid") == 0) {
        uint32_t gid = pl_ensure_builtin_global(p, SSIR_BUILTIN_LOCAL_INVOCATION_ID,
                                                 vec3u_t, "gl_LocalInvocationID");
        uint32_t vec = ssir_build_load(p->mod, p->func_id, p->block_id, vec3u_t, gid);
        uint32_t tidx = ssir_build_extract(p->mod, p->func_id, p->block_id, u32_t, vec, 0);
        uint32_t c32 = ssir_const_u32(p->mod, 32);
        uint32_t val = ssir_build_div(p->mod, p->func_id, p->block_id, u32_t, tidx, c32);
        if (expected_type != u32_t && expected_type != 0)
            return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
        return val;
    }

    uint32_t gid = pl_ensure_builtin_global(p, builtin, vec3u_t, gname);
    uint32_t vec = ssir_build_load(p->mod, p->func_id, p->block_id, vec3u_t, gid);
    uint32_t val = ssir_build_extract(p->mod, p->func_id, p->block_id, u32_t, vec, comp);
    if (expected_type != u32_t && expected_type != 0)
        return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
    return val;
}

/* ===== Operand Resolution ===== */

static bool pl_is_special_reg(const char *name) {
    return strncmp(name, "%tid.", 5) == 0 ||
           strncmp(name, "%ntid.", 6) == 0 ||
           strncmp(name, "%ctaid.", 7) == 0 ||
           strncmp(name, "%nctaid.", 8) == 0 ||
           strcmp(name, "%laneid") == 0 ||
           strcmp(name, "%warpid") == 0;
}

static uint32_t pl_resolve_global_addr(PtxLower *p, const char *name,
                                        uint32_t expected_type) {
    for (uint32_t i = 0; i < (uint32_t)p->mod->global_count; i++) {
        if (p->mod->globals[i].name &&
            strcmp(p->mod->globals[i].name, name) == 0) {
            uint32_t u64_t = ssir_type_u64(p->mod);
            uint32_t zero = ssir_const_u64(p->mod, 0);
            if (expected_type && expected_type != u64_t)
                zero = ssir_build_bitcast(p->mod, p->func_id,
                    p->block_id, expected_type, zero);
            return zero;
        }
    }
    return 0;
}

static uint32_t pl_resolve_operand(PtxLower *p, const PtxOperand *op,
                                    uint32_t expected_type) {
    if (p->had_error) return 0;
    switch (op->kind) {
    case PTX_OPER_IMM_INT:
        return pl_const_for_type(p, expected_type, (uint64_t)op->ival, 0.0, false);
    case PTX_OPER_IMM_FLT:
        return pl_const_for_type(p, expected_type, 0, op->fval, true);
    case PTX_OPER_REG: {
        if (pl_is_special_reg(op->name))
            return pl_load_special_reg(p, op->name, expected_type);
        PlReg *r = pl_find_reg(p, op->name);
        if (r) {
            uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                            r->val_type, r->ptr_id);
            if (expected_type && r->val_type != expected_type) {
                SsirType *rt = ssir_get_type(p->mod, r->val_type);
                SsirType *et = ssir_get_type(p->mod, expected_type);
                if (rt && et) {
                    if (pl_scalar_bit_width(rt) == pl_scalar_bit_width(et))
                        val = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                                 expected_type, val);
                    else
                        val = ssir_build_convert(p->mod, p->func_id, p->block_id,
                                                 expected_type, val);
                }
            }
            if (op->negated)
                val = ssir_build_neg(p->mod, p->func_id, p->block_id,
                                     expected_type, val);
            return val;
        }
        uint32_t gaddr = pl_resolve_global_addr(p, op->name, expected_type);
        if (gaddr) return gaddr;
        pl_error(p, "undefined register '%s'", op->name);
        return 0;
    }
    case PTX_OPER_ADDR: {
        uint32_t u64_t = ssir_type_u64(p->mod);
        PlReg *r = pl_find_reg(p, op->base);
        uint32_t addr;
        if (r) {
            addr = ssir_build_load(p->mod, p->func_id, p->block_id,
                                    r->val_type, r->ptr_id);
            if (r->val_type != u64_t)
                addr = ssir_build_convert(p->mod, p->func_id, p->block_id,
                                           u64_t, addr);
        } else {
            addr = pl_resolve_global_addr(p, op->base, u64_t);
            if (!addr) {
                pl_error(p, "undefined register '%s'", op->base);
                return 0;
            }
        }
        if (op->offset != 0) {
            uint32_t off = ssir_const_u64(p->mod, (uint64_t)op->offset);
            addr = ssir_build_add(p->mod, p->func_id, p->block_id, u64_t, addr, off);
        }
        return addr;
    }
    case PTX_OPER_LABEL:
        return pl_get_or_create_label(p, op->name);
    default:
        return 0;
    }
}

/* ===== Register Declaration from AST ===== */

static void pl_declare_regs(PtxLower *p, const PtxRegDecl *decls, int count) {
    for (int d = 0; d < count; d++) {
        const PtxRegDecl *rd = &decls[d];
        uint32_t scalar_type = pl_map_type(p, rd->type);
        bool is_pred = (rd->type == PTX_TYPE_PRED);
        uint32_t val_type = scalar_type;
        if (rd->vec_width > 1)
            val_type = ssir_type_vec(p->mod, scalar_type, rd->vec_width);

        if (rd->is_parameterized) {
            for (int i = 0; i < rd->count; i++) {
                char pname[80];
                snprintf(pname, sizeof(pname), "%s%d", rd->name, i);
                pl_add_reg(p, pname, val_type, is_pred);
            }
        } else {
            pl_add_reg(p, rd->name, val_type, is_pred);
        }
    }
}

/* ===== Function registration ===== */

static void pl_register_func(PtxLower *p, const char *name,
                              uint32_t func_id, uint32_t ret_type) {
    if (p->func_count >= p->func_cap) {
        p->func_cap = p->func_cap ? p->func_cap * 2 : 16;
        p->funcs = PTX_REALLOC(p->funcs, p->func_cap * sizeof(p->funcs[0]));
    }
    snprintf(p->funcs[p->func_count].name, 80, "%s", name);
    p->funcs[p->func_count].func_id = func_id;
    p->funcs[p->func_count].ret_type = ret_type;
    p->func_count++;
}

/* ===== Comparison Op Mapping ===== */

static SsirOpcode pl_cmp_op(PtxCmpOp cmp) {
    switch (cmp) {
    case PTX_CMP_EQ: case PTX_CMP_EQU: return SSIR_OP_EQ;
    case PTX_CMP_NE: case PTX_CMP_NEU: return SSIR_OP_NE;
    case PTX_CMP_LT: case PTX_CMP_LTU: case PTX_CMP_LO: return SSIR_OP_LT;
    case PTX_CMP_LE: case PTX_CMP_LEU: case PTX_CMP_LS: return SSIR_OP_LE;
    case PTX_CMP_GT: case PTX_CMP_GTU: case PTX_CMP_HI: return SSIR_OP_GT;
    case PTX_CMP_GE: case PTX_CMP_GEU: case PTX_CMP_HS: return SSIR_OP_GE;
    default: return SSIR_OP_EQ;
    }
}

static SsirAtomicOp pl_atomic_op(PtxAtomicOp op) {
    switch (op) {
    case PTX_ATOMIC_ADD: return SSIR_ATOMIC_ADD;
    case PTX_ATOMIC_MIN: return SSIR_ATOMIC_MIN;
    case PTX_ATOMIC_MAX: return SSIR_ATOMIC_MAX;
    case PTX_ATOMIC_AND: return SSIR_ATOMIC_AND;
    case PTX_ATOMIC_OR:  return SSIR_ATOMIC_OR;
    case PTX_ATOMIC_XOR: return SSIR_ATOMIC_XOR;
    case PTX_ATOMIC_EXCH: return SSIR_ATOMIC_EXCHANGE;
    case PTX_ATOMIC_CAS: return SSIR_ATOMIC_COMPARE_EXCHANGE;
    case PTX_ATOMIC_INC: return SSIR_ATOMIC_ADD;
    case PTX_ATOMIC_DEC: return SSIR_ATOMIC_SUB;
    }
    return SSIR_ATOMIC_ADD;
}

static SsirAddressSpace pl_mem_space(PtxMemSpace space) {
    switch (space) {
    case PTX_SPACE_GLOBAL: return SSIR_ADDR_STORAGE;
    case PTX_SPACE_SHARED: return SSIR_ADDR_WORKGROUP;
    case PTX_SPACE_LOCAL:  return SSIR_ADDR_FUNCTION;
    case PTX_SPACE_CONST:  return SSIR_ADDR_UNIFORM;
    case PTX_SPACE_PARAM:  return SSIR_ADDR_UNIFORM;
    default:               return SSIR_ADDR_STORAGE;
    }
}

static SsirTextureDim pl_tex_dim(PtxTexGeom geom, int *coord_count) {
    switch (geom) {
    case PTX_TEX_1D:   if (coord_count) *coord_count = 1; return SSIR_TEX_1D;
    case PTX_TEX_2D:   if (coord_count) *coord_count = 2; return SSIR_TEX_2D;
    case PTX_TEX_3D:   if (coord_count) *coord_count = 3; return SSIR_TEX_3D;
    case PTX_TEX_A1D:  if (coord_count) *coord_count = 2; return SSIR_TEX_1D_ARRAY;
    case PTX_TEX_A2D:  if (coord_count) *coord_count = 4; return SSIR_TEX_2D_ARRAY;
    case PTX_TEX_CUBE: if (coord_count) *coord_count = 4; return SSIR_TEX_CUBE;
    }
    if (coord_count) *coord_count = 2;
    return SSIR_TEX_2D;
}

/* ===== Instruction Lowering ===== */

static void pl_lower_arith(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
    if (p->had_error) return;

    bool is_wide = (inst->modifiers & PTX_MOD_WIDE) != 0;
    uint32_t result_type = type;

    if (is_wide && inst->opcode == PTX_OP_MUL) {
        SsirType *ty = ssir_get_type(p->mod, type);
        if (ty && (ty->kind == SSIR_TYPE_I32 || ty->kind == SSIR_TYPE_U32)) {
            result_type = (ty->kind == SSIR_TYPE_I32)
                ? ssir_type_i64(p->mod) : ssir_type_u64(p->mod);
            a = ssir_build_convert(p->mod, p->func_id, p->block_id, result_type, a);
            b = ssir_build_convert(p->mod, p->func_id, p->block_id, result_type, b);
        } else if (ty && (ty->kind == SSIR_TYPE_I16 || ty->kind == SSIR_TYPE_U16)) {
            result_type = (ty->kind == SSIR_TYPE_I16)
                ? ssir_type_i32(p->mod) : ssir_type_u32(p->mod);
            a = ssir_build_convert(p->mod, p->func_id, p->block_id, result_type, a);
            b = ssir_build_convert(p->mod, p->func_id, p->block_id, result_type, b);
        }
    }

    uint32_t result = 0;
    switch (inst->opcode) {
    case PTX_OP_ADD: result = ssir_build_add(p->mod, p->func_id, p->block_id, result_type, a, b); break;
    case PTX_OP_SUB: result = ssir_build_sub(p->mod, p->func_id, p->block_id, result_type, a, b); break;
    case PTX_OP_MUL: case PTX_OP_MUL24:
        result = ssir_build_mul(p->mod, p->func_id, p->block_id, result_type, a, b); break;
    case PTX_OP_DIV: result = ssir_build_div(p->mod, p->func_id, p->block_id, result_type, a, b); break;
    case PTX_OP_REM: result = ssir_build_rem(p->mod, p->func_id, p->block_id, result_type, a, b); break;
    default: break;
    }

    if (result) {
        pl_store_reg_typed(p, inst->dst.name, result, result_type);
        if (inst->src[0].kind == PTX_OPER_REG &&
            (inst->opcode == PTX_OP_ADD || inst->opcode == PTX_OP_SUB)) {
            SsirType *t = ssir_get_type(p->mod, result_type);
            if (t && (t->kind == SSIR_TYPE_U64 || t->kind == SSIR_TYPE_I64))
                pl_propagate_buffer(p, inst->dst.name, inst->src[0].name);
        }
    }
}

static void pl_lower_unary(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;

    uint32_t result = 0;
    switch (inst->opcode) {
    case PTX_OP_NEG:
        result = ssir_build_neg(p->mod, p->func_id, p->block_id, type, a);
        break;
    case PTX_OP_ABS: {
        uint32_t args[] = { a };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_ABS, args, 1);
        break;
    }
    case PTX_OP_NOT: {
        SsirType *ty = ssir_get_type(p->mod, type);
        if (ty && ty->kind == SSIR_TYPE_BOOL)
            result = ssir_build_not(p->mod, p->func_id, p->block_id, type, a);
        else
            result = ssir_build_bit_not(p->mod, p->func_id, p->block_id, type, a);
        break;
    }
    case PTX_OP_CNOT: {
        uint32_t zero = pl_const_for_type(p, type, 0, 0.0, false);
        uint32_t bool_t = ssir_type_bool(p->mod);
        uint32_t cmp = ssir_build_eq(p->mod, p->func_id, p->block_id, bool_t, a, zero);
        uint32_t one = pl_const_for_type(p, type, 1, 0.0, false);
        uint32_t args[] = { cmp, one, zero };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_SELECT, args, 3);
        break;
    }
    default: break;
    }
    if (result) pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_minmax(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
    if (p->had_error) return;
    SsirBuiltinId bid = (inst->opcode == PTX_OP_MIN) ? SSIR_BUILTIN_MIN : SSIR_BUILTIN_MAX;
    uint32_t args[] = { a, b };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, bid, args, 2);
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_mad(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    bool is_wide = (inst->modifiers & PTX_MOD_WIDE) != 0;
    SsirType *ty = ssir_get_type(p->mod, type);

    if (ty && pl_is_float_type(ty->kind)) {
        uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
        uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
        uint32_t c = pl_resolve_operand(p, &inst->src[2], type);
        if (p->had_error) return;
        uint32_t args[] = { a, b, c };
        uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                             type, SSIR_BUILTIN_FMA, args, 3);
        pl_store_reg_typed(p, inst->dst.name, result, type);
    } else if (is_wide && ty &&
               (ty->kind == SSIR_TYPE_U32 || ty->kind == SSIR_TYPE_I32 ||
                ty->kind == SSIR_TYPE_U16 || ty->kind == SSIR_TYPE_I16)) {
        uint32_t wide_type;
        if (ty->kind == SSIR_TYPE_U32 || ty->kind == SSIR_TYPE_U16)
            wide_type = (ty->kind == SSIR_TYPE_U16) ? ssir_type_u32(p->mod)
                                                     : ssir_type_u64(p->mod);
        else
            wide_type = (ty->kind == SSIR_TYPE_I16) ? ssir_type_i32(p->mod)
                                                     : ssir_type_i64(p->mod);
        uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
        uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
        uint32_t c = pl_resolve_operand(p, &inst->src[2], wide_type);
        if (p->had_error) return;
        uint32_t wa = ssir_build_convert(p->mod, p->func_id, p->block_id, wide_type, a);
        uint32_t wb = ssir_build_convert(p->mod, p->func_id, p->block_id, wide_type, b);
        uint32_t mul = ssir_build_mul(p->mod, p->func_id, p->block_id, wide_type, wa, wb);
        uint32_t result = ssir_build_add(p->mod, p->func_id, p->block_id, wide_type, mul, c);
        pl_store_reg_typed(p, inst->dst.name, result, wide_type);
    } else {
        uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
        uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
        uint32_t c = pl_resolve_operand(p, &inst->src[2], type);
        if (p->had_error) return;
        uint32_t mul = ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
        uint32_t result = ssir_build_add(p->mod, p->func_id, p->block_id, type, mul, c);
        pl_store_reg_typed(p, inst->dst.name, result, type);
    }
}

static void pl_lower_fma(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
    uint32_t c = pl_resolve_operand(p, &inst->src[2], type);
    if (p->had_error) return;
    bool round_floor = (inst->modifiers & PTX_MOD_RM) != 0;
    bool round_ceil = (inst->modifiers & PTX_MOD_RP) != 0;
    uint32_t result;
    if (round_floor || round_ceil) {
        uint32_t prod = ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
        SsirBuiltinId bid = round_floor ? SSIR_BUILTIN_FLOOR : SSIR_BUILTIN_CEIL;
        uint32_t rargs[] = { prod };
        uint32_t rprod = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                             type, bid, rargs, 1);
        result = ssir_build_add(p->mod, p->func_id, p->block_id, type, rprod, c);
    } else {
        uint32_t args[] = { a, b, c };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                     type, SSIR_BUILTIN_FMA, args, 3);
    }
    pl_store_reg(p, inst->dst.name, result);
}

static void pl_lower_bitwise(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
    if (p->had_error) return;
    SsirType *ty = ssir_get_type(p->mod, type);
    bool is_bool = ty && ty->kind == SSIR_TYPE_BOOL;
    uint32_t result = 0;
    switch (inst->opcode) {
    case PTX_OP_AND:
        result = is_bool ? ssir_build_and(p->mod, p->func_id, p->block_id, type, a, b)
                         : ssir_build_bit_and(p->mod, p->func_id, p->block_id, type, a, b);
        break;
    case PTX_OP_OR:
        result = is_bool ? ssir_build_or(p->mod, p->func_id, p->block_id, type, a, b)
                         : ssir_build_bit_or(p->mod, p->func_id, p->block_id, type, a, b);
        break;
    case PTX_OP_XOR:
        result = ssir_build_bit_xor(p->mod, p->func_id, p->block_id, type, a, b);
        break;
    default: break;
    }
    if (result) pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_shift(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], ssir_type_u32(p->mod));
    if (p->had_error) return;
    SsirType *ty = ssir_get_type(p->mod, type);
    uint32_t result;
    if (inst->opcode == PTX_OP_SHL)
        result = ssir_build_shl(p->mod, p->func_id, p->block_id, type, a, b);
    else if (ty && pl_is_signed_type(ty->kind))
        result = ssir_build_shr(p->mod, p->func_id, p->block_id, type, a, b);
    else
        result = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, type, a, b);
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_setp(PtxLower *p, const PtxInst *inst) {
    uint32_t src_type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], src_type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], src_type);
    if (p->had_error) return;

    uint32_t bool_t = ssir_type_bool(p->mod);
    SsirOpcode cmp_opcode = pl_cmp_op(inst->cmp_op);
    uint32_t cmp_result = 0;
    switch (cmp_opcode) {
    case SSIR_OP_EQ: cmp_result = ssir_build_eq(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    case SSIR_OP_NE: cmp_result = ssir_build_ne(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    case SSIR_OP_LT: cmp_result = ssir_build_lt(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    case SSIR_OP_LE: cmp_result = ssir_build_le(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    case SSIR_OP_GT: cmp_result = ssir_build_gt(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    case SSIR_OP_GE: cmp_result = ssir_build_ge(p->mod, p->func_id, p->block_id, bool_t, a, b); break;
    default: break;
    }

    if (inst->has_combine && inst->src_count >= 3) {
        uint32_t pred_src = pl_resolve_operand(p, &inst->src[2], bool_t);
        if (strcmp(inst->combine, "and") == 0)
            cmp_result = ssir_build_and(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
        else if (strcmp(inst->combine, "or") == 0)
            cmp_result = ssir_build_or(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
        else if (strcmp(inst->combine, "xor") == 0)
            cmp_result = ssir_build_ne(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
    }

    pl_store_reg(p, inst->dst.name, cmp_result);
    if (inst->has_dst2) {
        uint32_t neg = ssir_build_not(p->mod, p->func_id, p->block_id, bool_t, cmp_result);
        pl_store_reg(p, inst->dst2.name, neg);
    }
}

static void pl_lower_selp(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], type);
    uint32_t pred = pl_resolve_operand(p, &inst->src[2], ssir_type_bool(p->mod));
    if (p->had_error) return;
    uint32_t args[] = { b, a, pred };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_SELECT, args, 3);
    pl_store_reg(p, inst->dst.name, result);
}

static void pl_lower_mov(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);

    if (inst->src[0].kind == PTX_OPER_REG) {
        SsirType *t = ssir_get_type(p->mod, type);
        if (t && (t->kind == SSIR_TYPE_U64 || t->kind == SSIR_TYPE_I64)) {
            PlReg *sr = pl_find_reg(p, inst->src[0].name);
            if (sr) {
                SsirType *svt = ssir_get_type(p->mod, sr->val_type);
                if (svt && (svt->kind == SSIR_TYPE_ARRAY ||
                            svt->kind == SSIR_TYPE_RUNTIME_ARRAY)) {
                    uint32_t zero = ssir_const_u64(p->mod, 0);
                    pl_store_reg_typed(p, inst->dst.name, zero, type);
                    PlReg *dr = pl_find_reg(p, inst->dst.name);
                    if (dr) dr->local_ptr_id = sr->ptr_id;
                    return;
                }
                uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
                if (p->had_error) return;
                pl_store_reg_typed(p, inst->dst.name, src, type);
                pl_propagate_buffer(p, inst->dst.name, inst->src[0].name);
                return;
            }
            for (int gi = 0; gi < p->mod->global_count; gi++) {
                if (p->mod->globals[gi].name &&
                    strcmp(p->mod->globals[gi].name, inst->src[0].name) == 0) {
                    uint32_t zero = ssir_const_u64(p->mod, 0);
                    pl_store_reg_typed(p, inst->dst.name, zero, type);
                    PlReg *dr = pl_find_reg(p, inst->dst.name);
                    if (dr) {
                        dr->global_id = p->mod->globals[gi].id;
                        dr->is_bda_ptr = false;
                    }
                    pl_add_iface(p, p->mod->globals[gi].id);
                    return;
                }
            }
        }
    }

    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;
    pl_store_reg_typed(p, inst->dst.name, src, type);
}

static void pl_lower_ld(PtxLower *p, const PtxInst *inst) {
    SsirAddressSpace space = pl_mem_space(inst->space);
    uint32_t type = pl_map_type(p, inst->type);

    if (inst->vec_width <= 1) {
        if (space == SSIR_ADDR_UNIFORM) {
            const char *param_name = inst->src[0].kind == PTX_OPER_ADDR
                ? inst->src[0].base : inst->src[0].name;
            PlReg *param_reg = pl_find_reg(p, param_name);
            if (param_reg) {
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                               param_reg->val_type, param_reg->ptr_id);
                if (param_reg->val_type != type)
                    val = ssir_build_convert(p->mod, p->func_id, p->block_id, type, val);
                pl_store_reg(p, inst->dst.name, val);
                pl_propagate_buffer(p, inst->dst.name, param_name);
            } else {
                /* Check if it's a global (e.g. ld.const [coeffs]) */
                SsirGlobalVar *gv = NULL;
                uint32_t gv_id = 0;
                for (uint32_t gi = 0; gi < (uint32_t)p->mod->global_count; gi++) {
                    if (p->mod->globals[gi].name &&
                        strcmp(p->mod->globals[gi].name, param_name) == 0) {
                        gv = &p->mod->globals[gi];
                        gv_id = gi;
                        break;
                    }
                }
                if (gv) {
                    uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_UNIFORM);
                    int64_t off = inst->src[0].kind == PTX_OPER_ADDR ? inst->src[0].offset : 0;
                    uint32_t byte_size = pl_type_byte_size(p, type);
                    uint32_t idx_val = byte_size > 0 ? (uint32_t)(off / byte_size) : 0;
                    uint32_t cidx = ssir_const_u32(p->mod, idx_val);
                    uint32_t indices[] = { cidx };
                    uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                        elem_ptr_type, gv_id, indices, 1);
                    uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, ptr);
                    pl_store_reg(p, inst->dst.name, val);
                } else {
                    pl_error(p, "unknown parameter '%s'", param_name);
                }
            }
            return;
        }

        if (space == SSIR_ADDR_FUNCTION) {
            const char *base_name = inst->src[0].kind == PTX_OPER_ADDR
                ? inst->src[0].base : inst->src[0].name;
            PlReg *local_reg = pl_find_reg(p, base_name);
            if (local_reg && local_reg->local_ptr_id) {
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t byte_offset = pl_resolve_operand(p, &inst->src[0], u64_t);
                if (p->had_error) return;
                uint32_t elem_sz = ssir_const_u64(p->mod, pl_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id,
                    u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id,
                    u32_t, idx_u64);
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_FUNCTION);
                uint32_t indices[] = { idx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                    elem_ptr_type, local_reg->local_ptr_id, indices, 1);
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, ptr);
                pl_store_reg(p, inst->dst.name, val);
            } else if (local_reg) {
                int64_t off = inst->src[0].kind == PTX_OPER_ADDR ? inst->src[0].offset : 0;
                uint32_t byte_size = pl_type_byte_size(p, type);
                uint32_t idx_val = byte_size > 0 ? (uint32_t)(off / byte_size) : 0;
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_FUNCTION);
                uint32_t cidx = ssir_const_u32(p->mod, idx_val);
                uint32_t indices[] = { cidx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                    elem_ptr_type, local_reg->ptr_id, indices, 1);
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, ptr);
                pl_store_reg(p, inst->dst.name, val);
            } else {
                pl_error(p, "undefined local variable '%s'", base_name);
            }
            return;
        }

        if (space == SSIR_ADDR_STORAGE) {
            const char *base_name = inst->src[0].kind == PTX_OPER_ADDR
                ? inst->src[0].base : inst->src[0].name;
            uint32_t byte_offset = pl_resolve_operand(p, &inst->src[0], ssir_type_u64(p->mod));
            if (p->had_error) return;

            PlReg *base_reg = base_name[0] ? pl_find_reg(p, base_name) : NULL;

            if (base_reg && base_reg->is_bda_ptr && p->use_bda) {
                uint32_t psb_ptr_type = ssir_type_ptr(p->mod, type,
                    SSIR_ADDR_PHYSICAL_STORAGE_BUFFER);
                uint32_t typed_ptr = ssir_build_bitcast(p->mod, p->func_id,
                    p->block_id, psb_ptr_type, byte_offset);
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                    type, typed_ptr);
                pl_store_reg(p, inst->dst.name, val);
            } else if (base_reg && (base_reg->pending_binding != UINT32_MAX || base_reg->global_id != 0)) {
                uint32_t buf_global = pl_materialize_buffer(p, base_reg, type);
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t elem_sz = ssir_const_u64(p->mod, pl_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id, u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id, u32_t, idx_u64);
                bool is_direct_global = (base_reg->pending_binding == UINT32_MAX &&
                                          base_reg->global_id != 0);
                uint32_t ptr;
                if (is_direct_global) {
                    SsirGlobalVar *gv = ssir_get_global(p->mod, buf_global);
                    SsirType *gvt = gv ? ssir_get_type(p->mod, gv->type) : NULL;
                    SsirAddressSpace gv_space = (gvt && gvt->kind == SSIR_TYPE_PTR) ? gvt->ptr.space : SSIR_ADDR_STORAGE;
                    uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, gv_space);
                    uint32_t indices[] = { idx };
                    ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                        elem_ptr_type, buf_global, indices, 1);
                } else {
                    uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_STORAGE);
                    uint32_t const_0 = ssir_const_u32(p->mod, 0);
                    uint32_t indices[] = { const_0, idx };
                    ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                        elem_ptr_type, buf_global, indices, 2);
                }
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, ptr);
                pl_store_reg(p, inst->dst.name, val);
            } else {
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, byte_offset);
                pl_store_reg(p, inst->dst.name, val);
            }
        } else {
            uint32_t addr = pl_resolve_operand(p, &inst->src[0], ssir_type_u64(p->mod));
            if (p->had_error) return;
            uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, addr);
            pl_store_reg(p, inst->dst.name, val);
        }
    } else {
        /* Vector load: dst is a VEC operand */
        uint32_t vec_type = ssir_type_vec(p->mod, type, inst->vec_width);
        uint32_t addr = pl_resolve_operand(p, &inst->src[0], ssir_type_u64(p->mod));
        if (p->had_error) return;
        uint32_t vec_val = ssir_build_load(p->mod, p->func_id, p->block_id, vec_type, addr);
        if (inst->dst.kind == PTX_OPER_VEC) {
            for (int i = 0; i < inst->dst.vec_count && i < inst->vec_width; i++) {
                uint32_t comp = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                                   type, vec_val, i);
                pl_store_reg(p, inst->dst.regs[i], comp);
            }
        }
    }
}

static void pl_lower_st(PtxLower *p, const PtxInst *inst) {
    SsirAddressSpace space = pl_mem_space(inst->space);
    uint32_t type = pl_map_type(p, inst->type);

    /* For st, parser puts address in src[0] and value in src[1] */
    const PtxOperand *addr_op = &inst->src[0];
    const PtxOperand *val_op = &inst->src[1];

    if (inst->vec_width <= 1) {
        if (space == SSIR_ADDR_FUNCTION) {
            const char *base_name = addr_op->kind == PTX_OPER_ADDR
                ? addr_op->base : addr_op->name;
            uint32_t val = pl_resolve_operand(p, val_op, type);
            if (p->had_error) return;
            PlReg *local_reg = pl_find_reg(p, base_name);
            if (local_reg && local_reg->local_ptr_id) {
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t byte_offset = pl_resolve_operand(p, addr_op, u64_t);
                if (p->had_error) return;
                uint32_t elem_sz = ssir_const_u64(p->mod, pl_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id,
                    u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id,
                    u32_t, idx_u64);
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_FUNCTION);
                uint32_t indices[] = { idx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                    elem_ptr_type, local_reg->local_ptr_id, indices, 1);
                ssir_build_store(p->mod, p->func_id, p->block_id, ptr, val);
            } else if (local_reg) {
                int64_t off = addr_op->kind == PTX_OPER_ADDR ? addr_op->offset : 0;
                uint32_t byte_size = pl_type_byte_size(p, type);
                uint32_t idx_val = byte_size > 0 ? (uint32_t)(off / byte_size) : 0;
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_FUNCTION);
                uint32_t cidx = ssir_const_u32(p->mod, idx_val);
                uint32_t indices[] = { cidx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                    elem_ptr_type, local_reg->ptr_id, indices, 1);
                ssir_build_store(p->mod, p->func_id, p->block_id, ptr, val);
            } else {
                pl_error(p, "undefined local variable '%s'", base_name);
            }
            return;
        }

        if (space == SSIR_ADDR_STORAGE) {
            const char *base_name = addr_op->kind == PTX_OPER_ADDR
                ? addr_op->base : addr_op->name;
            uint32_t byte_offset = pl_resolve_operand(p, addr_op, ssir_type_u64(p->mod));
            uint32_t val = pl_resolve_operand(p, val_op, type);
            if (p->had_error) return;

            PlReg *base_reg = base_name[0] ? pl_find_reg(p, base_name) : NULL;

            if (base_reg && base_reg->is_bda_ptr && p->use_bda) {
                uint32_t psb_ptr_type = ssir_type_ptr(p->mod, type,
                    SSIR_ADDR_PHYSICAL_STORAGE_BUFFER);
                uint32_t typed_ptr = ssir_build_bitcast(p->mod, p->func_id,
                    p->block_id, psb_ptr_type, byte_offset);
                ssir_build_store(p->mod, p->func_id, p->block_id, typed_ptr, val);
            } else if (base_reg && (base_reg->pending_binding != UINT32_MAX || base_reg->global_id != 0)) {
                uint32_t buf_global = pl_materialize_buffer(p, base_reg, type);
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t elem_sz = ssir_const_u64(p->mod, pl_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id, u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id, u32_t, idx_u64);
                bool is_direct_global = (base_reg->pending_binding == UINT32_MAX &&
                                          base_reg->global_id != 0);
                uint32_t ptr;
                if (is_direct_global) {
                    SsirGlobalVar *gv = ssir_get_global(p->mod, buf_global);
                    SsirType *gvt = gv ? ssir_get_type(p->mod, gv->type) : NULL;
                    SsirAddressSpace gv_space = (gvt && gvt->kind == SSIR_TYPE_PTR) ? gvt->ptr.space : SSIR_ADDR_STORAGE;
                    uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, gv_space);
                    uint32_t indices[] = { idx };
                    ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                        elem_ptr_type, buf_global, indices, 1);
                } else {
                    uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_STORAGE);
                    uint32_t const_0 = ssir_const_u32(p->mod, 0);
                    uint32_t indices[] = { const_0, idx };
                    ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
                        elem_ptr_type, buf_global, indices, 2);
                }
                ssir_build_store(p->mod, p->func_id, p->block_id, ptr, val);
            } else {
                ssir_build_store(p->mod, p->func_id, p->block_id, byte_offset, val);
            }
        } else {
            uint32_t addr = pl_resolve_operand(p, addr_op, ssir_type_u64(p->mod));
            uint32_t val = pl_resolve_operand(p, val_op, type);
            if (p->had_error) return;
            ssir_build_store(p->mod, p->func_id, p->block_id, addr, val);
        }
    } else {
        /* Vector store: addr in src[0], values in src[1] (vec operand) */
        uint32_t vec_type = ssir_type_vec(p->mod, type, inst->vec_width);
        uint32_t addr = pl_resolve_operand(p, addr_op, ssir_type_u64(p->mod));
        if (p->had_error) return;
        uint32_t comps[4];
        for (int i = 0; i < inst->vec_width && i < val_op->vec_count; i++)
            comps[i] = pl_load_reg(p, val_op->regs[i]);
        uint32_t vec_val = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                                vec_type, comps, inst->vec_width);
        ssir_build_store(p->mod, p->func_id, p->block_id, addr, vec_val);
    }
}

static void pl_lower_cvt(PtxLower *p, const PtxInst *inst) {
    uint32_t dst_type = pl_map_type(p, inst->type);
    uint32_t src_type = pl_map_type(p, inst->type2);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], src_type);
    if (p->had_error) return;

    uint32_t result;
    SsirType *dt = ssir_get_type(p->mod, dst_type);
    SsirType *st = ssir_get_type(p->mod, src_type);
    if (dt && st && dt->kind != st->kind) {
        src = ssir_build_bitcast(p->mod, p->func_id, p->block_id, src_type, src);
        result = ssir_build_convert(p->mod, p->func_id, p->block_id, dst_type, src);
    } else {
        /* Same-type: check for integer rounding modes */
        bool same_type = (dt && st && dt->kind == st->kind);
        SsirBuiltinId rnd_bid = (SsirBuiltinId)-1;
        if (same_type && pl_is_float_type(dt->kind)) {
            if (inst->modifiers & PTX_MOD_RMI) rnd_bid = SSIR_BUILTIN_FLOOR;
            else if (inst->modifiers & PTX_MOD_RPI) rnd_bid = SSIR_BUILTIN_CEIL;
            else if (inst->modifiers & PTX_MOD_RZI) rnd_bid = SSIR_BUILTIN_TRUNC;
            else if (inst->modifiers & PTX_MOD_RNI) rnd_bid = SSIR_BUILTIN_ROUND;
        }
        if ((int)rnd_bid != -1) {
            uint32_t rargs[] = { src };
            result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         dst_type, rnd_bid, rargs, 1);
        } else {
            result = src;
        }
    }

    bool has_sat = (inst->modifiers & PTX_MOD_SAT) != 0;
    if (has_sat && dt &&
        (dt->kind == SSIR_TYPE_F32 || dt->kind == SSIR_TYPE_F64)) {
        uint32_t zero, one;
        if (dt->kind == SSIR_TYPE_F64) {
            zero = ssir_const_f64(p->mod, 0.0);
            one = ssir_const_f64(p->mod, 1.0);
        } else {
            zero = ssir_const_f32(p->mod, 0.0f);
            one = ssir_const_f32(p->mod, 1.0f);
        }
        uint32_t args[] = { result, zero, one };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    dst_type, SSIR_BUILTIN_CLAMP, args, 3);
    }
    pl_store_reg_typed(p, inst->dst.name, result, dst_type);
}

static void pl_lower_cvta(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;
    pl_store_reg(p, inst->dst.name, src);
    if (inst->src[0].kind == PTX_OPER_REG)
        pl_propagate_buffer(p, inst->dst.name, inst->src[0].name);
}

static void pl_lower_math_unary(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;

    uint32_t result;
    if (inst->opcode == PTX_OP_RCP) {
        uint32_t one = ssir_const_f32(p->mod, 1.0f);
        SsirType *ty = ssir_get_type(p->mod, type);
        if (ty && ty->kind == SSIR_TYPE_F64) one = ssir_const_f64(p->mod, 1.0);
        result = ssir_build_div(p->mod, p->func_id, p->block_id, type, one, src);
    } else {
        SsirBuiltinId bid = SSIR_BUILTIN_SQRT;
        switch (inst->opcode) {
        case PTX_OP_SQRT:  bid = SSIR_BUILTIN_SQRT; break;
        case PTX_OP_RSQRT: bid = SSIR_BUILTIN_INVERSESQRT; break;
        case PTX_OP_SIN:   bid = SSIR_BUILTIN_SIN; break;
        case PTX_OP_COS:   bid = SSIR_BUILTIN_COS; break;
        case PTX_OP_LG2:   bid = SSIR_BUILTIN_LOG2; break;
        case PTX_OP_EX2:   bid = SSIR_BUILTIN_EXP2; break;
        default: break;
        }
        uint32_t args[] = { src };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, bid, args, 1);
    }
    pl_store_reg(p, inst->dst.name, result);
}

static void pl_lower_atom(PtxLower *p, const PtxInst *inst) {
    SsirAddressSpace space = pl_mem_space(inst->space);
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
    uint32_t addr = pl_resolve_operand(p, &inst->src[0], ssir_type_u64(p->mod));
    uint32_t val = pl_resolve_operand(p, &inst->src[1], type);
    uint32_t cmp = 0;
    if (inst->atomic_op == PTX_ATOMIC_CAS && inst->src_count >= 3)
        cmp = pl_resolve_operand(p, &inst->src[2], type);
    if (p->had_error) return;
    (void)ptr_type;
    uint32_t result = ssir_build_atomic(p->mod, p->func_id, p->block_id,
                                        type, pl_atomic_op(inst->atomic_op),
                                        addr, val, cmp);
    pl_store_reg(p, inst->dst.name, result);
}

static void pl_lower_bar(PtxLower *p, const PtxInst *inst) {
    (void)inst;
    ssir_build_barrier(p->mod, p->func_id, p->block_id, SSIR_BARRIER_WORKGROUP);
}

static void pl_lower_membar(PtxLower *p, const PtxInst *inst) {
    SsirBarrierScope scope = SSIR_BARRIER_WORKGROUP;
    if (inst->membar_scope == PTX_MEMBAR_GL || inst->membar_scope == PTX_MEMBAR_SYS)
        scope = SSIR_BARRIER_STORAGE;
    ssir_build_barrier(p->mod, p->func_id, p->block_id, scope);
}

/* ===== Loop Restructuring ===== */

static void pl_create_loop_structure(PtxLower *p, uint32_t target_block,
                                      uint32_t continue_block,
                                      uint32_t merge_block) {
    SsirBlock *target = ssir_get_block(p->mod, p->func_id, target_block);
    if (!target || target->inst_count == 0) return;

    uint32_t n = target->inst_count;
    SsirInst *saved = (SsirInst *)PTX_REALLOC(NULL, n * sizeof(SsirInst));
    if (!saved) return;
    memcpy(saved, target->insts, n * sizeof(SsirInst));
    target->inst_count = 0;

    uint32_t body_block = ssir_block_insert_after(p->mod, p->func_id,
                                                   target_block, NULL);
    SsirBlock *body = ssir_get_block(p->mod, p->func_id, body_block);
    if (!body) { free(saved); return; }
    body->insts = saved;
    body->inst_count = n;
    body->inst_capacity = n;

    uint32_t actual_continue = continue_block;
    if (continue_block == target_block && body->inst_count > 0) {
        SsirInst last = body->insts[body->inst_count - 1];
        if (last.op == SSIR_OP_BRANCH_COND || last.op == SSIR_OP_BRANCH) {
            body->inst_count--;
            actual_continue = ssir_block_insert_after(p->mod, p->func_id,
                                                       body_block, NULL);
            body = ssir_get_block(p->mod, p->func_id, body_block);
            SsirBlock *cont = ssir_get_block(p->mod, p->func_id, actual_continue);
            if (cont) {
                cont->insts = (SsirInst *)PTX_REALLOC(NULL, sizeof(SsirInst));
                if (!cont->insts) return;
                cont->insts[0] = last;
                cont->inst_count = 1;
                cont->inst_capacity = 1;
            }
            ssir_build_branch(p->mod, p->func_id, body_block, actual_continue);
            body = ssir_get_block(p->mod, p->func_id, body_block);
        }
    }

    for (uint32_t i = 0; i < body->inst_count; i++) {
        SsirInst *ii = &body->insts[i];
        if (ii->op == SSIR_OP_BRANCH_COND) {
            for (int j = 1; j <= 2; j++) {
                uint32_t t = ii->operands[j];
                if (t != target_block && t != body_block &&
                    t != actual_continue && t != merge_block)
                    ii->operands[j] = merge_block;
            }
            if (ii->operand_count >= 4 && ii->operands[3] != 0) {
                uint32_t m = ii->operands[3];
                if (m != target_block && m != body_block &&
                    m != actual_continue && m != merge_block)
                    ii->operands[3] = 0;
            }
        }
    }

    SsirBlock *cont = ssir_get_block(p->mod, p->func_id, actual_continue);
    if (cont && actual_continue != continue_block) {
        for (uint32_t i = 0; i < cont->inst_count; i++) {
            SsirInst *ii = &cont->insts[i];
            if (ii->op == SSIR_OP_BRANCH_COND) {
                for (int j = 1; j <= 2; j++) {
                    uint32_t t = ii->operands[j];
                    if (t != target_block && t != body_block &&
                        t != actual_continue && t != merge_block)
                        ii->operands[j] = merge_block;
                }
                if (ii->operand_count >= 4 && ii->operands[3] != 0)
                    ii->operands[3] = 0;
            }
        }
    }

    target = ssir_get_block(p->mod, p->func_id, target_block);
    ssir_build_loop_merge(p->mod, p->func_id, target_block,
                          merge_block, actual_continue);
    ssir_build_branch(p->mod, p->func_id, target_block, body_block);
}

/* ===== If-Else Merge Pre-computation ===== */

static const char *pl_lookup_precomputed_merge(PtxLower *p, const char *cond_target) {
    for (int i = 0; i < p->precomputed_merge_count; i++)
        if (strcmp(p->precomputed_merges[i].cond_target, cond_target) == 0)
            return p->precomputed_merges[i].merge_label;
    return NULL;
}

static void pl_add_precomputed_merge(PtxLower *p, const char *cond_target,
                                      const char *merge_label) {
    if (p->precomputed_merge_count >= p->precomputed_merge_cap) {
        p->precomputed_merge_cap = p->precomputed_merge_cap ? p->precomputed_merge_cap * 2 : 16;
        p->precomputed_merges = PTX_REALLOC(p->precomputed_merges,
            p->precomputed_merge_cap * sizeof(p->precomputed_merges[0]));
    }
    int idx = p->precomputed_merge_count++;
    snprintf(p->precomputed_merges[idx].cond_target, 80, "%s", cond_target);
    snprintf(p->precomputed_merges[idx].merge_label, 80, "%s", merge_label);
}

static int pl_find_label_index(const PtxStmt *body, int body_count, const char *name) {
    for (int i = 0; i < body_count; i++)
        if (body[i].kind == PTX_STMT_LABEL && strcmp(body[i].label, name) == 0)
            return i;
    return -1;
}

static const char *pl_find_region_exit(const PtxStmt *body, int body_count,
                                       int true_start, const char *false_label) {
    int false_idx = pl_find_label_index(body, body_count, false_label);
    if (false_idx < 0) return NULL;
    int search_end = false_idx;

    const char *last_uncond = NULL;
    for (int i = true_start; i < search_end; i++) {
        const PtxStmt *s = &body[i];
        if (s->kind != PTX_STMT_INST) continue;
        const PtxInst *inst = &s->inst;
        if (inst->opcode == PTX_OP_BRA && !inst->has_pred &&
            inst->src_count > 0 && inst->src[0].kind == PTX_OPER_LABEL) {
            last_uncond = inst->src[0].name;
        }
    }
    return last_uncond;
}

static void pl_precompute_merges(PtxLower *p, const PtxStmt *body, int body_count) {
    for (int i = 0; i < body_count - 1; i++) {
        if (body[i].kind != PTX_STMT_INST) continue;
        const PtxInst *cond = &body[i].inst;
        if (cond->opcode != PTX_OP_BRA || !cond->has_pred) continue;
        if (cond->src_count < 1 || cond->src[0].kind != PTX_OPER_LABEL) continue;

        const char *true_label = cond->src[0].name;

        if (i + 1 < body_count && body[i + 1].kind == PTX_STMT_INST) {
            const PtxInst *uncond = &body[i + 1].inst;
            if (uncond->opcode == PTX_OP_BRA && !uncond->has_pred &&
                uncond->src_count > 0 && uncond->src[0].kind == PTX_OPER_LABEL) {
                const char *false_label = uncond->src[0].name;
                int true_idx = pl_find_label_index(body, body_count, true_label);
                if (true_idx < 0) continue;
                const char *true_exit = pl_find_region_exit(body, body_count,
                                                             true_idx + 1, false_label);
                if (!true_exit) continue;
                if (strcmp(true_exit, true_label) == 0) continue;
                if (strcmp(true_exit, false_label) != 0) {
                    int false_idx = pl_find_label_index(body, body_count, false_label);
                    if (false_idx < 0) continue;
                    int merge_idx = pl_find_label_index(body, body_count, true_exit);
                    if (merge_idx < 0) continue;
                    if (true_idx < false_idx && false_idx < merge_idx)
                        pl_add_precomputed_merge(p, true_label, true_exit);
                }
                continue;
            }
        }

        int true_idx = pl_find_label_index(body, body_count, true_label);
        if (true_idx < 0) continue;
        const char *false_exit = pl_find_region_exit(body, body_count,
                                                      i + 1, true_label);
        if (!false_exit) continue;
        int true_end = -1;
        for (int j = true_idx + 1; j < body_count; j++) {
            if (body[j].kind == PTX_STMT_LABEL) { true_end = j; break; }
        }
        if (true_end < 0) continue;
        const char *true_next_label = body[true_end].label;
        const char *true_exit = NULL;
        for (int j = true_idx + 1; j < true_end; j++) {
            if (body[j].kind != PTX_STMT_INST) continue;
            const PtxInst *inst = &body[j].inst;
            if (inst->opcode == PTX_OP_BRA && !inst->has_pred &&
                inst->src_count > 0 && inst->src[0].kind == PTX_OPER_LABEL)
                true_exit = inst->src[0].name;
        }
        if (!true_exit) true_exit = true_next_label;
        if (strcmp(false_exit, true_exit) == 0) {
            int merge_idx = pl_find_label_index(body, body_count, false_exit);
            if (merge_idx > true_idx)
                pl_add_precomputed_merge(p, true_label, false_exit);
        }
    }
}

/* ===== Control Flow ===== */

static void pl_lower_bra(PtxLower *p, const PtxInst *inst,
                          uint32_t pred_val, bool has_pred, bool pred_negated) {
    const char *label = inst->src[0].name;
    bool is_back_edge = pl_is_label_defined(p, label);
    uint32_t target = pl_get_or_create_label(p, label);

    if (has_pred) {
        if (pred_negated)
            pred_val = ssir_build_not(p->mod, p->func_id, p->block_id,
                                      ssir_type_bool(p->mod), pred_val);
        uint32_t fallthrough = ssir_block_create(p->mod, p->func_id, NULL);

        const char *precomputed = pl_lookup_precomputed_merge(p, label);

        if (is_back_edge) {
            ssir_build_branch_cond(p->mod, p->func_id, p->block_id,
                                   pred_val, target, fallthrough);
            pl_create_loop_structure(p, target, p->block_id, fallthrough);
            pl_mark_merge(p, fallthrough);
        } else if (precomputed) {
            uint32_t real_merge = pl_get_or_create_label(p, precomputed);
            uint32_t deferred_id = ssir_module_alloc_id(p->mod);
            ssir_build_branch_cond_merge(p->mod, p->func_id, p->block_id,
                                   pred_val, target, fallthrough, deferred_id);
            pl_mark_merge(p, deferred_id);
            pl_push_construct(p, deferred_id, real_merge, true);
        } else if (!pl_is_merge_used(p, target) &&
                   !pl_has_deferred_merge_for(p, target)) {
            ssir_build_branch_cond_merge(p->mod, p->func_id, p->block_id,
                                   pred_val, target, fallthrough, target);
            pl_mark_merge(p, target);
            pl_push_construct(p, target, target, false);
        } else {
            uint32_t new_fallthrough = ssir_block_create(p->mod, p->func_id, NULL);
            ssir_build_branch_cond_merge(p->mod, p->func_id, p->block_id,
                                   pred_val, fallthrough, new_fallthrough, fallthrough);
            pl_mark_merge(p, fallthrough);
            pl_push_construct(p, fallthrough, target, false);
            fallthrough = new_fallthrough;
        }
        p->block_id = fallthrough;
        p->block_from_label = false;
    } else {
        uint32_t actual_target = target;
        for (int i = p->construct_depth - 1; i >= 0; i--) {
            if (p->construct_stack[i].target_label == target) {
                if (p->construct_stack[i].merge_block != target)
                    actual_target = p->construct_stack[i].merge_block;
                break;
            }
            if (p->construct_stack[i].merge_block != target) {
                bool has_outer = false;
                for (int j = i - 1; j >= 0; j--) {
                    if (p->construct_stack[j].target_label == target) {
                        has_outer = true;
                        break;
                    }
                }
                if (has_outer) {
                    actual_target = p->construct_stack[i].merge_block;
                    break;
                }
            }
        }
        ssir_build_branch(p->mod, p->func_id, p->block_id, actual_target);
        p->block_id = ssir_block_create(p->mod, p->func_id, NULL);
        p->block_from_label = false;
    }
}

static void pl_lower_ret(PtxLower *p, const PtxInst *inst) {
    (void)inst;
    ssir_build_return_void(p->mod, p->func_id, p->block_id);
    p->block_id = ssir_block_create(p->mod, p->func_id, NULL);
    p->block_from_label = false;
}

static void pl_lower_call(PtxLower *p, const PtxInst *inst) {
    uint32_t callee = 0;
    uint32_t ret_type = ssir_type_void(p->mod);
    for (int i = 0; i < p->func_count; i++) {
        if (strcmp(p->funcs[i].name, inst->src[0].name) == 0) {
            callee = p->funcs[i].func_id;
            ret_type = p->funcs[i].ret_type;
            break;
        }
    }
    if (!callee) {
        pl_error(p, "undefined function '%s'", inst->src[0].name);
        return;
    }
    uint32_t args[16];
    int arg_count = 0;
    for (int i = 1; i < inst->src_count && arg_count < 16; i++)
        args[arg_count++] = pl_resolve_operand(p, &inst->src[i], 0);
    if (p->had_error) return;
    uint32_t result = ssir_build_call(p->mod, p->func_id, p->block_id,
                                      ret_type, callee, args, arg_count);
    if (inst->dst.kind == PTX_OPER_REG && inst->dst.name[0] && result)
        pl_store_reg(p, inst->dst.name, result);
}

/* ===== Texture Operations ===== */

static int pl_ssir_coord_count(SsirTextureDim dim) {
    switch (dim) {
    case SSIR_TEX_1D:       return 1;
    case SSIR_TEX_2D:       return 2;
    case SSIR_TEX_3D:       return 3;
    case SSIR_TEX_CUBE:     return 3;
    case SSIR_TEX_1D_ARRAY: return 2;
    case SSIR_TEX_2D_ARRAY: return 3;
    default:                return 2;
    }
}

static void pl_lower_tex(PtxLower *p, const PtxInst *inst) {
    int coord_count = 0;
    SsirTextureDim dim = pl_tex_dim(inst->tex_geom, &coord_count);
    uint32_t dst_type = pl_map_type(p, inst->type);
    uint32_t coord_type = pl_map_type(p, inst->type2);

    /* dst is VEC: {d0,d1,d2,d3} */
    /* src[0] is the texture name (LABEL), src[1..] are coords */
    const char *tex_name = inst->src[0].name;

    /* Collect coordinate values */
    uint32_t all_coords[16] = {0};
    int nall = inst->src_count - 1;
    for (int i = 0; i < nall && i < 16; i++)
        all_coords[i] = pl_resolve_operand(p, &inst->src[i + 1], coord_type);
    if (p->had_error) return;

    int base_coords = coord_count;
    int ncoords = base_coords < nall ? base_coords : nall;

    uint32_t lod_val = 0;
    uint32_t ddx_val = 0, ddy_val = 0;
    if (inst->mip_mode == PTX_MIP_LEVEL && nall > base_coords) {
        lod_val = all_coords[base_coords];
    } else if (inst->mip_mode == PTX_MIP_GRAD && nall > base_coords) {
        int grad_start = base_coords;
        int grad_dim = base_coords;
        uint32_t f32_t = ssir_type_f32(p->mod);
        if (grad_dim == 1) {
            ddx_val = (grad_start < nall) ? all_coords[grad_start] : ssir_const_f32(p->mod, 0.0f);
            ddy_val = (grad_start + 1 < nall) ? all_coords[grad_start + 1] : ssir_const_f32(p->mod, 0.0f);
        } else {
            uint32_t ddx_comps[4] = {0}, ddy_comps[4] = {0};
            for (int i = 0; i < grad_dim && i < 4; i++) {
                ddx_comps[i] = (grad_start + i < nall) ? all_coords[grad_start + i] : ssir_const_f32(p->mod, 0.0f);
                ddy_comps[i] = (grad_start + grad_dim + i < nall) ? all_coords[grad_start + grad_dim + i] : ssir_const_f32(p->mod, 0.0f);
            }
            uint32_t gvec_t = ssir_type_vec(p->mod, f32_t, grad_dim);
            ddx_val = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                            gvec_t, ddx_comps, grad_dim);
            ddy_val = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                            gvec_t, ddy_comps, grad_dim);
        }
    }

    PlTexRef *tref = pl_find_texref(p, tex_name);
    if (!tref) {
        uint32_t f32_t = ssir_type_f32(p->mod);
        uint32_t tex_t = ssir_type_texture(p->mod, dim, f32_t);
        uint32_t ptr_t = ssir_type_ptr(p->mod, tex_t, SSIR_ADDR_UNIFORM_CONSTANT);
        uint32_t gid = ssir_global_var(p->mod, tex_name, ptr_t);
        ssir_global_set_group(p->mod, gid, 0);
        ssir_global_set_binding(p->mod, gid, p->next_binding++);
        pl_add_iface(p, gid);
        tref = pl_add_texref(p, tex_name, gid, tex_t, false, dim);
    }

    PlSamplerRef *sref = pl_get_implicit_sampler(p, tex_name);

    uint32_t tex_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                        tref->type_id, tref->global_id);
    uint32_t sampler_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                            sref->type_id, sref->global_id);

    uint32_t f32_t = ssir_type_f32(p->mod);
    int ssir_cc = pl_ssir_coord_count(dim);

    uint32_t f32_coords[4];
    for (int i = 0; i < ncoords && i < 4; i++) {
        SsirType *ct = ssir_get_type(p->mod, coord_type);
        if (ct && ct->kind != SSIR_TYPE_F32)
            f32_coords[i] = ssir_build_convert(p->mod, p->func_id, p->block_id,
                                                f32_t, all_coords[i]);
        else
            f32_coords[i] = all_coords[i];
    }

    uint32_t coord_vec;
    if (ssir_cc == 1) {
        coord_vec = f32_coords[0];
    } else {
        uint32_t vec_t = ssir_type_vec(p->mod, f32_t, ssir_cc);
        if (dim == SSIR_TEX_2D_ARRAY && ncoords >= 3) {
            uint32_t comps[3] = { f32_coords[1], f32_coords[2], f32_coords[0] };
            coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id, vec_t, comps, 3);
        } else if (dim == SSIR_TEX_1D_ARRAY && ncoords >= 2) {
            uint32_t comps[2] = { f32_coords[1], f32_coords[0] };
            coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id, vec_t, comps, 2);
        } else {
            coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                             vec_t, f32_coords, ssir_cc);
        }
    }

    uint32_t vec4_t = ssir_type_vec(p->mod, dst_type, 4);
    uint32_t result = 0;
    if (inst->mip_mode == PTX_MIP_LEVEL) {
        result = ssir_build_tex_sample_level(p->mod, p->func_id, p->block_id,
                                              vec4_t, tex_val, sampler_val,
                                              coord_vec, lod_val);
    } else if (inst->mip_mode == PTX_MIP_GRAD) {
        result = ssir_build_tex_sample_grad(p->mod, p->func_id, p->block_id,
                                             vec4_t, tex_val, sampler_val,
                                             coord_vec, ddx_val, ddy_val);
    } else {
        uint32_t lod0 = ssir_const_f32(p->mod, 0.0f);
        result = ssir_build_tex_sample_level(p->mod, p->func_id, p->block_id,
                                              vec4_t, tex_val, sampler_val,
                                              coord_vec, lod0);
    }

    if (inst->dst.kind == PTX_OPER_VEC) {
        for (int i = 0; i < inst->dst.vec_count && i < 4; i++) {
            uint32_t comp = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                                dst_type, result, i);
            pl_store_reg(p, inst->dst.regs[i], comp);
        }
    }
}

static void pl_lower_tld4(PtxLower *p, const PtxInst *inst) {
    int coord_count = 0;
    SsirTextureDim dim = pl_tex_dim(inst->tex_geom, &coord_count);
    uint32_t dst_type = pl_map_type(p, inst->type);
    uint32_t coord_type = pl_map_type(p, inst->type2);
    const char *tex_name = inst->src[0].name;

    uint32_t coords[4] = {0};
    int ncoords = inst->src_count - 1;
    for (int i = 0; i < ncoords && i < 4; i++)
        coords[i] = pl_resolve_operand(p, &inst->src[i + 1], coord_type);
    if (p->had_error) return;

    PlTexRef *tref = pl_find_texref(p, tex_name);
    if (!tref) {
        uint32_t f32_t = ssir_type_f32(p->mod);
        uint32_t tex_t = ssir_type_texture(p->mod, dim, f32_t);
        uint32_t ptr_t = ssir_type_ptr(p->mod, tex_t, SSIR_ADDR_UNIFORM_CONSTANT);
        uint32_t gid = ssir_global_var(p->mod, tex_name, ptr_t);
        ssir_global_set_group(p->mod, gid, 0);
        ssir_global_set_binding(p->mod, gid, p->next_binding++);
        pl_add_iface(p, gid);
        tref = pl_add_texref(p, tex_name, gid, tex_t, false, dim);
    }
    PlSamplerRef *sref = pl_get_implicit_sampler(p, tex_name);

    uint32_t tex_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                        tref->type_id, tref->global_id);
    uint32_t sampler_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                            sref->type_id, sref->global_id);

    uint32_t f32_t = ssir_type_f32(p->mod);
    uint32_t f32_coords[4];
    for (int i = 0; i < ncoords && i < 4; i++) {
        SsirType *ct = ssir_get_type(p->mod, coord_type);
        if (ct && ct->kind != SSIR_TYPE_F32)
            f32_coords[i] = ssir_build_convert(p->mod, p->func_id, p->block_id,
                                                f32_t, coords[i]);
        else
            f32_coords[i] = coords[i];
    }

    int ssir_cc = (dim == SSIR_TEX_2D) ? 2 : 3;
    uint32_t coord_vec;
    if (ssir_cc == 1) {
        coord_vec = f32_coords[0];
    } else {
        uint32_t vec_t = ssir_type_vec(p->mod, f32_t, ssir_cc);
        coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                         vec_t, f32_coords, ssir_cc);
    }

    uint32_t vec4_t = ssir_type_vec(p->mod, dst_type, 4);
    uint32_t comp_val = ssir_const_u32(p->mod, (uint32_t)inst->tex_gather_comp);
    uint32_t result = ssir_build_tex_gather(p->mod, p->func_id, p->block_id,
                                             vec4_t, tex_val, sampler_val,
                                             coord_vec, comp_val);

    if (inst->dst.kind == PTX_OPER_VEC) {
        for (int i = 0; i < inst->dst.vec_count && i < 4; i++) {
            uint32_t comp = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                                dst_type, result, i);
            pl_store_reg(p, inst->dst.regs[i], comp);
        }
    }
}

static void pl_lower_suld(PtxLower *p, const PtxInst *inst) {
    int coord_count = 0;
    SsirTextureDim dim = pl_tex_dim(inst->tex_geom, &coord_count);
    uint32_t elem_type = pl_map_type(p, inst->type);
    const char *surf_name = inst->src[0].name;

    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t coords[4] = {0};
    int ncoords = inst->src_count - 1;
    for (int i = 0; i < ncoords && i < 4; i++)
        coords[i] = pl_resolve_operand(p, &inst->src[i + 1], u32_t);
    if (p->had_error) return;

    PlTexRef *sref = pl_find_texref(p, surf_name);
    if (!sref) {
        uint32_t fmt = pl_surface_format_for_type(p, elem_type);
        uint32_t surf_t = ssir_type_texture_storage(p->mod, dim,
            fmt, SSIR_ACCESS_READ_WRITE);
        uint32_t ptr_t = ssir_type_ptr(p->mod, surf_t, SSIR_ADDR_UNIFORM_CONSTANT);
        uint32_t gid = ssir_global_var(p->mod, surf_name, ptr_t);
        ssir_global_set_group(p->mod, gid, 0);
        ssir_global_set_binding(p->mod, gid, p->next_binding++);
        pl_add_iface(p, gid);
        sref = pl_add_texref(p, surf_name, gid, surf_t, true, dim);
    }

    uint32_t surf_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                         sref->type_id, sref->global_id);

    int ssir_cc = 0;
    switch (dim) {
    case SSIR_TEX_1D: ssir_cc = 1; break;
    case SSIR_TEX_2D: ssir_cc = 2; break;
    case SSIR_TEX_3D: ssir_cc = 3; break;
    default:          ssir_cc = 2; break;
    }

    uint32_t coord_vec;
    if (ssir_cc == 1) {
        coord_vec = coords[0];
    } else {
        uint32_t vec_t = ssir_type_vec(p->mod, u32_t, ssir_cc);
        coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                         vec_t, coords, ssir_cc);
    }

    uint32_t vec4_t = ssir_type_vec(p->mod, elem_type, 4);
    uint32_t level0 = ssir_const_i32(p->mod, 0);
    uint32_t result = ssir_build_tex_load(p->mod, p->func_id, p->block_id,
                                           vec4_t, surf_val, coord_vec, level0);

    if (inst->dst.kind == PTX_OPER_VEC) {
        for (int i = 0; i < inst->dst.vec_count && i < 4; i++) {
            uint32_t comp = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                                elem_type, result, i);
            pl_store_reg(p, inst->dst.regs[i], comp);
        }
    }
}

static void pl_lower_sust(PtxLower *p, const PtxInst *inst) {
    int coord_count = 0;
    SsirTextureDim dim = pl_tex_dim(inst->tex_geom, &coord_count);
    uint32_t elem_type = pl_map_type(p, inst->type);
    /* Parser puts: src[0] = surface name, src[1..coord_count] = coords,
     * remaining = values */
    const char *surf_name = inst->src[0].name;

    uint32_t u32_t = ssir_type_u32(p->mod);
    int avail = inst->src_count - 1;
    int ncrd = coord_count < avail ? coord_count : avail;
    uint32_t coords[4] = {0};
    for (int i = 0; i < ncrd && i < 4; i++)
        coords[i] = pl_resolve_operand(p, &inst->src[1 + i], u32_t);

    int val_start = 1 + ncrd;
    int nsrc = inst->src_count - val_start;
    uint32_t src_vals[4] = {0};
    for (int i = 0; i < nsrc && i < 4; i++)
        src_vals[i] = pl_resolve_operand(p, &inst->src[val_start + i], elem_type);
    if (p->had_error) return;

    PlTexRef *sref = pl_find_texref(p, surf_name);
    if (!sref) {
        uint32_t fmt = pl_surface_format_for_type(p, elem_type);
        uint32_t surf_t = ssir_type_texture_storage(p->mod, dim,
            fmt, SSIR_ACCESS_READ_WRITE);
        uint32_t ptr_t = ssir_type_ptr(p->mod, surf_t, SSIR_ADDR_UNIFORM_CONSTANT);
        uint32_t gid = ssir_global_var(p->mod, surf_name, ptr_t);
        ssir_global_set_group(p->mod, gid, 0);
        ssir_global_set_binding(p->mod, gid, p->next_binding++);
        pl_add_iface(p, gid);
        sref = pl_add_texref(p, surf_name, gid, surf_t, true, dim);
    }

    uint32_t surf_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                         sref->type_id, sref->global_id);

    int ssir_cc = 0;
    switch (dim) {
    case SSIR_TEX_1D: ssir_cc = 1; break;
    case SSIR_TEX_2D: ssir_cc = 2; break;
    case SSIR_TEX_3D: ssir_cc = 3; break;
    default:          ssir_cc = 2; break;
    }

    uint32_t coord_vec;
    if (ssir_cc == 1) {
        coord_vec = coords[0];
    } else {
        uint32_t vec_t = ssir_type_vec(p->mod, u32_t, ssir_cc);
        coord_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                         vec_t, coords, ssir_cc);
    }

    uint32_t vec4_t = ssir_type_vec(p->mod, elem_type, 4);
    uint32_t value_vec;
    if (nsrc >= 4) {
        value_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                         vec4_t, src_vals, 4);
    } else {
        uint32_t zero = pl_const_for_type(p, elem_type, 0, 0.0, false);
        uint32_t padded[4];
        for (int i = 0; i < 4; i++)
            padded[i] = (i < nsrc) ? src_vals[i] : zero;
        value_vec = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                         vec4_t, padded, 4);
    }

    ssir_build_tex_store(p->mod, p->func_id, p->block_id,
                          surf_val, coord_vec, value_vec);
}

static void pl_lower_txq(PtxLower *p, const PtxInst *inst) {
    uint32_t res_type = pl_map_type(p, inst->type);
    const char *tex_name = inst->src[0].name;

    PlTexRef *tref = pl_find_texref(p, tex_name);
    if (!tref) {
        uint32_t f32_t = ssir_type_f32(p->mod);
        uint32_t tex_t = ssir_type_texture(p->mod, SSIR_TEX_2D, f32_t);
        uint32_t ptr_t = ssir_type_ptr(p->mod, tex_t, SSIR_ADDR_UNIFORM_CONSTANT);
        uint32_t gid = ssir_global_var(p->mod, tex_name, ptr_t);
        ssir_global_set_group(p->mod, gid, 0);
        ssir_global_set_binding(p->mod, gid, p->next_binding++);
        pl_add_iface(p, gid);
        tref = pl_add_texref(p, tex_name, gid, tex_t, false, SSIR_TEX_2D);
    }

    uint32_t tex_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                        tref->type_id, tref->global_id);
    uint32_t u32_t = ssir_type_u32(p->mod);

    /* The txq_query field is encoded in tex_gather_comp for simplicity:
     * 0=width, 1=height, 2=depth, 3=num_mipmap_levels */
    int query = inst->tex_gather_comp;
    if (query == 3) {
        uint32_t result = ssir_build_tex_query_levels(p->mod, p->func_id,
                                                       p->block_id, u32_t, tex_val);
        if (res_type != u32_t)
            result = ssir_build_convert(p->mod, p->func_id, p->block_id, res_type, result);
        pl_store_reg(p, inst->dst.name, result);
    } else {
        int size_components = 1;
        switch (tref->dim) {
        case SSIR_TEX_1D:       size_components = 1; break;
        case SSIR_TEX_2D:       size_components = 2; break;
        case SSIR_TEX_3D:       size_components = 3; break;
        case SSIR_TEX_CUBE:     size_components = 2; break;
        case SSIR_TEX_1D_ARRAY: size_components = 2; break;
        case SSIR_TEX_2D_ARRAY: size_components = 3; break;
        default:                size_components = 2; break;
        }
        uint32_t level0 = ssir_const_i32(p->mod, 0);
        uint32_t result;
        if (size_components == 1) {
            result = ssir_build_tex_size(p->mod, p->func_id, p->block_id,
                                          u32_t, tex_val, level0);
        } else {
            uint32_t vec_t = ssir_type_vec(p->mod, u32_t, size_components);
            uint32_t size_vec = ssir_build_tex_size(p->mod, p->func_id, p->block_id,
                                                     vec_t, tex_val, level0);
            result = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                         u32_t, size_vec, query);
        }
        if (res_type != u32_t)
            result = ssir_build_convert(p->mod, p->func_id, p->block_id, res_type, result);
        pl_store_reg(p, inst->dst.name, result);
    }
}

static void pl_lower_popc(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;
    uint32_t args[] = { src };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_COUNTBITS, args, 1);
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_brev(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;
    uint32_t args[] = { src };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_REVERSEBITS, args, 1);
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_clz(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], type);
    if (p->had_error) return;
    uint32_t args[] = { src };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_FIRSTLEADINGBIT, args, 1);
    bool need_invert = (inst->opcode == PTX_OP_CLZ) ||
                       (inst->opcode == PTX_OP_BFIND &&
                        (inst->modifiers & PTX_MOD_SHIFTAMT));
    if (need_invert) {
        SsirType *ty = ssir_get_type(p->mod, type);
        int bits = ty ? pl_scalar_bit_width(ty) : 32;
        uint32_t width = pl_const_for_type(p, type, (uint64_t)(bits - 1), 0.0, false);
        result = ssir_build_sub(p->mod, p->func_id, p->block_id, type, width, result);
    }
    if (inst->opcode == PTX_OP_CLZ) {
        uint32_t zero = pl_const_for_type(p, type, 0, 0.0, false);
        uint32_t bool_t = ssir_type_bool(p->mod);
        uint32_t is_zero = ssir_build_eq(p->mod, p->func_id, p->block_id,
                                          bool_t, src, zero);
        SsirType *ty = ssir_get_type(p->mod, type);
        int bits = ty ? pl_scalar_bit_width(ty) : 32;
        uint32_t full_width = pl_const_for_type(p, type, (uint64_t)bits, 0.0, false);
        uint32_t sel_args[] = { result, full_width, is_zero };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_SELECT, sel_args, 3);
    } else if (inst->opcode == PTX_OP_BFIND) {
        uint32_t zero = pl_const_for_type(p, type, 0, 0.0, false);
        uint32_t bool_t = ssir_type_bool(p->mod);
        uint32_t is_zero = ssir_build_eq(p->mod, p->func_id, p->block_id,
                                          bool_t, src, zero);
        uint32_t neg1 = pl_const_for_type(p, type, 0xFFFFFFFF, 0.0, false);
        uint32_t sel_args[] = { result, neg1, is_zero };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_SELECT, sel_args, 3);
    }
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

static void pl_lower_shf(PtxLower *p, const PtxInst *inst) {
    /* shf.{l|r}.{wrap|clamp}.b32 dst, lo, hi, shift
     * Use 64-bit concat + shift to avoid UB when shift==0 or shift==32.
     * Left:  result = u32((u64(hi) << 32 | u64(lo)) << (shift & 31)) >> 32)
     * Right: result = u32((u64(hi) << 32 | u64(lo)) >> (shift & 31)) */
    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t u64_t = ssir_type_u64(p->mod);
    uint32_t lo = pl_resolve_operand(p, &inst->src[0], u32_t);
    uint32_t hi = pl_resolve_operand(p, &inst->src[1], u32_t);
    uint32_t shift = pl_resolve_operand(p, &inst->src[2], u32_t);
    if (p->had_error) return;

    uint32_t lo64 = ssir_build_convert(p->mod, p->func_id, p->block_id, u64_t, lo);
    uint32_t hi64 = ssir_build_convert(p->mod, p->func_id, p->block_id, u64_t, hi);
    uint32_t c32_u32 = ssir_const_u32(p->mod, 32);
    uint32_t hi_shifted = ssir_build_shl(p->mod, p->func_id, p->block_id, u64_t, hi64, c32_u32);
    uint32_t concat = ssir_build_bit_or(p->mod, p->func_id, p->block_id, u64_t, hi_shifted, lo64);

    bool wrap = (inst->modifiers & PTX_MOD_WRAP) != 0;
    uint32_t amt = shift;
    if (wrap) {
        uint32_t c31 = ssir_const_u32(p->mod, 31);
        amt = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, shift, c31);
    }

    uint32_t shifted;
    if (inst->modifiers & PTX_MOD_LEFT) {
        shifted = ssir_build_shl(p->mod, p->func_id, p->block_id, u64_t, concat, amt);
        shifted = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, u64_t, shifted, c32_u32);
    } else {
        shifted = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, u64_t, concat, amt);
    }
    uint32_t result = ssir_build_convert(p->mod, p->func_id, p->block_id, u32_t, shifted);
    pl_store_reg_typed(p, inst->dst.name, result, u32_t);
}

static void pl_lower_bfi(PtxLower *p, const PtxInst *inst) {
    /* bfi.b32 dst, src, base, offset, width
     * dst = (base & ~mask) | ((src << offset) & mask)
     * where mask = ((1 << width) - 1) << offset */
    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t src = pl_resolve_operand(p, &inst->src[0], u32_t);
    uint32_t base = pl_resolve_operand(p, &inst->src[1], u32_t);
    uint32_t offset = pl_resolve_operand(p, &inst->src[2], u32_t);
    uint32_t width = pl_resolve_operand(p, &inst->src[3], u32_t);
    if (p->had_error) return;

    uint32_t c1 = ssir_const_u32(p->mod, 1);
    uint32_t shifted_1 = ssir_build_shl(p->mod, p->func_id, p->block_id, u32_t, c1, width);
    uint32_t width_mask = ssir_build_sub(p->mod, p->func_id, p->block_id, u32_t, shifted_1, c1);
    uint32_t mask = ssir_build_shl(p->mod, p->func_id, p->block_id, u32_t, width_mask, offset);
    uint32_t inv_mask = ssir_build_bit_not(p->mod, p->func_id, p->block_id, u32_t, mask);
    uint32_t base_cleared = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, base, inv_mask);
    uint32_t src_shifted = ssir_build_shl(p->mod, p->func_id, p->block_id, u32_t, src, offset);
    uint32_t src_masked = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, src_shifted, mask);
    uint32_t result = ssir_build_bit_or(p->mod, p->func_id, p->block_id, u32_t, base_cleared, src_masked);
    pl_store_reg_typed(p, inst->dst.name, result, u32_t);
}

static void pl_lower_prmt(PtxLower *p, const PtxInst *inst) {
    /* prmt.b32 dst, a, b, selector
     * Concatenates {b, a} as 8 bytes, then selects 4 bytes by selector nibbles.
     * Each nibble in selector (bits 0-3, 4-7, 8-11, 12-15) picks a byte index. */
    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t a = pl_resolve_operand(p, &inst->src[0], u32_t);
    uint32_t b = pl_resolve_operand(p, &inst->src[1], u32_t);
    uint32_t sel = pl_resolve_operand(p, &inst->src[2], u32_t);
    if (p->had_error) return;

    uint32_t c0xff = ssir_const_u32(p->mod, 0xFF);
    uint32_t c0xf = ssir_const_u32(p->mod, 0xF);
    uint32_t c8 = ssir_const_u32(p->mod, 8);
    uint32_t c4 = ssir_const_u32(p->mod, 4);
    uint32_t c0 = ssir_const_u32(p->mod, 0);

    uint32_t result = c0;
    for (int i = 0; i < 4; i++) {
        uint32_t nibble_shift = ssir_const_u32(p->mod, (uint32_t)(i * 4));
        uint32_t nibble = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, u32_t, sel, nibble_shift);
        nibble = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, nibble, c0xf);
        /* nibble < 4 -> byte from a, else byte from b (index - 4) */
        uint32_t from_a_idx = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, nibble, ssir_const_u32(p->mod, 3));
        uint32_t byte_shift = ssir_build_mul(p->mod, p->func_id, p->block_id, u32_t, from_a_idx, c8);
        uint32_t byte_a = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, u32_t, a, byte_shift);
        byte_a = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, byte_a, c0xff);
        uint32_t byte_b = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, u32_t, b, byte_shift);
        byte_b = ssir_build_bit_and(p->mod, p->func_id, p->block_id, u32_t, byte_b, c0xff);
        uint32_t bool_t = ssir_type_bool(p->mod);
        uint32_t use_b = ssir_build_ge(p->mod, p->func_id, p->block_id, bool_t, nibble, c4);
        uint32_t sel_args[] = { use_b, byte_b, byte_a };
        uint32_t byte_val = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                                u32_t, SSIR_BUILTIN_SELECT, sel_args, 3);
        uint32_t out_shift = ssir_const_u32(p->mod, (uint32_t)(i * 8));
        uint32_t shifted = ssir_build_shl(p->mod, p->func_id, p->block_id, u32_t, byte_val, out_shift);
        result = ssir_build_bit_or(p->mod, p->func_id, p->block_id, u32_t, result, shifted);
    }
    pl_store_reg_typed(p, inst->dst.name, result, u32_t);
}

static void pl_lower_copysign(PtxLower *p, const PtxInst *inst) {
    uint32_t type = pl_map_type(p, inst->type);
    uint32_t sgn = pl_resolve_operand(p, &inst->src[0], type);
    uint32_t mag = pl_resolve_operand(p, &inst->src[1], type);
    if (p->had_error) return;
    uint32_t u32_type = ssir_type_u32(p->mod);
    uint32_t mag_bits = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                            u32_type, mag);
    uint32_t sgn_bits = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                            u32_type, sgn);
    uint32_t mag_mask = ssir_const_u32(p->mod, 0x7FFFFFFF);
    uint32_t sgn_mask = ssir_const_u32(p->mod, 0x80000000);
    uint32_t mag_cleared = ssir_build_bit_and(p->mod, p->func_id, p->block_id,
                                               u32_type, mag_bits, mag_mask);
    uint32_t sgn_bit = ssir_build_bit_and(p->mod, p->func_id, p->block_id,
                                           u32_type, sgn_bits, sgn_mask);
    uint32_t combined = ssir_build_bit_or(p->mod, p->func_id, p->block_id,
                                           u32_type, mag_cleared, sgn_bit);
    uint32_t result = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                          type, combined);
    pl_store_reg_typed(p, inst->dst.name, result, type);
}

/* ===== Instruction Dispatch ===== */

static void pl_lower_inst(PtxLower *p, const PtxInst *inst) {
    if (p->had_error) return;

    /* Handle predication */
    uint32_t pred_val = 0;
    bool has_pred = inst->has_pred;
    bool pred_negated = inst->pred_negated;
    if (has_pred) {
        PlReg *pr = pl_find_reg(p, inst->pred);
        if (pr) {
            pred_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                        pr->val_type, pr->ptr_id);
        } else {
            pl_error(p, "undefined predicate register '%s'", inst->pred);
            return;
        }
    }

    /* Branch needs special predicate handling */
    if (inst->opcode == PTX_OP_BRA) {
        pl_lower_bra(p, inst, pred_val, has_pred, pred_negated);
        return;
    }

    /* For other predicated instructions, wrap in if-then using selection merge */
    uint32_t saved_block = 0;
    uint32_t merge_block = 0;
    uint32_t then_block = 0;
    if (has_pred) {
        if (pred_negated)
            pred_val = ssir_build_not(p->mod, p->func_id, p->block_id,
                                      ssir_type_bool(p->mod), pred_val);
        then_block = ssir_block_create(p->mod, p->func_id, NULL);
        merge_block = ssir_block_create(p->mod, p->func_id, NULL);
        ssir_build_branch_cond_merge(p->mod, p->func_id, p->block_id,
                               pred_val, then_block, merge_block, merge_block);
        saved_block = p->block_id;
        p->block_id = then_block;
        p->block_from_label = false;
    }

    switch (inst->opcode) {
    case PTX_OP_ADD: case PTX_OP_SUB: case PTX_OP_MUL: case PTX_OP_MUL24:
    case PTX_OP_DIV: case PTX_OP_REM:
        pl_lower_arith(p, inst); break;
    case PTX_OP_NEG: case PTX_OP_ABS: case PTX_OP_NOT: case PTX_OP_CNOT:
        pl_lower_unary(p, inst); break;
    case PTX_OP_MIN: case PTX_OP_MAX:
        pl_lower_minmax(p, inst); break;
    case PTX_OP_MAD:
        pl_lower_mad(p, inst); break;
    case PTX_OP_FMA:
        pl_lower_fma(p, inst); break;
    case PTX_OP_AND: case PTX_OP_OR: case PTX_OP_XOR:
        pl_lower_bitwise(p, inst); break;
    case PTX_OP_SHL: case PTX_OP_SHR:
        pl_lower_shift(p, inst); break;
    case PTX_OP_SETP: case PTX_OP_SET:
        pl_lower_setp(p, inst); break;
    case PTX_OP_SELP:
        pl_lower_selp(p, inst); break;
    case PTX_OP_MOV:
        pl_lower_mov(p, inst); break;
    case PTX_OP_LD:
        pl_lower_ld(p, inst); break;
    case PTX_OP_ST:
        pl_lower_st(p, inst); break;
    case PTX_OP_CVT:
        pl_lower_cvt(p, inst); break;
    case PTX_OP_CVTA:
        pl_lower_cvta(p, inst); break;
    case PTX_OP_RET: case PTX_OP_EXIT:
        pl_lower_ret(p, inst); break;
    case PTX_OP_CALL:
        pl_lower_call(p, inst); break;
    case PTX_OP_BAR:
        pl_lower_bar(p, inst); break;
    case PTX_OP_MEMBAR:
        pl_lower_membar(p, inst); break;
    case PTX_OP_ATOM:
        pl_lower_atom(p, inst); break;
    case PTX_OP_RCP: case PTX_OP_SQRT: case PTX_OP_RSQRT:
    case PTX_OP_SIN: case PTX_OP_COS: case PTX_OP_LG2: case PTX_OP_EX2:
        pl_lower_math_unary(p, inst); break;
    case PTX_OP_TEX:
        pl_lower_tex(p, inst); break;
    case PTX_OP_TLD4:
        pl_lower_tld4(p, inst); break;
    case PTX_OP_SULD:
        pl_lower_suld(p, inst); break;
    case PTX_OP_SUST:
        pl_lower_sust(p, inst); break;
    case PTX_OP_TXQ: case PTX_OP_SUQ:
        pl_lower_txq(p, inst); break;
    case PTX_OP_POPC:
        pl_lower_popc(p, inst); break;
    case PTX_OP_BREV:
        pl_lower_brev(p, inst); break;
    case PTX_OP_CLZ: case PTX_OP_BFIND:
        pl_lower_clz(p, inst); break;
    case PTX_OP_COPYSIGN:
        pl_lower_copysign(p, inst); break;
    case PTX_OP_SHF:
        pl_lower_shf(p, inst); break;
    case PTX_OP_BFI:
        pl_lower_bfi(p, inst); break;
    case PTX_OP_PRMT:
        pl_lower_prmt(p, inst); break;
    default:
        break;
    }

    if (has_pred && merge_block) {
        ssir_build_branch(p->mod, p->func_id, p->block_id, merge_block);
        p->block_id = merge_block;
        p->block_from_label = false;
        (void)saved_block;
    }
}

/* ===== Label Handling ===== */

static void pl_handle_label(PtxLower *p, const char *name) {
    uint32_t lbl_block = pl_get_or_create_label(p, name);

    typedef struct { uint32_t merge_block; uint32_t target_label; bool has_inner_merge; } PopEntry;
    PopEntry popped[64];
    int popped_count = 0;
    int write = 0;
    for (int i = 0; i < p->construct_depth; i++) {
        if (p->construct_stack[i].merge_block == lbl_block ||
            p->construct_stack[i].target_label == lbl_block) {
            if (popped_count < 64) {
                popped[popped_count].merge_block = p->construct_stack[i].merge_block;
                popped[popped_count].target_label = p->construct_stack[i].target_label;
                popped[popped_count].has_inner_merge = p->construct_stack[i].has_inner_merge;
                popped_count++;
            }
        } else {
            p->construct_stack[write++] = p->construct_stack[i];
        }
    }
    p->construct_depth = write;

    for (int i = popped_count - 1; i >= 0; i--) {
        uint32_t mb = popped[i].merge_block;
        if (mb != lbl_block && popped[i].has_inner_merge)
            ssir_block_create_with_id(p->mod, p->func_id, mb, NULL);
    }
    uint32_t bridge_target = lbl_block;
    for (int i = 0; i < popped_count; i++) {
        uint32_t mb = popped[i].merge_block;
        if (mb != lbl_block) {
            ssir_build_branch(p->mod, p->func_id, mb, bridge_target);
            bridge_target = mb;
        }
    }

    if (p->block_from_label && bridge_target != lbl_block) {
        uint32_t target = lbl_block;
        for (int i = popped_count - 1; i >= 0; i--) {
            if (popped[i].merge_block != lbl_block && popped[i].has_inner_merge) {
                target = popped[i].merge_block;
                break;
            }
        }
        ssir_build_branch(p->mod, p->func_id, p->block_id, target);
    } else {
        ssir_build_branch(p->mod, p->func_id, p->block_id, bridge_target);
    }

    ssir_block_create_with_id(p->mod, p->func_id, lbl_block, name);
    p->block_id = lbl_block;
    p->block_from_label = true;

    for (int i = 0; i < p->label_count; i++) {
        if (p->labels[i].block_id == lbl_block)
            p->labels[i].defined = true;
    }
}

/* ===== Function Body Lowering ===== */

static void pl_lower_body(PtxLower *p, const PtxStmt *body, int body_count) {
    for (int i = 0; i < body_count && !p->had_error; i++) {
        const PtxStmt *stmt = &body[i];
        if (stmt->kind == PTX_STMT_LABEL) {
            pl_handle_label(p, stmt->label);
        } else {
            pl_lower_inst(p, &stmt->inst);
        }
    }

    SsirBlock *blk = ssir_get_block(p->mod, p->func_id, p->block_id);
    if (blk && blk->inst_count == 0) {
        ssir_build_return_void(p->mod, p->func_id, p->block_id);
    } else if (blk && blk->inst_count > 0) {
        SsirInst *last = &blk->insts[blk->inst_count - 1];
        if (last->op != SSIR_OP_BRANCH && last->op != SSIR_OP_BRANCH_COND &&
            last->op != SSIR_OP_RETURN && last->op != SSIR_OP_RETURN_VOID &&
            last->op != SSIR_OP_UNREACHABLE && last->op != SSIR_OP_SWITCH) {
            ssir_build_return_void(p->mod, p->func_id, p->block_id);
        }
    }
}

/* ===== BDA Param List ===== */

static void pl_build_bda_params(PtxLower *p, const PtxParam *params,
                                 int param_count) {
    typedef struct { char name[80]; uint32_t type; bool is_ptr; } BdaP;
    BdaP bda_params[64];
    int bda_count = 0;

    for (int i = 0; i < param_count; i++) {
        uint32_t param_type = pl_map_type(p, params[i].type);
        SsirType *ty = ssir_get_type(p->mod, param_type);
        bool is_pointer = ty && (ty->kind == SSIR_TYPE_U64 || ty->kind == SSIR_TYPE_I64);

        uint32_t loc_ptr_type = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_FUNCTION);
        uint32_t loc_id = ssir_function_add_local(p->mod, p->func_id,
            params[i].name, loc_ptr_type);

        if (p->reg_count >= p->reg_cap) {
            p->reg_cap = p->reg_cap ? p->reg_cap * 2 : 64;
            p->regs = (PlReg *)PTX_REALLOC(p->regs, p->reg_cap * sizeof(PlReg));
        }
        PlReg *r = &p->regs[p->reg_count++];
        memset(r, 0, sizeof(*r));
        snprintf(r->name, sizeof(r->name), "%s", params[i].name);
        r->val_type = param_type;
        r->ptr_type = loc_ptr_type;
        r->ptr_id = loc_id;
        r->pending_binding = UINT32_MAX;
        r->is_bda_ptr = is_pointer;

        if (bda_count < 64) {
            snprintf(bda_params[bda_count].name, 80, "%s", params[i].name);
            bda_params[bda_count].type = param_type;
            bda_params[bda_count].is_ptr = is_pointer;
            bda_count++;
        }
    }

    if (bda_count == 0) return;

    uint32_t member_types[67];
    uint32_t offsets[67];
    const char *member_names[67];
    uint32_t current_offset = 0;

    for (int i = 0; i < bda_count; i++) {
        member_types[i] = bda_params[i].type;
        member_names[i] = bda_params[i].name;
        uint32_t sz = pl_type_byte_size(p, bda_params[i].type);
        uint32_t align = sz;
        current_offset = (current_offset + align - 1) & ~(align - 1);
        offsets[i] = current_offset;
        current_offset += sz;
    }

    uint32_t u32_type = ssir_type_u32(p->mod);
    int total_members = bda_count + 3;
    static const char *ntid_names[] = { "__ntid_x", "__ntid_y", "__ntid_z" };
    for (int k = 0; k < 3; k++) {
        int idx = bda_count + k;
        member_types[idx] = u32_type;
        member_names[idx] = ntid_names[k];
        current_offset = (current_offset + 3) & ~3u;
        offsets[idx] = current_offset;
        current_offset += 4;
    }
    p->bda_param_count = (uint32_t)bda_count;

    uint32_t struct_type = ssir_type_struct_named(p->mod, "KernelParams",
        member_types, (uint32_t)total_members, offsets, member_names);
    uint32_t pc_ptr_type = ssir_type_ptr(p->mod, struct_type, SSIR_ADDR_PUSH_CONSTANT);
    uint32_t pc_global = ssir_global_var(p->mod, "kernel_params", pc_ptr_type);
    p->bda_pc_global = pc_global;
    pl_add_iface(p, pc_global);

    for (int i = 0; i < bda_count; i++) {
        PlReg *r = pl_find_reg(p, bda_params[i].name);
        if (!r) continue;
        uint32_t member_ptr_type = ssir_type_ptr(p->mod, bda_params[i].type,
                                                  SSIR_ADDR_PUSH_CONSTANT);
        uint32_t idx = ssir_const_u32(p->mod, (uint32_t)i);
        uint32_t indices[] = { idx };
        uint32_t member_ptr = ssir_build_access(p->mod, p->func_id, p->block_id,
            member_ptr_type, pc_global, indices, 1);
        uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                       bda_params[i].type, member_ptr);
        ssir_build_store(p->mod, p->func_id, p->block_id, r->ptr_id, val);
    }
}

/* ===== Descriptor Mode Params ===== */

static void pl_build_descriptor_params(PtxLower *p, const PtxParam *params,
                                        int param_count) {
    for (int i = 0; i < param_count; i++) {
        uint32_t param_type = pl_map_type(p, params[i].type);
        SsirType *ty = ssir_get_type(p->mod, param_type);
        bool is_pointer = ty && (ty->kind == SSIR_TYPE_U64 || ty->kind == SSIR_TYPE_I64);

        uint32_t loc_ptr_type = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_FUNCTION);
        uint32_t loc_id = ssir_function_add_local(p->mod, p->func_id,
            params[i].name, loc_ptr_type);

        uint32_t init_val = pl_const_for_type(p, param_type, 0, 0.0, false);
        ssir_build_store(p->mod, p->func_id, p->block_id, loc_id, init_val);

        if (p->reg_count >= p->reg_cap) {
            p->reg_cap = p->reg_cap ? p->reg_cap * 2 : 64;
            p->regs = (PlReg *)PTX_REALLOC(p->regs, p->reg_cap * sizeof(PlReg));
        }
        PlReg *r = &p->regs[p->reg_count++];
        memset(r, 0, sizeof(*r));
        snprintf(r->name, sizeof(r->name), "%s", params[i].name);
        r->val_type = param_type;
        r->ptr_type = loc_ptr_type;
        r->ptr_id = loc_id;
        r->global_id = 0;
        r->pending_binding = is_pointer ? p->next_binding++ : UINT32_MAX;
    }
}

/* ===== Global Variable Lowering ===== */

static void pl_lower_globals(PtxLower *p, const PtxModule *mod) {
    for (int i = 0; i < mod->global_count; i++) {
        const PtxGlobalDecl *g = &mod->globals[i];
        uint32_t elem_type = pl_map_type(p, g->type);
        SsirAddressSpace addr_space = SSIR_ADDR_STORAGE;
        if (g->space == PTX_SPACE_SHARED) addr_space = SSIR_ADDR_WORKGROUP;
        else if (g->space == PTX_SPACE_CONST) addr_space = SSIR_ADDR_UNIFORM;
        if (g->has_init) addr_space = SSIR_ADDR_PRIVATE;

        uint32_t arr_count = g->array_size;
        bool repack_as_u32 = false;
        if (g->has_init && g->type == PTX_TYPE_B8 && arr_count > 0 &&
            (arr_count % 4) == 0) {
            elem_type = ssir_type_u32(p->mod);
            arr_count /= 4;
            repack_as_u32 = true;
        }

        uint32_t var_type = arr_count > 0
            ? ssir_type_array(p->mod, elem_type, arr_count)
            : elem_type;
        uint32_t ptr_type = ssir_type_ptr(p->mod, var_type, addr_space);
        uint32_t gid = ssir_global_var(p->mod, g->name, ptr_type);

        if (g->has_init && g->init_count > 0) {
            if (repack_as_u32) {
                uint32_t n_u32 = arr_count;
                uint32_t *comps = (uint32_t *)malloc(n_u32 * sizeof(uint32_t));
                for (uint32_t j = 0; j < n_u32; j++) {
                    uint32_t word = 0;
                    for (int b = 0; b < 4; b++) {
                        uint32_t bi = j * 4 + b;
                        uint8_t byte = bi < (uint32_t)g->init_count
                            ? (uint8_t)g->init_vals[bi] : 0;
                        word |= (uint32_t)byte << (b * 8);
                    }
                    comps[j] = ssir_const_u32(p->mod, word);
                }
                uint32_t arr_type = ssir_type_array(p->mod, elem_type, n_u32);
                uint32_t init_id = ssir_const_composite(p->mod, arr_type,
                    comps, n_u32);
                ssir_global_set_initializer(p->mod, gid, init_id);
                free(comps);
            } else {
                uint32_t n = (uint32_t)g->init_count;
                uint32_t *comps = (uint32_t *)malloc(n * sizeof(uint32_t));
                for (uint32_t j = 0; j < n; j++)
                    comps[j] = pl_const_for_type(p, elem_type,
                        g->init_vals[j], 0.0, false);
                uint32_t init_id = ssir_const_composite(p->mod, var_type,
                    comps, n);
                ssir_global_set_initializer(p->mod, gid, init_id);
                free(comps);
            }
        }

        pl_add_iface(p, gid);
    }
}

/* ===== Ref Decl Lowering ===== */

static void pl_lower_refs(PtxLower *p, const PtxModule *mod) {
    for (int i = 0; i < mod->ref_count; i++) {
        const PtxRefDecl *ref = &mod->refs[i];
        if (ref->kind == PTX_REF_TEXREF) {
            uint32_t f32_t = ssir_type_f32(p->mod);
            uint32_t tex_t = ssir_type_texture(p->mod, SSIR_TEX_2D, f32_t);
            uint32_t ptr_t = ssir_type_ptr(p->mod, tex_t, SSIR_ADDR_UNIFORM_CONSTANT);
            uint32_t gid = ssir_global_var(p->mod, ref->name, ptr_t);
            ssir_global_set_group(p->mod, gid, 0);
            ssir_global_set_binding(p->mod, gid, p->next_binding++);
            pl_add_iface(p, gid);
            pl_add_texref(p, ref->name, gid, tex_t, false, SSIR_TEX_2D);
        } else if (ref->kind == PTX_REF_SAMPLERREF) {
            uint32_t sampler_t = ssir_type_sampler(p->mod);
            uint32_t ptr_t = ssir_type_ptr(p->mod, sampler_t, SSIR_ADDR_UNIFORM_CONSTANT);
            uint32_t gid = ssir_global_var(p->mod, ref->name, ptr_t);
            ssir_global_set_group(p->mod, gid, 0);
            ssir_global_set_binding(p->mod, gid, p->next_binding++);
            pl_add_iface(p, gid);
            pl_add_sampler(p, ref->name, gid, sampler_t);
        } else if (ref->kind == PTX_REF_SURFREF) {
            uint32_t surf_t = ssir_type_texture_storage(p->mod, SSIR_TEX_2D,
                30, SSIR_ACCESS_READ_WRITE);
            uint32_t ptr_t = ssir_type_ptr(p->mod, surf_t, SSIR_ADDR_UNIFORM_CONSTANT);
            uint32_t gid = ssir_global_var(p->mod, ref->name, ptr_t);
            ssir_global_set_group(p->mod, gid, 0);
            ssir_global_set_binding(p->mod, gid, p->next_binding++);
            pl_add_iface(p, gid);
            pl_add_texref(p, ref->name, gid, surf_t, true, SSIR_TEX_2D);
        }
    }
    p->module_binding_base = p->next_binding;
}

/* ===== Entry Point Lowering ===== */

static void pl_lower_entry(PtxLower *p, const PtxEntry *entry) {
    /* Reset per-function state */
    p->reg_count = 0;
    p->label_count = 0;
    p->iface_count = 0;
    p->merge_block_count = 0;
    p->construct_depth = 0;
    p->precomputed_merge_count = 0;
    p->is_entry = true;
    p->next_binding = p->module_binding_base;
    p->wg_size[0] = entry->wg_size[0];
    p->wg_size[1] = entry->wg_size[1];
    p->wg_size[2] = entry->wg_size[2];
    p->bda_pc_global = 0;

    for (int i = 0; i < p->texref_count; i++)
        pl_add_iface(p, p->texrefs[i].global_id);
    for (int i = 0; i < p->sampler_count; i++)
        pl_add_iface(p, p->samplers[i].global_id);

    uint32_t void_t = ssir_type_void(p->mod);
    p->func_id = ssir_function_create(p->mod, entry->name, void_t);
    p->block_id = ssir_block_create(p->mod, p->func_id, "entry");
    pl_register_func(p, entry->name, p->func_id, void_t);

    /* Declare registers */
    pl_declare_regs(p, entry->reg_decls, entry->reg_decl_count);

    for (int i = 0; i < entry->local_decl_count; i++) {
        const PtxLocalDecl *ld = &entry->local_decls[i];
        uint32_t elem_type = pl_map_type(p, ld->type);
        uint32_t arr_count = ld->array_size;
        if (ld->type == PTX_TYPE_B8 && arr_count > 0 && (arr_count % 4) == 0) {
            elem_type = ssir_type_u32(p->mod);
            arr_count /= 4;
        }
        uint32_t var_type = arr_count > 0
            ? ssir_type_array(p->mod, elem_type, arr_count)
            : elem_type;
        uint32_t ptr_type = ssir_type_ptr(p->mod, var_type, SSIR_ADDR_FUNCTION);
        uint32_t loc_id = ssir_function_add_local(p->mod, p->func_id,
            ld->name, ptr_type);
        PlReg *r = pl_add_reg(p, ld->name, var_type, false);
        r->ptr_id = loc_id;
        r->ptr_type = ptr_type;
        r->val_type = var_type;
    }

    if (p->use_bda)
        pl_build_bda_params(p, entry->params, entry->param_count);
    else
        pl_build_descriptor_params(p, entry->params, entry->param_count);

    /* Pre-compute merge blocks for if-else patterns */
    pl_precompute_merges(p, entry->body, entry->body_count);

    /* Lower body */
    pl_lower_body(p, entry->body, entry->body_count);

    /* Materialize remaining buffers (descriptor mode only) */
    if (!p->use_bda) {
        for (int i = 0; i < p->reg_count; i++) {
            PlReg *r = &p->regs[i];
            if (r->pending_binding != UINT32_MAX && r->global_id == 0) {
                uint32_t default_elem = ssir_type_u32(p->mod);
                pl_materialize_buffer(p, r, default_elem);
            }
        }
    }

    /* Create entry point */
    p->ep_index = ssir_entry_point_create(p->mod, SSIR_STAGE_COMPUTE,
                                           p->func_id, entry->name);
    for (int i = 0; i < p->iface_count; i++)
        ssir_entry_point_add_interface(p->mod, p->ep_index, p->iface[i]);

    if (p->wg_size[0] > 0)
        ssir_entry_point_set_workgroup_size(p->mod, p->ep_index,
            p->wg_size[0],
            p->wg_size[1] > 0 ? p->wg_size[1] : 1,
            p->wg_size[2] > 0 ? p->wg_size[2] : 1);
    else
        ssir_entry_point_set_workgroup_size(p->mod, p->ep_index, 1, 1, 1);
}

/* ===== Device Function Lowering ===== */

static void pl_lower_func(PtxLower *p, const PtxFunc *func) {
    p->reg_count = 0;
    p->label_count = 0;
    p->merge_block_count = 0;
    p->construct_depth = 0;
    p->precomputed_merge_count = 0;
    p->is_entry = false;

    uint32_t ret_type = func->has_return
        ? pl_map_type(p, func->return_type) : ssir_type_void(p->mod);

    p->func_id = ssir_function_create(p->mod, func->name, ret_type);
    p->block_id = ssir_block_create(p->mod, p->func_id, "entry");
    pl_register_func(p, func->name, p->func_id, ret_type);

    /* Add function params */
    for (int i = 0; i < func->param_count; i++) {
        uint32_t param_type = pl_map_type(p, func->params[i].type);
        ssir_function_add_param(p->mod, p->func_id,
                                func->params[i].name, param_type);
        pl_add_reg(p, func->params[i].name, param_type, false);
    }

    /* Declare registers */
    pl_declare_regs(p, func->reg_decls, func->reg_decl_count);

    for (int i = 0; i < func->local_decl_count; i++) {
        const PtxLocalDecl *ld = &func->local_decls[i];
        uint32_t elem_type = pl_map_type(p, ld->type);
        uint32_t arr_count = ld->array_size;
        if (ld->type == PTX_TYPE_B8 && arr_count > 0 && (arr_count % 4) == 0) {
            elem_type = ssir_type_u32(p->mod);
            arr_count /= 4;
        }
        uint32_t var_type = arr_count > 0
            ? ssir_type_array(p->mod, elem_type, arr_count)
            : elem_type;
        uint32_t ptr_type = ssir_type_ptr(p->mod, var_type, SSIR_ADDR_FUNCTION);
        uint32_t loc_id = ssir_function_add_local(p->mod, p->func_id,
            ld->name, ptr_type);
        PlReg *r = pl_add_reg(p, ld->name, var_type, false);
        r->ptr_id = loc_id;
        r->ptr_type = ptr_type;
        r->val_type = var_type;
    }

    if (func->is_decl_only) {
        ssir_build_return_void(p->mod, p->func_id, p->block_id);
    } else {
        pl_precompute_merges(p, func->body, func->body_count);
        pl_lower_body(p, func->body, func->body_count);
    }
}

/* ===== Cleanup ===== */

static void pl_cleanup(PtxLower *p) {
    PTX_FREE(p->regs);
    PTX_FREE(p->labels);
    PTX_FREE(p->iface);
    PTX_FREE(p->funcs);
    PTX_FREE(p->texrefs);
    PTX_FREE(p->samplers);
    PTX_FREE(p->merge_blocks);
    PTX_FREE(p->construct_stack);
    PTX_FREE(p->precomputed_merges);
}

/* ===== Public API ===== */

PtxToSsirResult ptx_lower(const PtxModule *mod, const PtxToSsirOptions *opts,
    SsirModule **out_module, char **out_error) {
    if (!mod || !out_module) {
        if (out_error) *out_error = ptx_strdup("Invalid input: null module or output pointer");
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    PtxLower lower;
    memset(&lower, 0, sizeof(lower));
    if (opts) {
        lower.opts = *opts;
        lower.use_bda = opts->use_bda != 0;
    }

    lower.mod = ssir_module_create();
    if (!lower.mod) {
        if (out_error) *out_error = ptx_strdup("Out of memory");
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    /* Lower globals and ref decls first */
    pl_lower_globals(&lower, mod);
    pl_lower_refs(&lower, mod);

    /* Lower device functions first so they can be called */
    for (int i = 0; i < mod->func_count; i++) {
        if (!lower.had_error)
            pl_lower_func(&lower, &mod->functions[i]);
    }

    /* Lower entry points */
    for (int i = 0; i < mod->entry_count; i++) {
        if (!lower.had_error)
            pl_lower_entry(&lower, &mod->entries[i]);
    }

    if (lower.had_error) {
        if (out_error) *out_error = ptx_strdup(lower.error);
        ssir_module_destroy(lower.mod);
        pl_cleanup(&lower);
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    *out_module = lower.mod;
    pl_cleanup(&lower);
    return PTX_TO_SSIR_OK;
}

PtxToSsirResult ptx_to_ssir(const char *ptx_source, const PtxToSsirOptions *opts,
    SsirModule **out_module, char **out_error) {
    if (!ptx_source || !out_module) {
        if (out_error) *out_error = ptx_strdup("Invalid input: null source or output pointer");
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    char *parse_error = NULL;
    PtxModule *ast = ptx_parse(ptx_source, &parse_error);
    if (!ast) {
        if (out_error) *out_error = parse_error;
        else free(parse_error);
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    PtxToSsirResult result = ptx_lower(ast, opts, out_module, out_error);
    ptx_parse_free(ast);
    return result;
}

void ptx_to_ssir_free(char *str) {
    PTX_FREE(str);
}

const char *ptx_to_ssir_result_string(PtxToSsirResult r) {
    switch (r) {
    case PTX_TO_SSIR_OK:          return "Success";
    case PTX_TO_SSIR_PARSE_ERROR: return "Parse error";
    case PTX_TO_SSIR_UNSUPPORTED: return "Unsupported feature";
    default:                      return "Unknown error";
    }
}
