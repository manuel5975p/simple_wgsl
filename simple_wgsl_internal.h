/*
 * simple_wgsl_internal.h - Shared infrastructure for simple_wgsl backends
 *
 * Internal header. Not part of the public API.
 * All functions static inline to preserve single-TU compilation model.
 */

#ifndef SIMPLE_WGSL_INTERNAL_H
#define SIMPLE_WGSL_INTERNAL_H

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ============================================================================
 * SW_GROW - Dynamic array growth macro
 * ============================================================================ */

#define SW_GROW(ptr, count, cap, T, realloc_fn) do { \
    if ((count) >= (cap)) { \
        uint32_t _nc = (cap) ? (cap) * 2 : 8; \
        while (_nc < (count)) _nc *= 2; \
        (ptr) = (T *)(realloc_fn)((ptr), _nc * sizeof(T)); \
        (cap) = _nc; \
    } \
} while(0)

/* ============================================================================
 * SwStringBuffer - Growable text buffer (replaces 4 backend copies)
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int indent;
} SwStringBuffer;

static inline void sw_sb_init(SwStringBuffer *sb) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->indent = 0;
}

static inline void sw_sb_free_with(SwStringBuffer *sb, void (*free_fn)(void *)) {
    free_fn(sb->data);
    sb->data = NULL;
    sb->len = sb->cap = 0;
}

static inline void sw_sb_free(SwStringBuffer *sb) {
    sw_sb_free_with(sb, free);
}

static inline int sw_sb_reserve_with(SwStringBuffer *sb, size_t need,
                                     void *(*realloc_fn)(void *, size_t)) {
    if (sb->len + need + 1 <= sb->cap) return 1;
    size_t ncap = sb->cap ? sb->cap : 256;
    while (ncap < sb->len + need + 1) ncap *= 2;
    char *nd = (char *)realloc_fn(sb->data, ncap);
    if (!nd) return 0;
    sb->data = nd;
    sb->cap = ncap;
    return 1;
}

static inline int sw_sb_reserve(SwStringBuffer *sb, size_t need) {
    return sw_sb_reserve_with(sb, need, realloc);
}

static inline void sw_sb_append_with(SwStringBuffer *sb, const char *s,
                                     void *(*realloc_fn)(void *, size_t)) {
    size_t sl = strlen(s);
    if (!sw_sb_reserve_with(sb, sl, realloc_fn)) return;
    memcpy(sb->data + sb->len, s, sl);
    sb->len += sl;
    sb->data[sb->len] = '\0';
}

static inline void sw_sb_append(SwStringBuffer *sb, const char *s) {
    sw_sb_append_with(sb, s, realloc);
}

static inline void sw_sb_appendf_with(SwStringBuffer *sb,
                                      void *(*realloc_fn)(void *, size_t),
                                      const char *fmt, ...) {
    char buf[1024];
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    if (n > 0) sw_sb_append_with(sb, buf, realloc_fn);
}

static inline void sw_sb_appendf(SwStringBuffer *sb, const char *fmt, ...) {
    char buf[1024];
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    if (n > 0) sw_sb_append(sb, buf);
}

static inline void sw_sb_indent_with(SwStringBuffer *sb,
                                     void *(*realloc_fn)(void *, size_t)) {
    for (int i = 0; i < sb->indent; i++)
        sw_sb_append_with(sb, "    ", realloc_fn);
}

static inline void sw_sb_indent(SwStringBuffer *sb) {
    sw_sb_indent_with(sb, realloc);
}

static inline void sw_sb_nl_with(SwStringBuffer *sb,
                                 void *(*realloc_fn)(void *, size_t)) {
    sw_sb_append_with(sb, "\n", realloc_fn);
}

static inline void sw_sb_nl(SwStringBuffer *sb) {
    sw_sb_append(sb, "\n");
}

/* ============================================================================
 * Shared text-backend context helpers
 * ============================================================================ */

static inline SsirFunctionParam *sw_find_param(const SsirFunction *fn,
                                               uint32_t id) {
    if (!fn) return NULL;
    for (uint32_t i = 0; i < fn->param_count; i++) {
        if (fn->params[i].id == id) return &fn->params[i];
    }
    return NULL;
}

static inline SsirLocalVar *sw_find_local(const SsirFunction *fn,
                                          uint32_t id) {
    if (!fn) return NULL;
    for (uint32_t i = 0; i < fn->local_count; i++) {
        if (fn->locals[i].id == id) return &fn->locals[i];
    }
    return NULL;
}

static inline SsirInst *sw_find_inst(SsirInst **inst_map, uint32_t cap,
                                     uint32_t id) {
    if (inst_map && id < cap) return inst_map[id];
    return NULL;
}

static inline const char *sw_get_id_name(char **id_names, uint32_t cap,
                                         uint32_t id) {
    if (id < cap && id_names[id]) return id_names[id];
    static char buffers[4][64];
    static int next_buf = 0;
    char *buf = buffers[next_buf];
    next_buf = (next_buf + 1) % 4;
    snprintf(buf, 64, "_v%u", id);
    return buf;
}

/* ============================================================================
 * Parser lexer primitives
 * ============================================================================ */

static inline char *sw_strndup(const char *s, size_t n,
                               void *(*alloc)(size_t)) {
    wgsl_compiler_assert(s != NULL, "sw_strndup: s is NULL");
    char *r = (char *)alloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

static inline char *sw_strdup(const char *s, void *(*alloc)(size_t)) {
    return s ? sw_strndup(s, strlen(s), alloc) : NULL;
}

static inline void *sw_grow_array(void *p, int needed, int *cap, size_t elem,
                                  void *(*realloc_fn)(void *, size_t)) {
    wgsl_compiler_assert(cap != NULL, "sw_grow_array: cap is NULL");
    if (needed <= *cap) return p;
    int nc = (*cap == 0) ? 4 : (*cap * 2);
    while (nc < needed) nc *= 2;
    void *np = realloc_fn(p, (size_t)nc * elem);
    if (!np) return p;
    *cap = nc;
    return np;
}

/* ============================================================================
 * SsirBuildCtx - Common (mod, func_id, block_id) triple
 * ============================================================================ */

typedef struct {
    SsirModule *mod;
    uint32_t func_id;
    uint32_t block_id;
} SsirBuildCtx;

#define BCTX(c) (c)->mod, (c)->func_id, (c)->block_id

/* ============================================================================
 * SPIR-V reader enums and structs (shared by wgsl_raise.c, spirv_to_ssir.c)
 * Requires <spirv/unified1/spirv.h> included BEFORE this header.
 * ============================================================================ */

#ifdef SPV_VERSION

typedef enum {
    SW_SPV_ID_UNKNOWN = 0,
    SW_SPV_ID_TYPE,
    SW_SPV_ID_CONSTANT,
    SW_SPV_ID_VARIABLE,
    SW_SPV_ID_FUNCTION,
    SW_SPV_ID_LABEL,
    SW_SPV_ID_INSTRUCTION,
    SW_SPV_ID_EXT_INST_IMPORT,
    SW_SPV_ID_PARAM,
} SwSpvIdKind;

typedef enum {
    SW_SPV_TYPE_VOID = 0,
    SW_SPV_TYPE_BOOL,
    SW_SPV_TYPE_INT,
    SW_SPV_TYPE_FLOAT,
    SW_SPV_TYPE_VECTOR,
    SW_SPV_TYPE_MATRIX,
    SW_SPV_TYPE_ARRAY,
    SW_SPV_TYPE_RUNTIME_ARRAY,
    SW_SPV_TYPE_STRUCT,
    SW_SPV_TYPE_POINTER,
    SW_SPV_TYPE_FUNCTION,
    SW_SPV_TYPE_IMAGE,
    SW_SPV_TYPE_SAMPLER,
    SW_SPV_TYPE_SAMPLED_IMAGE,
} SwSpvTypeKind;

typedef struct {
    SwSpvTypeKind kind;
    union {
        struct { uint32_t width; uint32_t signedness; } int_type;
        struct { uint32_t width; } float_type;
        struct { uint32_t component_type; uint32_t count; } vector;
        struct { uint32_t column_type; uint32_t columns; } matrix;
        struct { uint32_t element_type; uint32_t length_id; } array;
        struct { uint32_t element_type; } runtime_array;
        struct { uint32_t *member_types; int member_count; } struct_type;
        struct { uint32_t pointee_type; SpvStorageClass storage; } pointer;
        struct {
            uint32_t return_type;
            uint32_t *param_types;
            int param_count;
        } function;
        struct {
            uint32_t sampled_type;
            SpvDim dim;
            uint32_t depth;
            uint32_t arrayed;
            uint32_t ms;
            uint32_t sampled;
            SpvImageFormat format;
        } image;
        struct { uint32_t image_type; } sampled_image;
    };
} SwSpvTypeInfo;

typedef struct {
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SwSpvDecorationEntry;

typedef struct {
    uint32_t member_index;
    SpvDecoration decoration;
    uint32_t *literals;
    int literal_count;
} SwSpvMemberDecoration;

typedef struct {
    uint32_t label_id;
    uint32_t *instructions;
    int instruction_count;
    int instruction_cap;
    uint32_t merge_block;
    uint32_t continue_block;
    int is_loop_header;
    int is_selection_header;
    int emitted;
} SwSpvBasicBlock;

typedef struct {
    uint32_t id;
    char *name;
    uint32_t return_type;
    uint32_t func_type;
    uint32_t *params;
    int param_count;
    SwSpvBasicBlock *blocks;
    int block_count;
    int block_cap;
    SpvExecutionModel exec_model;
    int is_entry_point;
    uint32_t *interface_vars;
    int interface_var_count;
    int workgroup_size[3];
    bool depth_replacing;
    bool origin_upper_left;
    bool early_fragment_tests;
    uint32_t *local_vars;
    int local_var_count;
    int local_var_cap;
} SwSpvFunction;

/* ============================================================================
 * SPIR-V reader helpers
 * ============================================================================ */

static inline const char *sw_spv_read_string(const uint32_t *words,
                                             int word_count,
                                             int *out_words_read) {
    const char *str = (const char *)words;
    int max_chars = word_count * 4;
    int len = 0;
    while (len < max_chars && str[len] != '\0') len++;
    if (out_words_read) *out_words_read = (len / 4) + 1;
    return str;
}

static inline void sw_spv_add_decoration(SwSpvDecorationEntry **decorations,
                                         int *decoration_count,
                                         SpvDecoration decor,
                                         const uint32_t *literals,
                                         int lit_count,
                                         void *(*realloc_fn)(void *, size_t)) {
    int idx = (*decoration_count)++;
    *decorations = (SwSpvDecorationEntry *)realloc_fn(
        *decorations,
        (size_t)(*decoration_count) * sizeof(SwSpvDecorationEntry));
    (*decorations)[idx].decoration = decor;
    (*decorations)[idx].literals = NULL;
    (*decorations)[idx].literal_count = lit_count;
    if (lit_count > 0) {
        (*decorations)[idx].literals =
            (uint32_t *)realloc_fn(NULL, (size_t)lit_count * sizeof(uint32_t));
        memcpy((*decorations)[idx].literals, literals,
               (size_t)lit_count * sizeof(uint32_t));
    }
}

static inline void sw_spv_add_member_decoration(
    SwSpvMemberDecoration **member_decorations, int *member_decoration_count,
    uint32_t member, SpvDecoration decor,
    const uint32_t *literals, int lit_count,
    void *(*realloc_fn)(void *, size_t)) {
    int idx = (*member_decoration_count)++;
    *member_decorations = (SwSpvMemberDecoration *)realloc_fn(
        *member_decorations,
        (size_t)(*member_decoration_count) * sizeof(SwSpvMemberDecoration));
    (*member_decorations)[idx].member_index = member;
    (*member_decorations)[idx].decoration = decor;
    (*member_decorations)[idx].literals = NULL;
    (*member_decorations)[idx].literal_count = lit_count;
    if (lit_count > 0) {
        (*member_decorations)[idx].literals =
            (uint32_t *)realloc_fn(NULL, (size_t)lit_count * sizeof(uint32_t));
        memcpy((*member_decorations)[idx].literals, literals,
               (size_t)lit_count * sizeof(uint32_t));
    }
}

static inline int sw_spv_has_decoration(const SwSpvDecorationEntry *decorations,
                                        int decoration_count,
                                        SpvDecoration decor,
                                        uint32_t *out_value) {
    for (int i = 0; i < decoration_count; i++) {
        if (decorations[i].decoration == decor) {
            if (out_value && decorations[i].literal_count > 0)
                *out_value = decorations[i].literals[0];
            return 1;
        }
    }
    return 0;
}

static inline SwSpvFunction *sw_spv_add_function(SwSpvFunction **functions,
                                                 int *count, int *cap,
                                                 uint32_t id,
                                                 void *(*realloc_fn)(void *,
                                                                     size_t)) {
    if (*count >= *cap) {
        int ncap = *cap ? *cap * 2 : 8;
        *functions =
            (SwSpvFunction *)realloc_fn(*functions,
                                        (size_t)ncap * sizeof(SwSpvFunction));
        *cap = ncap;
    }
    SwSpvFunction *fn = &(*functions)[*count];
    memset(fn, 0, sizeof(*fn));
    fn->id = id;
    (*count)++;
    return fn;
}

static inline SwSpvBasicBlock *sw_spv_add_block(SwSpvFunction *fn,
                                                uint32_t label_id,
                                                void *(*realloc_fn)(void *,
                                                                    size_t)) {
    if (fn->block_count >= fn->block_cap) {
        int ncap = fn->block_cap ? fn->block_cap * 2 : 8;
        fn->blocks = (SwSpvBasicBlock *)realloc_fn(
            fn->blocks, (size_t)ncap * sizeof(SwSpvBasicBlock));
        fn->block_cap = ncap;
    }
    SwSpvBasicBlock *blk = &fn->blocks[fn->block_count++];
    memset(blk, 0, sizeof(*blk));
    blk->label_id = label_id;
    return blk;
}

static inline void sw_spv_add_block_instr(SwSpvBasicBlock *blk, uint32_t id,
                                          void *(*realloc_fn)(void *,
                                                              size_t)) {
    if (blk->instruction_count >= blk->instruction_cap) {
        int ncap = blk->instruction_cap ? blk->instruction_cap * 2 : 16;
        blk->instructions = (uint32_t *)realloc_fn(
            blk->instructions, (size_t)ncap * sizeof(uint32_t));
        blk->instruction_cap = ncap;
    }
    blk->instructions[blk->instruction_count++] = id;
}

static inline void sw_spv_add_local_var(SwSpvFunction *fn, uint32_t var_id,
                                        void *(*realloc_fn)(void *, size_t)) {
    if (fn->local_var_count >= fn->local_var_cap) {
        int ncap = fn->local_var_cap ? fn->local_var_cap * 2 : 8;
        fn->local_vars = (uint32_t *)realloc_fn(
            fn->local_vars, (size_t)ncap * sizeof(uint32_t));
        fn->local_var_cap = ncap;
    }
    fn->local_vars[fn->local_var_count++] = var_id;
}

/* ============================================================================
 * SSIR mapping tables: SpvBuiltIn <-> SsirBuiltinVar
 * ============================================================================ */

static inline SsirBuiltinVar sw_spv_builtin_to_ssir(SpvBuiltIn b) {
    switch (b) {
        case SpvBuiltInVertexIndex: return SSIR_BUILTIN_VERTEX_INDEX;
        case SpvBuiltInInstanceIndex: return SSIR_BUILTIN_INSTANCE_INDEX;
        case SpvBuiltInPosition: return SSIR_BUILTIN_POSITION;
        case SpvBuiltInFrontFacing: return SSIR_BUILTIN_FRONT_FACING;
        case SpvBuiltInFragDepth: return SSIR_BUILTIN_FRAG_DEPTH;
        case SpvBuiltInSampleId: return SSIR_BUILTIN_SAMPLE_INDEX;
        case SpvBuiltInSampleMask: return SSIR_BUILTIN_SAMPLE_MASK;
        case SpvBuiltInLocalInvocationId: return SSIR_BUILTIN_LOCAL_INVOCATION_ID;
        case SpvBuiltInLocalInvocationIndex: return SSIR_BUILTIN_LOCAL_INVOCATION_INDEX;
        case SpvBuiltInGlobalInvocationId: return SSIR_BUILTIN_GLOBAL_INVOCATION_ID;
        case SpvBuiltInWorkgroupId: return SSIR_BUILTIN_WORKGROUP_ID;
        case SpvBuiltInNumWorkgroups: return SSIR_BUILTIN_NUM_WORKGROUPS;
        case SpvBuiltInPointSize: return SSIR_BUILTIN_POINT_SIZE;
        case SpvBuiltInClipDistance: return SSIR_BUILTIN_CLIP_DISTANCE;
        case SpvBuiltInCullDistance: return SSIR_BUILTIN_CULL_DISTANCE;
        case SpvBuiltInLayer: return SSIR_BUILTIN_LAYER;
        case SpvBuiltInViewportIndex: return SSIR_BUILTIN_VIEWPORT_INDEX;
        case SpvBuiltInFragCoord: return SSIR_BUILTIN_FRAG_COORD;
        case SpvBuiltInHelperInvocation: return SSIR_BUILTIN_HELPER_INVOCATION;
        case SpvBuiltInPrimitiveId: return SSIR_BUILTIN_PRIMITIVE_ID;
        case SpvBuiltInBaseVertex: return SSIR_BUILTIN_BASE_VERTEX;
        case SpvBuiltInBaseInstance: return SSIR_BUILTIN_BASE_INSTANCE;
        case SpvBuiltInSubgroupSize: return SSIR_BUILTIN_SUBGROUP_SIZE;
        case SpvBuiltInSubgroupLocalInvocationId: return SSIR_BUILTIN_SUBGROUP_INVOCATION_ID;
        case SpvBuiltInSubgroupId: return SSIR_BUILTIN_SUBGROUP_ID;
        case SpvBuiltInNumSubgroups: return SSIR_BUILTIN_NUM_SUBGROUPS;
        default: return SSIR_BUILTIN_NONE;
    }
}

static inline SpvBuiltIn sw_ssir_builtin_to_spv(SsirBuiltinVar b) {
    switch (b) {
        case SSIR_BUILTIN_VERTEX_INDEX: return SpvBuiltInVertexIndex;
        case SSIR_BUILTIN_INSTANCE_INDEX: return SpvBuiltInInstanceIndex;
        case SSIR_BUILTIN_POSITION: return SpvBuiltInPosition;
        case SSIR_BUILTIN_FRONT_FACING: return SpvBuiltInFrontFacing;
        case SSIR_BUILTIN_FRAG_DEPTH: return SpvBuiltInFragDepth;
        case SSIR_BUILTIN_SAMPLE_INDEX: return SpvBuiltInSampleId;
        case SSIR_BUILTIN_SAMPLE_MASK: return SpvBuiltInSampleMask;
        case SSIR_BUILTIN_LOCAL_INVOCATION_ID: return SpvBuiltInLocalInvocationId;
        case SSIR_BUILTIN_LOCAL_INVOCATION_INDEX: return SpvBuiltInLocalInvocationIndex;
        case SSIR_BUILTIN_GLOBAL_INVOCATION_ID: return SpvBuiltInGlobalInvocationId;
        case SSIR_BUILTIN_WORKGROUP_ID: return SpvBuiltInWorkgroupId;
        case SSIR_BUILTIN_NUM_WORKGROUPS: return SpvBuiltInNumWorkgroups;
        case SSIR_BUILTIN_POINT_SIZE: return SpvBuiltInPointSize;
        case SSIR_BUILTIN_CLIP_DISTANCE: return SpvBuiltInClipDistance;
        case SSIR_BUILTIN_CULL_DISTANCE: return SpvBuiltInCullDistance;
        case SSIR_BUILTIN_LAYER: return SpvBuiltInLayer;
        case SSIR_BUILTIN_VIEWPORT_INDEX: return SpvBuiltInViewportIndex;
        case SSIR_BUILTIN_FRAG_COORD: return SpvBuiltInFragCoord;
        case SSIR_BUILTIN_HELPER_INVOCATION: return SpvBuiltInHelperInvocation;
        case SSIR_BUILTIN_PRIMITIVE_ID: return SpvBuiltInPrimitiveId;
        case SSIR_BUILTIN_BASE_VERTEX: return SpvBuiltInBaseVertex;
        case SSIR_BUILTIN_BASE_INSTANCE: return SpvBuiltInBaseInstance;
        case SSIR_BUILTIN_SUBGROUP_SIZE: return SpvBuiltInSubgroupSize;
        case SSIR_BUILTIN_SUBGROUP_INVOCATION_ID: return SpvBuiltInSubgroupLocalInvocationId;
        case SSIR_BUILTIN_SUBGROUP_ID: return SpvBuiltInSubgroupId;
        case SSIR_BUILTIN_NUM_SUBGROUPS: return SpvBuiltInNumSubgroups;
        default: return SpvBuiltInMax;
    }
}

/* ============================================================================
 * SSIR mapping tables: SpvStorageClass <-> SsirAddressSpace
 * ============================================================================ */

static inline SsirAddressSpace sw_spv_storage_class_to_ssir(SpvStorageClass sc) {
    switch (sc) {
        case SpvStorageClassFunction: return SSIR_ADDR_FUNCTION;
        case SpvStorageClassPrivate: return SSIR_ADDR_PRIVATE;
        case SpvStorageClassWorkgroup: return SSIR_ADDR_WORKGROUP;
        case SpvStorageClassUniform: return SSIR_ADDR_UNIFORM;
        case SpvStorageClassUniformConstant: return SSIR_ADDR_UNIFORM_CONSTANT;
        case SpvStorageClassStorageBuffer: return SSIR_ADDR_STORAGE;
        case SpvStorageClassInput: return SSIR_ADDR_INPUT;
        case SpvStorageClassOutput: return SSIR_ADDR_OUTPUT;
        case SpvStorageClassPushConstant: return SSIR_ADDR_PUSH_CONSTANT;
        case SpvStorageClassPhysicalStorageBuffer: return SSIR_ADDR_PHYSICAL_STORAGE_BUFFER;
        default: return SSIR_ADDR_FUNCTION;
    }
}

static inline SpvStorageClass sw_ssir_addr_space_to_spv(SsirAddressSpace a) {
    switch (a) {
        case SSIR_ADDR_FUNCTION: return SpvStorageClassFunction;
        case SSIR_ADDR_PRIVATE: return SpvStorageClassPrivate;
        case SSIR_ADDR_WORKGROUP: return SpvStorageClassWorkgroup;
        case SSIR_ADDR_UNIFORM: return SpvStorageClassUniform;
        case SSIR_ADDR_UNIFORM_CONSTANT: return SpvStorageClassUniformConstant;
        case SSIR_ADDR_STORAGE: return SpvStorageClassStorageBuffer;
        case SSIR_ADDR_INPUT: return SpvStorageClassInput;
        case SSIR_ADDR_OUTPUT: return SpvStorageClassOutput;
        case SSIR_ADDR_PUSH_CONSTANT: return SpvStorageClassPushConstant;
        case SSIR_ADDR_PHYSICAL_STORAGE_BUFFER: return SpvStorageClassPhysicalStorageBuffer;
        default: return SpvStorageClassFunction;
    }
}

/* ============================================================================
 * SSIR mapping tables: SpvDim <-> SsirTextureDim
 * ============================================================================ */

static inline SsirTextureDim sw_spv_dim_to_ssir(SpvDim dim, uint32_t arrayed,
                                                uint32_t ms) {
    if (ms) return arrayed ? SSIR_TEX_MULTISAMPLED_2D_ARRAY : SSIR_TEX_MULTISAMPLED_2D;
    switch (dim) {
        case SpvDim1D: return arrayed ? SSIR_TEX_1D_ARRAY : SSIR_TEX_1D;
        case SpvDim2D: return arrayed ? SSIR_TEX_2D_ARRAY : SSIR_TEX_2D;
        case SpvDim3D: return SSIR_TEX_3D;
        case SpvDimCube: return arrayed ? SSIR_TEX_CUBE_ARRAY : SSIR_TEX_CUBE;
        case SpvDimBuffer: return SSIR_TEX_BUFFER;
        default: return SSIR_TEX_2D;
    }
}

static inline SpvDim sw_ssir_tex_dim_to_spv(SsirTextureDim d) {
    switch (d) {
        case SSIR_TEX_1D: return SpvDim1D;
        case SSIR_TEX_2D: return SpvDim2D;
        case SSIR_TEX_3D: return SpvDim3D;
        case SSIR_TEX_CUBE: return SpvDimCube;
        case SSIR_TEX_2D_ARRAY: return SpvDim2D;
        case SSIR_TEX_CUBE_ARRAY: return SpvDimCube;
        case SSIR_TEX_MULTISAMPLED_2D: return SpvDim2D;
        case SSIR_TEX_1D_ARRAY: return SpvDim1D;
        case SSIR_TEX_BUFFER: return SpvDimBuffer;
        case SSIR_TEX_MULTISAMPLED_2D_ARRAY:
        default: return SpvDim2D;
    }
}

#endif /* SPV_VERSION */

#endif /* SIMPLE_WGSL_INTERNAL_H */
