/*
 * SSIR to SPIR-V Converter - Implementation
 */

#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>
#include "simple_wgsl_internal.h"

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef STS_MALLOC
#define STS_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STS_REALLOC
#define STS_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STS_FREE
#define STS_FREE(p) free((p))
#endif

/* ============================================================================
 * Word Buffer
 * ============================================================================ */

typedef struct {
    uint32_t *data;
    size_t len;
    size_t cap;
} StsWordBuf;

// wb nonnull
static void sts_wb_init(StsWordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_init: wb is NULL");
    wb->data = NULL;
    wb->len = 0;
    wb->cap = 0;
}

// wb nonnull
static void sts_wb_free(StsWordBuf *wb) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_free: wb is NULL");
    STS_FREE(wb->data);
    wb->data = NULL;
    wb->len = wb->cap = 0;
}

// wb nonnull
static int sts_wb_reserve(StsWordBuf *wb, size_t need) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_reserve: wb is NULL");
    if (wb->len + need <= wb->cap) return 1;
    size_t ncap = wb->cap ? wb->cap : 64;
    while (ncap < wb->len + need) ncap *= 2;
    void *nd = STS_REALLOC(wb->data, ncap * sizeof(uint32_t));
    if (!nd) return 0;
    wb->data = (uint32_t *)nd;
    wb->cap = ncap;
    return 1;
}

// wb nonnull
static int wb_push(StsWordBuf *wb, uint32_t w) {
    wgsl_compiler_assert(wb != NULL, "wb_push: wb is NULL");
    if (!sts_wb_reserve(wb, 1)) return 0;
    wb->data[wb->len++] = w;
    return 1;
}

// wb nonnull, src nonnull
static int sts_wb_push_many(StsWordBuf *wb, const uint32_t *src, size_t n) {
    wgsl_compiler_assert(wb != NULL, "sts_wb_push_many: wb is NULL");
    wgsl_compiler_assert(src != NULL, "sts_wb_push_many: src is NULL");
    if (!sts_wb_reserve(wb, n)) return 0;
    memcpy(wb->data + wb->len, src, n * sizeof(uint32_t));
    wb->len += n;
    return 1;
}

/* ============================================================================
 * SPIR-V Sections
 * ============================================================================ */

typedef struct {
    StsWordBuf capabilities;
    StsWordBuf extensions;
    StsWordBuf ext_inst_imports;
    StsWordBuf memory_model;
    StsWordBuf entry_points;
    StsWordBuf execution_modes;
    StsWordBuf debug_names;
    StsWordBuf annotations;
    StsWordBuf types_constants;
    StsWordBuf globals;
    StsWordBuf functions;
} StsSpvSections;

// s nonnull
static void sections_init(StsSpvSections *s) {
    wgsl_compiler_assert(s != NULL, "sections_init: s is NULL");
    sts_wb_init(&s->capabilities);
    sts_wb_init(&s->extensions);
    sts_wb_init(&s->ext_inst_imports);
    sts_wb_init(&s->memory_model);
    sts_wb_init(&s->entry_points);
    sts_wb_init(&s->execution_modes);
    sts_wb_init(&s->debug_names);
    sts_wb_init(&s->annotations);
    sts_wb_init(&s->types_constants);
    sts_wb_init(&s->globals);
    sts_wb_init(&s->functions);
}

// s nonnull
static void sections_free(StsSpvSections *s) {
    wgsl_compiler_assert(s != NULL, "sections_free: s is NULL");
    sts_wb_free(&s->capabilities);
    sts_wb_free(&s->extensions);
    sts_wb_free(&s->ext_inst_imports);
    sts_wb_free(&s->memory_model);
    sts_wb_free(&s->entry_points);
    sts_wb_free(&s->execution_modes);
    sts_wb_free(&s->debug_names);
    sts_wb_free(&s->annotations);
    sts_wb_free(&s->types_constants);
    sts_wb_free(&s->globals);
    sts_wb_free(&s->functions);
}

/* ============================================================================
 * Converter Context
 * ============================================================================ */

typedef struct {
    const SsirModule *mod;
    SsirToSpirvOptions opts;
    StsSpvSections sections;

    /* ID mapping: SSIR ID -> SPIR-V ID */
    uint32_t *id_map;
    uint32_t id_map_size;
    uint32_t next_spv_id;

    /* Type mapping: SSIR ID -> SSIR type ID */
    uint32_t *type_map;
    uint32_t type_map_size;

    /* Type IDs for common types (cached) */
    uint32_t spv_void;
    uint32_t spv_bool;
    uint32_t spv_i32;
    uint32_t spv_u32;
    uint32_t spv_f32;
    uint32_t spv_f16;

    /* GLSL.std.450 import ID */
    uint32_t glsl_ext_id;

    /* Capability tracking */
    int has_shader_cap;
    int has_float16_cap;
    int has_image_query_cap;
    int has_psb_cap;       /* PhysicalStorageBufferAddresses */
    int has_binding_array; /* RuntimeDescriptorArrayEXT */
    int glsl_ext_emitted;

    /* Pre-computed function type IDs (indexed by function index) */
    uint32_t *func_type_cache;
    uint32_t func_type_cache_count;

    /* Deduplication cache for OpTypeFunction signatures */
    struct {
        uint32_t return_type;
        uint32_t param_count;
        uint32_t param_hash; /* simple hash of param types for quick reject */
        uint32_t spv_id;
    } func_type_dedup[64];
    uint32_t func_type_dedup_count;

    /* Deduplication cache for OpTypeSampledImage */
    struct {
        uint32_t image_type;
        uint32_t spv_id;
    } sampled_image_cache[16];
    uint32_t sampled_image_cache_count;

    /* Block decoration dedup: SPIR-V type IDs that already have Block */
    uint32_t block_decorated[64];
    uint32_t block_decorated_count;
} Ctx;

// c nonnull
static uint32_t sts_fresh_id(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_fresh_id: c is NULL");
    return c->next_spv_id++;
}

// c nonnull
static uint32_t get_spv_id(Ctx *c, uint32_t ssir_id) {
    wgsl_compiler_assert(c != NULL, "get_spv_id: c is NULL");
    if (ssir_id < c->id_map_size && c->id_map[ssir_id] != 0) {
        return c->id_map[ssir_id];
    }
    /* Allocate new SPIR-V ID */
    uint32_t spv_id = sts_fresh_id(c);
    if (ssir_id < c->id_map_size) {
        c->id_map[ssir_id] = spv_id;
    }
    return spv_id;
}

// c nonnull
static void set_spv_id(Ctx *c, uint32_t ssir_id, uint32_t spv_id) {
    wgsl_compiler_assert(c != NULL, "set_spv_id: c is NULL");
    if (ssir_id < c->id_map_size) {
        c->id_map[ssir_id] = spv_id;
    }
}

// c nonnull
static void set_ssir_type(Ctx *c, uint32_t ssir_id, uint32_t ssir_type) {
    wgsl_compiler_assert(c != NULL, "set_ssir_type: c is NULL");
    if (ssir_id < c->type_map_size) {
        c->type_map[ssir_id] = ssir_type;
    }
}

// c nonnull
static uint32_t get_ssir_type(Ctx *c, uint32_t ssir_id) {
    wgsl_compiler_assert(c != NULL, "get_ssir_type: c is NULL");
    if (ssir_id < c->type_map_size) {
        return c->type_map[ssir_id];
    }
    return 0;
}

/* ============================================================================
 * String Literal Encoding
 * ============================================================================ */

// out_words nonnull, out_count nonnull
static uint32_t encode_string(const char *s, uint32_t **out_words, size_t *out_count) {
    wgsl_compiler_assert(out_words != NULL, "encode_string: out_words is NULL");
    wgsl_compiler_assert(out_count != NULL, "encode_string: out_count is NULL");
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    size_t words = (n + 3) / 4;
    uint32_t *buf = (uint32_t *)STS_MALLOC(words * sizeof(uint32_t));
    if (!buf) {
        *out_words = NULL;
        *out_count = 0;
        return 0;
    }
    memset(buf, 0, words * sizeof(uint32_t));
    memcpy(buf, s, n);
    *out_words = buf;
    *out_count = words;
    return (uint32_t)words;
}

/* ============================================================================
 * Instruction Emission Helpers
 * ============================================================================ */

// wb nonnull
static int sts_emit_op(StsWordBuf *wb, SpvOp op, size_t word_count) {
    wgsl_compiler_assert(wb != NULL, "sts_emit_op: wb is NULL");
    return wb_push(wb, ((uint32_t)word_count << 16) | (uint32_t)op);
}

// c nonnull
static int sts_emit_capability(Ctx *c, SpvCapability cap) {
    wgsl_compiler_assert(c != NULL, "sts_emit_capability: c is NULL");
    StsWordBuf *wb = &c->sections.capabilities;
    if (!sts_emit_op(wb, SpvOpCapability, 2)) return 0;
    return wb_push(wb, cap);
}

// c nonnull
static void sts_ensure_glsl_import(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_ensure_glsl_import: c is NULL");
    if (c->glsl_ext_emitted) return;
    c->glsl_ext_emitted = 1;
    if (!c->glsl_ext_id) c->glsl_ext_id = sts_fresh_id(c);
    StsWordBuf *wb = &c->sections.ext_inst_imports;
    uint32_t *str;
    size_t wn;
    encode_string("GLSL.std.450", &str, &wn);
    sts_emit_op(wb, SpvOpExtInstImport, 2 + wn);
    wb_push(wb, c->glsl_ext_id);
    sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
}

// c nonnull
static int sts_emit_memory_model(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_memory_model: c is NULL");
    StsWordBuf *wb = &c->sections.memory_model;
    if (!sts_emit_op(wb, SpvOpMemoryModel, 3)) return 0;
    if (!wb_push(wb, SpvAddressingModelLogical)) return 0;
    return wb_push(wb, SpvMemoryModelGLSL450);
}

// c nonnull
static int sts_emit_name(Ctx *c, uint32_t target, const char *name) {
    wgsl_compiler_assert(c != NULL, "sts_emit_name: c is NULL");
    if (!name || !*name) return 1;
    StsWordBuf *wb = &c->sections.debug_names;
    uint32_t *str;
    size_t wn;
    encode_string(name, &str, &wn);
    if (!sts_emit_op(wb, SpvOpName, 2 + wn)) {
        STS_FREE(str);
        return 0;
    }
    if (!wb_push(wb, target)) {
        STS_FREE(str);
        return 0;
    }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    return ok;
}

// c nonnull
static int sts_emit_member_name(Ctx *c, uint32_t struct_id, uint32_t member, const char *name) {
    wgsl_compiler_assert(c != NULL, "sts_emit_member_name: c is NULL");
    if (!name || !*name) return 1;
    StsWordBuf *wb = &c->sections.debug_names;
    uint32_t *str;
    size_t wn;
    encode_string(name, &str, &wn);
    if (!sts_emit_op(wb, SpvOpMemberName, 3 + wn)) {
        STS_FREE(str);
        return 0;
    }
    if (!wb_push(wb, struct_id)) {
        STS_FREE(str);
        return 0;
    }
    if (!wb_push(wb, member)) {
        STS_FREE(str);
        return 0;
    }
    int ok = sts_wb_push_many(wb, str, wn);
    STS_FREE(str);
    return ok;
}

// c nonnull
static int sts_emit_decorate(Ctx *c, uint32_t target, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_decorate: c is NULL");
    StsWordBuf *wb = &c->sections.annotations;
    if (!sts_emit_op(wb, SpvOpDecorate, 3 + lit_count)) return 0;
    if (!wb_push(wb, target)) return 0;
    if (!wb_push(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push(wb, literals[i])) return 0;
    }
    return 1;
}

// c nonnull
// Emit Block decoration on target, but only once per target ID.
static int sts_emit_block_decoration(Ctx *c, uint32_t target) {
    wgsl_compiler_assert(c != NULL, "sts_emit_block_decoration: c is NULL");
    for (uint32_t i = 0; i < c->block_decorated_count; i++) {
        if (c->block_decorated[i] == target) return 1; // already decorated
    }
    if (c->block_decorated_count < 64) {
        c->block_decorated[c->block_decorated_count++] = target;
    }
    return sts_emit_decorate(c, target, SpvDecorationBlock, NULL, 0);
}

// c nonnull
static int sts_emit_member_decorate(Ctx *c, uint32_t struct_id, uint32_t member, SpvDecoration decor, const uint32_t *literals, int lit_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_member_decorate: c is NULL");
    StsWordBuf *wb = &c->sections.annotations;
    if (!sts_emit_op(wb, SpvOpMemberDecorate, 4 + lit_count)) return 0;
    if (!wb_push(wb, struct_id)) return 0;
    if (!wb_push(wb, member)) return 0;
    if (!wb_push(wb, decor)) return 0;
    for (int i = 0; i < lit_count; ++i) {
        if (!wb_push(wb, literals[i])) return 0;
    }
    return 1;
}

/* ============================================================================
 * Type Emission
 * ============================================================================ */

static uint32_t sts_emit_type(Ctx *c, uint32_t ssir_type_id);
static uint32_t sts_emit_const_u32(Ctx *c, uint32_t value);

/* Compute the size and alignment of a type (for std430 layout).
 * Returns the size in bytes, and stores alignment in *out_align. */
static uint32_t compute_type_size_depth(Ctx *c, uint32_t ssir_type_id,
                                        uint32_t *out_align, int depth) {
    if (depth > 64) {
        if (out_align) *out_align = 4;
        return 4; /* runaway / cyclic type — bail */
    }
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type_id);
    if (!t) {
        if (out_align) *out_align = 4;
        return 4;
    }
    switch (t->kind) {
        case SSIR_TYPE_BOOL:
        case SSIR_TYPE_I32:
        case SSIR_TYPE_U32:
        case SSIR_TYPE_F32:
            if (out_align) *out_align = 4;
            return 4;
        case SSIR_TYPE_F16:
            if (out_align) *out_align = 2;
            return 2;
        case SSIR_TYPE_VEC: {
            uint32_t elem_size = compute_type_size_depth(c, t->vec.elem, NULL, depth + 1);
            uint32_t size = t->vec.size;
            uint32_t align_factor = (size == 2) ? 2 : 4;
            if (out_align) *out_align = elem_size * align_factor;
            return elem_size * size;
        }
        case SSIR_TYPE_MAT: {
            uint32_t col_align;
            uint32_t col_size = compute_type_size_depth(c, t->mat.elem, &col_align, depth + 1);
            uint32_t stride = (col_size + col_align - 1) & ~(col_align - 1);
            if (out_align) *out_align = col_align;
            return stride * t->mat.cols;
        }
        case SSIR_TYPE_ARRAY: {
            uint32_t elem_align;
            uint32_t elem_size = compute_type_size_depth(c, t->array.elem, &elem_align, depth + 1);
            uint32_t stride = (elem_size + elem_align - 1) & ~(elem_align - 1);
            if (out_align) *out_align = elem_align;
            uint64_t total = (uint64_t)stride * (uint64_t)t->array.length;
            if (total > UINT32_MAX) return 0;
            return (uint32_t)total;
        }
        case SSIR_TYPE_RUNTIME_ARRAY: {
            uint32_t elem_align;
            compute_type_size_depth(c, t->runtime_array.elem, &elem_align, depth + 1);
            if (out_align) *out_align = elem_align;
            return 0;
        }
        case SSIR_TYPE_STRUCT: {
            uint32_t max_align = 1;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                uint32_t mem_align;
                uint32_t mem_size = compute_type_size_depth(c, t->struc.members[i], &mem_align, depth + 1);
                if (mem_align > max_align) max_align = mem_align;
                offset = (offset + mem_align - 1) & ~(mem_align - 1);
                offset += mem_size;
            }
            offset = (offset + max_align - 1) & ~(max_align - 1);
            if (out_align) *out_align = max_align;
            return offset;
        }
        default:
            if (out_align) *out_align = 4;
            return 4;
    }
}

// c nonnull
static uint32_t compute_type_size(Ctx *c, uint32_t ssir_type_id, uint32_t *out_align) {
    wgsl_compiler_assert(c != NULL, "compute_type_size: c is NULL");
    return compute_type_size_depth(c, ssir_type_id, out_align, 0);
}

/* Compute the array stride for a given element type */
// c nonnull
static uint32_t compute_array_stride(Ctx *c, uint32_t elem_type_id) {
    wgsl_compiler_assert(c != NULL, "compute_array_stride: c is NULL");
    uint32_t elem_align;
    uint32_t elem_size = compute_type_size(c, elem_type_id, &elem_align);
    /* Stride is element size rounded up to element alignment */
    return (elem_size + elem_align - 1) & ~(elem_align - 1);
}

// c nonnull
static uint32_t sts_emit_type_void(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_void: c is NULL");
    if (c->spv_void) return c->spv_void;
    StsWordBuf *wb = &c->sections.types_constants;
    c->spv_void = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeVoid, 2)) return 0;
    if (!wb_push(wb, c->spv_void)) return 0;
    return c->spv_void;
}

// c nonnull
static uint32_t sts_emit_type_bool(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_bool: c is NULL");
    if (c->spv_bool) return c->spv_bool;
    StsWordBuf *wb = &c->sections.types_constants;
    c->spv_bool = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeBool, 2)) return 0;
    if (!wb_push(wb, c->spv_bool)) return 0;
    return c->spv_bool;
}

// c nonnull
static uint32_t sts_emit_type_int(Ctx *c, uint32_t width, uint32_t signedness) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_int: c is NULL");
    if (width == 32 && signedness == 1 && c->spv_i32) return c->spv_i32;
    if (width == 32 && signedness == 0 && c->spv_u32) return c->spv_u32;
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeInt, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, width)) return 0;
    if (!wb_push(wb, signedness)) return 0;
    if (width == 32 && signedness == 1) c->spv_i32 = id;
    if (width == 32 && signedness == 0) c->spv_u32 = id;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_float(Ctx *c, uint32_t width) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_float: c is NULL");
    if (width == 32 && c->spv_f32) return c->spv_f32;
    if (width == 16 && c->spv_f16) return c->spv_f16;
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeFloat, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, width)) return 0;
    if (width == 32) c->spv_f32 = id;
    if (width == 16) c->spv_f16 = id;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_vector(Ctx *c, uint32_t elem_type, uint32_t count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_vector: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeVector, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    if (!wb_push(wb, count)) return 0;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_matrix(Ctx *c, uint32_t col_type, uint32_t col_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_matrix: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeMatrix, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, col_type)) return 0;
    if (!wb_push(wb, col_count)) return 0;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_array(Ctx *c, uint32_t elem_type, uint32_t length_id) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_array: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeArray, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    if (!wb_push(wb, length_id)) return 0;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_runtime_array(Ctx *c, uint32_t elem_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_runtime_array: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeRuntimeArray, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, elem_type)) return 0;
    return id;
}

// c nonnull, member_types nonnull
static uint32_t sts_emit_type_struct(Ctx *c, uint32_t *member_types, uint32_t member_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_struct: c is NULL");
    wgsl_compiler_assert(member_types != NULL, "sts_emit_type_struct: member_types is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeStruct, 2 + member_count)) return 0;
    if (!wb_push(wb, id)) return 0;
    for (uint32_t i = 0; i < member_count; ++i) {
        if (!wb_push(wb, member_types[i])) return 0;
    }
    return id;
}

#define addr_space_to_storage_class sw_ssir_addr_space_to_spv

// c nonnull
static uint32_t sts_emit_type_pointer(Ctx *c, SpvStorageClass sc, uint32_t pointee) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_pointer: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypePointer, 4)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, sc)) return 0;
    if (!wb_push(wb, pointee)) return 0;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_function(Ctx *c, uint32_t return_type, uint32_t *param_types, uint32_t param_count) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_function: c is NULL");

    /* Compute a simple hash of param types for quick comparison */
    uint32_t param_hash = 0;
    for (uint32_t i = 0; i < param_count; ++i)
        param_hash = param_hash * 31 + param_types[i];

    /* Check dedup cache for matching signature */
    for (uint32_t d = 0; d < c->func_type_dedup_count; ++d) {
        if (c->func_type_dedup[d].return_type == return_type &&
            c->func_type_dedup[d].param_count == param_count &&
            c->func_type_dedup[d].param_hash == param_hash) {
            return c->func_type_dedup[d].spv_id;
        }
    }

    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeFunction, 3 + param_count)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, return_type)) return 0;
    for (uint32_t i = 0; i < param_count; ++i) {
        if (!wb_push(wb, param_types[i])) return 0;
    }

    /* Store in dedup cache */
    if (c->func_type_dedup_count < 64) {
        c->func_type_dedup[c->func_type_dedup_count].return_type = return_type;
        c->func_type_dedup[c->func_type_dedup_count].param_count = param_count;
        c->func_type_dedup[c->func_type_dedup_count].param_hash = param_hash;
        c->func_type_dedup[c->func_type_dedup_count].spv_id = id;
        c->func_type_dedup_count++;
    }

    return id;
}

// c nonnull
static uint32_t sts_emit_type_sampler(Ctx *c) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_sampler: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeSampler, 2)) return 0;
    if (!wb_push(wb, id)) return 0;
    return id;
}

#define texture_dim_to_spv sw_ssir_tex_dim_to_spv

// c nonnull
static uint32_t sts_emit_type_image(Ctx *c, SsirTextureDim dim, uint32_t sampled_type,
    uint32_t depth, uint32_t arrayed, uint32_t ms, uint32_t sampled, SpvImageFormat format) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_image: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeImage, 9)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, sampled_type)) return 0;
    if (!wb_push(wb, texture_dim_to_spv(dim))) return 0;
    if (!wb_push(wb, depth)) return 0;
    if (!wb_push(wb, arrayed)) return 0;
    if (!wb_push(wb, ms)) return 0;
    if (!wb_push(wb, sampled)) return 0;
    if (!wb_push(wb, format)) return 0;
    return id;
}

// c nonnull
static uint32_t sts_emit_type_sampled_image(Ctx *c, uint32_t image_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type_sampled_image: c is NULL");
    /* Deduplicate: check if we already emitted a sampled image type for this image type */
    for (uint32_t i = 0; i < c->sampled_image_cache_count; i++) {
        if (c->sampled_image_cache[i].image_type == image_type)
            return c->sampled_image_cache[i].spv_id;
    }
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t id = sts_fresh_id(c);
    if (!sts_emit_op(wb, SpvOpTypeSampledImage, 3)) return 0;
    if (!wb_push(wb, id)) return 0;
    if (!wb_push(wb, image_type)) return 0;
    if (c->sampled_image_cache_count < 16) {
        c->sampled_image_cache[c->sampled_image_cache_count].image_type = image_type;
        c->sampled_image_cache[c->sampled_image_cache_count].spv_id = id;
        c->sampled_image_cache_count++;
    }
    return id;
}

/* Emit SSIR type, returns SPIR-V type ID */
// c nonnull
static uint32_t sts_emit_type(Ctx *c, uint32_t ssir_type_id) {
    wgsl_compiler_assert(c != NULL, "sts_emit_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type_id);
    if (!t) return 0;

    /* Check if already emitted */
    uint32_t cached = 0;
    if (ssir_type_id < c->id_map_size) {
        cached = c->id_map[ssir_type_id];
        if (cached != 0) return cached;
    }

    uint32_t spv_id = 0;
    switch (t->kind) {
        case SSIR_TYPE_VOID:
            spv_id = sts_emit_type_void(c);
            break;
        case SSIR_TYPE_BOOL:
            spv_id = sts_emit_type_bool(c);
            break;
        case SSIR_TYPE_I32:
            spv_id = sts_emit_type_int(c, 32, 1);
            break;
        case SSIR_TYPE_U32:
            spv_id = sts_emit_type_int(c, 32, 0);
            break;
        case SSIR_TYPE_F32:
            spv_id = sts_emit_type_float(c, 32);
            break;
        case SSIR_TYPE_F16:
            spv_id = sts_emit_type_float(c, 16);
            if (!c->has_float16_cap) {
                sts_emit_capability(c, SpvCapabilityFloat16);
                c->has_float16_cap = 1;
            }
            break;
        case SSIR_TYPE_F64:
            spv_id = sts_emit_type_float(c, 64);
            sts_emit_capability(c, SpvCapabilityFloat64);
            break;
        case SSIR_TYPE_I8:
            spv_id = sts_emit_type_int(c, 8, 1);
            sts_emit_capability(c, SpvCapabilityInt8);
            break;
        case SSIR_TYPE_U8:
            spv_id = sts_emit_type_int(c, 8, 0);
            sts_emit_capability(c, SpvCapabilityInt8);
            break;
        case SSIR_TYPE_I16:
            spv_id = sts_emit_type_int(c, 16, 1);
            sts_emit_capability(c, SpvCapabilityInt16);
            break;
        case SSIR_TYPE_U16:
            spv_id = sts_emit_type_int(c, 16, 0);
            sts_emit_capability(c, SpvCapabilityInt16);
            break;
        case SSIR_TYPE_I64:
            spv_id = sts_emit_type_int(c, 64, 1);
            sts_emit_capability(c, SpvCapabilityInt64);
            break;
        case SSIR_TYPE_U64:
            spv_id = sts_emit_type_int(c, 64, 0);
            sts_emit_capability(c, SpvCapabilityInt64);
            break;
        case SSIR_TYPE_VEC: {
            uint32_t elem_spv = sts_emit_type(c, t->vec.elem);
            spv_id = sts_emit_type_vector(c, elem_spv, t->vec.size);
            break;
        }
        case SSIR_TYPE_MAT: {
            /* Matrix element is column type (vector) */
            uint32_t col_spv = sts_emit_type(c, t->mat.elem);
            spv_id = sts_emit_type_matrix(c, col_spv, t->mat.cols);
            break;
        }
        case SSIR_TYPE_ARRAY: {
            uint32_t elem_spv = sts_emit_type(c, t->array.elem);
            /* Emit length as constant */
            uint32_t len_id = sts_fresh_id(c);
            StsWordBuf *wb = &c->sections.types_constants;
            uint32_t u32_type = c->spv_u32 ? c->spv_u32 : sts_emit_type_int(c, 32, 0);
            sts_emit_op(wb, SpvOpConstant, 4);
            wb_push(wb, u32_type);
            wb_push(wb, len_id);
            wb_push(wb, t->array.length);
            spv_id = sts_emit_type_array(c, elem_spv, len_id);
            /* Decorate array stride only when explicitly set (buffer arrays).
               Workgroup/private arrays must NOT have ArrayStride. */
            if (t->array.stride) {
                uint32_t stride = t->array.stride;
                sts_emit_decorate(c, spv_id, SpvDecorationArrayStride, &stride, 1);
            }
            break;
        }
        case SSIR_TYPE_RUNTIME_ARRAY: {
            uint32_t elem_spv = sts_emit_type(c, t->runtime_array.elem);
            spv_id = sts_emit_type_runtime_array(c, elem_spv);
            /* Decorate runtime array stride */
            uint32_t stride = compute_array_stride(c, t->runtime_array.elem);
            sts_emit_decorate(c, spv_id, SpvDecorationArrayStride, &stride, 1);
            break;
        }
        case SSIR_TYPE_STRUCT: {
            uint32_t *member_spv = (uint32_t *)STS_MALLOC(t->struc.member_count * sizeof(uint32_t));
            if (!member_spv) return 0;
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                member_spv[i] = sts_emit_type(c, t->struc.members[i]);
            }
            spv_id = sts_emit_type_struct(c, member_spv, t->struc.member_count);
            STS_FREE(member_spv);
            /* Emit debug names and member decorations */
            if (t->struc.name) {
                sts_emit_name(c, spv_id, t->struc.name);
            }
            if (t->struc.member_names) {
                for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                    if (t->struc.member_names[i])
                        sts_emit_member_name(c, spv_id, i, t->struc.member_names[i]);
                }
            }
            if (t->struc.offsets) {
                for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                    sts_emit_member_decorate(c, spv_id, i, SpvDecorationOffset, &t->struc.offsets[i], 1);
                }
            }
            /* Emit ColMajor/RowMajor + MatrixStride for matrix members */
            for (uint32_t i = 0; i < t->struc.member_count; ++i) {
                SsirType *mt = ssir_get_type((SsirModule *)c->mod, t->struc.members[i]);
                if (mt && mt->kind == SSIR_TYPE_MAT) {
                    uint8_t major = (t->struc.matrix_major && t->struc.matrix_major[i]) ? t->struc.matrix_major[i] : 1;
                    SpvDecoration maj_dec = (major == 2) ? SpvDecorationRowMajor : SpvDecorationColMajor;
                    sts_emit_member_decorate(c, spv_id, i, maj_dec, NULL, 0);
                    uint32_t stride;
                    if (t->struc.matrix_strides && t->struc.matrix_strides[i]) {
                        stride = t->struc.matrix_strides[i];
                    } else {
                        uint32_t comp_size = 4;
                        SsirType *elem_t = ssir_get_type((SsirModule *)c->mod, mt->mat.elem);
                        if (elem_t && elem_t->kind == SSIR_TYPE_F16) comp_size = 2;
                        stride = mt->mat.rows * comp_size;
                        if (stride < 16) stride = 16;
                    }
                    sts_emit_member_decorate(c, spv_id, i, SpvDecorationMatrixStride, &stride, 1);
                }
            }
            break;
        }
        case SSIR_TYPE_PTR: {
            uint32_t pointee_spv = sts_emit_type(c, t->ptr.pointee);
            SpvStorageClass sc = addr_space_to_storage_class(t->ptr.space);
            spv_id = sts_emit_type_pointer(c, sc, pointee_spv);
            if (t->ptr.space == SSIR_ADDR_PHYSICAL_STORAGE_BUFFER) {
                if (!c->has_psb_cap) {
                    sts_emit_capability(c, SpvCapabilityPhysicalStorageBufferAddresses);
                    c->has_psb_cap = 1;
                }
                /* Block decoration for PSB pointee structs (required for runtime arrays) */
                SsirType *pointee_t = ssir_get_type((SsirModule *)c->mod, t->ptr.pointee);
                if (pointee_t && pointee_t->kind == SSIR_TYPE_STRUCT) {
                    sts_emit_block_decoration(c, pointee_spv);
                }
            }
            break;
        }
        case SSIR_TYPE_SAMPLER:
        case SSIR_TYPE_SAMPLER_COMPARISON:
            spv_id = sts_emit_type_sampler(c);
            break;
        case SSIR_TYPE_TEXTURE: {
            uint32_t sampled_spv = sts_emit_type(c, t->texture.sampled_type);
            SsirTextureDim dim = t->texture.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_CUBE_ARRAY ||
                                   dim == SSIR_TEX_1D_ARRAY || dim == SSIR_TEX_MULTISAMPLED_2D_ARRAY)
                                   ? 1
                                   : 0;
            uint32_t ms = (dim == SSIR_TEX_MULTISAMPLED_2D || dim == SSIR_TEX_MULTISAMPLED_2D_ARRAY) ? 1 : 0;
            spv_id = sts_emit_type_image(c, dim, sampled_spv, 0, arrayed, ms, 1, SpvImageFormatUnknown);
            break;
        }
        case SSIR_TYPE_TEXTURE_STORAGE: {
            /* Pick sampled type based on image format:
             * Rgba32i/Rgba16i/etc. → int, Rgba32ui/Rgba16ui/etc. → uint,
             * everything else (Rgba32f, Rgba16f, etc.) → float */
            SpvImageFormat fmt = (SpvImageFormat)t->texture_storage.format;
            uint32_t sampled_type;
            if (fmt == SpvImageFormatRgba32i || fmt == SpvImageFormatRgba16i ||
                fmt == SpvImageFormatRgba8i || fmt == SpvImageFormatR32i ||
                fmt == SpvImageFormatRg32i || fmt == SpvImageFormatRg16i ||
                fmt == SpvImageFormatRg8i || fmt == SpvImageFormatR16i ||
                fmt == SpvImageFormatR8i) {
                sampled_type = sts_emit_type_int(c, 32, 1);
            } else if (fmt == SpvImageFormatRgba32ui || fmt == SpvImageFormatRgba16ui ||
                       fmt == SpvImageFormatRgba8ui || fmt == SpvImageFormatR32ui ||
                       fmt == SpvImageFormatRg32ui || fmt == SpvImageFormatRg16ui ||
                       fmt == SpvImageFormatRg8ui || fmt == SpvImageFormatR16ui ||
                       fmt == SpvImageFormatR8ui) {
                sampled_type = sts_emit_type_int(c, 32, 0);
            } else {
                sampled_type = c->spv_f32 ? c->spv_f32 : sts_emit_type_float(c, 32);
            }
            SsirTextureDim dim = t->texture_storage.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_1D_ARRAY) ? 1 : 0;
            uint32_t sampled = 2; /* storage image */
            spv_id = sts_emit_type_image(c, dim, sampled_type, 0, arrayed, 0, sampled, fmt);
            break;
        }
        case SSIR_TYPE_TEXTURE_DEPTH: {
            uint32_t f32_type = c->spv_f32 ? c->spv_f32 : sts_emit_type_float(c, 32);
            SsirTextureDim dim = t->texture_depth.dim;
            uint32_t arrayed = (dim == SSIR_TEX_2D_ARRAY || dim == SSIR_TEX_CUBE_ARRAY || dim == SSIR_TEX_1D_ARRAY) ? 1 : 0;
            spv_id = sts_emit_type_image(c, dim, f32_type, 1, arrayed, 0, 1, SpvImageFormatUnknown);
            break;
        }
        case SSIR_TYPE_BINDING_ARRAY: {
            uint32_t elem_spv = sts_emit_type(c, t->binding_array.elem);
            if (t->binding_array.length > 0) {
                /* Fixed-size binding array */
                uint32_t len_id = sts_emit_const_u32(c, t->binding_array.length);
                spv_id = sts_emit_type_array(c, elem_spv, len_id);
            } else {
                /* Unsized binding array → OpTypeRuntimeArray */
                spv_id = sts_emit_type_runtime_array(c, elem_spv);
            }
            c->has_binding_array = 1;
            break;
        }
    }

    /* Cache */
    if (spv_id && ssir_type_id < c->id_map_size) {
        c->id_map[ssir_type_id] = spv_id;
    }

    return spv_id;
}

/* ============================================================================
 * Constant Emission
 * ============================================================================ */

// c nonnull
static uint32_t sts_emit_const_u32(Ctx *c, uint32_t value) {
    wgsl_compiler_assert(c != NULL, "sts_emit_const_u32: c is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t u32_type = c->spv_u32 ? c->spv_u32 : sts_emit_type_int(c, 32, 0);
    uint32_t id = sts_fresh_id(c);
    sts_emit_op(wb, SpvOpConstant, 4);
    wb_push(wb, u32_type);
    wb_push(wb, id);
    wb_push(wb, value);
    return id;
}

// c nonnull, cnst nonnull
static uint32_t sts_emit_constant(Ctx *c, const SsirConstant *cnst) {
    wgsl_compiler_assert(c != NULL, "sts_emit_constant: c is NULL");
    wgsl_compiler_assert(cnst != NULL, "sts_emit_constant: cnst is NULL");
    StsWordBuf *wb = &c->sections.types_constants;
    uint32_t type_spv = sts_emit_type(c, cnst->type);
    uint32_t id = get_spv_id(c, cnst->id);

    bool spec = cnst->is_specialization;
    switch (cnst->kind) {
        case SSIR_CONST_BOOL:
            if (cnst->bool_val) {
                sts_emit_op(wb, spec ? SpvOpSpecConstantTrue : SpvOpConstantTrue, 3);
            } else {
                sts_emit_op(wb, spec ? SpvOpSpecConstantFalse : SpvOpConstantFalse, 3);
            }
            wb_push(wb, type_spv);
            wb_push(wb, id);
            break;
        case SSIR_CONST_I32:
        case SSIR_CONST_U32:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, cnst->u32_val);
            break;
        case SSIR_CONST_F32: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t bits;
            memcpy(&bits, &cnst->f32_val, sizeof(float));
            wb_push(wb, bits);
            break;
        }
        case SSIR_CONST_F16:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, cnst->f16_val);
            break;
        case SSIR_CONST_F64: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 5);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t dw[2];
            memcpy(dw, &cnst->f64_val, sizeof(double));
            wb_push(wb, dw[0]);
            wb_push(wb, dw[1]);
            break;
        }
        case SSIR_CONST_I8:
        case SSIR_CONST_U8:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, (uint32_t)cnst->u8_val);
            break;
        case SSIR_CONST_I16:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            /* Sign-extend i16 to 32 bits as required by SPIR-V */
            wb_push(wb, (uint32_t)(int32_t)cnst->i16_val);
            break;
        case SSIR_CONST_U16:
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 4);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            wb_push(wb, (uint32_t)cnst->u16_val);
            break;
        case SSIR_CONST_I64:
        case SSIR_CONST_U64: {
            sts_emit_op(wb, spec ? SpvOpSpecConstant : SpvOpConstant, 5);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            uint32_t qw[2];
            memcpy(qw, &cnst->u64_val, sizeof(uint64_t));
            wb_push(wb, qw[0]);
            wb_push(wb, qw[1]);
            break;
        }
        case SSIR_CONST_COMPOSITE: {
            sts_emit_op(wb, spec ? SpvOpSpecConstantComposite : SpvOpConstantComposite, 3 + cnst->composite.count);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            for (uint32_t i = 0; i < cnst->composite.count; ++i) {
                wb_push(wb, get_spv_id(c, cnst->composite.components[i]));
            }
            break;
        }
        case SSIR_CONST_NULL:
            sts_emit_op(wb, SpvOpConstantNull, 3);
            wb_push(wb, type_spv);
            wb_push(wb, id);
            break;
    }

    /* Emit SpecId decoration for specialization constants */
    if (spec) {
        sts_emit_decorate(c, id, SpvDecorationSpecId, &cnst->spec_id, 1);
    }

    /* Emit debug name for named constants */
    sts_emit_name(c, id, cnst->name);

    /* Record type for this constant */
    set_ssir_type(c, cnst->id, cnst->type);

    return id;
}

/* ============================================================================
 * Global Variable Emission
 * ============================================================================ */

#define builtin_var_to_spv sw_ssir_builtin_to_spv

// c nonnull, g nonnull
static uint32_t sts_emit_global_var(Ctx *c, const SsirGlobalVar *g) {
    wgsl_compiler_assert(c != NULL, "sts_emit_global_var: c is NULL");
    wgsl_compiler_assert(g != NULL, "sts_emit_global_var: g is NULL");
    uint32_t type_spv = sts_emit_type(c, g->type);
    uint32_t id = get_spv_id(c, g->id);

    /* Get storage class from pointer type */
    SsirType *ptr_type = ssir_get_type((SsirModule *)c->mod, g->type);
    SpvStorageClass sc = SpvStorageClassPrivate;
    if (ptr_type && ptr_type->kind == SSIR_TYPE_PTR) {
        sc = addr_space_to_storage_class(ptr_type->ptr.space);
    }

    /* Emit OpVariable */
    StsWordBuf *wb = &c->sections.globals;
    sts_emit_op(wb, SpvOpVariable, g->has_initializer ? 5 : 4);
    wb_push(wb, type_spv);
    wb_push(wb, id);
    wb_push(wb, sc);
    if (g->has_initializer) {
        wb_push(wb, get_spv_id(c, g->initializer));
    }

    /* Emit debug name */
    sts_emit_name(c, id, g->name);

    /* Emit decorations */
    if (g->has_group) {
        sts_emit_decorate(c, id, SpvDecorationDescriptorSet, &g->group, 1);
    }
    if (g->has_binding) {
        sts_emit_decorate(c, id, SpvDecorationBinding, &g->binding, 1);
    }
    if (g->has_location) {
        sts_emit_decorate(c, id, SpvDecorationLocation, &g->location, 1);
    }
    if (g->builtin != SSIR_BUILTIN_NONE) {
        uint32_t b = builtin_var_to_spv(g->builtin);
        sts_emit_decorate(c, id, SpvDecorationBuiltIn, &b, 1);
    }
    if (g->interp == SSIR_INTERP_FLAT) {
        sts_emit_decorate(c, id, SpvDecorationFlat, NULL, 0);
    } else if (g->interp == SSIR_INTERP_LINEAR) {
        sts_emit_decorate(c, id, SpvDecorationNoPerspective, NULL, 0);
    }
    if (g->interp_sampling == SSIR_INTERP_SAMPLING_CENTROID) {
        sts_emit_decorate(c, id, SpvDecorationCentroid, NULL, 0);
    } else if (g->interp_sampling == SSIR_INTERP_SAMPLING_SAMPLE) {
        sts_emit_decorate(c, id, SpvDecorationSample, NULL, 0);
    }
    if (g->non_writable) {
        sts_emit_decorate(c, id, SpvDecorationNonWritable, NULL, 0);
    }
    if (g->invariant) {
        sts_emit_decorate(c, id, SpvDecorationInvariant, NULL, 0);
    }

    /* Block decoration for uniform/storage/push-constant buffers */
    if (sc == SpvStorageClassUniform || sc == SpvStorageClassStorageBuffer ||
        sc == SpvStorageClassPushConstant) {
        if (ptr_type && ptr_type->kind == SSIR_TYPE_PTR) {
            SsirType *pointee = ssir_get_type((SsirModule *)c->mod, ptr_type->ptr.pointee);
            if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
                uint32_t struct_spv = sts_emit_type(c, ptr_type->ptr.pointee);
                sts_emit_block_decoration(c, struct_spv);
                /* Ensure ArrayStride on array members that lack it */
                for (uint32_t mi = 0; mi < pointee->struc.member_count; ++mi) {
                    SsirType *mt = ssir_get_type((SsirModule *)c->mod, pointee->struc.members[mi]);
                    if (mt && mt->kind == SSIR_TYPE_ARRAY && !mt->array.stride) {
                        uint32_t arr_spv = sts_emit_type(c, pointee->struc.members[mi]);
                        uint32_t stride = compute_array_stride(c, mt->array.elem);
                        if (stride)
                            sts_emit_decorate(c, arr_spv, SpvDecorationArrayStride, &stride, 1);
                    }
                }
            }
        }
    }

    /* Record type for this global */
    set_ssir_type(c, g->id, g->type);

    return id;
}

/* ============================================================================
 * Instruction Emission
 * ============================================================================ */

/* Type helper to check if SSIR type is float-based */
// c nonnull
static int is_float_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_float_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16 || t->kind == SSIR_TYPE_F64) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_float_ssir_type(c, t->vec.elem);
    }
    if (t->kind == SSIR_TYPE_MAT) {
        SsirType *col = ssir_get_type((SsirModule *)c->mod, t->mat.elem);
        if (col && col->kind == SSIR_TYPE_VEC) {
            return is_float_ssir_type(c, col->vec.elem);
        }
    }
    return 0;
}

// c nonnull
static int is_signed_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_signed_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_I8 || t->kind == SSIR_TYPE_I16 ||
        t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_I64) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_signed_ssir_type(c, t->vec.elem);
    }
    return 0;
}

// c nonnull
static int is_unsigned_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_unsigned_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_U8 || t->kind == SSIR_TYPE_U16 ||
        t->kind == SSIR_TYPE_U32 || t->kind == SSIR_TYPE_U64) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_unsigned_ssir_type(c, t->vec.elem);
    }
    return 0;
}

// c nonnull
static int is_bool_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_bool_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_BOOL) return 1;
    if (t->kind == SSIR_TYPE_VEC) {
        return is_bool_ssir_type(c, t->vec.elem);
    }
    return 0;
}

// c nonnull
static int is_matrix_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_matrix_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    return t && t->kind == SSIR_TYPE_MAT;
}

// c nonnull
static int is_vector_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_vector_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    return t && t->kind == SSIR_TYPE_VEC;
}

// c nonnull
static int is_scalar_ssir_type(Ctx *c, uint32_t type_id) {
    wgsl_compiler_assert(c != NULL, "is_scalar_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, type_id);
    if (!t) return 0;
    return t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16 ||
           t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_U32 ||
           t->kind == SSIR_TYPE_BOOL;
}

/* Map SSIR builtin to GLSL.std.450 instruction */
static int builtin_to_glsl_op(SsirBuiltinId id) {
    switch (id) {
        case SSIR_BUILTIN_SIN: return GLSLstd450Sin;
        case SSIR_BUILTIN_COS: return GLSLstd450Cos;
        case SSIR_BUILTIN_TAN: return GLSLstd450Tan;
        case SSIR_BUILTIN_ASIN: return GLSLstd450Asin;
        case SSIR_BUILTIN_ACOS: return GLSLstd450Acos;
        case SSIR_BUILTIN_ATAN: return GLSLstd450Atan;
        case SSIR_BUILTIN_ATAN2: return GLSLstd450Atan2;
        case SSIR_BUILTIN_SINH: return GLSLstd450Sinh;
        case SSIR_BUILTIN_COSH: return GLSLstd450Cosh;
        case SSIR_BUILTIN_TANH: return GLSLstd450Tanh;
        case SSIR_BUILTIN_ASINH: return GLSLstd450Asinh;
        case SSIR_BUILTIN_ACOSH: return GLSLstd450Acosh;
        case SSIR_BUILTIN_ATANH: return GLSLstd450Atanh;
        case SSIR_BUILTIN_EXP: return GLSLstd450Exp;
        case SSIR_BUILTIN_EXP2: return GLSLstd450Exp2;
        case SSIR_BUILTIN_LOG: return GLSLstd450Log;
        case SSIR_BUILTIN_LOG2: return GLSLstd450Log2;
        case SSIR_BUILTIN_POW: return GLSLstd450Pow;
        case SSIR_BUILTIN_SQRT: return GLSLstd450Sqrt;
        case SSIR_BUILTIN_INVERSESQRT: return GLSLstd450InverseSqrt;
        case SSIR_BUILTIN_FLOOR: return GLSLstd450Floor;
        case SSIR_BUILTIN_CEIL: return GLSLstd450Ceil;
        case SSIR_BUILTIN_ROUND: return GLSLstd450Round;
        case SSIR_BUILTIN_TRUNC: return GLSLstd450Trunc;
        case SSIR_BUILTIN_FRACT: return GLSLstd450Fract;
        case SSIR_BUILTIN_SIGN: return GLSLstd450FSign;
        case SSIR_BUILTIN_LENGTH: return GLSLstd450Length;
        case SSIR_BUILTIN_DISTANCE: return GLSLstd450Distance;
        case SSIR_BUILTIN_NORMALIZE: return GLSLstd450Normalize;
        case SSIR_BUILTIN_FACEFORWARD: return GLSLstd450FaceForward;
        case SSIR_BUILTIN_REFLECT: return GLSLstd450Reflect;
        case SSIR_BUILTIN_REFRACT: return GLSLstd450Refract;
        case SSIR_BUILTIN_CROSS: return GLSLstd450Cross;
        case SSIR_BUILTIN_MIX: return GLSLstd450FMix;
        case SSIR_BUILTIN_STEP: return GLSLstd450Step;
        case SSIR_BUILTIN_SMOOTHSTEP: return GLSLstd450SmoothStep;
        case SSIR_BUILTIN_FMA: return GLSLstd450Fma;
        case SSIR_BUILTIN_DEGREES: return GLSLstd450Degrees;
        case SSIR_BUILTIN_RADIANS: return GLSLstd450Radians;
        case SSIR_BUILTIN_MODF: return GLSLstd450Modf;
        case SSIR_BUILTIN_FREXP: return GLSLstd450Frexp;
        case SSIR_BUILTIN_LDEXP: return GLSLstd450Ldexp;
        case SSIR_BUILTIN_DETERMINANT: return GLSLstd450Determinant;
        case SSIR_BUILTIN_PACK4X8SNORM: return GLSLstd450PackSnorm4x8;
        case SSIR_BUILTIN_PACK4X8UNORM: return GLSLstd450PackUnorm4x8;
        case SSIR_BUILTIN_PACK2X16SNORM: return GLSLstd450PackSnorm2x16;
        case SSIR_BUILTIN_PACK2X16UNORM: return GLSLstd450PackUnorm2x16;
        case SSIR_BUILTIN_PACK2X16FLOAT: return GLSLstd450PackHalf2x16;
        case SSIR_BUILTIN_UNPACK4X8SNORM: return GLSLstd450UnpackSnorm4x8;
        case SSIR_BUILTIN_UNPACK4X8UNORM: return GLSLstd450UnpackUnorm4x8;
        case SSIR_BUILTIN_UNPACK2X16SNORM: return GLSLstd450UnpackSnorm2x16;
        case SSIR_BUILTIN_UNPACK2X16UNORM: return GLSLstd450UnpackUnorm2x16;
        case SSIR_BUILTIN_UNPACK2X16FLOAT: return GLSLstd450UnpackHalf2x16;
        default: return -1;
    }
}

// c nonnull
static uint32_t get_unsigned_ssir_type(Ctx *c, uint32_t ssir_type) {
    wgsl_compiler_assert(c != NULL, "get_unsigned_ssir_type: c is NULL");
    SsirType *t = ssir_get_type((SsirModule *)c->mod, ssir_type);
    if (!t) return 0;
    if (t->kind == SSIR_TYPE_I32) return ssir_type_u32((SsirModule *)c->mod);
    if (t->kind == SSIR_TYPE_I64) return ssir_type_u64((SsirModule *)c->mod);
    if (t->kind == SSIR_TYPE_I16) return ssir_type_u16((SsirModule *)c->mod);
    if (t->kind == SSIR_TYPE_I8) return ssir_type_u8((SsirModule *)c->mod);
    if (t->kind == SSIR_TYPE_VEC) {
        uint32_t unsigned_elem = get_unsigned_ssir_type(c, t->vec.elem);
        if (unsigned_elem) return ssir_type_vec((SsirModule *)c->mod, unsigned_elem, t->vec.size);
    }
    return 0;
}

// c nonnull
static void emit_signed_int_binop(Ctx *c, SpvOp spv_op,
    uint32_t ssir_result, uint32_t type_spv,
    uint32_t ssir_type, uint32_t op0, uint32_t op1) {
    wgsl_compiler_assert(c != NULL, "emit_signed_int_binop: c is NULL");
    StsWordBuf *wb = &c->sections.functions;
    uint32_t unsigned_ssir = get_unsigned_ssir_type(c, ssir_type);
    uint32_t unsigned_spv = sts_emit_type(c, unsigned_ssir);
    uint32_t cast_a = sts_fresh_id(c);
    uint32_t cast_b = sts_fresh_id(c);
    uint32_t op_result = sts_fresh_id(c);
    uint32_t result_spv = sts_fresh_id(c);
    set_spv_id(c, ssir_result, result_spv);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, unsigned_spv);
    wb_push(wb, cast_a);
    wb_push(wb, op0);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, unsigned_spv);
    wb_push(wb, cast_b);
    wb_push(wb, op1);
    sts_emit_op(wb, spv_op, 5);
    wb_push(wb, unsigned_spv);
    wb_push(wb, op_result);
    wb_push(wb, cast_a);
    wb_push(wb, cast_b);
    sts_emit_op(wb, SpvOpBitcast, 4);
    wb_push(wb, type_spv);
    wb_push(wb, result_spv);
    wb_push(wb, op_result);
}

// c nonnull, inst nonnull
static int emit_instruction(Ctx *c, const SsirInst *inst, uint32_t func_type_hint) {
    (void)func_type_hint;
    wgsl_compiler_assert(c != NULL, "emit_instruction: c is NULL");
    wgsl_compiler_assert(inst != NULL, "emit_instruction: inst is NULL");
    StsWordBuf *wb = &c->sections.functions;
    uint32_t type_spv = inst->type ? sts_emit_type(c, inst->type) : 0;

    /* Get operand types for type-driven opcode selection.
     * For most ops, the result type matches operand types.
     * For comparison ops (result is bool), we need the actual operand type. */
    uint32_t op0_type = 0;
    uint32_t op1_type = 0;
    if (inst->operand_count > 0) {
        op0_type = get_ssir_type(c, inst->operands[0]);
        if (op0_type == 0) {
            op0_type = inst->type;
        }
    }
    if (inst->operand_count > 1) {
        op1_type = get_ssir_type(c, inst->operands[1]);
        if (op1_type == 0) {
            op1_type = inst->type;
        }
    }

    /* Defer result ID allocation for signed int binops (they allocate lazily) */
    int defer_result = 0;
    if ((inst->op == SSIR_OP_ADD || inst->op == SSIR_OP_SUB || inst->op == SSIR_OP_MUL) &&
        !is_float_ssir_type(c, op0_type) && is_signed_ssir_type(c, op0_type)) {
        defer_result = 1;
    }
    uint32_t result_spv = (!defer_result && inst->result) ? get_spv_id(c, inst->result) : 0;

    /* Record result type for this instruction (for future lookups) */
    if (inst->result && inst->type) {
        set_ssir_type(c, inst->result, inst->type);
    }

    switch (inst->op) {
        /* Arithmetic */
        case SSIR_OP_ADD:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFAdd, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpIAdd, inst->result, type_spv, inst->type,
                    get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpIAdd, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_SUB:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFSub, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpISub, inst->result, type_spv, inst->type,
                    get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpISub, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_MUL:
            if (is_float_ssir_type(c, op0_type)) {
                /* Check for vector * scalar or scalar * vector */
                int op0_is_vec = is_vector_ssir_type(c, op0_type);
                int op1_is_vec = is_vector_ssir_type(c, op1_type);
                int op0_is_scalar = is_scalar_ssir_type(c, op0_type);
                int op1_is_scalar = is_scalar_ssir_type(c, op1_type);

                if (op0_is_vec && op1_is_scalar) {
                    /* vec * scalar */
                    sts_emit_op(wb, SpvOpVectorTimesScalar, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[0]));
                    wb_push(wb, get_spv_id(c, inst->operands[1]));
                } else if (op0_is_scalar && op1_is_vec) {
                    /* scalar * vec -> swap operands for VectorTimesScalar */
                    sts_emit_op(wb, SpvOpVectorTimesScalar, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[1])); /* vec first */
                    wb_push(wb, get_spv_id(c, inst->operands[0])); /* scalar second */
                } else {
                    sts_emit_op(wb, SpvOpFMul, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[0]));
                    wb_push(wb, get_spv_id(c, inst->operands[1]));
                }
            } else if (is_signed_ssir_type(c, op0_type)) {
                emit_signed_int_binop(c, SpvOpIMul, inst->result, type_spv, inst->type,
                    get_spv_id(c, inst->operands[0]), get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpIMul, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;

        case SSIR_OP_DIV:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFDiv, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSDiv, 5);
            } else {
                sts_emit_op(wb, SpvOpUDiv, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_MOD:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFMod, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSMod, 5);
            } else {
                sts_emit_op(wb, SpvOpUMod, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_REM:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFRem, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSRem, 5);
            } else {
                sts_emit_op(wb, SpvOpUMod, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NEG:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFNegate, 4);
            } else {
                sts_emit_op(wb, SpvOpSNegate, 4);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Matrix */
        case SSIR_OP_MAT_MUL: {
            /* Determine operation based on operand types */
            int op0_mat = is_matrix_ssir_type(c, op0_type);
            int op1_mat = is_matrix_ssir_type(c, op1_type);
            int op0_vec = is_vector_ssir_type(c, op0_type);
            int op1_vec = is_vector_ssir_type(c, op1_type);
            int op0_scalar = is_scalar_ssir_type(c, op0_type);
            int op1_scalar = is_scalar_ssir_type(c, op1_type);

            if (op0_mat && op1_mat) {
                sts_emit_op(wb, SpvOpMatrixTimesMatrix, 5);
            } else if (op0_mat && op1_vec) {
                sts_emit_op(wb, SpvOpMatrixTimesVector, 5);
            } else if (op0_vec && op1_mat) {
                sts_emit_op(wb, SpvOpVectorTimesMatrix, 5);
            } else if (op0_mat && op1_scalar) {
                sts_emit_op(wb, SpvOpMatrixTimesScalar, 5);
            } else if (op0_scalar && op1_mat) {
                /* scalar * mat -> emit as MatrixTimesScalar with swapped operands */
                sts_emit_op(wb, SpvOpMatrixTimesScalar, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[1]));
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                break;
            } else {
                sts_emit_op(wb, SpvOpFMul, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;
        }

        case SSIR_OP_MAT_TRANSPOSE:
            sts_emit_op(wb, SpvOpTranspose, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Bitwise */
        case SSIR_OP_BIT_AND:
            sts_emit_op(wb, SpvOpBitwiseAnd, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_OR:
            sts_emit_op(wb, SpvOpBitwiseOr, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_XOR:
            sts_emit_op(wb, SpvOpBitwiseXor, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_BIT_NOT:
            sts_emit_op(wb, SpvOpNot, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_SHL:
            sts_emit_op(wb, SpvOpShiftLeftLogical, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_SHR:
            if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpShiftRightArithmetic, 5);
            } else {
                sts_emit_op(wb, SpvOpShiftRightLogical, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_SHR_LOGICAL:
            sts_emit_op(wb, SpvOpShiftRightLogical, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        /* Comparison */
        case SSIR_OP_EQ:
            if (is_bool_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpLogicalEqual, 5);
            } else if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpIEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NE:
            if (is_bool_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpLogicalNotEqual, 5);
            } else if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdNotEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpINotEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_LT:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdLessThan, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSLessThan, 5);
            } else {
                sts_emit_op(wb, SpvOpULessThan, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_LE:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdLessThanEqual, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSLessThanEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpULessThanEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_GT:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdGreaterThan, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSGreaterThan, 5);
            } else {
                sts_emit_op(wb, SpvOpUGreaterThan, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_GE:
            if (is_float_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpFOrdGreaterThanEqual, 5);
            } else if (is_signed_ssir_type(c, op0_type)) {
                sts_emit_op(wb, SpvOpSGreaterThanEqual, 5);
            } else {
                sts_emit_op(wb, SpvOpUGreaterThanEqual, 5);
            }
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        /* Logical */
        case SSIR_OP_AND:
            sts_emit_op(wb, SpvOpLogicalAnd, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_OR:
            sts_emit_op(wb, SpvOpLogicalOr, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_NOT:
            sts_emit_op(wb, SpvOpLogicalNot, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Composite */
        case SSIR_OP_CONSTRUCT: {
            uint32_t count = inst->operand_count;
            if (inst->extra_count > 0) count = inst->extra_count;
            sts_emit_op(wb, SpvOpCompositeConstruct, 3 + count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            if (inst->extra_count > 0) {
                for (uint32_t i = 0; i < inst->extra_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->extra[i]));
                }
            } else {
                for (uint8_t i = 0; i < inst->operand_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->operands[i]));
                }
            }
            break;
        }

        case SSIR_OP_EXTRACT:
            sts_emit_op(wb, SpvOpCompositeExtract, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, inst->operands[1]); /* literal index */
            break;

        case SSIR_OP_INSERT:
            sts_emit_op(wb, SpvOpCompositeInsert, 6);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* object to insert */
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* composite */
            wb_push(wb, inst->operands[2]);                /* literal index */
            break;

        case SSIR_OP_SHUFFLE: {
            sts_emit_op(wb, SpvOpVectorShuffle, 5 + inst->extra_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            for (uint32_t i = 0; i < inst->extra_count; ++i) {
                wb_push(wb, inst->extra[i]); /* literal indices */
            }
            break;
        }

        case SSIR_OP_SPLAT: {
            /* OpCompositeConstruct with repeated scalar */
            SsirType *t = ssir_get_type((SsirModule *)c->mod, inst->type);
            uint32_t count = t && t->kind == SSIR_TYPE_VEC ? t->vec.size : 4;
            sts_emit_op(wb, SpvOpCompositeConstruct, 3 + count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            uint32_t scalar = get_spv_id(c, inst->operands[0]);
            for (uint32_t i = 0; i < count; ++i) {
                wb_push(wb, scalar);
            }
            break;
        }

        case SSIR_OP_EXTRACT_DYN:
            sts_emit_op(wb, SpvOpVectorExtractDynamic, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            break;

        case SSIR_OP_INSERT_DYN:
            sts_emit_op(wb, SpvOpVectorInsertDynamic, 6);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* vector */
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* component */
            wb_push(wb, get_spv_id(c, inst->operands[2])); /* index */
            break;

        /* Memory */
        case SSIR_OP_LOAD: {
            /* PRE: operand_count >= 1 */
            wgsl_compiler_assert(inst->operand_count >= 1, "SSIR_OP_LOAD: operand_count < 1");
            /* Check if pointer is PhysicalStorageBuffer — requires Aligned operand */
            uint32_t load_ptr_ssir_type = get_ssir_type(c, inst->operands[0]);
            SsirType *load_ptr_ty = load_ptr_ssir_type ? ssir_get_type((SsirModule *)c->mod, load_ptr_ssir_type) : NULL;
            bool load_is_psb = load_ptr_ty && load_ptr_ty->kind == SSIR_TYPE_PTR &&
                               load_ptr_ty->ptr.space == SSIR_ADDR_PHYSICAL_STORAGE_BUFFER;
            if (load_is_psb) {
                /* OpLoad with Aligned memory operand */
                SsirType *pointee = ssir_get_type((SsirModule *)c->mod, load_ptr_ty->ptr.pointee);
                uint32_t align = 4; /* default */
                if (pointee) {
                    switch (pointee->kind) {
                    case SSIR_TYPE_U8: case SSIR_TYPE_I8: align = 1; break;
                    case SSIR_TYPE_U16: case SSIR_TYPE_I16: case SSIR_TYPE_F16: align = 2; break;
                    case SSIR_TYPE_U64: case SSIR_TYPE_I64: case SSIR_TYPE_F64: align = 8; break;
                    default: align = 4; break;
                    }
                }
                sts_emit_op(wb, SpvOpLoad, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, SpvMemoryAccessAlignedMask);
                wb_push(wb, align);
            } else {
                sts_emit_op(wb, SpvOpLoad, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
            }
            break;
        }

        case SSIR_OP_STORE: {
            /* PRE: operand_count >= 2 */
            wgsl_compiler_assert(inst->operand_count >= 2, "SSIR_OP_STORE: operand_count < 2");
            /* Check if pointer is PhysicalStorageBuffer — requires Aligned operand */
            uint32_t store_ptr_ssir_type = get_ssir_type(c, inst->operands[0]);
            SsirType *store_ptr_ty = store_ptr_ssir_type ? ssir_get_type((SsirModule *)c->mod, store_ptr_ssir_type) : NULL;
            bool store_is_psb = store_ptr_ty && store_ptr_ty->kind == SSIR_TYPE_PTR &&
                                store_ptr_ty->ptr.space == SSIR_ADDR_PHYSICAL_STORAGE_BUFFER;
            if (store_is_psb) {
                SsirType *pointee = ssir_get_type((SsirModule *)c->mod, store_ptr_ty->ptr.pointee);
                uint32_t align = 4;
                if (pointee) {
                    switch (pointee->kind) {
                    case SSIR_TYPE_U8: case SSIR_TYPE_I8: align = 1; break;
                    case SSIR_TYPE_U16: case SSIR_TYPE_I16: case SSIR_TYPE_F16: align = 2; break;
                    case SSIR_TYPE_U64: case SSIR_TYPE_I64: case SSIR_TYPE_F64: align = 8; break;
                    default: align = 4; break;
                    }
                }
                sts_emit_op(wb, SpvOpStore, 5);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
                wb_push(wb, SpvMemoryAccessAlignedMask);
                wb_push(wb, align);
            } else {
                sts_emit_op(wb, SpvOpStore, 3);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;
        }

        case SSIR_OP_ACCESS: {
            uint32_t idx_count = inst->extra_count;
            sts_emit_op(wb, SpvOpAccessChain, 4 + idx_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            for (uint32_t i = 0; i < idx_count; ++i) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));
            }
            break;
        }

        case SSIR_OP_ARRAY_LEN:
            sts_emit_op(wb, SpvOpArrayLength, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, 0); /* member index - typically 0 for runtime arrays */
            break;

        /* Control Flow */
        case SSIR_OP_BRANCH:
            sts_emit_op(wb, SpvOpBranch, 2);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_BRANCH_COND:
            /* Emit OpSelectionMerge if merge block is specified (operands[3]) */
            if (inst->operand_count >= 4 && inst->operands[3] != 0) {
                sts_emit_op(wb, SpvOpSelectionMerge, 3);
                wb_push(wb, get_spv_id(c, inst->operands[3]));
                wb_push(wb, 0); /* SelectionControlMaskNone */
            }
            sts_emit_op(wb, SpvOpBranchConditional, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;

        case SSIR_OP_SWITCH: {
            sts_emit_op(wb, SpvOpSwitch, 3 + inst->extra_count);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* default */
            for (uint32_t i = 0; i < inst->extra_count; i += 2) {
                wb_push(wb, inst->extra[i]);                    /* literal value */
                wb_push(wb, get_spv_id(c, inst->extra[i + 1])); /* label */
            }
            break;
        }

        case SSIR_OP_PHI: {
            sts_emit_op(wb, SpvOpPhi, 3 + inst->extra_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            for (uint32_t i = 0; i < inst->extra_count; i += 2) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));     /* value */
                wb_push(wb, get_spv_id(c, inst->extra[i + 1])); /* parent block */
            }
            break;
        }

        case SSIR_OP_RETURN:
            sts_emit_op(wb, SpvOpReturnValue, 2);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_RETURN_VOID:
            sts_emit_op(wb, SpvOpReturn, 1);
            break;

        case SSIR_OP_UNREACHABLE:
            sts_emit_op(wb, SpvOpUnreachable, 1);
            break;

        case SSIR_OP_LOOP_MERGE:
            sts_emit_op(wb, SpvOpLoopMerge, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* merge block */
            wb_push(wb, get_spv_id(c, inst->operands[1])); /* continue block */
            wb_push(wb, 0);                                /* LoopControlMaskNone */
            break;

        case SSIR_OP_SELECTION_MERGE:
            sts_emit_op(wb, SpvOpSelectionMerge, 3);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* merge block */
            wb_push(wb, 0);                                /* SelectionControlMaskNone */
            break;

        case SSIR_OP_DISCARD:
            sts_emit_op(wb, SpvOpKill, 1);
            break;

        /* Call */
        case SSIR_OP_CALL: {
            uint32_t arg_count = inst->extra_count;
            sts_emit_op(wb, SpvOpFunctionCall, 4 + arg_count);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0])); /* callee */
            for (uint32_t i = 0; i < arg_count; ++i) {
                wb_push(wb, get_spv_id(c, inst->extra[i]));
            }
            break;
        }

        case SSIR_OP_BUILTIN: {
            SsirBuiltinId builtin_id = (SsirBuiltinId)inst->operands[0];
            int glsl_op = builtin_to_glsl_op(builtin_id);
            sts_ensure_glsl_import(c);

            if (builtin_id == SSIR_BUILTIN_DOT) {
                /* OpDot is native */
                sts_emit_op(wb, SpvOpDot, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
                wb_push(wb, get_spv_id(c, inst->extra[1]));
            } else if (builtin_id == SSIR_BUILTIN_ABS) {
                /* Type-dependent */
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FAbs);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SAbs);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                }
            } else if (builtin_id == SSIR_BUILTIN_MIN) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UMin);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                }
            } else if (builtin_id == SSIR_BUILTIN_MAX) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UMax);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                }
            } else if (builtin_id == SSIR_BUILTIN_CLAMP) {
                if (is_float_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450FClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                } else if (is_signed_ssir_type(c, inst->type)) {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450SClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                } else {
                    sts_emit_op(wb, SpvOpExtInst, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, c->glsl_ext_id);
                    wb_push(wb, GLSLstd450UClamp);
                    wb_push(wb, get_spv_id(c, inst->extra[0]));
                    wb_push(wb, get_spv_id(c, inst->extra[1]));
                    wb_push(wb, get_spv_id(c, inst->extra[2]));
                }
            } else if (builtin_id == SSIR_BUILTIN_SELECT) {
                sts_emit_op(wb, SpvOpSelect, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[2])); /* condition */
                wb_push(wb, get_spv_id(c, inst->extra[1])); /* true value */
                wb_push(wb, get_spv_id(c, inst->extra[0])); /* false value */
            } else if (builtin_id == SSIR_BUILTIN_ALL) {
                sts_emit_op(wb, SpvOpAll, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ANY) {
                sts_emit_op(wb, SpvOpAny, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_DPDX || builtin_id == SSIR_BUILTIN_DPDX_COARSE || builtin_id == SSIR_BUILTIN_DPDX_FINE) {
                sts_emit_op(wb, SpvOpDPdx, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_DPDY || builtin_id == SSIR_BUILTIN_DPDY_COARSE || builtin_id == SSIR_BUILTIN_DPDY_FINE) {
                sts_emit_op(wb, SpvOpDPdy, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_FWIDTH) {
                sts_emit_op(wb, SpvOpFwidth, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ISINF) {
                sts_emit_op(wb, SpvOpIsInf, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_ISNAN) {
                sts_emit_op(wb, SpvOpIsNan, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_TRANSPOSE) {
                sts_emit_op(wb, SpvOpTranspose, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_COUNTBITS) {
                sts_emit_op(wb, SpvOpBitCount, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_REVERSEBITS) {
                sts_emit_op(wb, SpvOpBitReverse, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_FIRSTLEADINGBIT) {
                sts_emit_op(wb, SpvOpExtInst, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, c->glsl_ext_id);
                wb_push(wb, is_signed_ssir_type(c, inst->type) ? GLSLstd450FindSMsb : GLSLstd450FindUMsb);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (builtin_id == SSIR_BUILTIN_FIRSTTRAILINGBIT) {
                sts_emit_op(wb, SpvOpExtInst, 6);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, c->glsl_ext_id);
                wb_push(wb, GLSLstd450FindILsb);
                wb_push(wb, get_spv_id(c, inst->extra[0]));
            } else if (glsl_op >= 0) {
                /* Generic GLSL.std.450 function */
                sts_emit_op(wb, SpvOpExtInst, 5 + inst->extra_count);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, c->glsl_ext_id);
                wb_push(wb, glsl_op);
                for (uint32_t i = 0; i < inst->extra_count; ++i) {
                    wb_push(wb, get_spv_id(c, inst->extra[i]));
                }
            }
            break;
        }

        /* Conversion */
        case SSIR_OP_CONVERT: {
            /* Get source operand type */
            uint32_t src_type = get_ssir_type(c, inst->operands[0]);
            if (src_type == 0) src_type = op0_type;

            int from_float = is_float_ssir_type(c, src_type);
            int from_signed = is_signed_ssir_type(c, src_type);
            int from_unsigned = is_unsigned_ssir_type(c, src_type);
            int to_float = is_float_ssir_type(c, inst->type);
            int to_signed = is_signed_ssir_type(c, inst->type);
            int to_unsigned = is_unsigned_ssir_type(c, inst->type);

            SpvOp conv_op = SpvOpNop;

            if (from_float && to_signed) {
                conv_op = SpvOpConvertFToS;
            } else if (from_float && to_unsigned) {
                conv_op = SpvOpConvertFToU;
            } else if (from_signed && to_float) {
                conv_op = SpvOpConvertSToF;
            } else if (from_unsigned && to_float) {
                conv_op = SpvOpConvertUToF;
            } else if ((from_signed && to_signed) || (from_unsigned && to_unsigned)) {
                conv_op = from_signed ? SpvOpSConvert : SpvOpUConvert;
            } else if (from_float && to_float) {
                conv_op = SpvOpFConvert;
            } else {
                conv_op = SpvOpBitcast; /* sign reinterpretation or fallback */
            }

            sts_emit_op(wb, conv_op, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;
        }

        case SSIR_OP_BITCAST: {
            /* Check if this is u64 -> PhysicalStorageBuffer pointer conversion */
            SsirType *result_ty = ssir_get_type((SsirModule *)c->mod, inst->type);
            bool is_u_to_ptr = result_ty && result_ty->kind == SSIR_TYPE_PTR &&
                               result_ty->ptr.space == SSIR_ADDR_PHYSICAL_STORAGE_BUFFER;
            SpvOp bc_op = is_u_to_ptr ? SpvOpConvertUToPtr : SpvOpBitcast;
            sts_emit_op(wb, bc_op, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;
        }

        /* Texture operations */
        case SSIR_OP_TEX_SAMPLE:
        case SSIR_OP_TEX_SAMPLE_BIAS:
        case SSIR_OP_TEX_SAMPLE_LEVEL:
        case SSIR_OP_TEX_SAMPLE_GRAD:
        case SSIR_OP_TEX_SAMPLE_CMP:
        case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
        case SSIR_OP_TEX_SAMPLE_OFFSET:
        case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
        case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
        case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
        case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
        case SSIR_OP_TEX_GATHER:
        case SSIR_OP_TEX_GATHER_CMP:
        case SSIR_OP_TEX_GATHER_OFFSET: {
            /* Create OpSampledImage from texture + sampler */
            uint32_t tex_ssir_type = get_ssir_type(c, inst->operands[0]);
            uint32_t img_spv = sts_emit_type(c, tex_ssir_type);
            uint32_t si_type = sts_emit_type_sampled_image(c, img_spv);
            uint32_t si_id = sts_fresh_id(c);
            sts_emit_op(wb, SpvOpSampledImage, 5);
            wb_push(wb, si_type);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));

            uint32_t coord_spv = get_spv_id(c, inst->operands[2]);

            switch (inst->op) {
                case SSIR_OP_TEX_SAMPLE:
                    sts_emit_op(wb, SpvOpImageSampleImplicitLod, 5);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    break;
                case SSIR_OP_TEX_SAMPLE_BIAS:
                    sts_emit_op(wb, SpvOpImageSampleImplicitLod, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x1); /* Bias */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_SAMPLE_LEVEL:
                    sts_emit_op(wb, SpvOpImageSampleExplicitLod, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x2); /* Lod */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_SAMPLE_GRAD:
                    sts_emit_op(wb, SpvOpImageSampleExplicitLod, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x4); /* Grad */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                case SSIR_OP_TEX_SAMPLE_CMP:
                    sts_emit_op(wb, SpvOpImageSampleDrefImplicitLod, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_SAMPLE_CMP_LEVEL:
                    sts_emit_op(wb, SpvOpImageSampleDrefExplicitLod, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, 0x2); /* Lod */
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                case SSIR_OP_TEX_SAMPLE_OFFSET:
                    sts_emit_op(wb, SpvOpImageSampleImplicitLod, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x10); /* ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_SAMPLE_BIAS_OFFSET:
                    sts_emit_op(wb, SpvOpImageSampleImplicitLod, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x1 | 0x10); /* Bias | ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                case SSIR_OP_TEX_SAMPLE_LEVEL_OFFSET:
                    sts_emit_op(wb, SpvOpImageSampleExplicitLod, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x2 | 0x10); /* Lod | ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                case SSIR_OP_TEX_SAMPLE_GRAD_OFFSET:
                    sts_emit_op(wb, SpvOpImageSampleExplicitLod, 9);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, 0x4 | 0x10); /* Grad | ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    wb_push(wb, get_spv_id(c, inst->operands[5]));
                    break;
                case SSIR_OP_TEX_SAMPLE_CMP_OFFSET:
                    sts_emit_op(wb, SpvOpImageSampleDrefImplicitLod, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, 0x10); /* ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                case SSIR_OP_TEX_GATHER:
                    sts_emit_op(wb, SpvOpImageGather, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_GATHER_CMP:
                    sts_emit_op(wb, SpvOpImageDrefGather, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    break;
                case SSIR_OP_TEX_GATHER_OFFSET:
                    sts_emit_op(wb, SpvOpImageGather, 8);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, si_id);
                    wb_push(wb, coord_spv);
                    wb_push(wb, get_spv_id(c, inst->operands[3]));
                    wb_push(wb, 0x10); /* ConstOffset */
                    wb_push(wb, get_spv_id(c, inst->operands[4]));
                    break;
                default: break;
            }
            break;
        }

        case SSIR_OP_TEX_LOAD: {
            /* Use OpImageFetch for sampled textures, OpImageRead for storage */
            uint32_t tex_ssir_type = get_ssir_type(c, inst->operands[0]);
            SsirType *tex_type = ssir_get_type((SsirModule *)c->mod, tex_ssir_type);
            int is_sampled = (tex_type && (tex_type->kind == SSIR_TYPE_TEXTURE || tex_type->kind == SSIR_TYPE_TEXTURE_DEPTH));
            uint32_t lod_operand = (inst->operand_count >= 3 && inst->operands[2]) ? get_spv_id(c, inst->operands[2]) : 0;
            if (is_sampled && lod_operand) {
                sts_emit_op(wb, SpvOpImageFetch, 7);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
                wb_push(wb, 0x2); /* ImageOperands: Lod */
                wb_push(wb, lod_operand);
            } else if (is_sampled) {
                sts_emit_op(wb, SpvOpImageFetch, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            } else {
                sts_emit_op(wb, SpvOpImageRead, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;
        }

        case SSIR_OP_TEX_STORE:
            sts_emit_op(wb, SpvOpImageWrite, 4);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;

        case SSIR_OP_TEX_SIZE: {
            if (!c->has_image_query_cap) {
                sts_emit_capability(c, SpvCapabilityImageQuery);
                c->has_image_query_cap = 1;
            }
            /* Check if texture is multisampled — use OpImageQuerySize (no lod) */
            uint32_t tex_ssir_type_ts = get_ssir_type(c, inst->operands[0]);
            SsirType *tex_type_ts = ssir_get_type((SsirModule *)c->mod, tex_ssir_type_ts);
            int is_multisampled = (tex_type_ts && tex_type_ts->kind == SSIR_TYPE_TEXTURE &&
                                   (tex_type_ts->texture.dim == SSIR_TEX_MULTISAMPLED_2D));
            int has_level = (inst->operand_count >= 2 && inst->operands[1] != 0);
            if (is_multisampled || !has_level) {
                sts_emit_op(wb, SpvOpImageQuerySize, 4);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
            } else {
                sts_emit_op(wb, SpvOpImageQuerySizeLod, 5);
                wb_push(wb, type_spv);
                wb_push(wb, result_spv);
                wb_push(wb, get_spv_id(c, inst->operands[0]));
                wb_push(wb, get_spv_id(c, inst->operands[1]));
            }
            break;
        }

        case SSIR_OP_TEX_QUERY_LOD: {
            uint32_t tex_ssir_type = get_ssir_type(c, inst->operands[0]);
            uint32_t img_spv = sts_emit_type(c, tex_ssir_type);
            uint32_t si_type = sts_emit_type_sampled_image(c, img_spv);
            uint32_t si_id = sts_fresh_id(c);
            sts_emit_op(wb, SpvOpSampledImage, 5);
            wb_push(wb, si_type);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            wb_push(wb, get_spv_id(c, inst->operands[1]));
            sts_emit_op(wb, SpvOpImageQueryLod, 5);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, si_id);
            wb_push(wb, get_spv_id(c, inst->operands[2]));
            break;
        }

        case SSIR_OP_TEX_QUERY_LEVELS:
            if (!c->has_image_query_cap) {
                sts_emit_capability(c, SpvCapabilityImageQuery);
                c->has_image_query_cap = 1;
            }
            sts_emit_op(wb, SpvOpImageQueryLevels, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        case SSIR_OP_TEX_QUERY_SAMPLES:
            if (!c->has_image_query_cap) {
                sts_emit_capability(c, SpvCapabilityImageQuery);
                c->has_image_query_cap = 1;
            }
            sts_emit_op(wb, SpvOpImageQuerySamples, 4);
            wb_push(wb, type_spv);
            wb_push(wb, result_spv);
            wb_push(wb, get_spv_id(c, inst->operands[0]));
            break;

        /* Sync */
        case SSIR_OP_BARRIER: {
            SsirBarrierScope bscope = (SsirBarrierScope)inst->operands[0];
            switch (bscope) {
                case SSIR_BARRIER_WORKGROUP: {
                    uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                    uint32_t sem = sts_emit_const_u32(c, 0x108); /* AcquireRelease | WorkgroupMemory */
                    sts_emit_op(wb, SpvOpControlBarrier, 4);
                    wb_push(wb, wg);
                    wb_push(wb, wg);
                    wb_push(wb, sem);
                    break;
                }
                case SSIR_BARRIER_STORAGE: {
                    uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                    uint32_t sem = sts_emit_const_u32(c, 0x48); /* AcquireRelease | UniformMemory */
                    sts_emit_op(wb, SpvOpMemoryBarrier, 3);
                    wb_push(wb, wg);
                    wb_push(wb, sem);
                    break;
                }
                case SSIR_BARRIER_SUBGROUP: {
                    uint32_t sg = sts_emit_const_u32(c, SpvScopeSubgroup);
                    uint32_t sem = sts_emit_const_u32(c, 0x108); /* AcquireRelease | WorkgroupMemory */
                    sts_emit_op(wb, SpvOpControlBarrier, 4);
                    wb_push(wb, sg);
                    wb_push(wb, sg);
                    wb_push(wb, sem);
                    break;
                }
                case SSIR_BARRIER_IMAGE: {
                    uint32_t wg = sts_emit_const_u32(c, SpvScopeWorkgroup);
                    uint32_t sem = sts_emit_const_u32(c, 0x808); /* AcquireRelease | ImageMemory */
                    sts_emit_op(wb, SpvOpMemoryBarrier, 3);
                    wb_push(wb, wg);
                    wb_push(wb, sem);
                    break;
                }
            }
            break;
        }

        case SSIR_OP_ATOMIC: {
            SsirAtomicOp aop = (SsirAtomicOp)inst->operands[0];
            uint32_t a_ptr = get_spv_id(c, inst->operands[1]);
            uint32_t a_val = inst->operands[2] ? get_spv_id(c, inst->operands[2]) : 0;
            uint32_t a_cmp = inst->operands[3] ? get_spv_id(c, inst->operands[3]) : 0;
            /* Scope and semantics: use from extended form or defaults */
            uint32_t scope_val = (inst->operand_count >= 6) ? inst->operands[4] : 0; /* SSIR_SCOPE_DEVICE */
            uint32_t sem_val = (inst->operand_count >= 6) ? inst->operands[5] : 0;   /* SSIR_SEMANTICS_RELAXED */
            /* Convert SSIR scope to SPIR-V scope */
            SpvScope spv_scope;
            switch ((SsirMemoryScope)scope_val) {
                case SSIR_SCOPE_WORKGROUP: spv_scope = SpvScopeWorkgroup; break;
                case SSIR_SCOPE_SUBGROUP: spv_scope = SpvScopeSubgroup; break;
                case SSIR_SCOPE_INVOCATION: spv_scope = SpvScopeInvocation; break;
                default: spv_scope = SpvScopeDevice; break;
            }
            /* Convert SSIR semantics to SPIR-V memory semantics */
            SpvMemorySemanticsMask spv_sem;
            switch ((SsirMemorySemantics)sem_val) {
                case SSIR_SEMANTICS_ACQUIRE: spv_sem = SpvMemorySemanticsMaskNone | 0x2; break;         /* Acquire */
                case SSIR_SEMANTICS_RELEASE: spv_sem = SpvMemorySemanticsMaskNone | 0x4; break;         /* Release */
                case SSIR_SEMANTICS_ACQUIRE_RELEASE: spv_sem = SpvMemorySemanticsMaskNone | 0x8; break; /* AcquireRelease */
                case SSIR_SEMANTICS_SEQ_CST: spv_sem = SpvMemorySemanticsMaskNone | 0x10; break;        /* SequentiallyConsistent */
                default: spv_sem = SpvMemorySemanticsMaskNone; break;                                   /* Relaxed */
            }
            /* Emit scope and semantics as constants */
            uint32_t scope_id = sts_emit_const_u32(c, (uint32_t)spv_scope);
            uint32_t sem_id = sts_emit_const_u32(c, (uint32_t)spv_sem);
            uint32_t sem_equal_id = sem_id;                                                        /* For compare exchange, equal semantics */
            uint32_t sem_unequal_id = sts_emit_const_u32(c, (uint32_t)SpvMemorySemanticsMaskNone); /* Unequal gets relaxed */
            switch (aop) {
                case SSIR_ATOMIC_LOAD:
                    sts_emit_op(wb, SpvOpAtomicLoad, 6);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    break;
                case SSIR_ATOMIC_STORE:
                    sts_emit_op(wb, SpvOpAtomicStore, 5);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_ADD:
                    sts_emit_op(wb, SpvOpAtomicIAdd, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_SUB:
                    sts_emit_op(wb, SpvOpAtomicISub, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_MAX:
                    sts_emit_op(wb, is_signed_ssir_type(c, inst->type) ? SpvOpAtomicSMax : SpvOpAtomicUMax, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_MIN:
                    sts_emit_op(wb, is_signed_ssir_type(c, inst->type) ? SpvOpAtomicSMin : SpvOpAtomicUMin, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_AND:
                    sts_emit_op(wb, SpvOpAtomicAnd, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_OR:
                    sts_emit_op(wb, SpvOpAtomicOr, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_XOR:
                    sts_emit_op(wb, SpvOpAtomicXor, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_EXCHANGE:
                    sts_emit_op(wb, SpvOpAtomicExchange, 7);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_id);
                    wb_push(wb, a_val);
                    break;
                case SSIR_ATOMIC_COMPARE_EXCHANGE:
                    sts_emit_op(wb, SpvOpAtomicCompareExchange, 9);
                    wb_push(wb, type_spv);
                    wb_push(wb, result_spv);
                    wb_push(wb, a_ptr);
                    wb_push(wb, scope_id);
                    wb_push(wb, sem_equal_id);
                    wb_push(wb, sem_unequal_id);
                    wb_push(wb, a_val);
                    wb_push(wb, a_cmp);
                    break;
            }
            break;
        }

        default:
            break;
    }

    return 1;
}

/* ============================================================================
 * Function Emission
 * ============================================================================ */

static bool sts_is_trivial_branch_bridge(const SsirBlock *block) {
    if (!block || block->inst_count == 0) return false;
    const SsirInst *term = &block->insts[block->inst_count - 1];
    if (term->op != SSIR_OP_BRANCH || term->operand_count < 1) return false;
    for (uint32_t ii = 0; ii + 1 < block->inst_count; ++ii) {
        SsirOpcode op = block->insts[ii].op;
        if (op != SSIR_OP_SELECTION_MERGE && op != SSIR_OP_LOOP_MERGE)
            return false;
    }
    return true;
}

// c nonnull, func nonnull
static int sts_emit_function(Ctx *c, const SsirFunction *func, uint32_t func_type) {
    wgsl_compiler_assert(c != NULL, "sts_emit_function: c is NULL");
    wgsl_compiler_assert(func != NULL, "sts_emit_function: func is NULL");
    StsWordBuf *wb = &c->sections.functions;

    uint32_t return_spv = sts_emit_type(c, func->return_type);

    /* Emit OpFunction */
    uint32_t func_spv = get_spv_id(c, func->id);
    sts_emit_op(wb, SpvOpFunction, 5);
    wb_push(wb, return_spv);
    wb_push(wb, func_spv);
    wb_push(wb, SpvFunctionControlMaskNone);
    wb_push(wb, func_type);

    /* Debug name */
    sts_emit_name(c, func_spv, func->name);

    /* Emit parameters */
    for (uint32_t i = 0; i < func->param_count; ++i) {
        uint32_t param_type_spv = sts_emit_type(c, func->params[i].type);
        uint32_t param_spv = get_spv_id(c, func->params[i].id);
        sts_emit_op(wb, SpvOpFunctionParameter, 3);
        wb_push(wb, param_type_spv);
        wb_push(wb, param_spv);
        sts_emit_name(c, param_spv, func->params[i].name);
        set_ssir_type(c, func->params[i].id, func->params[i].type);
    }

    /* Dominator-tree-based block ordering for SPIR-V structural domination.
     * Uses ipdom (immediate post-dominator) to determine merge points for
     * selection constructs, and LOOP_MERGE annotations for loop constructs.
     * A recursive traversal with merge_bound prevents crossing construct
     * boundaries, producing valid SPIR-V layouts regardless of how merge
     * annotations were assigned. */
    uint32_t *dfs_order = (uint32_t *)STS_MALLOC((func->block_count + 64) * sizeof(uint32_t));
    uint32_t dfs_count = 0;
    const char *order_debug_env = getenv("SSIR_TO_SPIRV_DEBUG");
    bool order_debug = order_debug_env && order_debug_env[0] != '\0' &&
                       order_debug_env[0] != '0';

    if (dfs_order && func->block_count > 0) {
        const int n = (int)func->block_count;
        /* --- Build lightweight CFG: successors and predecessors --- */
        /* succ[i] stores successor block indices; nsuc[i] is count */
        int *succ_flat = (int *)STS_MALLOC(n * 140 * sizeof(int));
        int *nsuc = (int *)STS_MALLOC(n * sizeof(int));
        int *pred_flat = (int *)STS_MALLOC(n * 140 * sizeof(int));
        int *npred = (int *)STS_MALLOC(n * sizeof(int));
        int *ipdom = (int *)STS_MALLOC(n * sizeof(int));
        int *idom_fwd = NULL; /* forward dominators, computed alongside ipdom */
        uint8_t *visited = (uint8_t *)STS_MALLOC(n);
        uint8_t *is_known_merge_target = (uint8_t *)STS_MALLOC(n);
        uint8_t *is_active_merge = (uint8_t *)STS_MALLOC(n);
        uint8_t *is_continue_target = (uint8_t *)STS_MALLOC(n);

        if (succ_flat && nsuc && pred_flat && npred && ipdom && visited) {
            memset(nsuc, 0, n * sizeof(int));
            memset(npred, 0, n * sizeof(int));
            memset(visited, 0, n);
            if (is_known_merge_target) memset(is_known_merge_target, 0, n);
            if (is_active_merge) memset(is_active_merge, 0, n);
            if (is_continue_target) memset(is_continue_target, 0, n);

            /* Helper: find block index by ID */
            #define BO_IDX(id_val) ({ \
                int _r = -1; \
                for (int _i = 0; _i < n; _i++) \
                    if (func->blocks[_i].id == (id_val)) { _r = _i; break; } \
                _r; })
            #define BO_SUCC(bi, si) succ_flat[(bi) * 140 + (si)]
            #define BO_PRED(bi, pi) pred_flat[(bi) * 140 + (pi)]
            #define BO_ADD_EDGE(f, t) do { \
                if ((f) >= 0 && (t) >= 0 && nsuc[f] < 140) { \
                    bool _dup = false; \
                    for (int _i = 0; _i < nsuc[f]; _i++) \
                        if (BO_SUCC(f, _i) == (t)) { _dup = true; break; } \
                    if (!_dup) { \
                        BO_SUCC(f, nsuc[f]++) = (t); \
                        if (npred[t] < 140) { \
                            bool _dup2 = false; \
                            for (int _i = 0; _i < npred[t]; _i++) \
                                if (BO_PRED(t, _i) == (f)) { _dup2 = true; break; } \
                            if (!_dup2) BO_PRED(t, npred[t]++) = (f); \
                        } \
                    } \
                } \
            } while(0)

            /* Build CFG edges */
            for (int bi = 0; bi < n; bi++) {
                const SsirBlock *b = &func->blocks[bi];
                for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                    const SsirInst *si = &b->insts[ii];
                    if (si->op == SSIR_OP_BRANCH && si->operand_count >= 1)
                        BO_ADD_EDGE(bi, BO_IDX(si->operands[0]));
                    else if (si->op == SSIR_OP_BRANCH_COND && si->operand_count >= 2) {
                        BO_ADD_EDGE(bi, BO_IDX(si->operands[1]));
                        if (si->operand_count >= 3)
                            BO_ADD_EDGE(bi, BO_IDX(si->operands[2]));
                        if (si->operand_count >= 4 &&
                            si->operands[3] != 0 &&
                            is_known_merge_target) {
                            int mi = BO_IDX(si->operands[3]);
                            if (mi >= 0 && mi < n)
                                is_known_merge_target[mi] = 1;
                        }
                    } else if (si->op == SSIR_OP_SWITCH) {
                        if (si->operand_count >= 2)
                            BO_ADD_EDGE(bi, BO_IDX(si->operands[1]));
                        for (uint32_t ei = 1; ei < si->extra_count; ei += 2)
                            BO_ADD_EDGE(bi, BO_IDX(si->extra[ei]));
                    } else if (si->op == SSIR_OP_SELECTION_MERGE &&
                               si->operand_count >= 1 &&
                               is_known_merge_target) {
                        int mi = BO_IDX(si->operands[0]);
                        if (mi >= 0 && mi < n)
                            is_known_merge_target[mi] = 1;
                    } else if (si->op == SSIR_OP_LOOP_MERGE &&
                               si->operand_count >= 1) {
                        if (is_known_merge_target) {
                            int mi = BO_IDX(si->operands[0]);
                            if (mi >= 0 && mi < n)
                                is_known_merge_target[mi] = 1;
                        }
                        if (si->operand_count >= 2 && is_continue_target) {
                            int ci = BO_IDX(si->operands[1]);
                            if (ci >= 0 && ci < n)
                                is_continue_target[ci] = 1;
                        }
                    }
                }
            }

            /* --- Compute reachability and reverse-RPO for ipdom --- */
            bool *reach = (bool *)STS_MALLOC(n * sizeof(bool));
            int *rpo_stk = (int *)STS_MALLOC(n * sizeof(int));
            int *rpo_num = (int *)STS_MALLOC(n * sizeof(int));
            bool *rvis = (bool *)STS_MALLOC(n * sizeof(bool));
            int rpo_sp = 0;

            if (reach && rpo_stk && rpo_num && rvis) {
                /* Forward DFS to find reachable blocks and compute RPO numbers */
                memset(reach, 0, n * sizeof(bool));
                memset(rvis, 0, n * sizeof(bool));
                /* Iterative DFS for RPO */
                {
                    int dfs_cap = n * 12 + 64;
                    int *dfs_stk = (int *)STS_MALLOC(dfs_cap * sizeof(int));
                    int dsp = 0;
                    if (dfs_stk) {
                        dfs_stk[dsp++] = 0;
                        dfs_stk[dsp++] = 0; /* phase 0=enter, 1=post */
                        while (dsp > 0) {
                            int phase = dfs_stk[--dsp];
                            int node = dfs_stk[--dsp];
                            if (node < 0 || node >= n) continue;
                            if (phase == 1) {
                                rpo_stk[rpo_sp++] = node;
                                continue;
                            }
                            if (rvis[node]) continue;
                            rvis[node] = true;
                            reach[node] = true;
                            /* Push post-visit marker */
                            if (dsp + 2 <= dfs_cap) {
                                dfs_stk[dsp++] = node;
                                dfs_stk[dsp++] = 1;
                            }
                            /* Push successors in reverse order */
                            for (int si = nsuc[node] - 1; si >= 0; si--) {
                                int s = BO_SUCC(node, si);
                                if (s >= 0 && !rvis[s] && dsp + 2 <= dfs_cap) {
                                    dfs_stk[dsp++] = s;
                                    dfs_stk[dsp++] = 0;
                                }
                            }
                        }
                        STS_FREE(dfs_stk);
                    }
                }

                /* RPO numbering (rpo_stk is in reverse-postorder when read backwards) */
                for (int i = 0; i < n; i++) rpo_num[i] = n + i; /* unreachable = high */
                { int rn = 0;
                  for (int i = rpo_sp - 1; i >= 0; i--)
                      rpo_num[rpo_stk[i]] = rn++;
                }

                /* Reverse-RPO for post-dominator computation */
                /* Find exit blocks and do reverse DFS */
                int *rrpo_stk = (int *)STS_MALLOC(n * sizeof(int));
                int *rrpo_num = (int *)STS_MALLOC(n * sizeof(int));
                int rrpo_sp = 0;
                if (rrpo_stk && rrpo_num) {
                    bool *rr_vis = (bool *)STS_MALLOC(n * sizeof(bool));
                    if (rr_vis) {
                        memset(rr_vis, 0, n * sizeof(bool));
                        /* Iterative reverse DFS from exit blocks */
                        int rdfs_cap = n * 12 + 64;
                        int *rdfs_stk = (int *)STS_MALLOC(rdfs_cap * sizeof(int));
                        int rdsp = 0;
                        if (rdfs_stk) {
                            /* Seed: exit blocks (no successors, or has return/unreachable) */
                            for (int i = 0; i < n; i++) {
                                if (!reach[i]) continue;
                                bool is_exit = (nsuc[i] == 0);
                                const SsirBlock *b = &func->blocks[i];
                                for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                                    SsirOpcode op = b->insts[ii].op;
                                    if (op == SSIR_OP_RETURN || op == SSIR_OP_RETURN_VOID ||
                                        op == SSIR_OP_UNREACHABLE)
                                        is_exit = true;
                                }
                                if (is_exit && !rr_vis[i]) {
                                    /* DFS from this exit using predecessors */
                                    rdfs_stk[rdsp++] = i;
                                    rdfs_stk[rdsp++] = 0;
                                    while (rdsp > 0) {
                                        int ph = rdfs_stk[--rdsp];
                                        int nd = rdfs_stk[--rdsp];
                                        if (nd < 0 || nd >= n) continue;
                                        if (ph == 1) { rrpo_stk[rrpo_sp++] = nd; continue; }
                                        if (rr_vis[nd]) continue;
                                        rr_vis[nd] = true;
                                        if (rdsp + 2 <= rdfs_cap) {
                                            rdfs_stk[rdsp++] = nd;
                                            rdfs_stk[rdsp++] = 1;
                                        }
                                        for (int pi = npred[nd] - 1; pi >= 0; pi--) {
                                            int p = BO_PRED(nd, pi);
                                            if (p >= 0 && !rr_vis[p] && rdsp + 2 <= rdfs_cap) {
                                                rdfs_stk[rdsp++] = p;
                                                rdfs_stk[rdsp++] = 0;
                                            }
                                        }
                                    }
                                }
                            }
                            STS_FREE(rdfs_stk);
                        }
                        STS_FREE(rr_vis);
                    }
                    for (int i = 0; i < n; i++) rrpo_num[i] = n + i;
                    { int rn = 0;
                      for (int i = rrpo_sp - 1; i >= 0; i--)
                          rrpo_num[rrpo_stk[i]] = rn++;
                    }

                    /* --- Compute ipdom (immediate post-dominator) --- */
                    for (int i = 0; i < n; i++) ipdom[i] = -1;
                    /* Seed exit blocks as their own post-dominators */
                    for (int i = 0; i < n; i++) {
                        if (!reach[i]) continue;
                        bool is_exit = (nsuc[i] == 0);
                        const SsirBlock *b = &func->blocks[i];
                        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                            SsirOpcode op = b->insts[ii].op;
                            if (op == SSIR_OP_RETURN || op == SSIR_OP_RETURN_VOID ||
                                op == SSIR_OP_UNREACHABLE)
                                is_exit = true;
                        }
                        if (is_exit) ipdom[i] = i;
                    }
                    /* Cooper-Harvey-Kennedy iterative post-dominator */
                    { bool changed = true;
                      while (changed) {
                          changed = false;
                          for (int ri = 0; ri < rrpo_sp; ri++) {
                              int b = rrpo_stk[rrpo_sp - 1 - ri];
                              if (!reach[b] || ipdom[b] == b) continue;
                              int nd = -1;
                              for (int si = 0; si < nsuc[b]; si++) {
                                  int s = BO_SUCC(b, si);
                                  if (ipdom[s] == -1) continue;
                                  if (nd < 0) { nd = s; continue; }
                                  /* Intersect (with guard against infinite loop) */
                                  int a2 = nd, b2 = s, _steps = 0;
                                  while (a2 != b2 && _steps < n * 2) {
                                      while (rrpo_num[a2] > rrpo_num[b2] && _steps < n * 2) { a2 = ipdom[a2]; _steps++; }
                                      while (rrpo_num[b2] > rrpo_num[a2] && _steps < n * 2) { b2 = ipdom[b2]; _steps++; }
                                  }
                                  nd = (a2 == b2) ? a2 : -1;
                              }
                              if (nd >= 0 && nd != ipdom[b]) { ipdom[b] = nd; changed = true; }
                          }
                      }
                    }

                    /* --- Compute forward dominators (idom_fwd) for construct membership --- */
                    idom_fwd = (int *)STS_MALLOC(n * sizeof(int));
                    if (idom_fwd) {
                        for (int i = 0; i < n; i++) idom_fwd[i] = -1;
                        idom_fwd[0] = 0; /* entry block is its own dominator */
                        bool idom_changed = true;
                        while (idom_changed) {
                            idom_changed = false;
                            /* Process in RPO order (rpo_stk[rpo_sp-1] is RPO #0) */
                            for (int ri = rpo_sp - 1; ri >= 0; ri--) {
                                int b_idx = rpo_stk[ri];
                                if (b_idx == 0 || !reach[b_idx]) continue;
                                int nd = -1;
                                for (int pi = 0; pi < npred[b_idx]; pi++) {
                                    int p = BO_PRED(b_idx, pi);
                                    if (p < 0 || idom_fwd[p] == -1) continue;
                                    if (nd < 0) { nd = p; continue; }
                                    /* Intersect using rpo_num (forward RPO) */
                                    int a2 = nd, b2 = p;
                                    int _steps = 0;
                                    while (a2 != b2 && _steps < n * 2) {
                                        while (rpo_num[a2] > rpo_num[b2] && _steps < n * 2)
                                            { a2 = idom_fwd[a2]; _steps++; }
                                        while (rpo_num[b2] > rpo_num[a2] && _steps < n * 2)
                                            { b2 = idom_fwd[b2]; _steps++; }
                                    }
                                    nd = (a2 == b2) ? a2 : -1;
                                }
                                if (nd >= 0 && nd != idom_fwd[b_idx]) {
                                    idom_fwd[b_idx] = nd;
                                    idom_changed = true;
                                }
                            }
                        }
                    }

                    STS_FREE(rrpo_stk);
                    STS_FREE(rrpo_num);
                }


                /* Check if 'header' dominates 'block' using idom_fwd[] */
                #define BO_DOMINATES(header, block) ({ \
                    bool _dom = false; \
                    if (idom_fwd) { \
                        int _x = (block), _s = 0; \
                        while (_x >= 0 && _s < n + 1) { \
                            if (_x == (header)) { _dom = true; break; } \
                            if (_x == idom_fwd[_x]) break; \
                            _x = idom_fwd[_x]; _s++; \
                        } \
                    } \
                    _dom; })
                #define BO_POSTDOMINATES(merge, block) ({ \
                    bool _pdom = false; \
                    if (ipdom && (merge) >= 0 && (merge) < n) { \
                        int _x = (block), _s = 0; \
                        while (_x >= 0 && _s < n + 1) { \
                            if (_x == (merge)) { _pdom = true; break; } \
                            if (ipdom[_x] < 0 || ipdom[_x] == _x) break; \
                            _x = ipdom[_x]; _s++; \
                        } \
                    } \
                    _pdom; })

                /* --- Late loop detection: add OpLoopMerge to blocks that ---
                 * --- are back-edge targets but were not annotated by the  ---
                 * --- structurizer (non-trivial headers or loops created   ---
                 * --- by structurization transforms).                      --- */
                if (idom_fwd) {
                    for (int bi = 0; bi < n; bi++) {
                        SsirBlock *bk = &func->blocks[bi];
                        bool has_lm = false;
                        for (uint32_t ii = 0; ii < bk->inst_count; ii++)
                            if (bk->insts[ii].op == SSIR_OP_LOOP_MERGE) { has_lm = true; break; }
                        if (has_lm) continue;

                        /* Check if any predecessor edge is a back-edge
                         * (target dominates source). */
                        bool has_backedge_to = false;
                        for (int pi = 0; pi < npred[bi]; pi++) {
                            int p = BO_PRED(bi, pi);
                            if (p < 0 || p == bi) continue;
                            if (BO_DOMINATES(bi, p)) {
                                has_backedge_to = true;
                                break;
                            }
                        }
                        if (!has_backedge_to) continue;

                        /* bi is a back-edge target (loop header) without
                         * OpLoopMerge.  Find a merge (exit) block and a
                         * continue (latch) block. */

                        /* Collect natural loop body */
                        uint8_t *lbody = (uint8_t *)STS_MALLOC(n);
                        if (!lbody) continue;
                        memset(lbody, 0, n);
                        lbody[bi] = 1;
                        /* Reverse walk from each latch */
                        int lstack[512];
                        int lsp = 0;
                        for (int pi = 0; pi < npred[bi]; pi++) {
                            int p = BO_PRED(bi, pi);
                            if (p >= 0 && BO_DOMINATES(bi, p) && !lbody[p]) {
                                lbody[p] = 1;
                                if (lsp < 512) lstack[lsp++] = p;
                            }
                        }
                        while (lsp > 0) {
                            int x = lstack[--lsp];
                            for (int pi = 0; pi < npred[x]; pi++) {
                                int p = BO_PRED(x, pi);
                                if (p >= 0 && p < n && !lbody[p]) {
                                    lbody[p] = 1;
                                    if (lsp < 512) lstack[lsp++] = p;
                                }
                            }
                        }

                        /* Merge block: first successor of a body block that
                         * is NOT in the body, preferring ipdom of header. */
                        int lmerge = -1;
                        if (ipdom[bi] >= 0 && ipdom[bi] < n && !lbody[ipdom[bi]])
                            lmerge = ipdom[bi];
                        if (lmerge < 0) {
                            int best_rpo = n + 1;
                            for (int j = 0; j < n; j++) {
                                if (!lbody[j]) continue;
                                for (int si = 0; si < nsuc[j]; si++) {
                                    int s = BO_SUCC(j, si);
                                    if (s >= 0 && s < n && !lbody[s]) {
                                        if (rpo_num[s] < best_rpo) {
                                            best_rpo = rpo_num[s];
                                            lmerge = s;
                                        }
                                    }
                                }
                            }
                        }

                        /* Continue block: last latch by RPO */
                        int lcont = -1;
                        int best_rpo_cont = -1;
                        for (int pi = 0; pi < npred[bi]; pi++) {
                            int p = BO_PRED(bi, pi);
                            if (p >= 0 && p < n && lbody[p] &&
                                BO_DOMINATES(bi, p) &&
                                rpo_num[p] > best_rpo_cont) {
                                best_rpo_cont = rpo_num[p];
                                lcont = p;
                            }
                        }

                        STS_FREE(lbody);

                        if (lmerge < 0 || lcont < 0) continue;

                        /* Check for conflicting SelectionMerge */
                        bool has_sm = false;
                        for (uint32_t ii = 0; ii < bk->inst_count; ii++)
                            if (bk->insts[ii].op == SSIR_OP_SELECTION_MERGE)
                                { has_sm = true; break; }

                        if (has_sm) {
                            /* Replace the SelectionMerge with LoopMerge:
                             * the loop annotation takes precedence. */
                            for (uint32_t ii = 0; ii < bk->inst_count; ii++) {
                                if (bk->insts[ii].op == SSIR_OP_SELECTION_MERGE) {
                                    bk->insts[ii].op = SSIR_OP_LOOP_MERGE;
                                    bk->insts[ii].operands[0] = func->blocks[lmerge].id;
                                    bk->insts[ii].operands[1] = func->blocks[lcont].id;
                                    bk->insts[ii].operand_count = 2;
                                    break;
                                }
                            }
                        } else {
                            /* Insert OpLoopMerge before terminator */
                            if (bk->inst_count > 0) {
                                SsirInst lm;
                                memset(&lm, 0, sizeof(lm));
                                lm.op = SSIR_OP_LOOP_MERGE;
                                lm.operands[0] = func->blocks[lmerge].id;
                                lm.operands[1] = func->blocks[lcont].id;
                                lm.operand_count = 2;
                                /* Make room */
                                if (bk->inst_count >= bk->inst_capacity) {
                                    uint32_t nc = bk->inst_capacity ? bk->inst_capacity * 2 : 8;
                                    bk->insts = (SsirInst *)STS_REALLOC(bk->insts,
                                        nc * sizeof(SsirInst));
                                    bk->inst_capacity = nc;
                                }
                                uint32_t tpos = bk->inst_count - 1;
                                memmove(&bk->insts[tpos + 1], &bk->insts[tpos],
                                    sizeof(SsirInst));
                                bk->insts[tpos] = lm;
                                bk->inst_count++;
                            }
                        }

                        if (is_known_merge_target && lmerge >= 0 && lmerge < n)
                            is_known_merge_target[lmerge] = 1;
                    }
                }

                /* --- Recursive dominator-tree block ordering --- */
                /* Use an explicit work stack to avoid C recursion depth issues.
                 * Each frame: (block_idx, merge_bound, successor_index).
                 * successor_index tracks which successors have been visited. */
                {
                    typedef struct { int block; int merge_bound; int phase; int flags; } Frame;
                    #define FRAME_FROM_CONSTRUCT 1
                    int frame_cap = n * 4 + 64;
                    Frame *frames = (Frame *)STS_MALLOC(frame_cap * sizeof(Frame));
                    if (frames) {
                        int fsp = 0;
                        frames[fsp++] = (Frame){0, -1, 0, 0};

                        while (fsp > 0) {
                            Frame *f = &frames[fsp - 1];
                            int bi = f->block;
                            int mb = f->merge_bound;

                            /* Skip if invalid or at merge_bound */
                            if (bi < 0 || bi >= n) {
                                fsp--;
                                continue;
                            }
                            if (bi == mb && !(f->flags & FRAME_FROM_CONSTRUCT)) {
                                fsp--;
                                continue;
                            }
                            /* Skip blocks that are active merge targets of enclosing
                             * constructs. These blocks will be visited later when
                             * their owning header's phase 2/3 fires. Without this
                             * check, an inner construct can prematurely visit the
                             * merge of an outer construct, placing it too early. */
                            if (f->phase == 0 &&
                                !(f->flags & FRAME_FROM_CONSTRUCT) &&
                                is_known_merge_target &&
                                bi >= 0 && bi < n &&
                                is_known_merge_target[bi]) {
                                fsp--;
                                continue;
                            }
                            if (f->phase == 0 &&
                                !(f->flags & FRAME_FROM_CONSTRUCT) &&
                                is_active_merge &&
                                bi >= 0 && bi < n && is_active_merge[bi]) {
                                fsp--;
                                continue;
                            }
                            /* Skip already-visited blocks ONLY for phase 0 (first visit).
                             * Phase 2/3 frames are return visits to already-visited
                             * headers for merge block processing — don't skip them. */
                            if (visited[bi] && f->phase == 0) {
                                fsp--;
                                continue;
                            }

                            if (f->phase == 0) {
                                /* First visit: emit this block */
                                visited[bi] = 1;
                                dfs_order[dfs_count++] = bi;
                                f->phase = 1;

                                /* Determine block type from instructions */
                                const SsirBlock *b = &func->blocks[bi];
                                int loop_merge_idx = -1;
                                bool has_loop_merge = false;
                                bool is_multi_succ = false;
                                SsirOpcode term_op = (SsirOpcode)0;

                                for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                                    const SsirInst *si = &b->insts[ii];
                                    if (si->op == SSIR_OP_LOOP_MERGE && si->operand_count >= 1) {
                                        loop_merge_idx = BO_IDX(si->operands[0]);
                                        has_loop_merge = true;
                                    }
                                }
                                if (b->inst_count > 0)
                                    term_op = b->insts[b->inst_count - 1].op;
                                is_multi_succ = (term_op == SSIR_OP_BRANCH_COND ||
                                                 term_op == SSIR_OP_SWITCH);

                                if (has_loop_merge && loop_merge_idx >= 0) {
                                    /* Loop header: visit all successors with
                                     * merge_bound = loop_merge, then visit loop_merge
                                     * with the outer merge_bound. */
                                    f->phase = 2; /* mark: come back for loop merge */
                                    if (is_active_merge && loop_merge_idx >= 0 && loop_merge_idx < n)
                                        is_active_merge[loop_merge_idx] = 1;
                                    /* Push loop merge visit (will execute after body) */
                                    /* Push successor visits in reverse order */
                                    int saved_fsp = fsp;
                                    for (int si = nsuc[bi] - 1; si >= 0; si--) {
                                        int s = BO_SUCC(bi, si);
                                        if (s != loop_merge_idx && fsp < frame_cap)
                                            frames[fsp++] = (Frame){s, loop_merge_idx, 0};
                                    }
                                    (void)saved_fsp;
                                } else if (is_multi_succ) {
                                    /* Selection header: use explicit merge annotation
                                     * if present, otherwise fall back to ipdom. */
                                    int merge_idx = -1;
                                    bool has_explicit_merge = false;
                                    for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                                        const SsirInst *ai = &b->insts[ii];
                                        if (ai->op == SSIR_OP_SELECTION_MERGE &&
                                            ai->operand_count >= 1) {
                                            merge_idx = BO_IDX(ai->operands[0]);
                                            has_explicit_merge = true;
                                            break;
                                        }
                                        if (ai->op == SSIR_OP_BRANCH_COND &&
                                            ai->operand_count >= 4 &&
                                            ai->operands[3] != 0) {
                                            merge_idx = BO_IDX(ai->operands[3]);
                                            has_explicit_merge = true;
                                            break;
                                        }
                                    }
                                    if (merge_idx < 0)
                                        merge_idx = ipdom[bi];
                                    /* Trust explicit merge annotations even when the
                                     * CFG RPO does not make the merge look "forward".
                                     * The orderer is responsible for placing it after
                                     * the construct body. */
                                    bool valid_merge = (merge_idx >= 0 && merge_idx != bi);
                                    if (valid_merge && !has_explicit_merge &&
                                        rpo_num[merge_idx] <= rpo_num[bi])
                                        valid_merge = false;
                                    if (valid_merge) {
                                        f->phase = 3; /* come back for ipdom merge */
                                        if (is_active_merge && merge_idx >= 0 && merge_idx < n)
                                            is_active_merge[merge_idx] = 1;
                                        /* Push successors except merge, in reverse */
                                        for (int si = nsuc[bi] - 1; si >= 0; si--) {
                                            int s = BO_SUCC(bi, si);
                                            if (s != merge_idx && fsp < frame_cap)
                                                frames[fsp++] = (Frame){s, merge_idx, 0, 0};
                                        }
                                    } else {
                                        /* No valid ipdom: visit all successors */
                                        f->phase = -1;
                                        for (int si = nsuc[bi] - 1; si >= 0; si--) {
                                            int s = BO_SUCC(bi, si);
                                            if (fsp < frame_cap)
                                                frames[fsp++] = (Frame){s, mb, 0, 0};
                                        }
                                    }
                                } else {
                                    /* Single-successor or return: visit successors.
                                     * Current frame stays (phase -1); will be popped
                                     * via the visited[] check on next encounter. */
                                    if ((f->flags & FRAME_FROM_CONSTRUCT) &&
                                        sts_is_trivial_branch_bridge(b) &&
                                        nsuc[bi] == 1) {
                                        int s = BO_SUCC(bi, 0);
                                        bool delay_succ = false;
                                        if (s >= 0 && s < n) {
                                            if (is_continue_target && is_continue_target[s])
                                                delay_succ = true;
                                            if (is_active_merge && is_active_merge[s])
                                                delay_succ = true;
                                        }
                                        if (delay_succ) {
                                            fsp--;
                                            continue;
                                        }
                                    }
                                    f->phase = -1;
                                    for (int si = nsuc[bi] - 1; si >= 0; si--) {
                                        int s = BO_SUCC(bi, si);
                                        if (fsp < frame_cap)
                                            frames[fsp++] = (Frame){s, mb, 0, 0};
                                    }
                                    if (nsuc[bi] == 0) fsp--; /* return/unreachable: no successors, just pop */
                                }
                            } else if (f->phase == 2) {
                                /* Loop: body done, now visit the loop merge block */
                                const SsirBlock *b = &func->blocks[bi];
                                int lm_idx = -1;
                                for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                                    const SsirInst *si = &b->insts[ii];
                                    if (si->op == SSIR_OP_LOOP_MERGE && si->operand_count >= 1) {
                                        lm_idx = BO_IDX(si->operands[0]);
                                        break;
                                    }
                                }
                                if (is_active_merge && lm_idx >= 0 && lm_idx < n)
                                    is_active_merge[lm_idx] = 0;
                                fsp--; /* remove current frame */
                                if (lm_idx >= 0 && fsp < frame_cap)
                                    frames[fsp++] = (Frame){lm_idx, mb, 0,
                                                             FRAME_FROM_CONSTRUCT};
                            } else if (f->phase == 3) {
                                /* Selection: body done. Before visiting merge,
                                 * rescue unvisited blocks that belong to this
                                 * construct but were unreachable by the forward
                                 * DFS (bridge chain blocks and their targets). */
                                const SsirBlock *b3 = &func->blocks[bi];
                                int merge_idx = -1;
                                for (uint32_t ii = 0; ii < b3->inst_count; ii++) {
                                    const SsirInst *ai = &b3->insts[ii];
                                    if (ai->op == SSIR_OP_SELECTION_MERGE &&
                                        ai->operand_count >= 1) {
                                        merge_idx = BO_IDX(ai->operands[0]);
                                        break;
                                    }
                                    if (ai->op == SSIR_OP_BRANCH_COND &&
                                        ai->operand_count >= 4 &&
                                        ai->operands[3] != 0) {
                                        merge_idx = BO_IDX(ai->operands[3]);
                                        break;
                                    }
                                }
                                if (merge_idx < 0)
                                    merge_idx = ipdom[bi];

                                /* --- Rescue unvisited construct members --- */
                                int rescue[512];
                                int resc_count = 0;
                                if (merge_idx >= 0 && merge_idx < n && idom_fwd) {
                                    int bfs_q[512];
                                    int bfs_head = 0, bfs_tail = 0;
                                    /* Seed: unvisited predecessors of merge */
                                    for (int pi = 0; pi < npred[merge_idx]; pi++) {
                                        int p = BO_PRED(merge_idx, pi);
                                        if (p >= 0 && p < n && !visited[p] &&
                                            bfs_tail < 512)
                                            bfs_q[bfs_tail++] = p;
                                    }
                                    while (bfs_head < bfs_tail) {
                                        int bp = bfs_q[bfs_head++];
                                        if (bp < 0 || bp >= n || visited[bp]) continue;
                                        if (bp == bi || bp == merge_idx) continue;
                                        if (!BO_DOMINATES(bi, bp)) continue;
                                        if (!BO_POSTDOMINATES(merge_idx, bp)) continue;
                                        /* Dedup */
                                        bool dup = false;
                                        for (int ri = 0; ri < resc_count; ri++)
                                            if (rescue[ri] == bp) { dup = true; break; }
                                        if (dup) continue;
                                        if (resc_count < 512) rescue[resc_count++] = bp;
                                        /* Continue BFS only through bridge blocks */
                                        const SsirBlock *bpb = &func->blocks[bp];
                                        bool is_br = (bpb->inst_count >= 1 &&
                                            bpb->insts[bpb->inst_count-1].op == SSIR_OP_BRANCH);
                                        if (is_br) {
                                            bool pure = true;
                                            for (uint32_t ii = 0; ii < bpb->inst_count-1; ii++) {
                                                SsirOpcode op = bpb->insts[ii].op;
                                                if (op != SSIR_OP_SELECTION_MERGE &&
                                                    op != SSIR_OP_LOOP_MERGE)
                                                    { pure = false; break; }
                                            }
                                            if (pure) {
                                                for (int pi = 0; pi < npred[bp]; pi++) {
                                                    int pp = BO_PRED(bp, pi);
                                                    if (pp >= 0 && pp < n && !visited[pp] &&
                                                        bfs_tail < 512)
                                                        bfs_q[bfs_tail++] = pp;
                                                }
                                            }
                                        }
                                    }
                                    /* Forward extension: for rescued bridge blocks,
                                     * also rescue their branch targets if in-construct */
                                    for (int ri = 0; ri < resc_count; ri++) {
                                        int rb = rescue[ri];
                                        const SsirBlock *rbb = &func->blocks[rb];
                                        if (rbb->inst_count < 1) continue;
                                        if (rbb->insts[rbb->inst_count-1].op != SSIR_OP_BRANCH)
                                            continue;
                                        int tgt = BO_IDX(rbb->insts[rbb->inst_count-1].operands[0]);
                                        if (tgt < 0 || tgt >= n || visited[tgt]) continue;
                                        if (tgt == merge_idx) continue;
                                        if (!BO_DOMINATES(bi, tgt)) continue;
                                        if (!BO_POSTDOMINATES(merge_idx, tgt)) continue;
                                        bool dup = false;
                                        for (int ri2 = 0; ri2 < resc_count; ri2++)
                                            if (rescue[ri2] == tgt) { dup = true; break; }
                                        if (!dup && resc_count < 512)
                                            rescue[resc_count++] = tgt;
                                    }
                                    /* Sort descending by RPO (LIFO: lowest RPO on top) */
                                    for (int i = 0; i < resc_count - 1; i++)
                                        for (int j = i + 1; j < resc_count; j++)
                                            if (rpo_num[rescue[i]] < rpo_num[rescue[j]]) {
                                                int tmp = rescue[i];
                                                rescue[i] = rescue[j];
                                                rescue[j] = tmp;
                                            }
                                }

                                if (is_active_merge && merge_idx >= 0 && merge_idx < n)
                                    is_active_merge[merge_idx] = 0;
                                fsp--; /* pop selection header */
                                /* Push merge (visited last) */
                                if (merge_idx >= 0 && fsp < frame_cap)
                                    frames[fsp++] = (Frame){merge_idx, mb, 0,
                                                             FRAME_FROM_CONSTRUCT};
                                /* Push rescued blocks on top (visited before merge) */
                                for (int ri = 0; ri < resc_count; ri++) {
                                    if (!visited[rescue[ri]] && fsp < frame_cap)
                                        frames[fsp++] = (Frame){rescue[ri], merge_idx, 0, 0};
                                }
                            } else {
                                fsp--;
                            }
                        }
                        STS_FREE(frames);
                        #undef FRAME_FROM_CONSTRUCT
                    }
                }

                /* Place unvisited bridge chain blocks before their merge
                 * target in the output. Bridge blocks are created by the
                 * structurizer and contain only OpBranch (+ optional merge
                 * annotations). They form chains leading to merge blocks.
                 * Placing them before the merge inside the construct fixes
                 * SPIR-V validation errors about blocks branching to
                 * selection constructs from outside. */
                {
                    /* Process unvisited reachable blocks: bridge blocks get
                     * inserted before their target; others go at the end.
                     * Unreachable blocks are excluded — the structurizer's
                     * final pass already fixed dangling merge references. */
                    int unvis[512];
                    int unvis_count = 0;
                    for (int k = 0; k < n; k++) {
                        if (!visited[k] && reach[k] && unvis_count < 512)
                            unvis[unvis_count++] = k;
                    }

                    /* For each unvisited bridge block, find where its chain
                     * connects to the ordered output and insert it there. */
                    for (int ui = 0; ui < unvis_count; ui++) {
                        int blk = unvis[ui];
                        if (visited[blk]) continue; /* already inserted */
                        if (is_known_merge_target &&
                            blk >= 0 && blk < n &&
                            is_known_merge_target[blk])
                            continue;

                        /* Check if this is a bridge block */
                        const SsirBlock *bk = &func->blocks[blk];
                        bool is_bridge = false;
                        if (bk->inst_count >= 1 &&
                            bk->insts[bk->inst_count - 1].op == SSIR_OP_BRANCH) {
                            is_bridge = true;
                            for (uint32_t ii = 0; ii < bk->inst_count - 1; ii++) {
                                SsirOpcode op = bk->insts[ii].op;
                                if (op != SSIR_OP_SELECTION_MERGE &&
                                    op != SSIR_OP_LOOP_MERGE) {
                                    is_bridge = false;
                                    break;
                                }
                            }
                        }
                        if (!is_bridge) continue;

                        /* Collect the bridge chain starting from this block.
                         * Follow OpBranch targets until we reach a visited block
                         * (the connection point). */
                        int chain[512];
                        int chain_len = 0;
                        int cur = blk;
                        int connect = -1;
                        while (cur >= 0 && cur < n && !visited[cur] &&
                               chain_len < 512) {
                            const SsirBlock *cb = &func->blocks[cur];
                            bool cb_bridge = false;
                            if (cb->inst_count >= 1 &&
                                cb->insts[cb->inst_count - 1].op == SSIR_OP_BRANCH) {
                                cb_bridge = true;
                                for (uint32_t ii = 0;
                                     ii < cb->inst_count - 1; ii++) {
                                    SsirOpcode op = cb->insts[ii].op;
                                    if (op != SSIR_OP_SELECTION_MERGE &&
                                        op != SSIR_OP_LOOP_MERGE) {
                                        cb_bridge = false;
                                        break;
                                    }
                                }
                            }
                            if (!cb_bridge) break;
                            chain[chain_len++] = cur;
                            int tgt = BO_IDX(cb->insts[cb->inst_count - 1].operands[0]);
                            if (tgt >= 0 && tgt < n && visited[tgt]) {
                                connect = tgt;
                                break;
                            }
                            cur = tgt;
                        }

                        if (chain_len > 0) {
                            bool can_insert = (connect >= 0);
                            if (can_insert) {
                                for (int ci = 0; ci < chain_len && can_insert; ci++) {
                                    int cb = chain[ci];
                                    for (int pi = 0; pi < npred[cb]; pi++) {
                                        int pred = BO_PRED(cb, pi);
                                        if (pred < 0) continue;
                                        if (visited[pred]) continue;
                                        bool in_chain = false;
                                        for (int cj = 0; cj < chain_len; cj++) {
                                            if (chain[cj] == pred) {
                                                in_chain = true;
                                                break;
                                            }
                                        }
                                        if (!in_chain) {
                                            can_insert = false;
                                            break;
                                        }
                                    }
                                }
                            }

                            if (can_insert) {
                                for (int ci = 0; ci < chain_len; ci++)
                                    visited[chain[ci]] = 1;
                                /* Find the position of 'connect' in dfs_order
                                 * and insert the chain before it. */
                                int pos = -1;
                                for (uint32_t di = 0; di < dfs_count; di++) {
                                    if ((int)dfs_order[di] == connect) {
                                        pos = (int)di;
                                        break;
                                    }
                                }
                                if (pos >= 0) {
                                    /* Make room: shift dfs_order[pos..] right by chain_len */
                                    if (dfs_count + chain_len <= func->block_count + 64) {
                                        for (int si = (int)dfs_count - 1; si >= pos; si--)
                                            dfs_order[si + chain_len] = dfs_order[si];
                                        /* Insert chain */
                                        for (int ci = 0; ci < chain_len; ci++)
                                            dfs_order[pos + ci] = (uint32_t)chain[ci];
                                        dfs_count += chain_len;
                                    }
                                } else {
                                    /* Connection point not found; append at end */
                                    for (int ci = 0; ci < chain_len; ci++)
                                        dfs_order[dfs_count++] = (uint32_t)chain[ci];
                                }
                            } else if (connect < 0) {
                                /* No connection: chain didn't reach a visited block.
                                 * Append at end to avoid losing these blocks. */
                                for (int ci = 0; ci < chain_len; ci++)
                                    visited[chain[ci]] = 1;
                                for (int ci = 0; ci < chain_len; ci++)
                                    dfs_order[dfs_count++] = (uint32_t)chain[ci];
                            }
                        }
                    }

                    /* Append any remaining unvisited reachable blocks in
                     * forward-RPO.  Unreachable blocks are omitted unless
                     * they are merge targets (needed for SPIR-V structure). */
                    {
                        int remaining[512];
                        int remaining_count = 0;
                        for (int k = 0; k < n; k++) {
                            bool keep = reach[k];
                            if (!keep && is_known_merge_target &&
                                k < n && is_known_merge_target[k])
                                keep = true;
                            if (!visited[k] && keep && remaining_count < 512)
                                remaining[remaining_count++] = k;
                        }
                        /* Also include any NEW blocks (index >= n) that
                         * are merge targets created by the structurizer.
                         * These have index >= original n but < block_count. */
                        for (uint32_t k = (uint32_t)n; k < func->block_count; k++) {
                            /* Check if any emitted block references this
                             * block as a merge target. */
                            bool is_merge_ref = false;
                            uint32_t bid = func->blocks[k].id;
                            for (uint32_t di = 0; di < dfs_count && !is_merge_ref; di++) {
                                int eb = (int)dfs_order[di];
                                if (eb < 0 || eb >= (int)func->block_count) continue;
                                const SsirBlock *eb_b = &func->blocks[eb];
                                for (uint32_t ii = 0; ii < eb_b->inst_count; ii++) {
                                    const SsirInst *si = &eb_b->insts[ii];
                                    if ((si->op == SSIR_OP_SELECTION_MERGE ||
                                         si->op == SSIR_OP_LOOP_MERGE) &&
                                        si->operand_count >= 1) {
                                        for (uint32_t oi = 0; oi < si->operand_count; oi++) {
                                            if (si->operands[oi] == bid) {
                                                is_merge_ref = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (is_merge_ref) break;
                                }
                            }
                            if (is_merge_ref && remaining_count < 512)
                                remaining[remaining_count++] = (int)k;
                        }
                        for (int i = 0; i + 1 < remaining_count; i++) {
                            for (int j = i + 1; j < remaining_count; j++) {
                                if (rpo_num[remaining[i]] > rpo_num[remaining[j]]) {
                                    int tmp = remaining[i];
                                    remaining[i] = remaining[j];
                                    remaining[j] = tmp;
                                }
                            }
                        }
                        for (int i = 0; i < remaining_count; i++)
                            dfs_order[dfs_count++] = (uint32_t)remaining[i];
                    }

                    /* Some private bridge blocks are discovered during the
                     * main walk, so they never go through the "unvisited
                     * chain" rescue above. If such a bridge chain still ends
                     * up after the block it feeds, hoist the whole chain just
                     * before that target. */
                    {
                        bool changed = true;
                        int passes = 0;
                        int *pos_of = (int *)STS_MALLOC((n > 0 ? n : 1) * sizeof(int));
                        uint32_t *new_order = (uint32_t *)STS_MALLOC(
                            (func->block_count + 64) * sizeof(uint32_t));
                        if (!pos_of || !new_order) {
                            if (pos_of) STS_FREE(pos_of);
                            if (new_order) STS_FREE(new_order);
                        } else while (changed && passes++ < n * 4) {
                            changed = false;

                            for (int i = 0; i < n; i++)
                                pos_of[i] = -1;
                            for (uint32_t di = 0; di < dfs_count; di++) {
                                int blk = (int)dfs_order[di];
                                if (blk >= 0 && blk < n)
                                    pos_of[blk] = (int)di;
                            }

                            for (uint32_t di = 0; di < dfs_count && !changed; di++) {
                                int blk = (int)dfs_order[di];
                                if (blk < 0 || blk >= n) continue;

                                const SsirBlock *bk = &func->blocks[blk];
                                bool is_bridge = false;
                                if (bk->inst_count >= 1 &&
                                    bk->insts[bk->inst_count - 1].op == SSIR_OP_BRANCH) {
                                    is_bridge = true;
                                    for (uint32_t ii = 0; ii < bk->inst_count - 1; ii++) {
                                        SsirOpcode op = bk->insts[ii].op;
                                        if (op != SSIR_OP_SELECTION_MERGE &&
                                            op != SSIR_OP_LOOP_MERGE) {
                                            is_bridge = false;
                                            break;
                                        }
                                    }
                                }
                                if (!is_bridge) continue;

                                int chain[512];
                                int chain_len = 0;
                                int final_target = -1;
                                int cur = blk;
                                while (cur >= 0 && cur < n && chain_len < 512) {
                                    bool seen = false;
                                    for (int cj = 0; cj < chain_len; cj++) {
                                        if (chain[cj] == cur) {
                                            seen = true;
                                            break;
                                        }
                                    }
                                    if (seen) break;

                                    const SsirBlock *cb = &func->blocks[cur];
                                    bool cb_bridge = false;
                                    if (cb->inst_count >= 1 &&
                                        cb->insts[cb->inst_count - 1].op == SSIR_OP_BRANCH) {
                                        cb_bridge = true;
                                        for (uint32_t ii = 0; ii < cb->inst_count - 1; ii++) {
                                            SsirOpcode op = cb->insts[ii].op;
                                            if (op != SSIR_OP_SELECTION_MERGE &&
                                                op != SSIR_OP_LOOP_MERGE) {
                                                cb_bridge = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (!cb_bridge) break;

                                    chain[chain_len++] = cur;
                                    int tgt = BO_IDX(cb->insts[cb->inst_count - 1].operands[0]);
                                    if (tgt < 0 || tgt >= n) {
                                        final_target = -1;
                                        break;
                                    }

                                    const SsirBlock *tb = &func->blocks[tgt];
                                    bool tgt_bridge = false;
                                    if (tb->inst_count >= 1 &&
                                        tb->insts[tb->inst_count - 1].op == SSIR_OP_BRANCH) {
                                        tgt_bridge = true;
                                        for (uint32_t ii = 0; ii < tb->inst_count - 1; ii++) {
                                            SsirOpcode op = tb->insts[ii].op;
                                            if (op != SSIR_OP_SELECTION_MERGE &&
                                                op != SSIR_OP_LOOP_MERGE) {
                                                tgt_bridge = false;
                                                break;
                                            }
                                        }
                                    }
                                    if (!tgt_bridge) {
                                        final_target = tgt;
                                        break;
                                    }
                                    cur = tgt;
                                }

                                if (chain_len == 0) continue;

                                /* Check dominator of first block in chain */
                                int chain_idom = -1;
                                if (idom_fwd && chain[0] >= 0 && chain[0] < n)
                                    chain_idom = idom_fwd[chain[0]];

                                bool dom_violation = (chain_idom >= 0 &&
                                    chain_idom < n &&
                                    pos_of[chain_idom] >= 0 &&
                                    pos_of[chain_idom] > (int)di);

                                bool target_violation = (final_target >= 0 &&
                                    final_target < n &&
                                    pos_of[final_target] >= 0 &&
                                    pos_of[final_target] < (int)di);

                                if (dom_violation) {
                                    /* Bridge chain appears before its dominator.
                                     * Move it to just after the dominator. */
                                    uint32_t out = 0;
                                    bool inserted = false;
                                    for (uint32_t oi = 0; oi < dfs_count; oi++) {
                                        int cur_blk = (int)dfs_order[oi];
                                        bool in_chain = false;
                                        for (int cj = 0; cj < chain_len; cj++) {
                                            if (chain[cj] == cur_blk) {
                                                in_chain = true;
                                                break;
                                            }
                                        }
                                        if (!in_chain)
                                            new_order[out++] = (uint32_t)cur_blk;
                                        if (!inserted && cur_blk == chain_idom) {
                                            for (int cj = 0; cj < chain_len; cj++)
                                                new_order[out++] = (uint32_t)chain[cj];
                                            inserted = true;
                                        }
                                    }
                                    if (!inserted || out != dfs_count) continue;
                                    for (uint32_t oi = 0; oi < dfs_count; oi++)
                                        dfs_order[oi] = new_order[oi];
                                    changed = true;
                                } else if (target_violation) {
                                    /* Bridge chain appears after its target.
                                     * Hoist it to just before the target. */
                                    uint32_t out = 0;
                                    bool inserted = false;
                                    for (uint32_t oi = 0; oi < dfs_count; oi++) {
                                        int cur_blk = (int)dfs_order[oi];
                                        bool in_chain = false;
                                        for (int cj = 0; cj < chain_len; cj++) {
                                            if (chain[cj] == cur_blk) {
                                                in_chain = true;
                                                break;
                                            }
                                        }
                                        if (!inserted && cur_blk == final_target) {
                                            for (int cj = 0; cj < chain_len; cj++)
                                                new_order[out++] = (uint32_t)chain[cj];
                                            inserted = true;
                                        }
                                        if (!in_chain)
                                            new_order[out++] = (uint32_t)cur_blk;
                                    }
                                    if (!inserted || out != dfs_count) continue;
                                    for (uint32_t oi = 0; oi < dfs_count; oi++)
                                        dfs_order[oi] = new_order[oi];
                                    changed = true;
                                }
                            }
                        }
                        if (pos_of) STS_FREE(pos_of);
                        if (new_order) STS_FREE(new_order);
                    }
                }

                /* Recompute idom_fwd from the filtered block set.
                 * Compute proper RPO via forward DFS from block 0,
                 * considering only blocks that are in dfs_order. */
                if (idom_fwd && dfs_count > 0) {
                    uint8_t *in_out = (uint8_t *)STS_MALLOC(n);
                    int *lrpo = (int *)STS_MALLOC(n * sizeof(int));
                    int *lrpo_num = (int *)STS_MALLOC(n * sizeof(int));
                    if (in_out && lrpo && lrpo_num) {
                        memset(in_out, 0, n);
                        for (uint32_t di = 0; di < dfs_count; di++) {
                            int blk = (int)dfs_order[di];
                            if (blk >= 0 && blk < n) in_out[blk] = 1;
                        }
                        /* Forward DFS from block 0 on filtered CFG */
                        uint8_t *lvis = (uint8_t *)STS_MALLOC(n);
                        int *ldfs = (int *)STS_MALLOC(n * 12 * sizeof(int));
                        int lrpo_n = 0;
                        if (lvis && ldfs) {
                            memset(lvis, 0, n);
                            int ldsp = 0, ldcap = n * 12;
                            ldfs[ldsp++] = 0; ldfs[ldsp++] = 0;
                            while (ldsp > 0) {
                                int ph = ldfs[--ldsp];
                                int nd = ldfs[--ldsp];
                                if (nd < 0 || nd >= n) continue;
                                if (ph == 1) { lrpo[lrpo_n++] = nd; continue; }
                                if (lvis[nd]) continue;
                                lvis[nd] = true;
                                if (ldsp + 2 <= ldcap) { ldfs[ldsp++] = nd; ldfs[ldsp++] = 1; }
                                for (int si = nsuc[nd] - 1; si >= 0; si--) {
                                    int s = BO_SUCC(nd, si);
                                    if (s >= 0 && s < n && in_out[s] && !lvis[s] && ldsp + 2 <= ldcap) {
                                        ldfs[ldsp++] = s; ldfs[ldsp++] = 0;
                                    }
                                }
                            }
                        }
                        if (lvis) STS_FREE(lvis);
                        if (ldfs) STS_FREE(ldfs);
                        /* Reverse lrpo to get RPO order */
                        for (int i = 0; i < lrpo_n / 2; i++) {
                            int tmp = lrpo[i]; lrpo[i] = lrpo[lrpo_n - 1 - i]; lrpo[lrpo_n - 1 - i] = tmp;
                        }
                        for (int i = 0; i < n; i++) { idom_fwd[i] = -1; lrpo_num[i] = n + i; }
                        for (int i = 0; i < lrpo_n; i++) lrpo_num[lrpo[i]] = i;
                        if (lrpo_n > 0) idom_fwd[lrpo[0]] = lrpo[0];
                        bool lch = true;
                        while (lch) {
                            lch = false;
                            for (int ri = 1; ri < lrpo_n; ri++) {
                                int b = lrpo[ri];
                                int nd = -1;
                                for (int pi = 0; pi < npred[b]; pi++) {
                                    int p = BO_PRED(b, pi);
                                    if (p < 0 || !in_out[p] || idom_fwd[p] == -1) continue;
                                    if (nd < 0) { nd = p; continue; }
                                    int a2 = nd, b2 = p, _s = 0;
                                    while (a2 != b2 && _s < n * 2) {
                                        while (lrpo_num[a2] > lrpo_num[b2] && _s < n * 2)
                                            { a2 = idom_fwd[a2]; _s++; }
                                        while (lrpo_num[b2] > lrpo_num[a2] && _s < n * 2)
                                            { b2 = idom_fwd[b2]; _s++; }
                                    }
                                    nd = (a2 == b2) ? a2 : -1;
                                }
                                if (nd >= 0 && nd != idom_fwd[b]) {
                                    idom_fwd[b] = nd; lch = true;
                                }
                            }
                        }
                    }
                    if (in_out) STS_FREE(in_out);
                    if (lrpo) STS_FREE(lrpo);
                    if (lrpo_num) STS_FREE(lrpo_num);
                }

                /* General dominator-order enforcement: if a block appears
                 * before its immediate forward dominator, hoist the dominator
                 * to just before the block.  Hoisting the dominator earlier
                 * is safer than pushing the dominated block later, because
                 * the dominated block is already inside the right construct
                 * and moving it could break nesting. */
                if (idom_fwd) {
                    int *gpos = (int *)STS_MALLOC((n > 0 ? n : 1) * sizeof(int));
                    uint32_t *gord = (uint32_t *)STS_MALLOC(
                        (func->block_count + 64) * sizeof(uint32_t));
                    if (gpos && gord) {
                        bool gchanged = true;
                        int gpasses = 0;
                        while (gchanged && gpasses++ < n * 4) {
                            gchanged = false;

                            for (int i = 0; i < n; i++)
                                gpos[i] = -1;
                            for (uint32_t di = 0; di < dfs_count; di++) {
                                int blk = (int)dfs_order[di];
                                if (blk >= 0 && blk < n)
                                    gpos[blk] = (int)di;
                            }

                            for (uint32_t di = 0; di < dfs_count && !gchanged; di++) {
                                int blk = (int)dfs_order[di];
                                if (blk < 0 || blk >= n) continue;
                                int dom = idom_fwd[blk];
                                if (dom < 0 || dom >= n || dom == blk) continue;
                                if (gpos[dom] < 0 || gpos[dom] <= (int)di) continue;

                                /* dom at gpos[dom] is after blk at di.
                                 * Hoist dom to just before blk. */
                                uint32_t out = 0;
                                bool inserted = false;
                                for (uint32_t oi = 0; oi < dfs_count; oi++) {
                                    if ((int)dfs_order[oi] == dom) continue;
                                    if (!inserted && (int)dfs_order[oi] == blk) {
                                        gord[out++] = (uint32_t)dom;
                                        inserted = true;
                                    }
                                    gord[out++] = dfs_order[oi];
                                }
                                if (inserted && out == dfs_count) {
                                    for (uint32_t oi = 0; oi < dfs_count; oi++)
                                        dfs_order[oi] = gord[oi];
                                    gchanged = true;
                                }
                            }
                        }
                    }
                    if (gpos) STS_FREE(gpos);
                    if (gord) STS_FREE(gord);
                }

            }

            if (reach) STS_FREE(reach);
            if (rpo_stk) STS_FREE(rpo_stk);
            if (rpo_num) STS_FREE(rpo_num);
            if (rvis) STS_FREE(rvis);
        }

        if (succ_flat) STS_FREE(succ_flat);
        if (nsuc) STS_FREE(nsuc);
        if (pred_flat) STS_FREE(pred_flat);
        if (npred) STS_FREE(npred);
        if (ipdom) STS_FREE(ipdom);
        if (idom_fwd) STS_FREE(idom_fwd);
        if (visited) STS_FREE(visited);
        if (is_known_merge_target) STS_FREE(is_known_merge_target);
        if (is_active_merge) STS_FREE(is_active_merge);
        if (is_continue_target) STS_FREE(is_continue_target);

        #undef BO_IDX
        #undef BO_SUCC
        #undef BO_PRED
        #undef BO_ADD_EDGE
        #undef BO_DOMINATES
        #undef BO_POSTDOMINATES
    }
    /* Append any blocks not reached by DFS (unreachable / bounce blocks).
     * Instead of falling back to original order (which loses DFS ordering),
     * keep the DFS-ordered blocks and append the rest at the end. */
    if (dfs_count != func->block_count) {
        if (order_debug) {
            fprintf(stderr,
                    "[sts_order] appending %u unreached blocks (func=%s dfs=%u total=%u)\n",
                    func->block_count - dfs_count,
                    func->name ? func->name : "<anon-func>",
                    dfs_count, func->block_count);
        }
        /* Build a visited set from the DFS order */
        uint8_t *in_dfs = (uint8_t *)STS_MALLOC(func->block_count);
        if (in_dfs) {
            memset(in_dfs, 0, func->block_count);
            for (uint32_t k = 0; k < dfs_count; k++)
                if (dfs_order[k] < func->block_count)
                    in_dfs[dfs_order[k]] = 1;
            /* Drop unreachable blocks — they have no valid predecessors
             * and emitting them can produce invalid SPIR-V (undefined IDs,
             * loads from non-pointer types, unterminated blocks). */
            if (order_debug) {
                for (uint32_t k = 0; k < func->block_count; k++) {
                    if (!in_dfs[k]) {
                        const SsirBlock *ub = &func->blocks[k];
                        fprintf(stderr, "[sts_order]   dropped unreached[%u] id=%u name=%s insts=%u\n",
                                k, ub->id, ub->name ? ub->name : "?", ub->inst_count);
                    }
                }
            }
            STS_FREE(in_dfs);
        }
    }

    /* Emit blocks in DFS order */
    for (uint32_t dfs_i = 0; dfs_i < dfs_count; ++dfs_i) {
        const SsirBlock *block = &func->blocks[dfs_order[dfs_i]];
        uint32_t block_spv = get_spv_id(c, block->id);

        /* Emit OpLabel */
        sts_emit_op(wb, SpvOpLabel, 2);
        wb_push(wb, block_spv);

        /* Debug: emit block name */
        if (block->name && block->name[0]) {
            sts_emit_name(c, block_spv, block->name);
        } else {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "blk_%u", block->id);
            sts_emit_name(c, block_spv, tmp);
        }

        /* Emit local variables in first block */
        if (dfs_i == 0) {
            for (uint32_t li = 0; li < func->local_count; ++li) {
                const SsirLocalVar *local = &func->locals[li];
                uint32_t local_type_spv = sts_emit_type(c, local->type);
                uint32_t local_spv = get_spv_id(c, local->id);
                sts_emit_op(wb, SpvOpVariable, 4);
                wb_push(wb, local_type_spv);
                wb_push(wb, local_spv);
                wb_push(wb, SpvStorageClassFunction);
                sts_emit_name(c, local_spv, local->name);
                set_ssir_type(c, local->id, local->type);
            }
        }

        /* Emit instructions.
         * Skip SelectionMerge if the block also has a LoopMerge
         * (only one merge can precede a branch in SPIR-V). */
        {
            bool has_loop_merge = false;
            for (uint32_t ii = 0; ii < block->inst_count; ii++) {
                if (block->insts[ii].op == SSIR_OP_LOOP_MERGE) {
                    has_loop_merge = true;
                    break;
                }
            }
            for (uint32_t ii = 0; ii < block->inst_count; ++ii) {
                /* Skip SelectionMerge in blocks that have LoopMerge */
                if (has_loop_merge && block->insts[ii].op == SSIR_OP_SELECTION_MERGE)
                    continue;
                /* Skip SelectionMerge embedded in BRANCH_COND when LoopMerge exists */
                if (has_loop_merge && block->insts[ii].op == SSIR_OP_BRANCH_COND &&
                    block->insts[ii].operand_count >= 4 && block->insts[ii].operands[3] != 0) {
                    /* Temporarily clear the merge annotation so emit_instruction doesn't emit SelectionMerge */
                    uint32_t saved = block->insts[ii].operands[3];
                    ((SsirInst *)&block->insts[ii])->operands[3] = 0;
                    emit_instruction(c, &block->insts[ii], func->return_type);
                    ((SsirInst *)&block->insts[ii])->operands[3] = saved;
                    continue;
                }
                emit_instruction(c, &block->insts[ii], func->return_type);
            }
        }

        /* Ensure every block ends with a terminator.  Unreachable blocks
         * (or blocks whose terminator was lost) get OpUnreachable. */
        {
            bool has_term = false;
            if (block->inst_count > 0) {
                SsirOpcode last_op = block->insts[block->inst_count - 1].op;
                if (last_op == SSIR_OP_BRANCH || last_op == SSIR_OP_BRANCH_COND ||
                    last_op == SSIR_OP_RETURN || last_op == SSIR_OP_RETURN_VOID ||
                    last_op == SSIR_OP_UNREACHABLE || last_op == SSIR_OP_DISCARD ||
                    last_op == SSIR_OP_SWITCH)
                    has_term = true;
            }
            if (!has_term) {
                sts_emit_op(wb, SpvOpUnreachable, 1);
            }
        }
    }

    if (dfs_order) STS_FREE(dfs_order);

    /* Emit OpFunctionEnd */
    sts_emit_op(wb, SpvOpFunctionEnd, 1);

    return 1;
}

/* ============================================================================
 * Entry Point Emission
 * ============================================================================ */

static SpvExecutionModel stage_to_execution_model(SsirStage stage) {
    switch (stage) {
        case SSIR_STAGE_VERTEX: return SpvExecutionModelVertex;
        case SSIR_STAGE_FRAGMENT: return SpvExecutionModelFragment;
        case SSIR_STAGE_COMPUTE: return SpvExecutionModelGLCompute;
        case SSIR_STAGE_GEOMETRY: return SpvExecutionModelGeometry;
        case SSIR_STAGE_TESS_CONTROL: return SpvExecutionModelTessellationControl;
        case SSIR_STAGE_TESS_EVAL: return SpvExecutionModelTessellationEvaluation;
        default: return SpvExecutionModelVertex;
    }
}

// c nonnull, ep nonnull
static int sts_emit_entry_point(Ctx *c, const SsirEntryPoint *ep) {
    wgsl_compiler_assert(c != NULL, "sts_emit_entry_point: c is NULL");
    wgsl_compiler_assert(ep != NULL, "sts_emit_entry_point: ep is NULL");
    /* Emit OpEntryPoint */
    StsWordBuf *wb = &c->sections.entry_points;
    uint32_t *name_words;
    size_t name_count;
    encode_string(ep->name, &name_words, &name_count);

    uint32_t func_spv = get_spv_id(c, ep->function);

    /* For SPIR-V 1.3, only Input/Output globals go in the interface.
     * For SPIR-V 1.4+, all used globals must be in the interface. */
    uint32_t spv_ver = c->opts.spirv_version ? c->opts.spirv_version : 0x00010300;
    bool include_all_globals = (spv_ver >= 0x00010400);
    uint32_t iface_count = 0;
    uint32_t *iface_ids = NULL;
    if (ep->interface_count > 0) {
        iface_ids = (uint32_t *)STS_MALLOC(ep->interface_count * sizeof(uint32_t));
        /* PRE: iface_ids != NULL */
        wgsl_compiler_assert(iface_ids != NULL, "sts_emit_entry_point: iface_ids alloc failed");
        for (uint32_t i = 0; i < ep->interface_count; ++i) {
            SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, ep->interface[i]);
            if (g) {
                SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
                if (include_all_globals) {
                    /* SPIR-V 1.4+: all globals in interface */
                    iface_ids[iface_count++] = get_spv_id(c, ep->interface[i]);
                } else if (pt && pt->kind == SSIR_TYPE_PTR &&
                           (pt->ptr.space == SSIR_ADDR_INPUT || pt->ptr.space == SSIR_ADDR_OUTPUT)) {
                    iface_ids[iface_count++] = get_spv_id(c, ep->interface[i]);
                }
            }
        }
    }

    sts_emit_op(wb, SpvOpEntryPoint, 3 + name_count + iface_count);
    wb_push(wb, stage_to_execution_model(ep->stage));
    wb_push(wb, func_spv);
    sts_wb_push_many(wb, name_words, name_count);
    STS_FREE(name_words);

    for (uint32_t i = 0; i < iface_count; ++i) {
        wb_push(wb, iface_ids[i]);
    }
    STS_FREE(iface_ids);

    /* Emit execution modes */
    wb = &c->sections.execution_modes;

    if (ep->stage == SSIR_STAGE_FRAGMENT) {
        /* Always emit OriginUpperLeft for fragment shaders */
        sts_emit_op(wb, SpvOpExecutionMode, 3);
        wb_push(wb, func_spv);
        wb_push(wb, SpvExecutionModeOriginUpperLeft);
        if (ep->depth_replacing) {
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeDepthReplacing);
        }
        if (ep->early_fragment_tests) {
            sts_emit_op(wb, SpvOpExecutionMode, 3);
            wb_push(wb, func_spv);
            wb_push(wb, SpvExecutionModeEarlyFragmentTests);
        }
    }

    if (ep->stage == SSIR_STAGE_COMPUTE) {
        /* LocalSize for compute shaders */
        sts_emit_op(wb, SpvOpExecutionMode, 6);
        wb_push(wb, func_spv);
        wb_push(wb, SpvExecutionModeLocalSize);
        wb_push(wb, ep->workgroup_size[0]);
        wb_push(wb, ep->workgroup_size[1]);
        wb_push(wb, ep->workgroup_size[2]);
    }

    return 1;
}

/* ============================================================================
 * Main Conversion
 * ============================================================================ */

SsirToSpirvResult ssir_to_spirv(const SsirModule *mod,
    const SsirToSpirvOptions *opts,
    uint32_t **out_words,
    size_t *out_count) {
    if (!mod || !out_words || !out_count) {
        return SSIR_TO_SPIRV_ERR_INVALID_INPUT;
    }

    *out_words = NULL;
    *out_count = 0;

    /* Initialize context */
    Ctx c = {0};
    c.mod = mod;
    if (opts) {
        c.opts = *opts;
    }
    sections_init(&c.sections);
    c.next_spv_id = 1;

    /* Allocate ID map */
    c.id_map_size = mod->next_id + 1;
    c.id_map = (uint32_t *)STS_MALLOC(c.id_map_size * sizeof(uint32_t));
    if (!c.id_map) {
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }
    memset(c.id_map, 0, c.id_map_size * sizeof(uint32_t));

    /* Allocate type map (SSIR ID -> SSIR type) */
    c.type_map_size = mod->next_id + 1;
    c.type_map = (uint32_t *)STS_MALLOC(c.type_map_size * sizeof(uint32_t));
    if (!c.type_map) {
        STS_FREE(c.id_map);
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }
    memset(c.type_map, 0, c.type_map_size * sizeof(uint32_t));

    /* Pre-scan: detect if module uses PhysicalStorageBuffer pointers */
    {
        for (uint32_t ti = 0; ti < mod->type_count; ti++) {
            SsirType *t = &mod->types[ti];
            if (t->kind == SSIR_TYPE_PTR &&
                t->ptr.space == SSIR_ADDR_PHYSICAL_STORAGE_BUFFER) {
                c.has_psb_cap = 1;
                break;
            }
        }
    }

    /* Emit capabilities */
    sts_emit_capability(&c, SpvCapabilityShader);
    c.has_shader_cap = 1;
    if (c.has_psb_cap) {
        sts_emit_capability(&c, SpvCapabilityPhysicalStorageBufferAddresses);
        sts_emit_capability(&c, SpvCapabilityInt64);
    }
    /* binding_array capability emitted later (after types are processed) */

    /* Extensions */
    if (c.has_psb_cap) {
        /* SPV_KHR_physical_storage_buffer extension (required for SPIR-V < 1.5) */
        StsWordBuf *ewb = &c.sections.extensions;
        uint32_t *str_words = NULL;
        size_t str_count = 0;
        encode_string("SPV_KHR_physical_storage_buffer", &str_words, &str_count);
        if (str_words) {
            sts_emit_op(ewb, SpvOpExtension, 1 + (uint32_t)str_count);
            sts_wb_push_many(ewb, str_words, str_count);
            STS_FREE(str_words);
        }
    }

    /* GLSL.std.450 import ID allocated lazily on first use */
    c.glsl_ext_id = 0;

    /* Emit memory model */
    if (c.has_psb_cap) {
        /* PhysicalStorageBuffer64 addressing + GLSL450 memory model */
        StsWordBuf *wb = &c.sections.memory_model;
        sts_emit_op(wb, SpvOpMemoryModel, 3);
        wb_push(wb, SpvAddressingModelPhysicalStorageBuffer64);
        wb_push(wb, SpvMemoryModelGLSL450);
    } else {
        sts_emit_memory_model(&c);
    }

    /* Pre-emit function signature types and pre-allocate function/block IDs
     * Order: return type, function ID, param types, function type, block IDs */
    if (mod->function_count > 0) {
        c.func_type_cache = (uint32_t *)STS_MALLOC(mod->function_count * sizeof(uint32_t));
        /* PRE: func_type_cache != NULL */
        wgsl_compiler_assert(c.func_type_cache != NULL, "ssir_to_spirv: func_type_cache alloc failed");
        c.func_type_cache_count = mod->function_count;
        for (uint32_t i = 0; i < mod->function_count; ++i) {
            const SsirFunction *func = &mod->functions[i];
            uint32_t return_spv = sts_emit_type(&c, func->return_type);
            get_spv_id(&c, func->id);
            uint32_t *param_types = NULL;
            if (func->param_count > 0) {
                param_types = (uint32_t *)STS_MALLOC(func->param_count * sizeof(uint32_t));
                /* PRE: param_types != NULL */
                wgsl_compiler_assert(param_types != NULL, "ssir_to_spirv: param_types alloc failed");
                for (uint32_t j = 0; j < func->param_count; ++j) {
                    /* PRE: func->params[j] valid (guarded by param_count loop) */
                    wgsl_compiler_assert(j < func->param_count, "ssir_to_spirv: params[%u] OOB", j);
                    param_types[j] = sts_emit_type(&c, func->params[j].type);
                }
            }
            c.func_type_cache[i] = sts_emit_type_function(&c, return_spv, param_types, func->param_count);
            STS_FREE(param_types);
            for (uint32_t bi = 0; bi < func->block_count; ++bi)
                get_spv_id(&c, func->blocks[bi].id);
        }
    }

    /* Emit constants with correct type ordering:
     * For composite constants, emit their type just before the first component constant */
    {
        /* Build map: for each composite, find the earliest component constant index */
        uint32_t *comp_type_at = NULL; /* comp_type_at[i] = ssir type to emit before constant i, or 0 */
        if (mod->constant_count > 0) {
            comp_type_at = (uint32_t *)STS_MALLOC(mod->constant_count * sizeof(uint32_t));
            if (comp_type_at) memset(comp_type_at, 0, mod->constant_count * sizeof(uint32_t));
        }
        if (comp_type_at) {
            for (uint32_t ci = 0; ci < mod->constant_count; ++ci) {
                if (mod->constants[ci].kind != SSIR_CONST_COMPOSITE) continue;
                /* Find earliest component index */
                uint32_t earliest = ci;
                for (uint32_t k = 0; k < mod->constants[ci].composite.count; ++k) {
                    uint32_t comp_id = mod->constants[ci].composite.components[k];
                    for (uint32_t j = 0; j < mod->constant_count; ++j) {
                        if (mod->constants[j].id == comp_id && j < earliest) {
                            earliest = j;
                            break;
                        }
                    }
                }
                if (earliest < ci && comp_type_at[earliest] == 0)
                    comp_type_at[earliest] = mod->constants[ci].type;
            }
        }
        for (uint32_t i = 0; i < mod->constant_count; ++i) {
            if (comp_type_at && comp_type_at[i])
                sts_emit_type(&c, comp_type_at[i]);
            sts_emit_constant(&c, &mod->constants[i]);
        }
        STS_FREE(comp_type_at);
    }

    /* Emit global variables */
    for (uint32_t i = 0; i < mod->global_count; ++i) {
        sts_emit_global_var(&c, &mod->globals[i]);
    }

    /* Emit functions */
    for (uint32_t i = 0; i < mod->function_count; ++i) {
        sts_emit_function(&c, &mod->functions[i], c.func_type_cache[i]);
    }

    /* Emit names from name table (for let/const SSA results) */
    for (uint32_t i = 0; i < mod->name_count; ++i) {
        uint32_t spv_id = get_spv_id(&c, mod->names[i].id);
        if (spv_id) sts_emit_name(&c, spv_id, mod->names[i].name);
    }

    /* Emit entry points */
    for (uint32_t i = 0; i < mod->entry_point_count; ++i) {
        sts_emit_entry_point(&c, &mod->entry_points[i]);
    }

    /* Emit deferred capabilities (set during type/global emission) */
    if (c.has_binding_array) {
        sts_emit_capability(&c, 5302 /* RuntimeDescriptorArrayEXT */);
        StsWordBuf *ewb = &c.sections.extensions;
        uint32_t *str_words = NULL;
        size_t str_count = 0;
        encode_string("SPV_EXT_descriptor_indexing", &str_words, &str_count);
        if (str_words) {
            sts_emit_op(ewb, SpvOpExtension, 1 + (uint32_t)str_count);
            sts_wb_push_many(ewb, str_words, str_count);
            STS_FREE(str_words);
        }
    }

    /* Finalize SPIR-V binary */
    size_t total = 5; /* header */
    total += c.sections.capabilities.len;
    total += c.sections.extensions.len;
    total += c.sections.ext_inst_imports.len;
    total += c.sections.memory_model.len;
    total += c.sections.entry_points.len;
    total += c.sections.execution_modes.len;
    total += c.sections.debug_names.len;
    total += c.sections.annotations.len;
    total += c.sections.types_constants.len;
    total += c.sections.globals.len;
    total += c.sections.functions.len;

    uint32_t *words = (uint32_t *)STS_MALLOC(total * sizeof(uint32_t));
    if (!words) {
        STS_FREE(c.id_map);
        STS_FREE(c.type_map);
        sections_free(&c.sections);
        return SSIR_TO_SPIRV_ERR_OOM;
    }

    size_t pos = 0;

    /* Header */
    words[pos++] = SpvMagicNumber;
    words[pos++] = c.opts.spirv_version ? c.opts.spirv_version : 0x00010300;
    words[pos++] = 0;             /* generator */
    words[pos++] = c.next_spv_id; /* bound */
    words[pos++] = 0;             /* reserved */

/* Copy sections */
#define COPY_SEC(sec)                                                                        \
    do {                                                                                     \
        if (c.sections.sec.len > 0) {                                                        \
            memcpy(words + pos, c.sections.sec.data, c.sections.sec.len * sizeof(uint32_t)); \
            pos += c.sections.sec.len;                                                       \
        }                                                                                    \
    } while (0)

    COPY_SEC(capabilities);
    COPY_SEC(extensions);
    COPY_SEC(ext_inst_imports);
    COPY_SEC(memory_model);
    COPY_SEC(entry_points);
    COPY_SEC(execution_modes);
    COPY_SEC(debug_names);
    COPY_SEC(annotations);
    COPY_SEC(types_constants);
    COPY_SEC(globals);
    COPY_SEC(functions);

#undef COPY_SEC

    /* Cleanup */
    STS_FREE(c.func_type_cache);
    STS_FREE(c.id_map);
    STS_FREE(c.type_map);
    sections_free(&c.sections);

    *out_words = words;
    *out_count = total;

    return SSIR_TO_SPIRV_OK;
}

void ssir_to_spirv_free(void *p) {
    STS_FREE(p);
}

const char *ssir_to_spirv_result_string(SsirToSpirvResult result) {
    switch (result) {
        case SSIR_TO_SPIRV_OK: return "OK";
        case SSIR_TO_SPIRV_ERR_INVALID_INPUT: return "Invalid input";
        case SSIR_TO_SPIRV_ERR_UNSUPPORTED: return "Unsupported feature";
        case SSIR_TO_SPIRV_ERR_INTERNAL: return "Internal error";
        case SSIR_TO_SPIRV_ERR_OOM: return "Out of memory";
        default: return "Unknown error";
    }
}
