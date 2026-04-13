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
 * Unified allocator macros — backends can still override via #define before
 * including this header (existing #ifndef guards are preserved).
 * ============================================================================ */

#ifndef SW_MALLOC
#define SW_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef SW_REALLOC
#define SW_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef SW_FREE
#define SW_FREE(p) free((p))
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define SW_UNUSED __attribute__((unused))
#else
#  define SW_UNUSED
#endif

/* ============================================================================
 * SW_GROW - Dynamic array growth macro
 * ============================================================================ */

#define SW_GROW(ptr, count, cap, T, realloc_fn) do { \
    if ((count) >= (cap)) { \
        uint32_t _nc = (cap) ? (cap) * 2 : 8; \
        while (_nc < (count)) _nc *= 2; \
        T *_np = (T *)(realloc_fn)((ptr), _nc * sizeof(T)); \
        if (_np) { (ptr) = _np; (cap) = _nc; } \
    } \
} while(0)

/* ============================================================================
 * Vector swizzle char -> index
 * ============================================================================ */

static inline int sw_swizzle_char_to_idx(char c) {
    switch (c) {
        case 'x': case 'r': case 's': return 0;
        case 'y': case 'g': case 't': return 1;
        case 'z': case 'b': case 'p': return 2;
        case 'w': case 'a': case 'q': return 3;
        default: return -1;
    }
}

/* ============================================================================
 * WGSL AST address-space name <-> enum conversion
 * ============================================================================ */

static inline int sw_str_eq_n(const char *s, size_t n, const char *lit) {
    size_t ll = strlen(lit);
    return n == ll && memcmp(s, lit, ll) == 0;
}

static inline WgslAstAddrSpace sw_parse_addr_space(const char *s, size_t n) {
    if (!s || n == 0) return WGSL_ADDR_NONE;
    if (sw_str_eq_n(s, n, "function")) return WGSL_ADDR_FUNCTION;
    if (sw_str_eq_n(s, n, "private")) return WGSL_ADDR_PRIVATE;
    if (sw_str_eq_n(s, n, "workgroup")) return WGSL_ADDR_WORKGROUP;
    if (sw_str_eq_n(s, n, "uniform")) return WGSL_ADDR_UNIFORM;
    if (sw_str_eq_n(s, n, "storage")) return WGSL_ADDR_STORAGE;
    if (sw_str_eq_n(s, n, "immediate")) return WGSL_ADDR_IMMEDIATE;
    if (sw_str_eq_n(s, n, "device")) return WGSL_ADDR_DEVICE;
    if (sw_str_eq_n(s, n, "in")) return WGSL_ADDR_IN;
    if (sw_str_eq_n(s, n, "out")) return WGSL_ADDR_OUT;
    if (sw_str_eq_n(s, n, "push_constant")) return WGSL_ADDR_PUSH_CONSTANT;
    if (sw_str_eq_n(s, n, "handle")) return WGSL_ADDR_HANDLE;
    return WGSL_ADDR_UNKNOWN;
}

/* ============================================================================
 * WGSL built-in type name parsing
 * ============================================================================ */

typedef enum WgslScalarKind {
    WGSL_SCALAR_NONE = 0,
    WGSL_SCALAR_VOID,
    WGSL_SCALAR_BOOL,
    WGSL_SCALAR_I32,
    WGSL_SCALAR_U32,
    WGSL_SCALAR_F32,
    WGSL_SCALAR_F16
} WgslScalarKind;

typedef enum WgslTypeTag {
    WGSL_TYPE_UNKNOWN = 0,
    WGSL_TYPE_SCALAR,
    WGSL_TYPE_VEC,
    WGSL_TYPE_MAT,
    WGSL_TYPE_ARRAY,
    WGSL_TYPE_BINDING_ARRAY,
    WGSL_TYPE_PTR,
    WGSL_TYPE_ATOMIC,
    WGSL_TYPE_SAMPLER,
    WGSL_TYPE_SAMPLER_COMPARISON,
    WGSL_TYPE_TEXTURE_1D,
    WGSL_TYPE_TEXTURE_2D,
    WGSL_TYPE_TEXTURE_3D,
    WGSL_TYPE_TEXTURE_CUBE,
    WGSL_TYPE_TEXTURE_2D_ARRAY,
    WGSL_TYPE_TEXTURE_CUBE_ARRAY,
    WGSL_TYPE_TEXTURE_MULTISAMPLED_2D,
    WGSL_TYPE_TEXTURE_DEPTH_2D,
    WGSL_TYPE_TEXTURE_DEPTH_MULTISAMPLED_2D,
    WGSL_TYPE_TEXTURE_DEPTH_CUBE,
    WGSL_TYPE_TEXTURE_STORAGE_1D,
    WGSL_TYPE_TEXTURE_STORAGE_2D,
    WGSL_TYPE_TEXTURE_STORAGE_3D,
    WGSL_TYPE_TEXTURE_STORAGE_2D_ARRAY
} WgslTypeTag;

typedef struct WgslBuiltinTypeInfo {
    WgslTypeTag tag;
    WgslScalarKind scalar;   /* for SCALAR/VEC: element kind; 0 if no implicit */
    uint8_t vec_n;            /* for VEC: 2/3/4 */
    uint8_t mat_c;            /* for MAT: 2/3/4 */
    uint8_t mat_r;            /* for MAT: 2/3/4 */
} WgslBuiltinTypeInfo;

static inline WgslScalarKind sw_parse_scalar_name(const char *s, size_t n) {
    if (sw_str_eq_n(s, n, "void")) return WGSL_SCALAR_VOID;
    if (sw_str_eq_n(s, n, "bool")) return WGSL_SCALAR_BOOL;
    if (sw_str_eq_n(s, n, "i32")) return WGSL_SCALAR_I32;
    if (sw_str_eq_n(s, n, "u32")) return WGSL_SCALAR_U32;
    if (sw_str_eq_n(s, n, "f32")) return WGSL_SCALAR_F32;
    if (sw_str_eq_n(s, n, "f16")) return WGSL_SCALAR_F16;
    return WGSL_SCALAR_NONE;
}

static inline WgslScalarKind sw_vec_suffix_to_scalar(char c) {
    switch (c) {
        case 'f': return WGSL_SCALAR_F32;
        case 'i': return WGSL_SCALAR_I32;
        case 'u': return WGSL_SCALAR_U32;
        case 'h': return WGSL_SCALAR_F16;
        default:  return WGSL_SCALAR_NONE;
    }
}

static inline int sw_parse_builtin_type_name(const char *s, size_t n,
                                             WgslBuiltinTypeInfo *out) {
    out->tag = WGSL_TYPE_UNKNOWN;
    out->scalar = WGSL_SCALAR_NONE;
    out->vec_n = 0; out->mat_c = 0; out->mat_r = 0;
    if (!s || n == 0) return 0;
    WgslScalarKind sk = sw_parse_scalar_name(s, n);
    if (sk != WGSL_SCALAR_NONE) {
        out->tag = WGSL_TYPE_SCALAR;
        out->scalar = sk;
        return 1;
    }
    if (n >= 4 && s[0] == 'v' && s[1] == 'e' && s[2] == 'c') {
        char d = s[3];
        if (d >= '2' && d <= '4') {
            out->tag = WGSL_TYPE_VEC;
            out->vec_n = (uint8_t)(d - '0');
            if (n == 4) { out->scalar = WGSL_SCALAR_NONE; return 1; }
            if (n == 5) {
                WgslScalarKind ss = sw_vec_suffix_to_scalar(s[4]);
                if (ss != WGSL_SCALAR_NONE) { out->scalar = ss; return 1; }
            }
            out->tag = WGSL_TYPE_UNKNOWN;
            out->vec_n = 0;
        }
    }
    if (n >= 5 && s[0] == 'm' && s[1] == 'a' && s[2] == 't') {
        char c = s[3], x = s[4];
        if (c >= '2' && c <= '4' && x == 'x' && n >= 6) {
            char r = s[5];
            if (r >= '2' && r <= '4') {
                if (n == 6) {
                    out->tag = WGSL_TYPE_MAT;
                    out->mat_c = (uint8_t)(c - '0');
                    out->mat_r = (uint8_t)(r - '0');
                    out->scalar = WGSL_SCALAR_NONE;
                    return 1;
                }
                if (n == 7) {
                    WgslScalarKind ss = sw_vec_suffix_to_scalar(s[6]);
                    if (ss != WGSL_SCALAR_NONE) {
                        out->tag = WGSL_TYPE_MAT;
                        out->mat_c = (uint8_t)(c - '0');
                        out->mat_r = (uint8_t)(r - '0');
                        out->scalar = ss;
                        return 1;
                    }
                }
            }
        }
    }
    if (sw_str_eq_n(s, n, "array"))         { out->tag = WGSL_TYPE_ARRAY; return 1; }
    if (sw_str_eq_n(s, n, "binding_array")) { out->tag = WGSL_TYPE_BINDING_ARRAY; return 1; }
    if (sw_str_eq_n(s, n, "ptr"))           { out->tag = WGSL_TYPE_PTR; return 1; }
    if (sw_str_eq_n(s, n, "atomic"))        { out->tag = WGSL_TYPE_ATOMIC; return 1; }
    if (sw_str_eq_n(s, n, "sampler"))       { out->tag = WGSL_TYPE_SAMPLER; return 1; }
    if (sw_str_eq_n(s, n, "sampler_comparison")) { out->tag = WGSL_TYPE_SAMPLER_COMPARISON; return 1; }
    if (sw_str_eq_n(s, n, "texture_1d"))    { out->tag = WGSL_TYPE_TEXTURE_1D; return 1; }
    if (sw_str_eq_n(s, n, "texture_2d"))    { out->tag = WGSL_TYPE_TEXTURE_2D; return 1; }
    if (sw_str_eq_n(s, n, "texture_3d"))    { out->tag = WGSL_TYPE_TEXTURE_3D; return 1; }
    if (sw_str_eq_n(s, n, "texture_cube"))  { out->tag = WGSL_TYPE_TEXTURE_CUBE; return 1; }
    if (sw_str_eq_n(s, n, "texture_2d_array")) { out->tag = WGSL_TYPE_TEXTURE_2D_ARRAY; return 1; }
    if (sw_str_eq_n(s, n, "texture_cube_array")) { out->tag = WGSL_TYPE_TEXTURE_CUBE_ARRAY; return 1; }
    if (sw_str_eq_n(s, n, "texture_multisampled_2d")) { out->tag = WGSL_TYPE_TEXTURE_MULTISAMPLED_2D; return 1; }
    if (sw_str_eq_n(s, n, "texture_depth_2d")) { out->tag = WGSL_TYPE_TEXTURE_DEPTH_2D; return 1; }
    if (sw_str_eq_n(s, n, "texture_depth_multisampled_2d")) { out->tag = WGSL_TYPE_TEXTURE_DEPTH_MULTISAMPLED_2D; return 1; }
    if (sw_str_eq_n(s, n, "texture_depth_cube")) { out->tag = WGSL_TYPE_TEXTURE_DEPTH_CUBE; return 1; }
    if (sw_str_eq_n(s, n, "texture_storage_1d")) { out->tag = WGSL_TYPE_TEXTURE_STORAGE_1D; return 1; }
    if (sw_str_eq_n(s, n, "texture_storage_2d")) { out->tag = WGSL_TYPE_TEXTURE_STORAGE_2D; return 1; }
    if (sw_str_eq_n(s, n, "texture_storage_3d")) { out->tag = WGSL_TYPE_TEXTURE_STORAGE_3D; return 1; }
    if (sw_str_eq_n(s, n, "texture_storage_2d_array")) { out->tag = WGSL_TYPE_TEXTURE_STORAGE_2D_ARRAY; return 1; }
    return 0;
}

static inline WgslAttrKind sw_parse_attr_kind(const char *s, size_t n) {
    if (!s || n == 0) return WGSL_ATTR_UNKNOWN;
    if (sw_str_eq_n(s, n, "group")) return WGSL_ATTR_GROUP;
    if (sw_str_eq_n(s, n, "binding")) return WGSL_ATTR_BINDING;
    if (sw_str_eq_n(s, n, "location")) return WGSL_ATTR_LOCATION;
    if (sw_str_eq_n(s, n, "builtin")) return WGSL_ATTR_BUILTIN;
    if (sw_str_eq_n(s, n, "workgroup_size")) return WGSL_ATTR_WORKGROUP_SIZE;
    if (sw_str_eq_n(s, n, "interpolate")) return WGSL_ATTR_INTERPOLATE;
    if (sw_str_eq_n(s, n, "align")) return WGSL_ATTR_ALIGN;
    if (sw_str_eq_n(s, n, "size")) return WGSL_ATTR_SIZE;
    if (sw_str_eq_n(s, n, "id")) return WGSL_ATTR_ID;
    if (sw_str_eq_n(s, n, "vertex")) return WGSL_ATTR_VERTEX;
    if (sw_str_eq_n(s, n, "fragment")) return WGSL_ATTR_FRAGMENT;
    if (sw_str_eq_n(s, n, "compute")) return WGSL_ATTR_COMPUTE;
    if (sw_str_eq_n(s, n, "invariant")) return WGSL_ATTR_INVARIANT;
    if (sw_str_eq_n(s, n, "must_use")) return WGSL_ATTR_MUST_USE;
    if (sw_str_eq_n(s, n, "stride")) return WGSL_ATTR_STRIDE;
    if (sw_str_eq_n(s, n, "flat")) return WGSL_ATTR_FLAT;
    if (sw_str_eq_n(s, n, "push_constant")) return WGSL_ATTR_PUSH_CONSTANT;
    return WGSL_ATTR_UNKNOWN;
}

static inline const char *sw_addr_space_name(WgslAstAddrSpace a) {
    switch (a) {
        case WGSL_ADDR_FUNCTION: return "function";
        case WGSL_ADDR_PRIVATE: return "private";
        case WGSL_ADDR_WORKGROUP: return "workgroup";
        case WGSL_ADDR_UNIFORM: return "uniform";
        case WGSL_ADDR_STORAGE: return "storage";
        case WGSL_ADDR_IMMEDIATE: return "immediate";
        case WGSL_ADDR_DEVICE: return "device";
        case WGSL_ADDR_IN: return "in";
        case WGSL_ADDR_OUT: return "out";
        case WGSL_ADDR_PUSH_CONSTANT: return "push_constant";
        case WGSL_ADDR_HANDLE: return "handle";
        case WGSL_ADDR_UNKNOWN: return "unknown";
        case WGSL_ADDR_NONE: default: return NULL;
    }
}

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
    if (n >= (int)sizeof(buf)) {
        char *dyn = (char *)malloc(n + 1);
        if (dyn) {
            va_start(a, fmt);
            vsnprintf(dyn, n + 1, fmt, a);
            va_end(a);
            sw_sb_append_with(sb, dyn, realloc_fn);
            free(dyn);
            return;
        }
    }
    if (n > 0) sw_sb_append_with(sb, buf, realloc_fn);
}

static inline void sw_sb_appendf(SwStringBuffer *sb, const char *fmt, ...) {
    char buf[1024];
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    if (n >= (int)sizeof(buf)) {
        char *dyn = (char *)malloc(n + 1);
        if (dyn) {
            va_start(a, fmt);
            vsnprintf(dyn, n + 1, fmt, a);
            va_end(a);
            sw_sb_append(sb, dyn);
            free(dyn);
            return;
        }
    }
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
    while (nc < needed) {
        int doubled = nc * 2;
        if (doubled <= nc) return NULL;  /* overflow */
        nc = doubled;
    }
    void *np = realloc_fn(p, (size_t)nc * elem);
    if (!np) return NULL;  /* OOM: old pointer preserved in *p, caller must check */
    *cap = nc;
    return np;
}

/* Function wrappers for allocator macros (for use as function pointers) */
static inline void *sw_malloc_fn(size_t sz) { return SW_MALLOC(sz); }
static inline void sw_free_fn(void *p) { SW_FREE(p); }

/* ============================================================================
 * Diagnostic emission — implemented in wgsl_diagnostics.c
 * Internal; not part of the public simple_wgsl.h API.
 * ============================================================================ */

WgslDiagnosticList *wgsl_diag_list_new(void);

void wgsl_diag_list_append(WgslDiagnosticList *list,
                           WgslDiagnosticSeverity sev,
                           WgslDiagnosticCode code,
                           const char *owned_message,
                           const char *begin,
                           const char *end);

/* Append all items of `b` to `a`, then free the outer struct of `b`.
 * Transfers item ownership. If a is NULL, returns b unchanged. If b is
 * NULL, returns a unchanged. On OOM while growing a, b's items are still
 * freed (message + struct) and a is returned as-is. */
WgslDiagnosticList *wgsl_diag_list_concat(WgslDiagnosticList *a,
                                          WgslDiagnosticList *b);

/* Format into a newly SW_MALLOC'd heap buffer. Returns NULL on failure. */
char *wgsl_diag_vformat(const char *fmt, va_list ap);

/* ============================================================================
 * Shared emitter helpers — use-count + inst-map building
 * ============================================================================ */

static inline void sw_emitter_build_maps(
    const SsirModule *mod, const SsirFunction *fn,
    uint32_t **use_counts, SsirInst ***inst_map, uint32_t *inst_map_cap,
    void (*free_fn)(void *), void *(*malloc_fn)(size_t))
{
    uint32_t next_id = mod->next_id;

    /* Compute use counts for inlining decisions */
    free_fn(*use_counts);
    *use_counts = (uint32_t *)malloc_fn(next_id * sizeof(uint32_t));
    if (*use_counts) {
        memset(*use_counts, 0, next_id * sizeof(uint32_t));
        ssir_count_uses((SsirFunction *)fn, *use_counts, next_id);
    }

    /* Build instruction lookup map */
    free_fn(*inst_map);
    *inst_map_cap = next_id;
    *inst_map = (SsirInst **)malloc_fn(next_id * sizeof(SsirInst *));
    if (*inst_map) {
        memset(*inst_map, 0, next_id * sizeof(SsirInst *));
        for (uint32_t bi = 0; bi < fn->block_count; bi++) {
            SsirBlock *blk = &fn->blocks[bi];
            for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
                uint32_t rid = blk->insts[ii].result;
                if (rid && rid < next_id)
                    (*inst_map)[rid] = &blk->insts[ii];
            }
        }
    }
}

/* Resolve base type for access chain type tracing */
static inline uint32_t sw_resolve_access_base_type(
    SsirModule *mod, uint32_t base_id,
    SsirInst *(*find_inst)(void *ctx, uint32_t id), void *ctx)
{
    uint32_t cur_type_id = 0;
    SsirGlobalVar *ag = ssir_get_global(mod, base_id);
    if (ag) {
        SsirType *pt = ssir_get_type(mod, ag->type);
        if (pt && pt->kind == SSIR_TYPE_PTR)
            cur_type_id = pt->ptr.pointee;
    }
    if (!cur_type_id) {
        SsirInst *ai = find_inst(ctx, base_id);
        if (ai && ai->op == SSIR_OP_LOAD) {
            SsirGlobalVar *lg = ssir_get_global(mod, ai->operands[0]);
            if (lg) {
                SsirType *pt = ssir_get_type(mod, lg->type);
                if (pt && pt->kind == SSIR_TYPE_PTR)
                    cur_type_id = pt->ptr.pointee;
            }
        }
    }
    return cur_type_id;
}

/* Advance type through one access chain index, returning new type_id.
 * Sets *out_member_idx and *out_member_name for struct member accesses. */
static inline uint32_t sw_advance_access_type(
    SsirModule *mod, uint32_t cur_type_id, uint32_t index_id,
    uint32_t *out_member_idx, const char **out_member_name, int *out_is_const)
{
    *out_member_idx = 0;
    *out_member_name = NULL;
    *out_is_const = 0;

    SsirConstant *ic = ssir_get_constant(mod, index_id);
    SsirType *cur_st = cur_type_id ? ssir_get_type(mod, cur_type_id) : NULL;

    if (ic && (ic->kind == SSIR_CONST_U32 || ic->kind == SSIR_CONST_I32)) {
        *out_is_const = 1;
        uint32_t midx = (ic->kind == SSIR_CONST_U32) ? ic->u32_val : (uint32_t)ic->i32_val;
        *out_member_idx = midx;
        if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT &&
            cur_st->struc.member_names && midx < cur_st->struc.member_count)
            *out_member_name = cur_st->struc.member_names[midx];
        /* Advance type through struct member */
        if (cur_st && cur_st->kind == SSIR_TYPE_STRUCT && midx < cur_st->struc.member_count)
            return cur_st->struc.members[midx];
        return 0;
    } else {
        /* Advance type through array element */
        if (cur_st && (cur_st->kind == SSIR_TYPE_ARRAY || cur_st->kind == SSIR_TYPE_RUNTIME_ARRAY))
            return cur_st->array.elem;
        return 0;
    }
}

/* ============================================================================
 * sw_set_name_escaped - Shared reserved-word escaping for non-WGSL emitters
 * ============================================================================ */

static inline void sw_set_name_escaped(char **names, uint32_t cap, uint32_t id,
                                        const char *name,
                                        const char *const *reserved, int reserved_count,
                                        void *(*malloc_fn)(size_t))
{
    if (id >= cap || !name) return;
    for (int i = 0; i < reserved_count; i++) {
        if (strcmp(name, reserved[i]) == 0) {
            size_t len = strlen(name) + 2;
            char *escaped = (char *)malloc_fn(len);
            if (escaped) {
                escaped[0] = '_';
                memcpy(escaped + 1, name, len - 1);
                names[id] = escaped;
            }
            return;
        }
    }
    names[id] = sw_strdup(name, malloc_fn);
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
