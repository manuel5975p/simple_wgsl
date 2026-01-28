/*
 * SSIR (Simple Shader IR) - Implementation
 */

#include "ssir.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

static int ssir_grow_array(void **data, uint32_t *capacity, uint32_t elem_size, uint32_t needed) {
    if (*capacity >= needed) return 1;
    uint32_t new_cap = *capacity ? *capacity : 8;
    while (new_cap < needed) new_cap *= 2;
    void *new_data = SSIR_REALLOC(*data, (size_t)new_cap * elem_size);
    if (!new_data) return 0;
    *data = new_data;
    *capacity = new_cap;
    return 1;
}

static char *ssir_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *dup = (char *)SSIR_MALLOC(len);
    if (dup) memcpy(dup, s, len);
    return dup;
}

/* ============================================================================
 * Module API
 * ============================================================================ */

SsirModule *ssir_module_create(void) {
    SsirModule *mod = (SsirModule *)SSIR_MALLOC(sizeof(SsirModule));
    if (!mod) return NULL;
    memset(mod, 0, sizeof(SsirModule));
    mod->next_id = 1; /* start IDs at 1 (0 is invalid/null) */
    return mod;
}

static void ssir_free_type(SsirType *t) {
    if (!t) return;
    if (t->kind == SSIR_TYPE_STRUCT) {
        SSIR_FREE((void *)t->struc.name);
        SSIR_FREE(t->struc.members);
        SSIR_FREE(t->struc.offsets);
    }
}

static void ssir_free_constant(SsirConstant *c) {
    if (!c) return;
    if (c->kind == SSIR_CONST_COMPOSITE) {
        SSIR_FREE(c->composite.components);
    }
}

static void ssir_free_global(SsirGlobalVar *g) {
    if (!g) return;
    SSIR_FREE((void *)g->name);
}

static void ssir_free_block(SsirBlock *b) {
    if (!b) return;
    SSIR_FREE((void *)b->name);
    for (uint32_t i = 0; i < b->inst_count; i++) {
        SSIR_FREE(b->insts[i].extra);
    }
    SSIR_FREE(b->insts);
}

static void ssir_free_function(SsirFunction *f) {
    if (!f) return;
    SSIR_FREE((void *)f->name);
    for (uint32_t i = 0; i < f->param_count; i++) {
        SSIR_FREE((void *)f->params[i].name);
    }
    SSIR_FREE(f->params);
    for (uint32_t i = 0; i < f->local_count; i++) {
        SSIR_FREE((void *)f->locals[i].name);
    }
    SSIR_FREE(f->locals);
    for (uint32_t i = 0; i < f->block_count; i++) {
        ssir_free_block(&f->blocks[i]);
    }
    SSIR_FREE(f->blocks);
}

static void ssir_free_entry_point(SsirEntryPoint *ep) {
    if (!ep) return;
    SSIR_FREE((void *)ep->name);
    SSIR_FREE(ep->interface);
}

void ssir_module_destroy(SsirModule *mod) {
    if (!mod) return;

    for (uint32_t i = 0; i < mod->type_count; i++) {
        ssir_free_type(&mod->types[i]);
    }
    SSIR_FREE(mod->types);

    for (uint32_t i = 0; i < mod->constant_count; i++) {
        ssir_free_constant(&mod->constants[i]);
    }
    SSIR_FREE(mod->constants);

    for (uint32_t i = 0; i < mod->global_count; i++) {
        ssir_free_global(&mod->globals[i]);
    }
    SSIR_FREE(mod->globals);

    for (uint32_t i = 0; i < mod->function_count; i++) {
        ssir_free_function(&mod->functions[i]);
    }
    SSIR_FREE(mod->functions);

    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        ssir_free_entry_point(&mod->entry_points[i]);
    }
    SSIR_FREE(mod->entry_points);

    SSIR_FREE(mod);
}

uint32_t ssir_module_alloc_id(SsirModule *mod) {
    return mod->next_id++;
}

/* ============================================================================
 * Type API - Internal
 * ============================================================================ */

static uint32_t ssir_add_type(SsirModule *mod, SsirType *t) {
    if (!ssir_grow_array((void **)&mod->types, &mod->type_capacity,
                         sizeof(SsirType), mod->type_count + 1)) {
        return 0;
    }
    uint32_t id = ssir_module_alloc_id(mod);
    t->id = id;
    mod->types[mod->type_count++] = *t;
    return id;
}

/* Find existing type that matches - returns type ID (not array index) */
static uint32_t ssir_find_type(SsirModule *mod, SsirTypeKind kind) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == kind) return mod->types[i].id;
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_vec_type(SsirModule *mod, uint32_t elem, uint8_t size) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_VEC &&
            mod->types[i].vec.elem == elem &&
            mod->types[i].vec.size == size) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_mat_type(SsirModule *mod, uint32_t elem, uint8_t cols, uint8_t rows) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_MAT &&
            mod->types[i].mat.elem == elem &&
            mod->types[i].mat.cols == cols &&
            mod->types[i].mat.rows == rows) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_array_type(SsirModule *mod, uint32_t elem, uint32_t length) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_ARRAY &&
            mod->types[i].array.elem == elem &&
            mod->types[i].array.length == length) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_runtime_array_type(SsirModule *mod, uint32_t elem) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_RUNTIME_ARRAY &&
            mod->types[i].runtime_array.elem == elem) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_ptr_type(SsirModule *mod, uint32_t pointee, SsirAddressSpace space) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_PTR &&
            mod->types[i].ptr.pointee == pointee &&
            mod->types[i].ptr.space == space) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_texture_type(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE &&
            mod->types[i].texture.dim == dim &&
            mod->types[i].texture.sampled_type == sampled_type) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_texture_storage_type(SsirModule *mod, SsirTextureDim dim,
                                               uint32_t format, SsirAccessMode access) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE_STORAGE &&
            mod->types[i].texture_storage.dim == dim &&
            mod->types[i].texture_storage.format == format &&
            mod->types[i].texture_storage.access == access) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

static uint32_t ssir_find_texture_depth_type(SsirModule *mod, SsirTextureDim dim) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_TEXTURE_DEPTH &&
            mod->types[i].texture_depth.dim == dim) {
            return mod->types[i].id;
        }
    }
    return UINT32_MAX;
}

/* ============================================================================
 * Type API - Public
 * ============================================================================ */

uint32_t ssir_type_void(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_VOID);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_VOID };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_bool(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_BOOL);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_BOOL };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_i32(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_I32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_I32 };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_u32(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_U32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_U32 };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_f32(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_F32);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_F32 };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_f16(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_F16);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_F16 };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_vec(SsirModule *mod, uint32_t elem_type, uint8_t size) {
    uint32_t id = ssir_find_vec_type(mod, elem_type, size);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_VEC };
    t.vec.elem = elem_type;
    t.vec.size = size;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_mat(SsirModule *mod, uint32_t col_type, uint8_t cols, uint8_t rows) {
    uint32_t id = ssir_find_mat_type(mod, col_type, cols, rows);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_MAT };
    t.mat.elem = col_type;
    t.mat.cols = cols;
    t.mat.rows = rows;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_array(SsirModule *mod, uint32_t elem_type, uint32_t length) {
    uint32_t id = ssir_find_array_type(mod, elem_type, length);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_ARRAY };
    t.array.elem = elem_type;
    t.array.length = length;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_runtime_array(SsirModule *mod, uint32_t elem_type) {
    uint32_t id = ssir_find_runtime_array_type(mod, elem_type);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_RUNTIME_ARRAY };
    t.runtime_array.elem = elem_type;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_struct(SsirModule *mod, const char *name,
                          const uint32_t *members, uint32_t member_count,
                          const uint32_t *offsets) {
    /* Structs are not deduplicated (they can have the same layout but different names) */
    SsirType t = { .kind = SSIR_TYPE_STRUCT };
    t.struc.name = ssir_strdup(name);
    if (member_count > 0) {
        t.struc.members = (uint32_t *)SSIR_MALLOC(member_count * sizeof(uint32_t));
        if (!t.struc.members) return UINT32_MAX;
        memcpy(t.struc.members, members, member_count * sizeof(uint32_t));

        if (offsets) {
            t.struc.offsets = (uint32_t *)SSIR_MALLOC(member_count * sizeof(uint32_t));
            if (!t.struc.offsets) {
                SSIR_FREE(t.struc.members);
                return UINT32_MAX;
            }
            memcpy(t.struc.offsets, offsets, member_count * sizeof(uint32_t));
        }
    }
    t.struc.member_count = member_count;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_ptr(SsirModule *mod, uint32_t pointee_type, SsirAddressSpace space) {
    uint32_t id = ssir_find_ptr_type(mod, pointee_type, space);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_PTR };
    t.ptr.pointee = pointee_type;
    t.ptr.space = space;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_sampler(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_SAMPLER);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_SAMPLER };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_sampler_comparison(SsirModule *mod) {
    uint32_t id = ssir_find_type(mod, SSIR_TYPE_SAMPLER_COMPARISON);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_SAMPLER_COMPARISON };
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_texture(SsirModule *mod, SsirTextureDim dim, uint32_t sampled_type) {
    uint32_t id = ssir_find_texture_type(mod, dim, sampled_type);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE };
    t.texture.dim = dim;
    t.texture.sampled_type = sampled_type;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_texture_storage(SsirModule *mod, SsirTextureDim dim,
                                   uint32_t format, SsirAccessMode access) {
    uint32_t id = ssir_find_texture_storage_type(mod, dim, format, access);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE_STORAGE };
    t.texture_storage.dim = dim;
    t.texture_storage.format = format;
    t.texture_storage.access = access;
    return ssir_add_type(mod, &t);
}

uint32_t ssir_type_texture_depth(SsirModule *mod, SsirTextureDim dim) {
    uint32_t id = ssir_find_texture_depth_type(mod, dim);
    if (id != UINT32_MAX) return id;
    SsirType t = { .kind = SSIR_TYPE_TEXTURE_DEPTH };
    t.texture_depth.dim = dim;
    return ssir_add_type(mod, &t);
}

SsirType *ssir_get_type(SsirModule *mod, uint32_t type_id) {
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].id == type_id) {
            return &mod->types[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Type Classification Helpers
 * ============================================================================ */

bool ssir_type_is_scalar(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_BOOL ||
           t->kind == SSIR_TYPE_I32 ||
           t->kind == SSIR_TYPE_U32 ||
           t->kind == SSIR_TYPE_F32 ||
           t->kind == SSIR_TYPE_F16;
}

bool ssir_type_is_integer(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_I32 || t->kind == SSIR_TYPE_U32;
}

bool ssir_type_is_signed(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_I32;
}

bool ssir_type_is_unsigned(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_U32;
}

bool ssir_type_is_float(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_F32 || t->kind == SSIR_TYPE_F16;
}

bool ssir_type_is_bool(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_BOOL;
}

bool ssir_type_is_vector(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_VEC;
}

bool ssir_type_is_matrix(const SsirType *t) {
    if (!t) return false;
    return t->kind == SSIR_TYPE_MAT;
}

bool ssir_type_is_numeric(const SsirType *t) {
    if (!t) return false;
    return ssir_type_is_integer(t) || ssir_type_is_float(t);
}

uint32_t ssir_type_scalar_of(SsirModule *mod, uint32_t type_id) {
    SsirType *t = ssir_get_type(mod, type_id);
    if (!t) return UINT32_MAX;

    switch (t->kind) {
        case SSIR_TYPE_VEC:
            return t->vec.elem;
        case SSIR_TYPE_MAT:
            /* Matrix element type is a vector, get its element */
            return ssir_type_scalar_of(mod, t->mat.elem);
        default:
            /* Already scalar or not applicable */
            return type_id;
    }
}

/* ============================================================================
 * Constant API
 * ============================================================================ */

static uint32_t ssir_add_constant(SsirModule *mod, SsirConstant *c) {
    if (!ssir_grow_array((void **)&mod->constants, &mod->constant_capacity,
                         sizeof(SsirConstant), mod->constant_count + 1)) {
        return 0;
    }
    c->id = ssir_module_alloc_id(mod);
    mod->constants[mod->constant_count++] = *c;
    return c->id;
}

uint32_t ssir_const_bool(SsirModule *mod, bool val) {
    /* Check for existing constant */
    uint32_t bool_type = ssir_type_bool(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_BOOL &&
            mod->constants[i].bool_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = bool_type,
        .kind = SSIR_CONST_BOOL,
        .bool_val = val
    };
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_i32(SsirModule *mod, int32_t val) {
    uint32_t i32_type = ssir_type_i32(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_I32 &&
            mod->constants[i].i32_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = i32_type,
        .kind = SSIR_CONST_I32,
        .i32_val = val
    };
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_u32(SsirModule *mod, uint32_t val) {
    uint32_t u32_type = ssir_type_u32(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_U32 &&
            mod->constants[i].u32_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = u32_type,
        .kind = SSIR_CONST_U32,
        .u32_val = val
    };
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_f32(SsirModule *mod, float val) {
    uint32_t f32_type = ssir_type_f32(mod);
    /* Note: comparing floats with memcmp to handle NaN/negative zero correctly */
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_F32 &&
            memcmp(&mod->constants[i].f32_val, &val, sizeof(float)) == 0) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = f32_type,
        .kind = SSIR_CONST_F32,
        .f32_val = val
    };
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_f16(SsirModule *mod, uint16_t val) {
    uint32_t f16_type = ssir_type_f16(mod);
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].kind == SSIR_CONST_F16 &&
            mod->constants[i].f16_val == val) {
            return mod->constants[i].id;
        }
    }
    SsirConstant c = {
        .type = f16_type,
        .kind = SSIR_CONST_F16,
        .f16_val = val
    };
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_composite(SsirModule *mod, uint32_t type_id,
                              const uint32_t *components, uint32_t count) {
    SsirConstant c = {
        .type = type_id,
        .kind = SSIR_CONST_COMPOSITE,
    };
    if (count > 0) {
        c.composite.components = (uint32_t *)SSIR_MALLOC(count * sizeof(uint32_t));
        if (!c.composite.components) return 0;
        memcpy(c.composite.components, components, count * sizeof(uint32_t));
    }
    c.composite.count = count;
    return ssir_add_constant(mod, &c);
}

uint32_t ssir_const_null(SsirModule *mod, uint32_t type_id) {
    SsirConstant c = {
        .type = type_id,
        .kind = SSIR_CONST_NULL,
    };
    return ssir_add_constant(mod, &c);
}

SsirConstant *ssir_get_constant(SsirModule *mod, uint32_t const_id) {
    for (uint32_t i = 0; i < mod->constant_count; i++) {
        if (mod->constants[i].id == const_id) {
            return &mod->constants[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Global Variable API
 * ============================================================================ */

uint32_t ssir_global_var(SsirModule *mod, const char *name, uint32_t ptr_type) {
    if (!ssir_grow_array((void **)&mod->globals, &mod->global_capacity,
                         sizeof(SsirGlobalVar), mod->global_count + 1)) {
        return 0;
    }
    SsirGlobalVar *g = &mod->globals[mod->global_count++];
    memset(g, 0, sizeof(SsirGlobalVar));
    g->id = ssir_module_alloc_id(mod);
    g->name = ssir_strdup(name);
    g->type = ptr_type;
    return g->id;
}

SsirGlobalVar *ssir_get_global(SsirModule *mod, uint32_t global_id) {
    for (uint32_t i = 0; i < mod->global_count; i++) {
        if (mod->globals[i].id == global_id) {
            return &mod->globals[i];
        }
    }
    return NULL;
}

void ssir_global_set_group(SsirModule *mod, uint32_t global_id, uint32_t group) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_group = true;
        g->group = group;
    }
}

void ssir_global_set_binding(SsirModule *mod, uint32_t global_id, uint32_t binding) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_binding = true;
        g->binding = binding;
    }
}

void ssir_global_set_location(SsirModule *mod, uint32_t global_id, uint32_t location) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_location = true;
        g->location = location;
    }
}

void ssir_global_set_builtin(SsirModule *mod, uint32_t global_id, SsirBuiltinVar builtin) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->builtin = builtin;
    }
}

void ssir_global_set_interpolation(SsirModule *mod, uint32_t global_id, SsirInterpolation interp) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->interp = interp;
    }
}

void ssir_global_set_initializer(SsirModule *mod, uint32_t global_id, uint32_t const_id) {
    SsirGlobalVar *g = ssir_get_global(mod, global_id);
    if (g) {
        g->has_initializer = true;
        g->initializer = const_id;
    }
}

/* ============================================================================
 * Function API
 * ============================================================================ */

uint32_t ssir_function_create(SsirModule *mod, const char *name, uint32_t return_type) {
    if (!ssir_grow_array((void **)&mod->functions, &mod->function_capacity,
                         sizeof(SsirFunction), mod->function_count + 1)) {
        return 0;
    }
    SsirFunction *f = &mod->functions[mod->function_count++];
    memset(f, 0, sizeof(SsirFunction));
    f->id = ssir_module_alloc_id(mod);
    f->name = ssir_strdup(name);
    f->return_type = return_type;
    return f->id;
}

SsirFunction *ssir_get_function(SsirModule *mod, uint32_t func_id) {
    for (uint32_t i = 0; i < mod->function_count; i++) {
        if (mod->functions[i].id == func_id) {
            return &mod->functions[i];
        }
    }
    return NULL;
}

uint32_t ssir_function_add_param(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t type) {
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    uint32_t new_count = f->param_count + 1;
    SsirFunctionParam *new_params = (SsirFunctionParam *)SSIR_REALLOC(
        f->params, new_count * sizeof(SsirFunctionParam));
    if (!new_params) return 0;
    f->params = new_params;

    SsirFunctionParam *p = &f->params[f->param_count++];
    p->id = ssir_module_alloc_id(mod);
    p->name = ssir_strdup(name);
    p->type = type;
    return p->id;
}

uint32_t ssir_function_add_local(SsirModule *mod, uint32_t func_id,
                                 const char *name, uint32_t ptr_type) {
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->locals, &f->local_capacity,
                         sizeof(SsirLocalVar), f->local_count + 1)) {
        return 0;
    }

    SsirLocalVar *l = &f->locals[f->local_count++];
    memset(l, 0, sizeof(SsirLocalVar));
    l->id = ssir_module_alloc_id(mod);
    l->name = ssir_strdup(name);
    l->type = ptr_type;
    return l->id;
}

/* ============================================================================
 * Block API
 * ============================================================================ */

uint32_t ssir_block_create(SsirModule *mod, uint32_t func_id, const char *name) {
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->blocks, &f->block_capacity,
                         sizeof(SsirBlock), f->block_count + 1)) {
        return 0;
    }

    SsirBlock *b = &f->blocks[f->block_count++];
    memset(b, 0, sizeof(SsirBlock));
    b->id = ssir_module_alloc_id(mod);
    b->name = ssir_strdup(name);
    return b->id;
}

uint32_t ssir_block_create_with_id(SsirModule *mod, uint32_t func_id, uint32_t block_id, const char *name) {
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return 0;

    if (!ssir_grow_array((void **)&f->blocks, &f->block_capacity,
                         sizeof(SsirBlock), f->block_count + 1)) {
        return 0;
    }

    SsirBlock *b = &f->blocks[f->block_count++];
    memset(b, 0, sizeof(SsirBlock));
    b->id = block_id;
    b->name = ssir_strdup(name);
    return b->id;
}

SsirBlock *ssir_get_block(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    SsirFunction *f = ssir_get_function(mod, func_id);
    if (!f) return NULL;

    for (uint32_t i = 0; i < f->block_count; i++) {
        if (f->blocks[i].id == block_id) {
            return &f->blocks[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Instruction Builder - Internal
 * ============================================================================ */

static SsirInst *ssir_add_inst(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    SsirBlock *b = ssir_get_block(mod, func_id, block_id);
    if (!b) return NULL;

    if (!ssir_grow_array((void **)&b->insts, &b->inst_capacity,
                         sizeof(SsirInst), b->inst_count + 1)) {
        return NULL;
    }

    SsirInst *inst = &b->insts[b->inst_count++];
    memset(inst, 0, sizeof(SsirInst));
    return inst;
}

static uint32_t ssir_emit_binary(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                 SsirOpcode op, uint32_t type, uint32_t a, uint32_t b) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = op;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = a;
    inst->operands[1] = b;
    inst->operand_count = 2;
    return inst->result;
}

static uint32_t ssir_emit_unary(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                SsirOpcode op, uint32_t type, uint32_t a) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = op;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = a;
    inst->operand_count = 1;
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Arithmetic
 * ============================================================================ */

uint32_t ssir_build_add(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_ADD, type, a, b);
}

uint32_t ssir_build_sub(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SUB, type, a, b);
}

uint32_t ssir_build_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MUL, type, a, b);
}

uint32_t ssir_build_div(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_DIV, type, a, b);
}

uint32_t ssir_build_mod(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MOD, type, a, b);
}

uint32_t ssir_build_neg(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_NEG, type, a);
}

/* ============================================================================
 * Instruction Builder - Matrix
 * ============================================================================ */

uint32_t ssir_build_mat_mul(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_MAT_MUL, type, a, b);
}

uint32_t ssir_build_mat_transpose(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t type, uint32_t m) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_MAT_TRANSPOSE, type, m);
}

/* ============================================================================
 * Instruction Builder - Bitwise
 * ============================================================================ */

uint32_t ssir_build_bit_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_AND, type, a, b);
}

uint32_t ssir_build_bit_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_OR, type, a, b);
}

uint32_t ssir_build_bit_xor(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_BIT_XOR, type, a, b);
}

uint32_t ssir_build_bit_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t a) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_BIT_NOT, type, a);
}

uint32_t ssir_build_shl(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHL, type, a, b);
}

uint32_t ssir_build_shr(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHR, type, a, b);
}

uint32_t ssir_build_shr_logical(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_SHR_LOGICAL, type, a, b);
}

/* ============================================================================
 * Instruction Builder - Comparison
 * ============================================================================ */

uint32_t ssir_build_eq(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_EQ, type, a, b);
}

uint32_t ssir_build_ne(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_NE, type, a, b);
}

uint32_t ssir_build_lt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_LT, type, a, b);
}

uint32_t ssir_build_le(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_LE, type, a, b);
}

uint32_t ssir_build_gt(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_GT, type, a, b);
}

uint32_t ssir_build_ge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_GE, type, a, b);
}

/* ============================================================================
 * Instruction Builder - Logical
 * ============================================================================ */

uint32_t ssir_build_and(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_AND, type, a, b);
}

uint32_t ssir_build_or(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t type, uint32_t a, uint32_t b) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_OR, type, a, b);
}

uint32_t ssir_build_not(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, uint32_t a) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_NOT, type, a);
}

/* ============================================================================
 * Instruction Builder - Composite
 * ============================================================================ */

uint32_t ssir_build_construct(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t type, const uint32_t *components, uint32_t count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_CONSTRUCT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;

    /* Store in operands if fits, otherwise in extra */
    if (count <= SSIR_MAX_OPERANDS) {
        memcpy(inst->operands, components, count * sizeof(uint32_t));
        inst->operand_count = (uint8_t)count;
    } else {
        inst->operand_count = 0;
        inst->extra = (uint32_t *)SSIR_MALLOC(count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, components, count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)count;
    }
    return inst->result;
}

uint32_t ssir_build_extract(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t composite, uint32_t index) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_EXTRACT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = composite;
    inst->operands[1] = index;
    inst->operand_count = 2;
    return inst->result;
}

uint32_t ssir_build_insert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t composite, uint32_t value, uint32_t index) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_INSERT;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = composite;
    inst->operands[1] = value;
    inst->operands[2] = index;
    inst->operand_count = 3;
    return inst->result;
}

uint32_t ssir_build_shuffle(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t v1, uint32_t v2,
                            const uint32_t *indices, uint32_t index_count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_SHUFFLE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = v1;
    inst->operands[1] = v2;
    inst->operand_count = 2;

    if (index_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(index_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, indices, index_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)index_count;
    }
    return inst->result;
}

uint32_t ssir_build_splat(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t type, uint32_t scalar) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_SPLAT, type, scalar);
}

uint32_t ssir_build_extract_dyn(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                uint32_t type, uint32_t composite, uint32_t index) {
    return ssir_emit_binary(mod, func_id, block_id, SSIR_OP_EXTRACT_DYN, type, composite, index);
}

/* ============================================================================
 * Instruction Builder - Memory
 * ============================================================================ */

uint32_t ssir_build_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t ptr) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_LOAD, type, ptr);
}

void ssir_build_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                      uint32_t ptr, uint32_t value) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_STORE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = ptr;
    inst->operands[1] = value;
    inst->operand_count = 2;
}

uint32_t ssir_build_access(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, uint32_t base,
                           const uint32_t *indices, uint32_t index_count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_ACCESS;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = base;
    inst->operand_count = 1;

    if (index_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(index_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, indices, index_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)index_count;
    }
    return inst->result;
}

uint32_t ssir_build_array_len(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                              uint32_t ptr) {
    uint32_t u32_type = ssir_type_u32(mod);
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_ARRAY_LEN, u32_type, ptr);
}

/* ============================================================================
 * Instruction Builder - Control Flow
 * ============================================================================ */

void ssir_build_branch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t target_block) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BRANCH;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = target_block;
    inst->operand_count = 1;
}

void ssir_build_branch_cond_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                  uint32_t cond, uint32_t true_block, uint32_t false_block,
                                  uint32_t merge_block) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BRANCH_COND;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = cond;
    inst->operands[1] = true_block;
    inst->operands[2] = false_block;
    inst->operands[3] = merge_block;  // 0 = no merge (for unstructured), non-0 = merge block ID
    inst->operand_count = 4;
}

void ssir_build_branch_cond(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t cond, uint32_t true_block, uint32_t false_block) {
    ssir_build_branch_cond_merge(mod, func_id, block_id, cond, true_block, false_block, 0);
}

void ssir_build_loop_merge(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t merge_block, uint32_t continue_block) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_LOOP_MERGE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = merge_block;
    inst->operands[1] = continue_block;
    inst->operand_count = 2;
}

void ssir_build_switch(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t selector, uint32_t default_block,
                       const uint32_t *cases, uint32_t case_count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_SWITCH;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = selector;
    inst->operands[1] = default_block;
    inst->operand_count = 2;

    /* cases is pairs of (value, label) */
    if (case_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(case_count * 2 * sizeof(uint32_t));
        if (!inst->extra) return;
        memcpy(inst->extra, cases, case_count * 2 * sizeof(uint32_t));
        inst->extra_count = (uint16_t)(case_count * 2);
    }
}

uint32_t ssir_build_phi(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        uint32_t type, const uint32_t *incoming, uint32_t count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_PHI;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operand_count = 0;

    /* incoming is pairs of (value_id, block_id) */
    if (count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(count * 2 * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, incoming, count * 2 * sizeof(uint32_t));
        inst->extra_count = (uint16_t)(count * 2);
    }
    return inst->result;
}

void ssir_build_return(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                       uint32_t value) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_RETURN;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = value;
    inst->operand_count = 1;
}

void ssir_build_return_void(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_RETURN_VOID;
    inst->result = 0;
    inst->type = 0;
    inst->operand_count = 0;
}

void ssir_build_unreachable(SsirModule *mod, uint32_t func_id, uint32_t block_id) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_UNREACHABLE;
    inst->result = 0;
    inst->type = 0;
    inst->operand_count = 0;
}

/* ============================================================================
 * Instruction Builder - Call
 * ============================================================================ */

uint32_t ssir_build_call(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                         uint32_t type, uint32_t callee,
                         const uint32_t *args, uint32_t arg_count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_CALL;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = callee;
    inst->operand_count = 1;

    if (arg_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(arg_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, args, arg_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)arg_count;
    }
    return inst->result;
}

uint32_t ssir_build_builtin(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, SsirBuiltinId builtin,
                            const uint32_t *args, uint32_t arg_count) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_BUILTIN;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = (uint32_t)builtin;
    inst->operand_count = 1;

    if (arg_count > 0) {
        inst->extra = (uint32_t *)SSIR_MALLOC(arg_count * sizeof(uint32_t));
        if (!inst->extra) return 0;
        memcpy(inst->extra, args, arg_count * sizeof(uint32_t));
        inst->extra_count = (uint16_t)arg_count;
    }
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Conversion
 * ============================================================================ */

uint32_t ssir_build_convert(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_CONVERT, type, value);
}

uint32_t ssir_build_bitcast(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                            uint32_t type, uint32_t value) {
    return ssir_emit_unary(mod, func_id, block_id, SSIR_OP_BITCAST, type, value);
}

/* ============================================================================
 * Instruction Builder - Texture
 * ============================================================================ */

uint32_t ssir_build_tex_sample(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                               uint32_t type, uint32_t texture, uint32_t sampler,
                               uint32_t coord) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operand_count = 3;
    return inst->result;
}

uint32_t ssir_build_tex_sample_bias(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t bias) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_BIAS;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = bias;
    inst->operand_count = 4;
    return inst->result;
}

uint32_t ssir_build_tex_sample_level(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                     uint32_t type, uint32_t texture, uint32_t sampler,
                                     uint32_t coord, uint32_t lod) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_LEVEL;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = lod;
    inst->operand_count = 4;
    return inst->result;
}

uint32_t ssir_build_tex_sample_grad(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                    uint32_t type, uint32_t texture, uint32_t sampler,
                                    uint32_t coord, uint32_t ddx, uint32_t ddy) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_GRAD;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ddx;
    inst->operands[4] = ddy;
    inst->operand_count = 5;
    return inst->result;
}

uint32_t ssir_build_tex_sample_cmp(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                                   uint32_t type, uint32_t texture, uint32_t sampler,
                                   uint32_t coord, uint32_t ref) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SAMPLE_CMP;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = sampler;
    inst->operands[2] = coord;
    inst->operands[3] = ref;
    inst->operand_count = 4;
    return inst->result;
}

uint32_t ssir_build_tex_load(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t coord, uint32_t level) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_LOAD;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = coord;
    inst->operands[2] = level;
    inst->operand_count = 3;
    return inst->result;
}

void ssir_build_tex_store(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                          uint32_t texture, uint32_t coord, uint32_t value) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_TEX_STORE;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = texture;
    inst->operands[1] = coord;
    inst->operands[2] = value;
    inst->operand_count = 3;
}

uint32_t ssir_build_tex_size(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                             uint32_t type, uint32_t texture, uint32_t level) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_TEX_SIZE;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = texture;
    inst->operands[1] = level;
    inst->operand_count = 2;
    return inst->result;
}

/* ============================================================================
 * Instruction Builder - Sync
 * ============================================================================ */

void ssir_build_barrier(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                        SsirBarrierScope scope) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return;
    inst->op = SSIR_OP_BARRIER;
    inst->result = 0;
    inst->type = 0;
    inst->operands[0] = (uint32_t)scope;
    inst->operand_count = 1;
}

uint32_t ssir_build_atomic(SsirModule *mod, uint32_t func_id, uint32_t block_id,
                           uint32_t type, SsirAtomicOp op, uint32_t ptr,
                           uint32_t value, uint32_t comparator) {
    SsirInst *inst = ssir_add_inst(mod, func_id, block_id);
    if (!inst) return 0;
    inst->op = SSIR_OP_ATOMIC;
    inst->result = ssir_module_alloc_id(mod);
    inst->type = type;
    inst->operands[0] = (uint32_t)op;
    inst->operands[1] = ptr;
    inst->operands[2] = value;
    inst->operands[3] = comparator;
    inst->operand_count = 4;
    return inst->result;
}

/* ============================================================================
 * Entry Point API
 * ============================================================================ */

uint32_t ssir_entry_point_create(SsirModule *mod, SsirStage stage,
                                 uint32_t func_id, const char *name) {
    if (!ssir_grow_array((void **)&mod->entry_points, &mod->entry_point_capacity,
                         sizeof(SsirEntryPoint), mod->entry_point_count + 1)) {
        return UINT32_MAX;
    }

    uint32_t index = mod->entry_point_count++;
    SsirEntryPoint *ep = &mod->entry_points[index];
    memset(ep, 0, sizeof(SsirEntryPoint));
    ep->stage = stage;
    ep->function = func_id;
    ep->name = ssir_strdup(name);
    ep->workgroup_size[0] = 1;
    ep->workgroup_size[1] = 1;
    ep->workgroup_size[2] = 1;
    return index;
}

SsirEntryPoint *ssir_get_entry_point(SsirModule *mod, uint32_t index) {
    if (index >= mod->entry_point_count) return NULL;
    return &mod->entry_points[index];
}

void ssir_entry_point_add_interface(SsirModule *mod, uint32_t ep_index, uint32_t global_id) {
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (!ep) return;

    uint32_t new_count = ep->interface_count + 1;
    uint32_t *new_interface = (uint32_t *)SSIR_REALLOC(
        ep->interface, new_count * sizeof(uint32_t));
    if (!new_interface) return;
    ep->interface = new_interface;
    ep->interface[ep->interface_count++] = global_id;
}

void ssir_entry_point_set_workgroup_size(SsirModule *mod, uint32_t ep_index,
                                         uint32_t x, uint32_t y, uint32_t z) {
    SsirEntryPoint *ep = ssir_get_entry_point(mod, ep_index);
    if (!ep) return;
    ep->workgroup_size[0] = x;
    ep->workgroup_size[1] = y;
    ep->workgroup_size[2] = z;
}

/* ============================================================================
 * Validation
 * ============================================================================ */

static void ssir_add_validation_error(SsirValidationResult *result,
                                      SsirResult code, const char *message,
                                      uint32_t func_id, uint32_t block_id, uint32_t inst_index) {
    if (!ssir_grow_array((void **)&result->errors, &result->error_capacity,
                         sizeof(SsirValidationError), result->error_count + 1)) {
        return;
    }
    SsirValidationError *err = &result->errors[result->error_count++];
    err->code = code;
    err->message = message;
    err->func_id = func_id;
    err->block_id = block_id;
    err->inst_index = inst_index;
}

static bool ssir_is_terminator(SsirOpcode op) {
    return op == SSIR_OP_BRANCH ||
           op == SSIR_OP_BRANCH_COND ||
           op == SSIR_OP_SWITCH ||
           op == SSIR_OP_RETURN ||
           op == SSIR_OP_RETURN_VOID ||
           op == SSIR_OP_UNREACHABLE;
}

SsirValidationResult *ssir_validate(SsirModule *mod) {
    SsirValidationResult *result = (SsirValidationResult *)SSIR_MALLOC(sizeof(SsirValidationResult));
    if (!result) return NULL;
    memset(result, 0, sizeof(SsirValidationResult));
    result->valid = true;

    /* Validate each function */
    for (uint32_t fi = 0; fi < mod->function_count; fi++) {
        SsirFunction *f = &mod->functions[fi];

        /* Each function must have at least one block */
        if (f->block_count == 0) {
            ssir_add_validation_error(result, SSIR_ERROR_INVALID_BLOCK,
                "Function has no blocks", f->id, 0, 0);
            result->valid = false;
            continue;
        }

        /* Validate each block */
        for (uint32_t bi = 0; bi < f->block_count; bi++) {
            SsirBlock *b = &f->blocks[bi];

            /* Block must have instructions */
            if (b->inst_count == 0) {
                ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                    "Block has no instructions", f->id, b->id, 0);
                result->valid = false;
                continue;
            }

            /* Last instruction must be a terminator */
            SsirInst *last = &b->insts[b->inst_count - 1];
            if (!ssir_is_terminator(last->op)) {
                ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                    "Block does not end with terminator", f->id, b->id, b->inst_count - 1);
                result->valid = false;
            }

            /* Phi nodes must be at start of block */
            bool seen_non_phi = false;
            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *inst = &b->insts[ii];
                if (inst->op == SSIR_OP_PHI) {
                    if (seen_non_phi) {
                        ssir_add_validation_error(result, SSIR_ERROR_PHI_PLACEMENT,
                            "Phi node not at start of block", f->id, b->id, ii);
                        result->valid = false;
                    }
                } else {
                    seen_non_phi = true;
                }

                /* No instructions after terminator */
                if (ssir_is_terminator(inst->op) && ii < b->inst_count - 1) {
                    ssir_add_validation_error(result, SSIR_ERROR_TERMINATOR_MISSING,
                        "Instruction after terminator", f->id, b->id, ii + 1);
                    result->valid = false;
                }
            }
        }
    }

    /* Validate entry points */
    for (uint32_t ei = 0; ei < mod->entry_point_count; ei++) {
        SsirEntryPoint *ep = &mod->entry_points[ei];

        /* Entry point must reference valid function */
        SsirFunction *f = ssir_get_function(mod, ep->function);
        if (!f) {
            ssir_add_validation_error(result, SSIR_ERROR_ENTRY_POINT,
                "Entry point references invalid function", 0, 0, 0);
            result->valid = false;
        }

        /* Compute shader must have workgroup_size > 0 */
        if (ep->stage == SSIR_STAGE_COMPUTE) {
            if (ep->workgroup_size[0] == 0 ||
                ep->workgroup_size[1] == 0 ||
                ep->workgroup_size[2] == 0) {
                ssir_add_validation_error(result, SSIR_ERROR_ENTRY_POINT,
                    "Compute shader workgroup size must be > 0", 0, 0, 0);
                result->valid = false;
            }
        }
    }

    return result;
}

void ssir_validation_result_free(SsirValidationResult *result) {
    if (!result) return;
    SSIR_FREE(result->errors);
    SSIR_FREE(result);
}

/* ============================================================================
 * Debug/Utility API
 * ============================================================================ */

static const char *opcode_names[] = {
    [SSIR_OP_ADD] = "add",
    [SSIR_OP_SUB] = "sub",
    [SSIR_OP_MUL] = "mul",
    [SSIR_OP_DIV] = "div",
    [SSIR_OP_MOD] = "mod",
    [SSIR_OP_NEG] = "neg",
    [SSIR_OP_MAT_MUL] = "mat_mul",
    [SSIR_OP_MAT_TRANSPOSE] = "mat_transpose",
    [SSIR_OP_BIT_AND] = "bit_and",
    [SSIR_OP_BIT_OR] = "bit_or",
    [SSIR_OP_BIT_XOR] = "bit_xor",
    [SSIR_OP_BIT_NOT] = "bit_not",
    [SSIR_OP_SHL] = "shl",
    [SSIR_OP_SHR] = "shr",
    [SSIR_OP_SHR_LOGICAL] = "shr_logical",
    [SSIR_OP_EQ] = "eq",
    [SSIR_OP_NE] = "ne",
    [SSIR_OP_LT] = "lt",
    [SSIR_OP_LE] = "le",
    [SSIR_OP_GT] = "gt",
    [SSIR_OP_GE] = "ge",
    [SSIR_OP_AND] = "and",
    [SSIR_OP_OR] = "or",
    [SSIR_OP_NOT] = "not",
    [SSIR_OP_CONSTRUCT] = "construct",
    [SSIR_OP_EXTRACT] = "extract",
    [SSIR_OP_INSERT] = "insert",
    [SSIR_OP_SHUFFLE] = "shuffle",
    [SSIR_OP_SPLAT] = "splat",
    [SSIR_OP_EXTRACT_DYN] = "extract_dyn",
    [SSIR_OP_LOAD] = "load",
    [SSIR_OP_STORE] = "store",
    [SSIR_OP_ACCESS] = "access",
    [SSIR_OP_ARRAY_LEN] = "array_len",
    [SSIR_OP_BRANCH] = "branch",
    [SSIR_OP_BRANCH_COND] = "branch_cond",
    [SSIR_OP_SWITCH] = "switch",
    [SSIR_OP_PHI] = "phi",
    [SSIR_OP_RETURN] = "return",
    [SSIR_OP_RETURN_VOID] = "return_void",
    [SSIR_OP_UNREACHABLE] = "unreachable",
    [SSIR_OP_CALL] = "call",
    [SSIR_OP_BUILTIN] = "builtin",
    [SSIR_OP_CONVERT] = "convert",
    [SSIR_OP_BITCAST] = "bitcast",
    [SSIR_OP_TEX_SAMPLE] = "tex_sample",
    [SSIR_OP_TEX_SAMPLE_BIAS] = "tex_sample_bias",
    [SSIR_OP_TEX_SAMPLE_LEVEL] = "tex_sample_level",
    [SSIR_OP_TEX_SAMPLE_GRAD] = "tex_sample_grad",
    [SSIR_OP_TEX_SAMPLE_CMP] = "tex_sample_cmp",
    [SSIR_OP_TEX_LOAD] = "tex_load",
    [SSIR_OP_TEX_STORE] = "tex_store",
    [SSIR_OP_TEX_SIZE] = "tex_size",
    [SSIR_OP_BARRIER] = "barrier",
    [SSIR_OP_ATOMIC] = "atomic",
};

const char *ssir_opcode_name(SsirOpcode op) {
    if (op >= SSIR_OP_COUNT) return "unknown";
    return opcode_names[op] ? opcode_names[op] : "unknown";
}

static const char *builtin_names[] = {
    [SSIR_BUILTIN_SIN] = "sin",
    [SSIR_BUILTIN_COS] = "cos",
    [SSIR_BUILTIN_TAN] = "tan",
    [SSIR_BUILTIN_ASIN] = "asin",
    [SSIR_BUILTIN_ACOS] = "acos",
    [SSIR_BUILTIN_ATAN] = "atan",
    [SSIR_BUILTIN_ATAN2] = "atan2",
    [SSIR_BUILTIN_SINH] = "sinh",
    [SSIR_BUILTIN_COSH] = "cosh",
    [SSIR_BUILTIN_TANH] = "tanh",
    [SSIR_BUILTIN_ASINH] = "asinh",
    [SSIR_BUILTIN_ACOSH] = "acosh",
    [SSIR_BUILTIN_ATANH] = "atanh",
    [SSIR_BUILTIN_EXP] = "exp",
    [SSIR_BUILTIN_EXP2] = "exp2",
    [SSIR_BUILTIN_LOG] = "log",
    [SSIR_BUILTIN_LOG2] = "log2",
    [SSIR_BUILTIN_POW] = "pow",
    [SSIR_BUILTIN_SQRT] = "sqrt",
    [SSIR_BUILTIN_INVERSESQRT] = "inversesqrt",
    [SSIR_BUILTIN_ABS] = "abs",
    [SSIR_BUILTIN_SIGN] = "sign",
    [SSIR_BUILTIN_FLOOR] = "floor",
    [SSIR_BUILTIN_CEIL] = "ceil",
    [SSIR_BUILTIN_ROUND] = "round",
    [SSIR_BUILTIN_TRUNC] = "trunc",
    [SSIR_BUILTIN_FRACT] = "fract",
    [SSIR_BUILTIN_MIN] = "min",
    [SSIR_BUILTIN_MAX] = "max",
    [SSIR_BUILTIN_CLAMP] = "clamp",
    [SSIR_BUILTIN_SATURATE] = "saturate",
    [SSIR_BUILTIN_MIX] = "mix",
    [SSIR_BUILTIN_STEP] = "step",
    [SSIR_BUILTIN_SMOOTHSTEP] = "smoothstep",
    [SSIR_BUILTIN_DOT] = "dot",
    [SSIR_BUILTIN_CROSS] = "cross",
    [SSIR_BUILTIN_LENGTH] = "length",
    [SSIR_BUILTIN_DISTANCE] = "distance",
    [SSIR_BUILTIN_NORMALIZE] = "normalize",
    [SSIR_BUILTIN_FACEFORWARD] = "faceforward",
    [SSIR_BUILTIN_REFLECT] = "reflect",
    [SSIR_BUILTIN_REFRACT] = "refract",
    [SSIR_BUILTIN_ALL] = "all",
    [SSIR_BUILTIN_ANY] = "any",
    [SSIR_BUILTIN_SELECT] = "select",
    [SSIR_BUILTIN_COUNTBITS] = "countbits",
    [SSIR_BUILTIN_REVERSEBITS] = "reversebits",
    [SSIR_BUILTIN_FIRSTLEADINGBIT] = "firstleadingbit",
    [SSIR_BUILTIN_FIRSTTRAILINGBIT] = "firsttrailingbit",
    [SSIR_BUILTIN_EXTRACTBITS] = "extractbits",
    [SSIR_BUILTIN_INSERTBITS] = "insertbits",
    [SSIR_BUILTIN_DPDX] = "dpdx",
    [SSIR_BUILTIN_DPDY] = "dpdy",
    [SSIR_BUILTIN_FWIDTH] = "fwidth",
    [SSIR_BUILTIN_DPDX_COARSE] = "dpdx_coarse",
    [SSIR_BUILTIN_DPDY_COARSE] = "dpdy_coarse",
    [SSIR_BUILTIN_DPDX_FINE] = "dpdx_fine",
    [SSIR_BUILTIN_DPDY_FINE] = "dpdy_fine",
};

const char *ssir_builtin_name(SsirBuiltinId id) {
    if (id >= SSIR_BUILTIN_COUNT) return "unknown";
    return builtin_names[id] ? builtin_names[id] : "unknown";
}

static const char *type_kind_names[] = {
    [SSIR_TYPE_VOID] = "void",
    [SSIR_TYPE_BOOL] = "bool",
    [SSIR_TYPE_I32] = "i32",
    [SSIR_TYPE_U32] = "u32",
    [SSIR_TYPE_F32] = "f32",
    [SSIR_TYPE_F16] = "f16",
    [SSIR_TYPE_VEC] = "vec",
    [SSIR_TYPE_MAT] = "mat",
    [SSIR_TYPE_ARRAY] = "array",
    [SSIR_TYPE_RUNTIME_ARRAY] = "runtime_array",
    [SSIR_TYPE_STRUCT] = "struct",
    [SSIR_TYPE_PTR] = "ptr",
    [SSIR_TYPE_SAMPLER] = "sampler",
    [SSIR_TYPE_SAMPLER_COMPARISON] = "sampler_comparison",
    [SSIR_TYPE_TEXTURE] = "texture",
    [SSIR_TYPE_TEXTURE_STORAGE] = "texture_storage",
    [SSIR_TYPE_TEXTURE_DEPTH] = "texture_depth",
};

const char *ssir_type_kind_name(SsirTypeKind kind) {
    if (kind > SSIR_TYPE_TEXTURE_DEPTH) return "unknown";
    return type_kind_names[kind];
}

/* ============================================================================
 * Module to String (Debug)
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StringBuilder;

static void sb_append(StringBuilder *sb, const char *str) {
    size_t slen = strlen(str);
    if (sb->len + slen + 1 > sb->cap) {
        size_t new_cap = sb->cap ? sb->cap * 2 : 256;
        while (new_cap < sb->len + slen + 1) new_cap *= 2;
        char *new_data = (char *)SSIR_REALLOC(sb->data, new_cap);
        if (!new_data) return;
        sb->data = new_data;
        sb->cap = new_cap;
    }
    memcpy(sb->data + sb->len, str, slen);
    sb->len += slen;
    sb->data[sb->len] = '\0';
}

static void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    sb_append(sb, buf);
}

static void ssir_type_to_string(SsirModule *mod, uint32_t type_id, StringBuilder *sb) {
    SsirType *t = ssir_get_type(mod, type_id);
    if (!t) {
        sb_appendf(sb, "type_%u", type_id);
        return;
    }

    switch (t->kind) {
        case SSIR_TYPE_VOID: sb_append(sb, "void"); break;
        case SSIR_TYPE_BOOL: sb_append(sb, "bool"); break;
        case SSIR_TYPE_I32: sb_append(sb, "i32"); break;
        case SSIR_TYPE_U32: sb_append(sb, "u32"); break;
        case SSIR_TYPE_F32: sb_append(sb, "f32"); break;
        case SSIR_TYPE_F16: sb_append(sb, "f16"); break;
        case SSIR_TYPE_VEC:
            sb_append(sb, "vec");
            sb_appendf(sb, "%u<", t->vec.size);
            ssir_type_to_string(mod, t->vec.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_MAT:
            sb_appendf(sb, "mat%ux%u<", t->mat.cols, t->mat.rows);
            ssir_type_to_string(mod, t->mat.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_ARRAY:
            sb_append(sb, "array<");
            ssir_type_to_string(mod, t->array.elem, sb);
            sb_appendf(sb, ", %u>", t->array.length);
            break;
        case SSIR_TYPE_RUNTIME_ARRAY:
            sb_append(sb, "array<");
            ssir_type_to_string(mod, t->runtime_array.elem, sb);
            sb_append(sb, ">");
            break;
        case SSIR_TYPE_STRUCT:
            sb_appendf(sb, "struct %s", t->struc.name ? t->struc.name : "anon");
            break;
        case SSIR_TYPE_PTR: {
            static const char *space_names[] = {
                "function", "private", "workgroup", "uniform",
                "storage", "input", "output", "push_constant"
            };
            sb_append(sb, "ptr<");
            sb_append(sb, space_names[t->ptr.space]);
            sb_append(sb, ", ");
            ssir_type_to_string(mod, t->ptr.pointee, sb);
            sb_append(sb, ">");
            break;
        }
        case SSIR_TYPE_SAMPLER:
            sb_append(sb, "sampler");
            break;
        case SSIR_TYPE_SAMPLER_COMPARISON:
            sb_append(sb, "sampler_comparison");
            break;
        case SSIR_TYPE_TEXTURE:
            sb_append(sb, "texture");
            break;
        case SSIR_TYPE_TEXTURE_STORAGE:
            sb_append(sb, "texture_storage");
            break;
        case SSIR_TYPE_TEXTURE_DEPTH:
            sb_append(sb, "texture_depth");
            break;
    }
}

char *ssir_module_to_string(SsirModule *mod) {
    StringBuilder sb = {0};

    sb_append(&sb, "; SSIR Module\n\n");

    /* Types */
    sb_append(&sb, "; Types\n");
    for (uint32_t i = 0; i < mod->type_count; i++) {
        sb_appendf(&sb, "%%type_%u = ", i);
        ssir_type_to_string(mod, i, &sb);
        sb_append(&sb, "\n");
    }
    sb_append(&sb, "\n");

    /* Constants */
    if (mod->constant_count > 0) {
        sb_append(&sb, "; Constants\n");
        for (uint32_t i = 0; i < mod->constant_count; i++) {
            SsirConstant *c = &mod->constants[i];
            sb_appendf(&sb, "%%%u : ", c->id);
            ssir_type_to_string(mod, c->type, &sb);
            sb_append(&sb, " = ");
            switch (c->kind) {
                case SSIR_CONST_BOOL:
                    sb_append(&sb, c->bool_val ? "true" : "false");
                    break;
                case SSIR_CONST_I32:
                    sb_appendf(&sb, "%d", c->i32_val);
                    break;
                case SSIR_CONST_U32:
                    sb_appendf(&sb, "%uu", c->u32_val);
                    break;
                case SSIR_CONST_F32:
                    sb_appendf(&sb, "%f", c->f32_val);
                    break;
                case SSIR_CONST_F16:
                    sb_appendf(&sb, "0x%04xh", c->f16_val);
                    break;
                case SSIR_CONST_COMPOSITE:
                    sb_append(&sb, "composite(");
                    for (uint32_t j = 0; j < c->composite.count; j++) {
                        if (j > 0) sb_append(&sb, ", ");
                        sb_appendf(&sb, "%%%u", c->composite.components[j]);
                    }
                    sb_append(&sb, ")");
                    break;
                case SSIR_CONST_NULL:
                    sb_append(&sb, "null");
                    break;
            }
            sb_append(&sb, "\n");
        }
        sb_append(&sb, "\n");
    }

    /* Globals */
    if (mod->global_count > 0) {
        sb_append(&sb, "; Globals\n");
        for (uint32_t i = 0; i < mod->global_count; i++) {
            SsirGlobalVar *g = &mod->globals[i];
            if (g->has_group) sb_appendf(&sb, "@group(%u) ", g->group);
            if (g->has_binding) sb_appendf(&sb, "@binding(%u) ", g->binding);
            if (g->has_location) sb_appendf(&sb, "@location(%u) ", g->location);
            if (g->builtin != SSIR_BUILTIN_NONE) sb_appendf(&sb, "@builtin(%d) ", g->builtin);
            sb_appendf(&sb, "%%%u", g->id);
            if (g->name) sb_appendf(&sb, " \"%s\"", g->name);
            sb_append(&sb, " : ");
            ssir_type_to_string(mod, g->type, &sb);
            sb_append(&sb, "\n");
        }
        sb_append(&sb, "\n");
    }

    /* Functions */
    for (uint32_t fi = 0; fi < mod->function_count; fi++) {
        SsirFunction *f = &mod->functions[fi];
        sb_appendf(&sb, "fn %%%u", f->id);
        if (f->name) sb_appendf(&sb, " \"%s\"", f->name);
        sb_append(&sb, "(");
        for (uint32_t pi = 0; pi < f->param_count; pi++) {
            if (pi > 0) sb_append(&sb, ", ");
            SsirFunctionParam *p = &f->params[pi];
            sb_appendf(&sb, "%%%u : ", p->id);
            ssir_type_to_string(mod, p->type, &sb);
        }
        sb_append(&sb, ") -> ");
        ssir_type_to_string(mod, f->return_type, &sb);
        sb_append(&sb, " {\n");

        /* Locals */
        for (uint32_t li = 0; li < f->local_count; li++) {
            SsirLocalVar *l = &f->locals[li];
            sb_appendf(&sb, "    var %%%u : ", l->id);
            ssir_type_to_string(mod, l->type, &sb);
            sb_append(&sb, "\n");
        }
        if (f->local_count > 0) sb_append(&sb, "\n");

        /* Blocks */
        for (uint32_t bi = 0; bi < f->block_count; bi++) {
            SsirBlock *b = &f->blocks[bi];
            sb_appendf(&sb, "  block_%u", b->id);
            if (b->name) sb_appendf(&sb, " \"%s\"", b->name);
            sb_append(&sb, ":\n");

            for (uint32_t ii = 0; ii < b->inst_count; ii++) {
                SsirInst *inst = &b->insts[ii];
                sb_append(&sb, "    ");
                if (inst->result) {
                    sb_appendf(&sb, "%%%u : ", inst->result);
                    ssir_type_to_string(mod, inst->type, &sb);
                    sb_append(&sb, " = ");
                }
                sb_append(&sb, ssir_opcode_name(inst->op));
                for (uint8_t oi = 0; oi < inst->operand_count; oi++) {
                    sb_appendf(&sb, " %%%u", inst->operands[oi]);
                }
                if (inst->extra_count > 0) {
                    sb_append(&sb, " [");
                    for (uint16_t ei = 0; ei < inst->extra_count; ei++) {
                        if (ei > 0) sb_append(&sb, ", ");
                        sb_appendf(&sb, "%u", inst->extra[ei]);
                    }
                    sb_append(&sb, "]");
                }
                sb_append(&sb, "\n");
            }
        }
        sb_append(&sb, "}\n\n");
    }

    /* Entry points */
    if (mod->entry_point_count > 0) {
        sb_append(&sb, "; Entry points\n");
        static const char *stage_names[] = { "vertex", "fragment", "compute" };
        for (uint32_t ei = 0; ei < mod->entry_point_count; ei++) {
            SsirEntryPoint *ep = &mod->entry_points[ei];
            sb_appendf(&sb, "@%s entry \"%s\" = %%%u\n",
                      stage_names[ep->stage], ep->name ? ep->name : "", ep->function);
            if (ep->stage == SSIR_STAGE_COMPUTE) {
                sb_appendf(&sb, "  workgroup_size(%u, %u, %u)\n",
                          ep->workgroup_size[0], ep->workgroup_size[1], ep->workgroup_size[2]);
            }
            if (ep->interface_count > 0) {
                sb_append(&sb, "  interface: [");
                for (uint32_t ii = 0; ii < ep->interface_count; ii++) {
                    if (ii > 0) sb_append(&sb, ", ");
                    sb_appendf(&sb, "%%%u", ep->interface[ii]);
                }
                sb_append(&sb, "]\n");
            }
        }
    }

    return sb.data;
}
