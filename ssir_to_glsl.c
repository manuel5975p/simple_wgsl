/*
 * SSIR to GLSL 450 Converter
 *
 * Converts SSIR (Simple Shader IR) to GLSL 450 (Vulkan-compatible) text.
 * Modeled after ssir_to_wgsl.c with GLSL-specific syntax and conventions.
 */

#include "simple_wgsl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef STG_MALLOC
#define STG_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef STG_REALLOC
#define STG_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef STG_FREE
#define STG_FREE(p) free((p))
#endif

/* ============================================================================
 * String Buffer
 * ============================================================================ */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int indent;
} GlslBuf;

static void gb_init(GlslBuf *b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    b->indent = 0;
}

static void gb_free(GlslBuf *b) {
    STG_FREE(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static int gb_reserve(GlslBuf *b, size_t need) {
    if (b->len + need + 1 <= b->cap) return 1;
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->len + need + 1) nc *= 2;
    char *nd = (char *)STG_REALLOC(b->data, nc);
    if (!nd) return 0;
    b->data = nd;
    b->cap = nc;
    return 1;
}

static void gb_append(GlslBuf *b, const char *s) {
    size_t sl = strlen(s);
    if (!gb_reserve(b, sl)) return;
    memcpy(b->data + b->len, s, sl);
    b->len += sl;
    b->data[b->len] = '\0';
}

static void gb_appendf(GlslBuf *b, const char *fmt, ...) {
    char buf[1024];
    va_list a;
    va_start(a, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, a);
    va_end(a);
    if (n > 0) gb_append(b, buf);
}

static void gb_indent(GlslBuf *b) {
    for (int i = 0; i < b->indent; i++) gb_append(b, "    ");
}

static void gb_nl(GlslBuf *b) { gb_append(b, "\n"); }

/* ============================================================================
 * Converter Context
 * ============================================================================ */

typedef struct {
    const SsirModule *mod;
    SsirToGlslOptions opts;
    GlslBuf sb;

    char **id_names;
    uint32_t id_names_cap;

    const SsirFunction *current_func;
    SsirEntryPoint *active_ep;

    char last_error[256];
} GlslCtx;

static void gctx_init(GlslCtx *c, const SsirModule *m, const SsirToGlslOptions *o) {
    memset(c, 0, sizeof(*c));
    c->mod = m;
    if (o) c->opts = *o;
    gb_init(&c->sb);
    c->id_names_cap = m->next_id;
    c->id_names = (char **)STG_MALLOC(c->id_names_cap * sizeof(char *));
}

static void gctx_free(GlslCtx *c) {
    if (c->id_names) {
        for (uint32_t i = 0; i < c->id_names_cap; i++) STG_FREE(c->id_names[i]);
        STG_FREE(c->id_names);
    }
    gb_free(&c->sb);
}

static const char *get_name(GlslCtx *c, uint32_t id) {
    if (id < c->id_names_cap && c->id_names[id]) return c->id_names[id];
    static char buf[32];
    snprintf(buf, sizeof(buf), "_v%u", id);
    return buf;
}

static int is_glsl_reserved(const char *n) {
    static const char *kw[] = {
        "input", "output", "attribute", "varying", "buffer",
        "sampler", "filter", "sizeof", "cast", "namespace", "using",
        "smooth", "flat", "noperspective", "patch", "subroutine",
        "common", "partition", "active", "asm", "class", "union",
        "enum", "typedef", "template", "this", "resource",
        "goto", "inline", "noinline", "public", "static",
        "extern", "external", "interface", "long", "short",
        "half", "fixed", "unsigned", "superp", "lowp", "mediump", "highp",
        NULL
    };
    for (const char **p = kw; *p; p++)
        if (strcmp(n, *p) == 0) return 1;
    return 0;
}

static void set_name(GlslCtx *c, uint32_t id, const char *n) {
    if (id >= c->id_names_cap) return;
    STG_FREE(c->id_names[id]);
    if (n && is_glsl_reserved(n)) {
        size_t len = strlen(n) + 2;
        char *tmp = (char *)STG_MALLOC(len);
        snprintf(tmp, len, "_%s", n);
        c->id_names[id] = tmp;
    } else {
        c->id_names[id] = n ? strdup(n) : NULL;
    }
}

/* ============================================================================
 * GLSL Builtin Variable Mappings
 * ============================================================================ */

static const char *builtin_to_glsl(SsirBuiltinVar bv, SsirAddressSpace space) {
    if (space == SSIR_ADDR_OUTPUT) {
        switch (bv) {
        case SSIR_BUILTIN_POSITION:    return "gl_Position";
        case SSIR_BUILTIN_FRAG_DEPTH:  return "gl_FragDepth";
        case SSIR_BUILTIN_SAMPLE_MASK: return "gl_SampleMask[0]";
        default: break;
        }
    }
    switch (bv) {
    case SSIR_BUILTIN_VERTEX_INDEX:           return "gl_VertexIndex";
    case SSIR_BUILTIN_INSTANCE_INDEX:         return "gl_InstanceIndex";
    case SSIR_BUILTIN_POSITION:               return "gl_FragCoord";
    case SSIR_BUILTIN_FRONT_FACING:           return "gl_FrontFacing";
    case SSIR_BUILTIN_FRAG_DEPTH:             return "gl_FragDepth";
    case SSIR_BUILTIN_SAMPLE_INDEX:           return "gl_SampleID";
    case SSIR_BUILTIN_SAMPLE_MASK:            return "gl_SampleMaskIn[0]";
    case SSIR_BUILTIN_LOCAL_INVOCATION_ID:    return "gl_LocalInvocationID";
    case SSIR_BUILTIN_LOCAL_INVOCATION_INDEX: return "gl_LocalInvocationIndex";
    case SSIR_BUILTIN_GLOBAL_INVOCATION_ID:   return "gl_GlobalInvocationID";
    case SSIR_BUILTIN_WORKGROUP_ID:           return "gl_WorkGroupID";
    case SSIR_BUILTIN_NUM_WORKGROUPS:         return "gl_NumWorkGroups";
    default: return NULL;
    }
}

/* ============================================================================
 * GLSL Builtin Function Mappings
 * ============================================================================ */

static const char *bfunc_to_glsl(SsirBuiltinId id) {
    switch (id) {
    case SSIR_BUILTIN_SIN:           return "sin";
    case SSIR_BUILTIN_COS:           return "cos";
    case SSIR_BUILTIN_TAN:           return "tan";
    case SSIR_BUILTIN_ASIN:          return "asin";
    case SSIR_BUILTIN_ACOS:          return "acos";
    case SSIR_BUILTIN_ATAN:          return "atan";
    case SSIR_BUILTIN_ATAN2:         return "atan";
    case SSIR_BUILTIN_SINH:          return "sinh";
    case SSIR_BUILTIN_COSH:          return "cosh";
    case SSIR_BUILTIN_TANH:          return "tanh";
    case SSIR_BUILTIN_ASINH:         return "asinh";
    case SSIR_BUILTIN_ACOSH:         return "acosh";
    case SSIR_BUILTIN_ATANH:         return "atanh";
    case SSIR_BUILTIN_EXP:           return "exp";
    case SSIR_BUILTIN_EXP2:          return "exp2";
    case SSIR_BUILTIN_LOG:           return "log";
    case SSIR_BUILTIN_LOG2:          return "log2";
    case SSIR_BUILTIN_POW:           return "pow";
    case SSIR_BUILTIN_SQRT:          return "sqrt";
    case SSIR_BUILTIN_INVERSESQRT:   return "inversesqrt";
    case SSIR_BUILTIN_ABS:           return "abs";
    case SSIR_BUILTIN_SIGN:          return "sign";
    case SSIR_BUILTIN_FLOOR:         return "floor";
    case SSIR_BUILTIN_CEIL:          return "ceil";
    case SSIR_BUILTIN_ROUND:         return "round";
    case SSIR_BUILTIN_TRUNC:         return "trunc";
    case SSIR_BUILTIN_FRACT:         return "fract";
    case SSIR_BUILTIN_MIN:           return "min";
    case SSIR_BUILTIN_MAX:           return "max";
    case SSIR_BUILTIN_CLAMP:         return "clamp";
    case SSIR_BUILTIN_SATURATE:      return NULL; /* special: clamp(x,0,1) */
    case SSIR_BUILTIN_MIX:           return "mix";
    case SSIR_BUILTIN_STEP:          return "step";
    case SSIR_BUILTIN_SMOOTHSTEP:    return "smoothstep";
    case SSIR_BUILTIN_DOT:           return "dot";
    case SSIR_BUILTIN_CROSS:         return "cross";
    case SSIR_BUILTIN_LENGTH:        return "length";
    case SSIR_BUILTIN_DISTANCE:      return "distance";
    case SSIR_BUILTIN_NORMALIZE:     return "normalize";
    case SSIR_BUILTIN_FACEFORWARD:   return "faceforward";
    case SSIR_BUILTIN_REFLECT:       return "reflect";
    case SSIR_BUILTIN_REFRACT:       return "refract";
    case SSIR_BUILTIN_ALL:           return "all";
    case SSIR_BUILTIN_ANY:           return "any";
    case SSIR_BUILTIN_SELECT:        return NULL; /* special: mix */
    case SSIR_BUILTIN_COUNTBITS:     return "bitCount";
    case SSIR_BUILTIN_REVERSEBITS:   return "bitfieldReverse";
    case SSIR_BUILTIN_FIRSTLEADINGBIT: return "findMSB";
    case SSIR_BUILTIN_FIRSTTRAILINGBIT: return "findLSB";
    case SSIR_BUILTIN_EXTRACTBITS:   return "bitfieldExtract";
    case SSIR_BUILTIN_INSERTBITS:    return "bitfieldInsert";
    case SSIR_BUILTIN_DPDX:          return "dFdx";
    case SSIR_BUILTIN_DPDY:          return "dFdy";
    case SSIR_BUILTIN_FWIDTH:        return "fwidth";
    case SSIR_BUILTIN_DPDX_COARSE:   return "dFdxCoarse";
    case SSIR_BUILTIN_DPDY_COARSE:   return "dFdyCoarse";
    case SSIR_BUILTIN_DPDX_FINE:     return "dFdxFine";
    case SSIR_BUILTIN_DPDY_FINE:     return "dFdyFine";
    default: return "unknown_builtin";
    }
}

/* ============================================================================
 * GLSL Type Emission
 * ============================================================================ */

static void emit_type(GlslCtx *c, uint32_t tid, GlslBuf *b);

static void emit_type(GlslCtx *c, uint32_t tid, GlslBuf *b) {
    SsirType *t = ssir_get_type((SsirModule *)c->mod, tid);
    if (!t) { gb_appendf(b, "_type%u", tid); return; }

    switch (t->kind) {
    case SSIR_TYPE_VOID: gb_append(b, "void"); break;
    case SSIR_TYPE_BOOL: gb_append(b, "bool"); break;
    case SSIR_TYPE_I32:  gb_append(b, "int"); break;
    case SSIR_TYPE_U32:  gb_append(b, "uint"); break;
    case SSIR_TYPE_F32:  gb_append(b, "float"); break;
    case SSIR_TYPE_F16:  gb_append(b, "float"); break;

    case SSIR_TYPE_VEC: {
        SsirType *el = ssir_get_type((SsirModule *)c->mod, t->vec.elem);
        const char *prefix = "";
        if (el) {
            switch (el->kind) {
            case SSIR_TYPE_I32:  prefix = "i"; break;
            case SSIR_TYPE_U32:  prefix = "u"; break;
            case SSIR_TYPE_BOOL: prefix = "b"; break;
            default: break;
            }
        }
        gb_appendf(b, "%svec%u", prefix, t->vec.size);
        break;
    }

    case SSIR_TYPE_MAT:
        if (t->mat.cols == t->mat.rows)
            gb_appendf(b, "mat%u", t->mat.cols);
        else
            gb_appendf(b, "mat%ux%u", t->mat.cols, t->mat.rows);
        break;

    case SSIR_TYPE_ARRAY:
        emit_type(c, t->array.elem, b);
        gb_appendf(b, "[%u]", t->array.length);
        break;

    case SSIR_TYPE_RUNTIME_ARRAY:
        emit_type(c, t->runtime_array.elem, b);
        gb_append(b, "[]");
        break;

    case SSIR_TYPE_STRUCT:
        gb_append(b, t->struc.name ? t->struc.name : "_Struct");
        break;

    case SSIR_TYPE_PTR:
        emit_type(c, t->ptr.pointee, b);
        break;

    case SSIR_TYPE_SAMPLER:
        gb_append(b, "sampler");
        break;

    case SSIR_TYPE_SAMPLER_COMPARISON:
        gb_append(b, "samplerShadow");
        break;

    case SSIR_TYPE_TEXTURE: {
        SsirType *st = ssir_get_type((SsirModule *)c->mod, t->texture.sampled_type);
        const char *prefix = "";
        if (st) {
            if (st->kind == SSIR_TYPE_I32) prefix = "i";
            else if (st->kind == SSIR_TYPE_U32) prefix = "u";
        }
        switch (t->texture.dim) {
        case SSIR_TEX_1D:              gb_appendf(b, "%ssampler1D", prefix); break;
        case SSIR_TEX_2D:              gb_appendf(b, "%ssampler2D", prefix); break;
        case SSIR_TEX_3D:              gb_appendf(b, "%ssampler3D", prefix); break;
        case SSIR_TEX_CUBE:            gb_appendf(b, "%ssamplerCube", prefix); break;
        case SSIR_TEX_2D_ARRAY:        gb_appendf(b, "%ssampler2DArray", prefix); break;
        case SSIR_TEX_CUBE_ARRAY:      gb_appendf(b, "%ssamplerCubeArray", prefix); break;
        case SSIR_TEX_MULTISAMPLED_2D: gb_appendf(b, "%ssampler2DMS", prefix); break;
        default:                       gb_appendf(b, "%ssampler2D", prefix); break;
        }
        break;
    }

    case SSIR_TYPE_TEXTURE_STORAGE: {
        switch (t->texture_storage.dim) {
        case SSIR_TEX_1D: gb_append(b, "image1D"); break;
        case SSIR_TEX_2D: gb_append(b, "image2D"); break;
        case SSIR_TEX_3D: gb_append(b, "image3D"); break;
        default: gb_append(b, "image2D"); break;
        }
        break;
    }

    case SSIR_TYPE_TEXTURE_DEPTH: {
        switch (t->texture_depth.dim) {
        case SSIR_TEX_2D:       gb_append(b, "sampler2DShadow"); break;
        case SSIR_TEX_CUBE:     gb_append(b, "samplerCubeShadow"); break;
        case SSIR_TEX_2D_ARRAY: gb_append(b, "sampler2DArrayShadow"); break;
        default:                gb_append(b, "sampler2DShadow"); break;
        }
        break;
    }

    default:
        gb_appendf(b, "_type%u", tid);
        break;
    }
}

/* Emit declaration with GLSL array syntax: type name[N] */
static void emit_decl(GlslCtx *c, uint32_t tid, const char *name, GlslBuf *b) {
    SsirType *t = ssir_get_type((SsirModule *)c->mod, tid);
    if (t && t->kind == SSIR_TYPE_ARRAY) {
        emit_type(c, t->array.elem, b);
        gb_appendf(b, " %s[%u]", name, t->array.length);
    } else if (t && t->kind == SSIR_TYPE_RUNTIME_ARRAY) {
        emit_type(c, t->runtime_array.elem, b);
        gb_appendf(b, " %s[]", name);
    } else {
        emit_type(c, tid, b);
        gb_appendf(b, " %s", name);
    }
}

/* ============================================================================
 * Constant Emission
 * ============================================================================ */

static void emit_constant(GlslCtx *c, SsirConstant *k, GlslBuf *b);

static void emit_constant(GlslCtx *c, SsirConstant *k, GlslBuf *b) {
    switch (k->kind) {
    case SSIR_CONST_BOOL:
        gb_append(b, k->bool_val ? "true" : "false");
        break;
    case SSIR_CONST_I32:
        gb_appendf(b, "%d", k->i32_val);
        break;
    case SSIR_CONST_U32:
        gb_appendf(b, "%uu", k->u32_val);
        break;
    case SSIR_CONST_F32: {
        float f = k->f32_val;
        if (floorf(f) == f && fabsf(f) < 1e6f)
            gb_appendf(b, "%.1f", (double)f);
        else
            gb_appendf(b, "%g", (double)f);
        break;
    }
    case SSIR_CONST_F16:
        gb_appendf(b, "%.1f", 0.0);
        break;
    case SSIR_CONST_COMPOSITE: {
        emit_type(c, k->type, b);
        gb_append(b, "(");
        for (uint32_t i = 0; i < k->composite.count; i++) {
            if (i > 0) gb_append(b, ", ");
            SsirConstant *elem = ssir_get_constant((SsirModule *)c->mod, k->composite.components[i]);
            if (elem) emit_constant(c, elem, b);
            else gb_appendf(b, "_const%u", k->composite.components[i]);
        }
        gb_append(b, ")");
        break;
    }
    case SSIR_CONST_NULL:
        emit_type(c, k->type, b);
        gb_append(b, "(0)");
        break;
    default:
        gb_appendf(b, "_const%u", k->id);
        break;
    }
}

/* ============================================================================
 * Expression Emission
 * ============================================================================ */

static void emit_expr(GlslCtx *c, uint32_t id, GlslBuf *b);

static SsirInst *find_inst(GlslCtx *c, uint32_t id) {
    if (!c->current_func) return NULL;
    for (uint32_t bi = 0; bi < c->current_func->block_count; bi++) {
        SsirBlock *blk = &c->current_func->blocks[bi];
        for (uint32_t ii = 0; ii < blk->inst_count; ii++) {
            if (blk->insts[ii].result == id) return &blk->insts[ii];
        }
    }
    return NULL;
}

static SsirFunctionParam *find_param(GlslCtx *c, uint32_t id) {
    if (!c->current_func) return NULL;
    for (uint32_t i = 0; i < c->current_func->param_count; i++) {
        if (c->current_func->params[i].id == id) return &c->current_func->params[i];
    }
    return NULL;
}

static SsirLocalVar *find_local(GlslCtx *c, uint32_t id) {
    if (!c->current_func) return NULL;
    for (uint32_t i = 0; i < c->current_func->local_count; i++) {
        if (c->current_func->locals[i].id == id) return &c->current_func->locals[i];
    }
    return NULL;
}

static void emit_binop(GlslCtx *c, SsirInst *inst, const char *op, GlslBuf *b) {
    gb_append(b, "(");
    emit_expr(c, inst->operands[0], b);
    gb_append(b, op);
    emit_expr(c, inst->operands[1], b);
    gb_append(b, ")");
}

static void emit_unop(GlslCtx *c, SsirInst *inst, const char *op, GlslBuf *b) {
    gb_append(b, "(");
    gb_append(b, op);
    emit_expr(c, inst->operands[0], b);
    gb_append(b, ")");
}

static void emit_expr(GlslCtx *c, uint32_t id, GlslBuf *b) {
    /* Constant? */
    SsirConstant *k = ssir_get_constant((SsirModule *)c->mod, id);
    if (k) { emit_constant(c, k, b); return; }

    /* Global? */
    SsirGlobalVar *g = ssir_get_global((SsirModule *)c->mod, id);
    if (g) {
        if (g->builtin != SSIR_BUILTIN_NONE) {
            SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
            SsirAddressSpace sp = (pt && pt->kind == SSIR_TYPE_PTR) ? pt->ptr.space : SSIR_ADDR_FUNCTION;
            const char *gl = builtin_to_glsl(g->builtin, sp);
            if (gl) { gb_append(b, gl); return; }
        }
        gb_append(b, get_name(c, id));
        return;
    }

    /* Param? */
    if (find_param(c, id)) { gb_append(b, get_name(c, id)); return; }

    /* Local? */
    if (find_local(c, id)) { gb_append(b, get_name(c, id)); return; }

    /* Instruction? */
    SsirInst *inst = find_inst(c, id);
    if (!inst) { gb_append(b, get_name(c, id)); return; }

    switch (inst->op) {
    /* Arithmetic */
    case SSIR_OP_ADD: emit_binop(c, inst, " + ", b); break;
    case SSIR_OP_SUB: emit_binop(c, inst, " - ", b); break;
    case SSIR_OP_MUL: emit_binop(c, inst, " * ", b); break;
    case SSIR_OP_DIV: emit_binop(c, inst, " / ", b); break;
    case SSIR_OP_MOD: emit_binop(c, inst, " % ", b); break;
    case SSIR_OP_NEG: emit_unop(c, inst, "-", b); break;

    /* Matrix */
    case SSIR_OP_MAT_MUL: emit_binop(c, inst, " * ", b); break;
    case SSIR_OP_MAT_TRANSPOSE:
        gb_append(b, "transpose(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ")");
        break;

    /* Bitwise */
    case SSIR_OP_BIT_AND: emit_binop(c, inst, " & ", b); break;
    case SSIR_OP_BIT_OR:  emit_binop(c, inst, " | ", b); break;
    case SSIR_OP_BIT_XOR: emit_binop(c, inst, " ^ ", b); break;
    case SSIR_OP_BIT_NOT: emit_unop(c, inst, "~", b); break;
    case SSIR_OP_SHL:     emit_binop(c, inst, " << ", b); break;
    case SSIR_OP_SHR:
    case SSIR_OP_SHR_LOGICAL:
        emit_binop(c, inst, " >> ", b);
        break;

    /* Comparison */
    case SSIR_OP_EQ: emit_binop(c, inst, " == ", b); break;
    case SSIR_OP_NE: emit_binop(c, inst, " != ", b); break;
    case SSIR_OP_LT: emit_binop(c, inst, " < ", b); break;
    case SSIR_OP_LE: emit_binop(c, inst, " <= ", b); break;
    case SSIR_OP_GT: emit_binop(c, inst, " > ", b); break;
    case SSIR_OP_GE: emit_binop(c, inst, " >= ", b); break;

    /* Logical */
    case SSIR_OP_AND: emit_binop(c, inst, " && ", b); break;
    case SSIR_OP_OR:  emit_binop(c, inst, " || ", b); break;
    case SSIR_OP_NOT: emit_unop(c, inst, "!", b); break;

    /* Composite */
    case SSIR_OP_CONSTRUCT: {
        emit_type(c, inst->type, b);
        gb_append(b, "(");
        uint32_t cnt = inst->operand_count;
        const uint32_t *comps = inst->operands;
        if (inst->extra_count > 0) { cnt = inst->extra_count; comps = inst->extra; }
        for (uint32_t i = 0; i < cnt; i++) {
            if (i > 0) gb_append(b, ", ");
            emit_expr(c, comps[i], b);
        }
        gb_append(b, ")");
        break;
    }

    case SSIR_OP_EXTRACT: {
        emit_expr(c, inst->operands[0], b);
        uint32_t idx = inst->operands[1];
        if (idx < 4) {
            const char sw[] = "xyzw";
            gb_appendf(b, ".%c", sw[idx]);
        } else {
            gb_appendf(b, "[%u]", idx);
        }
        break;
    }

    case SSIR_OP_INSERT:
        gb_append(b, get_name(c, id));
        break;

    case SSIR_OP_SHUFFLE: {
        emit_type(c, inst->type, b);
        gb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) gb_append(b, ", ");
            uint32_t si = inst->extra[i];
            uint32_t v1_sz = 4;
            if (si < v1_sz) {
                emit_expr(c, inst->operands[0], b);
            } else {
                emit_expr(c, inst->operands[1], b);
                si -= v1_sz;
            }
            if (si < 4) {
                const char sw[] = "xyzw";
                gb_appendf(b, ".%c", sw[si]);
            }
        }
        gb_append(b, ")");
        break;
    }

    case SSIR_OP_SPLAT:
        emit_type(c, inst->type, b);
        gb_append(b, "(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_EXTRACT_DYN:
        emit_expr(c, inst->operands[0], b);
        gb_append(b, "[");
        emit_expr(c, inst->operands[1], b);
        gb_append(b, "]");
        break;

    /* Memory */
    case SSIR_OP_LOAD:
        emit_expr(c, inst->operands[0], b);
        break;

    case SSIR_OP_ACCESS: {
        emit_expr(c, inst->operands[0], b);
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            uint32_t idx = inst->extra[i];
            SsirConstant *ic = ssir_get_constant((SsirModule *)c->mod, idx);
            if (ic && ic->kind == SSIR_CONST_U32)
                gb_appendf(b, ".member%u", ic->u32_val);
            else if (ic && ic->kind == SSIR_CONST_I32)
                gb_appendf(b, ".member%d", ic->i32_val);
            else {
                gb_append(b, "[");
                emit_expr(c, idx, b);
                gb_append(b, "]");
            }
        }
        break;
    }

    case SSIR_OP_ARRAY_LEN:
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ".length()");
        break;

    /* Call */
    case SSIR_OP_CALL: {
        uint32_t callee_id = inst->operands[0];
        SsirFunction *callee = ssir_get_function((SsirModule *)c->mod, callee_id);
        if (callee && callee->name)
            gb_append(b, callee->name);
        else
            gb_append(b, get_name(c, callee_id));
        gb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) gb_append(b, ", ");
            emit_expr(c, inst->extra[i], b);
        }
        gb_append(b, ")");
        break;
    }

    case SSIR_OP_BUILTIN: {
        SsirBuiltinId bid = (SsirBuiltinId)inst->operands[0];

        if (bid == SSIR_BUILTIN_SATURATE) {
            gb_append(b, "clamp(");
            if (inst->extra_count > 0) emit_expr(c, inst->extra[0], b);
            gb_append(b, ", 0.0, 1.0)");
            break;
        }

        if (bid == SSIR_BUILTIN_SELECT) {
            gb_append(b, "mix(");
            for (uint16_t i = 0; i < inst->extra_count; i++) {
                if (i > 0) gb_append(b, ", ");
                emit_expr(c, inst->extra[i], b);
            }
            gb_append(b, ")");
            break;
        }

        const char *name = bfunc_to_glsl(bid);
        gb_append(b, name ? name : "unknown_builtin");
        gb_append(b, "(");
        for (uint16_t i = 0; i < inst->extra_count; i++) {
            if (i > 0) gb_append(b, ", ");
            emit_expr(c, inst->extra[i], b);
        }
        gb_append(b, ")");
        break;
    }

    /* Conversion */
    case SSIR_OP_CONVERT:
        emit_type(c, inst->type, b);
        gb_append(b, "(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_BITCAST: {
        SsirType *dst = ssir_get_type((SsirModule *)c->mod, inst->type);
        if (dst && dst->kind == SSIR_TYPE_F32) {
            gb_append(b, "intBitsToFloat(");
            emit_expr(c, inst->operands[0], b);
            gb_append(b, ")");
        } else if (dst && dst->kind == SSIR_TYPE_I32) {
            gb_append(b, "floatBitsToInt(");
            emit_expr(c, inst->operands[0], b);
            gb_append(b, ")");
        } else if (dst && dst->kind == SSIR_TYPE_U32) {
            gb_append(b, "floatBitsToUint(");
            emit_expr(c, inst->operands[0], b);
            gb_append(b, ")");
        } else {
            emit_type(c, inst->type, b);
            gb_append(b, "(");
            emit_expr(c, inst->operands[0], b);
            gb_append(b, ")");
        }
        break;
    }

    /* Texture */
    case SSIR_OP_TEX_SAMPLE:
        gb_append(b, "texture(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_BIAS:
        gb_append(b, "texture(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_LEVEL:
        gb_append(b, "textureLod(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_GRAD:
        gb_append(b, "textureGrad(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[4], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_TEX_SAMPLE_CMP:
        gb_append(b, "texture(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", vec3(");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[3], b);
        gb_append(b, "))");
        break;

    case SSIR_OP_TEX_LOAD:
        gb_append(b, "texelFetch(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[1], b);
        gb_append(b, ", ");
        emit_expr(c, inst->operands[2], b);
        gb_append(b, ")");
        break;

    case SSIR_OP_TEX_SIZE:
        gb_append(b, "textureSize(");
        emit_expr(c, inst->operands[0], b);
        gb_append(b, ", ");
        if (inst->operand_count > 1 && inst->operands[1] != 0)
            emit_expr(c, inst->operands[1], b);
        else
            gb_append(b, "0");
        gb_append(b, ")");
        break;

    case SSIR_OP_PHI:
        gb_append(b, get_name(c, id));
        break;

    default:
        gb_append(b, get_name(c, id));
        break;
    }
}

/* ============================================================================
 * Block and Statement Emission
 * ============================================================================ */

typedef struct {
    bool *emitted;
    uint32_t count;
} BlkState;

static void emit_block(GlslCtx *c, SsirBlock *blk, SsirFunction *fn, BlkState *bs);

static void emit_stmt(GlslCtx *c, SsirInst *inst, SsirBlock *blk,
                       SsirFunction *fn, BlkState *bs) {
    switch (inst->op) {
    case SSIR_OP_STORE:
        gb_indent(&c->sb);
        emit_expr(c, inst->operands[0], &c->sb);
        gb_append(&c->sb, " = ");
        emit_expr(c, inst->operands[1], &c->sb);
        gb_append(&c->sb, ";\n");
        break;

    case SSIR_OP_TEX_STORE:
        gb_indent(&c->sb);
        gb_append(&c->sb, "imageStore(");
        emit_expr(c, inst->operands[0], &c->sb);
        gb_append(&c->sb, ", ");
        emit_expr(c, inst->operands[1], &c->sb);
        gb_append(&c->sb, ", ");
        emit_expr(c, inst->operands[2], &c->sb);
        gb_append(&c->sb, ");\n");
        break;

    case SSIR_OP_RETURN_VOID:
        gb_indent(&c->sb);
        gb_append(&c->sb, "return;\n");
        break;

    case SSIR_OP_RETURN:
        gb_indent(&c->sb);
        gb_append(&c->sb, "return ");
        emit_expr(c, inst->operands[0], &c->sb);
        gb_append(&c->sb, ";\n");
        break;

    case SSIR_OP_BRANCH: {
        uint32_t target = inst->operands[0];
        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == target && !bs->emitted[i]) {
                emit_block(c, &fn->blocks[i], fn, bs);
                break;
            }
        }
        break;
    }

    case SSIR_OP_BRANCH_COND: {
        uint32_t cond = inst->operands[0];
        uint32_t true_blk = inst->operands[1];
        uint32_t false_blk = inst->operands[2];

        gb_indent(&c->sb);
        gb_append(&c->sb, "if (");
        emit_expr(c, cond, &c->sb);
        gb_append(&c->sb, ") {\n");
        c->sb.indent++;

        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == true_blk && !bs->emitted[i]) {
                emit_block(c, &fn->blocks[i], fn, bs);
                break;
            }
        }

        c->sb.indent--;
        gb_indent(&c->sb);
        gb_append(&c->sb, "} else {\n");
        c->sb.indent++;

        for (uint32_t i = 0; i < fn->block_count; i++) {
            if (fn->blocks[i].id == false_blk && !bs->emitted[i]) {
                emit_block(c, &fn->blocks[i], fn, bs);
                break;
            }
        }

        c->sb.indent--;
        gb_indent(&c->sb);
        gb_append(&c->sb, "}\n");
        break;
    }

    case SSIR_OP_BARRIER:
        gb_indent(&c->sb);
        if (inst->operands[0] == SSIR_BARRIER_WORKGROUP)
            gb_append(&c->sb, "barrier();\n");
        else
            gb_append(&c->sb, "memoryBarrierBuffer();\n");
        break;

    case SSIR_OP_LOOP_MERGE:
    case SSIR_OP_UNREACHABLE:
        break;

    default:
        if (inst->result != 0) {
            /* Check if the type is void - skip if so */
            SsirType *rt = ssir_get_type((SsirModule *)c->mod, inst->type);
            if (rt && rt->kind == SSIR_TYPE_VOID) break;
            gb_indent(&c->sb);
            emit_decl(c, inst->type, get_name(c, inst->result), &c->sb);
            gb_append(&c->sb, " = ");
            emit_expr(c, inst->result, &c->sb);
            gb_append(&c->sb, ";\n");
        }
        break;
    }
}

static void emit_block(GlslCtx *c, SsirBlock *blk, SsirFunction *fn, BlkState *bs) {
    for (uint32_t i = 0; i < fn->block_count; i++) {
        if (fn->blocks[i].id == blk->id) {
            if (bs->emitted[i]) return;
            bs->emitted[i] = true;
            break;
        }
    }

    for (uint32_t i = 0; i < blk->inst_count; i++) {
        emit_stmt(c, &blk->insts[i], blk, fn, bs);
    }
}

/* ============================================================================
 * Struct Emission (for non-interface-block types)
 * ============================================================================ */

static bool type_is_in_interface_block(GlslCtx *c, uint32_t type_id) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];
        SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
        if (!pt || pt->kind != SSIR_TYPE_PTR) continue;
        if (pt->ptr.space != SSIR_ADDR_UNIFORM && pt->ptr.space != SSIR_ADDR_STORAGE)
            continue;
        if (pt->ptr.pointee == type_id) return true;
    }
    return false;
}

static void emit_struct_def(GlslCtx *c, SsirType *t) {
    if (t->kind != SSIR_TYPE_STRUCT) return;
    if (type_is_in_interface_block(c, t->id)) return;

    gb_appendf(&c->sb, "struct %s {\n", t->struc.name ? t->struc.name : "_Struct");
    for (uint32_t i = 0; i < t->struc.member_count; i++) {
        gb_append(&c->sb, "    ");
        char mname[32];
        snprintf(mname, sizeof(mname), "member%u", i);
        emit_decl(c, t->struc.members[i], mname, &c->sb);
        gb_append(&c->sb, ";\n");
    }
    gb_append(&c->sb, "};\n\n");
}

/* ============================================================================
 * Global Variable Emission
 * ============================================================================ */

static void emit_global(GlslCtx *c, SsirGlobalVar *g) {
    SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
    if (!pt || pt->kind != SSIR_TYPE_PTR) return;

    SsirAddressSpace space = pt->ptr.space;
    uint32_t pointee_id = pt->ptr.pointee;
    SsirType *pointee = ssir_get_type((SsirModule *)c->mod, pointee_id);

    /* Skip builtins - they use gl_* names directly */
    if (g->builtin != SSIR_BUILTIN_NONE) return;

    const char *vname = get_name(c, g->id);

    switch (space) {
    case SSIR_ADDR_INPUT: {
        gb_append(&c->sb, "layout(location = ");
        gb_appendf(&c->sb, "%u", g->has_location ? g->location : 0);
        gb_append(&c->sb, ") ");
        if (g->interp == SSIR_INTERP_FLAT) gb_append(&c->sb, "flat ");
        else if (g->interp == SSIR_INTERP_LINEAR) gb_append(&c->sb, "noperspective ");
        gb_append(&c->sb, "in ");
        emit_decl(c, pointee_id, vname, &c->sb);
        gb_append(&c->sb, ";\n");
        break;
    }

    case SSIR_ADDR_OUTPUT: {
        gb_append(&c->sb, "layout(location = ");
        gb_appendf(&c->sb, "%u", g->has_location ? g->location : 0);
        gb_append(&c->sb, ") out ");
        emit_decl(c, pointee_id, vname, &c->sb);
        gb_append(&c->sb, ";\n");
        break;
    }

    case SSIR_ADDR_UNIFORM: {
        gb_append(&c->sb, "layout(std140");
        if (g->has_group && !c->opts.target_opengl) gb_appendf(&c->sb, ", set = %u", g->group);
        if (g->has_binding) gb_appendf(&c->sb, ", binding = %u", g->binding);
        gb_append(&c->sb, ") uniform ");

        if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
            gb_appendf(&c->sb, "_UB_%s {\n", vname);
            for (uint32_t m = 0; m < pointee->struc.member_count; m++) {
                gb_append(&c->sb, "    ");
                char mn[32];
                snprintf(mn, sizeof(mn), "member%u", m);
                emit_decl(c, pointee->struc.members[m], mn, &c->sb);
                gb_append(&c->sb, ";\n");
            }
            gb_appendf(&c->sb, "} %s;\n", vname);
        } else {
            gb_appendf(&c->sb, "_UB_%s {\n    ", vname);
            emit_decl(c, pointee_id, "member0", &c->sb);
            gb_append(&c->sb, ";\n");
            gb_appendf(&c->sb, "} %s;\n", vname);
        }
        break;
    }

    case SSIR_ADDR_STORAGE: {
        gb_append(&c->sb, "layout(std430");
        if (g->has_group && !c->opts.target_opengl) gb_appendf(&c->sb, ", set = %u", g->group);
        if (g->has_binding) gb_appendf(&c->sb, ", binding = %u", g->binding);
        gb_append(&c->sb, ") buffer ");

        if (pointee && pointee->kind == SSIR_TYPE_STRUCT) {
            gb_appendf(&c->sb, "_SB_%s {\n", vname);
            for (uint32_t m = 0; m < pointee->struc.member_count; m++) {
                gb_append(&c->sb, "    ");
                char mn[32];
                snprintf(mn, sizeof(mn), "member%u", m);
                emit_decl(c, pointee->struc.members[m], mn, &c->sb);
                gb_append(&c->sb, ";\n");
            }
            gb_appendf(&c->sb, "} %s;\n", vname);
        } else {
            gb_appendf(&c->sb, "_SB_%s {\n    ", vname);
            emit_decl(c, pointee_id, "member0", &c->sb);
            gb_append(&c->sb, ";\n");
            gb_appendf(&c->sb, "} %s;\n", vname);
        }
        break;
    }

    case SSIR_ADDR_WORKGROUP:
        gb_append(&c->sb, "shared ");
        emit_decl(c, pointee_id, vname, &c->sb);
        gb_append(&c->sb, ";\n");
        break;

    case SSIR_ADDR_UNIFORM_CONSTANT: {
        /* Samplers and textures */
        if (g->has_group || g->has_binding) {
            gb_append(&c->sb, "layout(");
            int need_comma = 0;
            if (g->has_group && !c->opts.target_opengl) { gb_appendf(&c->sb, "set = %u", g->group); need_comma = 1; }
            if (g->has_binding) {
                if (need_comma) gb_append(&c->sb, ", ");
                gb_appendf(&c->sb, "binding = %u", g->binding);
            }
            gb_append(&c->sb, ") ");
        }
        gb_append(&c->sb, "uniform ");
        emit_decl(c, pointee_id, vname, &c->sb);
        gb_append(&c->sb, ";\n");
        break;
    }

    default:
        break;
    }
}

/* ============================================================================
 * Function Emission
 * ============================================================================ */

static SsirEntryPoint *find_ep(GlslCtx *c, uint32_t func_id) {
    for (uint32_t i = 0; i < c->mod->entry_point_count; i++) {
        if (c->mod->entry_points[i].function == func_id)
            return &c->mod->entry_points[i];
    }
    return NULL;
}

static void emit_function(GlslCtx *c, SsirFunction *fn) {
    SsirEntryPoint *ep = find_ep(c, fn->id);
    int is_entry = (ep != NULL && ep == c->active_ep);

    if (is_entry) {
        /* Entry point is void main() in GLSL */
        gb_append(&c->sb, "void main() {\n");
    } else {
        /* Regular function */
        SsirType *ret = ssir_get_type((SsirModule *)c->mod, fn->return_type);
        if (ret && ret->kind != SSIR_TYPE_VOID) {
            emit_type(c, fn->return_type, &c->sb);
        } else {
            gb_append(&c->sb, "void");
        }
        gb_appendf(&c->sb, " %s(", fn->name ? fn->name : get_name(c, fn->id));

        for (uint32_t i = 0; i < fn->param_count; i++) {
            if (i > 0) gb_append(&c->sb, ", ");
            emit_decl(c, fn->params[i].type, get_name(c, fn->params[i].id), &c->sb);
        }
        gb_append(&c->sb, ") {\n");
    }

    c->sb.indent++;
    c->current_func = fn;

    /* Local variables */
    for (uint32_t i = 0; i < fn->local_count; i++) {
        SsirLocalVar *l = &fn->locals[i];
        SsirType *lpt = ssir_get_type((SsirModule *)c->mod, l->type);
        uint32_t pointee = (lpt && lpt->kind == SSIR_TYPE_PTR) ? lpt->ptr.pointee : l->type;

        gb_indent(&c->sb);
        emit_decl(c, pointee, get_name(c, l->id), &c->sb);

        if (l->has_initializer) {
            gb_append(&c->sb, " = ");
            SsirConstant *init = ssir_get_constant((SsirModule *)c->mod, l->initializer);
            if (init) emit_constant(c, init, &c->sb);
        }
        gb_append(&c->sb, ";\n");
    }

    /* Function body */
    if (fn->block_count > 0) {
        BlkState bs;
        bs.count = fn->block_count;
        bs.emitted = (bool *)STG_MALLOC(fn->block_count * sizeof(bool));
        if (bs.emitted) {
            memset(bs.emitted, 0, fn->block_count * sizeof(bool));
            emit_block(c, &fn->blocks[0], fn, &bs);
            STG_FREE(bs.emitted);
        }
    }

    c->current_func = NULL;
    c->sb.indent--;
    gb_append(&c->sb, "}\n\n");
}

/* ============================================================================
 * Name Assignment
 * ============================================================================ */

static SsirAddressSpace global_addr_space(GlslCtx *c, SsirGlobalVar *g) {
    SsirType *pt = ssir_get_type((SsirModule *)c->mod, g->type);
    if (pt && pt->kind == SSIR_TYPE_PTR) return pt->ptr.space;
    return SSIR_ADDR_FUNCTION;
}

/* Check if name is already used by another global */
static int global_name_taken(GlslCtx *c, const char *name, uint32_t exclude_id) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];
        if (g->id == exclude_id || g->builtin != SSIR_BUILTIN_NONE) continue;
        if (g->id < c->id_names_cap && c->id_names[g->id] &&
            strcmp(c->id_names[g->id], name) == 0) return 1;
    }
    return 0;
}

static void assign_names(GlslCtx *c) {
    for (uint32_t i = 0; i < c->mod->global_count; i++) {
        SsirGlobalVar *g = &c->mod->globals[i];

        /* Builtins get gl_* names via emit_expr, no need for id names */
        if (g->builtin != SSIR_BUILTIN_NONE) continue;

        if (g->name && c->opts.preserve_names) {
            set_name(c, g->id, g->name);
            /* Disambiguate if name collides with an already-assigned global */
            const char *assigned = c->id_names[g->id];
            if (assigned && global_name_taken(c, assigned, g->id)) {
                SsirAddressSpace sp = global_addr_space(c, g);
                const char *prefix = (sp == SSIR_ADDR_OUTPUT) ? "_out_" : "_in_";
                size_t len = strlen(prefix) + strlen(g->name) + 1;
                char *tmp = (char *)STG_MALLOC(len);
                snprintf(tmp, len, "%s%s", prefix, g->name);
                STG_FREE(c->id_names[g->id]);
                c->id_names[g->id] = tmp;
            }
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "_g%u", g->id);
            set_name(c, g->id, buf);
        }
    }

    for (uint32_t i = 0; i < c->mod->function_count; i++) {
        SsirFunction *fn = &c->mod->functions[i];
        if (fn->name && c->opts.preserve_names)
            set_name(c, fn->id, fn->name);

        for (uint32_t j = 0; j < fn->param_count; j++) {
            SsirFunctionParam *p = &fn->params[j];
            if (p->name && c->opts.preserve_names) {
                set_name(c, p->id, p->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_p%u", p->id);
                set_name(c, p->id, buf);
            }
        }

        for (uint32_t j = 0; j < fn->local_count; j++) {
            SsirLocalVar *l = &fn->locals[j];
            if (l->name && c->opts.preserve_names) {
                set_name(c, l->id, l->name);
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "_l%u", l->id);
                set_name(c, l->id, buf);
            }
        }
    }
}

/* ============================================================================
 * Public API
 * ============================================================================ */

SsirToGlslResult ssir_to_glsl(const SsirModule *mod,
                               SsirStage stage,
                               const SsirToGlslOptions *opts,
                               char **out_glsl,
                               char **out_error) {
    if (!mod || !out_glsl) {
        if (out_error) *out_error = strdup("Invalid input: null module or output pointer");
        return SSIR_TO_GLSL_ERR_INVALID_INPUT;
    }

    /* Find the matching entry point */
    SsirEntryPoint *ep = NULL;
    for (uint32_t i = 0; i < mod->entry_point_count; i++) {
        if (mod->entry_points[i].stage == stage) {
            ep = &mod->entry_points[i];
            break;
        }
    }
    if (!ep) {
        /* If no matching stage, use first entry point */
        if (mod->entry_point_count > 0)
            ep = &mod->entry_points[0];
    }

    GlslCtx ctx;
    gctx_init(&ctx, mod, opts);
    if (!ctx.id_names) {
        if (out_error) *out_error = strdup("Out of memory");
        return SSIR_TO_GLSL_ERR_OOM;
    }
    ctx.active_ep = ep;

    /* Phase 1: Assign names */
    assign_names(&ctx);

    /* Phase 2: Emit GLSL header */
    gb_append(&ctx.sb, "#version 450\n\n");

    /* Phase 3: Emit compute local size if needed */
    if (ep && ep->stage == SSIR_STAGE_COMPUTE) {
        gb_appendf(&ctx.sb, "layout(local_size_x = %u, local_size_y = %u, local_size_z = %u) in;\n\n",
                   ep->workgroup_size[0], ep->workgroup_size[1], ep->workgroup_size[2]);
    }

    /* Phase 4: Emit struct definitions (non-interface-block types) */
    for (uint32_t i = 0; i < mod->type_count; i++) {
        if (mod->types[i].kind == SSIR_TYPE_STRUCT) {
            emit_struct_def(&ctx, &mod->types[i]);
        }
    }

    /* Phase 5: Emit global variables */
    for (uint32_t i = 0; i < mod->global_count; i++) {
        emit_global(&ctx, &mod->globals[i]);
    }
    if (mod->global_count > 0) gb_nl(&ctx.sb);

    /* Phase 6: Emit non-entry functions first */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        SsirEntryPoint *fep = find_ep(&ctx, mod->functions[i].id);
        if (!fep || fep != ep) {
            emit_function(&ctx, &mod->functions[i]);
        }
    }

    /* Phase 7: Emit entry point as void main() */
    if (ep) {
        SsirFunction *entry_fn = ssir_get_function((SsirModule *)mod, ep->function);
        if (entry_fn) {
            emit_function(&ctx, entry_fn);
        }
    }

    *out_glsl = ctx.sb.data ? strdup(ctx.sb.data) : strdup("");
    gctx_free(&ctx);
    return SSIR_TO_GLSL_OK;
}

void ssir_to_glsl_free(void *p) {
    STG_FREE(p);
}

const char *ssir_to_glsl_result_string(SsirToGlslResult result) {
    switch (result) {
    case SSIR_TO_GLSL_OK:                return "Success";
    case SSIR_TO_GLSL_ERR_INVALID_INPUT: return "Invalid input";
    case SSIR_TO_GLSL_ERR_UNSUPPORTED:   return "Unsupported feature";
    case SSIR_TO_GLSL_ERR_INTERNAL:      return "Internal error";
    case SSIR_TO_GLSL_ERR_OOM:           return "Out of memory";
    default:                             return "Unknown error";
    }
}
