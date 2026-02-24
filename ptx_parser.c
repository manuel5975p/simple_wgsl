/*
 * ptx_parser.c - PTX (Parallel Thread Execution) Parser -> SSIR
 *
 * Parses PTX assembly source and produces an SSIR module directly using
 * the SSIR builder APIs. PTX's flat register-based assembly semantics are
 * too different from WGSL/GLSL to share their AST, so this follows the
 * same direct-to-SSIR pattern as msl_parser.c.
 */

#include "simple_wgsl.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

#ifndef PTX_MALLOC
#define PTX_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef PTX_REALLOC
#define PTX_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef PTX_FREE
#define PTX_FREE(p) free((p))
#endif

/* ============================================================================
 * String Utilities
 * ============================================================================ */

static char *ptx_strndup(const char *s, size_t n) {
    wgsl_compiler_assert(s != NULL, "ptx_strndup: s is NULL");
    char *r = (char *)PTX_MALLOC(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

static char *ptx_strdup(const char *s) {
    return s ? ptx_strndup(s, strlen(s)) : NULL;
}

/* ============================================================================
 * Token Types
 * ============================================================================ */

typedef enum {
    /* structural */
    PTK_EOF = 0,
    PTK_SEMI,       /* ; */
    PTK_COMMA,      /* , */
    PTK_LBRACE,     /* { */
    PTK_RBRACE,     /* } */
    PTK_LBRACKET,   /* [ */
    PTK_RBRACKET,   /* ] */
    PTK_LPAREN,     /* ( */
    PTK_RPAREN,     /* ) */
    PTK_COLON,      /* : */
    PTK_PIPE,       /* | */
    PTK_AT,         /* @ */
    PTK_BANG,       /* ! */
    PTK_PLUS,       /* + */
    PTK_MINUS,      /* - */
    PTK_STAR,       /* * */
    PTK_LANGLE,     /* < */
    PTK_RANGLE,     /* > */

    /* identifiers and literals */
    PTK_IDENT,
    PTK_INT_LIT,
    PTK_FLOAT_LIT,

    /* dot-prefixed token — the text field contains the full ".xxx" */
    PTK_DOT_TOKEN,
} PtxTokType;

typedef struct {
    PtxTokType type;
    const char *start;
    int length;
    int line, col;
    uint64_t int_val;    /* parsed integer value for PTK_INT_LIT */
    double float_val;    /* parsed float value for PTK_FLOAT_LIT */
} PtxToken;

/* ============================================================================
 * Lexer
 * ============================================================================ */

typedef struct {
    const char *src;
    size_t pos;
    int line, col;
} PtxLexer;

static void plx_init(PtxLexer *L, const char *src) {
    L->src = src;
    L->pos = 0;
    L->line = 1;
    L->col = 1;
}

static char plx_peek(const PtxLexer *L) { return L->src[L->pos]; }
static char plx_peek2(const PtxLexer *L) { return L->src[L->pos + 1]; }

static void plx_advance(PtxLexer *L) {
    if (L->src[L->pos] == '\n') { L->line++; L->col = 1; }
    else { L->col++; }
    L->pos++;
}

static void plx_skip_whitespace_and_comments(PtxLexer *L) {
    for (;;) {
        char c = plx_peek(L);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            plx_advance(L);
        } else if (c == '/' && plx_peek2(L) == '/') {
            /* single-line comment */
            while (plx_peek(L) && plx_peek(L) != '\n') plx_advance(L);
        } else if (c == '/' && plx_peek2(L) == '*') {
            /* multi-line comment */
            plx_advance(L); plx_advance(L);
            while (plx_peek(L)) {
                if (plx_peek(L) == '*' && plx_peek2(L) == '/') {
                    plx_advance(L); plx_advance(L);
                    break;
                }
                plx_advance(L);
            }
        } else {
            break;
        }
    }
}

static bool plx_is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_' || c == '%' || c == '$';
}

static bool plx_is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '$';
}

static PtxToken plx_next(PtxLexer *L) {
    plx_skip_whitespace_and_comments(L);

    PtxToken t;
    memset(&t, 0, sizeof(t));
    t.start = &L->src[L->pos];
    t.line = L->line;
    t.col = L->col;

    char c = plx_peek(L);
    if (c == '\0') { t.type = PTK_EOF; return t; }

    /* dot-prefixed token */
    if (c == '.') {
        plx_advance(L);
        while (isalnum((unsigned char)plx_peek(L)) || plx_peek(L) == '_')
            plx_advance(L);
        t.type = PTK_DOT_TOKEN;
        t.length = (int)(&L->src[L->pos] - t.start);
        return t;
    }

    /* identifier or %-prefixed register */
    if (plx_is_ident_start(c)) {
        bool is_reg = (c == '%');
        plx_advance(L);
        /* Allow '.' in %-register names (e.g. %tid.x, %ctaid.y) but NOT
         * in instruction opcodes (add.f32 must tokenize as "add" + ".f32") */
        while (plx_is_ident_char(plx_peek(L)) || (is_reg && plx_peek(L) == '.'))
            plx_advance(L);
        t.type = PTK_IDENT;
        t.length = (int)(&L->src[L->pos] - t.start);
        return t;
    }

    /* numeric literal */
    if (isdigit((unsigned char)c) || (c == '-' && isdigit((unsigned char)plx_peek2(L)))) {
        bool negative = false;
        if (c == '-') { negative = true; plx_advance(L); c = plx_peek(L); }

        /* 0f / 0d hex float */
        if (c == '0' && (plx_peek2(L) == 'f' || plx_peek2(L) == 'F' ||
                         plx_peek2(L) == 'd' || plx_peek2(L) == 'D')) {
            bool is_double = (plx_peek2(L) == 'd' || plx_peek2(L) == 'D');
            plx_advance(L); plx_advance(L); /* skip 0f/0d */
            uint64_t hex = 0;
            while (isxdigit((unsigned char)plx_peek(L))) {
                char h = plx_peek(L);
                int d = (h >= 'a') ? h - 'a' + 10 : (h >= 'A') ? h - 'A' + 10 : h - '0';
                hex = (hex << 4) | d;
                plx_advance(L);
            }
            t.type = PTK_FLOAT_LIT;
            if (is_double) {
                double dval;
                memcpy(&dval, &hex, sizeof(double));
                t.float_val = negative ? -dval : dval;
            } else {
                uint32_t bits = (uint32_t)hex;
                float fval;
                memcpy(&fval, &bits, sizeof(float));
                t.float_val = negative ? -fval : fval;
            }
            t.length = (int)(&L->src[L->pos] - t.start);
            return t;
        }

        /* hex integer */
        if (c == '0' && (plx_peek2(L) == 'x' || plx_peek2(L) == 'X')) {
            plx_advance(L); plx_advance(L);
            uint64_t val = 0;
            while (isxdigit((unsigned char)plx_peek(L))) {
                char h = plx_peek(L);
                int d = (h >= 'a') ? h - 'a' + 10 : (h >= 'A') ? h - 'A' + 10 : h - '0';
                val = (val << 4) | d;
                plx_advance(L);
            }
            /* handle U suffix */
            if (plx_peek(L) == 'U' || plx_peek(L) == 'u') plx_advance(L);
            t.type = PTK_INT_LIT;
            t.int_val = negative ? (uint64_t)(-(int64_t)val) : val;
            t.length = (int)(&L->src[L->pos] - t.start);
            return t;
        }

        /* decimal integer or float */
        bool is_float = false;
        while (isdigit((unsigned char)plx_peek(L))) plx_advance(L);
        if (plx_peek(L) == '.' && isdigit((unsigned char)plx_peek2(L))) {
            is_float = true;
            plx_advance(L); /* skip . */
            while (isdigit((unsigned char)plx_peek(L))) plx_advance(L);
        }
        if (plx_peek(L) == 'e' || plx_peek(L) == 'E') {
            is_float = true;
            plx_advance(L);
            if (plx_peek(L) == '+' || plx_peek(L) == '-') plx_advance(L);
            while (isdigit((unsigned char)plx_peek(L))) plx_advance(L);
        }
        if (plx_peek(L) == 'U' || plx_peek(L) == 'u') plx_advance(L);

        t.length = (int)(&L->src[L->pos] - t.start);
        if (is_float) {
            t.type = PTK_FLOAT_LIT;
            t.float_val = strtod(t.start, NULL);
        } else {
            t.type = PTK_INT_LIT;
            int64_t v = strtoll(t.start, NULL, 10);
            t.int_val = (uint64_t)v;
        }
        return t;
    }

    /* single-char punctuation */
    plx_advance(L);
    t.length = 1;
    switch (c) {
        case ';': t.type = PTK_SEMI; break;
        case ',': t.type = PTK_COMMA; break;
        case '{': t.type = PTK_LBRACE; break;
        case '}': t.type = PTK_RBRACE; break;
        case '[': t.type = PTK_LBRACKET; break;
        case ']': t.type = PTK_RBRACKET; break;
        case '(': t.type = PTK_LPAREN; break;
        case ')': t.type = PTK_RPAREN; break;
        case ':': t.type = PTK_COLON; break;
        case '|': t.type = PTK_PIPE; break;
        case '@': t.type = PTK_AT; break;
        case '!': t.type = PTK_BANG; break;
        case '+': t.type = PTK_PLUS; break;
        case '-': t.type = PTK_MINUS; break;
        case '*': t.type = PTK_STAR; break;
        case '<': t.type = PTK_LANGLE; break;
        case '>': t.type = PTK_RANGLE; break;
        default:  t.type = PTK_EOF; break; /* unexpected char */
    }
    return t;
}

/* ============================================================================
 * Dot-token helpers
 * ============================================================================ */

static bool dot_eq(const PtxToken *t, const char *s) {
    if (t->type != PTK_DOT_TOKEN) return false;
    return t->length == (int)strlen(s) && strncmp(t->start, s, t->length) == 0;
}

static void tok_to_str(const PtxToken *t, char *buf, int bufsz) {
    int len = t->length < bufsz - 1 ? t->length : bufsz - 1;
    memcpy(buf, t->start, len);
    buf[len] = '\0';
}

/* ============================================================================
 * Parser Context
 * ============================================================================ */

/* Register entry: maps PTX register name to SSIR local variable */
typedef struct {
    char name[80];
    uint32_t ptr_id;     /* SSIR pointer (local var) ID */
    uint32_t ptr_type;   /* SSIR pointer type ID */
    uint32_t val_type;   /* SSIR value type ID */
    uint32_t global_id;  /* SSIR global var ID (for kernel pointer params), 0 if none */
    uint32_t pending_binding; /* Binding index for unmaterialized buffer, UINT32_MAX = N/A */
    bool is_pred;
} PtxReg;

/* Label entry: maps label name to block ID */
typedef struct {
    char name[80];
    uint32_t block_id;
} PtxLabel;

/* Unresolved forward branch */
typedef struct {
    uint32_t block_id;        /* block containing the branch */
    char target_label[80];
    bool is_conditional;
    uint32_t cond_id;         /* condition value ID (for cond branches) */
    uint32_t fallthrough_block;
} PtxUnresolved;

/* Interface global entry for entry point */
typedef struct {
    uint32_t global_id;
} PtxIfaceEntry;

typedef struct {
    PtxLexer lex;
    PtxToken cur;
    SsirModule *mod;
    PtxToSsirOptions opts;

    /* PTX module info */
    int version_major, version_minor;
    int address_size;
    char target[32];

    /* Current function */
    uint32_t func_id;
    uint32_t block_id;
    bool is_entry;
    uint32_t ep_index;

    /* Registers (per-function) */
    PtxReg *regs;
    int reg_count, reg_cap;

    /* Labels (per-function) */
    PtxLabel *labels;
    int label_count, label_cap;

    /* Unresolved branches (per-function) */
    PtxUnresolved *unresolved;
    int unresolved_count, unresolved_cap;

    /* Interface globals for entry point */
    uint32_t *iface;
    int iface_count, iface_cap;

    /* Workgroup size */
    uint32_t wg_size[3];

    /* Next binding index for kernel params */
    uint32_t next_binding;

    /* Predicate state for current instruction */
    uint32_t pred_reg_ptr;      /* pointer to predicate register's local var */
    uint32_t pred_val_type;     /* type of predicate value */
    bool pred_negated;
    bool has_pred;

    /* Device functions: name -> func_id mapping */
    struct { char name[80]; uint32_t func_id; uint32_t ret_type; } *funcs;
    int func_count, func_cap;

    /* Error */
    char error[1024];
    int had_error;
} PtxParser;

/* ============================================================================
 * Parser Helpers
 * ============================================================================ */

static void pp_error(PtxParser *p, const char *fmt, ...) {
    if (p->had_error) return;
    p->had_error = 1;
    int n = snprintf(p->error, sizeof(p->error), "line %d col %d: ",
                     p->cur.line, p->cur.col);
    va_list a;
    va_start(a, fmt);
    vsnprintf(p->error + n, sizeof(p->error) - n, fmt, a);
    va_end(a);
}

static void pp_next(PtxParser *p) {
    p->cur = plx_next(&p->lex);
}

static bool pp_check(PtxParser *p, PtxTokType t) {
    return p->cur.type == t;
}

static bool pp_check_dot(PtxParser *p, const char *s) {
    return dot_eq(&p->cur, s);
}

static bool pp_eat(PtxParser *p, PtxTokType t) {
    if (p->cur.type == t) { pp_next(p); return true; }
    return false;
}

static bool pp_eat_dot(PtxParser *p, const char *s) {
    if (dot_eq(&p->cur, s)) { pp_next(p); return true; }
    return false;
}

static void pp_expect(PtxParser *p, PtxTokType t, const char *what) {
    if (!pp_eat(p, t)) {
        char buf[64];
        tok_to_str(&p->cur, buf, sizeof(buf));
        pp_error(p, "expected %s, got '%s'", what, buf);
    }
}

/* Read current token text into a buffer */
static void pp_read_ident(PtxParser *p, char *buf, int bufsz) {
    tok_to_str(&p->cur, buf, bufsz);
    pp_next(p);
}

/* ============================================================================
 * Register Management
 * ============================================================================ */

static PtxReg *pp_find_reg(PtxParser *p, const char *name) {
    for (int i = 0; i < p->reg_count; i++) {
        if (strcmp(p->regs[i].name, name) == 0)
            return &p->regs[i];
    }
    return NULL;
}

static PtxReg *pp_add_reg(PtxParser *p, const char *name,
                           uint32_t val_type, bool is_pred) {
    /* Check if already exists */
    PtxReg *existing = pp_find_reg(p, name);
    if (existing) return existing;

    if (p->reg_count >= p->reg_cap) {
        p->reg_cap = p->reg_cap ? p->reg_cap * 2 : 64;
        p->regs = (PtxReg *)PTX_REALLOC(p->regs, p->reg_cap * sizeof(PtxReg));
    }
    PtxReg *r = &p->regs[p->reg_count++];
    memset(r, 0, sizeof(*r));
    snprintf(r->name, sizeof(r->name), "%s", name);
    r->val_type = val_type;
    r->is_pred = is_pred;
    r->pending_binding = UINT32_MAX;

    /* Create local variable in SSIR */
    uint32_t ptr_type = ssir_type_ptr(p->mod, val_type, SSIR_ADDR_FUNCTION);
    r->ptr_type = ptr_type;
    r->ptr_id = ssir_function_add_local(p->mod, p->func_id,
        p->opts.preserve_names ? name : NULL, ptr_type);
    return r;
}

/* Load a register value (emit SSIR_OP_LOAD) */
static uint32_t pp_load_reg(PtxParser *p, const char *name) {
    PtxReg *r = pp_find_reg(p, name);
    if (!r) {
        pp_error(p, "undefined register '%s'", name);
        return 0;
    }
    return ssir_build_load(p->mod, p->func_id, p->block_id,
                           r->val_type, r->ptr_id);
}

/* Store to a register (emit SSIR_OP_STORE) */
static void pp_store_reg(PtxParser *p, const char *name, uint32_t value) {
    PtxReg *r = pp_find_reg(p, name);
    if (!r) {
        pp_error(p, "undefined register '%s'", name);
        return;
    }
    ssir_build_store(p->mod, p->func_id, p->block_id, r->ptr_id, value);
}

static void pp_add_iface(PtxParser *p, uint32_t global_id); /* forward decl */

/* Propagate buffer association from source register to destination register.
 * Used for address arithmetic: if %rd_dst = %rd_src + offset, then
 * %rd_dst inherits %rd_src's buffer association (global_id + pending_binding). */
static void pp_propagate_buffer(PtxParser *p, const char *dst, const char *src) {
    PtxReg *dr = pp_find_reg(p, dst);
    PtxReg *sr = pp_find_reg(p, src);
    if (dr && sr) {
        dr->global_id = sr->global_id;
        dr->pending_binding = sr->pending_binding;
    }
}

/* Get element size in bytes for a scalar SSIR type */
static uint32_t pp_type_byte_size(PtxParser *p, uint32_t type_id) {
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

/* Create a storage buffer global variable for a pointer param on first
 * typed access (ld.global/st.global). The element type determines the
 * runtime array element type. Returns the global_id. */
static uint32_t pp_materialize_buffer(PtxParser *p, PtxReg *r,
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
    pp_add_iface(p, gid);

    r->global_id = gid;

    /* Propagate global_id to all registers sharing the same binding */
    for (int i = 0; i < p->reg_count; i++) {
        if (p->regs[i].pending_binding == r->pending_binding)
            p->regs[i].global_id = gid;
    }

    return gid;
}

/* ============================================================================
 * Label Management
 * ============================================================================ */

static uint32_t pp_get_or_create_label(PtxParser *p, const char *name) {
    for (int i = 0; i < p->label_count; i++) {
        if (strcmp(p->labels[i].name, name) == 0)
            return p->labels[i].block_id;
    }
    /* Create new block for this label */
    if (p->label_count >= p->label_cap) {
        p->label_cap = p->label_cap ? p->label_cap * 2 : 32;
        p->labels = (PtxLabel *)PTX_REALLOC(p->labels,
            p->label_cap * sizeof(PtxLabel));
    }
    PtxLabel *l = &p->labels[p->label_count++];
    snprintf(l->name, sizeof(l->name), "%s", name);
    l->block_id = ssir_block_create(p->mod, p->func_id, name);
    return l->block_id;
}

/* ============================================================================
 * Interface helpers
 * ============================================================================ */

static void pp_add_iface(PtxParser *p, uint32_t global_id) {
    if (p->iface_count >= p->iface_cap) {
        p->iface_cap = p->iface_cap ? p->iface_cap * 2 : 16;
        p->iface = (uint32_t *)PTX_REALLOC(p->iface,
            p->iface_cap * sizeof(uint32_t));
    }
    p->iface[p->iface_count++] = global_id;
}

/* ============================================================================
 * Type Helpers
 * ============================================================================ */

static uint32_t pp_ptx_type(PtxParser *p) {
    /* Parse a PTX type suffix like .f32, .u64, .pred, etc. */
    if (pp_check_dot(p, ".pred")) { pp_next(p); return ssir_type_bool(p->mod); }
    if (pp_check_dot(p, ".b8"))   { pp_next(p); return ssir_type_u8(p->mod); }
    if (pp_check_dot(p, ".b16"))  { pp_next(p); return ssir_type_u16(p->mod); }
    if (pp_check_dot(p, ".b32"))  { pp_next(p); return ssir_type_u32(p->mod); }
    if (pp_check_dot(p, ".b64"))  { pp_next(p); return ssir_type_u64(p->mod); }
    if (pp_check_dot(p, ".u8"))   { pp_next(p); return ssir_type_u8(p->mod); }
    if (pp_check_dot(p, ".u16"))  { pp_next(p); return ssir_type_u16(p->mod); }
    if (pp_check_dot(p, ".u32"))  { pp_next(p); return ssir_type_u32(p->mod); }
    if (pp_check_dot(p, ".u64"))  { pp_next(p); return ssir_type_u64(p->mod); }
    if (pp_check_dot(p, ".s8"))   { pp_next(p); return ssir_type_i8(p->mod); }
    if (pp_check_dot(p, ".s16"))  { pp_next(p); return ssir_type_i16(p->mod); }
    if (pp_check_dot(p, ".s32"))  { pp_next(p); return ssir_type_i32(p->mod); }
    if (pp_check_dot(p, ".s64"))  { pp_next(p); return ssir_type_i64(p->mod); }
    if (pp_check_dot(p, ".f16"))  { pp_next(p); return ssir_type_f16(p->mod); }
    if (pp_check_dot(p, ".f32"))  { pp_next(p); return ssir_type_f32(p->mod); }
    if (pp_check_dot(p, ".f64"))  { pp_next(p); return ssir_type_f64(p->mod); }
    return 0;
}

static bool pp_is_signed_type(uint32_t type_kind) {
    /* Check if an SSIR type is signed integer */
    return type_kind == SSIR_TYPE_I8 || type_kind == SSIR_TYPE_I16 ||
           type_kind == SSIR_TYPE_I32 || type_kind == SSIR_TYPE_I64;
}

static bool pp_is_float_type(uint32_t type_kind) {
    return type_kind == SSIR_TYPE_F16 || type_kind == SSIR_TYPE_F32 ||
           type_kind == SSIR_TYPE_F64;
}

/* ============================================================================
 * Constant Helpers
 * ============================================================================ */

static uint32_t pp_const_for_type(PtxParser *p, uint32_t type_id, uint64_t ival,
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
        case SSIR_TYPE_F16:
            /* Simplified: store as zero bits for f16 constants */
            (void)is_float; (void)fval; (void)ival;
            return ssir_const_f16(p->mod, 0);
        case SSIR_TYPE_F32:   return ssir_const_f32(p->mod, is_float ? (float)fval : (float)ival);
        case SSIR_TYPE_F64:   return ssir_const_f64(p->mod, is_float ? fval : (double)ival);
        default: return 0;
    }
}

/* Forward declarations */
static uint32_t pp_load_special_reg(PtxParser *p, const char *name,
                                     uint32_t expected_type);

/* ============================================================================
 * Operand Parsing
 * ============================================================================ */

/* Parse a single operand: register, immediate, or special register */
static uint32_t pp_parse_operand(PtxParser *p, uint32_t expected_type) {
    if (p->had_error) return 0;

    /* Immediate integer */
    if (pp_check(p, PTK_INT_LIT)) {
        uint64_t val = p->cur.int_val;
        pp_next(p);
        return pp_const_for_type(p, expected_type, val, 0.0, false);
    }

    /* Immediate float */
    if (pp_check(p, PTK_FLOAT_LIT)) {
        double val = p->cur.float_val;
        pp_next(p);
        return pp_const_for_type(p, expected_type, 0, val, true);
    }

    /* Negative immediate */
    if (pp_check(p, PTK_MINUS)) {
        pp_next(p);
        if (pp_check(p, PTK_INT_LIT)) {
            int64_t val = -(int64_t)p->cur.int_val;
            pp_next(p);
            return pp_const_for_type(p, expected_type, (uint64_t)val, 0.0, false);
        }
        if (pp_check(p, PTK_FLOAT_LIT)) {
            double val = -p->cur.float_val;
            pp_next(p);
            return pp_const_for_type(p, expected_type, 0, val, true);
        }
        pp_error(p, "expected number after '-'");
        return 0;
    }

    /* Register or special register */
    if (pp_check(p, PTK_IDENT)) {
        char name[80];
        tok_to_str(&p->cur, name, sizeof(name));

        /* Special registers: %tid.x, %ctaid.x, etc. */
        if (strncmp(name, "%tid.", 5) == 0 ||
            strncmp(name, "%ntid.", 6) == 0 ||
            strncmp(name, "%ctaid.", 7) == 0 ||
            strncmp(name, "%nctaid.", 8) == 0 ||
            strcmp(name, "%laneid") == 0 ||
            strcmp(name, "%warpid") == 0) {
            pp_next(p);
            return pp_load_special_reg(p, name, expected_type);
        }

        pp_next(p);
        return pp_load_reg(p, name);
    }

    pp_error(p, "expected operand");
    return 0;
}

/* ============================================================================
 * Special Register Handling
 * ============================================================================ */

static uint32_t pp_ensure_builtin_global(PtxParser *p, SsirBuiltinVar builtin,
                                          uint32_t val_type,
                                          const char *name) {
    /* Check if we already created this global */
    for (int i = 0; i < p->iface_count; i++) {
        SsirGlobalVar *g = ssir_get_global(p->mod, p->iface[i]);
        if (g && g->builtin == builtin) return p->iface[i];
    }

    uint32_t ptr_type = ssir_type_ptr(p->mod, val_type, SSIR_ADDR_INPUT);
    uint32_t gid = ssir_global_var(p->mod, name, ptr_type);
    ssir_global_set_builtin(p->mod, gid, builtin);
    pp_add_iface(p, gid);
    return gid;
}

static int pp_component_index(char c) {
    switch (c) {
        case 'x': return 0;
        case 'y': return 1;
        case 'z': return 2;
        default: return 0;
    }
}

static uint32_t pp_load_special_reg(PtxParser *p, const char *name,
                                     uint32_t expected_type) {
    uint32_t u32_t = ssir_type_u32(p->mod);
    uint32_t vec3u_t = ssir_type_vec(p->mod, u32_t, 3);
    SsirBuiltinVar builtin = SSIR_BUILTIN_LOCAL_INVOCATION_ID;
    int comp = 0;
    const char *gname = "gl_LocalInvocationID";

    if (strncmp(name, "%tid.", 5) == 0) {
        builtin = SSIR_BUILTIN_LOCAL_INVOCATION_ID;
        comp = pp_component_index(name[5]);
        gname = "gl_LocalInvocationID";
    } else if (strncmp(name, "%ctaid.", 7) == 0) {
        builtin = SSIR_BUILTIN_WORKGROUP_ID;
        comp = pp_component_index(name[7]);
        gname = "gl_WorkGroupID";
    } else if (strncmp(name, "%nctaid.", 8) == 0) {
        builtin = SSIR_BUILTIN_NUM_WORKGROUPS;
        comp = pp_component_index(name[8]);
        gname = "gl_NumWorkGroups";
    } else if (strncmp(name, "%ntid.", 6) == 0) {
        /* Workgroup size - emit as constant if known, else use built-in */
        comp = pp_component_index(name[6]);
        if (p->wg_size[comp] > 0) {
            uint32_t val = ssir_const_u32(p->mod, p->wg_size[comp]);
            if (expected_type != u32_t && expected_type != 0)
                return ssir_build_convert(p->mod, p->func_id, p->block_id,
                                          expected_type, val);
            return val;
        }
        /* Fall through — treat like a built-in if size unknown */
        builtin = SSIR_BUILTIN_NUM_WORKGROUPS; /* best approximation */
        gname = "gl_NumWorkGroups";
    } else if (strcmp(name, "%laneid") == 0) {
        builtin = SSIR_BUILTIN_SUBGROUP_INVOCATION_ID;
        uint32_t gid = pp_ensure_builtin_global(p, builtin, u32_t, "gl_SubgroupInvocationID");
        uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, u32_t, gid);
        if (expected_type != u32_t && expected_type != 0)
            return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
        return val;
    } else if (strcmp(name, "%warpid") == 0) {
        /* warpid = tid.x / 32 (approximation) */
        uint32_t gid = pp_ensure_builtin_global(p, SSIR_BUILTIN_LOCAL_INVOCATION_ID,
                                                 vec3u_t, "gl_LocalInvocationID");
        uint32_t vec = ssir_build_load(p->mod, p->func_id, p->block_id, vec3u_t, gid);
        uint32_t tidx = ssir_build_extract(p->mod, p->func_id, p->block_id, u32_t, vec, 0);
        uint32_t c32 = ssir_const_u32(p->mod, 32);
        uint32_t val = ssir_build_div(p->mod, p->func_id, p->block_id, u32_t, tidx, c32);
        if (expected_type != u32_t && expected_type != 0)
            return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
        return val;
    }

    /* Vec3 built-ins: load vec3, extract component */
    uint32_t gid = pp_ensure_builtin_global(p, builtin, vec3u_t, gname);
    uint32_t vec = ssir_build_load(p->mod, p->func_id, p->block_id, vec3u_t, gid);
    uint32_t val = ssir_build_extract(p->mod, p->func_id, p->block_id, u32_t, vec, comp);

    if (expected_type != u32_t && expected_type != 0)
        return ssir_build_convert(p->mod, p->func_id, p->block_id, expected_type, val);
    return val;
}

/* ============================================================================
 * Module-Level Parsing
 * ============================================================================ */

static void pp_parse_version(PtxParser *p) {
    /* .version X.Y */
    pp_next(p); /* skip .version */
    if (pp_check(p, PTK_FLOAT_LIT)) {
        p->version_major = (int)p->cur.float_val;
        p->version_minor = (int)((p->cur.float_val - p->version_major) * 10 + 0.5);
        pp_next(p);
    } else if (pp_check(p, PTK_INT_LIT)) {
        p->version_major = (int)p->cur.int_val;
        pp_next(p);
        p->version_minor = 0;
    } else {
        pp_error(p, "expected version number");
    }
}

static void pp_parse_target(PtxParser *p) {
    /* .target sm_XX [, ...] */
    pp_next(p); /* skip .target */
    tok_to_str(&p->cur, p->target, sizeof(p->target));
    pp_next(p);
    /* Skip optional extra targets */
    while (pp_eat(p, PTK_COMMA)) {
        pp_next(p); /* skip extra target name */
    }
}

static void pp_parse_address_size(PtxParser *p) {
    /* .address_size 32|64 */
    pp_next(p); /* skip .address_size */
    if (pp_check(p, PTK_INT_LIT)) {
        p->address_size = (int)p->cur.int_val;
        pp_next(p);
    } else {
        pp_error(p, "expected 32 or 64 for .address_size");
    }
}

static void pp_parse_global_decl(PtxParser *p, const char *space_name) {
    /* .global/.shared/.const [.align N] .type name[size]; */
    pp_next(p); /* skip space directive */

    SsirAddressSpace addr_space = SSIR_ADDR_STORAGE;
    if (strcmp(space_name, ".shared") == 0) addr_space = SSIR_ADDR_WORKGROUP;
    else if (strcmp(space_name, ".const") == 0) addr_space = SSIR_ADDR_UNIFORM;

    /* Optional .extern */
    /* Optional .align */
    uint32_t alignment = 0;
    if (pp_check_dot(p, ".align")) {
        pp_next(p);
        if (pp_check(p, PTK_INT_LIT)) { alignment = (uint32_t)p->cur.int_val; pp_next(p); }
    }

    /* Type */
    uint32_t elem_type = pp_ptx_type(p);
    if (!elem_type) { pp_error(p, "expected type in global declaration"); return; }

    /* Name and optional array size */
    char name[80];
    pp_read_ident(p, name, sizeof(name));

    uint32_t array_size = 0;
    if (pp_eat(p, PTK_LBRACKET)) {
        if (pp_check(p, PTK_INT_LIT)) {
            array_size = (uint32_t)p->cur.int_val;
            pp_next(p);
        }
        /* Handle multi-dim arrays: [32][32] -> flatten */
        pp_expect(p, PTK_RBRACKET, "]");
        while (pp_eat(p, PTK_LBRACKET)) {
            if (pp_check(p, PTK_INT_LIT)) {
                array_size *= (uint32_t)p->cur.int_val;
                pp_next(p);
            }
            pp_expect(p, PTK_RBRACKET, "]");
        }
    }

    /* Build SSIR global variable */
    uint32_t var_type = array_size > 0
        ? ssir_type_array(p->mod, elem_type, array_size)
        : elem_type;
    uint32_t ptr_type = ssir_type_ptr(p->mod, var_type, addr_space);
    uint32_t gid = ssir_global_var(p->mod, name, ptr_type);
    (void)alignment; /* alignment info stored but not needed in SSIR currently */

    pp_add_iface(p, gid);
    pp_eat(p, PTK_SEMI);
}

/* ============================================================================
 * Register Declaration Parsing
 * ============================================================================ */

static void pp_parse_reg_decl(PtxParser *p) {
    /* .reg [.v2|.v4] .type name[<N>] [, name2, ...]; */
    pp_next(p); /* skip .reg */

    /* Optional vector prefix */
    int vec_width = 1;
    if (pp_check_dot(p, ".v2")) { vec_width = 2; pp_next(p); }
    else if (pp_check_dot(p, ".v4")) { vec_width = 4; pp_next(p); }

    /* Type */
    uint32_t scalar_type = pp_ptx_type(p);
    if (!scalar_type) { pp_error(p, "expected type in .reg"); return; }

    bool is_pred = false;
    {
        SsirType *ty = ssir_get_type(p->mod, scalar_type);
        if (ty && ty->kind == SSIR_TYPE_BOOL) is_pred = true;
    }

    uint32_t val_type = scalar_type;
    if (vec_width > 1)
        val_type = ssir_type_vec(p->mod, scalar_type, vec_width);

    /* Names: can be "name", "name<N>", or comma-separated list */
    do {
        char name[80];
        pp_read_ident(p, name, sizeof(name));

        /* Parameterized: name<N> */
        if (pp_eat(p, PTK_LANGLE)) {
            if (pp_check(p, PTK_INT_LIT)) {
                int count = (int)p->cur.int_val;
                pp_next(p);
                pp_expect(p, PTK_RANGLE, ">");
                for (int i = 0; i < count; i++) {
                    char pname[80];
                    snprintf(pname, sizeof(pname), "%s%d", name, i);
                    pp_add_reg(p, pname, val_type, is_pred);
                }
            } else {
                pp_error(p, "expected count in parameterized register");
            }
        } else {
            pp_add_reg(p, name, val_type, is_pred);
        }
    } while (pp_eat(p, PTK_COMMA));

    pp_eat(p, PTK_SEMI);
}

/* ============================================================================
 * Instruction Parsing - Arithmetic
 * ============================================================================ */

static void pp_skip_modifiers(PtxParser *p) {
    /* Skip rounding, saturation, approx, ftz, lo/hi/wide modifiers */
    while (pp_check_dot(p, ".rn") || pp_check_dot(p, ".rz") ||
           pp_check_dot(p, ".rm") || pp_check_dot(p, ".rp") ||
           pp_check_dot(p, ".rni") || pp_check_dot(p, ".rzi") ||
           pp_check_dot(p, ".rmi") || pp_check_dot(p, ".rpi") ||
           pp_check_dot(p, ".sat") || pp_check_dot(p, ".approx") ||
           pp_check_dot(p, ".ftz") || pp_check_dot(p, ".lo") ||
           pp_check_dot(p, ".hi") || pp_check_dot(p, ".wide")) {
        pp_next(p);
    }
}

static void pp_parse_arith(PtxParser *p, const char *opname) {
    /* opcode[.modifiers].type dst, src1, src2 */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");

    /* Capture first operand's register name for buffer propagation */
    char src1_name[80] = {0};
    if (pp_check(p, PTK_IDENT))
        tok_to_str(&p->cur, src1_name, sizeof(src1_name));

    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t result = 0;
    if (strcmp(opname, "add") == 0)
        result = ssir_build_add(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "sub") == 0)
        result = ssir_build_sub(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "mul") == 0)
        result = ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "div") == 0)
        result = ssir_build_div(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "rem") == 0)
        result = ssir_build_rem(p->mod, p->func_id, p->block_id, type, a, b);

    if (result) {
        pp_store_reg(p, dst, result);
        /* For address arithmetic (add/sub with u64/s64), propagate buffer
         * association from the base pointer operand to the result. */
        if (src1_name[0] && (strcmp(opname, "add") == 0 || strcmp(opname, "sub") == 0)) {
            SsirType *t = ssir_get_type(p->mod, type);
            if (t && (t->kind == SSIR_TYPE_U64 || t->kind == SSIR_TYPE_I64))
                pp_propagate_buffer(p, dst, src1_name);
        }
    }
}

static void pp_parse_unary_arith(PtxParser *p, const char *opname) {
    /* opcode[.modifiers].type dst, src */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t result = 0;
    if (strcmp(opname, "neg") == 0)
        result = ssir_build_neg(p->mod, p->func_id, p->block_id, type, a);
    else if (strcmp(opname, "abs") == 0) {
        uint32_t args[] = { a };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_ABS, args, 1);
    } else if (strcmp(opname, "not") == 0) {
        SsirType *ty = ssir_get_type(p->mod, type);
        if (ty && ty->kind == SSIR_TYPE_BOOL)
            result = ssir_build_not(p->mod, p->func_id, p->block_id, type, a);
        else
            result = ssir_build_bit_not(p->mod, p->func_id, p->block_id, type, a);
    } else if (strcmp(opname, "cnot") == 0) {
        /* cnot.type d, a => d = (a == 0) ? 1 : 0 */
        uint32_t zero = pp_const_for_type(p, type, 0, 0.0, false);
        uint32_t bool_t = ssir_type_bool(p->mod);
        uint32_t cmp = ssir_build_eq(p->mod, p->func_id, p->block_id, bool_t, a, zero);
        uint32_t one = pp_const_for_type(p, type, 1, 0.0, false);
        uint32_t args[] = { cmp, one, zero };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, SSIR_BUILTIN_SELECT, args, 3);
    }

    if (result) pp_store_reg(p, dst, result);
}

static void pp_parse_minmax(PtxParser *p, const char *opname) {
    /* min/max[.modifiers].type dst, src1, src2 */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    SsirBuiltinId bid = (strcmp(opname, "min") == 0) ? SSIR_BUILTIN_MIN : SSIR_BUILTIN_MAX;
    uint32_t args[] = { a, b };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, bid, args, 2);
    pp_store_reg(p, dst, result);
}

static void pp_parse_mad(PtxParser *p) {
    /* mad[.lo|.hi|.wide][.modifiers].type dst, a, b, c => dst = a*b + c */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for mad"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t c = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    SsirType *ty = ssir_get_type(p->mod, type);
    if (ty && pp_is_float_type(ty->kind)) {
        /* Use FMA for float */
        uint32_t args[] = { a, b, c };
        uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                             type, SSIR_BUILTIN_FMA, args, 3);
        pp_store_reg(p, dst, result);
    } else {
        /* mul + add for integer */
        uint32_t mul = ssir_build_mul(p->mod, p->func_id, p->block_id, type, a, b);
        uint32_t result = ssir_build_add(p->mod, p->func_id, p->block_id, type, mul, c);
        pp_store_reg(p, dst, result);
    }
}

static void pp_parse_fma(PtxParser *p) {
    /* fma[.rn|.rz|.rm|.rp].type dst, a, b, c */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for fma"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t c = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t args[] = { a, b, c };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_FMA, args, 3);
    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Bitwise
 * ============================================================================ */

static void pp_parse_bitwise(PtxParser *p, const char *opname) {
    /* and/or/xor.type dst, src1, src2 */
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    SsirType *ty = ssir_get_type(p->mod, type);
    bool is_bool = ty && ty->kind == SSIR_TYPE_BOOL;

    uint32_t result = 0;
    if (strcmp(opname, "and") == 0)
        result = is_bool ? ssir_build_and(p->mod, p->func_id, p->block_id, type, a, b)
                         : ssir_build_bit_and(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "or") == 0)
        result = is_bool ? ssir_build_or(p->mod, p->func_id, p->block_id, type, a, b)
                         : ssir_build_bit_or(p->mod, p->func_id, p->block_id, type, a, b);
    else if (strcmp(opname, "xor") == 0)
        result = ssir_build_bit_xor(p->mod, p->func_id, p->block_id, type, a, b);

    if (result) pp_store_reg(p, dst, result);
}

static void pp_parse_shift(PtxParser *p, const char *opname) {
    /* shl/shr.type dst, src1, src2 */
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, ssir_type_u32(p->mod));
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    SsirType *ty = ssir_get_type(p->mod, type);
    uint32_t result;
    if (strcmp(opname, "shl") == 0)
        result = ssir_build_shl(p->mod, p->func_id, p->block_id, type, a, b);
    else if (ty && pp_is_signed_type(ty->kind))
        result = ssir_build_shr(p->mod, p->func_id, p->block_id, type, a, b);
    else
        result = ssir_build_shr_logical(p->mod, p->func_id, p->block_id, type, a, b);

    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Comparison & Selection
 * ============================================================================ */

static SsirOpcode pp_cmp_op(const char *cmp) {
    if (strcmp(cmp, "eq") == 0 || strcmp(cmp, "equ") == 0) return SSIR_OP_EQ;
    if (strcmp(cmp, "ne") == 0 || strcmp(cmp, "neu") == 0) return SSIR_OP_NE;
    if (strcmp(cmp, "lt") == 0 || strcmp(cmp, "lo") == 0 || strcmp(cmp, "ltu") == 0) return SSIR_OP_LT;
    if (strcmp(cmp, "le") == 0 || strcmp(cmp, "ls") == 0 || strcmp(cmp, "leu") == 0) return SSIR_OP_LE;
    if (strcmp(cmp, "gt") == 0 || strcmp(cmp, "hi") == 0 || strcmp(cmp, "gtu") == 0) return SSIR_OP_GT;
    if (strcmp(cmp, "ge") == 0 || strcmp(cmp, "hs") == 0 || strcmp(cmp, "geu") == 0) return SSIR_OP_GE;
    return SSIR_OP_EQ; /* fallback */
}

static void pp_parse_setp(PtxParser *p) {
    /* setp.cmp[.combine].type dst[|dst2], a, b [, pred_src] */

    /* Parse comparison operator */
    char cmp_str[16] = {0};
    if (pp_check(p, PTK_DOT_TOKEN)) {
        tok_to_str(&p->cur, cmp_str, sizeof(cmp_str));
        /* Remove leading dot */
        memmove(cmp_str, cmp_str + 1, strlen(cmp_str));
        pp_next(p);
    }

    /* Optional combine mode (.and, .or, .xor) */
    bool has_combine = false;
    char combine[8] = {0};
    if (pp_check_dot(p, ".and") || pp_check_dot(p, ".or") || pp_check_dot(p, ".xor")) {
        tok_to_str(&p->cur, combine, sizeof(combine));
        memmove(combine, combine + 1, strlen(combine));
        has_combine = true;
        pp_next(p);
    }

    /* Type */
    uint32_t src_type = pp_ptx_type(p);
    if (!src_type) { pp_error(p, "expected type for setp"); return; }

    /* Destination predicate(s) */
    char dst1[80];
    pp_read_ident(p, dst1, sizeof(dst1));

    char dst2[80] = {0};
    if (pp_eat(p, PTK_PIPE)) {
        pp_read_ident(p, dst2, sizeof(dst2));
    }

    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, src_type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, src_type);

    /* Optional predicate source for combine */
    uint32_t pred_src = 0;
    if (has_combine && pp_eat(p, PTK_COMMA)) {
        pred_src = pp_parse_operand(p, ssir_type_bool(p->mod));
    }
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t bool_t = ssir_type_bool(p->mod);
    SsirOpcode cmp_opcode = pp_cmp_op(cmp_str);

    /* Build comparison */
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

    /* Apply combine if present */
    if (has_combine && pred_src) {
        if (strcmp(combine, "and") == 0)
            cmp_result = ssir_build_and(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
        else if (strcmp(combine, "or") == 0)
            cmp_result = ssir_build_or(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
        else if (strcmp(combine, "xor") == 0) {
            /* xor on bools = ne */
            cmp_result = ssir_build_ne(p->mod, p->func_id, p->block_id, bool_t, cmp_result, pred_src);
        }
    }

    pp_store_reg(p, dst1, cmp_result);

    /* Second predicate = complement */
    if (dst2[0]) {
        uint32_t neg = ssir_build_not(p->mod, p->func_id, p->block_id, bool_t, cmp_result);
        pp_store_reg(p, dst2, neg);
    }
}

static void pp_parse_selp(PtxParser *p) {
    /* selp.type dst, src1, src2, pred */
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for selp"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t a = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t b = pp_parse_operand(p, type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t pred = pp_parse_operand(p, ssir_type_bool(p->mod));
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    /* SSIR SELECT expects {falseVal, trueVal, condition} */
    uint32_t args[] = { b, a, pred };
    uint32_t result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                         type, SSIR_BUILTIN_SELECT, args, 3);
    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Data Movement
 * ============================================================================ */

static void pp_parse_mov(PtxParser *p) {
    /* mov.type dst, src */
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for mov"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t src = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;
    pp_store_reg(p, dst, src);
}

/* ============================================================================
 * Instruction Parsing - Memory Operations
 * ============================================================================ */

static SsirAddressSpace pp_mem_space(PtxParser *p) {
    if (pp_check_dot(p, ".global")) { pp_next(p); return SSIR_ADDR_STORAGE; }
    if (pp_check_dot(p, ".shared")) { pp_next(p); return SSIR_ADDR_WORKGROUP; }
    if (pp_check_dot(p, ".local"))  { pp_next(p); return SSIR_ADDR_FUNCTION; }
    if (pp_check_dot(p, ".const"))  { pp_next(p); return SSIR_ADDR_UNIFORM; }
    if (pp_check_dot(p, ".param"))  { pp_next(p); return SSIR_ADDR_UNIFORM; }
    return SSIR_ADDR_STORAGE; /* default to global */
}

static void pp_skip_cache_ops(PtxParser *p) {
    while (pp_check_dot(p, ".ca") || pp_check_dot(p, ".cg") ||
           pp_check_dot(p, ".cs") || pp_check_dot(p, ".cv") ||
           pp_check_dot(p, ".wb") || pp_check_dot(p, ".wt")) {
        pp_next(p);
    }
}

/* Parse address expression: [reg], [reg + offset], [symbol], [symbol + offset]
 * If base_name_out is non-NULL, writes the base register name into it. */
static uint32_t pp_parse_address_ex(PtxParser *p, uint32_t ptr_type,
                                     char *base_name_out, int base_name_sz) {
    (void)ptr_type;
    pp_expect(p, PTK_LBRACKET, "[");
    char base[80];
    pp_read_ident(p, base, sizeof(base));
    if (base_name_out)
        snprintf(base_name_out, base_name_sz, "%s", base);

    uint32_t addr = pp_load_reg(p, base);

    /* Optional + offset */
    if (pp_eat(p, PTK_PLUS)) {
        uint32_t u64_t = ssir_type_u64(p->mod);
        uint32_t offset;
        if (pp_check(p, PTK_INT_LIT)) {
            offset = ssir_const_u64(p->mod, p->cur.int_val);
            pp_next(p);
        } else {
            char off_reg[80];
            pp_read_ident(p, off_reg, sizeof(off_reg));
            offset = pp_load_reg(p, off_reg);
        }
        addr = ssir_build_add(p->mod, p->func_id, p->block_id, u64_t, addr, offset);
    }

    pp_expect(p, PTK_RBRACKET, "]");
    return addr;
}

static uint32_t pp_parse_address(PtxParser *p, uint32_t ptr_type) {
    return pp_parse_address_ex(p, ptr_type, NULL, 0);
}

static void pp_parse_ld(PtxParser *p) {
    /* ld.space[.cache].type dst, [addr] */
    SsirAddressSpace space = pp_mem_space(p);
    pp_skip_cache_ops(p);

    /* Optional vector */
    int vec_width = 1;
    if (pp_check_dot(p, ".v2")) { vec_width = 2; pp_next(p); }
    else if (pp_check_dot(p, ".v4")) { vec_width = 4; pp_next(p); }

    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for ld"); return; }

    if (vec_width == 1) {
        /* Scalar load */
        char dst[80];
        pp_read_ident(p, dst, sizeof(dst));
        pp_expect(p, PTK_COMMA, ",");

        /* For ld.param, we load from parameter variable directly */
        if (space == SSIR_ADDR_UNIFORM) {
            pp_expect(p, PTK_LBRACKET, "[");
            char param_name[80];
            pp_read_ident(p, param_name, sizeof(param_name));
            /* skip optional offset */
            if (pp_eat(p, PTK_PLUS)) {
                if (pp_check(p, PTK_INT_LIT)) pp_next(p);
                else { char tmp[80]; pp_read_ident(p, tmp, sizeof(tmp)); }
            }
            pp_expect(p, PTK_RBRACKET, "]");
            pp_eat(p, PTK_SEMI);

            /* Load from the parameter's local variable */
            PtxReg *param_reg = pp_find_reg(p, param_name);
            if (param_reg) {
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                               param_reg->val_type, param_reg->ptr_id);
                /* Convert if types differ */
                if (param_reg->val_type != type)
                    val = ssir_build_convert(p->mod, p->func_id, p->block_id, type, val);
                pp_store_reg(p, dst, val);
                /* Propagate buffer association for pointer params */
                pp_propagate_buffer(p, dst, param_name);
            } else {
                pp_error(p, "unknown parameter '%s'", param_name);
            }
            return;
        }

        /* Check if address register has buffer info for access chain path */
        if (space == SSIR_ADDR_STORAGE) {
            char base_name[80] = {0};
            uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
            uint32_t byte_offset = pp_parse_address_ex(p, ptr_type, base_name, sizeof(base_name));
            pp_eat(p, PTK_SEMI);
            if (p->had_error) return;

            PtxReg *base_reg = base_name[0] ? pp_find_reg(p, base_name) : NULL;
            if (base_reg && (base_reg->pending_binding != UINT32_MAX || base_reg->global_id != 0)) {
                /* Buffer-aware access chain path */
                uint32_t buf_global = pp_materialize_buffer(p, base_reg, type);
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t elem_sz = ssir_const_u64(p->mod, pp_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id, u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id, u32_t, idx_u64);
                uint32_t const_0 = ssir_const_u32(p->mod, 0);
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_STORAGE);
                uint32_t indices[] = { const_0, idx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id, elem_ptr_type, buf_global, indices, 2);
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, ptr);
                pp_store_reg(p, dst, val);
            } else {
                /* No buffer info — raw pointer load */
                uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, byte_offset);
                pp_store_reg(p, dst, val);
            }
        } else {
            uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
            uint32_t addr = pp_parse_address(p, ptr_type);
            pp_eat(p, PTK_SEMI);
            if (p->had_error) return;

            uint32_t val = ssir_build_load(p->mod, p->func_id, p->block_id, type, addr);
            pp_store_reg(p, dst, val);
        }
    } else {
        /* Vector load: ld.v4.f32 {r1,r2,r3,r4}, [addr] */
        pp_expect(p, PTK_LBRACE, "{");
        char dsts[4][80];
        int n = 0;
        while (n < vec_width) {
            pp_read_ident(p, dsts[n], sizeof(dsts[n]));
            n++;
            if (n < vec_width) pp_expect(p, PTK_COMMA, ",");
        }
        pp_expect(p, PTK_RBRACE, "}");
        pp_expect(p, PTK_COMMA, ",");

        uint32_t vec_type = ssir_type_vec(p->mod, type, vec_width);
        uint32_t ptr_type = ssir_type_ptr(p->mod, vec_type, space);
        uint32_t addr = pp_parse_address(p, ptr_type);
        pp_eat(p, PTK_SEMI);
        if (p->had_error) return;

        uint32_t vec_val = ssir_build_load(p->mod, p->func_id, p->block_id, vec_type, addr);
        for (int i = 0; i < vec_width; i++) {
            uint32_t comp = ssir_build_extract(p->mod, p->func_id, p->block_id,
                                               type, vec_val, i);
            pp_store_reg(p, dsts[i], comp);
        }
    }
}

static void pp_parse_st(PtxParser *p) {
    /* st.space[.cache].type [addr], src */
    SsirAddressSpace space = pp_mem_space(p);
    pp_skip_cache_ops(p);

    int vec_width = 1;
    if (pp_check_dot(p, ".v2")) { vec_width = 2; pp_next(p); }
    else if (pp_check_dot(p, ".v4")) { vec_width = 4; pp_next(p); }

    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for st"); return; }

    if (vec_width == 1) {
        /* Check if address register has buffer info for access chain path */
        if (space == SSIR_ADDR_STORAGE) {
            char base_name[80] = {0};
            uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
            uint32_t byte_offset = pp_parse_address_ex(p, ptr_type, base_name, sizeof(base_name));
            pp_expect(p, PTK_COMMA, ",");
            uint32_t val = pp_parse_operand(p, type);
            pp_eat(p, PTK_SEMI);
            if (p->had_error) return;

            PtxReg *base_reg = base_name[0] ? pp_find_reg(p, base_name) : NULL;
            if (base_reg && (base_reg->pending_binding != UINT32_MAX || base_reg->global_id != 0)) {
                /* Buffer-aware access chain path */
                uint32_t buf_global = pp_materialize_buffer(p, base_reg, type);
                uint32_t u64_t = ssir_type_u64(p->mod);
                uint32_t u32_t = ssir_type_u32(p->mod);
                uint32_t elem_sz = ssir_const_u64(p->mod, pp_type_byte_size(p, type));
                uint32_t idx_u64 = ssir_build_div(p->mod, p->func_id, p->block_id, u64_t, byte_offset, elem_sz);
                uint32_t idx = ssir_build_convert(p->mod, p->func_id, p->block_id, u32_t, idx_u64);
                uint32_t const_0 = ssir_const_u32(p->mod, 0);
                uint32_t elem_ptr_type = ssir_type_ptr(p->mod, type, SSIR_ADDR_STORAGE);
                uint32_t indices[] = { const_0, idx };
                uint32_t ptr = ssir_build_access(p->mod, p->func_id, p->block_id, elem_ptr_type, buf_global, indices, 2);
                ssir_build_store(p->mod, p->func_id, p->block_id, ptr, val);
            } else {
                /* No buffer info — raw pointer store */
                ssir_build_store(p->mod, p->func_id, p->block_id, byte_offset, val);
            }
        } else {
            uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
            uint32_t addr = pp_parse_address(p, ptr_type);
            pp_expect(p, PTK_COMMA, ",");
            uint32_t val = pp_parse_operand(p, type);
            pp_eat(p, PTK_SEMI);
            if (p->had_error) return;

            ssir_build_store(p->mod, p->func_id, p->block_id, addr, val);
        }
    } else {
        /* Vector store: st.v4.f32 [addr], {r1,r2,r3,r4} */
        uint32_t vec_type = ssir_type_vec(p->mod, type, vec_width);
        uint32_t ptr_type = ssir_type_ptr(p->mod, vec_type, space);
        uint32_t addr = pp_parse_address(p, ptr_type);
        pp_expect(p, PTK_COMMA, ",");

        pp_expect(p, PTK_LBRACE, "{");
        uint32_t comps[4];
        for (int i = 0; i < vec_width; i++) {
            comps[i] = pp_parse_operand(p, type);
            if (i < vec_width - 1) pp_expect(p, PTK_COMMA, ",");
        }
        pp_expect(p, PTK_RBRACE, "}");
        pp_eat(p, PTK_SEMI);
        if (p->had_error) return;

        uint32_t vec_val = ssir_build_construct(p->mod, p->func_id, p->block_id,
                                                vec_type, comps, vec_width);
        ssir_build_store(p->mod, p->func_id, p->block_id, addr, vec_val);
    }
}

static void pp_parse_cvta(PtxParser *p) {
    /* cvta.to.space.type dst, src */
    pp_eat_dot(p, ".to");
    pp_mem_space(p); /* consume space, we don't change anything */
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for cvta"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");

    /* Capture source register name for buffer propagation */
    char src_name[80] = {0};
    if (pp_check(p, PTK_IDENT))
        tok_to_str(&p->cur, src_name, sizeof(src_name));

    uint32_t src = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;
    /* cvta is essentially a no-op in our model */
    pp_store_reg(p, dst, src);
    if (src_name[0]) pp_propagate_buffer(p, dst, src_name);
}

/* ============================================================================
 * Instruction Parsing - Type Conversion
 * ============================================================================ */

static void pp_parse_cvt(PtxParser *p) {
    /* cvt[.rn|.rz|.rm|.rp][.ftz][.sat].dst_type.src_type dst, src */
    pp_skip_modifiers(p);
    uint32_t dst_type = pp_ptx_type(p);
    if (!dst_type) { pp_error(p, "expected dest type for cvt"); return; }
    uint32_t src_type = pp_ptx_type(p);
    if (!src_type) { pp_error(p, "expected src type for cvt"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t src = pp_parse_operand(p, src_type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t result;
    SsirType *dt = ssir_get_type(p->mod, dst_type);
    SsirType *st = ssir_get_type(p->mod, src_type);
    if (dt && st && dt->kind != st->kind) {
        /* Bitcast to declared src_type first. This handles cases like
         * ld.global.u32 + cvt.f32.s32 where the register holds u32 but
         * the conversion needs s32 interpretation for correct signedness. */
        src = ssir_build_bitcast(p->mod, p->func_id, p->block_id,
                                 src_type, src);
        result = ssir_build_convert(p->mod, p->func_id, p->block_id, dst_type, src);
    } else {
        result = src; /* same type = nop */
    }
    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Math Functions
 * ============================================================================ */

static void pp_parse_math_unary(PtxParser *p, const char *opname) {
    /* rcp/sqrt/rsqrt/sin/cos/lg2/ex2[.approx][.rn].type dst, src */
    pp_skip_modifiers(p);
    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for %s", opname); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");
    uint32_t src = pp_parse_operand(p, type);
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t result;
    if (strcmp(opname, "rcp") == 0) {
        /* 1.0 / x */
        uint32_t one = ssir_const_f32(p->mod, 1.0f);
        SsirType *ty = ssir_get_type(p->mod, type);
        if (ty && ty->kind == SSIR_TYPE_F64) one = ssir_const_f64(p->mod, 1.0);
        result = ssir_build_div(p->mod, p->func_id, p->block_id, type, one, src);
    } else {
        SsirBuiltinId bid = SSIR_BUILTIN_SQRT;
        if (strcmp(opname, "sqrt") == 0) bid = SSIR_BUILTIN_SQRT;
        else if (strcmp(opname, "rsqrt") == 0) bid = SSIR_BUILTIN_INVERSESQRT;
        else if (strcmp(opname, "sin") == 0) bid = SSIR_BUILTIN_SIN;
        else if (strcmp(opname, "cos") == 0) bid = SSIR_BUILTIN_COS;
        else if (strcmp(opname, "lg2") == 0) bid = SSIR_BUILTIN_LOG2;
        else if (strcmp(opname, "ex2") == 0) bid = SSIR_BUILTIN_EXP2;
        uint32_t args[] = { src };
        result = ssir_build_builtin(p->mod, p->func_id, p->block_id,
                                    type, bid, args, 1);
    }
    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Atomics
 * ============================================================================ */

static void pp_parse_atom(PtxParser *p) {
    /* atom.space.op.type dst, [addr], src [, src2] */
    SsirAddressSpace space = pp_mem_space(p);

    /* Parse atomic operation */
    SsirAtomicOp atom_op = SSIR_ATOMIC_ADD;
    bool is_cas = false;
    if (pp_check_dot(p, ".add")) { atom_op = SSIR_ATOMIC_ADD; pp_next(p); }
    else if (pp_check_dot(p, ".min")) { atom_op = SSIR_ATOMIC_MIN; pp_next(p); }
    else if (pp_check_dot(p, ".max")) { atom_op = SSIR_ATOMIC_MAX; pp_next(p); }
    else if (pp_check_dot(p, ".and")) { atom_op = SSIR_ATOMIC_AND; pp_next(p); }
    else if (pp_check_dot(p, ".or"))  { atom_op = SSIR_ATOMIC_OR;  pp_next(p); }
    else if (pp_check_dot(p, ".xor")) { atom_op = SSIR_ATOMIC_XOR; pp_next(p); }
    else if (pp_check_dot(p, ".exch")) { atom_op = SSIR_ATOMIC_EXCHANGE; pp_next(p); }
    else if (pp_check_dot(p, ".cas")) { atom_op = SSIR_ATOMIC_COMPARE_EXCHANGE; is_cas = true; pp_next(p); }
    else if (pp_check_dot(p, ".inc")) { atom_op = SSIR_ATOMIC_ADD; pp_next(p); } /* approximate */
    else if (pp_check_dot(p, ".dec")) { atom_op = SSIR_ATOMIC_SUB; pp_next(p); } /* approximate */

    uint32_t type = pp_ptx_type(p);
    if (!type) { pp_error(p, "expected type for atom"); return; }

    char dst[80];
    pp_read_ident(p, dst, sizeof(dst));
    pp_expect(p, PTK_COMMA, ",");

    uint32_t ptr_type = ssir_type_ptr(p->mod, type, space);
    uint32_t addr = pp_parse_address(p, ptr_type);
    pp_expect(p, PTK_COMMA, ",");
    uint32_t val = pp_parse_operand(p, type);

    uint32_t cmp = 0;
    if (is_cas) {
        pp_expect(p, PTK_COMMA, ",");
        cmp = pp_parse_operand(p, type);
    }
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t result = ssir_build_atomic(p->mod, p->func_id, p->block_id,
                                        type, atom_op, addr, val, cmp);
    pp_store_reg(p, dst, result);
}

/* ============================================================================
 * Instruction Parsing - Barriers
 * ============================================================================ */

static void pp_parse_bar(PtxParser *p) {
    /* bar.sync N [, count]; */
    if (pp_check_dot(p, ".sync")) pp_next(p);
    else if (pp_check_dot(p, ".arrive")) pp_next(p);
    else if (pp_check_dot(p, ".red")) pp_next(p);

    /* Barrier ID (usually 0) */
    if (pp_check(p, PTK_INT_LIT)) pp_next(p);
    /* Optional thread count */
    if (pp_eat(p, PTK_COMMA)) {
        if (pp_check(p, PTK_INT_LIT)) pp_next(p);
        else if (pp_check(p, PTK_IDENT)) pp_next(p);
    }
    pp_eat(p, PTK_SEMI);

    ssir_build_barrier(p->mod, p->func_id, p->block_id, SSIR_BARRIER_WORKGROUP);
}

static void pp_parse_membar(PtxParser *p) {
    /* membar.scope; */
    SsirBarrierScope scope = SSIR_BARRIER_WORKGROUP;
    if (pp_check_dot(p, ".cta")) { scope = SSIR_BARRIER_WORKGROUP; pp_next(p); }
    else if (pp_check_dot(p, ".gl")) { scope = SSIR_BARRIER_STORAGE; pp_next(p); }
    else if (pp_check_dot(p, ".sys")) { scope = SSIR_BARRIER_STORAGE; pp_next(p); }
    pp_eat(p, PTK_SEMI);

    ssir_build_barrier(p->mod, p->func_id, p->block_id, scope);
}

/* ============================================================================
 * Instruction Parsing - Control Flow
 * ============================================================================ */

static void pp_parse_bra(PtxParser *p) {
    /* bra[.uni] label; */
    if (pp_check_dot(p, ".uni")) pp_next(p);

    char label[80];
    pp_read_ident(p, label, sizeof(label));
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    uint32_t target = pp_get_or_create_label(p, label);

    if (p->has_pred) {
        /* Predicated branch: @%p bra LABEL */
        uint32_t pred_val = ssir_build_load(p->mod, p->func_id, p->block_id,
                                            p->pred_val_type, p->pred_reg_ptr);
        if (p->pred_negated) {
            pred_val = ssir_build_not(p->mod, p->func_id, p->block_id,
                                      ssir_type_bool(p->mod), pred_val);
        }
        uint32_t fallthrough = ssir_block_create(p->mod, p->func_id, NULL);
        ssir_build_branch_cond(p->mod, p->func_id, p->block_id,
                               pred_val, target, fallthrough);
        p->block_id = fallthrough;
        p->has_pred = false;
    } else {
        ssir_build_branch(p->mod, p->func_id, p->block_id, target);
        /* Create a new block for unreachable code after unconditional branch */
        p->block_id = ssir_block_create(p->mod, p->func_id, NULL);
    }
}

static void pp_parse_ret(PtxParser *p) {
    pp_eat(p, PTK_SEMI);
    ssir_build_return_void(p->mod, p->func_id, p->block_id);
    p->block_id = ssir_block_create(p->mod, p->func_id, NULL);
}

static void pp_parse_exit(PtxParser *p) {
    pp_eat(p, PTK_SEMI);
    ssir_build_return_void(p->mod, p->func_id, p->block_id);
    p->block_id = ssir_block_create(p->mod, p->func_id, NULL);
}

static void pp_parse_call(PtxParser *p) {
    /* call (retval), func_name, (arg1, arg2, ...); */
    /* call func_name, (arg1, arg2, ...); */

    /* Check for return value */
    char ret_reg[80] = {0};
    if (pp_eat(p, PTK_LPAREN)) {
        pp_read_ident(p, ret_reg, sizeof(ret_reg));
        pp_expect(p, PTK_RPAREN, ")");
        pp_expect(p, PTK_COMMA, ",");
    }

    /* Function name */
    char fname[80];
    pp_read_ident(p, fname, sizeof(fname));

    /* Arguments */
    uint32_t args[16];
    int arg_count = 0;

    pp_expect(p, PTK_COMMA, ",");
    pp_expect(p, PTK_LPAREN, "(");
    while (!pp_check(p, PTK_RPAREN) && !pp_check(p, PTK_EOF)) {
        if (arg_count > 0) pp_expect(p, PTK_COMMA, ",");
        args[arg_count] = pp_parse_operand(p, 0);
        arg_count++;
    }
    pp_expect(p, PTK_RPAREN, ")");
    pp_eat(p, PTK_SEMI);

    if (p->had_error) return;

    /* Look up function */
    uint32_t callee = 0;
    uint32_t ret_type = ssir_type_void(p->mod);
    for (int i = 0; i < p->func_count; i++) {
        if (strcmp(p->funcs[i].name, fname) == 0) {
            callee = p->funcs[i].func_id;
            ret_type = p->funcs[i].ret_type;
            break;
        }
    }

    if (!callee) {
        pp_error(p, "undefined function '%s'", fname);
        return;
    }

    uint32_t result = ssir_build_call(p->mod, p->func_id, p->block_id,
                                      ret_type, callee, args, arg_count);

    if (ret_reg[0] && result) {
        pp_store_reg(p, ret_reg, result);
    }
}

/* ============================================================================
 * Instruction Dispatch
 * ============================================================================ */

static void pp_parse_instruction(PtxParser *p) {
    if (p->had_error) return;

    /* Check for predicate guard: @%p or @!%p */
    p->has_pred = false;
    if (pp_eat(p, PTK_AT)) {
        p->pred_negated = pp_eat(p, PTK_BANG);
        char pred_name[80];
        pp_read_ident(p, pred_name, sizeof(pred_name));

        PtxReg *pr = pp_find_reg(p, pred_name);
        if (pr) {
            p->pred_reg_ptr = pr->ptr_id;
            p->pred_val_type = pr->val_type;
            p->has_pred = true;
        } else {
            pp_error(p, "undefined predicate register '%s'", pred_name);
            return;
        }
    }

    /* Label check: IDENT followed by : */
    if (pp_check(p, PTK_IDENT)) {
        /* Lookahead for colon */
        PtxLexer save = p->lex;
        PtxToken save_tok = p->cur;
        char name[80];
        tok_to_str(&p->cur, name, sizeof(name));
        pp_next(p);
        if (pp_check(p, PTK_COLON)) {
            pp_next(p); /* consume : */
            /* Terminate current block, switch to label's block */
            uint32_t lbl_block = pp_get_or_create_label(p, name);
            ssir_build_branch(p->mod, p->func_id, p->block_id, lbl_block);
            p->block_id = lbl_block;
            return;
        }
        /* Not a label — restore and parse as instruction */
        p->lex = save;
        p->cur = save_tok;
    }

    /* Instruction opcode */
    char op[80];
    tok_to_str(&p->cur, op, sizeof(op));

    if (pp_check(p, PTK_IDENT)) {
        pp_next(p);

        /* Dispatch by opcode name */
        if (strcmp(op, "add") == 0) { pp_parse_arith(p, "add"); }
        else if (strcmp(op, "sub") == 0) { pp_parse_arith(p, "sub"); }
        else if (strcmp(op, "mul") == 0) { pp_parse_arith(p, "mul"); }
        else if (strcmp(op, "div") == 0) { pp_parse_arith(p, "div"); }
        else if (strcmp(op, "rem") == 0) { pp_parse_arith(p, "rem"); }
        else if (strcmp(op, "neg") == 0) { pp_parse_unary_arith(p, "neg"); }
        else if (strcmp(op, "abs") == 0) { pp_parse_unary_arith(p, "abs"); }
        else if (strcmp(op, "not") == 0) { pp_parse_unary_arith(p, "not"); }
        else if (strcmp(op, "cnot") == 0) { pp_parse_unary_arith(p, "cnot"); }
        else if (strcmp(op, "min") == 0) { pp_parse_minmax(p, "min"); }
        else if (strcmp(op, "max") == 0) { pp_parse_minmax(p, "max"); }
        else if (strcmp(op, "mad") == 0) { pp_parse_mad(p); }
        else if (strcmp(op, "fma") == 0) { pp_parse_fma(p); }
        else if (strcmp(op, "and") == 0) { pp_parse_bitwise(p, "and"); }
        else if (strcmp(op, "or") == 0)  { pp_parse_bitwise(p, "or"); }
        else if (strcmp(op, "xor") == 0) { pp_parse_bitwise(p, "xor"); }
        else if (strcmp(op, "shl") == 0) { pp_parse_shift(p, "shl"); }
        else if (strcmp(op, "shr") == 0) { pp_parse_shift(p, "shr"); }
        else if (strcmp(op, "setp") == 0) { pp_parse_setp(p); }
        else if (strcmp(op, "selp") == 0) { pp_parse_selp(p); }
        else if (strcmp(op, "set") == 0)  { pp_parse_setp(p); } /* similar to setp */
        else if (strcmp(op, "mov") == 0)  { pp_parse_mov(p); }
        else if (strcmp(op, "ld") == 0)   { pp_parse_ld(p); }
        else if (strcmp(op, "st") == 0)   { pp_parse_st(p); }
        else if (strcmp(op, "cvt") == 0)  { pp_parse_cvt(p); }
        else if (strcmp(op, "cvta") == 0) { pp_parse_cvta(p); }
        else if (strcmp(op, "bra") == 0)  { pp_parse_bra(p); }
        else if (strcmp(op, "ret") == 0)  { pp_parse_ret(p); }
        else if (strcmp(op, "exit") == 0) { pp_parse_exit(p); }
        else if (strcmp(op, "call") == 0) { pp_parse_call(p); }
        else if (strcmp(op, "bar") == 0)  { pp_parse_bar(p); }
        else if (strcmp(op, "membar") == 0) { pp_parse_membar(p); }
        else if (strcmp(op, "atom") == 0) { pp_parse_atom(p); }
        else if (strcmp(op, "rcp") == 0)  { pp_parse_math_unary(p, "rcp"); }
        else if (strcmp(op, "sqrt") == 0) { pp_parse_math_unary(p, "sqrt"); }
        else if (strcmp(op, "rsqrt") == 0) { pp_parse_math_unary(p, "rsqrt"); }
        else if (strcmp(op, "sin") == 0)  { pp_parse_math_unary(p, "sin"); }
        else if (strcmp(op, "cos") == 0)  { pp_parse_math_unary(p, "cos"); }
        else if (strcmp(op, "lg2") == 0)  { pp_parse_math_unary(p, "lg2"); }
        else if (strcmp(op, "ex2") == 0)  { pp_parse_math_unary(p, "ex2"); }
        else {
            /* Unknown instruction — skip to semicolon */
            while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF) &&
                   !pp_check(p, PTK_RBRACE))
                pp_next(p);
            pp_eat(p, PTK_SEMI);
        }
    } else if (pp_check(p, PTK_DOT_TOKEN)) {
        /* Directives inside function body */
        if (pp_check_dot(p, ".reg")) {
            pp_parse_reg_decl(p);
        } else if (pp_check_dot(p, ".local")) {
            /* Skip .local declarations for now */
            while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF)) pp_next(p);
            pp_eat(p, PTK_SEMI);
        } else if (pp_check_dot(p, ".shared")) {
            pp_parse_global_decl(p, ".shared");
        } else if (pp_check_dot(p, ".pragma")) {
            while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF)) pp_next(p);
            pp_eat(p, PTK_SEMI);
        } else {
            /* Skip unknown directives */
            while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF)) pp_next(p);
            pp_eat(p, PTK_SEMI);
        }
    }
}

/* ============================================================================
 * Function Body Parsing
 * ============================================================================ */

static void pp_parse_function_body(PtxParser *p) {
    pp_expect(p, PTK_LBRACE, "{");

    while (!pp_check(p, PTK_RBRACE) && !pp_check(p, PTK_EOF) && !p->had_error) {
        pp_parse_instruction(p);
    }

    /* Ensure the last block has a terminator */
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

    pp_expect(p, PTK_RBRACE, "}");
}

/* ============================================================================
 * Kernel/Function Parsing
 * ============================================================================ */

static void pp_register_func(PtxParser *p, const char *name,
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

static void pp_parse_param_list(PtxParser *p) {
    /* Parse kernel parameters: .param .type name [, ...] */
    if (!pp_eat(p, PTK_LPAREN)) return;

    while (!pp_check(p, PTK_RPAREN) && !pp_check(p, PTK_EOF)) {
        /* .param or .reg prefix for parameters */
        if (pp_check_dot(p, ".param")) { pp_next(p); }
        else if (pp_check_dot(p, ".reg")) { pp_next(p); }
        else break;

        /* Optional .align */
        if (pp_check_dot(p, ".align")) {
            pp_next(p);
            if (pp_check(p, PTK_INT_LIT)) pp_next(p);
        }

        uint32_t param_type = pp_ptx_type(p);
        if (!param_type) { pp_error(p, "expected type in param"); return; }

        char param_name[80];
        pp_read_ident(p, param_name, sizeof(param_name));

        /* Optional array size for .param .align N .b8 name[size] */
        if (pp_eat(p, PTK_LBRACKET)) {
            if (pp_check(p, PTK_INT_LIT)) pp_next(p);
            pp_expect(p, PTK_RBRACKET, "]");
        }

        if (p->is_entry) {
            /* Kernel param: create as local variable in function.
             * Entry points must NOT have function parameters (SPIR-V
             * requires void() signature for entry points).
             * For pointer params (.u64/.i64), buffer creation is DEFERRED
             * until the first ld.global/st.global, so we know the element
             * type for the runtime array. */
            SsirType *ty = ssir_get_type(p->mod, param_type);
            bool is_pointer = ty && (ty->kind == SSIR_TYPE_U64 || ty->kind == SSIR_TYPE_I64);

            /* Create local variable for ld.param access */
            uint32_t loc_ptr_type = ssir_type_ptr(p->mod, param_type, SSIR_ADDR_FUNCTION);
            uint32_t loc_id = ssir_function_add_local(p->mod, p->func_id,
                param_name, loc_ptr_type);

            /* Initialize to zero (base address = start of buffer, byte
             * offsets computed by PTX address arithmetic). */
            uint32_t init_val = pp_const_for_type(p, param_type, 0, 0.0, false);
            ssir_build_store(p->mod, p->func_id, p->block_id, loc_id, init_val);

            /* Register it */
            if (p->reg_count >= p->reg_cap) {
                p->reg_cap = p->reg_cap ? p->reg_cap * 2 : 64;
                p->regs = (PtxReg *)PTX_REALLOC(p->regs, p->reg_cap * sizeof(PtxReg));
            }
            PtxReg *r = &p->regs[p->reg_count++];
            memset(r, 0, sizeof(*r));
            snprintf(r->name, sizeof(r->name), "%s", param_name);
            r->val_type = param_type;
            r->ptr_type = loc_ptr_type;
            r->ptr_id = loc_id;
            r->global_id = 0;
            r->pending_binding = is_pointer ? p->next_binding++ : UINT32_MAX;
        } else {
            /* Device function param: add as function parameter */
            ssir_function_add_param(p->mod, p->func_id, param_name, param_type);
            /* Also add as local for register access */
            pp_add_reg(p, param_name, param_type, false);
        }

        if (!pp_eat(p, PTK_COMMA)) break;
    }
    pp_expect(p, PTK_RPAREN, ")");
}

static void pp_parse_entry(PtxParser *p) {
    /* [.visible] .entry name(params) { body } */
    pp_next(p); /* skip .entry */

    char name[80];
    pp_read_ident(p, name, sizeof(name));

    /* Reset per-function state */
    p->reg_count = 0;
    p->label_count = 0;
    p->unresolved_count = 0;
    p->is_entry = true;
    p->next_binding = 0;
    p->wg_size[0] = p->wg_size[1] = p->wg_size[2] = 0;

    uint32_t void_t = ssir_type_void(p->mod);
    p->func_id = ssir_function_create(p->mod, name, void_t);
    p->block_id = ssir_block_create(p->mod, p->func_id, "entry");

    pp_register_func(p, name, p->func_id, void_t);

    pp_parse_param_list(p);

    /* Performance directives before body */
    while (pp_check_dot(p, ".maxntid") || pp_check_dot(p, ".reqntid") ||
           pp_check_dot(p, ".minnctapersm") || pp_check_dot(p, ".maxnreg") ||
           pp_check_dot(p, ".pragma") || pp_check_dot(p, ".noreturn")) {
        if (pp_check_dot(p, ".maxntid") || pp_check_dot(p, ".reqntid")) {
            pp_next(p);
            if (pp_check(p, PTK_INT_LIT)) { p->wg_size[0] = (uint32_t)p->cur.int_val; pp_next(p); }
            if (pp_eat(p, PTK_COMMA) && pp_check(p, PTK_INT_LIT)) { p->wg_size[1] = (uint32_t)p->cur.int_val; pp_next(p); }
            if (pp_eat(p, PTK_COMMA) && pp_check(p, PTK_INT_LIT)) { p->wg_size[2] = (uint32_t)p->cur.int_val; pp_next(p); }
        } else {
            pp_next(p);
            while (!pp_check(p, PTK_LBRACE) && !pp_check(p, PTK_DOT_TOKEN) &&
                   !pp_check(p, PTK_EOF))
                pp_next(p);
        }
    }

    pp_parse_function_body(p);

    /* Eagerly materialize any remaining unmaterialized buffer params.
     * This ensures storage buffer globals exist even if the kernel body
     * doesn't contain ld.global/st.global for every param. Default to u32
     * element type when no typed access was observed. */
    for (int i = 0; i < p->reg_count; i++) {
        PtxReg *r = &p->regs[i];
        if (r->pending_binding != UINT32_MAX && r->global_id == 0) {
            uint32_t default_elem = ssir_type_u32(p->mod);
            pp_materialize_buffer(p, r, default_elem);
        }
    }

    /* Create entry point */
    p->ep_index = ssir_entry_point_create(p->mod, SSIR_STAGE_COMPUTE, p->func_id, name);
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

static void pp_parse_func(PtxParser *p) {
    /* [.visible] .func [(ret_type retval)] name(params) { body } */
    pp_next(p); /* skip .func */

    /* Reset per-function state */
    p->reg_count = 0;
    p->label_count = 0;
    p->unresolved_count = 0;
    p->is_entry = false;

    /* Optional return value */
    uint32_t ret_type = ssir_type_void(p->mod);
    if (pp_eat(p, PTK_LPAREN)) {
        /* (.reg .type retval) */
        if (pp_check_dot(p, ".reg")) pp_next(p);
        ret_type = pp_ptx_type(p);
        if (!ret_type) ret_type = ssir_type_void(p->mod);
        char retname[80];
        pp_read_ident(p, retname, sizeof(retname));
        pp_expect(p, PTK_RPAREN, ")");
    }

    char name[80];
    pp_read_ident(p, name, sizeof(name));

    p->func_id = ssir_function_create(p->mod, name, ret_type);
    p->block_id = ssir_block_create(p->mod, p->func_id, "entry");

    pp_register_func(p, name, p->func_id, ret_type);

    pp_parse_param_list(p);

    /* Only parse body if present (could be just a declaration) */
    if (pp_check(p, PTK_LBRACE)) {
        pp_parse_function_body(p);
    } else {
        /* Declaration only - add a return */
        ssir_build_return_void(p->mod, p->func_id, p->block_id);
        pp_eat(p, PTK_SEMI);
    }
}

/* ============================================================================
 * Top-Level Parsing
 * ============================================================================ */

static void pp_parse_toplevel(PtxParser *p) {
    while (!pp_check(p, PTK_EOF) && !p->had_error) {
        if (pp_check_dot(p, ".version")) {
            pp_parse_version(p);
        } else if (pp_check_dot(p, ".target")) {
            pp_parse_target(p);
        } else if (pp_check_dot(p, ".address_size")) {
            pp_parse_address_size(p);
        } else if (pp_check_dot(p, ".visible") || pp_check_dot(p, ".weak")) {
            pp_next(p); /* skip linkage */
            if (pp_check_dot(p, ".entry")) pp_parse_entry(p);
            else if (pp_check_dot(p, ".func")) pp_parse_func(p);
            else if (pp_check_dot(p, ".global")) pp_parse_global_decl(p, ".global");
            else if (pp_check_dot(p, ".const")) pp_parse_global_decl(p, ".const");
            else if (pp_check_dot(p, ".shared")) pp_parse_global_decl(p, ".shared");
            else {
                while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF)) pp_next(p);
                pp_eat(p, PTK_SEMI);
            }
        } else if (pp_check_dot(p, ".extern")) {
            pp_next(p);
            /* skip extern declarations */
            while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF) &&
                   !pp_check(p, PTK_LBRACE)) pp_next(p);
            if (pp_check(p, PTK_LBRACE)) {
                int depth = 1;
                pp_next(p);
                while (depth > 0 && !pp_check(p, PTK_EOF)) {
                    if (pp_check(p, PTK_LBRACE)) depth++;
                    if (pp_check(p, PTK_RBRACE)) depth--;
                    pp_next(p);
                }
            }
            pp_eat(p, PTK_SEMI);
        } else if (pp_check_dot(p, ".entry")) {
            pp_parse_entry(p);
        } else if (pp_check_dot(p, ".func")) {
            pp_parse_func(p);
        } else if (pp_check_dot(p, ".global")) {
            pp_parse_global_decl(p, ".global");
        } else if (pp_check_dot(p, ".shared")) {
            pp_parse_global_decl(p, ".shared");
        } else if (pp_check_dot(p, ".const")) {
            pp_parse_global_decl(p, ".const");
        } else {
            /* Skip unknown top-level tokens */
            pp_next(p);
        }
    }
}

/* ============================================================================
 * Cleanup
 * ============================================================================ */

static void pp_cleanup(PtxParser *p) {
    PTX_FREE(p->regs);
    PTX_FREE(p->labels);
    PTX_FREE(p->unresolved);
    PTX_FREE(p->iface);
    PTX_FREE(p->funcs);
}

/* ============================================================================
 * Public API
 * ============================================================================ */

PtxToSsirResult ptx_to_ssir(const char *ptx_source, const PtxToSsirOptions *opts,
    SsirModule **out_module, char **out_error) {
    if (!ptx_source || !out_module) {
        if (out_error) *out_error = ptx_strdup("Invalid input: null source or output pointer");
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    PtxParser parser;
    memset(&parser, 0, sizeof(parser));
    plx_init(&parser.lex, ptx_source);
    if (opts) parser.opts = *opts;

    parser.mod = ssir_module_create();
    if (!parser.mod) {
        if (out_error) *out_error = ptx_strdup("Out of memory");
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    /* Read first token */
    pp_next(&parser);

    /* Parse */
    pp_parse_toplevel(&parser);

    if (parser.had_error) {
        if (out_error) *out_error = ptx_strdup(parser.error);
        ssir_module_destroy(parser.mod);
        pp_cleanup(&parser);
        return PTX_TO_SSIR_PARSE_ERROR;
    }

    *out_module = parser.mod;
    pp_cleanup(&parser);
    return PTX_TO_SSIR_OK;
}

void ptx_to_ssir_free(char *str) {
    PTX_FREE(str);
}

const char *ptx_to_ssir_result_string(PtxToSsirResult r) {
    switch (r) {
        case PTX_TO_SSIR_OK: return "Success";
        case PTX_TO_SSIR_PARSE_ERROR: return "Parse error";
        case PTX_TO_SSIR_UNSUPPORTED: return "Unsupported feature";
        default: return "Unknown error";
    }
}
