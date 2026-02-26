/*
 * ptx_parser.c - PTX Parser -> AST (PtxModule)
 *
 * Parses PTX assembly source and produces a PtxModule AST.
 * The lowerer (ptx_lower.c) then converts the AST to SSIR.
 */

#include "simple_wgsl.h"
#include "ptx_ast.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>

#ifndef PTX_MALLOC
#define PTX_MALLOC(sz) calloc(1, (sz))
#endif
#ifndef PTX_REALLOC
#define PTX_REALLOC(p, sz) realloc((p), (sz))
#endif
#ifndef PTX_FREE
#define PTX_FREE(p) free((p))
#endif

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

/* ===== Token Types ===== */

typedef enum {
  PTK_EOF = 0,
  PTK_SEMI, PTK_COMMA, PTK_LBRACE, PTK_RBRACE,
  PTK_LBRACKET, PTK_RBRACKET, PTK_LPAREN, PTK_RPAREN,
  PTK_COLON, PTK_PIPE, PTK_AT, PTK_BANG,
  PTK_PLUS, PTK_MINUS, PTK_STAR,
  PTK_LANGLE, PTK_RANGLE, PTK_EQ,
  PTK_IDENT, PTK_INT_LIT, PTK_FLOAT_LIT,
  PTK_DOT_TOKEN,
} PtxTokType;

typedef struct {
  PtxTokType type;
  const char *start;
  int length;
  int line, col;
  uint64_t int_val;
  double float_val;
} PtxToken;

/* ===== Lexer ===== */

typedef struct {
  const char *src;
  size_t pos;
  int line, col;
} PtxLexer;

static void plx_init(PtxLexer *L, const char *src) {
  L->src = src; L->pos = 0; L->line = 1; L->col = 1;
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
      while (plx_peek(L) && plx_peek(L) != '\n') plx_advance(L);
    } else if (c == '/' && plx_peek2(L) == '*') {
      plx_advance(L); plx_advance(L);
      while (plx_peek(L)) {
        if (plx_peek(L) == '*' && plx_peek2(L) == '/') {
          plx_advance(L); plx_advance(L); break;
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

  if (c == '.') {
    plx_advance(L);
    while (isalnum((unsigned char)plx_peek(L)) || plx_peek(L) == '_')
      plx_advance(L);
    t.type = PTK_DOT_TOKEN;
    t.length = (int)(&L->src[L->pos] - t.start);
    return t;
  }

  if (plx_is_ident_start(c)) {
    bool is_reg = (c == '%');
    plx_advance(L);
    while (plx_is_ident_char(plx_peek(L)) ||
           (is_reg && plx_peek(L) == '.'))
      plx_advance(L);
    t.type = PTK_IDENT;
    t.length = (int)(&L->src[L->pos] - t.start);
    return t;
  }

  if (isdigit((unsigned char)c) ||
      (c == '-' && isdigit((unsigned char)plx_peek2(L)))) {
    bool negative = false;
    if (c == '-') { negative = true; plx_advance(L); c = plx_peek(L); }

    if (c == '0' && (plx_peek2(L) == 'f' || plx_peek2(L) == 'F' ||
                     plx_peek2(L) == 'd' || plx_peek2(L) == 'D')) {
      bool is_double = (plx_peek2(L) == 'd' || plx_peek2(L) == 'D');
      plx_advance(L); plx_advance(L);
      uint64_t hex = 0;
      while (isxdigit((unsigned char)plx_peek(L))) {
        char h = plx_peek(L);
        int d = (h >= 'a') ? h - 'a' + 10 :
                (h >= 'A') ? h - 'A' + 10 : h - '0';
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

    if (c == '0' && (plx_peek2(L) == 'x' || plx_peek2(L) == 'X')) {
      plx_advance(L); plx_advance(L);
      uint64_t val = 0;
      while (isxdigit((unsigned char)plx_peek(L))) {
        char h = plx_peek(L);
        int d = (h >= 'a') ? h - 'a' + 10 :
                (h >= 'A') ? h - 'A' + 10 : h - '0';
        val = (val << 4) | d;
        plx_advance(L);
      }
      if (plx_peek(L) == 'U' || plx_peek(L) == 'u') plx_advance(L);
      t.type = PTK_INT_LIT;
      t.int_val = negative ? (uint64_t)(-(int64_t)val) : val;
      t.length = (int)(&L->src[L->pos] - t.start);
      return t;
    }

    bool is_float = false;
    while (isdigit((unsigned char)plx_peek(L))) plx_advance(L);
    if (plx_peek(L) == '.' &&
        isdigit((unsigned char)plx_peek2(L))) {
      is_float = true;
      plx_advance(L);
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

  if (c == '"') {
    plx_advance(L);
    while (plx_peek(L) != '"' && plx_peek(L) != '\0')
      plx_advance(L);
    if (plx_peek(L) == '"') plx_advance(L);
    t.type = PTK_IDENT;
    t.length = (int)(&L->src[L->pos] - t.start);
    return t;
  }

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
    case '=': t.type = PTK_EQ; break;
    default:  t.type = PTK_EOF; break;
  }
  return t;
}

static bool dot_eq(const PtxToken *t, const char *s) {
  if (t->type != PTK_DOT_TOKEN) return false;
  return t->length == (int)strlen(s) &&
         strncmp(t->start, s, t->length) == 0;
}

static void tok_to_str(const PtxToken *t, char *buf, int bufsz) {
  int len = t->length < bufsz - 1 ? t->length : bufsz - 1;
  memcpy(buf, t->start, len);
  buf[len] = '\0';
}

/* AST types are defined in ptx_ast.h */

/* ===== Parser Context ===== */

typedef struct {
  PtxLexer lex;
  PtxToken cur;
  PtxModule *mod;
  PtxEntry *cur_entry;
  PtxFunc *cur_func;
  char pending_pred[80];
  bool pending_pred_negated;
  bool pending_has_pred;
  char error[1024];
  int had_error;
} PtxParser;

/* ===== Parser Helpers ===== */

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

static void pp_next(PtxParser *p) { p->cur = plx_next(&p->lex); }

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
static void pp_read_ident(PtxParser *p, char *buf, int bufsz) {
  tok_to_str(&p->cur, buf, bufsz);
  pp_next(p);
}

/* ===== AST Builder Helpers ===== */

#define GROW(arr, cnt, cap, T) do { \
  if ((cnt) >= (cap)) { \
    (cap) = (cap) ? (cap) * 2 : 16; \
    (arr) = (T *)PTX_REALLOC((arr), (cap) * sizeof(T)); \
  } \
} while(0)

static void pp_add_stmt(PtxParser *p, const PtxStmt *s) {
  if (p->cur_entry) {
    PtxEntry *e = p->cur_entry;
    GROW(e->body, e->body_count, e->body_cap, PtxStmt);
    e->body[e->body_count++] = *s;
  } else if (p->cur_func) {
    PtxFunc *f = p->cur_func;
    GROW(f->body, f->body_count, f->body_cap, PtxStmt);
    f->body[f->body_count++] = *s;
  }
}

static void pp_emit_inst(PtxParser *p, PtxInst *inst) {
  if (p->pending_has_pred) {
    inst->has_pred = true;
    snprintf(inst->pred, sizeof(inst->pred), "%s",
             p->pending_pred);
    inst->pred_negated = p->pending_pred_negated;
    p->pending_has_pred = false;
  }
  PtxStmt stmt;
  memset(&stmt, 0, sizeof(stmt));
  stmt.kind = PTX_STMT_INST;
  stmt.inst = *inst;
  pp_add_stmt(p, &stmt);
}

static void pp_add_reg_decl(PtxParser *p, const PtxRegDecl *d) {
  if (p->cur_entry) {
    PtxEntry *e = p->cur_entry;
    GROW(e->reg_decls, e->reg_decl_count, e->reg_decl_cap,
         PtxRegDecl);
    e->reg_decls[e->reg_decl_count++] = *d;
  } else if (p->cur_func) {
    PtxFunc *f = p->cur_func;
    GROW(f->reg_decls, f->reg_decl_count, f->reg_decl_cap,
         PtxRegDecl);
    f->reg_decls[f->reg_decl_count++] = *d;
  }
}

static void pp_add_local_decl(PtxParser *p, const PtxLocalDecl *d) {
  if (p->cur_entry) {
    PtxEntry *e = p->cur_entry;
    GROW(e->local_decls, e->local_decl_count,
         e->local_decl_cap, PtxLocalDecl);
    e->local_decls[e->local_decl_count++] = *d;
  } else if (p->cur_func) {
    PtxFunc *f = p->cur_func;
    GROW(f->local_decls, f->local_decl_count,
         f->local_decl_cap, PtxLocalDecl);
    f->local_decls[f->local_decl_count++] = *d;
  }
}

static PtxEntry *pp_add_entry(PtxParser *p) {
  PtxModule *m = p->mod;
  GROW(m->entries, m->entry_count, m->entry_cap, PtxEntry);
  PtxEntry *e = &m->entries[m->entry_count++];
  memset(e, 0, sizeof(*e));
  return e;
}

static PtxFunc *pp_add_func(PtxParser *p) {
  PtxModule *m = p->mod;
  GROW(m->functions, m->func_count, m->func_cap, PtxFunc);
  PtxFunc *f = &m->functions[m->func_count++];
  memset(f, 0, sizeof(*f));
  return f;
}

static void pp_add_global(PtxParser *p, const PtxGlobalDecl *g) {
  PtxModule *m = p->mod;
  GROW(m->globals, m->global_count, m->global_cap, PtxGlobalDecl);
  m->globals[m->global_count++] = *g;
}

static void pp_add_ref(PtxParser *p, const PtxRefDecl *r) {
  PtxModule *m = p->mod;
  GROW(m->refs, m->ref_count, m->ref_cap, PtxRefDecl);
  m->refs[m->ref_count++] = *r;
}

/* ===== Type Parsing ===== */

static bool pp_parse_type(PtxParser *p, PtxDataType *out) {
  if (pp_check_dot(p, ".pred")) { pp_next(p); *out = PTX_TYPE_PRED; return true; }
  if (pp_check_dot(p, ".b8"))   { pp_next(p); *out = PTX_TYPE_B8;   return true; }
  if (pp_check_dot(p, ".b16"))  { pp_next(p); *out = PTX_TYPE_B16;  return true; }
  if (pp_check_dot(p, ".b32"))  { pp_next(p); *out = PTX_TYPE_B32;  return true; }
  if (pp_check_dot(p, ".b64"))  { pp_next(p); *out = PTX_TYPE_B64;  return true; }
  if (pp_check_dot(p, ".u8"))   { pp_next(p); *out = PTX_TYPE_U8;   return true; }
  if (pp_check_dot(p, ".u16"))  { pp_next(p); *out = PTX_TYPE_U16;  return true; }
  if (pp_check_dot(p, ".u32"))  { pp_next(p); *out = PTX_TYPE_U32;  return true; }
  if (pp_check_dot(p, ".u64"))  { pp_next(p); *out = PTX_TYPE_U64;  return true; }
  if (pp_check_dot(p, ".s8"))   { pp_next(p); *out = PTX_TYPE_S8;   return true; }
  if (pp_check_dot(p, ".s16"))  { pp_next(p); *out = PTX_TYPE_S16;  return true; }
  if (pp_check_dot(p, ".s32"))  { pp_next(p); *out = PTX_TYPE_S32;  return true; }
  if (pp_check_dot(p, ".s64"))  { pp_next(p); *out = PTX_TYPE_S64;  return true; }
  if (pp_check_dot(p, ".f16"))  { pp_next(p); *out = PTX_TYPE_F16;  return true; }
  if (pp_check_dot(p, ".f32"))  { pp_next(p); *out = PTX_TYPE_F32;  return true; }
  if (pp_check_dot(p, ".f64"))  { pp_next(p); *out = PTX_TYPE_F64;  return true; }
  return false;
}

/* ===== Modifier Parsing ===== */

static uint32_t pp_parse_modifiers(PtxParser *p) {
  uint32_t m = 0;
  for (;;) {
    if      (pp_check_dot(p, ".wide"))   { m |= PTX_MOD_WIDE;   pp_next(p); }
    else if (pp_check_dot(p, ".hi"))     { m |= PTX_MOD_HI;     pp_next(p); }
    else if (pp_check_dot(p, ".lo"))     { m |= PTX_MOD_LO;     pp_next(p); }
    else if (pp_check_dot(p, ".rn"))     { m |= PTX_MOD_RN;     pp_next(p); }
    else if (pp_check_dot(p, ".rz"))     { m |= PTX_MOD_RZ;     pp_next(p); }
    else if (pp_check_dot(p, ".rm"))     { m |= PTX_MOD_RM;     pp_next(p); }
    else if (pp_check_dot(p, ".rp"))     { m |= PTX_MOD_RP;     pp_next(p); }
    else if (pp_check_dot(p, ".rni"))    { m |= PTX_MOD_RNI;    pp_next(p); }
    else if (pp_check_dot(p, ".rzi"))    { m |= PTX_MOD_RZI;    pp_next(p); }
    else if (pp_check_dot(p, ".rmi"))    { m |= PTX_MOD_RMI;    pp_next(p); }
    else if (pp_check_dot(p, ".rpi"))    { m |= PTX_MOD_RPI;    pp_next(p); }
    else if (pp_check_dot(p, ".sat"))    { m |= PTX_MOD_SAT;    pp_next(p); }
    else if (pp_check_dot(p, ".approx")) { m |= PTX_MOD_APPROX; pp_next(p); }
    else if (pp_check_dot(p, ".ftz"))    { m |= PTX_MOD_FTZ;    pp_next(p); }
    else break;
  }
  return m;
}

/* ===== Operand Parsing ===== */

static PtxOperand pp_parse_operand(PtxParser *p) {
  PtxOperand op;
  memset(&op, 0, sizeof(op));
  if (pp_check(p, PTK_INT_LIT)) {
    op.kind = PTX_OPER_IMM_INT;
    op.ival = (int64_t)p->cur.int_val;
    pp_next(p);
    return op;
  }
  if (pp_check(p, PTK_FLOAT_LIT)) {
    op.kind = PTX_OPER_IMM_FLT;
    op.fval = p->cur.float_val;
    pp_next(p);
    return op;
  }
  if (pp_check(p, PTK_MINUS)) {
    pp_next(p);
    if (pp_check(p, PTK_INT_LIT)) {
      op.kind = PTX_OPER_IMM_INT;
      op.ival = -(int64_t)p->cur.int_val;
      pp_next(p);
      return op;
    }
    if (pp_check(p, PTK_FLOAT_LIT)) {
      op.kind = PTX_OPER_IMM_FLT;
      op.fval = -p->cur.float_val;
      pp_next(p);
      return op;
    }
    pp_error(p, "expected number after '-'");
    return op;
  }
  if (pp_check(p, PTK_IDENT)) {
    op.kind = PTX_OPER_REG;
    tok_to_str(&p->cur, op.name, sizeof(op.name));
    pp_next(p);
    return op;
  }
  pp_error(p, "expected operand");
  return op;
}

static PtxOperand pp_parse_address(PtxParser *p) {
  PtxOperand op;
  memset(&op, 0, sizeof(op));
  op.kind = PTX_OPER_ADDR;
  pp_expect(p, PTK_LBRACKET, "[");
  pp_read_ident(p, op.base, sizeof(op.base));

  if (pp_check(p, PTK_PLUS) || pp_check(p, PTK_MINUS)) {
    bool is_sub = pp_check(p, PTK_MINUS);
    pp_next(p);
    if (!is_sub && pp_check(p, PTK_MINUS)) {
      is_sub = true;
      pp_next(p);
    }
    if (pp_check(p, PTK_INT_LIT)) {
      op.offset = (int64_t)p->cur.int_val;
      if (is_sub) op.offset = -op.offset;
      pp_next(p);
    } else if (pp_check(p, PTK_IDENT)) {
      tok_to_str(&p->cur, op.name, sizeof(op.name));
      pp_next(p);
      op.negated = is_sub;
    }
  }
  pp_expect(p, PTK_RBRACKET, "]");
  return op;
}

static PtxCmpOp pp_parse_cmp_str(const char *s) {
  if (strcmp(s, "eq") == 0)  return PTX_CMP_EQ;
  if (strcmp(s, "ne") == 0)  return PTX_CMP_NE;
  if (strcmp(s, "lt") == 0)  return PTX_CMP_LT;
  if (strcmp(s, "le") == 0)  return PTX_CMP_LE;
  if (strcmp(s, "gt") == 0)  return PTX_CMP_GT;
  if (strcmp(s, "ge") == 0)  return PTX_CMP_GE;
  if (strcmp(s, "equ") == 0) return PTX_CMP_EQU;
  if (strcmp(s, "neu") == 0) return PTX_CMP_NEU;
  if (strcmp(s, "ltu") == 0) return PTX_CMP_LTU;
  if (strcmp(s, "leu") == 0) return PTX_CMP_LEU;
  if (strcmp(s, "gtu") == 0) return PTX_CMP_GTU;
  if (strcmp(s, "geu") == 0) return PTX_CMP_GEU;
  if (strcmp(s, "lo") == 0)  return PTX_CMP_LO;
  if (strcmp(s, "ls") == 0)  return PTX_CMP_LS;
  if (strcmp(s, "hi") == 0)  return PTX_CMP_HI;
  if (strcmp(s, "hs") == 0)  return PTX_CMP_HS;
  return PTX_CMP_EQ;
}

static PtxMemSpace pp_parse_mem_space(PtxParser *p) {
  if (pp_check_dot(p, ".global")) { pp_next(p); return PTX_SPACE_GLOBAL; }
  if (pp_check_dot(p, ".shared")) { pp_next(p); return PTX_SPACE_SHARED; }
  if (pp_check_dot(p, ".local"))  { pp_next(p); return PTX_SPACE_LOCAL; }
  if (pp_check_dot(p, ".const"))  { pp_next(p); return PTX_SPACE_CONST; }
  if (pp_check_dot(p, ".param"))  { pp_next(p); return PTX_SPACE_PARAM; }
  return PTX_SPACE_GLOBAL;
}

static void pp_skip_cache_ops(PtxParser *p) {
  while (pp_check_dot(p, ".ca") || pp_check_dot(p, ".cg") ||
         pp_check_dot(p, ".cs") || pp_check_dot(p, ".cv") ||
         pp_check_dot(p, ".wb") || pp_check_dot(p, ".wt") ||
         pp_check_dot(p, ".nc") || pp_check_dot(p, ".lu")) {
    pp_next(p);
  }
}

static PtxTexGeom pp_parse_tex_geom(PtxParser *p, int *ncoords) {
  PtxTexGeom g = PTX_TEX_2D; int nc = 2;
  if      (pp_check_dot(p, ".1d"))  { g = PTX_TEX_1D;   nc = 1; pp_next(p); }
  else if (pp_check_dot(p, ".2d"))  { g = PTX_TEX_2D;   nc = 2; pp_next(p); }
  else if (pp_check_dot(p, ".3d"))  { g = PTX_TEX_3D;   nc = 3; pp_next(p); }
  else if (pp_check_dot(p, ".a1d")) { g = PTX_TEX_A1D;  nc = 2; pp_next(p); }
  else if (pp_check_dot(p, ".a2d")) { g = PTX_TEX_A2D;  nc = 4; pp_next(p); }
  else if (pp_check_dot(p, ".cube")){ g = PTX_TEX_CUBE;  nc = 4; pp_next(p); }
  if (ncoords) *ncoords = nc;
  return g;
}

/* ===== Instruction Parsers ===== */

static void pp_parse_arith(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  inst.src_count = 2;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_unary(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_mad(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_MAD;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for mad"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  inst.src_count = 3;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_fma(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_FMA;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for fma"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  inst.src_count = 3;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_setp(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.cmp_op = PTX_CMP_NONE;
  if (pp_check(p, PTK_DOT_TOKEN)) {
    char cmp[16];
    tok_to_str(&p->cur, cmp, sizeof(cmp));
    inst.cmp_op = pp_parse_cmp_str(cmp + 1);
    pp_next(p);
  }
  if (pp_check_dot(p, ".and") || pp_check_dot(p, ".or") ||
      pp_check_dot(p, ".xor")) {
    tok_to_str(&p->cur, inst.combine, sizeof(inst.combine));
    memmove(inst.combine, inst.combine + 1, strlen(inst.combine));
    inst.has_combine = true;
    pp_next(p);
  }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for setp"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  if (pp_eat(p, PTK_PIPE)) {
    inst.dst2 = pp_parse_operand(p);
    inst.has_dst2 = true;
  }
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  inst.src_count = 2;
  if (inst.has_combine && pp_eat(p, PTK_COMMA)) {
    inst.src[2] = pp_parse_operand(p);
    inst.src_count = 3;
  }
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_selp(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_SELP;
  inst.line = p->cur.line; inst.col = p->cur.col;
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for selp"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  inst.src_count = 3;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_mov(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_MOV;
  inst.line = p->cur.line; inst.col = p->cur.col;
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for mov"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_ld(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_LD;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.space = pp_parse_mem_space(p);
  pp_skip_cache_ops(p);
  inst.vec_width = 1;
  if (pp_check_dot(p, ".v2")) { inst.vec_width = 2; pp_next(p); }
  else if (pp_check_dot(p, ".v4")) { inst.vec_width = 4; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for ld"); return;
  }
  inst.type = type;
  if (inst.vec_width > 1) {
    inst.dst.kind = PTX_OPER_VEC;
    pp_expect(p, PTK_LBRACE, "{");
    int n = 0;
    while (n < inst.vec_width) {
      pp_read_ident(p, inst.dst.regs[n], sizeof(inst.dst.regs[n]));
      n++;
      if (n < inst.vec_width) pp_expect(p, PTK_COMMA, ",");
    }
    inst.dst.vec_count = n;
    pp_expect(p, PTK_RBRACE, "}");
    pp_expect(p, PTK_COMMA, ",");
  } else {
    inst.dst = pp_parse_operand(p);
    pp_expect(p, PTK_COMMA, ",");
  }
  inst.src[0] = pp_parse_address(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_st(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_ST;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.space = pp_parse_mem_space(p);
  pp_skip_cache_ops(p);
  inst.vec_width = 1;
  if (pp_check_dot(p, ".v2")) { inst.vec_width = 2; pp_next(p); }
  else if (pp_check_dot(p, ".v4")) { inst.vec_width = 4; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for st"); return;
  }
  inst.type = type;
  if (inst.vec_width > 1) {
    inst.src[0] = pp_parse_address(p);
    pp_expect(p, PTK_COMMA, ",");
    PtxOperand vec_op;
    memset(&vec_op, 0, sizeof(vec_op));
    vec_op.kind = PTX_OPER_VEC;
    pp_expect(p, PTK_LBRACE, "{");
    int n = 0;
    while (n < inst.vec_width) {
      pp_read_ident(p, vec_op.regs[n], sizeof(vec_op.regs[n]));
      n++;
      if (n < inst.vec_width) pp_expect(p, PTK_COMMA, ",");
    }
    vec_op.vec_count = n;
    pp_expect(p, PTK_RBRACE, "}");
    inst.src[1] = vec_op;
    inst.src_count = 2;
  } else {
    inst.src[0] = pp_parse_address(p);
    pp_expect(p, PTK_COMMA, ",");
    inst.src[1] = pp_parse_operand(p);
    inst.src_count = 2;
  }
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_cvt(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_CVT;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType dst_type, src_type;
  if (!pp_parse_type(p, &dst_type)) {
    pp_error(p, "expected dest type for cvt"); return;
  }
  if (!pp_parse_type(p, &src_type)) {
    pp_error(p, "expected src type for cvt"); return;
  }
  inst.type = dst_type;
  inst.type2 = src_type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_cvta(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_CVTA;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".to")) { inst.modifiers |= PTX_MOD_TO; pp_next(p); }
  inst.space = pp_parse_mem_space(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for cvta"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_bra(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_BRA;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".uni")) { inst.modifiers |= PTX_MOD_UNI; pp_next(p); }
  inst.src[0].kind = PTX_OPER_LABEL;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_ret(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_RET;
  inst.line = p->cur.line; inst.col = p->cur.col;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_exit(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_EXIT;
  inst.line = p->cur.line; inst.col = p->cur.col;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_call(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_CALL;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_eat(p, PTK_LPAREN)) {
    inst.dst = pp_parse_operand(p);
    pp_expect(p, PTK_RPAREN, ")");
    pp_expect(p, PTK_COMMA, ",");
  }
  inst.src[0].kind = PTX_OPER_LABEL;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LPAREN, "(");
  while (!pp_check(p, PTK_RPAREN) && !pp_check(p, PTK_EOF)) {
    if (inst.src_count > 1) pp_expect(p, PTK_COMMA, ",");
    if (inst.src_count < 8) {
      inst.src[inst.src_count] = pp_parse_operand(p);
      inst.src_count++;
    } else {
      pp_parse_operand(p);
    }
  }
  pp_expect(p, PTK_RPAREN, ")");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_bar(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_BAR;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".sync")) pp_next(p);
  else if (pp_check_dot(p, ".arrive")) pp_next(p);
  else if (pp_check_dot(p, ".red")) pp_next(p);
  if (pp_check(p, PTK_INT_LIT)) {
    inst.bar_id = (int)p->cur.int_val;
    pp_next(p);
  }
  if (pp_eat(p, PTK_COMMA)) {
    if (pp_check(p, PTK_INT_LIT)) pp_next(p);
    else if (pp_check(p, PTK_IDENT)) pp_next(p);
  }
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_membar(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_MEMBAR;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.membar_scope = PTX_MEMBAR_CTA;
  if (pp_check_dot(p, ".cta")) { inst.membar_scope = PTX_MEMBAR_CTA; pp_next(p); }
  else if (pp_check_dot(p, ".gl")) { inst.membar_scope = PTX_MEMBAR_GL; pp_next(p); }
  else if (pp_check_dot(p, ".sys")) { inst.membar_scope = PTX_MEMBAR_SYS; pp_next(p); }
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_atom(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_ATOM;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.space = pp_parse_mem_space(p);
  bool is_cas = false;
  if      (pp_check_dot(p, ".add"))  { inst.atomic_op = PTX_ATOMIC_ADD;  pp_next(p); }
  else if (pp_check_dot(p, ".min"))  { inst.atomic_op = PTX_ATOMIC_MIN;  pp_next(p); }
  else if (pp_check_dot(p, ".max"))  { inst.atomic_op = PTX_ATOMIC_MAX;  pp_next(p); }
  else if (pp_check_dot(p, ".and"))  { inst.atomic_op = PTX_ATOMIC_AND;  pp_next(p); }
  else if (pp_check_dot(p, ".or"))   { inst.atomic_op = PTX_ATOMIC_OR;   pp_next(p); }
  else if (pp_check_dot(p, ".xor"))  { inst.atomic_op = PTX_ATOMIC_XOR;  pp_next(p); }
  else if (pp_check_dot(p, ".exch")) { inst.atomic_op = PTX_ATOMIC_EXCH; pp_next(p); }
  else if (pp_check_dot(p, ".cas"))  { inst.atomic_op = PTX_ATOMIC_CAS;  is_cas = true; pp_next(p); }
  else if (pp_check_dot(p, ".inc"))  { inst.atomic_op = PTX_ATOMIC_INC;  pp_next(p); }
  else if (pp_check_dot(p, ".dec"))  { inst.atomic_op = PTX_ATOMIC_DEC;  pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for atom"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_address(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  inst.src_count = 2;
  if (is_cas) {
    pp_expect(p, PTK_COMMA, ",");
    inst.src[2] = pp_parse_operand(p);
    inst.src_count = 3;
  }
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_math_unary(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.modifiers = pp_parse_modifiers(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_tex(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_TEX;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.mip_mode = PTX_MIP_NONE;
  if (pp_check_dot(p, ".level")) { inst.mip_mode = PTX_MIP_LEVEL; pp_next(p); }
  else if (pp_check_dot(p, ".grad")) { inst.mip_mode = PTX_MIP_GRAD; pp_next(p); }
  int coord_count = 2;
  inst.tex_geom = pp_parse_tex_geom(p, &coord_count);
  inst.vec_width = 4;
  if (pp_check_dot(p, ".v4")) { inst.vec_width = 4; pp_next(p); }
  else if (pp_check_dot(p, ".v2")) { inst.vec_width = 2; pp_next(p); }
  PtxDataType dst_type, coord_type;
  if (!pp_parse_type(p, &dst_type)) {
    pp_error(p, "expected dest type for tex"); return;
  }
  inst.type = dst_type;
  if (!pp_parse_type(p, &coord_type)) {
    pp_error(p, "expected coord type for tex"); return;
  }
  inst.type2 = coord_type;
  inst.dst.kind = PTX_OPER_VEC;
  pp_expect(p, PTK_LBRACE, "{");
  int ndst = 0;
  while (ndst < inst.vec_width && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (ndst > 0) pp_expect(p, PTK_COMMA, ",");
    pp_read_ident(p, inst.dst.regs[ndst], sizeof(inst.dst.regs[ndst]));
    ndst++;
  }
  inst.dst.vec_count = ndst;
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACKET, "[");
  inst.src[0].kind = PTX_OPER_REG;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACE, "{");
  while (inst.src_count < 8 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (inst.src_count > 1) pp_expect(p, PTK_COMMA, ",");
    inst.src[inst.src_count] = pp_parse_operand(p);
    inst.src_count++;
  }
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_RBRACKET, "]");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_tld4(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_TLD4;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.tex_gather_comp = 0;
  if      (pp_check_dot(p, ".r")) { inst.tex_gather_comp = 0; pp_next(p); }
  else if (pp_check_dot(p, ".g")) { inst.tex_gather_comp = 1; pp_next(p); }
  else if (pp_check_dot(p, ".b")) { inst.tex_gather_comp = 2; pp_next(p); }
  else if (pp_check_dot(p, ".a")) { inst.tex_gather_comp = 3; pp_next(p); }
  int coord_count = 2;
  inst.tex_geom = pp_parse_tex_geom(p, &coord_count);
  if (pp_check_dot(p, ".v4")) pp_next(p);
  PtxDataType dst_type, coord_type;
  if (!pp_parse_type(p, &dst_type)) {
    pp_error(p, "expected dest type for tld4"); return;
  }
  inst.type = dst_type;
  if (!pp_parse_type(p, &coord_type)) {
    pp_error(p, "expected coord type for tld4"); return;
  }
  inst.type2 = coord_type;
  inst.dst.kind = PTX_OPER_VEC;
  pp_expect(p, PTK_LBRACE, "{");
  int ndst = 0;
  while (ndst < 4 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (ndst > 0) pp_expect(p, PTK_COMMA, ",");
    pp_read_ident(p, inst.dst.regs[ndst], sizeof(inst.dst.regs[ndst]));
    ndst++;
  }
  inst.dst.vec_count = ndst;
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACKET, "[");
  inst.src[0].kind = PTX_OPER_REG;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACE, "{");
  while (inst.src_count < 8 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (inst.src_count > 1) pp_expect(p, PTK_COMMA, ",");
    inst.src[inst.src_count] = pp_parse_operand(p);
    inst.src_count++;
  }
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_RBRACKET, "]");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_suld(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_SULD;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".b") || pp_check_dot(p, ".p")) pp_next(p);
  int coord_count = 2;
  inst.tex_geom = pp_parse_tex_geom(p, &coord_count);
  inst.vec_width = 4;
  if (pp_check_dot(p, ".v4")) { inst.vec_width = 4; pp_next(p); }
  else if (pp_check_dot(p, ".v2")) { inst.vec_width = 2; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for suld"); return;
  }
  inst.type = type;
  inst.dst.kind = PTX_OPER_VEC;
  pp_expect(p, PTK_LBRACE, "{");
  int ndst = 0;
  while (ndst < inst.vec_width && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (ndst > 0) pp_expect(p, PTK_COMMA, ",");
    pp_read_ident(p, inst.dst.regs[ndst], sizeof(inst.dst.regs[ndst]));
    ndst++;
  }
  inst.dst.vec_count = ndst;
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACKET, "[");
  inst.src[0].kind = PTX_OPER_REG;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACE, "{");
  while (inst.src_count < 8 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (inst.src_count > 1) pp_expect(p, PTK_COMMA, ",");
    inst.src[inst.src_count] = pp_parse_operand(p);
    inst.src_count++;
  }
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_RBRACKET, "]");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_sust(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_SUST;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".b") || pp_check_dot(p, ".p")) pp_next(p);
  int coord_count = 2;
  inst.tex_geom = pp_parse_tex_geom(p, &coord_count);
  inst.vec_width = 4;
  if (pp_check_dot(p, ".v4")) { inst.vec_width = 4; pp_next(p); }
  else if (pp_check_dot(p, ".v2")) { inst.vec_width = 2; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for sust"); return;
  }
  inst.type = type;
  pp_expect(p, PTK_LBRACKET, "[");
  inst.src[0].kind = PTX_OPER_REG;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACE, "{");
  while (inst.src_count < 5 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (inst.src_count > 1) pp_expect(p, PTK_COMMA, ",");
    inst.src[inst.src_count] = pp_parse_operand(p);
    inst.src_count++;
  }
  pp_expect(p, PTK_RBRACE, "}");
  pp_expect(p, PTK_RBRACKET, "]");
  pp_expect(p, PTK_COMMA, ",");
  int coord_start = inst.src_count;
  pp_expect(p, PTK_LBRACE, "{");
  while (inst.src_count < 8 && !pp_check(p, PTK_RBRACE) &&
         !pp_check(p, PTK_EOF)) {
    if (inst.src_count > coord_start) pp_expect(p, PTK_COMMA, ",");
    inst.src[inst.src_count] = pp_parse_operand(p);
    inst.src_count++;
  }
  pp_expect(p, PTK_RBRACE, "}");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_txq(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  inst.tex_gather_comp = 0;
  if (pp_check_dot(p, ".width")) { inst.tex_gather_comp = 0; pp_next(p); }
  else if (pp_check_dot(p, ".height")) { inst.tex_gather_comp = 1; pp_next(p); }
  else if (pp_check_dot(p, ".depth")) { inst.tex_gather_comp = 2; pp_next(p); }
  else if (pp_check_dot(p, ".num_mipmap_levels")) { inst.tex_gather_comp = 3; pp_next(p); }
  else if (pp_check(p, PTK_DOT_TOKEN)) pp_next(p);
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for txq/suq"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  pp_expect(p, PTK_LBRACKET, "[");
  inst.src[0].kind = PTX_OPER_REG;
  pp_read_ident(p, inst.src[0].name, sizeof(inst.src[0].name));
  inst.src_count = 1;
  pp_expect(p, PTK_RBRACKET, "]");
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_bit_unary(PtxParser *p, PtxOpcode opcode) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = opcode;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (opcode == PTX_OP_BFIND && pp_check_dot(p, ".shiftamt")) {
    inst.modifiers |= PTX_MOD_SHIFTAMT;
    pp_next(p);
  }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  inst.src_count = 1;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_shf(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_SHF;
  inst.line = p->cur.line; inst.col = p->cur.col;
  if (pp_check_dot(p, ".l")) { inst.modifiers |= PTX_MOD_LEFT; pp_next(p); }
  else if (pp_check_dot(p, ".r")) { inst.modifiers |= PTX_MOD_RIGHT; pp_next(p); }
  if (pp_check_dot(p, ".wrap")) { inst.modifiers |= PTX_MOD_WRAP; pp_next(p); }
  else if (pp_check_dot(p, ".clamp")) { inst.modifiers |= PTX_MOD_CLAMP; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for shf"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  inst.src_count = 3;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_bfi(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_BFI;
  inst.line = p->cur.line; inst.col = p->cur.col;
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for bfi"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[3] = pp_parse_operand(p);
  inst.src_count = 4;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_prmt(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_PRMT;
  inst.line = p->cur.line; inst.col = p->cur.col;
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for prmt"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[2] = pp_parse_operand(p);
  inst.src_count = 3;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

static void pp_parse_copysign(PtxParser *p) {
  PtxInst inst;
  memset(&inst, 0, sizeof(inst));
  inst.opcode = PTX_OP_COPYSIGN;
  inst.line = p->cur.line; inst.col = p->cur.col;
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type for copysign"); return;
  }
  inst.type = type;
  inst.dst = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[0] = pp_parse_operand(p);
  pp_expect(p, PTK_COMMA, ",");
  inst.src[1] = pp_parse_operand(p);
  inst.src_count = 2;
  pp_eat(p, PTK_SEMI);
  pp_emit_inst(p, &inst);
}

/* ===== Declaration Parsing ===== */

static void pp_parse_reg_decl(PtxParser *p) {
  pp_next(p);
  int vec_width = 1;
  if (pp_check_dot(p, ".v2")) { vec_width = 2; pp_next(p); }
  else if (pp_check_dot(p, ".v4")) { vec_width = 4; pp_next(p); }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type in .reg"); return;
  }
  do {
    PtxRegDecl decl;
    memset(&decl, 0, sizeof(decl));
    decl.type = type;
    decl.vec_width = vec_width;
    pp_read_ident(p, decl.name, sizeof(decl.name));
    if (pp_eat(p, PTK_LANGLE)) {
      if (pp_check(p, PTK_INT_LIT)) {
        decl.count = (int)p->cur.int_val;
        decl.is_parameterized = true;
        pp_next(p);
        pp_expect(p, PTK_RANGLE, ">");
      }
    } else {
      decl.count = 1;
      decl.is_parameterized = false;
    }
    pp_add_reg_decl(p, &decl);
  } while (pp_eat(p, PTK_COMMA));
  pp_eat(p, PTK_SEMI);
}

static void pp_parse_local_decl(PtxParser *p) {
  pp_next(p);
  PtxLocalDecl decl;
  memset(&decl, 0, sizeof(decl));
  if (pp_check_dot(p, ".align")) {
    pp_next(p);
    if (pp_check(p, PTK_INT_LIT)) {
      decl.alignment = (uint32_t)p->cur.int_val;
      pp_next(p);
    }
  }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type in .local"); return;
  }
  decl.type = type;
  pp_read_ident(p, decl.name, sizeof(decl.name));
  if (pp_eat(p, PTK_LBRACKET)) {
    if (pp_check(p, PTK_INT_LIT)) {
      decl.array_size = (uint32_t)p->cur.int_val;
      pp_next(p);
    }
    pp_expect(p, PTK_RBRACKET, "]");
    while (pp_eat(p, PTK_LBRACKET)) {
      if (pp_check(p, PTK_INT_LIT)) {
        decl.array_size *= (uint32_t)p->cur.int_val;
        pp_next(p);
      }
      pp_expect(p, PTK_RBRACKET, "]");
    }
  }
  pp_add_local_decl(p, &decl);
  pp_eat(p, PTK_SEMI);
}

/* ===== Module-Level Parsing ===== */

static void pp_parse_version(PtxParser *p) {
  pp_next(p);
  if (pp_check(p, PTK_FLOAT_LIT)) {
    p->mod->version_major = (int)p->cur.float_val;
    p->mod->version_minor =
      (int)((p->cur.float_val - p->mod->version_major) * 10 + 0.5);
    pp_next(p);
  } else if (pp_check(p, PTK_INT_LIT)) {
    p->mod->version_major = (int)p->cur.int_val;
    pp_next(p);
  }
}

static void pp_parse_target(PtxParser *p) {
  pp_next(p);
  tok_to_str(&p->cur, p->mod->target, sizeof(p->mod->target));
  pp_next(p);
  while (pp_eat(p, PTK_COMMA)) pp_next(p);
}

static void pp_parse_address_size(PtxParser *p) {
  pp_next(p);
  if (pp_check(p, PTK_INT_LIT)) {
    p->mod->address_size = (int)p->cur.int_val;
    pp_next(p);
  }
}

static void pp_parse_texref_decl(PtxParser *p) {
  PtxRefDecl ref;
  memset(&ref, 0, sizeof(ref));
  if (pp_check_dot(p, ".texref"))      ref.kind = PTX_REF_TEXREF;
  else if (pp_check_dot(p, ".samplerref")) ref.kind = PTX_REF_SAMPLERREF;
  else if (pp_check_dot(p, ".surfref"))    ref.kind = PTX_REF_SURFREF;
  pp_next(p);
  pp_read_ident(p, ref.name, sizeof(ref.name));
  pp_eat(p, PTK_SEMI);
  pp_add_ref(p, &ref);
}

static void pp_parse_global_decl(PtxParser *p, PtxMemSpace space) {
  pp_next(p);
  PtxGlobalDecl g;
  memset(&g, 0, sizeof(g));
  g.space = space;

  if (pp_check_dot(p, ".extern")) pp_next(p);
  if (pp_check_dot(p, ".align")) {
    pp_next(p);
    if (pp_check(p, PTK_INT_LIT)) {
      g.alignment = (uint32_t)p->cur.int_val;
      pp_next(p);
    }
  }
  if (pp_check_dot(p, ".texref") || pp_check_dot(p, ".samplerref") ||
      pp_check_dot(p, ".surfref")) {
    pp_parse_texref_decl(p);
    return;
  }
  PtxDataType type;
  if (!pp_parse_type(p, &type)) {
    pp_error(p, "expected type in global declaration"); return;
  }
  g.type = type;
  pp_read_ident(p, g.name, sizeof(g.name));
  if (pp_eat(p, PTK_LBRACKET)) {
    if (pp_check(p, PTK_INT_LIT)) {
      g.array_size = (uint32_t)p->cur.int_val;
      pp_next(p);
    }
    pp_expect(p, PTK_RBRACKET, "]");
    while (pp_eat(p, PTK_LBRACKET)) {
      if (pp_check(p, PTK_INT_LIT)) {
        g.array_size *= (uint32_t)p->cur.int_val;
        pp_next(p);
      }
      pp_expect(p, PTK_RBRACKET, "]");
    }
  }
  if (pp_eat(p, PTK_EQ)) {
    g.has_init = true;
    if (pp_eat(p, PTK_LBRACE)) {
      while (!pp_check(p, PTK_RBRACE) && !pp_check(p, PTK_EOF)) {
        if (pp_check(p, PTK_INT_LIT) && g.init_count < 1024) {
          g.init_vals[g.init_count++] = p->cur.int_val;
          pp_next(p);
        } else {
          pp_next(p);
        }
        pp_eat(p, PTK_COMMA);
      }
      pp_expect(p, PTK_RBRACE, "}");
    } else {
      pp_next(p);
    }
  }
  pp_eat(p, PTK_SEMI);
  pp_add_global(p, &g);
}

/* ===== Param List Parsing ===== */

static void pp_parse_param_list(PtxParser *p, PtxParam **params,
                                int *count, int *cap) {
  if (!pp_eat(p, PTK_LPAREN)) return;
  while (!pp_check(p, PTK_RPAREN) && !pp_check(p, PTK_EOF)) {
    if (pp_check_dot(p, ".param")) pp_next(p);
    else if (pp_check_dot(p, ".reg")) pp_next(p);
    else break;
    PtxParam param;
    memset(&param, 0, sizeof(param));
    if (pp_check_dot(p, ".align")) {
      pp_next(p);
      if (pp_check(p, PTK_INT_LIT)) {
        param.alignment = (uint32_t)p->cur.int_val;
        pp_next(p);
      }
    }
    PtxDataType type;
    if (!pp_parse_type(p, &type)) {
      pp_error(p, "expected type in param"); return;
    }
    param.type = type;
    pp_read_ident(p, param.name, sizeof(param.name));
    if (pp_eat(p, PTK_LBRACKET)) {
      if (pp_check(p, PTK_INT_LIT)) {
        param.array_size = (uint32_t)p->cur.int_val;
        pp_next(p);
      }
      pp_expect(p, PTK_RBRACKET, "]");
    }
    GROW(*params, *count, *cap, PtxParam);
    (*params)[(*count)++] = param;
    if (!pp_eat(p, PTK_COMMA)) break;
  }
  pp_expect(p, PTK_RPAREN, ")");
}

/* ===== Instruction Dispatch ===== */

static void pp_parse_instruction(PtxParser *p) {
  if (p->had_error) return;

  if (pp_eat(p, PTK_AT)) {
    p->pending_pred_negated = pp_eat(p, PTK_BANG);
    pp_read_ident(p, p->pending_pred, sizeof(p->pending_pred));
    p->pending_has_pred = true;
  }

  if (pp_check(p, PTK_IDENT)) {
    PtxLexer save = p->lex;
    PtxToken save_tok = p->cur;
    char name[80];
    tok_to_str(&p->cur, name, sizeof(name));
    pp_next(p);
    if (pp_check(p, PTK_COLON)) {
      pp_next(p);
      PtxStmt stmt;
      memset(&stmt, 0, sizeof(stmt));
      stmt.kind = PTX_STMT_LABEL;
      snprintf(stmt.label, sizeof(stmt.label), "%s", name);
      pp_add_stmt(p, &stmt);
      return;
    }
    p->lex = save;
    p->cur = save_tok;
  }

  char op[80];
  tok_to_str(&p->cur, op, sizeof(op));

  if (pp_check(p, PTK_IDENT)) {
    pp_next(p);
    if      (strcmp(op, "add") == 0)  pp_parse_arith(p, PTX_OP_ADD);
    else if (strcmp(op, "sub") == 0)  pp_parse_arith(p, PTX_OP_SUB);
    else if (strcmp(op, "mul") == 0)  pp_parse_arith(p, PTX_OP_MUL);
    else if (strcmp(op, "mul24") == 0) pp_parse_arith(p, PTX_OP_MUL24);
    else if (strcmp(op, "div") == 0)  pp_parse_arith(p, PTX_OP_DIV);
    else if (strcmp(op, "rem") == 0)  pp_parse_arith(p, PTX_OP_REM);
    else if (strcmp(op, "neg") == 0)  pp_parse_unary(p, PTX_OP_NEG);
    else if (strcmp(op, "abs") == 0)  pp_parse_unary(p, PTX_OP_ABS);
    else if (strcmp(op, "not") == 0)  pp_parse_unary(p, PTX_OP_NOT);
    else if (strcmp(op, "cnot") == 0) pp_parse_unary(p, PTX_OP_CNOT);
    else if (strcmp(op, "min") == 0)  pp_parse_arith(p, PTX_OP_MIN);
    else if (strcmp(op, "max") == 0)  pp_parse_arith(p, PTX_OP_MAX);
    else if (strcmp(op, "mad") == 0)  pp_parse_mad(p);
    else if (strcmp(op, "fma") == 0)  pp_parse_fma(p);
    else if (strcmp(op, "and") == 0)  pp_parse_arith(p, PTX_OP_AND);
    else if (strcmp(op, "or") == 0)   pp_parse_arith(p, PTX_OP_OR);
    else if (strcmp(op, "xor") == 0)  pp_parse_arith(p, PTX_OP_XOR);
    else if (strcmp(op, "shl") == 0)  pp_parse_arith(p, PTX_OP_SHL);
    else if (strcmp(op, "shr") == 0)  pp_parse_arith(p, PTX_OP_SHR);
    else if (strcmp(op, "setp") == 0) pp_parse_setp(p, PTX_OP_SETP);
    else if (strcmp(op, "set") == 0)  pp_parse_setp(p, PTX_OP_SET);
    else if (strcmp(op, "selp") == 0) pp_parse_selp(p);
    else if (strcmp(op, "mov") == 0)  pp_parse_mov(p);
    else if (strcmp(op, "ld") == 0)   pp_parse_ld(p);
    else if (strcmp(op, "st") == 0)   pp_parse_st(p);
    else if (strcmp(op, "cvt") == 0)  pp_parse_cvt(p);
    else if (strcmp(op, "cvta") == 0) pp_parse_cvta(p);
    else if (strcmp(op, "bra") == 0)  pp_parse_bra(p);
    else if (strcmp(op, "ret") == 0)  pp_parse_ret(p);
    else if (strcmp(op, "exit") == 0) pp_parse_exit(p);
    else if (strcmp(op, "call") == 0) pp_parse_call(p);
    else if (strcmp(op, "bar") == 0)  pp_parse_bar(p);
    else if (strcmp(op, "membar") == 0) pp_parse_membar(p);
    else if (strcmp(op, "atom") == 0) pp_parse_atom(p);
    else if (strcmp(op, "rcp") == 0)  pp_parse_math_unary(p, PTX_OP_RCP);
    else if (strcmp(op, "sqrt") == 0) pp_parse_math_unary(p, PTX_OP_SQRT);
    else if (strcmp(op, "rsqrt") == 0) pp_parse_math_unary(p, PTX_OP_RSQRT);
    else if (strcmp(op, "sin") == 0)  pp_parse_math_unary(p, PTX_OP_SIN);
    else if (strcmp(op, "cos") == 0)  pp_parse_math_unary(p, PTX_OP_COS);
    else if (strcmp(op, "lg2") == 0)  pp_parse_math_unary(p, PTX_OP_LG2);
    else if (strcmp(op, "ex2") == 0)  pp_parse_math_unary(p, PTX_OP_EX2);
    else if (strcmp(op, "tex") == 0)  pp_parse_tex(p);
    else if (strcmp(op, "tld4") == 0) pp_parse_tld4(p);
    else if (strcmp(op, "suld") == 0) pp_parse_suld(p);
    else if (strcmp(op, "sust") == 0) pp_parse_sust(p);
    else if (strcmp(op, "txq") == 0)  pp_parse_txq(p, PTX_OP_TXQ);
    else if (strcmp(op, "suq") == 0)  pp_parse_txq(p, PTX_OP_SUQ);
    else if (strcmp(op, "popc") == 0) pp_parse_bit_unary(p, PTX_OP_POPC);
    else if (strcmp(op, "brev") == 0) pp_parse_bit_unary(p, PTX_OP_BREV);
    else if (strcmp(op, "clz") == 0)  pp_parse_bit_unary(p, PTX_OP_CLZ);
    else if (strcmp(op, "bfind") == 0) pp_parse_bit_unary(p, PTX_OP_BFIND);
    else if (strcmp(op, "shf") == 0)  pp_parse_shf(p);
    else if (strcmp(op, "bfi") == 0)  pp_parse_bfi(p);
    else if (strcmp(op, "prmt") == 0) pp_parse_prmt(p);
    else if (strcmp(op, "copysign") == 0) pp_parse_copysign(p);
    else {
      while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF) &&
             !pp_check(p, PTK_RBRACE))
        pp_next(p);
      pp_eat(p, PTK_SEMI);
    }
  } else if (pp_check(p, PTK_DOT_TOKEN)) {
    if (pp_check_dot(p, ".reg")) {
      pp_parse_reg_decl(p);
    } else if (pp_check_dot(p, ".local")) {
      pp_parse_local_decl(p);
    } else if (pp_check_dot(p, ".shared")) {
      pp_parse_global_decl(p, PTX_SPACE_SHARED);
    } else if (pp_check_dot(p, ".pragma")) {
      while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF))
        pp_next(p);
      pp_eat(p, PTK_SEMI);
    } else {
      while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF))
        pp_next(p);
      pp_eat(p, PTK_SEMI);
    }
  } else if (pp_check(p, PTK_LBRACE)) {
    pp_next(p);
    while (!pp_check(p, PTK_RBRACE) && !pp_check(p, PTK_EOF) &&
           !p->had_error) {
      pp_parse_instruction(p);
    }
    pp_expect(p, PTK_RBRACE, "}");
  }
}

static void pp_parse_function_body(PtxParser *p) {
  pp_expect(p, PTK_LBRACE, "{");
  while (!pp_check(p, PTK_RBRACE) && !pp_check(p, PTK_EOF) &&
         !p->had_error) {
    pp_parse_instruction(p);
  }
  pp_expect(p, PTK_RBRACE, "}");
}

/* ===== Entry/Function Parsing ===== */

static void pp_parse_entry(PtxParser *p) {
  pp_next(p);
  PtxEntry *e = pp_add_entry(p);
  pp_read_ident(p, e->name, sizeof(e->name));
  p->cur_entry = e;
  p->cur_func = NULL;

  pp_parse_param_list(p, &e->params, &e->param_count,
                      &e->param_cap);

  while (pp_check_dot(p, ".maxntid") || pp_check_dot(p, ".reqntid") ||
         pp_check_dot(p, ".minnctapersm") || pp_check_dot(p, ".maxnreg") ||
         pp_check_dot(p, ".pragma") || pp_check_dot(p, ".noreturn")) {
    if (pp_check_dot(p, ".maxntid") || pp_check_dot(p, ".reqntid")) {
      pp_next(p);
      if (pp_check(p, PTK_INT_LIT)) {
        e->wg_size[0] = (uint32_t)p->cur.int_val; pp_next(p);
      }
      if (pp_eat(p, PTK_COMMA) && pp_check(p, PTK_INT_LIT)) {
        e->wg_size[1] = (uint32_t)p->cur.int_val; pp_next(p);
      }
      if (pp_eat(p, PTK_COMMA) && pp_check(p, PTK_INT_LIT)) {
        e->wg_size[2] = (uint32_t)p->cur.int_val; pp_next(p);
      }
    } else {
      pp_next(p);
      while (!pp_check(p, PTK_LBRACE) &&
             !pp_check(p, PTK_DOT_TOKEN) &&
             !pp_check(p, PTK_EOF))
        pp_next(p);
    }
  }

  pp_parse_function_body(p);
  p->cur_entry = NULL;
}

static void pp_parse_func(PtxParser *p) {
  pp_next(p);
  PtxFunc *f = pp_add_func(p);
  p->cur_func = f;
  p->cur_entry = NULL;
  f->return_type = PTX_TYPE_NONE;

  if (pp_eat(p, PTK_LPAREN)) {
    if (pp_check_dot(p, ".reg")) pp_next(p);
    PtxDataType ret_type;
    if (pp_parse_type(p, &ret_type)) {
      f->return_type = ret_type;
      f->has_return = true;
    }
    pp_read_ident(p, f->return_reg, sizeof(f->return_reg));
    pp_expect(p, PTK_RPAREN, ")");
  }

  pp_read_ident(p, f->name, sizeof(f->name));

  pp_parse_param_list(p, &f->params, &f->param_count,
                      &f->param_cap);

  if (pp_check(p, PTK_LBRACE)) {
    pp_parse_function_body(p);
  } else {
    f->is_decl_only = true;
    pp_eat(p, PTK_SEMI);
  }
  p->cur_func = NULL;
}

/* ===== Top-Level Parsing ===== */

static void pp_parse_toplevel(PtxParser *p) {
  while (!pp_check(p, PTK_EOF) && !p->had_error) {
    if (pp_check_dot(p, ".version")) {
      pp_parse_version(p);
    } else if (pp_check_dot(p, ".target")) {
      pp_parse_target(p);
    } else if (pp_check_dot(p, ".address_size")) {
      pp_parse_address_size(p);
    } else if (pp_check_dot(p, ".visible") ||
               pp_check_dot(p, ".weak")) {
      pp_next(p);
      if (pp_check_dot(p, ".entry")) pp_parse_entry(p);
      else if (pp_check_dot(p, ".func")) pp_parse_func(p);
      else if (pp_check_dot(p, ".global"))
        pp_parse_global_decl(p, PTX_SPACE_GLOBAL);
      else if (pp_check_dot(p, ".const"))
        pp_parse_global_decl(p, PTX_SPACE_CONST);
      else if (pp_check_dot(p, ".shared"))
        pp_parse_global_decl(p, PTX_SPACE_SHARED);
      else if (pp_check_dot(p, ".texref") ||
               pp_check_dot(p, ".samplerref") ||
               pp_check_dot(p, ".surfref")) {
        pp_parse_texref_decl(p);
      } else {
        while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF))
          pp_next(p);
        pp_eat(p, PTK_SEMI);
      }
    } else if (pp_check_dot(p, ".extern")) {
      pp_next(p);
      while (!pp_check(p, PTK_SEMI) && !pp_check(p, PTK_EOF) &&
             !pp_check(p, PTK_LBRACE))
        pp_next(p);
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
      pp_parse_global_decl(p, PTX_SPACE_GLOBAL);
    } else if (pp_check_dot(p, ".shared")) {
      pp_parse_global_decl(p, PTX_SPACE_SHARED);
    } else if (pp_check_dot(p, ".const")) {
      pp_parse_global_decl(p, PTX_SPACE_CONST);
    } else if (pp_check_dot(p, ".texref") ||
               pp_check_dot(p, ".samplerref") ||
               pp_check_dot(p, ".surfref")) {
      pp_parse_texref_decl(p);
    } else {
      pp_next(p);
    }
  }
}

/* ===== Public API ===== */

PtxModule *ptx_parse(const char *source, char **out_error) {
  if (!source) {
    if (out_error) *out_error = ptx_strdup("null source");
    return NULL;
  }
  PtxParser parser;
  memset(&parser, 0, sizeof(parser));
  plx_init(&parser.lex, source);
  parser.mod = (PtxModule *)PTX_MALLOC(sizeof(PtxModule));
  memset(parser.mod, 0, sizeof(PtxModule));
  pp_next(&parser);
  pp_parse_toplevel(&parser);
  if (parser.had_error) {
    if (out_error) *out_error = ptx_strdup(parser.error);
    ptx_parse_free(parser.mod);
    return NULL;
  }
  return parser.mod;
}

void ptx_parse_free(PtxModule *mod) {
  if (!mod) return;
  for (int i = 0; i < mod->entry_count; i++) {
    PtxEntry *e = &mod->entries[i];
    PTX_FREE(e->params);
    PTX_FREE(e->reg_decls);
    PTX_FREE(e->local_decls);
    PTX_FREE(e->body);
  }
  PTX_FREE(mod->entries);
  for (int i = 0; i < mod->func_count; i++) {
    PtxFunc *f = &mod->functions[i];
    PTX_FREE(f->params);
    PTX_FREE(f->reg_decls);
    PTX_FREE(f->local_decls);
    PTX_FREE(f->body);
  }
  PTX_FREE(mod->functions);
  PTX_FREE(mod->globals);
  PTX_FREE(mod->refs);
  PTX_FREE(mod);
}
