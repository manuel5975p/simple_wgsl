/*
 * glsl_parser.c - GLSL 450 (Vulkan) Lexer and Parser
 *
 * Produces the same WgslAstNode AST as wgsl_parser.c.
 * Maps GLSL constructs to WGSL-compatible AST nodes:
 *   - layout(set=S, binding=B) → @group(S) @binding(B) attributes
 *   - layout(location=N) → @location(N) attribute
 *   - in/out/uniform/buffer/shared → address_space field
 *   - GLSL types (vec3, mat4) → TypeNode with GLSL name
 *   - Interface blocks → struct + global var
 */

#include "simple_wgsl.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ============================================================================
 * String Utilities
 * ============================================================================ */

static char *glsl_strndup(const char *s, size_t n) {
    char *r = (char *)NODE_MALLOC(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

static char *glsl_strdup(const char *s) {
    return glsl_strndup(s, s ? strlen(s) : 0);
}

static void *grow_ptr_array(void *p, int needed, int *cap, size_t elem) {
    if (needed <= *cap) return p;
    int nc = (*cap == 0) ? 4 : (*cap * 2);
    while (nc < needed) nc *= 2;
    void *np = NODE_REALLOC(p, (size_t)nc * elem);
    if (!np) return p;
    *cap = nc;
    return np;
}

static void vec_push_node(WgslAstNode ***arr, int *count, int *cap,
                           WgslAstNode *v) {
    *arr = (WgslAstNode **)grow_ptr_array(*arr, *count + 1, cap,
                                           sizeof(WgslAstNode *));
    (*arr)[(*count)++] = v;
}

/* ============================================================================
 * Lexer
 * ============================================================================ */

typedef enum TokenType {
    TOK_EOF = 0,
    TOK_IDENT,
    TOK_NUMBER,
    /* punctuation */
    TOK_COLON,
    TOK_SEMI,
    TOK_COMMA,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LT,
    TOK_GT,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_DOT,
    TOK_STAR,
    TOK_SLASH,
    TOK_PLUS,
    TOK_MINUS,
    TOK_PERCENT,
    TOK_EQ,
    TOK_BANG,
    TOK_TILDE,
    TOK_QMARK,
    TOK_AMP,
    TOK_PIPE,
    TOK_CARET,
    /* multi-char operators */
    TOK_LE,
    TOK_GE,
    TOK_EQEQ,
    TOK_NEQ,
    TOK_ANDAND,
    TOK_OROR,
    TOK_PLUSPLUS,
    TOK_MINUSMINUS,
    TOK_SHL,
    TOK_SHR,
    TOK_PLUSEQ,
    TOK_MINUSEQ,
    TOK_STAREQ,
    TOK_SLASHEQ,
    TOK_PERCENTEQ,
    TOK_AMPEQ,
    TOK_PIPEEQ,
    TOK_CARETEQ,
    TOK_SHLEQ,
    TOK_SHREQ,
    /* keywords */
    TOK_STRUCT,
    TOK_IF,
    TOK_ELSE,
    TOK_WHILE,
    TOK_FOR,
    TOK_DO,
    TOK_SWITCH,
    TOK_CASE,
    TOK_DEFAULT,
    TOK_BREAK,
    TOK_CONTINUE,
    TOK_RETURN,
    TOK_DISCARD,
    TOK_LAYOUT,
    TOK_CONST,
    TOK_PRECISION
} TokenType;

typedef struct Token {
    TokenType type;
    const char *start;
    int length;
    int line;
    int col;
    bool is_float;
} Token;

typedef struct Lexer {
    const char *src;
    size_t pos;
    int line;
    int col;
} Lexer;

static bool lx_peek2(const Lexer *L, char a, char b) {
    return (L->src[L->pos] == a && L->src[L->pos + 1] == b);
}

static void lx_advance(Lexer *L) {
    char c = L->src[L->pos];
    if (!c) return;
    L->pos++;
    if (c == '\n') { L->line++; L->col = 1; }
    else { L->col++; }
}

static void lx_skip_ws_comments(Lexer *L) {
    for (;;) {
        char c = L->src[L->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            lx_advance(L);
            continue;
        }
        if (c == '/' && L->src[L->pos + 1] == '/') {
            while (L->src[L->pos] && L->src[L->pos] != '\n')
                lx_advance(L);
            continue;
        }
        if (c == '/' && L->src[L->pos + 1] == '*') {
            lx_advance(L); lx_advance(L);
            while (L->src[L->pos]) {
                if (L->src[L->pos] == '*' && L->src[L->pos + 1] == '/') {
                    lx_advance(L); lx_advance(L);
                    break;
                }
                lx_advance(L);
            }
            continue;
        }
        /* Skip preprocessor directives: # until end of line */
        if (c == '#') {
            while (L->src[L->pos] && L->src[L->pos] != '\n')
                lx_advance(L);
            continue;
        }
        break;
    }
}

static bool is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}
static bool is_ident_part(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

static Token make_token(Lexer *L, TokenType t, const char *s, int len, bool f) {
    Token tok;
    tok.type = t;
    tok.start = s;
    tok.length = len;
    tok.line = L->line;
    tok.col = L->col;
    tok.is_float = f;
    return tok;
}

static int is_dec_digit_or_us(char c) { return isdigit((unsigned char)c) || c == '_'; }
static int is_hex_digit_or_us(char c) { return isxdigit((unsigned char)c) || c == '_'; }

typedef struct { const char *word; int len; TokenType tok; } Keyword;

static const Keyword glsl_keywords[] = {
    {"struct",    6, TOK_STRUCT},
    {"if",        2, TOK_IF},
    {"else",      4, TOK_ELSE},
    {"while",     5, TOK_WHILE},
    {"for",       3, TOK_FOR},
    {"do",        2, TOK_DO},
    {"switch",    6, TOK_SWITCH},
    {"case",      4, TOK_CASE},
    {"default",   7, TOK_DEFAULT},
    {"break",     5, TOK_BREAK},
    {"continue",  8, TOK_CONTINUE},
    {"return",    6, TOK_RETURN},
    {"discard",   7, TOK_DISCARD},
    {"layout",    6, TOK_LAYOUT},
    {"const",     5, TOK_CONST},
    {"precision", 9, TOK_PRECISION},
    {NULL, 0, TOK_EOF}
};

static Token lx_next(Lexer *L) {
    lx_skip_ws_comments(L);
    const char *s = &L->src[L->pos];
    char c = *s;
    if (!c) return make_token(L, TOK_EOF, s, 0, false);

    /* multi-char operators (3-char first) */
    if (L->src[L->pos] == '<' && L->src[L->pos+1] == '<' && L->src[L->pos+2] == '=') {
        lx_advance(L); lx_advance(L); lx_advance(L);
        return make_token(L, TOK_SHLEQ, s, 3, false);
    }
    if (L->src[L->pos] == '>' && L->src[L->pos+1] == '>' && L->src[L->pos+2] == '=') {
        lx_advance(L); lx_advance(L); lx_advance(L);
        return make_token(L, TOK_SHREQ, s, 3, false);
    }

    /* 2-char operators */
    if (lx_peek2(L, '<', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_LE, s, 2, false); }
    if (lx_peek2(L, '<', '<')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_SHL, s, 2, false); }
    if (lx_peek2(L, '>', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_GE, s, 2, false); }
    if (lx_peek2(L, '>', '>')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_SHR, s, 2, false); }
    if (lx_peek2(L, '=', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_EQEQ, s, 2, false); }
    if (lx_peek2(L, '!', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_NEQ, s, 2, false); }
    if (lx_peek2(L, '&', '&')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_ANDAND, s, 2, false); }
    if (lx_peek2(L, '|', '|')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_OROR, s, 2, false); }
    if (lx_peek2(L, '+', '+')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_PLUSPLUS, s, 2, false); }
    if (lx_peek2(L, '-', '-')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_MINUSMINUS, s, 2, false); }
    if (lx_peek2(L, '+', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_PLUSEQ, s, 2, false); }
    if (lx_peek2(L, '-', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_MINUSEQ, s, 2, false); }
    if (lx_peek2(L, '*', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_STAREQ, s, 2, false); }
    if (lx_peek2(L, '/', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_SLASHEQ, s, 2, false); }
    if (lx_peek2(L, '%', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_PERCENTEQ, s, 2, false); }
    if (lx_peek2(L, '&', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_AMPEQ, s, 2, false); }
    if (lx_peek2(L, '|', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_PIPEEQ, s, 2, false); }
    if (lx_peek2(L, '^', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_CARETEQ, s, 2, false); }

    /* single-char punctuation */
    switch (c) {
    case ':': lx_advance(L); return make_token(L, TOK_COLON, s, 1, false);
    case ';': lx_advance(L); return make_token(L, TOK_SEMI, s, 1, false);
    case ',': lx_advance(L); return make_token(L, TOK_COMMA, s, 1, false);
    case '{': lx_advance(L); return make_token(L, TOK_LBRACE, s, 1, false);
    case '}': lx_advance(L); return make_token(L, TOK_RBRACE, s, 1, false);
    case '(': lx_advance(L); return make_token(L, TOK_LPAREN, s, 1, false);
    case ')': lx_advance(L); return make_token(L, TOK_RPAREN, s, 1, false);
    case '<': lx_advance(L); return make_token(L, TOK_LT, s, 1, false);
    case '>': lx_advance(L); return make_token(L, TOK_GT, s, 1, false);
    case '[': lx_advance(L); return make_token(L, TOK_LBRACKET, s, 1, false);
    case ']': lx_advance(L); return make_token(L, TOK_RBRACKET, s, 1, false);
    case '.': lx_advance(L); return make_token(L, TOK_DOT, s, 1, false);
    case '*': lx_advance(L); return make_token(L, TOK_STAR, s, 1, false);
    case '/': lx_advance(L); return make_token(L, TOK_SLASH, s, 1, false);
    case '+': lx_advance(L); return make_token(L, TOK_PLUS, s, 1, false);
    case '-': lx_advance(L); return make_token(L, TOK_MINUS, s, 1, false);
    case '%': lx_advance(L); return make_token(L, TOK_PERCENT, s, 1, false);
    case '=': lx_advance(L); return make_token(L, TOK_EQ, s, 1, false);
    case '!': lx_advance(L); return make_token(L, TOK_BANG, s, 1, false);
    case '~': lx_advance(L); return make_token(L, TOK_TILDE, s, 1, false);
    case '?': lx_advance(L); return make_token(L, TOK_QMARK, s, 1, false);
    case '&': lx_advance(L); return make_token(L, TOK_AMP, s, 1, false);
    case '|': lx_advance(L); return make_token(L, TOK_PIPE, s, 1, false);
    case '^': lx_advance(L); return make_token(L, TOK_CARET, s, 1, false);
    default: break;
    }

    /* identifiers and keywords */
    if (is_ident_start(c)) {
        size_t p = L->pos;
        lx_advance(L);
        while (is_ident_part(L->src[L->pos]))
            lx_advance(L);
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        for (const Keyword *kw = glsl_keywords; kw->word; kw++) {
            if (len == kw->len && strncmp(start, kw->word, (size_t)len) == 0)
                return make_token(L, kw->tok, start, len, false);
        }
        return make_token(L, TOK_IDENT, start, len, false);
    }

    /* numbers */
    if (isdigit((unsigned char)c)) {
        size_t p = L->pos;
        bool is_float = false;
        if (c == '0' && (L->src[L->pos + 1] == 'x' || L->src[L->pos + 1] == 'X')) {
            lx_advance(L); lx_advance(L);
            while (is_hex_digit_or_us(L->src[L->pos]))
                lx_advance(L);
            if (L->src[L->pos] == 'u' || L->src[L->pos] == 'U')
                lx_advance(L);
            const char *start = &L->src[p];
            int len = (int)(&L->src[L->pos] - start);
            return make_token(L, TOK_NUMBER, start, len, false);
        }
        lx_advance(L);
        while (is_dec_digit_or_us(L->src[L->pos]))
            lx_advance(L);
        if (L->src[L->pos] == '.' && isdigit((unsigned char)L->src[L->pos + 1])) {
            is_float = true;
            lx_advance(L);
            while (is_dec_digit_or_us(L->src[L->pos]))
                lx_advance(L);
        } else if (L->src[L->pos] == '.' && !is_ident_start(L->src[L->pos + 1])) {
            is_float = true;
            lx_advance(L);
        }
        if (L->src[L->pos] == 'e' || L->src[L->pos] == 'E') {
            is_float = true;
            lx_advance(L);
            if (L->src[L->pos] == '+' || L->src[L->pos] == '-')
                lx_advance(L);
            while (is_dec_digit_or_us(L->src[L->pos]))
                lx_advance(L);
        }
        if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F') {
            is_float = true;
            lx_advance(L);
        } else if (L->src[L->pos] == 'l' || L->src[L->pos] == 'L') {
            lx_advance(L); /* long suffix */
            if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F') {
                is_float = true;
                lx_advance(L);
            }
        } else if (!is_float && (L->src[L->pos] == 'u' || L->src[L->pos] == 'U')) {
            lx_advance(L);
        }
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        return make_token(L, TOK_NUMBER, start, len, is_float);
    }

    /* . followed by digit is a float literal */
    if (c == '.' && isdigit((unsigned char)L->src[L->pos + 1])) {
        size_t p = L->pos;
        lx_advance(L); /* consume '.' */
        while (is_dec_digit_or_us(L->src[L->pos]))
            lx_advance(L);
        if (L->src[L->pos] == 'e' || L->src[L->pos] == 'E') {
            lx_advance(L);
            if (L->src[L->pos] == '+' || L->src[L->pos] == '-')
                lx_advance(L);
            while (is_dec_digit_or_us(L->src[L->pos]))
                lx_advance(L);
        }
        if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F')
            lx_advance(L);
        const char *start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        return make_token(L, TOK_NUMBER, start, len, true);
    }

    lx_advance(L);
    return make_token(L, TOK_EOF, s, 0, false);
}

/* ============================================================================
 * Parser
 * ============================================================================ */

/* Known struct names for declaration/expression disambiguation */
typedef struct KnownType {
    char *name;
} KnownType;

typedef struct Parser {
    Lexer L;
    Token cur;
    bool had_error;
    KnownType *known_types;
    int known_type_count;
    int known_type_cap;
} Parser;

static void advance(Parser *P) { P->cur = lx_next(&P->L); }
static bool check(Parser *P, TokenType t) { return P->cur.type == t; }
static bool match(Parser *P, TokenType t) {
    if (check(P, t)) { advance(P); return true; }
    return false;
}
static void parse_error(Parser *P, const char *msg) {
    fprintf(stderr, "[glsl-parser] error at %d:%d: %s\n",
            P->cur.line, P->cur.col, msg);
    P->had_error = true;
}
static void expect(Parser *P, TokenType t, const char *msg) {
    if (!match(P, t)) parse_error(P, msg);
}

static WgslAstNode *new_node(Parser *P, WgslNodeType k) {
    WgslAstNode *n = NODE_ALLOC(WgslAstNode);
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->type = k;
    n->line = P->cur.line;
    n->col = P->cur.col;
    return n;
}

static bool tok_eq(const Token *t, const char *s) {
    int len = (int)strlen(s);
    return t->length == len && strncmp(t->start, s, (size_t)len) == 0;
}

static WgslAstNode *new_ident(Parser *P, const Token *t) {
    WgslAstNode *n = new_node(P, WGSL_NODE_IDENT);
    n->ident.name = glsl_strndup(t->start, (size_t)t->length);
    return n;
}
static WgslAstNode *new_literal(Parser *P, const Token *t) {
    WgslAstNode *n = new_node(P, WGSL_NODE_LITERAL);
    n->literal.lexeme = glsl_strndup(t->start, (size_t)t->length);
    n->literal.kind = t->is_float ? WGSL_LIT_FLOAT : WGSL_LIT_INT;
    return n;
}
static WgslAstNode *new_type(Parser *P, const char *name) {
    WgslAstNode *n = new_node(P, WGSL_NODE_TYPE);
    n->type_node.name = glsl_strdup(name);
    return n;
}

/* Type keyword detection */
static const char *glsl_type_keywords[] = {
    "void", "bool", "int", "uint", "float", "double",
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "bvec2", "bvec3", "bvec4",
    "dvec2", "dvec3", "dvec4",
    "mat2", "mat3", "mat4",
    "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4",
    "mat4x2", "mat4x3", "mat4x4",
    "dmat2", "dmat3", "dmat4",
    "sampler2D", "sampler3D", "samplerCube",
    "sampler2DShadow", "samplerCubeShadow",
    "sampler2DArray", "sampler2DArrayShadow",
    "samplerBuffer", "sampler2DMS",
    "isampler2D", "isampler3D", "isamplerCube",
    "isampler2DArray", "isamplerBuffer", "isampler2DMS",
    "usampler2D", "usampler3D", "usamplerCube",
    "usampler2DArray", "usamplerBuffer", "usampler2DMS",
    "image2D", "image3D", "imageCube",
    "image2DArray", "imageBuffer",
    "iimage2D", "iimage3D", "uimage2D", "uimage3D",
    NULL
};

static bool is_type_keyword_str(const char *name, int len) {
    for (const char **kw = glsl_type_keywords; *kw; kw++) {
        if ((int)strlen(*kw) == len && strncmp(*kw, name, (size_t)len) == 0)
            return true;
    }
    return false;
}

static bool is_type_keyword_tok(const Token *t) {
    if (t->type != TOK_IDENT) return false;
    return is_type_keyword_str(t->start, t->length);
}

/* Known struct tracking */
static void register_struct(Parser *P, const char *name) {
    P->known_types = (KnownType *)grow_ptr_array(
        P->known_types, P->known_type_count + 1,
        &P->known_type_cap, sizeof(KnownType));
    P->known_types[P->known_type_count].name = glsl_strdup(name);
    P->known_type_count++;
}

static bool is_known_struct(const Parser *P, const Token *t) {
    if (t->type != TOK_IDENT) return false;
    for (int i = 0; i < P->known_type_count; i++) {
        if ((int)strlen(P->known_types[i].name) == t->length &&
            strncmp(P->known_types[i].name, t->start, (size_t)t->length) == 0)
            return true;
    }
    return false;
}

static bool is_type_start(const Parser *P, const Token *t) {
    return is_type_keyword_tok(t) || is_known_struct(P, t);
}

/* Storage qualifier detection */
static bool is_storage_qualifier_tok(const Token *t) {
    if (t->type != TOK_IDENT) return false;
    return tok_eq(t, "in") || tok_eq(t, "out") || tok_eq(t, "inout") ||
           tok_eq(t, "uniform") || tok_eq(t, "buffer") || tok_eq(t, "shared");
}

/* Interpolation qualifier detection */
static bool is_interp_qualifier_tok(const Token *t) {
    if (t->type != TOK_IDENT) return false;
    return tok_eq(t, "flat") || tok_eq(t, "smooth") || tok_eq(t, "noperspective");
}

/* Precision qualifier detection */
static bool is_precision_qualifier_tok(const Token *t) {
    if (t->type != TOK_IDENT) return false;
    return tok_eq(t, "highp") || tok_eq(t, "mediump") || tok_eq(t, "lowp");
}

/* Forward declarations */
static WgslAstNode *parse_program(Parser *P);
static WgslAstNode *parse_block(Parser *P);
static WgslAstNode *parse_statement(Parser *P);
static WgslAstNode *parse_expr(Parser *P);
static WgslAstNode *parse_assignment(Parser *P);
static WgslAstNode *parse_conditional(Parser *P);
static WgslAstNode *parse_logical_or(Parser *P);
static WgslAstNode *parse_logical_and(Parser *P);
static WgslAstNode *parse_bitwise_or(Parser *P);
static WgslAstNode *parse_bitwise_xor(Parser *P);
static WgslAstNode *parse_bitwise_and(Parser *P);
static WgslAstNode *parse_equality(Parser *P);
static WgslAstNode *parse_relational(Parser *P);
static WgslAstNode *parse_shift(Parser *P);
static WgslAstNode *parse_additive(Parser *P);
static WgslAstNode *parse_multiplicative(Parser *P);
static WgslAstNode *parse_unary(Parser *P);
static WgslAstNode *parse_postfix(Parser *P);
static WgslAstNode *parse_primary(Parser *P);

/* ============================================================================
 * Layout Qualifier Parsing
 * ============================================================================ */

/*
 * Parse layout(...) and return a list of Attribute nodes.
 * Maps: set→group, binding→binding, location→location,
 *       local_size_x/y/z→workgroup_size
 */
static int parse_layout_qualifier(Parser *P, WgslAstNode ***out_attrs) {
    *out_attrs = NULL;
    if (!check(P, TOK_LAYOUT)) return 0;
    advance(P); /* consume 'layout' */
    expect(P, TOK_LPAREN, "expected '(' after 'layout'");

    int cap = 0, count = 0;
    WgslAstNode **attrs = NULL;

    /* Track local_size components for workgroup_size attribute */
    int has_wg = 0;
    int wg_x = 1, wg_y = 1, wg_z = 1;

    while (!check(P, TOK_RPAREN) && !check(P, TOK_EOF)) {
        if (count > 0 || has_wg) {
            if (!match(P, TOK_COMMA)) break;
        }
        if (!check(P, TOK_IDENT)) { parse_error(P, "expected layout qualifier name"); break; }
        Token name = P->cur;
        advance(P);

        WgslAstNode *value = NULL;
        if (match(P, TOK_EQ)) {
            if (check(P, TOK_NUMBER)) {
                Token num = P->cur;
                advance(P);
                value = new_literal(P, &num);
            } else if (check(P, TOK_IDENT)) {
                Token id = P->cur;
                advance(P);
                value = new_ident(P, &id);
            } else {
                parse_error(P, "expected value after '=' in layout qualifier");
            }
        }

        /* Map layout qualifiers to WGSL-style attributes */
        if (tok_eq(&name, "set") && value) {
            WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("group");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "binding") && value) {
            WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("binding");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "location") && value) {
            WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("location");
            attr->attribute.arg_count = 1;
            attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            attr->attribute.args[0] = value;
            vec_push_node(&attrs, &count, &cap, attr);
        } else if (tok_eq(&name, "local_size_x") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_x = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "local_size_y") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_y = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "local_size_z") && value) {
            has_wg = 1;
            if (value->type == WGSL_NODE_LITERAL && value->literal.lexeme)
                wg_z = atoi(value->literal.lexeme);
            wgsl_free_ast(value);
        } else if (tok_eq(&name, "push_constant")) {
            WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strdup("push_constant");
            vec_push_node(&attrs, &count, &cap, attr);
            if (value) wgsl_free_ast(value);
        } else {
            /* Unknown layout qualifier - store as-is */
            WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
            attr->attribute.name = glsl_strndup(name.start, (size_t)name.length);
            if (value) {
                attr->attribute.arg_count = 1;
                attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
                attr->attribute.args[0] = value;
            }
            vec_push_node(&attrs, &count, &cap, attr);
        }
    }
    expect(P, TOK_RPAREN, "expected ')'");

    /* Emit workgroup_size attribute if local_size was found */
    if (has_wg) {
        WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
        attr->attribute.name = glsl_strdup("workgroup_size");

        char buf_x[32], buf_y[32], buf_z[32];
        snprintf(buf_x, sizeof(buf_x), "%d", wg_x);
        snprintf(buf_y, sizeof(buf_y), "%d", wg_y);
        snprintf(buf_z, sizeof(buf_z), "%d", wg_z);

        attr->attribute.arg_count = 3;
        attr->attribute.args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *) * 3);
        WgslAstNode *ax = new_node(P, WGSL_NODE_LITERAL);
        ax->literal.lexeme = glsl_strdup(buf_x);
        ax->literal.kind = WGSL_LIT_INT;
        WgslAstNode *ay = new_node(P, WGSL_NODE_LITERAL);
        ay->literal.lexeme = glsl_strdup(buf_y);
        ay->literal.kind = WGSL_LIT_INT;
        WgslAstNode *az = new_node(P, WGSL_NODE_LITERAL);
        az->literal.lexeme = glsl_strdup(buf_z);
        az->literal.kind = WGSL_LIT_INT;
        attr->attribute.args[0] = ax;
        attr->attribute.args[1] = ay;
        attr->attribute.args[2] = az;
        vec_push_node(&attrs, &count, &cap, attr);
    }

    *out_attrs = attrs;
    return count;
}

/* ============================================================================
 * Type Parsing
 * ============================================================================ */

static WgslAstNode *parse_type_node(Parser *P) {
    if (!check(P, TOK_IDENT) && !check(P, TOK_STRUCT)) {
        parse_error(P, "expected type name");
        return NULL;
    }
    Token name = P->cur;
    advance(P);
    char *tname = glsl_strndup(name.start, (size_t)name.length);
    WgslAstNode *T = new_type(P, tname);
    NODE_FREE(tname);
    return T;
}

/* Parse C-style array dimensions after a variable name: [N][M]... */
static WgslAstNode *wrap_type_with_array_dims(Parser *P, WgslAstNode *base_type) {
    while (match(P, TOK_LBRACKET)) {
        WgslAstNode *dim = NULL;
        if (!check(P, TOK_RBRACKET)) {
            if (check(P, TOK_NUMBER)) {
                Token num = P->cur;
                advance(P);
                dim = new_literal(P, &num);
            } else {
                dim = parse_expr(P);
            }
        }
        expect(P, TOK_RBRACKET, "expected ']'");

        /* Wrap base_type in array<base_type, dim> */
        WgslAstNode *arr_type = new_node(P, WGSL_NODE_TYPE);
        arr_type->type_node.name = glsl_strdup("array");
        arr_type->type_node.type_arg_count = 1;
        arr_type->type_node.type_args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
        arr_type->type_node.type_args[0] = base_type;
        if (dim) {
            arr_type->type_node.expr_arg_count = 1;
            arr_type->type_node.expr_args = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            arr_type->type_node.expr_args[0] = dim;
        }
        base_type = arr_type;
    }
    return base_type;
}

/* ============================================================================
 * Declaration Parsing
 * ============================================================================ */

/* Parse a struct definition: struct Name { type field; ... }; */
static WgslAstNode *parse_struct_def(Parser *P) {
    expect(P, TOK_STRUCT, "expected 'struct'");
    if (!check(P, TOK_IDENT)) {
        parse_error(P, "expected struct name");
        return NULL;
    }
    Token name = P->cur;
    advance(P);

    char *sname = glsl_strndup(name.start, (size_t)name.length);
    register_struct(P, sname);

    WgslAstNode *S = new_node(P, WGSL_NODE_STRUCT);
    S->struct_decl.name = sname;

    expect(P, TOK_LBRACE, "expected '{'");
    int cap = 0, count = 0;
    WgslAstNode **fields = NULL;

    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        /* Optional layout qualifier on struct members */
        WgslAstNode **field_attrs = NULL;
        int field_attr_count = 0;
        if (check(P, TOK_LAYOUT))
            field_attr_count = parse_layout_qualifier(P, &field_attrs);

        WgslAstNode *ftype = parse_type_node(P);
        if (!check(P, TOK_IDENT)) {
            parse_error(P, "expected field name");
            if (ftype) wgsl_free_ast(ftype);
            break;
        }
        Token fname = P->cur;
        advance(P);

        /* C-style array dimensions on field */
        ftype = wrap_type_with_array_dims(P, ftype);

        expect(P, TOK_SEMI, "expected ';'");

        WgslAstNode *F = new_node(P, WGSL_NODE_STRUCT_FIELD);
        F->struct_field.name = glsl_strndup(fname.start, (size_t)fname.length);
        F->struct_field.type = ftype;
        F->struct_field.attr_count = field_attr_count;
        F->struct_field.attrs = field_attrs;
        vec_push_node(&fields, &count, &cap, F);
    }
    expect(P, TOK_RBRACE, "expected '}'");
    match(P, TOK_SEMI); /* optional semicolon after struct */

    S->struct_decl.field_count = count;
    S->struct_decl.fields = fields;
    return S;
}

/*
 * Parse an interface block:
 *   layout(...) storage_qual BlockName { members } instance_name;
 *
 * Decomposes into: struct BlockName + global var instance_name
 */
static void parse_interface_block(Parser *P, WgslAstNode **out_struct,
                                  WgslAstNode **out_var,
                                  WgslAstNode **attrs, int attr_count,
                                  const char *address_space) {
    *out_struct = NULL;
    *out_var = NULL;

    if (!check(P, TOK_IDENT)) {
        parse_error(P, "expected interface block name");
        return;
    }
    Token block_name = P->cur;
    advance(P);

    char *bname = glsl_strndup(block_name.start, (size_t)block_name.length);
    register_struct(P, bname);

    /* Parse block body as struct */
    WgslAstNode *S = new_node(P, WGSL_NODE_STRUCT);
    S->struct_decl.name = glsl_strdup(bname);

    expect(P, TOK_LBRACE, "expected '{'");
    int cap = 0, count = 0;
    WgslAstNode **fields = NULL;

    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        WgslAstNode **field_attrs = NULL;
        int field_attr_count = 0;
        if (check(P, TOK_LAYOUT))
            field_attr_count = parse_layout_qualifier(P, &field_attrs);

        WgslAstNode *ftype = parse_type_node(P);
        if (!check(P, TOK_IDENT)) {
            parse_error(P, "expected field name");
            if (ftype) wgsl_free_ast(ftype);
            break;
        }
        Token fname = P->cur;
        advance(P);
        ftype = wrap_type_with_array_dims(P, ftype);
        expect(P, TOK_SEMI, "expected ';'");

        WgslAstNode *F = new_node(P, WGSL_NODE_STRUCT_FIELD);
        F->struct_field.name = glsl_strndup(fname.start, (size_t)fname.length);
        F->struct_field.type = ftype;
        F->struct_field.attr_count = field_attr_count;
        F->struct_field.attrs = field_attrs;
        vec_push_node(&fields, &count, &cap, F);
    }
    expect(P, TOK_RBRACE, "expected '}'");
    S->struct_decl.field_count = count;
    S->struct_decl.fields = fields;
    *out_struct = S;

    /* Instance name (optional) */
    char *inst_name = NULL;
    if (check(P, TOK_IDENT)) {
        Token iname = P->cur;
        advance(P);
        inst_name = glsl_strndup(iname.start, (size_t)iname.length);
    } else {
        inst_name = glsl_strdup(bname);
    }
    expect(P, TOK_SEMI, "expected ';'");

    /* Create global var of block type */
    WgslAstNode *G = new_node(P, WGSL_NODE_GLOBAL_VAR);
    G->global_var.name = inst_name;
    G->global_var.type = new_type(P, bname);
    G->global_var.address_space = address_space ? glsl_strdup(address_space) : NULL;
    G->global_var.attr_count = attr_count;
    G->global_var.attrs = attrs;
    *out_var = G;

    NODE_FREE(bname);
}

/* Parse function parameters: (in/out/inout type name, ...) */
static WgslAstNode *parse_param(Parser *P) {
    /* Optional parameter qualifier */
    char *param_qual = NULL;
    if (check(P, TOK_IDENT) && (tok_eq(&P->cur, "in") || tok_eq(&P->cur, "out") ||
                                 tok_eq(&P->cur, "inout"))) {
        param_qual = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        advance(P);
    } else if (check(P, TOK_CONST)) {
        advance(P); /* skip const on params */
    }

    WgslAstNode *type = parse_type_node(P);
    if (!check(P, TOK_IDENT)) {
        parse_error(P, "expected parameter name");
        if (param_qual) NODE_FREE(param_qual);
        if (type) wgsl_free_ast(type);
        return NULL;
    }
    Token name = P->cur;
    advance(P);

    type = wrap_type_with_array_dims(P, type);

    WgslAstNode *Par = new_node(P, WGSL_NODE_PARAM);
    Par->param.name = glsl_strndup(name.start, (size_t)name.length);
    Par->param.type = type;

    /* Store parameter qualifier as attribute if not "in" (default) */
    if (param_qual && (strcmp(param_qual, "out") == 0 || strcmp(param_qual, "inout") == 0)) {
        Par->param.attr_count = 1;
        Par->param.attrs = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
        WgslAstNode *attr = new_node(P, WGSL_NODE_ATTRIBUTE);
        attr->attribute.name = param_qual;
        Par->param.attrs[0] = attr;
    } else {
        if (param_qual) NODE_FREE(param_qual);
    }
    return Par;
}

/* Parse function definition: type name(params) { body } */
static WgslAstNode *parse_function_def(Parser *P, WgslAstNode *ret_type,
                                        const char *name,
                                        WgslAstNode **attrs, int attr_count) {
    expect(P, TOK_LPAREN, "expected '('");
    int pcap = 0, pcount = 0;
    WgslAstNode **params = NULL;

    if (!check(P, TOK_RPAREN)) {
        /* Check for void parameter list: void f(void) */
        if (check(P, TOK_IDENT) && tok_eq(&P->cur, "void") && P->cur.length == 4) {
            /* peek ahead: if next is ')', this is void param list */
            Lexer L2 = P->L;
            Token t2 = lx_next(&L2);
            if (t2.type == TOK_RPAREN) {
                advance(P); /* consume "void" */
            } else {
                goto parse_params;
            }
        } else {
parse_params:;
            WgslAstNode *par = parse_param(P);
            if (par) vec_push_node(&params, &pcount, &pcap, par);
            while (match(P, TOK_COMMA)) {
                WgslAstNode *par2 = parse_param(P);
                if (par2) vec_push_node(&params, &pcount, &pcap, par2);
            }
        }
    }
    expect(P, TOK_RPAREN, "expected ')'");

    WgslAstNode *body = parse_block(P);

    WgslAstNode *F = new_node(P, WGSL_NODE_FUNCTION);
    F->function.name = glsl_strdup(name);
    F->function.param_count = pcount;
    F->function.params = params;
    F->function.return_type = ret_type;
    F->function.body = body;
    F->function.attr_count = attr_count;
    F->function.attrs = attrs;
    return F;
}

/*
 * Parse a top-level external declaration.
 * Handles: layout? interp? storage? precision? type name → function or global var
 *          struct definition
 *          interface block
 *          precision declaration
 */
static WgslAstNode *parse_external_declaration(Parser *P,
                                                WgslAstNode ***extra_decls,
                                                int *extra_count,
                                                int *extra_cap) {
    /* Layout qualifier */
    WgslAstNode **attrs = NULL;
    int attr_count = 0;
    if (check(P, TOK_LAYOUT))
        attr_count = parse_layout_qualifier(P, &attrs);

    /* Interpolation qualifier */
    char *interp = NULL;
    if (is_interp_qualifier_tok(&P->cur)) {
        interp = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        advance(P);
    }

    /* Storage qualifier */
    char *storage = NULL;
    if (is_storage_qualifier_tok(&P->cur)) {
        storage = glsl_strndup(P->cur.start, (size_t)P->cur.length);
        advance(P);
    } else if (check(P, TOK_CONST)) {
        storage = glsl_strdup("const");
        advance(P);
    }

    /* Precision qualifier (skip) */
    if (is_precision_qualifier_tok(&P->cur)) {
        advance(P); /* skip precision qualifier */
    }

    /* Precision declaration: precision highp float; */
    if (check(P, TOK_PRECISION)) {
        advance(P); /* consume 'precision' */
        if (is_precision_qualifier_tok(&P->cur))
            advance(P);
        if (check(P, TOK_IDENT))
            advance(P); /* type name */
        expect(P, TOK_SEMI, "expected ';' after precision declaration");
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        /* Free unused attrs */
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL; /* skip precision declarations */
    }

    /* Struct definition */
    if (check(P, TOK_STRUCT)) {
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return parse_struct_def(P);
    }

    /* At this point we need a type name.
     * If we have a storage qualifier and the next token is an identifier
     * that's NOT a type, it might be an interface block. */
    if (storage && check(P, TOK_IDENT) && !is_type_start(P, &P->cur)) {
        /* Could be an interface block: uniform BlockName { ... } instance; */
        /* Check if identifier is followed by '{' */
        Lexer L2 = P->L;
        Token t2 = lx_next(&L2);
        if (t2.type == TOK_LBRACE) {
            /* Map storage qualifier to address space */
            const char *addr = NULL;
            if (strcmp(storage, "uniform") == 0) addr = "uniform";
            else if (strcmp(storage, "buffer") == 0) addr = "storage";
            else if (strcmp(storage, "shared") == 0) addr = "workgroup";
            else addr = storage;

            WgslAstNode *s_node = NULL, *v_node = NULL;
            parse_interface_block(P, &s_node, &v_node, attrs, attr_count, addr);
            NODE_FREE(storage);
            if (interp) NODE_FREE(interp);

            /* Add struct as extra declaration, return the global var */
            if (s_node)
                vec_push_node(extra_decls, extra_count, extra_cap, s_node);
            return v_node;
        }
    }

    /* Handle standalone layout qualifier declaration: layout(...) in; */
    if (check(P, TOK_SEMI) && storage && attrs) {
        advance(P); /* consume ';' */
        /* Create a global var node with the attributes (e.g. workgroup_size) */
        WgslAstNode *gv = new_node(P, WGSL_NODE_GLOBAL_VAR);
        gv->global_var.name = glsl_strdup("__layout_decl");
        gv->global_var.address_space = storage; /* transfer ownership */
        gv->global_var.type = NULL;
        gv->global_var.attrs = attrs;
        gv->global_var.attr_count = attr_count;
        if (interp) NODE_FREE(interp);
        return gv;
    }

    /* Parse type */
    if (!check(P, TOK_IDENT)) {
        parse_error(P, "expected type name or declaration");
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL;
    }
    WgslAstNode *type = parse_type_node(P);

    /* Parse name */
    if (!check(P, TOK_IDENT)) {
        parse_error(P, "expected declaration name");
        if (type) wgsl_free_ast(type);
        if (interp) NODE_FREE(interp);
        if (storage) NODE_FREE(storage);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        return NULL;
    }
    Token name = P->cur;
    advance(P);
    char *decl_name = glsl_strndup(name.start, (size_t)name.length);

    /* Function definition: type name(...) { ... } */
    if (check(P, TOK_LPAREN)) {
        /* Add interpolation as attribute if present */
        if (interp) {
            WgslAstNode *iattr = new_node(P, WGSL_NODE_ATTRIBUTE);
            iattr->attribute.name = interp;
            vec_push_node(&attrs, &attr_count, &(int){0}, iattr);
        }
        WgslAstNode *fn = parse_function_def(P, type, decl_name, attrs, attr_count);
        NODE_FREE(decl_name);
        if (storage) NODE_FREE(storage);
        return fn;
    }

    /* Global variable declaration */
    type = wrap_type_with_array_dims(P, type);

    /* Optional initializer */
    WgslAstNode *init = NULL;
    if (match(P, TOK_EQ)) {
        init = parse_expr(P);
    }
    expect(P, TOK_SEMI, "expected ';'");

    /* If no storage qualifier and has init, treat as const-like (VarDecl) */
    if (!storage || strcmp(storage, "const") == 0) {
        if (storage) NODE_FREE(storage);
        if (interp) NODE_FREE(interp);
        if (attrs) {
            for (int i = 0; i < attr_count; i++)
                wgsl_free_ast(attrs[i]);
            NODE_FREE(attrs);
        }
        WgslAstNode *V = new_node(P, WGSL_NODE_VAR_DECL);
        V->var_decl.name = decl_name;
        V->var_decl.type = type;
        V->var_decl.init = init;
        return V;
    }

    /* Map storage qualifier to address space */
    char *addr_space = NULL;
    if (strcmp(storage, "in") == 0) addr_space = glsl_strdup("in");
    else if (strcmp(storage, "out") == 0) addr_space = glsl_strdup("out");
    else if (strcmp(storage, "uniform") == 0) addr_space = glsl_strdup("uniform");
    else if (strcmp(storage, "buffer") == 0) addr_space = glsl_strdup("storage");
    else if (strcmp(storage, "shared") == 0) addr_space = glsl_strdup("workgroup");
    else addr_space = glsl_strdup(storage);
    NODE_FREE(storage);

    /* Add interpolation as attribute */
    if (interp) {
        WgslAstNode *iattr = new_node(P, WGSL_NODE_ATTRIBUTE);
        iattr->attribute.name = interp;
        int acap = attr_count;
        vec_push_node(&attrs, &attr_count, &acap, iattr);
    }

    WgslAstNode *G = new_node(P, WGSL_NODE_GLOBAL_VAR);
    G->global_var.name = decl_name;
    G->global_var.type = type;
    G->global_var.address_space = addr_space;
    G->global_var.attr_count = attr_count;
    G->global_var.attrs = attrs;

    if (init) wgsl_free_ast(init); /* discard init for qualified globals */

    return G;
}

/* ============================================================================
 * Statement Parsing
 * ============================================================================ */

static WgslAstNode *parse_block(Parser *P) {
    expect(P, TOK_LBRACE, "expected '{'");
    WgslAstNode *B = new_node(P, WGSL_NODE_BLOCK);
    int cap = 0, count = 0;
    WgslAstNode **stmts = NULL;
    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        WgslAstNode *s = parse_statement(P);
        if (s) vec_push_node(&stmts, &count, &cap, s);
        else break;
    }
    expect(P, TOK_RBRACE, "expected '}'");
    B->block.stmt_count = count;
    B->block.stmts = stmts;
    return B;
}

static WgslAstNode *parse_if_stmt(Parser *P) {
    expect(P, TOK_IF, "expected 'if'");
    expect(P, TOK_LPAREN, "expected '('");
    WgslAstNode *cond = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')'");

    WgslAstNode *then_b = NULL;
    if (check(P, TOK_LBRACE))
        then_b = parse_block(P);
    else {
        /* single statement */
        then_b = new_node(P, WGSL_NODE_BLOCK);
        WgslAstNode *s = parse_statement(P);
        if (s) {
            then_b->block.stmt_count = 1;
            then_b->block.stmts = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
            then_b->block.stmts[0] = s;
        }
    }

    WgslAstNode *else_b = NULL;
    if (match(P, TOK_ELSE)) {
        if (check(P, TOK_IF))
            else_b = parse_if_stmt(P);
        else if (check(P, TOK_LBRACE))
            else_b = parse_block(P);
        else {
            else_b = new_node(P, WGSL_NODE_BLOCK);
            WgslAstNode *s = parse_statement(P);
            if (s) {
                else_b->block.stmt_count = 1;
                else_b->block.stmts = (WgslAstNode **)NODE_MALLOC(sizeof(WgslAstNode *));
                else_b->block.stmts[0] = s;
            }
        }
    }

    WgslAstNode *I = new_node(P, WGSL_NODE_IF);
    I->if_stmt.cond = cond;
    I->if_stmt.then_branch = then_b;
    I->if_stmt.else_branch = else_b;
    return I;
}

static WgslAstNode *parse_while_stmt(Parser *P) {
    expect(P, TOK_WHILE, "expected 'while'");
    expect(P, TOK_LPAREN, "expected '('");
    WgslAstNode *cond = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')'");
    WgslAstNode *body = check(P, TOK_LBRACE) ? parse_block(P) : parse_statement(P);
    WgslAstNode *W = new_node(P, WGSL_NODE_WHILE);
    W->while_stmt.cond = cond;
    W->while_stmt.body = body;
    return W;
}

static WgslAstNode *parse_do_while_stmt(Parser *P) {
    expect(P, TOK_DO, "expected 'do'");
    WgslAstNode *body = parse_block(P);
    expect(P, TOK_WHILE, "expected 'while'");
    expect(P, TOK_LPAREN, "expected '('");
    WgslAstNode *cond = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')'");
    expect(P, TOK_SEMI, "expected ';'");
    WgslAstNode *D = new_node(P, WGSL_NODE_DO_WHILE);
    D->do_while_stmt.body = body;
    D->do_while_stmt.cond = cond;
    return D;
}

static WgslAstNode *parse_for_stmt(Parser *P) {
    expect(P, TOK_FOR, "expected 'for'");
    expect(P, TOK_LPAREN, "expected '('");

    /* init */
    WgslAstNode *init = NULL;
    if (check(P, TOK_SEMI)) {
        advance(P);
    } else if (is_type_start(P, &P->cur)) {
        /* type-led declaration in for init */
        WgslAstNode *type = parse_type_node(P);
        if (!check(P, TOK_IDENT)) {
            parse_error(P, "expected variable name");
            if (type) wgsl_free_ast(type);
        } else {
            Token vname = P->cur;
            advance(P);
            type = wrap_type_with_array_dims(P, type);
            WgslAstNode *vinit = NULL;
            if (match(P, TOK_EQ))
                vinit = parse_expr(P);
            expect(P, TOK_SEMI, "expected ';'");
            init = new_node(P, WGSL_NODE_VAR_DECL);
            init->var_decl.name = glsl_strndup(vname.start, (size_t)vname.length);
            init->var_decl.type = type;
            init->var_decl.init = vinit;
        }
    } else {
        WgslAstNode *e = parse_expr(P);
        expect(P, TOK_SEMI, "expected ';'");
        init = new_node(P, WGSL_NODE_EXPR_STMT);
        init->expr_stmt.expr = e;
    }

    /* cond */
    WgslAstNode *cond = NULL;
    if (!check(P, TOK_SEMI))
        cond = parse_expr(P);
    expect(P, TOK_SEMI, "expected ';'");

    /* continue */
    WgslAstNode *cont = NULL;
    if (!check(P, TOK_RPAREN))
        cont = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')'");

    WgslAstNode *body = check(P, TOK_LBRACE) ? parse_block(P) : parse_statement(P);

    WgslAstNode *F = new_node(P, WGSL_NODE_FOR);
    F->for_stmt.init = init;
    F->for_stmt.cond = cond;
    F->for_stmt.cont = cont;
    F->for_stmt.body = body;
    return F;
}

static WgslAstNode *parse_switch_stmt(Parser *P) {
    expect(P, TOK_SWITCH, "expected 'switch'");
    expect(P, TOK_LPAREN, "expected '('");
    WgslAstNode *expr = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')'");
    expect(P, TOK_LBRACE, "expected '{'");

    int cap = 0, count = 0;
    WgslAstNode **cases = NULL;

    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        WgslAstNode *C = new_node(P, WGSL_NODE_CASE);
        if (match(P, TOK_CASE)) {
            C->case_clause.expr = parse_expr(P);
            expect(P, TOK_COLON, "expected ':'");
        } else if (match(P, TOK_DEFAULT)) {
            C->case_clause.expr = NULL;
            expect(P, TOK_COLON, "expected ':'");
        } else {
            parse_error(P, "expected 'case' or 'default'");
            wgsl_free_ast(C);
            break;
        }

        int scap = 0, scount = 0;
        WgslAstNode **stmts = NULL;
        while (!check(P, TOK_CASE) && !check(P, TOK_DEFAULT) &&
               !check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
            WgslAstNode *s = parse_statement(P);
            if (s) vec_push_node(&stmts, &scount, &scap, s);
            else break;
        }
        C->case_clause.stmt_count = scount;
        C->case_clause.stmts = stmts;
        vec_push_node(&cases, &count, &cap, C);
    }
    expect(P, TOK_RBRACE, "expected '}'");

    WgslAstNode *S = new_node(P, WGSL_NODE_SWITCH);
    S->switch_stmt.expr = expr;
    S->switch_stmt.case_count = count;
    S->switch_stmt.cases = cases;
    return S;
}

/* Try to parse a local variable declaration (type-led) */
static WgslAstNode *try_parse_local_var_decl(Parser *P) {
    WgslAstNode *type = parse_type_node(P);
    if (!check(P, TOK_IDENT)) {
        /* Not a declaration, backtrack not possible - error */
        parse_error(P, "expected variable name after type");
        if (type) wgsl_free_ast(type);
        return NULL;
    }
    Token vname = P->cur;
    advance(P);
    type = wrap_type_with_array_dims(P, type);

    WgslAstNode *init = NULL;
    if (match(P, TOK_EQ))
        init = parse_expr(P);
    expect(P, TOK_SEMI, "expected ';'");

    WgslAstNode *V = new_node(P, WGSL_NODE_VAR_DECL);
    V->var_decl.name = glsl_strndup(vname.start, (size_t)vname.length);
    V->var_decl.type = type;
    V->var_decl.init = init;
    return V;
}

static WgslAstNode *parse_statement(Parser *P) {
    /* Empty statement */
    if (match(P, TOK_SEMI))
        return NULL;

    /* Block */
    if (check(P, TOK_LBRACE))
        return parse_block(P);

    /* Control flow */
    if (check(P, TOK_IF)) return parse_if_stmt(P);
    if (check(P, TOK_WHILE)) return parse_while_stmt(P);
    if (check(P, TOK_DO)) return parse_do_while_stmt(P);
    if (check(P, TOK_FOR)) return parse_for_stmt(P);
    if (check(P, TOK_SWITCH)) return parse_switch_stmt(P);

    /* Jump statements */
    if (match(P, TOK_BREAK)) {
        expect(P, TOK_SEMI, "expected ';'");
        return new_node(P, WGSL_NODE_BREAK);
    }
    if (match(P, TOK_CONTINUE)) {
        expect(P, TOK_SEMI, "expected ';'");
        return new_node(P, WGSL_NODE_CONTINUE);
    }
    if (match(P, TOK_DISCARD)) {
        expect(P, TOK_SEMI, "expected ';'");
        return new_node(P, WGSL_NODE_DISCARD);
    }
    if (match(P, TOK_RETURN)) {
        WgslAstNode *e = NULL;
        if (!check(P, TOK_SEMI))
            e = parse_expr(P);
        expect(P, TOK_SEMI, "expected ';'");
        WgslAstNode *R = new_node(P, WGSL_NODE_RETURN);
        R->return_stmt.expr = e;
        return R;
    }

    /* const declaration inside function */
    if (check(P, TOK_CONST)) {
        advance(P);
        return try_parse_local_var_decl(P);
    }

    /* Type-led local variable declaration */
    if (is_type_keyword_tok(&P->cur)) {
        return try_parse_local_var_decl(P);
    }

    /* Check if IDENT is a known struct type followed by another IDENT */
    if (check(P, TOK_IDENT) && is_known_struct(P, &P->cur)) {
        Lexer L2 = P->L;
        Token t2 = lx_next(&L2);
        if (t2.type == TOK_IDENT) {
            return try_parse_local_var_decl(P);
        }
    }

    /* Expression statement */
    WgslAstNode *e = parse_expr(P);
    expect(P, TOK_SEMI, "expected ';'");
    WgslAstNode *ES = new_node(P, WGSL_NODE_EXPR_STMT);
    ES->expr_stmt.expr = e;
    return ES;
}

/* ============================================================================
 * Expression Parsing (Precedence Climbing)
 * ============================================================================ */

static WgslAstNode *parse_expr(Parser *P) { return parse_assignment(P); }

static WgslAstNode *parse_assignment(Parser *P) {
    WgslAstNode *left = parse_conditional(P);

    /* Simple assignment */
    if (match(P, TOK_EQ)) {
        WgslAstNode *right = parse_assignment(P);
        WgslAstNode *A = new_node(P, WGSL_NODE_ASSIGN);
        A->assign.op = glsl_strdup("=");
        A->assign.lhs = left;
        A->assign.rhs = right;
        return A;
    }

    /* Compound assignment operators */
    struct { TokenType tok; const char *op; } compounds[] = {
        {TOK_PLUSEQ, "+="}, {TOK_MINUSEQ, "-="}, {TOK_STAREQ, "*="},
        {TOK_SLASHEQ, "/="}, {TOK_PERCENTEQ, "%="},
        {TOK_AMPEQ, "&="}, {TOK_PIPEEQ, "|="}, {TOK_CARETEQ, "^="},
        {TOK_SHLEQ, "<<="}, {TOK_SHREQ, ">>="},
        {TOK_EOF, NULL}
    };
    for (int i = 0; compounds[i].op; i++) {
        if (match(P, compounds[i].tok)) {
            WgslAstNode *right = parse_assignment(P);
            WgslAstNode *A = new_node(P, WGSL_NODE_ASSIGN);
            A->assign.op = glsl_strdup(compounds[i].op);
            A->assign.lhs = left;
            A->assign.rhs = right;
            return A;
        }
    }
    return left;
}

static WgslAstNode *parse_conditional(Parser *P) {
    WgslAstNode *c = parse_logical_or(P);
    if (match(P, TOK_QMARK)) {
        WgslAstNode *t = parse_assignment(P);
        expect(P, TOK_COLON, "expected ':'");
        WgslAstNode *e = parse_assignment(P);
        WgslAstNode *T = new_node(P, WGSL_NODE_TERNARY);
        T->ternary.cond = c;
        T->ternary.then_expr = t;
        T->ternary.else_expr = e;
        return T;
    }
    return c;
}

static WgslAstNode *parse_logical_or(Parser *P) {
    WgslAstNode *left = parse_logical_and(P);
    while (match(P, TOK_OROR)) {
        WgslAstNode *right = parse_logical_and(P);
        WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("||");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode *parse_logical_and(Parser *P) {
    WgslAstNode *left = parse_bitwise_or(P);
    while (match(P, TOK_ANDAND)) {
        WgslAstNode *right = parse_bitwise_or(P);
        WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("&&");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode *parse_bitwise_or(Parser *P) {
    WgslAstNode *left = parse_bitwise_xor(P);
    while (match(P, TOK_PIPE)) {
        WgslAstNode *right = parse_bitwise_xor(P);
        WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("|");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode *parse_bitwise_xor(Parser *P) {
    WgslAstNode *left = parse_bitwise_and(P);
    while (match(P, TOK_CARET)) {
        WgslAstNode *right = parse_bitwise_and(P);
        WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("^");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode *parse_bitwise_and(Parser *P) {
    WgslAstNode *left = parse_equality(P);
    while (match(P, TOK_AMP)) {
        WgslAstNode *right = parse_equality(P);
        WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = glsl_strdup("&");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode *parse_equality(Parser *P) {
    WgslAstNode *left = parse_relational(P);
    for (;;) {
        if (match(P, TOK_EQEQ)) {
            WgslAstNode *r = parse_relational(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("==");
            B->binary.left = left;
            B->binary.right = r;
            left = B;
            continue;
        }
        if (match(P, TOK_NEQ)) {
            WgslAstNode *r = parse_relational(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("!=");
            B->binary.left = left;
            B->binary.right = r;
            left = B;
            continue;
        }
        break;
    }
    return left;
}

static WgslAstNode *parse_relational(Parser *P) {
    WgslAstNode *left = parse_shift(P);
    for (;;) {
        if (match(P, TOK_LT)) {
            WgslAstNode *r = parse_shift(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_GT)) {
            WgslAstNode *r = parse_shift(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_LE)) {
            WgslAstNode *r = parse_shift(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<=");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_GE)) {
            WgslAstNode *r = parse_shift(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">=");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

static WgslAstNode *parse_shift(Parser *P) {
    WgslAstNode *left = parse_additive(P);
    for (;;) {
        if (match(P, TOK_SHL)) {
            WgslAstNode *r = parse_additive(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("<<");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_SHR)) {
            WgslAstNode *r = parse_additive(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup(">>");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

static WgslAstNode *parse_additive(Parser *P) {
    WgslAstNode *left = parse_multiplicative(P);
    for (;;) {
        if (match(P, TOK_PLUS)) {
            WgslAstNode *r = parse_multiplicative(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("+");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_MINUS)) {
            WgslAstNode *r = parse_multiplicative(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("-");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

static WgslAstNode *parse_multiplicative(Parser *P) {
    WgslAstNode *left = parse_unary(P);
    for (;;) {
        if (match(P, TOK_STAR)) {
            WgslAstNode *r = parse_unary(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("*");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_SLASH)) {
            WgslAstNode *r = parse_unary(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("/");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        if (match(P, TOK_PERCENT)) {
            WgslAstNode *r = parse_unary(P);
            WgslAstNode *B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = glsl_strdup("%");
            B->binary.left = left; B->binary.right = r; left = B; continue;
        }
        break;
    }
    return left;
}

static WgslAstNode *parse_unary(Parser *P) {
    if (match(P, TOK_PLUSPLUS)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("++"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (match(P, TOK_MINUSMINUS)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("--"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (match(P, TOK_PLUS)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("+"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (match(P, TOK_MINUS)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("-"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (match(P, TOK_BANG)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("!"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    if (match(P, TOK_TILDE)) {
        WgslAstNode *e = parse_unary(P);
        WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
        U->unary.op = glsl_strdup("~"); U->unary.is_postfix = 0; U->unary.expr = e;
        return U;
    }
    return parse_postfix(P);
}

static WgslAstNode *parse_postfix(Parser *P) {
    WgslAstNode *expr = parse_primary(P);
    for (;;) {
        if (match(P, TOK_LPAREN)) {
            /* Function call / constructor call */
            int cap = 0, count = 0;
            WgslAstNode **args = NULL;
            if (!check(P, TOK_RPAREN)) {
                WgslAstNode *a = parse_assignment(P);
                if (a) vec_push_node(&args, &count, &cap, a);
                while (match(P, TOK_COMMA)) {
                    WgslAstNode *a2 = parse_assignment(P);
                    if (a2) vec_push_node(&args, &count, &cap, a2);
                }
            }
            expect(P, TOK_RPAREN, "expected ')'");
            WgslAstNode *C = new_node(P, WGSL_NODE_CALL);
            C->call.callee = expr;
            C->call.arg_count = count;
            C->call.args = args;
            expr = C;
            continue;
        }
        if (match(P, TOK_LBRACKET)) {
            WgslAstNode *idx = parse_expr(P);
            expect(P, TOK_RBRACKET, "expected ']'");
            WgslAstNode *I = new_node(P, WGSL_NODE_INDEX);
            I->index.object = expr;
            I->index.index = idx;
            expr = I;
            continue;
        }
        if (match(P, TOK_DOT)) {
            if (!check(P, TOK_IDENT)) {
                parse_error(P, "expected member name");
                break;
            }
            Token mem = P->cur;
            advance(P);
            WgslAstNode *M = new_node(P, WGSL_NODE_MEMBER);
            M->member.object = expr;
            M->member.member = glsl_strndup(mem.start, (size_t)mem.length);
            expr = M;
            continue;
        }
        if (match(P, TOK_PLUSPLUS)) {
            WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
            U->unary.op = glsl_strdup("++"); U->unary.is_postfix = 1; U->unary.expr = expr;
            expr = U;
            continue;
        }
        if (match(P, TOK_MINUSMINUS)) {
            WgslAstNode *U = new_node(P, WGSL_NODE_UNARY);
            U->unary.op = glsl_strdup("--"); U->unary.is_postfix = 1; U->unary.expr = expr;
            expr = U;
            continue;
        }
        break;
    }
    return expr;
}

static WgslAstNode *parse_primary(Parser *P) {
    /* Type keyword as constructor: vec3(...), mat4(...), etc. */
    if (check(P, TOK_IDENT) && is_type_keyword_tok(&P->cur)) {
        Token t = P->cur;
        /* Look ahead: if followed by '(', treat as constructor */
        Lexer L2 = P->L;
        Token t2 = lx_next(&L2);
        if (t2.type == TOK_LPAREN) {
            advance(P); /* consume type keyword */
            return new_type(P, glsl_strndup(t.start, (size_t)t.length));
        }
        /* Otherwise fall through to identifier */
    }

    /* Boolean literals */
    if (check(P, TOK_IDENT) && (tok_eq(&P->cur, "true") || tok_eq(&P->cur, "false"))) {
        Token t = P->cur;
        advance(P);
        WgslAstNode *n = new_node(P, WGSL_NODE_LITERAL);
        n->literal.lexeme = glsl_strndup(t.start, (size_t)t.length);
        n->literal.kind = WGSL_LIT_INT; /* treat booleans as int-like for now */
        return n;
    }

    /* Identifier */
    if (check(P, TOK_IDENT)) {
        Token name = P->cur;
        advance(P);
        return new_ident(P, &name);
    }

    /* Number */
    if (check(P, TOK_NUMBER)) {
        Token t = P->cur;
        advance(P);
        return new_literal(P, &t);
    }

    /* Parenthesized expression */
    if (match(P, TOK_LPAREN)) {
        WgslAstNode *e = parse_expr(P);
        expect(P, TOK_RPAREN, "expected ')'");
        return e;
    }

    parse_error(P, "expected expression");
    return new_node(P, WGSL_NODE_LITERAL);
}

/* ============================================================================
 * Program (entry point)
 * ============================================================================ */

static WgslAstNode *parse_program(Parser *P) {
    WgslAstNode *root = new_node(P, WGSL_NODE_PROGRAM);
    int cap = 0, count = 0;
    WgslAstNode **decls = NULL;

    while (!check(P, TOK_EOF)) {
        int ecap = 0, ecount = 0;
        WgslAstNode **extra = NULL;

        WgslAstNode *d = parse_external_declaration(P, &extra, &ecount, &ecap);

        /* Add any extra declarations (e.g., struct from interface block) first */
        for (int i = 0; i < ecount; i++)
            vec_push_node(&decls, &count, &cap, extra[i]);
        if (extra) NODE_FREE(extra);

        if (d)
            vec_push_node(&decls, &count, &cap, d);
        else if (P->had_error)
            break;
    }

    root->program.decl_count = count;
    root->program.decls = decls;
    return root;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

WgslAstNode *glsl_parse(const char *source) {
    Parser P;
    memset(&P, 0, sizeof(P));
    P.L.src = source ? source : "";
    P.L.pos = 0;
    P.L.line = 1;
    P.L.col = 1;
    advance(&P);

    WgslAstNode *ast = parse_program(&P);

    /* Free known type tracking */
    for (int i = 0; i < P.known_type_count; i++)
        NODE_FREE(P.known_types[i].name);
    NODE_FREE(P.known_types);

    return ast;
}
