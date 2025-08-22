/* wgsl_parser.c */
#include "wgsl_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* =========================
   Small utilities
   ========================= */

static char* wgsl_strndup(const char* s, size_t n) {
    char* r = (char*)NODE_MALLOC(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}
static char* wgsl_strdup(const char* s) {
    return wgsl_strndup(s, s ? strlen(s) : 0);
}

static void* grow_ptr_array(void* p, int needed, int* cap, size_t elem_size) {
    if (needed <= *cap) return p;
    int newcap = (*cap == 0) ? 4 : (*cap * 2);
    while (newcap < needed) newcap *= 2;
    void* np = NODE_REALLOC(p, (size_t)newcap * elem_size);
    if (!np) return p;
    *cap = newcap;
    return np;
}

static void vec_push_node(WgslAstNode*** arr, int* count, int* cap, WgslAstNode* v) {
    *arr = (WgslAstNode**)grow_ptr_array(*arr, *count + 1, cap, sizeof(WgslAstNode*));
    (*arr)[(*count)++] = v;
}

/* =========================
   Lexer
   ========================= */

typedef enum TokenType {
    TOK_EOF = 0,
    TOK_IDENT,
    TOK_NUMBER,
    TOK_AT,
    TOK_COLON,
    TOK_SEMI,
    TOK_COMMA,
    TOK_LBRACE, TOK_RBRACE,
    TOK_LPAREN, TOK_RPAREN,
    TOK_LT, TOK_GT,
    TOK_LBRACKET, TOK_RBRACKET,
    TOK_DOT,
    TOK_STAR,
    TOK_SLASH,
    TOK_PLUS,
    TOK_MINUS,
    TOK_EQ,
    TOK_ARROW,

    /* two-char operators */
    TOK_LE,      /* <= */
    TOK_GE,      /* >= */
    TOK_EQEQ,    /* == */
    TOK_NEQ,     /* != */
    TOK_ANDAND,  /* && */
    TOK_OROR,    /* || */

    /* keywords */
    TOK_STRUCT,
    TOK_FN,
    TOK_VAR,
    TOK_LET,
    TOK_CONST,
    TOK_OVERRIDE,
    TOK_RETURN,
    TOK_IF,
    TOK_ELSE,
    TOK_WHILE,
    TOK_FOR
} TokenType;





typedef struct Token {
    TokenType type;
    const char* start;
    int length;
    int line;
    int col;
    bool is_float; /* for TOK_NUMBER */
} Token;

typedef struct Lexer {
    const char* src;
    size_t pos;
    int line;
    int col;
} Lexer;

static bool lx_peek2(const Lexer* L, char a, char b) {
    return (L->src[L->pos] == a && L->src[L->pos + 1] == b);
}

static void lx_advance(Lexer* L) {
    char c = L->src[L->pos];
    if (c == '\0') return;
    L->pos++;
    if (c == '\n') {
        L->line++;
        L->col = 1;
    } else {
        L->col++;
    }
}

static void lx_skip_ws_comments(Lexer* L) {
    for (;;) {
        char c = L->src[L->pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            lx_advance(L);
            continue;
        }
        if (c == '/' && L->src[L->pos + 1] == '/') {
            /* line comment */
            while (L->src[L->pos] && L->src[L->pos] != '\n') lx_advance(L);
            continue;
        }
        if (c == '/' && L->src[L->pos + 1] == '*') {
            /* block comment */
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
        break;
    }
}

static bool is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_' ;
}
static bool is_ident_part(char c) {
    return isalnum((unsigned char)c) || c == '_' ;
}

static Token make_token(Lexer* L, TokenType t, const char* s, int len, bool is_float) {
    Token tok;
    tok.type = t;
    tok.start = s;
    tok.length = len;
    tok.line = L->line;
    tok.col = L->col;
    tok.is_float = is_float;
    return tok;
}

static Token lx_next(Lexer* L) {
    lx_skip_ws_comments(L);
    const char* s = &L->src[L->pos];
    char c = *s;
    if (!c) return make_token(L, TOK_EOF, s, 0, false);

    /* two-char punctuators: check first */
    if (lx_peek2(L, '-', '>')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_ARROW, s, 2, false); }
    if (lx_peek2(L, '<', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_LE, s, 2, false); }
    if (lx_peek2(L, '>', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_GE, s, 2, false); }
    if (lx_peek2(L, '=', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_EQEQ, s, 2, false); }
    if (lx_peek2(L, '!', '=')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_NEQ, s, 2, false); }
    if (lx_peek2(L, '&', '&')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_ANDAND, s, 2, false); }
    if (lx_peek2(L, '|', '|')) { lx_advance(L); lx_advance(L); return make_token(L, TOK_OROR, s, 2, false); }

    /* single-char */
    switch (c) {
        case '@': lx_advance(L); return make_token(L, TOK_AT, s, 1, false);
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
        case '=': lx_advance(L); return make_token(L, TOK_EQ, s, 1, false);
        default: break;
    }

    /* ... keep the rest of lx_next identical (ident/number/unknown) ... */
    /* (no change to identifier/number scanning) */
    if (is_ident_start(c)) {
        size_t p = L->pos;
        lx_advance(L);
        while (is_ident_part(L->src[L->pos])) lx_advance(L);
        /* keyword block replaced above */
        const char* start = &L->src[p];
        int len = (int)(&L->src[L->pos] - start);
        if (len == 6 && strncmp(start, "struct", 6) == 0) return make_token(L, TOK_STRUCT, start, len, false);
        if (len == 2 && strncmp(start, "fn", 2) == 0)      return make_token(L, TOK_FN, start, len, false);
        if (len == 3 && strncmp(start, "var", 3) == 0)     return make_token(L, TOK_VAR, start, len, false);
        if (len == 3 && strncmp(start, "let", 3) == 0)     return make_token(L, TOK_LET, start, len, false);
        if (len == 5 && strncmp(start, "const", 5) == 0)   return make_token(L, TOK_CONST, start, len, false);
        if (len == 8 && strncmp(start, "override", 8) == 0)return make_token(L, TOK_OVERRIDE, start, len, false);
        if (len == 6 && strncmp(start, "return", 6) == 0)  return make_token(L, TOK_RETURN, start, len, false);
        if (len == 2 && strncmp(start, "if", 2) == 0)      return make_token(L, TOK_IF, start, len, false);
        if (len == 4 && strncmp(start, "else", 4) == 0)    return make_token(L, TOK_ELSE, start, len, false);
        if (len == 5 && strncmp(start, "while", 5) == 0)   return make_token(L, TOK_WHILE, start, len, false);
        if (len == 3 && strncmp(start, "for", 3) == 0)     return make_token(L, TOK_FOR, start, len, false);
        return make_token(L, TOK_IDENT, start, len, false);
    }

    if (isdigit((unsigned char)c)) {
        size_t p = L->pos; bool has_dot = false;
        lx_advance(L);
        while (isdigit((unsigned char)L->src[L->pos])) lx_advance(L);
        if (L->src[L->pos] == '.') { has_dot = true; lx_advance(L); while (isdigit((unsigned char)L->src[L->pos])) lx_advance(L); }
        if (L->src[L->pos] == 'e' || L->src[L->pos] == 'E') { has_dot = true; lx_advance(L); if (L->src[L->pos] == '+' || L->src[L->pos] == '-') lx_advance(L); while (isdigit((unsigned char)L->src[L->pos])) lx_advance(L); }
        if (L->src[L->pos] == 'f' || L->src[L->pos] == 'F') lx_advance(L);
        const char* start = &L->src[p]; int len = (int)(&L->src[L->pos] - start);
        return make_token(L, TOK_NUMBER, start, len, has_dot);
    }

    lx_advance(L);
    return make_token(L, TOK_EOF, s, 0, false);
}



/* =========================
   Parser
   ========================= */

typedef struct Parser {
    Lexer L;
    Token cur;
    bool had_error;
} Parser;

static void advance(Parser* P) { P->cur = lx_next(&P->L); }
static bool check(Parser* P, TokenType t) { return P->cur.type == t; }
static bool match(Parser* P, TokenType t) { if (check(P,t)) { advance(P); return true; } return false; }

static void parse_error(Parser* P, const char* msg) {
    fprintf(stderr, "[wgsl-parser] error at %d:%d: %s\n", P->cur.line, P->cur.col, msg);
    P->had_error = true;
}

static void expect(Parser* P, TokenType t, const char* msg) {
    if (!match(P, t)) parse_error(P, msg);
}

/* Node creation helpers */
static WgslAstNode* new_node(Parser* P, WgslNodeType kind) {
    (void)P;
    WgslAstNode* n = NODE_ALLOC(WgslAstNode);
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->type = kind;
    n->line = P->cur.line;
    n->col  = P->cur.col;
    return n;
}

static WgslAstNode* new_ident(Parser* P, const Token* t) {
    WgslAstNode* n = new_node(P, WGSL_NODE_IDENT);
    n->ident.name = wgsl_strndup(t->start, (size_t)t->length);
    return n;
}
static WgslAstNode* new_literal(Parser* P, const Token* t) {
    WgslAstNode* n = new_node(P, WGSL_NODE_LITERAL);
    n->literal.lexeme = wgsl_strndup(t->start, (size_t)t->length);
    n->literal.kind = t->is_float ? WGSL_LIT_FLOAT : WGSL_LIT_INT;
    return n;
}
static WgslAstNode* new_type(Parser* P, const char* name) {
    WgslAstNode* n = new_node(P, WGSL_NODE_TYPE);
    n->type_node.name = wgsl_strdup(name);
    return n;
}

/* Forward decls for recursive descent */
static WgslAstNode* parse_program(Parser* P);
static WgslAstNode* parse_decl_or_stmt(Parser* P);
static WgslAstNode* parse_if_stmt(Parser* P);
static WgslAstNode* parse_while_stmt(Parser* P);
static WgslAstNode* parse_for_stmt(Parser* P);
static WgslAstNode* parse_struct(Parser* P, WgslAstNode** opt_attrs, int attr_count);
static WgslAstNode* parse_global_var(Parser* P, WgslAstNode** attrs, int attr_count);
static WgslAstNode* parse_function(Parser* P, WgslAstNode** attrs, int attr_count);
static WgslAstNode* parse_type_node(Parser* P);
static WgslAstNode* parse_attribute(Parser* P);
static int          parse_attribute_list(Parser* P, WgslAstNode*** out);
static WgslAstNode* parse_block(Parser* P);
static WgslAstNode* parse_statement(Parser* P);
static WgslAstNode* parse_param(Parser* P);
static WgslAstNode* parse_expr(Parser* P);
static WgslAstNode* parse_assignment(Parser* P);
static WgslAstNode* parse_multiplicative(Parser* P);
static WgslAstNode* parse_additive(Parser* P);
static WgslAstNode* parse_logical_or(Parser* P);
static WgslAstNode* parse_logical_and(Parser* P);
static WgslAstNode* parse_equality(Parser* P);
static WgslAstNode* parse_relational(Parser* P);
static WgslAstNode* parse_postfix(Parser* P);
static WgslAstNode* parse_primary(Parser* P);
static void         skip_optional_comma(Parser* P);
static WgslAstNode* parse_const_decl(Parser* P);
static WgslAstNode* parse_override_decl(Parser* P);
static void free_node(WgslAstNode* n);
static void discard_attrs(WgslAstNode** attrs, int count);


/* attribute_list := {@ attribute}* */
static int parse_attribute_list(Parser* P, WgslAstNode*** out) {
    int cap = 0, count = 0;
    WgslAstNode** list = NULL;
    while (match(P, TOK_AT)) {
        /* after '@' we expect attribute */
        /* Attribute := ident '(' [expr {',' expr}*] ')' | ident */
        /* rewind one token so parse_attribute sees name */
        /* Actually we already consumed '@', so just parse attribute name etc. */
        WgslAstNode* attr = parse_attribute(P);
        if (!attr) break;
        vec_push_node(&list, &count, &cap, attr);
        /* multiple attributes can stack: @a @b ... */
    }
    *out = list;
    return count;
}

static void discard_attrs(WgslAstNode** attrs, int count) {
    if (!attrs) return;
    for (int i = 0; i < count; ++i) free_node(attrs[i]);
    NODE_FREE(attrs);
}

/* const-decl := 'const' IDENT [':' type] '=' expr ';'  (initializer required) */
static WgslAstNode* parse_const_decl(Parser* P) {
    expect(P, TOK_CONST, "expected 'const'");
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected constant name"); return NULL; }
    Token name = P->cur; advance(P);

    WgslAstNode* type = NULL;
    if (match(P, TOK_COLON)) {
        type = parse_type_node(P);
    }

    if (!match(P, TOK_EQ)) {
        parse_error(P, "const declaration requires an initializer (use '= expr')");
    }
    WgslAstNode* init = parse_expr(P);

    expect(P, TOK_SEMI, "expected ';' after const declaration");

    WgslAstNode* V = new_node(P, WGSL_NODE_VAR_DECL);
    V->var_decl.name = wgsl_strndup(name.start, (size_t)name.length);
    V->var_decl.type = type;
    V->var_decl.init = init;
    return V;
}

/* override-decl := 'override' IDENT [':' type] [ '=' expr ] ';' (initializer optional) */
static WgslAstNode* parse_override_decl(Parser* P) {
    expect(P, TOK_OVERRIDE, "expected 'override'");
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected override name"); return NULL; }
    Token name = P->cur; advance(P);

    WgslAstNode* type = NULL;
    if (match(P, TOK_COLON)) {
        type = parse_type_node(P);
    }

    WgslAstNode* init = NULL;
    if (match(P, TOK_EQ)) {
        init = parse_expr(P);
    }

    expect(P, TOK_SEMI, "expected ';' after override declaration");

    WgslAstNode* V = new_node(P, WGSL_NODE_VAR_DECL);
    V->var_decl.name = wgsl_strndup(name.start, (size_t)name.length);
    V->var_decl.type = type;
    V->var_decl.init = init; /* may be NULL */
    return V;
}



static WgslAstNode* parse_attribute(Parser* P) {
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected attribute name after '@'"); return NULL; }
    Token name = P->cur; advance(P);
    WgslAstNode* A = new_node(P, WGSL_NODE_ATTRIBUTE);
    A->attribute.name = wgsl_strndup(name.start, (size_t)name.length);
    int cap = 0, count = 0; WgslAstNode** args = NULL;
    if (match(P, TOK_LPAREN)) {
        if (!check(P, TOK_RPAREN)) {
            /* parse first expr */
            WgslAstNode* e = parse_expr(P);
            if (e) vec_push_node(&args, &count, &cap, e);
            while (match(P, TOK_COMMA)) {
                WgslAstNode* e2 = parse_expr(P);
                if (e2) vec_push_node(&args, &count, &cap, e2);
            }
        }
        expect(P, TOK_RPAREN, "expected ')' to close attribute arguments");
    }
    A->attribute.arg_count = count;
    A->attribute.args = args;
    return A;
}

/* type := ident [ '<' type_args '>' ]
   type_args := (type (',' expr)? (',' type (',' expr)? )* )?   (loose, but works for array<T,N> and texture_2d<f32>) */
static WgslAstNode* parse_type_node(Parser* P) {
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected type name"); return NULL; }
    Token name = P->cur; advance(P);
    char* tname = wgsl_strndup(name.start, (size_t)name.length);
    WgslAstNode* T = new_type(P, tname);
    NODE_FREE(tname);

    if (match(P, TOK_LT)) {
        /* allow a comma-separated list of entries; each entry may be a type, optionally followed by ',' expr (for array size) */
        int tcap = 0, tcount = 0; WgslAstNode** targs = NULL;
        int ecap = 0, ecount = 0; WgslAstNode** eargs = NULL;

        if (!check(P, TOK_GT)) {
            /* parse first type */
            WgslAstNode* t = parse_type_node(P);
            if (t) vec_push_node(&targs, &tcount, &tcap, t);

            /* If we see a comma and then something that's not a type, treat it as expr (array size) */
            if (match(P, TOK_COMMA)) {
                /* could be another type (for multiple generics) or an expression (for array size) */
                if (check(P, TOK_IDENT)) {
                    /* Heuristic: try to parse a type; if it fails later, no big deal for sample */
                    /* For array<T,N>, N is number; for texture_2d<f32>, this branch won't be taken since there's no comma */
                    /* Look ahead: if it's clearly a number or '(' etc., parse expr; else try type */
                }
                /* parse expression (array size) */
                WgslAstNode* ex = parse_expr(P);
                if (ex) { vec_push_node(&eargs, &ecount, &ecap, ex); }
                /* After expr, more ', type' pairs allowed (rare; skip for simplicity) */
            }

            while (match(P, TOK_COMMA)) {
                /* additional type args (e.g., hypothetically multi generics) */
                WgslAstNode* t2 = parse_type_node(P);
                if (t2) vec_push_node(&targs, &tcount, &tcap, t2);
            }
        }
        expect(P, TOK_GT, "expected '>' to close type arguments");

        T->type_node.type_arg_count = tcount;
        T->type_node.type_args = targs;
        T->type_node.expr_arg_count = ecount;
        T->type_node.expr_args = eargs;
    }
    return T;
}

/* struct-decl := 'struct' IDENT '{' fields '}' ';'
   field := [attributes] IDENT ':' type [',' ] */
static WgslAstNode* parse_struct(Parser* P, WgslAstNode** opt_attrs, int attr_count) {
    expect(P, TOK_STRUCT, "expected 'struct'");
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected struct name"); return NULL; }
    Token name = P->cur; advance(P);

    WgslAstNode* S = new_node(P, WGSL_NODE_STRUCT);
    S->struct_decl.name = wgsl_strndup(name.start, (size_t)name.length);

    /* attach any pre-read attributes (not used in sample) */
    S->struct_decl.attr_count = attr_count;
    S->struct_decl.attrs = opt_attrs;

    expect(P, TOK_LBRACE, "expected '{' to start struct body");

    int cap = 0, count = 0;
    WgslAstNode** fields = NULL;

    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        /* optional attributes before field */
        WgslAstNode** attrs = NULL; int acount = parse_attribute_list(P, &attrs);

        if (!check(P, TOK_IDENT)) { parse_error(P, "expected field name"); break; }
        Token fname = P->cur; advance(P);

        expect(P, TOK_COLON, "expected ':' after field name");
        WgslAstNode* ftype = parse_type_node(P);

        /* optional comma after field */
        skip_optional_comma(P);

        WgslAstNode* F = new_node(P, WGSL_NODE_STRUCT_FIELD);
        F->struct_field.name = wgsl_strndup(fname.start, (size_t)fname.length);
        F->struct_field.type = ftype;
        F->struct_field.attr_count = acount;
        F->struct_field.attrs = attrs;

        vec_push_node(&fields, &count, &cap, F);
    }

    expect(P, TOK_RBRACE, "expected '}' after struct fields");
    expect(P, TOK_SEMI, "expected ';' after struct declaration");

    S->struct_decl.field_count = count;
    S->struct_decl.fields = fields;
    return S;
}

/* replacement for parse_global_var in wgsl_parser.c */
/* global-var := [attributes] 'var' [ '<' (address_space [',' access_mode] | access_mode ',' address_space) '>' ]
                 IDENT ':' type ';' */
static WgslAstNode* parse_global_var(Parser* P, WgslAstNode** attrs, int attr_count) {
    expect(P, TOK_VAR, "expected 'var'");

    char* addr_space = NULL;   /* we keep this in the AST (existing field) */
    /* char* access_mode = NULL; */ /* uncomment + add to GlobalVar if you want to store it */

    if (match(P, TOK_LT)) {
        /* Collect one or two identifiers inside the angle brackets. */
        if (!check(P, TOK_IDENT)) {
            parse_error(P, "expected identifier inside '<>' after 'var'");
        }
        Token firstTok = P->cur; advance(P);
        char* first = wgsl_strndup(firstTok.start, (size_t)firstTok.length);

        char* second = NULL;
        if (match(P, TOK_COMMA)) {
            if (!check(P, TOK_IDENT)) parse_error(P, "expected identifier after ',' in var<...>");
            Token secondTok = P->cur; advance(P);
            second = wgsl_strndup(secondTok.start, (size_t)secondTok.length);
        }

        expect(P, TOK_GT, "expected '>' after var<...>");

        /* Classify strings: address space vs access mode, and accept either order. */
        int first_is_access =
            (strcmp(first, "read") == 0) || (strcmp(first, "write") == 0) || (strcmp(first, "read_write") == 0);
        int second_is_access = second ? (
            (strcmp(second, "read") == 0) || (strcmp(second, "write") == 0) || (strcmp(second, "read_write") == 0)
        ) : 0;

        /* Common WGSL address spaces to recognize */
        int first_is_addr =
            (strcmp(first, "uniform") == 0) || (strcmp(first, "storage") == 0) ||
            (strcmp(first, "workgroup") == 0) || (strcmp(first, "private") == 0);
        int second_is_addr = second ? (
            (strcmp(second, "uniform") == 0) || (strcmp(second, "storage") == 0) ||
            (strcmp(second, "workgroup") == 0) || (strcmp(second, "private") == 0)
        ) : 0;

        if (first_is_addr) {
            addr_space = first;   /* keep */
            /* if (second_is_access) access_mode = second; else free it */
            if (second) NODE_FREE(second);
        } else if (second && second_is_addr) {
            addr_space = second;  /* keep */
            /* if (first_is_access) access_mode = first; else free it */
            NODE_FREE(first);
        } else {
            /* Fallback: keep the first token as the "address_space" string to preserve information. */
            addr_space = first;
            if (second) NODE_FREE(second);
        }
    }

    if (!check(P, TOK_IDENT)) { parse_error(P, "expected variable name"); return NULL; }
    Token name = P->cur; advance(P);

    expect(P, TOK_COLON, "expected ':' after variable name");
    WgslAstNode* T = parse_type_node(P);
    expect(P, TOK_SEMI, "expected ';' after global var");

    WgslAstNode* G = new_node(P, WGSL_NODE_GLOBAL_VAR);
    G->global_var.attr_count = attr_count;
    G->global_var.attrs = attrs;
    G->global_var.address_space = addr_space; /* access_mode ignored by AST on purpose */
    G->global_var.name = wgsl_strndup(name.start, (size_t)name.length);
    G->global_var.type = T;
    return G;
}

/* param := [attributes] IDENT ':' type */
static WgslAstNode* parse_param(Parser* P) {
    WgslAstNode** attrs = NULL; int acount = parse_attribute_list(P, &attrs);
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected parameter name"); return NULL; }
    Token name = P->cur; advance(P);
    expect(P, TOK_COLON, "expected ':' after parameter name");
    WgslAstNode* T = parse_type_node(P);

    WgslAstNode* Par = new_node(P, WGSL_NODE_PARAM);
    Par->param.attr_count = acount;
    Par->param.attrs = attrs;
    Par->param.name = wgsl_strndup(name.start, (size_t)name.length);
    Par->param.type = T;
    return Par;
}

/* function := [attributes] 'fn' IDENT '(' [param {',' param}*] ')' [ '->' [attributes] type ] block */
static WgslAstNode* parse_function(Parser* P, WgslAstNode** attrs, int attr_count) {
    expect(P, TOK_FN, "expected 'fn'");
    if (!check(P, TOK_IDENT)) { parse_error(P, "expected function name"); return NULL; }
    Token name = P->cur; advance(P);

    expect(P, TOK_LPAREN, "expected '(' after function name");
    int pcap = 0, pcount = 0; WgslAstNode** params = NULL;
    if (!check(P, TOK_RPAREN)) {
        WgslAstNode* par = parse_param(P);
        if (par) vec_push_node(&params, &pcount, &pcap, par);
        while (match(P, TOK_COMMA)) {
            WgslAstNode* par2 = parse_param(P);
            if (par2) vec_push_node(&params, &pcount, &pcap, par2);
        }
    }
    expect(P, TOK_RPAREN, "expected ')' after parameter list");

    WgslAstNode** ret_attrs = NULL; int ret_acount = 0; WgslAstNode* ret_type = NULL;
    if (match(P, TOK_ARROW)) {
        ret_acount = parse_attribute_list(P, &ret_attrs);
        ret_type = parse_type_node(P);
    }

    WgslAstNode* body = parse_block(P);

    WgslAstNode* F = new_node(P, WGSL_NODE_FUNCTION);
    F->function.attr_count = attr_count;
    F->function.attrs = attrs;
    F->function.name = wgsl_strndup(name.start, (size_t)name.length);
    F->function.param_count = pcount;
    F->function.params = params;
    F->function.ret_attr_count = ret_acount;
    F->function.ret_attrs = ret_attrs;
    F->function.return_type = ret_type;
    F->function.body = body;
    return F;
}

/* block := '{' { statement } '}' */
static WgslAstNode* parse_block(Parser* P) {
    expect(P, TOK_LBRACE, "expected '{' to start block");
    WgslAstNode* B = new_node(P, WGSL_NODE_BLOCK);
    int cap = 0, count = 0; WgslAstNode** stmts = NULL;

    while (!check(P, TOK_RBRACE) && !check(P, TOK_EOF)) {
        WgslAstNode* s = parse_statement(P);
        if (s) vec_push_node(&stmts, &count, &cap, s);
    }
    expect(P, TOK_RBRACE, "expected '}' to close block");
    B->block.stmt_count = count;
    B->block.stmts = stmts;
    return B;
}

/* statement := 'var' IDENT ':' type ['=' expr] ';'
              | 'return' [expr] ';'
              | expr ';' */
static WgslAstNode* parse_statement(Parser* P) {
    /* control flow first */
    if (check(P, TOK_IF))    return parse_if_stmt(P);
    if (check(P, TOK_WHILE)) return parse_while_stmt(P);
    if (check(P, TOK_FOR))   return parse_for_stmt(P);

    /* local var (mutable) */
    if (check(P, TOK_VAR)) {
        advance(P);
        if (!check(P, TOK_IDENT)) { parse_error(P, "expected local var name"); return NULL; }
        Token name = P->cur; advance(P);

        WgslAstNode* type = NULL;
        if (match(P, TOK_COLON)) type = parse_type_node(P);

        WgslAstNode* init = NULL;
        if (match(P, TOK_EQ)) init = parse_expr(P);

        expect(P, TOK_SEMI, "expected ';' after local var");
        WgslAstNode* V = new_node(P, WGSL_NODE_VAR_DECL);
        V->var_decl.name = wgsl_strndup(name.start, (size_t)name.length);
        V->var_decl.type = type;
        V->var_decl.init = init;
        return V;
    }

    /* local let (immutable) — initializer required */
    if (check(P, TOK_LET)) {
        advance(P);
        if (!check(P, TOK_IDENT)) { parse_error(P, "expected local let name"); return NULL; }
        Token name = P->cur; advance(P);

        WgslAstNode* type = NULL;
        if (match(P, TOK_COLON)) type = parse_type_node(P);

        if (!match(P, TOK_EQ)) parse_error(P, "let declaration requires an initializer");
        WgslAstNode* init = parse_expr(P);

        expect(P, TOK_SEMI, "expected ';' after let declaration");
        WgslAstNode* V = new_node(P, WGSL_NODE_VAR_DECL);
        V->var_decl.name = wgsl_strndup(name.start, (size_t)name.length);
        V->var_decl.type = type;
        V->var_decl.init = init;
        return V;
    }

    if (check(P, TOK_RETURN)) {
        advance(P);
        WgslAstNode* e = NULL;
        if (!check(P, TOK_SEMI)) e = parse_expr(P);
        expect(P, TOK_SEMI, "expected ';' after return");
        WgslAstNode* R = new_node(P, WGSL_NODE_RETURN);
        R->return_stmt.expr = e;
        return R;
    }

    /* expr-stmt */
    WgslAstNode* e = parse_expr(P);
    expect(P, TOK_SEMI, "expected ';' after expression");
    WgslAstNode* ES = new_node(P, WGSL_NODE_EXPR_STMT);
    ES->expr_stmt.expr = e;
    return ES;
}

/* if-stmt := 'if' '(' expr ')' block [ 'else' ( block | if-stmt ) ] */
static WgslAstNode* parse_if_stmt(Parser* P) {
    expect(P, TOK_IF, "expected 'if'");
    expect(P, TOK_LPAREN, "expected '(' after 'if'");
    WgslAstNode* cond = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')' after if condition");
    WgslAstNode* then_b = parse_block(P);

    WgslAstNode* else_b = NULL;
    if (match(P, TOK_ELSE)) {
        if (check(P, TOK_IF)) {
            else_b = parse_if_stmt(P);   /* else-if chains nest IF nodes */
        } else {
            else_b = parse_block(P);
        }
    }

    WgslAstNode* I = new_node(P, WGSL_NODE_IF);
    I->if_stmt.cond = cond;
    I->if_stmt.then_branch = then_b;
    I->if_stmt.else_branch = else_b;
    return I;
}

/* while-stmt := 'while' '(' expr ')' block */
static WgslAstNode* parse_while_stmt(Parser* P) {
    expect(P, TOK_WHILE, "expected 'while'");
    expect(P, TOK_LPAREN, "expected '(' after 'while'");
    WgslAstNode* cond = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')' after while condition");
    WgslAstNode* body = parse_block(P);

    WgslAstNode* W = new_node(P, WGSL_NODE_WHILE);
    W->while_stmt.cond = cond;
    W->while_stmt.body = body;
    return W;
}

/* for-stmt := 'for' '(' [init] ';' [cond] ';' [cont] ')' block
   init can be: local 'var'/'let' declaration, or an expression statement.
   cond, cont are expressions (optional). */
static WgslAstNode* parse_for_stmt(Parser* P) {
    expect(P, TOK_FOR, "expected 'for'");
    expect(P, TOK_LPAREN, "expected '(' after 'for'");

    WgslAstNode* init = NULL;

    if (check(P, TOK_SEMI)) {
        advance(P); /* empty init */
    } else if (check(P, TOK_VAR) || check(P, TOK_LET)) {
        /* reuse statement parser (it consumes the trailing ';') */
        init = parse_statement(P);
    } else {
        /* expression initializer ending with ';' */
        WgslAstNode* e = parse_expr(P);
        expect(P, TOK_SEMI, "expected ';' after for-init expression");
        WgslAstNode* es = new_node(P, WGSL_NODE_EXPR_STMT);
        es->expr_stmt.expr = e;
        init = es;
    }

    WgslAstNode* cond = NULL;
    if (!check(P, TOK_SEMI)) cond = parse_expr(P);
    expect(P, TOK_SEMI, "expected ';' after for condition");

    WgslAstNode* cont = NULL;
    if (!check(P, TOK_RPAREN)) cont = parse_expr(P);
    expect(P, TOK_RPAREN, "expected ')' after for header");

    WgslAstNode* body = parse_block(P);

    WgslAstNode* F = new_node(P, WGSL_NODE_FOR);
    F->for_stmt.init = init;
    F->for_stmt.cond = cond;
    F->for_stmt.cont = cont;
    F->for_stmt.body = body;
    return F;
}



/* expressions with precedence:
   assignment:  lhs '=' rhs
   multiplicative: term ('*' term)*
   postfix: primary ( '(' args ')' | '[' expr ']' | '.' ident )*
   primary: IDENT | NUMBER | '(' expr ')'
*/
static WgslAstNode* parse_expr(Parser* P) {
    return parse_assignment(P);
}


static WgslAstNode* parse_assignment(Parser* P) {
    WgslAstNode* left = parse_logical_or(P);
    if (match(P, TOK_EQ)) {
        WgslAstNode* right = parse_assignment(P); /* right associative */
        WgslAstNode* A = new_node(P, WGSL_NODE_ASSIGN);
        A->assign.lhs = left;
        A->assign.rhs = right;
        return A;
    }
    return left;
}


static WgslAstNode* parse_logical_or(Parser* P) {
    WgslAstNode* left = parse_logical_and(P);
    while (match(P, TOK_OROR)) {
        WgslAstNode* right = parse_logical_and(P);
        WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = wgsl_strdup("||");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode* parse_logical_and(Parser* P) {
    WgslAstNode* left = parse_equality(P);
    while (match(P, TOK_ANDAND)) {
        WgslAstNode* right = parse_equality(P);
        WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
        B->binary.op = wgsl_strdup("&&");
        B->binary.left = left;
        B->binary.right = right;
        left = B;
    }
    return left;
}

static WgslAstNode* parse_equality(Parser* P) {
    WgslAstNode* left = parse_relational(P);
    for (;;) {
        if (match(P, TOK_EQEQ)) {
            WgslAstNode* right = parse_relational(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("==");
            B->binary.left = left; B->binary.right = right; left = B; continue;
        }
        if (match(P, TOK_NEQ)) {
            WgslAstNode* right = parse_relational(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("!=");
            B->binary.left = left; B->binary.right = right; left = B; continue;
        }
        break;
    }
    return left;
}

static WgslAstNode* parse_relational(Parser* P) {
    WgslAstNode* left = parse_additive(P);
    for (;;) {
        if (match(P, TOK_LT))  { WgslAstNode* r = parse_additive(P); WgslAstNode* B = new_node(P, WGSL_NODE_BINARY); B->binary.op = wgsl_strdup("<");  B->binary.left = left; B->binary.right = r; left = B; continue; }
        if (match(P, TOK_GT))  { WgslAstNode* r = parse_additive(P); WgslAstNode* B = new_node(P, WGSL_NODE_BINARY); B->binary.op = wgsl_strdup(">");  B->binary.left = left; B->binary.right = r; left = B; continue; }
        if (match(P, TOK_LE))  { WgslAstNode* r = parse_additive(P); WgslAstNode* B = new_node(P, WGSL_NODE_BINARY); B->binary.op = wgsl_strdup("<="); B->binary.left = left; B->binary.right = r; left = B; continue; }
        if (match(P, TOK_GE))  { WgslAstNode* r = parse_additive(P); WgslAstNode* B = new_node(P, WGSL_NODE_BINARY); B->binary.op = wgsl_strdup(">="); B->binary.left = left; B->binary.right = r; left = B; continue; }
        break;
    }
    return left;
}



static WgslAstNode* parse_additive(Parser* P) {
    WgslAstNode* left = parse_multiplicative(P);
    for (;;) {
        if (match(P, TOK_PLUS)) {
            WgslAstNode* right = parse_multiplicative(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("+");
            B->binary.left = left;
            B->binary.right = right;
            left = B;
            continue;
        }
        if (match(P, TOK_MINUS)) {
            WgslAstNode* right = parse_multiplicative(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("-");
            B->binary.left = left;
            B->binary.right = right;
            left = B;
            continue;
        }
        break;
    }
    return left;
}

/* replace the old parse_multiplicative with this */
static WgslAstNode* parse_multiplicative(Parser* P) {
    WgslAstNode* left = parse_postfix(P);
    for (;;) {
        if (match(P, TOK_STAR)) {
            WgslAstNode* right = parse_postfix(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("*");
            B->binary.left = left;
            B->binary.right = right;
            left = B;
            continue;
        }
        if (match(P, TOK_SLASH)) {
            WgslAstNode* right = parse_postfix(P);
            WgslAstNode* B = new_node(P, WGSL_NODE_BINARY);
            B->binary.op = wgsl_strdup("/");
            B->binary.left = left;
            B->binary.right = right;
            left = B;
            continue;
        }
        break;
    }
    return left;
}


static WgslAstNode* parse_postfix(Parser* P) {
    WgslAstNode* expr = parse_primary(P);
    for (;;) {
        if (match(P, TOK_LPAREN)) {
            /* call */
            int cap = 0, count = 0; WgslAstNode** args = NULL;
            if (!check(P, TOK_RPAREN)) {
                WgslAstNode* a = parse_expr(P);
                if (a) vec_push_node(&args, &count, &cap, a);
                while (match(P, TOK_COMMA)) {
                    WgslAstNode* a2 = parse_expr(P);
                    if (a2) vec_push_node(&args, &count, &cap, a2);
                }
            }
            expect(P, TOK_RPAREN, "expected ')' after arguments");
            WgslAstNode* C = new_node(P, WGSL_NODE_CALL);
            C->call.callee = expr;
            C->call.arg_count = count;
            C->call.args = args;
            expr = C;
            continue;
        }
        if (match(P, TOK_LBRACKET)) {
            /* index */
            WgslAstNode* idx = parse_expr(P);
            expect(P, TOK_RBRACKET, "expected ']' after index");
            WgslAstNode* I = new_node(P, WGSL_NODE_INDEX);
            I->index.object = expr;
            I->index.index = idx;
            expr = I;
            continue;
        }
        if (match(P, TOK_DOT)) {
            if (!check(P, TOK_IDENT)) { parse_error(P, "expected member after '.'"); break; }
            Token mem = P->cur; advance(P);
            WgslAstNode* M = new_node(P, WGSL_NODE_MEMBER);
            M->member.object = expr;
            M->member.member = wgsl_strndup(mem.start, (size_t)mem.length);
            expr = M;
            continue;
        }
        break;
    }
    return expr;
}

static WgslAstNode* parse_primary(Parser* P) {
    if (check(P, TOK_IDENT)) {
        Token t = P->cur; advance(P);
        return new_ident(P, &t);
    }
    if (check(P, TOK_NUMBER)) {
        Token t = P->cur; advance(P);
        return new_literal(P, &t);
    }
    if (match(P, TOK_LPAREN)) {
        WgslAstNode* e = parse_expr(P);
        expect(P, TOK_RPAREN, "expected ')'");
        return e;
    }
    parse_error(P, "expected expression");
    return new_node(P, WGSL_NODE_LITERAL); /* dummy */
}

static void skip_optional_comma(Parser* P) {
    (void)match(P, TOK_COMMA);
}

/* top-level:
   program := { [attributes] (struct | var | fn) }
*/
/* replace your whole parse_decl_or_stmt with this */
static WgslAstNode* parse_decl_or_stmt(Parser* P) {
    /* attributes may precede top-level decls */
    WgslAstNode** attrs = NULL; int acount = parse_attribute_list(P, &attrs);

    if (check(P, TOK_STRUCT)) {
        return parse_struct(P, attrs, acount);
    }
    if (check(P, TOK_VAR)) {
        return parse_global_var(P, attrs, acount);
    }
    if (check(P, TOK_FN)) {
        return parse_function(P, attrs, acount);
    }
    if (check(P, TOK_CONST)) {
        if (acount) discard_attrs(attrs, acount);  /* VAR_DECL has no attr slots */
        return parse_const_decl(P);
    }
    if (check(P, TOK_OVERRIDE)) {
        if (acount) discard_attrs(attrs, acount);  /* drop attrs for now */
        return parse_override_decl(P);
    }

    parse_error(P, "expected 'struct', 'var', 'fn', 'const', or 'override' at top level");
    if (acount) discard_attrs(attrs, acount);
    return NULL;
}


static WgslAstNode* parse_program(Parser* P) {
    WgslAstNode* root = new_node(P, WGSL_NODE_PROGRAM);
    int cap = 0, count = 0; WgslAstNode** decls = NULL;
    while (!check(P, TOK_EOF)) {
        WgslAstNode* d = parse_decl_or_stmt(P);
        if (d) vec_push_node(&decls, &count, &cap, d);
        else break;
    }
    root->program.decl_count = count;
    root->program.decls = decls;
    return root;
}



/* =========================
   Public API
   ========================= */



WgslAstNode* wgsl_parse(const char *source) {
    Parser P;
    memset(&P, 0, sizeof(P));
    P.L.src = source ? source : "";
    P.L.pos = 0;
    P.L.line = 1;
    P.L.col = 1;
    advance(&P); /* prime first token */
    WgslAstNode* ast = parse_program(&P);
    (void)P.had_error; /* you can inspect or elevate to NULL if desired */
    return ast;
}

static void free_node(WgslAstNode* n);

static void free_node_list(WgslAstNode** list, int count) {
    if (!list) return;
    for (int i = 0; i < count; ++i) free_node(list[i]);
    NODE_FREE(list);
}

static void free_string(char* s) { if (s) NODE_FREE(s); }

static void free_attribute(Attribute* a) {
    free_string(a->name);
    free_node_list(a->args, a->arg_count);
}

static void free_type(TypeNode* t) {
    free_string(t->name);
    free_node_list(t->type_args, t->type_arg_count);
    free_node_list(t->expr_args, t->expr_arg_count);
}

static void free_struct_field(StructField* f) {
    free_node_list(f->attrs, f->attr_count);
    free_string(f->name);
    free_node(f->type);
}

static void free_struct_decl(StructDecl* s) {
    free_node_list(s->attrs, s->attr_count);
    free_string(s->name);
    free_node_list(s->fields, s->field_count);
}

static void free_global_var(GlobalVar* g) {
    free_node_list(g->attrs, g->attr_count);
    free_string(g->address_space);
    free_string(g->name);
    free_node(g->type);
}

static void free_param(Param* p) {
    free_node_list(p->attrs, p->attr_count);
    free_string(p->name);
    free_node(p->type);
}

static void free_block(Block* b) {
    free_node_list(b->stmts, b->stmt_count);
}

static void free_var_decl(VarDecl* v) {
    free_string(v->name);
    free_node(v->type);
    free_node(v->init);
}

static void free_function(Function* f) {
    free_node_list(f->attrs, f->attr_count);
    free_string(f->name);
    free_node_list(f->params, f->param_count);
    free_node_list(f->ret_attrs, f->ret_attr_count);
    free_node(f->return_type);
    free_node(f->body);
}

static void free_binary(Binary* b) {
    free_string(b->op);
    free_node(b->left);
    free_node(b->right);
}

static void free_call(Call* c) {
    free_node(c->callee);
    free_node_list(c->args, c->arg_count);
}

static void free_member(Member* m) {
    free_node(m->object);
    free_string(m->member);
}

static void free_index(Index* i) {
    free_node(i->object);
    free_node(i->index);
}

static void free_node(WgslAstNode* n) {
    if (!n) return;
    switch (n->type) {
        case WGSL_NODE_PROGRAM:
            free_node_list(n->program.decls, n->program.decl_count); break;
        case WGSL_NODE_ATTRIBUTE:
            free_attribute(&n->attribute); break;
        case WGSL_NODE_TYPE:
            free_type(&n->type_node); break;
        case WGSL_NODE_STRUCT:
            free_struct_decl(&n->struct_decl); break;
        case WGSL_NODE_STRUCT_FIELD:
            free_struct_field(&n->struct_field); break;
        case WGSL_NODE_GLOBAL_VAR:
            free_global_var(&n->global_var); break;
        case WGSL_NODE_FUNCTION:
            free_function(&n->function); break;
        case WGSL_NODE_PARAM:
            free_param(&n->param); break;
        case WGSL_NODE_BLOCK:
            free_block(&n->block); break;
        case WGSL_NODE_VAR_DECL:
            free_var_decl(&n->var_decl); break;
        case WGSL_NODE_RETURN:
            free_node(n->return_stmt.expr); break;
        case WGSL_NODE_EXPR_STMT:
            free_node(n->expr_stmt.expr); break;
        case WGSL_NODE_IDENT:
            free_string(n->ident.name); break;
        case WGSL_NODE_LITERAL:
            free_string(n->literal.lexeme); break;
        case WGSL_NODE_BINARY:
            free_binary(&n->binary); break;
        case WGSL_NODE_ASSIGN:
            free_node(n->assign.lhs);
            free_node(n->assign.rhs); break;
        case WGSL_NODE_CALL:
            free_call(&n->call); break;
        case WGSL_NODE_MEMBER:
            free_member(&n->member); break;
        case WGSL_NODE_INDEX:
            free_index(&n->index); break;
        case WGSL_NODE_IF:
            free_node(n->if_stmt.cond);
            free_node(n->if_stmt.then_branch);
            free_node(n->if_stmt.else_branch);
            break;
        case WGSL_NODE_WHILE:
            free_node(n->while_stmt.cond);
            free_node(n->while_stmt.body);
            break;
        case WGSL_NODE_FOR:
            free_node(n->for_stmt.init);
            free_node(n->for_stmt.cond);
            free_node(n->for_stmt.cont);
            free_node(n->for_stmt.body);
            break;

        default: break;
    }
    NODE_FREE(n);
}

void wgsl_free_ast(WgslAstNode *node) { free_node(node); }

/* =========================
   Helpers
   ========================= */

const char* wgsl_node_type_name(WgslNodeType t) {
    switch (t) {
        case WGSL_NODE_PROGRAM: return "PROGRAM";
        case WGSL_NODE_STRUCT: return "STRUCT";
        case WGSL_NODE_STRUCT_FIELD: return "STRUCT_FIELD";
        case WGSL_NODE_GLOBAL_VAR: return "GLOBAL_VAR";
        case WGSL_NODE_FUNCTION: return "FUNCTION";
        case WGSL_NODE_PARAM: return "PARAM";
        case WGSL_NODE_TYPE: return "TYPE";
        case WGSL_NODE_ATTRIBUTE: return "ATTRIBUTE";
        case WGSL_NODE_BLOCK: return "BLOCK";
        case WGSL_NODE_VAR_DECL: return "VAR_DECL";
        case WGSL_NODE_RETURN: return "RETURN";
        case WGSL_NODE_EXPR_STMT: return "EXPR_STMT";
        case WGSL_NODE_IDENT: return "IDENT";
        case WGSL_NODE_LITERAL: return "LITERAL";
        case WGSL_NODE_BINARY: return "BINARY";
        case WGSL_NODE_ASSIGN: return "ASSIGN";
        case WGSL_NODE_CALL: return "CALL";
        case WGSL_NODE_MEMBER: return "MEMBER";
        case WGSL_NODE_INDEX: return "INDEX";
        default: return "UNKNOWN";
    }
}

/* Very lightweight debug printer (optional). */
static void print_indent(int n) { for (int i=0;i<n;i++) fputs("  ", stdout); }
static void dbg_print_node(const WgslAstNode* n, int ind);

static void dbg_print_list(const char* label, WgslAstNode* const* list, int count, int ind) {
    print_indent(ind); printf("%s (%d):\n", label, count);
    for (int i=0;i<count;i++) dbg_print_node(list[i], ind+1);
}

static void dbg_print_node(const WgslAstNode* n, int ind) {
    if (!n) { print_indent(ind); puts("(null)"); return; }
    print_indent(ind); printf("%s\n", wgsl_node_type_name(n->type));
    switch (n->type) {
        case WGSL_NODE_PROGRAM:
            dbg_print_list("decls", n->program.decls, n->program.decl_count, ind+1); break;
        case WGSL_NODE_STRUCT:
            print_indent(ind+1); printf("name: %s\n", n->struct_decl.name);
            dbg_print_list("fields", n->struct_decl.fields, n->struct_decl.field_count, ind+1);
            break;
        case WGSL_NODE_STRUCT_FIELD:
            print_indent(ind+1); printf("name: %s\n", n->struct_field.name);
            print_indent(ind+1); printf("type:\n");
            dbg_print_node(n->struct_field.type, ind+2);
            if (n->struct_field.attr_count)
                dbg_print_list("attrs", n->struct_field.attrs, n->struct_field.attr_count, ind+1);
            break;
        case WGSL_NODE_GLOBAL_VAR:
            print_indent(ind+1); printf("name: %s\n", n->global_var.name);
            print_indent(ind+1); printf("address_space: %s\n", n->global_var.address_space ? n->global_var.address_space : "(none)");
            print_indent(ind+1); printf("type:\n");
            dbg_print_node(n->global_var.type, ind+2);
            if (n->global_var.attr_count)
                dbg_print_list("attrs", n->global_var.attrs, n->global_var.attr_count, ind+1);
            break;
        case WGSL_NODE_FUNCTION:
            print_indent(ind+1); printf("name: %s\n", n->function.name);
            if (n->function.attr_count)
                dbg_print_list("attrs", n->function.attrs, n->function.attr_count, ind+1);
            if (n->function.param_count)
                dbg_print_list("params", n->function.params, n->function.param_count, ind+1);
            if (n->function.ret_attr_count)
                dbg_print_list("ret_attrs", n->function.ret_attrs, n->function.ret_attr_count, ind+1);
            if (n->function.return_type) {
                print_indent(ind+1); puts("return_type:");
                dbg_print_node(n->function.return_type, ind+2);
            }
            print_indent(ind+1); puts("body:");
            dbg_print_node(n->function.body, ind+2);
            break;
        case WGSL_NODE_PARAM:
            print_indent(ind+1); printf("name: %s\n", n->param.name);
            if (n->param.attr_count)
                dbg_print_list("attrs", n->param.attrs, n->param.attr_count, ind+1);
            print_indent(ind+1); puts("type:");
            dbg_print_node(n->param.type, ind+2);
            break;
        case WGSL_NODE_BLOCK:
            dbg_print_list("stmts", n->block.stmts, n->block.stmt_count, ind+1); break;
        case WGSL_NODE_VAR_DECL:
            print_indent(ind+1); printf("name: %s\n", n->var_decl.name);
            if (n->var_decl.type) { print_indent(ind+1); puts("type:"); dbg_print_node(n->var_decl.type, ind+2); }
            if (n->var_decl.init) { print_indent(ind+1); puts("init:"); dbg_print_node(n->var_decl.init, ind+2); }
            break;
        case WGSL_NODE_RETURN:
            if (n->return_stmt.expr) { print_indent(ind+1); puts("expr:"); dbg_print_node(n->return_stmt.expr, ind+2); }
            break;
        case WGSL_NODE_EXPR_STMT:
            dbg_print_node(n->expr_stmt.expr, ind+1); break;
        case WGSL_NODE_TYPE:
            print_indent(ind+1); printf("name: %s\n", n->type_node.name);
            if (n->type_node.type_arg_count) dbg_print_list("type_args", n->type_node.type_args, n->type_node.type_arg_count, ind+1);
            if (n->type_node.expr_arg_count) dbg_print_list("expr_args", n->type_node.expr_args, n->type_node.expr_arg_count, ind+1);
            break;
        case WGSL_NODE_ATTRIBUTE:
            print_indent(ind+1); printf("name: %s\n", n->attribute.name);
            if (n->attribute.arg_count) dbg_print_list("args", n->attribute.args, n->attribute.arg_count, ind+1);
            break;
        case WGSL_NODE_IDENT:
            print_indent(ind+1); printf("name: %s\n", n->ident.name); break;
        case WGSL_NODE_LITERAL:
            print_indent(ind+1); printf("literal: %s\n", n->literal.lexeme); break;
        case WGSL_NODE_BINARY:
            print_indent(ind+1); printf("op: %s\n", n->binary.op);
            dbg_print_node(n->binary.left, ind+1);
            dbg_print_node(n->binary.right, ind+1);
            break;
        case WGSL_NODE_ASSIGN:
            print_indent(ind+1); puts("lhs:");
            dbg_print_node(n->assign.lhs, ind+2);
            print_indent(ind+1); puts("rhs:");
            dbg_print_node(n->assign.rhs, ind+2);
            break;
        case WGSL_NODE_CALL:
            print_indent(ind+1); puts("callee:");
            dbg_print_node(n->call.callee, ind+2);
            if (n->call.arg_count) dbg_print_list("args", n->call.args, n->call.arg_count, ind+1);
            break;
        case WGSL_NODE_MEMBER:
            print_indent(ind+1); printf("member: %s\n", n->member.member);
            dbg_print_node(n->member.object, ind+1);
            break;
        case WGSL_NODE_INDEX:
            print_indent(ind+1); puts("object:");
            dbg_print_node(n->index.object, ind+2);
            print_indent(ind+1); puts("index:");
            dbg_print_node(n->index.index, ind+2);
            break;
        case WGSL_NODE_IF:
            print_indent(ind+1); puts("cond:");
            dbg_print_node(n->if_stmt.cond, ind+2);
            print_indent(ind+1); puts("then:");
            dbg_print_node(n->if_stmt.then_branch, ind+2);
            if (n->if_stmt.else_branch) {
                print_indent(ind+1); puts("else:");
                dbg_print_node(n->if_stmt.else_branch, ind+2);
            }
            break;
        
        case WGSL_NODE_WHILE:
            print_indent(ind+1); puts("cond:");
            dbg_print_node(n->while_stmt.cond, ind+2);
            print_indent(ind+1); puts("body:");
            dbg_print_node(n->while_stmt.body, ind+2);
            break;
        
        case WGSL_NODE_FOR:
            if (n->for_stmt.init) { print_indent(ind+1); puts("init:"); dbg_print_node(n->for_stmt.init, ind+2); }
            if (n->for_stmt.cond) { print_indent(ind+1); puts("cond:"); dbg_print_node(n->for_stmt.cond, ind+2); }
            if (n->for_stmt.cont) { print_indent(ind+1); puts("cont:"); dbg_print_node(n->for_stmt.cont, ind+2); }
            print_indent(ind+1); puts("body:");
            dbg_print_node(n->for_stmt.body, ind+2);
            break;
        
        default: break;
    }
}

void wgsl_debug_print(const WgslAstNode *node, int indent) {
    dbg_print_node(node, indent);
}

